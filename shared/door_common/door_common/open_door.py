"""Open a door: detect the handle, plan the whole open, grasp it, swing the door open -- what every appliance's open
shares. The capability sets what differs at startup (`configure`) and says what follows the swing (`AfterSwing`).

1. Detect (`detect_handle`, the capability's detector): looks until two agree within 3 cm (handle and hinge).
2. Plan everything before any motion: the grasp (squared to the door face: cuRobo to a line-up point 25 cm out (or
   less) -- only that goal is checked, cuRobo's path is not previewed -- then straight in along the door normal), and
   the largest swing whose whole arc passes the sim gates (IK, reach, joint jump, J4/J6 guards). cuRobo can't follow a
   line or an arc, so the straight-in leg and the swing stay on joint paths planned and checked here.
3. Execute: cuRobo to the line-up point, straight in, close, one blended swing. The hand stays on the
   handle at the end.

Refusals raise common.Refused: the arm holds where it is.
"""

import time

import numpy as np
import pybullet as p
import rammp_arm
from capability_engine import sim
from capability_engine.cornell_commands import CloseGripperCommand, OpenGripperCommand
from scipy.spatial.transform import Rotation as R

from . import common as dc
from .common import (
    ARM,
    Pose,
    Refused,
    door_arc_waypoints,
    ee_pose,
    fit_circle_xy,
    keep_out_message,
    make_sim,
    multiply_poses,
    plan_cartesian,
    quiet,
    record_door_angle,
    run_cartesian_trajectory,
    save_door_geometry,
    save_frame,
    say,
    short_delta,
)

# --- detection ---
MAX_LOOKS = 5  # keep looking until two detections agree within DETECT_AGREE
MAX_EMPTY_RETRIES = 8  # per look: retries on an empty read (viewpoint-dependent YOLO misses)
DETECT_AGREE = 0.03
HINGE_AGREE = 0.03  # the two agreeing looks' hinges must agree this well too (horizontal)
PLAUSIBLE_X = (0.30, 0.90)  # snorlax 10-30: was 0.85; the planner checks reach itself
PLAUSIBLE_Y = (-0.50, 0.20)
PLAUSIBLE_Z = (
    0.05,
    0.65,
)  # was 0.15 (rchi-cpu-5); Jetson+sheppy 10-07: handle z 0.142, microwave ~4 cm above the arm base
MAX_FACE_YAW_DEG = 70.0  # beyond this the plane fit is more likely wrong than the door

# --- grasp ---
# the capability's corrections from the detected handle to the grasp (grasp_pose); no defaults, see `configure`
GRIP_EXT = None  # along the approach axis (negative = past the detected handle)
VERTICAL_CORR = None  # world z
GRASP_X_ADJUST = None  # along the approach axis
# middle of the finger pads, past the tool point along the approach: ~2 cm in from the fingertips (FINGERTIP_PAST_TOOL,
# measured). An estimate -- what a tilted grasp keeps on the handle (grasp_pose)
GRASP_PADS_PAST_TOOL = 0.042
PREGRASP_STANDOFFS = (0.25, 0.20, 0.15, 0.10)  # line up this far straight out from the grasp (door normal, level),
# already squared, then go straight in along the normal; the largest that passes the gates
# (10 cm still looked diagonal on 10-21: most of the approach was the view -> line-up leg)
GRASP_PITCHES_DEG = (0, 10, 14, 17, 25)  # snorlax 10-30: level first, up to 25. Was (14, 10, 17, 25, 30, 35, 0)
# TEMPORARY 14 first (10-21): the arm's base is tilted a little on this rig, so the user's hand-placed grasp
# "perpendicular to the door" was 14 deg nose-down (demos/user_grasp_pose_2026-10-21.json). REAL DEPLOYMENT: the
# base will be level -> go back to (0, 10, 17, 25, 30, 35) (level first).
# Nose-down tilts tried about the finger axis, IN ORDER (fingers stay
# horizontal across the vertical bar); the smallest reaching the largest swing wins. A low
# handle needs one: 10-07, handle z ~0.14, level -> swing max 25-35 deg (J6 hits its guard),
# 17 -> 70, 25 -> 75 with J6 ending at 92 (sim); the user's hand-placed touches were ~17.
MAX_REACH, MIN_Z, MAX_Z = 0.915, 0.15, 0.75
GRASP_TRACK_ABORT = 0.03


# --- swing ---
SWING_WAYPOINT_SPACING_M = 0.02  # sim-checked spacing
SWING_DONE_TOL_M = 0.02  # the swing counts as done with the hand this close to its last waypoint
SWING_SEND_EVERY = 4  # Kortex slows at every waypoint: send every 4th (8 cm chords) + the last
SWING_DIRECTION = -1
SWING_MAX_TRY_DEG = 75  # tries this down to 45 in 5-deg steps; 80 dragged the microwave (09-29)
SWING_MIN_DEG = 45
SWING_MAX_JUMP_DEG = 25.0
SWING_MAX_REACH = 0.90
SWING_IK_PROBE_M = 0.01  # 200-iteration IK must get within 1 cm (else Kortex hits SINGULARITY_REGION)
J6_GUARD_DEG = 115.0  # J6 hard limit is +-119.7
GRASP_FORCE = 0.25  # gripper current ceiling on the handle (self.arm default 0.5): a looser hold lets the
# bar turn/shift in the fingers instead of tipping the microwave forward (10-07)
SWING_FIT_MIN_DEG = 15  # fit a hinge to the measured path only over at least this much arc
HINGE_RADIUS_RANGE = (0.25, 0.45)

# --- release ---
BACK_OFF_M = 0.20

# --- what differs per appliance ---
MAKE_DETECTOR = None  # () -> the detector (perception.HandleDetector's detect()); one per detect_handle
FIND_HINGE = None  # hinge_detected(...) or hinge_measured(...)
HANDLE_SOURCE = "depth"  # the node's handle_source: "depth" (what sticks out of the door) or "sam3" (SAM 3's mask)
PLACEMENT_NOTE = None  # (handle, door normal, start joints) -> how to place the appliance, printed after detection and
#   added to the refusals. None: nothing
DOOR_MODEL = {}  # door-model geometry written into the door file (common.door_dims). Empty: common's defaults
DOOR_Z_FROM_HANDLE = None  # door bottom / top relative to the handle, when it is taller than any detection sees
KEEP_ARM_IN_FRAMES = False  # each saved detection frame also keeps the arm's joints + tool pose
GRASPED_GAINS = None  # joint impedance {kq, zeta, torque_limit} from the grasp on (the swing, what follows it); None:
#   stiff position control. The approach (line-up, straight in) stays in position control either way
SWING_SPEED_SCALE = 1.0  # the swing's speed as a fraction of the free-air one (CART_SPEED x the goal's speed)
REQUIRED = ("MAKE_DETECTOR", "FIND_HINGE", "GRIP_EXT", "VERTICAL_CORR", "GRASP_X_ADJUST")


def configure(**values):
    """Set this module's constants at startup (the capability's process, the capability's values)."""
    dc.set_constants(globals(), values, REQUIRED)


def grasped_impedance():
    """self.arm's `impedance` for a move holding or pushing the door: GRASPED_GAINS with the usual tracking tolerance,
    or None (stiff position control)."""
    return None if GRASPED_GAINS is None else {**GRASPED_GAINS, "path_tol": rammp_arm.PATH_TOL_RAD}


def _swing_waypoints(start_pose, hinge, deg):
    """Door-arc waypoints [x, y, z, qx, qy, qz, qw] from `start_pose` about `hinge`."""
    radius = float(np.linalg.norm(np.asarray(start_pose.position)[:2] - np.asarray(hinge)[:2]))
    wps = door_arc_waypoints(
        start_pose=start_pose,
        hinge_position=tuple(hinge),
        arc_length_m=radius * np.radians(deg),
        waypoint_spacing_m=SWING_WAYPOINT_SPACING_M,
        direction=SWING_DIRECTION,
        rotate_orientation=True,
    )
    return [list(w.position) + list(w.orientation) for w in wps]


class AfterSwing:
    """What follows the swing, planned with it: a capability subclasses this. This default: nothing."""

    def check(self, scene, rb, geom, cand, explain):
        """Plan what follows from candidate `cand` (plan_open's dict: its "wps"[-1] and "arc"[-1] are the swing's end)
        before any motion. `geom`: the detection. `explain`: if it fails, print why in full. Returns (ok, message)."""
        return True, ""

    def recheck(self, scene, rb, geom, cand, advice):
        """After the grasp, before the swing, from what the swing will really run. Raises Refused to stop there."""


def _check_arc(scene, rb, wps, q0):
    """Chained sim IK over every swing waypoint from joints `q0`. Returns (ok, q_end, message, joints at each waypoint
    -- what the swing sends, so the arm runs exactly what was checked)."""
    q, qs = np.asarray(q0, dtype=float), []
    for i, w in enumerate(wps):
        pos, quat = w[:3], w[3:]
        for j, jj in enumerate(ARM):
            p.resetJointState(rb.robot_id, jj, float(q[j]), physicsClientId=rb.physics_client_id)
        wpose = multiply_poses(scene.robot_base_pose, Pose(tuple(pos), tuple(quat)))
        sol = p.calculateInverseKinematics(
            rb.robot_id,
            rb.end_effector_id,
            list(wpose.position),
            list(wpose.orientation),
            physicsClientId=rb.physics_client_id,
            maxNumIterations=200,
        )
        nq = np.array(sol[:7])
        for j, jj in enumerate(ARM):
            p.resetJointState(rb.robot_id, jj, float(nq[j]), physicsClientId=rb.physics_client_id)
        ikerr = np.linalg.norm(np.array(ee_pose(rb).position) - np.array(wpose.position))
        jump = float(np.max(np.degrees(np.abs(nq - q))))
        j6 = float(np.degrees(nq[5]))
        if (
            ikerr > SWING_IK_PROBE_M
            or np.linalg.norm(pos) > SWING_MAX_REACH
            or jump > SWING_MAX_JUMP_DEG
            or abs(j6) > J6_GUARD_DEG
            or abs(np.degrees(nq[3])) > sim.J4_GUARD_DEG
        ):
            return (
                False,
                q,
                (
                    f"waypoint {i + 1}: ik_err {ikerr * 100:.2f}cm, reach {np.linalg.norm(pos):.3f}, "
                    f"jump {jump:.1f}deg, J4 {np.degrees(nq[3]):.1f}deg, J6 {j6:.1f}deg"
                ),
                qs,
            )
        ko = keep_out_message(rb, nq)  # None unless the capability enabled the keep-out
        if ko is not None:
            return False, q, f"waypoint {i + 1}: {ko}", qs
        q = nq
        qs.append(nq)
    return True, q, f"all {len(wps)} waypoints OK, final J6 {np.degrees(q[5]):.1f}deg", qs


FRAME_ARM = None  # KEEP_ARM_IN_FRAMES: the arm, so each saved detection frame also keeps its joints + tool pose


def _save_frame(frame, tag, attempt, det):
    """Keep every detection's colour + depth frame (and the result) for looking at misses afterwards: real
    depth (uint16 mm), K and the camera pose (common.save_frame). Until 10-21 the depth was written
    as a float32 PNG, which saturated to 0/255 -- those older frames have no usable depth."""
    try:
        arm_state = None
        if FRAME_ARM is not None:
            try:
                arm_state = FRAME_ARM.get_state()
            except Exception as e:
                say(f"  (could not read the arm for the detection frame: {e})")
        save_frame(frame, f"look{tag}_try{attempt}", dc.DETECT_FRAMES_DIR, result=det, arm_state=arm_state)
    except Exception as e:  # never let debugging output stop a run
        say(f"  (could not save the detection frame: {e})")


def _rounded(x, decimals):
    return "-" if x is None else np.round(x, decimals)


def detect_handle():
    """Two agreeing looks. Returns (handle position, handle orientation, door normal, door_z, door_mid, hinge)."""
    cam = dc.camera()
    if not cam.wait_for_frames(30.0):
        raise Refused("No RGB-D frames")
    detector = MAKE_DETECTOR()

    def look(tag):
        for attempt in range(1, MAX_EMPTY_RETRIES + 1):
            f = cam.grab()
            if getattr(cam, "box_topic", "") and f["box"] is None:
                det = None  # boxer has no usable microwave box yet (grab said why): retry, never YOLO instead
            else:
                # handle_source "sam3": SAM 3 masks the handle on the wrist image (the door still comes from depth)
                mask = cam.sam3_handle_mask() if HANDLE_SOURCE == "sam3" else None
                det = (
                    None
                    if HANDLE_SOURCE == "sam3" and mask is None
                    else detector.detect(
                        f["rgb"], f["K"], f["depth_mm"], f["base_T_cam"], box=f.get("box"), handle_mask=mask
                    )
                )
            _save_frame(f, tag, attempt, det)
            if det is not None:
                break
            say(f"  no detection on look {tag} (retry {attempt}/{MAX_EMPTY_RETRIES})")
            time.sleep(0.5)
        else:
            raise Refused(f"NO DETECTION ({tag}) after {MAX_EMPTY_RETRIES} retries")
        say(
            f"  look {tag}: handle {np.round(det['handle'], 4)}  hinge {_rounded(det['hinge'], 4)}  "
            f"door normal {np.round(det['normal'], 3)}  door z {_rounded(det['door_z'], 3)}"
        )
        return det["handle"], det["quat"], det["normal"], det["door_z"], det["door_mid"], det["hinge"]

    looks, pair = [], None
    for i in range(1, MAX_LOOKS + 1):
        looks.append(look(str(i)))
        pair = next(
            (looks[j] for j in range(len(looks) - 1) if np.linalg.norm(looks[j][0] - looks[-1][0]) <= DETECT_AGREE),
            None,
        )
        if pair is not None:
            break
    if pair is None:
        raise Refused(f"No two of {len(looks)} detections agreed within {DETECT_AGREE * 100:.0f} cm -- refusing.")
    (ha, _, na, zra, mida, hga), (hb, orient, nb, zrb, midb, hgb) = pair, looks[-1]
    say(
        f"detections agree to {np.linalg.norm(ha - hb) * 100:.1f} cm after {len(looks)} look(s); "
        + (f"hinges to {np.linalg.norm(hga[:2] - hgb[:2]) * 100:.1f} cm" if hga is not None else "no hinge detected")
    )
    if hga is not None and np.linalg.norm(hga[:2] - hgb[:2]) > HINGE_AGREE:
        raise Refused(f"the two hinge detections differ by more than {HINGE_AGREE * 100:.0f} cm -- refusing.")
    h = (ha + hb) / 2.0
    hinge = None if hga is None else (hga + hgb) / 2.0  # None: the detector sees no door
    n = (na + nb) / np.linalg.norm(na + nb)
    door_z = None if zra is None else np.mean([zra, zrb], axis=0)
    door_mid = None if mida is None else np.mean([mida, midb], axis=0)
    if door_z is not None and not (door_z[0] < h[2] < door_z[1] and 0.15 <= door_z[1] - door_z[0] <= 0.60):
        say(f"door z range {np.round(door_z, 3)} doesn't fit a door around the handle -- not saving it")
        door_z = None
    for nm, v, (lo, hi) in (("x", h[0], PLAUSIBLE_X), ("y", h[1], PLAUSIBLE_Y), ("z", h[2], PLAUSIBLE_Z)):
        if not lo <= v <= hi:
            raise Refused(f"handle {nm}={v:.3f} outside plausible {(lo, hi)} -- refusing.")
    say(f"handle (mean) {np.round(h, 4)}  hinge (mean) {_rounded(hinge, 4)}")
    return h, orient, n, door_z, door_mid, hinge


def grasp_pose(h, orient, n, pitch_deg=0.0):
    """Grasp pose for handle position `h` (vertical-corrected here), squared to the door face, tilted
    `pitch_deg` nose-down about the finger axis. The tool point doesn't depend on the tilt."""
    h = h + np.array([0.0, 0.0, VERTICAL_CORR])
    quat = R.from_quat(orient) * R.from_euler("y", np.pi)
    # The detector's orientation is a fixed constant (right only for a door facing the arm
    # head-on); yaw it about world z so the approach axis points into the face (-normal).
    # Yaw only, so the roll -- fingers closing across the vertical bar -- is kept.
    z0, into = quat.as_matrix()[:, 2], -n
    yaw = float(np.arctan2(z0[0] * into[1] - z0[1] * into[0], z0[0] * into[0] + z0[1] * into[1]))
    say(f"door normal {np.round(n, 3)} -> yaw {np.degrees(yaw):+.1f} deg to face the door square")
    if abs(np.degrees(yaw)) > MAX_FACE_YAW_DEG:
        raise Refused(
            f"Face yaw {np.degrees(yaw):.1f} deg > {MAX_FACE_YAW_DEG} deg -- implausible plane fit? Refusing."
        )
    quat = (R.from_euler("z", yaw) * quat * R.from_euler("x", pitch_deg, degrees=True)).as_quat()
    tilted = R.from_quat(quat).as_matrix()[:, 2]
    # the LEVEL grasp is where the touch check put it; tilted, the finger pads must stay there (10-30: placing the
    # tool point there instead left the pads ~2 cm low and ~0.4 cm short at 25 deg), so the tool point backs out
    # along the tilted approach
    approach = np.r_[tilted[:2], 0.0] / np.linalg.norm(tilted[:2])
    pos = h + approach * (GRASP_X_ADJUST - GRIP_EXT) + (approach - tilted) * GRASP_PADS_PAST_TOOL
    say(f"grasp {np.round(pos, 4)}  range {np.linalg.norm(pos):.3f}")
    if np.linalg.norm(pos) > MAX_REACH or not MIN_Z <= pos[2] <= MAX_Z:
        raise Refused(
            f"GATE FAILED at grasp: range {np.linalg.norm(pos):.3f} (max {MAX_REACH}), "
            f"z {pos[2]:.3f} (range {MIN_Z}-{MAX_Z})"
        )
    return h, Pose(tuple(pos), tuple(quat))


def pregrasp_point(grasp, standoff):
    """`standoff` straight back from the grasp along its level approach axis (= into the door face)."""
    approach = R.from_quat(grasp.orientation).as_matrix()[:, 2]
    approach[2] = 0.0
    approach /= np.linalg.norm(approach)
    return np.asarray(grasp.position) - approach * standoff


def hinge_detected(behind_face, past_edge):
    """The hinge from the detected door edge: the real pivot sits `behind_face` behind the face and `past_edge` beyond
    the visible edge, away from the handle. For FIND_HINGE, which returns (pivot, handle, grasp, what the door file
    saves)."""

    def find(h_raw, orient, n, hinge_seen):
        # the real pivot sits behind the face and beyond the visible edge, expressed in the door's own frame
        away = np.r_[hinge_seen[:2] - h_raw[:2], 0.0]
        away /= np.linalg.norm(away)
        hinge = hinge_seen - np.r_[n[:2], 0.0] / np.linalg.norm(n[:2]) * behind_face + away * past_edge
        say(
            f"hinge: detected edge {np.round(hinge_seen, 4)} -> corrected pivot {np.round(hinge, 4)} "
            f"({behind_face * 100:.1f} cm behind the face, {past_edge * 100:.1f} cm past the edge)"
        )
        h, grasp = grasp_pose(h_raw, orient, n)  # runs the grasp gates once, prints the grasp point
        # demos + door model use the detected edge; swing uses the pivot
        return hinge, h, grasp, dict(hinge=hinge_seen, hinge_pivot=hinge)

    return find


def hinge_measured(offset):
    """The hinge from a measured `offset` from the closed-door grasp point, in the door's frame: into it, along it
    toward the hinge, up. For FIND_HINGE, as hinge_detected."""

    def find(h_raw, orient, n, hinge_seen):
        h, grasp = grasp_pose(h_raw, orient, n)
        into, along = -np.r_[n[:2], 0.0] / np.linalg.norm(n[:2]), np.cross(np.r_[n[:2], 0.0], [0.0, 0.0, 1.0])
        along /= np.linalg.norm(along)
        if hinge_seen is not None:
            say(f"detected door edge {np.round(hinge_seen, 4)} (not used: measured hinge)")
        hinge = np.asarray(grasp.position) + into * offset[0] + along * offset[1] + np.array([0.0, 0.0, offset[2]])
        say(f"hinge: grasp + measured offset {np.round(offset, 3)} (door frame) = {np.round(hinge, 4)}")
        # the door model turns about the measured pivot too
        return hinge, h, grasp, dict(hinge=hinge, hinge_pivot=hinge, detected_edge=hinge_seen)

    return find


# --- the planner's search (plan_open): which arm configurations to try for each grasp, and how far to shift the
# appliance when looking for a placement that works ---
SEARCH_J3_OFFSETS_DEG = (0.0, -30.0, 30.0, -60.0, 60.0)  # the 7-DOF arm's redundant elbow angle, about the start's
# J3; each also with the wrist flipped (J5 + 180, -J6, J7 + 180: the same hand, the other wrist branch)
SEARCH_AFTER_SWING_MAX = 6  # how many swing-feasible candidates (best first) to try the push-open on
SAME_CONFIG_DEG = 10.0  # grasp IK solutions closer than this (every joint) are one configuration
REPLAN_SWING_DEG = 3.0  # after the grasp, re-plan the swing from the real joints if they are further than this off
PLACEMENT_SHIFTS_M = ((0.05, 0.0), (0.10, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05), (0.05, 0.05), (0.05, -0.05))
# (x away from the arm, y to the arm's left), tried on a refusal to say where the appliance would work


CUROBO_LIMIT_MARGIN_DEG = 6.0  # cuRobo refuses to plan to or from joints this close to a limit (its
# joint_limit_margin_rad 0.1 = 5.7 deg, 10-28 START_IN_LIMIT_MARGIN) -- the line-up joints are a cuRobo goal


def _near_limit(rb, q, margin_deg):
    """Which joint of `q` is within `margin_deg` of its limit in the model (continuous joints have none), or ""."""
    for j, jj in enumerate(ARM):
        info = p.getJointInfo(rb.robot_id, jj, physicsClientId=rb.physics_client_id)
        lo, hi = info[8], info[9]
        if lo < hi and not (lo + np.radians(margin_deg) <= q[j] <= hi - np.radians(margin_deg)):
            return f"J{j + 1} {np.degrees(q[j]):.1f} deg is within {margin_deg:g} deg of its limit"
    return ""


def _config_seeds(q0, quick=False):
    """IK seeds spanning the arm's configurations: the start joints with J3 offset (SEARCH_J3_OFFSETS_DEG), each with
    and without the wrist flipped. `quick`: the first three."""
    seeds = []
    for flip in (False, True):
        for d3 in SEARCH_J3_OFFSETS_DEG:
            q = np.array(q0, dtype=float)
            q[2] += np.radians(d3)
            if flip:
                q[4], q[5], q[6] = q[4] + np.pi, -q[5], q[6] + np.pi
            seeds.append(q)
    return seeds[:3] if quick else seeds


def _guard_margin(qs):
    """Smallest distance (deg) of J4 and J6 to their guards over joint configurations `qs`."""
    d = np.degrees(np.asarray(qs, dtype=float))
    return float(min(sim.J4_GUARD_DEG - np.abs(d[:, 3]).max(), J6_GUARD_DEG - np.abs(d[:, 5]).max()))


def _approach_from(scene, rb, grasp, q_g):
    """Line-up point -> straight in to `grasp`, on the configuration of grasp joints `q_g` (the line-up IK is seeded
    there), for the largest PREGRASP_STANDOFFS that passes. The line-up joints are a cuRobo goal, so they must also
    clear cuRobo's own limit margin. Returns (standoff, line-up point, line-up joints, straight-in joint plan), or
    None."""
    g_p, g_q = np.asarray(grasp.position), np.asarray(grasp.orientation)
    for standoff in PREGRASP_STANDOFFS:
        pre_p = pregrasp_point(grasp, standoff)
        if not MIN_Z <= pre_p[2] <= MAX_Z:
            continue
        q_pre, why = dc._ik_within_guards(scene, rb, pre_p, g_q, q_g)
        if q_pre is None:
            say(f"  to line-up {standoff * 100:.0f} cm out: {why}")
            continue
        if keep_out_message(rb, q_pre) is not None:
            continue
        near = _near_limit(rb, q_pre, CUROBO_LIMIT_MARGIN_DEG)
        if near:
            say(f"  to line-up {standoff * 100:.0f} cm out: {near} -- cuRobo refuses goals that close")
            continue
        leg = plan_cartesian(scene, rb, pre_p, g_q, g_p, g_q, q_pre, "straight in")
        if leg is not None:
            return standoff, pre_p, q_pre, leg
    return None


def plan_open(scene, rb, geom, q0, max_swing, min_swing, after_swing, quick=False):
    """Plan the whole open before any motion, searching the arm's configurations: for each nose-down tilt
    (GRASP_PITCHES_DEG), each distinct grasp configuration from _config_seeds, its approach and the largest swing that
    fits; then, best first (largest swing, then most margin to the J4/J6 guards over the whole path), what follows the
    swing (`after_swing`). `geom`: the detection (h_raw, orient, n, hinge, hinge_seen, door_z). Prints a
    line per tilt. Returns (plan dict, or None; the reason when None)."""
    h_raw, orient, n, hinge = geom["h_raw"], geom["orient"], geom["n"], geom["hinge"]
    swings = list(range(max_swing, min(min_swing, max_swing) - 1, -10 if quick else -5))
    if swings[-1] != min(min_swing, max_swing):
        swings.append(min(min_swing, max_swing))
    cands, explained = [], quick  # explained: the first failing approach has been printed in full
    for pitch in GRASP_PITCHES_DEG:
        try:
            with quiet():
                h, g = grasp_pose(h_raw, orient, n, pitch)
        except Refused as e:
            say(f"  tilt {pitch:2d} deg: {e}")
            continue
        g_p, g_q = np.asarray(g.position), np.asarray(g.orientation)
        configs = []
        for seed in _config_seeds(q0, quick):
            q_g, _ = dc._ik_within_guards(scene, rb, g_p, g_q, seed)
            if q_g is not None and all(
                np.degrees(np.max(np.abs(short_delta(c, q_g)))) > SAME_CONFIG_DEG for c in configs
            ):
                configs.append(q_g)
        best, why = None, "no grasp configuration within the J4/J6 guards"
        for q_g in configs:
            with quiet():
                appr = _approach_from(scene, rb, g, q_g)
            if appr is None:
                why = "no line-up + straight-in passes"
                if not explained:  # once, out loud: which standoff, step and check fail
                    say(f"  tilt {pitch:2d} deg: no line-up + straight-in passes -- why, the first configuration's:")
                    _approach_from(scene, rb, g, q_g)
                    explained = True
                continue
            standoff, pre_p, q_pre, leg = appr
            for d in swings:
                wps = _swing_waypoints(g, hinge, d)
                ok, q_end, msg, arc = _check_arc(scene, rb, wps, leg[-1])
                if ok:
                    c = dict(
                        pitch=pitch,
                        swing=d,
                        grasp=g,
                        h=h,
                        standoff=standoff,
                        pre_p=pre_p,
                        q_pre=q_pre,
                        leg=leg,
                        wps=wps,
                        arc=arc,
                        margin=_guard_margin([q_pre, *leg, *arc]),
                        msg=msg,
                    )
                    cands.append(c)
                    if best is None or (c["swing"], c["margin"]) > (best["swing"], best["margin"]):
                        best = c
                    break
                why = f"swing: {msg}"
        say(
            f"  tilt {pitch:2d} deg: {len(configs)} arm configuration(s) -> "
            + (f"swing {best['swing']} deg fits, {best['margin']:.0f} deg from the J4/J6 guards" if best else why)
        )
    if not cands:
        return None, "no tilt and arm configuration gives a grasp + swing that passes the sim checks"
    cands.sort(key=lambda c: (c["swing"], c["margin"]), reverse=True)
    reason = None
    for c in cands[: 3 if quick else SEARCH_AFTER_SWING_MAX]:
        ok, m = after_swing.check(scene, rb, geom, c, reason is None and not quick)
        c["after"] = m
        if ok:
            return c, None
        reason = reason or f"tilt {c['pitch']} deg, swing {c['swing']} deg fits but {m}"
    return None, reason


def _placement_advice(scene, rb, geom, q0, max_swing, min_swing, after_swing):
    """Where the appliance would work: the open re-planned (quick) with the detection shifted by PLACEMENT_SHIFTS_M."""
    works = []
    for dx, dy in PLACEMENT_SHIFTS_M:
        shift = np.array([dx, dy, 0.0])
        g2 = dict(geom, h_raw=geom["h_raw"] + shift, hinge=geom["hinge"] + shift, hinge_seen=geom["hinge_seen"] + shift)
        with quiet():
            plan, _ = plan_open(scene, rb, g2, q0, max_swing, min_swing, after_swing, quick=True)
        if plan is not None:
            where = []
            if dx:
                where.append(f"{abs(dx) * 100:.0f} cm {'further from' if dx > 0 else 'closer to'} the arm")
            if dy:
                where.append(f"{abs(dy) * 100:.0f} cm to the arm's {'left' if dy > 0 else 'right'}")
            works.append(f"{' and '.join(where)} (tilt {plan['pitch']} deg, swing {plan['swing']} deg)")
    if not works:
        return f"\nNo shift of the {dc.APPLIANCE} tried (up to 10 cm) makes it work from here."
    return f"\nWould work with the {dc.APPLIANCE} moved: " + "; or ".join(works) + "."


def open_task(
    ai,
    execute,
    max_swing=SWING_MAX_TRY_DEG,
    min_swing=SWING_MIN_DEG,
    after_swing=None,
    before_detect=None,
    on_grasp=None,
):
    """Detect, plan, grasp, swing. `before_detect` `(ai, scene, rb, execute)`: runs first; returning True ends the task
    there. `after_swing`: what follows the swing (AfterSwing), planned with it before any motion. `on_grasp()`: called
    once the gripper has closed on the handle -- from then on a stop leaves the hand holding it.
    Returns True when the swing finished (or the dry run passed); raises Refused on any refusal."""
    scene, rb = make_sim()
    after_swing = after_swing or AfterSwing()
    if before_detect is not None and before_detect(ai, scene, rb, execute):
        return True

    st = ai.get_state()
    q0 = np.asarray(st["position"], dtype=float)
    if float(st["gripper_pos"]) > 0.2:
        say("gripper is closed -- " + ("opening it first" if execute else "the real run opens it first"))
        if execute:
            ai.execute_command(OpenGripperCommand())
            time.sleep(1.5)
            if float(ai.get_state()["gripper_pos"]) > 0.2:
                raise Refused("Gripper still closed after opening -- refusing to grasp.")

    global FRAME_ARM
    if KEEP_ARM_IN_FRAMES:
        FRAME_ARM = ai
    h_raw, orient, n, door_z, door_mid, hinge_seen = detect_handle()
    advice = ""
    if PLACEMENT_NOTE is not None:
        advice = "\n" + PLACEMENT_NOTE(h_raw, n, q0)
        say(advice[1:])
    try:
        hinge, h, grasp, hinge_fields = FIND_HINGE(h_raw, orient, n, hinge_seen)
    except Refused as e:
        raise Refused(f"{e}{advice}") from e
    radius = float(np.linalg.norm(np.asarray(grasp.position)[:2] - hinge[:2]))
    say(f"hinge (detected) {np.round(hinge, 4)}  radius {radius * 100:.1f} cm")
    if not HINGE_RADIUS_RANGE[0] <= radius <= HINGE_RADIUS_RANGE[1]:
        raise Refused(f"hinge radius {radius * 100:.1f} cm is implausible (door ~35 cm) -- refusing.{advice}")

    # --- plan everything before any motion: tilts x arm configurations x swings, then what follows the swing ---
    geom = dict(h_raw=h_raw, orient=orient, n=n, hinge=hinge, hinge_seen=hinge_seen, door_z=door_z)
    plan, why = plan_open(scene, rb, geom, q0, max_swing, min_swing, after_swing)
    if plan is None:
        say(f"  (refused -- looking for a {dc.APPLIANCE} placement that works ...)")
        advice += _placement_advice(scene, rb, geom, q0, max_swing, min_swing, after_swing)
        raise Refused(f"{why} -- refusing before any motion. Start joints {np.round(np.degrees(q0), 1)}.{advice}")
    deg, pitch, grasp, h = plan["swing"], plan["pitch"], plan["grasp"], plan["h"]
    g_p, g_q = np.asarray(grasp.position), np.asarray(grasp.orientation)
    say(
        f"chosen: tilt {pitch} deg nose-down, swing {deg} deg, line-up {plan['standoff'] * 100:.0f} cm out, "
        f"{plan['margin']:.0f} deg from the J4/J6 guards; grasp {np.round(g_p, 4)}; line-up joints "
        f"{np.round(np.degrees(plan['q_pre']), 1)}{'; ' + plan['after'] if plan.get('after') else ''}"
    )

    if not execute:
        say(f"\nDRY RUN -- would grasp, then swing {deg} deg. Nothing commanded.")
        return True

    # --- grasp: cuRobo to the planned line-up JOINTS, then the planned straight-in joints ---
    if not ai.go_to_joint_config(plan["q_pre"], "to line-up"):
        raise Refused("Did not reach the line-up point -- ABORT before touching the handle, gripper untouched.")
    if hasattr(ai, "move_joints"):  # the checked joint path itself, timed as the straight line it is
        step = float(np.linalg.norm(g_p - plan["pre_p"])) / len(plan["leg"])
        ai.move_joints(list(plan["leg"]), cart_lengths=[step] * len(plan["leg"]))
        time.sleep(0.3)
        e = float(np.linalg.norm(np.asarray(ai.get_state()["ee_pos"][:3]) - g_p))
    else:
        _, e, _ = run_cartesian_trajectory(ai, [(list(g_p), list(g_q))])
    say(f"grasp: tracking {e * 100:.1f} cm")
    if e > GRASP_TRACK_ABORT:
        raise Refused(f"Tracking {e * 100:.1f} cm at the grasp -- ABORT, gripper untouched.")
    # closed-door geometry for the close task (door_z/door_mid written even when None, so stale ones can't linger)
    if DOOR_Z_FROM_HANDLE is not None:  # the camera sees only part of the door's height
        door_z = h[2] + np.asarray(DOOR_Z_FROM_HANDLE)
    save_door_geometry(
        closed_normal=n,
        closed_grasp_pos=g_p,
        closed_handle=h,
        door_z=door_z,
        door_mid=door_mid,
        **DOOR_MODEL,
    )
    if hasattr(ai, "gripper"):
        ai.gripper(1.0, force=GRASP_FORCE)  # looser hold (sheppy)
    else:
        ai.execute_command(CloseGripperCommand())
    t0, last = time.time(), -1.0
    time.sleep(0.6)
    while time.time() - t0 < 3.5:  # until the fingers stop closing
        g = float(ai.get_state()["gripper_pos"])
        if abs(g - last) < 0.003:
            break
        last = g
        time.sleep(0.2)
    say(f"gripper after close: {g:.4f} (can't confirm a grasp on this rig -- watch it)")
    if g < 0.2:
        raise Refused("Gripper is OPEN -- nothing grasped. Refusing to swing.")
    if on_grasp is not None:
        on_grasp()

    # --- swing: the planned joints, unless the arm isn't where the plan put it -- then re-planned from where it is ---
    st = ai.get_state()
    ee = list(st["ee_pos"])
    off = float(np.degrees(np.max(np.abs(short_delta(plan["leg"][-1], st["position"])))))
    if off <= REPLAN_SWING_DEG:
        wps, arc = plan["wps"], plan["arc"]
    else:
        say(f"  the grasp joints are {off:.1f} deg off the plan's -- re-planning the swing from them")
        wps = _swing_waypoints(Pose(tuple(ee[:3]), tuple(ee[3:7])), hinge, deg)
        ok, _, msg, arc = _check_arc(scene, rb, wps, st["position"])
        if not ok:
            raise Refused(f"swing re-check from the real grasp failed ({msg}) -- stopping, still holding the handle.")
    after_swing.recheck(scene, rb, geom, {**plan, "wps": wps, "arc": arc}, advice)
    save_door_geometry(**hinge_fields)
    if hasattr(ai, "move_joints"):  # the checked joints at every waypoint, as one smooth trajectory
        say(f"swing {deg} deg: sending the {len(arc)} checked waypoints' joints ...")
        ai.move_joints(
            list(arc),
            cart_lengths=[SWING_WAYPOINT_SPACING_M] * len(arc),
            speed_scale=SWING_SPEED_SCALE,
            impedance=grasped_impedance(),
        )
        time.sleep(0.3)
        err = float(np.linalg.norm(np.asarray(ai.get_state()["ee_pos"][:3]) - np.asarray(wps[-1][:3])))
        ok = err <= SWING_DONE_TOL_M
    else:
        sent = wps[SWING_SEND_EVERY - 1 :: SWING_SEND_EVERY]
        if sent[-1] is not wps[-1]:
            sent.append(wps[-1])
        say(f"swing {deg} deg: {len(wps)} waypoints checked, sending {len(sent)} ...")
        ok, err, _ = run_cartesian_trajectory(
            ai, [(w[:3], w[3:]) for w in sent], impedance=grasped_impedance(), speed_scale=SWING_SPEED_SCALE
        )
    path = np.array([p_[1:] for p_ in getattr(ai, "last_path", [])])
    final = ai.get_state()
    arc_deg = 0.0
    if len(path) > 1:
        a0, a1 = path[0, :2] - hinge[:2], path[-1, :2] - hinge[:2]
        arc_deg = abs(np.degrees(np.arctan2(a0[0] * a1[1] - a0[1] * a1[0], a0 @ a1)))
    if len(path) > 20 and arc_deg >= SWING_FIT_MIN_DEG:
        c, r, rms = fit_circle_xy(path[:, :2])
        cmd = np.array([w[:3] for w in wps])
        dev = np.max(np.linalg.norm(path[:, None, :2] - cmd[None, :, :2], axis=2).min(axis=1))
        say(
            f"  measured hand path: {len(path)} samples, max {dev * 100:.1f} cm off the commanded arc; circle fit "
            f"hinge {np.round(c, 4)} r {r * 100:.1f} cm (rms {rms * 1000:.1f} mm) vs detected hinge "
            f"{np.round(hinge[:2], 4)} r {np.linalg.norm(np.asarray(ee[:2]) - hinge[:2]) * 100:.1f} cm -- "
            f"off by {np.linalg.norm(c - hinge[:2]) * 100:.1f} cm"
        )
    say(
        f"SWING {'DONE' if ok else 'STOPPED SHORT'}: EE {np.round(final['ee_pos'][:3], 4)} "
        f"({err * 100:.1f} cm from the last waypoint)"
    )
    record_door_angle(final["ee_pos"])
    return bool(ok)
