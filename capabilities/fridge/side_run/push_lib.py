"""PART 3 (11-06), planned before any motion and run right after the swing in run_general.py (--push):
1. let go: gripper open, the hand slides off the bar back along its own axis turning back to the grasp's orientation,
   then out from the open door's face (spare-joint IK, impedance);
2. round the door's free edge: Cailyn's push_open keypoints (past the free edge, round it turning the hand), joints by the
   spare-joint IK, >= her DEMO_MIN_CLEAR off the door model and >= SELF_MIN_M between the arm's own links, in POSITION
   control (11-06: under impedance it jammed into itself);
3. ONE swing about the hinge from there under the HARDEST impedance (no back-out along the door first): closing in on the
   inner face over its first CLOSE_IN_STEPS steps to her push point, then on to F.PUSH_OPEN_TO_DEG."""
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from capability_engine import sim
from door_common import common as dc
from door_common import demo_replay
from fridge import constants as F
from fridge import push_open as PO
import pull_lib

SLIDE_OFF_M, OUT_M, STEP_M = 0.05, 0.05, 0.01  # 11-06: a 5 cm slide-off and 5 cm out (room for the elbow at the free edge)
FIRST_OUT_M = 0.0  # 11-06: out from the door's face WHILE the gripper opens -- the angled fingertip would
# poke the door shut as it opens; dragging the door a little more open is fine
SELF_MIN_M = 0.04
CLEAR_MIN_M = demo_replay.DEMO_MIN_CLEAR
CLOSE_IN_STEPS = 2
PRE_OUT_M = 0.0  # 11-06: the rounding starts straight out from the open door's face (no slide-off before it)
PUSH_BACK_DEG = 30.0  # 11-06: the final push straight, this far back from pure left
FORWARD_M = 0.18  # 11-06: after rounding the edge, this far forward along the fridge's normal (toward it)
# the scene camera (not in the sim; 11-06 the arm hit it): arm_mount -> scene_camera_color_optical_frame from TF, a sphere
CAMERA_XYZ = (0.132, -0.441, 0.328)
CAMERA_R_M = 0.09
CAMERA_MIN_M = -1.0  # off (11-06)
_CAM = {}


def camera_body(rb):
    import pybullet as pb
    c = rb.physics_client_id
    if c not in _CAM:
        shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=CAMERA_R_M, physicsClientId=c)
        _CAM[c] = pb.createMultiBody(0, shape, -1, list(CAMERA_XYZ), physicsClientId=c)
    return [(_CAM[c], "scene camera")]
ROUND_SPEED = 1.0
DOOR_DRIFT_DEG = 0.0  # 11-06: NO extra reach forward before the swing (it ran the arm into the scene camera)
SLIDE_AFTER_OPEN_S = 0.3  # the slide starts this long after the gripper starts opening (no finger dragging the bar)
# the let-go (11-06: at full speed under the hardest gains it pushed the door shut): half speed, medium joint
# impedance (the driver's medium profile: a finger touching the bar gives), the hand's turn-back ramped in (smoothstep)
# the release (11-06): while the gripper opens, the hand keeps pulling the door RELEASE_ARC_DEG further along
# its arc (the angled fingertip can't push it back); only once the gripper is fully open does it slide off the bar
RELEASE_ARC_DEG = 8.0
OPEN_DURING_S = 2.0  # the pull runs on along the arc at least this long after 45 deg: the gripper opens while it moves
OPEN_LEAD_S = 0.3  # the gripper starts opening this long before the hand passes the swing's end
GRIP_OPEN_SPEED = 0.2
EXIT_SPEED = 0.5  # the slide off the bar and out from the door
SETTLE_S = 1.0  # after the slide, wait up to this long for the joints to arrive before judging it
SWING_SPEED = 1.0
AFTER_OPEN_FRACTION = 0.5  # of the driver max joint speed (80% elsewhere): impedance tracking
AFTER_OPEN_ACCEL = 1.0  # rad/s2 (1.5 elsewhere)
AFTER_OPEN_PATH_TOL = 0.25  # rad: the tracking limit for this free-space motion (0.15 elsewhere)
AFTER_OPEN_SPEED = 1.0  # 11-06: slide-off + round the edge + forward + swing as ONE trajectory under the hardest
# impedance (no stop, and no impedance -> position switch: the per-actuator mode switch jerked the arm)
PLACEMENTS = [(0.44, 45, 1.0), (0.44, 30, 1.0), (0.42, 45, 1.0), (0.42, 30, 1.0), (0.40, 45, 1.0), (0.40, 30, 1.0),
              (0.44, 15, 1.0), (0.44, 0, 1.0), (0.35, 30, 1.0), (0.30, 30, 1.0)]  # (u from the hinge, fingers, tool y)
F.PUSH_OPEN_EDGE_CLEAR_M = 0.10  # hers 0.16: ~37 cm from the arm at this placement


def door_uv(df, pos, deg):
    d = df._rot(deg).T @ (np.asarray(pos, float)[:2] - df.hinge[:2])
    return float(d @ df.e), float(d @ df.n) - df.mid


def door_deg_from_hand(p, ee):
    """The door's angle from the hand on the bar: its turn about the hinge pin since the grasp."""
    hinge, grasp = np.asarray(p["hinge"], float), np.asarray(p["grasp"], float)
    a0, a1 = grasp[:2] - hinge[:2], np.asarray(ee, float)[:2] - hinge[:2]
    return abs(float(np.degrees(np.arctan2(a0[0] * a1[1] - a0[1] * a1[0], a0 @ a1))))


def plan_exit(scene, rb, p, ee, q, deg):
    """The release from the swing's end (tool pose ee / joints q, the door at deg): {arc, arc_lens} -- the hand on the
    bar RELEASE_ARC_DEG further along the door's arc about the hinge pin (its orientation kept) while the gripper opens --
    then {slide, slide_lens, slide_poses} -- back off the bar along the hand's own axis turning back to the grasp's
    orientation, and out from the door's face; deg_after: the door then. (None, why) if either doesn't solve."""
    from door_common.common import door_arc_waypoints
    hinge, sgn = np.asarray(p["hinge"], float), int(p["hinge_sign"])
    ee = np.asarray(ee, float)
    rad = float(np.linalg.norm(ee[:2] - hinge[:2]))
    wps = door_arc_waypoints(sim.Pose(tuple(ee[:3]), tuple(ee[3:7])), tuple(hinge), rad * np.radians(RELEASE_ARC_DEG),
                             0.01, direction=-sgn, rotate_orientation=False)
    arc = [(np.asarray(w.position, float), ee[3:7]) for w in wps]
    arc_k, _, why = pull_lib.swing_knots_ns(scene, rb, np.asarray(q, float), arc)
    if arc_k is None:
        return None, f"the release arc ({RELEASE_ARC_DEG:g} deg on) doesn't solve: {why}"
    deg_after = deg + RELEASE_ARC_DEG
    e1 = np.r_[arc[-1][0], arc[-1][1]]
    n = np.array([p["normal"][0], p["normal"][1], 0.0]); n /= np.linalg.norm(n)
    out_open = R.from_euler("z", -sgn * np.radians(deg_after)).apply(n)
    back = -R.from_quat(e1[3:7]).as_matrix()[:, 2]; back[2] = 0.0; back /= np.linalg.norm(back)
    turn_back = Slerp([0.0, 1.0], R.from_quat([e1[3:7], p["quat"]])); q_end = turn_back(1.0).as_quat()
    k1, k2 = int(np.ceil(SLIDE_OFF_M / STEP_M)), int(np.ceil(OUT_M / STEP_M))
    p1 = e1[:3] + back * SLIDE_OFF_M
    ease = lambda f: f * f * (3 - 2 * f)  # noqa: E731 -- the turn mostly after the fingertips are off the bar
    sl = ([(e1[:3] + back * SLIDE_OFF_M * i / k1, turn_back(ease(i / k1)).as_quat()) for i in range(1, k1 + 1)]
          + [(p1 + out_open * OUT_M * i / k2, q_end) for i in range(1, k2 + 1)])
    sl_k, _, why = pull_lib.swing_knots_ns(scene, rb, np.asarray(arc_k[-1], float), sl)
    if sl_k is None:
        return None, f"the slide off the bar doesn't solve: {why}"
    lens = lambda start, poses: [float(np.linalg.norm(b_[0] - a_[0])) for a_, b_ in zip([(start, None)] + poses[:-1], poses)]  # noqa: E731
    return {"arc": arc_k, "arc_lens": lens(ee[:3], arc), "slide": sl_k, "slide_lens": lens(e1[:3], sl),
            "slide_poses": sl, "deg_after": deg_after}, ""


def plan_push(scene, rb, door, st, say=print):
    """Round the edge + the swing from arm state st (off the bar): dict or (None, reasons)."""
    df = dc.DoorFrame(door, door["open_sign"])
    deg0 = float(door["door_open_deg"])
    bodies = df.bodies(scene, rb, deg0)
    why = []
    try:
        for u, a, y in PLACEMENTS:
            out_dir = np.r_[df._rot(deg0) @ df.n, 0.0]  # the open door's outward normal
            ee0 = np.asarray(st["ee_pos"], float)
            st_out = {"position": st["position"], "ee_pos": list(np.r_[ee0[:3] + out_dir * PRE_OUT_M, ee0[3:7]])}
            pts, first = PO.push_open_points(scene, rb, door, st_out, u, a, y, False)
            if pts is not None:  # from the hand: out first, then her route from there
                pts = [(ee0[:3], ee0[3:7], "release end")] + pts
                first += 1
            if pts is None:
                why.append(f"{u}/{a}: can't be placed at the inner face"); continue
            into = -np.array([door["closed_normal"][0], door["closed_normal"][1], 0.0]); into /= np.linalg.norm(into)
            fwd = (np.asarray(pts[3][0], float) + into * FORWARD_M, pts[3][1], "forward along the fridge's normal")
            pts = pts[:4] + [fwd] + pts[4:]
            first += 1
            poses, leg_of = [], []
            for li, ((p0, q0, _), (p1, q1, _)) in enumerate(((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[4]))):
                k = max(2, int(np.ceil(np.linalg.norm(np.asarray(p1) - p0) / 0.01)))
                sl = Slerp([0.0, 1.0], R.from_quat([q0, q1]))
                poses += [(np.asarray(p0) + (np.asarray(p1) - p0) * i / k, sl(i / k).as_quat()) for i in range(1, k + 1)]
                leg_of += [li] * k
            knots, _, w = pull_lib.swing_knots_ns(scene, rb, np.asarray(st["position"], float), poses)
            if knots is None:
                why.append(f"{u}/{a}: round the edge: {w}"); continue
            cls = [dc.clearance(rb, kq, bodies) for kq in knots]
            ci = int(np.argmin([c_[0] for c_ in cls])); cl = cls[ci][0]
            if cl < CLEAR_MIN_M:
                legs_n = ("out from the door", "along to the free edge", "round the edge", "forward along the normal")
                why.append(f"{u}/{a}: round the edge {cl * 100:.1f} cm off the door ({legs_n[leg_of[ci]]}, {cls[ci][1]})"); continue
            cam = camera_body(rb)
            cc = min(dc.clearance(rb, kq, cam)[0] for kq in knots)
            if cc < CAMERA_MIN_M:
                why.append(f"{u}/{a}: round the edge {cc * 100:.1f} cm from the scene camera"); continue
            sc = min((pull_lib.self_clearance(rb, kq) for kq in knots), key=lambda t: t[0])
            if sc[0] < SELF_MIN_M:
                why.append(f"{u}/{a}: round the edge {sc[0] * 100:.1f} cm arm to itself"); continue
            # the push from the forward point (11-06): a STRAIGHT move PUSH_BACK_DEG back from pure left (toward
            # the hinge side, angled back toward the robot) -- not the arc; straight left got stuck on the door. The hand's
            # orientation and height kept; as far along that line as her push point stands at PUSH_OPEN_TO_DEG
            th = np.asarray(door["hinge"], float) - np.asarray(door["closed_handle"], float)
            th[2] = 0.0
            th -= (th @ into) * into  # along the closed door, toward the hinge
            th /= np.linalg.norm(th)
            dirn = np.cos(np.radians(PUSH_BACK_DEG)) * th - np.sin(np.radians(PUSH_BACK_DEG)) * into
            dirn /= np.linalg.norm(dirn)
            up, vp = door_uv(df, pts[first][0], deg0)
            p90 = df.pos(up, vp, float(F.PUSH_OPEN_TO_DEG), float(pts[4][0][2]))
            left_m = float((p90 - np.asarray(pts[4][0], float)) @ dirn)
            if left_m <= 0.0:
                why.append(f"{u}/{a}: the door at 90 is not ahead of the push line"); continue
            nst = max(2, int(np.ceil(left_m / 0.04)))
            sw = [(np.asarray(pts[4][0], float) + dirn * left_m * kk / nst, pts[4][1], f"push {left_m * kk / nst * 100:.0f} cm")
                  for kk in range(1, nst + 1)]
            st_mid = {"position": list(knots[-1]), "ee_pos": list(np.r_[pts[4][0], pts[4][1]])}
            with dc.quiet():
                legs = demo_replay.plan(scene, rb, door, st_mid, [(a_, b_) for a_, b_, _ in sw], 0, lift_off=False)
            if legs is None:
                why.append(f"{u}/{a}: the swing fails a gate"); continue
            cc2 = min(dc.clearance(rb, kq, cam)[0] for leg in legs for kq in leg)
            if cc2 < CAMERA_MIN_M:
                why.append(f"{u}/{a}: the swing {cc2 * 100:.1f} cm from the scene camera"); continue
            sc2 = min((pull_lib.self_clearance(rb, kq) for leg in legs for kq in leg), key=lambda t: t[0])
            if sc2[0] < SELF_MIN_M:
                why.append(f"{u}/{a}: the swing {sc2[0] * 100:.1f} cm arm to itself"); continue
            lens = [float(np.linalg.norm(b_[0] - a_[0])) for a_, b_ in zip([(pts[0][0], None)] + poses[:-1], poses)]
            from fridge.motion import _leg_lengths
            kp = [(pts[4][0], pts[4][1])] + [(a_, b_) for a_, b_, _ in sw]
            return {"round": knots, "round_lens": lens, "swing": [qq for leg in legs for qq in leg],
                    "swing_lens": _leg_lengths(kp, legs), "pts": pts[:5] + sw, "placement": (u, a, y),
                    "left_m": left_m, "self_min_m": min(sc[0], sc2[0]), "door_min_m": cl, "camera_min_m": min(cc, cc2)}, ""
    finally:
        dc.drop(rb, bodies)
    return None, " | ".join(why)
