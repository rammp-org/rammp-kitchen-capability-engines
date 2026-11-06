"""Open the fridge: detect the handle, grasp it, swing the door open, let go, go back round the door -- one run.

1. Detect, plan, grasp, swing: door_common.open_door, with this fridge's values (`configure`). The handle comes from the
   whole image (perception.HandleDetector); the hinge is measured from the grasp, not detected. The grasp is at the
   middle of the bar, or `grasp_offset_m` above it (below if negative), within the measured bar (`set_grasp_offset`).
   Exactly the asked swing or the run refuses, and what follows the swing is planned with it, before anything moves.
2. By default (`push_open.PushOpen`): a 70-deg pull, then let go and push the door on to 90. On its one hardware run
   (10-24, after a 50-deg pull) it finished its trajectory but the door swung shut.
3. With push_open false (`ReleaseAround`, a 90-deg pull): let go, back off the handle, then back along the open door
   past its free edge, round it and in beside the door, facing the fridge -- so the arm ends on the door's inner side.
   It always ends upright, as at the grasp; it may go round the edge upside down and roll upright on the way in. No
   park.

Refusals raise door_common.common.Refused: the arm holds where it is.
"""

import json
from functools import partial

import numpy as np
from door_common import common as dc
from door_common import open_door as shared
from door_common.close_door import _chain
from door_common.common import DoorFrame, Refused, drop, make_sim, plan_straight_line, quiet, say
from scipy.spatial.transform import Rotation as R

from . import constants as F
from . import push_open as P
from .motion import AfterSwing, door_after, logged_run, send_legs, trajectory_keep_out
from .push_open import PushOpen, push_after_chair_move, push_open


def _make_detector(handle_centre_to_face_m, door_square_to_arm, door_normal_from_handle=False):
    from .perception import HandleDetector  # here, not at import: open3d, which the planning tests don't have

    return HandleDetector(handle_centre_to_face_m, door_square_to_arm, door_normal_from_handle)


def placement_advice(handle, normal, start_joints=None, level=True):
    """Where to move the wheelchair to get the handle into the placement that passed in sim."""
    x, y = float(handle[0]), float(handle[1])
    yaw = float(np.degrees(np.arctan2(-normal[1], -normal[0])))

    def cm(d):
        return f"~{max(1, round(d * 100))} cm"

    moves = []
    if x > F.PLACEMENT_X[1]:
        moves.append(f"move the wheelchair {cm(x - F.PLACEMENT_X[1])} closer to the fridge")
    elif x < F.PLACEMENT_X[0]:
        moves.append(f"move the wheelchair {cm(F.PLACEMENT_X[0] - x)} back from the fridge")
    if y < F.PLACEMENT_Y[0]:
        moves.append(f"move the wheelchair {cm(F.PLACEMENT_Y[0] - y)} to its right")
    elif y > F.PLACEMENT_Y[1]:
        moves.append(f"move the wheelchair {cm(y - F.PLACEMENT_Y[1])} to its left")
    if yaw > F.PLACEMENT_YAW_DEG[1]:
        moves.append(f"turn the wheelchair ~{yaw - F.PLACEMENT_YAW_DEG[1]:.0f} deg so its front swings left")
    elif yaw < F.PLACEMENT_YAW_DEG[0]:
        moves.append(f"turn the wheelchair ~{F.PLACEMENT_YAW_DEG[0] - yaw:.0f} deg so its front swings right")
    lines = [
        f"placement: handle x {x:.3f} y {y:.3f}, yaw {yaw:+.1f} deg (passed in sim: x {F.PLACEMENT_X[0]:.2f}-"
        f"{F.PLACEMENT_X[1]:.2f}, y {F.PLACEMENT_Y[0]:.2f} to {F.PLACEMENT_Y[1]:.2f}, yaw "
        f"{F.PLACEMENT_YAW_DEG[0]:+.0f} to {F.PLACEMENT_YAW_DEG[1]:+.0f})"
    ]
    lines += [f"  -> {m}" for m in moves] or ["  -> inside that placement"]
    if len(moves) > 1:
        lines.append("  (re-check after each move: turning the chair also shifts the handle)")
    if level and start_joints is not None and np.degrees(start_joints[0]) < 0:  # that pose is the level arm's
        lines.append(f"  -> start from the web app's Fridge start pose (J1 now {np.degrees(start_joints[0]):.0f} deg)")
    return "\n".join(lines)


_HANDLE_LENGTH = F.MEASURED["handle_length_m"]  # configure sets these two
_DOOR_Z_FROM_HANDLE = (-F.MEASURED["door_below_handle_m"], F.MEASURED["door_above_handle_m"])


def configure(data_dir, mount_tilt_deg=0, door_square_to_arm=False, door_normal_from_handle=False, **measured):
    """Everything door_common needs from the fridge: call once at startup. `mount_tilt_deg`: a key of constants.MOUNTS.
    `door_square_to_arm` (testing only): the door taken to face the arm squarely, no door plane fit (perception).
    `door_normal_from_handle`: the door's normal from the handle's own depth, no door plane fit (perception; SAM 3).
    `measured`: constants.MEASURED's names."""
    if door_square_to_arm and door_normal_from_handle:
        raise ValueError("door_square_to_arm and door_normal_from_handle both set the door's normal: pick one")
    global _HANDLE_LENGTH, _DOOR_Z_FROM_HANDLE
    m = {**F.MEASURED, **measured}
    _HANDLE_LENGTH = float(m["handle_length_m"])
    _DOOR_Z_FROM_HANDLE = (-m["door_below_handle_m"], m["door_above_handle_m"])
    F.configure(data_dir, mount_tilt_deg)
    mount = F.MOUNTS[mount_tilt_deg]
    # the hinge and the door's face from the handle (where the fingers close), as they are measured -- not from the
    # grasp's tool point ~5 cm in front of it (until 11-05 they were: the face came out 1 cm in front of the handle).
    # The offset in the door's frame: into it, along it, up. A wrong one drags the door or the fridge during the swing
    hinge_offset = np.array([m["handle_centre_to_face_m"] + m["pin_behind_face_m"], m["hinge_along_door_m"], 0.0])
    shared.configure(
        MAKE_DETECTOR=partial(
            _make_detector, m["handle_centre_to_face_m"], door_square_to_arm, door_normal_from_handle
        ),
        FIND_HINGE=shared.hinge_measured(hinge_offset, from_handle=True),
        GRIP_EXT=F.GRIP_EXT,
        VERTICAL_CORR=F.VERTICAL_CORR,
        GRASP_X_ADJUST=F.GRASP_X_ADJUST,
        GRASP_FORCE=F.GRASP_FORCE,
        # no gates on where the handle is or how high the grasp is: the arm's reach, IK, joint guards, the keep-out and
        # the door model's collision checks decide what can be done; a handle out of place refuses there, not here
        PLAUSIBLE_X=None,
        PLAUSIBLE_Y=None,
        PLAUSIBLE_Z=None,
        MIN_Z=None,
        MAX_Z=None,
        HINGE_RADIUS_RANGE=None,  # the measured hinge is always hinge_along_door_m from the grasp
        PREFER_NEAR_START=True,  # of the plans with the largest swing, the least reconfiguration for cuRobo
        # no detected hinge to compare: the two looks' doors directly (11-04: 31 deg apart). From the handle, 5: on the
        # RealSense frames, two looks within 5 deg put 90% of doors within 8.5 deg (13 without the check)
        NORMAL_AGREE_DEG=5.0 if door_normal_from_handle else 10.0,
        MAX_REACH=mount["max_reach"],
        SWING_MAX_REACH=mount["swing_max_reach"],
        GRASP_PITCHES_DEG=mount["grasp_pitches_deg"],
        PLACEMENT_NOTE=partial(placement_advice, level=mount_tilt_deg == 0),
        DOOR_MODEL=dict(
            face_behind_grasp=m["handle_centre_to_face_m"],
            door_t=m["door_depth_m"],
            door_past_handle=m["door_width_m"] - m["hinge_along_door_m"],
            body_depth=m["body_depth_m"],
        ),
        DOOR_FACE_BEHIND_HANDLE=m["handle_centre_to_face_m"],
        DOOR_Z_FROM_HANDLE=_DOOR_Z_FROM_HANDLE,  # the camera sees only part of it
        KEEP_ARM_IN_FRAMES=True,
    )
    dc.enable_keep_out(m["chair_keep_out_behind_m"])  # every plan and every trajectory sent is checked


def grasp_offset_limit():
    """How far above or below the middle of the bar the grasp may be (m): the finger pads stay GRASP_END_MARGIN_M in
    from either end of the measured bar."""
    return max(0.0, _HANDLE_LENGTH / 2 - F.GRASP_END_MARGIN_M)


def set_grasp_offset(offset):
    """Grasp `offset` (m) above the detected middle of the bar (below if negative) for this run; Refused beyond
    grasp_offset_limit. door_common's grasp height is the bar's middle + VERTICAL_CORR, and the door's top and bottom
    are placed from that grasp height (DOOR_Z_FROM_HANDLE), so both move with the offset."""
    limit = grasp_offset_limit()
    if abs(offset) > limit + 1e-9:
        raise Refused(
            f"grasp_offset_m {offset:+.3f} is off the handle: at most +-{limit:.3f} for a "
            f"{_HANDLE_LENGTH * 100:.1f} cm bar (handle_length_m), keeping the finger pads "
            f"{F.GRASP_END_MARGIN_M * 100:.0f} cm in from its ends."
        )
    lo, hi = _DOOR_Z_FROM_HANDLE
    shared.configure(VERTICAL_CORR=F.VERTICAL_CORR + offset, DOOR_Z_FROM_HANDLE=(lo - offset, hi - offset))
    if offset:
        say(f"grasping {abs(offset) * 100:.1f} cm {'above' if offset > 0 else 'below'} the middle of the bar")


def _facing_fridge(df, y_up):
    """Approach along the closed door's inward normal (into the fridge), tool y up (+1, upright -- as the grasp) or
    down (-1)."""
    fwd = -np.r_[df.n, 0.0]
    y = np.array([0.0, 0.0, y_up])
    return R.from_matrix(np.column_stack([np.cross(y, fwd), y, fwd])).as_quat()


def release_around_points(door, p_from, quat_from, edge_clear, y_round):
    """Keypoints [(pos, quat, label)] from the hand after the back-off (`p_from`, `quat_from`) back round the open door
    to its inner side: along the door to `edge_clear` past its free edge (outside), round it turning to face the fridge
    (tool y up/down `y_round`: the hand may be either way up on the way), then in beside the door -- always ending
    UPRIGHT (tool y up, as the grasp), rolling on that last leg if it went round the edge upside down."""
    df = DoorFrame(door, door["open_sign"])
    deg = float(door["door_open_deg"])
    z = float(p_from[2])
    d = df._rot(deg).T @ (np.asarray(p_from[:2]) - df.hinge[:2])
    v_now = float(d @ df.n) - df.mid
    u_past = df.length + edge_clear
    return [
        (np.asarray(p_from[:3], float), np.asarray(quat_from, float), "after the back-off"),
        (
            df.pos(u_past, max(v_now, F.AROUND_OUT_M), deg, z),
            np.asarray(quat_from, float),
            f"back past the free edge (+{edge_clear * 100:.0f} cm)",
        ),
        (df.pos(u_past, -F.AROUND_IN_M, deg, z), _facing_fridge(df, y_round), "round the edge, facing the fridge"),
        (
            df.pos(F.AROUND_END_M[0], -F.AROUND_END_M[1], deg, z),
            _facing_fridge(df, 1.0),
            "beside the door, inner side, upright",
        ),
    ]


def plan_release_around(scene, rb, door, ee, q):
    """From the hand on the handle at the end of the full pull (`ee` [pos, quat], joints `q`): the release's back-off
    (planned exactly as release_and_back_off plans it) and then the route back round the door
    (release_around_points), every leg checked against the door model at its current angle. Returns (keypoints, joint
    legs, message) or (None, None, message). Says nothing."""
    ee = np.asarray(ee, float)
    approach = R.from_quat(ee[3:7]).as_matrix()[:, 2]
    with quiet():
        back = plan_straight_line(scene, rb, ee[:3], ee[3:7], -approach, F.BACK_OFF_M, q, "back-off")
    if back is None:
        return None, None, f"the {F.BACK_OFF_M * 100:.0f} cm back-off off the handle fails a gate"
    p_bo = ee[:3] - approach * F.BACK_OFF_M
    df = DoorFrame(door, door["open_sign"])
    bodies = df.bodies(scene, rb, float(door["door_open_deg"]))
    try:
        for edge_clear in F.AROUND_EDGE_CLEAR_M:
            for y_round in (1.0, -1.0):
                pts = release_around_points(door, p_bo, ee[3:7], edge_clear, y_round)
                with quiet():
                    legs, _ = _chain(scene, rb, [(pp, qq) for pp, qq, _ in pts], back[-1], bodies, "around")
                if legs is not None and trajectory_keep_out(
                    back[-1], [(pp, qq) for pp, qq, _ in pts], [leg for _, leg in legs]
                ):
                    legs = None  # the arm would refuse to send it
                if legs is not None:
                    way_up = "upright" if y_round > 0 else "upside down, rolling upright after"
                    return (
                        pts,
                        [leg for _, leg in legs],
                        f"release + back-off + round the door OK ({edge_clear * 100:.0f} cm past its free edge; round "
                        f"it {way_up}; ends upright)",
                    )
    finally:
        drop(rb, bodies)
    return None, None, "the route back round the door fails a gate"


class ReleaseAround(AfterSwing):
    """Plans the release, back-off and route round the door with the swing; `release_around` runs them."""

    def check(self, scene, rb, geom, cand, explain):
        pts, _, msg = plan_release_around(scene, rb, door_after(geom, cand), cand["wps"][-1], cand["arc"][-1])
        return pts is not None, msg


def release_around(ai, execute):
    """After a full pull: open the gripper, back straight out of the handle, then back round the door to its inner side.
    All of it is planned from where the arm really is before letting go; if any of it fails a check, nothing moves and
    the hand stays on the handle."""
    say("\n===== release, back off, round the door =====")
    if not execute:
        say("  (planned above from the planned swing end -- see the chosen tilt's line; re-planned from the real one)")
        return True
    st = ai.get_state()
    scene, rb = make_sim()
    pts, legs, msg = plan_release_around(scene, rb, json.loads(dc.DOOR_FILE.read_text()), st["ee_pos"], st["position"])
    say(f"  {msg}")
    if pts is None:
        raise Refused(
            f"{msg} -- NOT releasing, the hand stays on the handle. A release_only goal lets go and backs off."
        )
    for pp, _, label in pts[1:]:
        say(f"    {label[:40]:40s} -> {np.round(pp, 3)}")
    if not dc.release_and_back_off(ai, F.BACK_OFF_M, True, park=False):
        return False
    ok, _ = send_legs(ai, [(pp, qq) for pp, qq, _ in pts], legs)
    say(f"ROUND THE DOOR {'DONE' if ok else 'STOPPED SHORT'}")
    return ok


class ReleaseBackOff(AfterSwing):
    """Plans the release and the straight back-off off the handle with the swing (`around` False: nothing else follows
    it). Unplanned, a back-off failing a gate (J6 at its guard) left the hand on the handle with the door open, and a
    release_only goal refusing the same back-off (11-05 replay)."""

    def check(self, scene, rb, geom, cand, explain):
        if P.back_off(scene, rb, cand["wps"][-1], cand["arc"][-1]) is None:
            return False, f"the {F.BACK_OFF_M * 100:.0f} cm back-off off the handle fails a gate"
        return True, "release + back-off OK"


def release_back_off(ai, execute):
    """After a full pull, with `around` False: open the gripper and back straight out of the handle BACK_OFF_M, and
    nothing else (no round the door). For testing the grasp + pull on its own. The hand ends just off the handle; the
    door is left wherever the pull put it."""
    say("\n===== release, back off the handle (no round the door) =====")
    if not execute:
        say("  (planned above with the swing; re-planned from the real end before letting go)")
        return True
    return dc.release_and_back_off(ai, F.BACK_OFF_M, True, park=False)


def save_planned_door(push, after=None):
    """Write DOOR_FILE as the open just planned (shared.LAST_PLAN) would leave it: the closed door, the hinge, the door
    model, and the door open to PUSH_OPEN_TO_DEG (`push`; where it stands after the wheelchair move the push-open
    planned, `after`) or the pull's swing, the way the planned swing opens it. The same values an executed open writes
    as it goes, except door_open_deg / open_sign, which come from the plan here and from the arm's real end there.
    Replaces the file (nothing from an older door stays)."""
    plan = shared.LAST_PLAN
    door = door_after(plan["geom"], plan["plan"])
    door.update(hinge_pivot=plan["geom"]["hinge"], detected_edge=plan["geom"]["hinge_seen"])
    if push:
        door["door_open_deg"] = float(F.PUSH_OPEN_TO_DEG)
        if after is not None and after.chair_move:
            door = P.shifted(door, after.chair_move, after.chair_dir)
    dc.DOOR_FILE.unlink(missing_ok=True)
    dc.save_door_geometry(**door, saved_by="dry-run open (save_door): the door as the planned open would leave it")
    say(
        f"door file {dc.DOOR_FILE}: saved as the planned open would leave it -- door open {door['door_open_deg']:.0f} "
        f"deg (sense {int(door['open_sign']):+d}), hinge {np.round(door['hinge'], 3)}. Open the door to that by hand "
        "before testing the close."
    )


def run(
    ai,
    execute,
    swing_deg=None,
    push=True,
    release_only=False,
    speed=1.0,
    log_args=None,
    grasp_offset=0.0,
    save_door=False,
    around=True,
    chair_moved=False,
    chair_back_dir=F.CHAIR_BACK_DIR,
):
    """The open as the node runs it, logged to RUNS_DIR/<time>_open. Exactly `swing_deg` (default PUSH_OPEN_PULL_DEG, or
    SWING_DEG without `push`), then what follows. `grasp_offset`: grasp this far (m) above the middle of the bar
    (set_grasp_offset). `release_only`: let go and back off the handle, nothing else (when a run stopped holding it).
    `save_door`: a dry run that plans OK also writes DOOR_FILE as the planned open would leave it (save_planned_door),
    so the close can be tested without an executed open; an executed open writes it anyway. `around` (the full-pull
    path only, `push` false): False tests the pull alone -- after the swing just let go and back off the handle, no
    round-the-door: the plan is gated by the back-off only (ReleaseBackOff). With `push`, the open may stop after the
    pull and the back-off, asking for a wheelchair move (push_open.NEXT_STEP); `chair_moved` then does the rest: the
    push, from where the hand is (push_after_chair_move). `chair_back_dir`: which way moving the wheelchair back moves
    the fridge in the arm's frame.
    Returns True when it finished (or the dry run passed); raises Refused."""
    P.NEXT_STEP = ""

    def task():
        if release_only:
            return dc.release_and_back_off(ai, F.BACK_OFF_M, execute, park=False)
        if chair_moved:
            if not push:
                raise Refused("chair_moved is for the push-open's push, after the open asked for a wheelchair move.")
            return push_after_chair_move(ai, execute, chair_back_dir)
        set_grasp_offset(grasp_offset)
        swing = swing_deg or (F.PUSH_OPEN_PULL_DEG if push else F.SWING_DEG)
        max_pull = F.PUSH_OPEN_TO_DEG - F.PUSH_OPEN_MIN_PUSH_DEG
        if push and swing > max_pull:  # push_open would refuse after the pull, holding the handle
            raise Refused(
                f"swing_deg {swing} leaves the push-open too little to push (it pushes on to {F.PUSH_OPEN_TO_DEG}): at "
                f"most {max_pull} with push_open, or push_open false for a full pull."
            )
        if execute and dc.DOOR_FILE.exists():
            # this open moves the door: a push an earlier open left waiting for a wheelchair move no longer fits it
            # (the door file is merged, not replaced, so chair_moved would push a door that isn't there)
            dc.save_door_geometry(push_after_chair_move=None)
        say("\n===== OPEN: grasp + swing =====")
        # what follows the swing, planned with it and gating it: push-open, round the door, or (around False) nothing
        after = PushOpen(chair_back_dir) if push else (ReleaseAround() if around else ReleaseBackOff())
        shared.LAST_PLAN = None  # this run's plan, not an earlier one's
        if not shared.open_task(ai, execute, swing, swing, after_swing=after):
            say("swing did not finish -- stopping (still holding the handle).")
            return False
        if push:
            ok = push_open(ai, execute, after)
        elif around:
            ok = release_around(ai, execute)
        else:
            ok = release_back_off(ai, execute)
        if ok and save_door and not execute:
            save_planned_door(push, after if push else None)
        return ok

    ok = logged_run("open", ai, log_args or {}, speed, task)
    say(f"\nOPEN {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
    return ok
