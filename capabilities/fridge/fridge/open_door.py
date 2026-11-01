"""Open the fridge: detect the handle, grasp it, swing the door open, let go, go back round the door -- one run.

1. Detect, plan, grasp, swing: door_common.open_door, with this fridge's values (`configure`). The handle comes from the
   whole image (perception.HandleDetector); the hinge is measured from the grasp, not detected. Exactly the asked swing
   or the run refuses, and what follows the swing is planned with it, before anything moves.
2. By default (`ReleaseAround`, a 90-deg pull): let go, back off the handle, then back along the open door past its free
   edge, round it and in beside the door, facing the fridge -- so the arm ends on the door's inner side. It always ends
   upright, as at the grasp; it may go round the edge upside down and roll upright on the way in. No park.
3. The option (`push_open.PushOpen`): a shorter pull, then let go and push the door further open. Not the default: on
   its one hardware run (10-24) it finished its trajectory but the door swung shut.

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
from .motion import AfterSwing, door_after, logged_run, send_legs
from .push_open import PushOpen, push_open


def _make_detector():
    from .perception import HandleDetector  # here, not at import: open3d, which the planning tests don't have

    return HandleDetector()


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


def configure(data_dir, mount_tilt_deg=0, **measured):
    """Everything door_common needs from the fridge: call once at startup. `mount_tilt_deg`: a key of constants.MOUNTS.
    `measured`: constants.MEASURED's names."""
    m = {**F.MEASURED, **measured}
    F.configure(data_dir, mount_tilt_deg)
    mount = F.MOUNTS[mount_tilt_deg]
    shared.configure(
        MAKE_DETECTOR=_make_detector,
        # the measured offset from the closed-door grasp point, in the door's frame: into it, along it, up. Its last
        # 09-20 runs swung ~100 deg about this hinge. A wrong one drags the door or the fridge during the swing
        FIND_HINGE=shared.hinge_measured(
            np.array([m["handle_centre_to_face_m"] + m["pin_behind_face_m"], m["hinge_along_door_m"], 0.0])
        ),
        GRIP_EXT=F.GRIP_EXT,
        VERTICAL_CORR=F.VERTICAL_CORR,
        GRASP_X_ADJUST=F.GRASP_X_ADJUST,
        GRASP_FORCE=F.GRASP_FORCE,
        PLAUSIBLE_Y=F.PLAUSIBLE_Y,
        PLAUSIBLE_Z=F.PLAUSIBLE_Z,
        MAX_REACH=mount["max_reach"],
        SWING_MAX_REACH=mount["swing_max_reach"],
        PLACEMENT_NOTE=partial(placement_advice, level=mount_tilt_deg == 0),
        DOOR_MODEL=dict(
            face_behind_grasp=m["handle_centre_to_face_m"],
            door_t=m["door_depth_m"],
            door_past_handle=m["door_width_m"] - m["hinge_along_door_m"],
            body_depth=m["body_depth_m"],
        ),
        DOOR_Z_FROM_HANDLE=(-m["door_below_handle_m"], m["door_above_handle_m"]),  # the camera sees only part of it
        KEEP_ARM_IN_FRAMES=True,
    )
    dc.enable_keep_out(m["chair_keep_out_behind_m"])  # every plan and every trajectory sent is checked


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


def run(ai, execute, swing_deg=None, push=False, release_only=False, speed=1.0, log_args=None):
    """The open as the node runs it, logged to RUNS_DIR/<time>_open. Exactly `swing_deg` (default SWING_DEG, or
    PUSH_OPEN_PULL_DEG with `push`), then what follows. `release_only`: let go and back off the handle, nothing else
    (when a run stopped holding it). Returns True when it finished (or the dry run passed); raises Refused."""

    def task():
        if release_only:
            return dc.release_and_back_off(ai, F.BACK_OFF_M, execute, park=False)
        swing = swing_deg or (F.PUSH_OPEN_PULL_DEG if push else F.SWING_DEG)
        say("\n===== OPEN: grasp + swing =====")
        after = PushOpen() if push else ReleaseAround()
        if not shared.open_task(ai, execute, swing, swing, after_swing=after):
            say("swing did not finish -- stopping (still holding the handle).")
            return False
        return push_open(ai, execute) if push else release_around(ai, execute)

    ok = logged_run("open", ai, log_args or {}, speed, task)
    say(f"\nOPEN {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
    return ok
