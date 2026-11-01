"""Open the microwave on `rchi-cpu-5`: detect the handle, grasp it, swing the door open, let
go and push it further open -- one run.

    python3 -m microwave.open_door             # dry run
    python3 -m microwave.open_door --execute   # grasp, 50-deg swing, push-open
    python3 -m microwave.open_door --no-push-open --execute   # stop after the swing
    python3 -m microwave.open_door --release --execute

1. Detect (`perception.HandleDetector`): YOLO box around the microwave, depth -> door plane, the
   vertical cluster sticking out of it = handle, the door edge farthest from it = hinge. Looks
   until two agree within 3 cm (handle and hinge).
2. Plan everything before any motion: the grasp (squared to the door face: cuRobo from the view pose to a
   line-up point 25 cm out (or less) -- only that goal is checked, cuRobo's path is not previewed -- then
   straight in along the door normal), and the largest swing (75 down to 45 deg) whose whole arc passes the
   sim gates (IK, reach, joint jump, J4/J6 guards). cuRobo can't follow a line or an arc, so the straight-in
   leg and the swing stay on joint paths planned and checked here.
3. Execute: cuRobo to the line-up point, straight in, close, one blended swing. The hand stays on the
   handle at the end.

4. Push-open (default; `--no-push-open` to stop after the swing): let go of the handle and replay the
   user's push-open demo relative to the door (`demo_replay.py`, demos/push_open_2026-10-07.door.json):
   up, over to the right of the microwave, down, into the opening, push it further open. The swing
   is then PUSH_OPEN_SWING_DEG (the demo starts from a ~50 deg swing). Planned and sim-gated from
   where the swing actually ends, before anything moves -- so a dry run only lists its keypoints.

`--release` (separate run, after the swing; instead of the push-open): open the gripper, back straight off the
handle, go toward the base to clear the door's free edge, joint move to `PARK_JOINTS` (constants).

Under the engine: the node's `open` action (node.py -> `run`, which logs the run with RunLog). On the bench:
the commands above (rammp_arm.RammpArm). Refusals raise door_common.common.Refused: the arm holds where it is.

The shared open (detect, plan, grasp, swing) is door_common.open_door; here is what is the microwave's: the view pose,
YOLO or boxer finding the microwave, the push-open that follows the swing.
"""

import argparse
import contextlib
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from door_common import common as dc
from door_common import open_door as shared
from door_common.common import (
    DoorFrame,
    Refused,
    RunLog,
    check_curobo_goal,
    quiet,
    release_and_back_off,
    say,
)
from door_common.open_door import BACK_OFF_M, SWING_MAX_TRY_DEG, SWING_MIN_DEG
from scipy.spatial.transform import Rotation as R

from . import constants
from .constants import (
    BODY_DEPTH,
    GAINS_RIGID,
    GRASP_X_ADJUST,
    GRIP_EXT,
    HINGE_BEHIND_FACE,
    HINGE_PAST_EDGE,
    PUSH_OPEN_SPEED_SCALE,
    SWING_SPEED_SCALE,
    VERTICAL_CORR,
    YOLO_MODEL,
)

# --- view pose: before detecting, the wrist camera goes square in front of the microwave (placed from boxer's box, or
# without boxer from a first detection) ---
VIEW_DISTS = (0.35, 0.30, 0.40, 0.45)  # camera this far out from the box's front face (horizontally) ...
# 10-30: was 0.50 first -- at 50 cm a 1-2 cm handle bar is ~1 pixel of the OAK's 320x200 near-range depth and comes and
# goes between frames ("nothing protruding" / "no vertical bar" retries, and its height wandered)
VIEW_HEIGHTS = (0.15, 0.20, 0.10, 0.25, 0.05)  # ... and this far above the box's middle, looking down at the face's
# centre; the first (height, distance) that passes. Level at mid-height folded the elbow past its limit (10-30: J4
# ~210 deg, door 72 cm out); the hand-placed view that worked (10-28) had the camera ~20 cm above the door, looking down
VIEW_SEED_DEG = (0.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0)  # IK seed: Kinova's Home, the elbow the usual way
# (a start with J1 ~180 / J4 > 0 used to carry its flipped elbow into every plan, 10-29)
BOX_WAIT_S = 10.0  # wait this long for boxer's first microwave box

PUSH_OPEN = Path(__file__).resolve().parent / "demos" / "push_open_2026-10-07.door.json"
PUSH_OPEN_SWING_DEG = 50  # the push-open demo starts from a ~50 deg swing
PUSH_OPEN_TRUST_FROM = 3  # the 50-deg door model contradicts the user's real push-open poses from leg 3 on
PRE_PARK_X = 0.216

HANDLE_DETECTOR_MODEL = YOLO_MODEL  # the node sets its yolo_model parameter here
SCENE_CAMERA = None  # without boxer, the camera whose view of the door places the view pose (set_scene_camera)
SCENE_WAIT_S = 10.0  # wait this long for the scene camera's first frame


def set_scene_camera(cam):
    """The scene camera (a perception.WristCamera on the scene camera's topics) for this action, or None."""
    global SCENE_CAMERA
    SCENE_CAMERA = cam


def _make_detector():
    from .perception import HandleDetector  # here, not at import: open3d, which the planning tests don't have

    return HandleDetector(HANDLE_DETECTOR_MODEL)


def configure(data_dir):
    """Everything door_common needs from the microwave: call once at startup."""
    constants.configure(data_dir)
    shared.configure(
        MAKE_DETECTOR=_make_detector,
        FIND_HINGE=shared.hinge_detected(HINGE_BEHIND_FACE, HINGE_PAST_EDGE),
        GRIP_EXT=GRIP_EXT,
        VERTICAL_CORR=VERTICAL_CORR,
        GRASP_X_ADJUST=GRASP_X_ADJUST,
        GRASPED_GAINS=GAINS_RIGID,
        SWING_SPEED_SCALE=SWING_SPEED_SCALE,
    )


def _push_open_check(scene, rb, n, g_p, h, door_z, hinge_seen, deg, end_wp, q_end, loud=False):
    """Plan the push-open (demo_replay) from a planned swing's end -- pose `end_wp`, joints `q_end` -- against the
    door as the open will have saved it then, so a push-open that can't run refuses BEFORE the grasp, not after the
    swing (10-30). `loud`: print the plan (every leg's steps and the failing check). Returns (ok, message). The real
    run re-plans it from where the swing really ends."""
    from . import demo_replay

    if door_z is None:
        return False, "push-open: no door height detected (door_z) to place it"
    door = dict(closed_normal=list(n), closed_grasp_pos=list(g_p), closed_handle=list(h), door_z=list(door_z))
    door.update(hinge=list(hinge_seen))  # as saved: the demo uses the detected edge
    # ... and the opening angle as the run will record it after the swing (record_door_angle: the hand's angle about
    # that edge), not the swing's own `deg` about the pivot -- 10-30: 50 deg about the pivot recorded 54.2, and the
    # door model 4 deg further open failed the real check after this one had passed
    a = DoorFrame(door, 1.0).angle_of(end_wp[:3])
    door.update(door_open_deg=abs(a), open_sign=float(np.sign(a) or 1.0))
    enc = json.loads(PUSH_OPEN.read_text())
    pts, grippers = demo_replay.decode(enc, door), [k["gripper"] for k in enc["keypoints"]]
    off = float(np.linalg.norm(np.asarray(end_wp[:3]) - pts[0][0]))
    if off > demo_replay.START_TOL:
        return False, f"push-open: the swing ends {off * 100:.1f} cm from the demo's start"
    st = {"position": list(q_end), "ee_pos": list(end_wp)}
    lift_off = grippers[0] > 0.2 and grippers[1] < 0.2
    if loud:
        for k, (p_, _) in zip(enc["keypoints"], pts, strict=False):
            say(f"    {k['label'][:30]:30s} -> {np.round(p_, 3)}")
    with contextlib.nullcontext() if loud else quiet():
        legs = demo_replay.plan(scene, rb, door, st, pts[1:], PUSH_OPEN_TRUST_FROM, lift_off, short_last=True)
    return legs is not None, ("push-open OK" if legs is not None else "push-open: a leg fails a gate")


class PushOpen(shared.AfterSwing):
    """Plans the push-open with the swing; `push_open` runs it. The real run re-plans it from where the swing ends."""

    def check(self, scene, rb, geom, cand, explain):
        g, d, end_wp, q_end = cand["grasp"], cand["swing"], cand["wps"][-1], cand["arc"][-1]
        n, h = geom["n"], cand["h"]
        args = (scene, rb, n, g.position, h, geom["door_z"], geom["hinge_seen"], d, end_wp, q_end)
        ok, m = _push_open_check(*args)
        if not ok and explain:  # the best candidate's failing push-open, out loud
            say(f"  tilt {cand['pitch']} deg, swing {d} deg: the push-open fails -- why, its plan in full:")
            _push_open_check(*args, loud=True)
        return ok, m


def _pose_T(pos, quat):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.from_quat(quat).as_matrix(), pos
    return T


def _box_face(cam):
    """The front of boxer's microwave box: of its four side faces, the one whose outward normal points most toward
    the arm base. Returns (face centre at the box's mid height, outward normal); raises Refused without a box."""
    box, t0 = None, time.time()
    while box is None and time.time() - t0 < BOX_WAIT_S:
        with quiet():
            box = cam.appliance_box()
        if box is None:
            time.sleep(0.5)
    if box is None:
        cam.appliance_box()  # says why
        raise Refused(f"No microwave box from boxer ({cam.box_topic}) in {BOX_WAIT_S:g} s -- is it running?")
    c, Rb, size = box["center"], box["R"], box["size"]
    to_base = -c[:2] / np.linalg.norm(c[:2])
    n, half = max(
        ((s * Rb[:, i], size[i] / 2) for i in (0, 1) for s in (1.0, -1.0)), key=lambda t: float(t[0][:2] @ to_base)
    )
    n = np.r_[n[:2], 0.0] / np.linalg.norm(n[:2])
    face = c + n * half
    say(
        f"boxer microwave {box['score']:.2f}: center {np.round(c, 3)} size {np.round(size, 3)}; front face "
        f"{np.round(face, 3)} facing {np.round(n, 3)}"
    )
    return face, n


def _door_box(door, score):
    """The microwave as a box behind the door the scene camera found -- `door` (door_face's) its front face, BODY_DEPTH
    deep -- in boxer's box form, so the wrist detection crops its points to it the same way (no YOLO)."""
    n = np.asarray(door["normal"], dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    lo, hi = door["door_z"]
    face = np.r_[door["door_mid"][:2], (lo + hi) / 2]
    return {
        "center": face - n * BODY_DEPTH / 2,
        "R": np.column_stack([n, np.cross(up, n), up]),
        "size": np.array([BODY_DEPTH, door["width"], hi - lo]),
        "score": score,
        "source": "scene camera",
    }


def _detected_face():
    """Without boxer: the scene camera's view of the door places the view pose -- the scene segmenter's microwave box
    (RF-DETR, not YOLO), the door plane in it (no handle: it is too small from there), the face's centre (the middle of
    its width and of its height) and its normal. That door, as a box, is also what the wrist detection crops to.
    Returns (face centre, outward normal); raises Refused if it finds no door."""
    cam = SCENE_CAMERA
    if cam is None:
        raise Refused("No scene camera to place the view pose -- set box_topic, or send view false.")
    if not cam.wait_for_frames(SCENE_WAIT_S):
        raise Refused(f"No scene camera frames on {', '.join(cam.topics)} in {SCENE_WAIT_S:g} s -- is it up?")
    detector = _make_detector()
    for attempt in range(1, shared.MAX_EMPTY_RETRIES + 1):
        f = cam.grab()
        door = None
        if f["box2d"] is not None:
            detector.region_2d = f["box2d"]["xyxy"]
            door = detector.door_face(f["rgb"], f["K"], f["depth_mm"], f["base_T_cam"])
        shared._save_frame(f, "scene", attempt, door)
        if door is not None:
            break
        say(f"  no door in the scene camera (retry {attempt}/{shared.MAX_EMPTY_RETRIES})")
        time.sleep(0.5)
    else:
        raise Refused(
            f"No door in the scene camera after {shared.MAX_EMPTY_RETRIES} tries -- stopping before any motion."
        )
    face = np.r_[door["door_mid"][:2], float(np.mean(door["door_z"]))]
    say(
        f"door face {np.round(face, 3)} facing {np.round(door['normal'], 3)}, {door['width'] * 100:.0f} cm wide "
        f"(scene camera, {cam.box_class} {f['box2d']['score']:.2f})"
    )
    dc.camera().fixed_box = _door_box(door, f["box2d"]["score"])  # the wrist detection crops to it, not YOLO
    return face, np.asarray(door["normal"], dtype=float)


def plan_view(ai, scene, rb):
    """The view pose square in front of the microwave: the wrist camera VIEW_DISTS out from its front face, VIEW_HEIGHTS
    above the face's middle, looking at the face's centre. The face comes from boxer's box (box_topic) or, without
    boxer, from a first detection made from where the arm is. The hand pose comes from where the camera sits on it now
    (TF), its joints from IK seeded at VIEW_SEED_DEG and checked like any cuRobo goal. Returns the goal joints (to
    go_to_joint_config, so the arm arrives on that branch); raises Refused."""
    cam = dc.camera()
    face, n = _box_face(cam) if getattr(cam, "box_topic", "") else _detected_face()
    f = cam.grab()
    if f["base_T_cam"] is None:
        raise Refused("No arm base <- camera transform -- can't place the camera for the view pose.")
    st = ai.get_state()
    tool_T_cam = np.linalg.inv(_pose_T(st["ee_pos"][:3], st["ee_pos"][3:7])) @ f["base_T_cam"]
    tried = []
    for height in VIEW_HEIGHTS:
        for dist in VIEW_DISTS:
            # optical frame: z at the face's centre (tilted down), y down in the image, x right
            pos = face + n * dist + np.array([0.0, 0.0, height])
            z = (face - pos) / np.linalg.norm(face - pos)
            y = np.array([0.0, 0.0, -1.0]) - z * -z[2]
            y /= np.linalg.norm(y)
            T_cam = np.eye(4)
            T_cam[:3, :3], T_cam[:3, 3] = np.column_stack([np.cross(y, z), y, z]), pos
            T_tool = T_cam @ np.linalg.inv(tool_T_cam)
            label = f"to view pose {dist * 100:.0f} cm out, {height * 100:.0f} cm up"
            with quiet():
                q = check_curobo_goal(
                    scene,
                    rb,
                    label,
                    pos=T_tool[:3, 3],
                    quat=R.from_matrix(T_tool[:3, :3]).as_quat(),
                    seed=np.radians(VIEW_SEED_DEG),
                )
            if q is not None:
                say(
                    f"view pose ({label}, looking {np.degrees(np.arcsin(-z[2])):.0f} deg down): camera "
                    f"{np.round(pos, 3)}, joints {np.round(np.degrees(q), 1)}"
                )
                return q
            tried.append((label, T_tool))
    label, T_tool = tried[0]  # say why the preferred one failed
    check_curobo_goal(
        scene,
        rb,
        label,
        pos=T_tool[:3, 3],
        quat=R.from_matrix(T_tool[:3, :3]).as_quat(),
        seed=np.radians(VIEW_SEED_DEG),
    )
    raise Refused(
        f"No view pose ({len(tried)} tried: {VIEW_DISTS} m out x {VIEW_HEIGHTS} m up) passes the checks -- "
        "move the microwave or the robot."
    )


def _view_step(view, view_only, ai, scene, rb, execute):
    """(`view`) go to the view pose, so the handle is detected from the same place every run. Returns True when the
    open ends here."""
    if not view:
        return False
    q_view = plan_view(ai, scene, rb)
    if not execute:
        say(
            "\nDRY RUN -- would move to the view pose and detect from there; nothing commanded. To dry-run the "
            "rest: a view_only goal (moves), then a dry run with view false."
        )
        return True
    if not ai.go_to_joint_config(q_view, "to view pose"):
        raise Refused("Did not reach the view pose -- stopping before detection.")
    time.sleep(1.0)  # let the camera frames catch up with the arm
    if view_only:
        say("\nAT THE VIEW POSE (view_only) -- stopping here.")
        return True
    return False


def open_task(
    ai,
    execute,
    max_swing=SWING_MAX_TRY_DEG,
    min_swing=SWING_MIN_DEG,
    view=True,
    view_only=False,
    push=False,
    on_grasp=None,
):
    """(`view`) go to the view pose, then detect, plan, grasp, swing. `view_only`: stop at the view pose.
    `push`: the push-open follows, so each candidate's push-open is planned too before any motion. `on_grasp()`: called
    once the gripper has closed on the handle.
    Returns True when the swing finished (or the dry run passed); raises Refused on any refusal."""
    return shared.open_task(
        ai,
        execute,
        max_swing,
        min_swing,
        after_swing=PushOpen() if push else None,
        before_detect=partial(_view_step, view, view_only),
        on_grasp=on_grasp,
    )


def push_open(ai, execute):
    """After the swing: release the handle and push the door further open (the user's 10-07 demo, door-relative)."""
    from . import demo_replay

    say("\n===== release + push further open =====")
    if not execute:
        say("  (planned above from the planned swing end -- see the chosen tilt's line; re-planned from the real one)")
        return True
    return demo_replay.run(
        PUSH_OPEN,
        True,
        PUSH_OPEN_TRUST_FROM,
        ai=ai,
        impedance=shared.grasped_impedance(),
        speed_scale=PUSH_OPEN_SPEED_SCALE,
    )


def run(ai, execute, swing_deg=None, push=True, speed=1.0, log_args=None, view=True, view_only=False, on_hold=None):
    """The open task as the node runs it (was door_task.py `open`): (`view`) to the view pose, grasp + swing, then
    (`push`) let go and replay the push-open demo, all logged to RUNS_DIR/<time>_open (RunLog). `speed` (0..1]
    scales every arm motion for this run only. `view_only`: stop at the view pose. `on_hold(True)` once the gripper
    closes on the handle, `on_hold(False)` once the push-open has let go of it. Returns True when it finished
    (or the dry run passed); raises Refused."""
    import rammp_arm

    saved = (rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE)
    rammp_arm.CART_SPEED *= speed
    rammp_arm.JOINT_SPEED *= speed
    rammp_arm.GO_TO_SPEED_SCALE = max(0.01, rammp_arm.GO_TO_SPEED_SCALE * speed)
    log = RunLog("open", log_args or {}, ai)
    ok = False
    try:
        log.frame("run_start")
        swing = swing_deg or (PUSH_OPEN_SWING_DEG if push else SWING_MAX_TRY_DEG)
        say("\n===== OPEN: grasp + swing =====")
        on_grasp = partial(on_hold, True) if on_hold else None
        ok = open_task(
            ai, execute, swing, view=view, view_only=view_only, push=push and not view_only, on_grasp=on_grasp
        )
        if not ok:
            say("swing did not finish -- stopping (still holding the handle).")
        elif push and not view_only:
            ok = push_open(ai, execute)
            if ok and execute and on_hold:
                on_hold(False)
        log.event("open", ok=ok)
        say(f"\nOPEN {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
        return ok
    except BaseException as e:
        log.event("stopped", error=repr(e))
        raise
    finally:
        log.close(ok)
        rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE = saved


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--execute", action="store_true", help="command the arm; omit for a dry run")
    a.add_argument(
        "--swing-deg",
        type=int,
        default=None,
        help=f"largest swing to try (deg; default {PUSH_OPEN_SWING_DEG} with the push-open, where the demo "
        f"starts, else {SWING_MAX_TRY_DEG})",
    )
    a.add_argument("--no-push-open", action="store_true", help="stop after the swing, still holding the handle")
    a.add_argument(
        "--release",
        action="store_true",
        help="after a --no-push-open swing: open the gripper, back off the handle, park",
    )
    args = a.parse_args()
    swing = args.swing_deg or (SWING_MAX_TRY_DEG if args.no_push_open else PUSH_OPEN_SWING_DEG)
    min_swing = SWING_MIN_DEG
    import rammp_arm

    from .perception import WristCamera

    configure(constants.BENCH_DATA_DIR)
    ai = rammp_arm.RammpArm()
    dc.set_camera(
        WristCamera(
            ai.node,
            "/wrist_camera/color/image_raw",
            "/wrist_camera/color/camera_info",
            "/wrist_camera/depth/image_rect_raw",
            rammp_arm.FRAME,
        )
    )
    set_scene_camera(
        WristCamera(
            ai.node,
            "/scene_camera/color/image_raw",
            "/scene_camera/color/camera_info",
            "/scene_camera/depth/image_rect_raw",
            rammp_arm.FRAME,
            box_class="microwave",
            box2d_topic="/perception/object_detections",
        )
    )
    try:
        _run(ai, args, swing, min_swing)
    except Refused as e:
        sys.exit(str(e))
    finally:
        ai.close()


def _run(ai, args, swing, min_swing):
    if args.release:
        release_and_back_off(ai, BACK_OFF_M, args.execute, pre_park_x=PRE_PARK_X)
    elif open_task(ai, args.execute, swing, min_swing) and not args.no_push_open:
        push_open(ai, args.execute)


if __name__ == "__main__":
    main()
