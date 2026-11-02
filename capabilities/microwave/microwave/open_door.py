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

4. Push-open (default; `--no-push-open` to stop after the swing), from the door the swing left: open the gripper,
   then three cuRobo moves, the hand pointing down (PUSH_TILT_TRY_DEG off vertical, the least that plans; of the two
   such hands the one nearest the grip, no wrist flip): up (the fingertips PUSH_CLEAR_TRY_M over the door top, the
   most that plans) and back (PUSH_BACK_M) from the handle; over to above the inside of the door; down until the
   tool is at the door top, PUSH_BEHIND_DOOR_M behind its inner face at the handle. Then push straight toward the
   hinge until level with it (its y), opening the door to ~90 deg. cuRobo is trusted with its moves (11-02) -- not
   sim-gated, and its world has no door: it is asked for them (plan_to_pose, nothing moves) from the planned swing's
   end before the grasp, and again from the real one before letting go. The push is a straight line, the hand held
   as it is (move_cartesian), not cuRobo.

`--release` (separate run, after the swing; instead of the push-open): open the gripper, back straight off the
handle, go toward the base to clear the door's free edge, joint move to `PARK_JOINTS` (constants).

Under the engine: the node's `open` action (node.py -> `run`, which logs the run with RunLog). On the bench:
the commands above (rammp_arm.RammpArm). Refusals raise door_common.common.Refused: the arm holds where it is.

The shared open (detect, plan, grasp, swing) is door_common.open_door; here is what is the microwave's: the view pose,
YOLO or boxer finding the microwave, the push-open that follows the swing.
"""

import argparse
import json
import sys
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import rammp_arm
from capability_engine.arm import IMPEDANCE as IMPEDANCE_MODE
from capability_engine.arm_messages import HOLD_FIXED, HOLD_NONE
from door_common import common as dc
from door_common import open_door as shared
from door_common.common import (
    FINGERTIP_PAST_TOOL,
    DoorFrame,
    Refused,
    RunLog,
    check_curobo_goal,
    quiet,
    release_and_back_off,
    save_door_geometry,
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
VIEW_AT_TOL_M = 0.05  # the hand this close to the view pose counts as there: a dry run goes on to detect and plan

# --- push-open (after the swing): let go, up and back, over to the inside of the door, down, push it further open ---
PUSH_OPEN_SWING_DEG = 50  # the swing before the push-open
PUSH_CLEAR_TRY_M = (
    0.05,
    0.0375,
    0.025,
)  # after letting go, cuRobo lifts the hand to cross the door with its fingertips
#   (pointing down) this far over the door top, the first that plans: half of 11-02's 10 cm (it went higher than
#   needed); the door top is a depth-fit estimate ...
PUSH_BACK_M = 0.05  # ... and this far back from the handle, along the open door's outward normal
PUSH_BEHIND_DOOR_M = 0.05  # (~2 in) then the tool goes down this far behind the door's inner face, at the handle
PUSH_TOOL_ABOVE_TOP_M = 0.0  # ... to this height above the door top, pointing down: the fingertips reach
#   FINGERTIP_PAST_TOOL (6 cm) below the top edge
# then push straight left (toward the hinge: further open), the hand's x and height kept, until it is level with the
# detected hinge (the door file's hinge): the door then stands square to the arm, ~90 deg (11-02: 6 and 12 cm along
# the door's normal didn't open it enough)
PUSH_TILT_TRY_DEG = (0, 30, 45, 60)  # the hand points down tilted this far from vertical, leaning away from the arm
#   base, the least that plans: straight down over the inside of the door the wrist runs out (11-02: J6 -104..-114,
#   at its guard, and IK 1-2 cm off on the way there)
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


def _push_poses(door, ee, clear_m, tilt_deg):
    """The push-open's targets from the door file `door` as the swing left it (door_open_deg, open_sign) and the
    hand pose `ee` [x, y, z, qx, qy, qz, qw] at the swing's end: the fingertips crossing `clear_m` over the door top,
    the hand pointing down tilted `tilt_deg` from vertical away from the arm base. SimpleNamespace(df, deg, sign,
    up_back, above, inside, pushed, down (the hand's quat throughout), out (the open door's outward normal: pushing
    along it opens it further), how (the variant))."""
    sign, deg = float(door.get("open_sign", 1.0)), float(door["door_open_deg"])
    df = DoorFrame(door, sign)
    top = float(door["door_z"][1])
    out = df.pos(df.u_handle, 1.0, deg, 0.0) - df.pos(df.u_handle, 0.0, deg, 0.0)
    inside = df.pos(df.u_handle, -(df.t / 2 + PUSH_BEHIND_DOOR_M), deg, top + PUSH_TOOL_ABOVE_TOP_M)
    up_back = np.asarray(ee[:3], dtype=float) + out * PUSH_BACK_M
    up_z = top + clear_m + FINGERTIP_PAST_TOOL  # the tool, with the fingertips (pointing down) clear_m over the top
    up_back[2] = up_z
    above = np.r_[inside[:2], up_z]
    # tool z down, tilted away from the arm base (the fingertips then lean further behind the door, not into it); tool
    # y along the push: a finger pad meets the door
    away = np.r_[inside[:2], 0.0] / np.linalg.norm(inside[:2])
    t = np.radians(tilt_deg)
    z = np.cos(t) * np.array([0.0, 0.0, -1.0]) + np.sin(t) * away
    y = out - z * float(out @ z)
    y /= np.linalg.norm(y)
    # either finger pad can push: of the two hands pointing down (tool y along +-out), the one nearest the hand now,
    # so cuRobo doesn't spin the wrist half a turn on the way up (11-02: it flipped the hand)
    downs = [R.from_matrix(np.column_stack([np.cross(s_ * y, z), s_ * y, z])) for s_ in (1.0, -1.0)]
    now = R.from_quat(np.asarray(ee[3:7], dtype=float))
    down = min(downs, key=lambda r: (r * now.inv()).magnitude()).as_quat()
    return SimpleNamespace(
        df=df,
        deg=deg,
        sign=sign,
        up_back=up_back,
        above=above,
        inside=inside,
        pushed=np.r_[inside[0], df.hinge[1], inside[2]],
        down=down,
        out=out,
        how=f"the hand {tilt_deg} deg off straight down, fingertips {clear_m * 100:.1f} cm over the door top",
    )


def _push_legs(P):
    """The push-open's cuRobo moves: (label, tool position, orientation hold) -- turning down on the first, then the
    hand held as it is. The push itself is not one: it is a straight line, the hand held (push_open)."""
    return (
        ("up and back", P.up_back, HOLD_NONE),
        ("over the door", P.above, HOLD_FIXED),
        ("down inside the door", P.inside, HOLD_FIXED),
    )


def _plan_push_open(ai, door, ee, q, loud=False):
    """Ask cuRobo -- nothing moves -- for the push-open from the hand pose `ee` / joints `q` at the swing's end: each
    of its moves (_push_legs) from where the one before ends, for the first variant that plans all of them -- the hand
    PUSH_TILT_TRY_DEG off straight down, the fingertips PUSH_CLEAR_TRY_M over the door top. Not sim-gated: cuRobo is
    trusted with it (11-02), and its world has no door. Returns (P, [(label, Plan)], None) or (None, None, why)."""
    whys = []
    for tilt_deg, clear_m in ((t, c) for t in PUSH_TILT_TRY_DEG for c in PUSH_CLEAR_TRY_M):
        P = _push_poses(door, ee, clear_m, tilt_deg)
        if float((P.pushed - P.inside)[:2] @ P.out[:2]) <= 0:  # toward the hinge must open the door further
            return None, None, "push-open: the hinge's y is on the door's closing side of the hand -- refusing"
        start, plans = np.asarray(q, dtype=float), []
        for label, pos, hold in _push_legs(P):
            pl = ai.plan_to_pose(pos, P.down, start, hold=hold)
            if not pl.success:
                whys.append(f"{P.how}: cuRobo can't plan '{label}' ({pl.message})")
                if loud:
                    say(f"  push-open, {whys[-1]}")
                break
            plans.append((label, pl))
            if len(pl.q):
                start = pl.q[-1]
        else:
            return P, plans, None
    return None, None, f"push-open: cuRobo plans none of {len(whys)} variants (first: {whys[0]})"


def _door_after_swing(geom, cand):
    """The door file as the open will have saved it after candidate `cand`'s swing (detection `geom`), or None without
    a detected door height."""
    if geom["door_z"] is None:
        return None
    door = dict(
        closed_normal=list(geom["n"]),
        closed_grasp_pos=list(cand["grasp"].position),
        closed_handle=list(cand["h"]),
        door_z=list(geom["door_z"]),
        hinge=list(geom["hinge_seen"]),  # as saved: the detected edge
    )
    # the opening as record_door_angle will save it after the swing (the hand's angle about that edge)
    a = DoorFrame(door, 1.0).angle_of(cand["wps"][-1][:3])
    door.update(door_open_deg=abs(a), open_sign=float(np.sign(a) or 1.0))
    return door


def _curobo_path(plan_fn, label):
    """cuRobo's own plan for one of the open's cuRobo moves (self.arm's plan_to_*: nothing moves), as joint lists, or
    None (says why)."""
    pl = plan_fn()
    if not pl.success:
        say(f"  cuRobo can't plan {label}: {pl.message}")
        return None
    say(f"  cuRobo planned {label}: {len(pl.q)} samples, {pl.t[-1]:.1f} s")
    return np.asarray(pl.q, dtype=float).tolist()


def _save_plan(ai, run_dir, push):
    """A dry run's plan, for render_plan: <run_dir>/plan.json -- the joints (rad) of the start, the line-up, the
    straight in, the swing, and the door as the open would save it; with self.arm, cuRobo's own paths for the line-up
    and (`push`) the push-open's moves -- asked of the planner, nothing moves."""
    if shared.LAST_PLAN is None:
        return
    q0, plan, geom = shared.LAST_PLAN["q0"], shared.LAST_PLAN["plan"], shared.LAST_PLAN["geom"]
    door = _door_after_swing(geom, plan)
    j = lambda qs: np.asarray(qs, dtype=float).tolist()  # noqa: E731
    out = dict(
        start=j(q0),
        line_up=j(plan["q_pre"]),
        straight_in=j(plan["leg"]),
        swing=j(plan["arc"]),
        swing_deg=plan["swing"],
        tilt_deg=plan["pitch"],
        door=door,
        push_open=None,
        line_up_curobo=None,
    )
    curobo = hasattr(ai, "plan_to_joints")
    if curobo:
        out["line_up_curobo"] = _curobo_path(lambda: ai.plan_to_joints(plan["q_pre"], ai.raw_joints()), "the line-up")
    if push and door is not None and curobo:
        P, plans, why = _plan_push_open(ai, door, list(plan["wps"][-1]), plan["arc"][-1])
        if plans is not None:
            say(f"  cuRobo planned the push-open: {P.how}")
            out["push_open"] = dict(
                how=P.how,
                legs=[dict(label=lb, joints=j(pl.q)) for lb, pl in plans],
                push=dict(to=j(P.pushed), quat=j(P.down)),  # the straight push, from where the last leg ends
            )
        else:
            say(f"  {why}")
    (run_dir / "plan.json").write_text(json.dumps(out))
    say(f"plan saved: {run_dir / 'plan.json'} -- render it: python3 -m microwave.render_plan {run_dir}")


class PushOpen(shared.AfterSwing):
    """Asks cuRobo for the push-open with the swing, against the door as the open will have saved it then, so a
    push-open cuRobo can't plan refuses BEFORE the grasp; `push_open` asks again from where the swing really ends.
    Without cuRobo (no plan_to_pose: the bench's mock) it can't check, and says so."""

    def __init__(self, ai):
        self.ai = ai

    def check(self, scene, rb, geom, cand, explain):
        if not hasattr(self.ai, "plan_to_pose"):
            return True, "push-open: not checked (no cuRobo)"
        door = _door_after_swing(geom, cand)
        if door is None:
            return False, "push-open: no door height detected (door_z) to place it"
        P, _, why = _plan_push_open(self.ai, door, list(cand["wps"][-1]), cand["arc"][-1], loud=explain)
        return why is None, why or f"push-open OK ({P.how})"


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
        raise Refused("No scene camera to place the view pose -- is it configured (scene_*_topic), or set box_topic.")
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
    go_to_joint_config, so the arm arrives on that branch) and the hand's goal position; raises Refused."""
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
                return q, T_tool[:3, 3]
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


def _view_step(view_only, ai, scene, rb, execute):
    """Go to the view pose, so the handle is always detected from the same place. Already there (VIEW_AT_TOL_M), a dry
    run goes on to detect and plan from it; elsewhere it stops, since it can't detect from where the arm isn't.
    Returns True when the open ends here."""
    q_view, p_view = plan_view(ai, scene, rb)
    off = float(np.linalg.norm(np.asarray(ai.get_state()["ee_pos"][:3]) - p_view))
    if off <= VIEW_AT_TOL_M:
        say(f"already at the view pose ({off * 100:.1f} cm) -- detecting from here")
        if view_only:
            say("\nAT THE VIEW POSE (view_only) -- stopping here.")
        return view_only
    if not execute:
        say(
            f"\nDRY RUN -- would move to the view pose ({off * 100:.0f} cm away) and detect from there; nothing "
            "commanded. To dry-run the rest: a view_only goal (moves there), then this dry run again."
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
    view_only=False,
    push=False,
    on_grasp=None,
):
    """Go to the view pose, then detect, plan, grasp, swing. `view_only`: stop at the view pose.
    `push`: the push-open follows, so each candidate's push-open is planned too before any motion. `on_grasp()`: called
    once the gripper has closed on the handle.
    Returns True when the swing finished (or the dry run passed); raises Refused on any refusal."""
    return shared.open_task(
        ai,
        execute,
        max_swing,
        min_swing,
        after_swing=PushOpen(ai) if push else None,
        before_detect=partial(_view_step, view_only),
        on_grasp=on_grasp,
    )


def push_open(ai, execute, on_release=None):
    """After the swing: let go of the handle and push the door further open (see the module's step 4), all cuRobo.
    `on_release()`: called once the gripper has let go. Saves the door's new opening to the door file. Returns True
    when the push finished (or the dry run passed); raises Refused."""
    say("\n===== release + push further open =====")
    if not execute:
        say("  (asked of cuRobo above from the planned swing's end -- see the chosen tilt's line; asked again from the")
        say("   real one)")
        return True
    door = json.loads(dc.DOOR_FILE.read_text())
    st = ai.get_state()
    start = ai.raw_joints() if hasattr(ai, "raw_joints") else st["position"]
    P, _, why = _plan_push_open(ai, door, list(st["ee_pos"]), start)
    if why:
        raise Refused(f"{why} from where the swing ended -- still holding the handle.")
    say(f"push-open: {P.how}")
    gains = shared.grasped_impedance()
    go_to = {} if gains is None else dict(control_mode=IMPEDANCE_MODE, gains=shared.GRASPED_GAINS)
    say("opening the gripper ...")
    ai.gripper(0.0)
    time.sleep(0.5)
    if on_release is not None:
        on_release()
    for label, pos, hold in _push_legs(P):
        say(f"push-open: {label} (cuRobo) ...")
        if not ai.go_to_ee_pose(pos, P.down, f"push-open: {label}", orientation_hold=hold, **go_to):
            raise Refused(
                f"push-open: '{label}' stopped ({getattr(ai, 'last_error', '')}) -- the arm holds where it is."
            )
    # the push: a straight line square to the door, the hand held exactly as it is -- not cuRobo, whose path needn't be
    # straight (11-02: it pushed the wrong way)
    push = np.asarray(P.pushed) - np.asarray(P.inside)
    say(
        f"push-open: push {np.linalg.norm(push) * 100:.0f} cm to the hinge's y {P.df.hinge[1]:.3f} (straight, the "
        "hand held) ..."
    )
    here = ai.get_state()["ee_pos"]
    ok = ai.move_cartesian(
        [(list(P.pushed), list(here[3:7]))],
        impedance=gains,
        speed_scale=rammp_arm.contact_speed_scale() * PUSH_OPEN_SPEED_SCALE,
    )
    if not ok:
        raise Refused(f"push-open: the push stopped ({getattr(ai, 'last_error', '')}) -- the arm holds where it is.")
    # the door turned with the point the hand pushed: its new opening for the close
    h, a0 = P.df.hinge[:2], P.inside[:2] - P.df.hinge[:2]
    a1 = np.asarray(ai.get_state()["ee_pos"][:2]) - h
    turned = float(np.degrees(np.arctan2(a0[0] * a1[1] - a0[1] * a1[0], a0 @ a1))) * P.sign
    save_door_geometry(door_open_deg=P.deg + turned)
    say(f"door file: open ~{P.deg + turned:.1f} deg after the push ({turned:+.1f})")
    return True


def run(ai, execute, swing_deg=None, push=True, speed=1.0, log_args=None, view_only=False, on_hold=None):
    """The open task as the node runs it (was door_task.py `open`): to the view pose, grasp + swing, then
    (`push`) let go and push the door further open, all logged to RUNS_DIR/<time>_open (RunLog). `speed` (0..1]
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
        shared.LAST_PLAN = None  # open_task sets it once it has a plan
        ok = open_task(ai, execute, swing, view_only=view_only, push=push and not view_only, on_grasp=on_grasp)
        if not ok:
            say("swing did not finish -- stopping (still holding the handle).")
        elif push and not view_only:
            ok = push_open(ai, execute, on_release=partial(on_hold, False) if on_hold else None)
        log.event("open", ok=ok)
        if ok and not execute and not view_only:
            _save_plan(ai, log.dir, push)
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
