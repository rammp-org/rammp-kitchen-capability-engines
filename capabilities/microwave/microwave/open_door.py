"""Open the microwave: go to the view pose, detect the handle, grasp it, swing the door open, let go and push it
further open -- one run.

On the robot, the microwave node's `open` action (started with sheppy; the arm moves only with the node's
allow_execute on AND the goal's execute), its params as JSON:

    make action CAP=microwave ACTION=open PARAMS='{"execute": false}'                     # dry run
    make action CAP=microwave ACTION=open PARAMS='{"execute": true, "view_only": true}'   # to the view pose, stop
    make action CAP=microwave ACTION=open PARAMS='{"execute": true, "speed": 0.5}'        # the open, half speed
    make action CAP=microwave ACTION=open PARAMS='{"execute": true, "push_open": false}'  # stop after the swing

On the bench (rammp_arm.RammpArm, no node):

    python3 -m microwave.open_door             # dry run
    python3 -m microwave.open_door --execute   # grasp, 50-deg swing, push-open
    python3 -m microwave.open_door --no-push-open --execute   # stop after the swing
    python3 -m microwave.open_door --release --execute       # after that: let go, back off, park

0. View pose (plan_view, _view_step): the scene camera's door face (or boxer's box) gives a pose in front of it,
   off to the side away from the user (VIEW_SIDES); its joints are asked of cuRobo (plan_to_joints) and refused if
   it can't plan them or would end elsewhere. The door it saw is kept (SCENE_DOOR) only from a look the arm didn't
   block, and reused while the arm is at the view pose. `view_only` stops there.
1. Detect (`perception.HandleDetector`, depth only -- no YOLO, no SAM 3): the door from the scene camera, depth ->
   door plane, the vertical cluster sticking out of it = handle, the door edge farthest from it = hinge. Looks
   until two agree within 3 cm (handle and hinge).
2. Plan everything before any motion (door_common.open_door.plan_open): over tilts and arm configurations, a
   line-up point 25 cm out (or less), straight in along the door normal, and the largest swing (75 down to 45 deg)
   whose whole arc passes the sim gates (IK, reach, joint jump, J4/J6 guards, the chair and user boxes with
   CHAIR_SIM_MARGIN_M to spare: cuRobo's arm is fatter). cuRobo can't be asked for a line or an arc, so it is
   given their checked joints as targets; each candidate, best first, is asked of it before any motion, and one it
   can't plan (or plans with a detour) is passed over for the next. Refused only when none passes, with where the
   microwave would work.
3. Execute, all cuRobo: to the line-up point and straight in (one blended motion), close, one blended swing. The
   hand stays on the handle at the end.

4. Push-open (default; `--no-push-open` to stop after the swing), from the door the swing left: open the gripper;
   cuRobo back PUSH_BACK_M off the handle, level, along the door's outward normal, the hand held (the fingers
   slide off the bar); cuRobo straight up from there (the fingertips PUSH_CLEAR_TRY_M over the door top, the most
   that plans), turning the hand to point down (PUSH_TILT_TRY_DEG off vertical, the least that plans; of the two
   such hands the one nearest the grip, no wrist flip); close the gripper (a narrower hand round the door); then ONE
   cuRobo motion through three poses, stitched (go_to_ee_poses), the hand held (HOLD_FIXED): over to above the
   inside of the door; down until the tool is at the door top, PUSH_BEHIND_DOOR_M behind its inner face at the
   handle; push toward the hinge until PUSH_HINGE_SHORT_M short of level with it, opening the door to ~90 deg (saved
   as PUSH_OPEN_DOOR_DEG); then back the way it came (_back_legs), stitched too. Where cuRobo can't stitch the push, the
   run refuses. cuRobo is trusted with its moves (11-02) -- not sim-gated, and its world has no door: it is
   asked for each (plan_to_pose, nothing moves) from the planned swing's end before the grasp, and again from the
   real one before letting go.

`--release` (separate run, after the swing; instead of the push-open): open the gripper, then ONE blended cuRobo
motion: back straight off the handle, toward the base to clear the door's free edge, on to `PARK_JOINTS`
(constants) -- every leg asked of cuRobo before the gripper opens.

Under the engine: the node's `open` action (node.py -> `run`). Every run is recorded in RUNS_DIR/<time>_open: the
arm's joints (RunLog, arm.npz), the node's log, errors too (console.log), and a size-capped rosbag (bag.RunBag:
an occasional camera frame, colour and depth, and TF). On the bench:
the commands above (rammp_arm.RammpArm). Refusals raise door_common.common.Refused: the arm holds where it is.

The shared open (detect, plan, grasp, swing) is door_common.open_door; here is what is the microwave's: the view pose,
the scene camera or boxer finding the microwave, the push-open that follows the swing.
"""

import argparse
import collections
import json
import re
import sys
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import rammp_arm
from capability_engine import sim
from capability_engine.arm import IMPEDANCE as IMPEDANCE_MODE
from capability_engine.arm_messages import HOLD_FIXED, HOLD_NONE, PoseTarget
from door_common import common as dc
from door_common import open_door as shared
from door_common.common import (
    FINGERTIP_PAST_TOOL,
    DoorFrame,
    Refused,
    RunLog,
    captured,
    check_curobo_goal,
    quiet,
    release_and_back_off,
    save_door_geometry,
    say,
    short_delta,
)
from door_common.open_door import BACK_OFF_M, SWING_MAX_TRY_DEG, SWING_MIN_DEG
from scipy.spatial.transform import Rotation as R

from . import constants, recording
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
)

# --- view pose: before detecting, the wrist camera goes square in front of the microwave (placed from boxer's box, or
# without boxer from a first detection) ---
VIEW_DISTS = (0.40, 0.35, 0.45, 0.30)  # camera this far out from the box's front face (horizontally) ...
# 11-10: 0.35 first again, now the camera looks square at the face: closer, the door (handle to hinge) didn't fit
# 11-09: was 0.35 first -- with the arm 8 in further forward on the chair that put the hand over the user's lap, and
# every view pose IK'd into the person keep-out. The wrist OAK sees down to ~24 cm, so 0.25 is the closest
# 10-30: was 0.50 first -- at 50 cm a 1-2 cm handle bar is ~1 pixel of the OAK's 320x200 near-range depth and comes and
# goes between frames ("nothing protruding" / "no vertical bar" retries, and its height wandered)
VIEW_HEIGHTS = (0.15, 0.20, 0.10, 0.25, 0.05)  # ... and this far above the box's middle, looking down at the face's
# centre; the first (height, distance) that passes. Level at mid-height folded the elbow past its limit (10-30: J4
# ~210 deg, door 72 cm out); the hand-placed view that worked (10-28) had the camera ~20 cm above the door, looking down
VIEW_SIDES = (0.08, 0.04, 0.12, 0.0)  # ... and this far sideways along the face, toward the arm's left (+y): away
# from the user, who sits to the arm's right (-y), and toward the hinge. The camera looks straight at the face (square
# to it, tilted down), not back at the face's centre. Tried before the heights and distances. 11-10: 10 cm aside,
# looking back at the centre, saw the door at a slant with its hinge edge out of the image; the scene camera's face
# centre sits on the handle side (its door is short of the dark window). 15 cm first then put the handle at the
# image's edge (seen 11 cm of its 14, one look missed it): 8 cm first, the camera further out
# The view pose as fixed joints (deg), or None to place it from the scene camera's door (VIEW_DISTS/HEIGHTS/SIDES).
# 11-10: hand-placed -- the camera 0.61 m up, square to the door, 27 deg down, J4 44 deg from its limit; the placed
# ones folded J4 to within 7-11 deg, and the scene camera reads the door 13 cm short. For the microwave where it
# stood then (door face ~0.69 m out, handle ~26 cm to the user's side)
VIEW_JOINTS_DEG = (21.4, -43.8, -92.2, -110.6, -58.8, -93.8, -165.7)  # 11-11: teleoped to the view the user chose
VIEW_SEED_DEG = (0.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0)  # IK seed: Kinova's Home, the elbow the usual way
# (a start with J1 ~180 / J4 > 0 used to carry its flipped elbow into every plan, 10-29)
# ... and these when IK from it misses, lands past a guard or in the keep-out: PyBullet's IK ignores joint limits and
# one seed can settle on an impossible elbow (11-09: J4 225.8 deg for a microwave 47 cm out). The elbow the same
# way, more and less folded, and the arm swung toward either side
VIEW_GOAL_MATCH_DEG = 5.0  # cuRobo's plan to the view pose must end within this of the joints asked (every joint)
VIEW_MORE_SEEDS_DEG = (
    (0.0, 40.0, 180.0, -100.0, 0.0, 70.0, 90.0),
    (0.0, -10.0, 180.0, -145.0, 0.0, 40.0, 90.0),
    (-25.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0),
    (25.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0),
)
BOX_WAIT_S = 10.0  # wait this long for boxer's first microwave box
VIEW_AT_TOL_M = 0.05  # the hand this close to the view pose counts as there: a dry run goes on to detect and plan
VIEW_AT_JOINTS_DEG = 20.0  # ... with every joint this close to the view pose's too. 11-10: the hand was there on an
# arm folded the other way (J1 -105, toward the user), and every plan seeded from it put the forearm in the lap

# --- push-open (after the swing): let go, up and back, over to the inside of the door, down, push it further open ---
PUSH_OPEN_SWING_DEG = 50  # the swing before the push-open
PUSH_CLEAR_TRY_M = (
    0.05,
    0.0375,
    0.025,
)  # after letting go, cuRobo lifts the hand to cross the door with its fingertips
#   (pointing down) this far over the door top, the first that plans: half of 11-02's 10 cm (it went higher than
#   needed); the door top is a depth-fit estimate ...
PUSH_BACK_M = 0.08  # first, before going up, straight back this far from the handle, level, along the open door's
#   outward normal (toward where the swing came from), the hand held: the open fingers come off the handle bar
PUSH_BEHIND_DOOR_M = 0.05  # (~2 in) then the tool goes down this far behind the door's inner face, at the handle
PUSH_TOOL_ABOVE_TOP_M = 0.0  # ... to this height above the door top, pointing down: the fingertips reach
#   FINGERTIP_PAST_TOOL (6 cm) below the top edge
# then push straight left (toward the hinge: further open), the hand's x and height kept, until it is level with the
# detected hinge (the door file's hinge): the door then stands square to the arm, ~90 deg (11-02: 6 and 12 cm along
# the door's normal didn't open it enough) -- stopping PUSH_HINGE_SHORT_M short of it
PUSH_OPEN_DOOR_DEG = 90.0  # the door's opening the push leaves, saved for the close: taken, not measured (11-02:
#   estimating it from the hand, which slides along the door as it pushes, said 106.7 deg for ~90 by eye)
PUSH_HINGE_SHORT_M = 0.03  # the tool stops this short of the hinge's y, so the hand's far side, not its middle, lines
#   up with the hinge (11-02: level with it opened the door further than needed)
PUSH_BLEND_M = 0.02  # the stitched cuRobo motion passes its corners (up and back, above the door) this close, not
#   stopping; small, so it doesn't cut the corner over the door top
PUSH_TILT_TRY_DEG = (0, 30, 45, 60)  # the hand points down tilted this far from vertical, leaning away from the arm
#   base, the least that plans: straight down over the inside of the door the wrist runs out (11-02: J6 -104..-114,
#   at its guard, and IK 1-2 cm off on the way there)
PRE_PARK_X = 0.216

SCENE_CAMERA = None  # without boxer, the camera whose view of the door places the view pose (set_scene_camera)
SCENE_WAIT_S = 10.0  # wait this long for the scene camera's first frame
# The door as the scene camera saw it for the last view pose: {t, face, n, box, p_view}. At the view pose the arm stands
# between the scene camera and the door (11-10: the door then read 32 cm wide, not 47, and the hinge came out 10 cm
# short), so a run that starts there reuses this instead of looking again
SCENE_DOOR = None
SCENE_DOOR_CLEAR_M = 0.15  # ... kept only from a look with the hand this far from the view pose: out of the way
SCENE_DOOR_REUSE_S = 900.0  # ... if it is this recent (s) and the hand is still at that view pose (VIEW_AT_TOL_M)


def set_scene_camera(cam):
    """The scene camera (a perception.WristCamera on the scene camera's topics) for this action, or None."""
    global SCENE_CAMERA
    SCENE_CAMERA = cam


def _make_detector():
    from .perception import HandleDetector  # here, not at import: open3d, which the planning tests don't have

    return HandleDetector()


# no arm link or the tool lower than this on a cuRobo path (door_common.common.FREE_MOVE_MIN_Z): the table top is
# ~0.19 in the arm's frame (11-10), + 5 cm
FREE_MOVE_MIN_Z = 0.24


def configure(data_dir):
    """Everything door_common needs from the microwave: call once at startup."""
    constants.configure(data_dir)
    dc.FREE_MOVE_MIN_Z = FREE_MOVE_MIN_Z
    shared.configure(
        MAKE_DETECTOR=_make_detector,
        FIND_HINGE=shared.hinge_detected(HINGE_BEHIND_FACE, HINGE_PAST_EDGE),
        GRIP_EXT=GRIP_EXT,
        VERTICAL_CORR=VERTICAL_CORR,
        GRASP_X_ADJUST=GRASP_X_ADJUST,
        GRASPED_GAINS=GAINS_RIGID,
        SWING_SPEED_SCALE=SWING_SPEED_SCALE,
        CUROBO_PATHS=True,
        GRASP_FORCE=0.4,  # firmer: 0.25 (10-07) slipped off the bar on the exact swing, 11-10
        SWING_EXACT=True,  # the swing on its checked arc joints (move_joints), not cuRobo: 11-10
        APPROACH_EXACT=True,  # the straight-in run exactly as checked (move_joints), 11-10
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
    back = np.asarray(ee[:3], dtype=float) + out * PUSH_BACK_M  # straight back off the handle, level
    up_back = back.copy()  # then straight up from there
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
        back=back,
        grip=np.asarray(ee[3:7], dtype=float),
        up_back=up_back,
        above=above,
        inside=inside,
        pushed=np.r_[inside[0], df.hinge[1] - np.sign(df.hinge[1] - inside[1]) * PUSH_HINGE_SHORT_M, inside[2]],
        down=down,
        out=out,
        how=f"the hand {tilt_deg} deg off straight down, fingertips {clear_m * 100:.1f} cm over the door top",
    )


def _push_legs(P):
    """The push-open's cuRobo moves: (label, tool position, orientation hold) -- turning down on the first, then the
    hand held as it is (HOLD_FIXED), the last the push."""
    return (
        ("up and back", P.up_back, HOLD_NONE),
        ("over the door", P.above, HOLD_FIXED),
        ("down inside the door", P.inside, HOLD_FIXED),
        ("push open", P.pushed, HOLD_FIXED),
    )


def _back_legs(P):
    """After the push, back the way it came (as _push_legs, reversed): off the door to where the push started, up,
    back over the door to up-and-back -- the hand held pointing down -- so the arm can go home from there."""
    return (
        ("back off the door", P.inside, HOLD_FIXED),
        ("back up", P.above, HOLD_FIXED),
        ("back over the door", P.up_back, HOLD_FIXED),
    )


PUSH_PLAN_TRIES = 2  # each push-open move is asked of cuRobo this many times for a plan that passes its checks
PUSH_AT_START_DEG = 3.0  # a previewed move runs as planned only if the arm is this close (every joint) to its start


def _push_moves(P):
    """EVERY push-open move in order, as it runs: (label, tool position, quat, hold, keep_shape) -- back off the
    handle (the hand as it gripped), up and back (the hand turning down: the wrist may turn far, no shape cap), over,
    down, push, then back the way it came (the hand held pointing down)."""
    (up_label, up_pos, up_hold), *legs = _push_legs(P)
    return (
        ("back off the handle", P.back, P.grip, HOLD_FIXED, True),
        (up_label, up_pos, P.down, up_hold, False),
        *((lb, pos, P.down, hold, True) for lb, pos, hold in legs),
        *((lb, pos, P.down, hold, True) for lb, pos, hold in _back_legs(P)),
    )


def _stitched(ai, P, legs, label, go_to, speeds=None):
    """`legs` (as _push_legs) as ONE stitched cuRobo motion, passing every corner but the last within PUSH_BLEND_M;
    `speeds` {leg label: speed_scale}. Returns True, or (False, why) -- and whether nothing moved (it failed planning,
    e.g. no blend server), so the legs can be run one at a time instead."""
    targets = [
        PoseTarget(tuple(pos), tuple(P.down), hold, PUSH_BLEND_M if k < len(legs) - 1 else 0.0, (speeds or {}).get(lb))
        for k, (lb, pos, hold) in enumerate(legs)
    ]
    say(f"push-open: {', '.join(lb for lb, _, _ in legs)} (cuRobo, one stitched motion) ...")
    if ai.go_to_ee_poses(targets, f"push-open: {label}", **go_to):
        return True, "", False
    why = getattr(ai, "last_error", "")
    return False, why, "PLANNING_FAILED" in why


def _plan_push_open(ai, door, ee, q, loud=False):
    """Ask cuRobo -- nothing moves -- for the push-open from the hand pose `ee` / joints `q` at the swing's end: EVERY
    move (_push_moves) from where the one before ends, each with the checks it runs under (dc.pose_plan_problem: the
    shape cap, the floor, joint swings, the end), for the first variant whose moves ALL pass -- the hand
    PUSH_TILT_TRY_DEG off straight down, the fingertips PUSH_CLEAR_TRY_M over the door top. 11-11: only "plans" was
    asked here, so a variant whose "over the door" reshaped J3 79 deg was chosen, the handle let go, and the run refused
    two moves in. cuRobo's world has no door or scene camera.
    Returns (P, [(label, Plan)], None) or (None, None, why)."""
    whys = []
    for tilt_deg, clear_m in ((t, c) for t in PUSH_TILT_TRY_DEG for c in PUSH_CLEAR_TRY_M):
        P = _push_poses(door, ee, clear_m, tilt_deg)
        if float((P.pushed - P.inside)[:2] @ P.out[:2]) <= 0:  # toward the hinge must open the door further
            return None, None, "push-open: the hinge's y is on the door's closing side of the hand -- refusing"
        start, plans = np.asarray(q, dtype=float), []
        for label, pos, quat, hold, keep_shape in _push_moves(P):
            pl, why = dc.pose_plan_problem(ai, start, pos, quat, hold, keep_shape, PUSH_PLAN_TRIES)
            if why:
                whys.append(f"{P.how}: '{label}': {why}")
                if loud:
                    say(f"  push-open, {whys[-1]}")
                break
            plans.append((label, pl))
            start = np.asarray(pl.q, dtype=float)[-1]
        else:
            return P, plans, None
    return None, None, f"push-open: none of {len(whys)} variants passes every move (first: {whys[0]})"


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
                back=dict(to=j(P.back), quat=j(P.grip)),  # the straight back off the handle, before the first leg
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


# The wrist detection's crop box is this much wider than the scene camera's door, each side. 11-10: the scene camera
# read the door 32 cm wide (its depth stops short of the dark window), and the wrist detection then found the "hinge"
# at the crop's edge (radius 23-25 cm, refused). Its hinge is the door edge farthest from the handle
DOOR_BOX_WIDTH_PAD_M = 0.05


def _door_box(door, score):
    """The microwave as a box behind the door the scene camera found -- `door` (door_face's) its front face, BODY_DEPTH
    deep, DOOR_BOX_WIDTH_PAD_M wider each side -- in boxer's box form, so the wrist detection crops its points to it
    the same way."""
    n = np.asarray(door["normal"], dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    lo, hi = door["door_z"]
    face = np.r_[door["door_mid"][:2], (lo + hi) / 2]
    return {
        "center": face - n * BODY_DEPTH / 2,
        "R": np.column_stack([n, np.cross(up, n), up]),
        "size": np.array([BODY_DEPTH, door["width"] + 2 * DOOR_BOX_WIDTH_PAD_M, hi - lo]),
        "score": score,
        "source": "scene camera",
    }


def _detected_face():
    """Without boxer: the scene camera's view of the door places the view pose -- the scene segmenter's microwave box
    (RF-DETR), the door plane in it (no handle: it is too small from there), the face's centre (the middle of
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
    dc.camera().fixed_box = _door_box(door, f["box2d"]["score"])  # the wrist detection crops to it
    return face, np.asarray(door["normal"], dtype=float)


def _why_kind(text):
    """A refused goal's reason, coarsely: which keep-out box, past the guards, out of reach, or IK missed."""
    # "<link> 3.2 cm into person/lap (the chair ...)", "<link> 0.4 cm from person/lap, under the 1.5 cm margin"
    m = re.search(r"([\w/]+) -?[\d.]+ cm (?:into|from) ([\w/]+)", text)
    if m:
        return f"{m.group(1)} in {m.group(2) if m.group(2) != 'the' else 'the wheelchair keep-out'}"
    if "past the guards" in text:
        return "J4/J6 past the guards (elbow folded too far)"
    if "m out (max" in text:
        return "beyond the arm's reach"
    if "unreachable" in text:
        return "IK can't reach it"
    return text.strip()[:80] or "unknown"


def plan_view(ai, scene, rb):
    """The view pose in front of the microwave: the wrist camera VIEW_DISTS out from its front face, VIEW_SIDES along it
    away from the user, VIEW_HEIGHTS above the face's middle, looking square at the face (tilted down). The face comes
    from boxer's box (box_topic) or, without boxer, from a first detection made from where the arm is. The hand pose
    comes from where the camera sits on it now (TF), its joints from IK seeded at VIEW_SEED_DEG (then
    VIEW_MORE_SEEDS_DEG), checked like any cuRobo goal and then asked of cuRobo itself (plan_to_joints, nothing moves):
    the first that cuRobo plans too. Returns the goal joints (to go_to_joint_config, so the arm arrives on that branch)
    and the hand's goal position; raises Refused."""
    cam = dc.camera()
    global SCENE_DOOR
    boxer = bool(getattr(cam, "box_topic", ""))
    hand = np.asarray(ai.get_state()["ee_pos"][:3], dtype=float)
    reuse = (
        not boxer
        and SCENE_DOOR is not None
        and time.time() - SCENE_DOOR["t"] <= SCENE_DOOR_REUSE_S
        and float(np.linalg.norm(hand - SCENE_DOOR["p_view"])) <= VIEW_AT_TOL_M
    )
    if reuse:  # the arm is at the view pose, in the scene camera's way: the door as seen before it came here
        face, n = SCENE_DOOR["face"], SCENE_DOOR["n"]
        cam.fixed_box = SCENE_DOOR["box"]
        say(
            f"scene camera: reusing the door seen {time.time() - SCENE_DOOR['t']:.0f} s ago, before the arm came to "
            "the view pose (it now blocks the camera's view)"
        )
    else:
        face, n = _box_face(cam) if boxer else _detected_face()
    if VIEW_JOINTS_DEG is not None:
        return _fixed_view(ai, scene, rb, face, n, hand, remember=not boxer and not reuse)
    f = cam.grab()
    if f["base_T_cam"] is None:
        raise Refused("No arm base <- camera transform -- can't place the camera for the view pose.")
    st = ai.get_state()
    tool_T_cam = np.linalg.inv(_pose_T(st["ee_pos"][:3], st["ee_pos"][3:7])) @ f["base_T_cam"]
    side_dir = np.array([-n[1], n[0], 0.0]) / max(float(np.linalg.norm(n[:2])), 1e-9)  # along the face, level ...
    side_dir *= 1.0 if side_dir[1] >= 0 else -1.0  # ... toward +y: away from the user
    seeds = [np.radians(s) for s in VIEW_MORE_SEEDS_DEG] + [np.asarray(st["position"], float)]  # + where it is now
    tried, whys = [], collections.Counter()
    # every (side, height, distance), away from the user first
    for side, height, dist in ((s, h, d) for s in VIEW_SIDES for h in VIEW_HEIGHTS for d in VIEW_DISTS):
        # optical frame: z square to the face, at the face point in front of the camera (tilted down), y down in the
        # image, x right
        aim = face + side_dir * side
        pos = aim + n * dist + np.array([0.0, 0.0, height])
        z = (aim - pos) / np.linalg.norm(aim - pos)
        y = np.array([0.0, 0.0, -1.0]) - z * -z[2]
        y /= np.linalg.norm(y)
        T_cam = np.eye(4)
        T_cam[:3, :3], T_cam[:3, 3] = np.column_stack([np.cross(y, z), y, z]), pos
        T_tool = T_cam @ np.linalg.inv(tool_T_cam)
        label = f"to view pose {dist * 100:.0f} cm out, {side * 100:.0f} cm aside, {height * 100:.0f} cm up"
        with captured() as lines:
            q = check_curobo_goal(
                scene,
                rb,
                label,
                pos=T_tool[:3, 3],
                quat=R.from_matrix(T_tool[:3, :3]).as_quat(),
                seed=np.radians(VIEW_SEED_DEG),
                more_seeds=seeds,
            )
        if q is None:
            whys[_why_kind(" ".join(lines))] += 1
        elif hasattr(ai, "plan_to_joints"):  # cuRobo has the last word: its arm (padded spheres) is fatter than ours
            start = ai.raw_joints() if hasattr(ai, "raw_joints") else st["position"]
            pl = ai.plan_to_joints(q, start)
            if not pl.success:  # 11-09: a pose our sim passed was cuRobo's IK_FAIL at motion time
                whys[f"cuRobo: {str(pl.message).split(':')[0][:60]}"] += 1
                q = None
            elif len(pl.q) and np.degrees(np.max(np.abs(short_delta(pl.q[-1], q)))) > VIEW_GOAL_MATCH_DEG:
                # cuRobo's pose fallback: the same hand pose on other joints (11-10: J1 swung 131 deg, J3 177, the arm
                # folded the other way, and the move then failed the joints check). Not these joints: the next
                whys["cuRobo ends on other joints (its pose fallback)"] += 1
                q = None
        if q is not None:
            say(
                f"view pose ({label}, looking {np.degrees(np.arcsin(-z[2])):.0f} deg down): camera "
                f"{np.round(pos, 3)}, joints {np.round(np.degrees(q), 1)}"
            )
            # remembered only if the arm was out of the camera's way (11-10: a look taken with the arm at the view
            # pose read the door 32 cm wide, and reusing it kept the hinge short)
            if not boxer and not reuse and float(np.linalg.norm(hand - T_tool[:3, 3])) > SCENE_DOOR_CLEAR_M:
                SCENE_DOOR = dict(t=time.time(), face=face, n=n, box=cam.fixed_box, p_view=T_tool[:3, 3])
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
        more_seeds=seeds,
    )
    say("  why they failed: " + ", ".join(f"{n} x {k}" for k, n in whys.most_common()))
    say(f"  the door face is {np.linalg.norm(face[:2]):.2f} m from the arm's base (60-65 cm is comfortable)")
    raise Refused(
        f"No view pose ({len(tried)} tried: {VIEW_DISTS} m out x {VIEW_SIDES} m aside x {VIEW_HEIGHTS} m up) passes "
        "the checks -- "
        "move the microwave or the robot."
    )


def _fixed_view(ai, scene, rb, face, n, hand, remember):
    """The view pose at VIEW_JOINTS_DEG: checked against the chair and user and asked of cuRobo (plan_to_joints,
    nothing moves). The scene camera's door (`face`, `n`) only crops the wrist detection, and is remembered as
    plan_view does. Returns (joints, the hand's position there); raises Refused."""
    global SCENE_DOOR
    q = np.radians(VIEW_JOINTS_DEG)
    pos = np.asarray(sim.tool_pose(scene, rb, q)[0], dtype=float)
    ko = sim.keep_out_message(rb, q)
    if ko is not None:
        raise Refused(f"The fixed view pose (VIEW_JOINTS_DEG) puts the arm in the keep-out: {ko}")
    if hasattr(ai, "plan_to_joints"):
        start = ai.raw_joints() if hasattr(ai, "raw_joints") else ai.get_state()["position"]
        pl = ai.plan_to_joints(q, start)
        if not pl.success:
            raise Refused(f"cuRobo can't plan to the fixed view pose (VIEW_JOINTS_DEG): {pl.message}")
    say(
        f"view pose (fixed joints {VIEW_JOINTS_DEG}): hand {np.round(pos, 3)}; the scene camera's door face "
        f"{np.round(face, 3)}, {float(np.linalg.norm(face[:2] - pos[:2])):.2f} m from the hand"
    )
    if remember and float(np.linalg.norm(hand - pos)) > SCENE_DOOR_CLEAR_M:
        SCENE_DOOR = dict(t=time.time(), face=face, n=n, box=dc.camera().fixed_box, p_view=pos)
    return q, pos


VIEW_STOP = None  # why the last open ended at the view pose (_view_step), or None: it went on past it


def _view_step(view_only, ai, scene, rb, execute):
    """Go to the view pose, so the handle is always detected from the same place. Already there (VIEW_AT_TOL_M), a dry
    run goes on to detect and plan from it; elsewhere it stops, since it can't detect from where the arm isn't.
    Returns True when the open ends here, saying why in VIEW_STOP."""
    global VIEW_STOP
    VIEW_STOP = None
    q_view, p_view = plan_view(ai, scene, rb)
    st = ai.get_state()
    off = float(np.linalg.norm(np.asarray(st["ee_pos"][:3]) - p_view))
    joints_off = float(np.degrees(np.max(np.abs(short_delta(st["position"], q_view)))))
    if off <= VIEW_AT_TOL_M and joints_off > VIEW_AT_JOINTS_DEG:
        say(
            f"the hand is at the view pose ({off * 100:.1f} cm) but the arm is folded differently (a joint "
            f"{joints_off:.0f} deg off): going to the view pose's own joints"
        )
        off = float("inf")  # not there: move (or, a dry run, stop) as if elsewhere
    if off <= VIEW_AT_TOL_M:
        say(f"already at the view pose ({off * 100:.1f} cm) -- detecting from here")
        if view_only:
            say("\nAT THE VIEW POSE (view_only) -- stopping here.")
            VIEW_STOP = "view_only"
        return view_only
    if not execute:
        where = "to its joints" if off == float("inf") else f"{off * 100:.0f} cm away"
        say(
            f"\nDRY RUN -- would move to the view pose ({where}) and detect from there; nothing "
            "commanded. To dry-run the rest: a view_only goal (moves there), then this dry run again."
        )
        VIEW_STOP = "dry run, not at the view pose"
        return True
    moved = (
        dc.checked_curobo_move(ai, q_view, "to view pose", keep_shape=False)  # from wherever the arm was left
        if hasattr(ai, "execute_joint_trajectory") and hasattr(ai, "raw_joints")
        else ai.go_to_joint_config(q_view, "to view pose")
    )
    if not moved:
        raise Refused("Did not reach the view pose -- stopping before detection.")
    time.sleep(1.0)  # let the camera frames catch up with the arm
    if view_only:
        say("\nAT THE VIEW POSE (view_only) -- stopping here.")
        VIEW_STOP = "view_only"
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


def _pose_move(ai, pos, quat, label, hold, speed=None, held=False, keep_shape=True):
    """One push-open move: cuRobo's plan checked and run exactly as checked (dc.checked_pose_move) where the arm can,
    else cuRobo's own go_to_ee_pose. `held`: under the grasp's joint impedance (GRASPED_GAINS)."""
    gains = shared.GRASPED_GAINS if held else None
    if hasattr(ai, "execute_joint_trajectory") and hasattr(ai, "raw_joints"):
        return dc.checked_pose_move(ai, pos, quat, label, hold, speed, gains, keep_shape)
    go_to = {} if gains is None else dict(control_mode=IMPEDANCE_MODE, gains=gains)
    return ai.go_to_ee_pose(pos, quat, label, orientation_hold=hold, **go_to)


# The push-open's moves run blended (one cuRobo motion through the checked plans' joints) in these groups; the
# others -- backing off the handle and off the door (the pull-backs) and "up and back" -- run exactly as checked.
PUSH_BLENDED = (("over the door", "down inside the door", "push open"), ("back up", "back over the door"))
PUSH_END_AT_VIEW = True  # after the push-open, on to the view pose in the same blended motion (11-12: the user)


def _run_push_exact(ai, P, label, pl, speed, held):
    """One push-open move EXACTLY as _plan_push_open checked it (dc.run_checked_plan); planned and checked afresh from
    where the arm is (dc.checked_pose_move) if it isn't at the checked start (PUSH_AT_START_DEG). Raises Refused."""
    off = float(np.degrees(np.max(np.abs(short_delta(ai.raw_joints(), np.asarray(pl.q, dtype=float)[0])))))
    say(f"push-open: {label} ...")
    if off <= PUSH_AT_START_DEG:
        ok = dc.run_checked_plan(ai, pl, f"push-open: {label}", speed, held)
    else:
        say(f"  push-open: {label}: the arm is {off:.1f} deg from the checked start -- planned again from here")
        moves = {lb: (pos, quat, hold, keep) for lb, pos, quat, hold, keep in _push_moves(P)}
        pos, quat, hold, keep = moves[label]
        ok = dc.checked_pose_move(ai, pos, quat, f"push-open: {label}", hold, speed, held, keep)
    if not ok:
        raise Refused(f"push-open: '{label}' stopped ({getattr(ai, 'last_error', '')}) -- the arm holds where it is.")


def _run_push_blended(ai, P, plans, labels, speeds, held, end=None, end_label=""):
    """The checked plans `labels` (in order) as ONE blended cuRobo motion: every dc.CUROBO_SEND_EVERY-th joints of
    each plan and its end a target, passed within dc.CUROBO_BLEND_M except each plan's end before the push (the push
    starts where it should); `speeds` {label: speed_scale}; `end` (joints) one more target after them (the view
    pose). Each leg asked of cuRobo first (dc.curobo_path_problem). If it can't be (says why) or the driver plans
    nothing, the plans run exactly one by one (and `end` through a checked cuRobo move). Raises Refused."""
    by = dict(plans)
    knots, sp, bl = [], [], []
    for lb in labels:
        qs = np.asarray(by[lb].q, dtype=float)
        idx = sorted(set(range(dc.CUROBO_SEND_EVERY - 1, len(qs), dc.CUROBO_SEND_EVERY)) | {len(qs) - 1})
        for i in idx:
            knots.append(qs[i])
            sp.append(speeds.get(lb))
            stop = i == len(qs) - 1 and lb != labels[-1] and labels[labels.index(lb) + 1] == "push open"
            bl.append(0.0 if stop else dc.CUROBO_BLEND_M)
    if end is not None:
        knots.append(np.asarray(end, dtype=float))
        sp.append(None)
        bl.append(dc.CUROBO_BLEND_M)
    name = ", ".join(labels) + (f", {end_label}" if end is not None else "")
    why = dc.curobo_path_problem(ai, ai.raw_joints(), knots, f"push-open: {name}", knots=knots)
    if not why:
        say(f"push-open: {name} -- cuRobo through {len(knots)} of their checked joints, one blended motion ...")
        if dc.through_curobo_knots(ai, knots, f"push-open: {name}", gains=held, speeds=sp, blends=bl):
            return
        why = getattr(ai, "last_error", "")
        if "PLANNING_FAILED" not in why:
            raise Refused(f"push-open: {name} stopped ({why}) -- the arm holds where it is.")
    say(f"push-open: {name} not blended ({why}) -- one checked move at a time")
    for lb in labels:
        _run_push_exact(ai, P, lb, by[lb], speeds.get(lb), held)
    if end is not None and not dc.checked_curobo_move(ai, end, f"push-open: {end_label}", keep_shape=False):
        raise Refused(f"push-open: {end_label} stopped ({getattr(ai, 'last_error', '')}) -- the arm holds where it is.")


def _run_push_plans(ai, P, plans, gains, on_release):
    """The push-open's moves as _plan_push_open checked them, in order: open the gripper, back off the handle
    (exact), up and back (exact, the gripper closing meanwhile: a narrower hand round the door), then over, down and
    the push (at its own speed) blended, back off the door (exact), back up and over the door blended -- with
    PUSH_END_AT_VIEW on to the view pose in the same motion. Saves the door's new opening; returns True."""
    by = dict(plans)
    held = shared.GRASPED_GAINS if gains is not None else None
    push_speed = rammp_arm.GO_TO_SPEED_SCALE * PUSH_OPEN_SPEED_SCALE
    say("opening the gripper ...")
    ai.gripper(0.0)
    time.sleep(0.5)
    if on_release is not None:
        on_release()
    _run_push_exact(ai, P, "back off the handle", by["back off the handle"], None, held)
    gripper_done = dc.gripper_on_the_move(ai, 1.0, "push-open: a narrower hand round the door")
    try:
        _run_push_exact(ai, P, "up and back", by["up and back"], None, held)
    except BaseException:
        try:  # never left running; the move's own error is the one raised
            gripper_done()
        except BaseException:  # noqa: BLE001
            pass
        raise
    gripper_done()  # closed before the hand goes over the door
    _run_push_blended(ai, P, plans, PUSH_BLENDED[0], {"push open": push_speed}, held)
    _run_push_exact(ai, P, "back off the door", by["back off the door"], None, held)
    save_door_geometry(door_open_deg=PUSH_OPEN_DOOR_DEG)
    say(f"door file: open {PUSH_OPEN_DOOR_DEG:.0f} deg after the push (taken, not measured)")
    end = np.radians(VIEW_JOINTS_DEG) if PUSH_END_AT_VIEW and VIEW_JOINTS_DEG is not None else None
    _run_push_blended(ai, P, plans, PUSH_BLENDED[1], {}, held, end=end, end_label="to the view pose")
    return True


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
    P, plans, why = _plan_push_open(ai, door, list(st["ee_pos"]), start)
    if why:
        raise Refused(f"{why} from where the swing ended -- still holding the handle.")
    say(f"push-open: {P.how}; every move checked from the real joints before letting go")
    gains = shared.grasped_impedance()
    if hasattr(ai, "execute_joint_trajectory") and hasattr(ai, "raw_joints"):
        return _run_push_plans(ai, P, plans, gains, on_release)
    go_to = {} if gains is None else dict(control_mode=IMPEDANCE_MODE, gains=shared.GRASPED_GAINS)
    say("opening the gripper ...")
    ai.gripper(0.0)
    time.sleep(0.5)
    if on_release is not None:
        on_release()
    # straight back off the handle, level, the hand held as it gripped: the open fingers slide off the bar
    say(f"push-open: back {PUSH_BACK_M * 100:.0f} cm off the handle (cuRobo, the hand held) ...")
    if not _pose_move(ai, P.back, P.grip, "push-open: back off the handle", HOLD_FIXED, held=gains is not None):
        raise Refused(f"push-open: backing off the handle stopped ({getattr(ai, 'last_error', '')}) -- the arm holds.")
    # up and back (the hand turning down) on its own, then close the gripper -- a narrower hand to take round the
    # door (11-02) -- then over the door, down behind it, the push: ONE cuRobo motion, stitched (go_to_ee_poses),
    # passing the corner above the door within PUSH_BLEND_M (none after: the push starts where it should), the hand
    # held, the push at its own speed
    (up_label, up_pos, up_hold), *legs = _push_legs(P)
    say(f"push-open: {up_label} (cuRobo) ...")
    # the hand turns from gripping to pointing down here: the wrist may turn far (no shape cap)
    if not _pose_move(ai, up_pos, P.down, f"push-open: {up_label}", up_hold, held=gains is not None, keep_shape=False):
        raise Refused(
            f"push-open: '{up_label}' stopped ({getattr(ai, 'last_error', '')}) -- the arm holds where it is."
        )
    say("closing the gripper (a narrower hand round the door) ...")
    ai.gripper(1.0)
    time.sleep(0.8)
    push_speed = rammp_arm.GO_TO_SPEED_SCALE * PUSH_OPEN_SPEED_SCALE
    targets = [
        PoseTarget(
            tuple(pos),
            tuple(P.down),
            hold,
            PUSH_BLEND_M if k == 0 else 0.0,
            push_speed if label == "push open" else None,
        )
        for k, (label, pos, hold) in enumerate(legs)
    ]
    say(f"push-open: {', '.join(label for label, _, _ in legs)} (cuRobo, one stitched motion) ...")
    if hasattr(ai, "execute_joint_trajectory") and hasattr(ai, "raw_joints"):  # one checked move per leg (11-10)
        for label, pos, hold in legs:
            speed = push_speed if label == "push open" else None
            if not _pose_move(ai, pos, P.down, f"push-open: {label}", hold, speed, held=gains is not None):
                raise Refused(f"push-open: '{label}' stopped -- the arm holds where it is.")
    elif not ai.go_to_ee_poses(targets, "push-open: over, down, push", **go_to):
        why = getattr(ai, "last_error", "")
        raise Refused(f"push-open: over, down, push stopped ({why}) -- the arm holds where it is.")
    # back the way it came, off the door and over it, so the arm can go home from there
    back = _back_legs(P)
    if hasattr(ai, "execute_joint_trajectory") and hasattr(ai, "raw_joints"):  # one checked move per leg (11-10)
        ok, why, unmoved = False, "", True
    else:
        ok, why, unmoved = _stitched(ai, P, back, "back the way it came", go_to)
    if not ok and not unmoved:
        raise Refused(f"push-open: back the way it came stopped ({why}) -- the arm holds where it is.")
    if not ok:
        say("push-open: cuRobo can't stitch them here -- back one move at a time")
        for label, pos, hold in back:
            say(f"push-open: {label} (cuRobo) ...")
            if not _pose_move(ai, pos, P.down, f"push-open: {label}", hold, held=gains is not None):
                raise Refused(
                    f"push-open: '{label}' stopped ({getattr(ai, 'last_error', '')}) -- the arm holds where it is."
                )
    # the door's new opening, for the close: taken as square to the arm -- the push stops at the hinge's y
    save_door_geometry(door_open_deg=PUSH_OPEN_DOOR_DEG)
    say(f"door file: open {PUSH_OPEN_DOOR_DEG:.0f} deg after the push (taken, not measured)")
    return True


def run(ai, execute, swing_deg=None, push=True, speed=1.0, log_args=None, view_only=False, on_hold=None):
    """The open task as the node runs it (was door_task.py `open`): to the view pose, grasp + swing, then
    (`push`) let go and push the door further open, all logged to RUNS_DIR/<time>_open (RunLog), with what the
    node logs in its console.log and (recording.BAG) a rosbag in its bag/. `speed` (0..1] scales every arm motion
    for this run only. `view_only`: stop at the view pose. `on_hold(True)` once the gripper
    closes on the handle, `on_hold(False)` once the push-open has let go of it. Returns True when it finished
    (or the dry run passed); raises Refused."""
    import rammp_arm

    saved = (rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE)
    rammp_arm.CART_SPEED *= speed
    rammp_arm.JOINT_SPEED *= speed
    rammp_arm.GO_TO_SPEED_SCALE = max(0.01, rammp_arm.GO_TO_SPEED_SCALE * speed)
    log = RunLog("open", log_args or {}, ai)
    recording.set_console(log.dir / "console.log")
    recording.to_console("INFO", f"run log: {log.dir}")
    bag = recording.start_bag(log.dir)
    ok = False
    try:
        log.frame("run_start")
        swing = swing_deg or (PUSH_OPEN_SWING_DEG if push else SWING_MAX_TRY_DEG)
        say("\n===== OPEN: grasp + swing =====")
        on_grasp = partial(on_hold, True) if on_hold else None
        shared.LAST_PLAN = None  # open_task sets it once it has a plan
        ok = open_task(ai, execute, swing, view_only=view_only, push=push and not view_only, on_grasp=on_grasp)
        if ok and VIEW_STOP:  # ended at the view pose: no grasp, swing or push-open was planned or checked
            log.event("open", ok=ok, stopped_at_view_pose=VIEW_STOP)
            say(f"\nSTOPPED AT THE VIEW POSE ({VIEW_STOP}) -- nothing past it was planned or checked")
            return ok
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
        recording.to_console("ERROR", f"stopped: {e!r}")
        raise
    finally:
        log.close(ok)
        recording.stop_bag(bag)
        recording.set_console(None)
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
        release_and_back_off(ai, BACK_OFF_M, args.execute, pre_park_x=PRE_PARK_X, curobo=shared.CUROBO_PATHS)
    elif open_task(ai, args.execute, swing, min_swing) and not args.no_push_open and not VIEW_STOP:
        push_open(ai, args.execute)


if __name__ == "__main__":
    main()
