"""Shared motion pieces for the door tasks: planning, the door model, files, output.

Used by `open_door.py`, `close_door.py` and `demo_replay.py`, here and in each capability.

* `release_and_back_off` -- records the current grasp (pose + joints) to
  `LAST_GRASP_FILE`, opens the gripper, then backs straight off the handle along the
  gripper's approach axis in ~2 cm Cartesian steps.
* `DoorFrame` / `add_door_model` -- the door's geometry (from `DOOR_FILE`, which the grasp and
  swing write) and its collision model in the sim.

cuRobo (RAMMP's planner, through the driver's go_to_ee_pose / go_to_joint_config) does the
free-space moves: to park, to the regrasp back-off point, to the open-door view -- as
rammp_button_press does. Everything on or along the door (straight on/off the handle, the
swings, the pushes) is a joint path planned and checked here: the planner takes a goal
only and can't follow a line or an arc. `check_curobo_goal` gates a cuRobo goal. A checked
path can still be sent through cuRobo (`through_curobo`: some of its joints as blended targets,
each leg asked of cuRobo first by `curobo_path_problem`) -- the microwave sends all of them that way.

All gripper motion goes through `execute_command(Open/CloseGripperCommand())` --
`ArmInterfaceClient` has no `open_gripper()` method.

The arm model (make_sim, solve_ik, tool_pose, clearance) is arm_model.ArmModel: numpy, on the driver's own URDF --
no PyBullet. The IK guards are self.arm's ROS parameters (arm.max_ik_err, arm.max_step_jump_deg, ...), written into
capability_engine.sim's constants, so they are read as sim.X at call time; the joint limits are cuRobo's band on the
model's own limits (sim.limit_message / limit_room, which the model loads: arm_model.model). Only those constants and
sim's pure helpers come from there. The chair and the user are cuRobo's to check: every motion is planned by it, or --
run exactly as planned here -- validated by it (curobo_validates).

Each capability is its own process, so module constants are safe: the values that differ per appliance are
constants here with no default (None), set once at startup with `configure` (and `configure_paths`).
"""

import json
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rammp_arm
import yaml
from capability_engine import sim
from capability_engine.arm import IMPEDANCE as IMPEDANCE_MODE
from capability_engine.arm_messages import JointTarget
from capability_engine.cornell_commands import (
    CartesianTrajectoryCommand,
    JointCommand,
    OpenGripperCommand,
)
from capability_engine.sim import (  # noqa: F401 -- re-exported for the task modules (pure numpy, no PyBullet)
    Pose,
    continuous_ok,
    short_delta,
    wrap_joints,
)
from scipy.spatial.transform import Rotation as R

from . import arm_model
from .arm_model import Box, Sphere  # noqa: F401 -- re-exported for the task modules


def make_sim():
    """(None, the arm model): the process's one arm_model.ArmModel, in the (scene, rb) shape the door code passes
    around (the scene was PyBullet's; nothing reads it now)."""
    return None, arm_model.model()


def solve_ik(scene, rb, pos, quat, seed_joints):
    """IK for the tool at (pos, quat) from `seed_joints` on the arm model: (joints wrapped as commanded, error in m) --
    sim.LIMIT_ERR for an answer inside cuRobo's joint-limit margin, as sim.solve_ik reports it."""
    q, err = rb.ik(pos, quat, seed_joints)
    return q, (sim.LIMIT_ERR if sim.limit_room(q) < 0 else err)


def tool_pose(scene, rb, q):
    """The tool's (position, quaternion xyzw) in arm_mount at joints `q`."""
    return rb.pose(q)


def clearance(rb, q, bodies, closed=False, held=None):
    """The smallest distance (m) from the arm at joints `q` -- links, gripper and, with `held` (a Sphere), what it
    holds -- to `bodies` [(Box, name)]: (gap, "<part> vs <body>"). See arm_model.ArmModel.clearance."""
    return rb.clearance(q, bodies, closed, held)


class Refused(Exception):
    """A check failed before (or instead of) a motion: the arm holds where it is. The message says why."""


def set_constants(module, values, required=()):
    """Set constants in `module` (a module's globals()): an unknown name raises, and so does any of `required` left
    unset (None) -- a capability that forgets one fails at startup, not on the arm."""
    unknown = sorted(k for k in values if k not in module or not k.isupper())
    if unknown:
        raise ValueError(f"unknown settings: {unknown}")
    module.update(values)
    missing = [k for k in required if module[k] is None]
    if missing:
        raise ValueError(f"missing settings: {missing}")


def configure(**values):
    """Set this module's per-appliance constants at startup: APPLIANCE, PARK_JOINTS and the door-size defaults."""
    set_constants(globals(), values)


# ---------------------------------------------------------------------------
# Progress output: stdout on the bench, the node's logger under the engine (set_output)
# ---------------------------------------------------------------------------
_output = None
_quiet = 0


def say(*parts, sep=" ", end="\n", flush=False):
    """A progress line, with print()'s signature (ruff bans print in capability code)."""
    if _quiet:
        return
    text = sep.join(map(str, parts))
    if _output is not None:
        for line in text.split("\n"):
            if line.strip():
                _output(line.rstrip())
        return
    sys.stdout.write(text + end)
    if flush:
        sys.stdout.flush()


def set_output(fn):
    """Send say() to `fn(line)` (e.g. the node's get_logger().info); None restores stdout."""
    global _output
    _output = fn


@contextmanager
def quiet():
    """say() prints nothing inside (planning tries whose output only clutters the log)."""
    global _quiet
    _quiet += 1
    try:
        yield
    finally:
        _quiet -= 1


@contextmanager
def captured():
    """say() lines inside go into the yielded list instead of the log (even inside quiet()): to sum up many tries."""
    global _output, _quiet
    saved, lines = (_output, _quiet), []
    _output, _quiet = lines.append, 0
    try:
        yield lines
    finally:
        _output, _quiet = saved


# ---------------------------------------------------------------------------
# Where a run's files go. The open saves the door geometry and the grasp; the close (a later
# run, maybe another process) reads them back -- so this must be persistent (the node's
# data_dir parameter, a host volume in the fragment). Each capability sets its own (configure_paths).
# ---------------------------------------------------------------------------
DATA_DIR = Path("/tmp/door_common")
LAST_GRASP_FILE = DATA_DIR / "last_grasp.json"
DOOR_FILE = DATA_DIR / "door.json"
RUNS_DIR = DATA_DIR / "runs"
DETECT_FRAMES_DIR = DATA_DIR / "detections"


def configure_paths(data_dir):
    """Point every file above into `data_dir` (created). Modules read them as dc.X at call time."""
    global DATA_DIR, LAST_GRASP_FILE, DOOR_FILE, RUNS_DIR, DETECT_FRAMES_DIR
    DATA_DIR = Path(data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LAST_GRASP_FILE = DATA_DIR / "last_grasp.json"
    DOOR_FILE = DATA_DIR / "door.json"
    RUNS_DIR = DATA_DIR / "runs"
    DETECT_FRAMES_DIR = DATA_DIR / "detections"


STEP_M = 0.02
APPLIANCE = "appliance"  # what the door belongs to: names its body in the door model and the refusal messages
PARK_JOINTS = None  # park ("end") joints in rad, if the capability parks (release_and_back_off's `park`)


def door_arc_waypoints(
    start_pose, hinge_position, arc_length_m, waypoint_spacing_m, direction=1, rotate_orientation=True
):
    """Waypoints along a door arc about a vertical hinge axis, starting from (not including)
    `start_pose`, at constant height. Copied from `PerceptionInterface._generate_door_arc_waypoints`
    so these scripts don't import the lab's whole perception stack for one function."""
    if arc_length_m <= 0:
        return []
    if waypoint_spacing_m <= 0:
        raise ValueError("waypoint_spacing_m must be > 0")
    if direction not in (-1, 1):
        raise ValueError("direction must be either +1 or -1")
    hx, hy, hz = start_pose.position
    cx, cy, _ = hinge_position
    radius_vec = np.array([hx - cx, hy - cy], dtype=float)
    radius = np.linalg.norm(radius_vec)
    if radius < 1e-8:
        raise ValueError("Handle pose is too close to hinge pose; radius is ~0.")
    start_theta = np.arctan2(radius_vec[1], radius_vec[0])
    total_angle = arc_length_m / radius
    num_segments = max(1, int(np.ceil(arc_length_m / waypoint_spacing_m)))
    start_rot = R.from_quat(start_pose.orientation)
    waypoints = []
    for i in range(1, num_segments + 1):
        delta_angle = direction * i / num_segments * total_angle
        theta = start_theta + delta_angle
        orientation = (
            (R.from_euler("z", delta_angle) * start_rot).as_quat() if rotate_orientation else start_pose.orientation
        )
        waypoints.append(
            Pose(position=(cx + radius * np.cos(theta), cy + radius * np.sin(theta), hz), orientation=orientation)
        )
    return waypoints


def wait_for_joints(ai, q, tol_deg=1.0, timeout_s=6.0):
    """Wait until the arm is within `tol_deg` of the COMMANDED joints (Kortex can
    silently drop a chained command, so velocity ~ 0 is not proof of arrival)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        cur = np.array(ai.get_state()["position"], dtype=float)
        if np.degrees(np.max(np.abs((cur - q + np.pi) % (2 * np.pi) - np.pi))) < tol_deg:
            return True
        time.sleep(0.05)
    return False


def fit_circle_xy(xy):
    """Least-squares circle through 2-D points (Kasa fit). Returns (centre xy, radius, rms residual m)."""
    xy = np.asarray(xy, dtype=float)
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy**2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = float(np.sqrt(c + cx**2 + cy**2))
    resid = np.linalg.norm(xy - [cx, cy], axis=1) - r
    return np.array([cx, cy]), r, float(np.sqrt(np.mean(resid**2)))


def run_cartesian_trajectory(ai, traj, tol_m=0.01, settle_s=2.0, timeout_s=60.0, impedance=None, speed_scale=1.0):
    """Send a CartesianTrajectoryCommand and judge it by where the arm ACTUALLY ends up, not
    the RPC return: the server's wait can be tripped by a stale END/ABORT notification and
    return False at once while Kortex runs the whole trajectory anyway (09-28: returned False
    with the arm still at the start, then reached every waypoint). Waits until the EE is within
    `tol_m` of the last waypoint, or has not moved for `settle_s` (after at least `settle_s`
    had passed, so a not-yet-started motion isn't mistaken for a stop). Returns (reached, err_m, rpc_ok).
    `impedance`: run it compliant, e.g. the door swing (rammp_arm.IMPEDANCE).
    `speed_scale`: e.g. rammp_arm.contact_speed_scale() for a move that pushes on the door."""
    if (impedance is not None or speed_scale != 1.0) and hasattr(ai, "move_cartesian"):
        # sheppy: compliant (joint impedance) and/or slowed execution -- see rammp_arm.IMPEDANCE
        rpc_ok = ai.move_cartesian(
            [(list(pp), list(qq)) for pp, qq in traj], impedance=impedance, speed_scale=speed_scale
        )
    else:
        rpc_ok = ai.execute_command(CartesianTrajectoryCommand([(list(pp), list(qq)) for pp, qq in traj]))
    goal = np.asarray(traj[-1][0], dtype=float)
    t0 = time.time()
    last = np.asarray(ai.get_state()["ee_pos"][:3], dtype=float)
    last_move = t0
    while time.time() - t0 < timeout_s:
        cur = np.asarray(ai.get_state()["ee_pos"][:3], dtype=float)
        err = float(np.linalg.norm(cur - goal))
        if err < tol_m:
            time.sleep(0.3)  # let the final taper finish
            cur = np.asarray(ai.get_state()["ee_pos"][:3], dtype=float)
            return True, float(np.linalg.norm(cur - goal)), rpc_ok
        if np.linalg.norm(cur - last) > 0.002:
            last, last_move = cur, time.time()
        elif time.time() - last_move > settle_s and time.time() - t0 > settle_s:
            return False, err, rpc_ok
        time.sleep(0.1)
    cur = np.asarray(ai.get_state()["ee_pos"][:3], dtype=float)
    return False, float(np.linalg.norm(cur - goal)), rpc_ok


def plan_straight_line(scene, rb, start_pos, quat, direction, dist, seed_joints, label):
    """Joint solutions for a straight Cartesian line, ~2 cm per step, fixed orientation.
    Returns the list of joint vectors, or None if any step fails a gate."""
    n = max(1, int(np.ceil(dist / STEP_M)))
    q, plan = np.asarray(seed_joints, dtype=float), []
    for k in range(1, n + 1):
        tgt = np.asarray(start_pos) + np.asarray(direction) * dist * k / n
        nq, err = solve_ik(scene, rb, tgt, quat, q)
        jump = float(np.degrees(np.max(np.abs(short_delta(q, nq)))))
        j6, j7 = float(np.degrees(nq[5])), float(np.degrees(nq[6]))
        say(
            f"  {label} {k}/{n} -> {np.round(tgt, 3)}  IK err {err * 100:.2f}cm  jump {jump:.1f}deg  "
            f"J6 {j6:.1f}deg  J7 {j7:.1f}deg"
        )
        if err > sim.MAX_IK_ERR or jump > sim.MAX_STEP_JUMP_DEG or sim.limit_message(nq) or not continuous_ok(q, nq):
            say(f"  {label}: step {k} fails the IK/jump/joint-limit/wrap gate.")
            return None
        plan.append(nq)
        q = nq
    return plan


BIG_MOVE_TIMEOUT_S = 40.0


def plan_cartesian(
    scene, rb, p0, q0_quat, p1, q1_quat, seed_joints, label, bodies=None, min_clear=None, closed=False, held=None
):
    """Straight line p0 -> p1 with the orientation slerped q0 -> q1, ~2 cm / <= 5 deg per step,
    each step IK-solved from the previous one. Gates: IK, per-step jump, J6, wrap and (if
    `bodies`) >= MIN_CLEAR from the door model, checked along each step too."""
    from scipy.spatial.transform import Slerp

    rots = R.from_quat([q0_quat, q1_quat])
    ang = float(np.degrees((rots[0].inv() * rots[1]).magnitude()))
    n = max(1, int(np.ceil(max(np.linalg.norm(np.asarray(p1) - p0) / STEP_M, ang / 5.0))))
    slerp = Slerp([0, 1], rots)
    q, plan, worst = np.asarray(seed_joints, float), [], (9.0, "")
    for k in range(1, n + 1):
        f = k / n
        tgt = np.asarray(p0) + (np.asarray(p1) - p0) * f
        nq, err = solve_ik(scene, rb, tgt, slerp([f]).as_quat()[0], q)
        # one PyBullet IK pass sometimes stalls (joints barely move, ~2.3 cm error), mostly on
        # rotate-in-place steps; re-solving from its own result converges (09-28 sim check)
        for _ in range(3):
            if err <= sim.MAX_IK_ERR:
                break
            nq, err = solve_ik(scene, rb, tgt, slerp([f]).as_quat()[0], nq)
        # wrapped: a free joint going 179.8 -> -179.8 moved 0.4 deg, not 359.6 (Kortex takes the short way)
        jump = float(np.degrees(np.max(np.abs((nq - q + np.pi) % (2 * np.pi) - np.pi))))
        if bodies is not None:
            worst = min(
                [worst]
                + [clearance(rb, q + short_delta(q, nq) * s_ / 3, bodies, closed, held=held) for s_ in (1, 2, 3)],
                key=lambda w: w[0],
            )
        bad = (
            err > sim.MAX_IK_ERR
            or jump > sim.MAX_STEP_JUMP_DEG
            or sim.limit_message(nq)
            or not continuous_ok(q, nq)
            or (bodies is not None and worst[0] < (min_clear or MIN_CLEAR))
        )
        if bad or k in (1, n) or k % 5 == 0:
            say(
                f"  {label} {k}/{n} -> {np.round(tgt, 3)}  IK err {err * 100:.2f}cm  jump {jump:.1f}deg  "
                f"J4 {np.degrees(nq[3]):.1f}  J6 {np.degrees(nq[5]):.1f}  clear {worst[0] * 100:.1f}cm ({worst[1]})"
                f"{'' if continuous_ok(q, nq) else '  WRAP'}"
            )
        if bad:
            say(f"  {label}: step {k} fails a gate.")
            return None
        plan.append(nq)
        q = nq
    say(f"  {label}: {n} steps, orientation change {ang:.0f} deg, worst clearance {worst[0] * 100:.1f} cm")
    return plan


def _ik_within_guards(scene, rb, pos, quat, seed):
    """IK for (pos, quat) from `seed`, re-solved up to 3 times: (joints, None), or (None, why it fails)."""
    q, err = solve_ik(scene, rb, pos, quat, seed)
    for _ in range(3):
        if err <= sim.MAX_IK_ERR:
            break
        q, err = solve_ik(scene, rb, pos, quat, q)
    if err == sim.LIMIT_ERR:  # reached, but too near a joint limit for cuRobo (solve_ik's code)
        return None, f"goal {np.round(pos, 3)}: {sim.limit_message(q)}"
    if err > sim.MAX_IK_ERR:
        return None, f"goal {np.round(pos, 3)} unreachable (IK err {err * 100:.1f} cm)"
    return np.asarray(q, float), None


def check_curobo_goal(scene, rb, label, q=None, pos=None, quat=None, seed=None, bodies=None, more_seeds=()):
    """Gate the GOAL of a cuRobo move (self.arm.go_to_*): cuRobo plans the path in the driver, which can't be
    previewed, so only where it ends is checked here -- joints `q`, or (pos, quat) IK'd from `seed`: IK, reach,
    cuRobo's joint-limit margin and, with `bodies`, >= MIN_CLEAR from the door model (the chair and user: cuRobo's).
    For a pose goal cuRobo may end on another joint branch than this IK; the checked branch is the one nearest the
    seed. `more_seeds`: tried in turn when IK from `seed` misses or lands near a joint limit -- local IK keeps the
    seed's branch, and a seed near the goal finds the reachable one (10-30, PyBullet's IK then: line-up J4 +160 deg).
    Returns the checked goal joints, or None (refused)."""
    if q is None:
        if np.linalg.norm(pos) > MAX_REACH_M:
            say(
                f"  {label}: goal {np.round(pos, 3)} is {np.linalg.norm(pos):.3f} m out (max {MAX_REACH_M})"
                " -- refusing."
            )
            return None
        for s in [seed, *more_seeds]:
            q, why = _ik_within_guards(scene, rb, pos, quat, s)
            if q is not None:
                break
            why = why if s is seed else f"{why} (and from {1 + len(more_seeds)} seeds)"
        if q is None:
            say(f"  {label}: {why} -- refusing.")
            return None
    q = np.asarray(q, float)
    limit = sim.limit_message(q)
    if limit:
        say(f"  {label}: goal joints: {limit} -- refusing.")
        return None
    if bodies is not None:
        c = clearance(rb, q, bodies)
        if c[0] < MIN_CLEAR:
            say(f"  {label}: goal {c[0] * 100:.1f} cm from the door model ({c[1]}) -- refusing.")
            return None
        say(f"  {label}: goal ok, {c[0] * 100:.1f} cm from the door model; cuRobo plans the path (not previewed)")
    else:
        say(f"  {label}: goal ok; cuRobo plans the path (not previewed)")
    return q


MAX_REACH_M = 0.90


def execute_joint_plan(ai, plan, label):
    """One JointCommand per step, waiting for convergence to the COMMANDED joints. Kortex
    sometimes rejects a command (ROBOT_MOVEMENT_IN_PROGRESS) and the arm just stays put --
    re-send that step once before giving up (kortex-dropped-substep-bug).
    With self.arm (move_joints) the whole plan is ONE smooth trajectory through every step instead:
    one JointCommand per ~2 cm step became a stop-and-go per step there (loud, 10-07)."""
    if hasattr(ai, "move_joints"):
        ok = ai.move_joints(list(plan))
        if not (ok and wait_for_joints(ai, plan[-1])):
            say(f"  {label}: not at the last step's joints -- stopping here.")
            return False
        return True
    for k, q in enumerate(plan, 1):
        for attempt in (1, 2):
            ai.execute_command(JointCommand(pos=q.tolist()))
            if wait_for_joints(ai, q):
                break
            if attempt == 1:
                say(f"  {label} step {k}: not at the commanded joints after 6 s -- re-sending once.")
                time.sleep(0.5)
        else:
            say(f"  {label} step {k}: still not there after a re-send -- stopping here.")
            return False
    return True


def load_park():
    if PARK_JOINTS is None:
        raise Refused("no PARK_JOINTS set (configure)")
    return wrap_joints(PARK_JOINTS)


# ---------------------------------------------------------------------------
# A checked joint path through cuRobo: every CUROBO_SEND_EVERY-th joints a target, asked of it first
# ---------------------------------------------------------------------------
CUROBO_SEND_EVERY = 4  # every 4th checked waypoint is a cuRobo target (+ the last)
CUROBO_BLEND_M = 0.01  # the tool passes those targets this close, not stopping on each
CUROBO_MAX_DETOUR = 1.5  # a leg may travel this many times the joints' direct distance (11-02: cuRobo alone took 78 cm
# for a 12 cm push: not with the hand on the handle)
CUROBO_RETRIES = 2  # a leg cuRobo's post-validation refuses (velocity / acceleration / jerk over a limit) is re-asked
# this many times: its plans vary a little. 11-10: J2 acceleration 25.5-27.5 > 25 rad/s^2 on every straight-in
CUROBO_DETOUR_FLOOR_DEG = 5.0  # legs shorter than this are measured against this
FREE_MOVE_MIN_Z = None  # m (arm base frame): no arm link or the tool lower than this on a cuRobo path (the table:
# cuRobo's world has none). None: no floor. 11-10: an unchecked cuRobo move took the fingertips to z 0.03, under the
# microwave's table (top ~0.19), and hit it
FREE_MOVE_MAX_SWING_RAD = np.radians(25.0)  # ... and no joint further than this past the straight start -> goal range
FREE_MOVE_END_DEG = 2.0  # ... and the plan ends within this of the joints asked (cuRobo's pose fallback ends elsewhere)


def curobo_knots(path):
    """The joints of a checked path that cuRobo is given as targets."""
    path = list(path)
    knots = path[CUROBO_SEND_EVERY - 1 :: CUROBO_SEND_EVERY]
    if not knots or knots[-1] is not path[-1]:
        knots.append(path[-1])
    return knots


def curobo_path_problem(ai, start, path, label, in_front=None, knots=None):
    """Ask cuRobo -- nothing moves -- for every leg of `path` from joints `start`: why it can't be used, or None.
    `knots`: the targets to ask for, instead of curobo_knots(path). Without cuRobo (no plan_to_joints: the bench's
    mock) it can't check, and passes."""
    if not hasattr(ai, "plan_to_joints"):
        return None
    q = start
    for k, goal in enumerate(knots if knots is not None else curobo_knots(path), 1):
        pl = ai.plan_to_joints(goal, q)
        for _ in range(CUROBO_RETRIES):
            if pl.success or _over_limit(pl.message) is None:
                break
            pl = ai.plan_to_joints(goal, q)
        if not pl.success:
            slowed = _passes_slowed(pl.message)
            if slowed is None:
                return f"cuRobo can't plan {label}, leg {k}: {pl.message}"
            say(f"  {label}, leg {k}: {slowed}")
            q = goal
            continue
        if len(pl.q) > 1:
            direct = max(float(np.linalg.norm(short_delta(q, goal))), np.radians(CUROBO_DETOUR_FLOOR_DEG))
            went = sum(float(np.linalg.norm(short_delta(a, b))) for a, b in zip(pl.q[:-1], pl.q[1:], strict=True))
            if went > CUROBO_MAX_DETOUR * direct:
                return f"cuRobo's path for {label}, leg {k}, takes a detour ({went / direct:.1f}x the direct distance)"
        low = path_below_floor(pl.q) or path_not_in_front(pl.q, in_front)
        if low:
            return f"cuRobo's path for {label}, leg {k}: {low}"
        q = goal
    return None


def _lowest(rb, q):
    """(z, name) of the lowest joint origin past J1 or the tool at joints `q`, in arm_mount."""
    return rb.lowest(q)


FLOOR_START_TOL_M = 0.005  # a path that starts under the floor may dip this much below its start (planning noise)


def path_below_floor(qs):
    """Why a cuRobo path's joints `qs` take the arm below FREE_MOVE_MIN_Z, or "" (also with no floor set). A path that
    STARTS under the floor (the arm already there: 11-16, the tool at z 0.222 pointing down) is not refused for its
    start: it must just never go lower than it starts (less FLOOR_START_TOL_M) -- the floor keeps paths off the table,
    and leaving a low pose is moving away from it."""
    if FREE_MOVE_MIN_Z is None or len(qs) == 0:
        return ""
    rb = make_sim()[1]
    z0, _ = _lowest(rb, qs[0])
    floor = min(FREE_MOVE_MIN_Z, z0 - FLOOR_START_TOL_M)
    for i, q in enumerate(qs):
        z, link = _lowest(rb, q)
        if z < floor:
            under = f"the floor {FREE_MOVE_MIN_Z:.2f}" if floor == FREE_MOVE_MIN_Z else f"its start's z {z0:.3f}"
            return f"{link} down to z {z:.3f} at sample {i}/{len(qs)}, under {under}"
    return ""


def path_not_in_front(qs, in_front):
    """Why a cuRobo path's joints `qs` take the tool behind `in_front` = (point, outward normal, least distance) --
    the door's plane through the grasp: the tool never further into the door than the grasp -- or "" (also for
    None)."""
    if in_front is None or len(qs) == 0:
        return ""
    point, n, least = np.asarray(in_front[0], float), np.asarray(in_front[1], float), float(in_front[2])
    n = n / np.linalg.norm(n)
    rb = make_sim()[1]
    for i, q in enumerate(qs):
        d = float(np.dot(rb.pose(q)[0] - point, n))
        if d < least:
            return f"tool {-d * 100:.1f} cm past the grasp into the door at sample {i}/{len(qs)}"
    return ""


def free_move_problem(ai, start, q_goal, in_front=None, end_tool_tol=None, keep_shape=True):
    """cuRobo's plan to joints `q_goal` from joints `start` (plan_to_joints, nothing moves) and why it can't be run as
    it is, "" when it can: the floor (path_below_floor), the tool in front (path_not_in_front), no joint swinging
    FREE_MOVE_MAX_SWING_RAD past its straight range, the end within FREE_MOVE_END_DEG of `q_goal` (or, with
    `end_tool_tol` (m, deg), the tool there). A plan refused only for a full-speed limit is asked again
    (CUROBO_RETRIES). Returns (plan, why)."""
    start = np.asarray(start, dtype=float)
    big = reconfiguration(start, q_goal) if keep_shape else ""
    if big:
        return None, big
    pl = ai.plan_to_joints(q_goal, start)
    for _ in range(CUROBO_RETRIES):
        if pl.success or _over_limit(pl.message) is None:
            break
        pl = ai.plan_to_joints(q_goal, start)
    if not pl.success or len(pl.q) == 0:
        return pl, f"cuRobo can't plan it ({pl.message})"
    qs = np.asarray(pl.q, dtype=float)
    low = path_below_floor(qs) or path_not_in_front(qs, in_front)
    if low:
        return pl, f"cuRobo's path takes the {low}"
    off = np.abs(np.array([short_delta(start, q) for q in qs]))
    swing = off.max(axis=0) - np.abs(short_delta(start, q_goal))
    if float(swing.max()) > FREE_MOVE_MAX_SWING_RAD:
        j = int(np.argmax(swing))
        return pl, (
            f"cuRobo's path swings J{j + 1} {np.degrees(swing[j]):.0f} deg past a straight move (max "
            f"{np.degrees(FREE_MOVE_MAX_SWING_RAD):.0f})"
        )
    end = float(np.degrees(np.max(np.abs(short_delta(qs[-1], q_goal)))))
    if end > FREE_MOVE_END_DEG:  # other joints: refused, unless `end_tool_tol` and the tool ends there
        why = f"cuRobo's plan ends {end:.1f} deg off the goal joints"
        if end_tool_tol is None:
            return pl, why
        rb = make_sim()[1]
        (p_end, o_end), (p_goal, o_goal) = rb.pose(qs[-1]), rb.pose(q_goal)
        d_m = float(np.linalg.norm(p_end - p_goal))
        d_deg = float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(o_end, o_goal)))))))
        if d_m > end_tool_tol[0] or d_deg > end_tool_tol[1]:
            return pl, f"{why}, the tool {d_m * 100:.1f} cm / {d_deg:.1f} deg off"
    return pl, ""


FREE_MOVE_MAX_JOINT_DEG = 60.0  # no joint turns more than this in one free move (the arm keeps its shape)
# 11-10: 90 let J1 +89 / J3 -81 through (the elbow swung round); every good line-up that day turned <= 43


def reconfiguration(start, q_goal, max_deg=None):
    """Why the move `start` -> `q_goal` changes the arm's shape (a joint turning more than `max_deg`, None:
    FREE_MOVE_MAX_JOINT_DEG, the free joints the short way), or ""."""
    limit = FREE_MOVE_MAX_JOINT_DEG if max_deg is None else max_deg
    d = np.degrees(np.abs(short_delta(np.asarray(start, dtype=float), q_goal)))
    j = int(np.argmax(d))
    if d[j] > limit:
        return f"J{j + 1} would turn {d[j]:.0f} deg (max {limit:.0f}): the arm would change its shape"
    return ""


FREE_MOVE_DIRECT_STEPS = 40  # the direct joint path (direct_move_problem) is checked for the floor at this many points


def direct_move_problem(ai, start, q_goal, keep_shape=True):
    """Why the direct joint path `start` -> `q_goal` (every joint the short way, together, as move_joints sends it)
    can't be run, or "": the floor (path_below_floor, at FREE_MOVE_DIRECT_STEPS points along it), then cuRobo's
    check of the path as timed -- the chair and user, the arm itself, the joint limits (curobo_validates)."""
    start = np.asarray(start, dtype=float)
    big = reconfiguration(start, q_goal) if keep_shape else ""
    if big:
        return f"the direct joint path: {big}"
    d = short_delta(start, q_goal)
    qs = [start + d * k / FREE_MOVE_DIRECT_STEPS for k in range(FREE_MOVE_DIRECT_STEPS + 1)]
    low = path_below_floor(qs)
    if low:
        return f"the direct joint path: {low}"
    return curobo_validates(ai, start, [start + d], "the direct joint path")


def checked_curobo_move(ai, q_goal, label, speed_scale=None, in_front=None, end_tool_tol=None, keep_shape=True):
    """cuRobo's plan to joints `q_goal` from where the arm is, checked (free_move_problem), then run EXACTLY
    (execute_joint_trajectory), not planned again. Raises Refused before anything moves when a check fails; returns
    whether the motion succeeded."""
    pl, why = free_move_problem(ai, ai.raw_joints(), q_goal, in_front, end_tool_tol, keep_shape)
    if why:  # cuRobo's path refused: the direct joint path, if it is clear (cuRobo validates it as sent)
        direct = (
            "no in_front check for it"
            if in_front is not None
            else direct_move_problem(ai, ai.raw_joints(), q_goal, keep_shape)
        )
        if direct:
            raise Refused(f"{label}: {why}; {direct} -- nothing moved.")
        say(f"  {label}: {why}; the direct joint path is clear (the chair and user, the floor): running that")
        return bool(ai.move_joints([np.asarray(q_goal, dtype=float)]))
    scale = float(rammp_arm.GO_TO_SPEED_SCALE if speed_scale is None else speed_scale)
    secs = float(pl.t[-1]) / scale
    say(f"  {label}: cuRobo's plan checked ({len(pl.q)} samples, {secs:.1f} s at {scale:g} x), run as it is")
    points = [(float(t), q, qd) for t, q, qd in zip(pl.t, pl.q, pl.qd, strict=True)]
    res = ai.execute_joint_trajectory(list(rammp_arm.JOINTS), points, speed_scale=scale)
    if res.error_code != 0:
        say(f"  {label}: ended {res.error_string or res.error_code}")
    return res.error_code == 0


POSE_END_TOL = (0.01, 5.0)  # a checked pose move's plan must end the tool this close (m, deg) to the pose asked


POSE_MOVE_TRIES = 3  # cuRobo's plans vary: a checked pose move is asked this many times for one that passes


def checked_pose_move(ai, pos, quat, label, hold=0, speed_scale=None, gains=None, keep_shape=True, max_joint_deg=None):
    """cuRobo's plan to the tool pose (plan_to_pose from where the arm is, nothing moves), checked -- the arm keeping
    its shape (reconfiguration with `max_joint_deg`, unless not `keep_shape`), the floor, no joint swinging
    FREE_MOVE_MAX_SWING_RAD past its straight range, the tool ending within POSE_END_TOL of the pose -- then run
    EXACTLY (execute_joint_trajectory), under joint impedance `gains` if given. Asked POSE_MOVE_TRIES times: the first
    plan that passes runs. Raises Refused before anything moves when none does; returns whether the motion
    succeeded."""
    start = np.asarray(ai.raw_joints(), dtype=float)
    pl, why = pose_plan_problem(ai, start, pos, quat, hold, keep_shape, POSE_MOVE_TRIES, max_joint_deg)
    if why:
        raise Refused(f"{label}: {why} -- nothing moved.")
    return run_checked_plan(ai, pl, label, speed_scale, gains)


def run_checked_plan(ai, pl, label, speed_scale=None, gains=None):
    """Run cuRobo's checked plan `pl` EXACTLY (execute_joint_trajectory), under joint impedance `gains` if given;
    whether the motion succeeded."""
    qs = np.asarray(pl.q, dtype=float)
    scale = float(rammp_arm.GO_TO_SPEED_SCALE if speed_scale is None else speed_scale)
    secs = float(pl.t[-1]) / scale
    say(f"  {label}: cuRobo's plan checked ({len(qs)} samples, {secs:.1f} s at {scale:g} x), run as it is")
    points = [(float(t), q, qd) for t, q, qd in zip(pl.t, pl.q, pl.qd, strict=True)]
    kw = {} if gains is None else dict(control_mode=IMPEDANCE_MODE, gains=gains)
    res = ai.execute_joint_trajectory(list(rammp_arm.JOINTS), points, speed_scale=scale, **kw)
    if res.error_code != 0:
        say(f"  {label}: ended {res.error_string or res.error_code}")
    return res.error_code == 0


def pose_plan_problem(ai, start, pos, quat, hold=0, keep_shape=True, tries=1, max_joint_deg=None):
    """checked_pose_move's plan and checks, nothing moves: cuRobo's plan to the tool pose from joints `start` -- the
    arm keeping its shape (reconfiguration with `max_joint_deg`, unless not `keep_shape`), the floor, no joint
    swinging FREE_MOVE_MAX_SWING_RAD past its straight range, the tool ending within POSE_END_TOL of the pose. Asked
    `tries` times (cuRobo's plans vary) for one that passes. (plan, "") when one does, else (the last plan or None,
    why)."""
    start = np.asarray(start, dtype=float)
    pl, why = None, ""
    for _ in range(max(1, int(tries))):
        pl = ai.plan_to_pose(pos, quat, start, hold)
        for _ in range(CUROBO_RETRIES):
            if pl.success or _over_limit(pl.message) is None:
                break
            pl = ai.plan_to_pose(pos, quat, start, hold)
        if not pl.success or len(pl.q) == 0:
            why = f"cuRobo can't plan it ({pl.message})"
            continue
        qs = np.asarray(pl.q, dtype=float)
        why = (reconfiguration(start, qs[-1], max_joint_deg) if keep_shape else "") or path_below_floor(qs)
        if not why:
            off = np.abs(np.array([short_delta(start, q) for q in qs]))
            swing = off.max(axis=0) - np.abs(short_delta(start, qs[-1]))
            if float(swing.max()) > FREE_MOVE_MAX_SWING_RAD:
                j = int(np.argmax(swing))
                why = f"cuRobo's path swings J{j + 1} {np.degrees(swing[j]):.0f} deg past a straight move"
        if not why:
            p_end, o_end = make_sim()[1].pose(qs[-1])
            d_m = float(np.linalg.norm(p_end - np.asarray(pos, dtype=float)))
            d_deg = float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(o_end, np.asarray(quat, dtype=float))))))))
            if d_m > POSE_END_TOL[0] or d_deg > POSE_END_TOL[1]:
                why = f"cuRobo's plan ends the tool {d_m * 100:.1f} cm / {d_deg:.1f} deg off the pose"
        if not why:
            return pl, ""
    return pl, why


_LIMIT_RE = re.compile(r"(\w+) (velocity|acceleration|jerk) ([\d.]+) > limit ([\d.]+)")
_LIMIT_POWER = {"velocity": 1, "acceleration": 2, "jerk": 3}  # how each scales with the motion's speed


def _over_limit(message):
    """(joint, quantity, value, limit) when cuRobo's post-validation refused a plan for a velocity, acceleration or
    jerk over its limit -- its plans for plan_to_joints run at full speed -- else None."""
    m = _LIMIT_RE.search(str(message))
    if m is None or "VALIDATION" not in str(message):
        return None
    return m.group(1), m.group(2), float(m.group(3)), float(m.group(4))


def _passes_slowed(message):
    """Why a leg refused only for a velocity / acceleration / jerk over its limit passes anyway, or None: plan_to_joints
    plans at full speed, the motion runs at rammp_arm.GO_TO_SPEED_SCALE, which scales velocity by it, acceleration by
    its square and jerk by its cube. That leg's path is then unchecked for a detour."""
    over = _over_limit(message)
    if over is None:
        return None
    joint, kind, value, limit = over
    scale = float(rammp_arm.GO_TO_SPEED_SCALE)
    slowed = value * scale ** _LIMIT_POWER[kind]
    if slowed > limit:
        return None
    return (
        f"cuRobo's full-speed plan has {joint} {kind} {value:g} > {limit:g}; the motion runs at {scale:g} x speed: "
        f"{slowed:.2f}, within it (passed; this leg's path not checked for a detour)"
    )


def through_curobo(ai, path, label, speed_scale=None, gains=None):
    """Move along a checked joint path as one cuRobo motion, stiff or (`gains`) compliant."""
    return through_curobo_knots(ai, curobo_knots(path), label, speed_scale, gains)


def through_curobo_knots(ai, knots, label, speed_scale=None, gains=None, speeds=None, blends=None):
    """One cuRobo motion through joints `knots`, passing each but the last within CUROBO_BLEND_M (not stopping).
    `speeds`: a speed_scale per knot (None = `speed_scale`), e.g. a slow push inside the motion. `blends`: a blend
    radius per knot instead (0 = reach it exactly, e.g. where a push must start); the last always stops."""
    speeds = speeds if speeds is not None else [speed_scale] * len(knots)
    blends = blends if blends is not None else [CUROBO_BLEND_M] * len(knots)
    targets = [
        JointTarget(tuple(q), 0.0 if k == len(knots) - 1 else bl, sp)
        for k, (q, sp, bl) in enumerate(zip(knots, speeds, blends, strict=True))
    ]
    go_to = {} if gains is None else dict(control_mode=IMPEDANCE_MODE, gains=gains)
    return ai.go_to_joint_configs(targets, label, **go_to)


def gripper_on_the_move(ai, position, label, **kw):
    """Start ai.gripper(position, **kw) in a thread so the arm can move meanwhile (self.arm may be called from any
    thread during an action; the gripper is its own setpoint stream). Returns wait(): blocks until the gripper call
    has ended, re-raises what it raised (Cancelled, Refused), returns its result."""
    out = {}

    def run():
        try:
            out["ok"] = ai.gripper(position, **kw)
        except BaseException as e:  # noqa: BLE001 -- handed to wait(), which re-raises it in the caller's thread
            out["error"] = e

    t = threading.Thread(target=run, name=f"gripper: {label}", daemon=True)
    say(f"{label}: gripper to {position:g} while the arm moves ...")
    t.start()

    def wait():
        t.join()
        if "error" in out:
            raise out["error"]
        return out.get("ok")

    return wait


def release_and_back_off(ai, back_off_m, execute, save_path=None, park=True, pre_park_x=None, curobo=False):
    """Record the current grasp, open the gripper, back straight off along -approach, then
    (park=True) cuRobo (go_to_joint_config) to the hand-placed park joints -- the end
    position of both the open and the close task. `pre_park_x`: before the park move, go
    straight along world x (toward the base) to this x, same height and orientation -- on
    the open task this clears the open door's free edge, which sticks out toward the arm.
    The straight legs are planned and door-checked, and the park goal checked, before
    anything moves; a failed check means nothing is commanded. cuRobo's path to park is
    not previewed and its world has no door in it -- the straight legs get the hand out
    of the door's way first. `curobo`: the straight legs go through cuRobo too, every leg asked of it
    (curobo_path_problem) before the gripper opens, and with park they and the park move are ONE blended
    cuRobo motion (through_curobo_knots), not stopping between."""
    save_path = Path(save_path or LAST_GRASP_FILE)
    st = ai.get_state()
    ee = list(st["ee_pos"])
    grasp_pos, quat = np.array(ee[:3]), tuple(ee[3:7])
    approach = R.from_quat(quat).as_matrix()[:, 2]
    q0 = np.array(st["position"], dtype=float)
    say(f"current EE {np.round(grasp_pos, 4)}  gripper {float(st.get('gripper_pos')):.4f}")
    scene, rb = make_sim()
    plan = plan_straight_line(scene, rb, grasp_pos, quat, -approach, back_off_m, q0, "back-off")
    if plan is None:
        say("back-off plan fails a gate -- NOT releasing, arm untouched.")
        return False
    q_park = load_park() if park else None
    legs = [("back-off", plan)]
    if park:
        if not DOOR_FILE.exists():
            say(f"Need {DOOR_FILE} to plan the move to park -- NOT releasing.")
            return False
        bodies, info = add_door_model(scene, rb, json.loads(DOOR_FILE.read_text()), grasp_pos)
        say(f"door model: open {abs(info['open_deg']):.1f} deg, free edge at {np.round(info['free_edge'], 3)}")
        worst = min((clearance(rb, q, bodies) for q in plan), key=lambda w: w[0])
        say(f"  back-off worst clearance {worst[0] * 100:.1f} cm ({worst[1]})")
        q_end = plan[-1]
        if pre_park_x is not None:
            start = grasp_pos - approach * back_off_m
            dx = pre_park_x - start[0]
            if dx < 0:
                leg = plan_straight_line(scene, rb, start, quat, np.array([-1.0, 0, 0]), -dx, q_end, "toward base")
                if leg is None:
                    say("toward-base leg fails a gate -- NOT releasing, arm untouched.")
                    return False
                worst = min((clearance(rb, q, bodies) for q in leg), key=lambda w: w[0])
                say(f"  toward-base leg worst clearance {worst[0] * 100:.1f} cm ({worst[1]})")
                if worst[0] < MIN_CLEAR:
                    say("toward-base leg too close to the door -- NOT releasing, arm untouched.")
                    return False
                legs.append(("toward base", leg))
        if check_curobo_goal(scene, rb, "to park", q=q_park, bodies=bodies) is None:
            say("park goal fails a check -- NOT releasing, arm untouched.")
            return False
    if curobo:
        start = q0
        for label, leg in legs:
            why = curobo_path_problem(ai, start, leg, label)
            if why:
                say(f"{why} -- NOT releasing, arm untouched.")
                return False
            start = leg[-1]
    if not execute:
        say(f"\nDRY RUN (release + back-off{' + park' if park else ''}) -- nothing commanded.")
        return True

    back_off_pos = grasp_pos - approach * back_off_m
    Path(save_path).write_text(
        json.dumps(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "grasp_pos": grasp_pos.tolist(),
                "quat": list(quat),
                "grasp_joints": q0.tolist(),
                "back_off_pos": back_off_pos.tolist(),
                "back_off_joints": plan[-1].tolist(),
                "back_off_m": back_off_m,
            },
            indent=1,
        )
    )
    say(f"grasp recorded to {save_path} (for a later --phase regrasp)")

    say("Releasing ...")
    ai.execute_command(OpenGripperCommand())
    time.sleep(1.0)
    if curobo and hasattr(ai, "go_to_joint_configs"):
        knots = [q for _, leg in legs for q in curobo_knots(leg)] + ([q_park] if park else [])
        label = " + ".join([lb for lb, _ in legs] + (["park"] if park else []))
        say(f"  {label}: one cuRobo motion through {len(knots)} targets ...")
        ok = through_curobo_knots(ai, knots, label)
        if not ok:
            say(f"  {label}: cuRobo stopped ({getattr(ai, 'last_error', '')}) -- stopping here.")
    else:
        ok = all(execute_joint_plan(ai, leg, label) for label, leg in legs)  # all(): stops at the first that fails
        if ok and park:
            ok = ai.go_to_joint_config(q_park, "to park")
            say(f"  park: {'reached' if ok else 'did NOT reach the park joints'}")
    final = ai.get_state()
    say(
        f"\nRELEASE {'DONE' if ok else 'STOPPED EARLY'}. final EE:",
        np.round(final["ee_pos"][:3], 4),
        "gripper:",
        final.get("gripper_pos"),
    )
    return ok


# ---------------------------------------------------------------------------
# Door model (for collision-checking moves near the OPEN door)
# ---------------------------------------------------------------------------
# tool_frame -> fingertip distance along the approach axis, measured twice on 09-23
# (6.3 / 6.1 cm); the closed door face sits just past the fingertips at grasp time.
FINGERTIP_PAST_TOOL = 0.062
# door-size defaults, for a door file without its own (door_dims, add_door_model): set by the capability (configure)
DOOR_T = None  # door thickness
DOOR_PAST_HANDLE = None  # free edge beyond the handle
DOOR_Z = None  # door (bottom, top), the fallback when the door file has no door_z
DOOR_Z_HANDLE_Z = None  # closed-handle z when DOOR_Z was set; the model keeps DOOR_Z's extent relative to the handle
BODY_DEPTH = None
DOOR_Z_MARGIN = 0.02  # padding on the detected door_z range (door file), when the grasp saved one
MIN_CLEAR = 0.03  # every arm/gripper link must stay >= 3 cm from door/body


def door_dims(door):
    """(face behind the closed grasp, door thickness, free edge past the grasp, body depth) for the door model. A door
    file may carry its own (DOOR_MODEL in open_door, which the open writes into it); else the constants above."""
    return (
        float(door.get("face_behind_grasp", FINGERTIP_PAST_TOOL + 0.005)),
        float(door.get("door_t", DOOR_T)),
        float(door.get("door_past_handle", DOOR_PAST_HANDLE)),
        float(door.get("body_depth", BODY_DEPTH)),
    )


def save_door_geometry(door_file=None, **kw):
    """Merge door facts into `door_file` (default DOOR_FILE): closed_normal / closed_grasp_pos (grasp), hinge
    (swing)."""
    door_file = door_file or DOOR_FILE
    d = json.loads(door_file.read_text()) if door_file.exists() else {}
    d.update({k: (np.asarray(v).tolist() if not isinstance(v, str) else v) for k, v in kw.items()})
    d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    door_file.write_text(json.dumps(d, indent=1))


def _yaw_about(center, xy, ang):
    d = np.asarray(xy) - center[:2]
    c, s_ = np.cos(ang), np.sin(ang)
    return center[:2] + np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])


def door_model_boxes(door, open_grasp_pos):
    """The door slab (rotated to its current opening) and the appliance's fixed body as boxes in the arm's frame
    (arm_mount: the door file's), each (name, centre xyz, half extents, yaw about z), plus info -- what add_door_model
    puts in the sim and door_in_curobo in cuRobo's world, so the two check the same door."""
    n = np.asarray(door["closed_normal"], float)
    n[2] = 0
    n /= np.linalg.norm(n)  # out of the door
    g0, hinge = np.asarray(door["closed_grasp_pos"], float), np.asarray(door["hinge"], float)
    t = np.cross([0, 0, 1.0], n)  # along the door
    if np.dot(t, hinge - g0) < 0:
        t = -t  # t points to the hinge
    face_off, door_t, past_handle, body_depth = door_dims(door)
    face = g0[:2] - n[:2] * face_off  # a point on the closed face
    lo = float(np.dot(g0[:2] - face, t[:2])) - past_handle  # free edge (along t)
    hi = float(np.dot(hinge[:2] - face, t[:2]))  # hinge edge
    ang = np.arctan2(*(np.asarray(open_grasp_pos[:2]) - hinge[:2])[::-1]) - np.arctan2(*(g0[:2] - hinge[:2])[::-1])
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    if door.get("door_z"):
        # the detector's door-face height range (1st/99th pct of the plane fit), saved by the grasp
        z_lo, z_hi = door["door_z"][0] - DOOR_Z_MARGIN, door["door_z"][1] + DOOR_Z_MARGIN
    else:
        dz = float(door["closed_handle"][2]) - DOOR_Z_HANDLE_Z if "closed_handle" in door else 0.0
        z_lo, z_hi = DOOR_Z[0] + dz, DOOR_Z[1] + dz
    zc, zh = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2
    yaw_closed = float(np.arctan2(n[1], n[0]))
    mid = (lo + hi) / 2
    door_c_closed = face + t[:2] * mid - n[:2] * door_t / 2
    door_c = _yaw_about(hinge, door_c_closed, ang)
    body_c = face + t[:2] * mid - n[:2] * (door_t + body_depth / 2)
    boxes = [
        ("door", (door_c[0], door_c[1], zc), (door_t / 2, (hi - lo) / 2, zh), yaw_closed + float(ang)),
        (f"{APPLIANCE} body", (body_c[0], body_c[1], zc), (body_depth / 2, (hi - lo) / 2, zh), yaw_closed),
    ]
    free_edge = _yaw_about(hinge, face + t[:2] * lo, ang)
    return boxes, {"open_deg": float(np.degrees(ang)), "free_edge": free_edge}


def add_sim_box(scene, rb, centre, half, yaw):
    """An obstacle box at `centre` (arm frame) turned `yaw` about z, for clearance(): an arm_model.Box."""
    return Box(tuple(float(v) for v in centre), tuple(float(v) for v in half), float(yaw))


def add_door_model(scene, rb, door, open_grasp_pos):
    """The door slab (rotated to its current opening) and the appliance's fixed body as obstacle boxes
    (door_model_boxes). Returns (bodies, info) -- bodies is [(Box, name)], for clearance()."""
    boxes, info = door_model_boxes(door, open_grasp_pos)
    return [(add_sim_box(scene, rb, centre, half, yaw), name) for name, centre, half, yaw in boxes], info


# --- the door in cuRobo's world: cuRobo plans AROUND the door model, instead of the sim refusing its paths after ---
# set_world takes a world YAML by path, read inside the planner's container. That container (sheppy's) mounts the host's
# robot config, SHEPPY_ROBOT_CONFIG, as /config/robot: a capability run natively on the host (ros2 run / launch) writes
# the world under it and the planner reads it there -- no mount of our own. Elsewhere (a container): set
# RAMMP_DOOR_WORLD_DIR to a directory the planner sees as RAMMP_DOOR_WORLD_PLANNER_DIR.
SHEPPY_ROBOT_CONFIG = Path(
    os.environ.get("SHEPPY_ROBOT_CONFIG", "~/sheppy-local/december_2026/config/robots/current")
).expanduser()
DOOR_WORLD_DIR = Path(os.environ.get("RAMMP_DOOR_WORLD_DIR", SHEPPY_ROBOT_CONFIG / "curobo" / "door_world"))
DOOR_WORLD_PLANNER_DIR = os.environ.get("RAMMP_DOOR_WORLD_PLANNER_DIR", "/config/robot/curobo/door_world")
CUROBO_BOOT_WORLD = "world.yaml"  # planner.yaml's own world, next to its config: what set_world goes back to
# a pushed world replaces world.yaml's floor, so the door world lists the planner's own floor layer (the chair and
# user are planner.yaml's layers: in every world, pushed ones too)
CUROBO_FLOOR_LAYER = "/config/robot/curobo/layer_mebot_floor.yaml"
DOOR_WORLD_PAD_M = 0.015  # each door box this much bigger per side in cuRobo than in the sim (on top of the
# planner's own world_padding): its plans must still pass the sim's MIN_CLEAR from the unpadded model after


def door_world(boxes, pad=DOOR_WORLD_PAD_M):
    """A cuRobo world (the dict a world YAML holds) of the planner's floor and `boxes` ((name, centre, half, yaw),
    door_model_boxes' form, the arm's frame = arm_mount), each grown `pad` per side."""
    obstacles = [
        {
            "name": "door_model_" + re.sub(r"\W+", "_", name).strip("_"),
            "position": [round(float(v), 5) for v in centre],
            "dims": [round(2 * (float(h) + pad), 5) for h in half],
            "rpy_deg": [0.0, 0.0, round(float(np.degrees(yaw)), 3)],
        }
        for name, centre, half, yaw in boxes
    ]
    return {"base_frame": "arm_mount", "layers": [CUROBO_FLOOR_LAYER], "obstacles": obstacles}


@contextmanager
def door_in_curobo(ai, boxes, label):
    """cuRobo's world with the door model in it (door_world(boxes)) for the `with` block, then the planner's boot world
    again (CUROBO_BOOT_WORLD) -- also on an exception. Yields whether the door went in; when it didn't (says why),
    cuRobo's world is as it was."""
    if not hasattr(ai, "set_world"):
        say(f"  {label}: no set_world on this arm interface -- the door can't go into cuRobo's world")
        yield False
        return
    name = f"door_world_{os.getpid()}.yaml"
    if not DOOR_WORLD_DIR.parent.is_dir():  # not where the planner looks: its set_world would find no file
        say(
            f"  {label}: no {DOOR_WORLD_DIR.parent} here (not on the host with sheppy's robot config?) -- set "
            "SHEPPY_ROBOT_CONFIG, or RAMMP_DOOR_WORLD_DIR + RAMMP_DOOR_WORLD_PLANNER_DIR"
        )
        yield False
        return
    try:
        DOOR_WORLD_DIR.mkdir(parents=True, exist_ok=True)
        (DOOR_WORLD_DIR / name).write_text(yaml.safe_dump(door_world(boxes), sort_keys=False))
    except OSError as e:
        say(f"  {label}: can't write the door world in {DOOR_WORLD_DIR} ({e})")
        yield False
        return
    ok, msg = ai.set_world(f"{DOOR_WORLD_PLANNER_DIR.rstrip('/')}/{name}")
    if not ok:
        say(f"  {label}: the planner refused the door world ({msg})")
        (DOOR_WORLD_DIR / name).unlink(missing_ok=True)
        yield False
        return
    say(f"  {label}: door in cuRobo's world ({', '.join(b[0] for b in boxes)}, +{DOOR_WORLD_PAD_M * 100:.1f} cm)")
    try:
        yield True
    finally:
        ok, msg = ai.set_world(CUROBO_BOOT_WORLD)
        if ok:  # the file is there exactly while cuRobo has the door (collision_viz shows it then)
            (DOOR_WORLD_DIR / name).unlink(missing_ok=True)
        else:  # every later plan would dodge a door that has since moved: say so loudly
            say(
                f"  !! {label}: cuRobo's world NOT put back to {CUROBO_BOOT_WORLD} ({msg}) -- it still has the door "
                "in it: `ros2 service call /rammp_curobo/set_world rammp_curobo_interfaces/srv/SetWorld "
                f'"{{world: {CUROBO_BOOT_WORLD}}}"`'
            )


class DoorFrame:
    """Door geometry from DOOR_FILE. Door-frame coordinates (u, v) at opening angle `deg`:
    u along the door from the hinge toward the free edge, v along the outward normal
    (toward the arm when closed) measured from the slab's mid-plane."""

    def __init__(self, door, sign):
        self.door, self.sign = door, float(sign)
        self.hinge = np.asarray(door["hinge"], float)
        self.g0 = np.asarray(door["closed_grasp_pos"], float)
        n = np.asarray(door["closed_normal"], float)
        n[2] = 0
        self.n = n[:2] / np.linalg.norm(n[:2])
        e = self.g0[:2] - self.hinge[:2]
        e -= self.n * np.dot(e, self.n)
        self.e = e / np.linalg.norm(e)  # hinge -> free edge
        face_off, self.t, past_handle, _ = door_dims(door)  # as add_door_model
        face = self.g0[:2] - self.n * face_off
        self.mid = float(np.dot(face - self.hinge[:2], self.n)) - self.t / 2
        self.u_handle = float(np.dot(self.g0[:2] - self.hinge[:2], self.e))
        self.length = self.u_handle + past_handle

    def _rot(self, deg):
        a = self.sign * np.radians(deg)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def angle_of(self, xy):
        """Opening angle (deg, positive = open) of a point that was on the grasp line when closed."""
        d = np.asarray(xy[:2]) - self.hinge[:2]
        g = self.g0[:2] - self.hinge[:2]
        a = np.arctan2(g[0] * d[1] - g[1] * d[0], g @ d)
        return float(np.degrees(a) * self.sign)

    def pos(self, u, v, deg, z):
        xy = self.hinge[:2] + self._rot(deg) @ (self.e * u + self.n * (self.mid + v))
        return np.array([xy[0], xy[1], z])

    def quat(self, deg, y_up, yaw_deg=0.0, roll_deg=0.0):
        """Approach (tool z) horizontal, along the door toward the hinge; tool y vertical
        (`y_up` = +-1, keep whatever the wrist has now). `yaw_deg` turns the approach off the
        door line, `roll_deg` rolls the hand about it (which face of the gripper meets the door)."""
        z2 = self._rot(deg + yaw_deg * self.sign) @ -self.e
        zc = np.array([z2[0], z2[1], 0.0])
        yc = np.array([0.0, 0.0, y_up])
        m = np.column_stack([np.cross(yc, zc), yc, zc])
        return (R.from_matrix(m) * R.from_euler("z", roll_deg, degrees=True)).as_quat()

    def bodies_point(self, deg):
        """The closed grasp point swung to `deg` -- what add_door_model reads the angle from."""
        return np.array([*(self.hinge[:2] + self._rot(deg) @ (self.g0[:2] - self.hinge[:2])), self.g0[2]])

    def bodies(self, scene, rb, deg):
        """Door slab at `deg` + the appliance's body, as [(Box, name)]."""
        return add_door_model(scene, rb, self.door, self.bodies_point(deg))[0]


def record_door_angle(ee, door_file=None):
    """After a pull (hand still on the handle): save the opening sense and angle to `door_file`
    (default DOOR_FILE), which the close reads."""
    door = json.loads((door_file or DOOR_FILE).read_text())
    a = DoorFrame(door, 1.0).angle_of(ee)
    save_door_geometry(door_open_deg=abs(a), open_sign=float(np.sign(a) or 1.0), door_file=door_file)
    say(f"door file: open ~{abs(a):.1f} deg (sense {int(np.sign(a) or 1):+d})")


def drop(rb, bodies):
    """Nothing to remove: the boxes are values (kept for the callers that dropped PyBullet bodies)."""


# ---------------------------------------------------------------------------
# The chair and user: cuRobo's (its world's chair layers). A joint path run exactly as planned here is validated by it
# ---------------------------------------------------------------------------
def curobo_validates(ai, q0, knots, label, cart_lengths=None, speed_scale=1.0):
    """Why cuRobo rejects the joint path self.arm.move_joints would send through `knots` from joints `q0` (timed as it
    times them, rammp_arm.joint_trajectory), or "": a one-leg blend checks every state against the planner's world --
    the chair and user layers, the floor -- the arm itself and the joint limits, as the fridge validates its own legs
    (nothing moves). A rejection only for a speed over cuRobo's limit passes (said): what it checks here is where the
    arm goes. Without a planner (no `blend`: a bare bench arm) it can't check, and passes."""
    if not hasattr(ai, "blend"):
        return ""
    samples = rammp_arm.joint_trajectory(np.asarray(q0, float), list(knots), cart_lengths, speed_scale)
    if samples is None:
        return ""
    t, q, qd = samples
    res = ai.blend([SimpleNamespace(t=np.asarray(t), q=np.asarray(q), qd=np.asarray(qd))], [])
    if res.success:
        return ""
    if _over_limit(res.message) is not None:
        say(f"  {label}: cuRobo's check refused it only for a speed over its limit ({res.message}) -- passed")
        return ""
    return f"cuRobo rejects {label} ({str(res.message)[:200]})"


CHAIR_DIR = Path(__file__).parent / "chair"  # copied from rammp-deployments by `make sync-chair`, see chair/SOURCE
CHAIR_LAYERS = ("platform", "seat", "person")  # as flatbot's planner.yaml; its floor is not a chair part


def load_chair_boxes():
    """Every chair and user box as (name, centre in the arm's base frame, half extents), padded as cuRobo pads them
    (for drawing them: cuRobo checks them). A missing or unreadable file raises."""
    boxes = []
    for layer in CHAIR_LAYERS:
        data = yaml.safe_load((CHAIR_DIR / f"layer_mebot_{layer}.yaml").read_text())
        pad = float(data["meta"]["padding_assumed_m"])
        for o in data["obstacles"]:
            half = [float(d) / 2 + pad for d in o["dims"]]
            boxes.append((f"{layer}/{o['name']}", tuple(float(v) for v in o["position"]), half))
    return boxes


# ---------------------------------------------------------------------------
# Wrist-camera frames for the run log (and the open task's detection frames)
# ---------------------------------------------------------------------------
_camera = None  # perception.WristCamera: set by the node (set_camera), or made on the bench


def set_camera(camera):
    global _camera
    _camera = camera


def camera():
    """The wrist camera. Under the engine the node set it; on the bench (rammp_arm.RammpArm) one is made
    on that arm's node with the default topics."""
    if _camera is None:
        raise Refused("no wrist camera (the node sets one with set_camera)")
    return _camera


def grab_frames(n=1, gap_s=0.3):
    """`n` RGB-D frames from the wrist camera, each with K and base_T_cam (tf2 arm base <- camera).
    Reads topics only -- nothing is commanded."""
    frames = []
    for i in range(n):
        if i:
            time.sleep(gap_s)
        f = camera().grab()
        if f["base_T_cam"] is None:
            raise RuntimeError("no arm base <- camera transform")
        frames.append(f)
    return frames


def save_frame(frame, tag, out_dir, result=None, arm_state=None):
    """<time>_<tag>: _rgb.jpg + .npz (uint16 depth mm, K, base_T_cam, arm joints/tool pose if given) +
    _result.json. ~0.3 MB a frame. Returns the path stem (None if it could not save)."""
    try:
        import cv2

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(frame.get('time', time.time())))}_{tag}"
        if frame.get("rgb") is not None:
            cv2.imwrite(str(stem) + "_rgb.jpg", frame["rgb"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        depth = np.nan_to_num(np.asarray(frame["depth_mm"], dtype=np.float32), nan=0.0)
        extra = {}
        if arm_state is not None:
            extra = {"joints": np.asarray(arm_state["position"], float), "ee": np.asarray(arm_state["ee_pos"], float)}
        if frame.get("base_T_cam") is not None:
            extra["base_T_cam"] = frame["base_T_cam"]
        if frame.get("handle_mask") is not None:  # the SAM 3 mask the detection used
            extra["handle_mask"] = np.asarray(frame["handle_mask"], bool)
        np.savez_compressed(
            str(stem) + ".npz", depth_mm=np.clip(depth, 0, 65535).astype(np.uint16), K=frame["K"], **extra
        )
        if result is not None:
            Path(str(stem) + "_result.json").write_text(json.dumps(_jsonable(result), indent=1))
        return stem
    except Exception as e:  # never let debugging output stop a run
        say(f"  (could not save the frame: {e})")
        return None


# ---------------------------------------------------------------------------
# Run log: one directory per run
# ---------------------------------------------------------------------------
STREAM_HZ = 25.0  # arm state sampling while a run is on
CMD_EVERY = 5  # keep every 5th commanded trajectory sample (20 ms -> 100 ms)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), 5).tolist()
    if isinstance(x, np.floating | np.integer | np.bool_):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def arm_snapshot(st):
    return {
        "joints_deg": np.degrees(np.asarray(st["position"], float)),
        "ee": np.asarray(st["ee_pos"], float),
        "gripper": float(st["gripper_pos"]),
        "effort": np.asarray(st.get("effort", []), float),
    }


class RunLog:
    """Everything a run leaves behind, in RUNS_DIR/<time>_<what>/:

    run.json  -- args and a timestamped event list: arm state at start/end, the door used
                 for the close (source + values) and its decoded keypoints, every trajectory goal sent
                 (commanded joints every CMD_EVERY samples, ok or not, measured end joints), every gripper
                 command, step outcomes. Rewritten after each event, so a crash keeps what came before.
    arm.npz   -- the arm sampled at STREAM_HZ for the whole run: t, joints, velocity, effort (torques),
                 tool pose, gripper (float32).
    frames    -- camera frames (save_frame: real depth, K, camera pose, arm joints): `frame(tag)`.
    Read-only towards the arm: it only wraps the arm's send/gripper calls to note them."""

    def __init__(self, what, args, ai):
        # read the arm before starting the sampler or wrapping anything: if this raises, nothing is left running
        start = arm_snapshot(ai.get_state())
        self.ai, self.t0 = ai, time.time()
        self.dir = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{what}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data = {
            "what": what,
            "args": args,
            "start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "events": [],
        }
        self._samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self._wrap(ai)
        self.event("start", arm=start)
        say(f"run log: {self.dir}")

    def event(self, kind, **kw):
        self.data["events"].append({"t": round(time.time() - self.t0, 3), "kind": kind, **_jsonable(kw)})
        try:
            (self.dir / "run.json").write_text(json.dumps(self.data, indent=1))
        except Exception as e:
            say(f"  (run log: could not write run.json: {e})")

    def frame(self, tag):
        """One RGB-D frame from the wrist camera, saved here with the arm's state (camera read only)."""
        try:
            f = grab_frames(1)[0]
            stem = save_frame(f, tag, self.dir, arm_state=self.ai.get_state())
            self.event("frame", tag=tag, path=stem)
        except Exception as e:  # a missing camera must not stop the run
            say(f"  (run log: no '{tag}' frame: {e})")
            self.event("frame_failed", tag=tag, error=str(e))

    def _sample(self):
        while not self._stop.is_set():
            try:
                st = self.ai.get_state()
                self._samples.append(
                    (
                        time.time() - self.t0,
                        st["position"],
                        st.get("velocity", np.zeros(7)),
                        st.get("effort", np.zeros(7)),
                        st["ee_pos"],
                        st["gripper_pos"],
                    )
                )
            except Exception:
                pass
            time.sleep(1.0 / STREAM_HZ)

    def _wrap(self, ai):
        """Note every trajectory goal and gripper command the arm sends (the calls themselves unchanged).
        The wrappers are instance attributes over the class's methods; close() removes them, since under
        the engine the same self.arm serves every run."""
        self._wrapped = []
        if hasattr(ai, "_send"):
            send = ai._send

            def logged_send(t, q, qd, *a, **kw):
                start = round(time.time() - self.t0, 3)
                ok = send(t, q, qd, *a, **kw)
                q = np.asarray(q)
                self.event(
                    "trajectory",
                    start=start,
                    ok=ok,
                    planned_s=float(t[-1]),
                    n_points=len(t),
                    impedance=kw.get("impedance", a[0] if a else None),
                    q_cmd_deg=np.degrees(np.vstack([q[::CMD_EVERY], q[-1:]])),
                    end=arm_snapshot(ai.get_state()),
                )
                return ok

            ai._send = logged_send
            self._wrapped.append("_send")
        if hasattr(ai, "gripper"):
            grip = ai.gripper

            def logged_gripper(position, *a, **kw):
                r = grip(position, *a, **kw)
                self.event(
                    "gripper", cmd=position, force=kw.get("force"), measured=float(ai.get_state()["gripper_pos"])
                )
                return r

            ai.gripper = logged_gripper
            self._wrapped.append("gripper")

    def close(self, ok):
        for name in self._wrapped:
            self.ai.__dict__.pop(name, None)
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.event("end", ok=bool(ok), arm=arm_snapshot(self.ai.get_state()))
        except Exception as e:
            self.event("end", ok=bool(ok), error=str(e))
        if self._samples:
            t, q, dq, tau, ee, g = zip(*self._samples, strict=False)
            np.savez_compressed(
                self.dir / "arm.npz",
                t=np.asarray(t, np.float32),
                joints=np.asarray(q, np.float32),
                velocity=np.asarray(dq, np.float32),
                effort=np.asarray(tau, np.float32),
                ee=np.asarray(ee, np.float32),
                gripper=np.asarray(g, np.float32),
            )
        say(f"run log: {self.dir} ({len(self._samples)} arm samples)")
