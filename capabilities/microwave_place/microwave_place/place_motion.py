"""The arm side of placing the held container in the open microwave: planning, checking and running it.

From Walter Hsieh's place_container (plan.py and lower.py, WalterHsieh0713/rammp-kitchen-capability-engines
feature/7-place-container), unchanged: the hold model, the arm model, the checked joint moves and straight
lines, the straight line down in the driver's joint impedance mode with the contact stop, the hold, the
recording. Added for the microwave:

  plan_line / check_line: a straight line of joint waypoints, and the check that every pose is on it;
  "Into the microwave": the oven and what stands in front of it as obstacle boxes, the hand's heading into
    the oven, the checks on the arm model (clearance per link, tilt, joint travel) of cuRobo's transit and
    the straight lines, the release lift and the back-out with the fingers opened to RELEASE_OPEN;
  "The place on the arm": the heading search, the plan's summary, and running the checked plan
    (run_place), retreat, and reset's check.

Everything is in arm_mount, the driver's gravity-aligned frame: z true up, x forward, y left, origin on
the arm base axis. The container's centre is `model.reach` ahead of the tool frame along the hand's
pointing direction (tool z): flange --TOOL_Z (12 cm)--> tool frame --tip_ahead--> fingertips
--BOWL_RADIUS--> container centre.
"""

import csv
import time

import numpy as np
import rammp_arm
from capability_engine import sim
from capability_engine.arm import HOLD_LEVEL, IMPEDANCE, LATEST_WINS, POSITION, SUCCESSFUL
from capability_engine.driver_state import EStopped
from capability_engine.driver_state import Refused as ArmRefused
from rammp_arm import CONTINUOUS, JOINTS, TOOL_Z, _unwrap_from
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from . import microwave_detection

# --- the container and how it is held ---
BOWL_RADIUS = 0.055  # m, centre to the wall the fingertips are pushed in to
FLANGE_TO_TIP = 0.17955  # m, flange face to the closed fingertips (Gen3 + 2F-85; not ruler-measured)
PLACE_PITCH_DEG = 5.0  # the hand points this many deg down while placing
# --- the motion ---
PRESS_BELOW = 0.10  # m, the line down is aimed this far below where the bottom meets the table: in
# impedance the hand lags its command by a few cm, and the table, not the command, stops it
MIN_TOOL_HEIGHT = 0.02  # m, the tool frame is never targeted lower than this above the table
PATH_STEP = 0.002  # m, between the line's joint waypoints
RETREAT_STEP = 0.004  # m, between the retreat's joint waypoints
MAX_BOWL_TILT_DEG = 10.0  # container tilt allowed anywhere along a joint move
# --- checks ---
IK_TOL = 0.005  # m
IK_ROT_TOL_DEG = 1.0
MAX_STEP_JUMP_DEG = 120.0  # one joint move
UP = np.array([0.0, 0.0, 1.0])


class Refused(Exception):
    """A plan or check that does not hold: nothing more is commanded."""


def wrap(rad):
    return (np.asarray(rad) + np.pi) % (2 * np.pi) - np.pi


def wrap_deg(rad):
    """Angle difference in radians -> degrees in [-180, 180)."""
    return np.degrees(wrap(rad))


def hold_reach(flange_to_tip=FLANGE_TO_TIP):
    """Container centre in front of the tool frame (m)."""
    return BOWL_RADIUS + flange_to_tip - TOOL_Z


def camera_on_left(R):
    """The hand's camera is on its left (tool x points down rather than up)."""
    return bool(R[:, 0] @ UP < 0)


def bowl_tilt_deg(R, pitch_deg=PLACE_PITCH_DEG):
    """How far the container is from how it should be held: tool x straight up (camera right) or
    down (camera left), leaned forward by pitch_deg when the hand points down."""
    h = np.array([R[0, 2], R[1, 2], 0.0])
    h = h / np.linalg.norm(h) if np.linalg.norm(h) > 1e-6 else np.zeros(3)
    p = np.radians(pitch_deg)
    want = (UP * np.cos(p) + h * np.sin(p)) * (-1.0 if camera_on_left(R) else 1.0)
    return float(np.degrees(np.arccos(np.clip(R[:, 0] @ want, -1.0, 1.0))))


class Model:
    """FK and IK of the arm on the process's shared sim copy (capability_engine.sim: the driver's
    model, rooted at arm_mount). fk() gives 4x4 poses in arm_mount of the tool frame (the sim's ee
    link, arm_tool_frame), the flange or the fingertips."""

    def __init__(self, flange_to_tip=FLANGE_TO_TIP, pitch_deg=PLACE_PITCH_DEG):
        import pybullet as p

        self.scene, self.rb = sim.make_sim()
        self.flange_to_tip = float(flange_to_tip)
        self.pitch = float(pitch_deg)
        self.reach = hold_reach(self.flange_to_tip)
        info = [p.getJointInfo(self.rb.robot_id, j, physicsClientId=self.rb.physics_client_id) for j in sim.ARM]
        # (lower, upper) in rad; continuous joints come back as lower > upper
        self.limits = [(float(i[8]), float(i[9])) for i in info]

    def fk(self, q, link="tool"):
        pos, quat = sim.tool_pose(self.scene, self.rb, q)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        offset = {"tool": 0.0, "flange": -TOOL_Z, "tip": self.flange_to_tip - TOOL_Z}[link]
        T[:3, 3] += offset * T[:3, 2]
        return T

    def tilt(self, q):
        return bowl_tilt_deg(self.fk(q)[:3, :3], self.pitch)

    def limit_violations(self, q):
        bad = []
        for i, (v, (lo, hi)) in enumerate(zip(q, self.limits, strict=True)):
            if lo < hi and not (lo <= v <= hi):
                bad.append(f"J{i + 1} {np.degrees(v):.0f} deg (limit {np.degrees(lo):.0f}..{np.degrees(hi):.0f})")
        return bad

    def ik(self, pos, quat, seed, restarts=10, rot_weight=0.2, stay=0.0):
        """Joints putting the tool frame at (pos, quat), or None. stay > 0 pulls the answer toward
        the seed (m per rad): 7 joints for a 6-number pose, so without it consecutive small steps
        slide along the spare freedom. Of the solutions within IK_TOL / IK_ROT_TOL_DEG and the
        joint limits, the one that moves the joints least from the seed; continuous joints in
        [-pi, pi)."""
        seed = np.asarray(seed, dtype=float)
        target = Rotation.from_quat(quat)
        pos = np.asarray(pos, dtype=float)

        def residual(x):
            T = self.fk(x)
            rot = (Rotation.from_matrix(T[:3, :3]) * target.inv()).as_rotvec()
            return np.concatenate([T[:3, 3] - pos, rot_weight * rot, stay * wrap(x - seed)])

        lo = np.array([a if a < b else -2 * np.pi for a, b in self.limits])
        hi = np.array([b if a < b else 2 * np.pi for a, b in self.limits])
        rng = np.random.default_rng(0)
        best = None
        for k in range(restarts + 1):
            x0 = (
                np.clip(seed, lo + 1e-6, hi - 1e-6)
                if k == 0
                else rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi))
            )
            # PyBullet FK is single precision: the default finite-difference step is too small.
            r = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-4, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            q = r.x.copy()
            q[CONTINUOUS] = wrap(q[CONTINUOUS])
            T = self.fk(q)
            err = np.linalg.norm(T[:3, 3] - pos)
            rot = np.degrees((Rotation.from_matrix(T[:3, :3]) * target.inv()).magnitude())
            if err > IK_TOL or rot > IK_ROT_TOL_DEG or self.limit_violations(q):
                continue
            move = float(np.max(np.abs(wrap_deg(q - seed))))
            if best is None or move < best[0]:
                best = (move, q)
            if k == 0 and move < 30.0:
                break  # the seed's own solution is close: good enough
        return None if best is None else best[1]


def check_joint_move(model, q_a, q_b, table_z, samples=20, tilt_check=True):
    """Problems along a straight joint-space move: the container tilts (tilt_check: only with something
    in the hand), or the hand dips."""
    allowed = max(MAX_BOWL_TILT_DEG, model.tilt(q_a) + 1.0, model.tilt(q_b) + 1.0)
    delta = wrap(np.asarray(q_b) - np.asarray(q_a))
    for s in np.linspace(0.0, 1.0, samples):
        T = model.fk(np.asarray(q_a) + s * delta)
        tilt = bowl_tilt_deg(T[:3, :3], model.pitch)
        if tilt_check and tilt > allowed:
            return [f"the container tilts {tilt:.0f} deg"]
        if table_z is not None and T[2, 3] < table_z + MIN_TOOL_HEIGHT:
            return [f"the hand dips to {(T[2, 3] - table_z) * 100:.1f} cm above the table"]
    return []


def plan_segment(model, q_from, pos_to, quat_to, table_z, name, depth=0):
    """Joint waypoints from q_from to the pose, split in Cartesian halves until each joint move
    keeps the container level and above the table. [(name, q), ...]."""
    q_to = model.ik(pos_to, quat_to, q_from)
    if q_to is None:
        raise Refused(f"{name}: no IK solution for the tool at {np.round(pos_to, 3).tolist()}")
    jump = float(np.max(np.abs(wrap_deg(q_to - q_from))))
    problems = check_joint_move(model, q_from, q_to, table_z)
    if jump <= MAX_STEP_JUMP_DEG and not problems:
        return [(name, q_to)]
    if depth >= 4:
        raise Refused(f"{name}: " + ("; ".join(problems) or f"a joint moves {jump:.0f} deg"))
    T_from = model.fk(q_from)
    mid_pos = (T_from[:3, 3] + np.asarray(pos_to)) / 2
    rots = Rotation.from_matrix(np.stack([T_from[:3, :3], Rotation.from_quat(quat_to).as_matrix()]))
    mid_quat = Slerp([0, 1], rots)([0.5])[0].as_quat()
    first = plan_segment(model, q_from, mid_pos, mid_quat, table_z, name + "'", depth + 1)
    return first + plan_segment(model, first[-1][1], pos_to, quat_to, table_z, name, depth + 1)


def plan_descent(model, q_start, floor_z, step=PATH_STEP, log=None):
    """Joint targets every `step` straight down from q_start's tool pose to floor_z, orientation
    kept. Returns (tool z values, joint arrays); the first entry is q_start itself. Ends early (with
    a note to `log`) where the arm can't go further straight down, if it got at least 1 cm."""
    T = model.fk(q_start)
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    zs = np.arange(T[2, 3], floor_z - 1e-9, -step)
    qs, seed = [np.asarray(q_start, dtype=float)], np.asarray(q_start, dtype=float)
    for z in zs[1:]:
        q = model.ik([T[0, 3], T[1, 3], z], quat, seed, restarts=0, stay=0.01)
        if q is None or np.max(np.abs(wrap_deg(q - seed))) > 3.0 * step / PATH_STEP:
            got = T[2, 3] - zs[len(qs) - 1]
            if got >= 0.01:
                if log is not None:
                    log(f"note: the arm can only go {got * 100:.0f} cm straight down from here (reach / joint limits)")
                zs = zs[: len(qs)]
                break
            raise Refused(f"lower: can't go straight down from z={T[2, 3]:.3f}")
        qs.append(q)
        seed = q
    return np.asarray(zs), np.array(qs)


LINE_TOL = 0.01  # m: every pose of the reach-in within this of its line ...
LINE_TOL_DEG = 2.0  # ... and pointing within this of the oven's axis


def check_line(model, qs, a, b, axis, name):
    """Refused unless the tool at every joints of qs lies within LINE_TOL of the segment a-b and points
    (horizontally) within LINE_TOL_DEG of `axis`."""
    a, b, axis = np.asarray(a, float), np.asarray(b, float), np.asarray(axis, float)
    ab = b - a
    for i, q in enumerate(qs):
        T = model.fk(q)
        p = T[:3, 3]
        t = 0.0 if ab @ ab < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        off = float(np.linalg.norm(p - (a + t * ab)))
        h = np.array([T[0, 2], T[1, 2]])
        turn = float(np.degrees(np.arccos(np.clip(h @ axis[:2] / max(np.linalg.norm(h), 1e-9), -1.0, 1.0))))
        if off > LINE_TOL or turn > LINE_TOL_DEG:
            raise Refused(
                f"{name}: pose {i} has the tool at {np.round(p, 3).tolist()}, {off * 100:.1f} cm off its line "
                f"{np.round(a, 3).tolist()} -> {np.round(b, 3).tolist()} and {turn:.1f} deg off the oven's axis"
            )


def plan_line(model, q_start, delta, name="line", step=RETREAT_STEP):
    """Joint waypoints every `step` along the straight line `delta` (m) from q_start's tool pose,
    orientation kept; the first entry is q_start. Refused where the arm can't follow it."""
    T = model.fk(q_start)
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    length = float(np.linalg.norm(delta))
    n = max(1, int(np.ceil(length / step)))
    qs, seed = [np.asarray(q_start, dtype=float)], np.asarray(q_start, dtype=float)
    for k in range(1, n + 1):
        q = model.ik(T[:3, 3] + k / n * np.asarray(delta, dtype=float), quat, seed, restarts=0, stay=0.01)
        if q is None or np.max(np.abs(wrap_deg(q - seed))) > 3.0 * step / PATH_STEP:
            raise Refused(f"{name}: the arm can't go straight {k / n * length * 100:.0f} cm of {length * 100:.0f}")
        qs.append(q)
        seed = q
    return qs


def reset_moves(model, q_now, q_start, empty=False):
    """Back to a saved start: straight UP to the start's height first (the open fingers rise clear
    of the container instead of sweeping across it), then one joint move to the start's joints.
    empty: nothing in the hand, so the joint move may turn the hand any way (no tilt check)."""
    T_now, T_start = model.fk(q_now), model.fk(q_start)
    moves, q = [], np.asarray(q_now, dtype=float)
    if T_start[2, 3] - T_now[2, 3] > 0.005:
        up = T_now[:3, 3].copy()
        up[2] = T_start[2, 3]
        moves = plan_segment(model, q, up, Rotation.from_matrix(T_now[:3, :3]).as_quat(), None, "up")
        q = moves[-1][1]
    problems = check_joint_move(model, q, q_start, None, tilt_check=not empty)
    jump = float(np.max(np.abs(wrap_deg(np.asarray(q_start) - q))))
    if problems or jump > MAX_STEP_JUMP_DEG:
        raise Refused("back to the start: " + ("; ".join(problems) or f"a joint moves {jump:.0f} deg"))
    return moves + [("start", np.asarray(q_start, dtype=float))]


def hold_point(line_q, line_z, z_cmd):
    """The joints on the planned line down (joint waypoints line_q at flange heights line_z,
    falling) where the flange is at z_cmd: interpolated between neighbouring waypoints (a few mm
    apart). Above the line's start: None (hold where the arm is)."""
    if z_cmd >= line_z[0]:
        return None
    for i in range(len(line_z) - 1):
        if line_z[i] >= z_cmd >= line_z[i + 1]:
            s = (line_z[i] - z_cmd) / max(line_z[i] - line_z[i + 1], 1e-9)
            return np.asarray(line_q[i]) + s * (np.asarray(line_q[i + 1]) - np.asarray(line_q[i]))
    return np.asarray(line_q[-1])


class VzContact:
    """Contact on the way down from the hand's z velocity (/ee_state, + = up): once the hand has
    moved down faster than `moving`, vz at or above `stop` for `hits` new messages in a row = the
    table stopped it. On the flatbot (2026-10-26) the default 0.0 came 4.5 s late: the wrist folded
    about the fingertips and vz sat at ~-0.5 cm/s; -1 cm/s stops at the touch."""

    def __init__(self, stop=-0.01, moving=0.01, hits=2):
        self.stop, self.moving, self.need = stop, moving, hits
        self.moved, self.count = False, 0

    def update(self, vz):
        self.moved = self.moved or vz < -self.moving
        self.count = self.count + 1 if self.moved and vz >= self.stop else 0
        return self.count >= self.need


# self.arm.move_joints times a path minimum-jerk over its whole length (rammp_arm.joint_trajectory):
# at time fraction u the hand is minjerk_distance(u) of the way along, at minjerk_speed(u) x the
# average speed. The descent is one such move, so it is slow at both ends.
CONTACT_GATE = 0.75  # contact counts only while the commanded speed is at least this x lower_speed
MAX_PEAK = 1.5  # the descent's peak speed is at most this x lower_speed


def minjerk_distance(u):
    u = min(max(u, 0.0), 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def minjerk_speed(u):
    u = min(max(u, 0.0), 1.0)
    return 30 * u**2 * (1 - u) ** 2


def minjerk_time(d):
    """The time fraction at which a minimum-jerk move is fraction d of the way along."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if minjerk_distance(mid) < d else (lo, mid)
    return (lo + hi) / 2


def descent_speed_scale(meet, lower_speed, cart_speed):
    """move_joints' speed_scale for the line down so the hand passes `meet` (0..1 of the way, where
    the table should be) at lower_speed, the speed the contact test was tuned at (RAMMP-table: a
    steady 2 cm/s). Its peak speed is cart_speed x the scale (joint_trajectory), at most
    MAX_PEAK x lower_speed."""
    u = minjerk_time(min(max(meet, 0.05), 0.95))
    peak = min(lower_speed / (minjerk_speed(u) / minjerk_speed(0.5)), MAX_PEAK * lower_speed)
    return peak / cart_speed


def contact_gate(u, length, duration, lower_speed):
    """Whether contact may be called at time fraction u of the descent: the commanded speed is up
    (at the start and end of a minimum-jerk move it is slow, and a hand that slows with its command
    must not read as stopped by the table)."""
    return 0.0 < u < 1.0 and minjerk_speed(u) * length / duration >= CONTACT_GATE * lower_speed


# ---------------------------------------------------------------------------------------------
# The arm side (lower.py):
# The arm side of placing, on self.arm as it is: the impedance lowering with the contact hold and leaving
# impedance. Ported from RAMMP-table's kinova_arm.py (2026-10-26).
#
# 1. settle: an IMPEDANCE goal holding the line's start for `settle_s` (execute_joint_trajectory);
#    the hand must then be still -- the arm sags a little when it goes compliant, and that must be
#    over before the descent, or it looks like motion.
# 2. descent: the planned straight line down, self.arm.move_joints in impedance. Its monitor records
#    every new /ee_state message (pose, twist, wrench, phase) for plot_descent.py and stops on
#    contact (plan.VzContact): move_joints CANCELS the goal, and the hold goes out right after.
#    move_joints times the whole line minimum-jerk (slow at both ends), so contact only counts while
#    the commanded speed is up (plan.contact_gate); the speed is chosen so the hand passes the
#    marked table at ~lower_speed (plan.descent_speed_scale).
#    RAMMP-table replaced the goal with the hold instead of cancelling it (a cancel made the compliant
#    arm spring 2.7 cm up on the HERL chair, 2026-10-08, SOFT gains); whether the flatbot does so is
#    the first thing to look at in the recording (z right after "contact").
# 3. hold `hold_press` below the contact point on the planned line.
# In the microwave the release that follows runs in position control (to_position, then
# release_and_back_out): impedance moves lag too far for the 1-2 cm gaps inside the oven.
# One set of impedance gains (`gains`) for all of it, so the stiffness never changes mid-place. Any
# ending before contact leaves impedance with the position setpoint ON the current joints
# (exit_impedance), so a failed run never leaves the arm compliant in mid-air.
# ---------------------------------------------------------------------------------------------
LOOSE_TOL_RAD = 0.5  # path/goal tolerance of the impedance moves: the compliant arm lags and is held back
# by the table; the monitor watches instead
HOLD_S = 0.5  # s, the settle's and hold's single point
Z_STILL_SPEED = 0.002  # m/s, |vz| under this = still
SETTLE_STILL_S = 0.5  # after the settle the hand must be still this long ...
SETTLE_WAIT_S = 5.0  # ... within this long
EXIT_SETTLE_S = 2.0  # leaving impedance: wait at most this long for the arm to be still ...
EXIT_HOLD_S = 0.5  # ... then a position goal on the current joints, this long
OPEN_STALLED = 0.3  # the gripper must open past this, or the hand would drag the container ...
OPEN_WAIT_S = 3.0  # ... within this long of the open command (sent twice at most)
GRIPPER_FULL_RAD = 0.8  # the finger knuckle's travel (the URDF's limit) = gripper position 1.0 (closed)
# In the oven the gripper opens only to RELEASE_OPEN (gripper position, 0 = fully open): the fingers stand
# vertically, and fully open they span ~15.5 cm of the ~17.5 cm between the glass and the ceiling (1.0 cm each
# side at best); at 0.5 they leave ~2 cm free on each side of the container's wall and need a ~1.5 cm lift for
# 1 cm to the glass and 2 cm to the ceiling (arm model, 2026-11-02).
RELEASE_OPEN = 0.5
RELEASE_RAD = RELEASE_OPEN * GRIPPER_FULL_RAD  # the fingers' angle the release and back-out are checked at
RELEASE_TOL = 0.05  # the gripper must reach RELEASE_OPEN within this
LOG_COLUMNS = ["t", "phase", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "fx", "fy", "fz", "tx", "ty", "tz"]


def points(times, qs):
    """[(t, positions, velocities)] for execute_joint_trajectory; no velocities (the driver
    interpolates), as the RAMMP-table goals ran."""
    return [(float(t), [float(v) for v in q], []) for t, q in zip(times, qs, strict=True)]


def compliant(gains):
    """move_joints' `impedance` dict for `gains` (kq, zeta, torque_limit)."""
    return {**gains, "path_tol": LOOSE_TOL_RAD}


class Recorder:
    """/ee_state rows (poll() from a monitor or a wait), with the phase they were in."""

    def __init__(self):
        self.rows, self.phase, self._last = [], "settle", None

    def poll(self, arm):
        ee = arm.ee_state()
        key = (*ee.position, *ee.linear)
        if key != self._last:
            self._last = key
            self.rows.append([time.time(), self.phase, *ee.position, *ee.linear, *ee.angular, *ee.force, *ee.torque])
            return ee, True
        return ee, False

    def write(self, directory):
        if not self.rows:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / time.strftime("descent_%Y%m%d_%H%M%S.csv")
        with open(path, "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(LOG_COLUMNS)
            out.writerows(self.rows)
        return path


def hold_goal(arm, q, gains, hold_s=HOLD_S):
    """An IMPEDANCE goal holding joints q (LATEST_WINS: it replaces whatever holds the arm)."""
    return arm.execute_joint_trajectory(
        JOINTS,
        points([hold_s], [q]),
        path_tolerance=LOOSE_TOL_RAD,
        goal_tolerance=LOOSE_TOL_RAD,
        control_mode=IMPEDANCE,
        preemption=LATEST_WINS,
        gains=gains,
    )


def wait_still(arm, timeout_s, hold_s=0.2, rec=None):
    """Wait until the flange's z velocity has been ~0 for hold_s (at most timeout_s)."""
    t0, since = time.monotonic(), None
    while time.monotonic() - t0 < timeout_s:
        ee = rec.poll(arm)[0] if rec is not None else arm.ee_state()
        still = abs(float(ee.linear[2])) < Z_STILL_SPEED
        since = (since or time.monotonic()) if still else None
        if since is not None and time.monotonic() - since >= hold_s:
            return True
        time.sleep(0.02)
    return False


def exit_impedance(arm, log):
    """Back to POSITION control without a jump: in impedance the arm sits a few cm off its
    setpoint, so switching on that setpoint would pull it there. Wait until still, then a POSITION
    goal whose setpoint IS the current joints."""
    wait_still(arm, EXIT_SETTLE_S)
    q = arm.raw_joints()
    res = arm.execute_joint_trajectory(
        JOINTS, points([EXIT_HOLD_S], [q]), control_mode=POSITION, preemption=LATEST_WINS
    )
    log(
        "impedance off: position setpoint = the current joints"
        + ("" if res.error_code == SUCCESSFUL else f" ({res.error_string})")
    )


def lower(arm, model, descent, cfg, rec, log):
    """Lower along `descent` (plan.plan_descent's (zs, qs)) until contact, then hold. Returns the
    hold's joints (where the retreat starts). Raises Refused (nothing let go) if it didn't settle,
    nothing stopped it, or it stopped well above the table; the arm is then out of impedance."""
    zs, qs = descent
    gains = cfg["gains"]
    line = _unwrap_from(arm.raw_joints(), qs)  # the short way round from the measured joints
    line_z = [float(model.fk(q, "flange")[2, 3]) for q in qs]

    def fail(why):
        exit_impedance(arm, log)
        return Refused(f"STOPPED: {why} -- nothing let go")

    # 1. settle in impedance on the line's start
    rec.phase = "settle"
    log(f"impedance on (kq {gains['kq'][0]:g}..), holding the start {cfg['settle_s']:.1f} s ...")
    res = hold_goal(arm, line[0], gains, hold_s=cfg["settle_s"])
    if res.error_code != SUCCESSFUL:
        raise fail(f"the impedance hold at the start ended {res.error_string}")
    if not wait_still(arm, SETTLE_WAIT_S, SETTLE_STILL_S, rec):
        raise fail(f"the hand is not still in impedance after {cfg['settle_s'] + SETTLE_WAIT_S:.0f} s -- not lowering")
    lag = float(arm.ee_state().position[2]) - line_z[0]
    log(f"settled, hand {lag * 100:+.1f} cm {'above' if lag >= 0 else 'below'} the planned start")

    # 2. the line down, stopped on contact
    lengths = np.abs(np.diff(zs))
    meet = (zs[0] - (cfg["table_z"] + cfg["bowl_depth"])) / (zs[0] - zs[-1])  # where the table should be, 0..1
    scale = descent_speed_scale(meet, cfg["lower_speed"], rammp_arm.CART_SPEED)
    samples = rammp_arm.joint_trajectory(arm.raw_joints(), line[1:], lengths, scale)
    if samples is None:
        raise fail("the line down is too short to move")
    duration, length = float(samples[0][-1]), float(np.sum(lengths))
    contact = VzContact(stop=cfg["vz_stop"])
    state = dict(t0=None, z=None)

    def monitor():
        ee, new = rec.poll(arm)
        state["t0"] = state["t0"] or time.monotonic()  # the goal's clock, from its first poll
        u = (time.monotonic() - state["t0"]) / duration
        hit = new and contact.update(float(ee.linear[2]))
        if hit and contact_gate(u, length, duration, cfg["lower_speed"]):
            state["z"] = float(ee.position[2])
            rec.phase = "contact"
            log(
                f"contact at flange z={state['z']:.3f}: vz {ee.linear[2] * 100:+.2f} cm/s, "
                f"{min(u, 1.0) * 100:.0f}% of the move -> stop, hold {cfg['hold_press'] * 1000:.0f} mm below it"
            )
            return True
        return False

    rec.phase = "descend"
    log(
        f"lowering: {length * 100:.0f} cm in {duration:.0f} s (impedance), ~{cfg['lower_speed'] * 100:.0f} cm/s "
        f"where the table should be ({meet * 100:.0f}% down); contact = vz back up to {cfg['vz_stop'] * 100:+.1f} cm/s"
    )
    ok = arm.move_joints(line[1:], cart_lengths=lengths, speed_scale=scale, impedance=compliant(gains), monitor=monitor)
    if state["z"] is None:
        raise fail(
            f"went down {length * 100:.0f} cm without stopping on anything"
            if ok
            else f"the descent failed ({arm.last_error})"
        )

    # 3. hold a little below the contact point
    q = hold_point(line, line_z, state["z"] - lag - cfg["hold_press"])
    hold_q = np.asarray(arm.raw_joints() if q is None else q, dtype=float)
    hold_goal(arm, hold_q, gains)
    wait_still(arm, 3.0, rec=rec)
    z_tool = float(model.fk(arm.raw_joints())[2, 3])
    rest = z_tool - cfg["table_z"] - cfg["bowl_depth"]
    if rest > cfg["early_stop"]:
        raise Refused(
            f"STOPPED: the hand stopped at tool z={z_tool:.3f}, {rest * 100:.0f} cm above where the bottom meets "
            "the table -- it hit something else; NOT letting go (holding in impedance)"
        )
    log(f"contact at tool z={z_tool:.3f}")
    return hold_q


def open_to(arm, rec, log, position=RELEASE_OPEN):
    """Open the gripper to `position` (0 = fully open); Refused (hand NOT backed out) unless it gets within
    RELEASE_TOL of it (the command is sent twice at most, OPEN_WAIT_S each)."""
    rec.phase = "open"
    pos = float(arm.gripper_state().position)
    for attempt in (1, 2):
        arm.gripper(position)
        t0 = time.monotonic()
        while time.monotonic() - t0 < OPEN_WAIT_S:
            pos = float(arm.gripper_state().position)
            if pos <= position + RELEASE_TOL:
                break
            time.sleep(0.05)
        if pos <= position + RELEASE_TOL:
            break
        log(
            f"gripper at {pos:.2f}, not {position:.2f}, after {OPEN_WAIT_S:.0f} s"
            + (" -- sending it again" if attempt == 1 else "")
        )
    log(f"gripper opened to {pos:.2f} (asked {position:.2f}; 0 = fully open)")
    if pos > position + RELEASE_TOL:
        raise Refused(
            f"STOPPED, hand NOT backed out: the gripper only opened to {pos:.2f} (asked {position:.2f}). Open it, "
            "then `retreat`"
        )


def joint_moves(arm, model, moves, cfg, log, tol_deg=3.0):
    """Each (name, q) as its own impedance joint move (reset), at `reset_speed_scale` of self.arm's
    joint speed, checked to end within tol_deg."""
    for name, q in moves:
        T = model.fk(q)
        log(f"-> {name}: hand to x={T[0, 3]:.3f} y={T[1, 3]:.3f} z={T[2, 3]:.3f} (impedance)")
        ok = arm.move_joints([q], speed_scale=cfg["reset_speed_scale"], impedance=compliant(cfg["gains"]))
        wait_still(arm, 3.0)
        (target,) = _unwrap_from(arm.raw_joints(), [q])
        off = float(np.max(np.abs(np.degrees(np.asarray(arm.raw_joints()) - target))))
        if not ok or off > tol_deg:
            raise Refused(f"STOPPED: {name} ended {off:.1f} deg off ({arm.last_error or 'done'})")


# ---------------------------------------------------------------------------------------------
# Into the microwave (the sequence that ran on the arm, 2026-11-02): the oven as obstacles, the hand's
# heading, cuRobo's transit to in front of the opening, a straight line in, and the check of all of it on
# the arm model against the TRUE obstacle boxes.
#
# The held tupperware is modelled as a BOX_SIDE x BOX_SIDE x (BOX_BELOW + BOX_ABOVE) box centred on the hand's
# axis, its centre model.reach ahead of the tool frame. The target puts that centre over the placement point,
# the tool `above` over the turntable glass, the hand pointing along the oven's axis turned by a heading
# offset (the whole tupperware fits through the opening turned; the arm often reaches only so). The
# pre-insert pose is back along that heading until the whole box is INSERT_OUT outside the outer front.
# ---------------------------------------------------------------------------------------------
BOX_SIDE, BOX_BELOW, BOX_ABOVE = 0.17, 0.05, 0.05  # m: the tupperware's box; BOX_BELOW = tool frame to its bottom
INSERT_OUT = 0.03  # m, the pre-insert pose: the whole box this far outside the outer front
LINE_STEP = 0.004  # m, between the straight line's joint waypoints
TRANSIT_STEP_DEG = 1.0  # the check interpolates cuRobo's transit to at most this per joint ...
LINE_STEP_DEG = 0.5  # ... and the straight lines (in, down, out) to this
END_TOL, END_TOL_DEG = 0.01, 2.0  # m, deg: the line in ends this close to the target, pointing along the heading
DESCENT_BELOW_MEET = 0.01  # m, the line down must reach this far below where the box's bottom meets the glass
# The world: cuRobo keeps at most 60 boxes in its cache
MAX_BOXES, SHELL_BOXES = 58, 5
PAD_COMP = 0.02  # m: cuRobo pads every box 2 cm; the gaps inside the oven are 1-2 cm, so the shell boxes cuRobo
# gets have their INNER faces pulled back this much. Our check uses the true boxes.
WALL_MIN = 0.05  # m, the shell's walls are at least this thick around the opening (sides) ...
SHELL_BELOW, SHELL_ABOVE, SHELL_BEHIND = 0.04, 0.06, 0.10  # ... and reach at least this below / above / behind it
OUTSIDE_MIN_POINTS = 200  # SAM3 "microwave" depth points at or behind the outer front needed for its outside
OUTSIDE_PCT = (2, 98)  # the outside: these percentiles of those points
SHELL_NEAR = 0.02  # m, depth points this close to the shell are the oven, not obstacles in front of it
DOOR_FRONT, DOOR_SIDE = 0.50, 0.45  # m: obstacles kept in front of the outer front, this far out and to the sides
DOOR_ABOVE_FLOOR, DOOR_OVER_TOP = 0.02, 0.15  # m, ... from this above the glass to this over the shell's top
CELLS = (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.2)  # m, column sizes tried, finest first
CELL_MIN_POINTS = 15  # a column needs this many depth points
DROP_NEAR = 0.03  # m: a column this close to the robot or the held box at the start is them, not an obstacle
DEPTH_RANGE = (0.15, 2.5)  # m, the scene depth used
FLOOR_BOX = "mw_bottom"  # the box under the glass: the lower finger lands on it lowering
CEILING_BOX = "mw_top"  # the box over the cavity
# The release (on the arm, 2026-11-02): fingers stacked vertically, the LOWER finger is what lands on the glass
# (closed +0.5 cm from it on the model, OPEN -0.7 cm: the opening lower finger moves down into the glass). So
# the hand is lifted straight up first, in position control, until the open fingers clear the glass and the
# ceiling; then it opens; then it goes back onto the line in's end and back along the line in's own joints.
LIFT_STEP, LIFT_MAX = 0.002, 0.04  # m, the release lift: the smallest multiple of LIFT_STEP up to LIFT_MAX
ONTO_MAX_DEG = 10.0  # back onto the line in's end: a joint move of at most this (it is a cm or two) ...
ONTO_SPEED_SCALE = 0.1  # ... at this speed_scale of self.arm's joint speed


class StageFailed(Refused):
    """A heading's plan failed at `stage` for `reason`."""

    def __init__(self, stage, reason):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason = stage, reason


def _stage(stage, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except StageFailed:
        raise
    except Refused as e:
        raise StageFailed(stage, str(e)) from e


class Oven:
    """The located opening (microwave_detection.fit_interior's result): b0 the middle of its bottom edge on the frame
    plane, n into the oven, w its left. Oven coordinates: A along w, S along n, Z up from b0."""

    def __init__(self, fit):
        self.n = np.asarray(fit["insertion_axis"], dtype=float)
        self.w = np.asarray(fit["lateral_axis_left"], dtype=float)
        corners = np.asarray(fit["opening_corners"], dtype=float)  # BR, BL, TL, TR
        self.b0 = 0.5 * (corners[0] + corners[1])
        self.point = np.asarray(fit["point"], dtype=float)
        self.floor_z = float(fit["floor_z"])
        self.yaw_deg = float(np.degrees(np.arctan2(self.n[1], self.n[0])))

    def to_oven(self, P):
        q = np.asarray(P, dtype=float) - self.b0
        return np.c_[q @ self.w, q @ self.n, q[:, 2]]

    def box(self, name, ra, rs, rz):
        """An obstacle box from its A, S, Z ranges: dict(name, position, rpy_deg, dims) in the world."""
        (a0, a1), (s0, s1), (z0, z1) = ra, rs, rz
        c = self.b0 + (a0 + a1) / 2 * self.w + (s0 + s1) / 2 * self.n + np.array([0.0, 0.0, (z0 + z1) / 2])
        return dict(
            name=name,
            position=[float(v) for v in c],
            rpy_deg=[0.0, 0.0, self.yaw_deg],
            dims=[float(s1 - s0), float(a1 - a0), float(z1 - z0)],
        )


def _world_pose(model, pos, quat=(0.0, 0.0, 0.0, 1.0)):
    """arm_mount -> the sim's world frame."""
    import pybullet as p

    base = model.scene.robot_base_pose
    return p.multiplyTransforms(
        list(base.position),
        list(base.orientation),
        [float(v) for v in pos],
        [float(v) for v in quat],
        physicsClientId=model.rb.physics_client_id,
    )


def _box_body(model, box):
    import pybullet as p

    cid = model.rb.physics_client_id
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d / 2 for d in box["dims"]], physicsClientId=cid)
    pos, orn = _world_pose(model, box["position"], Rotation.from_euler("xyz", box["rpy_deg"], degrees=True).as_quat())
    return p.createMultiBody(0, shape, basePosition=pos, baseOrientation=orn, physicsClientId=cid)


def _place_container(model, body, q):
    """Move the held box `body` to where it is with the arm at q."""
    import pybullet as p

    T = model.fk(q)
    R = T[:3, :3]
    z = R[:, 0] * (-1.0 if camera_on_left(R) else 1.0)
    z /= np.linalg.norm(z)
    c = T[:3, 3] + model.reach * R[:, 2] + 0.5 * (BOX_ABOVE - BOX_BELOW) * z
    x = R[:, 2] - (R[:, 2] @ z) * z  # along the hand's heading
    x /= np.linalg.norm(x)
    pos, orn = _world_pose(model, c, Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat())
    p.resetBasePositionAndOrientation(body, pos, orn, physicsClientId=model.rb.physics_client_id)


def finger_clearance(rb, q, bodies, closed):
    """sim.clearance with `closed` True/False, or with the fingers at the angle `closed` (rad, 0 = open)."""
    if isinstance(closed, bool):
        return sim.clearance(rb, q, bodies, closed=closed)
    import pybullet as p

    cid = rb.physics_client_id
    for j, jj in enumerate(sim.ARM):
        p.resetJointState(rb.robot_id, jj, float(q[j]), physicsClientId=cid)
    for jj in sim.FINGER_JOINTS:
        p.resetJointState(rb.robot_id, jj, sim.FINGER_CLOSE_SIGN[jj] * float(closed), physicsClientId=cid)
    best = (9.0, "")
    for bid, name in bodies:
        for pt in p.getClosestPoints(rb.robot_id, bid, sim.CLEARANCE_QUERY_M, physicsClientId=cid):
            if pt[8] < best[0]:
                ln = p.getJointInfo(rb.robot_id, pt[3], physicsClientId=cid)[12].decode() if pt[3] >= 0 else "base"
                best = (float(pt[8]), f"{ln} vs {name}")
    return best


class Checker:
    """The obstacle boxes (and the held box) as bodies in the shared sim, for clearance checks. close()
    removes them; use it in a finally."""

    def __init__(self, model, boxes):
        import pybullet as p

        self.model, self.bodies, self.container = model, [], None
        cid = model.rb.physics_client_id
        try:
            for b in boxes:
                self.bodies.append((_box_body(model, b), b["name"]))
            shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[BOX_SIDE / 2, BOX_SIDE / 2, (BOX_BELOW + BOX_ABOVE) / 2], physicsClientId=cid
            )
            self.container = p.createMultiBody(0, shape, basePosition=[0, 0, -10], physicsClientId=cid)
        except BaseException:
            self.close()
            raise

    def clearance(self, q, container=False, skip=(), closed=True):
        """(smallest distance, "link vs box") at joints q: every arm and gripper link (`closed`: True closed,
        False fully open, a number: the fingers at that angle in rad), and with `container` the held box, to
        every box not named in `skip`."""
        import pybullet as p

        bodies = [b for b in self.bodies if b[1] not in skip]
        best = finger_clearance(self.model.rb, q, bodies, closed)
        if container:
            _place_container(self.model, self.container, q)
            for bid, name in bodies:
                for pt in p.getClosestPoints(
                    self.container, bid, sim.CLEARANCE_QUERY_M, physicsClientId=self.model.rb.physics_client_id
                ):
                    if pt[8] < best[0]:
                        best = (float(pt[8]), f"container vs {name}")
        return best

    def distance_to(self, q, names, closed=True):
        """(smallest distance, "link vs box") at joints q from every arm and gripper link to the boxes `names`."""
        return finger_clearance(self.model.rb, q, [b for b in self.bodies if b[1] in names], closed)

    def close(self):
        import pybullet as p

        cid = self.model.rb.physics_client_id
        for bid in [b for b, _ in self.bodies] + ([] if self.container is None else [self.container]):
            p.removeBody(bid, physicsClientId=cid)
        self.bodies, self.container = [], None


def microwave_obstacles(model, q0, oven, depth_m, K, world_T_cam, mw_mask):
    """The obstacles from one scene frame (depth_m, K, world_T_cam) and SAM3's "microwave" mask on it.

    - The microwave as SHELL_BOXES boxes around its free cavity and opening (the located opening and the
      tape-measured interior), out to its outside (OUTSIDE_PCT of the "microwave" depth points at or behind
      the outer front, at least WALL_MIN / SHELL_* past the cavity).
    - What the camera sees in front of the outer front, within DOOR_FRONT / DOOR_SIDE of the opening (the
      open door): columns on a grid, coarsened from CELLS[0] until they fit in MAX_BOXES; a column within
      DROP_NEAR of the robot or the held box at joints q0 is them and is dropped.

    Returns (true boxes, the boxes for cuRobo (the shell's inner faces pulled back PAD_COMP), info).
    Raises Refused when the microwave's outside is not seen or the columns do not fit."""
    import pybullet as p

    W, H, D, F = (
        microwave_detection.OPENING[0],
        microwave_detection.OPENING[1],
        microwave_detection.DEPTH,
        microwave_detection.FLOOR,
    )
    valid = np.isfinite(depth_m) & (depth_m > DEPTH_RANGE[0]) & (depth_m < DEPTH_RANGE[1])
    Tc = np.asarray(world_T_cam, dtype=float)
    Pw = microwave_detection._backproject(np.where(valid, depth_m, 0), np.asarray(K, float)) @ Tc[:3, :3].T + Tc[:3, 3]
    if mw_mask is None or mw_mask.shape != valid.shape:
        raise Refused("no SAM3 'microwave' mask on the scene frame: the oven's outside is not known")
    Om = oven.to_oven(Pw[mw_mask & valid])
    Om = Om[Om[:, 1] > -microwave_detection.RECESS - 0.01]  # at or behind the outer front (the open door is in front)
    if len(Om) <= OUTSIDE_MIN_POINTS:
        raise Refused(
            f"only {len(Om)} SAM3 'microwave' depth points at or behind its front (need {OUTSIDE_MIN_POINTS}): "
            "the oven's outside is not known"
        )
    lo, hi = np.percentile(Om, OUTSIDE_PCT, axis=0)
    a0e, a1e = min(-W / 2 - WALL_MIN, lo[0]), max(W / 2 + WALL_MIN, hi[0])
    z0e, z1e = min(-SHELL_BELOW, lo[2]), max(H + SHELL_ABOVE, hi[2])
    sFe, sBe = -microwave_detection.RECESS, max(D + SHELL_BEHIND, hi[1])
    shell = [  # (name, A range, S range, Z range)
        ("mw_top", (a0e, a1e), (sFe, sBe), (H, z1e)),
        (FLOOR_BOX, (a0e, a1e), (sFe, sBe), (z0e, F)),
        ("mw_left", (W / 2, a1e), (sFe, sBe), (F, H)),
        ("mw_right", (a0e, -W / 2), (sFe, sBe), (F, H)),
        ("mw_back", (-W / 2, W / 2), (D, sBe), (F, H)),
    ]
    inner = {"mw_top": (2, 0), FLOOR_BOX: (2, 1), "mw_left": (0, 0), "mw_right": (0, 1), "mw_back": (1, 0)}
    boxes, curobo = [], []
    for name, ra, rs, rz in shell:
        boxes.append(oven.box(name, ra, rs, rz))
        rng = [list(ra), list(rs), list(rz)]
        axis, side = inner[name]  # the face toward the cavity: (A/S/Z, low / high end)
        rng[axis][side] += PAD_COMP if side == 0 else -PAD_COMP
        curobo.append(oven.box(name, *rng))

    Ov = oven.to_oven(Pw[valid])
    near_shell = (Ov[:, 0] > a0e - SHELL_NEAR) & (Ov[:, 0] < a1e + SHELL_NEAR)
    near_shell &= (Ov[:, 1] > sFe - SHELL_NEAR) & (Ov[:, 1] < sBe + SHELL_NEAR)
    near_shell &= (Ov[:, 2] > z0e - SHELL_NEAR) & (Ov[:, 2] < z1e + SHELL_NEAR)
    Q, Oq = Pw[valid][~near_shell], Ov[~near_shell]
    front = (Oq[:, 1] < sFe) & (Oq[:, 1] > sFe - DOOR_FRONT) & (np.abs(Oq[:, 0]) < W / 2 + DOOR_SIDE)
    front &= (Oq[:, 2] > F + DOOR_ABOVE_FLOOR) & (Oq[:, 2] < z1e + DOOR_OVER_TOP)
    Q = Q[front]

    cid = model.rb.physics_client_id
    check = Checker(model, [])
    try:
        sim.clearance(model.rb, q0, [], closed=True)  # the robot at the start, gripper closed
        _place_container(model, check.container, q0)
        cols, cell = [], CELLS[0]
        for cell in CELLS:
            keys = np.floor(Q[:, :2] / cell).astype(int)
            cols = []
            for k in {tuple(v) for v in keys}:
                sel = Q[(keys[:, 0] == k[0]) & (keys[:, 1] == k[1])]
                if len(sel) < CELL_MIN_POINTS:
                    continue
                zlo, zhi = np.percentile(sel[:, 2], [1, 99])
                col = dict(
                    name="",
                    position=[float((k[0] + 0.5) * cell), float((k[1] + 0.5) * cell), float((zlo + zhi) / 2)],
                    rpy_deg=[0.0, 0.0, 0.0],
                    dims=[float(cell), float(cell), float(max(zhi - zlo, 0.02))],
                )
                bid = _box_body(model, col)
                ours = p.getClosestPoints(model.rb.robot_id, bid, DROP_NEAR, physicsClientId=cid) or p.getClosestPoints(
                    check.container, bid, DROP_NEAR, physicsClientId=cid
                )
                p.removeBody(bid, physicsClientId=cid)
                if not ours:
                    cols.append(col)
            if len(cols) <= MAX_BOXES - SHELL_BOXES:
                break
    finally:
        check.close()
    if len(cols) > MAX_BOXES - SHELL_BOXES:
        raise Refused(
            f"{len(cols)} obstacle columns in front of the oven even at {cell * 100:.0f} cm cells "
            f"(cuRobo takes {MAX_BOXES - SHELL_BOXES} with the shell)"
        )
    for i, c in enumerate(cols):
        c["name"] = f"seen_{i:02d}"
    info = dict(columns=len(cols), cell=float(cell), outside=[lo.tolist(), hi.tolist()], front_points=int(len(Q)))
    return boxes + cols, curobo + cols, info


def write_world(path, boxes, frame):
    """cuRobo's world file (set_world): the boxes in `frame`."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(base_frame=frame, obstacles=boxes), f, sort_keys=False)


def heading_pose(model, q0, oven, offset_deg, above):
    """The place for a heading `offset_deg` off the oven's axis (+ = to the left, about up): heading,
    quat (the hand's orientation at q0 turned about up to face along the heading, so its tilt is kept),
    target (the tool with the box's centre over the placement point, `above` over the glass), pre (the
    pre-insert pose, insert_m back along the heading)."""
    T0 = model.fk(q0)
    h0 = np.array([T0[0, 2], T0[1, 2], 0.0])
    h0 /= np.linalg.norm(h0)
    off = np.radians(offset_deg)
    h = Rotation.from_rotvec([0.0, 0.0, off]).apply(np.r_[oven.n[:2], 0.0])
    turn = np.arctan2(np.cross(h0, h)[2], h0 @ h)
    quat = Rotation.from_matrix(Rotation.from_rotvec([0.0, 0.0, turn]).as_matrix() @ T0[:3, :3]).as_quat()
    target = np.r_[oven.point[:2] - model.reach * h[:2], oven.floor_z + above]
    # from the box's far side INSERT_OUT outside the outer front to the target, along the heading
    insert_m = ((oven.point - oven.b0) @ oven.n + BOX_SIDE / 2 + microwave_detection.RECESS + INSERT_OUT) / max(
        np.cos(off), 0.3
    )
    return dict(
        offset_deg=float(offset_deg), heading=h, quat=quat, target=target, pre=target - insert_m * h, insert_m=insert_m
    )


def probe_heading(model, checker, q0, cand, clear_min, container=False):
    """The quick tests of a heading before cuRobo: IK at the target with clear_min of clearance, IK at the
    pre-insert pose, the straight line in from there with clear_min at every waypoint. Raises StageFailed."""
    qg = model.ik(cand["target"], cand["quat"], q0)
    if qg is None:
        raise StageFailed("goal", "no IK solution")
    d, what = checker.clearance(qg, container)
    if d < clear_min:
        raise StageFailed("goal", f"clearance {d * 100:.1f} cm < {clear_min * 100:.1f} ({what})")
    qp = model.ik(cand["pre"], cand["quat"], q0)
    if qp is None:
        raise StageFailed("pre-insert", "no IK solution")
    line = _stage("line in", plan_line, model, qp, cand["insert_m"] * cand["heading"], "line in", step=LINE_STEP)
    d, what = min(checker.clearance(q, container) for q in line)
    if d < clear_min:
        raise StageFailed("line in", f"clearance {d * 100:.1f} cm < {clear_min * 100:.1f} ({what})")


TRANSIT_TRAVEL_MAX_DEG = 120.0  # any joint's total travel along cuRobo's transit; more = it changed the arm's posture


def joint_delta(a, b):
    """b - a per joint (rad): the continuous joints (rammp_arm.CONTINUOUS) the short way round, the limited ones
    as they are (they can't wrap)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    d[CONTINUOUS] = wrap(d[CONTINUOUS])
    return d


def joint_travel_deg(qs):
    """Each joint's total travel (deg) along the waypoints qs."""
    qs = [np.asarray(q, dtype=float) for q in qs]
    return np.degrees(sum((np.abs(joint_delta(a, b)) for a, b in zip(qs[:-1], qs[1:], strict=True)), np.zeros(7)))


def check_transit_travel(qs):
    """Refuse a transit that swings any joint more than TRANSIT_TRAVEL_MAX_DEG in total (cuRobo's pose goal
    lets it pick another posture at the end, e.g. the elbow flipped: J2/J4 200+ deg). Raises StageFailed."""
    travel = joint_travel_deg(qs)
    if np.max(travel) > TRANSIT_TRAVEL_MAX_DEG:
        raise StageFailed(
            "cuRobo",
            f"the transit swings the arm: joint travel {np.round(travel, 0).tolist()} deg "
            f"(max {TRANSIT_TRAVEL_MAX_DEG:g} per joint)",
        )
    return travel


def dense(qs, step_deg):
    """qs with joint-space points added so no joint moves more than step_deg between two: each step the
    short way round (wrapped), as the driver's interpolation goes."""
    qs = [np.asarray(q, dtype=float) for q in qs]
    out = [qs[0]]
    for a, b in zip(qs[:-1], qs[1:], strict=True):
        d = joint_delta(a, b)
        k = max(1, int(np.ceil(np.max(np.abs(d)) / np.radians(step_deg))))
        out += [a + d * (i / k) for i in range(1, k + 1)]
    return out


def check_path(model, checker, legs, R0, clear_min, tilt_max, container=False, skip=(), closed=True, stage="check"):
    """Every leg (name, joint waypoints, step_deg), densely interpolated: clearance to every box not in
    `skip` (with `container`: the held box too) and the tilt vs the tool orientation R0 (None: not checked).
    Returns {leg: [clearance, "link vs box"]} and the largest tilt; raises StageFailed(stage, ...) listing
    every leg under clear_min and a tilt over tilt_max."""
    up_tool = None if R0 is None else np.asarray(R0).T @ UP
    worst, tilt = {}, 0.0
    for name, qs, step_deg in legs:
        best = (9.0, "")
        for q in dense(qs, step_deg):
            best = min(best, checker.clearance(q, container, skip, closed))
            if up_tool is not None:
                Rq = model.fk(q)[:3, :3]
                tilt = max(tilt, float(np.degrees(np.arccos(np.clip((Rq @ up_tool) @ UP, -1.0, 1.0)))))
        worst[name] = [float(best[0]), best[1]]
    problems = [
        f"{n} clearance {d * 100:.1f} cm < {clear_min * 100:.1f} ({w})" for n, (d, w) in worst.items() if d < clear_min
    ]
    if tilt_max is not None and tilt > tilt_max:
        problems.append(f"tilt {tilt:.1f} deg > {tilt_max:.1f}")
    if problems:
        raise StageFailed(stage, "; ".join(problems))
    return worst, tilt


def end_pose_error(model, q, target, heading):
    """(m, deg): the tool at q from `target`, its horizontal pointing from `heading`."""
    T = model.fk(q)
    off = float(np.linalg.norm(T[:3, 3] - np.asarray(target, dtype=float)))
    h = np.array([T[0, 2], T[1, 2]])
    turn = float(np.degrees(np.arccos(np.clip(h @ np.asarray(heading)[:2] / max(np.linalg.norm(h), 1e-9), -1.0, 1.0))))
    return off, turn


def line_floor(floor_z):
    """Where the line down is aimed: PRESS_BELOW under where the box's bottom meets the glass, the tool at
    least MIN_TOOL_HEIGHT over it (place_container's rule, with the box's BOX_BELOW)."""
    return floor_z + max(BOX_BELOW - PRESS_BELOW, MIN_TOOL_HEIGHT)


def plan_down(model, q, floor_z, log=None):
    """The straight line down from q (plan_descent), required to reach DESCENT_BELOW_MEET below where the
    box's bottom meets the glass; and its part above that meeting point (what the check covers: the lower
    finger touches the glass ~1.6 cm before the box does, by design)."""
    zs, qs = _stage("descent", plan_descent, model, q, line_floor(floor_z), log=log)
    meet = floor_z + BOX_BELOW
    if zs[-1] > meet - DESCENT_BELOW_MEET:
        raise StageFailed(
            "descent",
            f"the line down reaches tool z {zs[-1]:.3f}, not {DESCENT_BELOW_MEET * 100:.0f} cm below where the box "
            f"meets the glass ({meet:.3f})",
        )
    upper = [qq for z, qq in zip(zs, qs, strict=True) if z >= meet]
    return (zs, qs), upper or [qs[0]], int(np.argmin(np.abs(zs - meet)))


def plan_place_in(model, checker, q0, cand, Qs, floor_z, cfg, log=None):
    """Everything after cuRobo's transit `Qs` for heading `cand`, checked: the transit must end at the
    pre-insert pose; the straight line in from there; the transit and the line in densely checked (clearance
    >= clear_min for every link, with check_container the held box too; tilt <= tilt_max_deg vs the start;
    the end within END_TOL / END_TOL_DEG of the target); the line down from the target (to where the box meets
    the glass) against every box but FLOOR_BOX; from where the model has the lowering stop, the release lift
    (plan_release_lift) and the back-out (plan_back_out). Raises StageFailed. Returns the line in, the
    descent, the check's numbers and the release's."""
    pre, target, h, L = cand["pre"], cand["target"], cand["heading"], cand["insert_m"]
    clear_min, tilt_max, container = cfg["clear_min"], cfg["tilt_max_deg"], cfg["check_container"]
    R0 = model.fk(q0)[:3, :3]
    _stage("pre-insert", check_line, model, [Qs[-1]], pre, pre, h, "cuRobo's end")
    line = _stage("line in", plan_line, model, Qs[-1], L * h, "line in", step=LINE_STEP)
    _stage("line in", check_line, model, line, pre, target, h, "line in")
    worst, tilt = check_path(
        model,
        checker,
        [("transit", Qs, TRANSIT_STEP_DEG), ("insert", line, LINE_STEP_DEG)],
        R0,
        clear_min,
        tilt_max,
        container,
    )
    end_off, end_turn = end_pose_error(model, line[-1], target, h)
    if end_off > END_TOL or end_turn > END_TOL_DEG:
        raise StageFailed("check", f"the line in ends {end_off * 100:.1f} cm / {end_turn:.1f} deg off the target")
    descent, upper, _ = plan_down(model, line[-1], floor_z, log)
    w_down, _ = check_path(
        model, checker, [("descent", upper, LINE_STEP_DEG)], R0, clear_min, tilt_max, container, skip=(FLOOR_BOX,)
    )
    # the release and back-out, from where the model has the lowering stop
    i_hold = contact_index(model, checker, descent, floor_z)
    q_hold = descent[1][i_hold]
    lift_line, lift, w_lift = plan_release_lift(model, checker, q_hold, clear_min)
    _, w_back = plan_back_out(model, checker, lift_line[-1], line, clear_min)
    return dict(
        line=line,
        descent=descent,
        clearance={**worst, **w_down, **w_lift, **w_back},
        tilt=tilt,
        end=(end_off, end_turn),
        release=dict(hold_z=float(descent[0][i_hold]), lift=float(lift)),
    )


def contact_index(model, checker, descent, floor_z):
    """Where on the line down (zs, qs) the model has the lowering stop: the first waypoint where the closed
    fingers touch the glass (FLOOR_BOX) or the box's bottom meets it, whichever comes first."""
    zs, qs = descent
    meet = floor_z + BOX_BELOW
    for i, (z, q) in enumerate(zip(zs, qs, strict=True)):
        if z <= meet or checker.distance_to(q, (FLOOR_BOX,))[0] <= 0.0:
            return i
    return len(qs) - 1


def plan_release_lift(model, checker, q_hold, clear_min, closed=True, released=RELEASE_RAD):
    """The smallest straight lift from q_hold (LIFT_STEP steps, at most LIFT_MAX; none if the opened fingers are
    clear already) after which the fingers opened to `released` (rad) are clear_min from every true box: the
    glass below, the ceiling above. The lift itself (gripper `closed` or not) is checked against every box but the
    glass,
    which it leaves. Returns (joint waypoints from q_hold, lift m, {leg: [clearance, what]}); raises
    StageFailed("release", ...) with the numbers when no lift does it."""

    def open_at(q):
        return (
            checker.clearance(q, closed=released),
            checker.distance_to(q, (FLOOR_BOX,), closed=released)[0],
            checker.distance_to(q, (CEILING_BOX,), closed=released)[0],
        )

    (d0, w0), floor0, ceil0 = open_at(q_hold)
    if d0 >= clear_min:
        return [np.asarray(q_hold, dtype=float)], 0.0, {"released": [float(d0), w0]}
    T = model.fk(q_hold)
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    best, why = None, ""
    for lift in np.arange(LIFT_STEP, LIFT_MAX + 1e-9, LIFT_STEP):
        q = model.ik(T[:3, 3] + lift * UP, quat, q_hold, restarts=0, stay=0.01)
        if q is None:
            why = f"; no IK {lift * 100:.1f} cm up"
            break
        (d, w), fl, ce = open_at(q)
        if best is None or d > best[0]:
            best = (d, w, lift, fl, ce)
        if d < clear_min:
            continue
        line = _stage("release", plan_line, model, q_hold, lift * UP, "release lift", step=LIFT_STEP)
        (d, w), fl, ce = open_at(line[-1])
        if d < clear_min:
            continue
        w_lift, _ = check_path(
            model,
            checker,
            [("lift", line, LINE_STEP_DEG)],
            None,
            clear_min,
            None,
            skip=(FLOOR_BOX,),
            closed=closed,
            stage="release",
        )
        return line, float(lift), {**w_lift, "released": [float(d), w]}
    best_txt = (
        "-"
        if best is None
        else f"best {best[0] * 100:+.1f} cm ({best[1]}) {best[2] * 100:.1f} cm up: glass {best[3] * 100:+.1f}, "
        f"ceiling {best[4] * 100:+.1f}"
    )
    raise StageFailed(
        "release",
        f"fingers opened to {released:.2f} rad at the hold: glass {floor0 * 100:+.1f} cm, ceiling "
        f"{ceil0 * 100:+.1f} cm; no straight lift up to {LIFT_MAX * 100:.0f} cm puts them {clear_min * 100:.1f} cm "
        f"from every box ({best_txt}){why}",
    )


def plan_back_out(model, checker, q_from, line_in, clear_min, stage="back-out", released=RELEASE_RAD):
    """From q_from (released, gripper open) a joint move back onto the line in's end (the target), then the
    line in's own joints reversed to its start (the pre-insert pose), both checked with the fingers opened to
    `released` (rad) against every true box. Returns (the reversed line, its first entry the line in's end;
    {leg: [clearance, what]}); raises StageFailed(stage, ...)."""
    end = np.asarray(line_in[-1], dtype=float)
    jump = float(np.max(np.abs(wrap_deg(end - np.asarray(q_from, dtype=float)))))
    if jump > ONTO_MAX_DEG:
        raise StageFailed(stage, f"getting back onto the line in moves a joint {jump:.1f} deg (max {ONTO_MAX_DEG:g})")
    back = [np.asarray(q, dtype=float) for q in reversed(line_in)]
    worst, _ = check_path(
        model,
        checker,
        [("onto the line", [q_from, end], LINE_STEP_DEG), ("back-out", back, LINE_STEP_DEG)],
        None,
        clear_min,
        None,
        closed=released,
        stage=stage,
    )
    return back, worst


def to_position(arm, log):
    """Out of impedance, the way exit_impedance does it (still first, then a POSITION goal on the measured
    joints, so nothing jumps and nothing presses), but refused if the hand doesn't come still or the driver
    doesn't take the goal."""
    if not wait_still(arm, EXIT_SETTLE_S):
        raise Refused(f"leaving impedance: the hand is not still after {EXIT_SETTLE_S:g} s")
    q = arm.raw_joints()
    res = arm.execute_joint_trajectory(
        JOINTS, points([EXIT_HOLD_S], [q]), control_mode=POSITION, preemption=LATEST_WINS
    )
    if res.error_code != SUCCESSFUL:
        raise Refused(f"leaving impedance: the position hold ended {res.error_string}")
    log("impedance off: position control on the measured joints")


# ---------------------------------------------------------------------------------------------
# The place on the arm: the heading search, the plan's summary, and running it (transit, line in,
# lowering, release, back-out), plus retreat and reset's checks.
# ---------------------------------------------------------------------------------------------
START_TOL_DEG = 1.0  # the arm must be this close (every joint) to where the checked transit starts
SETTLE_AFTER_S = 0.5  # after the transit and the line in, before the arm's joints are read back
TRANSIT_LEAD_S = 0.5  # the transit's first planned sample is sent this long after the measured joints


def cm(d):
    return round(float(d) * 100, 1)


def search_heading(arm, model, checker, q0, oven, cfg, log):
    """(cand, cuRobo's plan, plan_place_in's result, attempts) of the first heading in heading_offsets_deg
    that passes every stage; Refused listing every attempt when none does."""
    c, attempts = cfg, []
    for off in c["heading_offsets_deg"]:
        cand = heading_pose(model, q0, oven, float(off), c["above"])
        try:
            probe_heading(model, checker, q0, cand, c["clear_min"], c["check_container"])
            plan = arm.plan_to_pose(cand["pre"], cand["quat"], q0, hold=HOLD_LEVEL)
            if not plan.success:
                raise StageFailed("cuRobo", " ".join(plan.message.split())[:200] or "no plan")
            if len(plan.q) < 2:
                raise StageFailed("cuRobo", f"a plan of {len(plan.q)} samples")
            travel = check_transit_travel(np.asarray(plan.q, dtype=float))
            log(f"heading {float(off):+.0f} deg: cuRobo's transit, joint travel {np.round(travel, 0).tolist()} deg")
            res = plan_place_in(model, checker, q0, cand, np.asarray(plan.q, dtype=float), oven.floor_z, c, log)
        except StageFailed as e:
            attempts.append(dict(offset_deg=float(off), stage=e.stage, reason=e.reason))
            log(f"heading {float(off):+.0f} deg: FAILED at {e.stage}: {e.reason}")
            continue
        attempts.append(dict(offset_deg=float(off), stage="passed", reason=""))
        log(f"heading {float(off):+.0f} deg: passed every stage")
        return cand, plan, res, attempts
    raise Refused(
        "no heading passed: " + "; ".join(f"{a['offset_deg']:+.0f} {a['stage']}: {a['reason']}" for a in attempts)
    )


def summarize(model, q0, loc, oven, world, world_info, boxes, cand, plan, res, attempts, cfg):
    """(the dry run's summary, the plan record written to state_dir) for the chosen heading."""
    c = cfg
    Qs, ts, line = np.asarray(plan.q, dtype=float), np.asarray(plan.t, dtype=float), res["line"]
    lengths = path_lengths(model, line)
    samples = rammp_arm.joint_trajectory(line[0], line[1:], lengths, c["insert_speed"] / rammp_arm.CART_SPEED)
    zs = res["descent"][0]
    summary = dict(
        located=dict(
            point=np.round(oven.point, 3).tolist(),
            median_point=np.round(loc["point"], 3).tolist(),
            axis=np.round(oven.n, 3).tolist(),
            glass_z=round(oven.floor_z, 3),
            frames_used=loc["frames"],
            frames_detected=loc["detected"],
            spread_cm=cm(loc["spread_m"]),
            spread_deg=round(loc["spread_deg"], 1),
        ),
        world=dict(file=str(world), boxes=len(boxes), **world_info),
        heading=dict(offset_deg=cand["offset_deg"], vector=np.round(cand["heading"], 3).tolist()),
        target=np.round(cand["target"], 3).tolist(),
        pre_insert=np.round(cand["pre"], 3).tolist(),
        insert_cm=cm(cand["insert_m"]),
        clearance_cm={k: [cm(d), w] for k, (d, w) in res["clearance"].items()},
        max_tilt_deg=round(res["tilt"], 2),
        end_off_cm=cm(res["end"][0]),
        end_turn_deg=round(res["end"][1], 2),
        joint_change_deg=dict(
            transit=np.round(np.degrees(joint_delta(Qs[0], Qs[-1])), 1).tolist(),
            insert=np.round(np.degrees(joint_delta(line[0], line[-1])), 1).tolist(),
        ),
        joint_travel_deg=dict(
            transit=np.round(joint_travel_deg(Qs), 1).tolist(),
            insert=np.round(joint_travel_deg(line), 1).tolist(),
        ),
        joints_deg=dict(
            start=np.round(np.degrees(Qs[0]), 1).tolist(),
            pre_insert=np.round(np.degrees(Qs[-1]), 1).tolist(),
            target=np.round(np.degrees(np.asarray(line[-1])), 1).tolist(),
        ),
        duration_s=dict(
            transit=round(float(ts[-1] - ts[0]) / c["transit_speed"] + TRANSIT_LEAD_S, 1),
            insert=round(float(samples[0][-1]) if samples is not None else 0.0, 1),
        ),
        descent=dict(
            from_z=round(float(zs[0]), 3),
            to_z=round(float(zs[-1]), 3),
            meets_glass_z=round(oven.floor_z + BOX_BELOW, 3),
        ),
        attempts=attempts,
    )
    record = dict(
        summary=summary,
        time=time.time(),
        joint_names=list(rammp_arm.JOINTS),
        start_q=q0,
        q=Qs,
        t=ts,
        q_insert=np.asarray(line),
        heading=cand["heading"],
        heading_offset_deg=cand["offset_deg"],
        quat=cand["quat"],
        target=cand["target"],
        pre=cand["pre"],
        insert_m=cand["insert_m"],
        floor_z=oven.floor_z,
        fit=loc["fit"],
        boxes=boxes,
        world=str(world),
        box=dict(side=BOX_SIDE, below=BOX_BELOW, above=BOX_ABOVE, reach=model.reach),
    )
    return summary, record


def run_place(arm, model, checker, boxes, q0, oven, cand, plan, cfg, ctx, log, save, descents_dir):
    """The checked place, on the arm. save(name, record) keeps a JSON record (start.json, last_insert.json);
    the lowering's /ee_state recording goes to descents_dir."""
    c = cfg
    h, floor_z = cand["heading"], oven.floor_z
    R0 = model.fk(q0)[:3, :3]
    Qs, ts = np.asarray(plan.q, dtype=float), np.asarray(plan.t, dtype=float)
    ready(arm)
    q_now = np.asarray(arm.raw_joints(), dtype=float)
    off = float(np.max(np.abs(wrap_deg(q_now - Qs[0]))))
    if off > START_TOL_DEG:
        raise Refused(f"the arm is {off:.1f} deg from the checked transit's start (max {START_TOL_DEG:g}): re-plan")
    save("start.json", dict(q=[float(v) for v in q_now], time=time.time()))

    # the transit, exactly as checked, slowed to transit_speed, from the measured joints
    ctx.progress("transit", 0.0)
    T = (ts - ts[0]) / c["transit_speed"] + TRANSIT_LEAD_S
    Q = rammp_arm._unwrap_from(q_now, Qs)
    log(f"-> transit: {len(Q)} points over {T[-1]:.1f} s (cuRobo's plan x{c['transit_speed']:g})")
    try:
        result = arm.execute_joint_trajectory(rammp_arm.JOINTS, points([0.0, *T], [q_now, *Q]))
    except EStopped as e:
        raise Refused(f"STOPPED, nothing moved: the arm is e-stopped ({e})") from e
    except ArmRefused as e:
        raise Refused(
            f"STOPPED, nothing moved: the driver did not take the transit ({e}). The driver rejects goals while "
            "the arm is FAULTED: clear the fault on the arm, then re-run the place"
        ) from e
    if result.error_code != SUCCESSFUL:
        raise Refused(
            f"STOPPED: the transit ended {rammp_arm.ERRORS.get(result.error_code, result.error_code)} "
            f"{result.error_string} (a faulted arm rejects goals: check the arm)"
        )
    time.sleep(SETTLE_AFTER_S)
    q_pre = np.asarray(arm.raw_joints(), dtype=float)
    d, turn = end_pose_error(model, q_pre, cand["pre"], h)
    log(f"in front of the opening: {d * 100:.1f} cm, {turn:.1f} deg from the planned pre-insert pose")
    if d > END_TOL or turn > END_TOL_DEG:
        raise Refused(
            f"STOPPED in front of the opening: the transit ended {d * 100:.1f} cm / {turn:.1f} deg from the "
            f"planned pose (max {END_TOL * 100:.0f} cm / {END_TOL_DEG:g} deg) -- not going in"
        )

    # the line in, planned again from where the arm is and checked again
    ctx.progress("insert", 0.0)
    try:
        Tp = model.fk(q_pre)
        line = plan_line(model, q_pre, cand["target"] - Tp[:3, 3], "line in", step=LINE_STEP)
        check_line(model, line, cand["pre"], cand["target"], h, "line in")
        worst, _ = check_path(
            model,
            checker,
            [("insert", line, LINE_STEP_DEG)],
            R0,
            c["clear_min"],
            c["tilt_max_deg"],
            c["check_container"],
        )
    except Refused as e:
        raise Refused(f"STOPPED in front of the opening, the line in from where the arm is: {e}") from e
    lengths = path_lengths(model, line)
    log(
        f"-> line in: {sum(lengths) * 100:.1f} cm at {c['insert_speed'] * 100:.1f} cm/s, clearance "
        f"{worst['insert'][0] * 100:.1f} cm ({worst['insert'][1]})"
    )
    if not arm.move_joints(line[1:], cart_lengths=lengths, speed_scale=c["insert_speed"] / rammp_arm.CART_SPEED):
        raise Refused(f"STOPPED: the line in did not complete ({arm.last_error})")
    time.sleep(SETTLE_AFTER_S)
    # the line in as run: the back-out retraces its joints (also `retreat`'s)
    insert = dict(
        line=[np.asarray(q, dtype=float) for q in line],
        target=cand["target"],
        pre=cand["pre"],
        heading=h,
        boxes=boxes,
        time=time.time(),
    )
    save("last_insert.json", insert)
    q_in = np.asarray(arm.raw_joints(), dtype=float)
    d, turn = end_pose_error(model, q_in, cand["target"], h)
    log(f"in the oven: {d * 100:.1f} cm, {turn:.1f} deg from the target")
    if d > END_TOL or turn > END_TOL_DEG:
        raise Refused(
            f"STOPPED in the oven, holding the container: the hand is {d * 100:.1f} cm / {turn:.1f} deg from the "
            "target -- not lowering"
        )

    # the line down from where the arm is, checked; Walter's impedance lowering
    try:
        descent, upper, _ = plan_down(model, q_in, floor_z, log)
        check_path(
            model,
            checker,
            [("descent", upper, LINE_STEP_DEG)],
            R0,
            c["clear_min"],
            c["tilt_max_deg"],
            c["check_container"],
            skip=(FLOOR_BOX,),
        )
    except Refused as e:
        raise Refused(f"STOPPED in the oven, holding the container, not lowering: {e}") from e
    zs = descent[0]
    log(
        f"line down: tool z {zs[0]:.3f} -> {zs[-1]:.3f}; the box meets the glass at tool z "
        f"{floor_z + BOX_BELOW:.3f} (the lower finger touches ~1.6 cm before it)"
    )
    rec = Recorder()
    cfg = {**c, "table_z": floor_z, "bowl_depth": BOX_BELOW}
    path = None
    try:
        ctx.progress("lower", 0.0)
        lower(arm, model, descent, cfg, rec, log)
    finally:
        path = rec.write(descents_dir)
        if path is not None:
            log(f"recorded {len(rec.rows)} /ee_state messages -> {path} (plot: plot_descent.py)")
    # out of impedance (it undershoots in here, and the hand is pressing), then everything in position control
    try:
        to_position(arm, log)
    except Refused as e:
        raise Refused(f"STOPPED on the glass, gripper CLOSED, still in impedance: {e}") from e
    release_and_back_out(arm, model, checker, insert, cfg, ctx, log, open_gripper=True)
    log("Done: container placed, hand back at the pre-insert pose outside the opening (position control).")
    return {"executed": True, "recording": str(path)}


def release_and_back_out(arm, model, checker, insert, cfg, ctx, log, open_gripper, fingers=None):
    """From where the arm is (position control): the release lift (the open fingers clear of the glass and
    the ceiling), open (open_gripper), back onto the line in's end, the line in's joints reversed to the
    pre-insert pose. Each planned from the measured joints and checked first; each move's end measured."""
    c = cfg
    state = "gripper CLOSED" if open_gripper else "gripper open"
    # the fingers' angle (rad) once released: the release's opening, or (retreat) as they are now
    released = RELEASE_RAD if open_gripper else float(fingers)
    line_in = [np.asarray(q, dtype=float) for q in insert["line"]]
    target, pre, h = np.asarray(insert["target"]), np.asarray(insert["pre"]), np.asarray(insert["heading"])

    # 1. the lift
    q = np.asarray(arm.raw_joints(), dtype=float)
    try:
        lift_line, lift, w_lift = plan_release_lift(
            model, checker, q, c["clear_min"], closed=True if open_gripper else released, released=released
        )
    except Refused as e:
        raise Refused(f"STOPPED at the hold, {state}, position control: {e}") from e
    if lift > 0:
        ctx.progress("release lift", 0.0)
        top = model.fk(lift_line[-1])[:3, 3]
        log(
            f"-> release lift: {lift * 100:.1f} cm straight up (position control); then the open fingers are "
            f"{w_lift['released'][0] * 100:.1f} cm from every box ({w_lift['released'][1]})"
        )
        move_line(arm, model, lift_line, "the release lift", c["insert_speed"], state)
        q = np.asarray(arm.raw_joints(), dtype=float)
        d, turn = end_pose_error(model, q, top, h)
        d_open, what = checker.clearance(q, closed=released)
        log(f"lifted: {d * 100:.1f} cm from the planned pose; open fingers {d_open * 100:.1f} cm from every box")
        if d > END_TOL or turn > END_TOL_DEG or d_open < c["clear_min"]:
            raise Refused(
                f"STOPPED after the release lift, {state}: the hand is {d * 100:.1f} cm / {turn:.1f} deg from the "
                f"planned pose, the open fingers would be {d_open * 100:.1f} cm from {what} "
                f"(need {c['clear_min'] * 100:.1f})"
            )
    else:
        log("release: the open fingers clear every box here already, no lift")

    # 2. open
    if open_gripper:
        ctx.progress("open", 0.0)
        open_to(arm, Recorder(), log, RELEASE_OPEN)
        time.sleep(0.5)

    # 3. back onto the line in's end, then the line in's joints reversed
    q = np.asarray(arm.raw_joints(), dtype=float)
    try:
        back, worst = plan_back_out(model, checker, q, line_in, c["clear_min"], released=released)
    except Refused as e:
        raise Refused(f"STOPPED, gripper open, hand NOT backed out (position control): {e}") from e
    ctx.progress("back-out", 0.0)
    log(
        "-> back onto the line in's end, then back along it: clearance "
        + ", ".join(f"{k} {v[0] * 100:.1f} cm ({v[1]})" for k, v in worst.items())
    )
    if not arm.move_joints([back[0]], speed_scale=ONTO_SPEED_SCALE):
        raise Refused(f"STOPPED, gripper open: the move back onto the line in failed ({arm.last_error})")
    time.sleep(SETTLE_AFTER_S)
    d, turn = end_pose_error(model, np.asarray(arm.raw_joints(), dtype=float), target, h)
    if d > END_TOL or turn > END_TOL_DEG:
        raise Refused(
            f"STOPPED, gripper open, in the oven: back onto the line in ended {d * 100:.1f} cm / {turn:.1f} deg "
            "from its end -- not backing out"
        )
    move_line(arm, model, back, "the back-out", c["insert_speed"], "gripper open")
    d, turn = end_pose_error(model, np.asarray(arm.raw_joints(), dtype=float), pre, h)
    log(f"backed out: {d * 100:.1f} cm, {turn:.1f} deg from the pre-insert pose")
    if d > END_TOL or turn > END_TOL_DEG:
        raise Refused(
            f"STOPPED, gripper open: the back-out ended {d * 100:.1f} cm / {turn:.1f} deg from the pre-insert "
            f"pose (max {END_TOL * 100:.0f} cm / {END_TOL_DEG:g} deg)"
        )


def move_line(arm, model, qs, what, speed, state):
    """move_joints through qs[1:] (stiff), timed in metres at `speed` (m/s); Refused if it fails."""
    if not arm.move_joints(qs[1:], cart_lengths=path_lengths(model, qs), speed_scale=speed / rammp_arm.CART_SPEED):
        raise Refused(f"STOPPED, {state}: {what} did not complete ({arm.last_error})")
    time.sleep(SETTLE_AFTER_S)


def retreat(arm, model, insert, cfg, execute, ctx, log):
    """The release lift and back-out alone, after a place stopped with the gripper open (at most RELEASE_OPEN),
    retracing `insert` (the last executed line in, with its boxes), the fingers checked as they are now."""
    gripper = float(arm.gripper_state().position)
    if gripper > RELEASE_OPEN + RELEASE_TOL:
        raise Refused(
            f"the gripper is at {gripper:.2f} (need <= {RELEASE_OPEN + RELEASE_TOL:.2f}) -- "
            "backing out "
            "would drag what it holds; open it first"
        )
    fingers = gripper * GRIPPER_FULL_RAD  # the fingers as they are now
    q = np.asarray(arm.raw_joints(), dtype=float)
    checker = Checker(model, insert["boxes"])
    try:
        lift_line, lift, _ = plan_release_lift(model, checker, q, cfg["clear_min"], closed=fingers, released=fingers)
        _, worst = plan_back_out(model, checker, lift_line[-1], insert["line"], cfg["clear_min"], released=fingers)
        summary = dict(
            lift_cm=cm(lift),
            clearance_cm={k: [cm(d), w] for k, (d, w) in worst.items()},
            to=np.round(insert["pre"], 3).tolist(),
        )
        if not execute:
            return {"executed": False, **summary}
        ready(arm)
        to_position(arm, log)  # in case the place stopped in impedance
        release_and_back_out(arm, model, checker, insert, cfg, ctx, log, open_gripper=False, fingers=fingers)
    finally:
        checker.close()
    return {"executed": True, **summary}


def check_against(model, boxes, legs, clear_min, skip=(), closed=True):
    """check_path against `boxes` (the last place plan's obstacles; clear_min, no tilt check)."""
    checker = Checker(model, boxes)
    try:
        return check_path(model, checker, legs, None, clear_min, None, skip=skip, closed=closed)
    finally:
        checker.close()


def ready(arm):
    """Refused unless the arm may be commanded now."""
    why = arm.not_ready()
    if why is not None:
        raise Refused(why)


def check_tool_frame(arm, model, q, tol, log):
    """Refused unless the arm's reported tool frame is within `tol` (m) of the model's at q."""
    reported = np.asarray(arm.get_state()["ee_pos"][:3], dtype=float)
    modelled = model.fk(q)[:3, 3]
    off = float(np.linalg.norm(reported - modelled))
    log(
        f"tool frame: arm says {np.round(reported, 3).tolist()}, "
        f"model {np.round(modelled, 3).tolist()} ({off * 100:.1f} cm apart)"
    )
    if off > tol:
        raise Refused("the arm's tool frame doesn't match the model's -- check the tool configuration")


def path_lengths(model, qs):
    """The tool's travel (m) between consecutive joint waypoints."""
    return [
        float(np.linalg.norm(model.fk(b)[:3, 3] - model.fk(a)[:3, 3])) for a, b in zip(qs[:-1], qs[1:], strict=True)
    ]
