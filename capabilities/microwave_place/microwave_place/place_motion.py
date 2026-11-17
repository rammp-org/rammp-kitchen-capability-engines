"""The arm side of placing the held container in the open microwave: planning, checking and running it.

From Walter Hsieh's place_container (plan.py and lower.py, WalterHsieh0713/rammp-kitchen-capability-engines
feature/7-place-container), unchanged: the hold model, the arm model, the checked joint moves and straight
lines, the straight line down in the driver's joint impedance mode with the contact stop, the hold, the
recording. Added for the microwave:

  plan_line / check_line: a straight line of joint waypoints, and the check that every pose is on it;
  "Into the microwave": the microwave as one obstacle box, the hand's heading into the oven, the checks on
    the arm model (clearance per link against that box for the side shift and cuRobo's transit only; tilt and
    joint travel), the straight lines, the release tilt and the level back-out (not collision-checked);
  "The place on the arm": the side-shift and heading search, the plan's summary, and running the plan
    (run_place).

Everything is in arm_mount, the driver's gravity-aligned frame: z true up, x forward, y left, origin on
the arm base axis. The held container is modelled level (container_in_hand): an 18 x 18 x 10 cm box, its
centre CONTAINER_REACH ahead of the tool frame along the hand's level heading (grab_container's grasp). It
places the target and the pre-insert pose; it is not collision-checked.
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
from scipy.spatial.transform import Rotation

from . import microwave_detection

# --- the container and how it is held ---
# grab_container (grasp.py) pinches the brim with the hand pitched GRASP_PITCH_DEG down (90 - its
# APPROACH_TILT_DEG), the tool frame GRASP_DEPTH along the approach and GRASP_SETBACK level short of the brim's
# outer edge, the container level; the container turns with the hand after.
CONTAINER_RADIUS = 0.0843  # m, the brim's outer radius (grab_container RING_RADIUS_M: the 17 cm tupperware)
GRASP_PITCH_DEG = 15.0
GRASP_DEPTH = 0.008  # m (grab_container GRASP_DEPTH_M, -0.008: short of the edge)
GRASP_SETBACK = 0.01  # m (grab_container GRASP_SETBACK_M)
FLANGE_TO_TIP = 0.17955  # m, flange face to the closed fingertips (Gen3 + 2F-85; not ruler-measured)
# --- the motion ---
PRESS_BELOW = 0.10  # m, the line down is aimed this far below where the bottom meets the table: in
# impedance the hand lags its command by a few cm, and the table, not the command, stops it
MIN_TOOL_HEIGHT = 0.02  # m, the tool frame is never targeted lower than this above the table
PATH_STEP = 0.002  # m, between the line's joint waypoints
RETREAT_STEP = 0.004  # m, plan_line's default step between joint waypoints
# --- checks ---
IK_TOL = 0.005  # m
IK_ROT_TOL_DEG = 1.0
UP = np.array([0.0, 0.0, 1.0])


class Refused(Exception):
    """A plan or check that does not hold: nothing more is commanded."""


def wrap(rad):
    return (np.asarray(rad) + np.pi) % (2 * np.pi) - np.pi


def wrap_deg(rad):
    """Angle difference in radians -> degrees in [-180, 180)."""
    return np.degrees(wrap(rad))


CONTAINER_REACH = CONTAINER_RADIUS + GRASP_SETBACK + GRASP_DEPTH * np.cos(np.radians(GRASP_PITCH_DEG))  # m, level


def container_in_hand(R):
    """The held container with the hand at orientation R (arm_mount): (its box's centre from the tool frame, its
    up axis, its level heading), all in arm_mount. The container is level (up = world up, 10 cm tall), its
    centre CONTAINER_REACH ahead of the tool frame along the hand's level heading, at the brim's height (the
    tool's) plus half of BOX_ABOVE - BOX_BELOW."""
    z = R[:, 2]
    level = np.array([z[0], z[1], 0.0]) / np.linalg.norm(z[:2])
    centre = CONTAINER_REACH * level + 0.5 * (BOX_ABOVE - BOX_BELOW) * UP
    return centre, UP.copy(), level


def container_reach(R):
    """How far (m, level) the container's centre is ahead of the tool frame with the hand at R."""
    return float(np.linalg.norm(container_in_hand(R)[0][:2]))


def _so3_left_jacobian_inv(r):
    """The inverse left Jacobian of SO(3) at rotation vector r."""
    th = float(np.linalg.norm(r))
    K = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    if th < 1e-6:
        return np.eye(3) - 0.5 * K
    return np.eye(3) - 0.5 * K + (1.0 / th**2 - (1.0 + np.cos(th)) / (2.0 * th * np.sin(th))) * (K @ K)


class Model:
    """FK and IK of the arm on the process's shared sim copy (capability_engine.sim: the driver's
    model, rooted at arm_mount). fk() gives 4x4 poses in arm_mount of the tool frame (the sim's ee
    link, arm_tool_frame), the flange or the fingertips."""

    def __init__(self, flange_to_tip=FLANGE_TO_TIP):
        import pybullet as p

        self.scene, self.rb = sim.make_sim()
        self.flange_to_tip = float(flange_to_tip)
        info = [p.getJointInfo(self.rb.robot_id, j, physicsClientId=self.rb.physics_client_id) for j in sim.ARM]
        # (lower, upper) in rad; continuous joints come back as lower > upper
        self.limits = [(float(i[8]), float(i[9])) for i in info]
        cid, rid = self.rb.physics_client_id, self.rb.robot_id
        # calculateJacobian takes every movable joint (in index order); the arm's columns among them
        self._movable = [
            j
            for j in range(p.getNumJoints(rid, physicsClientId=cid))
            if p.getJointInfo(rid, j, physicsClientId=cid)[2] != p.JOINT_FIXED
        ]
        self._arm_cols = [self._movable.index(j) for j in sim.ARM]
        base = self.scene.robot_base_pose
        self._R_base_inv = Rotation.from_quat(base.orientation).inv().as_matrix()

    def jacobian(self, q):
        """(3x7 linear, 3x7 angular) Jacobian of the tool frame at arm joints q, in arm_mount (PyBullet's)."""
        import pybullet as p

        cid, rid = self.rb.physics_client_id, self.rb.robot_id
        self.fk(q)  # sets the arm joints
        qa = [float(p.getJointState(rid, j, physicsClientId=cid)[0]) for j in self._movable]
        zero = [0.0] * len(qa)
        lin, ang = p.calculateJacobian(
            rid, self.rb.end_effector_id, [0.0, 0.0, 0.0], qa, zero, zero, physicsClientId=cid
        )
        lin, ang = np.asarray(lin)[:, self._arm_cols], np.asarray(ang)[:, self._arm_cols]
        return self._R_base_inv @ lin, self._R_base_inv @ ang

    def fk(self, q, link="tool"):
        pos, quat = sim.tool_pose(self.scene, self.rb, q)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        offset = {"tool": 0.0, "flange": -TOOL_Z, "tip": self.flange_to_tip - TOOL_Z}[link]
        T[:3, 3] += offset * T[:3, 2]
        return T

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

        def jac(x):
            # d(rotvec of R(x) target^-1)/dx = Jl^-1(r) @ angular Jacobian (R changes by [w]x R)
            lin, ang = self.jacobian(x)
            T = self.fk(x)
            r = (Rotation.from_matrix(T[:3, :3]) * target.inv()).as_rotvec()
            return np.vstack([lin, rot_weight * (_so3_left_jacobian_inv(r) @ ang), stay * np.eye(len(x))])

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
            # 1e-8: the same poses as tighter tolerances (PyBullet's single-precision FK limits them to ~0.1 mm)
            r = least_squares(residual, x0, jac=jac, bounds=(lo, hi), xtol=1e-8, ftol=1e-8, gtol=1e-8)
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
    """Refused unless the tool at every joints of qs lies within LINE_TOL of the segment a-b and (axis not None)
    points (horizontally) within LINE_TOL_DEG of `axis`."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = None if axis is None else np.asarray(axis, float)
    ab = b - a
    for i, q in enumerate(qs):
        T = model.fk(q)
        p = T[:3, 3]
        t = 0.0 if ab @ ab < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        off = float(np.linalg.norm(p - (a + t * ab)))
        h = np.array([T[0, 2], T[1, 2]])
        turn = (
            0.0
            if axis is None
            else float(np.degrees(np.arccos(np.clip(h @ axis[:2] / max(np.linalg.norm(h), 1e-9), -1.0, 1.0))))
        )
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
    """Whether contact may be called at time fraction u of the descent: the commanded speed is at least
    CONTACT_GATE x lower_speed, or x the descent's own peak speed when that is lower (a short descent, e.g. one
    starting low: self.arm makes every move last at least a second) -- at the start and end of a minimum-jerk
    move it is slow, and a hand that slows with its command must not read as stopped by the table."""
    peak = 1.875 * length / duration  # a minimum-jerk move's peak speed
    return 0.0 < u < 1.0 and minjerk_speed(u) * length / duration >= CONTACT_GATE * min(lower_speed, peak)


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
SETTLE_STILL_S = 0.1  # after the settle the hand must be still this long ...
SETTLE_WAIT_S = 5.0  # ... within this long
EXIT_SETTLE_S = 1.0  # leaving impedance: wait at most this long for the arm to be still ...
EXIT_HOLD_S = 0.3  # ... then a position goal on the current joints, this long
STILL_AFTER_S = 1.0  # after an impedance move (the hold, the tilt): wait at most this long for the arm to be still
OPEN_WAIT_S = 3.0  # ... within this long of the open command (sent twice at most)
GRIPPER_FULL_RAD = 0.8  # the finger knuckle's travel (the URDF's limit) = gripper position 1.0 (closed)
# In the oven the gripper opens only to RELEASE_OPEN (gripper position, 0 = fully open): the fingers stand
# vertically, and fully open they span ~15.5 cm of the ~17.5 cm between the glass and the ceiling (1.0 cm each
# side at best); at 0.5 they leave ~2 cm free on each side of the container's wall and need a ~1.5 cm lift for
# 1 cm to the glass and 2 cm to the ceiling (arm model, 2026-11-02).
RELEASE_OPEN = 0.5
RELEASE_TOL = 0.05  # the gripper must reach RELEASE_OPEN within this
LOG_COLUMNS = ["t", "phase", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "fx", "fy", "fz", "tx", "ty", "tz"]


def points(times, qs):
    """[(t, positions, velocities)] for execute_joint_trajectory; no velocities (the driver
    interpolates), as the RAMMP-table goals ran."""
    return [(float(t), [float(v) for v in q], []) for t, q in zip(times, qs, strict=True)]


ROT_LENGTH = 0.25  # m per rad: how a turn of the hand, or of any joint, counts in a path's length -- at a peak
# path speed v, nothing turns faster than v / ROT_LENGTH (16 cm/s -> ~37 deg/s)


def pose_lengths(model, qs):
    """Per segment of joint waypoints qs, the length a move is timed by (m): the tool's travel, or the hand's
    turn x ROT_LENGTH, or the largest joint change x ROT_LENGTH, whichever is longest. Timed by the tool's
    travel alone, a stretch where the hand mostly turns (the transit re-aiming it at the oven) gets almost no time
    and the wrist whips round (119 deg/s, a joint at 150 deg/s, 2026-11-04)."""
    T = [model.fk(np.asarray(q, dtype=float)) for q in qs]
    out = []
    for (a, b), (qa, qb) in zip(zip(T[:-1], T[1:], strict=True), zip(qs[:-1], qs[1:], strict=True), strict=True):
        turn = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
        joint = float(np.max(np.abs(wrap(np.asarray(qb, dtype=float) - np.asarray(qa, dtype=float)))))
        out.append(max(float(np.linalg.norm(b[:3, 3] - a[:3, 3])), ROT_LENGTH * float(turn), ROT_LENGTH * joint))
    return out


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
    hold's joints (where the release starts). Raises Refused (nothing let go) if it didn't settle,
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
    log(f"lowering from {lag * 100:+.1f} cm {'above' if lag >= 0 else 'below'} the planned start")

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
    wait_still(arm, STILL_AFTER_S, rec=rec)
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
        raise Refused(f"STOPPED, hand NOT backed out: the gripper only opened to {pos:.2f} (asked {position:.2f})")


# ---------------------------------------------------------------------------------------------
# Into the microwave: the microwave as one obstacle box, the hand's heading, cuRobo's transit to in front of
# the opening (checked against that box), a straight line in and everything after it (not collision-checked).
#
# The held tupperware is modelled as a BOX_SIDE x BOX_SIDE x (BOX_BELOW + BOX_ABOVE) box centred on the hand's
# axis as container_in_hand has it. The target puts that centre over the placement point,
# the tool `above` over the turntable glass, the hand pointing along the oven's axis turned by a heading
# offset (default 0; turned, the line in runs off the opening's middle: the 17 cm tupperware's margin to the
# opening's sides is ~6.8 cm at 0 deg, 1.7 at 15, none from ~19 deg, and nothing checks it). The
# pre-insert pose is back along that heading until the whole box is INSERT_OUT outside the outer front.
# ---------------------------------------------------------------------------------------------
BOX_SIDE, BOX_BELOW, BOX_ABOVE = (
    0.18,
    0.05,
    0.05,
)  # m: the tupperware's box (17 cm across, 1 cm margin); BOX_BELOW = tool frame to its bottom
INSERT_OUT = 0.03  # m, the pre-insert pose: the whole box this far outside the outer front
LINE_STEP = 0.004  # m, between the straight line's joint waypoints
TRANSIT_STEP_DEG = 1.0  # the check interpolates cuRobo's transit to at most this per joint ...
LINE_STEP_DEG = 0.5  # ... and the straight lines (in, down, out) to this
END_TOL, END_TOL_DEG = 0.01, 2.0  # m, deg: the line in ends this close to the target, pointing along the heading
DESCENT_BELOW_MEET = 0.01  # m, the line down must reach this far below where the box's bottom meets the glass
# The world: the microwave as one box, from the depth points of SAM3's "microwave" mask.
MICROWAVE_PCT = (2, 98)  # the microwave's box: these percentiles of its depth points, in oven coordinates
MICROWAVE_MIN_POINTS = 200
MICROWAVE_NEAR = dict(across=0.5, behind=0.7, below=0.15, above=0.5)  # m: the mask's depth points used for the box lie
# within this of the located opening (across its middle, behind its plane, below / above its bottom edge): a
# few of the open door's pixels have depth 1-2 m off (seen 2026-11-03), and they are not the oven
DEPTH_RANGE = (0.15, 2.5)  # m, the scene depth used
MICROWAVE = "microwave"  # the one obstacle box; only the first motion (side shift, transit) avoids it
RELEASE_TILT_DEG = 10.0  # after opening on the glass: the hand pitched this far nose-down (about the tool frame,
# in joint impedance) so the container slides off the fingers; it backs out tilted
TILT_STEP_DEG = 2.0  # between the tilt's joint waypoints
TILT_SPEED_SCALE = 1.0  # the tilt's speed_scale of self.arm's joint speed
TILT_JUMP_DEG = 15.0  # a tilt step moving any joint more than this is the IK flipping the arm: refused
# The release (2026-11-03): the gripper opens in place on the glass, still in the lowering's joint impedance;
# then the hand pitches RELEASE_TILT_DEG nose-down about the tool frame, in impedance, so the container slides
# off the fingers; then position control and a straight level line out, the hand kept tilted, to above the
# pre-insert pose. None of it is collision-checked.


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
    """The obstacle boxes as bodies in the shared sim, for clearance checks (the arm and gripper; the held
    container is not checked). close()
    removes them; use it in a finally."""

    def __init__(self, model, boxes):
        self.model, self.bodies = model, []
        try:
            for b in boxes:
                self.bodies.append((_box_body(model, b), b["name"]))
        except BaseException:
            self.close()
            raise

    def clearance(self, q, closed=True):
        """(smallest distance, "link vs box") at joints q from every arm and gripper link (`closed`: True closed,
        False fully open, a number: the fingers at that angle in rad) to every box."""
        return finger_clearance(self.model.rb, q, self.bodies, closed)

    def close(self):
        import pybullet as p

        cid = self.model.rb.physics_client_id
        for bid, _ in self.bodies:
            p.removeBody(bid, physicsClientId=cid)
        self.bodies = []


def microwave_obstacles(oven, depth_m, K, world_T_cam, mw_mask):
    """The obstacle: the microwave's body as one box, aligned with the oven (oven.box): from its outer front
    back, spanning MICROWAVE_PCT of the depth points of SAM3's "microwave" mask at or behind that front, in
    oven coordinates (what is in front of the front, the open door, is not in it). Only the first motion
    avoids it.

    Returns ([box], [box] for cuRobo, info). Raises Refused without the mask or with too few points."""
    valid = np.isfinite(depth_m) & (depth_m > DEPTH_RANGE[0]) & (depth_m < DEPTH_RANGE[1])
    if mw_mask is None or mw_mask.shape != valid.shape:
        raise Refused("no SAM3 'microwave' mask on the scene frame")
    Tc = np.asarray(world_T_cam, dtype=float)
    P = microwave_detection._backproject(np.where(valid, depth_m, 0), np.asarray(K, float)) @ Tc[:3, :3].T + Tc[:3, 3]
    P = P[mw_mask & valid].astype(float)
    if len(P) < MICROWAVE_MIN_POINTS:
        raise Refused(f"{len(P)} depth points on SAM3's 'microwave' mask (need {MICROWAVE_MIN_POINTS})")
    Po = oven.to_oven(P)
    front = -microwave_detection.RECESS  # the oven's outer front (S = 0 is the opening's plane)
    Po = Po[Po[:, 1] >= front - 0.01]  # the body: what is in front of it (the open door, the arm) is not the box
    n = MICROWAVE_NEAR
    Po = Po[
        (np.abs(Po[:, 0]) <= n["across"])
        & (Po[:, 1] <= n["behind"])
        & (Po[:, 2] >= -n["below"])
        & (Po[:, 2] <= n["above"])
    ]
    if len(Po) < MICROWAVE_MIN_POINTS:
        raise Refused(f"{len(Po)} 'microwave' depth points at or behind its front (need {MICROWAVE_MIN_POINTS})")
    lo, hi = np.percentile(Po, MICROWAVE_PCT, axis=0)
    box = oven.box(MICROWAVE, (lo[0], hi[0]), (front, hi[1]), (lo[2], hi[2]))
    info = dict(
        points=int(len(Po)),
        along_cm=[cm(front), cm(hi[1])],
        across_cm=[cm(lo[0]), cm(hi[0])],
        up_cm=[cm(lo[2]), cm(hi[2])],
    )
    return [box], [box], info


def write_world(path, boxes, frame):
    """cuRobo's world file (set_world): the boxes in `frame`."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(base_frame=frame, obstacles=boxes), f, sort_keys=False)


def heading_pose(model, q0, oven, offset_deg, above, closer=0.0):
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
    point = oven.point - closer * oven.n  # the spot, `closer` (m) back toward the opening
    target = np.r_[point[:2] - container_reach(Rotation.from_quat(quat).as_matrix()) * h[:2], oven.floor_z + above]
    # from the box's far side INSERT_OUT outside the outer front to the target, along the heading
    insert_m = ((point - oven.b0) @ oven.n + BOX_SIDE / 2 + microwave_detection.RECESS + INSERT_OUT) / max(
        np.cos(off), 0.3
    )
    return dict(
        offset_deg=float(offset_deg), heading=h, quat=quat, target=target, pre=target - insert_m * h, insert_m=insert_m
    )


def probe_heading(model, q0, cand):
    """The quick tests of a heading before cuRobo: IK at the target and at the pre-insert pose, and the straight
    line in from there (no collision check: from the line in on nothing is). Raises StageFailed."""
    qg = model.ik(cand["target"], cand["quat"], q0)
    if qg is None:
        raise StageFailed("goal", "no IK solution")
    qp = model.ik(cand["pre"], cand["quat"], q0)
    if qp is None:
        raise StageFailed("pre-insert", "no IK solution")
    _stage("line in", plan_line, model, qp, cand["insert_m"] * cand["heading"], "line in", step=LINE_STEP)


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


BEYOND_TOOL = 0.30  # m: every point of the gripper and the held box is within this of the tool frame


def joint_radii(model, q):
    """Per arm joint at q (m): at most how far any point moved by that joint (the links after it, the gripper,
    the held box) is from its axis. A joint move dq (rad) moves no point more than |dq| @ joint_radii."""
    import pybullet as p

    tool = model.fk(q)[:3, 3]
    cid, rid = model.rb.physics_client_id, model.rb.robot_id
    return np.array(
        [
            np.linalg.norm(
                tool - np.asarray(p.getLinkState(rid, j, computeForwardKinematics=True, physicsClientId=cid)[4])
            )
            + BEYOND_TOOL
            for j in sim.ARM
        ]
    )


def check_tilt(model, legs, R0, tilt_max, stage="check"):
    """The hand's tilt vs the tool orientation R0 at every densely interpolated pose of the legs (name, joint
    waypoints, step_deg): the largest; StageFailed(stage, ...) above tilt_max. No collision check."""
    up_tool = np.asarray(R0).T @ UP
    tilt = 0.0
    for _, qs, step_deg in legs:
        for q in dense(qs, step_deg):
            Rq = model.fk(q)[:3, :3]
            tilt = max(tilt, float(np.degrees(np.arccos(np.clip((Rq @ up_tool) @ UP, -1.0, 1.0)))))
    if tilt > tilt_max:
        raise StageFailed(stage, f"tilt {tilt:.1f} deg > {tilt_max:.1f}")
    return tilt


def check_path(model, checker, legs, R0, clear_min, tilt_max, closed=True, stage="check"):
    """Every leg (name, joint waypoints, step_deg), densely interpolated: clearance to every box and the tilt vs
    the tool orientation R0 (None: not checked).
    Clearance is computed at the first and last sample and wherever the samples since the last computed one
    could have moved some point (joint_radii) more than that one's clearance above clear_min: the samples
    skipped are at least clear_min clear by that bound. The tilt is checked at every sample.
    Returns {leg: [clearance, "link vs box"]} and the largest tilt; raises StageFailed(stage, ...) listing
    every leg under clear_min and a tilt over tilt_max."""
    up_tool = None if R0 is None else np.asarray(R0).T @ UP
    worst, tilt = {}, 0.0
    for name, qs, step_deg in legs:
        best = (9.0, "")
        D = dense(qs, step_deg)
        covered = 0.0  # how far (m) any point can have moved since the last sample checked for clearance
        margin, radii = -1.0, np.zeros(len(JOINTS))
        for k, q in enumerate(D):
            if k > 0:
                covered += float(np.abs(wrap(np.asarray(q) - np.asarray(D[k - 1]))) @ radii)
            if k == 0 or k == len(D) - 1 or covered > margin:
                d = checker.clearance(q, closed)
                best = min(best, d)
                margin, covered = d[0] - clear_min, 0.0
                radii = joint_radii(model, q)
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
    finger touches the glass ~1.3 cm before the box does on the arm model)."""
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
    """Everything after cuRobo's transit `Qs` for heading `cand`: the transit must end at the pre-insert pose and
    is densely checked (every link clear_min from the microwave box); the straight line in from there (not
    collision-checked) must end within END_TOL / END_TOL_DEG of the target; the tilt stays within tilt_max_deg of
    the start on both; the line down from the target, and from where the model has the lowering stop the
    release tilt (plan_tilt) and the level back-out (plan_level_back_out), none of them
    collision-checked. Raises StageFailed. Returns the line in, the descent, the transit's clearance, the tilt,
    the line in's end error and the release's numbers."""
    pre, target, h, L = cand["pre"], cand["target"], cand["heading"], cand["insert_m"]
    clear_min, tilt_max = cfg["clear_min"], cfg["tilt_max_deg"]
    R0 = model.fk(q0)[:3, :3]
    _stage("pre-insert", check_line, model, [Qs[-1]], pre, pre, h, "cuRobo's end")
    line = _stage("line in", plan_line, model, Qs[-1], L * h, "line in", step=LINE_STEP)
    _stage("line in", check_line, model, line, pre, target, h, "line in")
    worst, tilt = check_path(model, checker, [("transit", Qs, TRANSIT_STEP_DEG)], R0, clear_min, tilt_max)
    tilt = max(tilt, check_tilt(model, [("insert", line, LINE_STEP_DEG)], R0, tilt_max))
    end_off, end_turn = end_pose_error(model, line[-1], target, h)
    if end_off > END_TOL or end_turn > END_TOL_DEG:
        raise StageFailed("check", f"the line in ends {end_off * 100:.1f} cm / {end_turn:.1f} deg off the target")
    descent, upper, _ = plan_down(model, line[-1], floor_z, log)
    check_tilt(model, [("descent", upper, LINE_STEP_DEG)], R0, tilt_max)
    # the release and back-out, from where the model has the lowering stop
    i_hold = contact_index(descent, floor_z)
    q_hold = descent[1][i_hold]
    tilt_line = plan_tilt(model, q_hold)
    plan_level_back_out(model, tilt_line[-1], pre)
    return dict(
        line=line,
        descent=descent,
        clearance=worst,
        tilt=tilt,
        end=(end_off, end_turn),
    )


def contact_index(descent, floor_z):
    """Where on the line down (zs, qs) the model has the lowering stop: the first waypoint where the box's bottom
    meets the glass."""
    zs = descent[0]
    meet = floor_z + BOX_BELOW
    return next((i for i, z in enumerate(zs) if z <= meet), len(zs) - 1)


def plan_tilt(model, q, deg=RELEASE_TILT_DEG):
    """The hand pitched `deg` nose-down about the tool frame (the tool point stays) from q, about the hand's level
    left axis, in TILT_STEP_DEG steps: its joint waypoints, the first q."""
    T = model.fk(q)
    z = T[:3, 2]
    left = np.cross(UP, np.array([z[0], z[1], 0.0]) / np.linalg.norm(z[:2]))
    n = max(1, int(np.ceil(deg / TILT_STEP_DEG)))
    qs, seed = [np.asarray(q, dtype=float)], np.asarray(q, dtype=float)
    for k in range(1, n + 1):
        R = Rotation.from_rotvec(np.radians(deg * k / n) * left).as_matrix() @ T[:3, :3]
        qk = model.ik(T[:3, 3], Rotation.from_matrix(R).as_quat(), seed, restarts=0, stay=0.01)
        if qk is None or np.max(np.abs(wrap_deg(qk - seed))) > TILT_JUMP_DEG:
            raise StageFailed("release", f"the hand can't pitch {deg * k / n:.0f} of {deg:g} deg nose-down here")
        qs.append(qk)
        seed = qk
    return qs


def plan_level_back_out(model, q_from, pre):
    """From q_from (gripper open, the hand as it is, tilted after the release) the straight level line back out
    to above the pre-insert pose, at q_from's height, the hand's orientation kept: its joint waypoints, every pose
    checked to be on the line (not collision-checked). Raises StageFailed("back-out", ...)."""
    a = model.fk(q_from)[:3, 3]
    b = np.r_[np.asarray(pre, dtype=float)[:2], a[2]]
    line = _stage("back-out", plan_line, model, q_from, b - a, "back-out", step=LINE_STEP)
    _stage("back-out", check_line, model, line, a, b, None, "back-out")
    return line


# ---------------------------------------------------------------------------------------------
# The place on the arm: the side-shift and heading search, the plan's summary, and running it (approach,
# lowering, release, back-out).
# ---------------------------------------------------------------------------------------------
START_TOL_DEG = 1.0  # the arm must be this close (every joint) to where the checked transit starts
SETTLE_AFTER_S = 0.5  # after the transit and the line in, before the arm's joints are read back


def cm(d):
    return round(float(d) * 100, 1)


SHIFT_STEP = 0.004  # m, between the side shift's joint waypoints


def side_left(model, q0):
    """The hand's level left at q0 (unit, arm_mount): + side shifts go this way."""
    z = model.fk(q0)[:3, 2]
    h = np.array([z[0], z[1], 0.0]) / np.linalg.norm(z[:2])
    return np.cross(UP, h)


def plan_side_shift(model, checker, q0, shift_m, cfg):
    """The straight level move of the hand `shift_m` to its left (- = right) from q0, orientation kept,
    checked (the arm and gripper clear_min from the microwave box, tilt). Returns its joint waypoints ([q0] for
    no shift)."""
    if abs(shift_m) < 1e-9:
        return [np.asarray(q0, dtype=float)]
    a = model.fk(q0)[:3, 3]
    b = a + shift_m * side_left(model, q0)
    line = _stage("side shift", plan_line, model, q0, b - a, "side shift", step=SHIFT_STEP)
    h0 = model.fk(q0)[:3, 2]
    _stage(
        "side shift",
        check_line,
        model,
        line,
        a,
        b,
        np.array([h0[0], h0[1], 0.0]) / np.linalg.norm(h0[:2]),
        "side shift",
    )
    check_path(
        model,
        checker,
        [("side shift", line, LINE_STEP_DEG)],
        model.fk(q0)[:3, :3],
        cfg["clear_min"],
        cfg["tilt_max_deg"],
        stage="side shift",
    )
    return line


def search_place(arm, model, checker, q0, oven, cfg, log):
    """For each of side_shifts_cm in turn (the hand moved that far level to its left, - = right, before the
    transit) and each of heading_offsets_deg: the side shift, the quick probe, cuRobo's transit from the
    shifted pose and plan_place_in. The first that passes every stage wins: (cand, with cand["shift"] the side
    shift's joints; cuRobo's plan; plan_place_in's result; attempts). Refused listing every attempt."""
    c, attempts = cfg, []
    for s_cm in c["side_shifts_cm"]:
        for off in c["heading_offsets_deg"]:
            tag = f"shift {float(s_cm):+.0f} cm, heading {float(off):+.0f} deg"
            try:
                shift = plan_side_shift(model, checker, q0, float(s_cm) / 100.0, c)
                q_s = shift[-1]
                cand = heading_pose(model, q_s, oven, float(off), c["above"], c["closer"])
                probe_heading(model, q_s, cand)
                plan = arm.plan_to_pose(cand["pre"], cand["quat"], q_s, hold=HOLD_LEVEL)
                if not plan.success:
                    raise StageFailed("cuRobo", " ".join(plan.message.split())[:200] or "no plan")
                if len(plan.q) < 2:
                    raise StageFailed("cuRobo", f"a plan of {len(plan.q)} samples")
                travel = check_transit_travel(np.asarray(plan.q, dtype=float))
                log(f"{tag}: cuRobo's transit, joint travel {np.round(travel, 0).tolist()} deg")
                res = plan_place_in(model, checker, q0, cand, np.asarray(plan.q, dtype=float), oven.floor_z, c, log)
            except StageFailed as e:
                attempts.append(dict(shift_cm=float(s_cm), offset_deg=float(off), stage=e.stage, reason=e.reason))
                log(f"{tag}: FAILED at {e.stage}: {e.reason}")
                continue
            attempts.append(dict(shift_cm=float(s_cm), offset_deg=float(off), stage="passed", reason=""))
            log(f"{tag}: passed every stage")
            cand = {**cand, "shift_cm": float(s_cm), "shift": shift}
            return cand, plan, res, attempts
    raise Refused(
        "nothing passed: "
        + "; ".join(
            f"shift {a['shift_cm']:+.0f} heading {a['offset_deg']:+.0f} {a['stage']}: {a['reason']}" for a in attempts
        )
    )


def summarize(model, q0, loc, oven, world, world_info, boxes, cand, plan, res, attempts, cfg):
    """(the dry run's summary, the plan record written to state_dir) for the chosen heading."""
    c = cfg
    Qs, ts, line = np.asarray(plan.q, dtype=float), np.asarray(plan.t, dtype=float), res["line"]
    knots = approach_knots(cand["shift"], Qs, line)
    samples = rammp_arm.joint_trajectory(
        q0, knots, pose_lengths(model, [q0, *knots]), c["insert_speed"] / rammp_arm.CART_SPEED
    )
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
        side_shift_cm=cand["shift_cm"],
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
        duration_s=dict(approach=round(float(samples[0][-1]) if samples is not None else 0.0, 1)),
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
        shift_cm=cand["shift_cm"],
        q_shift=np.asarray(cand["shift"]),
        quat=cand["quat"],
        target=cand["target"],
        pre=cand["pre"],
        insert_m=cand["insert_m"],
        floor_z=oven.floor_z,
        fit=loc["fit"],
        boxes=boxes,
        world=str(world),
        box=dict(
            side=BOX_SIDE,
            below=BOX_BELOW,
            above=BOX_ABOVE,
            reach=container_reach(model.fk(q0)[:3, :3]),
        ),
    )
    return summary, record


REUSE_TOL_DEG = 0.3  # the arm this close (every joint) to where a checked leg starts: that leg runs as checked


def _near(q, q_plan):
    return (
        float(np.max(np.abs(wrap_deg(np.asarray(q, dtype=float) - np.asarray(q_plan, dtype=float))))) <= REUSE_TOL_DEG
    )


def approach_knots(shift, Qs, line):
    """The joint knots of the one approach motion: the side shift, cuRobo's transit and the line in, each
    leg's first knot (the previous leg's last) left out."""
    knots = [np.asarray(q, dtype=float) for q in shift[1:]] + [np.asarray(q, dtype=float) for q in Qs[1:]]
    knots += [np.asarray(q, dtype=float) for q in line[1:]]
    return knots


def run_place(arm, model, q0, oven, cand, plan, cfg, ctx, log, descents_dir, res):
    """The checked place, on the arm (`res`: plan_place_in's result). The lowering's /ee_state recording goes
    to descents_dir."""
    c = cfg
    h, floor_z = cand["heading"], oven.floor_z
    R0 = model.fk(q0)[:3, :3]
    Qs = np.asarray(plan.q, dtype=float)
    ready(arm)
    q_now = np.asarray(arm.raw_joints(), dtype=float)
    off = float(np.max(np.abs(wrap_deg(q_now - q0))))
    if off > START_TOL_DEG:
        raise Refused(f"the arm is {off:.1f} deg from where the plan starts (max {START_TOL_DEG:g}): re-plan")

    # the side shift, cuRobo's transit and the line in, as planned and checked, in ONE motion (no stop between)
    line = [np.asarray(q, dtype=float) for q in res["line"]]
    knots = approach_knots(cand["shift"], Qs, line)
    lengths = pose_lengths(model, [q_now, *knots])
    ctx.progress("approach", 0.0)
    log(
        f"-> approach in one motion: side shift {cand['shift_cm']:+.0f} cm, transit, line in "
        f"{sum(path_lengths(model, line)) * 100:.0f} cm; {sum(lengths) * 100:.0f} cm of tool path at "
        f"{c['insert_speed'] * 100:.0f} cm/s peak (position control)"
    )
    try:
        ok = arm.move_joints(knots, cart_lengths=lengths, speed_scale=c["insert_speed"] / rammp_arm.CART_SPEED)
    except EStopped as e:
        raise Refused(f"STOPPED, nothing moved: the arm is e-stopped ({e})") from e
    except ArmRefused as e:
        raise Refused(
            f"STOPPED, nothing moved: the driver did not take the approach ({e}). The driver rejects goals while "
            "the arm is FAULTED: clear the fault on the arm, then re-run the place"
        ) from e
    if not ok:
        raise Refused(f"STOPPED: the approach did not complete ({arm.last_error})")
    time.sleep(SETTLE_AFTER_S)
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
        if _near(q_in, res["line"][-1]):
            descent = res["descent"]
            log(f"line down: as checked (the arm within {REUSE_TOL_DEG:g} deg of its start)")
        else:
            descent, upper, _ = plan_down(model, q_in, floor_z, log)
            check_tilt(model, [("descent", upper, LINE_STEP_DEG)], R0, c["tilt_max_deg"])
    except Refused as e:
        raise Refused(f"STOPPED in the oven, holding the container, not lowering: {e}") from e
    zs = descent[0]
    log(
        f"line down: tool z {zs[0]:.3f} -> {zs[-1]:.3f}; the box meets the glass at tool z "
        f"{floor_z + BOX_BELOW:.3f} (the lower finger touches ~1.3 cm before it)"
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
    # still in impedance at the hold: open in place, tilt, then position control and the back-out
    release_and_back_out(arm, model, cand["pre"], h, cfg, ctx, log)
    log("Done: container placed, hand back at the pre-insert pose outside the opening (position control).")
    return {"executed": True, "recording": str(path)}


def release_and_back_out(arm, model, pre, h, cfg, ctx, log):
    """At the hold, still in the lowering's joint impedance (gripper closed): open to RELEASE_OPEN in place, pitch
    the hand RELEASE_TILT_DEG nose-down about the tool frame (impedance), stop, back to position control, then the
    level back-out with the hand tilted (_level_back_out). Not collision-checked."""
    c = cfg
    pre, h = np.asarray(pre, dtype=float), np.asarray(h, dtype=float)
    ok = False
    try:
        # 1. open in place
        ctx.progress("open", 0.0)
        open_to(arm, Recorder(), log, RELEASE_OPEN)
        # 2. the tilt, in impedance
        ctx.progress("tilt", 0.0)
        q = np.asarray(arm.raw_joints(), dtype=float)
        tilt = plan_tilt(model, q)
        log(f"-> release tilt: the hand {RELEASE_TILT_DEG:g} deg nose-down about the tool frame (impedance)")
        ok = arm.move_joints(tilt[1:], speed_scale=TILT_SPEED_SCALE, impedance=compliant(c["gains"]))
        wait_still(arm, STILL_AFTER_S)
    finally:
        exit_impedance(arm, log)  # position control whatever happened
    if not ok:
        raise Refused(f"STOPPED, gripper open, in the oven (position control): the tilt failed ({arm.last_error})")
    # 3. the level back-out, tilted (position control)
    q = np.asarray(arm.raw_joints(), dtype=float)
    _level_back_out(arm, model, q, float(model.fk(q)[2, 3]), pre, c, ctx, log)


def _level_back_out(arm, model, q, z, pre, c, ctx, log):
    """The level line out from joints q (the arm there, position control, gripper open) to above the pre-insert
    pose's x,y at height z (plan_level_back_out, not collision-checked); its end measured."""
    try:
        back = plan_level_back_out(model, q, pre)
    except Refused as e:
        raise Refused(f"STOPPED, gripper open, hand NOT backed out (position control): {e}") from e
    ctx.progress("back-out", 0.0)
    log("-> back out level to above the pre-insert pose (not collision-checked)")
    move_line(arm, model, back, "the back-out", c["insert_speed"], "gripper open")
    end = np.r_[np.asarray(pre, dtype=float)[:2], z]
    d = float(np.linalg.norm(model.fk(np.asarray(arm.raw_joints(), dtype=float))[:3, 3] - end))
    log(f"backed out: {d * 100:.1f} cm from above the pre-insert pose")
    if d > END_TOL:
        raise Refused(
            f"STOPPED, gripper open: the back-out ended {d * 100:.1f} cm from its planned end "
            f"(max {END_TOL * 100:.0f} cm)"
        )


def move_line(arm, model, qs, what, speed, state):
    """move_joints through qs[1:] (stiff), timed in metres at `speed` (m/s); Refused if it fails."""
    if not arm.move_joints(qs[1:], cart_lengths=path_lengths(model, qs), speed_scale=speed / rammp_arm.CART_SPEED):
        raise Refused(f"STOPPED, {state}: {what} did not complete ({arm.last_error})")
    time.sleep(SETTLE_AFTER_S)


def ready(arm):
    """Refused unless the arm may be commanded now."""
    why = arm.not_ready()
    if why is not None:
        raise Refused(why)


def path_lengths(model, qs):
    """The tool's travel (m) between consecutive joint waypoints."""
    return [
        float(np.linalg.norm(model.fk(b)[:3, 3] - model.fk(a)[:3, 3])) for a, b in zip(qs[:-1], qs[1:], strict=True)
    ]
