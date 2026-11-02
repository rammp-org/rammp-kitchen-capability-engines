"""Placing the held container in the open microwave: plan and run. Everything is in arm_mount (the
driver's gravity-aligned frame: z true up, x forward, y left, origin on the arm base axis).

Built from place_container (WalterHsieh0713/rammp-kitchen-capability-engines, feature/7-place-container,
3b3ca64), which puts the container down on a table: the hold model, the arm model (FK/IK on the
driver's PyBullet copy), the checked joint moves, the straight line down in the driver's joint
IMPEDANCE mode with the contact stop, the hold, the release and the back-out are his, unchanged. What
is new is where the hand goes: the microwave's floor is too close under its ceiling for his 30 cm
drop, so the hand comes in low and reaches straight in along the oven's axis.

Where the container is, along the hand's pointing direction (tool z):
  flange --TOOL_Z (12 cm)--> tool frame --tip_ahead--> fingertips --BOWL_RADIUS--> container centre
The fingers pinch the lip, so the lip is at the tool frame's height and the bottom BOWL_DEPTH below.
The hand holds the container level with the camera on its right (tool x up, tool z = the heading),
or the same hold turned 180 deg about the hand's axis (camera left), pointed `pitch` deg down so the
far side touches first.

The place (plan_microwave plans and checks all of it before anything moves; run() executes it):
  1. adjust   square the hand in place to the nearest level hold            (move_joints)
  2. lift     straight up to the height it reaches in at, if lower          (move_joints)
  3. transit  to PRE-INSERT, in front of the oven: heading along its insertion axis at the height it
              reaches in at, the container's far edge `standoff` in front of the control panel's face
              (cuRobo's plan, checked, then run exactly as planned)         (execute_joint_trajectory)
  4. insert   straight in along the axis until the lip's centre is over the spot  (move_joints, stiff)
  5. lower    straight down in impedance until the floor stops it, then hold a little below the
              contact                                                       (move_joints, impedance)
  6. open     the gripper
  7. back out straight back along the axis the whole way to pre-insert, the first quarter lifting the
              hold's press                                                  (move_joints, impedance)
Every leg is planned again from the measured joints just before it runs, and checked again.
The arm is left in impedance, held where the back-out ended.

The height it reaches in at (insert_height): the container's lowest point `insert_clearance` above the
turntable glass at least, and every link and the container `clear_inside` below the oven's ceiling (the
opening's top is flush with it) at least, from the arm's own model; the room between goes half to each.

Obstacles (Oven, Obstacles, check_path): cuRobo's world is empty, so nothing it plans avoids the oven.
Every configuration of every leg is checked here instead, in the sim, before it may run: every moving link
of the arm and gripper, and the held container (a cylinder of the lip's radius, BOWL_DEPTH tall, hanging
from the lip at the tool frame as held), against
  - the located oven as five solid boxes around its free interior (Oven; the extents beyond the tape-
    measured cavity are OVEN_BEYOND), and
  - the scene camera's depth points as voxels (the open door, the table, whatever else is there),
    except points on the oven model, on the arm and on the container.
Clearances (m): every link clear_link and the container clear_container from everything; on the legs
into, in and out of the oven (INSIDE_LEGS), clear_inside from the oven model; on the way down the
container may reach the glass (that is the goal). A configuration that fails refuses the whole place with
the closest pair named.

Frames: the sim's ee link and go_to targets are the TOOL frame (arm_tool_frame, TOOL_Z past the flange
along its z); /ee_state (arm.ee_state()) is the FLANGE (arm_end_effector_link). Model.fk(q, "flange") is
what /ee_state is compared with, and only /ee_state's z and z velocity are used, always against the flange.
"""

import csv
import time

import numpy as np
import rammp_arm
from capability_engine import sim
from capability_engine.arm import IMPEDANCE, LATEST_WINS, POSITION, QUEUE, SUCCESSFUL
from rammp_arm import CONTINUOUS, JOINTS, TOOL_Z, _unwrap_from
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from . import interior

# --- the container and how it is held ---
BOWL_RADIUS = 0.055  # m, centre to the wall the fingertips are pushed in to
LIP_WIDTH = 0.03175  # m, 1.25 in: the flat lip past BOWL_RADIUS
BOWL_DEPTH = 0.08  # m, lip (= tool frame height) to the bottom
FLANGE_TO_TIP = 0.17955  # m, flange face to the closed fingertips (Gen3 + 2F-85; not ruler-measured)
PLACE_PITCH_DEG = 5.0  # the hand points this many deg down while placing
# --- the motion ---
PRESS_BELOW = 0.10  # m, the line down is aimed this far below where the bottom meets the table: in
# impedance the hand lags its command by a few cm, and the table, not the command, stops it
MIN_TOOL_HEIGHT = 0.02  # m, the tool frame is never targeted lower than this above the table
MIN_BELOW_MEET = 0.03  # m, the line down must reach at least this far below where the bottom meets the table
MAX_DROP = 0.40  # m, the longest straight line down
PATH_STEP = 0.002  # m, between the line's joint waypoints
RETREAT_STEP = 0.004  # m, between the retreat's joint waypoints
RETREAT_RELEASE = 0.25  # the first quarter of the retreat also lifts the setpoint by the hold's press
MAX_ADJUST_DEG = 15.0  # the squaring turn may only correct this much (a bigger one tips the container)
MAX_BOWL_TILT_DEG = 10.0  # container tilt allowed anywhere along a joint move
MAX_LOWER_TILT_DEG = 1.0  # ... and on the straight way down
# --- checks ---
IK_TOL = 0.005  # m
IK_ROT_TOL_DEG = 1.0
MAX_STEP_JUMP_DEG = 120.0  # one joint move
UP = np.array([0.0, 0.0, 1.0])

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
# the recording's columns: /ee_state as it comes, so x..z and vx..vz are the FLANGE's, not the tool frame's
LOG_COLUMNS = ["t", "phase", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "fx", "fy", "fz", "tx", "ty", "tz"]
TOOL_FRAME_TOL_DEG = 3.0  # /ee_state's orientation vs the model's flange


# -----------------------------------------------------------------------------------------------
# Planning (no rclpy)
# -----------------------------------------------------------------------------------------------


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


def hand_orientation(heading, camera_left=False, pitch_deg=PLACE_PITCH_DEG):
    """The hand pointing along `heading` (horizontal part), container level, camera on its right
    (camera_left: the same hold turned 180 deg about the hand's axis), pointed pitch_deg down.
    Level, camera right: tool z = heading, tool y = heading x up (right of the hand seen from
    behind), tool x = up."""
    z = np.array([heading[0], heading[1], 0.0], dtype=float)
    z /= np.linalg.norm(z)
    y = np.cross(z, UP)
    x = np.cross(y, z)
    if camera_left:
        x, y = -x, -y
    R = Rotation.from_matrix(np.column_stack([x, y, z]))
    # about z x up (horizontal, across the hand): a negative turn brings the tip down
    return (Rotation.from_rotvec(-np.radians(pitch_deg) * np.cross(z, UP)) * R).as_quat()


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


def reaches(descent, table_z):
    """The line down gets at least MIN_BELOW_MEET below where the bottom meets the table."""
    return descent[0][-1] <= table_z + BOWL_DEPTH - MIN_BELOW_MEET


def reach_refusal(descent, table_z, spot):
    meet = table_z + BOWL_DEPTH
    return (
        f"lower: the arm can only go down to tool z={descent[0][-1]:.3f} here, "
        f"{(descent[0][-1] - meet) * 100:.0f} cm above where the bottom meets the table -- "
        f"the spot is {np.hypot(*spot[:2]):.2f} m from the base axis"
    )


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


# -----------------------------------------------------------------------------------------------
# The arm side: impedance lowering, hold, release
# -----------------------------------------------------------------------------------------------


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


def let_go(arm, rec, log):
    """Open the gripper; Refused (hand NOT backed out) if it stays closed on the container."""
    rec.phase = "open"
    pos = 1.0
    for attempt in (1, 2):  # the open command once more if the fingers haven't moved
        arm.gripper(0.0)
        t0 = time.monotonic()
        while time.monotonic() - t0 < OPEN_WAIT_S:  # it may start late (flatbot, 2026-10-29: 0.95 right after)
            pos = float(arm.gripper_state().position)
            if pos <= OPEN_STALLED:
                break
            time.sleep(0.05)
        if pos <= OPEN_STALLED:
            break
        log(
            f"gripper still at {pos:.2f} after {OPEN_WAIT_S:.0f} s" + (" -- sending open again" if attempt == 1 else "")
        )
    log(f"gripper open: {pos:.2f} (0 = fully open)")
    if pos > OPEN_STALLED:
        raise Refused(f"STOPPED, hand NOT backed out: the gripper only opened to {pos:.2f}. Open it, then `retreat`")


# ---------------------------------------------------------------------------------------------
# The microwave (new)
# ---------------------------------------------------------------------------------------------

INSERT_STEP = 0.004  # m, between the joint waypoints of the lines in and out
LIP_RADIUS = BOWL_RADIUS + LIP_WIDTH  # the container is checked as a cylinder this wide, BOWL_DEPTH tall
# The cavity in the oven frame, from interior.py's tape-measured constants: S from the control panel's face
# (the frame plane is interior.RECESS behind it), A from the opening's centre (the side walls are flush with
# the opening's edges), z from the turntable glass.
WALL_A = interior.OPENING[0] / 2  # the left wall at A = +WALL_A, the right wall at -WALL_A
BACK_S = interior.RECESS + interior.DEPTH  # the back wall
SPOT_S = interior.RECESS + interior.DEPTH / 2  # the spot (interior's point), half-way from the frame plane in
# The oven's body beyond the tape-measured cavity (interior.py) is not measured. The solid around the cavity
# is taken to reach this far past the measured surfaces (m), each more than the real oven:
OVEN_BEYOND = dict(
    left=0.06,  # past the left wall (the door's hinge side)
    right=0.16,  # past the right wall: the control panel, ~14 cm wide as the scene camera sees it
    above=0.10,  # above the ceiling
    below=0.10,  # below the turntable glass (what the oven stands on comes from the depth)
    back=0.10,  # behind the back wall
)
CEILING_BOX = "oven ceiling / front above the opening"
FLOOR_BOX = "oven floor (glass) / front below the opening"
INSIDE_LEGS = ("insert", "lower", "back out")  # clear_inside to the oven model on these
VOXELS_PER_BODY = 127  # PyBullet keeps at most 127 links per body and 16 shapes per compound, dropping the rest
# without an error: so one cube per link, 127 links per body, and the count checked
QUERY_M = 0.25  # clearances to the oven are measured up to this far; further is reported as at least this ...
VOXEL_QUERY_M = 0.10  # ... to the voxels up to this far (more than any clearance needed)
CHECK_MAX_DEG = 1.0  # between two checked configurations no joint turns more than this
START_TOL_DEG = 1.0  # cuRobo's plan runs only if it starts this close to the measured joints
END_TOL_DEG = 2.0  # ... and ends with the tool turned at most this far from its target


class Oven:
    """The located microwave as five solid boxes around its free interior, in arm_mount, from
    interior.locate's point, insertion_axis and floor_z and interior.py's tape-measured constants.

    Oven frame (interior.py's axes): S into the oven from the control panel's face (S = 0), A to the
    oven's left from the opening's centre, z up. The free interior is A between the side walls, S from
    the panel's face to the back wall, z from the turntable glass to the ceiling; the rest of the body
    (out to OVEN_BEYOND) is solid. So the boxes are the side walls, the glass, the ceiling and the back
    wall, and together the front: a plate from the panel's face back, with the opening in it. Larger
    than the oven where it is not certain: the opening's bottom is taken at the glass (it is
    interior.FLOOR lower) and the frame around it at the panel's face (it is interior.RECESS further
    back). The open door is not in it (its angle varies): the scene voxels are."""

    def __init__(self, target):
        n = np.array([target["insertion_axis"][0], target["insertion_axis"][1], 0.0], dtype=float)
        self.n = n / np.linalg.norm(n)
        self.w = np.cross(UP, self.n)  # the oven's left
        self.floor = float(target["floor_z"])
        self.ceiling = self.floor + interior.HEIGHT
        point = np.asarray(target["point"], dtype=float)
        self.spot = np.array([point[0], point[1], self.floor])
        self.origin = np.array([point[0], point[1], 0.0]) - SPOT_S * self.n  # the spot is centred between the walls
        right, left, back, b = -WALL_A, WALL_A, BACK_S, OVEN_BEYOND
        s = (0.0, back + b["back"])
        z = (self.floor - b["below"], self.ceiling + b["above"])
        self.boxes = [  # (name, (S), (A), (z))
            ("oven left wall", s, (left, left + b["left"]), z),
            ("oven right wall / control panel", s, (right - b["right"], right), z),
            (CEILING_BOX, s, (right, left), (self.ceiling, z[1])),
            (FLOOR_BOX, s, (right, left), (z[0], self.floor)),
            ("oven back wall", (back, s[1]), (right, left), (self.floor, self.ceiling)),
        ]

    def axes(self):
        """Columns: the oven's S, A and z axes in arm_mount (right-handed)."""
        return np.column_stack([self.n, self.w, UP])

    def world_boxes(self):
        """[(name, centre, rotation, half extents)] in arm_mount; the box axes are (S, A, z)."""
        R = self.axes()
        out = []
        for name, *ranges in self.boxes:
            lo, hi = np.array([r[0] for r in ranges]), np.array([r[1] for r in ranges])
            out.append((name, self.origin + R @ ((lo + hi) / 2), R, (hi - lo) / 2))
        return out

    def distance(self, pts):
        """Each point's distance (m) to the oven's solid; 0 inside it."""
        d = np.asarray(pts, dtype=float).reshape(-1, 3) - self.origin
        local = np.column_stack([d @ self.n, d @ self.w, d[:, 2]])
        best = np.full(len(local), np.inf)
        for _, *ranges in self.boxes:
            lo, hi = np.array([r[0] for r in ranges]), np.array([r[1] for r in ranges])
            best = np.minimum(best, np.linalg.norm(np.maximum(np.maximum(lo - local, local - hi), 0.0), axis=1))
        return best


def container_pose(T, camera_left, reach):
    """The held container as checked: a cylinder (centre, rotation whose z is its axis, out of its
    top) for tool pose T. Its lip's centre is `reach` along the tool's z (hold_reach), its axis is the
    tool's x (camera right) or -x (camera left), its bottom BOWL_DEPTH below the lip."""
    up = -T[:3, 0] if camera_left else T[:3, 0]
    lip = T[:3, 3] + reach * T[:3, 2]
    return lip - 0.5 * BOWL_DEPTH * up, np.column_stack([T[:3, 2], np.cross(up, T[:3, 2]), up])


def container_lowest(T, camera_left, reach):
    """z of the container's lowest point (the bottom's rim, as it is tipped) for tool pose T."""
    centre, R = container_pose(T, camera_left, reach)
    up = R[:, 2]
    return float(centre[2] - 0.5 * BOWL_DEPTH * up[2] - LIP_RADIUS * np.sqrt(max(0.0, 1.0 - up[2] ** 2)))


class Obstacles:
    """The oven (Oven), the scene voxels and the held container as bodies in the process's sim
    (capability_engine.sim), and the distances of the arm and the container to them. Only the links
    that move are measured (not those fixed to arm_mount). remove() takes every body out again."""

    def __init__(self, model, oven, camera_left):
        import pybullet as p

        self._p, self.model, self.oven, self.camera_left = p, model, oven, bool(camera_left)
        self.robot, self.client = model.rb.robot_id, model.rb.physics_client_id
        c = self.client
        info = [
            p.getJointInfo(self.robot, j, physicsClientId=c)
            for j in range(p.getNumJoints(self.robot, physicsClientId=c))
        ]
        moving = {-1: False}
        for j, i in enumerate(info):  # a joint's index is its child link's, and parents come first
            moving[j] = i[2] != p.JOINT_FIXED or moving[i[16]]
        self.links = {j: info[j][12].decode() for j in range(len(info)) if moving[j]}
        self.all_links = list(range(-1, len(info)))
        self.bodies, self.container = [], None  # bodies: (id, name, kind "oven" | "voxels")
        self._boxes = {}  # voxel body -> (lo, hi): the box around its cubes
        self.voxels, self.voxel_m = np.zeros((0, 3)), 0.0
        try:
            for name, centre, R, half in oven.world_boxes():
                self.bodies.append(
                    (self._body(p.GEOM_BOX, centre, R, halfExtents=[float(h) for h in half]), name, "oven")
                )
            self.container = self._body(
                p.GEOM_CYLINDER, [0.0, 0.0, -10.0], np.eye(3), radius=LIP_RADIUS, height=BOWL_DEPTH
            )
        except BaseException:
            self.remove()
            raise

    def _body(self, geom, centre, R, **shape):
        p, c = self._p, self.client
        sid = p.createCollisionShape(geom, physicsClientId=c, **shape)
        quat = Rotation.from_matrix(R).as_quat()
        return p.createMultiBody(
            0, sid, basePosition=[float(v) for v in centre], baseOrientation=quat.tolist(), physicsClientId=c
        )

    def remove(self):
        p, c = self._p, self.client
        for bid in [b for b, _, _ in self.bodies] + ([] if self.container is None else [self.container]):
            p.removeBody(bid, physicsClientId=c)  # (PyBullet keeps their collision shapes: it can't remove them)
        self.bodies, self.container = [], None

    def _pose(self, q, gripper_open=False, container=True):
        """The sim's arm at joints q, the fingers open or closed (sim.clearance's angles), the container
        in the hand."""
        p, c = self._p, self.client
        T = self.model.fk(q)  # sets the arm's joints
        for jj in sim.FINGER_JOINTS:
            angle = sim.FINGERTIP_OPEN_RAD if gripper_open else sim.FINGER_CLOSE_SIGN[jj] * sim.FINGER_CLOSED_RAD
            p.resetJointState(self.robot, jj, angle, physicsClientId=c)
        if container:
            centre, R = container_pose(T, self.camera_left, self.model.reach)
            p.resetBasePositionAndOrientation(
                self.container, centre.tolist(), Rotation.from_matrix(R).as_quat().tolist(), physicsClientId=c
            )
        return T

    def add_scene(self, points, q, cfg):
        """The scene camera's points (arm_mount) as voxel obstacles (cfg['voxel_m'] cubes): those within
        cfg['region_m'] of arm_mount's origin, further than cfg['surface_band'] from the oven model (the
        oven's own surfaces, modelled already) and further than cfg['self_band'] from the arm and the
        held container at joints q (where they were when the points were taken). Returns the counts."""
        p, c, v = self._p, self.client, float(cfg["voxel_m"])
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        pts = pts[np.linalg.norm(pts, axis=1) < cfg["region_m"]]
        n_region = len(pts)
        pts = pts[self.oven.distance(pts) >= cfg["surface_band"]]
        centres = (np.unique(np.floor(pts / v).astype(np.int64), axis=0) + 0.5) * v if len(pts) else np.zeros((0, 3))
        own = self._near_robot(centres, q, cfg["self_band"])
        centres = centres[~own]
        box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[v / 2] * 3, physicsClientId=c)
        for i in range(0, len(centres), VOXELS_PER_BODY):  # one fixed link per cube
            chunk = centres[i : i + VOXELS_PER_BODY]
            n = len(chunk)
            bid = p.createMultiBody(
                0,
                -1,
                basePosition=[0.0, 0.0, 0.0],
                linkMasses=[0.0] * n,
                linkCollisionShapeIndices=[box] * n,
                linkVisualShapeIndices=[-1] * n,
                linkPositions=chunk.tolist(),
                linkOrientations=[[0.0, 0.0, 0.0, 1.0]] * n,
                linkInertialFramePositions=[[0.0, 0.0, 0.0]] * n,
                linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * n,
                linkParentIndices=[0] * n,
                linkJointTypes=[p.JOINT_FIXED] * n,
                linkJointAxis=[[0.0, 0.0, 1.0]] * n,
                physicsClientId=c,
            )
            self.bodies.append((bid, "scene voxel", "voxels"))
            self._boxes[bid] = (chunk.min(axis=0) - v / 2, chunk.max(axis=0) + v / 2)
            if p.getNumJoints(bid, physicsClientId=c) != n:
                raise Refused(f"the sim made {p.getNumJoints(bid, physicsClientId=c)} of {n} voxels in one body")
        self.voxels, self.voxel_m = centres, v
        return dict(points=n_region, off_oven=len(pts), voxels=len(centres), on_robot=int(own.sum()))

    def _near_robot(self, centres, q, band):
        """Which centres are within `band` of the arm (gripper closed) or the container at joints q."""
        p, c = self._p, self.client
        near = np.zeros(len(centres), bool)
        if not len(centres):
            return near
        self._pose(q)
        cand = np.zeros(len(centres), bool)
        for bid, link in [(self.robot, j) for j in self.all_links] + [(self.container, -1)]:
            lo, hi = p.getAABB(bid, link, physicsClientId=c)
            cand |= np.all((centres > np.asarray(lo) - band) & (centres < np.asarray(hi) + band), axis=1)
        sid = p.createCollisionShape(p.GEOM_SPHERE, radius=1e-4, physicsClientId=c)
        probe = p.createMultiBody(0, sid, basePosition=[0, 0, -10], physicsClientId=c)
        try:
            for i in np.flatnonzero(cand):
                p.resetBasePositionAndOrientation(probe, centres[i].tolist(), [0, 0, 0, 1], physicsClientId=c)
                near[i] = bool(
                    p.getClosestPoints(self.robot, probe, band, physicsClientId=c)
                    or p.getClosestPoints(self.container, probe, band, physicsClientId=c)
                )
        finally:
            p.removeBody(probe, physicsClientId=c)
        return near

    def distances(self, q, gripper_open=False, container=True, only=None):
        """Per obstacle body (or only the one named `only`): the closest moving link and, with
        `container`, the container, within QUERY_M of the oven's boxes and VOXEL_QUERY_M of the voxels:
        [(distance, who, body name, kind, on who, on body)]. A voxel body whose box is further than
        VOXEL_QUERY_M from every link's and the container's box is not asked (nothing in it is nearer)."""
        p, c = self._p, self.client
        self._pose(q, gripper_open, container)
        near = [p.getAABB(self.robot, j, physicsClientId=c) for j in self.links]
        if container:
            near.append(p.getAABB(self.container, -1, physicsClientId=c))
        near = [(np.asarray(lo) - VOXEL_QUERY_M, np.asarray(hi) + VOXEL_QUERY_M) for lo, hi in near]
        out = []
        for bid, name, kind in self.bodies:
            if only is not None and name != only:
                continue
            reach = QUERY_M
            if kind == "voxels":
                lo, hi = self._boxes[bid]
                if not any(np.all(lo <= b_hi) and np.all(b_lo <= hi) for b_lo, b_hi in near):
                    continue
                reach = VOXEL_QUERY_M
            best = None
            for pt in p.getClosestPoints(self.robot, bid, reach, physicsClientId=c):
                if pt[3] in self.links and (best is None or pt[8] < best[0]):
                    best = (float(pt[8]), self.links[pt[3]], name, kind, pt[5], pt[6])
            if best is not None:
                out.append(best)
            if container:
                pts = p.getClosestPoints(self.container, bid, reach, physicsClientId=c)
                if pts:
                    pt = min(pts, key=lambda x: x[8])
                    out.append((float(pt[8]), "container", name, kind, pt[5], pt[6]))
        return out

    def probes(self, q):
        """Points a move must not carry more than check_step_m between two checked configurations:
        every moving link's frame and the container's rim, top and bottom."""
        p, c = self._p, self.client
        T = self.model.fk(q)
        states = p.getLinkStates(self.robot, list(self.links), computeForwardKinematics=1, physicsClientId=c)
        centre, R = container_pose(T, self.camera_left, self.model.reach)
        rim = [
            centre + a * LIP_RADIUS * R[:, k] + h * BOWL_DEPTH * R[:, 2]
            for a in (-1, 1)
            for k in (0, 1)
            for h in (-0.5, 0.5)
        ]
        return np.vstack([np.array([s[4] for s in states]), np.array(rim)])


def required(leg, who, what, kind, cfg):
    """The clearance (m) `who` (a link's name, or "container") needs from body `what` on `leg`, or
    None when that pair is not checked: lowering the container onto the glass is the goal."""
    if leg == "lower" and who == "container" and what == FLOOR_BOX:
        return None
    if kind == "oven" and leg in INSIDE_LEGS:
        return cfg["clear_inside"]
    return cfg["clear_container"] if who == "container" else cfg["clear_link"]


def check_path(obs, leg, path, cfg, gripper_open=False, container=True, tilt=True, view=None):
    """Check every configuration the arm passes through on the joint path `path`: consecutive entries
    are joined straight in joint space, as move_joints and the driver run them, so give them continuous
    (_unwrap_from, or cuRobo's raw joints). It is sampled so that between two checked configurations no
    probe point (Obstacles.probes) moves more than cfg['check_step_m'] and no joint more than
    CHECK_MAX_DEG. In each, every moving link (fingers open or closed) and, with `container`, the held
    container must be required() from every obstacle; with `tilt` the container must stay level (as
    check_joint_move). The leg's report goes to view.legs[leg] (also when it fails) and is returned;
    Refused names the pair that fails."""
    model, name = obs.model, leg.rstrip("'")
    path = [np.asarray(q, dtype=float) for q in path]
    samples, prev = [path[0]], obs.probes(path[0])
    for a, b in zip(path[:-1], path[1:], strict=True):
        nxt = obs.probes(b)
        n = max(
            1,
            int(np.ceil(np.max(np.linalg.norm(nxt - prev, axis=1)) / cfg["check_step_m"])),
            int(np.ceil(np.degrees(np.max(np.abs(b - a))) / CHECK_MAX_DEG)),
        )
        samples += [a + (b - a) * k / n for k in range(1, n + 1)]
        prev = nxt
    allowed = max(MAX_BOWL_TILT_DEG, model.tilt(path[0]) + 1.0, model.tilt(path[-1]) + 1.0)
    worst, closest, tool, problem = None, None, [], None
    for k, q in enumerate(samples):
        for d, who, what, kind, on_who, on_what in obs.distances(q, gripper_open, container):
            need = required(name, who, what, kind, cfg)
            if need is None:
                continue
            if closest is None or d < closest[0]:
                closest = (d, who, what, need)
            if worst is None or d - need < worst[0] - worst[1]:
                worst = (d, need, who, what, k, on_who, on_what)
        T = model.fk(q)
        tool.append(T[:3, 3].copy())
        if container and tilt and problem is None:
            t = bowl_tilt_deg(T[:3, :3], model.pitch)
            if t > allowed:
                problem = f"the container tilts {t:.0f} deg (max {allowed:.0f}) {100 * k / len(samples):.0f}% along"
    if problem is None and worst is not None and worst[0] < worst[1]:
        d, need, who, what, k, *_ = worst
        problem = (
            f"the {who} comes {d * 100:.1f} cm from the {what} (needs {need * 100:.1f}), "
            f"{100 * k / len(samples):.0f}% along, tool at {np.round(tool[k], 3).tolist()}"
        )
    report = dict(
        leg=leg,
        ok=problem is None,
        problem=problem,
        samples=len(samples),
        tool=np.array(tool),
        # the tightest pair against what it needs (None: nothing within QUERY_M)
        min_m=None if worst is None else worst[0],
        need_m=None if worst is None else worst[1],
        pair=None if worst is None else f"{worst[2]} vs {worst[3]}",
        between=None if worst is None else (np.asarray(worst[5]), np.asarray(worst[6])),
        closest_m=None if closest is None else closest[0],
        closest_pair=None if closest is None else f"{closest[1]} vs {closest[2]}",
    )
    if view is not None:
        view.legs.pop(leg, None)  # re-checked legs go to the end, in the order they run
        view.legs[leg] = report
    if problem is not None:
        raise Refused(f"{leg}: {problem}")
    return report


def check_curobo(obs, leg, plan, q_start, pos, quat, cfg, view=None):
    """cuRobo's plan (a capability_engine Plan) for `leg`, checked so that it may run exactly as
    planned: it succeeded, starts within START_TOL_DEG of q_start, ends with the tool within
    GO_TO_TRACK_M and END_TOL_DEG of (pos, quat), and every configuration along it passes check_path."""
    if not plan.success or len(plan.q) < 2:
        raise Refused(f"{leg}: cuRobo has no plan ({plan.message})")
    off = float(np.max(np.abs(wrap_deg(plan.q[0] - np.asarray(q_start, dtype=float)))))
    if off > START_TOL_DEG:
        raise Refused(f"{leg}: cuRobo's plan starts {off:.1f} deg from the joints it was asked to start at")
    T = obs.model.fk(plan.q[-1])
    miss = float(np.linalg.norm(T[:3, 3] - np.asarray(pos)))
    turn = float(np.degrees((Rotation.from_matrix(T[:3, :3]) * Rotation.from_quat(quat).inv()).magnitude()))
    if miss > rammp_arm.GO_TO_TRACK_M or turn > END_TOL_DEG:
        raise Refused(f"{leg}: cuRobo's plan ends with the tool {miss * 100:.1f} cm, {turn:.1f} deg from its target")
    check_path(obs, leg, list(plan.q), cfg, view=view)
    return [np.asarray(q, dtype=float) for q in plan.q]


def run_plan(arm, plan, speed_scale, leg):
    """Run cuRobo's checked plan exactly as planned, slowed to speed_scale: its own samples, times and
    velocities (execute_joint_trajectory, stiff). Refused unless the arm is where it starts."""
    off = float(np.max(np.abs(wrap_deg(plan.q[0] - arm.raw_joints()))))
    if off > START_TOL_DEG:
        raise Refused(f"STOPPED before {leg}: the arm is {off:.1f} deg from where its checked plan starts")
    pts = [
        (float(t) / speed_scale, q.tolist(), (qd * speed_scale).tolist())
        for t, q, qd in zip(plan.t, plan.q, plan.qd, strict=True)
    ]
    res = arm.execute_joint_trajectory(JOINTS, pts, control_mode=POSITION, preemption=QUEUE)
    if res.error_code != SUCCESSFUL:
        raise Refused(f"STOPPED: {leg} did not complete ({res.error_string or res.error_code})")


def plan_line(model, q_start, delta, name, lift=0.0, step=INSERT_STEP):
    """Joint waypoints every `step` along the straight line `delta` (m) from q_start's tool pose,
    orientation kept; `lift` (m, up) is added over the first RETREAT_RELEASE of it. The first entry
    is q_start; the rest continue from it the short way round (as move_joints runs them). Refused where
    the arm can't follow it."""
    T = model.fk(q_start)
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    length = float(np.linalg.norm(delta))
    n = max(1, int(np.ceil(length / step)))
    qs, seed = [np.asarray(q_start, dtype=float)], np.asarray(q_start, dtype=float)
    for k in range(1, n + 1):
        f = k / n
        pos = T[:3, 3] + f * np.asarray(delta, dtype=float) + min(1.0, f / RETREAT_RELEASE) * lift * UP
        q = model.ik(pos, quat, seed, restarts=0, stay=0.01)
        if q is None or np.max(np.abs(wrap_deg(q - seed))) > 3.0 * step / PATH_STEP:
            raise Refused(f"{name}: the arm can't go straight {f * length * 100:.0f} cm of {length * 100:.0f}")
        qs.append(q)
        seed = q
    return [qs[0]] + _unwrap_from(qs[0], qs[1:])


def line_lengths(model, qs):
    return [
        float(np.linalg.norm(model.fk(b)[:3, 3] - model.fk(a)[:3, 3])) for a, b in zip(qs[:-1], qs[1:], strict=True)
    ]


def reach_in(standoff):
    """How far the hand goes straight in (m): from the container's far edge `standoff` in front of the
    oven's front (the control panel's face) to its lip's centre over the spot."""
    return SPOT_S + LIP_RADIUS + standoff


def spot_tool(oven, quat, reach, z):
    """The tool frame's position, held at quat at height z, with the lip's centre over the spot."""
    tz = Rotation.from_quat(quat).as_matrix()[:, 2]
    return np.array([oven.spot[0] - reach * tz[0], oven.spot[1] - reach * tz[1], z])


def insert_height(obs, quat, seed, cfg):
    """The tool frame's height (arm_mount z) for the way in, at orientation quat: the container's
    lowest point at least cfg['insert_clearance'] above the glass, and every link and the container at
    least cfg['clear_inside'] from the oven's ceiling / the front above the opening (CEILING_BOX; the
    opening's top is flush with the ceiling) at the end of the way in, measured on the arm's model in
    the sim. The room between the two goes half to each. Returns (z, info); Refused when no height
    satisfies both."""
    model, oven = obs.model, obs.oven
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    z_min = oven.floor + cfg["insert_clearance"] - container_lowest(T, obs.camera_left, model.reach)
    pos = spot_tool(oven, quat, model.reach, z_min)
    q = model.ik(pos, quat, seed)
    if q is None:
        raise Refused(f"insert: no IK solution for the tool inside the oven at {np.round(pos, 3).tolist()}")
    top = obs.distances(q, only=CEILING_BOX)
    d, who = min(((t[0], t[1]) for t in top), default=(QUERY_M, "nothing"))  # none within QUERY_M: at least that
    room = d - cfg["clear_inside"]
    if room < 0:
        raise Refused(
            f"no height to reach in at: with the container's lowest point {cfg['insert_clearance'] * 100:.1f} cm above "
            f"the glass (tool z {z_min:.3f}) the {who} is {d * 100:.1f} cm from the oven's ceiling / the opening's top "
            f"(needs {cfg['clear_inside'] * 100:.1f}); glass to ceiling is {interior.HEIGHT * 100:.2f} cm"
        )
    return z_min + room / 2, dict(z_min=z_min, z_max=z_min + room, top=who, top_m=d)


class View:
    """The plan as it is shown on the marker topic: the oven's boxes, the voxels, the spot, each
    checked leg (check_path's report: its tool path and tightest pair) and the container carried in,
    put down and let go of."""

    def __init__(self, oven, voxels=(), voxel_m=0.0):
        self.oven, self.voxels, self.voxel_m = oven, np.asarray(voxels, dtype=float).reshape(-1, 3), voxel_m
        self.legs, self.containers = {}, {}


def plan_microwave(model, obs, q_now, cfg, curobo, view=None, log=None):
    """The whole place from the measured joints q_now (raw) into obs.oven, every leg checked
    (check_path) before anything moves. curobo(pos, quat, start_joints): cuRobo's plan for the tool,
    level hold (self.arm.plan_to_pose). Returns dict(adjust=[(name, q)], lift, transit, insert, lower,
    back joint paths, info); Refused with the reason."""
    oven = obs.oven
    q_now = np.asarray(q_now, dtype=float)
    T_now = model.fk(q_now)
    h = T_now[:3, 2].copy()
    h[2] = 0.0
    if np.linalg.norm(h) < 0.3:
        raise Refused("the hand points too steeply up or down to tell which way it faces")
    left = camera_on_left(T_now[:3, :3])
    obs.camera_left = left
    adjust_quat = hand_orientation(h, left, model.pitch)
    adjust_deg = float(
        np.degrees((Rotation.from_matrix(T_now[:3, :3]) * Rotation.from_quat(adjust_quat).inv()).magnitude())
    )
    if adjust_deg > cfg["max_adjust_deg"]:
        raise Refused(
            f"the hand is {adjust_deg:.0f} deg from holding the container {model.pitch:.0f} deg down "
            f"(max {cfg['max_adjust_deg']:.0f} to correct with it in hand) -- re-grip it closer"
        )
    floor, axis = oven.floor, oven.n
    quat = hand_orientation(axis, left, model.pitch)

    adjust, q = plan_segment(model, q_now, T_now[:3, 3], adjust_quat, floor, "adjust"), q_now
    for name, qa in adjust:
        path = [q] + _unwrap_from(q, [qa])
        check_path(obs, name, path, cfg, view=view)
        q = path[-1]
    z_in, height = insert_height(obs, quat, q, cfg)
    end = spot_tool(oven, quat, model.reach, z_in)
    length = reach_in(cfg["standoff"])
    pre = end - length * axis
    lift = None
    if model.fk(q)[2, 3] < z_in - 0.01:
        lift = plan_line(model, q, [0.0, 0.0, z_in - model.fk(q)[2, 3]], "lift")
        check_path(obs, "lift", lift, cfg, view=view)
        q = lift[-1]
    transit = check_curobo(obs, "transit", curobo(pre, quat, q), q, pre, quat, cfg, view)
    insert = plan_line(model, transit[-1], length * axis, "insert")
    check_path(obs, "insert", insert, cfg, view=view)
    line_floor = floor + max(BOWL_DEPTH - PRESS_BELOW, MIN_TOOL_HEIGHT)
    descent = plan_descent(model, insert[-1], line_floor, log=log)
    if not reaches(descent, floor):
        raise Refused(reach_refusal(descent, floor, oven.spot) + " at the end of the line in")
    lower_path = resting(obs, descent, cfg)
    check_path(obs, "lower", lower_path, cfg, view=view)
    back = plan_line(model, lower_path[-1], -length * axis, "back out", lift=cfg["hold_press"])
    check_path(obs, "back out", back, cfg, gripper_open=True, container=False, view=view)
    if view is not None:
        for label, qc in (("pre-insert", transit[-1]), ("inserted", insert[-1]), ("resting", lower_path[-1])):
            view.containers[label] = container_pose(model.fk(qc), left, model.reach)
    info = dict(
        adjust_deg=adjust_deg,
        pre=pre.tolist(),
        end=end.tolist(),
        quat=quat.tolist(),
        z_in=z_in,
        height=height,
        reach_in=length,
        axis=axis.tolist(),
        floor_z=floor,
        line_floor_z=line_floor,
        camera_left=left,
    )
    return dict(adjust=adjust, lift=lift, transit=transit, insert=insert, lower=lower_path, back=back, info=info)


def resting(obs, descent, cfg):
    """The part of the line down (plan_descent's (zs, qs)) the arm can reach holding the container: down
    to where its lowest point is cfg['hold_press'] into the glass. Below that the glass holds it up.
    Continuous from the line's start."""
    zs, qs = descent
    T = obs.model.fk(qs[0])
    rest_z = obs.oven.floor - cfg["hold_press"] - (container_lowest(T, obs.camera_left, obs.model.reach) - T[2, 3])
    below = np.flatnonzero(np.asarray(zs) <= rest_z)
    k = int(below[0]) if len(below) else len(zs) - 1
    return [np.asarray(qs[0], dtype=float)] + _unwrap_from(qs[0], qs[1 : k + 1])


def back_out(arm, model, obs, q_start, axis, length, cfg, rec, log, view=None, show=lambda: None):
    """Pull the hand straight back `length` along -axis from q_start, in impedance, at most
    `retreat_speed`, the first quarter lifting `hold_press` (as place_container's back_out). The
    line, from where the arm is, is checked with the fingers open first."""
    line = plan_line(model, q_start, -length * np.asarray(axis, dtype=float), "back out", lift=cfg["hold_press"])
    now = arm.raw_joints()
    check_path(obs, "back out", [now] + _unwrap_from(now, line), cfg, gripper_open=True, container=False, view=view)
    show()
    lengths = line_lengths(model, line)
    rec.phase = "retreat"
    log(f"back out: {sum(lengths) * 100:.0f} cm straight back (impedance, up to {cfg['retreat_speed'] * 100:.0f} cm/s)")

    def record_only():
        rec.poll(arm)
        return False  # never stops the back-out

    ok = arm.move_joints(
        line[1:],
        cart_lengths=lengths,
        speed_scale=cfg["retreat_speed"] / rammp_arm.CART_SPEED,
        impedance=compliant(cfg["gains"]),
        monitor=record_only,
    )
    if not ok:
        raise Refused(f"STOPPED: the back-out did not complete ({arm.last_error})")
    wait_still(arm, 3.0, rec=rec)
    rec.phase = "done"


def run(arm, model, obs, plan, cfg, ctx, log, joints, rec, curobo, view=None, show=lambda: None):
    """Execute plan_microwave's plan. Each leg is planned again from the measured joints just before it
    runs and checked again (check_path / check_curobo): the arm is never exactly where a plan put it.
    joints(): the measured joints. rec: a Recorder, filled from the lowering on (the caller writes it,
    also when this raises). show(): publishes `view` again after each check."""
    info = plan["info"]
    axis, quat = np.asarray(info["axis"]), np.asarray(info["quat"])
    for name, q in plan["adjust"]:
        ctx.progress(name, 0.0)
        now = arm.raw_joints()
        check_path(obs, name, [now] + _unwrap_from(now, [q]), cfg, view=view)
        show()
        # a small turn in place: a checked joint move (place_container: cuRobo swung the hand sideways)
        if not arm.move_joints([q], speed_scale=cfg["adjust_speed_scale"]):
            raise Refused(f"STOPPED: {name} did not complete ({arm.last_error})")
    now = arm.raw_joints()
    z_now = model.fk(now)[2, 3]
    if z_now < info["z_in"] - 0.01:
        ctx.progress("lift", 0.0)
        line = plan_line(model, now, [0.0, 0.0, info["z_in"] - z_now], "lift")
        check_path(obs, "lift", line, cfg, view=view)
        show()
        log(f"-> lift ({(info['z_in'] - z_now) * 100:.0f} cm straight up)")
        if not arm.move_joints(line[1:], cart_lengths=line_lengths(model, line), speed_scale=cfg["go_to_speed_scale"]):
            raise Refused(f"STOPPED: lift did not complete ({arm.last_error})")
    ctx.progress("transit", 0.0)
    now = arm.raw_joints()
    tp = curobo(info["pre"], quat, now)
    check_curobo(obs, "transit", tp, now, info["pre"], quat, cfg, view)
    show()
    log(f"-> transit (cuRobo's checked plan, {len(tp.q)} samples, {tp.t[-1] / cfg['go_to_speed_scale']:.1f} s)")
    run_plan(arm, tp, cfg["go_to_speed_scale"], "transit")
    q = joints()
    miss = float(np.linalg.norm(model.fk(q)[:3, 3] - np.asarray(info["pre"])))
    if miss > rammp_arm.GO_TO_TRACK_M:
        raise Refused(f"STOPPED in front of the oven: the tool is {miss * 100:.1f} cm from pre-insert")
    tilt = model.tilt(q)
    if tilt > 1.0:
        raise Refused(f"STOPPED in front of the oven: the container is {tilt:.1f} deg off its hold")
    # straight in, planned again from where the arm IS
    ctx.progress("insert", 0.0)
    now = arm.raw_joints()
    line = plan_line(model, now, info["reach_in"] * axis, "insert")
    check_path(obs, "insert", line, cfg, view=view)
    show()
    log(f"-> insert ({info['reach_in'] * 100:.0f} cm straight in at {cfg['insert_speed'] * 100:.0f} cm/s, stiff)")
    if not arm.move_joints(
        line[1:], cart_lengths=line_lengths(model, line), speed_scale=cfg["insert_speed"] / rammp_arm.CART_SPEED
    ):
        raise Refused(f"STOPPED: insert did not complete ({arm.last_error})")
    q = joints()
    descent = plan_descent(model, q, info["line_floor_z"], log=log)
    if not reaches(descent, info["floor_z"]):
        raise Refused(
            f"STOPPED inside the oven: {reach_refusal(descent, info['floor_z'], obs.oven.spot)} -- nothing let go"
        )
    check_path(obs, "lower", resting(obs, descent, cfg), cfg, view=view)
    show()
    lcfg = {**cfg, "table_z": info["floor_z"], "bowl_depth": BOWL_DEPTH}
    ctx.progress("lower", 0.0)
    hold_q = lower(arm, model, descent, lcfg, rec, log)
    ctx.progress("open", 0.0)
    let_go(arm, rec, log)
    time.sleep(0.5)
    ctx.progress("back out", 0.0)
    back_out(arm, model, obs, hold_q, axis, info["reach_in"], lcfg, rec, log, view, show)
    log("Done: container in the microwave, hand out. The arm is still in impedance mode.")
