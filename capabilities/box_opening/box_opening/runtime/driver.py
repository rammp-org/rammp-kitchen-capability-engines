"""What box-opening holds true about the kinova-gen3-ros2 driver.

The planner-side executor this repo drove before (RAMMP-CuRobo
feat/perceived-world, executor.py) refused a goal unless the arm stood at
its first waypoint, its velocities sat under the URDF limits and its time
ran forward, and it dilated time by a speed_scale. The driver's
/execute_joint_trajectory does none of that: its executor samples the
trajectory from t=0 and commands the first waypoint at once, wherever the
arm is, and the ROS layer checks only that every point carries seven
positions (kinova-gen3-driver v1.0.0 trajectory_executor.cpp,
kinova-gen3-ros2 ros2_backend.cpp). So those gates run here, in front of
every goal, and the driver's result codes, gripper units and middleware
are translated in one place.
"""

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# REQUIRED_RMW re-exported: both containers of the arm module build on rammp-base, which selects Cyclone
from box_opening.constants import REQUIRED_RMW as REQUIRED_RMW
from box_opening.constants import START_GATE_RAD
from box_opening.geometry import ang_diff
from box_opening.runtime.retime import check_like_executor
from box_opening.runtime.stamps import secs, set_stamp

# ExecuteJointTrajectory.Result.error_code (rammp_arm_interfaces v1.0.0)
RESULT_NAMES = {
    0: "SUCCESSFUL",
    -1: "INVALID_GOAL",
    -4: "PATH_TOLERANCE_VIOLATED",
    -5: "GOAL_TOLERANCE_VIOLATED",
    -6: "PREEMPTED",
    -7: "PLANNING_FAILED",
    -8: "NOT_AUTHORIZED",
    -9: "HALTED",
    -10: "STREAM_REJECTED",
}
SUCCESSFUL = 0
NOT_AUTHORIZED = -8

# /joint_states reports the Robotiq knuckle as GripperSetpoint.position x 0.8,
# the knuckle's URDF upper limit (the driver's gripper tier). Every gripper
# number in this repo stays in knuckle radians; only the command on the wire
# is normalized.
KNUCKLE_CLOSED_RAD = 0.8


def dilate(traj, speed):
    """`traj` to be flown at `speed` (> 0): time stretched by 1/speed,
    velocities scaled by speed and accelerations by speed**2, so the
    driver's Hermite interpolation still passes along the same path. A new
    trajectory; the planned one is left as it was (replans reuse it)."""
    speed = float(speed)
    if speed <= 0.0:
        raise ValueError("speed must be > 0, got %r" % speed)
    out = JointTrajectory()
    out.header = traj.header
    out.joint_names = list(traj.joint_names)
    for p in traj.points:
        q = JointTrajectoryPoint()
        q.positions = list(p.positions)
        q.velocities = [v * speed for v in p.velocities]
        q.accelerations = [a * speed * speed for a in p.accelerations]
        set_stamp(q.time_from_start, secs(p.time_from_start) / speed)
        out.points.append(q)
    return out


def start_gap_rad(live, traj):
    """Largest wrap-aware distance between the live joints and the first waypoint."""
    return max(abs(ang_diff(a, b)) for a, b in zip(live, traj.points[0].positions, strict=False))


def refusal(traj, live, vmax):
    """Why `traj` must not be sent to the driver from the `live` joints, or None."""
    if not traj.points:
        return "empty trajectory"
    if any(len(p.velocities) != len(vmax) for p in traj.points):
        return "trajectory has no velocity profile — the driver would fly it linearly, unchecked"
    gap = start_gap_rad(live, traj)
    if gap > START_GATE_RAD:
        return (
            "trajectory starts %.3f rad from the live arm (gate %.2f) — the driver "
            "would jump to its first waypoint; re-plan from the current state" % (gap, START_GATE_RAD)
        )
    problems = check_like_executor(traj, vmax)
    return "; ".join(problems) if problems else None


def result_message(error_code, error_string=""):
    """A driver result, readable: the code's name and the driver's own reason."""
    code = int(error_code)
    text = RESULT_NAMES.get(code, "error %d" % code)
    if error_string:
        text += ": " + error_string
    if code == NOT_AUTHORIZED:
        text += (
            " — the driver is enforcing arbitration and box-opening sends no "
            "control token (run the arm node with arbitration_mode: disabled)"
        )
    return text


def setpoint_position(knuckle_rad):
    """A knuckle-radian gripper command as the driver's 0 (open) .. 1 (closed)."""
    return min(1.0, max(0.0, float(knuckle_rad) / KNUCKLE_CLOSED_RAD))


def rmw_refusal(env):
    """Why a process with environment `env` cannot talk to the driver, or None."""
    rmw = env.get("RMW_IMPLEMENTATION", "")
    if rmw == REQUIRED_RMW:
        return None
    return (
        "RMW_IMPLEMENTATION is %s — the driver and planner containers speak "
        "Cyclone DDS, and any other middleware discovers them and then loses "
        "their data. export RMW_IMPLEMENTATION=%s" % ("unset (Fast DDS)" if not rmw else repr(rmw), REQUIRED_RMW)
    )


class GripperWait:
    """When a gripper command has finished, read off the knuckle.

    The driver's gripper is a latest-wins setpoint topic with no result, so
    completion is observed in /joint_states: at the target; or moved and
    then still for SETTLE_S — closed on something, a stall the grip band
    then judges; or never moved within NO_MOTION_S, a failure. A sample
    taken while the joint-state stream is stale never counts as stillness.
    """

    # A 2F-85 closed on air stops short of the 0.8 knuckle target: 0.987 and
    # 0.991 of full travel on 2026-09-16, i.e. 0.7896 and 0.7928 rad. At 0.01
    # a gripper already shut read as not-at-target, nothing moved, and "never
    # moved" failed the close. On flatbot 2026-10-23 it shut at 0.7696 (0.962),
    # and 0.02 failed a close the same way ("gripper at 0.770 (overlapped)"),
    # stopping the run's home. 0.035 rad is 4.4 % of travel.
    AT_TARGET_TOL = 0.035
    MOVED_MIN = 0.01
    SETTLE_TOL = 0.003
    SETTLE_S = 0.15
    NO_MOTION_S = 1.0

    def __init__(self, target, start, t0):
        self.target = float(target)
        self.start = float(start)
        self.t0 = float(t0)
        self._last = float(start)
        self._still_since = None

    def update(self, pos, t, fresh):
        """(ok, position, stalled) once the command has finished, else None."""
        pos, t = float(pos), float(t)
        if not fresh:
            self._still_since = None
            return None
        if abs(pos - self.target) <= self.AT_TARGET_TOL:
            return True, pos, False
        moved = abs(pos - self.start) >= self.MOVED_MIN
        if abs(pos - self._last) > self.SETTLE_TOL:
            self._still_since = None
        elif moved and self._still_since is None:
            self._still_since = t
        self._last = pos
        if moved and self._still_since is not None and t - self._still_since >= self.SETTLE_S:
            return True, pos, True
        if not moved and t - self.t0 >= self.NO_MOTION_S:
            return False, pos, False
        return None
