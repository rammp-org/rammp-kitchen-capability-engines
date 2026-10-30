"""The driver contract: what box-opening checks before the kinova-gen3-ros2
driver moves the arm, and how it reads the driver's results and gripper."""

import math

import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening.constants import JOINT_VMAX
from box_opening.runtime.driver import (
    GripperWait,
    dilate,
    refusal,
    result_message,
    rmw_refusal,
    setpoint_position,
)

NAMES = ["joint_%d" % i for i in range(1, 8)]


def _traj(rows, times, vel=None, acc=None):
    t = JointTrajectory()
    t.joint_names = list(NAMES)
    for k, (row, tt) in enumerate(zip(rows, times, strict=False)):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [float(v) for v in (vel[k] if vel else [0.0] * 7)]
        p.accelerations = [float(v) for v in (acc[k] if acc else [0.0] * 7)]
        p.time_from_start.sec = int(tt)
        p.time_from_start.nanosec = int(round((tt - int(tt)) * 1e9))
        t.points.append(p)
    return t


# cuRobo stamps point k at (k + 1) * dt and the re-timer starts at first_dt:
# a real trajectory's first point is never at t=0, so these fixtures aren't.


def _secs(p):
    return p.time_from_start.sec + p.time_from_start.nanosec * 1e-9


def test_dilate_stretches_time_and_scales_the_profile_consistently():
    # the driver has no speed_scale: a leg flown at half speed must reach it
    # as twice the duration with half the velocity and a quarter of the
    # acceleration, or its Hermite interpolation no longer matches the path
    t = _traj(
        [[0.0] * 7, [0.1] * 7, [0.2] * 7],
        [0.0, 0.5, 1.0],
        vel=[[0.0] * 7, [0.2] * 7, [0.0] * 7],
        acc=[[0.4] * 7, [0.0] * 7, [-0.4] * 7],
    )
    out = dilate(t, 0.5)
    assert [_secs(p) for p in out.points] == pytest.approx([0.0, 1.0, 2.0])
    assert list(out.points[1].velocities) == pytest.approx([0.1] * 7)
    assert list(out.points[0].accelerations) == pytest.approx([0.1] * 7)
    assert [list(p.positions) for p in out.points] == [[0.0] * 7, [0.1] * 7, [0.2] * 7]
    # the planned trajectory itself is untouched: a replan or retry reuses it
    assert _secs(t.points[-1]) == pytest.approx(1.0)
    assert list(t.points[1].velocities) == pytest.approx([0.2] * 7)


def test_start_gate_is_wrap_aware_and_tolerates_small_drift():
    # joint_3 sits on the +/-pi seam at HOME: -pi+0.01 is 0.01 rad from +pi
    live = [0.0, 0.0, math.pi, 0.0, 0.0, 0.0, 0.0]
    start = [0.03, 0.0, -math.pi + 0.01, 0.0, 0.0, 0.0, 0.0]
    end = [0.05, 0.0, -math.pi + 0.01, 0.0, 0.0, 0.0, 0.0]
    assert refusal(_traj([start, end], [0.02, 1.0]), live, JOINT_VMAX) is None


def test_a_trajectory_starting_away_from_the_arm_is_refused():
    # the driver commands the first waypoint at once, wherever the arm is
    t = _traj([[0.1] + [0.0] * 6, [0.2] + [0.0] * 6], [0.02, 1.0])
    assert refusal(t, [0.0] * 7, JOINT_VMAX) is not None


def test_a_trajectory_over_the_joint_velocity_limits_is_refused():
    t = _traj([[0.0] * 7, [0.1] * 7], [0.02, 1.0], vel=[[0.0] * 7, [2.0] + [0.0] * 6])
    assert refusal(t, [0.0] * 7, JOINT_VMAX) is not None


def test_a_trajectory_without_a_velocity_profile_is_refused():
    t = _traj([[0.0] * 7, [0.1] * 7], [0.02, 1.0])
    for p in t.points:
        p.velocities = []
    assert refusal(t, [0.0] * 7, JOINT_VMAX) is not None


def test_a_trajectory_with_no_timing_is_refused():
    # every point stamped 0: the driver would complete it "instantly",
    # commanding the final waypoint in one step
    t = _traj([[0.0] * 7, [0.01] * 7, [0.02] * 7], [0.0, 0.0, 0.0])
    assert refusal(t, [0.0] * 7, JOINT_VMAX) is not None


def test_driver_failures_read_as_the_code_and_the_drivers_reason():
    msg = result_message(-9, "halted: emergency stop")
    assert "HALTED" in msg and "emergency stop" in msg


def test_an_unauthorized_goal_says_why():
    # the manifest runs arbitration disabled; if that changes, a client that
    # sends no control token must not read as a mystery failure
    assert "token" in result_message(-8, "")


@pytest.mark.parametrize("knuckle, setpoint", [(0.8, 1.0), (0.0, 0.0), (0.4, 0.5), (1.2, 1.0), (-0.1, 0.0)])
def test_knuckle_commands_become_the_drivers_normalized_setpoint(knuckle, setpoint):
    assert setpoint_position(knuckle) == pytest.approx(setpoint)


def test_gripper_already_at_its_target_is_done_at_once():
    w = GripperWait(target=0.0, start=0.002, t0=0.0)
    assert w.update(0.002, 0.02, fresh=True) == (True, 0.002, False)


@pytest.mark.parametrize("closed", [0.7696, 0.7896, 0.790, 0.7928])
def test_a_gripper_already_closed_on_air_is_done_not_failed(closed):
    """Field 2026-09-16: `press:close -> failed (gripper at 0.790)` with the
    fingers already shut from an earlier run. A 2F-85 closed on air never
    reaches the 0.8 knuckle target — preflight read 0.987 and 0.991 of full
    travel that morning, 0.7896 and 0.7928 rad — so the at-target check
    missed, nothing needed to move, and "never moved" called a closed
    gripper a failure. Flatbot 2026-10-23: shut at 0.7696, the same failure
    at the 0.02 tolerance ("gripper at 0.770 (overlapped)")."""
    from box_opening.constants import GRIPPER_CMD_CLOSED

    w = GripperWait(target=GRIPPER_CMD_CLOSED, start=closed, t0=0.0)
    assert w.update(closed, 8.0, fresh=True) == (True, closed, False)


def test_opening_fully_is_done_on_arrival_and_not_a_stall():
    w = GripperWait(target=0.0, start=0.8, t0=0.0)
    assert w.update(0.4, 0.1, fresh=True) is None
    assert w.update(0.0, 0.2, fresh=True) == (True, 0.0, False)


def test_closing_on_the_knob_settles_short_of_the_target_as_stalled():
    w = GripperWait(target=0.8, start=0.0, t0=0.0)
    for pos, t in [(0.2, 0.10), (0.38, 0.20), (0.386, 0.25), (0.387, 0.30), (0.387, 0.35)]:
        assert w.update(pos, t, fresh=True) is None
    assert w.update(0.387, 0.45, fresh=True) == (True, 0.387, True)


def test_a_gripper_that_never_moves_fails():
    w = GripperWait(target=0.8, start=0.0, t0=0.0)
    for t in (0.2, 0.4, 0.6, 0.8):
        assert w.update(0.0, t, fresh=True) is None
    ok, _pos, _stalled = w.update(0.0, 1.0, fresh=True)
    assert ok is False


def test_a_joint_state_gap_never_reads_as_settled():
    w = GripperWait(target=0.8, start=0.0, t0=0.0)
    assert w.update(0.4, 0.10, fresh=True) is None
    assert w.update(0.4, 0.15, fresh=True) is None  # still: settling starts
    assert w.update(0.4, 0.35, fresh=False) is None  # stream gap: no verdict
    assert w.update(0.4, 0.40, fresh=True) is None  # settling restarts here
    assert w.update(0.4, 0.50, fresh=True) is None
    assert w.update(0.4, 0.56, fresh=True) == (True, 0.4, True)


def test_a_shell_not_speaking_cyclone_is_refused():
    # Fast DDS discovers the driver's Cyclone nodes and then loses their data
    assert rmw_refusal({}) is not None
    assert rmw_refusal({"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"}) is not None
    assert rmw_refusal({"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"}) is None
