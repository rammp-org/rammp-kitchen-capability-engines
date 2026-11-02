"""PressArm on a stand-in for the engine's self.arm (no ROS, nothing moves): a motion the driver
fails raises MotionFailed, a cancel during the press leaves the stream (closed) at once, and a dry
run sends nothing."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

pa = pytest.importorskip("button_press.press_arm")  # needs rclpy + rammp interfaces
from capability_engine import Cancelled  # noqa: E402

from button_press import Abort, MotionFailed  # noqa: E402

N_OUT = np.array([-1.0, 0.0, 0.0])  # out of the panel, toward the arm
SPOT = np.array([0.675, -0.317, 0.077])  # the pre-press spot (the tool frame)


class FakeArm:
    """self.arm's reads and motions. The tool sits at `tool`; motions record what was sent."""

    def __init__(self, tool=SPOT, effort=None):
        self.tool = np.asarray(tool, float)
        self.effort = np.zeros(7) if effort is None else np.asarray(effort, float)
        self.sent, self.setpoints, self.closed = [], [], False
        self.last_error = ""
        self.go_to_ok, self.trajectory_code = True, 0
        self.cancel_after = None  # send() raises Cancelled after this many setpoints

    def joint_states(self):
        return SimpleNamespace(position=np.zeros(7), effort=self.effort)

    def ee_state(self):  # the flange: TOOL_Z behind the tool along z (identity orientation)
        return SimpleNamespace(position=self.tool - [0.0, 0.0, pa.TOOL_Z], orientation=np.array([0.0, 0.0, 0.0, 1.0]))

    def gripper_state(self):
        return SimpleNamespace(position=1.0, present=True)

    def go_to_ee_pose(self, pos, quat, label="", speed_scale=None, *, feedback=None):
        self.sent.append(("go_to_ee_pose", label))
        if not self.go_to_ok:
            self.last_error = f"  RammpArm: {label} ended HALTED "
            return False
        self.tool = np.asarray(pos, float)
        return True

    def execute_joint_trajectory(self, names, points, **kw):
        self.sent.append(("trajectory", len(points)))
        return SimpleNamespace(error_code=self.trajectory_code, error_string="")

    @contextmanager
    def stream(self, controller, gains=None, timeout_s=None):
        self.sent.append(("stream", controller))
        try:
            yield self
        finally:
            self.closed = True

    def send(self, setpoint):  # the Stream's send
        if self.cancel_after is not None and len(self.setpoints) >= self.cancel_after:
            raise Cancelled()
        self.setpoints.append(np.asarray(setpoint, float))


def press_arm(arm, execute=True):
    return pa.PressArm(arm, SimpleNamespace(base_pos=np.zeros(3)), execute, checkpoint=lambda: None)


def test_a_curobo_move_the_driver_failed_is_a_motion_failure():
    arm = FakeArm(tool=[0.577, 0.001, 0.433])
    arm.go_to_ok = False
    with pytest.raises(MotionFailed, match="HALTED"):
        press_arm(arm).go_to_ee_pose(np.array([0.46, -0.35, 0.20]), np.eye(3), "to staging", lambda rec: None)


def test_a_curobo_move_that_reaches_its_goal_returns_where_it_ended():
    arm = FakeArm(tool=[0.577, 0.001, 0.433])
    goal = np.array([0.46, -0.35, 0.20])
    fin = press_arm(arm).go_to_ee_pose(goal, np.eye(3), "to staging", lambda rec: None)
    assert np.allclose(fin, goal) and arm.sent == [("go_to_ee_pose", "to staging")]


def test_a_trajectory_the_driver_aborted_is_a_motion_failure():
    arm = FakeArm()
    arm.trajectory_code = -9  # HALTED: an e-stop, or control taken
    pose = [(SPOT, np.eye(3))]
    with pytest.raises(MotionFailed, match="HALTED"):
        press_arm(arm).smooth_poses(pose, [np.zeros(7)], (np.zeros(7), SPOT, np.eye(3)), "to pre-press", lambda r: None)


def test_a_leg_planned_from_elsewhere_is_refused_before_anything_is_sent():
    arm = FakeArm()
    pose = [(SPOT, np.eye(3))]
    with pytest.raises(Abort, match="nothing sent") as e:
        press_arm(arm).smooth_poses(pose, [np.zeros(7)], (np.full(7, 0.1), SPOT, np.eye(3)), "x", lambda r: None)
    assert not isinstance(e.value, MotionFailed) and arm.sent == []


def test_a_cancel_during_the_press_leaves_the_stream_at_once():
    """REVIEW_RULES C2: the push stops when the action is cancelled, not when the press ends."""
    arm = FakeArm()
    arm.cancel_after = 30  # in the baseline's hold, or just after
    with pytest.raises(Cancelled):
        press_arm(arm).press_contact(SPOT, N_OUT, "press 1", lambda rec: None)
    assert arm.closed and len(arm.setpoints) == 30


def test_the_press_pushes_along_the_normal_with_no_turn():
    arm = FakeArm()
    arm.cancel_after = 40  # well into the push (the baseline is ~20 setpoints)
    with pytest.raises(Cancelled):
        press_arm(arm).press_contact(SPOT, N_OUT, "press 1", lambda rec: None)
    push = arm.setpoints[-1]
    assert np.allclose(push, [pa.CONTACT_SPEED_MPS, 0, 0, 0, 0, 0])  # into the panel (-n_out), linear only
    assert np.allclose(arm.setpoints[0], np.zeros(6))  # the baseline holds still


def test_a_press_with_no_surface_stops_on_its_own_checks():
    arm = FakeArm()  # the torques never move: no touch within CONTACT_MAX_TOUCH_M
    with pytest.raises(Abort, match="no touch") as e:
        press_arm(arm).press_contact(SPOT, N_OUT, "press 1", lambda rec: None)
    assert not isinstance(e.value, MotionFailed) and arm.closed


def test_no_efforts_refuses_the_press_before_the_stream_opens():
    arm = FakeArm(effort=np.zeros(0))
    with pytest.raises(Abort, match="no arm efforts"):
        press_arm(arm).press_contact(SPOT, N_OUT, "press 1", lambda rec: None)
    assert arm.sent == []


def test_a_dry_run_sends_nothing():
    arm = FakeArm(tool=[0.577, 0.001, 0.433])
    p = press_arm(arm, execute=False)
    p.go_to_ee_pose(np.array([0.46, -0.35, 0.20]), np.eye(3), "to staging", lambda rec: None)
    assert p.press_contact(SPOT, N_OUT, "press 1", lambda rec: None) is None
    assert arm.sent == []
