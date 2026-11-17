"""PressArm on the real self.arm (capability_engine's ArmInterface) beside capability_engine's
FakeDriver: the press opens the driver's ee_twist stream and a cancel closes it mid-push, and a
goal the driver fails is a MotionFailed. Needs ROS 2 and the RAMMP interfaces (make test-arm's image)."""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy")
pa = pytest.importorskip("button_press.press_arm")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "capability_engine" / "test"))
import rclpy  # noqa: E402
from capability_engine import Cancelled  # noqa: E402
from capability_engine.arm import ArmInterface  # noqa: E402
from fake_driver import FakeCtx, FakeDriver  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from button_press import MotionFailed  # noqa: E402

N_OUT = np.array([-1.0, 0.0, 0.0])
TOOL = np.array([0.5, 0.0, 0.3])  # the fake's flange + TOOL_Z along its (identity) z


@pytest.fixture
def env():
    """(press_arm, ctx, fake): a PressArm on an ArmInterface in an action, beside a FakeDriver."""
    rclpy.init()
    fake = FakeDriver()
    node = Node("button_press_test_engine")
    arm = ArmInterface(node, ReentrantCallbackGroup())
    executor = MultiThreadedExecutor()
    executor.add_node(fake)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    fake.set_flange(TOOL - [0.0, 0.0, pa.TOOL_Z], (0.0, 0.0, 0.0, 1.0))
    ctx = FakeCtx()
    arm.begin_action(ctx, token=None)
    kin = SimpleNamespace(base_pos=np.zeros(3))
    yield pa.PressArm(arm, kin, True, ctx.checkpoint), ctx, fake
    arm.end_action()
    executor.shutdown()
    spinner.join()
    node.destroy_node()
    fake.destroy_node()
    rclpy.shutdown()


def wait_for(cond, timeout_s=5.0):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout_s
        time.sleep(0.01)


def test_a_cancel_mid_press_closes_the_stream_and_stops_the_push(env):
    press_arm, ctx, fake = env
    wait_for(lambda: np.allclose(press_arm.ee_pos(), TOOL, atol=1e-6))
    threading.Timer(0.6, ctx.cancel).start()  # the baseline is 0.2 s: this is mid-push
    with pytest.raises(Cancelled):
        press_arm.press_contact(TOOL, N_OUT, "press 1", lambda rec: None)
    assert fake.requests["open_stream"][-1].controller == "ee_twist"
    wait_for(lambda: not fake.session_open())
    twists = [
        np.array([m.twist.linear.x, m.twist.linear.y, m.twist.linear.z]) for m in fake.setpoints["/setpoint/twist"]
    ]
    assert any(np.allclose(v, [pa.CONTACT_SPEED_MPS, 0, 0]) for v in twists)  # it pushed into the panel
    assert np.allclose(twists[-1], 0.0)  # and the stream ended on the hold (zero) setpoint


def test_a_go_to_the_driver_halted_is_a_motion_failure(env):
    press_arm, _, fake = env
    wait_for(lambda: np.allclose(press_arm.ee_pos(), TOOL, atol=1e-6))
    fake.set_result("go_to_ee_pose", -9)  # HALTED
    with pytest.raises(MotionFailed, match="HALTED"):
        press_arm.go_to_ee_pose(TOOL + [0.0, 0.1, 0.0], np.eye(3), "to staging", lambda rec: None)


def test_a_straight_leg_goes_out_as_one_joint_trajectory_on_the_drivers_joints(env):
    press_arm, _, fake = env
    q0 = np.zeros(7)
    fake.set_joints(q0)
    wait_for(lambda: np.allclose(press_arm.raw_joints(), q0))
    fake.set_result("execute_joint_trajectory", -4)  # PATH_TOLERANCE_VIOLATED
    poses = [(TOOL + [0.0, 0.0, 0.01 * i], np.eye(3)) for i in (1, 2)]
    knots = [np.full(7, 0.001), np.full(7, 0.002)]
    with pytest.raises(MotionFailed, match="PATH_TOLERANCE_VIOLATED"):
        press_arm.smooth_poses(poses, knots, (q0, TOOL, np.eye(3)), "to pre-press", lambda rec: None)
    (goal,) = fake.goals["execute_joint_trajectory"]
    assert list(goal.trajectory.joint_names) == [f"arm_joint_{i}" for i in range(1, 8)]
    assert np.allclose(goal.trajectory.points[-1].positions, knots[-1])
