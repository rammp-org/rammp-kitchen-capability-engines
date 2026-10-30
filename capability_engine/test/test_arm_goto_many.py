"""go_to_ee_poses / go_to_joint_configs: several targets in one motion, planned, blended and run
by the driver (/go_to_ee_poses, /go_to_joint_configs). The blending is the driver's and cuRobo's
(kinova-gen3-ros2, RAMMP-CuRobo /rammp_curobo/blend); here the arm's half: what each target sends,
the defaults, the end check, and that a failure is a refusal."""

import numpy as np
import pytest
from fake_driver import FakeCtx
from rammp_arm import SENDER_ID, TOOL_Z
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from capability_engine.arm import (
    HOLD_FIXED,
    HOLD_LEVEL,
    IMPEDANCE,
    ArmInterface,
    JointTarget,
    PoseTarget,
)
from capability_engine.driver_state import ArmError

TOKEN = b"\x03" * 16
UP = [0.0, 0.0, 0.0, 1.0]
VIEW = PoseTarget([0.4, 0.1, 0.3], UP, blend_radius=0.05)
GRASP = PoseTarget([0.5, 0.1, 0.2], UP, hold=HOLD_FIXED, speed_scale=0.5)
END = [0.3, 0.2, 0.1, -1.0, 0.0, 0.5, 0.0]


def test_go_to_ee_poses_sends_each_target_as_its_own_leg(arm_env):
    arm, _, fake = arm_env
    fake.finish_at("go_to_ee_poses", flange=([0.5, 0.1, 0.2 - TOOL_Z], UP))
    arm.begin_action(FakeCtx(), TOKEN)
    assert arm.go_to_ee_poses([VIEW, GRASP], "grab") is True

    (g,) = fake.goals["go_to_ee_poses"]
    view, grasp = g.targets
    assert view.pose.header.frame_id == grasp.pose.header.frame_id == "arm_mount"
    p, o = view.pose.pose.position, view.pose.pose.orientation
    np.testing.assert_allclose([p.x, p.y, p.z, o.x, o.y, o.z, o.w], [0.4, 0.1, 0.3, *UP])
    assert (view.orientation_hold, grasp.orientation_hold) == (0, HOLD_FIXED)
    assert (view.blend_radius, grasp.blend_radius) == (pytest.approx(0.05), 0.0)
    # the default per leg is GO_TO_SPEED_SCALE; a target's own scale wins
    assert (view.speed_scale, grasp.speed_scale) == (pytest.approx(0.3), pytest.approx(0.5))
    assert (g.sender_id, bytes(g.token)) == (SENDER_ID, TOKEN)
    assert not fake.goals["go_to_ee_pose"]  # one goal, not one per target


def test_go_to_ee_poses_passes_the_mode_for_the_whole_motion(arm_env):
    arm, _, fake = arm_env
    fake.finish_at("go_to_ee_poses", flange=([0.4, 0.1, 0.3 - TOOL_Z], UP))
    arm.begin_action(FakeCtx(), None)
    held = PoseTarget([0.4, 0.1, 0.3], UP, hold=HOLD_LEVEL)
    assert arm.go_to_ee_poses([held], control_mode=IMPEDANCE, gains="soft") is True
    g = fake.goals["go_to_ee_poses"][-1]
    assert g.control_mode == IMPEDANCE and g.targets[0].orientation_hold == HOLD_LEVEL


def test_go_to_ee_poses_checks_where_the_tool_ended(arm_env):
    arm, _, fake = arm_env
    fake.finish_at("go_to_ee_poses", flange=([0.5, 0.1, 0.215 - TOOL_Z], UP))
    arm.begin_action(FakeCtx(), None)
    assert arm.go_to_ee_poses([VIEW, GRASP]) is False
    assert "tool 1.5 cm from the goal" in arm.last_error


def test_go_to_joint_configs_sends_each_target_and_checks_the_joints(arm_env):
    arm, _, fake = arm_env
    fake.finish_at("go_to_joint_configs", joints=END)
    arm.begin_action(FakeCtx(), None)
    targets = [JointTarget([0.0] * 7, blend_radius=0.05), JointTarget(END, speed_scale=0.8)]
    assert arm.go_to_joint_configs(targets, "sweep") is True
    (g,) = fake.goals["go_to_joint_configs"]
    np.testing.assert_allclose([list(t.target_joints) for t in g.targets], [[0.0] * 7, END])
    assert [t.speed_scale for t in g.targets] == pytest.approx([0.3, 0.8])
    assert [t.blend_radius for t in g.targets] == pytest.approx([0.05, 0.0])


def test_a_driver_failure_is_a_refusal(arm_env):
    arm, _, fake = arm_env
    fake.set_result("go_to_ee_poses", -7)
    arm.begin_action(FakeCtx(), None)
    assert arm.go_to_ee_poses([VIEW, GRASP], "grab") is False
    assert "grab ended PLANNING_FAILED" in arm.last_error


def test_no_targets_and_bad_gains_raise_before_anything_is_sent(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(ArmError, match="no targets"):
        arm.go_to_ee_poses([])
    with pytest.raises(ArmError, match="no targets"):
        arm.go_to_joint_configs([])
    with pytest.raises(ArmError, match="gains"):
        arm.go_to_ee_poses([VIEW], gains="bogus")
    assert not fake.goals["go_to_ee_poses"] and not fake.goals["go_to_joint_configs"]


def test_mock_mode_ends_on_the_last_target(ros):
    node = Node("arm_mock_goto_many")
    arm = ArmInterface(node, ReentrantCallbackGroup(), mock=True)
    try:
        arm.begin_action(FakeCtx(), None)
        assert arm.go_to_joint_configs([JointTarget([0.0] * 7), JointTarget(END)]) is True
        np.testing.assert_allclose(arm.raw_joints(), END)
    finally:
        arm.end_action()
        node.destroy_node()
