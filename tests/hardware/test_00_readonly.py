"""Stage 0, nothing moves: state reads, the driver's model, the controllers, the planner.

Safe beside a running cell: it only reads topics and asks cuRobo for plans."""

import time

import numpy as np
import pytest
import rammp_arm
from conftest import HOME, STATE_FRESH_S, angle_deg, diagnostic, joint_err

from capability_engine import NotInAction, sim
from capability_engine.arm import SETPOINTS

FK_POS_M = 0.005  # the sim's tool frame vs the driver's, at the measured joints
FK_ROT_DEG = 1.0
PLAN_START_RAD = 1e-3  # a plan's first sample vs the start it was given
PLAN_GOAL_RAD = 1e-3  # a joint plan's last sample vs its goal
PLAN_POSE_M = 0.01  # a pose plan's last sample (through the sim's FK) vs its target
PLAN_POSE_DEG = 2.0
UP = np.array([0.0, 0.0, 0.05])  # the pose the planning tests aim at: 5 cm above the tool
WRIST_TURN = np.array([0, 0, 0, 0, 0, 0, 0.2])  # the joint plan: J7 +0.2 rad
FAR = [2.0, 0.0, 0.5]  # 2 m out: no plan


def test_the_state_reads_are_fresh_and_ready(cell):
    def body(arm, ctx):
        return dict(
            js=arm.joint_states(),
            ee=arm.ee_state(),
            grip=arm.gripper_state(),
            control=arm.control_status(),
            stream=arm.stream_status(),
            diagnostics=arm.diagnostics(),
            why=arm.not_ready(),
        )

    s = cell.act(body)
    for key in ("js", "ee", "grip"):
        assert s[key].age_s < STATE_FRESH_S, (key, s[key].age_s)
    assert s["js"].position.shape == (7,)
    assert s["ee"].frame_id == rammp_arm.FRAME
    assert s["grip"].present, "the driver reports no gripper"
    assert not s["control"].estopped
    assert not s["stream"].open, f"a stream is open: {s['stream']}"
    for task in ("Arm", "Arm link"):  # the sim driver has no "Arm link"
        d = diagnostic(s["diagnostics"], task)
        assert d is not None or task == "Arm link", s["diagnostics"]
        assert d is None or d.level == 0, d
    assert s["why"] is None


def test_get_state_is_the_students_dict(cell):
    def body(arm, ctx):
        return arm.get_state(), arm.raw_joints()

    state, raw = cell.act(body)
    assert set(state) == {"position", "velocity", "effort", "ee_pos", "gripper_pos"}
    assert np.all(np.abs(state["position"]) <= np.pi)
    assert joint_err(state["position"], raw).max() < 1e-3  # the same joints (two reads)
    assert len(state["ee_pos"]) == 7
    pos, quat = cell.observer.tool()
    assert np.linalg.norm(state["ee_pos"][:3] - pos) < 0.002  # the tool frame, not the flange
    assert angle_deg(state["ee_pos"][3:], quat) < 0.5
    assert 0.0 <= state["gripper_pos"] <= 1.0


def test_the_sim_agrees_with_the_driver(cell):
    """make_sim() builds from the driver's URDF; its tool frame at the measured joints is where
    the driver says the tool is. If not, every pose a capability checks in the sim is off."""

    def body(arm, ctx):
        urdf = arm.robot_description()
        model = sim.make_sim()
        raw = arm.raw_joints()
        return urdf, sim.tool_pose(*model, raw), arm.get_state()["ee_pos"]

    urdf, (pos, quat), ee = cell.act(body)
    for joint in rammp_arm.JOINTS:
        assert f'name="{joint}"' in urdf
    assert np.linalg.norm(pos - ee[:3]) < FK_POS_M, (pos, ee[:3])
    assert angle_deg(quat, ee[3:]) < FK_ROT_DEG


def test_every_available_controller_can_be_streamed(cell):
    """stream() takes a controller that reads one channel it knows; the driver may list others
    (cartesian_impedance reads pose and wrench) as long as they are not available."""
    controllers = cell.act(lambda arm, ctx: arm.list_controllers())
    available = [c for c in controllers if c.available]
    for c in available:
        assert len(c.channels) == 1 and c.channels[0] in SETPOINTS, c
    channels = {c.channels[0] for c in available}
    assert {"joint_position", "joint_velocity", "pose", "twist"} <= channels, controllers


def test_plan_to_joints(cell):
    raw = np.array(HOME)  # wherever the arm is: plans start where they are told

    def body(arm, ctx):
        return raw, arm.plan_to_joints(raw + WRIST_TURN, raw)

    raw, plan = cell.act(body)
    assert plan.success, plan.message
    assert plan.q.shape[1] == 7 and plan.qd.shape == plan.q.shape and len(plan.t) == len(plan.q)
    assert np.all(np.diff(plan.t) > 0)
    assert np.abs(plan.q[0] - raw).max() < PLAN_START_RAD
    assert np.abs(plan.q[-1] - (raw + WRIST_TURN)).max() < PLAN_GOAL_RAD


def test_plan_to_pose_ends_at_the_tool_target(cell):
    """cuRobo's goal frame, self.arm's tool frame and the sim's are one frame: the plan's last
    joints put the sim's tool on the target."""

    def body(arm, ctx):
        model = sim.make_sim()
        pos, quat = sim.tool_pose(*model, HOME)
        target = pos + UP
        plan = arm.plan_to_pose(target, quat, HOME)
        end = sim.tool_pose(*model, plan.q[-1]) if plan.success else None
        return target, quat, plan, end

    target, quat, plan, end = cell.act(body)
    assert plan.success, plan.message
    assert np.linalg.norm(end[0] - target) < PLAN_POSE_M, (end[0], target)
    assert angle_deg(end[1], quat) < PLAN_POSE_DEG


def test_an_unreachable_plan_fails_without_raising(cell):
    def body(arm, ctx):
        return arm.plan_to_pose(FAR, [0, 0, 0, 1], arm.raw_joints()), arm.last_error

    plan, last_error = cell.act(body)
    assert not plan.success
    assert plan.message and last_error


def test_the_arm_is_only_available_inside_an_action(cell):
    with pytest.raises(NotInAction):
        cell.engine.arm.get_state()


def test_each_action_reads_the_driver_afresh(cell):
    """No state is carried from one action to the next: a value kept from the first would be
    stale by the second, and joint_states() refuses stale state."""
    cell.act(lambda arm, ctx: arm.joint_states())
    time.sleep(2 * STATE_FRESH_S)
    assert cell.act(lambda arm, ctx: arm.joint_states()).age_s < STATE_FRESH_S


@pytest.mark.motion  # changes the planner's world, then puts the cell's own back
def test_set_world(cell, request):
    world = request.config.getoption("--world")

    def body(arm, ctx):
        good = arm.set_world(world)
        bad = arm.set_world("no_such_world.yaml")
        plan = arm.plan_to_joints(np.array(HOME) + WRIST_TURN, HOME)  # the planner still works
        return good, bad, plan, arm.set_world(world)

    good, bad, plan, restored = cell.act(body)
    assert good[0], good[1]
    assert not bad[0] and bad[1]
    assert plan.success, plan.message
    assert restored[0], restored[1]
