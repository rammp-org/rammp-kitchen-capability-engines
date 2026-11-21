"""Offline take-out regressions: planner failures never fall back to unchecked approach IK."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def motion(monkeypatch):
    # Isolate the motion module from ROS/PyBullet; no robot resources are created.
    package = ModuleType("take_out_test_package")
    package.__path__ = []
    pm = ModuleType("take_out_test_package.place_motion")
    pm.Refused = type("Refused", (Exception,), {})
    pm.LINE_STEP = 0.004
    pm.LINE_STEP_DEG = 0.5
    pm.SETTLE_AFTER_S = 0.0
    pm.dense = lambda qs, step: qs
    pm.pose_lengths = lambda model, qs: [
        float(np.linalg.norm(model.fk(b)[:3, 3] - model.fk(a)[:3, 3])) for a, b in zip(qs[:-1], qs[1:], strict=True)
    ]
    # Exercise the production corridor check, without importing its ROS/PyBullet dependencies.
    tree = ast.parse((Path(__file__).parents[1] / "microwave_grab/place_motion.py").read_text())
    check = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "check_line")
    check_scope = dict(np=np, LINE_TOL=0.01, LINE_TOL_DEG=2.0)

    pm.END_TOL, pm.END_TOL_DEG = 0.01, 2.0
    pm.START_TOL_DEG, pm.SUCCESSFUL = 3.0, 0
    pm.wrap_deg = lambda q: np.degrees(q)
    pm.ready = Mock()
    check_scope["Refused"] = pm.Refused
    exec(compile(ast.Module(body=[check], type_ignores=[]), "check_line", "exec"), check_scope)
    pm.check_line = check_scope["check_line"]
    capability = ModuleType("capability_engine")
    capability.__path__ = []
    interface = ModuleType("capability_engine.arm")
    interface.HOLD_FIXED = 2
    monkeypatch.setitem(sys.modules, "capability_engine", capability)
    monkeypatch.setitem(sys.modules, "capability_engine.arm", interface)
    arm_api = ModuleType("rammp_arm")
    arm_api.JOINTS = [f"arm_joint_{i}" for i in range(1, 8)]
    for name, module in ((package.__name__, package), (pm.__name__, pm), ("rammp_arm", arm_api)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "take_out_test_package.take_out", Path(__file__).parents[1] / "microwave_grab" / "take_out.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, pm


WORLD = "world set (31 boxes: scene 0 | config layers: platform 16, seat 10, person 4 | scene layers: floor 1)"


def test_world_requires_persistent_layers_and_floor(motion):
    take, _ = motion
    arm = Mock()
    arm.set_world.return_value = True, WORLD
    assert take.set_approach_world(arm, "/config/robot/curobo/world.yaml", Mock()) == WORLD


@pytest.mark.parametrize(
    "message",
    [
        "world set",
        WORLD.replace("person 4", "person 0"),
        WORLD.replace("seat 10", "seat 0"),
        WORLD.replace("platform 16", "platform 0"),
        WORLD.replace("floor 1", "floor 0"),
        WORLD.replace("config layers:", "scene layers:"),
    ],
)
def test_world_missing_layers_refused(motion, message):
    take, pm = motion
    arm = Mock()
    arm.set_world.return_value = True, message
    with pytest.raises(pm.Refused):
        take.set_approach_world(arm, "world.yaml", Mock())


def test_world_load_failure_refused(motion):
    take, pm = motion
    arm = Mock()
    arm.set_world.return_value = False, "missing file"
    with pytest.raises(pm.Refused, match="missing file"):
        take.set_approach_world(arm, "world.yaml", Mock())


def test_planner_failure_never_falls_back_to_local_ik(motion):
    take, pm = motion
    arm, model = Mock(), Mock()
    arm.plan_to_pose.return_value = SimpleNamespace(success=False, message="INVALID_START person/torso")
    with pytest.raises(pm.Refused, match="INVALID_START person/torso"):
        take.plan_take_out(
            arm,
            model,
            np.zeros(7),
            {"pre": [0, 0, 0], "quat": [0, 0, 0, 1]},
            {"take_approach_speed_scale": 0.3, "take_exit_speed_scale": 0.3},
        )
    model.assert_not_called()
    model.ik.assert_not_called()
    arm.execute_joint_trajectory.assert_not_called()


def test_dry_plan_seeds_insertion_from_curobo_endpoint(motion, monkeypatch):
    take, _ = motion
    arm, model = Mock(), Mock()
    plan = SimpleNamespace(
        success=True, q=np.array([np.zeros(7), np.ones(7)]), t=np.array([0.0, 2.0]), qd=np.zeros((2, 7))
    )
    arm.plan_to_pose.return_value = plan
    tail = Mock(return_value={"line": [], "lift": [np.zeros(7)], "end": (0, 0)})
    monkeypatch.setattr(take, "plan_withdrawal", Mock(return_value={"out": plan.q}))
    monkeypatch.setattr(take, "plan_from_pregrasp", tail)
    cand, cfg = (
        {"pre": [0, 0, 0], "quat": [0, 0, 0, 1]},
        {"take_approach_speed_scale": 0.3, "take_exit_speed_scale": 0.3},
    )
    legs = take.plan_take_out(arm, model, np.zeros(7), cand, cfg)
    np.testing.assert_array_equal(tail.call_args.args[1], plan.q[-1])
    np.testing.assert_array_equal(legs["approach_t"], plan.t)
    arm.execute_joint_trajectory.assert_not_called()
    arm.move_joints.assert_not_called()


@pytest.mark.parametrize("speed", [0, -1, float("nan"), float("inf"), 1.1])
def test_invalid_speed_refused_before_planning(motion, speed):
    take, pm = motion
    arm = Mock()
    with pytest.raises(pm.Refused, match="speed_scale"):
        take.plan_take_out(
            arm, Mock(), np.zeros(7), {}, {"take_approach_speed_scale": speed, "take_exit_speed_scale": 0.3}
        )
    arm.plan_to_pose.assert_not_called()


@pytest.mark.parametrize("position, rotation", [(0.02, np.eye(3)), (0.0, np.diag([-1.0, -1.0, 1.0]))])
def test_measured_pregrasp_error_stops_insertion(motion, position, rotation):
    take, pm = motion
    model = Mock()
    pose = np.eye(4)
    pose[0, 3], pose[:3, :3] = position, rotation
    model.fk.return_value = pose
    with pytest.raises(pm.Refused, match="not inserting"):
        take.plan_from_pregrasp(model, np.zeros(7), {"pre": np.zeros(3), "quat": [0, 0, 0, 1]}, {})


def test_failed_approach_stops_before_insertion_and_preserves_samples(motion, monkeypatch):
    take, pm = motion
    arm, ctx = Mock(), Mock()
    arm.raw_joints.return_value = np.zeros(7)
    arm.execute_joint_trajectory.return_value = SimpleNamespace(error_code=1, error_string="tracking failed")
    monkeypatch.setattr(take, "set_gripper", Mock())
    tail = Mock()
    monkeypatch.setattr(take, "plan_from_pregrasp", tail)
    legs = {"approach_t": [0.0, 1.0], "approach": [np.zeros(7), np.ones(7)], "approach_qd": [np.zeros(7), np.zeros(7)]}
    with pytest.raises(pm.Refused, match="tracking failed"):
        take.run_take_out(
            arm, Mock(), np.zeros(7), {}, legs, {"grip_open": 0.0, "take_approach_speed_scale": 0.3}, ctx, Mock()
        )
    call = arm.execute_joint_trajectory.call_args
    sent = list(call.args[1])
    assert [point[0] for point in sent] == legs["approach_t"]
    np.testing.assert_array_equal(sent[-1][1], legs["approach"][-1])
    np.testing.assert_array_equal(sent[-1][2], legs["approach_qd"][-1])
    assert call.kwargs["speed_scale"] == 0.3
    tail.assert_not_called()
    arm.move_joints.assert_not_called()


@pytest.mark.parametrize("extra", [0.05, 0.07])
def test_deeper_grasp_extends_insertion_without_moving_pregrasp(motion, extra):
    take, _ = motion
    model = Mock()
    pose = np.eye(4)
    # Tool heading along base x, fingers initially along y.
    pose[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    model.fk.return_value = pose
    point = np.array([1.0, -0.2, 0.3])
    old = take.grasp_pose(model, np.zeros(7), point, 0.25, -0.05, 0.035)
    new = take.grasp_pose(model, np.zeros(7), point, 0.25 + extra, -0.05 + extra, 0.035)
    np.testing.assert_allclose(new["target"] - old["target"], extra * old["heading"])
    np.testing.assert_allclose(new["pre"], old["pre"])
    np.testing.assert_allclose(new["quat"], old["quat"])


def test_home_already_reached_does_not_move(motion):
    take, _ = motion
    arm = Mock()
    arm.raw_joints.return_value = np.zeros(7)
    q = take.prepare_home(arm, [0.0] * 7, True, 0.3, Mock(), Mock())
    np.testing.assert_array_equal(q, np.zeros(7))
    arm.go_to_joint_config.assert_not_called()


def test_dry_run_off_home_refuses_without_moving(motion):
    take, pm = motion
    arm = Mock()
    arm.raw_joints.return_value = np.ones(7)
    with pytest.raises(pm.Refused, match="dry run"):
        take.prepare_home(arm, [0.0] * 7, False, 0.3, Mock(), Mock())
    arm.go_to_joint_config.assert_not_called()


def test_home_move_uses_configured_target_and_verifies_endpoint(motion):
    take, _ = motion
    arm = Mock()
    home = np.arange(7, dtype=float) / 10
    arm.raw_joints.side_effect = [np.ones(7), home]
    arm.go_to_joint_config.return_value = True
    np.testing.assert_array_equal(take.prepare_home(arm, home, True, 0.3, Mock(), Mock()), home)
    np.testing.assert_array_equal(arm.go_to_joint_config.call_args.args[0], home)
    assert arm.go_to_joint_config.call_args.kwargs["speed_scale"] == 0.3


def test_home_move_failure_refuses(motion):
    take, pm = motion
    arm = Mock()
    arm.raw_joints.return_value = np.ones(7)
    arm.go_to_joint_config.return_value = False
    arm.last_error = "collision"
    with pytest.raises(pm.Refused, match="collision"):
        take.prepare_home(arm, [0.0] * 7, True, 0.3, Mock(), Mock())


def test_home_tracking_error_refuses(motion):
    take, pm = motion
    arm = Mock()
    arm.raw_joints.return_value = np.ones(7)
    arm.go_to_joint_config.return_value = True
    with pytest.raises(pm.Refused, match="not detecting"):
        take.prepare_home(arm, [0.0] * 7, True, 0.3, Mock(), Mock())


class ExitModel:
    def fk(self, q):
        from scipy.spatial.transform import Rotation

        T = np.eye(4)
        T[:3, 3] = np.asarray(q)[:3]
        T[:3, :3] = Rotation.from_euler("z", float(q[3])).as_matrix()
        return T


def exit_setup():
    start = np.array([1.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    plan = SimpleNamespace(
        success=True, message="", q=np.array([start, end]), t=np.array([0.0, 1.0]), qd=np.zeros((2, 7))
    )
    arm = Mock()
    arm.plan_to_pose.return_value = plan
    cfg = {"take_exit_speed_scale": 0.3, "insert_speed": 0.16}
    return arm, plan, cfg, start, end


def test_exit_holds_orientation_and_uses_lifted_height(motion):
    take, _ = motion
    arm, plan, cfg, start, end = exit_setup()
    out = take.plan_exit(arm, ExitModel(), start, [0.5, 0.0, 0.1], cfg)
    args = arm.plan_to_pose.call_args
    np.testing.assert_allclose(args.args[0], end[:3])
    np.testing.assert_allclose(args.args[2], start)
    assert args.kwargs["hold"] == 2
    np.testing.assert_array_equal(out["out"], plan.q)
    arm.execute_joint_trajectory.assert_not_called()


def test_exit_invalid_start_refuses_without_local_fallback(motion):
    take, pm = motion
    arm, _, cfg, start, _ = exit_setup()
    arm.plan_to_pose.return_value = SimpleNamespace(success=False, message="INVALID_START person/lap")
    with pytest.raises(pm.Refused, match="person/lap"):
        take.plan_exit(arm, ExitModel(), start, [0.5, 0.0, 0.1], cfg)
    arm.execute_joint_trajectory.assert_not_called()
    arm.move_joints.assert_not_called()


def test_exit_sideways_detour_rejected(motion):
    take, pm = motion
    arm, plan, cfg, start, end = exit_setup()
    middle = (start + end) / 2
    middle[1] = 0.03
    plan.q = np.array([start, middle, end])
    with pytest.raises(pm.Refused, match="off its line"):
        take.plan_exit(arm, ExitModel(), start, [0.5, 0.0, 0.1], cfg)
    arm.execute_joint_trajectory.assert_not_called()


def test_exit_rotation_rejected_even_if_position_straight(motion):
    take, pm = motion
    arm, plan, cfg, start, _ = exit_setup()
    plan.q[-1, 3] = np.radians(5.0)
    with pytest.raises(pm.Refused, match="turns the held container"):
        take.plan_exit(arm, ExitModel(), start, [0.5, 0.0, 0.1], cfg)


def test_exit_incomplete_endpoint_rejected(motion):
    take, pm = motion
    arm, plan, cfg, start, _ = exit_setup()
    plan.q[-1, 0] = 0.6
    with pytest.raises(pm.Refused, match="retreat endpoint"):
        take.plan_exit(arm, ExitModel(), start, [0.5, 0.0, 0.1], cfg)


def test_exit_capped_at_existing_path_speed(motion):
    take, _ = motion
    arm, plan, cfg, start, _ = exit_setup()
    plan.t[-1] = 0.1
    out = take.plan_exit(arm, ExitModel(), start, [0.5, 0.0, 0.1], cfg)
    assert out["out_speed_scale"] == pytest.approx(0.032)


def test_exit_infeasible_speed_refused(motion):
    take, pm = motion
    arm, plan, cfg, start, _ = exit_setup()
    plan.t[-1] = 0.001
    with pytest.raises(pm.Refused, match="driver minimum"):
        take.plan_exit(arm, ExitModel(), start, [0.5, 0.0, 0.1], cfg)


@pytest.mark.parametrize("times", [[0.0, 0.0], [0.0, -1.0], [0.0, float("nan")]])
def test_exit_bad_timing_refused(motion, times):
    take, pm = motion
    arm, plan, cfg, start, _ = exit_setup()
    plan.t = np.array(times)
    with pytest.raises(pm.Refused, match="invalid sample timing"):
        take.plan_exit(arm, ExitModel(), start, [0.5, 0.0, 0.1], cfg)


def test_execute_exit_uses_actual_start_and_exact_checked_samples(motion):
    take, _ = motion
    arm, plan, cfg, start, end = exit_setup()
    arm.raw_joints.side_effect = [start, start, end]
    sent = []

    def execute(names, points, **kwargs):
        sent.extend(list(points))
        return SimpleNamespace(error_code=0, error_string="")

    arm.execute_joint_trajectory.side_effect = execute
    take.execute_exit(arm, ExitModel(), [0.5, 0.0, 0.1], cfg, Mock(), Mock())
    np.testing.assert_array_equal(arm.plan_to_pose.call_args.args[2], start)
    np.testing.assert_array_equal([row[1] for row in sent], plan.q)
    np.testing.assert_array_equal([row[2] for row in sent], plan.qd)
    assert [row[0] for row in sent] == list(plan.t)
    arm.move_joints.assert_not_called()


def test_exit_planning_failure_after_lift_sends_nothing(motion):
    take, pm = motion
    arm, _, cfg, start, _ = exit_setup()
    arm.raw_joints.return_value = start
    arm.plan_to_pose.return_value = SimpleNamespace(success=False, message="collision")
    with pytest.raises(pm.Refused, match="collision"):
        take.execute_exit(arm, ExitModel(), [0.5, 0.0, 0.1], cfg, Mock(), Mock())
    arm.execute_joint_trajectory.assert_not_called()
    arm.move_joints.assert_not_called()
    arm.gripper.assert_not_called()


def test_exit_rechecks_start_before_execution(motion):
    take, pm = motion
    arm, _, cfg, start, _ = exit_setup()
    shifted = start.copy()
    shifted[-1] = 0.2
    arm.raw_joints.side_effect = [start, shifted]
    with pytest.raises(pm.Refused, match="arm moved"):
        take.execute_exit(arm, ExitModel(), [0.5, 0.0, 0.1], cfg, Mock(), Mock())
    arm.execute_joint_trajectory.assert_not_called()


def test_exit_must_plan_before_any_approach_motion(motion, monkeypatch):
    take, pm = motion
    arm, approach, cfg, start, _ = exit_setup()
    cfg["take_approach_speed_scale"] = 0.3
    arm.plan_to_pose.return_value = approach
    monkeypatch.setattr(take, "plan_from_pregrasp", Mock(return_value={"lift": [start]}))
    monkeypatch.setattr(take, "plan_withdrawal", Mock(side_effect=pm.Refused("invalid lift endpoint")))
    with pytest.raises(pm.Refused, match="invalid lift endpoint"):
        take.plan_take_out(arm, ExitModel(), start, {"pre": [0.5, 0.0, 0.1], "quat": [0, 0, 0, 1]}, cfg)
    arm.execute_joint_trajectory.assert_not_called()
    arm.move_joints.assert_not_called()
    arm.gripper.assert_not_called()


def test_local_withdrawal_never_calls_curobo(motion):
    take, pm = motion
    arm, _, cfg, start, end = exit_setup()
    pm.plan_line = Mock(return_value=[start, end])
    route = take.plan_withdrawal(arm, ExitModel(), start, end[:3], cfg)
    np.testing.assert_allclose(route["out_target"], end[:3])
    arm.plan_to_pose.assert_not_called()


def test_fully_closed_gripper_allows_lift_and_retreat(motion):
    take, _ = motion
    arm = Mock()
    arm.gripper_state.return_value = SimpleNamespace(position=1.0)
    assert take.close_on_brim(arm, Mock()) == 1.0


def test_gen15_layer_contract_without_platform(motion):
    take, pm = motion
    arm = Mock()
    message = "world: 27 boxes: scene 6 | config layers: seat 17, person 3 | scene layers: floor 1"
    arm.set_world.return_value = True, message
    assert take.set_approach_world(arm, "world.yaml", Mock(), {"seat": 17, "person": 3}) == message
    for expected in ({"seat": 16, "person": 3}, {"seat": 17}, {"seat": 17, "person": 3, "platform": 16}):
        with pytest.raises(pm.Refused):
            take.set_approach_world(arm, "world.yaml", Mock(), expected)



def test_insertion_collision_refuses_without_retrying_approach(motion, monkeypatch):
    take, pm = motion
    arm = Mock()
    arm.plan_to_pose.return_value = SimpleNamespace(success=True, q=[np.zeros(7), np.full(7, 0.1)])
    monkeypatch.setattr(take, "plan_from_pregrasp", Mock(side_effect=pm.Refused("door collision")))
    with pytest.raises(pm.Refused, match="door collision"):
        take.plan_take_out(
            arm,
            Mock(),
            np.zeros(7),
            dict(pre=[0, 0, 0], quat=[0, 0, 0, 1]),
            dict(take_exit_speed_scale=0.3, take_approach_speed_scale=0.3),
        )
    arm.plan_to_pose.assert_called_once()
    arm.plan_to_joints.assert_not_called()
    arm.execute_joint_trajectory.assert_not_called()


def test_insertion_requests_fixed_orientation_and_checks_drift(motion, monkeypatch):
    take, pm = motion
    arm, model = Mock(), Mock()
    q = np.zeros(7)
    model.fk.return_value = np.eye(4)
    plan = SimpleNamespace(success=True, q=np.array([q, q]), message="")
    arm.plan_to_pose.return_value = plan
    monkeypatch.setattr(pm, "end_pose_error", Mock(return_value=(0.0, 0.0)), raising=False)
    monkeypatch.setattr(pm, "plan_line", Mock(return_value=[q, q]), raising=False)
    cand = dict(pre=np.zeros(3), target=np.zeros(3), quat=[0, 0, 0, 1], heading=[1, 0, 0])
    cfg = dict(impedance_in=0.1, take_lift=0.06)
    take.plan_from_pregrasp(model, q, cand, cfg, arm)
    assert arm.plan_to_pose.call_count == 2
    assert all(call.args[3] == take.HOLD_FIXED for call in arm.plan_to_pose.call_args_list)
    # A planner claiming success must still fail the independent 2-degree check.
    tilted = np.eye(4)
    tilted[:3, :3] = take.Rotation.from_euler("x", 3, degrees=True).as_matrix()
    model.fk.side_effect = [np.eye(4), tilted]
    with pytest.raises(pm.Refused, match="orientation deviates"):
        take.plan_from_pregrasp(model, q, cand, cfg, arm)


def test_partial_run_plans_only_first_segment(motion, monkeypatch):
    take, pm = motion
    arm, model = Mock(), Mock()
    q = np.zeros(7)
    model.fk.return_value = np.eye(4)
    arm.plan_to_pose.return_value = SimpleNamespace(success=True, q=[q, q], message="")
    later = Mock(side_effect=AssertionError("partial run must not plan lift"))
    monkeypatch.setattr(pm, "plan_line", later, raising=False)
    result = take.plan_from_pregrasp(
        model, q, dict(pre=np.zeros(3), target=[0.1, 0, 0], quat=[0, 0, 0, 1], heading=[1, 0, 0]),
        dict(impedance_in=0.1, stop_after_first_insertion=True), arm,
    )
    assert len(result["insertion_plans"]) == 1
    assert arm.plan_to_pose.call_count == 1
    assert arm.plan_to_pose.call_args.args[3] == take.HOLD_FIXED
    later.assert_not_called()


def test_partial_run_stops_before_impedance_close_and_retreat(motion, monkeypatch):
    take, pm = motion
    q = np.zeros(7)
    arm = Mock()
    arm.raw_joints.return_value = q
    arm.execute_joint_trajectory.return_value = SimpleNamespace(error_code=pm.SUCCESSFUL)
    plan = SimpleNamespace(q=[q, q])
    monkeypatch.setattr(take, "set_gripper", Mock())
    monkeypatch.setattr(take, "plan_from_pregrasp", Mock(return_value=dict(insertion_plans=[plan])))
    insertion = Mock()
    monkeypatch.setattr(take, "run_insertion_plan", insertion)
    forbidden = Mock(side_effect=AssertionError("must not execute beyond first insertion"))
    monkeypatch.setattr(take, "plan_withdrawal", forbidden)
    monkeypatch.setattr(take, "close_on_brim", forbidden)
    result = take.run_take_out(
        arm, Mock(), q, {}, dict(approach_t=[0, 1], approach=[q, q], approach_qd=[q, q]),
        dict(grip_open=0, take_approach_speed_scale=0.3, insert_speed=0.048, stop_after_first_insertion=True),
        Mock(), Mock(),
    )
    assert result["stopped_at"] == "first_insertion"
    assert insertion.call_count == 1
    forbidden.assert_not_called()


def test_explicit_straight_final_insertion_keeps_pose_planned_first_segment(motion, monkeypatch):
    take, pm = motion
    q = np.zeros(7)
    model, arm = Mock(), Mock()
    model.fk.return_value = np.eye(4)
    line = [q, q]
    plan = SimpleNamespace(success=True, q=line)
    arm.plan_to_pose.return_value = plan
    planner = Mock(return_value=line)
    monkeypatch.setattr(pm, "plan_line", planner, raising=False)
    monkeypatch.setattr(pm, "end_pose_error", Mock(return_value=(0.0, 0.0)), raising=False)
    monkeypatch.setattr(pm, "check_line", Mock())
    monkeypatch.setattr(take, "straight_plan", Mock(return_value=plan))
    result = take.plan_from_pregrasp(
        model, q, dict(pre=np.zeros(3), target=[0.1, 0, 0], quat=[0, 0, 0, 1], heading=[1, 0, 0]),
        dict(impedance_in=0.1, take_lift=0.02, straight_insertion=True), arm,
    )
    assert len(result["insertion_plans"]) == 2
    assert planner.call_count == 2  # final insertion, then lift
    pm.check_line.assert_called_once()
    arm.plan_to_pose.assert_called_once()
    assert arm.plan_to_pose.call_args.args[3] == take.HOLD_FIXED


def test_explicit_motion_scale_uses_driver_trajectory_without_legacy_move(motion, monkeypatch):
    take, _ = motion
    arm, model = Mock(), Mock()
    qs = [np.zeros(7), np.ones(7)]
    plan = SimpleNamespace()
    monkeypatch.setattr(take, "straight_plan", Mock(return_value=plan))
    execute = Mock()
    monkeypatch.setattr(take, "run_insertion_plan", execute)
    take.move_timed(arm, model, qs, "lift", 0.05, "holding", max_scale=0.3)
    assert execute.call_args.args[2] is plan
    assert execute.call_args.args[4] == 0.3
    arm.move_joints.assert_not_called()
