"""Offline take-out regressions: planner failures never fall back to unchecked approach IK."""

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
    pm.END_TOL, pm.END_TOL_DEG = 0.01, 2.0
    pm.START_TOL_DEG, pm.SUCCESSFUL = 3.0, 0
    pm.wrap_deg = lambda q: np.degrees(q)
    pm.ready = Mock()
    arm_api = ModuleType("rammp_arm")
    arm_api.JOINTS = [f"arm_joint_{i}" for i in range(1, 8)]
    for name, module in ((package.__name__, package), (pm.__name__, pm), ("rammp_arm", arm_api)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "take_out_test_package.take_out", Path(__file__).parents[1] / "microwave_place" / "take_out.py"
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
            arm, model, np.zeros(7), {"pre": [0, 0, 0], "quat": [0, 0, 0, 1]}, {"take_approach_speed_scale": 0.3}
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
    tail = Mock(return_value={"line": [], "lift": [], "out": [], "end": (0, 0)})
    monkeypatch.setattr(take, "plan_from_pregrasp", tail)
    cand, cfg = {"pre": [0, 0, 0], "quat": [0, 0, 0, 1]}, {"take_approach_speed_scale": 0.3}
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
        take.plan_take_out(arm, Mock(), np.zeros(7), {}, {"take_approach_speed_scale": speed})
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
