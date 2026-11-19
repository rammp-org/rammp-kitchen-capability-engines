"""Microwave free-space detours retain endpoint/floor/planner validation, without ROS."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def checks(monkeypatch, tmp_path):
    settings = {}
    dc = SimpleNamespace(configure_paths=Mock(), configure=lambda **kw: settings.update(kw))
    package = ModuleType("door_common")
    package.common = dc
    monkeypatch.setitem(sys.modules, "door_common", package)
    spec = importlib.util.spec_from_file_location(
        "microwave_motion_constants", ROOT / "capabilities/microwave/microwave/constants.py"
    )
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    constants.configure(tmp_path)
    scope = dict(
        np=np,
        FREE_MOVE_MAX_SWING_RAD=settings["FREE_MOVE_MAX_SWING_RAD"],
        FREE_MOVE_END_DEG=2.0,
        CUROBO_RETRIES=0,
        short_delta=lambda a, b: np.asarray(b) - np.asarray(a),
        reconfiguration=Mock(return_value=""),
        path_below_floor=Mock(return_value=""),
        path_not_in_front=Mock(return_value=""),
        _over_limit=lambda message: None,
    )
    source = ROOT / "shared/door_common/door_common/common.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "free_move_problem")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    return scope


def plan(endpoint=0.1):
    return SimpleNamespace(
        success=True, q=[np.zeros(7), np.array([1.5, 0, 0, 0, 0, 0, 0]), np.full(7, endpoint)], message="planned"
    )


def check(scope, result):
    arm = SimpleNamespace(plan_to_joints=Mock(return_value=result))
    return scope["free_move_problem"](arm, np.zeros(7), np.full(7, 0.1))


def test_microwave_allows_large_detour_that_the_old_cap_refused(checks):
    assert np.isinf(checks["FREE_MOVE_MAX_SWING_RAD"])
    assert check(checks, plan())[1] == ""
    checks["FREE_MOVE_MAX_SWING_RAD"] = np.radians(25)
    assert "swings J1" in check(checks, plan())[1]


def test_wrong_joint_endpoint_is_still_refused(checks):
    assert "off the goal joints" in check(checks, plan(0.3))[1]


def test_floor_and_planner_refusals_are_still_respected(checks):
    checks["path_below_floor"].return_value = "tool below floor"
    assert "below floor" in check(checks, plan())[1]
    failed = SimpleNamespace(success=False, q=[], message="collision")
    assert "collision" in check(checks, failed)[1]


def test_reconfiguration_checks_are_unchanged(checks):
    checks["reconfiguration"].return_value = "arm would change its shape"
    assert "change its shape" in check(checks, plan())[1]
