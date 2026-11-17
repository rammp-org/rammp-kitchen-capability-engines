"""Offline take-out HOME and UI bridge tests; production methods run without ROS or hardware. The one-press UI flows
are in test_ui_flow.py."""

import ast
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_class(path, name, methods=None, **scope):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if name != "EngineError":
        cls.bases = []
    if methods is not None:
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


EngineError = load_class(ROOT / "capability_engine/capability_engine/engine.py", "EngineError")


@pytest.mark.parametrize("stage", ["world", "home"])
@pytest.mark.parametrize("driver_error", [False, True])
def test_home_refusal_prevents_camera_detection(stage, driver_error):
    refused = type("Refused", (Exception,), {})
    arm_refused = type("ArmRefused", (Exception,), {})
    error_type = arm_refused if driver_error else refused
    take = SimpleNamespace(set_approach_world=Mock(), prepare_home=Mock())
    failure = take.set_approach_world if stage == "world" else take.prepare_home
    failure.side_effect = error_type("no home plan")
    camera = Mock()
    cls = load_class(
        ROOT / "capabilities/microwave_place/microwave_place/microwave_place_node.py",
        "Capability",
        {"_take_out"},
        EngineError=EngineError,
        ArmRefused=arm_refused,
        take_out=take,
        place_motion=SimpleNamespace(Refused=refused),
        rim_detection=camera,
        _site_words=lambda text, site: text,
    )
    node = cls()
    node.cfg = {"take_approach_world": "world", "take_home_joints": [0] * 7, "take_approach_speed_scale": 0.3}
    node.arm = Mock()
    with pytest.raises(EngineError) as error:
        node._take_out(Mock(), True, Mock(), Mock())
    assert error.value.error_code == "home_failed"
    assert error.value.recoverable
    if stage == "world":
        take.prepare_home.assert_not_called()
    camera.WristCamera.assert_not_called()


@pytest.fixture
def bridge():
    source = ROOT / "capability_engine/capability_engine/engine.py"
    run_cls = load_class(source, "_Run", threading=threading, time=time)
    cls = load_class(
        source,
        "CapabilityEngine",
        {"_ui_update_panel", "_ui_open_goal", "_ui_close"},
        Context=object,
        _Run=run_cls,
        json=json,
        RequestUserInput=SimpleNamespace(Goal=lambda **kw: SimpleNamespace(**kw)),
    )
    node = cls()
    node._assigned_token = bytes([1] * 16)
    node.get_logger = Mock(return_value=Mock())
    node.get_name = Mock(return_value="microwave_place")
    node._ui_request_client = Mock()
    node._ui_on_goal_response = Mock()
    node._ui_on_feedback = Mock()
    node._ui_update_client = Mock()
    return node, run_cls("retrieve", None)


def test_bridge_can_reopen_after_server_temporarily_unavailable(bridge):
    node, run = bridge
    ctx = SimpleNamespace(_run=run)
    node._ui_request_client.server_is_ready.return_value = False
    assert not node._ui_update_panel(ctx, panel_json={"message": "retry"}, objects=None, selected_instance_id=None)
    assert not run.ui_opening
    node._ui_request_client.server_is_ready.return_value = True
    assert node._ui_update_panel(ctx, panel_json={"message": "retry"}, objects=None, selected_instance_id=None)
    node._ui_request_client.send_goal_async.assert_called_once()


def test_closed_run_never_opens_a_new_panel(bridge):
    node, run = bridge
    node._ui_close(run)
    assert not node._ui_open_goal(run, node._assigned_token)
    node._ui_request_client.send_goal_async.assert_not_called()
