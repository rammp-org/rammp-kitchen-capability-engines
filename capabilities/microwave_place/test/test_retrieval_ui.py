"""Offline UI flow tests; production methods run without ROS or hardware."""

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


@pytest.fixture
def flow(monkeypatch):
    cls = load_class(
        ROOT / "capabilities/microwave_place/microwave_place/microwave_place_node.py",
        "Capability",
        {"_retrieve", "_retrieval_panel", "_retrieval_notice", "_wait_for_button", "_take_out"},
        EngineError=EngineError,
        time=time,
    )
    node = cls()
    node.mock = False
    node.cfg = {"allow_execute": True}
    node.get_logger = Mock(return_value=Mock())
    node.execute = Mock(return_value={"executed": True})
    ctx = Mock()
    ctx.update_user_panel.return_value = True
    ctx.poll_ui_event.return_value = {"type": "button", "button_id": "done"}
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return node, ctx


def test_single_click_runs_whole_take_out(flow):
    node, ctx = flow
    assert node._retrieve(ctx) == {"executed": True}
    node.execute.assert_called_once_with("take_out", {"execute": True}, ctx)
    panels = [c.kwargs["panel_json"] for c in ctx.update_user_panel.call_args_list]
    assert panels[0]["buttons"] == []
    assert panels[-1]["message"] == "Tupperware retrieved."


def test_home_failure_waits_for_explicit_retry_then_runs_again(flow):
    node, ctx = flow
    node.execute.side_effect = [EngineError("home_failed", "PLANNING_FAILED", True), {"executed": True}]
    ctx.poll_ui_event.side_effect = [
        None,
        {"type": "object_tapped", "instance_id": "retrieve"},
        {"type": "button", "button_id": "unrelated"},
        {"type": "button", "button_id": "retrieve"},
        {"type": "button", "button_id": "done"},
    ]
    assert node._retrieve(ctx)["executed"]
    assert node.execute.call_count == 2
    panel = ctx.update_user_panel.call_args_list[1].kwargs["panel_json"]
    assert "PLANNING_FAILED" in panel["message"]
    assert panel["buttons"] == [{"button_id": "retrieve", "label": "Retrieve Tupperware from Microwave"}]


def test_cancel_during_retry_does_not_restart_motion(flow):
    node, ctx = flow
    node.execute.side_effect = EngineError("home_failed", "no plan", True)
    ctx.checkpoint.side_effect = [None, RuntimeError("cancelled")]
    with pytest.raises(RuntimeError, match="cancelled"):
        node._retrieve(ctx)
    assert node.execute.call_count == 1


def test_later_failure_is_reported_without_automatic_retry(flow):
    node, ctx = flow
    node.execute.side_effect = EngineError("refused", "grasp failed", True)
    with pytest.raises(EngineError, match="grasp failed"):
        node._retrieve(ctx)
    assert node.execute.call_count == 1
    assert "grasp failed" in ctx.update_user_panel.call_args.kwargs["panel_json"]["message"]


def test_execution_disabled_never_starts_take_out(flow):
    node, ctx = flow
    node.cfg["allow_execute"] = False
    with pytest.raises(EngineError, match="disabled"):
        node._retrieve(ctx)
    node.execute.assert_not_called()


def test_no_ui_session_home_failure_returns_recoverable_error(flow):
    node, ctx = flow
    ctx.update_user_panel.return_value = False
    node.execute.side_effect = EngineError("home_failed", "no plan", True)
    with pytest.raises(EngineError) as error:
        node._retrieve(ctx)
    assert error.value.recoverable
    ctx.poll_ui_event.assert_not_called()


def test_home_refusal_prevents_camera_detection():
    refused = type("Refused", (Exception,), {})
    take = SimpleNamespace(set_approach_world=Mock(), prepare_home=Mock(side_effect=refused("no home plan")))
    camera = Mock()
    cls = load_class(
        ROOT / "capabilities/microwave_place/microwave_place/microwave_place_node.py",
        "Capability",
        {"_take_out"},
        EngineError=EngineError,
        take_out=take,
        place_motion=SimpleNamespace(Refused=refused),
        rim_detection=camera,
    )
    node = cls()
    node.cfg = {"take_approach_world": "world", "take_home_joints": [0] * 7, "take_approach_speed_scale": 0.3}
    node.arm = Mock()
    with pytest.raises(EngineError) as error:
        node._take_out(Mock(), True, Mock(), Mock())
    assert error.value.error_code == "home_failed"
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
