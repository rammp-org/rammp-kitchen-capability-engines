"""Offline UI flow tests for the microwave's one-press open; production methods run without ROS or hardware."""

import ast
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / "capabilities/microwave/microwave/node.py"


def load(path, names, methods=None, **scope):
    """The module-level assignments `names` and (methods) the Capability class with only those methods, no bases."""
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in names for t in n.targets)]
    if methods is not None:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Capability")
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        body.append(cls)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope


def engine_error():
    tree = ast.parse((ROOT / "capability_engine/capability_engine/engine.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineError")
    scope = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "engine.py", "exec"), scope)
    return scope["EngineError"]


EngineError = engine_error()


@pytest.fixture
def flow(monkeypatch):
    scope = load(
        NODE,
        {"UI_ACTIONS", "UI_TITLE", "UI_DONE"},
        {"_ui_run", "_ui_panel", "_ui_notice", "_wait_for_button", "_panel_open"},
        EngineError=EngineError,
        time=time,
    )
    node = scope["Capability"]()
    scope["real_panel_open"] = scope["Capability"].__dict__["_panel_open"]
    scope["Capability"]._panel_open = staticmethod(lambda ctx: False)  # as if the bridge dropped it: re-send
    node.mock = False
    node.cfg = {"allow_execute": True}
    node.get_logger = Mock(return_value=Mock())
    node.execute = Mock(return_value={"executed": True})
    ctx = Mock()
    ctx.update_user_panel.return_value = True
    ctx.poll_ui_event.return_value = {"type": "button", "button_id": "done"}
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return node, ctx, scope


def panels(ctx):
    return [c.kwargs["panel_json"] for c in ctx.update_user_panel.call_args_list]


def test_one_press_open_runs_the_open_executing(flow):
    node, ctx, _ = flow
    assert node._ui_run("open_door_ui", ctx) == {"executed": True}
    node.execute.assert_called_once_with("open", {"execute": True}, ctx)
    assert panels(ctx)[0]["buttons"] == []
    assert panels(ctx)[-1]["message"] == "Microwave door opened."


def test_recoverable_refusal_waits_for_explicit_retry_then_runs_again(flow):
    node, ctx, _ = flow
    node.execute.side_effect = [EngineError("refused", "no grasp passes the checks", True), {"executed": True}]
    ctx.poll_ui_event.side_effect = [
        None,
        {"type": "button", "button_id": "unrelated"},
        {"type": "button", "button_id": "retry"},
        {"type": "button", "button_id": "done"},
    ]
    assert node._ui_run("open_door_ui", ctx)["executed"]
    assert node.execute.call_count == 2
    retry = panels(ctx)[1]
    assert "no grasp passes the checks" in retry["message"]
    assert retry["buttons"] == [{"button_id": "retry", "label": "Open Microwave Door"}]


def test_holding_the_handle_is_reported_and_never_retried(flow):
    node, ctx, _ = flow
    node.execute.side_effect = EngineError("stopped", "the open stopped short -- still holding the handle", True)
    with pytest.raises(EngineError, match="holding the handle"):
        node._ui_run("open_door_ui", ctx)
    assert node.execute.call_count == 1
    last = panels(ctx)[-1]
    assert "let go of it" in last["message"]
    assert last["buttons"] == [{"button_id": "done", "label": "Done"}]


def test_unrecoverable_failure_is_reported_without_retry(flow):
    node, ctx, _ = flow
    node.execute.side_effect = EngineError("bad_params", "bad speed", False)
    with pytest.raises(EngineError, match="bad speed"):
        node._ui_run("open_door_ui", ctx)
    assert node.execute.call_count == 1
    assert "bad speed" in panels(ctx)[-1]["message"]


def test_cancel_while_waiting_to_retry_does_not_move_again(flow):
    node, ctx, _ = flow
    node.execute.side_effect = EngineError("refused", "no plan", True)
    ctx.poll_ui_event.return_value = None
    ctx.checkpoint.side_effect = [None, RuntimeError("cancelled")]
    with pytest.raises(RuntimeError, match="cancelled"):
        node._ui_run("open_door_ui", ctx)
    assert node.execute.call_count == 1


def test_execution_disabled_never_moves(flow):
    node, ctx, _ = flow
    node.cfg["allow_execute"] = False
    with pytest.raises(EngineError, match="disabled"):
        node._ui_run("open_door_ui", ctx)
    node.execute.assert_not_called()


def test_no_ui_session_returns_the_recoverable_error(flow):
    node, ctx, _ = flow
    ctx.update_user_panel.return_value = False
    node.execute.side_effect = EngineError("refused", "no plan", True)
    with pytest.raises(EngineError) as error:
        node._ui_run("open_door_ui", ctx)
    assert error.value.recoverable
    ctx.poll_ui_event.assert_not_called()


def test_mock_says_nothing_moved(flow):
    node, ctx, _ = flow
    node.mock = True
    node.execute.return_value = {"executed": False, "mock": True}
    node._ui_run("open_door_ui", ctx)
    assert "No arm motion" in panels(ctx)[-1]["message"]


def test_open_panel_is_not_resent(flow, monkeypatch):
    node, ctx, scope = flow
    cap = scope["Capability"]
    cap._panel_open = scope["real_panel_open"]  # the real check: the run's UI goal is open
    ctx._run = type("Run", (), {"ui_goal_handle": object(), "ui_opening": False})()
    clock = iter(range(0, 100, 2))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    ctx.poll_ui_event.side_effect = [None, None, None, {"type": "button", "button_id": "done"}]
    node._wait_for_button(ctx, "done", {"title": "t"})
    ctx.update_user_panel.assert_not_called()


def test_dropped_panel_is_resent(flow, monkeypatch):
    node, ctx, scope = flow
    cap = scope["Capability"]
    cap._panel_open = scope["real_panel_open"]
    ctx._run = type("Run", (), {"ui_goal_handle": None, "ui_opening": False})()
    clock = iter(range(0, 100, 2))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    ctx.poll_ui_event.side_effect = [None, None, {"type": "button", "button_id": "done"}]
    node._wait_for_button(ctx, "done", {"title": "t"})
    assert ctx.update_user_panel.call_count >= 1
