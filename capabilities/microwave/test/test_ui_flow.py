"""Offline UI flow tests for the microwave's one-press open / close; production methods run without ROS or hardware.

The node's methods are loaded through `ast` (no ROS imports) with the shared capability_ui.RunUI and a fake ctx that
behaves as the engine's: one held UI event, a cancel flag that checkpoint() raises on, update_user_panel() False with
no UI session. cancel_own_run is stubbed to set that flag, as the engine's own cancel does."""

import ast
import importlib.util
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / "capabilities/microwave/microwave/node.py"
sys.path.insert(0, str(ROOT / "shared/capability_ui"))

from capability_ui import panels  # noqa: E402
from capability_ui.panels import DONE, NOTICE_TIMEOUT_S, PROMPT_TIMEOUT_S, STOP, STOPPED_MESSAGE, RunUI  # noqa: E402


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


def engine_classes():
    tree = ast.parse((ROOT / "capability_engine/capability_engine/engine.py").read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("EngineError", "Cancelled")]
    scope = {}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "engine.py", "exec"), scope)
    return scope["EngineError"], scope["Cancelled"]


EngineError, Cancelled = engine_classes()


def load_ui_text():
    """The node's panel wording (microwave/ui_text.py): no ROS, only capability_ui."""
    spec = importlib.util.spec_from_file_location(
        "microwave_ui_text", ROOT / "capabilities/microwave/microwave/ui_text.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UI_TEXT = load_ui_text()


class FakeCtx:
    """The engine's ctx as the UI sees it. `answers`: each panel with a button other than Stop takes the next one (a
    button_id tapped at once; None: never tapped)."""

    def __init__(self, answers=(), session=True):
        self.answers, self.session = list(answers), session
        self.sent = []
        self._event = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self.checkpoint_calls = 0

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        self.sent.append(panel_json)
        if self.session and any(b["button_id"] != "stop" for b in panel_json["buttons"]):
            answer = self.answers.pop(0) if self.answers else None
            if answer:
                self.tap(answer)
        return self.session

    def poll_ui_event(self):
        with self._lock:
            event, self._event = self._event, None
            return event

    def tap(self, button_id):
        with self._lock:
            self._event = {"type": "button", "button_id": button_id}

    def cancel(self):
        self._cancelled.set()

    def cancelled(self):
        return self._cancelled.is_set()

    def checkpoint(self):
        self.checkpoint_calls += 1
        if self.cancelled():
            raise Cancelled()


class Clock:
    """Virtual time for the waits; sleep() advances it."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


@pytest.fixture(autouse=True)
def fast_watch(monkeypatch):
    monkeypatch.setattr(panels, "POLL_S", 0.005)


@pytest.fixture
def flow():
    clock = Clock()

    class UI(RunUI):  # RunUI on the virtual clock
        def __init__(self, *args, **kwargs):
            super().__init__(*args, clock=clock, sleep=clock.sleep, **kwargs)

    scope = load(
        NODE,
        {"UI_ACTIONS", "UI_TITLE", "UI_DONE"},
        {"_ui_run", "_ui_stopped_note"},
        EngineError=EngineError,
        Cancelled=Cancelled,
        ui_text=UI_TEXT,
        RunUI=UI,
        cancel_own_run=lambda engine: engine.ctx.cancel(),
    )
    node = scope["Capability"]()
    node.mock = False
    node.cfg = {"allow_execute": True}
    node.get_logger = Mock(return_value=Mock())
    node.execute = Mock(return_value={"executed": True})
    node.ctx = FakeCtx(["done"])
    return node, node.ctx, scope, clock


def wait_until(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def until_stopped(ctx, holding=None, node=None):
    """An execute() that moves until cancelled, as self.arm does (Stop is tapped once it runs)."""

    def run(action_id, params, _ctx):
        if node is not None:
            node._holding_handle = holding
        ctx.tap("stop")
        assert wait_until(ctx.cancelled)
        raise Cancelled()

    return run


def test_one_press_open_runs_the_open_executing(flow):
    node, ctx, _, _ = flow
    assert node._ui_run("open_door_ui", ctx) == {"executed": True}
    node.execute.assert_called_once_with("open", {"execute": True, "push_open": True}, ctx)
    moving = ctx.sent[0]
    assert moving["message"] == "Open Microwave Door: the arm is moving."
    assert moving["buttons"] == [STOP]
    assert ctx.sent[-1]["message"] == "Microwave door opened."
    assert ctx.sent[-1]["buttons"] == [DONE]


def test_one_press_close_uses_the_settings_that_closed_it(flow):
    node, ctx, scope, _ = flow
    node._ui_run("close_door_ui", ctx)
    action, params, _ = node.execute.call_args.args
    assert action == "close"
    assert params == scope["UI_ACTIONS"]["close_door_ui"][2]
    assert params["execute"] and params["face_door"] and params["route_over"]


def test_stop_while_opening_with_the_hand_on_the_handle(flow):
    node, ctx, _, _ = flow
    node.execute.side_effect = until_stopped(ctx, holding=True, node=node)
    with pytest.raises(Cancelled):
        node._ui_run("open_door_ui", ctx)
    stopped = ctx.sent[-1]
    assert stopped["title"] == "Open microwave: stopped"
    assert stopped["message"].startswith(STOPPED_MESSAGE)
    assert "The hand is still on the handle: let go of it" in stopped["message"]
    assert stopped["buttons"] == [DONE]
    assert node.execute.call_count == 1


def test_stop_before_the_grasp_says_nothing_about_the_handle(flow):
    node, ctx, _, _ = flow
    node.execute.side_effect = until_stopped(ctx, holding=False, node=node)
    with pytest.raises(Cancelled):
        node._ui_run("open_door_ui", ctx)
    assert ctx.sent[-1]["message"] == STOPPED_MESSAGE


def test_stop_when_holding_is_unknown_says_to_let_go_if_holding(flow):
    node, ctx, _, _ = flow
    node.execute.side_effect = until_stopped(ctx)  # never says (holding stays None)
    with pytest.raises(Cancelled):
        node._ui_run("open_door_ui", ctx)
    assert ctx.sent[-1]["message"].endswith("If the hand is on the handle, let go of it before anything else.")


def test_stop_during_the_close_has_no_handle_note(flow):
    node, ctx, _, _ = flow
    node.execute.side_effect = until_stopped(ctx)
    with pytest.raises(Cancelled):
        node._ui_run("close_door_ui", ctx)
    assert ctx.sent[-1]["message"] == STOPPED_MESSAGE
    assert ctx.sent[-1]["title"] == "Close microwave: stopped"


def test_a_stop_that_ends_as_an_engine_error_is_not_retried_or_reported_twice(flow):
    node, ctx, _, _ = flow

    def run(action_id, params, _ctx):  # a legacy move returns False on a cancel: execute() raises "stopped"
        node._holding_handle = False
        ctx.tap("stop")
        assert wait_until(ctx.cancelled)
        raise EngineError("stopped", "the open stopped short -- the arm holds where it is", True)

    node.execute.side_effect = run
    with pytest.raises(EngineError, match="stopped short"):
        node._ui_run("open_door_ui", ctx)
    assert node.execute.call_count == 1
    assert [p["title"] for p in ctx.sent] == ["Open microwave", "Open microwave: stopped"]


def test_orchestrator_cancel_shows_no_stopped_notice(flow):
    node, ctx, _, _ = flow

    def run(action_id, params, _ctx):
        ctx.cancel()
        raise Cancelled()

    node.execute.side_effect = run
    with pytest.raises(Cancelled):
        node._ui_run("open_door_ui", ctx)
    assert [p["buttons"] for p in ctx.sent] == [[STOP]]


def test_recoverable_refusal_waits_for_explicit_retry_then_runs_again(flow):
    node, ctx, _, _ = flow
    ctx.answers = ["retry", "done"]
    node.execute.side_effect = [EngineError("refused", "no grasp passes the checks", True), {"executed": True}]
    assert node._ui_run("open_door_ui", ctx)["executed"]
    assert node.execute.call_count == 2
    retry = ctx.sent[1]
    assert retry["title"] == "Unable to open microwave"
    assert retry["message"] == (
        "No grasp passes the checks. The arm is holding still. Press Open Microwave Door to try again."
    )
    assert retry["buttons"] == [{"button_id": "retry", "label": "Open Microwave Door"}, STOP]
    assert ctx.sent[-1]["message"] == "Microwave door opened."


def test_stop_at_the_retry_question_cancels_and_does_not_move_again(flow):
    node, ctx, _, _ = flow
    ctx.answers = ["stop"]
    node.execute.side_effect = EngineError("refused", "no plan", True)
    with pytest.raises(Cancelled):
        node._ui_run("open_door_ui", ctx)
    assert node.execute.call_count == 1
    assert ctx.cancelled()


def test_no_answer_to_the_retry_returns_the_refusal(flow):
    node, ctx, _, clock = flow
    ctx.answers = [None]
    error = EngineError("refused", "no plan", True)
    node.execute.side_effect = error
    with pytest.raises(EngineError) as raised:
        node._ui_run("open_door_ui", ctx)
    assert raised.value is error and raised.value.recoverable
    assert PROMPT_TIMEOUT_S <= clock.t < PROMPT_TIMEOUT_S + 1.0
    assert node.execute.call_count == 1


def test_holding_the_handle_is_reported_and_never_retried(flow):
    node, ctx, _, _ = flow
    node.execute.side_effect = EngineError("stopped", "the open stopped short -- still holding the handle", True)
    with pytest.raises(EngineError, match="holding the handle"):
        node._ui_run("open_door_ui", ctx)
    assert node.execute.call_count == 1
    last = ctx.sent[-1]
    assert last["title"] == "Open microwave: failed"
    assert "still holding the handle" in last["message"] and "open the gripper" in last["message"]
    assert "stopped before the door was open" in last["message"]
    assert last["buttons"] == [DONE]


def test_unrecoverable_failure_is_reported_without_retry(flow):
    node, ctx, _, _ = flow
    node.execute.side_effect = EngineError("bad_params", "bad speed", False)
    with pytest.raises(EngineError, match="bad speed"):
        node._ui_run("close_door_ui", ctx)
    assert node.execute.call_count == 1
    assert "Bad speed." in ctx.sent[-1]["message"]


def test_cancel_while_waiting_to_retry_does_not_move_again(flow):
    node, ctx, _, clock = flow
    ctx.answers = [None]
    node.execute.side_effect = EngineError("refused", "no plan", True)
    sleep = clock.sleep

    def cancel_later(dt):  # the orchestrator cancels while the question is up
        sleep(dt)
        if clock.t > 5.0:
            ctx.cancel()

    clock.sleep = cancel_later
    with pytest.raises(Cancelled):
        node._ui_run("open_door_ui", ctx)
    assert node.execute.call_count == 1


def test_execution_disabled_never_moves(flow):
    node, ctx, _, _ = flow
    node.cfg["allow_execute"] = False
    with pytest.raises(EngineError, match="disabled") as raised:
        node._ui_run("close_door_ui", ctx)
    assert raised.value.recoverable and raised.value.error_code == "execution_disabled"
    node.execute.assert_not_called()
    assert ctx.sent[-1]["title"] == "Close microwave: not started"
    assert "arm motion off" in ctx.sent[-1]["message"] and "allow_execute" not in ctx.sent[-1]["message"]
    assert ctx.sent[-1]["buttons"] == [DONE]


def test_no_ui_session_returns_the_recoverable_error(flow):
    node, ctx, _, _ = flow
    ctx.session = False
    node.execute.side_effect = EngineError("refused", "no plan", True)
    with pytest.raises(EngineError) as error:
        node._ui_run("open_door_ui", ctx)
    assert error.value.recoverable
    assert node.execute.call_count == 1


def test_mock_says_nothing_moved(flow):
    node, ctx, _, _ = flow
    node.mock = True
    node.execute.return_value = {"executed": False, "mock": True}
    node._ui_run("open_door_ui", ctx)
    assert "No arm motion" in ctx.sent[-1]["message"]


def test_done_notice_is_bounded(flow):
    node, ctx, _, clock = flow
    ctx.answers = [None]  # Done never tapped
    assert node._ui_run("open_door_ui", ctx) == {"executed": True}
    assert NOTICE_TIMEOUT_S <= clock.t < NOTICE_TIMEOUT_S + 1.0


@pytest.fixture(autouse=True)
def _no_ui_wrap(monkeypatch):
    """Messages are compared whole here; capability_ui's own tests cover the UI's line wrapping."""
    from capability_ui import panels as ui_panels

    monkeypatch.setattr(ui_panels, "WRAP_CHARS", 10**6)


def test_a_refusal_shows_its_reason_and_what_to_do(flow):
    node, ctx, _, _ = flow
    ctx.answers = [None]
    node.execute.side_effect = EngineError(
        "refused", "NO DETECTION (1) after 3 retries -- the arm holds where it is", True
    )
    with pytest.raises(EngineError):
        node._ui_run("open_door_ui", ctx)
    message = ctx.sent[1]["message"].replace(f"</>\n<{panels.UI_TEXT_STYLE}>", " ")
    assert message == (
        "The wrist camera couldn't see the microwave handle (3 tries). The arm is holding still. Make sure the handle "
        "is in view of the wrist camera and nothing covers it. Press Open Microwave Door to try again."
    )


def test_a_close_that_ended_false_names_what_its_run_said(flow):
    node, ctx, _, _ = flow
    ctx.answers = [None]

    def run(action_id, params, _ctx):
        node._said.append("need /data/microwave/door.json -- refusing.")
        raise EngineError("stopped", "the close stopped short -- the arm holds where it is", True)

    node.execute.side_effect = run
    with pytest.raises(EngineError):
        node._ui_run("close_door_ui", ctx)
    message = ctx.sent[1]["message"].replace(f"</>\n<{panels.UI_TEXT_STYLE}>", " ")
    assert "no record of the microwave door being opened" in message
    assert "Open the door with Open Microwave Door first" in message


def test_the_arms_own_error_is_named_only_if_this_run_set_it(flow):
    node, ctx, _, _ = flow
    node.arm = Mock(last_error="  RammpArm: go_to: arm is E-STOPPED -- clear it first -- nothing sent.")
    ctx.answers = [None]

    def run(action_id, params, _ctx):
        node.arm.last_error = "swing: STOPPED (-11 halted: unexpected contact)"
        raise EngineError("stopped", "the open stopped short -- the arm holds where it is", True)

    node.execute.side_effect = run
    with pytest.raises(EngineError):
        node._ui_run("open_door_ui", ctx)
    message = ctx.sent[1]["message"].replace(f"</>\n<{panels.UI_TEXT_STYLE}>", " ")
    assert "unexpected push or pull" in message and "emergency stop" not in message


def test_an_arm_error_that_isnt_an_engine_error_is_shown_then_raised(flow):
    node, ctx, _, _ = flow

    class EStopped(Exception):
        pass

    node.execute.side_effect = EStopped("arm is E-STOPPED -- clear it first")
    with pytest.raises(EStopped):
        node._ui_run("close_door_ui", ctx)
    last = ctx.sent[-1]
    assert last["title"] == "Close microwave: failed" and last["buttons"] == [DONE]
    assert "Release the emergency stop" in last["message"].replace(f"</>\n<{panels.UI_TEXT_STYLE}>", " ")
