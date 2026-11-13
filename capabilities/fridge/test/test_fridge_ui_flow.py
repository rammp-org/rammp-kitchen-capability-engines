"""Offline UI flow tests for the fridge's buttons (UI_ACTIONS); the node's own methods run without ROS or hardware.

As the microwave's: the methods are loaded through `ast` (no ROS imports) with the shared capability_ui.RunUI and a
fake ctx that behaves as the engine's (one held UI event, a cancel flag checkpoint() raises on, update_user_panel()
False with no UI session); cancel_own_run is stubbed to set that flag, as the engine's own cancel does. `_execute` (the
direct action) is scripted. The waits run on a fake clock (the node's `time`). Nothing reaches the arm or the base."""

import ast
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / "capabilities/fridge/fridge/node.py"
CONSTANTS = ROOT / "capabilities/fridge/fridge/constants.py"
sys.path.insert(0, str(ROOT / "shared/capability_ui"))

from capability_ui import panels  # noqa: E402
from capability_ui.panels import DONE, STOP, STOPPED_MESSAGE, RunUI  # noqa: E402

UI_NAMES = {
    "UI_DRIVE",
    "UI_ACTIONS",
    "UI_TITLE",
    "UI_MOVING",
    "UI_DONE",
    "UI_NOT_MOVED",
    "UI_WHAT",
    "UI_ERRORS",
    "UI_ERROR_OTHER",
    "UI_MAYBE_HOLDING",
    "UI_STOPPED_OPEN",
    "UI_DROVE",
}
UI_METHODS = {
    "_ui_run",
    "_ui_close_move",
    "_ui_can_retry",
    "_ui_maybe_holding",
    "_ui_stopped_note",
    "_ui_failed",
    "_ui_answer",
    "_ui_ask_move",
    "_ui_ask_retry",
}


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
TIMEOUTS = load(CONSTANTS, {"UI_NOTICE_TIMEOUT_S", "UI_PROMPT_TIMEOUT_S"})
NOTICE_S, PROMPT_S = TIMEOUTS["UI_NOTICE_TIMEOUT_S"], TIMEOUTS["UI_PROMPT_TIMEOUT_S"]


class FakeCtx:
    """The engine's ctx as the UI sees it. `answers`: each panel with a button other than Stop takes the next one (a
    button_id tapped at once; None: never tapped)."""

    def __init__(self, answers=(), session=True):
        self.answers, self.session = list(answers), session
        self.sent = []
        self._event = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

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
        if self.cancelled():
            raise Cancelled()

    def messages(self):
        return [p["message"] for p in self.sent]


class Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


@pytest.fixture(autouse=True)
def fast_watch(monkeypatch):
    monkeypatch.setattr(panels, "POLL_S", 0.005)


def wait_until(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


OK = {"executed": True, "note": ""}


class Flow:
    """The node's UI methods on a bare Capability, `_execute` taking each of `outcomes` in turn: a result dict, an
    exception, or a callable(action_id, params) giving either. Calls are recorded in `.calls`."""

    def __init__(self, *outcomes, answers=("done",), mock=False, allow_execute=True, session=True):
        self.clock = Clock()
        self.push_open = types.SimpleNamespace(NEXT_MOVE=0.0)
        self.close_door = types.SimpleNamespace(LAST_CHAIR_BACK=None)
        scope = load(
            NODE,
            UI_NAMES,
            UI_METHODS,
            EngineError=EngineError,
            Cancelled=Cancelled,
            RunUI=RunUI,
            cancel_own_run=lambda engine: engine.ctx.cancel(),
            time=types.SimpleNamespace(monotonic=self.clock.monotonic, sleep=self.clock.sleep),
            constants=types.SimpleNamespace(UI_NOTICE_TIMEOUT_S=NOTICE_S, UI_PROMPT_TIMEOUT_S=PROMPT_S),
            push_open=self.push_open,
            close_door=self.close_door,
        )
        self.scope = scope
        self.node = node = scope["Capability"]()
        node.mock, node.cfg = mock, {"allow_execute": allow_execute}
        node.log = Mock()
        node.get_logger = lambda: node.log
        self.ctx = node.ctx = FakeCtx(answers, session)
        self.calls = []
        _with_outcomes(self, *outcomes)

    def press(self, ui_id):
        return self.node._ui_run(ui_id, self.ctx)


def pulled_asking(flow, move):
    """An _execute outcome: the open's pull is done and the push asks for the wheelchair moved `move` (m)."""

    def out(action_id, params):
        flow.push_open.NEXT_MOVE = move
        return {"executed": True, "note": "Pull done ..."}

    return out


def close_asking(flow, move):
    """An _execute outcome: the close refuses, having planned with the wheelchair moved `move` (m)."""

    def out(action_id, params):
        flow.close_door.LAST_CHAIR_BACK = move
        return EngineError("refused", "Move the WHEELCHAIR ... chair_moved true", recoverable=True)

    return out


def until_stopped(flow, drove=False):
    """An _execute outcome: moves until cancelled, as self.arm and self.base do (Stop is tapped once it runs)."""

    def out(action_id, params):
        flow.node._drove = drove
        flow.ctx.tap("stop")
        assert wait_until(flow.ctx.cancelled)
        return Cancelled()

    return out


# -- success -------------------------------------------------------------------------------------------------------


def test_go_to_fridge_moving_panel_has_stop_then_done():
    f = Flow(OK)
    assert f.press("go_to_start_ui") == OK
    assert f.calls == [("go_to_start", {"execute": True, "speed": 0.7})]
    moving, done = f.ctx.sent
    assert moving == {"title": "Go to fridge", "message": "Moving the arm to the fridge.", "buttons": [STOP]}
    assert done["message"] == "The arm is in front of the fridge." and done["buttons"] == [DONE]
    assert f.clock.t < 1.0  # ended at Done


def test_open_with_no_wheelchair_move():
    f = Flow(OK)
    f.press("open_door_ui")
    assert f.ctx.messages() == ["Opening the fridge door. The arm is moving.", "Fridge door opened."]


def test_a_done_notice_waits_at_most_its_timeout():
    f = Flow(OK, answers=[None])
    assert f.press("close_door_ui") == OK
    assert NOTICE_S <= f.clock.t < NOTICE_S + 1.0


def test_mock_says_no_arm_motion():
    f = Flow({"executed": False, "mock": True}, mock=True)
    f.press("open_door_ui")
    assert f.ctx.messages()[-1] == "Mock: fridge door opened. No arm motion."


# -- Stop while moving -----------------------------------------------------------------------------------------------


def test_stop_during_the_open_says_the_hand_may_be_on_the_handle():
    f = Flow()
    f = _with_outcomes(f, until_stopped(f))
    with pytest.raises(Cancelled):
        f.press("open_door_ui")
    assert f.ctx.cancelled()
    stopped = f.ctx.sent[-1]
    assert stopped["title"] == "Open fridge: stopped"
    assert stopped["message"].startswith(STOPPED_MESSAGE)
    assert "If the hand is on the handle, let go of it" in stopped["message"]
    assert stopped["buttons"] == [DONE]
    assert len(f.calls) == 1


def test_stop_during_go_to_start_has_no_handle_note():
    f = Flow()
    f = _with_outcomes(f, until_stopped(f))
    with pytest.raises(Cancelled):
        f.press("go_to_start_ui")
    assert f.ctx.sent[-1]["message"] == STOPPED_MESSAGE


def test_stop_after_the_wheelchair_drove_says_so():
    f = Flow()
    f = _with_outcomes(f, until_stopped(f, drove=True))
    with pytest.raises(Cancelled):
        f.press("close_door_drive_ui")
    assert "The wheelchair has already moved" in f.ctx.sent[-1]["message"]
    assert "handle" not in f.ctx.sent[-1]["message"]  # the close doesn't grasp


def test_a_stop_that_ends_as_an_engine_error_is_not_retried_or_reported_twice():
    f = Flow()

    def out(action_id, params):  # a motion that returns False on a cancel: _execute raises "stopped"
        f.ctx.tap("stop")
        assert wait_until(f.ctx.cancelled)
        return EngineError("stopped", "the open stopped short -- the arm holds where it is", True)

    f = _with_outcomes(f, out)
    with pytest.raises(EngineError, match="stopped short"):
        f.press("open_door_ui")
    assert len(f.calls) == 1
    assert [p["title"] for p in f.ctx.sent] == ["Open fridge", "Open fridge: stopped"]


def test_an_orchestrator_cancel_shows_no_panel_after_the_motion():
    f = Flow()

    def out(action_id, params):
        f.ctx.cancel()
        return Cancelled()

    f = _with_outcomes(f, out)
    with pytest.raises(Cancelled):
        f.press("open_door_ui")
    assert [p["buttons"] for p in f.ctx.sent] == [[STOP]]


# -- the wheelchair question ----------------------------------------------------------------------------------------


def test_open_asks_for_the_wheelchair_move_then_pushes_on_continue():
    f = Flow()
    f = _with_outcomes(f, pulled_asking(f, 0.05), OK)
    f.ctx.answers = ["continue", "done"]
    assert f.press("open_door_ui") == OK
    assert f.calls == [
        ("open", {"execute": True, "speed": 0.7}),
        ("open", {"execute": True, "speed": 0.7, "chair_moved": True}),
    ]
    ask = f.ctx.sent[1]
    assert "straight back, away from the fridge, about 5 cm" in ask["message"]
    assert "Continue to push the door open" in ask["message"]
    assert ask["buttons"] == [{"button_id": "continue", "label": "Continue"}, STOP]
    assert f.ctx.messages()[-1] == "Fridge door opened."


def test_close_asks_for_its_planned_move_then_closes():
    f = Flow()
    f = _with_outcomes(f, close_asking(f, -0.08), OK)
    f.ctx.answers = ["continue", "done"]
    assert f.press("close_door_ui") == OK
    assert f.calls == [("close", {"execute": True}), ("close", {"execute": True, "chair_moved": True})]
    assert "straight forward, toward the fridge, about 8 cm" in f.ctx.sent[1]["message"]


def test_stop_at_the_wheelchair_question_cancels_the_run():
    f = Flow()
    f = _with_outcomes(f, pulled_asking(f, 0.05))
    f.ctx.answers = ["stop"]
    with pytest.raises(Cancelled):
        f.press("open_door_ui")
    assert f.ctx.cancelled() and len(f.calls) == 1


def test_no_answer_to_the_wheelchair_question_fails_no_answer():
    f = Flow()
    f = _with_outcomes(f, close_asking(f, 0.1))
    f.ctx.answers = [None]
    with pytest.raises(EngineError) as raised:
        f.press("close_door_ui")
    assert raised.value.error_code == "no_answer"
    assert PROMPT_S <= f.clock.t < PROMPT_S + 1.0


def test_the_driving_buttons_never_ask_for_a_move():
    f = Flow()
    f = _with_outcomes(f, close_asking(f, 0.12), OK)
    f.ctx.answers = ["retry", "done"]
    f.press("close_door_drive_ui")
    assert f.calls[0] == ("close", {"execute": True, "drive_base": True, "base_speed": 0.03})
    assert "chair_moved" not in f.calls[1][1]
    assert not any("Move your wheelchair" in m for m in f.ctx.messages())


# -- refusals and failures ------------------------------------------------------------------------------------------


def test_a_refusal_offers_the_same_button_again():
    f = Flow(EngineError("refused", "NO DETECTION", True), OK, answers=["retry", "done"])
    assert f.press("open_door_ui") == OK
    retry = f.ctx.sent[1]
    assert retry["title"] == "Couldn't open the fridge"
    assert retry["message"].endswith("Press Open Fridge Door to try again.")
    assert retry["buttons"] == [{"button_id": "retry", "label": "Open Fridge Door"}, STOP]
    assert "NO DETECTION" not in retry["message"]  # the log has it, not the UI
    assert len(f.calls) == 2


def test_stop_instead_of_retry_cancels_the_run():
    f = Flow(EngineError("refused", "x", True), answers=["stop"])
    with pytest.raises(Cancelled):
        f.press("go_to_start_ui")
    assert len(f.calls) == 1


def test_a_failure_that_may_hold_the_handle_is_not_retried():
    error = EngineError("stopped", "the open stopped short", True)
    f = Flow(error)
    with pytest.raises(EngineError) as raised:
        f.press("open_door_ui")
    assert raised.value is error and len(f.calls) == 1
    failed = f.ctx.sent[-1]
    assert failed["title"] == "Couldn't open the fridge" and failed["buttons"] == [DONE]
    assert "stopped partway" in failed["message"] and "still be holding the handle" in failed["message"]


def test_a_bug_shows_the_generic_failure_and_is_reraised():
    bug = KeyError("door_deg")
    f = Flow(bug)
    with pytest.raises(KeyError) as raised:
        f.press("close_door_ui")
    assert raised.value is bug
    assert f.ctx.messages()[-1] == f.scope["UI_ERROR_OTHER"]


def test_execution_off_says_so_and_runs_nothing():
    f = Flow()
    f.node.cfg["allow_execute"] = False
    with pytest.raises(EngineError) as raised:
        f.press("open_door_ui")
    assert raised.value.error_code == "execution_disabled" and raised.value.recoverable and f.calls == []
    assert len(f.ctx.sent) == 1
    assert "allow_execute is false" in f.ctx.sent[-1]["message"] and f.ctx.sent[-1]["buttons"] == [DONE]
    assert "holding the handle" not in f.ctx.sent[-1]["message"]  # nothing moved


def test_no_ui_session():
    """Standalone: nothing waited for; the flow ends as the direct action would."""
    f = Flow(OK, session=False)
    assert f.press("go_to_start_ui") == OK
    f = Flow(session=False)
    f = _with_outcomes(f, pulled_asking(f, 0.05))
    assert f.press("open_door_ui")["executed"] and len(f.calls) == 1  # its note says the move
    f = Flow(session=False)
    f = _with_outcomes(f, close_asking(f, 0.1))
    with pytest.raises(EngineError) as raised:
        f.press("close_door_ui")
    assert raised.value.error_code == "refused"
    f = Flow(EngineError("refused", "x", True), session=False)
    with pytest.raises(EngineError):
        f.press("open_door_ui")
    assert f.clock.t == 0.0 and len(f.calls) == 1


def _with_outcomes(flow, *outcomes):
    """`flow` with its _execute taking `outcomes` (built from the flow, so they can reach its stubs)."""
    queue = list(outcomes)
    node = flow.node
    flow.calls.clear()

    def _execute(action_id, params, ctx):
        flow.calls.append((action_id, params))
        node._drove = False  # as the real _execute: per attempt
        out = queue.pop(0)
        if callable(out) and not isinstance(out, BaseException):
            out = out(action_id, params)
        if isinstance(out, BaseException):
            raise out
        return out

    node._execute = _execute
    return flow


@pytest.fixture(autouse=True)
def _no_ui_wrap(monkeypatch):
    """Messages are compared whole here; capability_ui's own tests cover the UI's line wrapping."""
    from capability_ui import panels as ui_panels

    monkeypatch.setattr(ui_panels, "WRAP_CHARS", 10**6)
