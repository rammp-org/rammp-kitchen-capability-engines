"""Offline UI flow tests for the five one-button actions (press_start, ...): node.py's own methods, loaded with `ast`
without ROS, the press itself faked, on the shared capability_ui.RunUI against a fake ctx that cancels as the engine's
does. Mock tests: nothing here is robot validation."""

import ast
import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "capabilities/button_press/button_press"
sys.path.insert(0, str(ROOT / "shared/capability_ui"))

from capability_ui import panels  # noqa: E402
from capability_ui.panels import RunUI  # noqa: E402


def classes(path, names):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    scope = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope


EngineError = classes(ROOT / "capability_engine/capability_engine/engine.py", {"EngineError"})["EngineError"]
_errors = classes(PKG / "__init__.py", {"Abort", "MotionFailed"})
Abort, MotionFailed = _errors["Abort"], _errors["MotionFailed"]
NAMES = ast.literal_eval(
    next(
        n.value
        for n in ast.parse((PKG / "dome_pattern.py").read_text()).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "NAMES"
    )
)
_spec = importlib.util.spec_from_file_location("bp_ui_panel", PKG / "ui_panel.py")
ui_panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ui_panel)


class Cancelled(Exception):
    pass


class ArmError(Exception):
    pass


class Action:
    def __init__(self, action_id, label, params_schema=None):
        self.action_id, self.label = action_id, label


class Base:
    def availability(self, instance_id, world_state):
        return list(self.actions)


def load():
    """node.py's module constants, failure(), and the Capability class with only its UI methods (base: Base)."""
    tree = ast.parse((PKG / "node.py").read_text())
    consts = {"MAX_PRESSES", "AT_PANEL", "AT_PANEL_NOTE", "UI_ACTIONS", "UI_MAX_PRESSES", "MOCK_RESULT"}
    body = [
        n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in consts for t in n.targets)
    ]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "failure"]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Capability")
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    keep = {"_ui_run", "_ui_presses", "availability", "execute"}
    cls.body = [
        n
        for n in cls.body
        if (isinstance(n, ast.FunctionDef) and n.name in keep)
        or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "actions")
    ]
    body.append(cls)
    scope = {
        "Action": Action,
        "Base": Base,
        "Abort": Abort,
        "MotionFailed": MotionFailed,
        "ArmError": ArmError,
        "Cancelled": Cancelled,
        "EngineError": EngineError,
        "RunUI": RunUI,
        "TEXT": ui_panel.TEXT,
        "dome_pattern": SimpleNamespace(NAMES=NAMES),
        "DEFAULTS": SimpleNamespace(target="start_30s"),
        # the engine's own cancel (Control.CANCEL on itself): here, the fake ctx's
        "cancel_own_run": lambda engine: engine.test_ctx.cancel(),
    }
    ast.fix_missing_locations(cls)
    exec(compile(ast.Module(body=body, type_ignores=[]), "node.py", "exec"), scope)
    return scope


SCOPE = load()
Capability, failure = SCOPE["Capability"], SCOPE["failure"]
AT_PANEL, AT_PANEL_NOTE, UI_ACTIONS = SCOPE["AT_PANEL"], SCOPE["AT_PANEL_NOTE"], SCOPE["UI_ACTIONS"]


class FakeCtx:
    """As the engine's ctx: one held UI event, cancel() sets what checkpoint() raises on, update_user_panel() False
    without a UI session. `replies`: the user's taps, one per panel that asks something (has more than Stop)."""

    def __init__(self, replies=(), session=True):
        self.session = session
        self.replies = list(replies)
        self.sent, self.progress_calls = [], []
        self._event = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        self.sent.append(panel_json)
        if self.session and [b["button_id"] for b in panel_json["buttons"]] != ["stop"] and self.replies:
            self.tap(self.replies.pop(0))
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

    def progress(self, phase, pct, message="", pausable=True):
        self.progress_calls.append((phase, pct))


def wait_until(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


class FakePress:
    """Node._press: walks `steps` as press_button does (checkpoint, phase, show) then runs `then`; each call takes
    the next outcome. An outcome: a dict (the result), an exception (raised), or a callable(ctx) (e.g. Stop)."""

    def __init__(self, ctx, steps, *outcomes):
        self.ctx, self.steps, self.outcomes, self.calls = ctx, steps, list(outcomes), []

    def __call__(self, args, ctx, phase, checkpoint, show):
        self.calls.append(args)
        outcome = self.outcomes.pop(0)
        for i, name in enumerate(self.steps):
            checkpoint()
            phase[0] = name
            show(name, 0.1 * (i + 1), True)
        if callable(outcome) and not isinstance(outcome, type):
            return outcome(ctx)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def user_stops_mid_motion(ctx):
    """Stop tapped while the arm moves; like self.arm, the motion runs until ctx.cancelled(), then raises."""
    ctx.tap("stop")
    assert wait_until(ctx.cancelled), "Stop did not cancel the motion in flight"
    raise Cancelled()


def quiet(*_):
    pass


OK = {"dry_run": False, "pressed": 3, "returned": True, "moved": True, "log_file": "/x"}


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(panels, "POLL_S", 0.002)


ALL_STEPS = ("preflight", "detect", "stage", "redetect", "plan", "approach", "press", "return")


def make(replies=(), steps=ALL_STEPS, outcomes=(OK,), session=True, allow=True, mock=False):
    ctx = FakeCtx(replies, session)
    node = Capability()
    node.test_ctx = ctx
    node.mock = mock
    node.settings = {"allow_execute": allow}
    node.get_logger = lambda: SimpleNamespace(info=quiet, warning=quiet, error=quiet)
    node._args = lambda params: SimpleNamespace(**params)
    node._press = FakePress(ctx, steps, *outcomes)
    return node, ctx


def test_ui_targets_are_the_detectors_names_and_the_ids_read_as_buttons():
    assert {t for t, _ in UI_ACTIONS.values()} == set(NAMES)
    assert list(UI_ACTIONS) == ["press_start", "press_power", "press_timer_clock", "press_defrost", "press_stop_eco"]
    assert UI_ACTIONS["press_start"][0] == "start_30s" and UI_ACTIONS["press_defrost"][0] == "wgt_time_defrost"


def test_availability_offers_only_the_five_ui_actions_and_press_button_stays_an_action():
    node, _ = make()
    assert [a.action_id for a in node.availability("microwave_1", {})] == list(UI_ACTIONS)
    assert "press_button" in [a.action_id for a in Capability.actions]


def test_the_count_asked_becomes_the_presses_param():
    node, ctx = make(replies=["presses_3", "done"])
    assert node.execute("press_start", {}, ctx) == OK
    (args,) = node._press.calls
    assert (args.target, args.presses) == ("start_30s", 3)
    ask = ctx.sent[0]
    assert ask["message"] == "How many presses of Start?"
    assert [b["label"] for b in ask["buttons"]] == ["1", "2", "3", "4", "5", "Stop"]
    moving = [p for p in ctx.sent if p.get("progress")]
    assert moving and all(p["buttons"] == [panels.STOP] for p in moving)
    assert moving[-1]["progress"]["phase"] == "return" and "3 presses" in moving[-1]["message"]
    assert ctx.sent[-1]["message"] == "Pressed Start 3 times." and ctx.sent[-1]["buttons"] == [panels.DONE]


@pytest.mark.parametrize("ui_id", list(UI_ACTIONS))
def test_each_ui_action_presses_its_own_button(ui_id):
    node, ctx = make(replies=["presses_1", "done"], outcomes=(dict(OK, pressed=1),))
    node.execute(ui_id, {}, ctx)
    assert node._press.calls[0].target == UI_ACTIONS[ui_id][0]
    assert ctx.sent[-1]["message"] == f"Pressed {UI_ACTIONS[ui_id][1]} 1 time."


def test_stop_on_the_count_prompt_cancels_before_anything_runs():
    node, ctx = make(replies=["stop"])
    with pytest.raises(Cancelled):
        node.execute("press_power", {}, ctx)
    assert ctx.cancelled() and node._press.calls == []


@pytest.mark.parametrize("steps,at_panel", [(("preflight", "detect", "stage"), False), (("approach", "press"), True)])
def test_stop_during_the_motion_cancels_the_motion_in_flight_and_is_not_retried(steps, at_panel):
    node, ctx = make(replies=["presses_2", "done"], steps=steps, outcomes=(user_stops_mid_motion,))
    with pytest.raises(Cancelled):
        node.execute("press_start", {}, ctx)
    assert len(node._press.calls) == 1
    stopped = ctx.sent[-1]
    assert stopped["title"] == "Press Start: stopped" and stopped["buttons"] == [panels.DONE]
    assert stopped["message"].startswith(panels.STOPPED_MESSAGE)
    assert (AT_PANEL_NOTE in stopped["message"]) is at_panel


def test_a_refused_check_before_the_panel_offers_the_same_press_again():
    refused = failure("redetect", Abort("dome detector: nothing in 7 frames"))
    assert refused.recoverable
    node, ctx = make(
        replies=["presses_2", "retry", "done"],
        steps=("preflight", "detect", "stage", "redetect"),
        outcomes=(refused, dict(OK, pressed=2)),
    )
    assert node.execute("press_timer_clock", {}, ctx)["pressed"] == 2
    assert [a.presses for a in node._press.calls] == [2, 2]  # the count is asked once
    retry = next(p for p in ctx.sent if any(b["button_id"] == "retry" for b in p["buttons"]))
    assert "nothing in 7 frames" in retry["message"] and retry["message"].endswith("Try again?")
    assert retry["buttons"] == [{"button_id": "retry", "label": "Press Timer/Clock"}, panels.STOP]
    assert ctx.sent[-1]["message"] == "Pressed Timer/Clock 2 times."


def test_stop_on_the_retry_prompt_ends_it_without_pressing_again():
    node, ctx = make(
        replies=["presses_1", "stop"], steps=("preflight", "detect"), outcomes=(failure("detect", Abort("no panel")),)
    )
    with pytest.raises(Cancelled):
        node.execute("press_start", {}, ctx)
    assert len(node._press.calls) == 1


@pytest.mark.parametrize("phase", AT_PANEL)
def test_a_failure_at_the_panel_is_shown_and_never_retried(phase):
    steps = ("preflight", "detect", "plan", *AT_PANEL[: AT_PANEL.index(phase) + 1])
    node, ctx = make(replies=["presses_1", "done"], steps=steps, outcomes=(failure(phase, Abort("x")), OK))
    with pytest.raises(EngineError) as error:
        node.execute("press_defrost", {}, ctx)
    assert error.value.error_code == f"{phase}_failed" and not error.value.recoverable
    assert len(node._press.calls) == 1
    last = ctx.sent[-1]
    assert last["title"] == "Press Defrost: failed" and last["buttons"] == [panels.DONE]
    assert AT_PANEL_NOTE in last["message"]
    assert not any(b["button_id"] == "retry" for p in ctx.sent for b in p["buttons"])


def test_a_recoverable_error_raised_at_the_panel_is_still_not_retried():
    node, ctx = make(
        replies=["presses_1", "done"], steps=("plan", "approach"), outcomes=(EngineError("x", "odd", True), OK)
    )
    with pytest.raises(EngineError):
        node.execute("press_start", {}, ctx)
    assert len(node._press.calls) == 1 and AT_PANEL_NOTE in ctx.sent[-1]["message"]


def test_a_driver_failure_before_the_panel_is_shown_and_not_retried():
    node, ctx = make(
        replies=["presses_1", "done"],
        steps=("preflight", "stage"),
        outcomes=(failure("stage", MotionFailed("trajectory ended HALTED -- HOLDING HERE")), OK),
    )
    with pytest.raises(EngineError):
        node.execute("press_start", {}, ctx)
    assert len(node._press.calls) == 1
    assert "HALTED" in ctx.sent[-1]["message"] and AT_PANEL_NOTE not in ctx.sent[-1]["message"]


def test_a_bug_is_shown_too():
    node, ctx = make(replies=["presses_1", "done"], steps=("preflight",), outcomes=(ValueError("boom"),))
    with pytest.raises(ValueError):
        node.execute("press_start", {}, ctx)
    assert "boom" in ctx.sent[-1]["message"] and ctx.sent[-1]["title"] == "Press Start: failed"


@pytest.mark.parametrize(
    "phase,error,recoverable",
    [
        ("detect", Abort("no microwave"), True),
        ("plan", Abort("pre-check failed"), True),
        ("redetect", MotionFailed("HOLDING HERE"), False),
        ("stage", ArmError("stream closed"), False),
        *((p, Abort("x"), False) for p in AT_PANEL),
    ],
)
def test_failure_is_recoverable_only_for_a_refused_check_away_from_the_panel(phase, error, recoverable):
    e = failure(phase, error)
    assert (e.error_code, e.recoverable) == (f"{phase}_failed", recoverable)
    assert str(error) in str(e)


def test_execution_disabled_refuses_on_the_panel_and_never_runs():
    node, ctx = make(replies=["done"], allow=False)
    with pytest.raises(EngineError) as error:
        node.execute("press_start", {}, ctx)
    assert error.value.error_code == "execution_disabled"
    assert node._press.calls == []
    (refusal,) = ctx.sent  # no count asked
    assert refusal["title"] == "Press Start: not started" and "allow_execute" in refusal["message"]
    assert refusal["buttons"] == [panels.DONE]


def test_standalone_without_a_ui_session_uses_the_presses_param_and_waits_for_nothing():
    node, ctx = make(session=False)
    node.execute("press_stop_eco", {"presses": 2}, ctx)
    assert (node._press.calls[0].target, node._press.calls[0].presses) == ("stop_eco", 2)
    node, ctx = make(session=False)
    node.execute("press_stop_eco", {}, ctx)
    assert node._press.calls[0].presses == 1


def test_mock_says_nothing_moved():
    node, ctx = make(replies=["presses_1", "done"], mock=True)
    assert node.execute("press_start", {}, ctx)["dry_run"] is True
    assert node._press.calls == []  # the mock never runs the press
    assert "No arm motion" in ctx.sent[-1]["message"]


@pytest.fixture(autouse=True)
def _no_ui_wrap(monkeypatch):
    """Messages are compared whole here; capability_ui's own tests cover the UI's line wrapping."""
    from capability_ui import panels as ui_panels

    monkeypatch.setattr(ui_panels, "WRAP_CHARS", 10**6)
