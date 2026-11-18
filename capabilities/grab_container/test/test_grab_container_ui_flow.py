"""Offline tests of grab_container's one-press UI action (grab_by_brim) on capability_ui.RunUI: the node's own
methods, loaded with `ast` (no ROS, no hardware), against a fake ctx that behaves as the engine's (one held UI event,
cancel sets a flag checkpoint() raises on, update_user_panel False without a UI). As microwave_place's
test_microwave_place_ui_flow.py.

    uv run --no-project --with pytest python3 -m pytest -q \
        capabilities/grab_container/test/test_grab_container_ui_flow.py
"""

import ast
import importlib.util
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / "capabilities/grab_container/grab_container/node.py"
sys.path.insert(0, str(ROOT / "shared/capability_ui"))

from capability_ui import panels  # noqa: E402
from capability_ui.panels import RunUI  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "grab_container_ui_text", ROOT / "capabilities/grab_container/grab_container/ui_text.py"
)
UI_TEXT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(UI_TEXT)

GLOBALS = {"EXECUTE", "GRAB", "CONTAINER_CLASSES", "UI_GRAB", "UI_GRAB_PARAMS", "UI_RETRY_STEPS"}
FUNCTIONS = {"_PhaseCtx"}
METHODS = {"_ui_run", "availability", "execute"}
STOP = [{"button_id": "stop", "label": "Stop"}]
DONE = [{"button_id": "done", "label": "Done"}]


class Cancelled(Exception):
    pass


def engine_class(name, methods=None, **scope):
    tree = ast.parse((ROOT / "capability_engine/capability_engine/engine.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "engine.py", "exec"), scope)
    return scope[name]


EngineError = engine_class("EngineError")
Action = engine_class("Action", dataclass=dataclass, field=field)
# the engine's own availability(): an action whose perception_classes don't include the tapped class is left out
Base = engine_class(
    "CapabilityEngine",
    {"availability"},
    ActionAvailability=lambda action_id, available: SimpleNamespace(action_id=action_id, available=available),
)


def load():
    """The node's UI globals, _PhaseCtx and Capability (its class attributes and METHODS) on the engine's
    availability()."""
    tree = ast.parse(NODE.read_text())
    body = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in GLOBALS for t in n.targets):
            body.append(n)
        elif isinstance(n, ast.ClassDef) and n.name in FUNCTIONS:
            body.append(n)
        elif isinstance(n, ast.ClassDef) and n.name == "Capability":
            n.bases = [ast.Name(id="Base", ctx=ast.Load())]
            n.body = [
                m
                for m in n.body
                if (isinstance(m, ast.FunctionDef) and m.name in METHODS)
                or (isinstance(m, ast.Assign) and getattr(m.targets[0], "id", "") in ("actions", "perception_classes"))
            ]
            body.append(n)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    scope = {
        "Base": Base,
        "Action": Action,
        "EngineError": EngineError,
        "RunUI": RunUI,
        "cancel_own_run": lambda engine: engine.fake_ctx.cancel(),  # the engine's cancel: sets the run's flag
        "ui_text": UI_TEXT,
    }
    exec(compile(module, str(NODE), "exec"), scope)
    return scope


class FakeCtx:
    """The engine's ctx as RunUI sees it. A panel with Done is tapped Done at once; one with a retry button gets the
    next of `answers` ("retry" or "stop")."""

    def __init__(self, session=True, answers=()):
        self.session = session
        self.answers = list(answers)
        self.sent = []
        self.phases = []
        self._event = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        self.sent.append(panel_json)
        ids = [b["button_id"] for b in panel_json.get("buttons", [])]
        if "done" in ids:
            self.tap("done")
        elif "retry" in ids and self.answers:
            self.tap(self.answers.pop(0))
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
        self.phases.append(phase)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(panels, "POLL_S", 0.002)
    monkeypatch.setattr(panels, "WRAP_CHARS", 10**6)  # messages compared whole; capability_ui tests the wrapping


@pytest.fixture
def scope():
    return load()


def make(scope, ctx, steps=(), result=None, error=None, **cfg):
    """The node with `_grab` / the direct goal replaced: it reports `steps` ((phase, pct) as grab's ctx.progress),
    then raises `error` or returns `result`."""
    node = scope["Capability"]()
    node.mock = False
    node.cfg = {"allow_execute": True, **cfg}
    node.get_logger = Mock(return_value=Mock())
    node.fake_ctx = ctx
    node.calls = []
    outcomes = list(error) if isinstance(error, list) else [error]

    def grab(action_id, params, run_ctx):
        node.calls.append((action_id, params))
        for phase, pct in steps:
            run_ctx.progress(phase, pct)
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if callable(outcome):
            outcome(run_ctx)
        elif outcome is not None:
            raise outcome
        return result or {"executed": True, "raised": True}

    real_execute = node.execute
    node.execute = lambda action_id, params, ctx: (
        real_execute(action_id, params, ctx) if action_id == scope["UI_GRAB"] else grab(action_id, params, ctx)
    )
    return node


def run(scope, node, ctx):
    return node.execute(scope["UI_GRAB"], {}, ctx)


GRAB_STEPS = [("home", 0.0), ("locate", 2.0), ("view", 5.0), ("detect", 0.0), ("grasp", 10.0), ("approach", 30.0)]
GRAB_STEPS += [("grasp", 70.0), ("raise", 85.0)]


def test_one_press_runs_grab_executing_with_each_step_on_the_stop_panel_then_done(scope):
    ctx = FakeCtx()
    node = make(scope, ctx, steps=GRAB_STEPS)
    assert run(scope, node, ctx) == {"executed": True, "raised": True}
    assert node.calls == [("grab", {"execute": True})]
    assert ctx.phases == [p for p, _ in GRAB_STEPS]
    *moving, final = ctx.sent
    assert all(p["buttons"] == STOP for p in moving)
    assert [p["message"] for p in moving] == [UI_TEXT.MOVING] + [UI_TEXT.progress(p, pct)[0] for p, pct in GRAB_STEPS]
    assert moving[-1]["progress"] == {"phase": "raise", "pct": 85.0, "pausable": False}
    assert "Opening the gripper." in moving[5]["message"] and "Closing" in moving[7]["message"]
    assert final == {"title": UI_TEXT.TITLE, "message": UI_TEXT.DONE, "buttons": DONE}


def test_a_skipped_lift_is_said_on_done(scope):
    ctx = FakeCtx()
    node = make(scope, ctx, steps=GRAB_STEPS, result={"executed": True, "raised": False})
    run(scope, node, ctx)
    assert ctx.sent[-1]["message"] == UI_TEXT.DONE_NOT_RAISED


def stop_now(run_ctx):
    """The user presses Stop while the arm moves: the cancel stops the motion and the arm call raises Cancelled,
    as self.arm does."""
    run_ctx.tap("stop")
    assert run_ctx._cancelled.wait(2.0), "Stop did not cancel the run"
    raise Cancelled()


@pytest.mark.parametrize(
    "steps, holds",
    [
        ([("home", 0.0)], "Nothing has been grasped."),
        ([("view", 5.0), ("grasp", 10.0)], "nothing is held"),
        ([("approach", 30.0)], "at or near the tupperware's brim; nothing is held"),
        ([("approach", 30.0), ("grasp", 70.0)], "it may be holding the tupperware"),
        ([("raise", 85.0)], "holding the tupperware by its brim"),
    ],
)
def test_stop_cancels_and_says_what_the_hand_holds(scope, steps, holds):
    ctx = FakeCtx()
    node = make(scope, ctx, steps=steps, error=stop_now)
    with pytest.raises(Cancelled):
        run(scope, node, ctx)
    assert ctx.cancelled()
    assert len(node.calls) == 1  # never restarted
    last = ctx.sent[-1]
    assert last["title"] == f"{UI_TEXT.TITLE}: stopped"
    assert last["message"].startswith(panels.STOPPED_MESSAGE) and holds in last["message"]
    assert last["buttons"] == DONE
    assert not any("failed" in p["title"] for p in ctx.sent)


@pytest.mark.parametrize(
    "steps, raw, says",
    [
        ([("home", 0.0)], "home: PLANNING_FAILED -- HOLDING HERE", "The arm couldn't get to its start position"),
        (
            [("locate", 2.0)],
            "no 'food container' >= 0.5 in the scene camera (sees ['table'])",
            "The scene camera can't see the tupperware.",
        ),
        ([("view", 5.0)], "view: IK_FAIL -- HOLDING HERE", "The arm couldn't move the wrist camera above"),
        (
            [("detect", 0.0)],
            "ring not seen stably (2/5 good frames; rejected: no ring detected x3)",
            "The wrist camera couldn't find the tupperware's brim steadily (2 of 5 frames).",
        ),
        (
            [("detect", 0.0)],
            "frames disagree by 9 mm (> 5) -- not grasping",
            "The wrist camera's views of the brim disagreed by 9 mm, so the hand didn't reach in.",
        ),
    ],
)
def test_refusal_before_the_reach_offers_the_grab_again(scope, steps, raw, says):
    ctx = FakeCtx(answers=["retry"])
    error = EngineError("refused", f"{raw} -- the arm holds where it is", True)
    node = make(scope, ctx, steps=steps, error=[error, None])
    assert run(scope, node, ctx)["executed"]
    assert len(node.calls) == 2
    retry = next(p for p in ctx.sent if any(b["button_id"] == "retry" for b in p["buttons"]))
    assert retry["title"] == f"{UI_TEXT.TITLE}: not done"
    assert retry["message"].startswith(says)
    assert "The arm is holding still." in retry["message"]
    assert retry["message"].endswith(f"Press {UI_TEXT.RETRY_LABEL} to try again.")
    for code in ("HOLDING HERE", "IK_FAIL", "PLANNING_FAILED", "food container'", "-- "):
        assert code not in retry["message"]
    assert retry["buttons"] == [{"button_id": "retry", "label": UI_TEXT.RETRY_LABEL}, *STOP]
    assert ctx.sent[-1]["message"] == UI_TEXT.DONE


def test_stop_on_the_retry_question_ends_without_moving_again(scope):
    ctx = FakeCtx(answers=["stop"])
    node = make(
        scope, ctx, steps=[("locate", 2.0)], error=EngineError("refused", "no scene camera colour + depth", True)
    )
    with pytest.raises(Cancelled):
        run(scope, node, ctx)
    assert len(node.calls) == 1


def test_failure_after_the_reach_is_reported_never_retried(scope):
    ctx = FakeCtx(answers=["retry"])
    raw = "approach: the arm is e-stopped -- HOLDING HERE -- the arm holds where it is"
    error = EngineError("refused", raw, True)
    node = make(scope, ctx, steps=[("grasp", 10.0), ("approach", 30.0)], error=error)
    with pytest.raises(EngineError):
        run(scope, node, ctx)
    assert len(node.calls) == 1
    assert not any(any(b["button_id"] == "retry" for b in p["buttons"]) for p in ctx.sent)
    last = ctx.sent[-1]
    assert last["title"] == f"{UI_TEXT.TITLE}: failed" and last["buttons"] == DONE
    assert last["message"].startswith("The arm stopped on its way in to the brim: the emergency stop is pressed")
    assert "nothing is held" in last["message"] and "Release the emergency stop" in last["message"]


def test_unexpected_errors_are_reported_without_retry(scope):
    ctx = FakeCtx(answers=["retry"])
    node = make(scope, ctx, steps=[("detect", 0.0)], error=RuntimeError("the arm is e-stopped"))
    with pytest.raises(RuntimeError):
        run(scope, node, ctx)
    assert len(node.calls) == 1
    assert "emergency stop is pressed" in ctx.sent[-1]["message"]
    assert ctx.sent[-1]["title"] == f"{UI_TEXT.TITLE}: failed"


def test_orchestrator_cancel_takes_the_ui_back_without_notices(scope):
    ctx = FakeCtx(answers=["retry"])

    def cancelled(run_ctx):
        ctx.cancel()  # the orchestrator's cancel, not our Stop
        raise EngineError("refused", "view: cancelled -- HOLDING HERE", True)

    node = make(scope, ctx, steps=[("view", 5.0)], error=cancelled)
    with pytest.raises(EngineError):
        run(scope, node, ctx)
    assert [p["message"] for p in ctx.sent] == [UI_TEXT.MOVING, UI_TEXT.PHASES["view"][0]]


def test_execution_disabled_is_refused_before_anything(scope):
    ctx = FakeCtx()
    node = make(scope, ctx, allow_execute=False)
    with pytest.raises(EngineError) as error:
        run(scope, node, ctx)
    assert error.value.error_code == "execution_disabled"
    assert node.calls == []
    (notice,) = ctx.sent
    assert notice["title"] == f"{UI_TEXT.TITLE}: not started"
    assert "arm motion off" in notice["message"] and "allow_execute" not in notice["message"]


def test_no_ui_session_returns_the_recoverable_error_without_waiting(scope):
    ctx = FakeCtx(session=False)
    ctx.poll_ui_event = Mock(return_value=None)
    node = make(scope, ctx, steps=[("home", 0.0)], error=EngineError("refused", "home: no plan", True))
    with pytest.raises(EngineError) as error:
        run(scope, node, ctx)
    assert error.value.recoverable
    assert len(node.calls) == 1
    ctx.poll_ui_event.assert_not_called()


def test_mock_says_nothing_moved(scope):
    ctx = FakeCtx()
    node = make(scope, ctx, steps=[("grab", 100.0)], result={"executed": False, "mock": True})
    node.mock = True
    node.cfg["allow_execute"] = False  # a mock never moves: not refused
    run(scope, node, ctx)
    assert ctx.sent[-1]["message"] == f"Mock: {UI_TEXT.DONE} No arm motion."


# -- what the UI is offered ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tapped, offered",
    [
        ("food_container", {"grab_by_brim"}),
        ("food container", {"grab_by_brim"}),  # the scene segmenter's class, with its space
        ("microwave", set()),
        ("table", set()),
        (None, {"grab_by_brim"}),  # class unknown: every UI action
    ],
)
def test_availability_offers_only_the_ui_grab_on_the_tupperware(scope, tapped, offered):
    node = scope["Capability"]()
    objects = {} if tapped is None else {"obj": {"perception_class": tapped}}
    assert {a.action_id for a in node.availability("obj", {"objects": objects})} == offered


def test_direct_actions_stay_declared_and_the_tupperware_activates_the_engine(scope):
    cap = scope["Capability"]
    assert {a.action_id for a in cap.actions} == {"grab", "park", "grab_by_brim"}
    assert set(cap.perception_classes) == {"food_container", "food container"}
    assert {a.action_id: a.label for a in cap.actions}["grab_by_brim"] == "Grab Tupperware by Brim"


# -- the words -----------------------------------------------------------------------------------------------------


def test_every_reported_step_has_words():
    for phase, pct in GRAB_STEPS:
        text, bar = UI_TEXT.progress(phase, pct)
        assert text.endswith(".") and bar is not None
    assert UI_TEXT.progress("grab", 100.0, "mock: nothing detected or moved")[1] is None


def test_an_unknown_failure_keeps_its_own_words():
    what, todo = UI_TEXT.reason("something new broke -- the arm holds where it is")
    assert what == "Something new broke." and todo is None
