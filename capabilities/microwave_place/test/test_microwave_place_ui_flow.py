"""Offline tests of microwave_place's one-press UI actions (retrieve, place_in_microwave, retrieve_from_fridge) on
capability_ui.RunUI: the node's own methods, loaded with `ast` (no ROS, no hardware), against a fake ctx that behaves as
the engine's (one held UI event, cancel sets a flag checkpoint() raises on, update_user_panel False without a UI)."""

import ast
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / "capabilities/microwave_place/microwave_place/microwave_place_node.py"
sys.path.insert(0, str(ROOT / "shared/capability_ui"))

from capability_ui import panels  # noqa: E402
from capability_ui.panels import RunUI  # noqa: E402

GLOBALS = {
    "EXECUTE",
    "PLACE",
    "TAKE_OUT",
    "FRIDGE_TAKE",
    "SITES",
    "FRIDGE_DOOR_GAP",
    "UI_ACTIONS",
    "UI_TITLE",
    "UI_MOVING",
    "UI_DONE",
    "UI_RETRY",
    "UI_RETRYABLE",
}
FUNCTIONS = {"_site_words", "_hand_note", "_started", "_PhaseCtx"}
METHODS = {"_ui_run", "availability", "_take_out"}


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


class Refused(Exception):
    pass


def load(**extra):
    """The node's UI globals, helpers and Capability (its class attributes and METHODS) on the engine's
    availability()."""
    tree = ast.parse(NODE.read_text())
    body = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in GLOBALS for t in n.targets):
            body.append(n)
        elif isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name in FUNCTIONS:
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
        "contextmanager": contextmanager,
        "Action": Action,
        "EngineError": EngineError,
        "RunUI": RunUI,
        "cancel_own_run": lambda engine: engine.fake_ctx.cancel(),  # the engine's cancel: sets the run's flag
        "place_motion": SimpleNamespace(Refused=Refused),
        "ArmRefused": type("ArmRefused", (Exception,), {}),
        **extra,
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


@pytest.fixture
def scope():
    return load()


def make(scope, ctx, **cfg):
    node = scope["Capability"]()
    node.mock = False
    node.cfg = {"allow_execute": True, "fridge_accept_unmodeled_door": False, **cfg}
    node.get_logger = Mock(return_value=Mock())
    node.execute = Mock(return_value={"executed": True})
    node.fake_ctx = ctx
    return node


def stop_during(phase):
    """An execute() that is moving in `phase` when the user presses Stop: the cancel stops the motion and the
    arm call raises Cancelled, as self.arm does."""

    def run(action_id, params, ctx):
        ctx.progress(phase, 0.0)
        ctx.tap("stop")
        assert ctx._cancelled.wait(2.0), "Stop did not cancel the run"
        raise Cancelled()

    return run


def titles(ctx):
    return [p["title"] for p in ctx.sent]


# -- every UI action ----------------------------------------------------------------------------------------------

RUNS = {
    "retrieve": ("take_out", {"execute": True}, "Tupperware retrieved from the microwave."),
    "place_in_microwave": ("place", {"execute": True}, "Tupperware placed in the microwave."),
    "retrieve_from_fridge": ("take_out", {"execute": True, "site": "fridge"}, "Tupperware retrieved from the fridge."),
}


@pytest.mark.parametrize("ui_id", list(RUNS))
def test_one_press_runs_the_action_executing_then_done(scope, ui_id):
    ctx = FakeCtx()
    node = make(scope, ctx, fridge_accept_unmodeled_door=True)
    assert node._ui_run(ui_id, ctx) == {"executed": True}
    action_id, params, done = RUNS[ui_id]
    (called_action, called_params, run_ctx), _ = node.execute.call_args
    assert (called_action, called_params) == (action_id, params)
    assert node.execute.call_count == 1
    moving, final = ctx.sent  # the progress panel once (never re-sent while open: no flashing), then Done
    assert moving["buttons"] == [{"button_id": "stop", "label": "Stop"}]
    assert moving["message"] == scope["UI_MOVING"][ui_id]
    assert final["message"] == done
    assert final["buttons"] == [{"button_id": "done", "label": "Done"}]


@pytest.mark.parametrize(
    "ui_id, phase, holds",
    [
        ("retrieve", "lift", "holding the container in the microwave"),
        ("retrieve", "reach", "nothing is held"),
        ("retrieve_from_fridge", "retract", "part way out of the fridge"),
        ("place_in_microwave", "approach", "holding the container"),
        ("place_in_microwave", "tilt", "gripper is open"),
    ],
)
def test_stop_during_motion_cancels_and_says_what_the_hand_holds(scope, ui_id, phase, holds):
    ctx = FakeCtx()
    node = make(scope, ctx, fridge_accept_unmodeled_door=True)
    node.execute.side_effect = stop_during(phase)
    with pytest.raises(Cancelled):
        node._ui_run(ui_id, ctx)
    assert ctx.cancelled()
    assert node.execute.call_count == 1  # never restarted
    last = ctx.sent[-1]
    assert last["title"].endswith(": stopped")
    assert last["message"].startswith(panels.STOPPED_MESSAGE)
    assert holds in last["message"]
    assert last["buttons"] == [{"button_id": "done", "label": "Done"}]
    assert not any("failed" in t for t in titles(ctx))


@pytest.mark.parametrize(
    "ui_id, error",
    [
        ("retrieve", EngineError("home_failed", "take-out home failed: PLANNING_FAILED", True)),
        ("retrieve", EngineError("refused", "no container found in 60 s", True)),
        ("place_in_microwave", EngineError("refused", "no side shift and heading passes the checks", True)),
        ("retrieve_from_fridge", EngineError("refused", "take-out approach: cuRobo refused: no plan", True)),
    ],
)
def test_refusal_before_motion_offers_the_action_again(scope, ui_id, error):
    ctx = FakeCtx(answers=["retry"])
    node = make(scope, ctx, fridge_accept_unmodeled_door=True)
    node.execute.side_effect = [error, {"executed": True}]
    assert node._ui_run(ui_id, ctx)["executed"]
    assert node.execute.call_count == 2
    retry = next(p for p in ctx.sent if any(b["button_id"] == "retry" for b in p["buttons"]))
    assert str(error) in retry["message"]
    assert retry["buttons"] == [
        {"button_id": "retry", "label": scope["UI_RETRY"][ui_id]},
        {"button_id": "stop", "label": "Stop"},
    ]
    assert ctx.sent[-1]["message"] == RUNS[ui_id][2]


def test_stop_on_the_retry_question_ends_without_moving_again(scope):
    ctx = FakeCtx(answers=["stop"])
    node = make(scope, ctx)
    node.execute.side_effect = EngineError("refused", "no plan", True)
    with pytest.raises(Cancelled):
        node._ui_run("place_in_microwave", ctx)
    assert node.execute.call_count == 1


@pytest.mark.parametrize(
    "ui_id, message",
    [
        ("place_in_microwave", "STOPPED in the oven, holding the container: the hand is 3.0 cm from the target"),
        ("place_in_microwave", "STOPPED, nothing moved: the arm is e-stopped (x)"),
        ("retrieve", "STOPPED in the oven: the gripper only closed to 0.400 -- not lifting"),
        ("retrieve", "STOPPED during the take-out: pre-grasp is 2.0 cm / 1.0 deg off target -- not inserting"),
    ],
)
def test_stopped_failure_is_reported_never_retried(scope, ui_id, message):
    ctx = FakeCtx(answers=["retry"])
    node = make(scope, ctx)
    node.execute.side_effect = EngineError("refused", message, True)
    with pytest.raises(EngineError, match="STOPPED"):
        node._ui_run(ui_id, ctx)
    assert node.execute.call_count == 1
    assert not any(any(b["button_id"] == "retry" for b in p["buttons"]) for p in ctx.sent)
    last = ctx.sent[-1]
    assert last["title"].endswith(": failed")
    assert message in last["message"]
    assert last["buttons"] == [{"button_id": "done", "label": "Done"}]


def test_unrecoverable_or_unexpected_errors_are_reported_without_retry(scope):
    ctx = FakeCtx(answers=["retry"])
    node = make(scope, ctx)
    node.execute.side_effect = RuntimeError("the arm is e-stopped")
    with pytest.raises(RuntimeError):
        node._ui_run("retrieve", ctx)
    assert node.execute.call_count == 1
    assert "e-stopped" in ctx.sent[-1]["message"] and ctx.sent[-1]["title"].endswith(": failed")


def test_fridge_failure_names_the_fridge(scope):
    ctx = FakeCtx()
    node = make(scope, ctx, fridge_accept_unmodeled_door=True)
    node.execute.side_effect = EngineError("refused", "STOPPED in the oven, gripper open: 4.0 cm off the grasp", True)
    with pytest.raises(EngineError):
        node._ui_run("retrieve_from_fridge", ctx)
    assert "STOPPED in the fridge, gripper open" in ctx.sent[-1]["message"]
    assert "oven" not in ctx.sent[-1]["message"]


def test_orchestrator_cancel_takes_the_ui_back_without_notices(scope):
    ctx = FakeCtx(answers=["retry"])
    node = make(scope, ctx)

    def cancelled_mid_motion(action_id, params, run_ctx):
        ctx.cancel()  # the orchestrator's cancel, not our Stop
        raise EngineError("refused", "STOPPED: the approach did not complete (cancelled)", True)

    node.execute.side_effect = cancelled_mid_motion
    with pytest.raises(EngineError):
        node._ui_run("place_in_microwave", ctx)
    assert len(ctx.sent) == 1  # the progress panel only


@pytest.mark.parametrize("ui_id", list(RUNS))
def test_execution_disabled_is_refused_before_anything(scope, ui_id):
    ctx = FakeCtx()
    node = make(scope, ctx, allow_execute=False)
    with pytest.raises(EngineError) as error:
        node._ui_run(ui_id, ctx)
    assert error.value.error_code == "execution_disabled"
    node.execute.assert_not_called()
    (notice,) = ctx.sent
    assert notice["title"].endswith(": not started")
    assert "allow_execute:=true" in notice["message"]


def test_fridge_retrieve_refused_while_the_door_is_not_modelled(scope):
    ctx = FakeCtx()
    node = make(scope, ctx)  # fridge_accept_unmodeled_door false (the default)
    with pytest.raises(EngineError) as error:
        node._ui_run("retrieve_from_fridge", ctx)
    assert error.value.error_code == "fridge_door_unmodeled"
    node.execute.assert_not_called()
    (notice,) = ctx.sent
    assert notice["title"] == "Retrieve from fridge: not started"
    assert "open door" in notice["message"] and "fridge_accept_unmodeled_door" in notice["message"]


def test_no_ui_session_returns_the_recoverable_error_without_waiting(scope):
    ctx = FakeCtx(session=False)
    ctx.poll_ui_event = Mock(return_value=None)
    node = make(scope, ctx)
    node.execute.side_effect = EngineError("home_failed", "no plan", True)
    with pytest.raises(EngineError) as error:
        node._ui_run("retrieve", ctx)
    assert error.value.recoverable
    assert node.execute.call_count == 1


def test_mock_says_nothing_moved(scope):
    ctx = FakeCtx()
    node = make(scope, ctx)
    node.mock = True
    node.cfg["allow_execute"] = False  # a mock never moves: not refused
    node.execute.return_value = {"executed": False, "mock": True}
    node._ui_run("retrieve_from_fridge", ctx)
    assert "No arm motion" in ctx.sent[-1]["message"]


# -- what the UI is offered ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tapped, offered",
    [
        ("microwave", {"retrieve", "place_in_microwave"}),
        ("fridge", {"retrieve_from_fridge"}),
        (None, {"retrieve", "place_in_microwave", "retrieve_from_fridge"}),  # class unknown: every UI action
    ],
)
def test_availability_offers_only_the_ui_actions_on_their_class(scope, tapped, offered):
    node = scope["Capability"]()
    objects = {} if tapped is None else {"obj": {"perception_class": tapped}}
    got = {a.action_id for a in node.availability("obj", {"objects": objects})}
    assert got == offered


def test_direct_actions_stay_declared_and_the_fridge_activates_the_engine(scope):
    cap = scope["Capability"]
    ids = {a.action_id for a in cap.actions}
    assert {"place", "take_out", "retrieve", "place_in_microwave", "retrieve_from_fridge"} <= ids
    assert set(cap.perception_classes) == {"microwave", "fridge"}


# -- the node's motion boundary and the fridge settings ------------------------------------------------------------


def test_a_refusal_from_the_running_arm_reads_stopped(scope):
    with pytest.raises(Refused, match="^STOPPED during the take-out: pre-grasp is off"):
        with scope["_started"]("take-out"):
            raise Refused("pre-grasp is off")
    with pytest.raises(Refused, match="^STOPPED in the oven$"):
        with scope["_started"]("place"):
            raise Refused("STOPPED in the oven")


@pytest.fixture
def take(monkeypatch):
    stub = SimpleNamespace(set_approach_world=Mock(return_value="ok"), prepare_home=Mock())
    camera = Mock()
    scope = load(take_out=stub, rim_detection=camera)
    node = scope["Capability"]()
    node.cfg = {
        "take_approach_world": "microwave_world",
        "take_home_joints": [0.0] * 7,
        "take_approach_speed_scale": 0.3,
        "grasp_up": 0.035,
        "reach_in": 0.30,
        "fridge_take_approach_world": "fridge_world",
        "fridge_take_home_joints": [1.0] * 7,
        "fridge_grasp_up": 0.05,
        "fridge_reach_in": 0.25,
        "fridge_accept_unmodeled_door": False,
    }
    node.arm = Mock()
    return node, stub, camera, scope


def test_fridge_take_out_refuses_to_move_before_anything_while_the_door_is_not_modelled(take):
    node, stub, camera, _ = take
    with pytest.raises(Refused, match="before anything moved"):
        node._take_out(Mock(), True, Mock(), Mock(), "fridge")
    stub.set_approach_world.assert_not_called()
    stub.prepare_home.assert_not_called()
    camera.WristCamera.assert_not_called()


def test_fridge_take_out_uses_the_fridge_settings(take):
    node, stub, _, scope = take
    node.cfg["fridge_accept_unmodeled_door"] = True
    stub.prepare_home.side_effect = Refused("take-out home failed: the microwave pose")
    with pytest.raises(EngineError) as error:
        node._take_out(Mock(), True, Mock(), Mock(), "fridge")
    assert error.value.error_code == "home_failed"
    assert "fridge pose" in str(error.value)
    assert stub.set_approach_world.call_args.args[1] == "fridge_world"
    assert stub.prepare_home.call_args.args[1] == [1.0] * 7


def test_fridge_dry_run_is_not_gated(take):
    node, stub, _, _ = take
    stub.prepare_home.side_effect = Refused("dry run: off home")
    with pytest.raises(EngineError):
        node._take_out(Mock(), False, Mock(), Mock(), "fridge")
    assert stub.set_approach_world.call_args.args[1] == "fridge_world"


def test_microwave_take_out_keeps_its_own_settings(take):
    node, stub, _, _ = take
    stub.prepare_home.side_effect = Refused("off home")
    with pytest.raises(EngineError):
        node._take_out(Mock(), True, Mock(), Mock())
    assert stub.set_approach_world.call_args.args[1] == "microwave_world"
    assert stub.prepare_home.call_args.args[1] == [0.0] * 7


def test_phase_ctx_remembers_the_phase_and_passes_everything_on(scope):
    ctx = FakeCtx()
    run_ctx = scope["_PhaseCtx"](ctx)
    run_ctx.progress("close", 0.0)
    assert run_ctx.phase == "close" and ctx.phases == ["close"]
    assert run_ctx.cancelled() is False


@pytest.fixture(autouse=True)
def _no_ui_wrap(monkeypatch):
    """Messages are compared whole here; capability_ui's own tests cover the UI's line wrapping."""
    from capability_ui import panels as ui_panels

    monkeypatch.setattr(ui_panels, "WRAP_CHARS", 10**6)
