"""Offline tests of fridge_peering_mode's one-press UI action (look_in_fridge) on
capability_ui.RunUI: ui_flow.run against a fake ctx that behaves as the engine's (one held UI
event, cancel sets a flag checkpoint() raises on, update_user_panel False without a UI) and a fake
execute() standing in for `peer`. No ROS, no hardware."""

import ast
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
for path in (
    "shared/capability_ui",
    "capabilities/microwave_place",
    "capabilities/fridge_peering_mode",
):
    sys.path.insert(0, str(ROOT / path))

from capability_ui import panels  # noqa: E402
from fridge_peering_mode import ui_flow, ui_text  # noqa: E402

NODE = ROOT / "capabilities/fridge_peering_mode/fridge_peering_mode/node.py"
STOP = {"button_id": "stop", "label": "Stop"}
DONE = {"button_id": "done", "label": "Done"}


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
Base = engine_class(
    "CapabilityEngine",
    {"availability"},
    ActionAvailability=lambda action_id, available: SimpleNamespace(
        action_id=action_id, available=available
    ),
)


@pytest.fixture(autouse=True)
def engine_module(monkeypatch):
    """ui_flow.run imports EngineError from capability_engine when it runs: the engine's own
    class, without ROS."""
    module = ModuleType("capability_engine")
    module.EngineError = EngineError
    monkeypatch.setitem(sys.modules, "capability_engine", module)
    monkeypatch.setattr(panels, "POLL_S", 0.002)


def unwrap(message):
    """A panel's message as one line (capability_ui.wrap breaks it into rich-text runs)."""
    return message.replace(f"</>\n<{panels.UI_TEXT_STYLE}>", " ")


class FakeCtx:
    """The engine's ctx as RunUI and FoodTargets see it. A panel with Done is tapped at once."""

    def __init__(self, session=True):
        self.session = session
        self.sent = []  # (panel_json, objects)
        self.phases = []
        self.overlay_visible = []
        self.overlay_frames = []
        self._event = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        self.sent.append((panel_json, objects))
        if DONE in panel_json.get("buttons", []):
            self.tap("done")
        return self.session

    def poll_ui_event(self):
        with self._lock:
            event, self._event = self._event, None
            return event

    def tap(self, button_id):
        with self._lock:
            self._event = {"type": "button", "button_id": button_id}

    def tap_object(self, instance_id):
        with self._lock:
            self._event = {"type": "object_tapped", "instance_id": instance_id}

    def cancel(self):
        self._cancelled.set()

    def cancelled(self):
        return self._cancelled.is_set()

    def checkpoint(self):
        if self.cancelled():
            raise Cancelled()

    def progress(self, phase, pct, message="", pausable=True):
        self.phases.append(phase)

    def set_overlay_visible(self, visible):
        self.overlay_visible.append(visible)
        return self.session

    def publish_overlay_image(self, image):
        self.overlay_frames.append(image)
        return self.session

    @property
    def panels(self):
        return [p for p, _ in self.sent]


def make(ctx, run=None, allow_execute=True, mock=False):
    """An engine as ui_flow.run sees it: `run(action_id, params, run_ctx)` is its execute()."""
    engine = SimpleNamespace(mock=mock, cfg={"allow_execute": allow_execute})
    engine.get_logger = Mock(return_value=Mock())
    engine.execute = Mock(
        side_effect=run or (lambda action_id, params, run_ctx: {"executed": True})
    )
    ui_flow.cancel_own_run = lambda _engine: (
        ctx.cancel()
    )  # the engine's cancel: sets the run's flag
    return engine


@pytest.fixture(autouse=True)
def restore_cancel():
    original = ui_flow.cancel_own_run
    yield
    ui_flow.cancel_own_run = original


# -- the press -------------------------------------------------------------------------------------


def test_one_press_runs_peer_executing_then_done():
    ctx = FakeCtx()
    engine = make(ctx, lambda a, p, c: {"executed": True, "ended": "max_teleop_s"})
    assert ui_flow.run(engine, "look_in_fridge", ctx)["executed"]
    (action_id, params, run_ctx), _ = engine.execute.call_args
    assert (action_id, params) == ("peer", {"execute": True})
    assert isinstance(run_ctx, ui_flow.RunCtx)
    moving, final = ctx.panels
    assert moving["buttons"] == [STOP]
    assert unwrap(moving["message"]) == ui_text.MOVING
    assert final["buttons"] == [DONE]
    assert unwrap(final["message"]) == ui_text.DONE_LOOKING


def test_a_pick_ends_with_the_container_held():
    ctx = FakeCtx()
    engine = make(
        ctx, lambda a, p, c: {"executed": True, "ended": "picked", "pick": {"executed": True}}
    )
    ui_flow.run(engine, "look_in_fridge", ctx)
    assert unwrap(ctx.panels[-1]["message"]) == ui_text.DONE_PICKED


def test_mock_says_nothing_moved():
    ctx = FakeCtx()
    engine = make(
        ctx, lambda a, p, c: {"executed": False, "mock": True}, allow_execute=False, mock=True
    )
    ui_flow.run(engine, "look_in_fridge", ctx)
    assert unwrap(ctx.panels[-1]["message"]).startswith("Mock: ")
    assert "No arm motion." in unwrap(ctx.panels[-1]["message"])


def test_disabled_refuses_before_anything_runs():
    ctx = FakeCtx()
    engine = make(ctx, allow_execute=False)
    with pytest.raises(EngineError) as raised:
        ui_flow.run(engine, "look_in_fridge", ctx)
    assert raised.value.error_code == "execution_disabled"
    assert raised.value.recoverable
    engine.execute.assert_not_called()
    (panel,) = ctx.panels
    assert panel["title"].endswith(": not started")
    assert panel["buttons"] == [DONE]


def test_standalone_runs_without_waiting():
    ctx = FakeCtx(session=False)
    engine = make(ctx, lambda a, p, c: {"executed": True, "ended": "max_teleop_s"})
    t0 = time.monotonic()
    assert ui_flow.run(engine, "look_in_fridge", ctx)["executed"]
    assert time.monotonic() - t0 < 1.0


# -- while it runs ---------------------------------------------------------------------------------


def test_the_panel_follows_the_steps_with_stop_on_every_one():
    def run(action_id, params, run_ctx):
        run_ctx.progress("peer", 0.0, "executing")  # no words of its own: the panel stays
        run_ctx.progress("scene_homing", 0.0, "locating the fridge in the scene camera")
        run_ctx.progress("homing", 0.0, "waiting for the fridge TF")
        run_ctx.progress(
            "homing", 10.0, "planning and moving to the fridge standoff"
        )  # same words: not re-sent
        run_ctx.progress("searching", 0.0, "no fridge frame: move the stick to look around")
        run_ctx.progress("searching", 50.0, "fridge detected: let go of the stick")
        run_ctx.progress(
            "teleop",
            50.0,
            "teleop: move the stick; hold still and tap a food container to pick it up",
        )
        run_ctx.progress("pick", 10.0, "detecting the container")
        run_ctx.progress("reach", 0.0)
        run_ctx.progress("leave", 80.0, "backing out of the fridge")
        return {"executed": True, "ended": "picked", "pick": {"executed": True}}

    ctx = FakeCtx()
    ui_flow.run(make(ctx, run), "look_in_fridge", ctx)
    running = ctx.panels[:-1]
    assert all(p["buttons"] == [STOP] for p in running)
    messages = [unwrap(p["message"]) for p in running]
    assert messages == [
        ui_text.MOVING,
        ui_text.phase_text("scene_homing"),
        ui_text.phase_text("homing"),
        ui_text.phase_text("searching"),
        ui_text.phase_text("searching", 50.0, "fridge detected"),
        ui_text.phase_text("teleop", 50.0, "tap"),
        ui_text.phase_text("pick", 10.0),
        ui_text.phase_text("reach"),
        ui_text.phase_text("leave"),
    ]
    assert "tap a container" in messages[5]
    assert ctx.phases == [
        "peer",
        "scene_homing",
        "homing",
        "homing",
        "searching",
        "searching",
        "teleop",
        "pick",
        "reach",
        "leave",
    ]  # the orchestrator still gets every step


def test_teleop_without_tap_to_pick_does_not_mention_tapping():
    assert "tap" not in ui_text.phase_text(
        "teleop", 50.0, "teleop: move the stick to move the camera, let go"
    )


def test_food_panels_keep_stop_and_their_boxes():
    """FoodTargets' panels (its own title, no buttons) go out with the run's title and Stop, its
    objects as they are."""
    boxes = [
        {
            "instance_id": "food_1",
            "perception_class": "food",
            "box": [1.0, 2.0, 3.0, 4.0],
            "source": "overlay",
        }
    ]
    seen = {}

    def run(action_id, params, run_ctx):
        seen["ui"] = run_ctx.update_user_panel(
            panel_json={"title": "Fridge", "message": "Tap a food container."}, objects=boxes
        )
        run_ctx.progress(
            "teleop", 50.0, "teleop: ... tap a food container"
        )  # boxes left as they are
        return {"executed": True, "ended": "max_teleop_s"}

    ctx = FakeCtx()
    ui_flow.run(make(ctx, run), "look_in_fridge", ctx)
    assert seen["ui"] is True
    (food, objects), (teleop, teleop_objects) = ctx.sent[1:3]
    assert food["title"] == ui_text.TITLE
    assert food["buttons"] == [STOP]
    assert unwrap(food["message"]) == "Tap a food container."
    assert objects == boxes
    assert teleop["buttons"] == [STOP]
    assert teleop_objects is None


def test_food_panels_say_standalone():
    seen = {}

    def run(action_id, params, run_ctx):
        seen["ui"] = run_ctx.update_user_panel(
            panel_json={"title": "Fridge", "message": "x"}, objects=[]
        )
        return {"executed": True}

    ctx = FakeCtx(session=False)
    ui_flow.run(make(ctx, run), "look_in_fridge", ctx)
    assert seen["ui"] is False  # FoodTargets then mirrors on ~/ui_panel and takes ~/simulate_tap


def test_a_food_tap_reaches_food_targets_and_does_not_stop():
    seen = {}

    def run(action_id, params, run_ctx):
        run_ctx.update_user_panel(panel_json={"message": "Tap a food container."}, objects=[])
        ctx.tap_object("food_3")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            event = run_ctx.poll_ui_event()
            if event is not None:
                seen["tap"] = event
                break
            time.sleep(0.002)
        return {"executed": True, "ended": "max_teleop_s"}

    ctx = FakeCtx()
    ui_flow.run(make(ctx, run), "look_in_fridge", ctx)
    assert seen["tap"] == {"type": "object_tapped", "instance_id": "food_3"}
    assert not ctx.cancelled()


def test_the_overlay_is_shown_once_with_the_first_frame():
    def run(action_id, params, run_ctx):
        run_ctx.publish_overlay_image("frame 1")
        run_ctx.publish_overlay_image("frame 2")
        return {"executed": True}

    ctx = FakeCtx()
    ui_flow.run(make(ctx, run), "look_in_fridge", ctx)
    assert ctx.overlay_visible == [True]
    assert ctx.overlay_frames == ["frame 1", "frame 2"]


# -- Stop, failure, cancel -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase, holds",
    [
        ("teleop", ui_text.NOTHING_HELD),
        ("homing", ui_text.NOTHING_HELD),
        ("reach", "nothing is held"),
        ("close", "may be holding it"),
        ("lift", "holding the container in the fridge"),
        ("leave", "part way out of the fridge"),
    ],
)
def test_stop_cancels_and_says_what_the_hand_holds(phase, holds):
    ctx = FakeCtx()

    def run(action_id, params, run_ctx):
        run_ctx.progress(phase, 0.0, "")
        ctx.tap("stop")
        assert ctx._cancelled.wait(2.0), "Stop did not cancel the run"
        raise Cancelled()

    engine = make(ctx, run)
    with pytest.raises(Cancelled):
        ui_flow.run(engine, "look_in_fridge", ctx)
    assert engine.execute.call_count == 1
    last = ctx.panels[-1]
    assert last["title"].endswith(": stopped")
    assert unwrap(last["message"]).startswith(panels.STOPPED_MESSAGE)
    assert holds in unwrap(last["message"])
    assert last["buttons"] == [DONE]
    assert not any(p["title"].endswith(": failed") for p in ctx.panels)


@pytest.mark.parametrize(
    "error, says",
    [
        (
            EngineError(
                "no_fridge",
                "no usable fridge TF within 5.0s: none -- the arm holds where it is",
                True,
            ),
            "back wall wasn't detected",
        ),
        (
            EngineError(
                "no_tf", "arm TF arm_end_effector_link -> arm_tool_frame not available (x)", True
            ),
            "arm model isn't being published",
        ),
        (
            EngineError(
                "refused", "homing: MotionGenStatus.IK_FAIL -- the arm holds where it is", True
            ),
            "couldn't move in front of the fridge",
        ),
        (EngineError("estopped", "e-stopped -- the arm holds where it is", True), "emergency stop"),
        (
            EngineError("stream_closed", "session 4 closed -- the arm holds where it is", True),
            "control stream",
        ),
        (
            EngineError(
                "pick_failed",
                "STOPPED: the gripper only closed to 0.91 -- the arm holds where it is",
            ),
            "gripper closed less than expected (0.91)",
        ),
    ],
)
def test_failure_says_what_happened_in_plain_words(error, says):
    ctx = FakeCtx()

    def run(action_id, params, run_ctx):
        run_ctx.progress("homing", 0.0, "")
        raise error

    with pytest.raises(EngineError):
        ui_flow.run(make(ctx, run), "look_in_fridge", ctx)
    last = ctx.panels[-1]
    assert last["title"].endswith(": failed")
    assert last["buttons"] == [DONE]
    assert says in unwrap(last["message"])
    assert "TF" not in unwrap(last["message"])
    assert "_" not in unwrap(last["message"])  # no parameter or frame names


def test_an_unexpected_error_still_ends_with_a_notice():
    ctx = FakeCtx()

    def run(action_id, params, run_ctx):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        ui_flow.run(make(ctx, run), "look_in_fridge", ctx)
    assert ctx.panels[-1]["title"].endswith(": failed")


def test_an_orchestrator_cancel_shows_nothing_more():
    ctx = FakeCtx()

    def run(action_id, params, run_ctx):
        ctx.cancel()
        raise Cancelled()

    with pytest.raises(Cancelled):
        ui_flow.run(make(ctx, run), "look_in_fridge", ctx)
    assert len(ctx.panels) == 1  # the progress panel only


# -- what the orchestrator offers ------------------------------------------------------------------


def node_capability():
    """node.Capability's class attributes and availability(), on the engine's own (no ROS)."""
    tree = ast.parse(NODE.read_text())
    body = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in (
            "_EXECUTE",
            "_LABEL",
            "_SPEED",
        ):
            body.append(n)
        elif isinstance(n, ast.ClassDef) and n.name == "Capability":
            n.bases = [ast.Name(id="Base", ctx=ast.Load())]
            n.body = [
                m
                for m in n.body
                if (isinstance(m, ast.FunctionDef) and m.name == "availability")
                or (
                    isinstance(m, ast.Assign)
                    and getattr(m.targets[0], "id", "") in ("actions", "perception_classes")
                )
            ]
            body.append(n)
    scope = {"Base": Base, "Action": Action, "ui_flow": ui_flow}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(NODE), "exec"
        ),
        scope,
    )
    return scope["Capability"]


def test_a_fridge_tap_offers_only_the_one_press_action():
    cap = node_capability()
    world = {"objects": {"fridge_1": {"perception_class": "fridge"}}}
    assert [a.action_id for a in cap().availability("fridge_1", world)] == ["look_in_fridge"]
    world = {"objects": {"microwave_1": {"perception_class": "microwave"}}}
    assert cap().availability("microwave_1", world) == []
    labels = {a.action_id: a.label for a in cap.actions}
    assert labels["look_in_fridge"] == "Look Inside Fridge"
    assert {"peer", "pick"} <= set(labels)  # still there for direct goals


def test_every_panel_message_is_a_sentence():
    for text in (
        ui_text.MOVING,
        ui_text.DONE_LOOKING,
        ui_text.DONE_PICKED,
        ui_text.disabled(ui_text.TITLE),
    ):
        assert text.endswith(".")
    for phase in (
        "scene_homing",
        "homing",
        "unwind",
        "searching",
        "teleop",
        "pick",
        "reach",
        "leave",
    ):
        assert ui_text.phase_text(phase).endswith(".")
