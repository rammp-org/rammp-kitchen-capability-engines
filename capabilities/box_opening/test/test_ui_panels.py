"""open_box's panels on the UI (node.PanelCtx, capability_ui.RunUI): the run's phases with Stop,
Stop as the run's own Ctrl+C through the base class's cancel, and one final notice with Done.
Standalone nothing is sent. The panel's design is Chris's (demo/ui e128789); the child is
test_node's stand-in script, the UI a fake ctx that records every panel and answers taps."""

import json
import shutil
import signal
import sys
import threading
import time

import pytest
from capability_engine.engine import Cancelled, EngineError, _Run
from capability_ui import panels
from rclpy.parameter import Parameter
from test_node import FAKE_RUN, FULL_RUN, FakeCtx, given

from box_opening import LOGGER, node

TOKEN = bytes(range(1, 17))
STOP = {"type": "button", "button_id": "stop"}


class UICtx(FakeCtx):
    """test_node's FakeCtx with the UI: every panel sent (as JSON round-trips it), taps answered.
    `stop_at`: Stop is tapped the first time a panel's message contains it. Done is tapped as soon
    as a notice shows it (`auto_done`). Cancelled as the base class's run says (`run`)."""

    def __init__(self, run=None, stop_at=None, auto_done=True, **kwargs):
        super().__init__(**kwargs)
        self.run = run
        self.stop_at = stop_at
        self.auto_done = auto_done
        self.panels = []
        self.events = []
        self.polls = 0
        self._lock = threading.Lock()

    def cancelled(self):
        return self.cancel or (self.run is not None and self.run.cancelled.is_set())

    def checkpoint(self):
        if self.cancelled():
            raise Cancelled()

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        with self._lock:
            self.panels.append(json.loads(json.dumps(panel_json)))
        return True

    def poll_ui_event(self):
        with self._lock:
            self.polls += 1
            if self.events:
                return self.events.pop(0)
            last = self.panels[-1] if self.panels else None
            if last is None:
                return None
            if self.stop_at and self.stop_at in last["message"]:
                self.stop_at = None
                return dict(STOP)
            if self.auto_done and any(b["button_id"] == "done" for b in last["buttons"]):
                return {"type": "button", "button_id": "done"}
        return None

    def messages(self):
        return [p["message"] for p in self.panels]

    def buttons(self, panel):
        return [b["button_id"] for b in panel["buttons"]]


@pytest.fixture(autouse=True)
def _no_ui_wrap(monkeypatch):
    """Whole messages compared: RunUI.send's wrapping for the UI (capability_ui WRAP_CHARS) off."""
    from capability_ui import panels as ui_panels

    monkeypatch.setattr(ui_panels, "WRAP_CHARS", 10**6)


@pytest.fixture(scope="module")
def ros():
    import rclpy

    rclpy.init()
    yield
    rclpy.shutdown()


def make_engine(tmp_path, monkeypatch, **overrides):
    script = tmp_path / "fake_run.py"
    script.write_text(FAKE_RUN)
    params = {"use_orchestrator": False, "allow_execute": True, "state_dir": str(tmp_path), **overrides}
    cap = node.Capability(parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
    real_argv = cap._argv
    monkeypatch.setattr(cap, "_argv", lambda *a: [sys.executable, str(script)] + real_argv(*a)[1:])
    return cap


@pytest.fixture
def engine(ros, tmp_path, monkeypatch):
    """A real engine node, its runs test_node's FAKE_RUN; the orchestrator's token pushed (the
    UI is on) and a goal running in the base class, so Stop's cancel has a run to cancel."""
    handlers = list(LOGGER.handlers)
    cap = make_engine(tmp_path, monkeypatch)
    cap._assigned_token = TOKEN
    cap._current = _Run("open_box", None)
    yield cap
    cap._current = None
    cap._assigned_token = bytes(16)
    cap.destroy_node()
    LOGGER.handlers[:] = handlers


def ctx_for(engine, **kwargs):
    return UICtx(run=engine._current, **kwargs)


def interrupts(monkeypatch):
    sent = []
    real = node.Run.interrupt
    monkeypatch.setattr(node.Run, "interrupt", lambda run: (sent.append(signal.SIGINT), real(run)))
    return sent


def test_a_full_run_shows_its_phases_with_stop_then_the_lid_off_with_done(engine, monkeypatch):
    given(monkeypatch, FULL_RUN)
    engine.instance_id = "food_container_1"
    ctx = ctx_for(engine)
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and out["lid_xyz"] == [0.597, -0.071, 0.209]
    running, notice = ctx.panels[:-1], ctx.panels[-1]
    assert running[0]["message"] == "Starting" and running[0]["title"] == node.PANEL_TITLE
    assert "Pressing the button" in ctx.messages() and "Letting go of the lid" in ctx.messages()
    for p in running:
        assert ctx.buttons(p) == ["stop"] and p["buttons"][0]["label"] == "Stop"
        assert p["progress"]["pausable"] is False
    pcts = [p["progress"]["pct"] for p in running]
    assert pcts == sorted(pcts) and pcts[0] == 0.0 and pcts[-1] == 100.0  # the UI's 0-100, never backwards
    # a panel is sent as it changes, never the same one twice in a row (a re-send redraws it)
    assert all(a != b for a, b in zip(running, running[1:], strict=False))
    assert notice["message"].startswith("Lid off") and ctx.buttons(notice) == ["done"]
    assert "progress" not in notice
    # the result is the run's, as without the UI
    assert ctx.objects == {"food_container_1": {"opened": True}}


def test_standalone_sends_nothing_and_the_run_is_as_it_was(ros, tmp_path, monkeypatch):
    handlers = list(LOGGER.handlers)
    cap = make_engine(tmp_path, monkeypatch)  # no token: standalone
    try:
        given(monkeypatch, FULL_RUN)
        ctx = UICtx()
        out = cap.execute("open_box", {}, ctx)
        assert out["lid_off"] and ctx.panels == [] and ctx.polls == 0
        assert [p for p, *_ in ctx.said][-1] == "home"
        assert cap._run_ui(ctx) is None
    finally:
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_stop_is_the_base_classs_cancel_the_runs_own_ctrl_c_and_says_stopped(engine, monkeypatch):
    """Stop on the panel is the orchestrator's cancel: the base class's run is cancelled, the run
    gets ONE Ctrl+C (the motion in flight cancelled, the arm holds), the action ends cancelled with
    nothing written after; the panel says it is stopping, with no button, then Stopped with Done."""
    given(monkeypatch, FULL_RUN[:5], wait=True)  # up to the press, then it waits
    sent = interrupts(monkeypatch)
    engine.instance_id = "food_container_1"
    ctx = ctx_for(engine, stop_at="Pressing the button")
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert engine._current.cancelled.is_set() and sent == [signal.SIGINT]
    assert time.monotonic() - t0 < 6.0
    stopping = [p for p in ctx.panels if p["message"] == node.PANEL_CANCELLED]
    assert stopping and all(p["buttons"] == [] for p in stopping)
    notice = ctx.panels[-1]
    assert notice["title"] == f"{node.PANEL_TITLE}: stopped" and ctx.buttons(notice) == ["done"]
    assert (
        notice["message"]
        == (  # no grip yet: no lid
            "Stopped. The arm is holding still where it was. "
            "Check where the arm and the container are before trying again."
        )
    )
    assert ctx.objects == {} and ctx.world == []


@pytest.mark.parametrize(
    "rows, stop_at, said",
    [
        # the grip's own check said it holds the lid
        (FULL_RUN[:8] + [FULL_RUN[10]], "Holding the lid", "The lid is still in the fingers."),
        # the fingers closed, the check not in yet
        (FULL_RUN[:8], "Gripping the lid", "The lid may still be in the fingers."),
    ],
)
def test_stopped_with_the_lid_says_it_may_still_be_in_the_fingers(engine, monkeypatch, rows, stop_at, said):
    given(monkeypatch, rows, wait=True)
    interrupts(monkeypatch)
    ctx = ctx_for(engine, stop_at=stop_at)
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert ctx.panels[-1]["message"].endswith(f"{said} {node.LID_NEXT_STEP}")


def test_an_orchestrator_cancel_shows_no_notice(engine, monkeypatch):
    """The orchestrator's own cancel (not Stop): the run stops its own way, and the orchestrator
    has the UI back, so no Stopped notice waits for a tap."""
    given(monkeypatch, FULL_RUN[:5], wait=True)
    sent = interrupts(monkeypatch)
    ctx = ctx_for(engine)
    threading.Timer(1.0, engine._current.cancelled.set).start()
    with pytest.raises(Cancelled):
        engine.execute("open_box", {}, ctx)
    assert sent == [signal.SIGINT]
    assert all(ctx.buttons(p) != ["done"] for p in ctx.panels)
    assert ctx.panels[-1]["message"] == node.PANEL_CANCELLED


@pytest.mark.parametrize(
    "say, code, error_code",
    [
        (["[press_demo] NO BOX: the scene camera found no container"], 2, "not_found"),
        (["[press_demo] STOP: pressed, but the knob did not pop"], 1, "not_popped"),
        (["[press_demo] STOP: the lid slipped out of the fingers on the way home"], 1, "lid_dropped"),
        (["Traceback (most recent call last):"], 1, "failed"),
    ],
)
def test_an_error_shows_its_code_in_a_persons_words_with_done(engine, monkeypatch, say, code, error_code):
    given(monkeypatch, FULL_RUN[:2], say, code=code)
    ctx = ctx_for(engine)
    with pytest.raises(EngineError) as raised:
        engine.execute("open_box", {}, ctx)
    assert raised.value.error_code == error_code
    notice = ctx.panels[-1]
    assert notice["title"] == f"{node.PANEL_TITLE}: failed" and ctx.buttons(notice) == ["done"]
    assert notice["message"].startswith(node.ERROR_TEXT[error_code])
    assert notice["message"].endswith(node.NEXT_STEP[error_code])


def test_every_error_code_in_the_interface_has_words():
    from pathlib import Path

    doc = (Path(node.__file__).resolve().parents[1] / "docs" / "interface.md").read_text()
    table = doc.split("| Code ", 1)[1].split("\n\n", 1)[0]
    codes = [row.split("`")[1] for row in table.splitlines()[2:]]
    assert codes and sorted(codes) == sorted(node.ERROR_TEXT) == sorted(node.NEXT_STEP)


def test_a_refusal_before_anything_ran_says_not_started(ros, tmp_path, monkeypatch):
    handlers = list(LOGGER.handlers)
    cap = make_engine(tmp_path, monkeypatch, allow_execute=False)
    cap._assigned_token = TOKEN
    try:
        ctx = UICtx()
        with pytest.raises(EngineError, match="allow_execute"):
            cap.execute("open_box", {}, ctx)
        assert len(ctx.panels) == 1 and ctx.panels[0]["title"] == f"{node.PANEL_TITLE}: not started"
        assert ctx.panels[0]["message"] == f"{node.ERROR_TEXT['not_allowed']} {node.NEXT_STEP['not_allowed']}"
        assert "allow_execute" not in ctx.panels[0]["message"]  # no parameter names on the panel
        ctx = UICtx()
        with pytest.raises(EngineError, match="no params"):
            cap.execute("open_box", {"x": 1}, ctx)
        assert ctx.panels[-1]["message"] == f"{node.ERROR_TEXT['bad_params']} {node.NEXT_STEP['bad_params']}"
    finally:
        cap._assigned_token = bytes(16)
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_the_budget_stop_says_so_then_timeout_and_the_notice_is_not_in_the_budget(engine, monkeypatch, tmp_path):
    """A run at RUN_BUDGET_S is stopped its own way (one Ctrl+C); its panel says it is taking too
    long, with no button, and the notice is the timeout's. The notice comes after the run ended:
    a slow tap on Done neither stops a run nor turns a result into a timeout."""
    given(monkeypatch, FULL_RUN[:5], wait=True)
    sent = interrupts(monkeypatch)
    monkeypatch.setattr(node, "RUN_BUDGET_S", 0.5)
    ctx = ctx_for(engine)
    with pytest.raises(EngineError) as raised:
        engine.execute("open_box", {}, ctx)
    assert raised.value.error_code == "timeout" and sent == [signal.SIGINT]
    assert any(p["message"] == node.PANEL_TIMED_OUT and p["buttons"] == [] for p in ctx.panels)
    assert ctx.panels[-1]["message"] == f"{node.ERROR_TEXT['timeout']} {node.NEXT_STEP['timeout']}"
    # a full run inside its budget, Done tapped only well after the budget: still the lid off
    monkeypatch.delenv("WAIT")
    shutil.rmtree(tmp_path / "runs")  # the stand-in writes the same run log name: a new one
    given(monkeypatch, FULL_RUN)
    monkeypatch.setattr(node, "RUN_BUDGET_S", 3.0)
    engine._current = _Run("open_box", None)
    ctx = ctx_for(engine, auto_done=False)

    def tap_done_late():
        while not (ctx.panels and ctx.buttons(ctx.panels[-1]) == ["done"]):
            time.sleep(0.05)
        time.sleep(4.0)
        ctx.events.append({"type": "button", "button_id": "done"})

    threading.Thread(target=tap_done_late, daemon=True).start()
    t0 = time.monotonic()
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and out["duration_s"] < 3.0 and time.monotonic() - t0 > 4.0


def test_an_untapped_notice_ends_after_notice_timeout_s(engine, monkeypatch):
    """Nobody taps Done: execute() still returns the result, after NOTICE_TIMEOUT_S (a fake clock)."""
    given(monkeypatch, FULL_RUN)
    clock = [0.0]

    def sleep(dt):
        clock[0] += dt

    real = node.RunUI
    monkeypatch.setattr(node, "RunUI", lambda *a, **k: real(*a, clock=lambda: clock[0], sleep=sleep, **k))
    ctx = ctx_for(engine, auto_done=False)
    out = engine.execute("open_box", {}, ctx)
    assert out["lid_off"] and ctx.buttons(ctx.panels[-1]) == ["done"]
    assert clock[0] >= panels.NOTICE_TIMEOUT_S


def test_the_mock_shows_the_same_panels_and_says_mock(ros, tmp_path, monkeypatch):
    handlers = list(LOGGER.handlers)
    cap = make_engine(tmp_path, monkeypatch, mock=True)
    cap._assigned_token = TOKEN
    try:
        cap.instance_id = "c1"
        ctx = UICtx()
        out = cap.execute("open_box", {}, ctx)
        assert out["mock"] and out["lid_off"]
        shown = ctx.messages()
        assert shown[:-1] == [node.sentence(d) for _, _, d in node.MOCK_PHASES["open_box"]]
        assert shown[-1].startswith("Mock: Lid off") and shown[-1].endswith("No arm motion.")
    finally:
        cap._assigned_token = bytes(16)
        cap.destroy_node()
        LOGGER.handlers[:] = handlers


def test_open_box_reads_as_a_button(engine):
    """The orchestrator shows the Action's label on its button, not the action id."""
    assert [a.label for a in engine.actions] == ["Open Container"]


def test_the_class_has_a_name_without_a_space_for_a_list_typed_by_hand(engine):
    """interactive_perception_node's `add <instance_id> <perception_class>` is split on spaces: a "food
    container" cannot be typed into it. Both names are claimed, the segmenter's first (Chris, c184e67)."""
    classes = list(engine._manifest().perception_classes)
    assert classes == ["food container", "food_container"]
    typed = "add food_container_1 food_container".split()
    assert typed[2] in classes and len(typed) == 3


# -- the failure notice's words: what happened, the run's own reason, the lid, what to do --------------------------


class Lid:
    def __init__(self, where=""):
        self.where = where

    def lid_in_fingers(self):
        return self.where


def notice_for(code, lines, lid=""):
    return node.error_text(node.error_for("open_box", code, lines), Lid(lid))


def test_a_stop_says_the_step_that_stopped_and_what_the_driver_meant_in_plain_words():
    text = notice_for(
        1, ["[press_demo] STOP: the way in stopped during approach (-11 halted: unexpected contact) " "— the arm holds"]
    )
    assert text.startswith(node.ERROR_TEXT["arm_holds"])
    assert "Reason: the move in to the button stopped during approach." in text
    assert "unexpected push or pull" in text
    assert text.endswith(node.NEXT_STEP["arm_holds"])
    for jargon in ("-11", "halted", "[press_demo]", "STOP:", "—"):
        assert jargon not in text


def test_a_planner_refusal_quoted_by_the_run_is_said_without_its_status():
    text = notice_for(
        1,
        [
            "[press_demo] STOP: no reach-out press plans for the scene camera's box (plan_to_pose: "
            "MotionGenStatus.IK_FAIL: no collision-free joint solution AT the goal); nothing moved"
        ],
    )
    assert "no reach-and-press move could be planned for the container the scene camera saw." in text
    assert "can't reach the position it needs" in text
    assert "MotionGenStatus" not in text and "nothing moved;" not in text
    assert text.endswith(node.NEXT_STEP["no_press"])


def test_the_run_saying_why_it_did_not_start_is_on_the_notice():
    text = notice_for(
        1, ["[lid_room_view] not starting: the scene camera is not calibrated " "(scripts/calibrate_scene_camera.py)"]
    )
    assert "Reason: the scene camera is not calibrated." in text and "scripts/" not in text


def test_the_lid_in_the_fingers_comes_first_in_what_to_do():
    text = notice_for(
        1, ["[press_demo] STOP: the way up is not a straight line the arm can fly (x) — the arm holds"], lid="is"
    )
    assert "The lid is still in the fingers." in text and text.endswith(node.LID_NEXT_STEP)
    assert "can't be done as one straight move" in text


def test_a_last_line_that_is_not_a_stop_is_no_reason():
    # killed mid-run: its last line is progress, not why it ended
    text = notice_for(-9, ["[press_demo] grip closed"], lid="may be")
    assert "Reason" not in text and "grip closed" not in text
    assert notice_for(1, ["Traceback (most recent call last):"]) == (
        f"{node.ERROR_TEXT['failed']} {node.NEXT_STEP['failed']}"
    )


def test_codes_whose_words_say_it_all_add_no_reason():
    text = notice_for(1, ["[press_demo] STOP: pressed, but the knob did not pop"])
    assert text == f"{node.ERROR_TEXT['not_popped']} {node.NEXT_STEP['not_popped']}"


def test_stopped_says_what_to_do():
    assert node.stopped_text(Lid("is")).endswith(node.LID_NEXT_STEP)
    assert node.stopped_text(Lid()).endswith("Check where the arm and the container are before trying again.")
