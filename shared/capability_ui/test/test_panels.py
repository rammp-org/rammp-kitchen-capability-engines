"""capability_ui against a fake ctx that behaves as the engine's: one held UI event, cancel sets a flag that
checkpoint() raises on, update_user_panel() False when there is no UI session. No ROS."""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capability_ui import panels  # noqa: E402
from capability_ui.panels import RunUI  # noqa: E402


class Cancelled(Exception):
    pass


class FakeCtx:
    def __init__(self, session=True):
        self.session = session
        self.sent = []
        self._event = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        self.sent.append({"panel": panel_json, "objects": objects, "selected": selected_instance_id})
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

    def panels(self):
        return [s["panel"] for s in self.sent]


class Clock:
    """Virtual time for waits; sleep() advances it and lets a scripted tap happen at a given time."""

    def __init__(self, ctx):
        self.t = 0.0
        self.ctx = ctx
        self.script = []  # (at, fn)

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        while self.script and self.script[0][0] <= self.t:
            self.script.pop(0)[1]()


@pytest.fixture(autouse=True)
def fast_watch(monkeypatch):
    monkeypatch.setattr(panels, "POLL_S", 0.005)


def make(session=True):
    ctx = FakeCtx(session)
    clock = Clock(ctx)
    ui = RunUI(ctx, "Do thing", cancel=ctx.cancel, clock=clock, sleep=clock.sleep)
    return ui, ctx, clock


def wait_until(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_moving_panel_has_stop_and_stop_cancels_the_motion():
    ui, ctx, _ = make()
    started = threading.Event()

    def motion():  # like self.arm: runs until ctx.cancelled(), then raises
        started.set()
        assert wait_until(ctx.cancelled)
        raise Cancelled()

    ctx_tap = threading.Timer(0.05, lambda: ctx.tap("stop"))
    ctx_tap.start()
    with pytest.raises(Cancelled):
        ui.motion("Moving.", motion, stopped_note=lambda: "The hand is still on the handle.")
    assert ctx.panels()[0]["buttons"] == [panels.STOP]
    assert ui.user_stopped and ctx.cancelled()
    stopped = dict(ctx.panels()[-1])
    stopped["message"] = stopped["message"].replace(f"</>\n<{panels.UI_TEXT_STYLE}>", " ")
    assert stopped["title"] == "Do thing: stopped"
    assert stopped["message"].startswith(panels.STOPPED_MESSAGE)
    assert stopped["message"].endswith("The hand is still on the handle.")
    assert stopped["buttons"] == [panels.DONE]


def test_stopped_notice_stays_up_until_done_although_the_run_is_cancelled():
    ui, ctx, clock = make()
    ctx.cancel()
    ui.user_stopped = True
    clock.script = [(5.0, lambda: ctx.tap("done"))]
    ui.notice("Stopped.")
    assert clock.t == pytest.approx(5.0, abs=0.2)


def test_orchestrator_cancel_ends_a_notice_and_a_motion_without_the_stopped_panel():
    ui, ctx, clock = make()

    def motion():
        ctx.cancel()  # the orchestrator's cancel, not our Stop
        raise Cancelled()

    with pytest.raises(Cancelled):
        ui.motion("Moving.", motion)
    assert not ui.user_stopped
    assert all(p["title"] != "Do thing: stopped" for p in ctx.panels())
    ui.notice("Done.")  # returns at once
    assert clock.t < 0.2


def test_success_returns_the_result_and_done_wait_is_bounded():
    ui, ctx, clock = make()
    assert ui.motion("Moving.", lambda: {"ok": 1}) == {"ok": 1}
    ui.done("Thing done.")
    assert ctx.panels()[-1] == {"title": "Do thing", "message": "Thing done.", "buttons": [panels.DONE]}
    assert clock.t == pytest.approx(panels.NOTICE_TIMEOUT_S, abs=0.2)  # nobody tapped: gave up, no error


def test_mock_done_message():
    ui, ctx, clock = make()
    clock.script = [(0.1, lambda: ctx.tap("done"))]
    ui.done("Thing done.", mock=True)
    assert ctx.panels()[-1]["message"] == "Mock: Thing done. No arm motion."


def test_ask_returns_the_answer_and_always_offers_stop():
    ui, ctx, clock = make()
    clock.script = [(2.0, lambda: ctx.tap("continue"))]
    assert ui.ask("Move the chair.", [{"button_id": "continue", "label": "Continue"}]) == "continue"
    assert [b["button_id"] for b in ctx.panels()[-1]["buttons"]] == ["continue", "stop"]


def test_stop_on_a_question_cancels_and_raises():
    ui, ctx, clock = make()
    clock.script = [(1.0, lambda: ctx.tap("stop"))]
    with pytest.raises(Cancelled):
        ui.ask("Move the chair.", [{"button_id": "continue", "label": "Continue"}])
    assert ui.user_stopped and ctx.cancelled()


def test_question_times_out_to_none():
    ui, ctx, clock = make()
    assert ui.ask("?", [{"button_id": "a", "label": "A"}], timeout_s=3.0) is None
    assert clock.t == pytest.approx(3.0, abs=0.2)


def test_retry():
    ui, ctx, clock = make()
    clock.script = [(0.5, lambda: ctx.tap("retry"))]
    assert ui.retry("Couldn't see the handle.", "Open Door") is True
    message = ctx.panels()[-1]["message"].replace(f"</>\n<{panels.UI_TEXT_STYLE}>", " ")
    assert message == "Couldn't see the handle. Press Open Door to try again."


def test_object_taps_go_to_on_tap_and_do_not_end_the_question():
    ui, ctx, clock = make()
    taps = []
    clock.script = [(0.5, lambda: ctx.tap_object("spot_3")), (1.0, lambda: ctx.tap("place"))]
    objects = [{"instance_id": "spot_3", "perception_class": "spot", "box": [0, 0, 1, 1], "source": "scene"}]
    answer = ui.ask("Tap a spot.", [{"button_id": "place", "label": "Place here"}], objects=objects, on_tap=taps.append)
    assert answer == "place" and taps == ["spot_3"]
    assert ctx.sent[-1]["objects"] == objects


def test_standalone_never_waits():
    ui, ctx, clock = make(session=False)
    assert ui.motion("Moving.", lambda: 7) == 7
    assert ui.ask("?", [{"button_id": "a", "label": "A"}]) is None
    ui.done("Done.")
    assert clock.t == 0.0
    assert ui._watch is None


def test_a_dropped_panel_is_resent_but_an_open_one_is_not():
    ui, ctx, clock = make()

    class Run:
        ui_goal_handle = object()

    ctx._run = Run()
    clock.script = [(3.05, lambda: setattr(Run, "ui_goal_handle", None)), (4.5, lambda: ctx.tap("a"))]
    ui.ask("?", [{"button_id": "a", "label": "A"}])
    # sent once; open for ~3 s (no re-send); then dropped: re-sent each second until the answer
    assert len(ctx.sent) == 2


def test_the_watcher_stops_reading_before_a_question():
    """A Continue tap meant for the question must not be eaten by the Stop watcher."""
    ui, ctx, clock = make()
    ui.moving("Moving.")
    ui.quiet()
    ctx.tap("continue")
    assert ui.ask("Continue?", [{"button_id": "continue", "label": "Continue"}]) == "continue"


def test_styles():
    ui, ctx, clock = make()
    ui.error("Broke.")
    assert ctx.panels()[-1]["style"] == "error" and ctx.panels()[-1]["title"] == "Do thing: failed"
    ui.done("Fine.")
    assert "style" not in ctx.panels()[-1]


def test_panel_open_takes_the_run_lock():
    class Run:
        ui_goal_handle = None
        ui_opening = True
        ui_lock = threading.Lock()

    class Ctx:
        _run = Run()

    assert panels.panel_open(Ctx())
    Run.ui_opening = False
    assert not panels.panel_open(Ctx())


def test_messages_are_wrapped_for_the_ui():
    ui, ctx, clock = make()
    long = "Couldn't open the fridge: NO DETECTION (1) after 8 retries -- the arm holds where it is."
    ui.send(long)
    sep = f"</>\n<{panels.UI_TEXT_STYLE}>"
    lines = ctx.panels()[-1]["message"].split(sep)
    assert len(lines) > 1 and all(len(line) <= panels.WRAP_CHARS for line in lines)
    assert " ".join(lines) == long


def test_wrap_keeps_paragraphs_and_long_words():
    sep = f"</>\n<{panels.UI_TEXT_STYLE}>"
    assert panels.wrap("a\n\nb") == f"a{sep}{sep}b"
    word = "x" * 60
    assert panels.wrap(word) == word
