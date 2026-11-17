"""fridge.node's UI buttons (UI_ACTIONS): the panels a press shows through ctx.update_user_panel(),
the wheelchair-move and retry questions, the Done notices, Stop (a cancel of the run), and that a
failure is still reported unchanged. A fake ctx (with a scripted user answering the panels), a fake
clock, a stubbed cancel_own_run and a scripted _execute stand in for the engine's UI-bridge
plumbing and cancel, time and the arm; nothing here starts a node or reaches the UI bridge, the arm
or the base."""

import importlib.util
import json
import sys
import threading
import time
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from door_common import common as dc
from fridge import close_door, constants, open_door, push_open
from fridge import constants as F
from fridge_fake import door_is_open, fridge  # noqa: F401 -- fridge: the fake-planner fixture

from capability_engine import Cancelled, EngineError
from capability_ui import DONE, STOP, STOPPED_MESSAGE, panel_open

fridge_node = None  # fridge.node, imported by the fixture below


@pytest.fixture(autouse=True)
def _fridge_node(monkeypatch):
    """fridge.node, without open3d / message_filters where the image has none (its perception
    imports them; nothing here uses the camera), as test_fridge_detection's fixture does. Its
    cancel_own_run (the engine's own CANCEL; it imports the ROS interfaces) is stubbed to cancel
    the press's ctx, as the engine's cancel does."""
    global fridge_node
    for name in ("open3d", "message_filters"):
        if importlib.util.find_spec(name) is None:
            monkeypatch.setitem(sys.modules, name, MagicMock())
    from fridge import node

    fridge_node = node
    monkeypatch.setattr(node, "cancel_own_run", lambda cap: setattr(cap.ui_ctx, "cancel", True))
    monkeypatch.setattr(push_open, "NEXT_MOVE", 0.0)
    monkeypatch.setattr(close_door, "LAST_CHAIR_BACK", None)
    return node


class Clock:
    def __init__(self):
        self.t = 0.0
        self.on_sleep = []

    def monotonic(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        for f in self.on_sleep:
            f(self.t)


@pytest.fixture
def clock(monkeypatch, _fridge_node):
    c = Clock()
    monkeypatch.setattr(
        fridge_node, "time", types.SimpleNamespace(monotonic=c.monotonic, sleep=c.sleep)
    )
    return c


class Run:
    """The engine's _Run, only what capability_ui.panel_open reads."""

    def __init__(self):
        self.ui_lock = threading.Lock()
        self.ui_goal_handle = None
        self.ui_opening = False


class Ctx:
    """ctx with a scripted user: each new panel with a button other than Stop takes the next of
    `answers` (a button_id, pressed 0.5 s after it shows; None: never pressed); the moving panel
    (Stop only) takes none. A pressed button ends the panel's goal, as the bridge does. `cancel`:
    the run is cancelled (checkpoint() raises Cancelled). `session` False: no UI session
    (update_user_panel is a no-op returning False)."""

    def __init__(self, clock, answers=(), session=True):
        self.clock, self.answers, self.session = clock, list(answers), session
        self.panels, self.cancel = [], False
        self._run = Run()
        self.pending = None  # (button_id, at)

    def cancelled(self):
        return self.cancel

    def checkpoint(self):
        if self.cancel:
            raise Cancelled()

    def update_user_panel(self, panel_json=None, objects=None, selected_instance_id=None):
        if not self.session:
            return False
        resend = bool(self.panels) and panel_json == self.panels[-1]  # sent again: no new answer
        self.panels.append(panel_json)
        self._run.ui_goal_handle = object()
        if any(b["button_id"] != STOP["button_id"] for b in panel_json["buttons"]) and not resend:
            answer = self.answers.pop(0) if self.answers else None
            self.pending = (answer, self.clock.t + 0.5) if answer else None
        return True

    def poll_ui_event(self):
        if self.pending and self.clock.t >= self.pending[1]:
            button, self.pending = self.pending[0], None
            self._run.ui_goal_handle = None
            return {"type": "button", "button_id": button}
        return None


class Log:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("info", m))

    def warning(self, m):
        self.lines.append(("warning", m))

    def error(self, m):
        self.lines.append(("error", m))


def engine(*outcomes, mock=False, allow_execute=True):
    """The real Capability's methods on an uninitialised instance (no node): _execute takes each
    of `outcomes` in turn -- a result dict, an exception, or a callable(action_id, params) giving
    either -- and records its calls in `.calls`."""
    cap = object.__new__(fridge_node.Capability)
    log = Log()
    cap.get_logger = lambda: log
    cap.log, cap.mock, cap.cfg, cap.calls = log, mock, {"allow_execute": allow_execute}, []
    queue = list(outcomes)

    def _execute(action_id, params, ctx):
        cap.calls.append((action_id, params))
        cap._drove = False  # as the real _execute: per goal attempt
        out = queue.pop(0)
        if callable(out) and not isinstance(out, BaseException):
            out = (
                out(action_id, params, cap)
                if getattr(out, "with_cap", False)
                else out(action_id, params)
            )
        if isinstance(out, BaseException):
            raise out
        return out

    cap._execute = _execute
    return cap


def press(cap, ui_id, ctx):
    cap.ui_ctx = ctx  # what the stubbed cancel_own_run cancels
    return cap.execute(ui_id, {}, ctx)


def messages(ctx):
    return [p["message"] for p in ctx.panels]


def asked_move(move):
    """An _execute outcome: the close refuses, having planned with the wheelchair moved `move`."""

    def out(action_id, params):
        close_door.LAST_CHAIR_BACK = move
        return EngineError("refused", "Move the WHEELCHAIR ... chair_moved true", recoverable=True)

    return out


def pulled_asking(move):
    """An _execute outcome: the open's pull is done and the push asks for the wheelchair moved."""

    def out(action_id, params):
        push_open.NEXT_MOVE = move
        return {"executed": True, "note": "Pull done ..."}

    return out


def stopped_while_moving(ctx):
    """An _execute outcome: Stop is pressed on the moving panel and the motion ends once the run is
    cancelled, as self.arm does (the Stop watcher runs in its own thread, on real time)."""

    def out(action_id, params):
        ctx.pending = ("stop", ctx.clock.t)
        end = time.time() + 2.0
        while not ctx.cancel and time.time() < end:
            time.sleep(0.005)
        assert ctx.cancel
        return Cancelled()

    return out


OK = {"executed": True, "note": ""}


def test_the_ui_is_offered_only_its_buttons(_fridge_node):
    cap = engine()
    offered = {a.action_id for a in cap.availability("fridge_1", {})}
    assert offered == set(fridge_node.UI_ACTIONS)


def test_each_button_runs_its_action_with_the_tested_settings(_fridge_node):
    ui = fridge_node.UI_ACTIONS
    # the start pose: the node's start_joints_deg (constants.START_JOINTS_DEG, or a profile's)
    assert ui["go_to_start_ui"] == ("go_to_start", "Go to Fridge", {"execute": True, "speed": 0.7})
    assert constants.START_JOINTS_DEG == (103.04, -30.7, -169.63, -123.05, 101.64, 110.04, 2.5)
    assert ui["open_door_ui"][2] == {"execute": True, "speed": 0.7}
    assert ui["open_door_drive_ui"][2]["speed"] == 0.7
    assert ui["close_door_ui"][2] == {
        "execute": True
    }  # the close's own 0.7: slower doesn't shut it
    for ui_id in ("open_door_drive_ui", "close_door_drive_ui"):
        assert ui[ui_id][2]["drive_base"] is True and ui[ui_id][2]["base_speed"] == 0.03
    declared = {a.action_id: a.label for a in fridge_node.Capability.actions}
    for ui_id, (action_id, label, _) in ui.items():
        assert declared[ui_id] == label and action_id in declared


def test_a_direct_action_shows_nothing(clock):
    cap, ctx = engine(OK), Ctx(clock)
    assert cap.execute("open", {"execute": False}, ctx) == OK
    assert cap.calls == [("open", {"execute": False})] and ctx.panels == []


def test_go_to_fridge(clock):
    cap, ctx = engine(OK), Ctx(clock, ["done"])
    assert press(cap, "go_to_start_ui", ctx) == OK
    assert cap.calls == [("go_to_start", {"execute": True, "speed": 0.7})]
    moving, done = ctx.panels
    assert moving == {
        "title": "Go to fridge",
        "message": "Moving the arm to the fridge.",
        "buttons": [STOP],
    }
    assert done["message"] == "The arm is in front of the fridge."
    assert done["buttons"] == [DONE] and "style" not in done
    assert clock.t < 1.0  # ended at Done


def test_open_with_no_wheelchair_move(clock):
    cap, ctx = engine(OK), Ctx(clock, ["done"])
    press(cap, "open_door_ui", ctx)
    assert messages(ctx) == ["Opening the fridge door. The arm is moving.", "Fridge door opened."]


def test_open_asks_for_the_wheelchair_move_then_pushes(clock):
    cap, ctx = engine(pulled_asking(0.05), OK), Ctx(clock, ["continue", "done"])
    assert press(cap, "open_door_ui", ctx) == OK
    assert cap.calls == [
        ("open", {"execute": True, "speed": 0.7}),
        ("open", {"execute": True, "speed": 0.7, "chair_moved": True}),
    ]
    ask = ctx.panels[1]
    assert "straight back, away from the fridge, about 5 cm" in ask["message"]
    assert "Continue to push the door open" in ask["message"]
    assert ask["buttons"] == [{"button_id": "continue", "label": "Continue"}, STOP]
    assert messages(ctx)[-1] == "Fridge door opened."


def test_stop_at_the_wheelchair_question_cancels_the_run(clock):
    cap, ctx = engine(pulled_asking(0.05)), Ctx(clock, ["stop"])
    with pytest.raises(Cancelled):
        press(cap, "open_door_ui", ctx)
    assert ctx.cancel  # through the engine's own cancel
    assert len(cap.calls) == 1


def test_stop_while_the_arm_moves_cancels_the_run_and_says_stopped(clock):
    """Stop on the moving panel cancels the run (the motion stops on it); the Stopped notice says
    the hand may be on the handle (an open), with Done."""
    ctx = Ctx(clock, ["done"])
    cap = engine(stopped_while_moving(ctx))
    with pytest.raises(Cancelled):
        press(cap, "open_door_ui", ctx)
    assert ctx.cancel and len(cap.calls) == 1
    stopped = ctx.panels[-1]
    assert stopped["title"] == "Open fridge: stopped"
    assert stopped["message"] == f"{STOPPED_MESSAGE} {fridge_node.UI_STOPPED_OPEN}"
    assert stopped["buttons"] == [DONE] and stopped["style"] == "warning"
    assert clock.t < 1.0  # ended at Done


def test_close_asks_for_its_planned_move_then_closes(clock):
    cap, ctx = engine(asked_move(0.12), OK), Ctx(clock, ["continue", "done"])
    assert press(cap, "close_door_ui", ctx) == OK
    assert cap.calls == [
        ("close", {"execute": True}),
        ("close", {"execute": True, "chair_moved": True}),
    ]
    assert "straight back, away from the fridge, about 12 cm" in ctx.panels[1]["message"]
    assert messages(ctx)[-1] == "Fridge door closed."


def test_close_forward_move(clock):
    cap, ctx = engine(asked_move(-0.08), OK), Ctx(clock, ["continue", "done"])
    press(cap, "close_door_ui", ctx)
    assert "straight forward, toward the fridge, about 8 cm" in ctx.panels[1]["message"]


def test_the_driving_buttons_never_ask_for_a_move(clock):
    """drive_base drives it; a refusal there is an ordinary refusal (try again)."""
    cap, ctx = engine(asked_move(0.12), OK), Ctx(clock, ["retry", "done"])
    press(cap, "close_door_drive_ui", ctx)
    assert cap.calls[0] == ("close", {"execute": True, "drive_base": True, "base_speed": 0.03})
    assert "chair_moved" not in cap.calls[1][1]
    assert not any("Move your wheelchair" in m for m in messages(ctx))


def test_a_refusal_offers_the_same_button_again(clock):
    cap, ctx = (
        engine(EngineError("refused", "NO DETECTION", True), OK),
        Ctx(clock, ["retry", "done"]),
    )
    assert press(cap, "open_door_ui", ctx) == OK
    retry = ctx.panels[1]
    assert retry["title"] == "Couldn't open the fridge"
    assert "Press Open Fridge Door to try again" in retry["message"]
    assert retry["buttons"] == [{"button_id": "retry", "label": "Open Fridge Door"}, STOP]
    assert "NO DETECTION" not in retry["message"]  # the log has it, not the UI
    assert len(cap.calls) == 2


def test_a_retry_panel_is_warning_styled(clock):
    cap, ctx = (
        engine(EngineError("refused", "NO DETECTION", True), OK),
        Ctx(clock, ["retry", "done"]),
    )
    press(cap, "open_door_ui", ctx)
    assert ctx.panels[1].get("style") == "warning"


def test_stop_instead_of_retry_cancels_the_run(clock):
    cap, ctx = engine(EngineError("refused", "x", True)), Ctx(clock, ["stop"])
    with pytest.raises(Cancelled):
        press(cap, "go_to_start_ui", ctx)
    assert ctx.cancel and len(cap.calls) == 1


@pytest.mark.parametrize(
    "error, ui_id, has, holding",
    [
        (
            EngineError("refused", "x -- stopping, the hand on the handle", True),
            "open_door_ui",
            "safety check",
            True,
        ),
        (
            EngineError("stopped", "the open stopped short", True),
            "open_door_ui",
            "stopped partway",
            True,
        ),
        (
            EngineError("stopped", "the close stopped short", True),
            "close_door_ui",
            "stopped partway",
            False,
        ),
        (
            EngineError("base_stopped", "x", True),
            "close_door_drive_ui",
            "Check where the wheelchair is",
            False,
        ),
    ],
)
def test_failures_that_are_not_retried(clock, error, ui_id, has, holding):
    cap, ctx = engine(error), Ctx(clock, ["done"])
    with pytest.raises(EngineError) as raised:
        press(cap, ui_id, ctx)
    assert raised.value is error  # reported unchanged
    failed = ctx.panels[-1]
    assert failed["style"] == "error" and failed["title"].startswith("Couldn't")
    assert has in failed["message"]
    assert ("still be holding the handle" in failed["message"]) == holding
    assert [b["button_id"] for b in failed["buttons"]] == ["done"]
    assert len(cap.calls) == 1


def test_a_bug_shows_the_generic_failure_and_is_reraised(clock):
    bug = KeyError("door_deg")
    cap, ctx = engine(bug), Ctx(clock, ["done"])
    with pytest.raises(KeyError) as raised:
        press(cap, "close_door_ui", ctx)
    assert raised.value is bug
    assert ctx.panels[-1]["message"] == fridge_node.UI_ERROR_OTHER


def test_a_cancel_shows_no_failure(clock):
    """An orchestrator cancel: only the moving panel was shown; no failure or Stopped notice."""
    cap, ctx = engine(Cancelled()), Ctx(clock)
    with pytest.raises(Cancelled):
        press(cap, "open_door_ui", ctx)
    assert [p["buttons"] for p in ctx.panels] == [[STOP]]


def test_execution_off_says_so_and_runs_nothing(clock):
    cap, ctx = engine(allow_execute=False), Ctx(clock, ["done"])
    with pytest.raises(EngineError) as raised:
        press(cap, "open_door_ui", ctx)
    assert raised.value.error_code == "execution_disabled" and cap.calls == []
    assert "allow_execute is false" in ctx.panels[-1]["message"]
    assert "holding the handle" not in ctx.panels[-1]["message"]  # nothing moved


@pytest.mark.parametrize(
    "code, message, holding",
    [
        ("stopped", "the open stopped short", True),
        ("refused", "swing re-check failed -- stopping, still holding the handle.", True),
        ("refused", "NO DETECTION", False),
        ("refused", "Did not reach the line-up point -- ABORT before touching the handle", False),
        ("bad_params", "x", False),
        ("base_stopped", "x", False),
        ("no_answer", "x", False),
    ],
)
def test_which_open_failures_may_hold_the_handle(_fridge_node, code, message, holding):
    cap = fridge_node.Capability
    assert cap._ui_maybe_holding("open", EngineError(code, message, True)) is holding
    assert cap._ui_maybe_holding("open", None) is True  # a bug: can't tell
    assert cap._ui_maybe_holding("close", EngineError("stopped", "x", True)) is False


def test_mock_says_no_arm_motion(clock):
    cap, ctx = engine({"executed": False, "mock": True}, mock=True), Ctx(clock, ["done"])
    press(cap, "open_door_ui", ctx)
    assert messages(ctx)[-1] == "Mock: fridge door opened. No arm motion."


def test_mock_go_to_fridge_plans_only_and_says_so(clock):
    """In mock go_to_start really plans, as a dry run (allow_execute false): not 'in front of'."""
    dry = {"executed": False, "note": "allow_execute is false: ran as a DRY RUN. dry run: ..."}
    cap, ctx = engine(dry, mock=True, allow_execute=False), Ctx(clock, ["done"])
    press(cap, "go_to_start_ui", ctx)
    assert messages(ctx)[-1] == "Mock: the move to the fridge planned. No arm motion."


def test_no_ui_session(clock):
    """Standalone: nothing shown or waited for; the flow ends as the direct action would."""
    cap = engine(OK)
    assert press(cap, "go_to_start_ui", Ctx(clock, session=False)) == OK
    cap = engine(pulled_asking(0.05))
    assert press(cap, "open_door_ui", Ctx(clock, session=False))[
        "executed"
    ]  # its note says the move
    assert len(cap.calls) == 1
    close_refusal = EngineError("refused", "Move the WHEELCHAIR ...", True)
    cap = engine(lambda a, p: (setattr(close_door, "LAST_CHAIR_BACK", 0.1), close_refusal)[1])
    with pytest.raises(EngineError) as raised:
        press(cap, "close_door_ui", Ctx(clock, session=False))
    assert raised.value is close_refusal
    assert clock.t == 0.0


def test_a_cancel_at_a_question(clock):
    cap, ctx = engine(pulled_asking(0.05)), Ctx(clock, [None])
    clock.on_sleep.append(lambda t: setattr(ctx, "cancel", ctx.cancel or t >= 2.0))
    with pytest.raises(Cancelled):
        press(cap, "open_door_ui", ctx)


def test_a_dropped_question_is_sent_again(clock):
    cap, ctx = engine(pulled_asking(0.05), OK), Ctx(clock, [None, "done"])
    asks = []

    def drop_then_answer(t):
        if 1.0 <= t < 1.1:  # the bridge drops it (ui_control reclaimed, a restart)
            ctx._run.ui_goal_handle = None
        if t >= 3.0 and ctx.pending is None and not asks:
            asks.append(t)
            ctx.pending = ("continue", t)

    clock.on_sleep.append(drop_then_answer)
    press(cap, "open_door_ui", ctx)
    resent = [p for p in ctx.panels if "Move your wheelchair" in p["message"]]
    assert len(resent) >= 2  # shown, then sent again after the drop
    assert cap.calls[-1][1].get("chair_moved") is True


def test_a_notice_waits_at_most_its_timeout(clock):
    cap, ctx = engine(OK), Ctx(clock, [None])
    press(cap, "open_door_ui", ctx)
    assert constants.UI_NOTICE_TIMEOUT_S <= clock.t < constants.UI_NOTICE_TIMEOUT_S + 0.2


def test_a_question_with_no_answer_fails_the_goal(clock):
    cap, ctx = engine(asked_move(0.1)), Ctx(clock, [None])
    with pytest.raises(EngineError) as raised:
        press(cap, "close_door_ui", ctx)
    assert raised.value.error_code == "no_answer"
    assert constants.UI_PROMPT_TIMEOUT_S <= clock.t < constants.UI_PROMPT_TIMEOUT_S + 0.2


def test_a_failure_panel_that_raises_never_hides_the_failure(clock):
    error = EngineError("stopped", "x", True)
    cap, ctx = engine(error), Ctx(clock)

    def broken(**kw):
        if kw["panel_json"].get("style") == "error":
            raise RuntimeError("bridge client blew up")
        return True

    ctx.update_user_panel = broken
    with pytest.raises(EngineError) as raised:
        press(cap, "close_door_ui", ctx)
    assert raised.value is error
    assert any(level == "warning" and "blew up" in m for level, m in cap.log.lines)


def test_every_panel_matches_the_bridge_schema(clock):
    """rammp-UI-bridge's InteractionPanel: title, message, buttons (button_id, label; no per-button
    style any more), and the panel's optional style."""
    cap = engine(EngineError("refused", "x", True), pulled_asking(0.05), OK)
    ctx = Ctx(clock, ["retry", "continue", "done"])
    press(cap, "open_door_ui", ctx)
    for panel in ctx.panels:
        assert (
            {"title", "message", "buttons"}
            <= set(panel)
            <= {"title", "message", "buttons", "style"}
        )
        assert all(set(b) == {"button_id", "label"} for b in panel["buttons"])
        assert STOP in panel["buttons"] or panel["buttons"] == [DONE]  # Stop, or the final Done
        json.dumps(panel)


def test_panel_open_without_a_run_record_counts_as_open():
    """capability_ui.panel_open (was fridge.node._panel_open): no run record, nothing re-sent."""
    assert panel_open(types.SimpleNamespace()) is True
    assert not hasattr(fridge_node, "_panel_open")


def test_on_the_robot_already_at_the_pose_is_done(clock):
    """go_to_start already there doesn't execute: on hardware that is done, not 'Mock'."""
    there = {"executed": False, "note": "already at the fridge start pose ...: nothing moved"}
    cap, ctx = engine(there), Ctx(clock, ["done"])
    press(cap, "go_to_start_ui", ctx)
    assert messages(ctx)[-1] == "The arm is in front of the fridge."


def drove_then(error):
    """An _execute outcome: the base drove (drive_base), then `error`."""

    def out(action_id, params, cap):
        cap._drove = True
        return error

    out.with_cap = True
    return out


@pytest.mark.parametrize("ui_id", ["close_door_drive_ui", "open_door_drive_ui"])
def test_no_retry_once_the_wheelchair_drove(clock, ui_id):
    """A retry would drive the picked move again from where the chair now is (TESTING.md)."""
    error = EngineError(
        "refused", "after the wheelchair move the close no longer plans -- the arm holds.", True
    )
    cap, ctx = engine(drove_then(error)), Ctx(clock, ["done"])
    with pytest.raises(EngineError) as raised:
        press(cap, ui_id, ctx)
    assert raised.value is error and len(cap.calls) == 1
    failed = ctx.panels[-1]
    assert failed["style"] == "error" and [b["button_id"] for b in failed["buttons"]] == ["done"]
    assert "wheelchair has already moved" in failed["message"]


def test_base_stopped_after_driving_says_it_once(clock):
    error = EngineError("base_stopped", "x", True)
    cap, ctx = engine(drove_then(error)), Ctx(clock, ["done"])
    with pytest.raises(EngineError):
        press(cap, "close_door_drive_ui", ctx)
    assert "already moved" not in ctx.panels[-1]["message"]
    assert "Check where the wheelchair is" in ctx.panels[-1]["message"]


# -- what the UI relies on in the real open and close, on the fake planner (fridge_fake) --


def test_an_executed_close_refuses_naming_its_move_and_continue_closes(fridge):  # noqa: F811
    """The UI runs the close executing, with no dry run first: a needed move refuses before
    anything moves, with LAST_CHAIR_BACK set (the UI's question), and chair_moved then closes.
    11-12's close only offers a move whose after-move plan plans too."""
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    first = {}

    def unreachable(
        pos, quat, obstacles
    ):  # the door where it stands: no plan; moved with the chair: plans
        if not (isinstance(obstacles, list) and door_is_open(obstacles)):
            return False
        door = np.round(next(b for b in obstacles if b["name"] == "door")["position"], 4).tolist()
        return first.setdefault("door", door) == door

    arm.unreachable = unreachable
    with pytest.raises(dc.Refused, match="WHEELCHAIR") as refused:
        close_door.run(arm, planner, True)
    assert arm.sent == []
    move = close_door.LAST_CHAIR_BACK
    assert move == F.CLOSE_CHAIR_MOVES_M[1]
    error = EngineError("refused", str(refused.value), recoverable=True)
    assert fridge_node.Capability._ui_close_move("close", error, {"execute": True}) == move
    arm.unreachable = lambda *a: False
    assert close_door.run(arm, planner, True, chair_moved=True)  # Continue
    assert len(arm.sent) == 1


def test_an_executed_open_that_needs_a_move_reports_it_in_next_move(fridge):  # noqa: F811
    arm, planner, _ = fridge
    arm.unreachable = lambda pos, quat, obstacles: (
        isinstance(obstacles, list) and door_is_open(obstacles)
    )
    assert open_door.run(arm, planner, True)  # the pull, off the handle, then it asks
    move = open_door.LAST_PLAN["after"].chair_move
    assert move and push_open.NEXT_MOVE == pytest.approx(move)
    arm.unreachable = lambda *a: False
    assert open_door.run(arm, planner, True, chair_moved=True)  # Continue: the push
    assert push_open.NEXT_MOVE == 0.0  # reset each run


@pytest.fixture(autouse=True)
def _no_ui_wrap(monkeypatch):
    """Messages are compared whole here; capability_uis own tests cover the UIs line wrapping."""
    from capability_ui import panels as ui_panels

    monkeypatch.setattr(ui_panels, "WRAP_CHARS", 10**6)
