"""open_box from a terminal (tasks/open_box_terminal): what the person is shown and what is sent (Session,
no ROS), the node's result read out of what the action client hands back (read_result, on the image's own
types), and a Ctrl+C followed to the node's result (follow, on a stand-in of the client's goal handle).
main() itself, with no node on the graph and against the mock node, is test_open_box_terminal_e2e (it
needs a ROS graph inside the test container, as test_node.py does)."""

import json
import queue

from box_opening import guide, line_up
from box_opening.tasks import open_box_terminal as term


def _session():
    shown, sent = [], []
    return term.Session(shown.append, sent.append), shown, sent


def test_it_shows_what_the_node_says_once_each_and_the_question_with_how_to_answer():
    """The node's progress comes five times a second while it watches the box: each thing it says is one
    line. A line-up's lines stand out from the run's own; the question says how to answer it."""
    session, shown, _sent = _session()
    session.accepted(True)
    for phase, pct, message in (
        ("locate", 0.0, "starting"),
        ("locate", 0.0, "starting"),
        (line_up.PHASE, 0.0, "Looking for the container"),
        (line_up.PHASE, 0.0, "Move closer, about 15 cm"),
        (line_up.PHASE, 0.0, "Move closer, about 15 cm"),
        (line_up.PHASE, 0.0, "Hold still, checking"),
        (line_up.PHASE, 0.0, guide.ASK),
        (line_up.PHASE, 0.0, guide.ASK),
        (line_up.PHASE, 0.0, "Opening the container"),
        ("press", 0.35, "pressing the button"),
        ("press", 0.35, ""),
    ):
        session.progress(phase, pct, message)
    assert shown == [
        "open_box sent: the node runs it. Type n and Enter to cancel at any time.",
        "     0%  starting",
        ">> Looking for the container",
        ">> Move closer, about 15 cm",
        ">> Hold still, checking",
        ">> Ready. Open the container?   y / n, then Enter",
        ">> Opening the container",
        "    35%  pressing the button",
    ]


def test_a_y_is_sent_only_while_the_question_stands_and_an_n_at_any_moment():
    """The node takes a y only on a fresh "Ready" whoever sends it; the terminal does not send one typed
    at any other time, and says so, so that nobody waits on an answer that went nowhere. n is the cancel."""
    session, shown, sent = _session()
    session.progress(line_up.PHASE, 0.0, "Move closer, about 15 cm")
    session.typed("y\n")
    assert sent == [] and shown[-1] == "   (nothing is asked yet: y is for the question, when it comes)"
    session.progress(line_up.PHASE, 0.0, guide.ASK)
    assert session.asking
    session.typed("maybe\n")
    session.typed("\n")  # a bare Enter says nothing
    assert sent == [] and shown[-1] == '   (type y or n, then Enter: "maybe" is neither)'
    session.typed(" Yes \n")
    assert sent == ["y"] and shown[-1] == "   sent: yes"
    session.progress(line_up.PHASE, 0.0, "Hold still, checking")  # the box rolled on: the question is gone
    assert not session.asking
    session.typed("y")
    assert sent == ["y"]
    session.progress("press", 0.35, "pressing the button")
    session.typed("n\n")
    assert sent == ["y", "n"] and shown[-1] == "   sent: no: cancelling"


def test_it_ends_with_the_result_in_a_persons_words():
    session, shown, sent = _session()
    payload = json.dumps({"lid_off": True, "lid_xyz": [0.597, -0.071, 0.209], "duration_s": 41.2})
    assert session.result(term.SUCCESS, "", False, payload) == 0
    assert shown[-1] == "DONE: the lid is off, set down at [0.60, -0.07] (41 s)"
    session.typed("y")  # after the end nothing is sent
    assert sent == [] and session.done
    session, shown, _sent = _session()
    assert session.result(term.FAILED, "not_lined_up", True, "") == 1
    assert shown[-2].startswith("FAILED: not_lined_up: not in place in time: nothing was pressed")
    assert shown[-1] == "   it can be started again"
    session, shown, _sent = _session()
    assert session.result(term.FAILED, "arm_holds", False) == 1
    assert "the arm holds where it stopped" in shown[-2] and "not recoverable" in shown[-1]
    session, shown, _sent = _session()
    assert session.result(term.CANCELLED) == 1 and shown[-1] == "CANCELLED: the run was stopped where it was"
    session, shown, _sent = _session()
    assert session.result(term.FAILED, "some_new_code", False, "not json") == 1
    assert shown[-2] == "FAILED: some_new_code"
    session.accepted(False)
    assert "refused the goal" in shown[-1]


def test_the_terminal_and_the_node_mean_the_same_question_and_the_same_answers():
    """The terminal knows the question by the guide's own words and phase, and sends the node's own two
    answers: nothing here is a second copy to drift."""
    from box_opening import node

    assert (line_up.PHASE, guide.ASK) == ("line_up", "Ready. Open the container?")
    assert (line_up.YES, line_up.NO) == ("y", "n")
    assert node.ANSWER_TOPIC == "~/answer" and term.NODE == "box_opening"
    assert "/box_opening/answer" in node.CONSOLE_HOW and "open_box_terminal" in node.CONSOLE_HOW
    setup = open("capabilities/box_opening/setup.py").read()
    assert '"open_box_terminal = box_opening.tasks.open_box_terminal:main"' in setup


def _wrapped(goal_status, **result):
    """What the action client hands back as a goal ends, built of the image's own types: the wrapper
    (ExecuteAction_GetResult_Response: the ACTION's status, and its result), the action's result
    (ExecuteAction.Result: one field, `result`) and the node's Result in it."""
    from rammp_orchestrator_interfaces.action import ExecuteAction
    from rammp_orchestrator_interfaces.msg import Result

    return ExecuteAction.Impl.GetResultService.Response(
        status=goal_status, result=ExecuteAction.Result(result=Result(**result))
    )


def test_the_result_is_read_out_of_the_wrapper_the_action_client_hands_back():
    """The review of 2026-11-17 ran the terminal against the mock node: every run, the successful ones
    too, ended `AttributeError: 'ExecuteAction_Result' object has no attribute 'status'`, exit 1. The
    action client's future gives a wrapper whose `.status` (how the action ended) and `.result` (the
    action's result, whose own `.result` is the node's Result) are separate, and the terminal read the
    node's fields one level too high. On the interface package's own types, as the engine fills them."""
    from action_msgs.msg import GoalStatus
    from rammp_orchestrator_interfaces.action import ExecuteAction
    from rammp_orchestrator_interfaces.msg import Result

    # the shape itself: what was read, and where the node's word is
    assert set(ExecuteAction.Result.get_fields_and_field_types()) == {"result"}
    assert not hasattr(ExecuteAction.Result(), "status") and hasattr(ExecuteAction.Result().result, "status")
    assert set(ExecuteAction.Impl.GetResultService.Response.get_fields_and_field_types()) == {"status", "result"}
    assert (term.SUCCESS, term.FAILED, term.CANCELLED) == (
        Result.STATUS_SUCCESS,
        Result.STATUS_FAILED,
        Result.STATUS_CANCELLED,
    )
    assert term.GOAL_SUCCEEDED == GoalStatus.STATUS_SUCCEEDED
    payload = json.dumps({"lid_off": True, "lid_xyz": [0.597, -0.071, 0.209], "duration_s": 41.2})
    done = _wrapped(GoalStatus.STATUS_SUCCEEDED, status=Result.STATUS_SUCCESS, payload_json=payload)
    assert term.read_result(done) == (term.SUCCESS, "", False, payload)
    session, shown, _sent = _session()
    assert session.result(*term.read_result(done)) == 0
    assert shown == ["DONE: the lid is off, set down at [0.60, -0.07] (41 s)"]
    # a failure with its code and whether it can be started again (the engine aborts the goal)
    failed = _wrapped(
        GoalStatus.STATUS_ABORTED, status=Result.STATUS_FAILED, error_code="not_lined_up", recoverable=True
    )
    assert term.read_result(failed) == (term.FAILED, "not_lined_up", True, "")
    # a cancel: the node's own (an n: the goal aborted, the Result cancelled) and the action's (Ctrl+C)
    for goal_status in (GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED):
        cancelled = _wrapped(goal_status, status=Result.STATUS_CANCELLED)
        session, shown, _sent = _session()
        assert session.result(*term.read_result(cancelled)) == 1
        assert shown == ["CANCELLED: the run was stopped where it was"]


def test_a_result_nobody_filled_in_is_never_read_as_done():
    """A blank Result reads status 0, success. An action that did not itself succeed and hands back such
    a Result (a goal the node dropped, a server that went) is a failure with no word of its own."""
    from action_msgs.msg import GoalStatus

    for goal_status in (GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_UNKNOWN):
        status, code, recoverable, payload = term.read_result(_wrapped(goal_status))
        assert (status, code, recoverable, payload) == (term.FAILED, term.NO_RESULT, False, "")
    session, shown, _sent = _session()
    assert session.result(*term.read_result(_wrapped(GoalStatus.STATUS_ABORTED))) == 1
    assert shown[0].startswith("FAILED: no_result: the action ended without a result of its own")


class _Future:
    def __init__(self):
        self._result, self._done = None, False

    def done(self):
        return self._done

    def result(self):
        return self._result

    def set(self, result):
        self._result, self._done = result, True


class _Handle:
    """rclpy's ClientGoalHandle as follow() uses it: the result's future, and the cancel's."""

    def __init__(self):
        self.ending, self.cancels = _Future(), 0

    def get_result_async(self):
        return self.ending

    def cancel_goal_async(self):
        self.cancels += 1
        return _Future()


def _following(ends_after=None, ctrl_c_at=(), typed_at=None, wait_s=45.0):
    """follow() on a stand-in handle, one turn of `spin` a tenth of a second on its clock. `ctrl_c_at`: the
    turns a Ctrl+C comes at; `ends_after`: how many turns after the cancel the node's result comes (None:
    never). Returns (what follow returned, the handle, what was shown, what was sent, the turns it took)."""
    session, shown, sent = _session()
    session.progress(line_up.PHASE, 0.0, "Move closer, about 15 cm")
    handle, stops, typed, turns, now = _Handle(), term.Stops(), queue.Queue(), [0], [0.0]
    cancelled_turn = []

    def spin(seconds):
        turns[0] += 1
        now[0] += seconds
        if turns[0] in ctrl_c_at:
            stops()  # as the signal's handler is called, between two turns
        if typed_at == turns[0]:
            typed.put("n\n")
        if handle.cancels and not cancelled_turn:
            cancelled_turn.append(turns[0])
        if ends_after is not None and cancelled_turn and turns[0] - cancelled_turn[0] >= ends_after:
            handle.ending.set("the node's result")

    got = term.follow(session, handle, spin, typed, stops, clock=lambda: now[0], wait_s=wait_s)
    return got, handle, shown, sent, turns[0]


def test_a_ctrl_c_is_the_cancel_once_and_the_nodes_result_is_waited_for():
    """The review, 2026-11-17: on Ctrl+C the terminal printed "no result from the node within 45 s of the
    cancel" 0.3 s later and left; the node's own stop takes seconds (the run's Ctrl+C, the driver's answer).
    The cancel goes out once and the loop goes on turning until the node's result is there, however long
    its stop takes inside RESULT_WAIT_S; the person is told the truth only: that it is waiting."""
    got, handle, shown, sent, turns = _following(ends_after=200, ctrl_c_at=(3,))  # its stop takes 20 s
    assert got == "the node's result" and handle.cancels == 1 and sent == []
    assert turns >= 203 and not any("no result" in line or "left without" in line for line in shown)
    assert [line for line in shown if "Ctrl+C" in line] == [
        "   Ctrl+C: cancelling. Waiting for the node to stop the run and say how it ended (Ctrl+C again leaves "
        "without that)"
    ]
    # an n typed while it waits is still sent (the node's own cancel: harmless twice), a result that comes
    # of itself needs no cancel
    got, handle, _shown, sent, _turns = _following(ends_after=5, ctrl_c_at=(3,), typed_at=4)
    assert got == "the node's result" and handle.cancels == 1 and sent == ["n"]
    assert term.RESULT_WAIT_S == 45.0  # more than the node's stop at its worst: 15 + 10 + 3 s (node.py)


def test_only_a_node_that_gives_no_result_in_time_or_a_second_ctrl_c_ends_the_wait_and_each_says_so():
    got, handle, shown, _sent, turns = _following(ends_after=None, ctrl_c_at=(3,), wait_s=45.0)
    assert got is None and handle.cancels == 1 and 450 <= turns - 3 <= 453  # 45 s of turns, not 0.3
    assert shown[-1] == "no result from the node within 45 s of the cancel: see its console"
    got, handle, shown, _sent, turns = _following(ends_after=None, ctrl_c_at=(3, 20))
    assert got is None and handle.cancels == 1 and turns == 20
    assert shown[-1].startswith("left without the node's result: the cancel was sent")
    # two Ctrl+C's before the cancel went out (a key held down) are the one cancel, and its result is waited for
    stops = term.Stops()
    stops(), stops()
    session, shown, _sent = _session()
    handle = _Handle()

    def spin(seconds, turns=[0]):  # noqa: B006 -- this call's own counter
        turns[0] += 1
        if turns[0] == 30:
            handle.ending.set("the node's result")

    assert term.follow(session, handle, spin, queue.Queue(), stops, clock=lambda: 0.0) == "the node's result"
    assert handle.cancels == 1 and not any("left without" in line for line in shown)
