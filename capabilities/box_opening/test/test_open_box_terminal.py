"""open_box from a terminal (tasks/open_box_terminal.Session): what the person is shown and what is sent.
No ROS: main() is the wiring (the action client, the answer's publisher, the keyboard)."""

import json

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
