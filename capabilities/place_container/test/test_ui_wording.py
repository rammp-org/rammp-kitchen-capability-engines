"""place_container's failures on the panel: plain words, the numbers kept, where things are, what to do -- never the
planner's or driver's raw words, exception class names or parameter names. The inputs are the messages the code
raises (plan.explain_failure / plan.reach_refusal called for real), so a change to them shows here."""

import re

import numpy as np
import pytest

from place_container import plan, ui

RAW_WORDS = re.compile(r"cuRobo|MotionGenStatus|IK_FAIL|\[driver|allow_\w+|\w+:=|Refused|Error\b|arm_mount|tool z=")


def plain(text):
    return text.replace("</>\n<body-sm-light>", " ")


def check_plain(text):
    assert not RAW_WORDS.search(text), text
    return text


@pytest.mark.parametrize(
    "raw, words, todo",
    [
        (
            "adjust: no arm pose puts the tool at [0.9, 0.1, 0.3] m (0.91 m from the base axis) -- out of reach or "
            "joint limits; move the spot or the start position",
            "out of the arm's reach (0.91 m",
            ui.TAP_OTHER,
        ),
        (
            "carry: no arm shape continues the line at 3/20 (tool at [0.5, 0.2, 0.4]): every elbow position is within "
            "4.0 cm of the chair and user boxes or needs a joint jump (last kept: link 4 3.1 cm from seat) -- nothing "
            "moved",
            "too close to the wheelchair",
            "further from the wheelchair",
        ),
        ("carry 4: the container would tilt 7.5 deg (max 5) -- nothing moved", "tilt 7.5 degrees", ui.TAP_OTHER),
        (
            "the gripper is at 0.01 (0 = open; under 0.05 holds nothing): it isn't holding the container. Close it on "
            "the lip, or set allow_empty_gripper for an empty test run",
            "isn't holding the container",
            "Put the",
        ),
        (
            "the hand is 23 deg from holding the container 30 deg down (max 15 to correct with it in hand) -- re-grip "
            "it closer",
            "23 degrees off",
            "Grip the container again",
        ),
        (
            "the marked spot is 95 min old -- the chair or table may have moved; mark it again",
            "95 minutes old",
            "mark the spot again",
        ),
        (
            "the arm moved 1.2 deg while the place was planned -- nothing moved; run it again",
            "moved while",
            "Keep the arm still",
        ),
        (
            "the arm can't be commanded now: arm is E-STOPPED. Next: clear that (e-stop, control), then run again",
            "emergency stop",
            "Release the emergency stop",
        ),
        ("give all of x, y and table_z, or none (the marked spot)", "settings placing can't use", "operator"),
        (
            "mock: a failure at 'lower', as asked (mock_fail). Next: run again without mock_fail",
            "the lower step",
            "without the test failure",
        ),
    ],
)
def test_place_containers_own_refusals_in_plain_words(raw, words, todo):
    said, then = ui.reason(raw)
    assert words in said and todo in (then or "")
    check_plain(f"{said} {then}")


def test_a_spot_the_lowering_cant_reach_keeps_its_numbers():
    raw = plan.reach_refusal(([0.5, 0.31],), 0.20, np.array([0.95, 0.10])) + " -- nothing moved"
    said, then = ui.reason(raw)
    assert "only lower the container to" in said and "0.96 m from the arm's base" in said and then == ui.TAP_OTHER


@pytest.mark.parametrize(
    "driver, words",
    [
        ("leg 2/3: MotionGenStatus.IK_FAIL: no collision-free joint solution", "can't reach"),
        ("go_to_ee_poses: arm is E-STOPPED -- nothing sent.", "emergency stop is pressed"),
        ("leg 1/2: INVALID_START_STATE_WORLD_COLLISION", "touching the wheelchair"),
        ("leg 1/1: goal pose COLLIDES with world", "too close to the wheelchair"),
        ("execute ended -11 halted: unexpected contact", "unexpected push"),
        ("leg 1/1: TRAJOPT_FAIL", "smooth path"),
        ("something no rule knows", "motion didn't complete"),
    ],
)
def test_a_motions_failure_reads_the_drivers_words_but_never_shows_them(driver, words):
    raw = "STOPPED: " + plan.explain_failure(driver, ["carry", "lower", "retreat"], {"carry": [0.6, 0.1, 0.4]})
    text = check_plain(ui.failure_message("place_here", raw))
    assert words in text


def test_the_state_says_where_the_container_is():
    assert "still in the hand" in ui.failure_message(
        "place_here", "STOPPED in the carry, nothing let go: x [driver: y]"
    )
    down = "STOPPED in the back-out (the container is down, the gripper open): x [driver: y]"
    assert "on the table and the gripper is open" in ui.failure_message("place_here", down)
    assert "Nothing moved: the container is still in the hand" in ui.failure_message(
        "place_here", "carry 4: the container would tilt 7.5 deg (max 5) -- nothing moved"
    )
    assert ui.failure_message("reset", "back to the start: a joint moves 200 deg").count("Nothing moved") == 1


def test_after_the_arm_moved_it_doesnt_ask_for_a_tap():
    raw = "STOPPED above the spot: " + plan.reach_refusal(([0.5, 0.31],), 0.20, np.array([0.95, 0.1]))
    text = ui.failure_message("place_here", raw + " in the posture cuRobo chose -- nothing let go")
    assert "Tap" not in text and "try again at a different spot" in text and "still in the hand" in text


def test_a_direct_action_has_no_picker_to_tap():
    raw = "carry 4: the container would tilt 7.5 deg (max 5) -- nothing moved"
    assert "Choose a different spot" in ui.failure_message("place", raw)


def test_the_pickers_note_says_why_and_leaves_the_tap_to_the_picker():
    note = ui.retry_note("adjust: no arm pose puts the tool at [0.9, 0.1, 0.3] m (0.91 m from the base axis) -- x")
    assert note == "Couldn't place it there: that spot is out of the arm's reach (0.91 m from the arm's base). "
    note = ui.retry_note("the arm moved 1.2 deg while the place was planned -- nothing moved; run it again")
    assert note.endswith("Keep the arm still, then try again. ")


@pytest.mark.parametrize(
    "problem, words",
    [
        ("the scene camera isn't ready (no depth or camera_info from the scene camera)", "isn't sending images"),
        ("the scene camera sees no table in reach", "Make sure the table"),
        ("too little of the table is seen", "blocking the camera's view"),
        ("", "no free space on the table"),
    ],
)
def test_no_spot_says_why_and_what_to_do(problem, words):
    text = ui.no_spot_message(problem, marked=False)
    assert words in text and "camera_info" not in text


def test_the_table_note_keeps_its_number():
    assert "3 cm higher or lower" in ui.table_note("the table there is 3 cm off the table's plane")
    assert ui.table_note("too few points") == "The table's height couldn't be measured at that spot. Tap another spot. "


def test_a_bug_and_the_switched_off_robot_say_no_internals():
    check_plain(ui.bug_message("place_here"))
    check_plain(ui.REFUSED_OFF_UI)
    assert "operator" in ui.bug_message("place_here") and "operator" in ui.REFUSED_OFF_UI
