"""fridge.ui_text: what the fridge's UI panels say for each failure, from the messages the run raises and logs.

The raw messages below are the run's own, as its code formats them today; test_the_messages_it_reads_are_the_runs
checks the phrases ui_text keys on are still in that code, so a reworded log message fails here, not on the robot."""

from pathlib import Path

import pytest

from fridge import ui_text as T

FRIDGE = Path(__file__).resolve().parents[1] / "fridge"


class Err(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.error_code = code


TAIL = " -- the arm holds where it is"
OPEN_REFUSED = (
    "no grasp gives a whole open that cuRobo plans (21 tried): 12x swing: IK_FAIL -- refusing before any motion. "
    "Start joints [103.0 -30.7 -169.6 -123.1 101.6 110.0 2.5]."
)
PLACEMENT = (
    "\n  Wheelchair placement (handle at x 1.10 y -0.09, door yaw +1 deg; the open planned in sim with x 0.65-1.05, "
    "y -0.30 to 0.00, yaw -10 to +10): move the wheelchair ~5 cm closer to the fridge; turn the wheelchair ~4 deg so "
    "its front swings left."
)


def test_open_says_the_wheelchair_move_the_placement_search_planned():
    raw = (
        OPEN_REFUSED
        + PLACEMENT
        + (
            "\n  -> checked by planning it: the open plans with the fridge shifted [-0.1, 0.05] m -- move the "
            "wheelchair ~10 cm closer to the fridge and move the wheelchair ~5 cm to its right, then "
            "dry-run it again (3 moves tried, 41 s)" + TAIL
        )
    )
    text = T.failure("open", Err("refused", raw))
    assert text.startswith("The arm can't find a way to open the door from where the wheelchair is now.")
    assert "Move the wheelchair ~10 cm closer to the fridge and ~5 cm to its right." in text
    assert "checked that it can open the door from there" in text
    for internal in ("cuRobo", "IK_FAIL", "dry-run", "Start joints", "[-0.1"):
        assert internal not in text


def test_open_falls_back_on_the_sim_placement_advice():
    raw = OPEN_REFUSED + PLACEMENT + TAIL
    text = T.failure("open", Err("refused", raw))
    assert (
        "Try this: Move the wheelchair ~5 cm closer to the fridge, and turn the wheelchair ~4 deg so its front "
        "swings left."
    ) in text


def test_open_says_when_no_wheelchair_move_helps():
    raw = OPEN_REFUSED + (
        "\n  -> checked by planning it: no wheelchair move of up to 15 cm made the open plan (24 tried, 300 s) -- "
        "within that, the refusal is not about where the wheelchair is" + TAIL
    )
    text = T.failure("open", Err("refused", raw))
    assert "up to 15 cm in any direction wouldn't help" in text and "door is fully closed" in text


def test_the_retry_prompt_names_the_button_once():
    raw = OPEN_REFUSED + PLACEMENT + TAIL
    text = T.retry_prompt("open", Err("refused", raw), "Open Fridge Door")
    assert text.endswith("Then press Open Fridge Door to try again.")
    assert text.count("Open Fridge Door") == 1


def test_a_handle_out_of_reach_says_which_way_to_move():
    raw = "handle x=1.234 outside plausible (0.5, 1.15) -- refusing." + TAIL
    text = T.failure("open", Err("refused", raw))
    assert "too far ahead" in text and "Move the wheelchair ~8 cm closer to the fridge" in text


@pytest.mark.parametrize(
    "action_id, raw, words",
    [
        ("open", "NO DETECTION (1) after 3 retries: no handle", "couldn't find the fridge handle"),
        ("open", "No two of 3 detections agreed within 3 cm -- refusing.", "3 looks, none within 3 cm"),
        ("open", "approach: the arm is 1.2 deg from where it was planned from -- not sent", "moved 1.2 degrees"),
        ("open", "Tracking 2.5 cm at the grasp -- ABORT, gripper untouched.", "2.5 cm away from the handle"),
        ("open", "the door is already ~95 deg open (push to 90).", "already about 95 degrees open"),
        ("close", "/fridge_data/door.json has no opened door: the swing writes it.", "no record of an opened"),
        ("close", "the hand would tilt -- refusing, arm untouched.", "tilt the container"),
        ("close", "drive_base: 40 cm is over base_max_distance -- nothing moved.", "drive 40 cm"),
        (
            "go_to_start",
            "cuRobo found no collision-free path to the fridge start pose: MotionGenStatus.TRAJOPT_FAIL.",
            "safe path to its fridge start position",
        ),
        (
            "go_to_start",
            "cuRobo found no collision-free path to the fridge start pose: "
            "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION. The arm's start is inside ...",
            "touching an obstacle",
        ),
    ],
)
def test_each_refusal_in_plain_words(action_id, raw, words):
    text = T.failure(action_id, Err("refused", raw + TAIL))
    assert words in text
    assert "MotionGenStatus" not in text and "cuRobo" not in text and "_" not in text.replace("(allow_execute)", "")


def test_the_close_says_the_range_of_wheelchair_moves_it_tried():
    raw = (
        "no wheelchair move from 10 cm forward to 20 cm back lets the close plan (last: plan_to_pose: "
        "MotionGenStatus.IK_FAIL ...) -- refusing." + TAIL
    )
    text = T.failure("close", Err("refused", raw))
    assert "between 10 cm forward and 20 cm back" in text
    assert "Move the wheelchair back to where it was for the open" in text


def test_a_stopped_motion_says_so_and_what_the_hand_may_hold():
    text = T.failure("open", Err("stopped", "the open stopped short -- the arm holds where it is"), maybe_holding=True)
    assert text.startswith("The arm stopped. It is holding still. The hand may still be holding the handle.")
    assert text.endswith(T.HOLDING_TODO)


def test_the_estop_during_go_to_start():
    raw = (
        "the move to the fridge start pose stopped short (  RammpArm: go_to_start: arm is E-STOPPED -- clear it "
        "first -- nothing sent.) -- the arm holds where it is"
    )
    text = T.failure("go_to_start", Err("stopped", raw))
    assert "emergency stop is pressed" in text and "Release the emergency stop" in text


@pytest.mark.parametrize(
    "raw, words",
    [
        ("the base moved the wrong way (-1.5 cm): stopped", "moving the wrong way (-1.5 cm)"),
        (
            "the base did not cover 12 cm in 20 s: stopped after moving +3.0 cm (4 twists sent; ...)",
            "didn't drive the 12 cm in 20 s (it moved +3.0 cm)",
        ),
        ("the base ended +2.1 cm from the distance: the arm doesn't go on", "stopped +2.1 cm from where"),
    ],
)
def test_the_wheelchair_drive_says_what_it_did(raw, words):
    text = T.failure(
        "close", Err("base_stopped", raw + " -- the base is stopped and the arm holds where it is"), drove=True
    )
    assert words in text and "Check where the wheelchair is" in text
    assert text.count("Check where the wheelchair is") == 1


def test_anything_else_shows_its_own_reason_not_a_generic_sentence():
    text = T.failure("open", Err("refused", "the gripper reports 0.9 closed with nothing in it" + TAIL))
    assert "The gripper reports 0.9 closed with nothing in it." in text


def test_a_bug():
    assert "Something unexpected went wrong" in T.failure("close", None)


def test_the_messages_it_reads_are_the_runs():
    """The phrases ui_text keys on, in the fridge's own code (and door_common's): reword one there, update ui_text."""
    code = "".join(p.read_text() for p in FRIDGE.glob("*.py"))
    code += (FRIDGE.parents[2] / "shared/door_common/door_common/open_door.py").read_text()
    for phrase in (
        "refusing before any motion",
        "the open plans with the fridge shifted",
        "then dry-run it again",
        "Wheelchair placement (",
        "no wheelchair move of up to",
        "none of the first",
        "Once the error above is fixed: ",
        "outside plausible",
        "NO DETECTION",
        "detections agreed within",
        "deg from where it was planned from",
        "at the grasp -- ABORT",
        "the hand is off the handle (gripper open)",
        "has no opened door",
        "cm back lets the close plan",
        "the close doesn't plan with the wheelchair moved",
        "after the wheelchair move the close no longer plans",
        "the hand would tilt",
        "cm is over base_max_distance",
        "the base moved the wrong way",
        "cm from the distance",
        "no collision-free path to the fridge start pose",
        "move the wheelchair ",
    ):
        assert phrase in code, phrase
