"""capability_ui.wording: the arm driver's and the planner's failures in plain words, with a next step."""

import pytest

from capability_ui import wording as X


@pytest.mark.parametrize(
    "raw, words",
    [
        ("  RammpArm: go_to_start: arm is E-STOPPED -- clear it first -- nothing sent.", "emergency stop is pressed"),
        ("off the handle + round the door + push-open: STOPPED (HALTED halted: emergency stop)", "emergency stop"),
        ("swing: STOPPED (-11 halted: unexpected contact)", "unexpected push or pull"),
        ("plan_to_pose: MotionGenStatus.IK_FAIL: no collision-free joint solution AT the goal", "can't reach"),
        ("MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION", "starting from a position"),
        (
            "blend: LIBRARY_VALIDATION_FAILED: plan rejected by post-validation: "
            "arm_joint_5 velocity 1.709 > limit 1.222",
            "too fast",
        ),
        ("cuRobo found no collision-free path to the fridge start pose: MotionGenStatus.TRAJOPT_FAIL", "safe path"),
        ("/set_contact_thresholds does not answer", "didn't respond"),
        ("no /joint_states in 2 s -- is the arm driver up?", "didn't respond"),
        ("approach: goal REJECTED -- nothing moved", "didn't accept"),
        ("take-out home failed: PLANNING_FAILED", "safe path"),
    ],
)
def test_the_drivers_and_the_planners_failures_in_plain_words(raw, words):
    reason, todo = X.explain(raw)
    assert words in reason
    assert "MotionGenStatus" not in reason and "RammpArm" not in reason


def test_the_estop_wins_over_what_it_stopped():
    # a motion the e-stop halted also "stopped short": the e-stop is what the user acts on
    reason, todo = X.explain("the move to the fridge start pose stopped short (arm is E-STOPPED) -- path tolerance")
    assert "emergency stop" in reason and "Release the emergency stop" in todo


def test_a_message_no_rule_knows_is_shown_cleaned_not_hidden():
    reason, todo = X.explain("the door is already ~95 deg open (push to 90) -- the arm holds where it is")
    assert reason == "The door is already ~95 deg open (push to 90)."
    assert todo is None


def test_clean_cuts_a_long_message_at_a_word():
    text = X.clean("word " * 100)
    assert len(text) <= X.MAX_REASON_CHARS and text.endswith("...")


def test_clean_never_returns_nothing():
    assert X.clean("") == "Something went wrong."
    assert X.clean(" -- the arm holds where it is") == "Something went wrong."


def test_failure_text_joins_the_parts_as_sentences_and_leaves_out_empty_ones():
    assert X.failure_text("It stopped", "The arm is holding still", None) == "It stopped. The arm is holding still."
    assert X.failure_text("It stopped.", "", "Try again") == "It stopped. Try again."


def test_describe_takes_the_capabilitys_own_next_step_over_the_rules():
    text = X.describe("swing: STOPPED (-11 halted: unexpected contact)", "The hand may be on the handle.", "Do X.")
    assert text.endswith("The hand may be on the handle. Do X.")
    assert "unexpected push" in text
