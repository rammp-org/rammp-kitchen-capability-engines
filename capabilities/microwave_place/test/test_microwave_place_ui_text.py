"""microwave_place/ui_text.py: each grab / place failure the log names, in plain words with the numbers and a next
step, at the right place (microwave or fridge)."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/capability_ui"))
spec = importlib.util.spec_from_file_location(
    "microwave_place_ui_text", ROOT / "capabilities/microwave_place/microwave_place/ui_text.py"
)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


@pytest.mark.parametrize(
    "raw, where, says, todo",
    [
        (
            "the container not seen in 3 agreeing frames within 20 s (12 tries; rejected: small x5)",
            "fridge",
            "couldn't find the container in the fridge (12 tries in 20 s)",
            "in the fridge, near the front",
        ),
        (
            "no 3 consecutive detected frames within 2.0 cm and 3.0 deg of their median in 15 s (4 of 30 SAM3 frames "
            "detected; last reasons: x)",
            "microwave",
            "found in 4 of 30 frames in 15 s",
            "door fully open",
        ),
        (
            "take-out home failed: plan_to_joints: MotionGenStatus.IK_FAIL: no collision-free joint solution",
            "fridge",
            "starting position in front of the fridge: the arm can't reach",
            None,
        ),
        (
            "take-out approach: cuRobo refused: MotionGenStatus.TRAJOPT_FAIL",
            "microwave",
            "safe path to the microwave's opening",
            "straight ahead",
        ),
        (
            "nothing passed: shift +0 heading +0 transit: x; shift +5 heading +0 line: y",
            "microwave",
            "No safe way to put the container in the microwave",
            "door fully open",
        ),
        (
            "STOPPED: the hand stopped at tool z=0.210, 4 cm above where the bottom meets the table -- it hit "
            "something else; NOT letting go (holding in impedance)",
            "microwave",
            "touched something 4 cm above the turntable",
            "inside the microwave",
        ),
        (
            "STOPPED, nothing moved: the driver did not take the approach (FAULTED). The driver rejects goals while "
            "the arm is FAULTED: clear the fault on the arm, then re-run the place",
            "microwave",
            "may have a fault",
            "Clear the fault",
        ),
        (
            "STOPPED: the approach did not complete (swing: STOPPED (-11 halted: unexpected contact))",
            "microwave",
            "stopped part way: the arm felt an unexpected push",
            "fault",
        ),
        ("the arm is 2.3 deg from where the plan starts (max 1): re-plan", "microwave", "2.3 degrees off", None),
    ],
)
def test_each_failure_in_plain_words(raw, where, says, todo):
    what, next_step = T.reason(raw, where)
    assert says in what
    if todo:
        assert todo in next_step
    for code in ("MotionGenStatus", "cuRobo", "SAM3", "STOPPED", "take_"):
        assert code not in what and code not in (next_step or "")


def test_a_message_no_rule_knows_is_shown_without_its_stopped_prefix():
    what, todo = T.reason("STOPPED in the oven: the shelf moved", "microwave")
    assert what == "The shelf moved." and todo is None


def test_failure_puts_the_hand_after_the_reason():
    text = T.failure(
        "STOPPED in the oven: the gripper only closed to 0.400 -- not lifting",
        "microwave",
        "The gripper was closing on the container in the microwave: it may be holding it.",
    )
    assert text.startswith("The gripper closed less than expected (0.400)")
    assert text.endswith("it may be holding it.")
