"""microwave/ui_text.py: each open / close failure the log names, in plain words with the numbers and a next step."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/capability_ui"))
spec = importlib.util.spec_from_file_location("microwave_ui_text", ROOT / "capabilities/microwave/microwave/ui_text.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

HOLDS = " -- the arm holds where it is"


@pytest.mark.parametrize(
    "raw, says, todo",
    [
        (
            "No microwave box from boxer (/boxer/detections) in 5 s -- is it running?" + HOLDS,
            "detector sent no microwave in 5 s",
            "boxer",
        ),
        (
            "No door in the scene camera after 3 tries -- stopping before any motion." + HOLDS,
            "didn't find the microwave door (3 tries)",
            "scene camera's view",
        ),
        (
            "No two of 4 detections agreed within 2 cm -- refusing." + HOLDS,
            "4 looks at the handle didn't agree",
            "still",
        ),
        ("Face yaw 17.3 deg > 15 deg -- implausible plane fit? Refusing." + HOLDS, "turned 17.3 degrees", "squarely"),
        (
            "GATE FAILED at grasp: range 0.912 (max 0.85), z 0.410 (0.20-0.60)" + HOLDS,
            "0.91 m away, at most 0.85 m",
            "closer",
        ),
        (
            "GATE FAILED at grasp: range 0.700 (max 0.85), z 0.710 (z 0.20-0.60)" + HOLDS,
            "height the arm can't use",
            None,
        ),
        ("Tracking 2.4 cm at the grasp -- ABORT, gripper untouched." + HOLDS, "2.4 cm from the handle", None),
        (
            "Gripper is OPEN -- nothing grasped. Refusing to swing." + HOLDS + " -- still holding the handle",
            "closed on nothing",
            None,
        ),
        (
            "push-open: 'over the door' stopped (swing: STOPPED (-11 halted: unexpected contact)) -- the arm holds "
            "where it is.",
            "unexpected push or pull",
            "fault",
        ),
    ],
)
def test_each_failure_in_plain_words(raw, says, todo):
    what, next_step = T.reason("open", raw)
    assert says in what
    if todo:
        assert todo in next_step
    for code in ("MotionGenStatus", "VIEW_JOINTS_DEG", "RammpArm", "cuRobo"):
        assert code not in what and code not in (next_step or "")


def test_a_plan_that_fails_carries_the_doors_own_placement_search():
    raw = (
        "no tilt and arm configuration gives a grasp + swing that passes the sim checks -- refusing before any "
        "motion. Start joints [1. 2.].\nWould work with the microwave moved: 5 cm closer to the arm (tilt 10 deg, "
        "swing 60 deg); or 5 cm to the arm's left (tilt 0 deg, swing 60 deg)." + HOLDS
    )
    what, todo = T.reason("open", raw)
    assert what.startswith("The arm can't find a safe way to open the microwave door from here")
    assert "It would work with the microwave 5 cm closer to the arm." in what
    assert todo == "Move the microwave 5 cm closer to the arm (or move the wheelchair the same distance the other way)."
    assert "Start joints" not in what


def test_no_placement_helps():
    raw = (
        "no grasp -- refusing before any motion.\nNo shift of the microwave tried (up to 10 cm) makes it work." + HOLDS
    )
    what, todo = T.reason("open", raw)
    assert "up to 10 cm wouldn't help" in what and "straight ahead" in todo


def test_a_stop_with_nothing_said_names_the_arms_error():
    what, todo = T.reason("close", "the close stopped short" + HOLDS, [], "arm is E-STOPPED -- clear it first")
    assert what.startswith("The arm stopped before the door was shut: the emergency stop is pressed")
    assert todo == "Release the emergency stop, then try again."


def test_a_close_the_gates_refused():
    what, todo = T.reason(
        "close", "the close stopped short" + HOLDS, ["\nno plan passes every gate -- refusing, arm " "untouched."]
    )
    assert "can't find a safe way to push the door shut" in what and "nothing is in front of it" in todo


def test_a_message_no_rule_knows_is_shown_not_hidden():
    what, todo = T.reason("open", "the door is already ~80 deg open" + HOLDS)
    assert what == "The door is already ~80 deg open." and todo is None


def test_failure_says_the_hand_is_on_the_handle():
    text = T.failure("open", "Gripper is OPEN -- nothing grasped." + HOLDS, holding=True)
    assert "still holding the handle" in text
