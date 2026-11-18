"""What fridge_peering_mode's UI panels say: the step the run is on, what the hand holds after a
Stop or a failure, and a failure in plain words (numbers kept) with what to do. The log keeps the
full message; the panel never shows TF frame names, parameter names or driver codes.

Text only: nothing here decides anything. No ROS imports (the UI tests load it).
"""

import re

from capability_ui import wording as W

TITLE = "Look inside the fridge"
MOVING = "Getting ready to look inside the fridge. The arm may move."
DONE_LOOKING = "Done looking inside the fridge. The arm is holding still in front of it."
DONE_PICKED = "Holding the container outside the fridge."
NOTHING_HELD = "Nothing has been grasped."


def phase_text(phase, pct=0.0, message=""):
    """The progress panel's message for a ctx.progress() of the run (`phase`, `pct`, its log
    `message`), or None to leave the panel as it is."""
    message = str(message or "")
    if phase == "scene_homing":
        return "Finding the fridge and moving the wrist camera in front of it."
    if phase == "homing":
        return "Moving the wrist camera in front of the fridge's back wall."
    if phase == "unwind":
        return "Turning the arm away from a joint limit first: the arm is moving."
    if phase == "searching":
        if message.startswith("fridge detected"):
            return "Fridge found. Let go of the joystick: the arm moves in front of it by itself."
        return (
            "The fridge isn't detected yet. Move the joystick to give the camera another view, "
            "then let go once the fridge is in view."
        )
    if phase == "teleop":
        if "tap" in message:
            return (
                "Move the joystick to look around. Hold still to find food, then tap a container "
                "to pick it up."
            )
        return "Move the joystick to look around. Let go to hold the camera still."
    if phase == "pick":
        if float(pct or 0.0) < 55.0:
            return "Finding the container and planning how to pick it up."
        return "Taking the container: the arm is moving."
    if phase in ("gripper", "approach", "reach", "close", "lift", "retract"):
        return "Taking the container: the arm is moving."
    if phase == "leave":
        return "Backing out of the fridge with the container: the arm is moving."
    return None


def hand_note(phase):
    """What the hand holds after a Stop or failure in `phase` (the run's last progress phase)."""
    notes = {
        "gripper": "The hand is open; nothing is held.",
        "approach": "The open hand is on its way to the container; nothing is held.",
        "reach": "The open hand is in the fridge; nothing is held.",
        "close": "The gripper was closing on the container in the fridge: it may be holding it.",
        "lift": "The hand is holding the container in the fridge.",
        "retract": "The hand is holding the container, part way out of the fridge.",
        "leave": "The hand is holding the container, part way out of the fridge.",
    }
    return notes.get(phase, NOTHING_HELD)


# node._error_code's arm errors (the driver's own exceptions, whatever their message says)
ARM_ERRORS = {
    "estopped": (
        "The emergency stop is pressed, so the arm stopped.",
        "Release the emergency stop, then try again.",
    ),
    "control_taken": (
        "Another program took control of the arm, so this one stopped.",
        "Check that no other program is controlling the arm, then try again.",
    ),
    "stream_closed": (
        "The arm stopped following the joystick: the arm driver closed its control stream.",
        "Check that the arm has no fault, then try again.",
    ),
}


def reason(raw, code=""):
    """(what happened, what to do or None) for a run's failure message `raw` (code `code`)."""
    if code in ARM_ERRORS:
        return ARM_ERRORS[code]
    text = " ".join(str(raw or "").split())

    def why(inner):
        what = W.explain(inner)[0]
        return what[:1].lower() + what[1:] if what[1:2].islower() else what

    rules = (
        (
            r"arm TF .* not available",
            lambda m: (
                "The robot's arm model isn't being published, so the camera's position is unknown.",
                "An operator has to check that the robot description is running.",
            ),
        ),
        (
            r"no usable fridge TF",
            lambda m: (
                "The fridge's back wall wasn't detected.",
                "Make sure the fridge door is open and the fridge is in view, then try again.",
            ),
        ),
        (
            r"camera looks straight up or down",
            lambda m: (
                "The fridge wasn't detected, and the camera points straight up or down, so it "
                "can't search.",
                "Move the arm so the camera looks ahead (teleop), then try again.",
            ),
        ),
        (
            r"homing: (.*)",
            lambda m: (
                f"The arm couldn't move in front of the fridge: {why(m[1])}",
                W.explain(m[1])[1],
            ),
        ),
        (
            r"STOPPED holding the container: (.*)",
            lambda m: (f"The arm stopped while leaving the fridge: {why(m[1])}", None),
        ),
    )
    for pattern, make in rules:
        m = re.search(pattern, text)
        if m:
            return make(m)
    if code == "pick_failed":  # microwave_place's take-out: its own words, naming the fridge
        from microwave_place import ui_text as take_out_text

        return take_out_text.reason(text, "fridge")
    return W.explain(text)


def failure(raw, hand="", code=""):
    """The failed panel's message (Done only): what happened, what the hand holds, what to do."""
    what, todo = reason(raw, code)
    return W.failure_text(what, hand, todo)


def unexpected(error, hand=""):
    """A failure that isn't one of the action's own (an arm error such as the e-stop, or a bug)."""
    what, todo = W.explain(f"{type(error).__name__}: {error}")
    return W.failure_text(what, hand, todo)


def done(result):
    """The Done notice for a successful run's result."""
    pick = result.get("pick") or {}
    if result.get("ended") == "picked" and pick.get("executed"):
        return DONE_PICKED
    return DONE_LOOKING


def disabled(title):
    return (
        f"{title} is turned off: this robot's fridge_peering_mode capability was started with "
        "arm motion off. An operator has to restart it with arm motion on, for a supervised run."
    )
