"""What grab_container's UI panels say: each step of the grab in words, what the hand holds after a stop or failure,
and a failure in plain words (numbers kept) with what to do. The log keeps the full message; the panel never shows
planner or driver codes or parameter names.

Text only: nothing here decides whether to retry or what the action returns. No ROS imports (the UI tests load it).
"""

import re

from capability_ui import wording as W

TITLE = "Grab tupperware"
# The button on a tupperware tap. microwave_place's take_out is the demo's "Grab Tupperware"; this one pinches the
# brim (grab_container's ring detector), so it says so.
LABEL = "Grab Tupperware by Brim"
RETRY_LABEL = "Grab"
MOVING = "Grabbing the tupperware by its brim: the arm is moving."
DONE = "Holding the tupperware by its brim, a little above where it stood."
DONE_NOT_RAISED = "Holding the tupperware by its brim where it stood: the lift didn't run, so check the grip by eye."

# (phase as execute() reports it, gripper closing?) -> the progress panel's words and bar (%). `grasp` is reported
# twice: opening (10 %) and closing (70 %).
PHASES = {
    "home": ("Moving to the start position. The arm is moving.", 0.0),
    "locate": ("Finding the tupperware with the scene camera.", 2.0),
    "view": ("Moving the wrist camera above the tupperware. The arm is moving.", 5.0),
    "detect": ("Finding the tupperware's brim with the wrist camera.", 8.0),
    "open": ("Opening the gripper.", 10.0),
    "approach": ("Reaching in to the brim. The arm is moving.", 30.0),
    "close": ("Closing the gripper on the brim.", 70.0),
    "raise": ("Lifting the tupperware a little. The arm is moving.", 85.0),
}


def step(phase, pct=0.0):
    """The step of the grab a progress report (`phase`, `pct`) is: PHASES' key, or `phase` itself."""
    if phase == "grasp":
        return "close" if float(pct) >= 50.0 else "open"
    return phase


def progress(phase, pct=0.0, message=""):
    """(words, bar %) for the progress panel."""
    return PHASES.get(step(phase, pct), (W.sentence(message or phase), None))


def hand(step_):
    """What the hand holds after a stop or failure in `step_` (step(): the run's last progress report)."""
    notes = {
        "open": "The hand is above the tupperware; nothing is held.",
        "approach": "The open hand is at or near the tupperware's brim; nothing is held.",
        "close": "The gripper was closing on the brim: it may be holding the tupperware.",
        "raise": "The hand is holding the tupperware by its brim.",
    }
    return notes.get(step_, "Nothing has been grasped.")


def _lower(text):
    """`text` with a lower-case first letter, for use mid-sentence; an all-caps word (a code) is left as it is."""
    if not text or text[1:2].isupper():
        return text
    return text[:1].lower() + text[1:]


def reason(raw):
    """(what happened, what to do or None) for grab's failure message `raw` (node.py's EngineError, grasp.Abort's
    words)."""
    text = re.sub(r"\s*--\s*(the arm holds where it is|HOLDING HERE|arm not commanded|not grasping)\b\.?", "", str(raw))
    text = " ".join(text.split())

    def why(inner):
        return _lower(W.explain(inner)[0])

    def todo(inner):
        return W.explain(inner)[1]

    rules = (
        (
            r"^home: (.*)",
            lambda m: (f"The arm couldn't get to its start position: {why(m[1])}", todo(m[1])),
        ),
        (
            r"no /perception/object_detections",
            lambda m: (
                "The scene camera's object detector isn't sending anything.",
                "Check that the scene camera and its detector are running.",
            ),
        ),
        (
            r"no '[^']*' >= [\d.]+ in the scene camera",
            lambda m: (
                "The scene camera can't see the tupperware.",
                "Make sure the tupperware is in the scene camera's view, nothing in front of it.",
            ),
        ),
        (
            r"no scene camera colour \+ depth",
            lambda m: ("The scene camera isn't sending images.", "Check that the scene camera is running."),
        ),
        (
            r"(\d+) mask pixels with depth",
            lambda m: (
                "The scene camera can't measure how far away the tupperware is.",
                "Make sure the tupperware is in the scene camera's view, nothing in front of it.",
            ),
        ),
        (
            r"^view: (.*)",
            lambda m: (
                f"The arm couldn't move the wrist camera above the tupperware: {why(m[1])}",
                todo(m[1]) or "Move the wheelchair so the tupperware is within reach.",
            ),
        ),
        (
            r"no camera_info|tf \S+ <- .* unavailable",
            lambda m: ("The wrist camera isn't sending usable images.", "Check that the wrist camera is running."),
        ),
        (
            r"ring not seen stably \((\d+)/(\d+) good frames",
            lambda m: (
                f"The wrist camera couldn't find the tupperware's brim steadily ({m[1]} of {m[2]} frames).",
                "Make sure the tupperware's brim is in view of the wrist camera, with nothing on top of it.",
            ),
        ),
        (
            r"frames disagree by (\d+) mm",
            lambda m: (
                f"The wrist camera's views of the brim disagreed by {m[1]} mm, so the hand didn't reach in.",
                "Keep the tupperware still.",
            ),
        ),
        (
            r"^approach: (.*)",
            lambda m: (f"The arm stopped on its way in to the brim: {why(m[1])}", todo(m[1])),
        ),
    )
    for pattern, make in rules:
        m = re.search(pattern, text)
        if m:
            return make(m)
    return W.explain(text)


def failure(raw, hand_note=""):
    """The failed panel's message (Done only): what happened, what the hand holds, what to do."""
    what, todo = reason(raw)
    return W.failure_text(what, hand_note, todo)


def retry(raw):
    """The retry panel's message (RunUI.retry adds "Press <label> to try again.")."""
    what, todo = reason(raw)
    return W.failure_text(what, "The arm is holding still. Nothing has been grasped", todo)


def unexpected(error, hand_note=""):
    """A failure that isn't one of grab's own (an arm error such as the e-stop, the detector's weights, or a bug)."""
    what, todo = W.explain(f"{type(error).__name__}: {error}")
    return W.failure_text(what, hand_note, todo)


def disabled():
    return (
        f"{TITLE} is turned off: this robot's grab_container capability was started with arm motion off. An "
        "operator has to restart it with arm motion on, for a supervised run."
    )
