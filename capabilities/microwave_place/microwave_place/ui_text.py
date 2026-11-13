"""What microwave_place's UI panels say when a grab (take_out) or a place doesn't finish: what happened in plain words
(numbers kept), what the hand holds and where (the node's _hand_note), and what to do. The log keeps the full message;
the panel never shows planner or driver codes or parameter names, and never hides the reason behind a generic sentence.

Text only: nothing here decides whether to retry or what the action returns. No ROS imports (the UI tests load it).
"""

import re

from capability_ui import wording as W


def _lower(text):
    """`text` with a lower-case first letter, for use mid-sentence; an all-caps word (a code) is left as it is."""
    if not text or text[1:2].isupper():
        return text
    return text[:1].lower() + text[1:]


def _places(where):
    """The phrases a message uses for the place of the run: `where` "microwave" or "fridge" names it; "any" (the UI's
    one grab, wherever the tupperware is: the node's site) names no place, as the node's _site_words and _hand_note."""
    if where in ("microwave", "fridge"):
        the = "the " + where
        return {
            "in": f" in {the}",
            "front": f"in front of {the}",
            "opening": f"{the}'s opening",
            "out": f"back out of {the}",
            "ahead": f"Move the wheelchair so {the} is straight ahead and within reach.",
            "seen": f"Make sure the container is in {the}, near the front, and in view of the wrist camera.",
            "nearer": f"Move the container nearer the front middle of {the}, or the wheelchair closer.",
        }
    return {
        "in": "",
        "front": "in front of the tupperware",
        "opening": "the tupperware",
        "out": "back from the tupperware",
        "ahead": "Move the wheelchair so the tupperware is within reach.",
        "seen": "Make sure the tupperware is in view of the wrist camera.",
        "nearer": "Move the tupperware or the wheelchair closer.",
    }


def reason(raw, where="microwave"):
    """(what happened, what to do or None) for the failure message `raw` of a run at `where` ("microwave",
    "fridge", or "any": the UI's grab, which names no place)."""
    text = " ".join(str(raw or "").split())
    at = _places(where)

    def why(inner):
        return _lower(W.explain(inner)[0])

    def todo(inner):
        return W.explain(inner)[1]

    rules = (
        # before anything moved: setup, perception, planning
        (
            r"rosbag|run bag|RecordingError",
            lambda m: (
                "The run recording couldn't start, so the action didn't run.",
                "An operator has to check the recording setup (disk space, the bag folder).",
            ),
        ),
        (
            r"SAM3 service \S+ not available within ([\d.]+) s",
            lambda m: (
                f"The object detector didn't respond in {m[1]} s.",
                "Check that the detector (SAM 3) is running.",
            ),
        ),
        (
            r"no SAM3 '?(\w+)'? frame in ([\d.]+) s",
            lambda m: (
                f"The scene camera's detector sent nothing in {m[2]} s.",
                "Check that the scene camera and the detector (SAM 3) are running.",
            ),
        ),
        (
            r"no (\d+) consecutive detected frames .*? in ([\d.]+) s \((\d+) of (\d+) SAM3 frames detected",
            lambda m: (
                f"The scene camera couldn't locate the microwave steadily (found in {m[3]} of {m[4]} frames in "
                f"{m[2]} s).",
                "Make sure the microwave is in the scene camera's view, its door fully open, and "
                "nothing in front of it.",
            ),
        ),
        (
            r"no cavity mask|cavity mask too small|depth points seen through the opening|best fit|known-size opening",
            lambda m: (
                "The scene camera couldn't make out the microwave's opening.",
                "Make sure the microwave door is fully open and nothing blocks the opening.",
            ),
        ),
        (
            r"no SAM3 'microwave' mask|depth points on SAM3's 'microwave' mask|'microwave' depth points at or behind",
            lambda m: (
                "The scene camera couldn't see the microwave's outline well enough.",
                "Make sure the whole microwave is in the scene camera's view.",
            ),
        ),
        (
            r"the container not seen in (\d+) agreeing frames within ([\d.]+) s \((\d+) tries",
            lambda m: (
                f"The wrist camera couldn't find the container{at['in']} ({m[3]} tries in {m[2]} s).",
                at["seen"],
            ),
        ),
        (
            r"no wrist camera_info|wrist depth \S+ px",
            lambda m: ("The wrist camera isn't sending usable images.", "Check that the wrist camera is running."),
        ),
        (
            r"scene depth encoding|SAM3's mask is \S+ px",
            lambda m: (
                "The scene camera's images don't match what the detector expects.",
                "An operator has to check the scene camera's settings.",
            ),
        ),
        (
            r"planner did not confirm (?:config layer (\S+)|the floor scene layer)",
            lambda m: (
                "The planner's model of the wheelchair, you and the floor isn't loaded, so the path couldn't be "
                "checked.",
                "Check that the planner is running with the robot's setup loaded.",
            ),
        ),
        (
            r"set_world\(.*?\) failed|worlds_dir|can't write cuRobo's world|can't save \S+ in",
            lambda m: (
                "The planner couldn't be given the microwave's model (a file couldn't be written or read).",
                "An operator has to check that the capability and the planner share the planner-world folder.",
            ),
        ),
        (
            r"take-out home failed: (.*)",
            lambda m: (
                f"The arm couldn't get to its starting position {at['front']}: {why(m[1])}",
                todo(m[1]) or "Move the arm a little clear (teleop).",
            ),
        ),
        (
            r"take-out approach: cuRobo refused: (.*)",
            lambda m: (
                f"The arm can't find a safe path to {at['opening']}: {why(m[1])}",
                at["ahead"],
            ),
        ),
        (
            r"pre-grasp is ([\d.]+) cm / ([\d.]+) deg off target",
            lambda m: (
                f"The arm couldn't line up in front of the container ({m[1]} cm / {m[2]} degrees off).",
                at["ahead"],
            ),
        ),
        (
            r"the line in ends ([\d.]+) cm / ([\d.]+) deg off the grasp|the arm can't go straight (\d+) cm of (\d+)|"
            r"can't go straight down",
            lambda m: (
                "The arm can't reach straight in to the container from here.",
                at["nearer"],
            ),
        ),
        (
            r"cuRobo exit turns the held container ([\d.]+) deg",
            lambda m: (
                f"The only way out found would tip the container {m[1]} degrees, so the grab didn't start.",
                None,
            ),
        ),
        (
            r"cuRobo exit|exit: trajectory|retreat endpoint",
            lambda m: (
                f"The arm can't find a safe way {at['out']} with the container.",
                at["ahead"],
            ),
        ),
        (
            r"nothing passed: (.*)",
            lambda m: (
                "No safe way to put the container in the microwave was found from here (every side shift and "
                "heading tried failed).",
                "Move the wheelchair so the microwave is straight ahead and within " "reach, with its door fully open.",
            ),
        ),
        (
            r"the arm is ([\d.]+) deg from where the plan starts",
            lambda m: (f"The arm moved after planning ({m[1]} degrees off), so nothing was sent.", None),
        ),
        # after the arm started
        (
            r"driver did not take the approach",
            lambda m: (
                "The arm driver refused the move; the arm may have a fault.",
                "Clear the fault on the arm, then try again.",
            ),
        ),
        (
            r"the hand is ([\d.]+) cm / ([\d.]+) deg from the target -- not lowering",
            lambda m: (
                f"The hand ended {m[1]} cm / {m[2]} degrees off its target, so it didn't lower the container.",
                None,
            ),
        ),
        (
            r"went down (\d+) cm without stopping on anything",
            lambda m: (f"The hand went down {m[1]} cm without touching the turntable, so it didn't let go.", None),
        ),
        (
            r"(\d+) cm above where the bottom meets the table -- it hit something else",
            lambda m: (
                f"The container touched something {m[1]} cm above the turntable, so the hand didn't let go.",
                "Check what is inside the microwave.",
            ),
        ),
        (
            r"the gripper only opened to ([\d.]+)",
            lambda m: (f"The gripper only opened part way ({m[1]}), so the hand didn't back out.", None),
        ),
        (
            r"the gripper only closed to ([\d.]+)",
            lambda m: (
                f"The gripper closed less than expected ({m[1]}): it may not have the container, so it didn't " "lift.",
                None,
            ),
        ),
        (
            r"the fingers are still moving",
            lambda m: ("The fingers were still moving after closing, so the arm didn't lift.", None),
        ),
        (
            r"([\d.]+) cm off the grasp",
            lambda m: (
                f"The hand ended {m[1]} cm from the grasp point{at['in']}, so the gripper didn't close.",
                None,
            ),
        ),
        (
            r"impedance hold (?:at the start )?ended ?(.*)|not still in impedance",
            lambda m: ("The arm didn't settle before lowering, so it stopped.", None),
        ),
        (
            r"the back-out ended ([\d.]+) cm from its planned end",
            lambda m: (f"Backing out ended {m[1]} cm from where it should have, so the arm stopped.", None),
        ),
        (
            r"(?:the tilt|the approach|cuRobo approach|cuRobo exit|[\w -]+) (?:failed|did not complete) \((.*)\)",
            lambda m: (f"The arm stopped part way: {why(m[1])}", todo(m[1])),
        ),
    )
    for pattern, make in rules:
        m = re.search(pattern, text)
        if m:
            return make(m)
    inner = re.sub(r"^STOPPED(?: [^:]*)?: ", "", text)
    return W.explain(inner)


def failure(raw, where="microwave", hand=""):
    """The failed panel's message (Done only): what happened, what the hand holds (`hand`, the node's _hand_note),
    what to do."""
    what, todo = reason(raw, where)
    return W.failure_text(what, hand, todo)


def retry(raw, where="microwave"):
    """The retry panel's message (RunUI.retry adds "Press <label> to try again.")."""
    what, todo = reason(raw, where)
    return W.failure_text(what, "The arm is holding still", todo)


def unexpected(error, hand=""):
    """A failure that isn't one of the action's own (an arm error such as the e-stop, or a bug)."""
    what, todo = W.explain(f"{type(error).__name__}: {error}")
    return W.failure_text(what, hand, todo)


def disabled(title):
    return (
        f"{title} is turned off: this robot's microwave_place capability was started with arm motion off. An "
        "operator has to restart it with arm motion on, for a supervised run."
    )


def fridge_unmodeled(title):
    return (
        f"{title} is turned off: the arm's path into the fridge isn't checked against the fridge, its shelves or "
        "the open door. An operator can turn it on only for a supervised run with a clear path."
    )
