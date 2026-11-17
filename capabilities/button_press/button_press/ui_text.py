"""What the UI panel says when a button press fails: what happened in plain words (numbers kept), what the arm and
fingertip are left like, and what to do next.

The run's own messages (press_button.py, press_arm.py, collision.py, perception.py) are written for its log: step
names, planner statuses, parameter names. This maps them to words for the person at the screen; anything it doesn't
know falls back on capability_ui.wording (the arm driver's and planner's failures), and failing that on the message
itself, cleaned: the panel never says less than the log. Nothing here decides anything. Nothing imports ROS.
"""

import re

from capability_ui import wording

# The steps before the fingertip goes near the panel: a failure in one has moved nothing yet (preflight may only have
# closed the gripper); after them the arm may have moved, and from "approach" on the fingertip may be at the panel.
NOT_MOVED = ("preflight", "detect")
AT_PANEL_STATE = "The fingertip may still be at the microwave's buttons, and the arm is holding still there."
AT_PANEL_TODO = "Move the arm clear of the microwave (teleop or Home) before running anything else."
HOLDING_STATE = "The arm is holding still where it stopped."
NOT_MOVED_STATE = "The arm hasn't moved."

_CAMERA_DOWN = "Check that the wrist camera is running, then try again."
_FACE = "Make sure the microwave's front faces the arm and nothing blocks the buttons, then try again."
_STILL = "Keep the wheelchair and the microwave still, then try again."
_TEAM = "This is a setup problem in the button press, not something to fix at the microwave: tell the robot team."

# A preflight problem (press_button.Run.preflight) -> (what happened, what to do)
PREFLIGHT = (
    (
        r"gripper (is open|must be CLOSED)",
        "The gripper is open, and the press uses the closed fingertips.",
        "Close the gripper (empty), then try again.",
    ),
    (r"no camera_info|no wrist colou?r image", "The wrist camera isn't sending images.", _CAMERA_DOWN),
    (
        r"tf .* unavailable",
        "The robot can't tell where the wrist camera is: the arm's position isn't being published.",
        "Check that the arm driver and its state publisher are running, then try again.",
    ),
)

# The run's other failures -> (what happened, what to do). `{n}`-style groups are filled from the match. First match
# wins: the most specific first.
RULES = (
    (
        r"gripper close failed|gripper did not close in (?P<s>[\d.]+) s",
        "The gripper didn't close (something between the fingers?).",
        "Make sure the gripper is empty and nothing is in its way, then try again.",
    ),
    (
        r"dome detector: nothing in (?P<n>\d+) frames \(TOO CLOSE",
        "The wrist camera was too close to the microwave to see its buttons ({n} frames tried).",
        "Move the wheelchair back from the microwave a little, then try again.",
    ),
    (
        r"dome detector: nothing in (?P<n>\d+) frames \(TOO FAR",
        "The wrist camera was too far from the microwave to see its buttons ({n} frames tried).",
        "Move the wheelchair a little closer to the microwave, then try again.",
    ),
    (
        r"dome detector: nothing in (?P<n>\d+) frames",
        "The wrist camera couldn't find the microwave's buttons ({n} frames tried).",
        _FACE,
    ),
    (
        r"dome detector: only (?P<k>\d+)/(?P<n>\d+) frames agree",
        "The buttons seemed to move between camera frames (only {k} of {n} agreed), so the press wasn't trusted.",
        _STILL,
    ),
    (
        r"dome detector: panel normal|front normal .* not near-horizontal|too close to vertical",
        "The camera's view of the microwave's front didn't make sense (it looked tilted).",
        _FACE,
    ),
    (
        r"scene camera: no frames in (?P<s>[\d.]+) s",
        "The scene camera sent no images in {s} seconds.",
        "Check that the scene camera is running, then try again.",
    ),
    (
        r"scene camera: found the microwave in (?P<k>\d+)/(?P<n>\d+) frames",
        "The scene camera found the microwave in only {k} of {n} frames.",
        "Make sure the microwave is in the scene camera's view and nothing blocks it, then try again.",
    ),
    (
        r"scene camera: unstable -- only (?P<k>\d+)/(?P<n>\d+) frames agree",
        "The microwave seemed to move between scene camera frames (only {k} of {n} agreed).",
        _STILL,
    ),
    (
        r"the pre-press spot puts the fingertip (?P<cm>[\d.]+) cm INTO the panel",
        "The press position would put the fingertip {cm} cm inside the microwave's panel, so the arm wasn't moved.",
        _TEAM,
    ),
    (
        r"wrist would turn (?P<deg>\d+) deg to the .* \(> (?P<max>\d+)\)",
        "Reaching the buttons would turn the wrist {deg} degrees, more than the {max} allowed.",
        "Turn the wheelchair so the arm faces the microwave squarely, then try again.",
    ),
    (
        r"collision check -- (?P<what>.+?) (?P<cm>[\d.]+) cm \(need >= (?P<need>[\d.]+)\)",
        "The planned motion would bring the arm within {cm} cm of something (it needs {need} cm), so it wasn't run.",
        "Check that nothing is near the microwave or the arm, or move the wheelchair a little further from the "
        "microwave, then try again.",
    ),
    (
        r"pre-check failed at point|goal fails the gates",
        "The planned motion didn't pass the arm's safety checks, so it wasn't run.",
        "Turn the wheelchair to face the microwave squarely at the usual distance, then try again.",
    ),
    (
        r"the arm moved (?P<deg>[\d.]+) deg since this leg was planned",
        "The arm moved {deg} degrees after the motion was planned, so it wasn't sent.",
        "Make sure nothing else is moving the arm, then try again.",
    ),
    (
        r"arm is (?P<mm>[\d.]+) mm from the pre-press spot",
        "The fingertip was {mm} mm off the spot in front of the button, so it didn't press.",
        None,
    ),
    (
        r"ended (?P<cm>[\d.]+) cm from the target",
        "The arm stopped {cm} cm short of where it was going.",
        "Check that nothing is blocking the arm.",
    ),
    (
        r"carries no arm efforts",
        "The arm isn't reporting its joint torques, which the press needs to feel the button.",
        "Check that the arm driver is running normally.",
    ),
    (
        r"can't write cuRobo's world|planner_world_dir .* not absolute|planner refused the microwave's collision world",
        "The planner couldn't load the microwave's model.",
        "Check that the planner is running and shares its world folder with the button press, then try again.",
    ),
    (
        r"no collision model yet|world has no microwave in it yet",
        "The microwave's model wasn't ready, so the arm wasn't moved.",
        "Try again.",
    ),
    (
        r"driver's model puts the tool",
        "The arm's model doesn't match the button press's own (its mount angle setting).",
        _TEAM,
    ),
    (r"no front model for target|dome detector knows", "The button press doesn't know that button.", _TEAM),
)

# Log phrases the panel says in its own words (the arm's state, step labels)
_NOISE = re.compile(
    r"\s*--\s*(HOLDING HERE|session closed, HOLDING HERE|arm not commanded|nothing (sent|commanded)|no session opened)"
    r"|^[a-z_ ]+: (?=[a-z])",
    re.IGNORECASE,
)


def _fill(text, match):
    return text.format(**{k: v for k, v in match.groupdict().items() if v is not None}) if text else text


def reason_and_todo(message):
    """(what happened, what to do or None) for a run's failure message, in plain words."""
    text = " ".join(str(message or "").split())
    if text.startswith("preflight failed"):
        problems = text.split(":", 1)[1] if ":" in text else ""
        found = [(r, t) for pattern, r, t in PREFLIGHT if re.search(pattern, problems)]
        if found:
            reasons = " ".join(dict.fromkeys(r for r, _ in found))
            todos = " ".join(dict.fromkeys(t for _, t in found if t))
            return f"The check before moving found a problem: {reasons}", todos or None
    for pattern, reason, todo in RULES:
        match = re.search(pattern, text)
        if match:
            return _fill(reason, match), _fill(todo, match)
    if any(re.search(pattern, text, re.IGNORECASE) for pattern, _, _ in wording.RULES):
        return wording.explain(text)  # the arm driver's or the planner's failure
    return wording.clean(_NOISE.sub("", text)), None  # no rule knows it: the message itself, log-only tails dropped


def state(phase, motion_failed=False):
    """What the panel says the arm is left like after a failure in step `phase`."""
    if phase in ("approach", "press", "return"):
        return AT_PANEL_STATE
    if phase in NOT_MOVED and not motion_failed:
        return NOT_MOVED_STATE
    return HOLDING_STATE


def failure_message(name, step_words, phase, message, motion_failed=False):
    """The failure panel's message for pressing `name`: "Couldn't press Start while <step>.", then what happened, the
    arm's state and what to do (at the panel: move the arm clear first)."""
    reason, todo = reason_and_todo(message)
    if phase in ("approach", "press", "return"):
        todo = AT_PANEL_TODO
    head = f"Couldn't press {name} while {step_words[0].lower() + step_words[1:]}." if step_words else ""
    return " ".join(p for p in (head, wording.failure_text(reason, state(phase, motion_failed), todo)) if p)


def bug_message(name, phase, error):
    """A failure the run didn't expect (a bug): said as one, with its own words kept short, and the state."""
    details = wording.clean(str(error)) if str(error).strip() else ""
    detail = f" (details: {details[:-1] if details.endswith('.') else details})" if details else ""
    todo = AT_PANEL_TODO if phase in ("approach", "press", "return") else "Tell the robot team; the log has the rest."
    return wording.failure_text(
        f"Couldn't press {name}: something unexpected went wrong in the button press software{detail}",
        state(phase, motion_failed=phase not in NOT_MOVED),
        todo,
    )


DISABLED = (
    "Arm motion is turned off for the button press on this robot, so nothing moved. To use it, the button press "
    "has to be restarted with arm motion allowed, with someone at the emergency stop."
)
NO_ANSWER = "No number of presses was chosen in time, so nothing was pressed. Tap the button again to start over."
