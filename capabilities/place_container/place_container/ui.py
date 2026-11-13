"""What place_container shows on the user's UI panel (ctx.update_user_panel; the shapes are in
docs/UI_JSON_SCHEMA.md). Pure: no ROS, so it is tested on its own.

Every phase the action reports with ctx.progress() is also put on the panel, in words, with a
progress bar; every failure is put on it with an OK button before the action ends with the error
(the panel closes when the action ends, so an error not shown before then is never seen).

place_here (the UI's tap on a table) runs on capability_ui.RunUI instead: the spot picker, then the
place under a progress panel with Stop, then Stopped / failed / Done notices. Its words are here too.
"""

import re

from capability_ui import clean, failure_text, sentence, wrap  # wrap: lines Unreal can show (one rich-text run a line)
from capability_ui import explain as plain_explain

OK = "ok"  # the error panel's button
PLACE_HERE = "Place Tupperware on Table"  # the one-tap action's label on the UI's action list

# place_here's spot picker: its buttons (RunUI adds Stop)
PLACE = {"button_id": "place", "label": "Place here"}
MARKED = {"button_id": "marked", "label": "Use marked spot"}
LOOK = {"button_id": "look", "label": "Look again"}
PICK = "Tap where to put the container, then press Place here."
TAP_FIRST = "Tap a spot on the table first. "
MOVING = "Placing the container: the arm is moving. Press Stop to stop it."
PLACED = "The container is on the table and the arm is back at its start."
REFUSED_OFF = (
    "Placing is switched off on this robot (allow_execute is false). Start place_container with "
    "allow_execute:=true, with someone at the e-stop."
)
# the same refusal on the panel, for the person in the chair (REFUSED_OFF stays the action's error, for the log)
REFUSED_OFF_UI = (
    "Placing is switched off on this robot, so nothing moved. Ask the operator to turn on arm motion for placing, "
    "with someone at the emergency stop."
)

TITLES = {
    "place_here": "Placing the container",
    "place": "Placing the container",
    "mark": "Marking the spot",
    "retreat": "Backing the hand away",
    "reset": "Going back to the start",
}
FAILED_TITLES = {
    "place_here": "Couldn't place the container",
    "place": "Couldn't place the container",
    "mark": "Couldn't mark the spot",
    "retreat": "Couldn't back the hand away",
    "reset": "Couldn't go back to the start",
}
# A phase (what execute() reports with ctx.progress) -> what the user reads, and how far along the
# action is then (%). The carry's own legs (adjust, lift, transit, ...) are reported under their
# names when they run as separate moves.
PHASES = {
    "plan": ("Checking the plan", 5),
    "carry": ("Carrying the container over to the spot", 15),
    "adjust": ("Squaring the hand", 10),
    "lift": ("Lifting the container", 12),
    "transit": ("Carrying the container over to the spot", 20),
    "transit-reconfigure": ("Carrying the container over to the spot", 20),
    "lower": ("Lowering it onto the table", 45),
    "open": ("Letting go", 70),
    "retreat": ("Backing the hand away", 85),
    "return": ("Backing the hand away and going back to the start", 85),
    "reset": ("Going back to the start", 50),
}
# mock place: the phases it walks through, and how long it shows each (s)
MOCK_PHASES = ("carry", "lower", "open", "return")
MOCK_PHASE_S = 1.0


def title(action_id):
    return TITLES.get(action_id, action_id)


def progress_panel(action_id, phase, message=""):
    """The panel while `action_id` is in `phase`: its title, the phase in words, a progress bar.
    A phase not in PHASES shows `message` (or its own name) and no bar."""
    text, pct = PHASES.get(phase, (message or phase, None))
    panel = {"title": title(action_id), "message": wrap(text)}
    if pct is not None:
        # no pause control: a pause in the middle of the lowering or the retreat is not offered
        panel["progress"] = {"phase": phase, "pct": float(pct), "pausable": False}
    return panel


def done_panel(action_id, message):
    return {
        "title": title(action_id),
        "message": wrap(message),
        "progress": {"phase": "done", "pct": 100.0, "pausable": False},
    }


def error_panel(action_id, message):
    """The panel for a failed action: what went wrong (the refusal's own words: what happened,
    where, what to do next) and an OK button."""
    return {
        "title": FAILED_TITLES.get(action_id, f"{title(action_id)}: failed"),
        "message": wrap(message),
        "buttons": [{"button_id": OK, "label": "OK"}],
    }


def pick_message(n_spots, note=""):
    """The picker's text: `note` (why it is asking again) first."""
    return f"{note}{PICK} ({n_spots} free spots shown.)"


# Why the picker has no spot to offer (spots.Search.problem, node._find_spots): what happened, what to do
LOOK_AGAIN = "then press Look again"
_PROBLEMS = (
    (
        r"camera isn't ready",
        "The scene camera isn't sending images.",
        f"Check that the scene camera is running, {LOOK_AGAIN}.",
    ),
    (
        r"depth image is empty",
        "The scene camera's depth image is empty.",
        f"Check that the scene camera is running, {LOOK_AGAIN}.",
    ),
    (
        r"sees no table in reach",
        "The scene camera doesn't see a table within the arm's reach.",
        f"Make sure the table is in front of the arm and in the camera's view, {LOOK_AGAIN}.",
    ),
    (
        r"no flat table surface",
        "The scene camera can't find a flat table surface.",
        f"Make sure the table is in front of the arm and in the camera's view, {LOOK_AGAIN}.",
    ),
    (
        r"too little of the table",
        "The scene camera sees too little of the table.",
        f"Move anything blocking the camera's view of the table, {LOOK_AGAIN}.",
    ),
    (
        r"calibration|tilted",
        None,  # the problem's own words: how far off, and that it is the camera's mount
        "The camera's mount on the wheelchair may have moved: ask the operator to check it.",
    ),
)


def no_spot_message(problem, marked, note=""):
    """No spot to offer: why, and what the user can do."""
    text, todo = clean(problem), "Press Look again to try again."
    if not str(problem or "").strip():  # every spot in reach taken, or too near the edge
        text = "There is no free space on the table within the arm's reach."
        todo = f"Clear a space on the table, {LOOK_AGAIN}."
    for pattern, said, then in _PROBLEMS:
        if re.search(pattern, str(problem)):
            text, todo = (said or text), then
            break
    marked_too = " Or press Use marked spot to place at the spot marked earlier." if marked else ""
    return f"{note}No spot to offer. {text} {todo}{marked_too}"


def table_note(why):
    """The picker's note when the table's height can't be measured at the tapped spot (node._table_z's why)."""
    m = re.search(r"(\d+) cm off the table's plane", str(why))
    if m:
        return (
            f"The table at that spot is {m[1]} cm higher or lower than the rest of the table (something may be "
            "there). Tap another spot. "
        )
    return "The table's height couldn't be measured at that spot. Tap another spot. "


# --- failures in plain words -------------------------------------------------------------------------------------
# A refusal or a stop (plan.Refused / EngineError messages, written for the log) -> what happened, what to do.
# The first match wins. Each entry: (pattern, function of the match -> (what happened, what to do or None)).
TAP_OTHER = "Tap a different spot."
_FAILURES = (
    (
        r"allow_execute is false",
        lambda m: (
            "Placing is switched off on this robot.",
            "Ask the operator to turn on arm motion for placing, with someone at the emergency stop.",
        ),
    ),
    (
        r"^mock: a failure at '([^']+)'",
        lambda m: (f"Mock test: a failure at the {m[1]} step, as set up.", "Run it again without the test failure."),
    ),
    (
        r"isn't holding the container",
        lambda m: (
            "The gripper isn't holding the container: it is open.",
            "Put the container in the hand, gripped by its lip, then try again.",
        ),
    ),
    (
        r"the gripper is at [\d.]+, still closed",
        lambda m: (
            "The gripper is still closed, so backing the hand away would drag the container.",
            "Open the gripper, then back the hand away.",
        ),
    ),
    (
        r"the gripper only opened to ([\d.]+)",
        lambda m: (
            "The gripper only opened partway, so the hand didn't back away from the container.",
            "Open the gripper fully, then back the hand away.",
        ),
    ),
    (
        r"the hand stopped at tool z=[-\d.]+, (-?\d+) cm above where the bottom meets the table",
        lambda m: (
            f"While lowering, the container touched something {m[1]} cm above the table, so the arm stopped "
            "and didn't let go.",
            "Look at what is under the container. Take the container out of the hand, then send the arm back "
            "to its start.",
        ),
    ),
    (
        r"the container is ([\d.]+) deg (?:from how the start held it|off level)\D*?\(max ([\d.]+)",
        lambda m: (
            f"The container tilted {m[1]} degrees on the way (at most {m[2]} is allowed), so the arm stopped "
            "above the spot without letting go.",
            "Send the arm back to its start, then try again.",
        ),
    ),
    (
        r"the arm ended ([\d.]+) deg from the planned line's start|the arm is ([\d.]+) deg from the approach's start",
        lambda m: (
            f"The arm ended {m[1] or m[2]} degrees away from where the lowering starts, so it stopped above "
            "the spot without letting go.",
            "Send the arm back to its start, then try again.",
        ),
    ),
    (
        r"can only go down to tool z=[-\d.]+ here, (-?\d+) cm above where the bottom meets the table -- the spot is "
        r"([\d.]+) m from the base axis",
        lambda m: (
            f"At that spot the arm can only lower the container to {m[1]} cm above the table (the spot is "
            f"{m[2]} m from the arm's base).",
            TAP_OTHER,
        ),
    ),
    (
        r"move the spot to ([\d.]+)-([\d.]+) m from the base axis",
        lambda m: (
            f"That spot is out of the arm's reach: it can place only {m[1]} to {m[2]} m from its base.",
            TAP_OTHER,
        ),
    ),
    (
        r"ANOTHER arm posture reaches it",
        lambda m: (
            "The arm could reach that spot only by changing its whole posture, which is turned off for " "placing.",
            TAP_OTHER,
        ),
    ),
    (
        r"no arm pose puts the tool at .*?\(([\d.]+) m from the base axis\)",
        lambda m: (f"That spot is out of the arm's reach ({m[1]} m from the arm's base).", TAP_OTHER),
    ),
    (r"no arm pose puts the tool", lambda m: ("That spot is out of the arm's reach.", TAP_OTHER)),
    (
        r"can't go straight down from tool z=[-\d.]+ m at ([\d.]+) m",
        lambda m: (
            f"The arm can't lower the container straight down at that spot ({m[1]} m from the arm's base).",
            TAP_OTHER,
        ),
    ),
    (
        r"chair and user boxes",
        lambda m: (
            "The arm would pass too close to the wheelchair or to you on the way to that spot.",
            "Tap a spot further from the wheelchair.",
        ),
    ),
    (
        r"would tilt ([\d.]+) deg \(max ([\d.]+)\)",
        lambda m: (
            f"The container would tilt {m[1]} degrees on the way to that spot (at most {m[2]} is allowed).",
            TAP_OTHER,
        ),
    ),
    (r"a joint at \d+ deg/s", lambda m: ("The motion to that spot would be too fast for the arm.", TAP_OTHER)),
    (
        r"^back to the start: ",
        lambda m: (
            "The arm can't find a way back to its start position from here.",
            "Move the arm a little by hand control (teleop), then try again.",
        ),
    ),
    (
        r"past a joint limit|a joint jumps|a joint moves \d+ deg",
        lambda m: ("The arm can't reach that spot without going past its joint limits.", TAP_OTHER),
    ),
    (
        r"bottom ([\d.]+) cm over the table \(min (\d+)\)",
        lambda m: (f"The container would pass too low over the table ({m[1]} cm; it needs {m[2]} cm).", TAP_OTHER),
    ),
    (
        r"no carry height plans",
        lambda m: ("The arm couldn't find a safe height to carry the container to that spot.", TAP_OTHER),
    ),
    (
        r"leaves the vertical|approach down to the slow line is not straight",
        lambda m: ("The arm can't lower the container straight down at that spot.", TAP_OTHER),
    ),
    (
        r"points too steeply up or down",
        lambda m: (
            "The hand is pointing too steeply up or down for the arm to plan the move.",
            "Turn the hand closer to level, then try again.",
        ),
    ),
    (
        r"the hand is (\d+) deg from holding the container (\d+) deg down \(max (\d+)",
        lambda m: (
            f"The container is held {m[1]} degrees off the angle placing needs (it can correct up to " f"{m[3]}).",
            "Grip the container again, closer to straight, then try again.",
        ),
    ),
    (
        r"the arm moved ([\d.]+) deg while the place was planned",
        lambda m: (
            "The arm moved while the place was being planned, so nothing moved.",
            "Keep the arm still, then try again.",
        ),
    ),
    (
        r"tool frame at .*?([\d.]+) cm apart",
        lambda m: (
            f"The arm's gripper setup doesn't match what placing expects ({m[1]} cm apart).",
            "Ask the operator to check the arm driver's gripper settings.",
        ),
    ),
    (
        r"the marked spot is (\d+) min old",
        lambda m: (
            f"The marked spot is {m[1]} minutes old, so the wheelchair or table may have moved since.",
            "Tap a spot on the table instead, or mark the spot again.",
        ),
    ),
    (
        r"before the arm moved on the chair",
        lambda m: (
            "A saved position is from before the arm was last moved on the wheelchair, so it can't be " "trusted.",
            "Ask the operator to save it again (mark the spot, or save the start position).",
        ),
    ),
    (
        r"no marked spot",
        lambda m: ("There is no marked spot to place at.", "Tap a spot on the table instead, or mark a spot first."),
    ),
    (
        r"no start position|start\.json",
        lambda m: (
            "The arm's start position isn't saved.",
            "Ask the operator to put the arm at its start and save it.",
        ),
    ),
    (
        r"can't save .* in ",
        lambda m: ("The robot couldn't save its records to disk.", "Ask the operator to check the robot's disk."),
    ),
    (
        r"rosbag|recording|bag",
        lambda m: (
            "The run's recording couldn't start, so nothing moved.",
            "Ask the operator to check the robot's disk, or to turn recording off.",
        ),
    ),
    (
        r"retreat: can't back out from here",
        lambda m: (
            "The hand can't back straight away from here.",
            "Move the hand clear by hand control (teleop), then try again.",
        ),
    ),
    (
        r"big_turn|give all of x, y and table_z|lower_speed|vz_stop",
        lambda m: (
            "The request had settings placing can't use, so nothing moved.",
            "Ask the operator to check the settings sent with the request.",
        ),
    ),
    (r"the arm can't be commanded now: (.*?)\. Next", lambda m: plain_explain(m[1])),
)
# A motion's failure as plan.explain_failure words it ("... Next: ... [driver: <the driver's words>]"): the
# driver's words read here (place_container's own cases first, then capability_ui's for the driver and planner)
_ARM_FAILURES = (
    (
        r"\bIK[_ ]?FAIL|\bNO IK\b|\bIK SOLUTION",
        ("The arm can't reach the position it needs there without hitting something.", TAP_OTHER),
    ),
    (
        r"INVALID_START|START COLLIDES",
        (
            "The arm is starting from a position the planner sees as touching the wheelchair.",
            "Move the arm a little clear of the wheelchair (teleop), then try again.",
        ),
    ),
    (
        r"COLLI(?!SION-FREE)",
        ("That spot is too close to the wheelchair for the arm.", "Tap a spot further from the wheelchair."),
    ),
    (r"PLANNER BUSY", ("The planner was busy with another task.", "Wait a moment, then try again.")),
    (
        r"CONSTRAINT_VIOLATED|\bHOLD\b",
        ("The planner couldn't keep the container level on the way.", "Try again: a new plan is usually different."),
    ),
    (
        r"\bBLEND:",
        ("The arm's moves couldn't be joined into one motion.", "Try again: a new plan is usually different."),
    ),
    (
        r"CM FROM THE GOAL|JOINTS WITHIN",
        (
            "The arm stopped short of where it was going: something may be in its way.",
            "Check for anything touching the arm or the container.",
        ),
    ),
)
_TRAJOPT = r"TRAJOPT|DT_EXCEPTION|FINETUNE|PLANNING_FAILED|NO PLAN"


def _arm_reason(raw):
    """(what happened, what to do) for a motion failure plan.explain_failure worded ("[driver: ...]" at its end)."""
    m = re.search(r"\[driver: (.*)\]\s*$", raw, re.S)
    driver = m[1] if m else raw
    up = driver.upper()
    for pattern, answer in _ARM_FAILURES:
        if re.search(pattern, up):
            return answer
    reason, todo = plain_explain(driver)
    if reason == clean(driver) and re.search(_TRAJOPT, up):  # no rule of capability_ui's knew it
        return "The planner couldn't find a smooth path for the arm.", "Try again: a new plan is usually different."
    if reason == clean(driver):  # nothing knew it: say what the arm was doing, never the driver's raw words
        return "The arm's motion didn't complete.", "Check for anything in the arm's way, then try again."
    return reason, todo


def reason(raw):
    """(what happened, what to do or None) for a place_container failure message `raw`, in plain words."""
    text = " ".join(str(raw or "").split())
    if "[driver:" in text:
        return _arm_reason(text)
    for pattern, said in _FAILURES:
        m = re.search(pattern, text)
        if m:
            return said(m)
    return plain_explain(text)


def state(action_id, raw):
    """Where the container, the gripper and the arm are left, from the failure's message."""
    text = str(raw or "")
    if action_id in ("place", "place_here"):
        if re.search(r"container is down, the gripper open|\(gripper open\)", text):
            return "The container is on the table and the gripper is open. The arm is holding still."
        if "STOPPED in reset" in text:
            return "The container is on the table. The arm stopped on its way back to its start and is holding still."
        if "gripper only opened" in text:
            return "The gripper is partly open around the container. The arm is holding still."
        if "NOT letting go" in text:
            return "The container is still in the hand. The arm is holding it still, gently."
        if "STOPPED" in text:
            return "The container is still in the hand. The arm is holding still."
        return "Nothing moved: the container is still in the hand."
    return "The arm is holding still where it stopped." if "STOPPED" in text else "Nothing moved."


def failure_message(action_id, raw):
    """The final failure panel's text: what happened, where things are left, what to do."""
    said, todo = reason(raw)
    if todo == TAP_OTHER or (todo or "").startswith("Tap a spot"):
        if "STOPPED" in str(raw):  # the arm has moved: no picker to tap on any more
            todo = "Once the arm is back at its start, try again at a different spot."
        elif action_id != "place_here":  # a direct action: no picker
            todo = todo.replace("Tap", "Choose", 1)
    return failure_text(said, state(action_id, raw), todo)


def retry_note(raw):
    """The picker's note after a refusal before anything moved: why, and what to do (the picker says to tap)."""
    said, todo = reason(raw)
    then = "" if not todo or todo == TAP_OTHER or todo.startswith("Tap a spot") else f" {sentence(todo)}"
    return f"Couldn't place it there: {sentence(said)[0].lower()}{sentence(said)[1:]}{then} "


def bug_message(action_id):
    """A bug's panel: no exception class or traceback, where things may be, who can look."""
    where = (
        "If the container is still in the hand, the arm is holding it still."
        if action_id in ("place", "place_here")
        else "The arm is holding still."
    )
    return failure_text(
        "Something went wrong in the placing software, so it stopped",
        where,
        "Ask the operator to check the place_container log",
    )


def stopped_note(phase):
    """What a Stop left behind, from the last phase the place reported."""
    if phase in ("retreat", "return", "reset"):
        return "The container is on the table; the gripper had let go of it."
    if phase == "open":
        return "The gripper was opening: the container may still be in the hand."
    return "The container may still be in the hand."


def is_button(event, button_id):
    """Whether a poll_ui_event() result is a tap on `button_id`."""
    return bool(event) and event.get("type") == "button" and event.get("button_id") == button_id


def is_ok(event):
    """Whether a poll_ui_event() result is the error panel's OK."""
    return is_button(event, OK)
