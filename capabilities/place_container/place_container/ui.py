"""What place_container shows on the user's UI panel (ctx.update_user_panel; the shapes are in
docs/UI_JSON_SCHEMA.md). Pure: no ROS, so it is tested on its own.

Every phase the action reports with ctx.progress() is also put on the panel, in words, with a
progress bar; every failure is put on it with an OK button before the action ends with the error
(the panel closes when the action ends, so an error not shown before then is never seen).

place_here (the UI's tap on a table) runs on capability_ui.RunUI instead: the spot picker, then the
place under a progress panel with Stop, then Stopped / failed / Done notices. Its words are here too.
"""

from capability_ui import wrap  # lines Unreal can show (it doesn't wrap; one rich-text run a line)

OK = "ok"  # the error panel's button
PLACE_HERE = "Place the container here"  # the one-tap action's label on the UI's action list

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


def no_spot_message(problem, marked, note=""):
    """No spot to offer: why, and what the user can do."""
    then = "Use the marked spot, or clear" if marked else "Clear"
    return f"{note}No free spot to offer: {problem}. {then} a spot on the table and press Look again."


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
