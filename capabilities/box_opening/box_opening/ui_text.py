"""What open_box's final notices on the UI say: Done, Stopped, or what went wrong, in plain words, with the state the
arm and the gripper are left in and what to do next. The log and the action's result keep the run's own message (its
last STOP line); the panel never shows leg names, exit codes or parameter names.

Text only: nothing here decides anything. No ROS imports (the UI tests load it).
"""

from capability_ui import wording as W

TITLE = "Opening the container"

DONE = "The container is open. The lid is on the table beside it."
# a run that ended its own way (exit 0) with no lid set down and let go in its rows: said, never hidden
DONE_NO_LID = "The run finished, but the lid may not be off. Check the container."
STOPPED = "Stopped. The arm is holding still where it was."
# the run did not end on its own Ctrl+C and was killed (node: "a motion may still be in flight"): no state is said
KILLED = "The run had to be ended by force."
HELD = "The lid is still in the gripper."
HELD_TODO = "Someone needs to take it out before the next try."
CHECK_ARM = "Check where the arm is before the next try."

# error_code (node.error_for and execute's own) -> (what happened, the state left in, what to do). None: not said.
# A state is said only where the run's own STOP line says it (node.STOP_CODES, docs/interface.md "Errors").
WORDS = {
    "bad_params": ("The request had settings this action doesn't take.", "Nothing moved.", None),
    "not_ready": (
        "The robot wasn't ready: the arm driver, planner, cameras or gripper didn't answer.",
        "Nothing moved.",
        "Check they are running, then try again.",
    ),
    "not_found": (
        "The container wasn't found.",
        "Nothing was pressed.",
        "Make sure the container is on the table in view of the camera, then try again.",
    ),
    "not_home": (
        "The arm couldn't get to its start pose.",
        "Nothing was pressed.",
        "Check nothing is in the arm's way, then try again.",
    ),
    "arm_holds": (
        "The arm stopped part way.",
        "The arm is holding still where it stopped.",
        "Someone needs to clear it before the next try.",
    ),
    # the STOP line is said before the way home, whose failure says no STOP of its own: where the arm is, unknown
    "contact_early": (
        "The hand touched something before the button, so the press was stopped.",
        None,
        "Check nothing is in the way, then try again.",
    ),
    "contact_late": (
        "The button wasn't where the camera saw it, so the press was stopped.",
        None,
        "Check the container hasn't moved, then try again.",
    ),
    "watch_failed": ("The arm couldn't feel the press, so it stopped.", None, CHECK_ARM),
    "box_moved": (
        "The press moved the container.",
        None,
        "Check the container is still standing on the table, then try again.",
    ),
    # both STOP lines are said once the arm is home (tasks/reach_lid_off: "the arm is home, but the lid slipped",
    # "the lid is back on the box, and the arm is home"); a put-back that holds is arm_holds
    "lid_dropped": ("The lid slipped out of the gripper.", "The arm is home.", "Check where the lid is."),
    "lid_put_back": (
        "The lid wouldn't come off, so the arm put it back on the container.",
        "The arm is home.",
        "Try again.",
    ),
    "not_popped": ("The button was pressed, but the lid didn't pop up.", None, "Try again."),
    "no_button": ("The camera couldn't find the container's button.", "Nothing was pressed.", None),
    "no_press": ("The arm couldn't press the button from here.", "Nothing was pressed.", None),
    "arm_halted": (
        "The arm was halted: the emergency stop, or another program took control.",
        "The arm is stopped where it was.",
        "Release the emergency stop and check the arm before the next try.",
    ),
    "stopped": (KILLED, None, CHECK_ARM),
    "timeout": (
        "The run took too long and was stopped.",
        "The arm is holding still where it stopped.",
        "Check the arm, then try again.",
    ),
    "not_lined_up": (
        "The container wasn't where the arm can open it in time.",
        "Nothing was pressed.",
        "Line up again, then press Open the container.",
    ),
    "line_up_unavailable": ("Lining up couldn't start.", "Nothing was pressed.", None),
}

NOT_ALLOWED = (
    "Opening the container is switched off on this robot: it runs only when someone is at the emergency stop. "
    "Nothing moved."
)


def done(result):
    """The Done notice for open_box's `result`."""
    return DONE if result.get("lid_off") else DONE_NO_LID


def stopped(held=False, killed=False):
    """The notice after the user's Stop: the run's Ctrl+C stopped the motion in flight. `killed`: the run did not
    end on it and was killed, so where the arm stopped is not known."""
    if killed:
        return W.failure_text(f"Stopped. {KILLED}", HELD if held else None, HELD_TODO if held else CHECK_ARM)
    return W.failure_text(STOPPED, HELD if held else None, HELD_TODO if held else None)


def failure(error_code, message, killed=False):
    """The failed notice for an EngineError (`error_code`, its `message`): the code's own words, else the shared
    rules' plain words for the raw message (an e-stop, a planner failure), and where the arm is. `killed`: the
    run was killed (its stop's last resort), and no state is said."""
    held = "still in the fingers" in str(message)
    if error_code in WORDS:
        reason, state, todo = WORDS[error_code]
    else:
        reason, todo = W.explain(message)
        state = None if todo else CHECK_ARM
    if killed:
        reason = reason if reason == KILLED else f"{reason} {KILLED}"
        state, todo = None, CHECK_ARM
    if held:
        # where the arm is stays said beside the lid; "nothing moved" or "pressed" cannot be true with it held
        state = f"{state} {HELD}" if state and state.startswith("The arm") else HELD
        todo = HELD_TODO
    return W.failure_text(reason, state, todo)


def unexpected(error):
    """The failed notice for an exception that is not an EngineError (a bug, a lost service)."""
    return W.failure_text(W.explain(f"{type(error).__name__}: {error}")[0], None, CHECK_ARM)
