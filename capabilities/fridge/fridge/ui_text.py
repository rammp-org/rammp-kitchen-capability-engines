"""What the fridge's UI panels say when a goal fails: the reason in plain words, with the numbers the run log has
(how far to move the wheelchair, which way, by how much the camera's looks disagreed, ...), what state it leaves the
arm, hand and wheelchair in, and what to do next.

Text only. The run raises its failures for the log and the orchestrator (dc.Refused, BaseError, EngineError); this
module reads those messages and says them for the person in the wheelchair. Nothing here changes what the run does.
"""

import re

from capability_ui import explain, failure_text, sentence

# what each action's button is called on the UI (node.UI_ACTIONS), for "open the door with ... first"
BUTTON = {"open": "Open Fridge Door", "close": "Close Fridge Door", "go_to_start": "Go to Fridge"}

HOLDING = "The hand may still be holding the handle."
HOLDING_TODO = "Open the gripper and move the arm off the handle (teleop) before anything else."
DROVE = "The wheelchair has already moved."
DROVE_TODO = "Check where the wheelchair is before trying again."
STOPPED_OPEN = (
    "If the hand is still on the handle, open the gripper and move the arm off the handle (teleop) before anything "
    "else."
)


def _moves_sentence(moves):
    """chair_moves' words ("move the wheelchair ~5 cm closer to the fridge and move the wheelchair ~5 cm to its
    left") as one sentence ("Move the wheelchair ~5 cm closer to the fridge and ~5 cm to its left.")."""
    text = " ".join(str(moves).split()).strip(" .;")
    text = re.sub(r"(;| and) move the wheelchair ", lambda m: (", " if m.group(1) == ";" else " and "), text)
    text = re.sub(r";\s*", ", and ", text)
    return sentence(text[:1].upper() + text[1:]) if text else ""


def _open_infeasible(raw):
    """The open refused for feasibility (no grasp gives a whole open that plans): (reason, todo), with the wheelchair
    move the placement search found by planning it, or the sim's placement advice; None if `raw` isn't that."""
    if "refusing before any motion" not in raw or not re.search(r"no grasp|grasps failed|search ran out", raw):
        return None
    reason = "The arm can't find a way to open the door from where the wheelchair is now."
    again = "Then try again."
    found = re.search(r"the open plans with the fridge shifted \[[^\]]*\] m -- (?P<moves>.+?), then dry-run", raw)
    if found:
        return (
            reason,
            f"{_moves_sentence(found.group('moves'))} The arm checked that it can open the door from "
            f"there. {again}",
        )
    advice = re.search(r"Wheelchair placement \([^)]*\): (?P<moves>[^\n]+?)\.(?=\s*(\n|$| -- |->))", raw)
    none_m = re.search(r"no wheelchair move of up to (?P<cm>\d+) cm made the open plan", raw)
    timed = re.search(r"none of the first (?P<n>\d+) wheelchair moves made the open plan", raw)
    if advice and not none_m:
        todo = f"Try this: {_moves_sentence(advice.group('moves'))} {again}"
        if timed:
            todo = f"Moving the wheelchair a little didn't help in the moves the arm had time to check. {todo}"
        return reason, todo
    if none_m:
        return reason, (
            f"Moving the wheelchair up to {none_m.group('cm')} cm in any direction wouldn't help either, so the "
            "problem is not where the wheelchair is. Check that the door is fully closed and nothing is in front of "
            f"the handle, then try again."
        )
    if timed:
        return reason, (
            "The arm ran out of time checking other wheelchair positions. Park the wheelchair square in front of the "
            "fridge, about where it was for earlier opens, then try again."
        )
    return reason, "Park the wheelchair square in front of the fridge, then try again."


def _secondary_placement(raw):
    m = re.search(r"Once the error above is fixed: (?P<moves>.+?)\.\)", raw)
    return f"Also, the wheelchair is outside its usual spot: {_moves_sentence(m.group('moves'))}" if m else ""


def _handle_out_of_range(raw):
    m = re.search(
        r"handle (?P<axis>[xyz])=(?P<v>-?[\d.]+) outside plausible \((?P<lo>-?[\d.]+), (?P<hi>-?[\d.]+)\)", raw
    )
    if not m:
        return None
    axis, v, lo, hi = m.group("axis"), float(m.group("v")), float(m.group("lo")), float(m.group("hi"))
    off = (v - hi) if v > hi else (lo - v)
    cm = max(1, round(abs(off) * 100))
    if axis == "x":
        where = "too far ahead" if v > hi else "too close"
        todo = (
            f"Move the wheelchair ~{cm} cm closer to the fridge"
            if v > hi
            else f"Move the wheelchair ~{cm} cm back from the fridge"
        )
    elif axis == "y":
        where = "too far to the left" if v > hi else "too far to the right"
        todo = f"Move the wheelchair ~{cm} cm to its {'left' if v > hi else 'right'}"
    else:
        where = "too high" if v > hi else "too low"
        todo = "Check that the camera is looking at the fridge's handle and not something else"
    return (
        f"The camera saw the handle {where} for the arm ({abs(off) * 100:.0f} cm outside where it can open " "from)."
    ), f"{todo}, then try again."


# (pattern on the raw message, reason, state or None, todo); {name} fields come from the pattern's groups
_REFUSALS = (
    # -- seeing the handle ---------------------------------------------------------------------------------------
    (
        r"NO DETECTION",
        "The wrist camera couldn't find the fridge handle.",
        None,
        "Make sure the arm is in front of the fridge (Go to Fridge) and nothing blocks the camera's view of the "
        "handle, then try again.",
    ),
    (
        r"No two of (?P<n>\d+) detections agreed within (?P<cm>\d+) cm",
        "The camera saw the handle in a different place each time it looked ({n} looks, none within {cm} cm of "
        "another).",
        None,
        "Keep the wheelchair still and make sure the handle is clearly in view, then try again.",
    ),
    (
        r"hinge detections differ by more than (?P<cm>\d+) cm",
        "The camera saw the door's hinge in two different places (more than {cm} cm apart).",
        None,
        "Keep the wheelchair still and make sure the door is in view, then try again.",
    ),
    (
        r"No RGB-D frames|no wrist camera",
        "The wrist camera isn't sending images.",
        None,
        "Check that the wrist camera is running, then try again.",
    ),
    # -- the arm, the gripper and the handle ---------------------------------------------------------------------
    (
        r"is (?P<deg>[\d.]+) deg from where it was planned from",
        "The arm had moved {deg} degrees since the motion was planned, so it wasn't started.",
        None,
        "Try again.",
    ),
    (
        r"Gripper still closed after opening",
        "The gripper didn't open before reaching for the handle.",
        None,
        "Check that the gripper works and isn't holding anything, then try again.",
    ),
    (r"Did not reach the grasp", "The arm didn't reach the handle.", "The gripper is open.", "Try again."),
    (
        r"Tracking (?P<cm>[\d.]+) cm at the grasp",
        "The hand stopped {cm} cm away from the handle, so it didn't grab it.",
        "The gripper is open.",
        "Try again.",
    ),
    (
        r"Gripper is OPEN -- nothing grasped",
        "The gripper closed on nothing, so the arm didn't pull.",
        None,
        "Check the hand is around the handle; move the arm away (teleop) and try again.",
    ),
    (
        r"gripper did not open -- the hand is still on the handle",
        "The gripper didn't open, so the hand is still holding the handle.",
        None,
        HOLDING_TODO,
    ),
    (
        r"swing from the real grasp|from the real grasp -- stopping BEFORE the swing",
        "After grabbing the handle, the arm couldn't find a safe way to pull the door from where it actually grabbed "
        "it.",
        "The hand is holding the handle and the door is still closed.",
        HOLDING_TODO,
    ),
    (
        r"back-off: .* NOT releasing",
        "The arm couldn't find a safe way to back off the handle, so it kept holding it.",
        "The hand is holding the handle.",
        HOLDING_TODO,
    ),
    (
        r"the hand is off the handle \(gripper open\)",
        "The arm let go of the handle but couldn't find a safe way to push the door further open.",
        "The door is partly open.",
        "Push it the rest of the way by hand, or close it and try again.",
    ),
    (
        r"the door is already ~(?P<deg>\d+) deg open",
        "The door is already about {deg} degrees open, so there's nothing left to push.",
        None,
        "Close the door first if you want to open it again.",
    ),
    (
        r"no hand placement behind the door plans from here",
        "After the wheelchair moved, the arm can't find a place behind the door to push from.",
        None,
        "Check the wheelchair moved straight, by the distance shown, without turning, then try again.",
    ),
    # -- the close -----------------------------------------------------------------------------------------------
    (
        r"has no opened door|need .*the open writes it",
        "There's no record of an opened fridge door to close.",
        None,
        f"Open the door with {BUTTON['open']} first: the close pushes the door from where the open left it. If the "
        "door was opened by hand, close it by hand.",
    ),
    (
        r"no wheelchair move from (?P<fwd>\d+) cm forward to (?P<back>\d+) cm back lets the close plan",
        "The arm can't find a safe way to close the door from any wheelchair position between {fwd} cm forward and "
        "{back} cm back.",
        None,
        "Check the door is still where the open left it (it may have swung). Move the wheelchair back to where it was "
        "for the open, then try again, or close the door by hand.",
    ),
    (
        r"the close doesn't plan with the wheelchair moved (?P<how>[^:]+):",
        "The arm can't find a safe way to close the door with the wheelchair moved {how}.",
        None,
        "Move the wheelchair back to where it was for the open, then try again.",
    ),
    (
        r"after the wheelchair move the close no longer plans",
        "After the wheelchair moved, the arm can't find a safe way to close the door.",
        None,
        "Check the wheelchair moved straight back by the distance asked, then try again.",
    ),
    (
        r"the hand would tilt",
        "Closing the door would tilt the container in the hand, so the arm didn't move.",
        None,
        "Put the container down first, or close the door by hand.",
    ),
    # -- the wheelchair driving itself (the "(wheelchair drives)" buttons) ------------------------------------------
    (
        r"drive_base: (?P<cm>\d+) cm is over base_max_distance",
        "The wheelchair would have to drive {cm} cm, more than it may drive on its own.",
        None,
        'Use the button without "(wheelchair drives)" and move the wheelchair yourself when asked.',
    ),
    (r"drive_base: a dry run never drives", "The wheelchair can't drive during a dry run.", None, None),
    # -- planning to the start pose, the planner -----------------------------------------------------------------
    (
        r"no collision-free path to the fridge start pose.*START_STATE",
        "The arm can't plan to its fridge start position: the planner sees it touching an obstacle where it is now.",
        None,
        "Move the arm a little clear (teleop), then try again.",
    ),
    (
        r"no collision-free path to the fridge start pose",
        "The arm can't find a safe path to its fridge start position from where it is now.",
        None,
        "Check nothing is near the arm, then try again. If it keeps failing, move the arm "
        "closer to the start position (teleop) first.",
    ),
    (
        r"planner refused the fridge's collision world",
        "The motion planner didn't accept the fridge's model.",
        None,
        "Check the planner is running, then try again.",
    ),
    (
        r"could not join the (?P<n>\d+) planned legs",
        "The planner couldn't combine the {n} planned moves into one motion.",
        None,
        "Try again.",
    ),
)

_BASE_ERRORS = (
    (
        r"moved the wrong way \((?P<cm>[-+\d.]+) cm\)",
        "The wheelchair started moving the wrong way ({cm} cm) and was stopped.",
    ),
    (
        r"the base moved (?P<cm>[\d.]+) cm, more than asked",
        "The wheelchair moved {cm} cm, more than planned, and was stopped.",
    ),
    (
        r"did not cover (?P<cm>\d+) cm in (?P<s>\d+) s: stopped after moving (?P<moved>[-+\d.]+) cm",
        "The wheelchair didn't drive the {cm} cm in {s} s (it moved {moved} cm) and was stopped.",
    ),
    (
        r"ended (?P<cm>[-+\d.]+) cm from the distance",
        "The wheelchair stopped {cm} cm from where it should have, so the arm didn't continue.",
    ),
)


def _match(rules, raw):
    for rule in rules:
        m = re.search(rule[0], raw, re.IGNORECASE | re.DOTALL)
        if m:
            fields = m.groupdict()
            return tuple(None if part is None else part.format(**fields) for part in rule[1:])
    return None


def refusal(action_id, raw):
    """(reason, state, todo) for a refusal (dc.Refused as the node raised it) in plain words."""
    raw = str(raw or "")
    if action_id == "open":
        found = _open_infeasible(raw)
        if found:
            return found[0], None, found[1]
        out = _handle_out_of_range(raw)
        if out:
            return out[0], None, out[1]
    hit = _match(_REFUSALS, raw)
    if hit:
        reason, state, todo = hit
    else:
        reason, todo = explain(raw)
        state = None
    extra = _secondary_placement(raw) if action_id == "open" else ""
    if extra:
        todo = " ".join(p for p in (sentence(todo), sentence(extra)) if p)
    return reason, state, todo


def failure(action_id, error, maybe_holding=False, drove=False):
    """The failure panel's message for `error` (an EngineError, or None for a bug), from its own message."""
    code = getattr(error, "error_code", None)
    raw = str(error) if error is not None else ""
    state = todo = None
    if error is None:
        reason = "Something unexpected went wrong in the fridge program, so it stopped."
        todo = "Check the fridge's log before trying again."
    elif code == "refused":
        reason, state, todo = refusal(action_id, raw)
    elif code == "stopped":
        # the error says what stopped; why (a contact stop, the e-stop) only if its message has it
        plain, rule_todo = explain(raw) if "(" in raw else (None, None)
        reason = "The arm stopped." + (f" {plain}" if plain else "")
        state = "It is holding still."
        todo = rule_todo or "Check that nothing is in the arm's or the door's way, then try again."
    elif code == "base_stopped":
        hit = _match(_BASE_ERRORS, raw)
        reason = hit[0] if hit else explain(raw)[0]
        state = "The wheelchair and the arm are stopped."
        todo = "Check where the wheelchair is (it may need to go back to its spot) before trying again."
        drove = False  # said already
    elif code == "no_answer":
        reason = "There was no answer on the screen in time, so the fridge action ended."
        state = "The arm is holding still."
    elif code == "execution_disabled":
        reason = "Arm motion is turned off for the fridge on this robot."
        todo = "Ask the operator to restart the fridge with arm motion allowed (allow_execute)."
    else:
        reason, todo = explain(raw)
    if maybe_holding:
        state = " ".join(p for p in (sentence(state), HOLDING) if p)
        todo = HOLDING_TODO
    if drove:
        state = " ".join(p for p in (sentence(state), DROVE) if p)
        todo = " ".join(p for p in (sentence(todo), DROVE_TODO) if p)
    return failure_text(reason, state, todo)


def retry_prompt(action_id, error, label):
    """A recoverable refusal before anything moved: what happened and what to change, then the same button
    (`label`) to try again."""
    reason, state, todo = refusal(action_id, str(error))
    todo = todo or ""
    if re.search(r"\btry again\b", todo, re.IGNORECASE):  # "..., then try again." -> "..., then press <label> ..."
        todo = re.sub(
            r"\b(t)ry again\b",
            lambda m: ("P" if m.group(1) == "T" else "p") + f"ress {label} to try again",
            todo,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        todo = " ".join(p for p in (sentence(todo), f"Press {label} to try again.") if p)
    return failure_text(reason, state, todo)
