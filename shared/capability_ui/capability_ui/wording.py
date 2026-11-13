"""Plain words for a failure on the UI: what happened, and what the user can do about it.

A capability's failures are written for its log (planner statuses, driver codes, parameter names). The UI panel is
read by the person in the wheelchair, so the panel says the same thing in plain words, with the numbers kept and a
next step: `explain()` maps the arm driver's and the planner's own failures (shared by every capability); each
capability maps its own refusals first and falls back on this.

Nothing here decides anything: it only turns a message into words. Nothing imports ROS.
"""

import re

# The arm driver's and the planner's failures, as self.arm and the capabilities log them: (pattern, what happened,
# what to do or None). The first match wins, so the most specific come first.
RULES = (
    (
        r"e-?stopped|estopped|emergency stop|e-stop",
        "The emergency stop is pressed, so the arm can't move.",
        "Release the emergency stop, then try again.",
    ),
    (
        r"unexpected contact|CONTACT_STOP",
        "The arm felt an unexpected push or pull and stopped to be safe.",
        "Check that nothing is blocking or holding the arm. The arm's fault has to be cleared before it can move "
        "again.",
    ),
    (
        r"START_STATE|start is inside an obstacle|start .{0,30}in collision",
        "The arm is starting from a position the planner sees as touching something.",
        "Move the arm a little clear (teleop), then try again.",
    ),
    (
        r"IK_FAIL|no collision-free joint solution",
        "The arm can't reach the position it needs without hitting something.",
        None,
    ),
    (
        r"VALIDATION_FAILED|velocity [\d.]+ > limit",
        "The planned motion was too fast for the arm's limits, so it wasn't run.",
        "Try again: a new plan is usually different.",
    ),
    (
        r"MotionGenStatus|PLANNING_FAILED|collision-free path|found no plan|no plan\b|could not plan|didn't plan|"
        r"does ?n[o']t plan",
        "The planner couldn't find a safe path for the arm.",
        None,
    ),
    (
        r"path.?tolerance|tracking error|off (its|the) path",
        "The arm fell too far behind its planned path and stopped.",
        "Check that nothing is holding the arm back, then try again.",
    ),
    (
        r"goal REJECTED|rejected the goal|not authorized|another (client|program) (owns|has)",
        "The arm driver didn't accept the motion.",
        "Check that no other program is controlling the arm and that it has no fault, then try again.",
    ),
    (
        r"does not answer|did not answer|didn't answer|no .{0,40}action server|is the arm driver up|no result in|"
        r"timed out|not available|no /\S+ in [\d.]+ s",
        "Part of the robot's software didn't respond.",
        "Check that the arm driver, planner and cameras are running, then try again.",
    ),
)

# Phrases in a log message that only repeat the arm's state, which the panel says on its own line
_STATE_TAILS = re.compile(
    r"\s*--\s*(the arm holds( where it is)?|nothing (moved|sent)|arm untouched|refusing( before any motion)?|"
    r"not sent|ABORT[^.]*)\.?",
    re.IGNORECASE,
)
MAX_REASON_CHARS = 220


def explain(raw):
    """(what happened, what to do or None) for a failure message `raw`, in plain words; the cleaned message itself
    when no rule knows it (so the panel never says less than the log's first line)."""
    text = " ".join(str(raw or "").split())
    for pattern, reason, todo in RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return reason, todo
    return clean(text), None


def clean(raw):
    """`raw` as a sentence for the panel: one line, the state tails ("-- the arm holds where it is") dropped, a
    capital first and a full stop last, at most MAX_REASON_CHARS (cut at a word)."""
    text = _STATE_TAILS.sub("", " ".join(str(raw or "").split())).strip(" -;:")
    if not text:
        return "Something went wrong."
    if len(text) > MAX_REASON_CHARS:
        text = text[: MAX_REASON_CHARS - 3].rsplit(" ", 1)[0].rstrip(" ,;:(") + "..."
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?", "...")) else text + "."


def sentence(text):
    """`text` with a full stop, or "" if it is empty."""
    text = " ".join(str(text or "").split()).strip()
    if not text:
        return ""
    return text if text.endswith((".", "!", "?", "...")) else text + "."


def failure_text(reason, state=None, todo=None):
    """A failure panel's message: what happened, the state the arm/hand/wheelchair is left in, what to do. Each
    part a sentence of its own; empty parts left out."""
    return " ".join(p for p in (sentence(reason), sentence(state), sentence(todo)) if p)


def describe(raw, state=None, todo=None):
    """failure_text for a raw failure message: explain()'s plain words, `todo` (the capability's own next step) in
    place of the rule's when given."""
    reason, rule_todo = explain(raw)
    return failure_text(reason, state, todo or rule_todo)
