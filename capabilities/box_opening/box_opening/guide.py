"""Lining the chair up with the box before the arm flies: what to tell the user, and when to ask.

The owner, 2026-11-16: "instead of the arm just saying it can't reach the box and erroring, I want it
to have a boundary of where it CAN and CANT open the box, and tell the user to either move closer,
further away, more left, or right. When the box is in the boundary, it will prompt the user asking if
they want to open it, when they can say yes or something and the arm flies. It should dynamically
check constantly if the arm can reach the box."

This is the part that decides what is said. It moves nothing and sees nothing: each tick it is told
where the box was last seen (Fix) and what the boundary says of that place (Reach), and answers with
what the panel shows (Say). The engine's loop feeds it and sends its panel, as the base class's
example does (capabilities/example: ctx.update_user_panel / ctx.poll_ui_event).

    LOOKING   no box seen lately                   "Looking for the container"
    MOVE      the box is outside the boundary      "Move closer, about 15 cm"   (or back / left / right)
    HOLD      inside, not yet steady or planned    "Hold still, checking"
    READY     inside, steady, and the run plans    "Ready. Open the container?"  [Open] [Cancel]
    GO        Open pressed on a fresh READY        the arm flies
    TIMED_OUT the time for lining up ran out       "Press open again when you are in place"

The words are the user's moves, not the box's place: the box too far away is "Move closer", the box
too far to the left is "Move left" (toward it). One instruction at a time, the larger miss first, and
no flicker: an instruction stands MIN_SAY_S, the axis it names changes only where the other miss is
AXIS_SWITCH_M larger, and READY is left only once the box has been out, or moving, for LEAVE_S.

Open is taken only from READY and only on a box seen within FRESH_S: a press on anything older is not
a yes to what is there now.
"""

import math
from dataclasses import dataclass, field

LOOKING, MOVE, HOLD, READY, GO, TIMED_OUT = "looking", "move", "hold", "ready", "go", "timed_out"

OPEN, CANCEL = "open", "cancel"  # the panel's button ids
TITLE = "Opening the container"

LOST_S = 1.5  # no box seen for this long: looking again
FRESH_S = 1.0  # Open counts only on a box seen this recently
STEADY_S = 1.0  # the box has stood still this long ...
STEADY_M = 0.015  # ... within this much: the chair is not moving
LEAVE_S = 0.5  # READY is left once the box has been outside, moving, or not judged, this long
MIN_SAY_S = 0.7  # an instruction stands at least this long before another replaces it
AXIS_SWITCH_M = 0.03  # the other axis is named only where its miss is this much the larger
ROUND_M = 0.05  # distances are said to the nearest 5 cm ...
LITTLE_M = 0.04  # ... and under this, "a little"
FAR_M = 0.5  # the progress bar is empty this far from the boundary, full inside it


@dataclass(frozen=True)
class Fix:
    """Where the box was last seen: `t` (s, the guide's clock), `xy` in the arm's frame (x ahead of the
    arm, y to its left), which camera saw it."""

    t: float
    xy: tuple
    source: str = "wrist"


@dataclass(frozen=True)
class Reach:
    """What the boundary says of the place a Fix names. `inside`: the arm opens a box there (the reach
    map's word, with its margin). `to_zone`: from the box to the nearest place it does, (dx, dy) in the
    arm's frame; None inside, or where no such place is known. `planned`: the run's own plan for the
    box where it stands passed (True), failed (False), or has not been made yet (None)."""

    inside: bool
    to_zone: tuple = None
    planned: bool = None
    why: str = ""


@dataclass(frozen=True)
class Say:
    """What the panel shows now: the state, the message, the buttons' (id, label), how full the bar is
    (0-100), and the instruction as (word, metres) where there is one."""

    state: str
    message: str
    buttons: tuple = ()
    pct: float = 0.0
    hint: tuple = None

    def panel(self):
        """The UI bridge's panel (docs/UI_JSON_SCHEMA.md), as the base class's example sends one."""
        return {
            "title": TITLE,
            "message": self.message,
            "progress": {"phase": self.state, "pct": round(float(self.pct), 1), "pausable": False},
            "buttons": [{"button_id": b, "label": label} for b, label in self.buttons],
        }


def instruction(to_zone, axis=None):
    """The user's move that brings a box `to_zone` away from the boundary into it: (word, metres, axis).
    The zone nearer the arm than the box (dx < 0) is "closer", further "back"; the zone to the box's
    right (dy < 0) means the box stands too far left: "left", toward it. The larger miss is named;
    `axis` ("x" | "y", the one named last) is kept unless the other is AXIS_SWITCH_M larger."""
    dx, dy = float(to_zone[0]), float(to_zone[1])
    ax, ay = abs(dx), abs(dy)
    if axis == "x":
        pick = "y" if ay > ax + AXIS_SWITCH_M else "x"
    elif axis == "y":
        pick = "x" if ax > ay + AXIS_SWITCH_M else "y"
    else:
        pick = "x" if ax >= ay else "y"
    if pick == "x":
        return ("closer" if dx < 0 else "back"), ax, "x"
    return ("left" if dy < 0 else "right"), ay, "y"


def words(word, metres):
    """An instruction in the user's words: "Move closer, about 15 cm"; "Move left a little"."""
    if metres < LITTLE_M:
        return "Move %s a little" % word
    return "Move %s, about %d cm" % (word, int(round(max(metres, ROUND_M) / ROUND_M) * ROUND_M * 100))


@dataclass
class Guide:
    """One line-up: update() every tick, press() for a button, until `state` is GO, TIMED_OUT or the
    caller's cancel. `budget_s`: how long the user has to line up (None: no limit)."""

    budget_s: float = None
    state: str = LOOKING
    started: float = None
    _track: list = field(default_factory=list)  # (t, x, y) of the fixes of the last STEADY_S
    _last_fix: Fix = None
    _axis: str = None
    _said: Say = None
    _said_at: float = None
    _bad_since: float = None  # in READY: since when the box has been outside, moving, or not judged

    # -- the box's own motion ------------------------------------------------------
    def _see(self, fix):
        if fix is None or (self._last_fix is not None and fix.t <= self._last_fix.t):
            return
        self._last_fix = fix
        self._track.append((fix.t, float(fix.xy[0]), float(fix.xy[1])))
        while self._track and fix.t - self._track[0][0] > STEADY_S + 0.5:
            self._track.pop(0)

    def steady(self, now):
        """The box seen all through the last STEADY_S, never more than STEADY_M from where it is now."""
        if self._last_fix is None or now - self._last_fix.t > FRESH_S:
            return False
        recent = [p for p in self._track if now - p[0] <= STEADY_S]
        if not recent or not any(now - p[0] >= STEADY_S * 0.8 for p in self._track):
            return False  # not watched for long enough yet
        _t, x, y = self._track[-1]
        return all(math.hypot(px - x, py - y) <= STEADY_M for _pt, px, py in recent)

    # -- what to say ---------------------------------------------------------------
    def _answer(self, now, say, at_once=False):
        """`say`, unless another instruction has not yet stood MIN_SAY_S (a change of state is said at once)."""
        prev = self._said
        hold = (
            not at_once
            and prev is not None
            and prev.state == say.state == MOVE
            and prev.hint is not None
            and say.hint is not None
            and prev.hint[0] != say.hint[0]
            and now - self._said_at < MIN_SAY_S
        )
        if hold:
            return prev
        if prev is None or (prev.state, prev.message, prev.buttons) != (say.state, say.message, say.buttons):
            self._said_at = now
        self._said = say
        self.state = say.state
        return say

    def update(self, now, fix, reach):
        """The tick: the newest Fix (or None: nothing new) and the Reach of the newest box seen (None: not
        judged yet). Returns what to show."""
        now = float(now)
        if self.started is None:
            self.started = now
        if self.state in (GO, TIMED_OUT):
            return self._said
        self._see(fix)
        if self.budget_s is not None and now - self.started > self.budget_s:
            return self._answer(now, Say(TIMED_OUT, "Press open again when you are in place"), at_once=True)
        cancel = ((CANCEL, "Cancel"),)
        seen = self._last_fix
        if seen is None or now - seen.t > LOST_S:
            self._bad_since = None
            return self._answer(now, Say(LOOKING, "Looking for the container", cancel), at_once=True)
        good = reach is not None and bool(reach.inside) and reach.planned is True and self.steady(now)
        if self.state == READY and not good:
            # a jitter of the box or of the boundary is not yet a reason to take the question away; nor is a
            # tick the boundary cannot judge at all (no Reach: one frame's lid height in another grid's bucket)
            self._bad_since = now if self._bad_since is None else self._bad_since
            if now - self._bad_since < LEAVE_S:
                return self._said
        else:
            self._bad_since = None
        if reach is None:
            return self._answer(now, Say(HOLD, "Hold still, checking", cancel, 50.0), at_once=self.state != READY)
        if good:
            ready = Say(READY, "Ready. Open the container?", ((OPEN, "Open"),) + cancel, 100.0)
            return self._answer(now, ready, at_once=True)
        if reach.inside and reach.planned is not False:
            return self._answer(now, Say(HOLD, "Hold still, checking", cancel, 100.0), at_once=True)
        if reach.to_zone is None:
            return self._answer(now, Say(MOVE, "The arm cannot open it from here", cancel, 0.0), at_once=True)
        word, metres, self._axis = instruction(reach.to_zone, self._axis)
        pct = 100.0 * max(0.0, 1.0 - math.hypot(*reach.to_zone) / FAR_M)
        return self._answer(now, Say(MOVE, words(word, metres), cancel, pct, (word, metres)))

    def press(self, button_id, now):
        """A button: Open flies only from READY on a box seen within FRESH_S; Cancel is the caller's.
        Returns True where the press is taken as the user's yes."""
        if button_id != OPEN or self.state != READY:
            return False
        if self._last_fix is None or float(now) - self._last_fix.t > FRESH_S:
            return False
        self._answer(float(now), Say(GO, "Opening the container", ((CANCEL, "Cancel"),), 100.0), at_once=True)
        return True
