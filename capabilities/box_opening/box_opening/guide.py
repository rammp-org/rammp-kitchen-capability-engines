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

    LOOKING   nothing has looked yet               "Looking for the container"
              no camera sees the container         "The arm cannot see the container. Move the chair in front of it"
              ... what no chair move cures         "The arm's vision is not working" and the like (NO_CURE)
    MOVE      the box is outside the boundary      "Move closer, about 15 cm"   (or back / left / right)
    HOLD      inside, not yet steady or planned    "Hold still, checking"
    READY     inside, steady, and the run plans    "Ready. Open the container?"  [Open] [Cancel]
    GO        Open pressed on a fresh READY        the arm flies
    TIMED_OUT the time for lining up ran out       "Press open again when you are in place"
    UNABLE    the run says what no waiting cures   "The arm is not at its rest pose: home it, then ..."  (unable())

The words are the user's moves, not the box's place: the box too far away is "Move closer", the box
too far to the left is "Move left" (toward it). One instruction at a time, the larger miss first, and
no flicker: an instruction stands MIN_SAY_S, the axis it names changes only where the other miss is
AXIS_SWITCH_M larger, "Hold still" takes an instruction's place only once the box has read inside the
boundary for HOLD_AFTER_MOVE_S, and READY is left only once the box has been out, or moving, for LEAVE_S.
Its opposite ("back" after "closer") by a few centimetres is said only once it has stood REVERSE_S, and
an instruction stands through a container out of view for up to OUT_STANDS_S (2026-11-20).

Open is taken only from READY and only on a box seen within FRESH_S: a press on anything older is not
a yes to what is there now.

A goal with no UI is guided by the same Guide (owner 2026-11-17: "don't end, just tell you how far to
move until it's close enough then asking if ready y/n"): its lines are said on the node's console and
its question (ASK) is answered y or n (line_up.Answers), a y taken exactly as Open is.

A container NO camera sees is guided too (owner 2026-11-19, asked whether a box no camera sees at all
should become guidance, "move so the container is in front of the arm": yes). Until then the panel stood
on "Looking for the container" for the whole line-up, which tells the person nothing to do, and a run
whose cameras found none ended `not_found`. Now, once what watches has looked and says it has no
container (`none`: its own word on why), or a container that was seen has not been for LOST_S, the person
is told so and what to do about it: move the CHAIR (UNSEEN). Never the container, nor anything on the
table: the scene is not changed, only where the robot stands (the owner's rule of 2026-11-11). No
direction and no distance are said then, because none is known; the moment a camera sees the container
they are (MOVE), as ever.

Guidance is for what moving the chair can cure (review of 2026-11-19). Three states it cannot cure wore
the same sentence: what watches delivering nothing, two things that could be the container side by side,
and something in view that nothing ties to the container the person chose. Under a five-minute wait each
was five minutes of a person driving a wheelchair for nothing. Each is now told as what it is (kind_of,
NO_CURE), and two of them, where they stand, end the line-up (ENDS, NO_CURE_S: line_up.BoxSource). A
container that is merely not in view never ends one: only its time does. And the sentence does not flash:
until what watches has had no container for LOST_S, as long as a lost one is given, the panel still says
"Looking for the container".
"""

import math
from dataclasses import dataclass, field

LOOKING, MOVE, HOLD, READY, GO, TIMED_OUT = "looking", "move", "hold", "ready", "go", "timed_out"

OPEN, CANCEL = "open", "cancel"  # the panel's button ids
# What the run's line-up says, in the place of where the container is, of an arm that does not stand at its rest
# pose (press_demo.line_up_watch): a line-up moves nothing but the wrist of an arm at rest, so no waiting and no
# driving the chair brings its Ready. The engine ends the line-up on it at once, the panel saying it (UNABLE)
ARM_NOT_AT_REST = "the arm is not at its rest pose: home it, then press open again"
UNABLE = "unable"  # the panel's state where the line-up cannot go on and says why: no button
TITLE = "Opening the container"
ASK = "Ready. Open the container?"  # the question: Open on the panel, y on the console (the terminal reads it too)
# what a line-up out of its time says last. The UI's words; a goal with no UI is given its own (Guide.again)
AGAIN = "Press open again when you are in place"
# ... and where no camera saw the container in all that time: what was missing is that it never came into
# view, not that it stood out of reach (Guide.again_unseen)
AGAIN_UNSEEN = "Container not seen in time. Press open again in front of it"
# What is said while no container is seen (state LOOKING). SEARCHING until what watches has looked and had
# none for LOST_S: while the wrist turns to its view, before the scene camera's first look ends, and through
# a first frame or two that show nothing to a person who is in place. Then the person is told what it is.
# Each is one short message on the UI's panel, no longer than the longest it showed before these
# (ARM_NOT_AT_REST's sentence, 63 characters)
SEARCHING = "Looking for the container"
# WHAT a watch's word on having no container is of (kind_of; a run says the same of its own find:
# press_demo.no_container_kind). K_UNSEEN: nothing that could be the container is in any view, which moving
# the chair cures: the one state that is guided
K_UNSEEN, K_SEVERAL, K_NOT_CHOSEN, K_FAULT = "unseen", "several", "not_chosen", "fault"
UNSEEN = "The arm cannot see the container. Move the chair in front of it"
# ... and the three that no moving of the chair cures, each in words that are true of it (review of
# 2026-11-19: all three wore UNSEEN's sentence, or a "Move in front of only one" that no chair can do):
#  K_SEVERAL     two things that could be the container, and nothing says which is meant. The second may be
#                any lid-sized top (a bowl, a roll of tape: flatbot 2026-11-12), so it is not called a
#                container; and two tops side by side within the arm's reach are not parted by where the
#                chair stands
#  K_NOT_CHOSEN  something that could be the container IS in view, and nothing ties it to the one the
#                person chose: the wrist has a lid and the selected id is in none of the scene camera's
#                detections (typed by hand, or a track perception has dropped), or the lid stands off the
#                scene camera's line to the selected one. The cure is to choose again. Never of a selected
#                id that is merely absent with nothing in view: that is K_UNSEEN (review of 2026-11-19, the
#                second: the scene camera's "was in none of its frames" read as this, and a container that
#                was only out of its view ended the line-up with "Choose it again")
#  K_FAULT       what watches delivers nothing: a camera with no frame, no pose or calibration for its
#                frames, the scene segmenter down or not for this container, the watch itself fallen silent.
#                "Vision", not "camera": half of these are not the camera (the same review)
NO_CURE = {
    K_SEVERAL: "The arm sees two things that could be the container",
    K_NOT_CHOSEN: "The arm cannot tell which container you chose",
    K_FAULT: "The arm's vision is not working",
}
# Which of them END a line-up once they have stood NO_CURE_S (line_up.BoxSource): said the while, the last
# line_up.LINGER_S with no button, as an arm that is not at rest is said. Not two tops in a frame: at
# 21aa7cc they ended no line-up, and the integrator put that back (2026-11-19): their sentence stands
# while they do, the line-up goes on reading sightings, and its own time bounds it. And never K_UNSEEN
ENDS = (K_NOT_CHOSEN, K_FAULT)
# Ten seconds: a frame late or a segmenter message lost are over in one or two, and a person is not left
# to drive for minutes at what no driving cures
NO_CURE_S = 10.0
# The words by which what watches says each, where it says no kind itself (the run's own watch does:
# press_demo.line_up_watch; tasks/box_watch and the scene locator say words alone). They are other
# modules' words, held to them by test_guide (each mark is in the source it is said to come from):
#  a camera that delivers nothing: tasks/survey.grab_frame and run_survey, perception/depth_source,
#  perception/scene_source (locate, _unplaced_why), perception/segmenter_source.describe, tasks/box_watch
FAULT_MARKS = (
    "no wrist camera",
    "no new wrist frame",
    "no wrist frame at",
    "no depth stamp yet",
    "depth not new",
    "depth not paired with colour",
    "no camera pose for the frame",
    "scene camera streams missing",
    "no TF between the scene camera",
    "to the scene camera (",
    "no TF from the scene camera",
    "the scene segmenter cannot be read",
    "no scene segmenter for this container",
    "frames upside down",
    "no scene depth cloud",
    "no frame newer than the look",
    "ABSENT: nothing publishes",
    "SILENT: no ",
    "not the scene camera (",
    "its frames are ",
    "older than the scene camera's",
    "no scene camera calibration",
)
#  two that could be it: the wrist finder's and the scene locator's "two places pass as this container";
#  the selection's "another lid-sized top ... in the same direction" and "two containers, and the selected
#  one ..." (press_demo.tie_refusal, alone_refusal)
SEVERAL_MARKS = ("two places pass", "another lid-sized top", "two containers")
#  something in view that is not tied to the selection: the wrist's watch and look, of a lid they have,
#  "... is in none of the scene camera's detections" (press_demo.UNTIED, wrist_first) and "... is not taken
#  for the selected container" (press_demo.tie_refusal). NOT the scene locator's "the selected container
#  (id) was in none of its N frames": that says only that the id is absent, of a container that may simply
#  be out of the scene camera's view, which is K_UNSEEN
NOT_CHOSEN_MARKS = ("in none of the scene camera's detections", "not taken for the selected container")
# one of those sentences stands this long before another of them takes its place (anything else replaces
# it at once: a container seen, the time out). What watches judges each frame alone, a word a second
# (survey.LINE_UP_NONE_S), and where one frame shows two lid-sized tops and the next shows one cut by the
# view, the panel would change between two sentences of a dozen words faster than either is read
UNSEEN_STANDS_S = 3.0

LOST_S = 1.5  # no box seen for this long: looking again
# ... and an instruction that is up ("Move ...", "Hold still, checking") stands through a container that has
# been out of every view for no longer than this; only then is the person told the arm cannot see it (or what
# else it is). 2026-11-20 13:40, RAMMP 1.5: sightings came 1.9 s and 5.1 s apart (the wrist's find sat on the
# edge of what ties it to the selection, with the scene camera's mount 14 deg off), and with LOST_S alone the
# panel read "Move closer, about 10 cm", "The arm cannot see the container. Move the chair in front of it",
# "Move closer, about 10 cm", "The arm cannot see ..." within 3.4 s: two sentences of a dozen words changing
# faster than either is read, of a container that stood in front of the arm throughout. Twice LOST_S: a gap
# of two seconds no longer shows, and one of five still does, once, after three (a container that has really
# gone out of view is said to have within that). It ends that long after the LAST SIGHTING, whatever the
# watch's words do meanwhile, and sooner where the watch delivers nothing: its "vision is not working" is
# said once that has stood LOST_S. Nothing is asked the while: the question needs a sighting within
# FRESH_S, and Open is taken on nothing older
OUT_STANDS_S = 3.0
FRESH_S = 1.0  # Open counts only on a box seen this recently
STEADY_S = 1.0  # the box has stood still this long ...
STEADY_M = 0.015  # ... within this much: the chair is not moving
LEAVE_S = 0.5  # READY is left once the box has been outside, moving, or not judged, this long
# "Hold still, checking" replaces a "Move ..." only once the box has read inside the boundary (or not judged)
# for this long without a break. A box on the boundary's edge reads inside on one wrist frame and a
# centimetre or two outside on the next (five frames a second, the 3 cm margin either side of it): without
# this the panel and the console changed between "Hold still, checking" and "Move closer a little" with
# every frame, 51 lines in 10 s in the review's replay of 2026-11-17, and the person was told to hold still
# where a small move was still to make. Half a second is two to three wrist frames. It delays no question:
# READY is asked the moment the box is inside, steady and planned, from whatever stood before it
HOLD_AFTER_MOVE_S = 0.5
MIN_SAY_S = 0.7  # an instruction stands at least this long before another replaces it
AXIS_SWITCH_M = 0.03  # the other axis is named only where its miss is this much the larger
# The OPPOSITE of the instruction last said ("back" after "closer", "right" after "left") by less than
# REVERSE_M is not said at once: "Hold still, checking" takes the old instruction's place at once (a reading
# on the other side is never answered with the old direction), and from then on every small instruction is
# held back until one side has been the answer for REVERSE_S without a break; that one is then said. The
# same 13:40: "Move closer, about 5 cm", 0.8 s later "Move back a little", a second later "Hold still,
# checking" and then the question, of a chair that had rolled 5 cm: the box read a centimetre or two past the
# boundary's far side for a moment (two cameras' sightings centimetres apart took each other's place), and
# the person was sent back out of where the arm opens it. A box that reads either side by turns is told to
# hold still, not to move a little one way throughout. A miss of REVERSE_M or more the other way is said at
# once, MIN_SAY_S or not
REVERSE_M = 0.05
REVERSE_S = 1.0
OPPOSITE = {"closer": "back", "back": "closer", "left": "right", "right": "left"}
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
            # each with its style, as the bridge's schema asks of every button and the orchestrator's own
            # progress panel sends them (Cancel "warning"): USER_INPUT_JSON_SCHEMA.md 6.3
            "buttons": [
                {"button_id": b, "label": label, "style": "warning" if b == CANCEL else "normal"}
                for b, label in self.buttons
            ],
        }


def unable(why):
    """The panel for a line-up that cannot go on, in the user's words: `why`, a sentence, and no button."""
    return Say(UNABLE, why[:1].upper() + why[1:])


def kind_of(why):
    """What ONE camera's word on having no container is of: K_FAULT (it delivers nothing), K_SEVERAL (two
    things that could be the container), K_NOT_CHOSEN (nothing ties what it sees to the selection), else
    K_UNSEEN (it looked and has nothing that could be it); "" for no word at all. A fault first: of a
    camera that delivers nothing, nothing else it says is known. The words of a whole find, both cameras'
    joined, are not for this: which camera a guidance depends on decides there (press_demo)."""
    why = str(why or "")
    if not why:
        return ""
    for kind, marks in ((K_FAULT, FAULT_MARKS), (K_SEVERAL, SEVERAL_MARKS), (K_NOT_CHOSEN, NOT_CHOSEN_MARKS)):
        if any(mark in why for mark in marks):
            return kind
    return K_UNSEEN


def says(kind):
    """What the person is told where no camera has the container, by what that is of (K_*; "": not said
    yet, which of a container that WAS seen is that the arm no longer sees it)."""
    return NO_CURE.get(kind, UNSEEN)


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
    caller's cancel. `budget_s`: how long the user has to line up (None: no limit). `again`: what it says
    last where that time runs out; `again_unseen`: ... where the container was never seen in it (saw_box).
    `instruction`: the newest "Move ..." it said (None: none yet), which the engine's error carries where
    the line-up ends without its yes."""

    budget_s: float = None
    again: str = AGAIN
    again_unseen: str = AGAIN_UNSEEN
    instruction: str = None
    _none_since: float = None  # with no container ever seen: since when what watches has said it has none
    _kind: str = ""  # what its newest word on having none is of ...
    _kind_since: float = None  # ... and since when without a break
    uncured: str = ""  # the kind no chair move cures whose sentence the time ran out on (update), else ""
    state: str = LOOKING
    started: float = None
    _track: list = field(default_factory=list)  # (t, x, y) of the fixes of the last STEADY_S
    _last_fix: Fix = None
    _axis: str = None
    _said: Say = None
    _said_at: float = None
    _bad_since: float = None  # in READY: since when the box has been outside, moving, or not judged
    _hold_since: float = None  # with a "Move ..." up: since when the box has read inside without a break
    _word: str = None  # the instruction last said ("closer"), until the question is asked
    _band: bool = False  # the box has read just past the boundary's other side: small instructions are held back
    _side: str = None  # ... the small instruction that is the answer now, and since when without a break
    _side_since: float = None

    # -- the box's own motion ------------------------------------------------------
    @property
    def saw_box(self):
        """Whether any sighting of the box came while it guided (it may have lost it since)."""
        return self._last_fix is not None

    def _see(self, fix):
        if fix is None or (self._last_fix is not None and fix.t <= self._last_fix.t):
            return
        self._last_fix = fix
        self._kind, self._kind_since = "", None  # (seen: whatever was said of having none starts anew)
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

    def _hold(self, now, say, at_once):
        """ "Hold still, checking" (`say`), unless a "Move ..." is up and the box has not yet read inside for
        HOLD_AFTER_MOVE_S without a break: the instruction then stands (a box on the boundary's edge)."""
        if self.state == MOVE and self._said is not None:
            self._hold_since = now if self._hold_since is None else self._hold_since
            if now - self._hold_since < HOLD_AFTER_MOVE_S:
                return self._said
        return self._answer(now, say, at_once=at_once)

    def _held_back(self, now, word, metres):
        """Whether the instruction `word` (a miss of `metres`) is not said yet, "Hold still, checking" in
        its place. The band opens where the box reads on the OTHER side of the boundary from the instruction
        last said, by less than REVERSE_M; while it is open every small instruction (either way: a box
        that reads a centimetre or two either side on alternate frames) is held back, until one side has
        been the answer for REVERSE_S without a break, which is then said. A miss of REVERSE_M or more is
        said at once and closes it; so does the question (update)."""
        if metres >= REVERSE_M or self._word is None:
            self._band, self._side, self._side_since = False, None, None
            return False
        if not self._band:
            if OPPOSITE.get(self._word) != word:
                return False
            self._band = True
        if word != self._side or self._side_since is None:
            self._side, self._side_since = word, now
        if now - self._side_since >= REVERSE_S:
            self._band, self._side, self._side_since = False, None, None
            return False
        return True

    def update(self, now, fix, reach, none="", looked=False, stands=False):
        """The tick: the newest Fix (or None: nothing new) and the Reach of the newest box seen (None: not
        judged yet). `none`: where no container is seen, what that is of by what watches' own word (K_*;
        "": it has not said, as while its wrist still turns to look). `looked`: what watches had looked,
        and found none, before this line-up began (a run that asks to be guided because its own look found
        no container): its word is said from the first tick. `stands`: `none` is a state of what watches
        that has stood by its own count whatever was seen meanwhile (the wrist's camera at fault while the
        scene camera has sightings, a watch fallen silent: line_up.BoxSource): said now, and no sighting
        on record hides it. Returns what to show."""
        now = float(now)
        if self.started is None:
            self.started = now
        if self.state in (GO, TIMED_OUT):
            return self._said
        self._see(fix)
        seen = self._last_fix
        stands = bool(stands) and none in NO_CURE
        if fix is None:
            # what the watch's word on having none is of, and since when without a break: counted from the
            # first tick after a sighting, not from when the sighting is given up as lost, so that a word
            # that began with the loss has stood its LOST_S by the time the loss is said
            if none != self._kind or self._kind_since is None:
                self._kind, self._kind_since = none, now
            if stands:
                self._kind_since = min(self._kind_since, now - LOST_S)  # (it has stood, by its source's count)
        lost = seen is None or now - seen.t > LOST_S or stands
        stood = none in NO_CURE and self._kind_since is not None and now - self._kind_since >= LOST_S
        if self.budget_s is not None and now - self.started > self.budget_s:
            last, kinds = self._said, {NO_CURE[kind]: kind for kind in ENDS}
            ends = ""
            if last is not None and last.state == LOOKING and last.message in kinds:
                # the time ran out on a state that ends a line-up (one the 120 s rule gave less than
                # NO_CURE_S). Its own sentence is the last word, and `uncured` tells the engine which it
                # was: "Press open again in front of it" is not true of vision that is not working. (Two
                # tops in view are not such a state: the time ran out, and that is what is said)
                ends = kinds[last.message]
            elif lost and none in ENDS and stood and last is not None and last.state in (MOVE, HOLD):
                # ... or on one whose sentence was due while an instruction still stood in its place
                # (OUT_STANDS_S): the same end, whatever the panel happened to show
                ends = none
            if ends:
                self.uncured = ends
                return self._answer(now, Say(TIMED_OUT, NO_CURE[ends]), at_once=True)
            # what was missing, in its last words: the container in reach, or the container in view at all
            return self._answer(now, Say(TIMED_OUT, self.again if self.saw_box else self.again_unseen), at_once=True)
        cancel = ((CANCEL, "Cancel"),)
        if lost:
            self._bad_since = self._hold_since = None
            self._side = self._side_since = None
            # no camera has it: lost (it was seen, and has not been for LOST_S), or what watches has had
            # none for as long. Before that it is still "looking": one frame without the container, the
            # first of a line-up or a late one, is not the arm unable to see what stands in front of it
            # (review of 2026-11-19: it showed "The arm cannot see the container. Move the chair in front
            # of it" to a person who was in place)
            prev = self._said
            if seen is None:
                self._none_since = None if not none else now if self._none_since is None else self._none_since
            up = prev is not None and seen is not None and not stands
            if up and now - seen.t <= OUT_STANDS_S and not (none == K_FAULT and stood):
                # an instruction is up and the container has only just gone out of view: it stands (the
                # question does not: it is taken back to "Hold still, checking", nothing to press on a box
                # that is not seen). Not against what watches delivering NOTHING once that has stood LOST_S:
                # the container is then not "out of view for a moment", and the instruction is of a place
                # nothing reads any more
                if prev.state in (MOVE, HOLD):
                    return prev
                if prev.state == READY:
                    return self._answer(now, Say(HOLD, "Hold still, checking", cancel, 50.0), at_once=True)
            waited = (
                stands
                or seen is not None
                or looked
                or (self._none_since is not None and now - self._none_since >= LOST_S)
            )
            if up and now - seen.t > OUT_STANDS_S:
                # a container that WAS seen and has been out past the hold: said as what the watch has now,
                # whatever its words have been doing meanwhile. Never "Looking for the container" again, and
                # never the instruction: two words of the watch taking each other's place once a second
                # (two tops in a frame, a lid nothing ties) started each other's LOST_S again for ever, and
                # the instruction stood for as long as they did (review of 2026-11-20)
                told = says(none)
                if none == K_FAULT and not stood and prev.state == LOOKING:
                    # ... but once the panel has moved past the hold to what the watch has, one late frame is
                    # not "the arm's vision is not working" until it has stood LOST_S (re-check of 2026-11-20):
                    # what was said meanwhile stands
                    told = prev.message
            elif none in NO_CURE and not stood:
                # ... nor is one word of a camera that delivers nothing, or of a second lid in the frame,
                # that state: until it has stood LOST_S what was said of having none stands, and where
                # nothing was, it is still looking (never, meanwhile, that the arm cannot see the container)
                told = prev.message if prev is not None and prev.state == LOOKING else SEARCHING
            else:
                told = says(none) if (none or seen is not None) and waited else SEARCHING
            if (
                prev is not None
                and prev.state == LOOKING
                and SEARCHING not in (prev.message, told)
                and prev.message != told
                and now - self._said_at < UNSEEN_STANDS_S
                and not stands  # (a state of the watch's own has stood by its source's count: said now)
            ):
                return prev
            return self._answer(now, Say(LOOKING, told, cancel), at_once=True)
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
            self._side = self._side_since = None
            return self._hold(now, Say(HOLD, "Hold still, checking", cancel, 50.0), at_once=self.state != READY)
        if good:
            # (in place: what follows is another approach, and no word of this one is held against it)
            self._hold_since = self._word = self._side = self._side_since = None
            self._band = False
            ready = Say(READY, ASK, ((OPEN, "Open"),) + cancel, 100.0)
            return self._answer(now, ready, at_once=True)
        if reach.inside and reach.planned is not False:
            self._side = self._side_since = None
            return self._hold(now, Say(HOLD, "Hold still, checking", cancel, 100.0), at_once=True)
        if reach.to_zone is None:
            self._hold_since = None
            return self._answer(now, Say(MOVE, "The arm cannot open it from here", cancel, 0.0), at_once=True)
        word, metres, axis = instruction(reach.to_zone, self._axis)
        pct = 100.0 * max(0.0, 1.0 - math.hypot(*reach.to_zone) / FAR_M)
        if self._held_back(now, word, metres):
            # the box reads just past the boundary's other side, or either side by turns: no instruction
            # yet, and never the old one (a reading on the other side is not answered with "closer")
            return self._answer(now, Say(HOLD, "Hold still, checking", cancel, pct), at_once=True)
        self._hold_since, self._axis = None, axis
        # the opposite of the instruction that is up is said at once: MIN_SAY_S would leave the old
        # direction standing over a box that reads on the other side
        turned = OPPOSITE.get(self._word) == word
        said = self._answer(now, Say(MOVE, words(word, metres), cancel, pct, (word, metres)), at_once=turned)
        self.instruction = said.message
        if said.hint is not None:
            self._word = said.hint[0]
        return said

    def press(self, button_id, now):
        """A button: Open flies only from READY on a box seen within FRESH_S; Cancel is the caller's.
        Returns True where the press is taken as the user's yes."""
        if button_id != OPEN or self.state != READY:
            return False
        if self._last_fix is None or float(now) - self._last_fix.t > FRESH_S:
            return False
        self._answer(float(now), Say(GO, "Opening the container", ((CANCEL, "Cancel"),), 100.0), at_once=True)
        return True
