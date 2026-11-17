"""The flight view: a run's console, one short line a step, for the person watching it fly.

A run says what it does in full sentences, every number in them (tasks/*: the lines that begin
"[press_demo]"). Flown attended, eight runs in an evening (flatbot 2026-11-12), those were too long
to read while watching the arm: the owner asked for "easier to read in flight terminal logs so I
know if and what things are working or not working". This reads the same lines as they come and
writes, beside them:

    BOX OPENING   20:53:57
       find      ...   looking for the box
     1 FIND      ok    box 0.66 m out (wrist camera, 1 view)                18.8 s
       look      ...   moving to look at the button
     2 LOOK      ok    button found, 43 mm across, 0.68 m out                3.9 s
       press     ...   pressing the button
     3 PRESS     ok    21 N                                                  4.3 s
                 note  lid touched 41 mm higher than the camera read it
       grip      ...   rising 45 mm, opening the fingers, coming down round the knob
     4 GRIP      ok    knob held                                             7.9 s
     5 LIFT      ok
     6 CARRY     ok    lid to 20 cm from the box
     7 SET DOWN  ok    let go where the lid met the table
       home      ...   letting go and going home
     8 HOME      ok                                                          5.0 s
    RESULT  DONE — box open, lid on the table, arm home                       40 s

A step's lower-case line is written as it starts, its numbered line as it ends: `ok`, `FAIL`, or
`??` where the run could not tell and went on. `note` is something that worked but looked off,
`next` what the run does after a failure. The result says where the box, the lid and the arm were
left, and says the lid is on the table only of a set-down the run said it felt meet the table (its
"SET-DOWN CONTACT" line): flatbot 2026-11-17 16:40 let the lid go over the floor and this said
"lid on the table". A set-down that felt none is a failed step, the lid carried back to the box.
A lid that slipped out of the fingers on the way (the run's "the lid slipped on the way" line, the
knuckle read at the table) is never "on the table" either, whatever the set-down felt: the step that
failed is the carry, and the result says the lid slipped out and that where it lies is not known.

The engine writes it to its stdout. The run's full lines are kept in a file of their own beside the run
log (runs/console-<stamp>.log in box opening's state folder: node.NodeLogHandler), which the result's last
line names, so the console (`sheppy logs box_opening`, which shows stdout and stderr as one) is the flight
view and the node's own few lines. With the node's `full_console` on they are on the console too, between
the flight view's lines under their "[INFO] [...]" prefix, as before 2026-11-17.

A line-up (the chair guided to where the arm opens the box, before a run or in the middle of one that
found no press from where the chair stands) is a step of its own, said by the engine (guide_start,
guide_say, guide_end): each thing the person is told as it changes, the question, and how it ended.

       line up   ...   the arm rests and the wrist looks: move the chair as told
                 say   Move closer, about 15 cm
                 ask   Ready. Open the container?   y / n
     3 LINE UP   ok    yes                                                  23.0 s

It only reads. Nothing a run does depends on it, and nothing it meets may stop the reader that
feeds it (node.Run._read: a reader that died leaves the run blocked on a full pipe, mid-motion):
feed(), finish() and the guide's three never raise, and a line it does not know is passed over.
"""

import math
import re
import sys
import threading
import time

WIDTH = 68  # the column a step's seconds end at
PREFIX = re.compile(r"^\[press_demo\] ")
NUM = r"([-+]?\d+(?:\.\d+)?)"

SURVEY_VIEW = re.compile(r"^WRIST SURVEY (survey:\S+) \(.*?\): (.*)$")
FIX = re.compile(r"^(WRIST SURVEY|SCENE): box at \[%s, %s\].*— reaching out" % (NUM, NUM))
WRIST_VS_SCENE = re.compile(r"^WRIST LOOK: the box at \[.*?\]; the scene camera had it at \[.*?\].*?, (\d+) mm away")
NO_CONTAINER = re.compile(r"^WRIST LOOK: no container .*— the scene camera's box is taken")
HOME_FIRST = re.compile(r"^HOME FIRST: ")
LOOK_START = re.compile(r"^REACH-OUT: hand tilted (\d+) deg; looking")
AIM = re.compile(r"^AIM: button circle at pixel .*?, (\d+) mm across")
AIMED = re.compile(r"^AIMED from the look pose: button at \[%s, %s, %s\]" % (NUM, NUM, NUM))
TABLE_WARN = re.compile(r"^WARNING: the wrist reads its table ([-+]?\d+) mm from the scene camera's")
TURN = re.compile(r"^TURN: (\d+) deg short")
PRESS_START = re.compile(r"^(?:PRESS (\d+) of \d+: )?ONE MOTION from the look pose to below the button")
PRESS_LEAN = re.compile(r"^PRESS \(compliant\), the hand leaning (\d+) deg")
CONTACT = re.compile(r"^CONTACT: the pads met the box at [\d.]+, ([-+]?\d+) mm from the top")
PRESSED = re.compile(r"^PRESSED with ~(\d+) N")
PRESS_REFUSED = re.compile(r"^PRESS REFUSED \((.*?)\)")
GRIP_FROM_PRESS = re.compile(r"^GRIP STRAIGHT FROM THE PRESS: up (\d+) mm")
GRIP_FROM_LOOK = re.compile(r"^GRIP[^:]*: the fingers open to the knob")
ROSE_LITTLE = re.compile(r"^the hand rose only (\d+) mm")
TOUCHED = re.compile(r"^GRIP: the open fingertips met the lid")
CLOSE = re.compile(r"^CLOSE, then ONE MOTION: (.*?) —")
LID_GOES = re.compile(r"^lid goes to \[.*?\] — (\d+) mm from the box")
GRIPPED = re.compile(r"^LID GRIPPED")
GRIP_MISS = re.compile(r"^the grip straight from the press did not hold the knob \((.*)\)")
GRIP_BACK = re.compile(r"^the grip straight from the press: (.*?) — back to the look pose")
GRIP_FAILED = re.compile(r"^GRIP (FAILED|MISSED)")
BAND = re.compile(r"grip %s vs band \[%s, %s\]" % (NUM, NUM, NUM))
WAY_BACK = re.compile(r"^ONE MOTION from the button back to the look pose")
AFTER_KNOB = re.compile(r"^AFTER PRESS (\d+): knob (.*)$")
AFTER_LOST = re.compile(r"^AFTER PRESS (\d+): the wrist does not find the button")
KNOB_DOWN = re.compile(r"^the knob is not up \(%s mm above the lid\): pressing once more" % NUM)
KNOB_UNJUDGED = re.compile(r"^the knob is not judged after press (\d+)")
SETDOWN = re.compile(r"^SET-DOWN CONTACT")
SETDOWN_NONE = re.compile(r"^SET-DOWN: no table felt")
SLIPPED = re.compile(r"^the fingers read \S+ at the table, outside the grip band .*: the lid slipped on the way")
SLIPPED_HOME = re.compile(r"^the arm is home, but the lid slipped out of the fingers")
LID_SLIPPED = "slipped out of the fingers on the way (where it lies is not known)"
BACK_ON_BOX = re.compile(r"the lid is back on the box, and the arm is home")
RELEASE = re.compile(r"^RELEASE as the retreat starts")
PUT_BACK = re.compile(r"^PUT BACK: ")
REFUSED = re.compile(r"^REFUSED: (\S+) — ")
DONE = re.compile(r"^DONE — ")
NO_BOX = re.compile(r"^NO BOX\b")
STOP = re.compile(r"^STOP: (.*)$")
# no press plans from where the chair stands and the engine guides the chair (tasks/reach_press.no_way_in,
# press_demo's wrist look): not the run's end, so no STOP
NO_WAY_IN = re.compile(r"^NO WAY IN: (.*)$")
FAR_BOX = re.compile(r"^OUT OF REACH: (.*)$")
RUNNER_STOP = re.compile(r"^STOP: leg \S+ -> ")  # the runner's own word on one leg: the task's line says what it means

# a leg named in a STOP ("... stopped during lid:carry (...)") -> the step it is
LEG_STEPS = (
    ("survey:", "FIND"),
    ("reach:look", "LOOK"),
    ("reach:", "PRESS"),
    ("depth:", "PRESS"),
    ("grip:", "GRIP"),
    ("lid:carry", "CARRY"),
    ("lid:", "LIFT"),
    ("place:down", "SET DOWN"),
    ("place:touch", "SET DOWN"),
    ("place:", "HOME"),
    ("home", "HOME"),
)
DURING = re.compile(r"(?:stopped during|in) ([a-z]+:[a-z-]+|home)\b")


# "30 deg: <why that tilt's press does not plan>", as plan_reach joins them (tasks/reach_press)
TILT = re.compile(r"(\d+) deg: ")
INTO = re.compile(r"into (\w+)/")
# the part of the way in a refusal names, innermost first -> the words for it
WAY_PARTS = (
    ("the press stroke", "the press stroke"),
    ("the slide in", "the slide in"),
    ("the look pose", "the look pose"),
    ("the way down", "the way down"),
)


def _reason(text):
    """Why one tilt's press does not plan, in a few plain words: the part of the way in it names and what
    refused it there (the run's own gates: runner's sanity gate, runtime/body, kinematics' lines)."""
    part = next((words for key, words in WAY_PARTS if key in text), "the way in")
    if "past the arm's reach" in text:
        return "out of the arm's reach"
    if "sanity gate" in text or "excursion" in text:
        return "%s swings the arm wide" % part
    into = INTO.search(text)
    if into or "the chair and user" in text:
        who = "the chair or the user" if not into else "the user" if into.group(1) == "person" else "the chair"
        return "%s would pass too close to %s" % (part, who)
    if "cabinet" in text or "counter" in text or "the table" in text:
        return "%s would meet the table" % part
    if "LIMIT_MARGIN" in text or "joint limit" in text:
        return "%s runs a joint to its limit" % part
    if "singularity" in text or "joint jump" in text:
        return "%s is not a line the arm can follow" % part
    if "IK" in text or "did not converge" in text:
        return "%s leaves what the arm reaches" % part
    return "%s does not plan" % part


def no_press_reason(stop, out_m=None):
    """Why no press plans, from the run's own line: each tilt's real reason, the tilts that share one
    said together. Flatbot 2026-11-17 16:43: the line read "the box is out of the arm's reach (0.79 m
    out)" for a box well within it, where the slide in swung the arm wide at 30 and 35 deg and the press
    stroke crossed the user's padding at 40 and 45: the person was told the one thing that was not so.
    Only a box whose every tilt's look stands past the arm's reach is said to be out of it."""
    far = "" if out_m is None else " (%.2f m out)" % out_m
    parts = TILT.split(stop)
    if len(parts) >= 3:
        tilts = [(parts[i], _reason(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
    else:
        tilts = [(None, _reason(stop))]
    kinds = list(dict.fromkeys(why for _tilt, why in tilts))
    if kinds == ["out of the arm's reach"]:
        return "the box is out of the arm's reach%s" % far
    said = []
    for why in kinds:
        at = [tilt for tilt, w in tilts if w == why and tilt is not None]
        said.append(why + (" (%s deg)" % ", ".join(at) if at else ""))
    return "no press plans from here%s: %s" % (far, "; ".join(said))


def _plain(stop, out_m):
    """A STOP's reason in a few plain words; `out_m` how far out the box is, where known."""
    if "no reach-out press" in stop or "no press reaches" in stop:
        return no_press_reason(stop, out_m)
    if "does not come down to the knob" in stop:
        return "the hand cannot come down to the knob from here"
    if "the knob did not pop" in stop:
        return "pressed, but the button did not pop"
    if "MOVED the box" in stop:
        return "the press moved the box"
    if "the lid has no way off the box" in stop:
        return "no way to lift the lid from here"
    if "did not get home" in stop:
        return "the arm did not get home"
    m = re.search(r"stopped(?: during \S+)? \((.*?)\)", stop)
    if m:
        return "stopped part way (%s)" % m.group(1)[:50]
    short = re.split(r" \(| — |: ", stop, maxsplit=1)[0].strip()
    return short[:60] + ("..." if len(short) > 60 else "")


class FlightView:
    """One run's flight view: start(), feed() each console line of the run, finish() with how it
    ended. `write(line)` takes each line of the view (default: stdout, flushed); `clock` the
    seconds its durations are measured by."""

    def __init__(self, write=None, clock=time.monotonic):
        self._write = write or self._stdout
        self._clock = clock
        self._lock = threading.Lock()
        self.n = 0  # steps ended
        self.cur = None  # (name, started) of the step in flight
        self.t0 = None
        self.failed = None  # (n, name, why): the first step that failed for good
        self.last_fail = None  # ... and the latest, which a recovery may follow
        self.done = False
        self.out_m = None  # how far out the box (then its button) is
        self.source = None
        self.views = 0
        self.rivals = self.far = False
        self.aim_mm = None
        self.tilt = None
        self.presses = 0
        self.contact_mm = None
        self.knob = None  # "up" | "down" | None: the last the run read of it
        self.touched = self.setdown_met = self.slipped = False
        self.spot_mm = None
        self.one_motion = False
        self.lid = "on the box"
        self.log = None  # where the run's full lines are kept (the node's, node.Capability._lines_to_file)
        self.arm = None  # "home" | "holding where it stopped"
        self.going_home = False
        self.finished = False  # the result is written: a second finish() writes nothing
        self.lined_up = 0  # line-ups that ended with the person's yes
        self._how_said = False  # how to answer the question was said (once a run)

    @staticmethod
    def _stdout(line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    # -- writing -------------------------------------------------------------------
    def _say(self, line):
        try:
            self._write(line)
        except Exception:  # noqa: BLE001 -- a console that fails is not the run's problem
            pass

    def _begin(self, name, doing):
        """A step starts: the one in flight, if any, has ended well (the run went on from it)."""
        if self.cur is not None and self.cur[0] == name:
            return
        self._end("ok")
        self.cur = (name, self._clock())
        self._say("   %-9s ...   %s" % (name.lower(), doing))

    def _end(self, outcome, text="", name=None):
        """The step in flight ends (`name`: a step that never said it started ends now too)."""
        if self.cur is None and name is None:
            return
        took = None
        if self.cur is not None:
            took = self._clock() - self.cur[1]
            name = self.cur[0]
        self.cur = None
        self.n += 1
        line = "%2d %-9s %-5s %s" % (self.n, name, outcome, text)
        if took is not None:
            line = "%s %5.1f s" % (line.ljust(WIDTH - 8), took)
        self._say(line.rstrip())
        if outcome == "FAIL":
            self.last_fail = (self.n, name, text)

    def _note(self, text, kind="note"):
        self._say("%13s%-5s %s" % ("", kind, text))

    # -- the run's lines -------------------------------------------------------------
    def start(self, what="open_box", instance=None, find=True):
        """The run's first line. `find` False: a line-up comes before anything looks for the box
        (guide_start), and the find starts at its yes."""
        with self._lock:
            try:
                self.t0 = self._clock()
                of = "" if not instance else "   %s" % instance
                self._say("BOX OPENING   %s%s" % (time.strftime("%H:%M:%S"), of))
                if find:
                    self._begin("FIND", "looking for the box")
            except Exception:  # noqa: BLE001
                pass

    # -- the line-up (the engine's: node.Capability._line_up) ---------------------------
    def guide_start(self, doing="move the chair as told; the arm rests and the wrist looks"):
        """A line-up begins: a step of its own, whatever was in flight ended."""
        with self._lock:
            try:
                self._begin("LINE UP", doing)
            except Exception:  # noqa: BLE001
                pass

    def guide_say(self, message, ask=None, how=None):
        """What the person is told now: one line each time it changes (the engine holds a distance that
        only shrinks to one line a second: line_up.LineUp's say_min_s). `ask`: this is the question, and
        how it is answered here (" y / n" on the console; None: an instruction). `how`: said once a
        run under the first question, where the answer is given."""
        with self._lock:
            try:
                if ask is None:
                    self._note(message, "say")
                    return
                self._note("%s   %s" % (message, ask), "ask")
                if how and not self._how_said:
                    self._how_said = True
                    self._note(how, "how")
            except Exception:  # noqa: BLE001
                pass

    def guide_end(self, ok, text="", arm=None, again=False, find=True):
        """The line-up ended: with the person's yes (`ok`: the run goes on, and with `find` looks for
        the box, `again` where it had found it before), or any other way (`text` says which, and it is
        the run's end: the result names it). `arm`: where the arm was left, for the result."""
        with self._lock:
            try:
                if ok:
                    self.lined_up += 1
                    self.views = 0  # the find that follows counts its own
                    self.going_home = False  # (a home before the line-up is not the way home of what follows)
                    # ... nor is the press that had no way in the failure of a run that goes on from here
                    self.last_fail = None
                    self._end("ok", text, name="LINE UP")
                    if find:
                        self._begin("FIND", "looking for the box again" if again else "looking for the box")
                    return
                if self.cur is not None and self.cur[0] != "LINE UP":
                    self._end("ok")
                self._end("FAIL", text, name="LINE UP")
                self.failed = self.last_fail
                if arm:
                    self.arm = arm
            except Exception:  # noqa: BLE001
                pass

    def feed(self, text):
        """One console line of the run. Never raises."""
        with self._lock:
            try:
                self._line(PREFIX.sub("", str(text).strip()))
            except Exception:  # noqa: BLE001 -- a line it cannot read is passed over
                pass

    def _line(self, s):  # noqa: C901 -- one line, one rule
        m = SURVEY_VIEW.match(s)
        if m:
            self.views += 1
            said = m.group(2)
            if "two places pass" in said and not self.rivals:
                self.rivals = True
                self._note("more than one thing looks like the box; looking from another angle")
            elif "out of the arm's reach" in said and not self.far:
                self.far = True
                self._note("what it saw there is out of the arm's reach; looking again")
            return
        m = WRIST_VS_SCENE.match(s)
        if m:
            if int(m.group(1)) >= 50:
                self._note("scene camera puts it %s mm away; using the wrist's" % m.group(1))
            return
        if NO_CONTAINER.match(s):
            self._note("wrist camera found no box; using the scene camera's")
            return
        if HOME_FIRST.match(s):
            self._note("the arm was not at its rest pose; going there first")
            return
        m = FIX.match(s)
        if m:
            self.out_m = math.hypot(float(m.group(2)), float(m.group(3)))
            self.source = "wrist" if m.group(1) == "WRIST SURVEY" else "scene"
            views = "" if not self.views else ", %d view%s" % (self.views, "" if self.views == 1 else "s")
            if self.cur is None or self.cur[0] != "FIND":
                self._begin("FIND", "looking for the box")
            self._end("ok", "box %.2f m out (%s camera%s)" % (self.out_m, self.source, views))
            return
        if NO_BOX.match(s):
            self._fail("no box found", "FIND", final=True)
            return
        m = LOOK_START.match(s)
        if m:
            self._begin("LOOK", "moving to look at the button")
            return
        m = AIM.match(s)
        if m:
            self.aim_mm = int(m.group(1))
            return
        m = TABLE_WARN.match(s)
        if m:
            self._note("the two cameras disagree on the table's height by %s mm" % m.group(1).lstrip("+-"))
            return
        m = AIMED.match(s)
        if m:
            self.out_m = math.hypot(float(m.group(1)), float(m.group(2)))
            if self.cur is not None and self.cur[0] == "LOOK":
                across = "" if self.aim_mm is None else ", %d mm across" % self.aim_mm
                self._end("ok", "button found%s, %.2f m out" % (across, self.out_m))
            return
        m = TURN.match(s) or PRESS_LEAN.match(s)
        if m:
            self.tilt = int(m.group(1))
            return
        m = PRESS_START.match(s)
        if m:
            self.presses = int(m.group(1)) if m.group(1) else self.presses + 1
            self.contact_mm = None
            name = "PRESS" if self.presses == 1 else "PRESS %d" % self.presses
            self._begin(name, "pressing the button")
            return
        m = CONTACT.match(s)
        if m:
            self.contact_mm = int(m.group(1))
            return
        m = PRESSED.match(s)
        if m:
            tilt = "" if not self.tilt or self.tilt < 5 else ", hand tilted %d°" % self.tilt
            self._end("ok", "%s N%s" % (m.group(1), tilt), name="PRESS")
            if self.contact_mm is not None and abs(self.contact_mm) >= 25:
                way = "higher" if self.contact_mm > 0 else "lower"
                self._note("lid touched %d mm %s than the camera read it" % (abs(self.contact_mm), way))
            return
        m = PRESS_REFUSED.match(s)
        if m:
            self._fail("the press was refused: %s" % m.group(1), "PRESS")
            return
        m = GRIP_FROM_PRESS.match(s)
        if m:
            self.touched = False
            self._begin("GRIP", "rising %s mm, opening the fingers, coming down round the knob" % m.group(1))
            return
        if GRIP_FROM_LOOK.match(s):
            self.touched = False
            self._begin("GRIP", "coming down round the knob")
            return
        m = ROSE_LITTLE.match(s)
        if m:
            self._end("??", "the hand rose only %s mm; not coming down on the knob" % m.group(1), name="GRIP")
            self._note("going back to look at the knob", "next")
            return
        if TOUCHED.match(s):
            self.touched = True
            return
        m = LID_GOES.match(s)
        if m:
            self.spot_mm = int(m.group(1))
            return
        m = CLOSE.match(s)
        if m:
            self.one_motion = "lid:carry" in m.group(1)
            return
        if GRIPPED.match(s):
            self.lid = "in the hand"
            self.knob = "up"
            self._end("ok", "knob held%s" % (", fingertips touched the lid first" if self.touched else ""), name="GRIP")
            return
        m = GRIP_MISS.match(s)
        if m:
            band = BAND.search(m.group(1))
            on_air = band is not None and float(band.group(1)) > float(band.group(3))
            if "sanity gate" in m.group(1):
                why = "a planned move was refused, so the fingers never closed"
            else:
                why = "fingers closed on nothing" if on_air else "fingers stopped on something that is not the knob"
            self._fail(why, "GRIP")
            return
        m = GRIP_BACK.match(s)
        if m:
            if "no way off the box" in m.group(1):
                self._fail("no way to lift the lid from here", "GRIP")
            self._note("going back to look at the knob", "next")
            return
        if GRIP_FAILED.match(s):
            self._fail("fingers closed on nothing", "GRIP")
            return
        if WAY_BACK.match(s):
            self._begin("CHECK", "going back to look at the knob")
            return
        m = AFTER_LOST.match(s)
        if m:
            if self.cur is None or self.cur[0] != "CHECK":
                self._begin("CHECK", "looking at the knob")
            return
        m = KNOB_UNJUDGED.match(s)
        if m:
            self.knob = None
            self._end("??", "camera cannot find the knob; trying the grip", name="CHECK")
            return
        m = AFTER_KNOB.match(s)
        if m:
            if self.cur is None or self.cur[0] != "CHECK":
                self._begin("CHECK", "looking at the knob")
            if m.group(2).startswith("UP"):
                self.knob = "up"
                self._end("ok", "knob is up: the box is open")
            return
        m = KNOB_DOWN.match(s)
        if m:
            self.knob = "down"
            self._end("ok", "knob is down (%s mm); pressing again" % m.group(1), name="CHECK")
            return
        if SETDOWN.match(s):
            self.setdown_met = True
            return
        if SETDOWN_NONE.match(s):
            # the arm felt no table under the lid: it is not let go (reach_lid_off.carry_back)
            self._carried()
            self._fail("the arm felt no table under the lid; the lid was not let go", "SET DOWN", final=True)
            self._note("carrying the lid back to the box", "next")
            return
        if SLIPPED.match(s):
            # the fingers read shut past the knob at the table: the lid left them on the way there, and
            # whatever the set-down felt was not the lid in the fingers meeting the table. Flatbot's view
            # said "SET DOWN ok let go where the lid met the table" and "lid: on the table" of such a run
            # (review 2026-11-17). Which of the lift, the carry and the way down it slipped on is not known
            self._slipped()
            return
        if RELEASE.match(s) and self.slipped:
            self.going_home = True
            self._begin("HOME", "opening the fingers and going home")
            return
        if RELEASE.match(s):
            self._carried()
            if self.setdown_met:
                self._end("ok", "let go where the lid met the table", name="SET DOWN")
                self.lid = "on the table"
            else:
                # a release with no contact said: where the lid went is not known to be the table
                self._end("??", "let go, and no table was felt under the lid", name="SET DOWN")
                self.lid = "let go where no table was felt"
            self.going_home = True
            self._begin("HOME", "letting go and going home")
            return
        if PUT_BACK.match(s):
            self.lid = "on the box"
            self._note("putting the lid back on the box", "next")
            return
        m = REFUSED.match(s)
        if m:
            self._note("the safety gate refused a planned move (%s)" % m.group(1))
            return
        if DONE.match(s):
            self.done = True
            self.arm = "home"
            self._end("ok", name="HOME")
            return
        if RUNNER_STOP.match(s):
            return
        m = NO_WAY_IN.match(s) or FAR_BOX.match(s)
        if m:
            self._no_way_in(m.group(1), far=s.startswith("OUT OF REACH"))
            return
        m = STOP.match(s)
        if m:
            self._stop(m.group(1))

    def _carried(self):
        """The lid's way to its spot is over: whatever was in flight, the lift and the carry ended well."""
        self._end("ok")
        self._end("ok", name="LIFT")
        self._end(
            "ok",
            "" if self.spot_mm is None else "lid to %.0f cm from the box" % (self.spot_mm / 10.0),
            name="CARRY",
        )

    def _slipped(self):
        """The lid slipped out of the fingers on its way to the spot: the lid's way there is the step that
        failed, no set-down is said of it, and where the lid lies is not known."""
        if self.slipped:
            return
        self.slipped, self.setdown_met, self.lid = True, False, LID_SLIPPED
        self._end("ok")
        self._fail("the lid slipped out of the fingers on the way", "CARRY", final=True)

    def _no_way_in(self, why, far=False):
        """No press plans from where the chair stands, and the run does not end on it: the step it found
        that at fails with the real reason, and what follows is the chair's move (the engine's line-up)."""
        if self.cur is not None and self.cur[0] in ("FIND", "LOOK") and not far:
            self._end("ok")  # the look itself ended well: it is the press that has no way in
        if far:
            self._fail("what the wrist camera sees is out of the arm's reach", "FIND")
        else:
            self._fail(no_press_reason(why, self.out_m), "PRESS")
        self._note("the chair is to move: the run does not end here", "next")
        if "returning home" in why:
            self.going_home = True
            self._begin("HOME", "going home")

    def _fail(self, why, name, final=False):
        """A step fails; `final`: nothing follows it but the way home."""
        self._end("FAIL", why, name=name if self.cur is None else None)
        if final and self.failed is None:
            self.failed = self.last_fail

    def _stop(self, stop):
        """The run's STOP: the step in flight (else the one it names) failed for good."""
        if SLIPPED_HOME.match(stop):
            # the run's last word of a lid that slipped out: the arm IS home (its way there ended well),
            # the fingers empty, and the step that failed is the lid's way to its spot
            self.arm = "home"
            if self.cur is not None and self.cur[0] == "HOME":
                self._end("ok")
            self._slipped()  # said already where its "slipped on the way" line was read; else here
            return
        back = BACK_ON_BOX.search(stop) is not None
        if back:
            self.lid, self.arm = "on the box", "home"  # the run's own words: nothing held, the arm home
            if self.failed is not None:
                # the step that failed is said already (a set-down that felt no table): this is how it ended
                self._end("ok", "lid back on the box, arm home", name="PUT BACK")
                return
        name = None
        m = DURING.search(stop)
        if m:
            name = next((step for leg, step in LEG_STEPS if m.group(1).startswith(leg)), None)
        if name is None and back:
            name = "CARRY"  # the lid had nowhere to go: not the way home, which it flew
        if name is None:
            if stop.startswith("pressed") or "the press " in stop:
                name = "PRESS"
            elif "knob" in stop or "grip" in stop or "fingers" in stop:
                name = "GRIP"
            elif "home" in stop and "returning home" not in stop:
                name = "HOME"
            elif "press" in stop or "button" in stop:
                name = "PRESS"
            elif "lid" in stop:
                name = "CARRY"
            else:
                name = "RUN"
        if self.cur is not None and self.cur[0] in ("FIND", "LOOK") and ("press" in stop):
            # no press reaches the box the look found: the look itself ended well
            self._end("ok")
        if self.cur is not None and self.cur[0] != name and name in ("LIFT", "CARRY", "SET DOWN", "HOME"):
            self._end("ok")
        self._fail(_plain(stop, self.out_m), name, final=True)
        if "holds the lid" in stop or "the arm holds the lid" in stop:
            self.lid = "in the hand"
        if "arm holds" in stop or "The arm holds" in stop or "the arm holds" in stop:
            self.arm = "holding where it stopped"
        if "did not get home" in stop:
            self.arm = "not home"
        if "nothing moved" in stop and self.arm is None:
            self.arm = "where the look left it (not sent home)"
        if "returning home" in stop:
            self.going_home = True
            self._begin("HOME", "going home")

    def finish(self, code, cancelled=False, timed_out=False, held=False):
        """The run ended with exit code `code`: the step in flight closed, and the result. Never raises."""
        with self._lock:
            try:
                if not self.finished:  # (a line-up that ended the run has written its result already)
                    self.finished = True
                    self._finish(code, cancelled, timed_out, held)
            except Exception:  # noqa: BLE001
                pass

    def _finish(self, code, cancelled, timed_out, held):
        took = "" if self.t0 is None else "%3.0f s" % (self._clock() - self.t0)
        if self.cur is not None:
            if cancelled or timed_out:
                self._fail("cancelled" if cancelled else "stopped: the run took too long", None, final=True)
            elif self.cur[0] == "HOME" and self.arm is None:
                self.arm = "home" if code is not None else None
                self._end("ok")
            elif code == 0:
                self._end("ok")
            else:
                self._fail("the run ended here" + ("" if code is None else " (exit code %s)" % code), None, final=True)
        if held:
            self.lid = "in the hand"
        if code == 0 and self.done and not cancelled:
            # "on the table" only of a set-down the run felt meet the table; else what is known
            lid = "lid on the table" if self.setdown_met else "lid let go (no table was felt under it)"
            self._say(("RESULT  DONE — box open, %s, arm home" % lid).ljust(WIDTH - 8) + " " + took)
            self._where()
            return
        failed = self.failed or self.last_fail
        word = "CANCELLED" if cancelled else "FAILED"
        where = "" if failed is None else " at step %d (%s)" % (failed[0], failed[1])
        self._say(("RESULT  %s%s" % (word, where)).ljust(WIDTH - 8) + " " + took)
        if failed is not None and failed[2]:
            self._say("        why: %s" % failed[2])
        if self.presses == 0:
            box = "not pressed"
        elif self.knob == "up":
            box = "open (knob up)"
        elif self.knob == "down":
            box = "shut (knob down)"
        else:
            box = "pressed %d time%s, not confirmed open" % (self.presses, "" if self.presses == 1 else "s")
        arm = self.arm or ("not confirmed home" if not self.going_home else "home")
        self._say("        box: %s    lid: %s    arm: %s" % (box, self.lid, arm))
        self._where()

    def _where(self):
        """Where the run's full lines were kept (`log`: a place in box opening's state folder), where they were."""
        if self.log:
            self._say("        full lines: %s" % self.log)

    def relay(self, log):
        """`log(level, text)` wrapped: each line logged as before, then read here."""

        def both(level, text):
            log(level, text)
            self.feed(text)

        return both
