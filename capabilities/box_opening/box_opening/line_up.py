"""The line-up before a run: the loop that tells the user which way to move, and asks before the arm flies.

The base class's example (capabilities/example, count_with_panel) is its pattern: one loop inside
execute() that sends a panel (ctx.update_user_panel), reads the buttons (ctx.poll_ui_event) and honours a
cancel, none of them waited on. What the panel says is guide.Guide's; what the guide is told each tick,
where the box was last seen and what the boundary says of that place, is a `source`'s:

    source.look(now) -> (Fix or None, Reach or None)     the newest sighting (None: nothing new since the
                                                         last tick) and the boundary's word on the box
    source.why_none                                      (where it has one) its newest word on why it sees
                                                         no container, "" while it has not looked or sees it:
                                                         the guide then says the arm cannot see it (owner
                                                         2026-11-19), and to move the chair
    source.none_kind                                     ... and what that is of (guide.K_*; a source without
                                                         it: guide.kind_of its word). What no chair move
                                                         cures is said as what it is, and where it stands
                                                         the source ends the line-up on it (`unable`,
                                                         `uncured`: BoxSource)

It ends one of five ways: GO (Open pressed on a fresh "Ready. Open the container?"), CANCELLED (the
panel's Cancel, or the orchestrator's), TIMED_OUT (the time for lining up ran out: the panel says to press
open again, and holds that for LINGER_S), GONE (what watched the box, or what judged its place, has ended:
`source.gone()`), UNABLE (the source says what no waiting cures, `source.unable`: the arm is not at its
rest pose; the panel says so for LINGER_S). Nothing here moves the arm or starts a run: the caller does,
on GO alone.

A goal with no UI lines up the same way (owner 2026-11-17, of a goal sent by hand: "don't end, just tell
you how far to move until it's close enough then asking if ready y/n"): no panel goes out (`live()` is
False), what the panel would say is the caller's to say on its console (`on_say`), and the question is
answered through `answers` (Answers: what the node heard on its answer topic). A y is Open's press, taken
only where Open would be (a fresh "Ready"); an n is Cancel's, read at every tick as Cancel is.

MockApproach is the mock's source: a box that comes into reach as a user driving up to a table brings it,
so the UI's side of a line-up can be rehearsed with no arm and no camera.

BoxSource is a real run's: two child processes for as long as the panel is up, neither of which can move
the arm. tasks/box_watch follows the container (the scene camera's sightings, as a run's own look gates
them), and tasks/reach_probe says what the boundary makes of where it stands (the run's own proof, on the
robot's URDF and chair model). Their lines are read on threads of their own, so a tick never waits on
either.
"""

import json
import math
import os
import subprocess
import threading
import time
from collections import deque

from box_opening import guide
from box_opening.reach_zone import Zone

GO, CANCELLED, TIMED_OUT, GONE, UNABLE = "go", "cancelled", "timed_out", "gone", "unable"

POLL_S = 0.05
# a panel that changed is sent at once where its words or buttons did, and no oftener than this where only its
# bar moved (the bar follows the box as it nears); unchanged, it is sent again every REFRESH_S: a goal the
# bridge ended on its own is opened again by the next send (node.Panel's period)
SEND_MIN_S = 0.2
REFRESH_S = 1.0
LINGER_S = 3.0  # "Press open again when you are in place" stays up this long before the action ends
PHASE = "line_up"  # the phase the orchestrator's progress names while the user lines up
YES, NO = "y", "n"  # the answers to the question where there is no panel to press on (Answers)


def read_answer(text):
    """An answer as it was typed or published: YES for y / yes, NO for n / no (any case, spaces round it);
    None for anything else, which is no answer at all (never a yes)."""
    word = str(text if text is not None else "").strip().lower()
    return YES if word in ("y", "yes") else NO if word in ("n", "no") else None


class Answers:
    """The answers a person gives where there is no panel: put() by whatever hears them (the node's answer
    topic, on the executor's thread), poll() by the loop that asks (the oldest not yet read). They are
    kept only between clear() and the action's end: an answer is to the action that is running, never
    to the next."""

    def __init__(self):
        self._lock = threading.Lock()
        self._got = deque()

    def put(self, text):
        """`text` as an answer (read_answer): kept where it is one, and returned; None where it is not."""
        word = read_answer(text)
        if word is not None:
            with self._lock:
                self._got.append(word)
        return word

    def poll(self):
        with self._lock:
            return self._got.popleft() if self._got else None

    def clear(self):
        with self._lock:
            self._got.clear()


class LineUp:
    """One line-up. `live()`: whether this execution has the UI (node.Capability._ui_live). `on_say(say)`:
    called where what the panel says changes (the node's progress, its log and its console). `say_min_s`:
    an instruction that only changes its distance ("Move closer, about 25 cm" after "about 30 cm"), and
    "Hold still, checking" and an instruction taking each other's place, are handed to `on_say` no oftener
    than this; the newest is said once that long has passed, and anything else (another direction, the
    question, the question taken away) at once. A chair driven in changes the distance several times a
    second, a box on the boundary's edge reads either side of it frame by frame, and a console of five
    lines a second is not read (0: every change, as the panel has it; the guide itself holds the edge's
    flicker for both: guide.HOLD_AFTER_MOVE_S). `answers`: an Answers, for the y or n of a person with no
    panel (None: the panel alone). `again`: what it says last where the time runs out (None: the panel's
    words); `again_unseen`: ... where the container was never seen in that time (None: the panel's).
    `on_unasked()`: called for a y that came with nothing asked, which is no yes."""

    def __init__(
        self,
        ctx,
        live,
        source,
        budget_s,
        on_say=None,
        clock=time.monotonic,
        sleep=time.sleep,
        answers=None,
        say_min_s=0.0,
        again=None,
        on_unasked=None,
        again_unseen=None,
    ):
        self._ctx = ctx
        self._live = live
        self._source = source
        self._clock = clock
        self._sleep = sleep
        self._on_say = on_say
        self._answers = answers
        self._say_min_s = float(say_min_s)
        self._on_unasked = on_unasked
        self.guide = guide.Guide(
            budget_s=budget_s, again=again or guide.AGAIN, again_unseen=again_unseen or guide.AGAIN_UNSEEN
        )
        self.cancel_pressed = False  # the panel's own Cancel ended it (the caller makes it the orchestrator's)
        self.said_no = False  # ... and it was the person's n, not a press on the panel
        self.said_yes = False  # it ended GO on the person's y (False: on Open, or not GO)
        self._sent = None  # the panel as last sent, None where the last send did not go out
        self._tried_at = None
        self._told = None  # (state, message) as last handed to on_say
        self._told_at = None
        self._told_hint = None  # ... and the word of its instruction ("closer"), where it was one

    def _press(self):
        """The newest press as (button id, whether it was an answer): the panel's Open or Cancel, or the
        person's y (Open's) or n (Cancel's) where there are answers to read. (None, False): none."""
        event = self._ctx.poll_ui_event()
        if event is not None and event.get("type") == "button":
            # any press ends the bridge's goal, and its panel with it: the next send opens it again
            self._sent = self._tried_at = None
            return event.get("button_id"), False
        answer = self._answers.poll() if self._answers is not None else None
        if answer == NO:
            self.said_no = True
            return guide.CANCEL, True
        if answer == YES:
            return guide.OPEN, True
        return None, False

    def _show(self, say, now):
        if (say.state, say.message) != self._told:
            # only the distance of the same kind of instruction changed, or "hold still" and an instruction
            # took each other's place (the box on the boundary's edge), and the last line was said a moment
            # ago: held, and said on a later tick where it still stands (this is called every tick with the
            # newest). The question, and its being taken away, are never held
            quick = (
                self._say_min_s > 0.0
                and self._told is not None
                and now - self._told_at < self._say_min_s
                and (
                    (
                        say.state == self._told[0] == guide.MOVE
                        and say.hint is not None
                        and self._told_hint == say.hint[0]
                    )
                    or {say.state, self._told[0]} == {guide.MOVE, guide.HOLD}
                )
            )
            if not quick:
                self._told, self._told_at = (say.state, say.message), now
                self._told_hint = None if say.hint is None else say.hint[0]
                if self._on_say is not None:
                    self._on_say(say)
        panel = say.panel()
        if self._tried_at is None or now - self._tried_at >= REFRESH_S:
            due = True
        elif self._sent is None or panel == self._sent:
            due = False  # a send that did not go out is tried again at the refresh, no oftener
        else:
            words = (panel["message"], panel["buttons"]) != (self._sent["message"], self._sent["buttons"])
            due = words or now - self._tried_at >= SEND_MIN_S
        if due:
            self._tried_at = now
            self._sent = panel if self._live() and self._ctx.update_user_panel(panel_json=panel) else None

    def run(self):
        """Until the user says yes, cancels, or the time runs out: GO, CANCELLED or TIMED_OUT. Or the source
        ended (GONE), or says what no waiting cures (its `unable`: the arm is not at rest): UNABLE, once the
        panel has said so for LINGER_S, with no button but a cancel still read."""
        timed_out_at = unable_at = None
        gone = getattr(self._source, "gone", None)
        while True:
            now = self._clock()
            if self._ctx.cancelled():
                return CANCELLED
            if gone is not None and gone():
                return GONE
            why = getattr(self._source, "unable", "")
            if why:
                unable_at = now if unable_at is None else unable_at
                self._show(guide.unable(why), now)
                if now - unable_at >= LINGER_S:
                    return UNABLE
                if self._press()[0] == guide.CANCEL:  # pressed on the panel before this one, or an n
                    self.cancel_pressed = True
                    return CANCELLED
                self._sleep(POLL_S)
                continue
            pressed, answered = self._press()
            if pressed == guide.CANCEL:
                self.cancel_pressed = True
                return CANCELLED
            if pressed is not None:
                if timed_out_at is None and self.guide.press(pressed, now):
                    self.said_yes = answered
                    self._show(self.guide.update(now, None, None), now)  # "Opening the container"
                    return GO
                if answered and self._on_unasked is not None:
                    self._on_unasked()  # a y with nothing asked (or asked of a box no longer seen): no yes
            fix, reach = (None, None) if timed_out_at is not None else self._source.look(now)
            if timed_out_at is None and getattr(self._source, "unable", ""):
                # heard on this tick (the arm is not at its rest pose; a state no chair move cures has
                # stood its time): said as what it is, at the top, and never first as a container the arm
                # cannot see. As first written the tick that heard the arm's word showed one panel and one
                # console line of "The arm cannot see the container. Move the chair in front of it" before
                # "The arm is not at its rest pose" (review of 2026-11-19)
                continue
            # (read after look(): what the source heard this tick is in it)
            kind = getattr(self._source, "none_kind", None)
            if kind is None:
                kind = guide.kind_of(getattr(self._source, "why_none", ""))
            say = self.guide.update(now, fix, reach, none=kind, looked=bool(getattr(self._source, "looked", False)))
            self._show(say, now)
            if say.state == guide.TIMED_OUT:
                timed_out_at = now if timed_out_at is None else timed_out_at
                if now - timed_out_at >= LINGER_S:
                    return TIMED_OUT
            self._sleep(POLL_S)


# The mock's table, as the arm stands to it (m, the arm's frame). Stand-ins: the mock sees nothing and reaches
# nothing. The band is about where flatbot's arm opened boxes on 2026-11-12 (0.63 and 0.68 m out).
MOCK_ZONE_X = (0.50, 0.80)
MOCK_ZONE_Y = (-0.20, 0.20)
MOCK_PATH = ((1.00, 0.35), (0.70, 0.35), (0.70, 0.0))  # too far and to the left; closer; then left, into reach


def mock_zone():
    zone = Zone.blank((0.0, 1.0), (-0.5, 0.5), key="mock")
    nx, ny = zone.cells.shape
    for i in range(nx):
        for j in range(ny):
            x, y = zone.centre(i, j)
            zone.set(
                i,
                j,
                MOCK_ZONE_X[0] - 1e-9 <= x <= MOCK_ZONE_X[1] + 1e-9
                and MOCK_ZONE_Y[0] - 1e-9 <= y <= MOCK_ZONE_Y[1] + 1e-9,
            )
    return zone


class MockApproach:
    """A box that comes along MOCK_PATH in `pace_s` seconds from the first look and then stands still (0:
    in place at once), seen every tick, judged by mock_zone no further out than `far_m` (the node's
    line_up_far_m; None: no bound); the run "plans" wherever it is inside."""

    def __init__(self, pace_s=0.0, path=MOCK_PATH, far_m=None):
        self._pace_s = max(0.0, float(pace_s))
        self._path = [tuple(float(v) for v in p) for p in path]
        self._zone = mock_zone().within(far_m=far_m)  # the deployment's bound, as a real line-up takes it
        self._t0 = None

    def at(self, t):
        """Where the box is `t` seconds into the approach."""
        legs = [math.dist(a, b) for a, b in zip(self._path, self._path[1:], strict=False)]
        if self._pace_s <= 0.0 or t >= self._pace_s:
            return self._path[-1]
        gone = max(0.0, t) / self._pace_s * sum(legs)
        for (a, b), length in zip(zip(self._path, self._path[1:], strict=False), legs, strict=False):
            if gone <= length:
                f = gone / length if length else 1.0
                return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            gone -= length
        return self._path[-1]

    def look(self, now):
        self._t0 = now if self._t0 is None else self._t0
        xy = self.at(now - self._t0)
        return guide.Fix(now, xy, "mock"), self._zone.reach(xy, planned=True)


# ------------------------------------------------------------------------------------- a real run's source
MARK = "@@ "  # a child's line that is for the engine (tasks/box_watch, tasks/reach_probe); the rest is its log
PROOF_NEAR_M = guide.STEADY_M  # a proof speaks for a box this near the place it was made for
VIEW_JOINTS = 7  # the arm's joints: a view the run says it looks from names them all, or is not taken
STOP_WAIT_S = 2.0  # a child asked to end is killed after this long
# reach_probe keeps the cells it asked this line-up only where it ends of itself: told to quit, it finishes
# the proof in flight (0.8 to 2 s by its own notes), writes its grid and exits. It is given this long to,
# before it is ended as any child is and what it had not yet written is asked again next time
PROBE_QUIT_S = 2.5
# reach_probe's proofs are numpy's, whose BLAS starts a thread for every core. It works beside the arm's driver
# and the run's own process on the robot's computer, so it is held to one
PROBE_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
# the end of a child's own log is kept, for the error of one that died: this many lines hold a traceback's
# last frame and its exception ...
TAIL_LINES = 4
TAIL_WAIT_S = 0.5  # ... and its reader is given this long to read what it wrote as it died
# a watch's word on having no container still stands where its newest saying is no older than this: it says
# one a second (survey.LINE_UP_NONE_S, box_watch.LOOK_S), and a watch that has fallen silent says nothing
NONE_FRESH_S = 3.0


class Child:
    """A child process whose marked stdout lines are messages: (when it was read, the message), oldest first.
    `log(text)` gets every other line it prints, and the last TAIL_LINES of them are kept (last_lines). It is
    ended by terminate(), then a kill (stop)."""

    def __init__(self, argv, env, log=None, clock=time.monotonic):
        self._log, self._clock = log, clock
        self._lock = threading.Lock()
        self._got = []
        self._said = deque(maxlen=TAIL_LINES)
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,  # a Ctrl+C at the node's terminal is the node's to pass on
        )
        self._reader = threading.Thread(target=self._read, name="line-up-child", daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(MARK):
                try:
                    msg = json.loads(line[len(MARK) :])
                except ValueError:
                    continue
                with self._lock:
                    self._got.append((self._clock(), msg))
                continue
            with self._lock:
                self._said.append(line)
            if self._log is not None:
                self._log(line)

    def messages(self):
        with self._lock:
            got, self._got = self._got, []
        return got

    def send(self, msg):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):  # it has ended: its absence is seen in what it no longer says
            pass

    def alive(self):
        return self.proc.poll() is None

    def last_lines(self, wait_s=TAIL_WAIT_S):
        """The end of its own log. Of a child that has ended: what it said as it died, its reader given
        `wait_s` to have read it."""
        if not self.alive():
            self._reader.join(wait_s)
        with self._lock:
            return list(self._said)

    def stop(self, wait_s=STOP_WAIT_S, quit_s=0.0):
        """terminate(), then a kill `wait_s` later. `quit_s`: a child already told to end of itself (its own
        quit line) is first given that long to."""
        if quit_s > 0.0 and self.proc.poll() is None:
            try:
                self.proc.wait(quit_s)
            except subprocess.TimeoutExpired:
                pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(wait_s)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class BoxSource:
    """The box as its watch sees it, judged by the boundary reach_probe fills. `watch`: box_watch's command
    line, run as a child of this source's own; or a process already running that says the same lines (the
    run itself, with --line-up: node.Run), which is its owner's to end. Such a run says the wrist camera's
    fixes ("by": "wrist") and, between them, the scene camera's: the wrist's are kept to while one is fresh
    (guide.LOST_S). `probe_argv(urdf, root)`: reach_probe's, once the watch has said which robot this is
    (run with PROBE_ENV). `far_m`: the deployment's bound on the boundary (None: none). `refused`: places
    ((x, y), or (x, y, lid z), the arm's frame) where a run itself found no press (a line-up in the middle
    of a run, whose box still stands there as it begins; every such place of the goal, and of the goal a
    line-up is carried from: node.Asking.refused): their cells do not work for this line-up whatever the
    probe's proof says of them, as a cell its own proof refuses does not. The run has the planner and the
    table's edges and the probe has neither, and where the two disagree the run's word is the last: without
    this the user would be asked "Ready?" on the very place the run refused. A place is refused by where it
    is on the table alone, at whatever height this line-up's watch reads the lid: the run's fix and the
    watch's are two readings of one lid (the scene camera's against the wrist's, centimetres apart in
    height), and as first written a refused place counted only within 15 mm of the run's own lid height and
    in that height's grid (review of 2026-11-17). `unseen`: where the line-up begins because the run's own
    look found NO container (press_demo's "NOT SEEN"), that look's word on why: the watch's word on why it
    sees none (why_none, `looked`) until the watch says its own or sees one, so that the person is told at
    once that the arm cannot see the container, not "Looking for the container" through the seconds the
    wrist takes to turn to its view. `looked` False: that look is not to be said (the engine's, after a
    yes of the same goal: the watch had the container a moment ago where the person now stands, and it
    looks first, as at any line-up's start). Nothing else of a line-up starts from a container having been seen:
    the wrist's view is chosen by the arm's own reach (tasks/survey.angled_views), and reach_probe does
    nothing before the first sighting, whose lid height names its grid (tasks/reach_probe.Server.step).
    So for such a line-up reach_probe is STARTED at the first sighting, not at the watch's first word: one
    that cannot start, or dies, ended the action line_up_unavailable, not recoverable, of a run that at
    21aa7cc ended a recoverable not_found and of a container that may never come into view (review of
    2026-11-19). The cost is the probe's own start-up after that sighting, before the first direction.

    What a "none" is OF (`none_kind`: the watch's own "kind" where it says one, the run's; else
    guide.kind_of its word) decides what the person is told, and whether moving the chair can cure it. A
    kind that cannot (guide.NO_CURE: a camera that delivers nothing, two things that could be the
    container, a container nothing ties to the selection) ends the line-up once it has been the watch's
    word without a break for guide.NO_CURE_S, that word still fresh (NONE_FRESH_S): `unable` (its
    sentence, said for the last LINGER_S of that with no button) and `uncured` (which it was). A word of
    another kind, or a sighting, starts that time again: a segmenter message late for one look, or a
    second lid passing through the view while the chair turns, does not end a guidance.

    close() ends the children that are this source's own, the probe by its own quit so that it keeps its
    grid; it can take PROBE_QUIT_S, so what must not wait is done before it."""

    def __init__(
        self,
        watch,
        probe_argv,
        env,
        far_m=None,
        log=None,
        clock=time.monotonic,
        spawn=Child,
        refused=(),
        unseen="",
        looked=True,
    ):
        self._probe_argv, self._env, self._far_m = probe_argv, env, far_m
        self._refused_at = [(float(xy[0]), float(xy[1])) for xy in refused or () if len(xy) >= 2]
        self._log, self._clock, self._spawn = log, clock, spawn
        self._owns_watch = isinstance(watch, list | tuple)
        self.watch = spawn(watch, env, log, clock) if self._owns_watch else watch
        self.probe = None
        self._fix = None  # the newest sighting: (Fix, lid z, table z, rough)
        self._wrist_at = None  # when the wrist camera last saw the box (its newest fix's time)
        self._q_from = None  # the joints the wrist looks from, once the run has said them ({"view": ...})
        self._zone = None
        self._proof = None  # the newest proof: ((x, y, z), ok)
        self._refuted = set()  # (lid z of the grid, i, j): cells a proof for the box itself refused this line-up
        # the watch's newest word on why it sees no box; none once it has seen it since. Until its first word,
        # the run's own where its look found none (`unseen`) and that look is to be said (`looked`)
        self.looked = bool(unseen) and bool(looked)  # the run HAD looked: its word is said from the first tick
        self.why_none = str(unseen or "") if self.looked else ""
        # what that is of, as the guide is told it (guide.K_*, "" while the watch has said nothing): the
        # run's own look's none is a container that is not in view
        self.none_kind = guide.K_UNSEEN if self.looked else ""
        self._lazy_probe = bool(unseen)  # reach_probe is started at the first sighting, not before
        self._robot = None  # which robot this is, as the watch said it, until the probe is started on it
        # since when the watch's "none" has been of the kind it is now, without a break, and when last said
        self._said_since = self._said_at = None
        self.uncured = ""  # the kind no chair move cures that this line-up ended on (`unable` says it)
        self.why_gone = ""  # where it is the probe that ended (gone): that, and the end of its log
        self.unable = ""  # what the watch said that no waiting cures (guide.ARM_NOT_AT_REST): the line-up ends
        self.viewed = False  # the run said its wrist stands at the line-up's view: it has turned

    def _hear(self):
        """What the children have said since the last tick. True where there is a new sighting."""
        new = False
        for t, msg in self.watch.messages():
            if "robot" in msg and self.probe is None:
                self._robot = msg["robot"]
                if not self._lazy_probe:
                    self._start_probe()
            elif "fix" in msg:
                f = msg["fix"]
                by_wrist = f.get("by") == "wrist"
                if not by_wrist and self._wrist_at is not None and t - self._wrist_at <= guide.LOST_S:
                    # the scene camera's sighting, said where a wrist frame showed no box: it has a box in
                    # front of the arm centimetres from where the wrist has it, often roughly and at another
                    # height. While the wrist's fix is fresh it is left out whole: it does not replace the
                    # fix, enter the guide's track (the box would have "moved") or go to the probe
                    continue
                xyz = [float(v) for v in f["xyz"]]
                fix = guide.Fix(t - float(f.get("age_s") or 0.0), (xyz[0], xyz[1]), "wrist" if by_wrist else "scene")
                if by_wrist:
                    self._wrist_at = fix.t
                self._fix = (fix, xyz[2], f.get("table_z"), bool(f.get("rough")))
                self.why_none = ""  # it is seen: whatever it said before is no longer why
                self.none_kind = ""
                self._said_since = self._said_at = None
                self.looked = False
                self._start_probe()  # (where it waited for this: a line-up that began with no container seen)
                new = True
            elif "none" in msg:
                self.why_none = str(msg["none"])
                if self.why_none == guide.ARM_NOT_AT_REST:
                    self.unable = self.why_none  # what no waiting cures: the line-up ends on it (LineUp.run)
                    continue  # (the arm's word, not a camera's on a container: it is of no kind)
                kind = str(msg.get("kind") or "") or guide.kind_of(self.why_none)
                if kind != self.none_kind or self._said_since is None:
                    self._said_since = t
                self.none_kind, self._said_at = kind, t
            elif "view" in msg:
                self.viewed = True
                # said once, as the wrist stands at the line-up's view: the run starts its reach from these
                # joints, not from rest, so the probe proves the box's own place from them ("q_from")
                try:
                    q = [float(v) for v in msg["view"]["joints"]]
                except (TypeError, KeyError, ValueError):
                    q = []
                if len(q) == VIEW_JOINTS:
                    self._q_from = q
                elif self._log is not None:
                    self._log(
                        "the view's joints were not heard as %d numbers: the probe is not told them" % VIEW_JOINTS
                    )
        if self.probe is not None:
            for _t, msg in self.probe.messages():
                if "zone" in msg:
                    self._zone = Zone.from_dict(msg["zone"]).within(far_m=self._far_m)
                    self._refute()
                elif "proof" in msg:
                    p = msg["proof"]
                    self._proof = (tuple(float(v) for v in p["xyz"]), bool(p["ok"]))
                    if not p["ok"] and self._log is not None:
                        # why the run's own proof refuses the place: the panel only shows the way to another
                        self._log(
                            "no press plans for the box at [%.3f, %.3f, %.3f] (proven from %s): %s"
                            % (
                                *self._proof[0],
                                "the view's joints" if p.get("q_from") else "rest",
                                str(p.get("why", ""))[:200],
                            )
                        )
                    zone = self._zone
                    if not p["ok"] and zone is not None and zone.serves(p["xyz"][2]) and zone.index(p["xyz"]):
                        # the grid has this cell working and the run's proof for the box itself does not plan
                        # there: for this line-up the cell does not work, and the user is shown the way to one
                        # that does (a cell is asked at its centre and its bucket's height, the box where it is)
                        self._refuted.add((zone.lid_z, *zone.index(p["xyz"])))
                        self._refute()
        return new

    def _start_probe(self):
        """reach_probe, on the robot the watch named: at the watch's first word, or at the first sighting
        where this line-up began with no container seen (`unseen`). Once."""
        robot, self._robot = self._robot, None
        if self.probe is not None or robot is None:
            return
        if robot.get("urdf"):
            env = dict(os.environ if self._env is None else self._env, **PROBE_ENV)
            self.probe = self._spawn(self._probe_argv(robot["urdf"], robot.get("root")), env, self._log)
        elif self._log is not None:
            self._log("the robot's URDF was not heard: no boundary can be worked out")

    def _stands(self, now):
        """Whether what the watch says of having no container is a state no chair move cures that has stood
        its time: the line-up then ends on it (`unable`, `uncured`), as on an arm that is not at rest."""
        kind = self.none_kind
        if self.unable or kind not in guide.NO_CURE or self._said_since is None:
            return
        fresh = now - self._said_at <= NONE_FRESH_S
        if fresh and now - self._said_since >= guide.NO_CURE_S - LINGER_S:
            self.uncured, self.unable = kind, guide.NO_CURE[kind]

    def _refute(self):
        """The grid in hand without the cells that do not work for this line-up: those a proof for the box
        itself refused (at that grid's height), and the cell of every place a run refused, in whichever
        grid this is (`refused`: by where it is on the table, at any lid height)."""
        if self._zone is not None:
            for lid_z, i, j in self._refuted:
                if lid_z == self._zone.lid_z:
                    self._zone.set(i, j, False)
            for xy in self._refused_at:
                ij = self._zone.index(xy)
                if ij is not None:
                    self._zone.set(*ij, False)

    def _planned(self, xy, z):
        """The proof's word for a box at `xy`, lid top `z`: True, False, or None where none was made there.
        False, whatever the proof and whatever `z`, within PROOF_NEAR_M of a place a run itself refused
        (`refused`)."""
        for rx, ry in self._refused_at:
            if math.hypot(rx - xy[0], ry - xy[1]) <= PROOF_NEAR_M:
                return False
        if self._proof is None:
            return None
        (px, py, pz), ok = self._proof
        near = math.hypot(px - xy[0], py - xy[1]) <= PROOF_NEAR_M and abs(pz - z) <= PROOF_NEAR_M
        return ok if near else None

    def _too_far(self, xy):
        """The boundary's word on a box further from the arm's base than the deployment's bound (`far_m`),
        which needs no grid: nothing out there is in the boundary whatever the proof finds, so the box is not
        inside, and the way in is toward the base, by what it is beyond the bound at the least. None where
        there is no bound, or the box and the cell that holds it are within it."""
        if self._far_m is None:
            return None
        r = math.hypot(xy[0], xy[1])
        beyond = r - self._far_m
        if beyond <= 0.0 and self._zone is not None:
            # within the bound itself, in a cell whose centre is beyond it: the bound goes by the cell's
            # centre (Zone.within, reach_probe --far-m), so that cell is never asked and never works. Without
            # this a box driven in from too far was told "closer" up to the bound and "hold still" just
            # inside it, where holding still gets no answer
            ij = self._zone.index(xy)
            if ij is not None:
                beyond = math.hypot(*self._zone.centre(*ij)) - self._far_m
        if beyond <= 0.0 or r <= 0.0:
            return None
        back = beyond / r
        return guide.Reach(False, (-xy[0] * back, -xy[1] * back))

    def look(self, now):
        new = self._hear()
        self._stands(now)
        if self._fix is None:
            return None, None
        fix, z, table_z, rough = self._fix
        reach = None
        if self._zone is not None and self._zone.serves(z):
            reach = self._zone.reach(fix.xy, None if rough else self._planned(fix.xy, z))
            if rough and reach is not None and reach.inside:
                reach = None  # a rough sighting says which way to move, and is never asked on
        if reach is None:
            # the grid cannot say yet (blank at this lid's height, or still filling): a box beyond the
            # deployment's bound is told to come closer all the same, not to hold still for it
            reach = self._too_far(fix.xy)
        if new and self.probe is not None:
            # the place is proven only where the boundary has it inside and no proof speaks for it yet
            prove = reach is not None and reach.inside and reach.planned is None
            line = {"box": [fix.xy[0], fix.xy[1], z], "table_z": table_z, "prove": prove}
            if self._q_from is not None:
                line["q_from"] = self._q_from
            self.probe.send(line)
        return (fix if new else None), reach

    def gone(self):
        """What the line-up cannot go on without has ended: the watch (nothing sees the box any more), or the
        probe once it was started (nothing judges where the box stands: the panel would say "Hold still,
        checking" until the time ran out, and the user be told the container was not in place). `why_gone`
        then says it was the probe, with the end of its log."""
        if not self.watch.alive():
            return True
        if self.probe is None or self.probe.alive():
            return False
        if not self.why_gone:
            said = " | ".join(line.strip()[:200] for line in self.probe.last_lines())
            self.why_gone = "reach_probe, which judges where the arm opens the container, ended" + (
                f" (it said last: {said})" if said else ""
            )
        return True

    def close(self, quit_s=PROBE_QUIT_S):
        """Both children ended (the watch only where it is this source's own). `quit_s`: how long the probe is
        given to end of itself on its quit before it is terminated; 0 where the caller cannot wait (the
        user's Open: the run's go comes after this). Its terminate keeps its grid too (reach_probe's
        end_on_terminate), cutting the proof in flight short where the quit lets it finish."""
        if self.probe is not None:
            self.probe.send({"quit": True})  # first: it finishes its proof while the watch is ended
        if self._owns_watch:
            self.watch.stop()
        if self.probe is not None:
            self.probe.stop(quit_s=quit_s)
