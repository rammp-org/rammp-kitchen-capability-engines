"""The line-up before a run: the loop that tells the user which way to move, and asks before the arm flies.

The base class's example (capabilities/example, count_with_panel) is its pattern: one loop inside
execute() that sends a panel (ctx.update_user_panel), reads the buttons (ctx.poll_ui_event) and honours a
cancel, none of them waited on. What the panel says is guide.Guide's; what the guide is told each tick,
where the box was last seen and what the boundary says of that place, is a `source`'s:

    source.look(now) -> (Fix or None, Reach or None)     the newest sighting (None: nothing new since the
                                                         last tick) and the boundary's word on the box

It ends one of four ways: GO (Open pressed on a fresh "Ready. Open the container?"), CANCELLED (the
panel's Cancel, or the orchestrator's), TIMED_OUT (the time for lining up ran out: the panel says to press
open again, and holds that for LINGER_S), GONE (what watched the box has ended: `source.gone()`). Nothing
here moves the arm or starts a run: the caller does, on GO alone.

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
import subprocess
import threading
import time

from box_opening import guide
from box_opening.reach_zone import Zone

GO, CANCELLED, TIMED_OUT, GONE = "go", "cancelled", "timed_out", "gone"

POLL_S = 0.05
# a panel that changed is sent at once where its words or buttons did, and no oftener than this where only its
# bar moved (the bar follows the box as it nears); unchanged, it is sent again every REFRESH_S: a goal the
# bridge ended on its own is opened again by the next send (node.Panel's period)
SEND_MIN_S = 0.2
REFRESH_S = 1.0
LINGER_S = 3.0  # "Press open again when you are in place" stays up this long before the action ends
PHASE = "line_up"  # the phase the orchestrator's progress names while the user lines up


class LineUp:
    """One line-up. `live()`: whether this execution has the UI (node.Capability._ui_live). `on_say(say)`:
    called where what the panel says changes (the node's progress and log)."""

    def __init__(self, ctx, live, source, budget_s, on_say=None, clock=time.monotonic, sleep=time.sleep):
        self._ctx = ctx
        self._live = live
        self._source = source
        self._clock = clock
        self._sleep = sleep
        self._on_say = on_say
        self.guide = guide.Guide(budget_s=budget_s)
        self.cancel_pressed = False  # the panel's own Cancel ended it (the caller makes it the orchestrator's)
        self._sent = None  # the panel as last sent, None where the last send did not go out
        self._tried_at = None
        self._told = None  # (state, message) as last handed to on_say

    def _show(self, say, now):
        if (say.state, say.message) != self._told:
            self._told = (say.state, say.message)
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
        """Until the user says yes, cancels, or the time runs out: GO, CANCELLED or TIMED_OUT."""
        timed_out_at = None
        gone = getattr(self._source, "gone", None)
        while True:
            now = self._clock()
            if self._ctx.cancelled():
                return CANCELLED
            if gone is not None and gone():
                return GONE
            event = self._ctx.poll_ui_event()
            if event is not None and event.get("type") == "button":
                # any press ends the bridge's goal, and its panel with it: the next send opens it again
                self._sent = self._tried_at = None
                if event.get("button_id") == guide.CANCEL:
                    self.cancel_pressed = True
                    return CANCELLED
                if timed_out_at is None and self.guide.press(event.get("button_id"), now):
                    self._show(self.guide.update(now, None, None), now)  # "Opening the container"
                    return GO
            fix, reach = (None, None) if timed_out_at is not None else self._source.look(now)
            say = self.guide.update(now, fix, reach)
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
STOP_WAIT_S = 2.0  # a child asked to end is killed after this long


class Child:
    """A child process whose marked stdout lines are messages: (when it was read, the message), oldest first.
    `log(text)` gets every other line it prints. It is ended by terminate(), then a kill."""

    def __init__(self, argv, env, log=None, clock=time.monotonic):
        self._log, self._clock = log, clock
        self._lock = threading.Lock()
        self._got = []
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
        threading.Thread(target=self._read, name="line-up-child", daemon=True).start()

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
            elif self._log is not None:
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

    def stop(self, wait_s=STOP_WAIT_S):
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
    run itself, with --line-up: node.Run), which is its owner's to end. `probe_argv(urdf, root)`:
    reach_probe's, once the watch has said which robot this is. `far_m`: the deployment's bound on the
    boundary (None: none). close() ends the children that are this source's own."""

    def __init__(self, watch, probe_argv, env, far_m=None, log=None, clock=time.monotonic, spawn=Child):
        self._probe_argv, self._env, self._far_m = probe_argv, env, far_m
        self._log, self._clock, self._spawn = log, clock, spawn
        self._owns_watch = isinstance(watch, list | tuple)
        self.watch = spawn(watch, env, log, clock) if self._owns_watch else watch
        self.probe = None
        self._fix = None  # the newest sighting: (Fix, lid z, table z, rough)
        self._zone = None
        self._proof = None  # the newest proof: ((x, y, z), ok)
        self._refuted = set()  # (lid z of the grid, i, j): cells a proof for the box itself refused this line-up
        self.why_none = ""  # the watch's last word on why it sees no box

    def _hear(self):
        """What the children have said since the last tick. True where there is a new sighting."""
        new = False
        for t, msg in self.watch.messages():
            if "robot" in msg and self.probe is None:
                robot = msg["robot"]
                if robot.get("urdf"):
                    self.probe = self._spawn(self._probe_argv(robot["urdf"], robot.get("root")), self._env, self._log)
                elif self._log is not None:
                    self._log("the robot's URDF was not heard: no boundary can be worked out")
            elif "fix" in msg:
                f = msg["fix"]
                xyz = [float(v) for v in f["xyz"]]
                fix = guide.Fix(t - float(f.get("age_s") or 0.0), (xyz[0], xyz[1]), "scene")
                self._fix = (fix, xyz[2], f.get("table_z"), bool(f.get("rough")))
                new = True
            elif "none" in msg:
                self.why_none = str(msg["none"])
        if self.probe is not None:
            for _t, msg in self.probe.messages():
                if "zone" in msg:
                    self._zone = Zone.from_dict(msg["zone"]).within(far_m=self._far_m)
                    self._refute()
                elif "proof" in msg:
                    p = msg["proof"]
                    self._proof = (tuple(float(v) for v in p["xyz"]), bool(p["ok"]))
                    zone = self._zone
                    if not p["ok"] and zone is not None and zone.serves(p["xyz"][2]) and zone.index(p["xyz"]):
                        # the grid has this cell working and the run's proof for the box itself does not plan
                        # there: for this line-up the cell does not work, and the user is shown the way to one
                        # that does (a cell is asked at its centre and its bucket's height, the box where it is)
                        self._refuted.add((zone.lid_z, *zone.index(p["xyz"])))
                        self._refute()
        return new

    def _refute(self):
        if self._zone is not None:
            for lid_z, i, j in self._refuted:
                if lid_z == self._zone.lid_z:
                    self._zone.set(i, j, False)

    def _planned(self, xy, z):
        """The proof's word for a box at `xy`, lid top `z`: True, False, or None where none was made there."""
        if self._proof is None:
            return None
        (px, py, pz), ok = self._proof
        near = math.hypot(px - xy[0], py - xy[1]) <= PROOF_NEAR_M and abs(pz - z) <= PROOF_NEAR_M
        return ok if near else None

    def look(self, now):
        new = self._hear()
        if self._fix is None:
            return None, None
        fix, z, table_z, rough = self._fix
        reach = None
        if self._zone is not None and self._zone.serves(z):
            reach = self._zone.reach(fix.xy, None if rough else self._planned(fix.xy, z))
            if rough and reach is not None and reach.inside:
                reach = None  # a rough sighting says which way to move, and is never asked on
        if new and self.probe is not None:
            # the place is proven only where the boundary has it inside and no proof speaks for it yet
            prove = reach is not None and reach.inside and reach.planned is None
            self.probe.send({"box": [fix.xy[0], fix.xy[1], z], "table_z": table_z, "prove": prove})
        return (fix if new else None), reach

    def gone(self):
        """The watch has ended: nothing sees the box any more."""
        return not self.watch.alive()

    def close(self):
        for child in (self.watch, self.probe) if self._owns_watch else (self.probe,):
            if child is not None:
                child.stop()
