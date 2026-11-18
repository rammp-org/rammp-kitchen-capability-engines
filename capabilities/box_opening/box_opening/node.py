"""The box_opening capability: press the food container's button, take its lid off, set the lid
down beside the container, and go home.

One action, open_box: the whole run (tasks/press_demo). It checks the robot is ready (driver,
planner, TF, gripper, cameras: nothing moves if not), brings the arm to its rest pose if it is
not there, then locates the container, looks, presses, grips, lifts, carries, sets the lid
down and goes home.

The action runs box opening's command line as a child process, exactly as it ran on flatbot
until now: its own process and executor, and Ctrl+C's stop path for a cancel (the motion in
flight is cancelled and the driver's answer awaited before it exits, runtime/abort.py). Its
console lines become this node's log lines, and the rows its run log gains become the
action's progress. A run is one motion from the press to the set-down, so
it cannot pause once it starts; a cancel stops it at once.

The run finds the container with sheppy's scene segmenter (/perception/object_detections),
the perception already running on the robot: only the object the orchestrator selected (its
instance_id, passed to the run as --instance-id), and a lid box stacked on it. No detector
runs in this node.

Started from the UI (the orchestrator's token for the execution in hand), the run shows a panel
for as long as it goes: what it is doing, a bar, and one button, Stop, which is the
orchestrator's cancel (Panel). No Pause: a run cannot pause. It ends on a final notice with Done
(capability_ui.RunUI: Done, failed in ui_text's plain words, or Stopped), waited on, bounded.

SAFETY: the arm moves only on a node with allow_execute:=true, set for an attended run (someone
at the e-stop). With it false, open_box is reported unavailable and a goal that comes anyway is
refused before anything runs. The mock moves nothing and is always available.

    ros2 launch box_opening box_opening.launch.py use_orchestrator:=false
    make action CAP=box_opening ACTION=open_box
"""

import json
import logging
import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from capability_engine import Action, Cancelled, CapabilityEngine, EngineError, run
from capability_ui import RunUI, cancel_own_run, wrap
from rammp_orchestrator_interfaces.msg import ActionAvailability
from rammp_orchestrator_interfaces.srv import Control

from . import LOGGER, line_up, ui_text
from .constants import AIR_CONTROL_ENV, state_dir
from .flight_view import FlightView
from .runtime.abort import EXIT_HALTED, EXIT_NOT_LINED_UP
from .worlds import PLANNER_WORLDS_AS_ENV, PLANNER_WORLDS_DIR_ENV

# ROS parameter -> default. Container and world name files in the image's config/ (or are
# absolute paths); the robot's per-run environment comes from its cameras, never from here.
PARAMS = {
    "allow_execute": False,
    "container": "ankou_pink.yaml",
    "world": "world_flatbot.yaml",
    # where runs keep their logs, captures and videos; "" = ~/.ros/box_opening
    "state_dir": "",
    # a video of both cameras from the grip to the set-down: for attended data collection
    "video": False,
    "viz": False,
    # debug: each --execute run keeps its camera frames (~10 MB; the newest 20 kept), and its
    # run record's flight files and room maps
    "mission_frames": False,
    "speed_scale": 1.0,
    # the cuRobo planner config whose layers (the chair and its user) and robot spheres every
    # plan-free line is held out of (runtime/body); unreadable, the run stops before it moves;
    # "" turns the check off (the bench). BOX_OPENING_BODY_LAYERS in the node's environment overrides
    "body_layers": "/config/robot/curobo/planner.yaml",
    # where the scene camera finds no container, look for it with the wrist camera from one or two
    # poses over the arm's side (a chair parked side-on to a counter: tasks/survey); off, such a
    # run stops with NO BOX before anything moves, as the owner had it (2026-10-22). Never with the
    # orchestrator's selection: that is the scene camera's detection
    "wrist_survey": False,
    # the wrist camera finds the container: the wrist turns where the arm rests so its camera sees
    # the table in front (up to four views, first where the scene camera has the box; a lid a view
    # cuts is looked at again), and what it finds there is the box; the scene camera's only where
    # the wrist finds none (a table faced from the front: the scene camera sees no lid whole where
    # the arm presses, tasks/survey). With the orchestrator's selection too: the wrist's box is
    # taken only where it stands in the direction the scene camera has the selected container
    # (press_demo.SelectionTie); the instance marked opened is the selected one either way
    "wrist_first": False,
    # with wrist_first, for a selection that is NO detection of the scene camera's (the selected id in none of
    # its frames: the UI's object list typed by hand): the wrist looks from its own views, tied to nothing, and
    # what it finds is taken only where it is the ONE container it sees; two or more, NO BOX. Off (the default):
    # such a run ends NO BOX without turning the wrist. On, the arm opens whichever single container stands in
    # front of it under the selected id: the owner's to turn on (press_demo.wrist_look)
    "wrist_alone": False,
    # how the arm flies the free air before the press (constants.AIR_CONTROL_ENV): "impedance" (the default),
    # every leg in the driver's joint impedance, one motion to under the button, as flown on 2026-11-12;
    # "position", the look and the way in's free-air lines in position control and slower, the arm then
    # standing about a second while the springs come in, and the press on them as ever (owner 2026-11-16:
    # the arm "a little floppy" on its way to the button). In position it follows its path closely and does
    # not give way to what it meets in the air. Not flown
    "air_control": "impedance",
    # the folder box opening hands the planner worlds from (worlds.PlannerTable): the run's measured table,
    # while a way HOME is planned. `planner_worlds_as` is that folder as the PLANNER's container mounts it
    # (read-only); "" (the default) and box opening sends the planner no world. `planner_worlds_dir` is the
    # same folder as this node writes it; "" = planner_worlds under state_dir
    "planner_worlds_as": "",
    "planner_worlds_dir": "",
    # the line-up before a run, on the UI (line_up, guide): the panel tells the user which way to move until
    # the container is where the arm opens it, then asks, and the arm moves only on its Open. Off (the
    # default) open_box starts its run at once, as it always has; so does an execution with no UI (standalone).
    # With wrist_first the wrist turns to look while the user lines up (the run itself, --line-up: gently,
    # and only where the arm stands at rest). Without it the SCENE camera alone follows the container, which
    # on flatbot does not see a lid whole in front of the arm: one it cannot see is never asked of
    "line_up": False,
    # how far from the arm's base (m) the line-up trusts the boundary: a container further out is told to come
    # closer, though the run's proof plans a press for it there. 0 (the default): as far as the proof plans.
    # A deployment's own: on flatbot the proof plans presses to about 0.90 m, and the arm has opened boxes
    # at 0.63 and 0.68 m (2026-11-12)
    "line_up_far_m": 0.0,
    # the mock only: how long each of its phases lasts (s). 0, the default, and its run is over at once, as it
    # always was; a second or two and the UI can be rehearsed on it (the panel, its bar, Cancel) with no arm
    "mock_phase_s": 0.0,
}
NOT_ALLOWED = "allow_execute is off: open_box moves the arm only on a node set for an attended run"


def check_params(params):
    """`params` against open_box's schema (REVIEW_RULES O4: the base class does not enforce
    it): open_box takes none, so anything but an empty JSON object is EngineError("bad_params")
    (the dry run's `execute` included: it is gone)."""
    if not isinstance(params, dict):
        raise EngineError("bad_params", f"params must be a JSON object, not {type(params).__name__}")
    if params:
        raise EngineError("bad_params", f"open_box takes no params, not {', '.join(sorted(map(str, params)))}")


# the command line each action runs: the package's executable, as `ros2 run box_opening <it>`
EXECUTABLES = {"open_box": "press_demo"}
POLL_S = 0.05
# a cancel is Ctrl+C's: one SIGINT, which cancels the motion in flight and waits for the
# driver to confirm. A second, after this long, is its escalation; a kill after that.
CANCEL_ESCALATE_S = 15.0
CANCEL_KILL_S = 10.0
SHUTDOWN_WAIT_S = 3.0  # on SIGTERM: the run's own stop first, inside docker stop's grace period
# REVIEW_RULES O3: the orchestrator cancels any action still running after 120 s (fixed in
# rammp-orchestrator; an engine cannot ask for more), releases the arm and hibernates the engine.
# A run still going at this budget is stopped its own way (one Ctrl+C) and fails with `timeout`,
# so the end is the run's: its stop's worst (a second Ctrl+C after CANCEL_ESCALATE_S, a kill
# CANCEL_KILL_S after that) and the last world-state writes (WORLD_WAIT_S) still end inside the
# 120 s. The longest of 48 logged missions took 45 s, plus ~5 s to start.
ORCHESTRATOR_TIMEOUT_S = 120.0
WORLD_WAIT_S = 3.0
RUN_BUDGET_S = ORCHESTRATOR_TIMEOUT_S - CANCEL_ESCALATE_S - CANCEL_KILL_S - WORLD_WAIT_S - 2.0  # 90 s
# The line-up's share of those 120 s (LINE_UP_S): what the run's budget leaves once the longest run logged
# has its time (45 s of 48 missions, plus ~5 s to start), so a run started on the line-up's last second
# still ends before its own stop. A user not in place by then is told to press open again
RUN_LONGEST_S = 50.0
LINE_UP_S = RUN_BUDGET_S - RUN_LONGEST_S  # 40 s
LINE_UP_END_S = 12.0  # a run told its line-up is over turns its wrist back and ends within this, or is stopped
NOT_LINED_UP = "the container was not where the arm opens it within %.0f s: nothing moved (press open again in place)"
TAIL_LINES = 40

# a leg's row -> (phase, the fraction of the run done it ends, what the run is doing then, for
# the user) (REVIEW_RULES U2: progress runs from 0 to 1, its message a person's words, not a leg's
# name); rows are written as their legs end. First match wins.
LEG_PHASES = (
    ("survey:", "locate", 0.03, "looking for the container with the wrist camera"),
    ("reach:look", "look", 0.1, "looking for the button"),
    ("press:close", "look", 0.12, "reaching for the button"),
    ("reach:", "press", 0.2, "reaching for the button"),
    ("depth:approach", "press", 0.25, "pressing the button"),
    ("depth:", "press", 0.35, "pressing the button"),
    ("grip:", "grip", 0.5, "gripping the lid"),
    ("lid:", "carry", 0.7, "lifting the lid off"),
    ("place:down", "carry", 0.8, "carrying the lid to the table"),
    ("place:touch", "carry", 0.85, "setting the lid down"),
    ("place:", "release", 0.9, "letting go of the lid"),
    ("home", "home", 1.0, "going home"),
)
# notes that say a run is past its start: it has looked for the container, or given up on it. A home row before
# any of them (and before any leg above but the last) is the run's FIRST motion, to its start pose where the arm
# does not rest there; from one of them on, a home row is the way home (Tracker.add)
PAST_START = ("scene_fix", "wrist_survey", "wrist_first", "counter_edges", "reach_plan", "reach_refused")
# the mock's run: the real run's phases, in its words. "grip" holds the lid, "release" lets it go
# on the table, as the real run's rows say (Tracker)
MOCK_PHASES = {
    "open_box": (
        ("locate", 0.05, "found the container"),
        ("look", 0.1, "looking for the button"),
        ("press", 0.35, "pressing the button"),
        ("grip", 0.5, "holding the lid"),
        ("carry", 0.85, "setting the lid down"),
        ("release", 0.9, "letting go of the lid"),
        ("home", 1.0, "going home"),
    ),
}
MOCK_APPROACH_PHASES = 6  # the mock's line-up: its box comes into reach over this many mock_phase_s
# where the mock says it set the lid down (arm_mount, m): a stand-in, the mock sees nothing.
# flatbot's 2026-10-30 13:28 run set it down here, beside the box
MOCK_LID_XYZ = [0.597, -0.071, 0.209]
VIDEO_LINE = re.compile(r"\] VIDEO: .*: (\S+)$")
# a line the run's own rclpy logger wrote: "[ERROR] [1793385477.635189323] [box_opening_run]: ..."
ROS_LINE = re.compile(r"^\[(DEBUG|INFO|WARN|ERROR|FATAL)\] \[[0-9.]+\] \[([^\]]*)\]: (.*)$")
LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": logging.CRITICAL,
}


def share(*parts):
    from ament_index_python.packages import get_package_share_directory

    return str(Path(get_package_share_directory("box_opening"), *parts))


def executable(name):
    """The installed executable `name` of this package (what `ros2 run box_opening <name>` runs)."""
    from ament_index_python.packages import get_package_prefix

    return os.path.join(get_package_prefix("box_opening"), "lib", "box_opening", name)


def config_path(name, *where):
    """A config file by its name in the image's config/<where>, or an absolute path as it is."""
    return name if os.path.isabs(name) else share("config", *where, name)


class Run:
    """One command line's run as a child process: its console lines logged (`log(level, text)`)
    and kept (the last TAIL_LINES), the video it announced, its exit code."""

    def __init__(self, argv, env, log, talk=False):
        self.lines = deque(maxlen=TAIL_LINES)
        self.video = None
        self.started = time.monotonic()
        self.timed_out = False  # stopped by the node at RUN_BUDGET_S
        self._log = log
        # `talk` (a run that lines up first, --line-up): its lines that begin with line_up.MARK are messages
        # for the line-up, kept apart from its log (messages()), and it is answered on its stdin (send())
        self._got, self._lock = [], threading.Lock()
        # its own session: a Ctrl+C at this node's terminal reaches it once, from here. Its
        # lines are decoded whatever they hold: a reader that died would leave the run
        # blocked on a full pipe, mid-motion.
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if talk else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self._reader = threading.Thread(target=self._read, name="box-opening-run", daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(line_up.MARK):  # for the line-up, not the log: five a second while it watches
                try:
                    msg = json.loads(line[len(line_up.MARK) :])
                except ValueError:
                    continue
                with self._lock:
                    self._got.append((time.monotonic(), msg))
                continue
            ros = ROS_LINE.match(line)
            level, text = (LEVELS[ros.group(1)], f"[{ros.group(2)}] {ros.group(3)}") if ros else (logging.INFO, line)
            self.lines.append(text)
            m = VIDEO_LINE.search(text)
            if m:
                self.video = m.group(1)
            self._log(level, text)

    def messages(self):
        """The line-up's lines it has said since the last call: (when read, message), oldest first."""
        with self._lock:
            got, self._got = self._got, []
        return got

    def send(self, msg):
        """A line on its stdin (a run started with `talk`): the line-up's go, or stop."""
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError, AttributeError):  # it has ended, or was never to be talked to
            pass

    def alive(self):
        return self.proc.poll() is None

    def interrupt(self):
        """Ctrl+C, to the run alone (not to the ffmpeg its video may be encoding with)."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)

    def stop(self, wait_s):
        """Its own stop, then a kill if it has not ended within `wait_s`."""
        self.interrupt()
        try:
            self.proc.wait(wait_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def finish(self):
        self._reader.join(2.0)
        return self.proc.returncode


class RunLog:
    """The rows of the run log a run creates in `runs` (the first run-*.jsonl that was not
    there before it started), as they are written."""

    def __init__(self, runs, before):
        self.runs = Path(runs)
        self.before = before
        self.path = None
        self._at = 0
        self._partial = ""

    @staticmethod
    def existing(runs):
        return set(Path(runs).glob("run-*.jsonl")) if Path(runs).is_dir() else set()

    def rows(self):
        if self.path is None:
            new = sorted(self.existing(self.runs) - self.before)
            if not new:
                return []
            self.path = new[0]
        try:
            with open(self.path) as f:
                f.seek(self._at)
                text = f.read()
                self._at = f.tell()
        except OSError:
            return []
        lines = (self._partial + text).split("\n")
        self._partial = lines.pop()  # a row not yet finished
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


class WorldWriter:
    """A run's world-state writes, made in order on a thread of their own (REVIEW_RULES R4): an
    orchestrator call waits up to 10 s for its service and 10 s for the answer, and the poll loop
    that follows the run (its budget, its cancel) must never wait on one. `put` queues a write,
    a callable that returns whether it was taken; `wait` joins what is queued, at most `timeout_s`."""

    def __init__(self, log):
        self._log = log
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._work, name="box-opening-world-state", daemon=True)
        self._thread.start()

    def put(self, what, write, done=None):
        """Queue `write` (said as `what` where it is refused); `done(ok)` once it is answered."""
        self._q.put((what, write, done))

    def _work(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            what, write, done = item
            try:
                ok = bool(write())
            except Exception as e:  # noqa: BLE001 -- Cancelled included: no write once cancelled
                self._log.error(f"the world state was not told {what}: {type(e).__name__}: {e}")
                ok = False
            else:
                if not ok:
                    self._log.error(f"the world state was not told {what}")
            if done is not None:
                done(ok)

    def wait(self, timeout_s):
        """Every write queued so far answered (True), or `timeout_s` gone (False: said)."""
        self._q.put(None)
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            self._log.error(f"the world-state writes did not finish in {timeout_s:.0f} s: some may be missing")
            return False
        return True


# The run's panel on the UI (Panel). The bar's scale is the UI's, 0 to 100 (docs/UI_JSON_SCHEMA.md, the
# base class's example); ctx.progress keeps REVIEW_RULES U2's fraction from 0 to 1.
PANEL_TITLE = ui_text.TITLE
PANEL_CANCEL = "cancel"  # the Stop button's id
PANEL_PCT = 100.0
# an unchanged panel is sent again this often: a goal the bridge ended on its own (its abort, a refusal) is opened
# again by the next send, and Cancel is back on the screen (the microwave engine's period)
PANEL_REFRESH_S = 1.0
PANEL_CANCELLED = "Cancelled: the arm is stopping where it is"
PANEL_TIMED_OUT = "Taking too long: the arm is stopping where it is"
# the final notice (Done, Stopped, failed) is waited on for its Done at most this long, and never past the
# orchestrator's 120 s less NOTICE_MARGIN_S from the goal's start: a notice must never become its timeout
NOTICE_S = 60.0
NOTICE_MARGIN_S = 10.0


class Panel:
    """open_box's panel on the UI for as long as it runs (REVIEW_RULES U3): what the run is doing (the phase and
    the message its progress reports), a bar, and ONE button, Stop. No Pause, and `pausable` false: a run is one
    motion and cannot pause. Once the run is being stopped (`stopping`: a cancel, its budget) it says so and has
    no button left: there is nothing more to ask of it.

    It asks the user nothing and waits for nothing: `tick()` (the follow loop's, every POLL_S) reads a press and
    sends the panel where it changed or PANEL_REFRESH_S has passed, both calls the base class answers at once.
    Its life is the run's: the base class closes it as execute() ends (at most RUN_BUDGET_S and the run's stop),
    with the token it was shown by. `live()`: whether this execution has the UI at all (Capability._ui_live);
    without it nothing is sent, and a run is as it was."""

    def __init__(self, ctx, live, clock=time.monotonic):
        self._ctx = ctx
        self._live = live
        self._clock = clock
        self._said = ("locate", 0.0, "starting")
        self._sent = None  # the panel as last sent, None where the last send did not go out
        self._tried_at = None  # when a send was last tried
        self.pressed = False  # Cancel was pressed on it (it stays said)
        self.stopping = ""  # the run is being stopped, in the user's words: no button is offered

    def say(self, phase, pct, message=""):
        """What the run does now: the progress it just reported."""
        self._said = (str(phase), float(pct), str(message))

    def panel(self):
        phase, pct, message = self._said
        text = self.stopping or (message[:1].upper() + message[1:])
        return {
            "title": PANEL_TITLE,
            "message": wrap(text),  # the UI does not wrap: a long line runs off its screen (capability_ui.wrap)
            "progress": {"phase": phase, "pct": round(PANEL_PCT * pct, 1), "message": text, "pausable": False},
            "buttons": [] if self.stopping else [{"button_id": PANEL_CANCEL, "label": "Stop"}],
        }

    def tick(self):
        """Read a press, and send the panel where it is due: at once where it changed (or a press took the
        bridge's panel down), every PANEL_REFRESH_S where it did not, and no oftener than that where the last
        send did not go out. True the once Cancel is pressed: the panel says the arm is stopping from then on."""
        event = self._ctx.poll_ui_event()
        pressed = False
        if event is not None and event.get("type") == "button":
            # any press ends the bridge's goal, and its panel with it: the next send opens it again
            self._sent = self._tried_at = None
            pressed = event.get("button_id") == PANEL_CANCEL and not self.pressed and not self.stopping
            if pressed:
                self.pressed, self.stopping = True, PANEL_CANCELLED
        panel, now = self.panel(), self._clock()
        due = self._tried_at is None or now - self._tried_at >= PANEL_REFRESH_S
        if due or (self._sent is not None and panel != self._sent):
            self._tried_at = now
            self._sent = panel if self._live() and self._ctx.update_user_panel(panel_json=panel) else None
        return pressed


class Tracker:
    """A run's rows as it goes: the progress each says, and what the result reports.
    `instance_id`: the container the orchestrator selected when the goal came, kept for the
    whole run (a hibernate clears the engine's own mid-run; REVIEW_RULES O3)."""

    def __init__(self, action_id, instance_id=""):
        self.action_id = action_id
        self.instance_id = instance_id
        self.pct = 0.0
        self.looking = False  # said once, as the run's log appears: it is up and looking for the container
        self.past_start = False  # a row past the run's start was read: a home row from here on is the way home
        self.presses = 0
        self.grips = 0
        self.lid_spot = None
        self.set_down = False
        self.released = False
        self.held = False  # the lid in the fingers: from the grip that held it to its release or put-back
        self.held_said = False  # ... as the world state was last told (Capability._sync_held)
        self.opened = None  # the object record's `opened`: None not asked, then the orchestrator's answer

    def add(self, row):
        """The (phase, %, message) this row says, or None."""
        leg, kind = str(row.get("leg", "")), row.get("kind")
        if kind == "note":
            self.past_start = self.past_start or leg in PAST_START
            return self._note(leg, row)
        ok = bool(row.get("ok"))
        if leg == "depth:press" and ok:
            self.presses += 1
        if leg == "grip:close":
            self.grips += 1
        if leg == "place:open" and ok:
            self.released = True
        if leg in ("place:open", "lid:release") and ok:
            self.held = False
        for start, phase, pct, doing in LEG_PHASES:
            if leg.startswith(start):
                if phase == "home" and not self.past_start:
                    # the run's first motion, to its start pose (press_demo.home_first): not the way home, and
                    # no part of the run done (flatbot 2026-11-11 18:06 began so, and reported 100 % at once)
                    phase, pct, doing = "locate", self.pct, "going to the start pose"
                else:
                    self.past_start = True
                self.pct = max(self.pct, pct)
                return phase, self.pct, doing if ok else f"{doing}: stopped"
        return None

    def up(self):
        """("locate", %, message) the once, as the run's log appears: the run is up (its start-up and readiness
        take a second or two) and looks for the container, which with the scene camera can take 15 s in which
        no row is written."""
        if self.looking:
            return None
        self.looking = True
        return "locate", self.pct, "looking for the container"

    def _note(self, leg, row):
        if leg == "scene_fix" and row.get("origin_xyz"):
            x, y = row["origin_xyz"][:2]
            self.pct = max(self.pct, 0.05)
            return "locate", self.pct, f"found the container at [{x:.2f}, {y:.2f}]"
        if leg == "lid_spot" and row.get("xyz"):
            self.lid_spot = [round(float(v), 3) for v in row["xyz"]]
            x, y = self.lid_spot[:2]
            return "grip", self.pct, f"the lid will go to [{x:.2f}, {y:.2f}] on the table"
        if leg == "reach_setdown":
            self.set_down = True
        if leg == "lid_put_back" and row.get("ok"):
            self.held = False
            return "release", self.pct, "put the lid back on the container"
        if leg == "reach_grip_result":
            self.held = self.held or bool(row.get("ok"))
            return "grip", self.pct, "holding the lid" if row.get("ok") else "the grip missed the lid"
        if leg == "reach_refused":
            return "locate", self.pct, "stopping: the run refused to go on"
        return None

    def lid_down(self):
        """The lid set down on the table and let go (the container is open, whatever follows)."""
        return self.set_down and self.released

    def lid_off(self, code):
        return self.action_id == "open_box" and code == 0 and self.lid_down()


# The run's last STOP line -> (error code, recoverable), first match wins (REVIEW_RULES U4).
# Recoverable only where the run says nothing moved, or that the arm got home with nothing
# held: a retry may then succeed. A STOP no rule names, and an exit with none (a traceback),
# is `failed`, not recoverable: nothing says where the arm is.
STOP_CODES = (
    (re.compile(r"not starting"), "not_ready", True),
    (re.compile(r"NO BOX"), "not_found", True),
    # "— the arm holds", "it holds", "The arm holds over where it was": a person clears it
    (re.compile(r"\bholds\b|did not get home"), "arm_holds", False),
    # the press's contact off the band around the cameras' top, none, or its watch failed
    # (tasks/press_contact): stopped there, back up and home
    (re.compile(r"something else is in the way"), "contact_early", True),
    (re.compile(r"the box is not where it was seen"), "contact_late", True),
    (re.compile(r"contact watch failed"), "watch_failed", False),
    (re.compile(r"MOVED the box"), "box_moved", True),
    (re.compile(r"slipped out of the fingers"), "lid_dropped", False),
    (re.compile(r"the lid is back on the box"), "lid_put_back", True),
    (re.compile(r"did not pop|the knob is not up|pressed nothing"), "not_popped", True),
    (re.compile(r"no button"), "no_button", True),
    (re.compile(r"nothing was pressed|nothing moved"), "no_press", True),
)


def error_for(action_id, code, lines):
    """The EngineError a run that ended with `code` reports, from what it said last."""
    said = [ln for ln in lines if "STOP" in ln or "not starting" in ln or "FAIL" in ln or "NO BOX" in ln]
    why = (said or list(lines) or [f"exit {code}"])[-1]
    if code == EXIT_HALTED:
        # the e-stop, or control lost: the run sent nothing after it (REVIEW_RULES C5)
        return EngineError("arm_halted", why, recoverable=False)
    if code == 2:
        return EngineError("not_found", why, recoverable=True)
    if code == 3:
        return EngineError("not_home", why, recoverable=True)
    if code == EXIT_NOT_LINED_UP:  # its line-up ended with no go (the engine's own stop is said by the engine)
        return EngineError("not_lined_up", why, recoverable=True)
    if code is not None and code < 0:  # killed: whatever it said last, it did not end its own way
        return EngineError("stopped", f"the run was ended by signal {-code}: {why}", recoverable=False)
    if said:
        for pattern, error_code, recoverable in STOP_CODES:
            if pattern.search(why):
                if error_code == "arm_holds":
                    why = f"{why} -- the arm holds where it stopped: clear it, attended, before the next run"
                return EngineError(error_code, why, recoverable=recoverable)
    return EngineError("failed", why, recoverable=False)


class Capability(CapabilityEngine):
    display_name = "Open the container"
    # "food container" is the class the scene segmenter publishes. "food_container" is the same class without
    # its space, for a selection typed by hand: the orchestrator's interactive_perception_node (the UI's object
    # list on snorlax) reads `add <instance_id> <perception_class>` split on spaces, so the first cannot be typed
    perception_classes = ["food container", "food_container"]
    required_resources = ["arm", "gripper"]
    actions = [
        Action(
            "open_box",
            "Open the container",
            params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("box_opening", **node_kwargs)
        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        self._run = None  # the Run under way
        self._cancelled_token = None  # the execution a CANCEL came for before its goal did (_on_control)
        self._ui_stopped = False  # this execution was stopped by the panel's own Stop (_cancel_from_ui)
        self._ui_held = False  # this execution's run ended with the lid in the fingers (for its Stopped notice)
        self._mock_pubs = self._mock_interface() if self.mock else []
        use_node_logger(self)
        self.context.on_shutdown(self._on_shutdown)
        self.get_logger().info(
            "allow_execute is on: open_box moves the arm"
            if self._allowed()
            else NOT_ALLOWED + " (open_box unavailable)"
        )

    def _mock_interface(self):
        """The mock's share of the real alternative's interface: a publisher, idle, on every topic
        the real run publishes (rammp-alternative.yaml `publishes`), with the run's own types and
        QoS — the gripper commands its run sends the driver, and with viz what its perception
        sees and decides. The real node holds none of them: its run does, and press_demo
        refuses to move an arm another program on the graph publishes gripper commands to. Nothing
        is ever sent on them (scripts/smoke.py checks that every declared topic appears)."""
        from rammp_arm_interfaces.msg import GripperSetpoint
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray

        from box_opening.constants import GRIPPER_SETPOINT_TOPIC
        from box_opening.perception import viz

        best_effort = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        return [
            self.create_publisher(GripperSetpoint, GRIPPER_SETPOINT_TOPIC, best_effort),
            self.create_publisher(CompressedImage, viz.SCENE_TOPIC, 1),
            self.create_publisher(CompressedImage, viz.WRIST_TOPIC, 1),
            self.create_publisher(MarkerArray, viz.MARKER_TOPIC, 1),
            self.create_publisher(String, viz.STATUS_TOPIC, 1),
        ]

    def _param(self, name):
        """A PARAMS value as it is now: each goal reads its own (a deployment, or the sheppy TUI,
        may set one while the node runs), never a copy taken at start-up."""
        return self.get_parameter(name).value

    def _allowed(self):
        """allow_execute as it is now (a deployment may set it while the node runs)."""
        return bool(self._param("allow_execute"))

    def _ui_live(self):
        """Whether this execution may show its panel: the orchestrator pushed its token for it (standalone: never)
        and the bridge's request_user_input is there. Asked before every send, of the base class's own state,
        which has no call for it: as of feat/engine_update 8e7f0ab update_user_panel() logs a warning for every
        send without a token, and a run whose first send finds the bridge's server not there never opens its
        panel again (its ui_opening stays set)."""
        return self._assigned_token != bytes(16) and self._ui_request_client.server_is_ready()

    def _cancel_from_ui(self):
        """The panel's Cancel is the orchestrator's: the base class's own Control.CANCEL, so the run ends the
        one way a cancel ends it (its Ctrl+C, the result cancelled, no world-state write after it)."""
        self.get_logger().warning("Stop pressed on the UI: the action is cancelled")
        self._ui_stopped = True  # the user's own Stop: the action ends on a Stopped notice (execute)
        CapabilityEngine._on_control(self, Control.Request(command=Control.Request.CANCEL), Control.Response())

    def _on_control(self, request, response):
        """The base class's, and one case it loses: a CANCEL that comes after the orchestrator has pushed its
        token for an execution and before that execution's goal (it pushes the token to the engine first, then
        to the bridge and the drivers, waiting 2 s for each that is not there, then activates and sends the
        goal). The base class has no run to cancel then, answers "ok", and the action ran to its end. It is
        kept for the execution that token is for, which ends cancelled before anything runs (execute)."""
        with self._lifecycle:
            idle = self._current is None
        response = super()._on_control(request, response)
        token = self._assigned_token
        if idle and request.command == Control.Request.CANCEL and token != bytes(16):
            self._cancelled_token = token
            response.message = "ok (the action about to start is cancelled)"
            self.get_logger().warning("cancelled before its goal came: the action about to start will not run")
        return response

    def _cancelled_early(self):
        """Whether a CANCEL came for THIS execution before its goal did (_on_control); asked once, as it starts."""
        token, self._cancelled_token = self._cancelled_token, None
        return token is not None and token == self._assigned_token

    def _far_m(self):
        """line_up_far_m as it is now, None where it sets no bound (0 or less)."""
        far = float(self._param("line_up_far_m") or 0.0)
        return far if far > 0.0 else None

    def _lines_up(self):
        """Whether this execution lines up first: the param is on and there is a UI to ask on."""
        return bool(self._param("line_up")) and self._ui_live()

    def _lined_up_run(self, ctx, env, action_id, instance_id, view):
        """The run started with --line-up and taken through its line-up: the Run, told to go on its Open.
        Until then it only watches (the wrist turned to look, gently, where the arm stands at rest). A
        line-up that ends any other way ends the run before this raises: a cancel is its Ctrl+C, as every
        cancel is (the arm stops where it is); out of time it is told so, turns its wrist back to rest and
        ends (LINE_UP_END_S, then its stop). A run that ended of itself while lining up (not ready, its
        wrist's turn stopped) raises what it said."""
        run_ = Run(self._argv(action_id, instance_id) + ["--line-up"], env, view.relay(LOGGER.log), talk=True)
        self._run = run_  # a shutdown stops it (_on_shutdown)
        try:
            self._line_up(ctx, lambda: self._box_source(env, instance_id, watch=run_))
        except BaseException as e:
            self._run = None
            cancelled = isinstance(e, Cancelled)
            if cancelled:
                self.get_logger().warning("cancelled while lining up: the run's own stop (Ctrl+C) sent")
                run_.stop(CANCEL_ESCALATE_S)
            else:
                run_.send({"stop": True})
                try:
                    run_.proc.wait(LINE_UP_END_S)
                except subprocess.TimeoutExpired:
                    self.get_logger().error(f"the run did not end its line-up in {LINE_UP_END_S:.0f} s: its stop")
                    run_.stop(CANCEL_ESCALATE_S)
            code = run_.finish()
            view.finish(code, cancelled=cancelled, timed_out=False, held=False)
            if isinstance(e, EngineError) and e.error_code == "line_up_unavailable" and code != EXIT_NOT_LINED_UP:
                raise error_for(action_id, code, run_.lines) from e  # it ended of itself: what it said
            raise
        run_.send({"go": True})
        self.get_logger().info("line-up: Open pressed: the run goes on")
        return run_

    def _box_source(self, env, instance_id, watch=None):
        """A real run's line-up source (line_up.BoxSource): a watch follows the container and reach_probe judges
        where it stands, for as long as the panel is up. `watch`: the run itself, started with --line-up (its
        wrist looks; _lined_up_run); None, and box_watch is run for it, which reads the scene camera alone and
        holds no client for the arm. reach_probe never can move it."""
        container = config_path(self._param("container"), "containers")
        if watch is None:
            watch = [executable("box_watch"), "--container", container]
            watch += ["--bench-world", config_path(self._param("world"))]
            if instance_id:
                watch += ["--instance-id", instance_id]
        layers = self._param("body_layers")
        layers = PARAMS["body_layers"] if layers is None else str(layers)

        def probe(urdf, root):
            argv = [executable("reach_probe"), "--serve", "--urdf", urdf, "--container", container]
            return argv + ["--root", root or "arm_mount", "--body-layers", layers]

        return line_up.BoxSource(
            watch, probe, env, far_m=self._far_m(), log=lambda text: LOGGER.log(logging.INFO, f"line-up: {text}")
        )

    def _line_up(self, ctx, make_source):
        """The line-up before a run (the `line_up` param; line_up.LineUp): the panel tells the user which way
        to move and asks, and this returns only on its Open, with the seconds it took. Cancel on it is the
        orchestrator's cancel (Cancelled); out of time, EngineError("not_lined_up"), recoverable: nothing
        moved. A source that cannot start is EngineError("line_up_unavailable"): with the line-up asked for,
        no run starts without its Open. 0.0 at once, and the run starts as it always has, where the param is
        off or this execution has no UI to ask on (standalone). `make_source()`: what sees the box and judges
        its place (BoxSource; the mock's MockApproach), made only where it is used, and closed before this
        returns: no run starts while it lives."""
        if not self._param("line_up"):
            return 0.0
        if not self._ui_live():
            self.get_logger().info("line_up: no UI for this execution, so no one to ask: the run starts as always")
            return 0.0
        t0 = time.monotonic()

        def said(say):
            ctx.progress(line_up.PHASE, 0.0, say.message, pausable=False)
            self.get_logger().info(f"line-up: {say.message}")

        try:
            source = make_source()
        except OSError as e:
            raise EngineError(
                "line_up_unavailable", f"the line-up could not start ({e}): nothing moved", recoverable=False
            ) from e
        try:
            lining = line_up.LineUp(ctx, self._ui_live, source, LINE_UP_S, on_say=said)
            outcome = lining.run()
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
        if outcome == line_up.CANCELLED:
            if lining.cancel_pressed:
                self._cancel_from_ui()
            raise Cancelled()
        if outcome == line_up.GONE:
            raise EngineError(
                "line_up_unavailable",
                "what watched the container ended while lining up: nothing ran",
                recoverable=False,
            )
        if outcome == line_up.TIMED_OUT:
            why = getattr(source, "why_none", "")
            raise EngineError(
                "not_lined_up", NOT_LINED_UP % LINE_UP_S + (f" -- {why}" if why else ""), recoverable=True
            )
        return time.monotonic() - t0

    def availability(self, instance_id, world_state):
        """open_box for a container the world state does not already hold opened (its object
        record's `opened`, which a run writes as it lets the lid go on the table), while the
        gripper holds nothing: a run that stopped with the lid in the fingers says so in
        `gripper.holding`, and a run started from there would start holding it."""
        opened = ((world_state.get("objects") or {}).get(instance_id) or {}).get("opened") is True
        holding = (world_state.get("gripper") or {}).get("holding")
        if holding:
            why = f"the gripper holds {holding}: empty it, attended, before the next run"
        elif opened:
            why = "the container is already opened"
        else:  # REVIEW_RULES O1: offered only where it will run. The mock moves nothing: always offered
            why = "" if self.mock or self._allowed() else NOT_ALLOWED
        return [ActionAvailability(action_id=a.action_id, available=not why, reason=why) for a in self.actions]

    def execute(self, action_id, params, ctx):
        """open_box (_open_box), and how it ended on the UI (capability_ui.RunUI, the shared panels): a Done
        notice, a failed notice in plain words (ui_text) saying where the arm and the lid are, or after the
        panel's own Stop a Stopped notice. Each waits for its Done, bounded (_notice). An orchestrator cancel
        shows nothing more: the orchestrator has taken the UI back. Standalone (no UI) nothing is sent."""
        t0 = time.monotonic()
        self._ui_stopped = self._ui_held = False
        ui = RunUI(ctx, PANEL_TITLE, cancel=lambda: cancel_own_run(self), logger=self.get_logger())
        try:
            result = self._open_box(action_id, params, ctx)
        except Cancelled:
            if self._ui_stopped:
                ui.user_stopped = True  # the notice stays up through the cancel it made, until Done
                self._notice(ui, t0, ui_text.stopped(self._ui_held), f"{PANEL_TITLE}: stopped", "warning")
            raise
        except EngineError as error:
            if not ctx.cancelled():
                if error.error_code == "not_allowed":
                    self._notice(ui, t0, ui_text.NOT_ALLOWED, f"{PANEL_TITLE}: not started", "warning")
                else:
                    words = ui_text.failure(error.error_code, str(error))
                    self._notice(ui, t0, words, f"{PANEL_TITLE}: failed", "error")
            raise
        except Exception as error:
            if not ctx.cancelled():
                self._notice(ui, t0, ui_text.unexpected(error), f"{PANEL_TITLE}: failed", "error")
            raise
        done = ui_text.done(result)
        self._notice(ui, t0, f"Mock: {done} No arm motion." if result.get("mock") else done)
        return result

    def _notice(self, ui, t0, message, title=None, style=None):
        """A final notice with Done, where this execution has the UI: waited on for at most NOTICE_S, and never
        past the orchestrator's 120 s (less NOTICE_MARGIN_S) from `t0`, the goal's start. Never raises."""
        try:
            if not self._ui_live():
                return
            wait_s = max(0.0, min(NOTICE_S, ORCHESTRATOR_TIMEOUT_S - NOTICE_MARGIN_S - (time.monotonic() - t0)))
            ui.notice(message, title=title, timeout_s=wait_s, style=style)
        except Exception as e:  # noqa: BLE001 -- the action's outcome is decided: a notice must not change it
            self.get_logger().warning(f"UI: the final notice was not shown: {type(e).__name__}: {e}")

    def _open_box(self, action_id, params, ctx):
        if self._cancelled_early():
            raise Cancelled()  # nothing ran
        check_params(params)
        if self.mock:
            return self._mock(action_id, ctx)
        if not self._allowed():
            raise EngineError("not_allowed", NOT_ALLOWED + ": nothing ran", recoverable=False)
        ctx.checkpoint()  # cancelled before it started: nothing runs
        env = self._env()  # this goal's params, read once (REVIEW_RULES: none is a start-up copy)
        # the selection as the goal came: read once, used for the line-up, every write and the run's argv
        instance_id = self.instance_id
        runs = state_dir(env) / "runs"
        log = RunLog(runs, RunLog.existing(runs))
        track = Tracker(action_id, instance_id)
        t0 = time.monotonic()
        # the flight view (flight_view): the run's lines read as they are logged, one short line a step
        # on this node's stdout. It only reads, and nothing it meets stops the run or its reader
        view = FlightView()
        lined_up_s, run_ = 0.0, None
        if self._lines_up() and self._param("wrist_first"):
            # the wrist looks while the user lines up (owner 2026-11-16: "keep the wrist look"): the run's own
            # process is the line-up's eyes, started now and held at its line-up until the Open, so that one
            # process alone ever commands the arm. Its budget runs from here, the line-up inside it
            view.start(action_id, instance_id)
            run_ = self._lined_up_run(ctx, env, action_id, instance_id, view)
        else:
            # nothing moves, and nothing runs but on the line-up's Open
            lined_up_s = self._line_up(ctx, lambda: self._box_source(env, instance_id))
        first = MOCK_PHASES[action_id][0][0]  # the first phase's name
        ctx.progress(first, 0.0, "starting", pausable=False)
        panel = Panel(ctx, self._ui_live)  # on the UI from now until the action ends
        panel.say(first, 0.0, "starting")
        panel.tick()
        world = WorldWriter(self.get_logger())
        if run_ is None:
            view.start(action_id, track.instance_id)
            run_ = Run(self._argv(action_id, track.instance_id), env, view.relay(LOGGER.log))
        self._run = run_
        code = None
        try:
            code = self._follow(self._run, log, track, ctx, world, panel, RUN_BUDGET_S - lined_up_s)
        finally:
            run_, self._run = self._run, None
            if run_.proc.poll() is None:
                # _follow raised with the run still going: nothing else would stop it, and it
                # would go on moving the arm after its goal ended. Its own stop first (Ctrl+C:
                # the motion in flight cancelled), a kill if it has not ended in time.
                self.get_logger().error(
                    f"the run outlived its goal: its own stop (Ctrl+C), a kill after {CANCEL_ESCALATE_S:.0f} s"
                )
                run_.stop(CANCEL_ESCALATE_S)
            self._progress(log, track, ctx, panel)  # the last rows, written as it ended
            if not (ctx.cancelled() or panel.pressed):
                self._sync_world(track, ctx, world)  # the last rows' word: a release, a put-back
            view.finish(
                code, cancelled=ctx.cancelled() or panel.pressed, timed_out=run_.timed_out, held=bool(track.held)
            )
            self._ui_held = bool(track.held)  # for a Stopped notice (execute)
        if ctx.cancelled() or panel.pressed:
            world.wait(WORLD_WAIT_S)  # those queued before the cancel
            raise Cancelled()
        result = {
            "executed": True,
            "lid_off": track.lid_off(code),
            "lid_xyz": track.lid_spot if track.lid_off(code) else None,
            "presses": track.presses,
            "grips": track.grips,
            "duration_s": round(time.monotonic() - t0, 1),
            "video": run_.video,
            "run_log": str(log.path) if log.path else None,
        }
        error = None
        if run_.timed_out:
            error = EngineError(
                "timeout",
                f"the run took longer than {RUN_BUDGET_S - lined_up_s:.0f} s and was stopped its own way (Ctrl+C)",
                recoverable=True,
            )
        elif code != 0:
            error = error_for(action_id, code, run_.lines)
        if error is not None:
            if track.held:
                # the next run would start holding it (availability): nothing to retry until it is emptied
                error = EngineError(error.error_code, f"{error} -- the lid is still in the fingers", recoverable=False)
            if not error.recoverable and self.orchestrated:  # standalone: no world state to tell
                self._report_arm_error(error, ctx, world)
            world.wait(WORLD_WAIT_S)
            raise error
        world.wait(WORLD_WAIT_S)  # every write answered before the result goes out
        if result["lid_off"] and track.instance_id:
            # `opened` is the object record's own field (the orchestrator refuses any other); where
            # the lid went is the result's lid_xyz. A refused write leaves open_box offered on an
            # open container: said, in the log and in the result (REVIEW_RULES O6)
            result["world_state"] = "opened" if track.opened else "not updated"
            if result["world_state"] != "opened":
                self.get_logger().error(
                    f"the lid is off, but the world state was not updated: {track.instance_id} is not marked opened"
                )
        return result

    def _follow(self, run_, log, track, ctx, world, panel, budget_s=RUN_BUDGET_S):
        """Progress from the run's rows until it ends, on `panel` too; a cancel (the orchestrator's, or the
        panel's Cancel, which is made the orchestrator's) is its Ctrl+C. Its exit code. The world-state writes
        the rows call for are `world`'s, never this loop's to wait on. `budget_s`: the run's own time, what a
        line-up before it left of RUN_BUDGET_S."""
        sent = escalated = None
        while run_.proc.poll() is None:
            self._progress(log, track, ctx, panel)
            if panel.tick():
                self._cancel_from_ui()
            cancelled = ctx.cancelled() or panel.pressed  # the press itself, whatever the base class made of it
            if not cancelled:
                self._sync_world(track, ctx, world)
            now = time.monotonic()
            if cancelled and sent is None:
                self.get_logger().warning("cancelled: the run's own stop (Ctrl+C) sent")
                panel.stopping = PANEL_CANCELLED
                run_.interrupt()
                sent = now
            elif sent is None and now - run_.started > budget_s:
                self.get_logger().error(
                    f"the run is still going after {budget_s:.0f} s: its own stop (Ctrl+C), before the "
                    "orchestrator's 120 s cancel"
                )
                run_.timed_out = True
                panel.stopping = PANEL_TIMED_OUT
                run_.interrupt()
                sent = now
            elif sent is not None and escalated is None and now - sent > CANCEL_ESCALATE_S:
                self.get_logger().error(f"the run did not stop in {CANCEL_ESCALATE_S:.0f} s: a second Ctrl+C")
                run_.interrupt()
                escalated = now
            elif escalated is not None and now - escalated > CANCEL_KILL_S:
                self.get_logger().error("the run did not stop: killed (a motion may still be in flight)")
                run_.proc.kill()
            time.sleep(POLL_S)
        return run_.finish()

    @staticmethod
    def _sync_held(track, ctx, world):
        """The world state's gripper as the run's rows say it (REVIEW_RULES C7): holding the
        selected container's lid from the grip that held it to its release or put-back. A cancel,
        or a run stopped between the two, then leaves it said: the base class takes no world-state
        write once an action is cancelled, so it is written as it happens, never after. Queued on
        `world` (WorldWriter), in order."""
        if not track.instance_id or track.held == track.held_said:
            return
        track.held_said = track.held
        holding = f"{track.instance_id}:lid" if track.held else None
        world.put(f"the gripper holds {holding}", lambda: ctx.write_world_state(gripper={"holding": holding}))

    @staticmethod
    def _report_arm_error(error, ctx, world):
        """A run that ended where a person must clear it (an e-stop, an arm held part way, the lid
        still in the fingers) says so in the world state's system_status, which the orchestrator
        honours: no object or action flow until the arm's domain is normal again."""
        status = {"arm": {"state": "error", "error_code": error.error_code, "message": str(error)}}
        world.put(f"the arm is in error ({error.error_code})", lambda: ctx.write_world_state(system_status=status))

    def _sync_world(self, track, ctx, world):
        """What the run's rows so far tell the world state: the gripper's `holding`, and the
        container `opened` once the lid is down on the table and let go."""
        self._sync_held(track, ctx, world)
        if track.lid_down():
            self._mark_opened(track, ctx, world)

    @staticmethod
    def _mark_opened(track, ctx, world):
        """The container's object record `opened`, once, as the lid is let go on the table
        (REVIEW_RULES C6): a run that fails or is cancelled after that, on its way home, leaves
        the container open, and open_box must not be offered on it again. track.opened is the
        orchestrator's answer."""
        if not track.instance_id or track.opened is not None:
            return
        track.opened = False  # asked; True once the orchestrator takes it
        world.put(
            f"{track.instance_id} is opened",
            lambda: ctx.update_object(track.instance_id, opened=True),
            lambda ok: setattr(track, "opened", ok),
        )

    @staticmethod
    def _progress(log, track, ctx, panel):
        """The progress each new row of the run log says (a run cannot pause once started), to the
        orchestrator and on the run's panel."""
        rows = log.rows()  # finds the run's log once it is there
        said = [track.up()] if log.path is not None else []
        said += [track.add(row) for row in rows]
        for progress in said:
            if progress is not None:
                ctx.progress(*progress, pausable=False)
                panel.say(*progress)

    def _argv(self, action_id, instance_id=""):
        argv = [executable(EXECUTABLES[action_id])]
        argv += ["--container", config_path(self._param("container"), "containers")]
        argv += ["--bench-world", config_path(self._param("world"))]
        argv += ["--speed-scale", str(float(self._param("speed_scale")))]
        layers = self._param("body_layers")  # only an explicit "" turns the check off; unset is the default
        argv += ["--body-layers", PARAMS["body_layers"] if layers is None else str(layers)]
        if instance_id:  # the container the orchestrator selected; standalone: any of its class
            argv += ["--instance-id", instance_id]
        if self._param("wrist_survey"):  # the run surveys only without a selection (tasks/survey)
            argv += ["--wrist-survey"]
        if self._param("wrist_first"):
            argv += ["--wrist-first"]
        if self._param("wrist_alone"):  # read by the wrist look alone (--wrist-first), and only with a selection
            argv += ["--wrist-alone"]
        return argv + ["--execute"]  # only ever run where allow_execute is on (execute())

    def _env(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env["BOX_OPENING_VIDEO"] = "1" if self._param("video") else "0"
        env["BOX_OPENING_VIZ"] = "1" if self._param("viz") else "0"
        env["BOX_OPENING_MISSION_FRAMES"] = "1" if self._param("mission_frames") else "0"
        if self._param("state_dir"):
            env["BOX_OPENING_STATE"] = self._param("state_dir")
        env[AIR_CONTROL_ENV] = str(self._param("air_control") or "impedance")
        for name, var in (("planner_worlds_as", PLANNER_WORLDS_AS_ENV), ("planner_worlds_dir", PLANNER_WORLDS_DIR_ENV)):
            if self._param(name):  # unset: whatever the node's own environment says, as BOX_OPENING_BODY_LAYERS
                env[var] = str(self._param(name))
        return env

    def _mock(self, action_id, ctx):
        """A successful real run at the interface, nothing moved (review round 2): its phases in
        its words (each `mock_phase_s` long), the gripper's `holding` from the grip to the release and
        the container `opened` at the release, written as the real run writes them, its panel on the UI
        and its result. Like the real run it cannot pause; a cancel ends it."""
        track = Tracker(action_id, self.instance_id)
        phase_s = max(0.0, float(self._param("mock_phase_s")))
        t0 = time.monotonic()
        # the real run's line-up, of a box that comes into reach over a few of the mock's phases
        self._line_up(ctx, lambda: line_up.MockApproach(MOCK_APPROACH_PHASES * phase_s, far_m=self._far_m()))
        world = WorldWriter(self.get_logger())
        panel = Panel(ctx, self._ui_live)  # the real run's, Cancel included

        def cancel_point():
            """A cancel (the orchestrator's, or the panel's Cancel) ends the mock here, as the real run."""
            if panel.tick():
                self._cancel_from_ui()
            if ctx.cancelled() or panel.pressed:
                world.wait(WORLD_WAIT_S)
                raise Cancelled()

        for phase, pct, doing in MOCK_PHASES[action_id]:
            cancel_point()
            ctx.progress(phase, pct, doing, pausable=False)
            panel.say(phase, pct, doing)
            if phase == "grip":
                track.held = True
            elif phase == "release":
                track.held, track.set_down, track.released = False, True, True
                track.lid_spot = list(MOCK_LID_XYZ)
            self._sync_world(track, ctx, world)
            until = time.monotonic() + phase_s  # the phase's length, a cancel read all through it
            while time.monotonic() < until:
                cancel_point()
                time.sleep(POLL_S)
        world.wait(WORLD_WAIT_S)
        result = {
            "executed": True,
            "mock": True,
            "lid_off": True,
            "lid_xyz": track.lid_spot,
            "presses": 1,
            "grips": 1,
            "duration_s": round(time.monotonic() - t0, 1),
            "video": None,
            "run_log": None,
        }
        if track.instance_id:
            result["world_state"] = "opened" if track.opened else "not updated"
        return result

    def _on_shutdown(self):
        """SIGTERM (docker stop) or Ctrl+C: a run under way stops its own way first."""
        run_ = self._run
        if run_ is not None:
            run_.stop(SHUTDOWN_WAIT_S)


class NodeLogHandler(logging.Handler):
    """Every box_opening log record as a line of the node's log, at its own level."""

    def __init__(self, node):
        super().__init__()
        self.node = node

    def emit(self, record):
        logger = self.node.get_logger()
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            logger.error(msg)
        elif record.levelno >= logging.WARNING:
            logger.warning(msg)
        else:
            logger.info(msg)


def use_node_logger(node):
    """The box_opening logger writes to `node`'s log instead of the console."""
    LOGGER.handlers.clear()
    LOGGER.addHandler(NodeLogHandler(node))


def main():
    run(Capability)


if __name__ == "__main__":
    main()
