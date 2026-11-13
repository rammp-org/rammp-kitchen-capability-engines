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
from rammp_orchestrator_interfaces.msg import ActionAvailability

from . import LOGGER
from .constants import state_dir
from .flight_view import FlightView
from .runtime.abort import EXIT_HALTED
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
    # the arm presses, tasks/survey). Never with the orchestrator's selection
    "wrist_first": False,
    # the folder box opening hands the planner worlds from (worlds.PlannerTable): the run's measured table,
    # while a way HOME is planned. `planner_worlds_as` is that folder as the PLANNER's container mounts it
    # (read-only); "" (the default) and box opening sends the planner no world. `planner_worlds_dir` is the
    # same folder as this node writes it; "" = planner_worlds under state_dir
    "planner_worlds_as": "",
    "planner_worlds_dir": "",
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

    def __init__(self, argv, env, log):
        self.lines = deque(maxlen=TAIL_LINES)
        self.video = None
        self.started = time.monotonic()
        self.timed_out = False  # stopped by the node at RUN_BUDGET_S
        self._log = log
        # its own session: a Ctrl+C at this node's terminal reaches it once, from here. Its
        # lines are decoded whatever they hold: a reader that died would leave the run
        # blocked on a full pipe, mid-motion.
        self.proc = subprocess.Popen(
            argv,
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
            ros = ROS_LINE.match(line)
            level, text = (LEVELS[ros.group(1)], f"[{ros.group(2)}] {ros.group(3)}") if ros else (logging.INFO, line)
            self.lines.append(text)
            m = VIDEO_LINE.search(text)
            if m:
                self.video = m.group(1)
            self._log(level, text)

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


class Tracker:
    """A run's rows as it goes: the progress each says, and what the result reports.
    `instance_id`: the container the orchestrator selected when the goal came, kept for the
    whole run (a hibernate clears the engine's own mid-run; REVIEW_RULES O3)."""

    def __init__(self, action_id, instance_id=""):
        self.action_id = action_id
        self.instance_id = instance_id
        self.pct = 0.0
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
                self.pct = max(self.pct, pct)
                return phase, self.pct, doing if ok else f"{doing}: stopped"
        return None

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
    perception_classes = ["food container"]
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
        check_params(params)
        if self.mock:
            return self._mock(action_id, ctx)
        if not self._allowed():
            raise EngineError("not_allowed", NOT_ALLOWED + ": nothing ran", recoverable=False)
        ctx.checkpoint()  # cancelled before it started: nothing runs
        env = self._env()  # this goal's params, read once (REVIEW_RULES: none is a start-up copy)
        runs = state_dir(env) / "runs"
        log = RunLog(runs, RunLog.existing(runs))
        # the selection as the goal came: read once, used for every write and the run's argv
        track = Tracker(action_id, self.instance_id)
        t0 = time.monotonic()
        first = MOCK_PHASES[action_id][0][0]  # the first phase's name
        ctx.progress(first, 0.0, "starting", pausable=False)
        world = WorldWriter(self.get_logger())
        # the flight view (flight_view): the run's lines read as they are logged, one short line a step
        # on this node's stdout. It only reads, and nothing it meets stops the run or its reader
        view = FlightView()
        view.start(action_id, track.instance_id)
        self._run = Run(self._argv(action_id, track.instance_id), env, view.relay(LOGGER.log))
        code = None
        try:
            code = self._follow(self._run, log, track, ctx, world)
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
            self._progress(log, track, ctx)  # the last rows, written as it ended
            if not ctx.cancelled():
                self._sync_world(track, ctx, world)  # the last rows' word: a release, a put-back
            view.finish(code, cancelled=ctx.cancelled(), timed_out=run_.timed_out, held=bool(track.held))
        if ctx.cancelled():
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
                f"the run took longer than {RUN_BUDGET_S:.0f} s and was stopped its own way (Ctrl+C)",
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

    def _follow(self, run_, log, track, ctx, world):
        """Progress from the run's rows until it ends; a cancel is its Ctrl+C. Its exit code. The
        world-state writes the rows call for are `world`'s, never this loop's to wait on."""
        sent = escalated = None
        while run_.proc.poll() is None:
            self._progress(log, track, ctx)
            if not ctx.cancelled():
                self._sync_world(track, ctx, world)
            now = time.monotonic()
            if ctx.cancelled() and sent is None:
                self.get_logger().warning("cancelled: the run's own stop (Ctrl+C) sent")
                run_.interrupt()
                sent = now
            elif sent is None and now - run_.started > RUN_BUDGET_S:
                self.get_logger().error(
                    f"the run is still going after {RUN_BUDGET_S:.0f} s: its own stop (Ctrl+C), before the "
                    "orchestrator's 120 s cancel"
                )
                run_.timed_out = True
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
    def _progress(log, track, ctx):
        """The progress each new row of the run log says (a run cannot pause once started)."""
        for row in log.rows():
            said = track.add(row)
            if said is not None:
                ctx.progress(*said, pausable=False)

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
        return argv + ["--execute"]  # only ever run where allow_execute is on (execute())

    def _env(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env["BOX_OPENING_VIDEO"] = "1" if self._param("video") else "0"
        env["BOX_OPENING_VIZ"] = "1" if self._param("viz") else "0"
        env["BOX_OPENING_MISSION_FRAMES"] = "1" if self._param("mission_frames") else "0"
        if self._param("state_dir"):
            env["BOX_OPENING_STATE"] = self._param("state_dir")
        for name, var in (("planner_worlds_as", PLANNER_WORLDS_AS_ENV), ("planner_worlds_dir", PLANNER_WORLDS_DIR_ENV)):
            if self._param(name):  # unset: whatever the node's own environment says, as BOX_OPENING_BODY_LAYERS
                env[var] = str(self._param(name))
        return env

    def _mock(self, action_id, ctx):
        """A successful real run at the interface, nothing moved (review round 2): its phases in
        its words, the gripper's `holding` from the grip to the release and the container `opened`
        at the release, written as the real run writes them, and its result. Like the real run it
        cannot pause; a cancel ends it."""
        track = Tracker(action_id, self.instance_id)
        world = WorldWriter(self.get_logger())
        t0 = time.monotonic()
        for phase, pct, doing in MOCK_PHASES[action_id]:
            if ctx.cancelled():
                world.wait(WORLD_WAIT_S)
                raise Cancelled()
            ctx.progress(phase, pct, doing, pausable=False)
            if phase == "grip":
                track.held = True
            elif phase == "release":
                track.held, track.set_down, track.released = False, True, True
                track.lid_spot = list(MOCK_LID_XYZ)
            self._sync_world(track, ctx, world)
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
