"""The box_opening capability: press the food container's button, take its lid off, set the lid
down beside the container, and go home.

Actions:
  open   the whole run: look, press, grip, lift, carry, set down, home (tasks/press_demo)
  press  the press alone: the knob pops, the lid stays on, home (press_demo --press-only)
  home   the arm home from wherever it holds (tasks/home_arm)
  check  the robot ready for a run: driver, planner, TF, gripper, cameras (tasks/preflight);
         nothing moves

Each action runs box opening's command line for it as a child process, exactly as it ran on
flatbot until now: its own process and executor, and Ctrl+C's stop path for a cancel (the
motion in flight is cancelled and the driver's answer awaited before it exits,
runtime/abort.py). Its console lines become this node's log lines, and the rows its run log
gains become the action's progress. A run is one motion from the press to the set-down, so
it cannot pause once it starts; a cancel stops it at once.

The wrist's and the scene camera's OWL detectors, which the run asks where the box is
(perception/owl_node), are nodes of this process: each loads OWLv2 onto the GPU once, in
the background from startup, so no run waits for it.

SAFETY: the arm moves only if the node runs with allow_execute:=true AND the goal sets
`execute`. Otherwise the run is a dry run: it finds the box, plans and checks every motion,
and sends none (its client's motion latch stays off). Someone stands at the e-stop whenever
allow_execute is on.

    ros2 launch box_opening box_opening.launch.py use_orchestrator:=false
    make action CAP=box_opening ACTION=open PARAMS='{"execute": false}'
"""

import json
import logging
import os
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

# ROS parameter -> default. Container and world name files in the image's config/ (or are
# absolute paths); the robot's per-run environment comes from its cameras, never from here.
PARAMS = {
    "allow_execute": False,
    "container": "ankou_pink.yaml",
    "world": "world_flatbot.yaml",
    # where runs keep their logs, captures and videos; "" = ~/.ros/box_opening
    "state_dir": "",
    "owl": True,
    "video": True,
    "viz": False,
    "speed_scale": 1.0,
}
EXECUTE = {"execute": {"type": "boolean", "default": False}}
HOME = {**EXECUTE, "lift_first_m": {"type": "number", "minimum": 0.0, "maximum": 0.15, "default": 0.0}}

# the command line each action runs: the package's executable, as `ros2 run box_opening <it>`
EXECUTABLES = {"open": "press_demo", "press": "press_demo", "home": "home_arm", "check": "preflight"}
POLL_S = 0.05
# a cancel is Ctrl+C's: one SIGINT, which cancels the motion in flight and waits for the
# driver to confirm. A second, after this long, is its escalation; a kill after that.
CANCEL_ESCALATE_S = 15.0
CANCEL_KILL_S = 10.0
SHUTDOWN_WAIT_S = 3.0  # on SIGTERM: the run's own stop first, inside docker stop's grace period
TAIL_LINES = 40

# a leg's row -> (phase, %) it ends; rows are written as their legs end. First match wins.
LEG_PHASES = (
    ("reach:look", "look", 10.0),
    ("press:close", "look", 12.0),
    ("reach:", "press", 20.0),
    ("depth:approach", "press", 25.0),
    ("depth:", "press", 35.0),
    ("grip:", "grip", 50.0),
    ("lid:", "carry", 70.0),
    ("place:down", "carry", 80.0),
    ("place:touch", "carry", 85.0),
    ("place:", "release", 90.0),
    ("home", "home", 100.0),
)
MOCK_PHASES = {
    "open": (
        ("locate", 5.0),
        ("look", 10.0),
        ("press", 35.0),
        ("grip", 50.0),
        ("carry", 85.0),
        ("release", 90.0),
        ("home", 100.0),
    ),
    "press": (("locate", 5.0), ("look", 10.0), ("press", 35.0), ("home", 100.0)),
    "home": (("home", 100.0),),
    "check": (("check", 100.0),),
}
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


class Tracker:
    """A run's rows as it goes: the progress each says, and what the result reports."""

    def __init__(self, action_id):
        self.action_id = action_id
        self.pct = 0.0
        self.presses = 0
        self.grips = 0
        self.lid_spot = None
        self.set_down = False
        self.released = False

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
        for start, phase, pct in LEG_PHASES:
            if leg.startswith(start):
                self.pct = max(self.pct, pct)
                return phase, self.pct, f"{leg}: {row.get('outcome') or ('ok' if ok else 'failed')}"
        return None

    def _note(self, leg, row):
        if leg == "scene_fix" and row.get("origin_xyz"):
            x, y = row["origin_xyz"][:2]
            self.pct = max(self.pct, 5.0)
            return "locate", self.pct, f"the box at [{x:.2f}, {y:.2f}] ({row.get('source', '?')})"
        if leg == "lid_spot" and row.get("xyz"):
            self.lid_spot = [round(float(v), 3) for v in row["xyz"]]
            x, y = self.lid_spot[:2]
            return "grip", self.pct, f"the lid goes to [{x:.2f}, {y:.2f}] ({row.get('tier', '')})"
        if leg == "reach_setdown":
            self.set_down = True
        if leg == "reach_grip_result":
            held = "held" if row.get("ok") else "missed"
            return "grip", self.pct, f"the grip {held} ({row.get('detail', '')})"
        if leg == "reach_refused":
            return "locate", self.pct, f"refused: {row.get('why', '')}"
        return None

    def lid_off(self, code):
        return self.action_id == "open" and code == 0 and self.set_down and self.released


def error_for(action_id, code, lines):
    """The EngineError a run that ended with `code` reports, from what it said last."""
    said = [ln for ln in lines if "STOP" in ln or "not starting" in ln or "FAIL" in ln or "NO BOX" in ln]
    why = (said or list(lines) or [f"exit {code}"])[-1]
    if action_id == "check":
        fails = [ln for ln in lines if ln.startswith("FAIL")]
        return EngineError("not_ready", "; ".join(fails) or why, recoverable=True)
    if "not starting" in why:
        return EngineError("not_ready", why, recoverable=True)
    if code == 2 or "NO BOX" in why:
        return EngineError("not_found", why, recoverable=True)
    if code == 3:
        return EngineError("not_home", why, recoverable=True)
    if "the arm holds" in why or "did not get home" in why:
        return EngineError("arm_holds", f"{why} -- home it with the `home` action", recoverable=False)
    if "MOVED the box" in why or "left the table" in why:
        return EngineError("box_moved", why, recoverable=True)
    if code is not None and code < 0:
        return EngineError("stopped", f"the run was ended by signal {-code}: {why}", recoverable=False)
    return EngineError("failed", why, recoverable=True)


class Capability(CapabilityEngine):
    display_name = "Open the container"
    perception_classes = ["food container"]
    required_resources = ["arm", "gripper"]
    actions = [
        Action("open", "Open the container", params_schema={"type": "object", "properties": EXECUTE}),
        Action("press", "Press the container's button", params_schema={"type": "object", "properties": EXECUTE}),
        Action("home", "Move the arm home", params_schema={"type": "object", "properties": HOME}),
        Action("check", "Check the robot is ready", params_schema={"type": "object", "properties": {}}),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("box_opening", **node_kwargs)
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        self._run = None  # the Run under way
        self._owl = []
        use_node_logger(self)
        self.context.on_shutdown(self._on_shutdown)
        if not self.mock and self.cfg["owl"]:
            threading.Thread(target=self._start_owl, name="owl-load", daemon=True).start()
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def availability(self, instance_id, world_state):
        """open and press for a container whose lid is on; home and check always."""
        lid_off = ((world_state.get("objects") or {}).get(instance_id) or {}).get("lid") == "off"
        return [
            ActionAvailability(action_id=a.action_id, available=a.action_id in ("home", "check") or not lid_off)
            for a in self.actions
        ]

    def execute(self, action_id, params, ctx):
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            return self._mock(action_id, ctx, note)
        ctx.checkpoint()  # cancelled before it started: nothing runs
        runs = state_dir(self._env()) / "runs"
        log = RunLog(runs, RunLog.existing(runs))
        track = Tracker(action_id)
        t0 = time.monotonic()
        first = MOCK_PHASES[action_id][0][0]
        ctx.progress(first, 0.0, "starting" + ("" if execute else " (dry run)"), pausable=False)
        self._run = Run(self._argv(action_id, params, execute), self._env(), LOGGER.log)
        try:
            code = self._follow(self._run, log, track, ctx)
        finally:
            self._progress(log, track, ctx)  # the last rows, written as it ended
            run_, self._run = self._run, None
        if ctx.cancelled():
            raise Cancelled()
        result = {
            "executed": execute,
            "lid_off": track.lid_off(code),
            "lid_xyz": track.lid_spot if track.lid_off(code) else None,
            "presses": track.presses,
            "grips": track.grips,
            "duration_s": round(time.monotonic() - t0, 1),
            "video": run_.video,
            "run_log": str(log.path) if log.path else None,
            "note": note,
        }
        if code != 0:
            raise error_for(action_id, code, run_.lines)
        if action_id == "check":
            result["checks"] = [ln for ln in run_.lines if ln[:4] in ("PASS", "WARN", "FAIL")]
        if result["lid_off"] and self.instance_id:
            ctx.update_object(self.instance_id, lid="off", lid_xyz=result["lid_xyz"])
        return result

    def _follow(self, run_, log, track, ctx):
        """Progress from the run's rows until it ends; a cancel is its Ctrl+C. Its exit code."""
        sent = escalated = None
        while run_.proc.poll() is None:
            self._progress(log, track, ctx)
            now = time.monotonic()
            if ctx.cancelled() and sent is None:
                self.get_logger().warning("cancelled: the run's own stop (Ctrl+C) sent")
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
    def _progress(log, track, ctx):
        """The progress each new row of the run log says (a run cannot pause once started)."""
        for row in log.rows():
            said = track.add(row)
            if said is not None:
                ctx.progress(*said, pausable=False)

    def _argv(self, action_id, params, execute):
        argv = [executable(EXECUTABLES[action_id])]
        argv += ["--container", config_path(self.cfg["container"], "containers")]
        argv += ["--bench-world", config_path(self.cfg["world"])]
        if action_id == "check":
            return argv
        argv += ["--speed-scale", str(float(self.cfg["speed_scale"]))]
        if action_id == "press":
            argv.append("--press-only")
        lift = float(params.get("lift_first_m", 0.0) or 0.0)
        if action_id == "home" and lift > 0.0:
            argv += ["--lift-first", str(lift)]
        return argv + (["--execute"] if execute else [])

    def _env(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env["BOX_OPENING_VIDEO"] = "1" if self.cfg["video"] else "0"
        env["BOX_OPENING_VIZ"] = "1" if self.cfg["viz"] else "0"
        if self.cfg["state_dir"]:
            env["BOX_OPENING_STATE"] = self.cfg["state_dir"]
        return env

    def _mock(self, action_id, ctx, note):
        for phase, pct in MOCK_PHASES[action_id]:
            ctx.checkpoint()
            ctx.progress(phase, pct, "mock: nothing detected or moved", pausable=False)
        return {
            "executed": False,
            "mock": True,
            "lid_off": False,
            "lid_xyz": None,
            "presses": 0,
            "grips": 0,
            "duration_s": 0.0,
            "video": None,
            "run_log": None,
            "note": note,
        }

    def _start_owl(self):
        """The two OWL detectors, as nodes of this process once each has loaded."""
        from rclpy.parameter import Parameter

        from .perception.owl_node import OwlDetector
        from .worlds import WorldStore

        store = WorldStore(config_path(self.cfg["world"]))
        container = config_path(self.cfg["container"], "containers")
        cameras = [("box_opening_owl_wrist", "wrist", store.wrist_camera_config or "")]
        scene_cfg = store._bench.get("scene_camera_config") or ""
        if scene_cfg:
            cameras.append(("box_opening_owl_scene", "scene", scene_cfg))
        for name, camera, camera_cfg in cameras:
            given = {"container": container, "camera": camera, "camera_config": camera_cfg}
            try:
                owl = OwlDetector(name, parameter_overrides=[Parameter(k, value=v) for k, v in given.items() if v])
            except Exception as e:  # no GPU, no weights: the run says no box, this says why
                self.get_logger().error(f"{name}: the OWL detector did not load: {e!r}")
                continue
            while self.executor is None:  # run() adds this node to its executor after __init__
                time.sleep(0.1)
            self.executor.add_node(owl)
            self._owl.append(owl)

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
