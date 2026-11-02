"""The button press capability: one action, `press_button`, that presses a microwave button.

It finds the button (scene camera, then the wrist camera's dome detector), plans and gates every
leg in its sim, goes to the pre-press spot, presses by touch and comes back: press_button.py.
Everything orchestrator-facing comes from capability_engine.CapabilityEngine.

SAFETY: the arm moves only with the allow_execute parameter true. Otherwise every action is a dry
run (detect, plan, gate; nothing moves) and its result says so. Someone stands at the e-stop
whenever allow_execute is on.

    ros2 launch button_press button_press.launch.py use_orchestrator:=false
    make action CAP=button_press ACTION=press_button PARAMS='{"target": "start_30s"}'
"""

from capability_engine import Action, ArmError, CapabilityEngine, EngineError, run
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import SetParametersResult

import button_press
from button_press import Abort, MotionFailed, dome_pattern
from button_press.kinematics import Kinematics
from button_press.press_button import Args, Run, press_button

MAX_PRESSES = 10
# The steps that take the fingertip to the panel. A failure in one may leave it there, and a retry
# would press the button again, so it is never recoverable (the Result carries no message).
AT_PANEL = ("approach", "press", "return")
DEFAULTS = Args()
# ROS parameter -> its default (the run's settings that are not per action)
PARAMS = {
    "allow_execute": False,
    "mount_rpy_deg": list(DEFAULTS.mount_rpy_deg),
    "camera_frame": DEFAULTS.camera_frame,
    "camera_ns": DEFAULTS.camera_ns,
    "depth_topic": DEFAULTS.depth_topic,
    "far": DEFAULTS.far,
    "scene_ns": DEFAULTS.scene_ns,
    "scene_depth_topic": DEFAULTS.scene_depth_topic,
    "detections": DEFAULTS.detections,
    "log_dir": DEFAULTS.log_dir,
}


class _LogLines:
    """say()'s sink: every complete line goes to the node's log."""

    def __init__(self, logger):
        self.logger, self.buf = logger, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.logger.info(line.rstrip())
        return len(s)

    def flush(self):
        pass


class Capability(CapabilityEngine):
    display_name = "Button press"
    perception_classes = ["microwave"]
    required_resources = ["arm", "gripper"]
    actions = [
        Action(
            "press_button",
            "Press a button",
            params_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "enum": list(dome_pattern.NAMES), "default": DEFAULTS.target},
                    "presses": {"type": "integer", "minimum": 0, "maximum": MAX_PRESSES, "default": 1},
                    "no_return": {"type": "boolean", "default": False},
                },
            },
        )
    ]

    def __init__(self):
        super().__init__("button_press")
        self.settings = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        self.add_on_set_parameters_callback(self._refuse_setting_changes)
        self.kin = Kinematics(base_rpy_deg=self.settings["mount_rpy_deg"])  # the sim every leg is gated in
        # For viewing the planned pre-press pose in RViz. A publisher receives nothing, so it lives
        # with the node.
        self.target_pub = self.create_publisher(PoseStamped, "~/target_pose", 1)
        button_press.set_output(_LogLines(self.get_logger()))
        moves = "the arm WILL move" if self.settings["allow_execute"] and not self.mock else "nothing moves"
        self.get_logger().info(f"button_press: {moves}")

    @staticmethod
    def _refuse_setting_changes(params):
        """The settings are read once, here: a change would be accepted and ignored (allow_execute
        false would not stop the arm)."""
        if any(p.name in PARAMS for p in params):
            return SetParametersResult(successful=False, reason="button_press parameters are read at startup; restart")
        return SetParametersResult(successful=True)

    def _args(self, params) -> Args:
        target = params.get("target", DEFAULTS.target)
        presses = params.get("presses", 1)
        no_return = params.get("no_return", False)
        if target not in dome_pattern.NAMES:
            raise EngineError("bad_params", f"target {target!r} is not one of {list(dome_pattern.NAMES)}")
        if isinstance(presses, bool) or not isinstance(presses, int) or not 0 <= presses <= MAX_PRESSES:
            raise EngineError("bad_params", f"presses must be an integer in [0, {MAX_PRESSES}], not {presses!r}")
        if not isinstance(no_return, bool):
            raise EngineError("bad_params", f"no_return must be true or false, not {no_return!r}")
        s = self.settings
        if s["far"] not in ("scene", "wrist"):
            raise EngineError("bad_config", f"the far parameter must be 'scene' or 'wrist', not {s['far']!r}")
        return Args(
            target=target,
            presses=presses,
            no_return=no_return,
            execute=bool(s["allow_execute"]),
            log_dir=s["log_dir"],
            mount_rpy_deg=tuple(s["mount_rpy_deg"]),
            camera_frame=s["camera_frame"],
            camera_ns=s["camera_ns"],
            depth_topic=s["depth_topic"],
            far=s["far"],
            scene_ns=s["scene_ns"],
            scene_depth_topic=s["scene_depth_topic"],
            detections=bool(s["detections"]),
        )

    def execute(self, action_id, params, ctx):
        args = self._args(params)
        if self.mock:
            ctx.progress("mock", 1.0, "mock: nothing detected or moved")
            return {"dry_run": True, "pressed": 0, "returned": False, "moved": False, "log_file": ""}
        phase = ["preflight"]

        def step(name, fraction, pausable=True):
            ctx.checkpoint()
            phase[0] = name
            ctx.progress(name, fraction, f"{args.target}: {name}", pausable=pausable)

        run = Run(args, self, self.kin, ctx.checkpoint, step)
        self.get_logger().info(
            f"press_button: {args.target} x{args.presses} {'EXECUTE' if args.execute else 'dry run'}"
        )
        try:
            done = press_button(run, args, self.target_pub)
        except Abort as e:
            # MotionFailed: the driver failed a motion (an e-stop, control lost) and the arm holds
            # where it stopped; someone has to look before it runs again.
            recoverable = not isinstance(e, MotionFailed) and phase[0] not in AT_PANEL
            raise EngineError(f"{phase[0]}_failed", str(e), recoverable=recoverable) from None
        except ArmError as e:
            raise EngineError(f"{phase[0]}_failed", f"{type(e).__name__}: {e}") from None
        finally:
            run.close()
        ctx.progress("done", 1.0, "dry run: nothing moved" if done["dry_run"] else "done")
        return dict(done, log_file=run.log_path, moved=run.arm.sent_goals > 0)


def main():
    run(Capability)


if __name__ == "__main__":
    main()
