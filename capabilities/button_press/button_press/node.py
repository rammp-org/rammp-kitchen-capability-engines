"""The button press capability: `press_button`, that presses a microwave button, and the UI's five
one-button actions (UI_ACTIONS: `press_start`, `press_power`, ...) that run it for one button each.

It finds the button (scene camera, then the wrist camera's dome detector), plans and gates every
leg in its sim, goes to the pre-press spot, presses by touch and comes back: press_button.py.
Everything orchestrator-facing comes from capability_engine.CapabilityEngine.

A UI action asks how many presses (1-5), runs the press under a progress panel whose Stop cancels the
action (the motion in flight stops), and ends with the result or the error on the panel
(capability_ui.RunUI). It never dry-runs: with allow_execute false it refuses.

SAFETY: the arm moves only with the allow_execute parameter true. Otherwise every action is a dry
run (detect, plan, gate; nothing moves) and its result says so. Someone stands at the e-stop
whenever allow_execute is on.

    ros2 launch button_press button_press.launch.py use_orchestrator:=false
    make action CAP=button_press ACTION=press_button PARAMS='{"target": "start_30s"}'
"""

from capability_engine import Action, ArmError, Cancelled, CapabilityEngine, EngineError, run
from capability_ui import RunUI, cancel_own_run
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import SetParametersResult

import button_press
from button_press import Abort, MotionFailed, dome_pattern
from button_press.kinematics import Kinematics
from button_press.press_button import Args, Run, press_button
from button_press.ui_panel import TEXT, Panel

MAX_PRESSES = 10
# The steps that take the fingertip to the panel. A failure in one may leave it there, and a retry
# would press the button again, so it is never recoverable (the Result carries no message).
AT_PANEL = ("approach", "press", "return")
AT_PANEL_NOTE = "The fingertip may be at the panel: move the arm clear of the microwave before anything else runs."
DEFAULTS = Args()
# The UI's buttons (the orchestrator labels them with the raw id): id -> (dome_pattern target, name on the panel).
# Each asks how many presses (1..UI_MAX_PRESSES), then presses and comes back.
UI_ACTIONS = {
    "press_start": ("start_30s", "Start"),
    "press_power": ("power_level", "Power"),
    "press_timer_clock": ("timer_clock", "Timer/Clock"),
    "press_defrost": ("wgt_time_defrost", "Defrost"),
    "press_stop_eco": ("stop_eco", "Stop/Eco"),
}
UI_MAX_PRESSES = 5
MOCK_RESULT = {"dry_run": True, "pressed": 0, "returned": False, "moved": False, "log_file": ""}
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
    "prepress_aim_mm": list(DEFAULTS.prepress_aim_mm),
    "record_bag": DEFAULTS.record_bag,
    "bag_frame_hz": DEFAULTS.bag_frame_hz,
    "planner_world_dir": DEFAULTS.planner_world_dir,
    "planner_world_dir_in_planner": DEFAULTS.planner_world_dir_in_planner,
    "planner_base_layers": list(DEFAULTS.planner_base_layers),
    "planner_restore_world": DEFAULTS.planner_restore_world,
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
        ),
        # the UI's buttons: each asks the count, then presses that button (UI_ACTIONS, _ui_run)
        *(Action(ui_id, f"Press {name}") for ui_id, (_, name) in UI_ACTIONS.items()),
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
            prepress_aim_mm=tuple(float(v) for v in s["prepress_aim_mm"]),
            record_bag=bool(s["record_bag"]),
            bag_frame_hz=float(s["bag_frame_hz"]),
            mount_rpy_deg=tuple(s["mount_rpy_deg"]),
            camera_frame=s["camera_frame"],
            camera_ns=s["camera_ns"],
            depth_topic=s["depth_topic"],
            far=s["far"],
            scene_ns=s["scene_ns"],
            scene_depth_topic=s["scene_depth_topic"],
            detections=bool(s["detections"]),
            planner_world_dir=s["planner_world_dir"],
            planner_world_dir_in_planner=s["planner_world_dir_in_planner"],
            planner_base_layers=tuple(s["planner_base_layers"]),
            planner_restore_world=s["planner_restore_world"],
        )

    def availability(self, instance_id, world_state):
        """The UI is offered only the one-button actions (UI_ACTIONS): `press_button` stays callable directly
        (from the UI, with no params, it would press start_30s once with no way to pick)."""
        return [a for a in super().availability(instance_id, world_state) if a.action_id in UI_ACTIONS]

    def execute(self, action_id, params, ctx):
        if action_id in UI_ACTIONS:
            return self._ui_run(action_id, params, ctx)
        args = self._args(params)
        if self.mock:
            ctx.progress("mock", 1.0, "mock: nothing detected or moved")
            return dict(MOCK_RESULT)
        phase = ["preflight"]
        # The user's screen (ui_panel.py): each step, and a Stop button. Nothing is shown standalone.
        panel = Panel(ctx, args.target, dry_run=not args.execute)

        def checkpoint():
            """ctx.checkpoint, and the panel's Stop: it ends the run as a cancel does (the arm holds
            where it is). Like a cancel's, its effect waits for a checkpoint: a move already sent
            to the driver finishes first."""
            if panel.wait_out_pause():
                self.get_logger().warning(f"press_button: Stop tapped on the panel during {phase[0]}")
                raise Cancelled()
            ctx.checkpoint()

        done = self._press(args, ctx, phase, checkpoint, panel.show)
        panel.show("done", 1.0, pausable=False, stop=False)
        return done

    def _press(self, args, ctx, phase, checkpoint, show):
        """One press run (press_button.py). `phase[0]` follows its step; `show(step, fraction, pausable)` puts each
        step on the user's screen. A failure is an EngineError named after its step (`failure`)."""

        def step(name, fraction, pausable=True):
            checkpoint()
            phase[0] = name
            ctx.progress(name, fraction, f"{args.target}: {name}", pausable=pausable)
            show(name, fraction, pausable)

        run = Run(args, self, self.kin, checkpoint, step)
        self.get_logger().info(
            f"press_button: {args.target} x{args.presses} {'EXECUTE' if args.execute else 'dry run'}"
        )
        try:
            done = press_button(run, args, self.target_pub)
            run.record_end(phase[0])
        except (Abort, ArmError) as e:
            run.record_end(phase[0], e)
            raise failure(phase[0], e) from None
        except BaseException as e:  # a cancel, or a bug: in the run's folder too, then as before
            run.record_end(phase[0], e)
            raise
        finally:
            run.close()
        ctx.progress("done", 1.0, "dry run: nothing moved" if done["dry_run"] else "done")
        return dict(done, log_file=run.log_path, moved=run.arm.sent_goals > 0)

    # -- the UI's one-button actions -------------------------------------------------------------------------------

    def _ui_run(self, ui_id, params, ctx):
        """Ask how many presses, press under a panel with Stop, show the result. A recoverable failure before the
        fingertip reached the panel (any step before AT_PANEL) offers the same press again; one at or after it, a
        driver failure, a bug or a Stop is shown and not retried. Every final notice waits (bounded) for Done."""
        target, name = UI_ACTIONS[ui_id]
        title = f"Press {name}"
        ui = RunUI(ctx, title, cancel=lambda: cancel_own_run(self), logger=self.get_logger())
        if not self.mock and not self.settings["allow_execute"]:
            message = (
                f"{title} is disabled: the button press moves the arm only with allow_execute:=true, with someone "
                "at the e-stop. Nothing moved."
            )
            ui.refuse(message)
            raise EngineError("execution_disabled", message, recoverable=True)
        presses = self._ui_presses(ui, name, params)
        args = self._args({"target": target, "presses": presses})
        count = f"{presses} press{'' if presses == 1 else 'es'}"
        phase = ["preflight"]

        def show(step, fraction, pausable=True):
            ui.moving(f"{TEXT.get(step, step)}: {name}, {count}. The arm is moving.", 100.0 * fraction, step)

        def press():
            if self.mock:
                ctx.progress("mock", 1.0, "mock: nothing detected or moved")
                return dict(MOCK_RESULT)
            return self._press(args, ctx, phase, ctx.checkpoint, show)

        while True:
            ctx.checkpoint()
            phase[0] = "preflight"
            try:
                result = ui.motion(
                    f"Pressing {name}, {count}. The arm is moving.",
                    press,
                    stopped_note=lambda: AT_PANEL_NOTE if phase[0] in AT_PANEL else "",
                )
            except Cancelled:  # Stop (its notice is shown) or the orchestrator's cancel
                raise
            except EngineError as error:
                self.get_logger().error(f"{ui_id}: {error.error_code} during {phase[0]}: {error}")
                message = f"Couldn't press {name} ({TEXT.get(phase[0], phase[0]).lower()}): {error}."
                if error.recoverable and phase[0] not in AT_PANEL:
                    # the same press again (same count), on its own button: "<title>" and Stop
                    if ui.ask(f"{message} Try again?", [{"button_id": "retry", "label": title}]) == "retry":
                        continue
                    raise
                ui.error(f"{message} {AT_PANEL_NOTE}" if phase[0] in AT_PANEL else message)
                raise
            except Exception as error:  # a bug: shown too, then as before
                self.get_logger().error(f"{ui_id}: {type(error).__name__} during {phase[0]}: {error}")
                note = f" {AT_PANEL_NOTE}" if phase[0] in AT_PANEL else ""
                ui.error(f"Couldn't press {name}: {type(error).__name__}: {error}.{note}")
                raise
            pressed = result.get("pressed", 0)
            ui.done(f"Pressed {name} {pressed} time{'' if pressed == 1 else 's'}.", mock=self.mock)
            return result

    def _ui_presses(self, ui, name, params):
        """How many presses, asked on the panel (1..UI_MAX_PRESSES, and Stop). With no UI session (standalone) it is
        the goal's `presses` param, 1 by default; no answer in time fails the action, nothing moved."""
        buttons = [{"button_id": f"presses_{n}", "label": str(n)} for n in range(1, UI_MAX_PRESSES + 1)]
        answer = ui.ask(f"How many presses of {name}?", buttons)
        if answer is not None:
            return int(answer.rsplit("_", 1)[1])
        if ui.connected is False:
            return params.get("presses", 1)
        message = "No answer: nothing was pressed."
        ui.notice(message)
        raise EngineError("no_answer", message, recoverable=True)


def failure(phase, error):
    """A failed run as the action's EngineError, `<phase>_failed`. Retried (recoverable) only away from the panel
    and only for a refused check: MotionFailed (the driver failed a motion, e.g. an e-stop; the arm holds where it
    stopped) and ArmError need someone to look first, and in AT_PANEL the fingertip may be at the button, where a
    retry would press it again."""
    if isinstance(error, Abort):
        recoverable = not isinstance(error, MotionFailed) and phase not in AT_PANEL
        return EngineError(f"{phase}_failed", str(error), recoverable=recoverable)
    return EngineError(f"{phase}_failed", f"{type(error).__name__}: {error}")


def main():
    run(Capability)


if __name__ == "__main__":
    main()
