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
from button_press.ui_text import DISABLED, NO_ANSWER, bug_message, failure_message

MAX_PRESSES = 10
# The steps that take the fingertip to the panel. A failure in one may leave it there, and a retry
# would press the button again, so it is never recoverable (the Result carries no message).
AT_PANEL = ("approach", "press", "return")
AT_PANEL_NOTE = (
    "The fingertip may still be at the microwave's buttons. Move the arm clear of the microwave (teleop or Home) "
    "before running anything else."
)
DEFAULTS = Args()
# The UI's button on the microwave: id -> (dome_pattern target, the button's name on the panels, its label).
# "Heat up food", then the arm holds at Start's pre-press spot: "Press 30 sec", then "+30 sec" or "Done" (_ui_run).
UI_ACTIONS = {"heat_food": ("start_30s", "Start", "Press Button")}
UI_TITLE = "Heat up food"
HOLD_S = float("inf")  # how long the arm holds at the pre-press spot for "Press 30 sec" / "+30 sec" / "Done"
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
    "start_preset": DEFAULTS.start_preset,
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
        # the UI's button: heat up food (UI_ACTIONS, _ui_run)
        *(Action(ui_id, label) for ui_id, (_, _, label) in UI_ACTIONS.items()),
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
            start_preset=str(s["start_preset"]).strip(),
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
            error = failure(phase[0], e)
            error.at_start = arm_at_start(run)  # for the UI's failure panel: where the arm was left
            raise error from None
        except BaseException as e:  # a cancel, or a bug: in the run's folder too, then as before
            run.record_end(phase[0], e)
            raise
        finally:
            run.close()
        ctx.progress("done", 1.0, "dry run: nothing moved" if done["dry_run"] else "done")
        return dict(done, log_file=run.log_path, moved=run.arm.sent_goals > 0)

    # -- the UI's one-button actions -------------------------------------------------------------------------------

    def _ui_run(self, ui_id, params, ctx):
        """Heat up food from the touchscreen. "Heat up food" (and Stop); then the run goes to Start's pre-press spot
        and the arm HOLDS there while the panel offers "Press 30 sec" (and Stop), for as long as it takes (HOLD_S).
        Each press goes in, back out to the pre-press spot, and the panel offers "+30 sec" (another press) or "Done";
        Done sends the arm home and ends the action. Stop on any panel cancels: the arm holds where it is.

        A recoverable failure before the fingertip reached the panel (any step before AT_PANEL) offers the run again;
        one at or after it, a driver failure, a bug or a Stop is shown and not retried. Standalone (no UI session):
        no question is asked, the run presses once and comes back."""
        target, name, _ = UI_ACTIONS[ui_id]
        ui = RunUI(ctx, UI_TITLE, cancel=lambda: cancel_own_run(self), logger=self.get_logger())
        if not self.mock and not self.settings["allow_execute"]:
            message = (
                f"{UI_TITLE} is disabled: the button press moves the arm only with allow_execute:=true, with someone "
                "at the e-stop. Nothing moved."
            )
            ui.refuse(DISABLED)  # the panel's words; the goal's error keeps the log's
            raise EngineError("execution_disabled", message, recoverable=True)
        self._ui_confirm(ui)
        args = self._args({"target": target, "presses": 1})
        phase = ["preflight"]

        def ask_press(pressed):
            """At the pre-press spot, after `pressed` presses: True to press once more, False to finish."""
            if pressed == 0:
                message = f"The arm is at the microwave's {name} button. Tap Press 30 sec to start heating."
                buttons = [{"button_id": "press", "label": "Press 30 sec"}]
            else:
                total = f"{30 * pressed} sec"
                message = f"Heating: {total} added. Tap +30 sec to add more, or Done to move the arm away."
                buttons = [{"button_id": "press", "label": "+30 sec"}, {"button_id": "done", "label": "Done"}]
            answer = ui.ask(message, buttons, timeout_s=HOLD_S)
            if answer is None:  # standalone (no UI): one press, then home
                return pressed == 0
            return answer == "press"

        args.ask_press = ask_press

        def show(step, fraction, pausable=True):
            ui.moving(f"{TEXT.get(step, step)}. The arm is moving.", 100.0 * fraction, step)

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
                    f"Heating up food: moving to the microwave's {name} button.",
                    press,
                    stopped_note=lambda: AT_PANEL_NOTE if phase[0] in AT_PANEL else "",
                )
            except Cancelled:  # Stop (its notice is shown) or the orchestrator's cancel
                raise
            except EngineError as error:
                self.get_logger().error(f"{ui_id}: {error.error_code} during {phase[0]}: {error}")
                # the panel's words (ui_text): what happened, the arm's state, what to do
                message = failure_message(
                    name,
                    TEXT.get(phase[0], phase[0]),
                    phase[0],
                    str(error),
                    motion_failed=not error.recoverable,
                    at_start=getattr(error, "at_start", None),
                )
                if error.recoverable and phase[0] not in AT_PANEL:
                    retry = [{"button_id": "retry", "label": UI_TITLE}]
                    if ui.ask(f"{message} Tap {UI_TITLE} to try again.", retry) == "retry":
                        continue
                    raise
                ui.error(message)
                raise
            except Exception as error:  # a bug: shown too, then as before
                self.get_logger().error(f"{ui_id}: {type(error).__name__} during {phase[0]}: {error}")
                ui.error(bug_message(name, phase[0], error))
                raise
            if self.mock:
                ui.done(f"{UI_TITLE}.", mock=True)
            return result  # the user's Done ended it; the arm is home

    def _ui_confirm(self, ui):
        """The first panel: "Heat up food" (and Stop). Standalone (no UI session) it goes ahead; no answer in time
        fails the action, nothing moved."""
        answer = ui.ask(
            "Heat up food? The arm moves to the microwave's Start button and waits there.",
            [{"button_id": "heat", "label": UI_TITLE}],
        )
        if answer == "heat" or (answer is None and ui.connected is False):
            return
        ui.notice(NO_ANSWER)
        raise EngineError("no_answer", "No answer: nothing was pressed.", recoverable=True)


def failure(phase, error):
    """A failed run as the action's EngineError, `<phase>_failed`. Retried (recoverable) only away from the panel
    and only for a refused check: MotionFailed (the driver failed a motion, e.g. an e-stop; the arm holds where it
    stopped) and ArmError need someone to look first, and in AT_PANEL the fingertip may be at the button, where a
    retry would press it again."""
    if isinstance(error, Abort):
        recoverable = not isinstance(error, MotionFailed) and phase not in AT_PANEL
        return EngineError(f"{phase}_failed", str(error), recoverable=recoverable)
    return EngineError(f"{phase}_failed", f"{type(error).__name__}: {error}")


def arm_at_start(run):
    """Whether a failed run left the arm at its start joints, for the failure panel: the run's own `at_home()` (a run
    that goes back to the start after a failed check has it), or None when the run can't tell (no such method, or the
    arm's state can't be read). Only reads the arm's state."""
    at_home = getattr(run, "at_home", None)
    if at_home is None:
        return None
    try:
        return bool(at_home())
    except Exception:  # the panel must still show
        return None


def main():
    run(Capability)


if __name__ == "__main__":
    main()
