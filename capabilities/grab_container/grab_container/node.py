"""The grab_container capability: grab the food container (tupperware) by its brim.

Action `grab`:
  0. home: a cuRobo move to `home_joints`, where every run starts.
  1. locate (scene.locate_container): the scene segmenter's food-container mask and the scene
     camera's depth give a coarse position.
  2. view (scene.view_pose): the wrist camera moves above the container, on the approach side.
  3. grasp (grasp.grab): the students' YOLO brim detector and a circle fit find the grasp point,
     then open, approach (view -> pre-grasp -> grasp in one motion, self.arm.go_to_ee_poses), close and
     raise a little with the gripper turned level, holding it.
The detector loads in the background from the node's start. Every move is a cuRobo move
(self.arm.go_to_*); the goal's `impedance` runs them all on the driver's joint
impedance controller (`impedance_gains`) instead of stiff position control. Action `park`: the scan
pose by a direct joint move, for when cuRobo refuses the start state. A failed check stops the run
with the arm holding where it is (EngineError "refused").

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets
`execute`. Otherwise the goal is a dry run (grab: locate and plan the view move; park: report the
move). Someone stands at the e-stop whenever allow_execute is on.

UI action `grab_by_brim` (a tupperware tap, one press): `grab` executing, under capability_ui.RunUI: a
progress panel with Stop that names each step, then Done, failed or Stopped (what the hand holds).
`grab` and `park` stay direct goals (their params need setting) and are not offered to the UI.

    ros2 launch grab_container grab_container.launch.py use_orchestrator:=false
    make action CAP=grab_container ACTION=grab PARAMS='{"execute": false}'
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run
from capability_engine.arm import HOLD_LEVEL, IMPEDANCE
from capability_ui import RunUI, cancel_own_run

from . import grasp, scene, ui_text

# ROS parameter -> default. Topics, frames and paths are per robot (the fragment's params:).
PARAMS = {
    "allow_execute": False,
    "model_path": "/models/best.pt",
    "color_topic": "/wrist_camera/color/image_raw",
    "camera_info_topic": "/wrist_camera/color/camera_info",
    "depth_topic": "/wrist_camera/depth/image_rect_raw",
    "camera_frame": "",
    "detections_topic": "/perception/object_detections",
    "scene_color_topic": "/scene_camera/color/image_raw",
    "scene_camera_info_topic": "/scene_camera/color/camera_info",
    "scene_depth_topic": "/scene_camera/depth/image_rect_raw",
    # the tool point along the approach from the brim's outer edge (m; negative: short of it)
    "grasp_depth_m": grasp.GRASP_DEPTH_M,
    # new frames whose brim edges must agree to AGREE_M (grasp.N_FRAMES)
    "detect_frames": grasp.N_FRAMES,
    # where grab starts (rad, arm_joint_1..7, as /joint_states reports them): the flatbot's, read
    # 2026-10-28 with the arm at home
    "home_joints": [0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725],
    # the driver's impedance profile for `impedance` goals: default, soft, medium or stiff
    "impedance_gains": "default",
}
EXECUTE = {"type": "object", "properties": {"execute": {"type": "boolean", "default": False}}}
GRAB = {"type": "object", "properties": {**EXECUTE["properties"], "impedance": {"type": "boolean", "default": False}}}

# The tupperware's perception classes: "food container" is the scene segmenter's class (scene.CONTAINER_CLASS);
# "food_container" the same without its space, for a selection typed by hand (as microwave_place's)
CONTAINER_CLASSES = ["food_container", "food container"]
# The UI's one press: grab, executing, its other params at their defaults (stiff position control)
UI_GRAB = "grab_by_brim"
UI_GRAB_PARAMS = {"execute": True}
# Steps (ui_text.step) whose failure comes before the hand reaches for the brim: the UI offers the grab again
UI_RETRY_STEPS = ("", "home", "locate", "view", "detect")
# How long the retry question waits for Grab (s): under the orchestrator's 120 s execute timeout (REVIEW_RULES O3),
# which feat/orchestrator resets on each progress report. No answer ends the action with its refusal.
UI_PROMPT_TIMEOUT_S = 90.0


class _PhaseCtx:
    """The action's ctx, remembering the run's last step (ui_text.step) and showing each on the UI's panel."""

    def __init__(self, ctx, ui):
        self._ctx = ctx
        self._ui = ui
        self.step = ""

    def progress(self, phase, pct, message="", pausable=True):
        self.step = ui_text.step(phase, pct)
        self._ctx.progress(phase, pct, message, pausable)
        text, bar = ui_text.progress(phase, pct, message)
        self._ui.moving(text, bar, phase)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


class Capability(CapabilityEngine):
    display_name = "Grab container"
    # a tupperware tap activates the engine and offers the UI's grab (availability)
    perception_classes = list(CONTAINER_CLASSES)
    required_resources = ["arm", "gripper"]
    actions = [
        Action("grab", "Grab the container", params_schema=GRAB),
        Action("park", "Park at the scan pose", params_schema=EXECUTE),
        Action(UI_GRAB, ui_text.LABEL, perception_classes=list(CONTAINER_CLASSES)),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("grab_container", **node_kwargs)
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        # the detector's loading (a Future): from startup, so the first grab finds it warm; a grab
        # starts it again after a failed load
        self.model = None if self.mock else ThreadPoolExecutor(1).submit(self._load_model)
        grasp.GRASP_DEPTH_M = self.cfg["grasp_depth_m"]
        grasp.N_FRAMES = self.cfg["detect_frames"]
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def availability(self, instance_id, world_state):
        """The UI is offered only the one-press grab (on a tupperware tap); `grab` and `park` stay direct goals."""
        return [a for a in super().availability(instance_id, world_state) if a.action_id == UI_GRAB]

    def execute(self, action_id, params, ctx):
        if action_id == UI_GRAB:
            return self._ui_run(ctx)
        return self._direct(action_id, params, ctx)

    def _direct(self, action_id, params, ctx, sensors=None):
        """`grab` or `park`. `sensors()`: the (wrist, scene camera, detections) to use; None makes them for this run
        (_sensors)."""
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing detected or moved")
            return {"executed": False, "mock": True}

        log = self.get_logger().info
        if action_id == "grab" and self.model is None:  # a failed load: again, while the arm moves
            self.model = ThreadPoolExecutor(1).submit(self._load_model)
        try:
            if action_id == "park":
                return {**grasp.park(self.arm, execute, log), "note": note}
            gains = self.cfg["impedance_gains"]
            motion = {"control_mode": IMPEDANCE, "gains": gains} if params.get("impedance") else {}
            wrist, scene_cam, detections = sensors() if sensors is not None else self._sensors()
            return {**self._grab(wrist, scene_cam, detections, execute, log, ctx, motion), "note": note}
        except grasp.Abort as e:
            raise EngineError("refused", f"{e} -- the arm holds where it is", recoverable=True) from e

    def _sensors(self):
        """(wrist camera, scene camera, scene detections) on one tf buffer. Their subscriptions are per action
        (create_action_subscription, create_action_tf_buffer): the engine drops them when the action ends."""
        cfg = self.cfg
        tfbuf = self.create_action_tf_buffer()
        wrist = self._camera(
            tfbuf, cfg["color_topic"], cfg["camera_info_topic"], cfg["depth_topic"], cfg["camera_frame"]
        )
        scene_cam = self._camera(
            tfbuf, cfg["scene_color_topic"], cfg["scene_camera_info_topic"], cfg["scene_depth_topic"]
        )
        return wrist, scene_cam, scene.SceneDetections(self, cfg["detections_topic"])

    def _ui_run(self, ctx):
        """The UI's one press: `grab` executing, under capability_ui.RunUI. A progress panel with Stop names each
        step (Stop cancels this run: the motion in flight stops), then Done, failed or Stopped (saying what the
        hand holds). A refusal before the hand reached for the brim (UI_RETRY_STEPS) shows its reason and offers
        the grab again, waiting for an explicit tap; anything later is shown with Done only, never retried. Every
        final notice is shown and waited on (bounded) before this returns or raises.

        The cameras', detections' and tf subscriptions are made once, by the first try, and kept for every retry
        (the engine drops them when this action ends). Before each wait for a tap, a ctx.progress tells the
        orchestrator the action is alive (feat/orchestrator resets its 120 s execute timeout on each), and no wait
        is longer than that: a notice NOTICE_TIMEOUT_S (60 s), the retry question UI_PROMPT_TIMEOUT_S."""
        ui = RunUI(ctx, ui_text.TITLE, cancel=lambda: cancel_own_run(self), logger=self.get_logger())

        def waiting(what):  # the orchestrator's progress before a wait (never shown on the UI)
            ctx.progress("ui", 100.0, f"waiting for {what} on the UI", pausable=False)

        if not self.mock and not self.cfg["allow_execute"]:
            waiting("Done")
            ui.refuse(ui_text.disabled())
            raise EngineError(
                "execution_disabled",
                f"{ui_text.TITLE} is disabled: launch with allow_execute:=true for an attended run.",
                recoverable=True,
            )
        kept = []

        def sensors():
            if not kept:
                kept.extend(self._sensors())
            kept[0].reset()  # the wrist camera's image turn is per run (set_upright), as a new Camera's
            return kept

        def stopped_note(run_ctx):
            waiting("Done")  # called just before the Stopped notice
            return ui_text.hand(run_ctx.step)

        while True:
            ctx.checkpoint()
            run_ctx = _PhaseCtx(ctx, ui)
            try:
                result = ui.motion(
                    ui_text.MOVING,
                    lambda run_ctx=run_ctx: self._direct("grab", dict(UI_GRAB_PARAMS), run_ctx, sensors),
                    stopped_note=lambda run_ctx=run_ctx: stopped_note(run_ctx),
                )
            except EngineError as error:
                self.get_logger().error(f"{UI_GRAB}: {error}")
                if ui.user_stopped or ctx.cancelled():
                    raise  # Stop showed its own notice; an orchestrator cancel has taken the UI back
                if error.recoverable and error.error_code == "refused" and run_ctx.step in UI_RETRY_STEPS:
                    waiting(ui_text.RETRY_LABEL)
                    message = f"{ui_text.retry(str(error))} Press {ui_text.RETRY_LABEL} to try again."
                    retry = [{"button_id": "retry", "label": ui_text.RETRY_LABEL}]
                    title = f"{ui_text.TITLE}: not done"
                    # RunUI.retry's question, with a wait under the orchestrator's 120 s
                    if ui.ask(message, retry, UI_PROMPT_TIMEOUT_S, title=title, style="warning") == "retry":
                        continue
                    raise  # no UI session, no answer in time
                waiting("Done")
                ui.error(ui_text.failure(str(error), ui_text.hand(run_ctx.step)))
                raise
            except Exception as error:  # Cancelled, an arm error (e-stop, control taken), the detector, a bug
                self.get_logger().error(f"{UI_GRAB}: {error!r}")
                if not (ui.user_stopped or ctx.cancelled()):
                    waiting("Done")
                    ui.error(ui_text.unexpected(error, ui_text.hand(run_ctx.step)))
                raise
            waiting("Done")
            if result.get("mock"):
                ui.done(ui_text.DONE, mock=True)
            else:
                ui.done(ui_text.DONE if result.get("raised", True) else ui_text.DONE_NOT_RAISED)
            return result

    def _grab(self, wrist, scene_cam, detections, execute, log, ctx, motion):
        why = self.arm.not_ready()
        if why is not None:
            raise grasp.Abort(why)
        home = self.cfg["home_joints"]
        if execute:
            ctx.progress("home", 0.0)
            if not self.arm.go_to_joint_config(home, "home", **motion):
                raise grasp.Abort(f"home: {self.arm.last_error} -- HOLDING HERE")
        else:
            plan = self.arm.plan_to_joints(home, self.arm.raw_joints())
            if not plan.success:
                raise grasp.Abort(f"home: cuRobo found no path ({plan.message}) -- arm not commanded")
            log(f"home             cuRobo {len(plan.t):3d} pts  {plan.t[-1]:.1f} s")
        ctx.progress("locate", 2.0)
        c, obj = scene.locate_container(detections, scene_cam)
        pos, quat = scene.view_pose(c, grasp.grasp_orientation(), scene.tool_T_cam(self.arm, wrist))
        log(f"{obj.instance_id} ({obj.score:.2f}) at {np.round(c, 3)}; view from {np.round(pos, 3)}")
        located = {"container": obj.instance_id, "container_point": c.tolist(), "view_pose": pos.tolist()}
        if not execute:
            plan = self.arm.plan_to_pose(pos, quat, home)  # from home, where the run will start
            if not plan.success:
                raise grasp.Abort(f"view: cuRobo found no path ({plan.message}) -- arm not commanded")
            log(f"view             cuRobo {len(plan.t):3d} pts  {plan.t[-1]:.1f} s")
            log("DRY: located and planned the view only (the grasp is detected from the view)")
            return {"executed": False, **located}
        ctx.progress("view", 5.0)
        if not self.arm.go_to_ee_pose(pos, quat, "view", **motion):
            raise grasp.Abort(f"view: {self.arm.last_error} -- HOLDING HERE")
        log(f"wrist image turned {90 * wrist.set_upright()} deg for the detector (world up = image up)")
        return {**located, **grasp.grab(self.arm, wrist, self._model(), True, log, ctx.progress, HOLD_LEVEL, **motion)}

    def _camera(self, tfbuf, color_topic, info_topic, depth_topic, frame=""):
        from .camera import Camera

        return Camera(self, tfbuf, rammp_arm.FRAME, color_topic, info_topic, depth_topic, frame)

    def _model(self):
        try:
            return self.model.result()
        except Exception:
            self.model = None  # the next grab tries again
            raise

    def _load_model(self):
        """The detector, run once on each image shape it will see (turned or not): the first inference
        initialises CUDA and takes seconds."""
        from ultralytics import YOLO

        model = YOLO(self.cfg["model_path"])
        for shape in ((400, 640, 3), (640, 400, 3)):
            model.predict(np.zeros(shape, np.uint8), conf=grasp.CONF, retina_masks=True, verbose=False)
        return model


def main():
    run(Capability)


if __name__ == "__main__":
    main()
