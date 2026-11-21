"""The microwave_grab capability: put the held container into the open microwave, and take it out again.

  microwave_detection.py   where the microwave is, from the scene camera (SAM3 masks + depth)
  place_motion.py          planning, checking and running the arm (Walter Hsieh's place_container motion,
                           with the transit, the straight lines and the release the microwave needs)
  rim_detection.py         take_out: where the container's rim is, from the wrist camera (grab_container's SAM3
                           rim detection)
  take_out.py              take_out: planning and running the arm
  microwave_grab_node.py  this action server: parameters, the actions, the plan and state files

Everything is in arm_mount (x forward, y left, z up).

Action:
  place    the tupperware, gripped by its side (after grab_container), into the open microwave:
             1. locate (microwave_detection.locate): the opening's known size fitted to SAM3's cavity mask,
                frame after frame until `frames` consecutive detected frames agree (agree_m / agree_deg);
             2. obstacle (place_motion.microwave_obstacles): the microwave as one box, from the depth of SAM3's
                "microwave" mask; only the first motion (side shift, transit to in front of the door) avoids it,
                nothing after it is collision-checked;
             3. plan (place_motion.search_place): for each of side_shifts_cm (the hand moved level to its
                left first) and each of heading_offsets_deg: the side shift, the target and pre-insert pose
                reachable, the straight line in, cuRobo's level transit to the pre-insert pose (refused if it
                swings any joint more than TRANSIT_TRAVEL_MAX_DEG), then on the arm model: the side shift and
                the transit clear_min from the microwave box, the tilt within tilt_max_deg everywhere, the line
                in ending on the target, the line down, the release tilt and the back-out planned (none of
                them collision-checked). The first that passes wins. A dry run stops here and returns the plan
                (also written to state_dir/plans);
             4. with execute (place_motion.run_place): the side shift, cuRobo's transit and the line in as ONE
                motion (insert_speed peak), the impedance lowering until the glass stops it, the gripper opened
                to RELEASE_OPEN in place (still in impedance), the hand pitched RELEASE_TILT_DEG nose-down
                (impedance) so the container slides off, a stop, then position control and the level line back
                out, the hand kept tilted, to above the pre-insert pose.

  take_out the container out of the open microwave, from the take-out start pose (the wrist camera looking at
             it, the gripper open; take_out.py has the steps): the container's nearest point from the wrist camera
             (rim_detection), cuRobo moves to pre-grasp with the hand turned 90 deg, then reaches straight in to
             `grasp_depth` /
             `grasp_up` from that point (position control, the last `impedance_in` in joint impedance), the gripper
             closed in position control, a `take_lift` raise and the level line back out. cuRobo plans the approach
             against flatbot collision layers; the insertion onward is planned on the arm model
             first; a dry run stops there and returns the plan (also written to state_dir/plans).

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets `execute`.
Otherwise every goal is a dry run that locates, plans and checks, and returns the plan. Someone stands at
the e-stop whenever allow_execute is on. Take-out checks the cuRobo approach and exit against flatbot layers;
insertion and lift remain local IK.
For place, only the first motion (side shift, transit) is checked against the
microwave; from the line in on nothing is collision-checked. Any check that fails refuses with the reason.

  take_out with `"site": "fridge"`: the same take-out with the fridge's settings (fridge_take_home_joints,
             fridge_take_approach_world, fridge_grasp_up, fridge_reach_in). Its approach world has no fridge, shelves
             or open door, so an executing fridge take-out is refused unless fridge_accept_unmodeled_door is true.
             Direct goals only: the UI has no fridge grab.

UI actions (one press each, fixed settings, panels with Stop on the UI bridge: capability_ui.RunUI):
  retrieve              (tupperware tap: food_container) take_out, executing (take_home_joints' view)
A recoverable refusal before the arm started (no "STOPPED" in front of it) offers the same action again; anything
after the arm started is shown with Done only. Stop cancels the run (the motion in flight stops) and says what the
hand was left holding.

    ros2 launch microwave_grab microwave_grab.launch.py use_orchestrator:=false allow_execute:=true
    make action CAP=microwave_grab ACTION=place PARAMS='{"execute": false}'
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run
from capability_engine.driver_state import Refused as ArmRefused
from capability_ui import RunUI, cancel_own_run

from . import collision_world, microwave_detection, place_motion, rim_detection, take_out, ui_text

# the fridge take-out's own settings: take_out's setting <- this node's parameter
FRIDGE_TAKE = {
    "take_home_joints": "fridge_take_home_joints",
    "take_approach_world": "fridge_take_approach_world",
    "grasp_up": "fridge_grasp_up",
    "reach_in": "fridge_reach_in",
}
SITES = ("microwave", "fridge")
FRIDGE_DOOR_GAP = (
    "the fridge take-out is not collision-checked against the fridge: cuRobo's approach world has only the chair, "
    "the user and the floor (not the cabinet, its shelves or the open door), and the reach in, lift and back-out are "
    "not checked at all. Set fridge_accept_unmodeled_door:=true only for an attended run with a clear path"
)

# ROS parameter -> default.
PARAMS = {
    "allow_execute": False,
    "door_geometry_file": "/microwave_data/door.json",
    "planner_config": "/config/robot/curobo/planner.yaml",
    "door_open_angle_deg": 90.0,
    "door_alignment_m": 0.08,
    "door_alignment_deg": 10.0,
    "microwave_wall_m": 0.02,
    "microwave_panel_width_m": 0.127,
    "microwave_border_m": 0.0254,
    "microwave_base_layers": ["/config/robot/curobo/layer_mebot_floor.yaml"],
    "record_bag": True,
    "bag_frame_hz": 1.0,
    "bag_state_hz": 25.0,
    # flange face to the closed fingertips (m)
    "flange_to_tip": place_motion.FLANGE_TO_TIP,
    # contact: vz (m/s, + up) back up to this after moving down; 0.0 came 4.5 s late on the flatbot
    "vz_stop": -0.01,
    # the hold after contact, this far (m) below the contact point, while the gripper opens
    "hold_press": 0.003,
    # held in impedance at the start of the lowering this long (s); must end still
    "settle_s": 0.2,
    # contact more than this (m) above where the box's bottom meets the glass = not the glass
    "early_stop": 0.03,
    # joint impedance for the settle, lowering, hold, gripper opening and release tilt (one set: the stiffness never
    # changes mid-place). Default: the driver's documented starting point (rammp_arm.IMPEDANCE)
    "impedance_kq": list(rammp_arm.IMPEDANCE["kq"]),
    "impedance_zeta": rammp_arm.IMPEDANCE["zeta"],
    "impedance_torque_limit": list(rammp_arm.IMPEDANCE["torque_limit"]),
    # the hand's speed where the glass should be (m/s): what the contact test was tuned at
    "lower_speed": 0.12,
    # the plans (plans/, last_plan.json) and the descent recordings (descents/)
    "state_dir": "/config/microwave_grab",
    # locate: the scene camera's depth and intrinsics (the 3D detections are not used)
    "depth_topic": "/scene_camera/depth/image_rect_raw",
    "camera_info_topic": "/scene_camera/color/camera_info",
    "detections_topic": "/perception/object_detections_3d",
    # detect continuously until `frames` consecutive detected frames lie within agree_m / agree_deg of their
    # median (at most timeout_s)
    "frames": 3,
    "timeout_s": 20.0,
    "agree_m": 0.01,
    "agree_deg": 2.0,
    # SAM3's prompt for the whole oven (its outside, for the obstacles)
    "microwave_prompt": "microwave",
    # the tool this far (m) over the turntable glass at the end of the line in (the box's bottom
    # place_motion.BOX_ABOVE lower), then lowered from there
    "above": 0.055,
    # the container's spot this far (m) back from the turntable's middle toward the opening, along the oven's axis
    "closer": 0.02,
    # the hand's heading off the oven's axis (deg, + = left), tried in this order; the first that passes wins
    "heading_offsets_deg": [0.0],
    # before the transit, the hand moved this far (cm) level to its left (- = right), tried in this order
    # (each with every heading); the first that passes wins
    "side_shifts_cm": [3.0, -3.0, 6.0, -6.0, 9.0, -9.0],
    # the side shift and the transit: every arm and gripper link at least this far (m) from the microwave box
    "clear_min": 0.01,
    "take_clear_min": 0.005,  # local grab path clearance, metres
    # the hand's tilt vs the start, anywhere on the transit and the lines (deg)
    "tilt_max_deg": 2.0,
    # cuRobo's world files go here; the planner's container must see this directory at the SAME path
    "worlds_dir": "/home/snorlax/.ros/rammp_box_opening/worlds",
    # the hand's peak speed (m/s; position control, min-jerk: about half of it on average) on the approach
    # (side shift + transit + line in, one motion) and the back-out
    "insert_speed": 0.16,
    # Resolved inside the planner container; its config must load flatbot's platform, seat and person.
    "take_approach_world": "/config/robot/curobo/world.yaml",
    "take_approach_speed_scale": rammp_arm.GO_TO_SPEED_SCALE,
    "take_exit_speed_scale": rammp_arm.GO_TO_SPEED_SCALE,
    "take_pullback_m": 0.05,
    # Measured stationary pose with successful wrist-camera container detection, 2026-11-18.
    "take_home_joints": [
        0.5830700922169674,
        -0.746675268515478,
        -1.6893474753519921,
        -2.0094160288537477,
        -1.0276813740962787,
        -1.7364811655886498,
        -3.0368779584762042,
    ],
    # take_out with site "fridge" (direct goals only; no UI button): its viewing pose, arm_joint_1..7 in radians.
    # Default: the fridge capability's start pose (fridge/constants.py START_JOINTS_DEG, its start_joints_deg,
    # 11-12), the only recorded fridge-facing pose; NOT yet tried as a take-out viewing pose
    "fridge_take_home_joints": [
        1.7983872612549572,
        -0.5358160803622591,
        -2.9606020101579813,
        -2.1476276445790226,
        1.7739526517270365,
        1.9205603088945604,
        0.04363323129985824,
    ],
    # ... its approach world (inside the planner container; same layer checks as take_approach_world). The default has
    # no fridge in it
    "fridge_take_approach_world": "/config/robot/curobo/world.yaml",
    # ... its grasp height over the detected point and line-in length (m): the microwave's values, untuned here
    "fridge_grasp_up": 0.035,
    "fridge_reach_in": 0.30,
    # ... an executing fridge take-out is refused unless this is true: nothing it plans or runs knows the fridge, its
    # shelves or its open door (FRIDGE_DOOR_GAP). Attended runs only
    "fridge_accept_unmodeled_door": False,
    # take_out: whole-container detection on the wrist camera (colour, aligned depth, camera_info)
    "wrist_topics": [
        "/wrist_camera/color/image_raw",
        "/wrist_camera/depth/image_rect_raw",
        "/wrist_camera/color/camera_info",
    ],
    # SAM3 prompt: the whole container with its sleeve
    "container_prompt": "food container together with its blue sleeve",
    "sam3_threshold": 0.5,
    # Historical parameter names: agreement of whole-container masks; no rim detection.
    "rim_frames": 3,
    "rim_mask_iou": 0.85,
    "rim_timeout_s": 60.0,
    # the grasp: the tool frame this far (m) past the detected container point along the hand's heading (0: on it)
    "grasp_depth": 0.0,
    # ... and this far (m) above it: the tool's height for the pinch over the detected rim point (2026-11-03: the hand
    # placed by hand at the right height, tool z 0.332, over the rim the wrist camera found at z 0.302; +0.5 cm
    # after the first runs)
    "grasp_up": 0.035,
    # the pre-grasp pose this far (m) back from the grasp along the hand's level heading: the line in's length
    "reach_in": 0.30,
    # the gripper's opening for the reach in (0 = fully open .. 1 closed)
    "grip_open": 0.0,
    # the last this far (m) of the line in, in joint impedance (a short stop to switch); position control before it, and
    # from the gripper's close on
    "impedance_in": 0.10,
    # take_out's joint impedance: the driver's cap (kq <= 400); the place's (impedance_*, kq 80/30) let the arm sag
    # under its own weight (2026-11-03). The torque limits are the place's (impedance_torque_limit)
    "take_impedance_kq": [400.0, 400.0, 400.0, 400.0, 200.0, 200.0, 200.0],
    "take_impedance_zeta": 0.7,
    # the hand's peak speed (m/s) on that impedance part of the line in: half the approach's insert_speed
    "impedance_speed": 0.08,
    # after closing: straight up this far (m), then the place's level retract
    "take_lift": 0.02,
}
EXECUTE = {"type": "object", "properties": {"execute": {"type": "boolean", "default": False}}}
PLACE = EXECUTE
TAKE_OUT = {
    "type": "object",
    "properties": {
        "execute": {"type": "boolean", "default": False},
        "stop_after_first_insertion": {"type": "boolean", "default": False},
        "height_offset_m": {"type": "number", "default": -0.020},
        "straight_insertion": {"type": "boolean", "default": False},
        "use_saved_scene": {"type": "boolean", "default": False},
        "motion_speed_scale": {"type": "number", "minimum": 0.01, "maximum": 1.0},
        # where from: the microwave's settings, or the fridge's (FRIDGE_TAKE)
        "site": {"type": "string", "enum": list(SITES), "default": "microwave"},
    },
}

# The tupperware's perception classes: "food container" is the scene segmenter's class; "food_container" the same
# without its space, for a selection typed by hand (as box_opening's perception_classes)
CONTAINER_CLASSES = ["food_container", "food container"]

# The UI's buttons (the orchestrator labels them with the id): what each runs, with fixed params, and on which tapped
# class it is offered. The grab is offered on the tupperware itself and runs take_out from take_home_joints (wherever
# that viewing pose looks); the fridge site stays a direct take_out goal, not a button.
UI_ACTIONS = {
    "retrieve": ("take_out", "Grab Tupperware", {"execute": True}, CONTAINER_CLASSES),
}
UI_TITLE = {
    "retrieve": "Grab tupperware",
}
UI_MOVING = {
    "retrieve": "Grabbing the tupperware: the arm is moving.",
}
UI_DONE = {
    "retrieve": "Holding the tupperware.",
}
UI_RETRY = {"retrieve": "Grab Tupperware"}
# error codes a UI action offers again (when not "STOPPED ..."): checks that refused before the arm started
UI_RETRYABLE = ("refused", "home_failed", "recording")


def _site_words(text, site):
    """take_out's messages name the oven; on the fridge they name the fridge; for the UI's one grab (site "any",
    wherever the tupperware is) they name no place."""
    if site == "fridge":
        return text.replace("microwave", "fridge").replace("the oven", "the fridge").replace("oven", "fridge")
    if site == "any":
        for place in ("in the oven", "in the microwave"):
            text = text.replace(place, "at the tupperware")
        for place in ("the oven", "the microwave"):
            text = text.replace(place, "the tupperware's spot")
    return text


def _hand_note(action_id, phase, site="microwave"):
    """What the hand holds after a stop or failure in `phase` (the last ctx.progress phase of the run)."""
    where = "fridge" if site == "fridge" else "microwave"
    if action_id == "take_out" and site == "any":  # the UI's one grab: wherever the tupperware is
        notes = {
            "reach": "The open hand is at the tupperware; nothing is held.",
            "close": "The gripper was closing on the tupperware: it may be holding it.",
            "lift": "The hand is holding the tupperware where it stood.",
            "retract": "The hand is holding the tupperware, part way back from where it stood.",
        }
        return notes.get(phase, "Nothing has been grasped.")
    if action_id == "take_out":
        notes = {
            "reach": f"The open hand is in the {where}; nothing is held.",
            "close": f"The gripper was closing on the container in the {where}: it may be holding it.",
            "lift": f"The hand is holding the container in the {where}.",
            "retract": f"The hand is holding the container, part way out of the {where}.",
        }
        return notes.get(phase, "Nothing has been grasped.")
    notes = {
        "approach": "The hand is holding the container, at or in the microwave's opening.",
        "lower": "The hand is holding the container in the microwave.",
        "open": "The container is on the turntable; the gripper may be open, still in the microwave.",
        "tilt": "The gripper is open in the microwave; the container is on the turntable.",
        "back-out": "The container is in the microwave; the open hand is part way out.",
    }
    return notes.get(phase, "The hand is still holding the container.")


@contextmanager
def _started(what):
    """Around the part of an action that runs the arm: any refusal from it reads "STOPPED ..." (the arm may have
    moved, so a UI action never offers it again), whatever its own wording (e.g. a check after the approach)."""
    try:
        yield
    except place_motion.Refused as e:
        if str(e).startswith("STOPPED"):
            raise
        raise place_motion.Refused(f"STOPPED during the {what}: {e}") from e


class _PhaseCtx:
    """The action's ctx, remembering the last progress phase (to say what the hand holds after a stop)."""

    def __init__(self, ctx):
        self._ctx = ctx
        self.phase = ""

    def progress(self, phase, *args, **kwargs):
        self.phase = phase
        return self._ctx.progress(phase, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _json_default(o):
    if isinstance(o, np.ndarray | np.generic):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


class Capability(CapabilityEngine):
    display_name = "Grab from microwave"
    # A tupperware tap offers retrieve; placement has no UI action in this engine.
    perception_classes = list(CONTAINER_CLASSES)
    required_resources = ["arm", "gripper"]
    actions = [
        Action("place", "Place the held container in the open microwave", params_schema=PLACE),
        Action("compare_alignment", "Compare wrist and scene microwave alignment without motion", params_schema=EXECUTE),
        Action("diagnose_grasp", "Diagnose saved grasp without moving", params_schema=EXECUTE),
        Action("finish_grasp", "Close, lift 2 cm, and retract from saved grasp", params_schema=EXECUTE),
        Action("move_to_grasp", "Continue to saved grasp point, leaving gripper open", params_schema=EXECUTE),
        Action("take_out", "Take the container out of the open microwave (or the fridge)", params_schema=TAKE_OUT),
        *(
            Action(ui_id, label, perception_classes=list(classes))
            for ui_id, (_, label, _, classes) in UI_ACTIONS.items()
        ),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("microwave_grab", **node_kwargs)
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from sensor_msgs.msg import Image

        # Advertise debug images while idle; retain the last result for Foxglove.
        self.sam3_overlay = self.create_publisher(
            Image, rim_detection.OVERLAY_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        self.cfg["gains"] = {
            "kq": [float(v) for v in self.cfg["impedance_kq"]],
            "zeta": float(self.cfg["impedance_zeta"]),
            "torque_limit": [float(v) for v in self.cfg["impedance_torque_limit"]],
        }
        self.cfg["take_gains"] = {
            "kq": [float(v) for v in self.cfg["take_impedance_kq"]],
            "zeta": float(self.cfg["take_impedance_zeta"]),
            "torque_limit": [float(v) for v in self.cfg["impedance_torque_limit"]],
        }
        self.state_dir = Path(self.cfg["state_dir"])
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def execute(self, action_id, params, ctx):
        if action_id in UI_ACTIONS:
            return self._ui_run(action_id, ctx)
        if not self.cfg["record_bag"]:
            return self._execute_behavior(action_id, params, ctx)
        # Lazy imports keep the pure planning helpers usable without ROS bag tools.
        try:
            from .bag import RecordingError, RunBag
        except ImportError as exc:
            raise EngineError("recording", f"rosbag dependencies unavailable: {exc}", recoverable=True) from exc
        c = self.cfg
        camera = (
            (c["wrist_topics"][0], c["wrist_topics"][2], c["wrist_topics"][1])
            if action_id == "take_out"
            else (microwave_detection.SAM3_IMAGE_TOPIC, c["camera_info_topic"], c["depth_topic"])
        )
        try:
            recorder = RunBag(
                self,
                self.state_dir,
                action_id,
                params,
                json.loads(json.dumps(c, default=_json_default)),
                camera,
                frame_hz=c["bag_frame_hz"],
                state_hz=c["bag_state_hz"],
            )
        except RecordingError as exc:
            raise EngineError("recording", str(exc), recoverable=True) from exc
        with recorder:
            self.get_logger().info(f"run bag: {recorder.path}")
            result = self._execute_behavior(action_id, params, ctx)
            recorder.result = json.loads(json.dumps(result, default=_json_default))
        return {**result, "bag_path": str(recorder.path), "recording_errors": list(recorder.errors)}

    def _ui_run(self, ui_id, ctx):
        """One UI press runs the whole action (UI_ACTIONS) with its fixed params, under capability_ui.RunUI: a progress
        panel with Stop while it runs (Stop cancels this run: the motion in flight stops), then Done, failed or
        Stopped (saying what the hand holds). A recoverable refusal before the arm started (UI_RETRYABLE, not
        "STOPPED ...") shows its reason and offers the action again, waiting for an explicit tap; anything after the
        arm started is shown with Done only, never retried. Every final notice is shown and waited on (bounded) before
        this returns or raises."""
        action_id, _, params, _ = UI_ACTIONS[ui_id]
        # the UI's grab is one action wherever the tupperware is: its notes name no place ("any"); place names the oven
        site = params.get("site", "any" if action_id == "take_out" else "microwave")
        title = UI_TITLE[ui_id]
        ui = RunUI(ctx, title, cancel=lambda: cancel_own_run(self), logger=self.get_logger())
        if not self.mock and not self.cfg["allow_execute"]:
            message = f"{title} is disabled: launch with allow_execute:=true for an attended run."
            ui.refuse(ui_text.disabled(title))
            raise EngineError("execution_disabled", message, recoverable=True)
        if site == "fridge" and not self.mock and not self.cfg["fridge_accept_unmodeled_door"]:
            message = f"{title} is disabled: {FRIDGE_DOOR_GAP}."
            ui.refuse(ui_text.fridge_unmodeled(title))
            raise EngineError("fridge_door_unmodeled", message, recoverable=True)
        while True:
            ctx.checkpoint()
            run_ctx = _PhaseCtx(ctx)
            try:
                result = ui.motion(
                    UI_MOVING[ui_id],
                    lambda run_ctx=run_ctx: self.execute(action_id, dict(params), run_ctx),
                    stopped_note=lambda run_ctx=run_ctx: _hand_note(action_id, run_ctx.phase, site),
                )
            except EngineError as error:
                self.get_logger().error(f"{ui_id}: {error}")
                if ui.user_stopped or ctx.cancelled():
                    raise  # Stop showed its own notice; an orchestrator cancel has taken the UI back
                message = _site_words(str(error), site)
                hand = _hand_note(action_id, run_ctx.phase, site)
                if error.recoverable and error.error_code in UI_RETRYABLE and not message.startswith("STOPPED"):
                    if ui.retry(ui_text.retry(str(error), site), UI_RETRY[ui_id], title=f"{title}: not done"):
                        continue
                    raise  # no UI session, no answer in time
                ui.error(ui_text.failure(str(error), site, hand))
                raise
            except Exception as error:  # Cancelled, an arm error (e-stop, control taken), a bug
                self.get_logger().error(f"{ui_id}: {error!r}")
                if not (ui.user_stopped or ctx.cancelled()):
                    ui.error(ui_text.unexpected(error, _hand_note(action_id, run_ctx.phase, site)))
                raise
            ui.done(UI_DONE[ui_id], mock=bool(result.get("mock")))
            return result

    def availability(self, instance_id, world_state):
        """Offer retrieve on a tupperware tap. `place` / `take_out` remain direct actions only."""
        return [a for a in super().availability(instance_id, world_state) if a.action_id in UI_ACTIONS]

    def _execute_behavior(self, action_id, params, ctx):
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing planned or moved")
            return {"executed": False, "mock": True}
        log = self.get_logger().info
        site = params.get("site", "microwave") if action_id == "take_out" else "microwave"
        if site not in SITES:
            raise EngineError("bad_params", f"'site' must be one of {', '.join(SITES)}, not {site!r}", recoverable=True)
        try:
            model = place_motion.Model(self.cfg["flange_to_tip"])
            if action_id == "compare_alignment":
                return self._compare_alignment(ctx, log)
            if action_id == "diagnose_grasp":
                return self._diagnose_grasp(model, ctx, log)
            if action_id == "finish_grasp":
                return self._finish_grasp(model, execute, ctx, log)
            if action_id == "move_to_grasp":
                return self._move_to_grasp(model, execute, ctx, log)
            if action_id == "take_out":
                if site == "microwave":
                    with self._grab_collision_world(ctx, log, bool(params.get("use_saved_scene", False))) as scene:
                        checker = place_motion.Checker(model, scene["metadata"]["obstacles"])
                        try:

                            def report_clearance(record):
                                detail = checker.closest_points(record["joints"], record["gripper_closed"])
                                record = {
                                    **record,
                                    "closest_points": detail,
                                    "obstacles": scene["metadata"]["obstacles"],
                                }
                                self._save("last_clearance.json", record)
                                if record["distance_mm"] < record["required_mm"]:
                                    self._save("last_clearance_refusal.json", record)
                                log_message = (
                                    f"{record['phase']}: minimum checked clearance "
                                    f"{record['distance_mm']:.3f} mm, required {record['required_mm']:.3f} mm "
                                    f"({record['pair']})"
                                )
                                self.get_logger().info(log_message)

                            scene["check_path"] = lambda phase, qs: collision_world.check_local_path(
                                checker,
                                qs,
                                place_motion.dense,
                                place_motion.LINE_STEP_DEG,
                                self.cfg["take_clear_min"],
                                ctx.checkpoint,
                                phase,
                                report_clearance,
                            )
                            return {**self._take_out(model, execute, ctx, log, site, scene, params.get("stop_after_first_insertion", False), params.get("height_offset_m", -0.020), motion_params=params), "note": note}
                        finally:
                            checker.close()
                return {**self._take_out(model, execute, ctx, log, site, stop_after_first_insertion=params.get("stop_after_first_insertion", False), height_offset_m=params.get("height_offset_m", -0.020), motion_params=params), "note": note}
            return {**self._place(model, execute, ctx, log), "note": note}
        except (
            place_motion.Refused,
            microwave_detection.NotLocated,
            rim_detection.NotDetected,
            collision_world.InvalidGeometry,
        ) as e:
            raise EngineError("refused", _site_words(str(e), site), recoverable=True) from e




    def _compare_alignment(self, ctx, log):
        import cv2
        c = self.cfg
        results, shots = {}, {}
        for name, color, depth, info in [
            ("scene", microwave_detection.SAM3_IMAGE_TOPIC, c["depth_topic"], c["camera_info_topic"]),
            ("wrist", c["wrist_topics"][0], c["wrist_topics"][1], c["wrist_topics"][2]),
        ]:
            ctx.checkpoint()
            ctx.progress("compare_"+name, 0.0)
            camera = microwave_detection.SceneCamera(self, rammp_arm.FRAME, (depth, info),
                                                     sam3_image_topic=color)
            try:
                located = microwave_detection.locate(
                    self, rammp_arm.FRAME, (depth, info), 3, 30.0, c["agree_m"], c["agree_deg"],
                    log, camera=camera)
            except microwave_detection.NotLocated as error:
                results[name] = {"success": False, "error": str(error)}
                continue
            shot = located["scene"]
            shots[name] = shot
            estimate = {key: value for key, value in located.items() if key not in ("scene", "body")}
            results[name] = {"success": True, "estimate": estimate,
                             "yaw_deg": float(np.degrees(np.arctan2(located["insertion_axis"][1], located["insertion_axis"][0]))),
                             "world_T_cam": shot["world_T_cam"].tolist(),
                             "stamp": shot["stamp"], "image_topic": color}
            directory = self.state_dir / "camera_alignment"
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory / f"{name}.npz", depth_m=shot["depth_m"], K=shot["K"],
                world_T_cam=shot["world_T_cam"], mask=shot["mask"])
        result = {"executed": False, "frame": rammp_arm.FRAME, "cameras": results}
        if all(results.get(name, {}).get("success") for name in ("scene", "wrist")):
            a, b = results["scene"], results["wrist"]
            diff = (a["yaw_deg"] - b["yaw_deg"] + 180) % 360 - 180
            corners_a = np.asarray(a["estimate"]["fit"]["opening_corners"])
            corners_b = np.asarray(b["estimate"]["fit"]["opening_corners"])
            result["comparison"] = {
                "yaw_difference_deg": float(diff),
                "point_difference_m": (np.asarray(a["estimate"]["point"])-np.asarray(b["estimate"]["point"])).tolist(),
                "point_distance_m": float(np.linalg.norm(np.asarray(a["estimate"]["point"])-np.asarray(b["estimate"]["point"]))),
                "opening_corner_distances_m": np.linalg.norm(corners_a-corners_b,axis=1).tolist(),
                "floor_difference_m": a["estimate"]["floor_z"]-b["estimate"]["floor_z"]}
        for name, shot in shots.items():
            depth = np.nan_to_num(shot["depth_m"])
            image = cv2.applyColorMap(np.uint8(np.clip(depth / 2.0,0,1)*255),cv2.COLORMAP_TURBO)
            image[depth<=0] = 0
            contours,_ = cv2.findContours(shot["mask"].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image,contours,-1,(255,255,255),1)
            for other in shots:
                corners = np.asarray(results[other]["estimate"]["fit"]["opening_corners"])
                camera_points = (np.linalg.inv(shot["world_T_cam"]) @ np.c_[corners,np.ones(len(corners))].T).T[:,:3]
                pixel = (shot["K"] @ camera_points.T).T
                pixel = np.round(pixel[:,:2]/pixel[:,2:3]).astype(np.int32)
                color = (0,255,0) if other==name else (255,0,255)
                cv2.polylines(image,[pixel],True,color,2)
            cv2.putText(image,name+" | own fit: green | other camera: magenta",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
            cv2.imwrite(str(self.state_dir/"camera_alignment"/f"{name}.png"),image)
        record = json.loads((self.state_dir/"grasp_reached_scene.json").read_text())
        saved = json.loads(Path(record["world_file"]).with_suffix(".json").read_text())
        corners = np.asarray(saved["opening_corners"])
        edge = corners[1]-corners[0]
        normal = np.cross(edge,[0,0,1])
        saved_yaw = float(np.degrees(np.arctan2(normal[1],normal[0])))
        result["saved_scene_yaw_deg_mod_180"] = saved_yaw % 180
        self._save("camera_alignment/comparison.json", result)
        return result

    def _diagnose_grasp(self, model, ctx, log):
        saved = json.loads((self.state_dir / "plans/first_insertion_latest.json").read_text())
        summary, c = saved["summary"], self.cfg
        q = np.asarray(self.arm.raw_joints(), dtype=float)
        T = model.fk(q)
        target = np.asarray(summary["grasp"])
        cand = take_out.grasp_pose(model, np.asarray(saved["q0"]), summary["container_point"],
                                  c["reach_in"], c["grasp_depth"], summary["grasp_up_m"])
        result = dict(executed=False, current=T[:3, 3].tolist(), target=target.tolist(),
                      delta=(target-T[:3, 3]).tolist(), current_joints=q.tolist())
        try:
            line = place_motion.plan_line(model, q, target-T[:3, 3], "diagnostic straight insertion",
                                          step=place_motion.LINE_STEP)
        except place_motion.Refused as error:
            result["straight_ik_error"] = str(error)
            return result
        result["straight_ik_samples"] = len(line)
        result["endpoint_joints"] = np.asarray(line[-1]).tolist()
        result["endpoint"] = model.fk(line[-1])[:3, 3].tolist()
        result["joint_change_deg"] = place_motion.wrap_deg(np.asarray(line[-1])-q).tolist()
        with self._grab_collision_world(ctx, log) as scene:
            result["scene"] = scene["metadata"]
            checker = place_motion.Checker(model, scene["metadata"]["obstacles"])
            try:
                records = []
                try:
                    collision_world.check_local_path(checker, line, place_motion.dense,
                        place_motion.LINE_STEP_DEG, c["take_clear_min"], ctx.checkpoint,
                        "diagnostic straight insertion", records.append)
                except collision_world.InvalidGeometry as error:
                    result["straight_collision_error"] = str(error)
                result["clearance"] = records
                result["endpoint_closest"] = checker.closest_points(line[-1], False)
            finally:
                checker.close()
            joint_plan = self.arm.plan_to_joints(line[-1], q)
            result["curobo_joint_goal"] = dict(success=joint_plan.success, message=joint_plan.message)
            pose_plan = self.arm.plan_to_pose(target, cand["quat"], q)
            result["curobo_free_pose_goal"] = dict(success=pose_plan.success, message=pose_plan.message)
        self._save("grasp_diagnostics.json", result)
        log(f"grasp diagnostics: {json.dumps(result, default=_json_default)}")
        return result


    def _finish_grasp(self, model, execute, ctx, log):
        """Continue the staged grasp without homing or repeating insertion."""
        import yaml
        saved = json.loads((self.state_dir / "plans/first_insertion_latest.json").read_text())
        summary, c = saved["summary"], dict(self.cfg)
        q = np.asarray(self.arm.raw_joints(), dtype=float)
        start = model.fk(q)[:3, 3]
        if np.linalg.norm(start - summary["grasp"]) > place_motion.END_TOL:
            raise place_motion.Refused("arm is not at the saved grasp point; not closing")
        if abs(c["take_lift"] - 0.02) > 1e-9:
            raise place_motion.Refused("finish_grasp requires configured 2 cm lift")
        def move(qs, name):
            qs = np.asarray(qs)
            distances = np.asarray(place_motion.pose_lengths(model, qs))
            length = float(np.sum(distances))
            if length < 0.001:
                return
            duration = 1.875 * length / 0.16
            fractions = np.r_[0.0, np.cumsum(distances)] / length
            times = np.array([place_motion.minjerk_time(float(f)) for f in fractions]) * duration
            times[0], times[-1] = 0.0, duration
            velocities = np.gradient(qs, times, axis=0)
            velocities[0] = velocities[-1] = 0.0
            ctx.checkpoint()
            result = self.arm.execute_joint_trajectory(take_out.rammp_arm.JOINTS,
                zip(times, qs, velocities, strict=True), speed_scale=0.1)
            if result.error_code != place_motion.SUCCESSFUL:
                raise place_motion.Refused(f"{name} stopped: {result.error_string}")
            endpoint = model.fk(np.asarray(self.arm.raw_joints(), dtype=float))[:3, 3]
            if np.linalg.norm(endpoint-model.fk(qs[-1])[:3, 3]) > place_motion.END_TOL:
                raise place_motion.Refused(f"{name} endpoint error; holding position")
        record = json.loads((self.state_dir / "grasp_reached_scene.json").read_text())
        scene_path = Path(record["world_file"])
        scene = {"metadata": {"obstacles": yaml.safe_load(scene_path.read_text())["obstacles"]}}
        expected_layers = collision_world.planner_layers(c["planner_config"])
        with collision_world.loaded_world(self.arm, scene_path, c["take_approach_world"],
                lambda arm, world, logger: take_out.set_approach_world(arm, world, logger, expected_layers),
                log):
            config_path = Path(c["planner_config"])
            config = yaml.safe_load(config_path.read_text())
            obstacles = list(scene["metadata"]["obstacles"])
            for layer in list(config["layers"]) + ["layer_mebot_floor.yaml"]:
                layer_path = Path(layer)
                if not layer_path.is_absolute():
                    layer_path = config_path.parent / layer_path
                obstacles.extend(yaml.safe_load(layer_path.read_text())["obstacles"])
            padded = [{**box, "rpy_deg": box.get("rpy_deg", [0, 0, 0]),
                       "dims": (np.asarray(box["dims"]) + 0.04).tolist()} for box in obstacles]
            checker = place_motion.Checker(model, padded)
            try:
                c["collision_check"] = lambda phase, qs: collision_world.check_local_path(
                    checker, qs, place_motion.dense, place_motion.LINE_STEP_DEG,
                    c["take_clear_min"], ctx.checkpoint, phase,
                    lambda record: log(f"{phase} clearance: {record}"))
                lift = place_motion.plan_line(model, q, [0, 0, c["take_lift"]], "2 cm lift",
                                             step=place_motion.LINE_STEP)
                take_out.check_local(c, "2 cm lift", lift)
                take_out.plan_withdrawal(self.arm, model, lift[-1], summary["pre_grasp"], c)
                if not execute:
                    return {"executed": False, "lift_m": c["take_lift"], "speed_scale": 0.1}
                ctx.checkpoint()
                ctx.progress("close", 0.0)
                pos = take_out.close_on_brim(self.arm, log)
                ctx.checkpoint()
                q = np.asarray(self.arm.raw_joints(), dtype=float)
                lift = place_motion.plan_line(model, q, [0, 0, c["take_lift"]], "2 cm lift",
                                             step=place_motion.LINE_STEP)
                take_out.check_local(c, "measured-start 2 cm lift", lift)
                take_out.plan_withdrawal(self.arm, model, lift[-1], summary["pre_grasp"], c)
                ctx.progress("lift", 0.0)
                move(lift, "2 cm lift")
                ctx.checkpoint()
                q = np.asarray(self.arm.raw_joints(), dtype=float)
                route = take_out.plan_withdrawal(self.arm, model, q, summary["pre_grasp"], c)
                ctx.progress("retract", 0.0)
                move(route["out"], "straight withdrawal")
                actual = model.fk(np.asarray(self.arm.raw_joints(), dtype=float))[:3, 3]
                return {"executed": True, "lift_m": c["take_lift"], "gripper": pos,
                        "stopped_at": "retracted_holding_container", "actual": actual.tolist()}
            finally:
                checker.close()

    def _move_to_grasp(self, model, execute, ctx, log):
        """Continue only the last saved first insertion to its grasp; never close."""
        saved = json.loads((self.state_dir / "plans/first_insertion_latest.json").read_text())
        summary, c = saved["summary"], self.cfg
        cand = take_out.grasp_pose(model, np.asarray(saved["q0"]), summary["container_point"],
                                  c["reach_in"], c["grasp_depth"], summary["grasp_up_m"])
        target = np.asarray(summary["grasp"], dtype=float)
        q = np.asarray(self.arm.raw_joints(), dtype=float)
        current = model.fk(q)
        if np.linalg.norm(current[:3, 3] - summary["first_insertion_target"]) > place_motion.END_TOL:
            raise place_motion.Refused("arm is not at the saved first insertion endpoint")
        with self._grab_collision_world(ctx, log) as scene:
            from types import SimpleNamespace
            import yaml

            # Explicit Cartesian insertion requested by the operator, solved from measured joints.
            qs = np.asarray(place_motion.plan_line(model, q, target-current[:3, 3],
                            "straight grasp insertion", step=place_motion.LINE_STEP))
            place_motion.check_line(model, qs, current[:3, 3], target, None, "straight grasp insertion")
            lengths = np.asarray(place_motion.pose_lengths(model, qs), dtype=float)
            distance = float(np.sum(lengths))
            if distance < 0.001:
                raise place_motion.Refused("already at grasp; no insertion required")
            fractions = np.r_[0.0, np.cumsum(lengths)] / distance
            duration = 1.875 * distance / 0.16
            times = np.array([place_motion.minjerk_time(float(f)) for f in fractions]) * duration
            times[0], times[-1] = 0.0, duration
            velocities = np.gradient(qs, times, axis=0)
            velocities[0] = velocities[-1] = 0.0
            plan = SimpleNamespace(q=qs, t=times, qd=velocities)
            reference = take_out.Rotation.from_quat(cand["quat"])
            for sample in place_motion.dense(plan.q, place_motion.LINE_STEP_DEG):
                actual = take_out.Rotation.from_matrix(model.fk(sample)[:3, :3])
                deviation = float(np.degrees((reference.inv() * actual).magnitude()))
                if deviation > take_out.INSERT_ORIENTATION_TOL_DEG:
                    raise place_motion.Refused(f"grasp continuation orientation deviation {deviation:.3f} deg")
            # Include persistent chair/person/floor layers and the planner's 20 mm padding.
            config_path = Path(c["planner_config"])
            config = yaml.safe_load(config_path.read_text())
            obstacles = list(scene["metadata"]["obstacles"])
            for layer in list(config["layers"]) + ["layer_mebot_floor.yaml"]:
                layer_path = Path(layer)
                if not layer_path.is_absolute():
                    layer_path = config_path.parent / layer_path
                obstacles.extend(yaml.safe_load(layer_path.read_text())["obstacles"])
            padded = [{**box, "rpy_deg": box.get("rpy_deg", [0.0, 0.0, 0.0]), "dims": (np.asarray(box["dims"]) + 0.040).tolist()} for box in obstacles]
            checker = place_motion.Checker(model, padded)
            try:
                collision_world.check_local_path(checker, plan.q, place_motion.dense,
                    place_motion.LINE_STEP_DEG, c["take_clear_min"], ctx.checkpoint,
                    "grasp continuation", lambda record: log(f"grasp continuation clearance: {record}"))
            finally:
                checker.close()
            endpoint = model.fk(np.asarray(plan.q[-1]))[:3, 3]
            if np.linalg.norm(endpoint - target) > place_motion.END_TOL:
                raise place_motion.Refused("grasp continuation plan misses the target")
            if not execute:
                return {"executed": False, "target": target.tolist(), "speed_scale": 0.1}
            ctx.checkpoint()
            ctx.progress("reach", 0.0)
            log("-> grasp point only: impedance, 10% maximum speed scale; gripper unchanged")
            try:
                take_out.run_insertion_plan(self.arm, model, plan, c["impedance_speed"],
                                           0.1, "grasp continuation", gains=c["take_gains"])
            finally:
                place_motion.exit_impedance(self.arm, log)
            actual = model.fk(np.asarray(self.arm.raw_joints(), dtype=float))[:3, 3]
            error = float(np.linalg.norm(actual - target))
            if error > place_motion.END_TOL:
                raise place_motion.Refused(f"stopped with gripper open: grasp endpoint error {error * 100:.2f} cm")
            ctx.progress("grasp_point_open", 100.0)
            return {"executed": True, "stopped_at": "grasp_point", "gripper_unchanged": True,
                    "speed_scale_max": 0.1, "target": target.tolist(),
                    "actual": actual.tolist(), "position_error_m": error}

    def _place(self, model, execute, ctx, log):
        c = self.cfg
        q0 = np.asarray(self.arm.raw_joints(), dtype=float)
        oven, loc, boxes, world, world_info, stamp = self._oven_world(ctx, log)
        checker = place_motion.Checker(model, boxes)
        try:
            # 4-5. the first heading that passes every stage
            ctx.progress("plan", 0.0)
            cand, plan, res, attempts = place_motion.search_place(self.arm, model, checker, q0, oven, c, log)
            summary, record = place_motion.summarize(
                model, q0, loc, oven, world, world_info, boxes, cand, plan, res, attempts, c
            )
            path = self._save_plan(record, stamp)
            summary["plan_file"] = str(path)
            log(
                f"plan: side shift {cand['shift_cm']:+.0f} cm, heading {cand['offset_deg']:+.0f} deg, clearance "
                + ", ".join(f"{k} {v[0]:.1f} cm" for k, v in summary["clearance_cm"].items())
                + f", tilt {summary['max_tilt_deg']:.1f} deg, approach (shift + transit + line in "
                f"{cand['insert_m'] * 100:.0f} cm) in {summary['duration_s']['approach']:.1f} s -> {path}"
            )
            if not execute:
                log("DRY RUN -- all checks passed, nothing commanded")
                return {"executed": False, "plan": summary}
            # 6. execute
            with _started("place"):
                result = place_motion.run_place(
                    self.arm,
                    model,
                    q0,
                    oven,
                    cand,
                    plan,
                    c,
                    ctx,
                    log,
                    self.state_dir / "descents",
                    res,
                )
            return {**result, "plan": summary}
        finally:
            checker.close()

    def _take_out(self, model, execute, ctx, log, site="microwave", scene=None, stop_after_first_insertion=False, height_offset_m=-0.020, motion_params=None):
        """Find the nearest container-mask point, approach, insert, close, lift and retract. `site` "fridge": with the
        fridge's settings (FRIDGE_TAKE), and executing only if fridge_accept_unmodeled_door (FRIDGE_DOOR_GAP)."""
        c = self.cfg
        if site != "microwave":
            c = {**self.cfg, **{key: self.cfg[name] for key, name in FRIDGE_TAKE.items()}}
            if execute and not self.cfg["fridge_accept_unmodeled_door"]:
                raise place_motion.Refused(f"refused before anything moved: {FRIDGE_DOOR_GAP}")
            log(f"take-out from the fridge settings. WARNING: {FRIDGE_DOOR_GAP.split('. Set')[0]}")
        motion_params = motion_params or {}
        c = {**c, "straight_insertion": bool(motion_params.get("straight_insertion", False))}
        if "motion_speed_scale" in motion_params:
            scale = float(motion_params["motion_speed_scale"])
            if not np.isfinite(scale) or not 0.01 <= scale <= 1.0:
                raise place_motion.Refused("motion_speed_scale must be in [0.01, 1]")
            c.update(motion_speed_scale=scale, take_approach_speed_scale=scale,
                     take_exit_speed_scale=scale, insert_speed=0.16*scale, impedance_speed=0.16*scale)
        ctx.checkpoint()
        try:
            world_message = (
                scene["message"] if scene else take_out.set_approach_world(self.arm, c["take_approach_world"], log)
            )
            q0 = take_out.prepare_home(
                self.arm, c["take_home_joints"], execute, c["take_approach_speed_scale"], ctx, log
            )
        except (place_motion.Refused, ArmRefused) as error:
            raise EngineError("home_failed", _site_words(str(error), site), recoverable=True) from error
        if scene:
            c = {**c, "collision_check": scene["check_path"]}
        ctx.progress("detect", 0.0)
        wrist = rim_detection.WristCamera(self, rammp_arm.FRAME, list(c["wrist_topics"]))
        rim = rim_detection.detect_container(
            wrist,
            c["container_prompt"],
            c["sam3_threshold"],
            int(c["rim_frames"]),
            c["rim_mask_iou"],
            c["rim_timeout_s"],
            log,
            ctx.checkpoint,
        )
        height_offset_m = float(height_offset_m)
        if not np.isfinite(height_offset_m):
            raise place_motion.Refused("height_offset_m must be finite")
        log(f"grasp height: {c['grasp_up']:.3f} m + {height_offset_m:.3f} m offset")
        cand = take_out.grasp_pose(model, q0, rim["point"], c["reach_in"], c["grasp_depth"], c["grasp_up"] + height_offset_m)
        ctx.progress("plan", 0.0)
        ctx.checkpoint()
        c = {**c, "stop_after_first_insertion": bool(stop_after_first_insertion)}
        legs = take_out.plan_take_out(self.arm, model, q0, cand, c)
        if stop_after_first_insertion:
            summary = dict(stopped_at="first_insertion", height_offset_m=height_offset_m, grasp_up_m=c["grasp_up"] + height_offset_m, grasp=np.asarray(cand["target"]).tolist(), container_point=np.round(rim["point"], 3).tolist(),
                           pre_grasp=np.round(cand["pre"], 3).tolist(),
                           first_insertion_target=(np.asarray(cand["target"]) - c["impedance_in"] * np.asarray(cand["heading"])).tolist())
            self._save("plans/first_insertion_latest.json", dict(summary=summary, q0=q0, **{key: value for key, value in legs.items() if key != "insertion_plans"}))
            if not execute:
                return {"executed": False, "plan": summary}
            with _started("first-insertion"):
                result = take_out.run_take_out(self.arm, model, q0, cand, legs, c, ctx, log)
            return {**result, "plan": summary}
        T_pre = model.fk(legs["approach"][-1])
        summary = dict(
            site=site,
            home_joints=list(c["take_home_joints"]),
            approach_world=world_message,
            microwave_collision_model=scene["metadata"] if scene else None,
            approach_collision_checked=True,
            insertion_collision_checked=scene is not None,
            lift_collision_checked=scene is not None,
            exit_collision_checked=scene is not None,
            local_collision_checker="PyBullet arm/gripper vs microwave boxes" if scene else None,
            held_container_collision_checked=False,
            exit_speed_scale=float(legs["out_speed_scale"]),
            exit_duration_s=round(
                float(legs["out_t"][-1]) / legs["out_speed_scale"]
                + float(legs["pullback"]["out_t"][-1]) / legs["pullback"]["out_speed_scale"],
                2,
            ),
            pullback_m=0.0,
            pullback_target=legs["pullback"]["out_target"].tolist(),
            approach_duration_s=round(float(legs["approach_t"][-1]) / c["take_approach_speed_scale"], 2),
            container_point=np.round(rim["point"], 3).tolist(),
            container_depths_m=rim["info"]["depths_m"],
            container_frame_points=rim["info"]["frame_points"],
            container_frames_spread_cm=round(rim["info"]["frames_spread_m"] * 100, 1),
            container_mask_iou_min=round(rim["iou_min"], 3),
            start_tool=np.round(model.fk(q0)[:3, 3], 3).tolist(),
            pre_grasp=np.round(cand["pre"], 3).tolist(),
            grasp=np.round(cand["target"], 3).tolist(),
            heading=np.round(cand["heading"], 3).tolist(),
            fingers_line_from_up_deg=round(float(np.degrees(np.arccos(np.clip(T_pre[2, 0], -1.0, 1.0)))), 1),
            hand_pitch_deg=round(float(np.degrees(np.arcsin(-T_pre[2, 2]))), 1),
            joint_change_deg=dict(
                approach=np.round(
                    np.degrees(place_motion.joint_delta(legs["approach"][0], legs["approach"][-1])), 1
                ).tolist(),
                line_in=np.round(np.degrees(place_motion.joint_delta(legs["line"][0], legs["line"][-1])), 1).tolist(),
            ),
            end_off_cm=round(legs["end"][0] * 100, 1),
            lift_cm=round(c["take_lift"] * 100, 1),
            out_end=np.round(model.fk(legs["out"][-1])[:3, 3], 3).tolist(),
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = "take_out" if site == "microwave" else f"take_out_{site}"
        self._save(f"plans/{name}_{stamp}.json", dict(summary=summary, q0=q0, **legs))
        log(f"take-out plan: {json.dumps(summary)}")
        if not execute:
            log("DRY RUN -- planned, nothing commanded")
            return {"executed": False, "plan": summary}
        c = {**c, "stop_after_first_insertion": bool(stop_after_first_insertion)}
        with _started("take-out"):
            result = take_out.run_take_out(self.arm, model, q0, cand, legs, c, ctx, log)
        return {**result, "plan": summary}

    @contextmanager
    def _grab_collision_world(self, ctx, log, use_saved_scene=False):
        """Observe the shell before HOME; keep it in cuRobo until retrieval ends."""
        c = self.cfg
        if use_saved_scene:
            record = json.loads((self.state_dir / "grasp_reached_scene.json").read_text())
            path = Path(record["world_file"])
            metadata = json.loads(path.with_suffix(".json").read_text())
            expected = collision_world.planner_layers(c["planner_config"])
            log(f"using verified static microwave scene: {path}")
            with collision_world.loaded_world(self.arm, path, c["take_approach_world"],
                    lambda arm, world, logger: take_out.set_approach_world(arm, world, logger, expected),
                    log) as message:
                yield dict(message=message, metadata=metadata)
            return
        ctx.checkpoint()
        ctx.progress("microwave_model", 0.0)
        try:
            expected_layers = collision_world.planner_layers(c["planner_config"])
            door = collision_world.read_door(c["door_geometry_file"])
            topics = (c["depth_topic"], c["camera_info_topic"], c["detections_topic"])
            loc = microwave_detection.locate(
                self,
                rammp_arm.FRAME,
                topics,
                int(c["frames"]),
                c["timeout_s"],
                c["agree_m"],
                c["agree_deg"],
                log,
                checkpoint=ctx.checkpoint,
            )
            ctx.checkpoint()
            oven = place_motion.Oven(loc["fit"])
            outer = collision_world.fixed_outer_bounds(
                microwave_detection.OPENING,
                c["microwave_panel_width_m"],
                c["microwave_border_m"],
                c["microwave_wall_m"],
            )
            boxes = collision_world.shell_boxes(
                oven,
                outer,
                microwave_detection.OPENING,
                microwave_detection.DEPTH,
                microwave_detection.RECESS,
                c["microwave_wall_m"],
            )
            boxes.append(
                collision_world.door_box(
                    oven,
                    door,
                    c["door_open_angle_deg"],
                    c["door_alignment_m"],
                    c["door_alignment_deg"],
                    microwave_detection.OPENING,
                    outer,
                    microwave_detection.RECESS,
                )
            )
            metadata = dict(
                outer_bounds=outer,
                opening_corners=loc["fit"]["opening_corners"],
                door_file=c["door_geometry_file"],
                door_angle_deg=c["door_open_angle_deg"],
                door_angle_source="configured assumption",
                door_registration="door right surface aligned to cavity left edge at 90 degrees",
                exterior_source="configured dimensions: right panel, top/bottom border, left wall",
                panel_width_m=c["microwave_panel_width_m"],
                border_m=c["microwave_border_m"],
                home_source="measured successful tupperware-detection pose, 2026-11-18",
                expected_config_layers=expected_layers,
                obstacles=boxes,
            )
            path = collision_world.write_scene(
                c["worlds_dir"], rammp_arm.FRAME, c["microwave_base_layers"], boxes, metadata
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise collision_world.InvalidGeometry(f"microwave collision model unavailable: {error}") from error
        metadata["world_file"] = str(path)
        log(f"microwave collision world: {path} (shell, control panel, left door)")
        ctx.checkpoint()
        with collision_world.loaded_world(
            self.arm,
            path,
            c["take_approach_world"],
            lambda arm, world, logger: take_out.set_approach_world(arm, world, logger, expected_layers),
            log,
        ) as message:
            yield dict(message=message, metadata=metadata)

    def _oven_world(self, ctx, log):
        """locate the oven (scene camera), its one obstacle box, cuRobo's world set: (oven, loc, boxes, world file,
        world info, stamp)."""
        c = self.cfg

        # 2. locate
        ctx.progress("locate", 0.0)
        topics = (c["depth_topic"], c["camera_info_topic"], c["detections_topic"])
        camera = microwave_detection.SceneCamera(self, rammp_arm.FRAME, topics)
        loc = microwave_detection.locate(
            self,
            rammp_arm.FRAME,
            topics,
            int(c["frames"]),
            c["timeout_s"],
            c["agree_m"],
            c["agree_deg"],
            log,
            camera=camera,
            extra_prompt=c["microwave_prompt"],
        )
        oven = place_motion.Oven(loc["fit"])
        log(
            f"microwave: point {np.round(oven.point, 3).tolist()}, axis {np.round(oven.n[:2], 3).tolist()}, glass z "
            f"{oven.floor_z:.3f} (the frame nearest the median of {loc['frames']} of {loc['detected']}, "
            f"within {loc['spread_m'] * 100:.1f} cm, {loc['spread_deg']:.1f} deg)"
        )

        # 3. obstacles
        ctx.progress("obstacles", 0.0)
        shot = loc["body"]  # the locate's frame of SAM3's "microwave" (the oven's body)
        boxes, curobo_boxes, world_info = place_motion.microwave_obstacles(
            oven, shot["depth_m"], shot["K"], shot["world_T_cam"], shot["mask"]
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        world = self._write_world(curobo_boxes, stamp)
        ok, msg = self.arm.set_world(str(world))
        log(
            f"world: the microwave as one box from {world_info['points']} depth points, in oven coordinates (cm) "
            f"along {world_info['along_cm']}, across {world_info['across_cm']}, up {world_info['up_cm']} -> {world}"
        )
        if not ok:
            raise place_motion.Refused(f"cuRobo's set_world({world}) failed: {msg}")

        return oven, loc, boxes, world, world_info, stamp

    # -- state files -----------------------------------------------------------------------------

    def _write_world(self, boxes, stamp):
        """cuRobo's world file in worlds_dir (the planner reads it at this same absolute path)."""
        directory = Path(self.cfg["worlds_dir"])
        if not directory.is_absolute():
            raise place_motion.Refused(
                f"worlds_dir {directory} is not an absolute path: the planner reads the file by its path"
            )
        path = directory / f"microwave_grab_{stamp}.yaml"
        try:
            place_motion.write_world(path, boxes, rammp_arm.FRAME)
        except OSError as e:
            raise place_motion.Refused(
                f"can't write cuRobo's world {path} ({e}) -- set worlds_dir, mounted in both containers"
            ) from e
        return path

    def _save_plan(self, record, stamp):
        path = self.state_dir / "plans" / f"plan_{stamp}.json"
        self._save(f"plans/plan_{stamp}.json", record)
        self._save("last_plan.json", record)
        return path

    def _save(self, name, record):
        try:
            path = self.state_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1, default=_json_default))
        except OSError as e:
            raise place_motion.Refused(f"can't save {name} in {self.state_dir} ({e}) -- set state_dir") from e


def main():
    run(Capability)


if __name__ == "__main__":
    main()
