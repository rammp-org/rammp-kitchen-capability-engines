"""The fridge capability: open and close the fridge door with the arm.

Action `open` (open_door.run): detect the handle from the wrist camera, grasp it, pull the door 60 deg open, let go and
push it on to 90 (or, with `push_open` false, a 90-deg pull and go back round the door to its inner side). Action
`close` (close_door.run): push the door most of the way shut with the hand -- no grasp, the gripper untouched (it may
hold a container) -- against the door the open saved; it swings shut by itself from there. cuRobo plans every motion
(planning.py) against the fridge it is shown (world.py: the cabinet, the door at its angle, the handle, pushed into the
planner with set_world) and the planner's own wheelchair layers, all of it before anything moves; any failed plan or
check stops the run with the arm holding where it is (EngineError "refused"). The planner's world is put back when the
action ends.

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets `execute`. Otherwise the goal
is a dry run (detect, plan and gate every leg; nothing moves) and the result says so. Someone stands at the e-stop
whenever allow_execute is on.

The wheelchair moves the open and the close plan (back ~30 cm before the close; 5-20 cm before the push-open's push,
when it needs one) are made by hand by default: the goal stops and asks, and a goal with `chair_moved` goes on. With
`drive_base` true an executing goal drives the base that fixed distance itself (base_drive, self.base) at `base_speed`
m/s and goes on in the same goal. A dry run with drive_base only checks the base's odometry is there.

    ros2 launch fridge fridge.launch.py use_orchestrator:=false
    make action CAP=fridge ACTION=open PARAMS='{"execute": false}'
    make action CAP=fridge ACTION=close PARAMS='{"execute": false}'
    make action CAP=fridge ACTION=go_to_start PARAMS='{"execute": false}'

Action `go_to_start`, sent before an open: the arm to the fridge start pose (the joints `start_joints_deg`, the web
app's Fridge pose), cuRobo's collision-free joint-space plan from where the arm is, against the robot's own planner
world (the wheelchair's layers; on the level mount the chair stand-in too) -- the fridge isn't in it: nothing has
detected it yet. The driver runs it (go_to_joint_config). The gripper is untouched. Gated like the others: a dry run
plans only.
"""

import time
from functools import partial

import numpy as np
import rammp_arm
from capability_engine import Action, BaseError, CapabilityEngine, EngineError, run
from door_common import common as dc
from door_common import open_door as shared_open

from . import base_drive, close_door, constants, motion, open_door, push_open
from .kinematics import Chain
from .perception import WristCamera
from .planning import Planner
from .tee_logger import TeeLogger
from .world import PlannerWorld, chair_stand_in

COMMON = {
    "execute": {"type": "boolean", "default": False},
    "speed": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 1.0},
    # the base drives the wheelchair moves the plan needs (fixed distances, base_drive), instead of asking for them
    "drive_base": {"type": "boolean", "default": False},
    # ... at this speed (m/s), at most the node's base_max_speed
    "base_speed": {"type": "number", "exclusiveMinimum": 0, "default": base_drive.SPEED_DEFAULT},
}


def _schema(default):
    """JSON Schema type of a close setting from its default (None: a number or unset)."""
    if isinstance(default, bool):
        return {"type": "boolean", "default": default}
    if default is None:
        return {"type": ["number", "null"], "default": None}
    return {"type": "number", "default": default}


# ROS parameter -> default. Topics, paths and what is measured on the fridge are per robot (the fragment's params:).
PARAMS = {
    "allow_execute": False,
    "data_dir": "/fridge_data",
    # the arm's forward lean in its mount (deg): 0 (dojo) or 30 (flatbot, gen_1_5). Picks constants.MOUNTS
    "mount_tilt_deg": 0,
    "color_topic": "/wrist_camera/color/image_raw",
    "camera_info_topic": "/wrist_camera/color/camera_info",
    # every wrist camera's depth, aligned to colour on the D405 and the OAK; the D435i needs aligned_depth_to_color
    "depth_topic": "/wrist_camera/depth/image_rect_raw",
    "camera_frame": "",
    # which camera finds the handle and measures the door: "wrist" (color_topic, camera_info_topic, depth_topic: what
    # every executed open has used, and what GRIP_EXT was tuned on) or "scene" (the scene_* topics below, as the side
    # grasp measured the door on 11-06; its handle point is not the wrist camera's: GRIP_EXT is unchecked with it)
    "detect_camera": "wrist",
    # where the handle comes from: "sam3" (SAM 3's mask of sam3_prompts, from sam3_service) or "depth" (the vertical bar
    # sticking out of the door)
    "handle_source": "sam3",
    # how each look measures the door's direction (open_door.DOOR_NORMALS): "handle_model" (the printed handle's model,
    # RAMMP.Fridge.Handle5.STL, fit to SAM 3's handle mask on the depth: door_normal.py, from the side grasp), "handle"
    # (the bar's front and its plate), "plane" (a door-plane fit over the whole image), "square" (TESTING ONLY: the door
    # taken as square to the arm). handle_model and handle need handle_source sam3
    "door_normal": "handle_model",
    "sam3_service": "/perception/sam3/detect",
    "sam3_prompts": ["handle"],
    "sam3_threshold": 0.3,
    **constants.MEASURED,
    # the fridge's collision world for the cuRobo planner (world.py): this node writes it into planner_world_dir, a host
    # directory the planner container mounts too, at planner_world_dir_in_planner ("": the same path), and pushes it
    # with set_world. planner_base_layers: the robot world's layers every pushed scene lists again (as the planner sees
    # them; the chair's own layers are the planner config's and stay in every world). planner_restore_world: the world
    # put back when the action ends, resolved next to the planner's config
    "planner_world_dir": "/rammp_planner_worlds",
    "planner_world_dir_in_planner": "",
    "planner_base_layers": ["/config/robot/curobo/layer_mebot_floor.yaml"],
    "planner_restore_world": "world.yaml",
    "chair_back_dir": list(constants.CHAIR_BACK_DIR),
    # go_to_start's pose: the arm's joints J1..J7 (deg)
    "start_joints_deg": list(constants.START_JOINTS_DEG),
    # every run also records a rosbag into its run folder (bag.RunBag): the wrist camera this many times a second
    # (~75 MB a minute at 1) and TF
    "record_bag": True,
    "bag_frame_hz": 1.0,
    # ... and the scene camera's frames too (any of its topics "": not recorded), and boxer's 3D boxes ("": not)
    "scene_color_topic": "/scene_camera/color/image_raw",
    "scene_camera_info_topic": "/scene_camera/color/camera_info",
    "scene_depth_topic": "/scene_camera/depth/image_rect_raw",
    # ... and the wrist camera's rectified stereo pair (the OAK's infra1 and infra2, each with its camera_info; any ""
    # or no publisher: not recorded), to run a learned stereo matcher on later. The OAK names its left mono infra2: the
    # camera_infos say which is left. ~30 MB a minute at 1 (two 640 x 400 mono images)
    "wrist_infra1_topic": "/wrist_camera/infra1/image_rect_raw",
    "wrist_infra1_info_topic": "/wrist_camera/infra1/camera_info",
    "wrist_infra2_topic": "/wrist_camera/infra2/image_rect_raw",
    "wrist_infra2_info_topic": "/wrist_camera/infra2/camera_info",
    "boxes_topic": "/perception/object_detections_3d",
    # drive_base: its limits and the MIB drive profile (base_drive.PARAMS)
    **base_drive.PARAMS,
}

ERROR_TITLES = {
    "bad_params": "Bad request",
    "refused": "Couldn't plan that move",
    "base_stopped": "Wheelchair stopped",
    "stopped": "Stopped short",
}
ERROR_PANEL_TIMEOUT_S = 60.0  # how long an error panel waits for the user to tap OK
BASE_CONTROL_STATUS_TOPIC = "/base/control_status"  # the MIB's rt/base/control_status (rammp-rtps control_message.hpp)


def _base_topics():
    """The drive base's topics for the run bag, every message (RunBag's `full`): what the MIB says (MibStatus: state,
    profile, ground speed, error; ControlStatus: arbitration on, owner, commands it rejected), what was sent to it
    (DriveCommand, and TwistSetpoint: this node's own twists, token included), and its odometry. Names from
    capability_engine.base, so they are the ones self.base uses. QoS as their publishers': best-effort, except
    ControlStatus (reliable, latched: transient_local, depth 1)."""
    from capability_engine import base as B
    from nav_msgs.msg import Odometry
    from rammp_base_interfaces.msg import DriveCommand, MibStatus, TwistSetpoint
    from rammp_common_interfaces.msg import ControlStatus
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    return [
        (B.MIB_STATUS_TOPIC, "rammp_base_interfaces/msg/MibStatus", MibStatus, qos_profile_sensor_data),
        (BASE_CONTROL_STATUS_TOPIC, "rammp_common_interfaces/msg/ControlStatus", ControlStatus, latched),
        (B.DRIVE_COMMAND_TOPIC, "rammp_base_interfaces/msg/DriveCommand", DriveCommand, qos_profile_sensor_data),
        (B.TWIST_TOPIC, "rammp_base_interfaces/msg/TwistSetpoint", TwistSetpoint, qos_profile_sensor_data),
        (B.ODOM_TOPIC, "nav_msgs/msg/Odometry", Odometry, qos_profile_sensor_data),
    ]


def _joint_gap_deg(q_a, q_b):
    """The largest joint difference between `q_a` and `q_b` (rad), deg; the free joints compared the short way."""
    d = np.asarray(q_a, float) - np.asarray(q_b, float)
    d[rammp_arm.CONTINUOUS] = (d[rammp_arm.CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.max(np.abs(d))))


class Capability(CapabilityEngine):
    display_name = "Fridge"
    perception_classes = ["fridge"]
    required_resources = ["arm", "gripper", "base"]  # the base only with drive_base
    actions = [
        Action(
            "go_to_start",
            "Go to the fridge start pose",
            params_schema={
                "type": "object",
                "properties": {"execute": COMMON["execute"], "speed": COMMON["speed"]},
            },
        ),
        Action(
            "open",
            "Open the door",
            params_schema={
                "type": "object",
                "properties": {
                    **COMMON,
                    "swing_deg": {"type": "integer", "minimum": 30, "maximum": 110},
                    "push_open": {"type": "boolean", "default": True},
                    # full-pull path only (push_open false): False tests the pull alone -- after the swing just let go
                    # and back off the handle, no round-the-door (the plan is gated by the back-off only)
                    "around": {"type": "boolean", "default": True},
                    # grasp this far (m) above the middle of the handle bar, below if negative; refused off the bar
                    # (handle_length_m)
                    "grasp_offset_m": {"type": "number", "default": 0.0},
                    # a run stopped holding the handle: let go, back off, nothing else
                    "release_only": {"type": "boolean", "default": False},
                    # the push-open asked for a wheelchair move after the pull and it was made: push the door on
                    "chair_moved": {"type": "boolean", "default": False},
                    # a dry run that plans OK also writes the door file as the planned open would leave it, so the
                    # close can be tested without an executed open (open the door to that angle by hand)
                    "save_door": {"type": "boolean", "default": False},
                },
            },
        ),
        Action(
            "close",
            "Close the door",
            params_schema={
                "type": "object",
                "properties": {
                    **COMMON,
                    # at least this, or the swing doesn't carry the door shut
                    "speed": {**COMMON["speed"], "default": constants.CLOSE_SPEED},
                    **{k: _schema(v) for k, v in close_door.GOAL_SETTINGS.items()},
                },
            },
        ),
    ]

    def get_logger(self):
        return TeeLogger(super().get_logger())

    def __init__(self):
        super().__init__("fridge")
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        # an empty string can arrive as "null" (sheppy's params file, 10-29: camera_frame) -- it means empty
        self.cfg = {
            k: "" if isinstance(v, str) and v.strip().lower() in ("null", "none") else v for k, v in self.cfg.items()
        }
        open_door.configure(
            self.cfg["data_dir"],
            self.cfg["mount_tilt_deg"],
            self.cfg["door_normal"],
            **{k: self.cfg[k] for k in constants.MEASURED},
        )
        if self.cfg["handle_source"] not in ("depth", "sam3"):
            raise ValueError(f"handle_source must be 'depth' or 'sam3', not {self.cfg['handle_source']!r}")
        if self.cfg["door_normal"] in ("handle_model", "handle") and self.cfg["handle_source"] != "sam3":
            raise ValueError(f"door_normal {self.cfg['door_normal']} needs handle_source sam3 (the handle's mask)")
        if self.cfg["detect_camera"] not in ("scene", "wrist"):
            raise ValueError(f"detect_camera must be 'scene' or 'wrist', not {self.cfg['detect_camera']!r}")
        scene_cam = self.cfg["detect_camera"] == "scene"
        self.detect_topics = tuple(
            self.cfg[k]
            for k in (
                ("scene_color_topic", "scene_camera_info_topic", "scene_depth_topic")
                if scene_cam
                else ("color_topic", "camera_info_topic", "depth_topic")
            )
        )
        if not all(self.detect_topics):
            raise ValueError(
                f"detect_camera {self.cfg['detect_camera']}: its colour, info and depth topics must be set"
            )
        shared_open.configure(HANDLE_SOURCE=self.cfg["handle_source"])
        if len(self.cfg["start_joints_deg"]) != len(constants.START_JOINTS_DEG):
            raise ValueError(f"start_joints_deg must be 7 joint angles (deg), not {self.cfg['start_joints_deg']!r}")
        if self.cfg["record_bag"] and not self.mock:
            from .bag import RunBag  # here, not at import: rosbag2, which the planning tests don't have

            cams = [(self.cfg["color_topic"], self.cfg["camera_info_topic"], self.cfg["depth_topic"])]
            scene = (self.cfg["scene_color_topic"], self.cfg["scene_camera_info_topic"], self.cfg["scene_depth_topic"])
            if all(scene):
                cams.append(scene)
            extra = []
            if self.cfg["boxes_topic"]:
                from rammp_perception_interfaces.msg import ObjectDetections

                extra.append(
                    (self.cfg["boxes_topic"], "rammp_perception_interfaces/msg/ObjectDetections", ObjectDetections)
                )
            stereo = tuple(
                self.cfg[f"wrist_{k}"]
                for k in ("infra1_topic", "infra1_info_topic", "infra2_topic", "infra2_info_topic")
            )
            stereo = [stereo] if all(stereo) else []
            full = _base_topics()
            motion.BAG = lambda run_dir: RunBag(self, run_dir, cams, self.cfg["bag_frame_hz"], extra, stereo, full)
        self._fk = None  # the arm's forward kinematics, from its model (read in the first action)
        # SAM 3's client lives as long as the node (a client gets no messages; only subscriptions are per action)
        self.sam3 = None
        if self.cfg["handle_source"] == "sam3":
            from rammp_perception_interfaces.srv import Detect
            from rclpy.callback_groups import ReentrantCallbackGroup

            self.sam3 = self.create_client(Detect, self.cfg["sam3_service"], callback_group=ReentrantCallbackGroup())
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def execute(self, action_id, params, ctx):
        try:
            return self._execute(action_id, params, ctx)
        except EngineError as e:
            self._show_error(ctx, ERROR_TITLES.get(e.error_code, "Error"), str(e))
            raise
        except Exception as e:  # a bug, not a refusal: say so rather than fail blank
            self._show_error(ctx, "Unexpected error", repr(e))
            raise

    def _show_error(self, ctx, title, message):
        """Show an error on the UI panel and wait (up to ERROR_PANEL_TIMEOUT_S) for the user to tap OK.
        The engine closes the panel when execute() ends, so a panel shown right before raising would vanish."""
        update = getattr(ctx, "update_user_panel", None)  # absent until the UI API (PR #42) is in this branch
        if update is None or ctx.cancelled():  # a cancel isn't an error to show
            return
        panel = {"title": title, "message": message, "buttons": [{"button_id": "dismiss", "label": "OK"}]}
        if not update(panel_json=panel):  # no session token (standalone) or the bridge is down: already logged
            return
        deadline = time.monotonic() + ERROR_PANEL_TIMEOUT_S
        while time.monotonic() < deadline and not ctx.cancelled():
            event = ctx.poll_ui_event()
            if event and event.get("type") == "button" and event.get("button_id") == "dismiss":
                return
            time.sleep(0.1)

    def _execute(self, action_id, params, ctx):
        speed = float(params.get("speed", constants.CLOSE_SPEED if action_id == "close" else 1.0))
        if not 0.0 < speed <= 1.0:
            raise EngineError("bad_params", "'speed' must be in (0, 1]", recoverable=True)
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        drive_base = bool(params.get("drive_base", False))
        base_speed = float(params.get("base_speed", base_drive.SPEED_DEFAULT))
        if drive_base and not 0.0 < base_speed <= self.cfg["base_max_speed"]:
            raise EngineError(
                "bad_params", f"'base_speed' must be in (0, {self.cfg['base_max_speed']}] m/s", recoverable=True
            )
        if drive_base and params.get("chair_moved"):
            raise EngineError(
                "bad_params",
                "drive_base drives the wheelchair itself; chair_moved is for after you moved it by hand -- send one",
                recoverable=True,
            )
        if action_id == "go_to_start":  # mock too: self.arm's mock plans and moves the mock joints
            return self._go_to_start(ctx, execute, speed, note)
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing detected or moved")
            return {"executed": False, "mock": True}

        ctx.checkpoint()
        if self._fk is None:  # the arm's own model, as the planner plans on it (/arm/robot_description)
            self._fk = Chain(self.arm.robot_description())
        world = self._planner_world()
        planner = Planner(self.arm, world, self._fk, ctx.checkpoint, budget_s=constants.PLAN_BUDGET_S)
        # the camera's subscriptions and tf buffer are this action's: the engine destroys them when it ends
        dc.set_camera(
            WristCamera(  # the detection camera (detect_camera): the wrist's or the scene's
                self,
                *self.detect_topics,
                rammp_arm.FRAME,
                self.cfg["camera_frame"] if self.cfg["detect_camera"] == "wrist" else "",
                sam3=self.sam3,
                sam3_prompts=self.cfg["sam3_prompts"],
                sam3_threshold=self.cfg["sam3_threshold"],
            )
        )
        dc.set_output(self.get_logger().info)
        shared_open.configure(CHECKPOINT=ctx.checkpoint)  # a cancel stops a long plan, not only a motion
        ctx.progress(action_id, 0.0, note or ("executing" if execute else "dry run"))
        log_args = {**params, "execute": execute}
        # a dry run gets it too, for its messages; it refuses to drive unless the goal executes
        drive_chair = None
        if drive_base:
            base_cfg = {k: self.cfg[k] for k in base_drive.PARAMS}
            drive_chair = partial(
                base_drive.move_chair, self.base, self.arm, ctx, speed=base_speed, cfg=base_cfg, execute=execute
            )
        try:
            # before the arm moves, if this goal may drive the base: its odometry and the arm's e-stop status are there
            if action_id == "open":
                may_drive = bool(params.get("push_open", True)) and not params.get("release_only")
            else:
                may_drive = params.get("chair_back") is None or float(params["chair_back"]) != 0
            if drive_base and may_drive:
                base_drive.check(self.base, self.arm)
            if action_id == "open":
                ok = open_door.run(
                    self.arm,
                    planner,
                    execute,
                    swing_deg=params.get("swing_deg"),
                    push=bool(params.get("push_open", True)),
                    release_only=bool(params.get("release_only", False)),
                    speed=speed,
                    log_args=log_args,
                    grasp_offset=float(params.get("grasp_offset_m", 0.0)),
                    save_door=bool(params.get("save_door", False)),
                    around=bool(params.get("around", True)),
                    chair_moved=bool(params.get("chair_moved", False)),
                    chair_back_dir=self.cfg["chair_back_dir"],
                    drive_chair=drive_chair,
                )
                if ok and push_open.NEXT_STEP:  # the wheelchair move it asks for, where the goal was sent
                    ctx.progress(action_id, 100.0, push_open.NEXT_STEP)
                    note = f"{note}. {push_open.NEXT_STEP}" if note else push_open.NEXT_STEP
            else:
                settings = {k: v for k, v in params.items() if k in close_door.GOAL_SETTINGS}
                ok = close_door.run(
                    self.arm,
                    planner,
                    execute,
                    chair_back_dir=self.cfg["chair_back_dir"],
                    speed=speed,
                    log_args=log_args,
                    drive_chair=drive_chair,
                    **settings,
                )
                chair_back = close_door.LAST_CHAIR_BACK or 0.0  # the move the plan used
                if ok and not execute and chair_back != 0 and drive_base:
                    next_step = (
                        f"Plan OK. With execute true and drive_base true the base drives the wheelchair "
                        f"{close_door.how_moved(chair_back)} at {base_speed:.2f} m/s, then closes."
                    )
                    ctx.progress(action_id, 100.0, next_step)
                    note = f"{note}. {next_step}" if note else next_step
                elif ok and not execute and chair_back != 0 and not settings.get("chair_moved"):
                    # the plan passed: say, where the goal was sent (feedback and the result), what comes next
                    next_step = (
                        f"Plan OK. {close_door.move_chair(chair_back)}, then send the close with execute true and "
                        "chair_moved true."
                    )
                    ctx.progress(action_id, 100.0, next_step)
                    note = f"{note}. {next_step}" if note else next_step
        except dc.Refused as e:
            raise EngineError("refused", f"{e} -- the arm holds where it is", recoverable=True) from e
        except BaseError as e:  # a wheelchair move stopped part way (or the MIB refused it): the base commanded still
            raise EngineError(
                "base_stopped",
                f"{e} -- the base is stopped and the arm holds where it is; check where the wheelchair is before any "
                "new goal",
                recoverable=True,
            ) from e
        finally:
            world.restore()  # the robot's own planner world back, also after a refusal or a cancel
            dc.set_output(None)
            dc.set_camera(None)
            shared_open.configure(CHECKPOINT=None)
        if not ok:
            raise EngineError(
                "stopped", f"the {action_id} stopped short -- the arm holds where it is", recoverable=True
            )
        # an open that stopped for a wheelchair move left the door part way, the push still to come: not opened yet (the
        # chair_moved goal that pushes it reports that)
        if execute and not (action_id == "open" and push_open.NEXT_STEP):
            ctx.update_object(self.instance_id, opened=action_id == "open")
        return {"executed": execute, "note": note, "runs_dir": str(dc.RUNS_DIR)}

    def _planner_world(self):
        """This action's PlannerWorld: the scenes it pushes list planner_base_layers, and on the level mount the chair
        stand-in (it has no chair layers in its planner; flatbot's layers, its scene camera included, are the
        planner's own)."""
        return PlannerWorld(
            self.arm,
            self.cfg["planner_world_dir"],
            self.cfg["planner_world_dir_in_planner"] or None,
            [x for x in self.cfg["planner_base_layers"] if x],
            self.cfg["planner_restore_world"],
            extra=[chair_stand_in(self.cfg["chair_keep_out_behind_m"])] if self.cfg["mount_tilt_deg"] == 0 else [],
        )

    def _go_to_start(self, ctx, execute, speed, note):
        """Action go_to_start: the arm to start_joints_deg by cuRobo's joint-space plan from where it is, against the
        robot's world (no fridge: it isn't detected yet). Planned here first, so a dry run says whether it would go
        and a refusal names the obstacle before anything moves; executing, the driver plans it again from the measured
        joints and runs it (go_to_joint_config). A refusal or a stop leaves the arm holding where it is."""
        action_id = "go_to_start"
        target = np.radians(np.asarray(self.cfg["start_joints_deg"], float))
        ctx.checkpoint()
        q_now = np.asarray(self.arm.joint_states().position, float)
        gap = _joint_gap_deg(target, q_now)
        if gap <= constants.START_REACHED_DEG:
            done = f"already at the fridge start pose (every joint within {gap:.1f} deg): nothing moved"
            ctx.progress(action_id, 100.0, done)
            return {"executed": False, "note": f"{note}. {done}" if note else done}
        world = self._planner_world()
        try:
            if not self.mock:  # the mock planner has no world, and the node may not write planner_world_dir
                world.push([], "go_to_start: the robot's world, no fridge (not detected yet)")
            ctx.progress(action_id, 0.0, f"planning to the fridge start pose (furthest joint {gap:.0f} deg away)")
            plan = self.arm.plan_to_joints(list(map(float, target)), list(map(float, q_now)))
            if not plan.success:
                msg = " ".join(str(plan.message).split())
                hint = (
                    " The arm's start is inside an obstacle of the planner's world (the boxes it names): teleop it "
                    "clear first."
                    if "START_STATE" in msg
                    else ""
                )
                raise EngineError(
                    "refused",
                    f"cuRobo found no collision-free path to the fridge start pose: {msg}.{hint} -- the arm holds "
                    "where it is",
                    recoverable=True,
                )
            planned_s = float(plan.t[-1]) if len(plan.t) else 0.0
            if not execute:
                done = f"dry run: the fridge start pose plans ({planned_s:.1f} s at full speed); nothing moved"
                ctx.progress(action_id, 100.0, f"{note}. {done}" if note else done)
                return {"executed": False, "note": f"{note}. {done}" if note else done}
            ctx.progress(action_id, 10.0, "moving to the fridge start pose")
            scale = max(0.01, rammp_arm.GO_TO_SPEED_SCALE * speed)  # `speed` scales it, as in every fridge run
            if not self.arm.go_to_joint_config(target, label="go_to_start", speed_scale=scale):
                raise EngineError(
                    "stopped",
                    f"the move to the fridge start pose stopped short ({self.arm.last_error}) -- the arm holds where "
                    "it is",
                    recoverable=True,
                )
        except dc.Refused as e:  # the planner refused the world
            raise EngineError("refused", f"{e} -- the arm holds where it is", recoverable=True) from e
        finally:
            world.restore()  # the robot's own planner world back (only if this action pushed one)
        ctx.progress(action_id, 100.0, "at the fridge start pose")
        return {"executed": True, "note": note}


def main():
    run(Capability)


if __name__ == "__main__":
    main()
