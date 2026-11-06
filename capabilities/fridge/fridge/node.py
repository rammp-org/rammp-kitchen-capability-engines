"""The fridge capability: open and close the fridge door with the arm.

Action `open` (open_door.run): detect the handle from the wrist camera, grasp it, pull the door 60 deg open, let go and
push it on to 90 (or, with `push_open` false, a 90-deg pull and go back round the door to its inner side). Action
`close` (close_door.run): push the door most of the way shut with the side of the hand -- no grasp, the gripper
untouched (it may hold a container) -- against the door the open saved; it swings shut by itself from there. Every leg
is planned and sim-gated before it moves, with the wheelchair keep-out on; any failed check stops the run with the arm
holding where it is (EngineError "refused").

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
"""

from functools import partial

import rammp_arm
from capability_engine import Action, BaseError, CapabilityEngine, EngineError, run
from door_common import common as dc
from door_common import open_door as shared_open

from . import base_drive, close_door, constants, motion, open_door, push_open
from .perception import WristCamera
from .tee_logger import TeeLogger

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
    # where the handle comes from: "depth" (the vertical bar sticking out of the door) or "sam3" (SAM 3's mask of
    # sam3_prompts on the wrist image, from sam3_service; the door plane still comes from depth)
    "handle_source": "depth",
    "sam3_service": "/perception/sam3/detect",
    "sam3_prompts": ["handle"],
    "sam3_threshold": 0.3,
    **constants.MEASURED,
    "chair_back_dir": list(constants.CHAIR_BACK_DIR),
    # TESTING ONLY: the door taken to face the arm squarely (normal along the arm base's -x), no door plane fit -- for a
    # door whose depth is its reflection. A door yawed off it gets a wrong hinge and swing
    "door_square_to_arm": False,
    # the door's normal from the handle's own depth (bar front + plate), no door plane fit. SAM 3 only; not with
    # door_square_to_arm
    "door_normal_from_handle": False,
    # every run also records a rosbag into its run folder (bag.RunBag): the wrist camera this many times a second
    # (~75 MB a minute at 1) and TF
    "record_bag": True,
    "bag_frame_hz": 1.0,
    # ... and the scene camera's frames too (any of its topics "": not recorded), and boxer's 3D boxes ("": not)
    "scene_color_topic": "/scene_camera/color/image_raw",
    "scene_camera_info_topic": "/scene_camera/color/camera_info",
    "scene_depth_topic": "/scene_camera/depth/image_rect_raw",
    "boxes_topic": "/perception/object_detections_3d",
    # drive_base: its limits and the MIB drive profile (base_drive.PARAMS)
    **base_drive.PARAMS,
}


class Capability(CapabilityEngine):
    display_name = "Fridge"
    perception_classes = ["fridge"]
    required_resources = ["arm", "gripper", "base"]  # the base only with drive_base
    actions = [
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
            self.cfg["door_square_to_arm"],
            self.cfg["door_normal_from_handle"],
            **{k: self.cfg[k] for k in constants.MEASURED},
        )
        if self.cfg["handle_source"] not in ("depth", "sam3"):
            raise ValueError(f"handle_source must be 'depth' or 'sam3', not {self.cfg['handle_source']!r}")
        if self.cfg["door_normal_from_handle"] and self.cfg["handle_source"] != "sam3":
            raise ValueError("door_normal_from_handle needs handle_source sam3 (the handle's mask)")
        shared_open.configure(HANDLE_SOURCE=self.cfg["handle_source"])
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
            motion.BAG = lambda run_dir: RunBag(self, run_dir, cams, self.cfg["bag_frame_hz"], extra)
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
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing detected or moved")
            return {"executed": False, "mock": True}

        ctx.checkpoint()
        # the camera's subscriptions and tf buffer are this action's: the engine destroys them when it ends
        dc.set_camera(
            WristCamera(
                self,
                self.cfg["color_topic"],
                self.cfg["camera_info_topic"],
                self.cfg["depth_topic"],
                rammp_arm.FRAME,
                self.cfg["camera_frame"],
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
                may_drive = float(params.get("chair_back", constants.CHAIR_BACK_M)) > 0
            if drive_base and may_drive:
                base_drive.check(self.base, self.arm)
            if action_id == "open":
                ok = open_door.run(
                    self.arm,
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
                    execute,
                    chair_back_dir=self.cfg["chair_back_dir"],
                    speed=speed,
                    log_args=log_args,
                    drive_chair=drive_chair,
                    **settings,
                )
                chair_back = float(settings.get("chair_back", constants.CHAIR_BACK_M))
                if ok and not execute and chair_back > 0 and drive_base:
                    next_step = (
                        f"Plan OK. With execute true and drive_base true the base drives the wheelchair "
                        f"~{chair_back * 100:.0f} cm back at {base_speed:.2f} m/s, then closes."
                    )
                    ctx.progress(action_id, 100.0, next_step)
                    note = f"{note}. {next_step}" if note else next_step
                elif ok and not execute and chair_back > 0 and not settings.get("chair_moved"):
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


def main():
    run(Capability)


if __name__ == "__main__":
    main()
