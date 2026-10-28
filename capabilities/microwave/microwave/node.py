"""The microwave capability: open and close the microwave door with the arm.

Action `open` (open_door.run): detect the handle from the wrist camera, grasp it, swing the door
open, let go and push it further open (the user's 10-07 push-open demo, replayed relative to the
door). Action `close` (close_door.run): push the door shut with the side of the hand -- no grasp,
the gripper untouched (it may hold a container) -- against the door the open saved. Every leg is
planned and sim-gated before it moves; any failed check stops the run with the arm holding where
it is (EngineError "refused").

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets
`execute`. Otherwise the goal is a dry run (detect, plan and gate every leg; nothing moves) and the
result says so. Someone stands at the e-stop whenever allow_execute is on.

    ros2 launch microwave microwave.launch.py use_orchestrator:=false
    make action CAP=microwave ACTION=open PARAMS='{"execute": false}'
    make action CAP=microwave ACTION=close PARAMS='{"execute": false, "route_over": true}'
"""

import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run

from . import close_door, open_door
from . import microwave_common as mc
from .perception import YOLO_MODEL, WristCamera

COMMON = {
    "execute": {"type": "boolean", "default": False},
    "speed": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 1.0},
}


def _schema(default):
    """JSON Schema type of a close setting from its default (None: a number or unset)."""
    if isinstance(default, bool):
        return {"type": "boolean", "default": default}
    if default is None:
        return {"type": ["number", "null"], "default": None}
    return {"type": "number", "default": default}


# ROS parameter -> default. Topics and paths are per robot (the fragment's params:).
PARAMS = {
    "allow_execute": False,
    "data_dir": "/microwave_data",
    "color_topic": "/wrist_camera/color/image_raw",
    "camera_info_topic": "/wrist_camera/color/camera_info",
    "depth_topic": "/wrist_camera/aligned_depth_to_color/image_raw",
    "camera_frame": "",
    # boxer's 3D boxes (scene camera), e.g. /perception/object_detections_3d: the open goes to a view pose in front
    # of the microwave's box and the handle detector crops the wrist camera's points to it. "" (default, 10-30: boxer's
    # box wasn't accurate enough on snorlax) = no view pose, YOLO on the wrist image (yolo_model).
    "box_topic": "",
    "box_class": "microwave",
    # the view pose in front of the box (open_door.VIEW_*): camera distances out from its front face and heights
    # above its middle (m), tried in order. Per camera and setup.
    "view_dists": list(open_door.VIEW_DISTS),
    "view_heights": list(open_door.VIEW_HEIGHTS),
    # where the handle comes from: "depth" (the vertical bar sticking out of the door) or "sam3" (SAM 3's mask of
    # sam3_prompts on the wrist image, from sam3_service; the door plane and hinge still come from depth)
    "handle_source": "depth",
    "sam3_service": "/perception/sam3/detect",
    "sam3_prompts": ["handle"],
    "sam3_threshold": 0.3,
    "yolo_model": YOLO_MODEL,
}


class Capability(CapabilityEngine):
    display_name = "Microwave"
    perception_classes = ["microwave"]
    required_resources = ["arm", "gripper"]
    actions = [
        Action(
            "open",
            "Open the door",
            params_schema={
                "type": "object",
                "properties": {
                    **COMMON,
                    "swing_deg": {"type": "integer", "minimum": 45, "maximum": 75},
                    "push_open": {"type": "boolean", "default": True},
                    # with boxer (box_topic): go to the view pose in front of the microwave before detecting
                    "view": {"type": "boolean", "default": True},
                    "view_only": {"type": "boolean", "default": False},
                },
            },
        ),
        Action(
            "close",
            "Close the door",
            params_schema={
                "type": "object",
                "properties": {**COMMON, **{k: _schema(v) for k, v in close_door.CLOSE_DEFAULTS.items()}},
            },
        ),
    ]

    def __init__(self):
        super().__init__("microwave")
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        # an empty string can arrive as "null" (sheppy's params file, 10-29: camera_frame) -- it means empty
        self.cfg = {
            k: "" if isinstance(v, str) and v.strip().lower() in ("null", "none") else v for k, v in self.cfg.items()
        }
        mc.configure_paths(self.cfg["data_dir"])
        open_door.HANDLE_DETECTOR_MODEL = self.cfg["yolo_model"]
        open_door.VIEW_DISTS = tuple(float(v) for v in self.cfg["view_dists"])
        open_door.VIEW_HEIGHTS = tuple(float(v) for v in self.cfg["view_heights"])
        if self.cfg["handle_source"] not in ("depth", "sam3"):
            raise ValueError(f"handle_source must be 'depth' or 'sam3', not {self.cfg['handle_source']!r}")
        open_door.HANDLE_SOURCE = self.cfg["handle_source"]
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
        speed = float(params.get("speed", 1.0))
        if not 0.0 < speed <= 1.0:
            raise EngineError("bad_params", "'speed' must be in (0, 1]", recoverable=True)
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing detected or moved")
            return {"executed": False, "mock": True}

        ctx.checkpoint()
        # the camera's subscriptions and tf buffer are this action's: the engine destroys them when it ends
        camera = WristCamera(
            self,
            self.cfg["color_topic"],
            self.cfg["camera_info_topic"],
            self.cfg["depth_topic"],
            rammp_arm.FRAME,
            self.cfg["camera_frame"],
            self.cfg["box_topic"],
            self.cfg["box_class"],
            self.sam3,
            self.cfg["sam3_prompts"],
            self.cfg["sam3_threshold"],
        )
        mc.set_camera(camera)
        mc.set_output(self.get_logger().info)
        ctx.progress(action_id, 0.0, note or ("executing" if execute else "dry run"))
        log_args = {**params, "execute": execute}
        try:
            if action_id == "open":
                ok = open_door.run(
                    self.arm,
                    execute,
                    swing_deg=params.get("swing_deg"),
                    push=bool(params.get("push_open", True)),
                    speed=speed,
                    log_args=log_args,
                    view=bool(params.get("view", True)),
                    view_only=bool(params.get("view_only", False)),
                )
            else:
                settings = {k: v for k, v in params.items() if k in close_door.CLOSE_DEFAULTS}
                ok = close_door.run(self.arm, execute, speed=speed, log_args=log_args, **settings)
        except mc.Refused as e:
            raise EngineError("refused", f"{e} -- the arm holds where it is", recoverable=True) from e
        finally:
            mc.set_output(None)
            mc.set_camera(None)
        if not ok:
            raise EngineError(
                "stopped", f"the {action_id} stopped short -- the arm holds where it is", recoverable=True
            )
        if execute:
            ctx.update_object(self.instance_id, opened=action_id == "open")
        return {"executed": execute, "note": note, "runs_dir": str(mc.RUNS_DIR)}


def main():
    run(Capability)


if __name__ == "__main__":
    main()
