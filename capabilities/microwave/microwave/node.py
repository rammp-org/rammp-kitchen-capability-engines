"""The microwave capability: open the microwave door with the arm.

Action `open` (open_door.run): detect the handle from the wrist camera, grasp it, swing the door
open, let go and push it further open (the user's 10-07 push-open demo, replayed relative to the
door), then park. Every leg is planned and sim-gated before it moves; any failed check stops the
run with the arm holding where it is (EngineError "refused").

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets
`execute`. Otherwise the goal is a dry run (detect, plan and gate every leg; nothing moves) and the
result says so. Someone stands at the e-stop whenever allow_execute is on.

    ros2 launch microwave microwave.launch.py use_orchestrator:=false
    make action CAP=microwave ACTION=open PARAMS='{"execute": false}'
"""

import rammp_arm
from capability_engine import Action, Cancelled, CapabilityEngine, EngineError, run
from door_common import common as dc

from . import open_door, recording
from .perception import WristCamera
from .tee_logger import TeeLogger

# ROS parameter -> default. Topics and paths are per robot (the fragment's params:).
PARAMS = {
    "allow_execute": False,
    "data_dir": "/microwave_data",
    "color_topic": "/wrist_camera/color/image_raw",
    "camera_info_topic": "/wrist_camera/color/camera_info",
    # every wrist camera's depth, aligned to colour on the D405 and the OAK; the D435i needs aligned_depth_to_color
    "depth_topic": "/wrist_camera/depth/image_rect_raw",
    "camera_frame": "",
    # the scene camera (depth aligned to colour): without boxer, its view of the door places the view pose
    "scene_color_topic": "/scene_camera/color/image_raw",
    "scene_camera_info_topic": "/scene_camera/color/camera_info",
    "scene_depth_topic": "/scene_camera/depth/image_rect_raw",
    # the scene segmenter's 2D boxes on the scene camera (RF-DETR): its box_class box is where the door is looked for
    "scene_detections_topic": "/perception/object_detections",
    # boxer's 3D boxes (scene camera), e.g. /perception/object_detections_3d: the open goes to a view pose in front
    # of the microwave's box and the handle detector crops the wrist camera's points to it. "" (default, 10-30: boxer's
    # box wasn't accurate enough on snorlax) = the scene camera places the view pose.
    "box_topic": "",
    "box_class": "microwave",
    # the view pose in front of the box (open_door.VIEW_*): camera distances out from its front face and heights
    # above its middle (m), tried in order. Per camera and setup.
    "view_dists": list(open_door.VIEW_DISTS),
    "view_heights": list(open_door.VIEW_HEIGHTS),
    # every run also records a rosbag into its run folder (bag.RunBag): the wrist and scene cameras' colour, depth
    # and intrinsics bag_frame_hz times a second, and TF. A 640x400 frame (colour + depth) is ~1.3 MB, so at 0.5 Hz
    # the two cameras are ~75 MB a minute; past bag_max_mb no more frames (TF still). The arm's joints are in the
    # run's arm.npz (RunLog), what the node logs (errors too) in its console.log
    "record_bag": True,
    "bag_frame_hz": 0.5,
    "bag_max_mb": 300.0,
    # ... and the wrist camera's rectified stereo pair, for a learned stereo matcher later (+~15 MB a minute at 0.5):
    # off ("") by default. The camera_infos say which is left (the OAK names its left mono infra2), e.g.
    # /wrist_camera/infra1/image_rect_raw, /wrist_camera/infra1/camera_info and the same for infra2
    "wrist_infra1_topic": "",
    "wrist_infra1_info_topic": "",
    "wrist_infra2_topic": "",
    "wrist_infra2_info_topic": "",
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
                    "execute": {"type": "boolean", "default": False},
                    "swing_deg": {"type": "integer", "minimum": 45, "maximum": 75},
                    "push_open": {"type": "boolean", "default": True},
                    "speed": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 1.0},
                    # stop at the view pose (the open always goes there first, to detect the handle from it)
                    "view_only": {"type": "boolean", "default": False},
                },
            },
        )
    ]

    def get_logger(self):
        return TeeLogger(super().get_logger())

    def __init__(self):
        super().__init__("microwave")
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        # an empty string can arrive as "null" (sheppy's params file, 10-29: camera_frame) -- it means empty
        self.cfg = {
            k: "" if isinstance(v, str) and v.strip().lower() in ("null", "none") else v for k, v in self.cfg.items()
        }
        open_door.configure(self.cfg["data_dir"])
        dc.enable_chair_collision()  # the moves PyBullet plans stay out of the chair and user, as cuRobo's do
        open_door.VIEW_DISTS = tuple(float(v) for v in self.cfg["view_dists"])
        open_door.VIEW_HEIGHTS = tuple(float(v) for v in self.cfg["view_heights"])
        if self.cfg["record_bag"] and not self.mock:
            self._set_up_bag()
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def _set_up_bag(self):
        from .bag import RunBag  # here, not at import: rosbag2, which the planning tests don't have

        cams = [(self.cfg["color_topic"], self.cfg["camera_info_topic"], self.cfg["depth_topic"])]
        scene = (self.cfg["scene_color_topic"], self.cfg["scene_camera_info_topic"], self.cfg["scene_depth_topic"])
        if all(scene):
            cams.append(scene)
        from rammp_perception_interfaces.msg import ObjectDetections

        # the scene segmenter's 2D boxes (where the view pose comes from without boxer) and boxer's 3D ones, if set
        extra = [
            (topic, "rammp_perception_interfaces/msg/ObjectDetections", ObjectDetections)
            for topic in (self.cfg["scene_detections_topic"], self.cfg["box_topic"])
            if topic
        ]
        stereo = tuple(
            self.cfg[f"wrist_{k}"] for k in ("infra1_topic", "infra1_info_topic", "infra2_topic", "infra2_info_topic")
        )
        stereo = [stereo] if all(stereo) else []
        recording.BAG = lambda run_dir: RunBag(
            self, run_dir, cams, self.cfg["bag_frame_hz"], extra, stereo, self.cfg["bag_max_mb"]
        )

    def execute(self, action_id, params, ctx):
        speed = float(params.get("speed", 1.0))
        if not 0.0 < speed <= 1.0:
            raise EngineError("bad_params", "'speed' must be in (0, 1]", recoverable=True)
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            ctx.progress("open", 100.0, "mock: nothing detected or moved")
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
        )
        dc.set_camera(camera)
        if action_id == "open" and not self.cfg["box_topic"]:
            open_door.set_scene_camera(
                WristCamera(
                    self,
                    self.cfg["scene_color_topic"],
                    self.cfg["scene_camera_info_topic"],
                    self.cfg["scene_depth_topic"],
                    rammp_arm.FRAME,
                    box_class=self.cfg["box_class"],
                    box2d_topic=self.cfg["scene_detections_topic"],
                )
            )
        dc.set_output(self.get_logger().info)
        ctx.progress(action_id, 0.0, note or ("executing" if execute else "dry run"))
        log_args = {**params, "execute": execute}
        holding = False

        def on_hold(now):
            """From the grasp until the push-open lets go the hand is on the handle: a pause or cancel there would
            leave it holding the door, so the step is not pausable and the result says so."""
            nonlocal holding
            holding = now
            ctx.progress(action_id, 50.0, "holding the handle" if now else "let go of the handle", pausable=not now)

        held = " -- still holding the handle"
        try:
            ok = open_door.run(
                self.arm,
                execute,
                swing_deg=params.get("swing_deg"),
                push=bool(params.get("push_open", True)),
                speed=speed,
                log_args=log_args,
                view_only=bool(params.get("view_only", False)),
                on_hold=on_hold,
            )
        except dc.Refused as e:
            raise EngineError(
                "refused", f"{e} -- the arm holds where it is{held if holding else ''}", recoverable=True
            ) from e
        except Cancelled:
            if holding:  # a cancelled action can't move the arm: someone has to let go of the handle
                self.get_logger().error(f"{action_id} cancelled{held}; release it with open_door --release")
            raise
        finally:
            dc.set_output(None)
            dc.set_camera(None)
            open_door.set_scene_camera(None)
        if not execute:
            ctx.checkpoint()  # a dry run makes no arm call after detecting and planning: a cancel there ends it here
        if not ok:
            raise EngineError(
                "stopped",
                f"the {action_id} stopped short -- the arm holds where it is{held if holding else ''}",
                recoverable=True,
            )
        if execute:
            ctx.update_object(self.instance_id, opened=True)
        return {"executed": execute, "note": note, "runs_dir": str(dc.RUNS_DIR), "holding_handle": holding}


def main():
    run(Capability)


if __name__ == "__main__":
    main()
