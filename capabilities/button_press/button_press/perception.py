"""Camera input for one action: the latest camera frames and tf. Never moves the arm.

Subscribes, for the running action only (CapabilityEngine.create_action_subscription), to the wrist
camera under ``camera_ns`` (colour, depth aligned to colour, camera_info) and, between
start_scene() and stop_scene(), the fixed scene camera under ``scene_ns`` (colour and depth NOT
aligned: each has its own camera_info and optical frame) and the scene segmenter's detections.
``press_button.Run`` reads frames and transforms from it.

tf: the arm's state publisher plus the deployment's camera mounts (rammp-deployments' static_tf,
transforms/cameras.yaml on flatbot), through the action's tf buffer.

The scene segmenter (rammp-perception) publishes /perception/object_detections
(rammp_perception_interfaces ObjectDetections, boxes in the scene colour image): microwave_box()
picks the microwave's, which narrows scene_finder's search.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from capability_engine.action_scope import ActionScope
from cv_bridge import CvBridge
from rammp_perception_interfaces.msg import DetectedObject, ObjectDetections
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data  # best effort: receives from reliable and best-effort publishers
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from button_press import Abort

FRESH_S = 1.0  # a frame older than this is stale
SCENE_NS = "/scene_camera"
SCENE_COLOR_FRAME = "scene_camera_color_optical_frame"
DETECTIONS_TOPIC = "/perception/object_detections"
DETECTION_CLASS = "microwave"
DETECTION_MIN_SCORE = 0.5  # YOLO's microwave scored 0.89-0.90 live (2026-10-15); not yet checked on the segmenter
DETECTION_FRESH_S = 3.0  # a box received longer ago than this is not used
SCENE_DEPTH_FRAME = "scene_camera_depth_optical_frame"
# Depth topics, relative to the camera's namespace. The RealSense's aligned depth and the Gemini's
# depth are the defaults; the OAKs (flatbot) publish depth/image_rect_raw for both.
DEPTH_TOPIC = "aligned_depth_to_color/image_raw"
SCENE_DEPTH_TOPIC = "depth/image_raw"
SCENE_KEYS = ("scene_color", "scene_depth", "scene_cinfo", "scene_dinfo", "detections")


def frame_of(msg, default: str) -> str:
    """The optical frame a camera message names in its header, or `default` if it names none."""
    frame = getattr(getattr(msg, "header", None), "frame_id", "") or ""
    return frame.lstrip("/") or default


def to_bgr_depth(bridge: CvBridge, color_msg: Image, depth_msg: Image):
    """(bgr8 image, float32 depth in metres) from a colour + depth message pair (mm depth converted)."""
    bgr = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, "passthrough").astype(np.float32)
    if depth_msg.encoding in ("16UC1", "mono16"):  # the RealSense driver publishes millimetres
        depth /= 1000.0
    return bgr, depth


def scene_frame_from_msgs(
    bridge: CvBridge,
    color_msg: Image,
    depth_msg: Image,
    color_info: CameraInfo,
    depth_info: CameraInfo,
    T_base_color,
    T_base_depth,
) -> dict:
    """Bundle the scene camera's messages and its two base_link <- optical-frame transforms (4x4)
    into the keyword arguments scene_finder.find_microwave takes."""
    bgr = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, "passthrough").astype(np.float32)
    if depth_msg.encoding in ("16UC1", "mono16"):
        depth /= 1000.0
    k_c, k_d = color_info.k, depth_info.k
    return {
        "bgr": bgr,
        "depth_m": depth,
        "color_K": (k_c[0], k_c[4], k_c[2], k_c[5]),
        "depth_K": (k_d[0], k_d[4], k_d[2], k_d[5]),
        "T_base_color": np.asarray(T_base_color, dtype=float),
        "T_base_depth": np.asarray(T_base_depth, dtype=float),
    }


def microwave_box(msg, frame=SCENE_COLOR_FRAME, cls=DETECTION_CLASS, min_score=DETECTION_MIN_SCORE):
    """(u0, v0, u1, v1, score) of the best visible `cls` object in an ObjectDetections, in `frame`'s
    colour pixels, or None (no message, another frame, nothing of that class at min_score)."""
    if msg is None or msg.header.frame_id.lstrip("/") != frame:
        return None
    best = None
    for o in msg.objects:
        if o.perception_class != cls or o.state != DetectedObject.STATE_VISIBLE or o.score < min_score:
            continue
        if best is None or o.score > best[4]:
            best = (*map(float, o.box), float(o.score))
    return best


def why_no_fit(bgr, depth_m, fx) -> str:
    """Why dome_pattern.detect() returned None on this frame: the layout was found but outside
    the detector's range (so the camera should move), or no 5-dome layout at all."""
    from button_press import dome_pattern as dp  # noqa: PLC0415

    fit = dp.fit_layout(dp.candidates(bgr, depth_m, fx), fx)
    if fit is None:
        return "no 5-dome layout"
    if fit.z < dp.MIN_RANGE_M:
        return f"TOO CLOSE: domes at {fit.z*100:.1f} cm (min {dp.MIN_RANGE_M*100:.0f})"
    if fit.z > dp.MAX_RANGE_M:
        return f"TOO FAR: domes at {fit.z*100:.0f} cm (max {dp.MAX_RANGE_M*100:.0f})"
    return "no 5-dome layout"


class Perception:
    """Latest camera messages (get / wait) and tf lookups into the arm base frame, for one action.

    `engine`: the running CapabilityEngine; make this inside execute(). `checkpoint`: the action's
    ctx.checkpoint, polled in wait()."""

    def __init__(
        self,
        engine,
        checkpoint,
        arm_frame: str,
        cam_frame: str,
        camera_ns: str = "/wrist_camera",
        scene_ns: str | None = None,
        depth_topic: str = DEPTH_TOPIC,
        scene_depth_topic: str = SCENE_DEPTH_TOPIC,
    ):
        """cam_frame "" = the wrist colour image's own header.frame_id. depth_topic and
        scene_depth_topic are relative to their camera's namespace; the wrist depth must be aligned
        to colour (pixel for pixel, same optical frame). scene_ns None: no scene camera."""
        self.engine, self.checkpoint = engine, checkpoint
        self.arm_frame, self.cam_frame = arm_frame, cam_frame
        self.scene_ns, self.scene_depth_topic = scene_ns, scene_depth_topic
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest: dict[str, tuple[float, object]] = {}
        self.tfbuf = engine.create_action_tf_buffer()
        self._scene = None  # the scene camera's subscriptions, between start_scene() and stop_scene()
        ns = camera_ns.rstrip("/")
        for msg_type, topic, name in (
            (CameraInfo, f"{ns}/color/camera_info", "info"),
            (Image, f"{ns}/color/image_raw", "color"),
            (Image, f"{ns}/{depth_topic.strip('/')}", "depth"),
        ):
            engine.create_action_subscription(msg_type, topic, self._keep(name), qos_profile_sensor_data)

    def _keep(self, name):
        def cb(msg):
            with self.lock:
                self.latest[name] = (time.monotonic(), msg)

        return cb

    def get_clock(self):
        return self.engine.get_clock()

    def start_scene(self):
        """Subscribe to the scene camera and the segmenter's detections, for the first look only.
        stop_scene() destroys them; the action's end does not, so call it in a finally."""
        if self.scene_ns is None or self._scene is not None:
            return
        sns = self.scene_ns.rstrip("/")
        self._scene = ActionScope(self.engine, ReentrantCallbackGroup())
        for msg_type, topic, name in (
            (Image, f"{sns}/color/image_raw", "scene_color"),
            (Image, f"{sns}/{self.scene_depth_topic.strip('/')}", "scene_depth"),
            (CameraInfo, f"{sns}/color/camera_info", "scene_cinfo"),
            (CameraInfo, f"{sns}/depth/camera_info", "scene_dinfo"),
            (ObjectDetections, DETECTIONS_TOPIC, "detections"),
        ):
            self._scene.create_subscription(msg_type, topic, self._keep(name), qos_profile_sensor_data)

    def stop_scene(self):
        """Drop the scene camera's subscriptions once the first look is done (and forget its
        frames). Left open, its 1280x720 colour + depth at 30 Hz starved the wrist camera: the
        staging re-detect got 4 frames in 8 s instead of 7 in under 1 s (2026-10-14)."""
        scene, self._scene = self._scene, None
        if scene is not None:
            scene.close()
        with self.lock:
            for k in SCENE_KEYS:
                self.latest.pop(k, None)

    # -- accessors ---------------------------------------------------------------------------
    def get(self, name, max_age=None):
        """Latest message on `name` ("color", "depth", "info", "scene_*", "detections"), or None if
        there is none yet or it is older than max_age seconds."""
        with self.lock:
            item = self.latest.get(name)
        if item is None:
            return None
        t, v = item
        if max_age is not None and time.monotonic() - t > max_age:
            return None
        return v

    def wait(self, name, timeout, max_age=FRESH_S):
        """Like get(), but waits up to `timeout` seconds for one (Cancelled once the action is
        cancelled). None on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            v = self.get(name, max_age)
            if v is not None:
                return v
            self.checkpoint()
            time.sleep(0.05)
        return None

    def intrinsics(self):
        """(fx, fy, cx, cy) of the wrist colour camera. Raises Abort without camera_info."""
        info = self.get("info")
        if info is None:
            raise Abort("no camera_info")
        return info.k[0], info.k[4], info.k[2], info.k[5]

    def camera_frame(self) -> str:
        """The wrist camera's optical frame: cam_frame, or (cam_frame "") the colour image's own."""
        if self.cam_frame:
            return self.cam_frame
        color = self.get("color")
        if color is None or not color.header.frame_id:
            raise Abort("no wrist colour image yet to take the camera frame from -- is the wrist camera up?")
        return color.header.frame_id.lstrip("/")

    def _cam_in_base(self):
        frame = self.camera_frame()
        try:
            return self.tfbuf.lookup_transform(self.arm_frame, frame, Time(), timeout=Duration(seconds=2.0)).transform
        except Exception as e:  # noqa: BLE001
            raise Abort(
                f"tf {self.arm_frame} <- {frame} unavailable: {e}\n"
                "  sheppy arm_state_publisher / static_tf nodes up? /joint_states flowing? verify with\n"
                f"  ros2 run tf2_ros tf2_echo {self.arm_frame} {frame}"
            ) from e

    def transform(self, frame: str) -> np.ndarray:
        """4x4 arm base <- `frame` (latest tf). Raises Abort."""
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        try:
            tr = self.tfbuf.lookup_transform(self.arm_frame, frame, Time(), timeout=Duration(seconds=2.0)).transform
        except Exception as e:  # noqa: BLE001
            raise Abort(f"tf {self.arm_frame} <- {frame} unavailable: {type(e).__name__}") from None
        m = np.eye(4)
        m[:3, :3] = R.from_quat([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]).as_matrix()
        m[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
        return m

    def cam_rotation_in_base(self) -> np.ndarray:
        """3x3 rotation taking camera-frame directions into arm-base directions."""
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        q = self._cam_in_base().rotation
        return R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def cam_pose_at(self, stamp, timeout_s: float = 2.0):
        """(position, 3x3 rotation) of the camera in the arm base frame AT `stamp` (an image header
        stamp), from ONE tf lookup that waits for tf to reach that time. Raises Abort.

        Not "latest", and not two lookups: right after a fast move this node's tf buffer can still be
        working through the backlog, and the two "latest" lookups (rotation, then position) once came
        from different moments -- the rotation from mid-move, 85 deg off, and a button 57 cm wrong
        from a perfect dome fit (2026-10-15)."""
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        frame = self.camera_frame()
        try:
            tr = self.tfbuf.lookup_transform(
                self.arm_frame, frame, Time.from_msg(stamp), timeout=Duration(seconds=timeout_s)
            ).transform
        except Exception as e:  # noqa: BLE001
            raise Abort(
                f"tf {self.arm_frame} <- {frame} at the image time unavailable: "
                f"{type(e).__name__} -- arm/tf publishing?"
            ) from None
        q, t = tr.rotation, tr.translation
        return np.array([t.x, t.y, t.z]), R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def cam_position_in_base(self) -> np.ndarray:
        """Camera optical-frame origin in the arm base frame (tf2), metres."""
        t = self._cam_in_base().translation
        return np.array([t.x, t.y, t.z])
