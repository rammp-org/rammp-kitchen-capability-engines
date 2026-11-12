"""A run's rosbag: enough to see again what the run saw and did, kept small.

From the fridge capability's bag.py (feature/15-fridge, RunBag), its stereo pair left out. In the bag:

  cameras   the wrist camera's (and, far "scene", the scene camera's) colour, camera_info and depth as matched
            sets, `frame_hz` of them a second (the cameras run at 15); the scene depth's own camera_info too
  arm       /joint_states, /ee_state, /gripper_state at ARM_HZ; /control_status as it changes
  the run   /button_press/target_pose (the planned pre-press pose) and /button_press/status, every message;
            the scene segmenter's /perception/object_detections at `frame_hz`
  errors    /rosout at WARN and above, from every node (the driver's and the planner's refusals included)
  tf        all of /tf_static, and /tf at TF_HZ per set of frames it carries

Raw, lossless, sqlite3. The run's own progress lines and every check's record are beside it in the run folder
(console.log, run.jsonl): press_button.Run.
"""

import shutil
import threading

import message_filters
import rosbag2_py
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

SYNC_SLOP_S = 0.05  # colour, depth and camera_info within this of each other are one frame
TF_HZ = 5.0
ARM_HZ = 10.0
ROSOUT_MIN_LEVEL = 30  # rcl_interfaces/Log: 30 WARN, 40 ERROR, 50 FATAL
MIN_FREE_GB = 2.0  # no bag with less than this free under the run folder (~75 MB a minute a camera at 1 Hz)
TF_QOS = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
LATCHED_QOS = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
EVERY = 0.0  # an extra topic's hz: record every message


def free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


class _Input(message_filters.SimpleFilter):
    """A message_filters input on the running action's subscription (message_filters.Subscriber makes a plain one)."""

    def __init__(self, node, msg_type, topic, qos):
        super().__init__()
        node.create_action_subscription(msg_type, topic, self.signalMessage, qos)


def topics(camera_ns, depth_topic, scene_ns=None, scene_depth_topic=None):
    """(cameras, extra) for RunBag from the run's camera settings: cameras = [(colour, camera_info, depth)], the
    wrist camera first; extra = [(topic, type name, message class, hz, qos)]."""
    from geometry_msgs.msg import PoseStamped
    from rammp_arm_interfaces.msg import EeState, GripperState
    from rammp_common_interfaces.msg import ControlStatus
    from rammp_perception_interfaces.msg import ObjectDetections
    from rcl_interfaces.msg import Log
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    from button_press.perception import DETECTIONS_TOPIC

    cams = [(f"{camera_ns}/color/image_raw", f"{camera_ns}/color/camera_info", f"{camera_ns}/{depth_topic}")]
    extra = [
        ("/joint_states", "sensor_msgs/msg/JointState", JointState, ARM_HZ, qos_profile_sensor_data),
        ("/ee_state", "rammp_arm_interfaces/msg/EeState", EeState, ARM_HZ, qos_profile_sensor_data),
        ("/gripper_state", "rammp_arm_interfaces/msg/GripperState", GripperState, ARM_HZ, qos_profile_sensor_data),
        ("/control_status", "rammp_common_interfaces/msg/ControlStatus", ControlStatus, EVERY, LATCHED_QOS),
        ("/button_press/target_pose", "geometry_msgs/msg/PoseStamped", PoseStamped, EVERY, 10),
        ("/button_press/status", "std_msgs/msg/String", String, EVERY, 10),
        ("/rosout", "rcl_interfaces/msg/Log", Log, EVERY, 100),
    ]
    if scene_ns:
        cams.append((f"{scene_ns}/color/image_raw", f"{scene_ns}/color/camera_info", f"{scene_ns}/{scene_depth_topic}"))
        extra.append(
            (f"{scene_ns}/depth/camera_info", "sensor_msgs/msg/CameraInfo", CameraInfo, None, qos_profile_sensor_data)
        )
        extra.append((DETECTIONS_TOPIC, "rammp_perception_interfaces/msg/ObjectDetections", ObjectDetections, None, 10))
    return cams, extra


class RunBag:
    """Records into `run_dir`/bag from now until `close()`. Make it inside the action (its subscriptions are the
    action's, as create_action_subscription's)."""

    def __init__(self, node, run_dir, cameras, frame_hz, extra=()):
        """`cameras`: [(colour, camera_info, depth topic)], the first the wrist camera's (close() counts its frames).
        `extra`: [(topic, type name, message class, hz, qos)]: hz None = `frame_hz`, EVERY (0) = every message
        (no publisher: nothing recorded). /rosout keeps only ROSOUT_MIN_LEVEL and above."""
        self.node, self.period_ns = node, int(1e9 / frame_hz)
        self.tf_period_ns = int(1e9 / TF_HZ)
        self._lock = threading.Lock()
        self._last_frame, self._last_tf = {}, {}
        self._counts = {cam: 0 for cam in cameras}
        self._periods = {}
        self._wrist = cameras[0]
        self._writer = rosbag2_py.SequentialWriter()
        self._writer.open(
            rosbag2_py.StorageOptions(uri=str(run_dir / "bag"), storage_id="sqlite3"),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        self.topics = {"/tf": "tf2_msgs/msg/TFMessage", "/tf_static": "tf2_msgs/msg/TFMessage"}
        for color_topic, info_topic, depth_topic in cameras:
            self.topics.update(
                {
                    color_topic: "sensor_msgs/msg/Image",
                    info_topic: "sensor_msgs/msg/CameraInfo",
                    depth_topic: "sensor_msgs/msg/Image",
                }
            )
        self.topics.update({topic: type_name for topic, type_name, *_ in extra})
        for name, msg_type in self.topics.items():
            self._writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=msg_type, serialization_format="cdr"))
        self._syncs = []
        for cam in cameras:
            inputs = [
                _Input(node, Image, cam[0], qos_profile_sensor_data),
                _Input(node, CameraInfo, cam[1], qos_profile_sensor_data),
                _Input(node, Image, cam[2], qos_profile_sensor_data),
            ]
            sync = message_filters.ApproximateTimeSynchronizer(inputs, 10, SYNC_SLOP_S)
            sync.registerCallback(lambda c, i, d, cam=cam: self._on_frame(cam, c, i, d))
            self._syncs.append(sync)
        for topic, _, msg_class, hz, qos in extra:
            self._periods[topic] = self.period_ns if hz is None else (int(1e9 / hz) if hz else 0)
            node.create_action_subscription(msg_class, topic, lambda m, topic=topic: self._on_extra(topic, m), qos)
        node.create_action_subscription(TFMessage, "/tf", self._on_tf, TF_QOS)
        node.create_action_subscription(TFMessage, "/tf_static", self._on_tf_static, LATCHED_QOS)

    def _now(self):
        return self.node.get_clock().now().nanoseconds

    def _write(self, topic, msg, t):
        if self._writer is not None:  # under the lock; None once closed (a late callback)
            self._writer.write(topic, serialize_message(msg), t)

    def _due(self, key, t, period_ns):
        """Under the lock: whether `key`'s message at `t` is due, noting it if so."""
        if period_ns and key in self._last_frame and t - self._last_frame[key] < period_ns:
            return False
        self._last_frame[key] = t
        return True

    def _on_frame(self, cam, color, info, depth):
        t = self._now()
        with self._lock:
            if not self._due(cam, t, self.period_ns):
                return
            for topic, msg in zip(cam, (color, info, depth), strict=True):
                self._write(topic, msg, t)
            self._counts[cam] += 1

    def _on_extra(self, topic, msg):
        if topic == "/rosout" and getattr(msg, "level", 0) < ROSOUT_MIN_LEVEL:
            return
        t = self._now()
        with self._lock:
            if self._due(topic, t, self._periods[topic]):
                self._write(topic, msg, t)

    def _on_tf(self, msg):
        t = self._now()
        key = tuple(sorted(tr.child_frame_id for tr in msg.transforms))  # each publisher's own set, thinned apart
        with self._lock:
            if t - self._last_tf.get(key, -self.tf_period_ns) < self.tf_period_ns:
                return
            self._last_tf[key] = t
            self._write("/tf", msg, t)

    def _on_tf_static(self, msg):
        with self._lock:
            self._write("/tf_static", msg, self._now())

    def close(self):
        """Finish the bag (its metadata too). Returns how many wrist camera frames it holds."""
        with self._lock:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            return self._counts[self._wrist]
