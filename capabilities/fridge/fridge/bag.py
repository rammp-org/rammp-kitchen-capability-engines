"""A run's rosbag: enough to see again what the run saw, kept small.

Each camera's colour, depth and intrinsics (the wrist camera's, and the scene camera's when it has topics) as matched
sets, `frame_hz` of them a second (the cameras run at 15); the extra topics (boxer's 3D boxes) at `frame_hz` too; all of
/tf_static, and /tf at TF_HZ for each set of frames it carries (with /tf_static: where each camera was for each frame).
The arm's joints, tool pose and gripper are not here: the run's arm.npz has them at 25 Hz. Raw, lossless, sqlite3.
"""

import threading

import message_filters
import rosbag2_py
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

SYNC_SLOP_S = 0.05  # colour, depth and camera_info within this of each other are one frame (perception.WristCamera's)
TF_HZ = 5.0
TF_QOS = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
TF_STATIC_QOS = QoSProfile(
    depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)


class _Input(message_filters.SimpleFilter):
    """A message_filters input on the running action's subscription (message_filters.Subscriber makes a plain one)."""

    def __init__(self, node, msg_type, topic, qos):
        super().__init__()
        node.create_action_subscription(msg_type, topic, self.signalMessage, qos)


class RunBag:
    """Records into `run_dir`/bag from now until `close()`. Make it inside the action (its subscriptions are the
    action's, as create_action_subscription's)."""

    def __init__(self, node, run_dir, cameras, frame_hz, extra=()):
        """`cameras`: [(colour, camera_info, depth topic)], the first the wrist camera's (close() counts its frames).
        `extra`: [(topic, type name, message class)] thinned to `frame_hz` (no publisher: nothing recorded)."""
        self.node, self.period_ns = node, int(1e9 / frame_hz)
        self.tf_period_ns = int(1e9 / TF_HZ)
        self._lock = threading.Lock()
        self._last_frame, self._last_tf = {}, {}
        self._counts = {cam: 0 for cam in cameras}
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
        self.topics.update({topic: type_name for topic, type_name, _ in extra})
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
        for topic, _, msg_class in extra:
            node.create_action_subscription(msg_class, topic, lambda m, topic=topic: self._on_extra(topic, m), 10)
        node.create_action_subscription(TFMessage, "/tf", self._on_tf, TF_QOS)
        node.create_action_subscription(TFMessage, "/tf_static", self._on_tf_static, TF_STATIC_QOS)

    def _now(self):
        return self.node.get_clock().now().nanoseconds

    def _write(self, topic, msg, t):
        if self._writer is not None:  # under the lock; None once closed (a late callback)
            self._writer.write(topic, serialize_message(msg), t)

    def _thinned(self, key, t):
        """Under the lock: whether `key`'s message at `t` is due (frame_hz), noting it if so."""
        if key in self._last_frame and t - self._last_frame[key] < self.period_ns:
            return False
        self._last_frame[key] = t
        return True

    def _on_frame(self, cam, color, info, depth):
        t = self._now()
        with self._lock:
            if not self._thinned(cam, t):
                return
            for topic, msg in zip(cam, (color, info, depth), strict=True):
                self._write(topic, msg, t)
            self._counts[cam] += 1

    def _on_extra(self, topic, msg):
        t = self._now()
        with self._lock:
            if self._thinned(topic, t):
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
