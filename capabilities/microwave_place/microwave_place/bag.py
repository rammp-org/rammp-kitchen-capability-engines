"""Per-action sqlite3 rosbag, using the fridge's synchronized-frame recording pattern.

Only action-owned subscriptions are created. No arm commands, subprocesses or idle
subscriptions. Camera sets are sampled together; state and TF are sampled separately.
"""

import json
import math
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import message_filters
import rosbag2_py
import yaml
from diagnostic_msgs.msg import DiagnosticArray
from rammp_arm_interfaces.msg import EeState, GripperState
from rcl_interfaces.msg import Log
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

# Humble may expose uint8 constants as bytes even though Log.level requires int.
LOG_WARN = 30
LOG_ERROR = 40


def _level(value):
    return value[0] if isinstance(value, bytes) else int(value)


SYNC_SLOP_S = 0.05
EVENT_TOPIC = "/microwave_place/run_event"  # bag-only JSON events, not a live publisher
STATIC_QOS = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
LOG_QOS = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)


class RecordingError(Exception):
    """Recording could not start: refuse the action before any motion."""


class _Input(message_filters.SimpleFilter):
    def __init__(self, node, msg_type, topic):
        super().__init__()
        node.create_action_subscription(msg_type, topic, self.signalMessage, qos_profile_sensor_data)


def _qos_yaml(qos):
    def duration(value):
        sec, nsec = divmod(value.nanoseconds, 1_000_000_000)
        return dict(sec=sec, nsec=nsec)

    return yaml.safe_dump(
        [
            dict(
                history=int(qos.history),
                depth=qos.depth,
                reliability=int(qos.reliability),
                durability=int(qos.durability),
                deadline=duration(qos.deadline),
                lifespan=duration(qos.lifespan),
                liveliness=int(qos.liveliness),
                liveliness_lease_duration=duration(qos.liveliness_lease_duration),
                avoid_ros_namespace_conventions=qos.avoid_ros_namespace_conventions,
            )
        ]
    )


class RunBag:
    """One action's bag and run.json; close on success, failure, cancellation or exception.

    `camera`: (color, color camera_info, depth), matching the camera used by the
    action. Start failures raise; later I/O errors are reported without interrupting
    arm cleanup. Subscriptions are destroyed by the engine's action scope, not here.
    """

    def __init__(self, node, state_dir, action, params, config, camera, frame_hz=1.0, state_hz=25.0):
        for name, value in (("bag_frame_hz", frame_hz), ("bag_state_hz", state_hz)):
            if not math.isfinite(value) or value <= 0:
                raise RecordingError(f"{name} must be finite and positive")
        self.node, self.camera = node, tuple(camera)
        self.frame_period, self.state_period = 1 / frame_hz, 1 / state_hz
        self._lock, self._last, self.counts = threading.Lock(), {}, {}
        self._writer, self._closed, self.result = None, False, None
        self.errors = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_dir = Path(state_dir) / "runs" / f"{action}_{stamp}_{uuid.uuid4().hex[:8]}"
        self.path = self.run_dir / "bag"
        self.metadata = dict(
            action=action,
            params=params,
            config=config,
            camera=list(camera),
            started_ns=self._now(),
            frame_hz=frame_hz,
            state_hz=state_hz,
        )
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            self._writer = rosbag2_py.SequentialWriter()
            self._writer.open(
                rosbag2_py.StorageOptions(uri=str(self.path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"),
            )
            self._subscribe()
            self.event("started", self.metadata)
            if self.errors:
                raise RecordingError("; ".join(self.errors))
            self._save_metadata(dict(self.metadata, status="running"))
        except Exception as exc:
            self.close(exc)
            raise RecordingError(f"cannot start run bag at {self.path}: {exc}") from exc

    def _now(self):
        return self.node.get_clock().now().nanoseconds

    def _topic(self, topic, msg_type, qos):
        self._writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic, type=msg_type, serialization_format="cdr", offered_qos_profiles=_qos_yaml(qos)
            )
        )
        self.counts[topic] = 0

    def _subscribe(self):
        for topic, name in zip(self.camera, ("Image", "CameraInfo", "Image"), strict=True):
            self._topic(topic, f"sensor_msgs/msg/{name}", qos_profile_sensor_data)
        inputs = [
            _Input(self.node, cls, topic) for cls, topic in zip((Image, CameraInfo, Image), self.camera, strict=True)
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(inputs, 10, SYNC_SLOP_S)
        self._sync.registerCallback(self._on_frame)
        streams = [
            ("/joint_states", "sensor_msgs/msg/JointState", JointState, qos_profile_sensor_data, self.state_period),
            ("/ee_state", "rammp_arm_interfaces/msg/EeState", EeState, qos_profile_sensor_data, self.state_period),
            (
                "/gripper_state",
                "rammp_arm_interfaces/msg/GripperState",
                GripperState,
                qos_profile_sensor_data,
                self.state_period,
            ),
            ("/tf", "tf2_msgs/msg/TFMessage", TFMessage, QoSProfile(depth=100), 0.1),
            ("/tf_static", "tf2_msgs/msg/TFMessage", TFMessage, STATIC_QOS, 0),
            ("/arm/robot_description", "std_msgs/msg/String", String, STATIC_QOS, 0),
            ("/diagnostics", "diagnostic_msgs/msg/DiagnosticArray", DiagnosticArray, LOG_QOS, 1.0),
            ("/rosout", "rcl_interfaces/msg/Log", Log, LOG_QOS, 0),
        ]
        if self.metadata["action"] == "take_out":
            streams.append(
                ("/microwave_place/sam3_overlay", "sensor_msgs/msg/Image", Image, STATIC_QOS, self.frame_period)
            )
        for topic, type_name, cls, qos, period in streams:
            self._topic(topic, type_name, qos)
            self.node.create_action_subscription(
                cls, topic, lambda msg, topic=topic, period=period: self._on_message(topic, msg, period), qos
            )
        self._topic(EVENT_TOPIC, "std_msgs/msg/String", QoSProfile(depth=1))

    def _due(self, key, period):
        now = time.monotonic()
        if period and key in self._last and now - self._last[key] < period:
            return False
        self._last[key] = now
        return True

    def _write(self, topic, message, stamp):
        # Caller holds the lock. A late callback after close must never touch the writer.
        if self._closed or self._writer is None or self.errors:
            return
        try:
            self._writer.write(topic, serialize_message(message), stamp)
            self.counts[topic] += 1
        except Exception as exc:
            self.errors.append(f"write {topic}: {exc}")

    def _on_frame(self, color, info, depth):
        with self._lock:
            if self._closed or not self._due("camera", self.frame_period):
                return
            stamp = self._now()
            for topic, msg in zip(self.camera, (color, info, depth), strict=True):
                self._write(topic, msg, stamp)

    def _on_message(self, topic, msg, period):
        # Keep this node's full log and all nodes' warnings/errors, without thinning.
        if topic == "/rosout" and _level(msg.level) < LOG_WARN and msg.name != self.node.get_logger().name:
            return
        key = topic
        if topic == "/tf":
            key = (topic, tuple(sorted((t.header.frame_id, t.child_frame_id) for t in msg.transforms)))
        # Never thin away an error diagnostic.
        if topic == "/diagnostics" and any(_level(s.level) >= 2 for s in msg.status):
            period = 0
        with self._lock:
            if self._closed or not self._due(key, period):
                return
            self._write(topic, msg, self._now())

    def event(self, status, detail):
        with self._lock:
            self._write(EVENT_TOPIC, String(data=json.dumps(dict(status=status, detail=detail))), self._now())

    def _save_metadata(self, metadata):
        temporary = self.run_dir / "run.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(self.run_dir / "run.json")

    def close(self, error=None):
        with self._lock:
            if self._closed:
                return
            status = (
                "succeeded" if error is None else ("cancelled" if type(error).__name__ == "Cancelled" else "failed")
            )
            outcome = dict(status=status, result=self.result, finished_ns=self._now())
            if error is not None:
                outcome.update(
                    error_type=type(error).__name__,
                    error=str(error),
                    traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)),
                )
                # Synchronous copy: the engine may emit its failure log only after execute returns.
                self._write(
                    "/rosout",
                    Log(
                        stamp=self.node.get_clock().now().to_msg(),
                        level=LOG_ERROR,
                        name=self.node.get_logger().name,
                        msg=str(error),
                    ),
                    self._now(),
                )
            self._write(EVENT_TOPIC, String(data=json.dumps(outcome)), self._now())
            self._closed = True
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception as exc:
                    self.errors.append(f"close: {exc}")
                finally:
                    self._writer = None
            metadata = dict(self.metadata, **outcome, counts=dict(self.counts), recording_errors=list(self.errors))
            try:
                self._save_metadata(metadata)
            except Exception as exc:
                self.errors.append(f"run.json: {exc}")
        if self.errors:
            self.node.get_logger().error(f"Run recording incomplete at {self.path}: {'; '.join(self.errors)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(exc)
        return False
