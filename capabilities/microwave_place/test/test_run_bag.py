"""Real rosbag write/read tests with synthetic messages; no ROS graph or arm commands."""

# ROS imports follow optional-dependency checks so pure planning tests can run without ROS.
# ruff: noqa: E402

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("message_filters")
from builtin_interfaces.msg import Time
from rcl_interfaces.msg import Log
from rclpy.clock import Clock
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

SPEC = importlib.util.spec_from_file_location("microwave_run_bag", Path(__file__).parents[1] / "microwave_place/bag.py")
bag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bag)
CAMERA = ("/test/color", "/test/info", "/test/depth")


class Node:
    def __init__(self):
        self.callbacks = {}
        self.logger = SimpleNamespace(name="recording_test", error=Mock(), info=Mock())

    def create_action_subscription(self, cls, topic, callback, qos):
        self.callbacks[topic] = callback

    def get_clock(self):
        return Clock()

    def get_logger(self):
        return self.logger


def recorder(tmp_path, node=None, **kwargs):
    return bag.RunBag(node or Node(), tmp_path, "take_out", {"execute": False}, {"mock": True}, CAMERA, **kwargs)


def messages(sec=1):
    color = Image(height=3, width=4, encoding="rgb8", step=12, data=bytes(36))
    depth = Image(height=3, width=4, encoding="16UC1", step=8, data=bytes(24))
    info = CameraInfo(height=3, width=4)
    for msg in (color, info, depth):
        msg.header.stamp = Time(sec=sec)
        msg.header.frame_id = "test_optical"
    return color, info, depth


def read(rec):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(rec.path), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr")
    )
    topics = {t.name: t for t in reader.get_all_topics_and_types()}
    records = []
    while reader.has_next():
        records.append(reader.read_next())
    return topics, records


def test_camera_pairs_sampled_together_and_state_saved(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(bag.time, "monotonic", lambda: clock[0])
    node = Node()
    with recorder(tmp_path, node) as rec:
        for topic, msg in zip(CAMERA, messages(), strict=True):
            node.callbacks[topic](msg)
        clock[0] += 0.1
        for topic, msg in zip(CAMERA, messages(2), strict=True):
            node.callbacks[topic](msg)
        clock[0] += 1.0
        for topic, msg in zip(CAMERA, messages(3), strict=True):
            node.callbacks[topic](msg)
        node.callbacks["/joint_states"](JointState(name=["arm_joint_1"], position=[0.3]))
        node.callbacks["/rosout"](Log(level=40, name="other_node", msg="test driver error"))
        rec.result = {"executed": False}
    topics, data = read(rec)
    for topic in CAMERA:
        assert sum(t == topic for t, _, _ in data) == 2
    colors = [deserialize_message(raw, Image) for t, raw, _ in data if t == CAMERA[0]]
    assert [m.header.stamp.sec for m in colors] == [1, 3]
    assert all(m.encoding == "rgb8" and len(m.data) == 36 for m in colors)
    joints = [deserialize_message(raw, JointState) for t, raw, _ in data if t == "/joint_states"]
    assert list(joints[0].position) == [0.3]
    assert "durability: 1" in topics["/tf_static"].offered_qos_profiles
    metadata = json.loads((rec.run_dir / "run.json").read_text())
    assert metadata["status"] == "succeeded"
    assert metadata["counts"][CAMERA[0]] == 2
    assert metadata["result"] == {"executed": False}
    assert (rec.path / "metadata.yaml").exists()


def test_failure_written_synchronously_and_bag_closed(tmp_path):
    rec = recorder(tmp_path)
    with pytest.raises(RuntimeError, match="detection failed"), rec:
        raise RuntimeError("detection failed")
    _, data = read(rec)
    logs = [deserialize_message(raw, Log) for t, raw, _ in data if t == "/rosout"]
    assert logs[-1].level == 40 and logs[-1].msg == "detection failed"
    events = [json.loads(deserialize_message(raw, String).data) for t, raw, _ in data if t == bag.EVENT_TOPIC]
    assert events[-1]["status"] == "failed"
    assert "RuntimeError" in events[-1]["traceback"]
    assert json.loads((rec.run_dir / "run.json").read_text())["status"] == "failed"


def test_cancellation_closed_and_late_callbacks_ignored(tmp_path):
    class Cancelled(Exception):
        pass

    rec = recorder(tmp_path)
    with pytest.raises(Cancelled), rec:
        raise Cancelled("cancelled by user")
    before = dict(rec.counts)
    rec._on_frame(*messages())
    rec._on_message("/joint_states", JointState(), 0)
    rec.close()
    assert rec.counts == before
    assert json.loads((rec.run_dir / "run.json").read_text())["status"] == "cancelled"
    read(rec)


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("key", ["frame_hz", "state_hz"])
def test_invalid_rates_refuse_before_creating_bag(tmp_path, key, rate):
    with pytest.raises(bag.RecordingError, match="positive"):
        recorder(tmp_path, **{key: rate})
    assert not list(tmp_path.iterdir())


def test_mismatched_camera_frames_not_recorded(tmp_path):
    node = Node()
    with recorder(tmp_path, node) as rec:
        color, info, depth = messages()
        depth.header.stamp.sec = 3
        for topic, msg in zip(CAMERA, (color, info, depth), strict=True):
            node.callbacks[topic](msg)
    assert rec.counts[CAMERA[0]] == 0


def test_io_failure_does_not_interrupt_cleanup(tmp_path):
    rec = recorder(tmp_path)
    actual = rec._writer
    rec._writer = Mock()
    rec._writer.write.side_effect = OSError("disk full")
    rec._on_message("/joint_states", JointState(), 0)
    rec.close(RuntimeError("original error"))
    actual.close()
    assert any("disk full" in e for e in rec.errors)
    metadata = json.loads((rec.run_dir / "run.json").read_text())
    assert metadata["error"] == "original error"
    assert metadata["recording_errors"]
    rec.node.get_logger().error.assert_called_once()


def test_start_failure_refuses_before_subscriptions(tmp_path, monkeypatch):
    writer = Mock()
    writer.open.side_effect = OSError("read-only storage")
    monkeypatch.setattr(rosbag2_py, "SequentialWriter", lambda: writer)
    node = Node()
    with pytest.raises(bag.RecordingError, match="read-only storage"):
        recorder(tmp_path, node)
    assert not node.callbacks


def test_run_names_unique_and_info_logs_filtered(tmp_path):
    first, second = recorder(tmp_path), recorder(tmp_path)
    assert first.path != second.path
    first._on_message("/rosout", Log(level=20, name="other_node", msg="routine"), 0)
    first._on_message("/rosout", Log(level=30, name="other_node", msg="warning"), 0)
    first._on_message("/rosout", Log(level=20, name="recording_test", msg="stage"), 0)
    first.close()
    second.close()
    assert first.counts["/rosout"] == 2


@pytest.mark.parametrize("action", ["place", "take_out"])
def test_capability_records_mock_run_without_arm_access(tmp_path, action):
    module = pytest.importorskip("microwave_place.microwave_place_node")
    node = Node()
    node.cfg = dict(
        record_bag=True,
        allow_execute=False,
        bag_frame_hz=1.0,
        bag_state_hz=25.0,
        wrist_topics=["/wrist/color", "/wrist/depth", "/wrist/info"],
        camera_info_topic="/scene/info",
        depth_topic="/scene/depth",
    )
    node.state_dir, node.mock = tmp_path, True
    node._execute_behavior = lambda *args: module.Capability._execute_behavior(node, *args)
    result = module.Capability.execute(node, action, {"execute": True}, Mock())
    assert result["mock"] and not result["executed"]
    assert Path(result["bag_path"]).exists()
    data = json.loads((Path(result["bag_path"]).parent / "run.json").read_text())
    assert data["status"] == "succeeded"
    assert data["camera"][0] == (
        "/wrist/color" if action == "take_out" else module.microwave_detection.SAM3_IMAGE_TOPIC
    )


def test_capability_failure_finalizes_bag(tmp_path):
    module = pytest.importorskip("microwave_place.microwave_place_node")
    node = Node()
    node.cfg = dict(
        record_bag=True,
        bag_frame_hz=1.0,
        bag_state_hz=25.0,
        wrist_topics=["/wrist/color", "/wrist/depth", "/wrist/info"],
    )
    node.state_dir = tmp_path
    node._execute_behavior = Mock(side_effect=RuntimeError("test action failure"))
    with pytest.raises(RuntimeError, match="test action failure"):
        module.Capability.execute(node, "take_out", {}, Mock())
    files = list(tmp_path.glob("runs/*/run.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["status"] == "failed"
    assert (files[0].parent / "bag/metadata.yaml").exists()


def test_error_diagnostics_not_throttled_with_humble_byte_levels(tmp_path, monkeypatch):
    from diagnostic_msgs.msg import DiagnosticStatus

    monkeypatch.setattr(bag.time, "monotonic", lambda: 10.0)
    status = DiagnosticStatus()
    status.level = bytes([2]) if isinstance(status.level, bytes) else 2
    rec = recorder(tmp_path)
    rec._on_message("/diagnostics", bag.DiagnosticArray(), 1.0)
    rec._on_message("/diagnostics", bag.DiagnosticArray(), 1.0)
    error = bag.DiagnosticArray(status=[status])
    rec._on_message("/diagnostics", error, 1.0)
    rec._on_message("/diagnostics", error, 1.0)
    rec.close()
    assert rec.counts["/diagnostics"] == 3
