"""The cameras' and detections' subscriptions are the engine's per-action ones: they receive while
the action runs and are gone when it ends. Needs rclpy and capability_engine (the arm test image)."""

import sys
import threading
import time
import types

import pytest

rclpy = pytest.importorskip("rclpy")

from capability_engine import Action, CapabilityEngine  # noqa: E402
from rammp_orchestrator_interfaces.action import ExecuteAction  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

# Neither is in the arm test image: cv_bridge is only constructed here, and the detections can
# ride on any message type for this test.
sys.modules.setdefault("cv_bridge", types.SimpleNamespace(CvBridge=lambda: None))
_perception = types.ModuleType("rammp_perception_interfaces")
_perception.msg = types.SimpleNamespace(ObjectDetections=String)
sys.modules.setdefault("rammp_perception_interfaces", _perception)
sys.modules.setdefault("rammp_perception_interfaces.msg", _perception.msg)

from grab_container import scene  # noqa: E402
from grab_container.camera import Camera  # noqa: E402

TOPICS = {"/t/color": Image, "/t/info": CameraInfo, "/t/depth": Image, "/t/detections": String}
WAIT_S = 10.0


def wait_until(ready, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not ready():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


class Engine(CapabilityEngine):
    actions = [Action("run", "run")]

    def __init__(self):
        super().__init__(
            "grab_container_entities_test",
            parameter_overrides=[Parameter("use_orchestrator", value=False), Parameter("mock", value=True)],
        )
        self.seen = None

    def execute(self, action_id, params, ctx):
        cam = Camera(self, self.create_action_tf_buffer(), "arm_mount", "/t/color", "/t/info", "/t/depth")
        detections = scene.SceneDetections(self, "/t/detections")
        wait_until(lambda: all(cam.get(n) is not None for n in ("color", "info", "depth")))
        self.seen = detections.wait(WAIT_S)
        self.tf_subs = [self.count_subscribers(t) for t in ("/tf", "/tf_static")]
        return {}


def test_subscriptions_live_only_while_the_action_runs():
    rclpy.init()
    try:
        engine, client = Engine(), Node("grab_container_entities_client")
        executor = MultiThreadedExecutor()
        executor.add_node(engine)
        executor.add_node(client)
        threading.Thread(target=executor.spin, daemon=True).start()
        pubs = [client.create_publisher(t, topic, qos_profile_sensor_data) for topic, t in TOPICS.items()]
        stop = threading.Event()

        def publish():
            while not stop.is_set():
                for p in pubs:
                    p.publish(p.msg_type())
                time.sleep(0.05)

        threading.Thread(target=publish, daemon=True).start()
        action = ActionClient(client, ExecuteAction, "/grab_container_entities_test/execute_action")
        assert action.wait_for_server(timeout_sec=WAIT_S)
        send = action.send_goal_async(ExecuteAction.Goal(action_id="run"))
        wait_until(send.done)
        result = send.result().get_result_async()
        wait_until(result.done)
        stop.set()

        assert engine.seen is not None  # every topic reached the action
        assert all(engine.tf_subs), engine.tf_subs  # the action's tf buffer was listening
        for topic in [*TOPICS, "/tf", "/tf_static"]:  # graph updates lag a little
            wait_until(lambda topic=topic: client.count_subscribers(topic) == 0)
        executor.shutdown()
        engine.destroy_node()
        client.destroy_node()
    finally:
        rclpy.shutdown()
