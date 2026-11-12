"""The scene camera's messages -> one scene_finder frame (perception.scene_frame_from_msgs), without
spinning ROS: the Gemini publishes rgb8 colour and 16UC1 millimetre depth, each with its own
camera_info and optical frame."""

import inspect

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

from button_press import scene_finder
from button_press.perception import scene_frame_from_msgs


def info(fx, fy, cx, cy, w, h):
    m = CameraInfo()
    m.width, m.height = w, h
    m.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    return m


def test_scene_messages_become_a_finder_frame():
    br = CvBridge()
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[0, 0] = (200, 100, 50)
    depth_mm = np.full((2, 4), 1234, np.uint16)
    color, depth = br.cv2_to_imgmsg(rgb, "rgb8"), br.cv2_to_imgmsg(depth_mm, "16UC1")
    T_c, T_d = np.eye(4), np.eye(4)
    T_d[:3, 3] = (0.01, 0.02, 0.0)
    fr = scene_frame_from_msgs(
        br, color, depth, info(612.7, 612.8, 635.4, 361.0, 3, 2), info(410.1, 410.1, 426.0, 237.4, 4, 2), T_c, T_d
    )
    assert tuple(fr["bgr"][0, 0]) == (50, 100, 200)
    assert fr["depth_m"].dtype == np.float32 and abs(float(fr["depth_m"][1, 3]) - 1.234) < 1e-6
    assert fr["color_K"] == (612.7, 612.8, 635.4, 361.0)
    assert fr["depth_K"] == (410.1, 410.1, 426.0, 237.4)
    assert np.array_equal(fr["T_base_depth"], T_d)
    want = [
        p
        for p in inspect.signature(scene_finder.find_microwave).parameters.values()
        if p.default is inspect.Parameter.empty
    ]
    assert set(fr) == {p.name for p in want}  # exactly the frame find_microwave needs


def detections(frame, *dets):
    """An ObjectDetections: (class, score, (x_min, y_min, x_max, y_max)[, state]) per object."""
    from rammp_perception_interfaces.msg import DetectedObject, ObjectDetections

    m = ObjectDetections()
    m.header.frame_id = frame
    for cls, score, box, *state in dets:
        o = DetectedObject()
        o.perception_class, o.score, o.box = cls, float(score), [float(v) for v in box]
        o.state = state[0] if state else DetectedObject.STATE_VISIBLE
        m.objects.append(o)
    return m


def test_the_microwave_box_is_the_best_visible_microwave():
    from button_press.perception import SCENE_COLOR_FRAME, microwave_box

    m = detections(
        SCENE_COLOR_FRAME,
        ("fridge", 0.57, (200, 300, 300, 400)),
        ("microwave", 0.9015, (552.75, 240.25, 846.0, 431.0)),
        ("microwave", 0.6, (75, 75, 125, 125)),
    )
    u0, v0, u1, v1, score = microwave_box(m)
    assert np.allclose((u0, v0, u1, v1), (552.75, 240.25, 846.0, 431.0)) and abs(score - 0.9015) < 1e-6


def test_the_microwave_box_refuses_weak_occluded_other_frame_and_missing():
    from rammp_perception_interfaces.msg import DetectedObject

    from button_press.perception import SCENE_COLOR_FRAME, microwave_box

    box = (552, 240, 846, 431)
    assert microwave_box(None) is None
    assert microwave_box(detections(SCENE_COLOR_FRAME, ("microwave", 0.3, box))) is None
    assert microwave_box(detections(SCENE_COLOR_FRAME, ("fridge", 0.9, box))) is None
    assert microwave_box(detections(SCENE_COLOR_FRAME, ("microwave", 0.9, box, DetectedObject.STATE_OCCLUDED))) is None
    assert microwave_box(detections("wrist_d405_color_optical", ("microwave", 0.9, box))) is None


def engine():
    """A node with the engine's two per-action entity makers, as CapabilityEngine gives them."""
    from capability_engine.action_scope import ActionScope
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node

    class Engine(Node):
        def __init__(self):
            super().__init__("button_press_test")
            self.scope = ActionScope(self, ReentrantCallbackGroup())

        def create_action_subscription(self, *a):
            return self.scope.create_subscription(*a)

        def create_action_tf_buffer(self):
            return self.scope.create_tf_buffer()

    return Engine()


def test_the_scene_camera_is_subscribed_only_between_start_and_stop_scene():
    import rclpy

    from button_press.perception import DETECTIONS_TOPIC, SCENE_NS, Perception

    rclpy.init()
    try:
        node = engine()
        per = Perception(node, lambda: None, "arm_mount", "", "/wrist_camera", scene_ns=SCENE_NS)
        topics = lambda: {s.topic_name for s in node.subscriptions}  # noqa: E731
        assert "/wrist_camera/color/image_raw" in topics() and "/tf" in topics()
        assert not any(t.startswith(SCENE_NS) for t in topics()) and DETECTIONS_TOPIC not in topics()
        per.start_scene()
        assert f"{SCENE_NS}/color/image_raw" in topics() and DETECTIONS_TOPIC in topics()
        per.stop_scene()
        assert not any(t.startswith(SCENE_NS) for t in topics()) and DETECTIONS_TOPIC not in topics()
        assert "/wrist_camera/color/image_raw" in topics()
        per.stop_scene()  # twice is fine
        node.scope.close()  # the action's end: the wrist camera and tf go too
        assert topics() == set()
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_without_a_scene_camera_start_scene_subscribes_nothing():
    import rclpy

    from button_press.perception import Perception

    rclpy.init()
    try:
        node = engine()
        per = Perception(node, lambda: None, "arm_mount", "", "/wrist_camera", scene_ns=None)
        before = {s.topic_name for s in node.subscriptions}
        per.start_scene()
        assert {s.topic_name for s in node.subscriptions} == before
        node.scope.close()
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_the_camera_pose_is_taken_at_the_image_time_not_the_latest():
    """2026-10-15: right after a fast move, "latest" tf gave a rotation from mid-move (85 deg off) and the
    button came out 57 cm wrong. cam_pose_at(stamp) answers for the image's own time, in one lookup."""
    import rclpy
    from builtin_interfaces.msg import Time as TimeMsg
    from geometry_msgs.msg import TransformStamped
    from scipy.spatial.transform import Rotation

    from button_press.perception import Perception

    rclpy.init()
    try:
        node = engine()
        per = Perception(node, lambda: None, "base_link", "cam", "/wrist_camera")
        for sec, yaw, x in ((100, 0.0, 0.5), (101, 85.0, 0.1)):
            tf = TransformStamped()
            tf.header.frame_id, tf.child_frame_id = "base_link", "cam"
            tf.header.stamp = TimeMsg(sec=sec)
            tf.transform.translation.x = x
            q = Rotation.from_euler("z", yaw, degrees=True).as_quat()
            tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = q
            per.tfbuf.set_transform(tf, "test")
        p, Rm = per.cam_pose_at(TimeMsg(sec=100))
        assert np.allclose(p, [0.5, 0, 0]) and np.allclose(Rm, np.eye(3), atol=1e-9)
        p, Rm = per.cam_pose_at(TimeMsg(sec=101))
        assert np.allclose(p, [0.1, 0, 0]) and abs(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])) - 85.0) < 1e-6
        node.scope.close()
        node.destroy_node()
    finally:
        rclpy.shutdown()
