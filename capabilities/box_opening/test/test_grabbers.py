"""The camera grabbers can rebuild their subscriptions in place."""

import rclpy


def _count(node):
    return len(list(node.subscriptions))


def test_scene_grabber_resubscribes_in_place():
    from box_opening.perception.scene import SceneGrabber

    rclpy.init()
    try:
        node = rclpy.create_node("grabber_test")
        g = SceneGrabber(node, keep=2, need_depth=True)
        before = _count(node)
        assert len(g._subs) == 4
        g.resubscribe()
        assert g.resubscribes == 1 and len(g._subs) == 4
        assert _count(node) == before  # torn down and rebuilt, not doubled
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_d405_grabber_resubscribes_in_place():
    from box_opening.perception.d405 import D405Grabber

    rclpy.init()
    try:
        node = rclpy.create_node("grabber_test_d405")
        g = D405Grabber(node, need_depth=True)
        before = _count(node)
        assert len(g._subs) == 3
        g.resubscribe()
        assert g.resubscribes == 1 and len(g._subs) == 3
        assert _count(node) == before
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_grabbers_spin_no_thread_of_their_own():
    """One executor per node: a grabber's TF listener must not start a
    spinning thread behind the node's owner (the OWL node went deaf under
    two executors, 2026-09-17)."""
    import threading

    from box_opening.perception.d405 import D405Grabber
    from box_opening.perception.scene import SceneGrabber

    rclpy.init()
    try:
        node = rclpy.create_node("grabber_threads")
        before = threading.active_count()
        SceneGrabber(node, keep=1, need_depth=False)
        D405Grabber(node, need_depth=False)
        assert threading.active_count() == before
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_the_scene_cloud_is_one_function_the_grabber_and_a_replay_share():
    """color_cloud(depth, kd, k, dist, T_color_depth): the scene points in
    the colour frame with their colour pixels — computed from recorded
    arrays exactly as the live grabber computes them."""
    import numpy as np

    from box_opening.perception.scene import color_cloud

    kd = np.array([[400.0, 0, 20], [0, 400.0, 15], [0, 0, 1]])
    k = np.array([[600.0, 0, 30], [0, 600.0, 20], [0, 0, 1]])
    depth = np.full((30, 40), 0.8, np.float32)
    T = np.eye(4)
    T[0, 3] = 0.025  # a 25 mm baseline
    pc, uv = color_cloud(depth, kd, k, np.zeros(5), T, stride=1)
    assert pc.shape == (1200, 3) and uv.shape == (1200, 2)
    assert np.allclose(pc[:, 2], 0.8) and np.allclose(pc[:, 0].min(), (0 - 20) / 400.0 * 0.8 + 0.025)
    # every point projects to its own colour pixel
    assert np.allclose(uv[:, 0], k[0, 0] * pc[:, 0] / pc[:, 2] + k[0, 2], atol=1e-6)
    pc2, _uv2 = color_cloud(depth, kd, k, np.zeros(5), T, stride=2)
    assert pc2.shape[0] == 300
