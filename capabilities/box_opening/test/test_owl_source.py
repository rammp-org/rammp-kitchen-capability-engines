"""The OWL bbox topic contract."""

import numpy as np
import pytest


def test_top_boxes_are_best_first_above_the_floor():
    from box_opening.perception.owl_source import bboxes_in_msg, top_boxes

    scores = [0.10, 0.255, 0.24, 0.30]
    boxes = [[0, 0, 1, 1], [775, 494, 874, 574], [923, 517, 1112, 608], [5, 5, 6, 6]]
    top = top_boxes(scores, [0] * 4, boxes, 0.12, k=3)
    assert [s for s, _b in top] == [0.30, 0.255, 0.24]
    m = []
    for s, b in top:
        m += [float(v) for v in b] + [s, 0.4]
    got = bboxes_in_msg(m)
    assert len(got) == 3 and got[0][:4] == [5.0, 5.0, 6.0, 6.0] and got[1][4] == 0.255
    assert bboxes_in_msg([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) == []  # a heartbeat
    assert bboxes_in_msg(None) == []


class _FakeNode:
    """Records publishers and what they publish."""

    def __init__(self):
        self.pubs = []  # (msg_type, topic, qos)
        self.published = []  # (topic, msg)

    def create_publisher(self, msg_type, topic, qos):
        node = self

        class Pub:
            def publish(self, msg):
                node.published.append((topic, msg))

        self.pubs.append((msg_type, topic, qos))
        return Pub()


def test_the_mission_tells_the_detectors_which_container_to_look_for(tmp_path):
    """The owl_detector nodes are started by the launch (sheppy's
    box_opening node) with no container of their own, so they loaded
    oxo_pop.yaml's prompts whatever the mission was run with:
    `press_demo --container ankou_pink.yaml` changed the geometry and the
    aim while both detectors kept looking for a white square box
    (2026-09-22). The mission now announces its container on ONE latched
    topic both instances read; the path is absolute, so a node started
    from any directory opens the same file."""
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
    from std_msgs.msg import String

    from box_opening.perception.owl_source import CONTAINER_TOPIC, announce_container, container_changed

    assert CONTAINER_TOPIC.startswith("/box_opening/")
    yaml = tmp_path / "pink.yaml"
    yaml.write_text("x: 1\n")
    node = _FakeNode()
    announce_container(node, str(yaml))
    ((msg_type, topic, qos),) = node.pubs
    assert msg_type is String and topic == CONTAINER_TOPIC
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL and qos.reliability == ReliabilityPolicy.RELIABLE
    ((t, m),) = node.published
    assert t == CONTAINER_TOPIC and m.data == str(yaml.resolve())
    # the node reloads only on a real change; the same file spelled two ways is no change
    assert container_changed(str(yaml.resolve()), str(tmp_path / "." / "pink.yaml")) is False
    assert container_changed(str(yaml.resolve()), str(tmp_path / "other.yaml")) is True
    assert container_changed(str(yaml.resolve()), "") is False  # an empty announcement is ignored


def test_a_live_node_with_no_frames_says_so_instead_of_reading_as_dead():
    """Bench 2026-09-22: the wrist owl_detector had been up for 400 s with
    no camera frame ever delivered (the camera re-enumerated as it
    started). It heartbeated only after a NEW frame, so the mission read
    "owl_detector node not running" — the wrong diagnosis, and the one
    that hides a camera problem. A frame-less node now heartbeats BLIND
    (score -2) every tick, and the rung declines at once, naming it."""
    from types import SimpleNamespace

    from box_opening.perception.owl_source import (
        BLIND,
        HEARTBEAT,
        OwlRung,
        bboxes_in_msg,
        classify_bbox_msg,
    )

    now = 100.0
    assert classify_bbox_msg([0, 0, 0, 0, HEARTBEAT, now - 0.2], now) == "alive"
    assert classify_bbox_msg([0, 0, 0, 0, BLIND, now - 0.2], now) == "blind"
    assert classify_bbox_msg([0, 0, 0, 0, BLIND, now - 9.0], now) == "stale"
    assert bboxes_in_msg([0, 0, 0, 0, BLIND, now]) == []  # never a box
    rung = OwlRung.__new__(OwlRung)
    rung.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now * 1e9)))
    )
    rung.camera = "wrist"
    rung.latest = [0, 0, 0, 0, BLIND, now - 0.1]
    rung.latest_box = None
    roi, why = rung(None, None)
    assert roi is None and "no wrist frame" in why and "not running" not in why


def test_blind_means_no_frames_for_a_while_not_one_quiet_tick():
    """Bench 2026-09-23: two runs said "OWL node is live but has had no
    wrist frame" while the node's own log showed frames flowing. An
    inference (0.65 s) outlasts the 0.5 s tick, so the next tick fires
    before a new frame lands — and the node sent BLIND on every such tick
    (added 2026-09-22). The mission reads the LATEST message: one quiet
    tick ended the wait. BLIND now means what it says."""
    from box_opening.perception.owl_source import BLIND_AFTER_S, blind_heartbeat_due

    assert not blind_heartbeat_due(0.7)  # one overrun tick
    assert not blind_heartbeat_due(BLIND_AFTER_S - 0.1)
    assert blind_heartbeat_due(BLIND_AFTER_S + 0.1)


def test_a_heartbeat_right_after_a_box_does_not_hide_it():
    """... and the same race hid the scene camera's boxes: a heartbeat
    landing milliseconds after a bbox replaced it as `latest` before the
    locate loop (polling every 20 ms) looked — "the scene OWL saw no box in
    3.0 s" with the box in view. The rung keeps the newest BOX apart from
    the newest message."""
    from types import SimpleNamespace

    from box_opening.perception.owl_source import HEARTBEAT, OwlRung

    now = [100.0]
    rung = OwlRung.__new__(OwlRung)
    rung.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now[0] * 1e9)))
    )
    rung.latest = rung.latest_box = None
    rung.enabled = False
    rung.watcher_holder = {}
    rung._cb(SimpleNamespace(data=[10.0, 20.0, 60.0, 80.0, 0.45, 0.1]))
    rung._cb(SimpleNamespace(data=[0.0, 0.0, 0.0, 0.0, HEARTBEAT, 0.0]))
    assert rung.latest[4] == HEARTBEAT  # the newest message is the heartbeat ...
    box = rung.fresh_box()
    assert box is not None and box[4] == 0.45  # ... and the box is still there
    now[0] += 10.0
    assert rung.fresh_box() is None  # but not forever


def test_each_camera_has_its_own_floor_in_the_container_config():
    """The scene instance's floor (0.12) lived in the launch file, out of
    reach of anything that tunes detection; it is the container's now, next
    to the prompts it goes with. The canister's is 0.05 since 2026-10-27,
    when its pale lid scored 0.07-0.10 and two runs saw no box in 15 s."""
    from box_opening.models.container import load_press_demo
    from box_opening.perception.owl_source import owl_floor

    cfg = load_press_demo("capabilities/box_opening/config/containers/ankou_pink.yaml")
    assert owl_floor(cfg, "scene") == cfg.owl_min_score_scene == 0.05
    assert owl_floor(cfg, "wrist") == cfg.owl_min_score == 0.18
    oxo = load_press_demo("capabilities/box_opening/config/containers/oxo_pop.yaml")
    assert owl_floor(oxo, "scene") == oxo.owl_min_score_scene == 0.12


def test_detection_is_one_function_the_node_and_an_offline_replay_share():
    """owl_detect: processor + model + frame -> the top boxes above the
    floor, best first — what the node publishes, and what an evaluator
    replaying recorded frames must compute the same way."""
    import torch

    from box_opening.perception.owl_source import owl_detect

    seen = {}

    class Batch(dict):
        def to(self, device):
            seen["device"] = device
            return self

    class Proc:
        def __call__(self, text, images, return_tensors):
            seen["text"], seen["shape"] = text, images[0].shape
            return Batch()

        def post_process_object_detection(self, out, threshold, target_sizes):
            seen["threshold"] = threshold
            return [
                {
                    "scores": torch.tensor([0.20, 0.47, 0.13]),
                    "labels": torch.tensor([0, 1, 0]),
                    "boxes": torch.tensor([[1.0, 2, 3, 4], [10, 20, 60, 80], [5, 5, 9, 9]]),
                }
            ]

    rgb = np.zeros((720, 1280, 3), np.uint8)
    top = owl_detect(Proc(), lambda **kw: "out", rgb, ["a pink lid", "a round pink lid"], 0.15, device="cpu")
    assert [s for s, _b in top] == pytest.approx([0.47, 0.20])  # best first; 0.13 is under the floor
    assert top[0][1] == [10, 20, 60, 80]
    assert (
        seen["text"] == [["a pink lid", "a round pink lid"]] and seen["threshold"] == 0.15 and seen["device"] == "cpu"
    )


def test_owl_is_shown_the_frame_as_it_comes_and_its_boxes_stay_in_its_pixels():
    """Since rammp-deployments #133 (2026-10-27) the OAKs rotate their frames
    upright on the device: OWLv2 is shown each frame as published, nothing
    turned, and its boxes are that frame's own pixels."""
    import torch

    from box_opening.perception.owl_source import owl_detect

    seen = {}

    class Batch(dict):
        def to(self, device):
            return self

    class Proc:
        def __call__(self, text, images, return_tensors):
            seen["image"] = images[0]
            return Batch()

        def post_process_object_detection(self, out, threshold, target_sizes):
            return [
                {
                    "scores": torch.tensor([0.44]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.tensor([[10.0, 20, 60, 80]]),
                }
            ]

    rgb = np.zeros((400, 640, 3), np.uint8)
    rgb[0, 0] = 255  # the published frame's top-left pixel
    top = owl_detect(Proc(), lambda **kw: "out", rgb, ["a pink lid"], 0.12, device="cpu")
    assert seen["image"][0, 0, 0] == 255  # shown as published
    assert top[0][0] == pytest.approx(0.44) and top[0][1] == [10, 20, 60, 80]  # in the frame's own pixels
    import inspect

    assert "upside_down" not in inspect.signature(owl_detect).parameters  # nothing turns frames any more
