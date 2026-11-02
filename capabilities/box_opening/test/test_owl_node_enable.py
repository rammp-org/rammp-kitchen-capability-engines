"""The OWL node infers on the enable edge, not at its next 0.5 s tick.

Pure unit test: the node is built with __new__, no rclpy.init, no model.
"""

import time
import types

import numpy as np

from box_opening.perception import owl_node
from box_opening.perception.owl_source import BLIND, HEARTBEAT


class _Stamp:
    def __init__(self, sec):
        self.sec, self.nanosec = sec, 0


class _Grab:
    def __init__(self):
        self.color = np.zeros((4, 4, 3), np.uint8)
        self.color_stamp = _Stamp(1)


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(list(msg.data))


class _Timer:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _Clock:
    def now(self):
        return types.SimpleNamespace(nanoseconds=int(2e9))


def _node(monkeypatch, calls):
    n = owl_node.OwlDetector.__new__(owl_node.OwlDetector)
    n.grab, n.pub, n._timer = _Grab(), _Pub(), _Timer()
    n._last_stamp = None
    n._last_new_frame_t = time.monotonic()
    n._last_resub_t = 0.0
    n._enabled_until = 0.0
    n._proc = n._model = None
    n.cfg = types.SimpleNamespace(owl_queries=["box"])
    n.min_score = 0.1
    n.camera = "scene"
    monkeypatch.setattr(n, "get_clock", lambda: _Clock(), raising=False)
    monkeypatch.setattr(owl_node, "owl_detect", lambda *a, **k: calls.append(1) or [(0.5, (1, 2, 3, 4))])
    return n


def _enable(n, on):
    n._on_enable(types.SimpleNamespace(data=on))


def test_a_rising_enable_infers_the_newest_frame_at_once(monkeypatch):
    calls = []
    n = _node(monkeypatch, calls)
    n._tick()  # idle: heartbeat only, the stamp is recorded
    assert calls == [] and n.pub.sent[-1][4] == HEARTBEAT
    n.grab.color_stamp = _Stamp(2)
    _enable(n, True)
    assert calls == [1] and n._timer.resets == 1
    assert n.pub.sent[-1][:5] == [1.0, 2.0, 3.0, 4.0, 0.5]


def test_a_repeated_enable_is_not_an_edge(monkeypatch):
    calls = []
    n = _node(monkeypatch, calls)
    n._tick()
    n.grab.color_stamp = _Stamp(2)
    _enable(n, True)
    n.grab.color_stamp = _Stamp(3)  # a new frame, but the window is already open
    _enable(n, True)
    assert calls == [1] and n._timer.resets == 1 and n.enabled


def test_a_disable_does_not_infer(monkeypatch):
    calls = []
    n = _node(monkeypatch, calls)
    n._tick()
    n.grab.color_stamp = _Stamp(2)
    _enable(n, False)
    assert calls == [] and not n.enabled and n._timer.resets == 0


def test_an_enable_with_no_new_frame_waits_without_blind(monkeypatch):
    calls = []
    n = _node(monkeypatch, calls)
    n._tick()
    sent = len(n.pub.sent)
    _enable(n, True)  # same frame as the idle tick saw
    assert calls == [] and n.enabled
    assert all(m[4] != BLIND for m in n.pub.sent[sent:])
