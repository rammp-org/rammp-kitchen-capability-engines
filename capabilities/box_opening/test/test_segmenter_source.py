"""The scene camera's box from sheppy's scene segmenter: which of its boxes
become candidates, where the lid band above a holder box lies, and when its
messages may be used at all. Pure: no ROS graph, no model."""

from types import SimpleNamespace

import pytest

from box_opening.perception import segmenter_source as sg

WH = (640, 400)
FOOD = ("food container",)
# what RF-DETR-S published on the 10-27 16:24 frame the OWL missed ("saw no
# box in 15.0 s"), and the microwave it boxes on 10-22 18:32
HOLDER_162417 = (157.75, 216.38, 255.62, 260.5)
OBJECTS_162417 = [
    ("food container", 0.881, HOLDER_162417),
    ("drink cup", 0.593, (378.0, 127.12, 448.75, 237.88)),
    ("microwave", 0.889, (201.0, 112.19, 418.75, 244.5)),
    ("table", 0.209, (1.41, 206.25, 638.0, 386.75)),
]


def test_only_the_containers_classes_become_candidates():
    cands = sg.segmenter_candidates(OBJECTS_162417, FOOD, WH)
    assert len(cands) == 1  # the microwave outscores it and is no candidate at all
    assert cands[0].score == pytest.approx(0.881) and cands[0].det_box == pytest.approx(HOLDER_162417)


def test_a_holder_box_becomes_the_band_where_its_lid_stands():
    (c,) = sg.segmenter_candidates(OBJECTS_162417, FOOD, WH)
    assert c.kind == "band"
    assert c.box == pytest.approx((171.45, 196.53, 241.92, 220.79), abs=0.01)
    assert c.holder == pytest.approx(HOLDER_162417)  # the band's anchor: the holder itself
    band, anchored = sg.lid_band((112, 217, 216, 260), WH)  # 10-27 16:22's holder
    assert anchored and band == pytest.approx((126.56, 197.65, 201.44, 221.30), abs=0.01)


def test_a_lid_boxed_on_its_holder_goes_as_it_is_and_a_cut_holder_is_grown():
    """Flatbot 2026-10-22 18:32: RF-DETR boxed the holder, cut by the frame's
    left edge, and the lid on it as a second bowl."""
    holder, lid = (0.0, 227.38, 159.0, 293.0), (25.31, 204.12, 132.38, 242.38)
    cands = sg.segmenter_candidates([("food container", 0.699, holder), ("food container", 0.314, lid)], FOOD, WH)
    assert [c.kind for c in cands] == ["lid", "grown"]
    assert cands[0].box == pytest.approx(lid) and cands[0].holder is None and cands[0].score == pytest.approx(0.314)
    assert cands[1].box == pytest.approx((0.0, 194.57, 159.0, 293.0), abs=0.01)
    assert cands[1].holder is None  # the holder lies inside a grown box: nothing to anchor
    assert cands[1].det_box == pytest.approx(holder)
    assert sg.stacked_on(lid, [holder]) and not sg.stacked_on(holder, [lid])


def test_the_three_best_are_offered_clamped_to_the_frame():
    objs = [
        ("food container", s, (100.0 + 120 * i, 200.0, 180.0 + 120 * i, 250.0))
        for i, s in enumerate((0.3, 0.9, 0.5, 0.7))
    ]
    cands = sg.segmenter_candidates(objs, FOOD, WH)
    assert [c.score for c in cands] == [0.9, 0.7, 0.5]
    edge = sg.segmenter_candidates([("food container", 0.6, (560.0, 200.0, 655.0, 260.0))], FOOD, WH)
    assert edge[0].det_box[2] == 640.0 and edge[0].kind == "grown"  # past the right edge: clamped, and cut there
    as_is = sg.segmenter_candidates(objs, FOOD, WH, box_kind="as_is")
    assert [c.kind for c in as_is] == ["as_is"] * 3
    assert as_is[0].box == pytest.approx(objs[1][2]) and as_is[0].holder is None
    with pytest.raises(ValueError):
        sg.segmenter_candidates(objs, FOOD, WH, box_kind="lid")


def _obj(cls="food container", score=0.88, box=HOLDER_162417, state=0):
    return SimpleNamespace(perception_class=cls, score=score, box=list(box), state=state)


def _msg(objects, stamp=(100, 5), frame_id="scene_camera_color_optical_frame", wh=WH):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp[0], nanosec=stamp[1]), frame_id=frame_id),
        objects=objects,
        label_image=SimpleNamespace(width=wh[0], height=wh[1]),
    )


def test_a_message_keeps_only_what_can_be_lifted():
    nan = float("nan")
    m = _msg(
        [
            _obj(),
            _obj(state=1),  # occluded: tracked, not seen this frame
            _obj(box=(nan, 216.0, 255.0, 260.0)),
            _obj(box=(250.0, 216.0, 150.0, 260.0)),  # empty
            _obj(score=1.2),
            _obj(box=(600.0, 216.0, 645.0, 260.0)),  # beyond the 640 px label image
            _obj(cls="drink cup"),
        ]
    )
    f = sg.frame_of(m, FOOD, now_s=12.5)
    assert f.objects == (("food container", pytest.approx(0.88), pytest.approx(HOLDER_162417)),)
    assert f.n_objects == 7 and f.recv_s == 12.5
    assert f.stamp_ns == 100 * 1_000_000_000 + 5 and f.frame_id == "scene_camera_color_optical_frame" and f.wh == WH
    # no masks sent (a 0 x 0 label image): nothing to check the box against
    f = sg.frame_of(_msg([_obj(box=(600.0, 216.0, 645.0, 260.0))], wh=(0, 0)), FOOD, now_s=0.0)
    assert len(f.objects) == 1 and f.wh == (0, 0)
    assert sg.stamp_ns(None) is None


class _Node:
    def __init__(self):
        self.subs = []

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs.append((msg_type, topic, cb))
        return object()


def _rung(import_types=None):
    class Detected:
        @staticmethod
        def get_fields_and_field_types():
            return {"perception_class": "string", "box": "float[4]", "size": "geometry_msgs/Vector3"}

    node = _Node()
    rung = sg.SegmenterRung(node, FOOD, import_types=import_types or (lambda: (object, Detected)))
    rung.t_subscribed = 0.0
    return rung, node


def _frame(recv_s, stamp_s=10.0, frame_id="scene_camera_color_optical_frame", wh=WH):
    return sg.SegFrame(recv_s, int(stamp_s * 1e9), frame_id, wh, (), 0)


def test_the_segmenter_is_live_only_from_the_scene_camera_at_its_size_and_fresh():
    rung, node = _rung()
    assert rung.available and node.subs[0][1] == sg.SEGMENTER_TOPIC == "/perception/object_detections"
    cam = ("scene_camera_color_optical_frame", WH, int(10.2e9))  # the grabber's own newest colour frame
    assert rung.state(0.5, *cam) == "waiting"  # its first second: no word yet is no failure yet
    assert rung.state(1.1, *cam) == "silent"
    assert "SILENT" in rung.describe("silent") and "scene_segmenter: rfdetr" in rung.describe("silent")
    rung.latest = _frame(recv_s=5.0)
    assert rung.state(5.3, *cam) == "live"
    assert rung.state(6.1, *cam) == "silent"  # nothing for over a second since the last
    rung.latest = _frame(recv_s=5.0, frame_id="")  # the mock: an empty frame_id and its own clock
    assert rung.state(5.3, *cam) == "wrong_camera"
    assert "the mock?" in rung.describe("wrong_camera", *cam)
    rung.latest = _frame(recv_s=5.0, frame_id="wrist_camera_color_optical_frame")
    assert rung.state(5.3, *cam) == "wrong_camera"
    assert "'wrist_camera_color_optical_frame', not the scene camera" in rung.describe("wrong_camera", *cam)
    rung.latest = _frame(recv_s=5.0, wh=(1280, 720))
    assert rung.state(5.3, *cam) == "wrong_size" and "1280x720" in rung.describe("wrong_size", *cam)
    rung.latest = _frame(recv_s=5.0, wh=(0, 0))  # no masks: no size to compare
    assert rung.state(5.3, *cam) == "live"
    rung.latest = _frame(recv_s=5.0, stamp_s=10.0)
    ahead = (cam[0], WH, int(11.5e9))  # the camera 1.5 s past the segmenter's newest frame
    assert rung.state(5.3, *ahead) == "stale" and "1.5 s older" in rung.describe("stale", *ahead)
    # an explicit frame is judged, not the newest
    assert rung.state(5.3, *cam, frame=None) == "silent"


class _GraphNode(_Node):
    """A node whose graph has `n[0]` publishers on a topic, counting the queries."""

    def __init__(self, n):
        super().__init__()
        self.n, self.asked = n, 0

    def count_publishers(self, topic):
        assert topic == sg.SEGMENTER_TOPIC
        self.asked += 1
        return self.n[0]


def test_a_segmenter_nothing_publishes_is_absent_at_once_not_after_a_silent_second():
    """A profile that does not select the segmenter: no message will ever
    come, and waiting out SEG_SILENT_S before the scene OWL is asked cost the
    first look its first second. One that has a publisher still gets its
    first second; a node that cannot say keeps the timer."""
    n = [0]
    node = _GraphNode(n)
    rung = sg.SegmenterRung(node, FOOD, import_types=lambda: (object, type("D", (), {"size": None})))
    rung.t_subscribed = 0.0
    cam = ("scene_camera_color_optical_frame", WH, int(10.2e9))
    assert rung.state(0.0, *cam) == "absent"
    d = rung.describe("absent")
    assert d.startswith("ABSENT: nothing publishes /perception/object_detections") and "scene_segmenter: rfdetr" in d
    n[0] = 1  # it comes up: the graph is asked again only after SEG_GRAPH_S
    assert rung.state(0.2, *cam) == "absent" and node.asked == 1
    assert rung.state(0.0 + sg.SEG_GRAPH_S, *cam) == "waiting" and node.asked == 2
    assert rung.state(1.1, *cam) == "silent"  # published, but nothing came in its first second
    rung.latest = _frame(recv_s=1.2)
    asked = node.asked
    assert rung.state(1.3, *cam) == "live" and node.asked == asked  # a message: the graph is not asked
    rung, _node = _rung()  # a node with no graph to ask: the timer alone
    assert rung.state(0.5, *cam) == "waiting" and rung.state(1.1, *cam) == "silent"


def test_a_message_the_rung_cannot_read_never_reaches_the_perception_executor():
    """An exception in the callback would end the executor's spin, and with
    it every camera and OWL rung the mission's perception thread serves."""
    rung, node = _rung()
    _type, _topic, cb = node.subs[0]
    cb(_msg([_obj()]))
    good = rung.latest
    cb(SimpleNamespace(header=SimpleNamespace(stamp=None, frame_id="x"), objects=None, label_image=None))
    assert rung.latest is good and rung.count == 1 and "TypeError" in rung.cb_error
    assert "could not be read: TypeError" in rung.describe("silent")


def test_a_segmenter_this_install_cannot_read_is_unavailable_and_says_why():
    def missing():
        raise ImportError("No module named 'rammp_perception_interfaces'")

    rung, node = _rung(import_types=missing)
    assert not rung.available and node.subs == [] and rung.state(0.0) == "unavailable"
    assert "does not import" in rung.why and rung.describe("unavailable") == rung.why

    class OldDetected:
        @staticmethod
        def get_fields_and_field_types():
            return {"perception_class": "string", "box": "float[4]"}  # before rammp-interfaces-ros2#16

    rung, node = _rung(import_types=lambda: (object, OldDetected))
    assert not rung.available and node.subs == [] and "no size" in rung.why


def test_the_installed_interfaces_are_read_as_the_running_node_publishes_them():
    """The callback on a real ObjectDetections, as rammp-perception's
    detection node builds it (header copied from the camera image, every
    object VISIBLE, a mono16 label image the image's size)."""
    msgs = pytest.importorskip("rammp_perception_interfaces.msg")
    from sensor_msgs.msg import Image

    node = _Node()
    rung = sg.SegmenterRung(node, FOOD)
    assert rung.available, rung.why
    msg_type, topic, cb = node.subs[0]
    assert msg_type is msgs.ObjectDetections and topic == sg.SEGMENTER_TOPIC
    m = msgs.ObjectDetections()
    m.header.stamp.sec, m.header.stamp.nanosec = 1761582257, 5
    m.header.frame_id = "scene_camera_color_optical_frame"
    m.label_image = Image(height=400, width=640, encoding="mono16")
    for cls, score, box in OBJECTS_162417:
        m.objects.append(
            msgs.DetectedObject(
                perception_class=cls,
                score=score,
                state=msgs.DetectedObject.STATE_VISIBLE,
                box=[float(v) for v in box],
                mask_id=1,
            )
        )
    cb(m)
    f = rung.latest
    assert rung.count == 1 and f.n_objects == 4 and f.wh == WH and f.frame_id == "scene_camera_color_optical_frame"
    assert [o[0] for o in f.objects] == ["food container"] and f.objects[0][2] == pytest.approx(HOLDER_162417)
    assert f.stamp_ns == 1761582257 * 1_000_000_000 + 5
