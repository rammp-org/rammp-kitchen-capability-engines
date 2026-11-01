"""The wrist D405 helpers box-opening owns since RAMMP-CuRobo v1.0.0 dropped
perception: the fix-agreement gate, quaternion math, the grabber's streams."""

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image

from box_opening.perception.d405 import (
    D405Grabber,
    mat_to_quat_xyzw,
    quat_to_mat,
    stable_fix,
)

# the camera sheppy runs (rammp-deployments december_2026: the `wrist_camera`
# node), not a driver this repo starts
COLOR = "/wrist_camera/color/image_raw"
DEPTH = "/wrist_camera/aligned_depth_to_color/image_raw"
INFO = "/wrist_camera/color/camera_info"


def test_three_agreeing_sightings_commit_their_median():
    samples = [
        ([0.450, 0.000, 0.085], 1.0),
        ([0.452, 0.001, 0.086], 1.1),
        ([0.451, -0.001, 0.084], 1.2),
    ]
    assert stable_fix(samples, tol=0.015, n=3) == pytest.approx([0.451, 0.0, 0.085])


def test_one_sighting_out_of_tolerance_blocks_the_commit():
    samples = [
        ([0.450, 0.0, 0.085], 1.0),
        ([0.451, 0.0, 0.085], 1.1),
        ([0.480, 0.0, 0.085], 1.2),
    ]
    assert stable_fix(samples, tol=0.015, n=3) is None


def test_only_the_newest_n_sightings_vote():
    samples = [([0.600, 0.0, 0.085], 0.5)] + [([0.450, 0.0, 0.085], t) for t in (1.0, 1.1, 1.2)]
    assert stable_fix(samples, tol=0.015, n=3) == pytest.approx([0.450, 0.0, 0.085])


def test_too_few_sightings_do_not_commit():
    assert stable_fix([([0.45, 0.0, 0.085], 1.0)] * 2, tol=0.015, n=3) is None


def test_the_wrist_mounts_half_turn_about_the_lens_axis():
    # mount_quat_xyzw [0, 0, 1, 0]: 180 deg about z flips x and y
    assert quat_to_mat(0.0, 0.0, 1.0, 0.0) == pytest.approx(np.diag([-1.0, -1.0, 1.0]))


def test_quarter_turn_about_x():
    s = np.sqrt(0.5)
    want = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert quat_to_mat(s, 0.0, 0.0, s) == pytest.approx(want)


def test_an_unnormalized_quaternion_is_normalized():
    assert quat_to_mat(0.0, 0.0, 2.0, 0.0) == pytest.approx(np.diag([-1.0, -1.0, 1.0]))


@pytest.mark.parametrize(
    "q",
    [[0.0, 0.0, 1.0, 0.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], [0.1, -0.3, 0.2, 0.927]],
)
def test_matrix_back_to_quaternion_is_the_same_rotation(q):
    q = np.asarray(q) / np.linalg.norm(q)
    back = mat_to_quat_xyzw(quat_to_mat(*q))
    assert quat_to_mat(*back) == pytest.approx(quat_to_mat(*q))


class _Node:
    """Records the subscriptions a grabber makes (tf2's listener included)."""

    def __init__(self):
        self.subscribed = {}

    def create_subscription(self, msg_type, topic, callback, qos, **_kw):
        self.subscribed[topic] = callback
        return object()

    def destroy_subscription(self, _sub):
        # tf2's listener unregisters at GC: without this the teardown noise
        # lands in every run's output
        return True


def test_the_grabber_reads_the_aligned_streams_of_the_configured_camera():
    node = _Node()
    D405Grabber(node, need_depth=True)
    assert {COLOR, DEPTH, INFO} <= set(node.subscribed)


def test_a_colour_only_grabber_leaves_depth_alone():
    node = _Node()
    g = D405Grabber(node, need_depth=False)
    assert DEPTH not in node.subscribed
    assert COLOR in node.subscribed
    assert "ALIGNED" not in " ".join(g.missing())


def test_frames_land_as_bgr_and_metres_and_complete_the_streams():
    node = _Node()
    g = D405Grabber(node, need_depth=True)
    assert len(g.missing()) == 3

    img = Image()
    img.height, img.width, img.encoding = 1, 2, "rgb8"
    img.data = bytes([10, 20, 30, 40, 50, 60])
    img.header.stamp.sec = 7
    node.subscribed[COLOR](img)
    assert g.color.tolist() == [[[30, 20, 10], [60, 50, 40]]]
    assert g.color_stamp.sec == 7

    d = Image()
    d.height, d.width, d.encoding = 1, 2, "16UC1"
    d.data = np.array([1500, 250], dtype=np.uint16).tobytes()
    node.subscribed[DEPTH](d)
    assert g.depth == pytest.approx(np.array([[1.5, 0.25]]))

    info = CameraInfo()
    info.k = [600.0, 0.0, 424.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0]
    node.subscribed[INFO](info)
    assert g.k[0, 0] == 600.0 and g.k[1, 2] == 240.0
    assert g.missing() == []


def test_each_image_is_kept_with_its_own_stamp_and_snapshots_read_them_once():
    """Colour and depth are separate streams: each is stored with its own
    stamp as one pair, so a consumer can tell a stalled or lagging depth
    from the depth of the colour frame it judges."""
    node = _Node()
    g = D405Grabber(node, need_depth=True)
    s = g.snapshot()
    assert s.color is None and s.color_stamp is None and s.depth is None and s.depth_stamp is None

    img = Image()
    img.height, img.width, img.encoding = 1, 1, "bgr8"
    img.data = bytes([1, 2, 3])
    img.header.stamp.sec, img.header.stamp.nanosec = 7, 100
    node.subscribed[COLOR](img)
    assert g.depth_stamp is None and any(m.startswith("depth") for m in g.missing())

    d = Image()
    d.height, d.width, d.encoding = 1, 1, "16UC1"
    d.data = np.array([500], dtype=np.uint16).tobytes()
    d.header.stamp.sec, d.header.stamp.nanosec = 7, 200
    node.subscribed[DEPTH](d)
    assert not any(m.startswith("depth") for m in g.missing())
    s = g.snapshot()
    assert s.color.tolist() == [[[1, 2, 3]]] and (s.color_stamp.sec, s.color_stamp.nanosec) == (7, 100)
    assert s.depth == pytest.approx(np.array([[0.5]])) and (s.depth_stamp.sec, s.depth_stamp.nanosec) == (7, 200)
    assert s.color is g.color and s.depth is g.depth and s.depth_stamp is g.depth_stamp

    # a new colour frame changes the colour pair only, and a snapshot taken
    # before it keeps the frame it read
    img2 = Image()
    img2.height, img2.width, img2.encoding = 1, 1, "bgr8"
    img2.data = bytes([4, 5, 6])
    img2.header.stamp.sec = 8
    node.subscribed[COLOR](img2)
    assert g.color_stamp.sec == 8 and g.color.tolist() == [[[4, 5, 6]]] and g.depth_stamp.nanosec == 200
    assert s.color_stamp.sec == 7 and s.color.tolist() == [[[1, 2, 3]]]


def test_frame_snapshot_reads_a_grabber_without_a_snapshot():
    from types import SimpleNamespace

    from box_opening.perception.d405 import MISSING, frame_snapshot

    fake = SimpleNamespace(color=1, color_stamp=2, depth=3, k=4)
    s = frame_snapshot(fake)
    assert (s.color, s.color_stamp, s.depth, s.k, s.dist) == (1, 2, 3, 4, None)
    assert s.depth_stamp is MISSING  # it keeps none: nothing to pair
    fake.depth_stamp = None
    assert frame_snapshot(fake).depth_stamp is None  # a real "no depth yet": the frame waits
