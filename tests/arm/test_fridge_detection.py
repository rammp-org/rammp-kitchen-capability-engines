"""The fridge's detection: the handle from SAM 3's masks and the depth, the door from it, the run's logs and bag.

The planning is tests/arm/test_fridge_curobo.py's.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from door_common import common as dc
from door_common import open_door as shared
from fridge import constants, open_door


@pytest.fixture
def perception(monkeypatch):
    """door_common.perception and the fridge's, without open3d / message_filters (not in these
    tests' image)."""
    for name in ("open3d", "message_filters"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    import door_common.perception
    from fridge import perception as fridge_perception

    return door_common.perception, fridge_perception


# a level camera 0.5 m up, looking along the arm's +x: image right is the arm's -y, image down is -z
CAM_K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 200.0], [0.0, 0.0, 1.0]])
CAM_T = np.eye(4)
CAM_T[:3, :3] = np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
CAM_T[2, 3] = 0.5


def _door_scene(door_m):
    """Depth (mm) of a door `door_m` from the camera, a 2 cm wide vertical bar on it at 0.50 m
    (columns 400-415, rows 100-300), and SAM 3's mask of the bar, a little wider than it (so it
    takes in some door)."""
    depth = np.full((400, 640), door_m * 1000.0, np.float32)
    depth[100:300, 400:416] = 500.0
    mask = np.zeros((400, 640), bool)
    mask[100:300, 396:420] = True
    return depth, mask


def test_the_handle_is_the_bar_front_from_its_own_depth_whatever_the_door_reads(perception):
    _, fp = perception
    for door_m in (0.55, 1.05):  # a true door; a glossy one returning its reflection
        depth, mask = _door_scene(door_m)
        h, _, n = fp.HandleDetector()._bar_front(mask, depth, CAM_K, CAM_T[2, :3])
        assert h[2] == pytest.approx(0.50, abs=0.005)  # on the bar, not the door behind it
        assert h[0] == pytest.approx((407.5 - 320) / 400 * 0.5, abs=0.005)
        assert n == 16 * 200


def _plane(door_m):
    """door_common's _door_plane result: a plane facing the camera `door_m` away (camera frame)."""
    return (
        None,
        None,
        np.array([0.0, 0.0, -1.0]),
        door_m,
        CAM_T[2, :3],
        CAM_T[:3, :3],
        CAM_T[:3, 3],
    )


@pytest.mark.parametrize(
    ("planes", "picked"),
    [
        ([0.55], 0.55),  # a true door: the first plane, as before
        ([1.05, 0.55], 0.55),  # its reflection first: refused, the door found next
        ([0.503, 0.55], 0.55),  # flush with the handle's front (the handle, its mount): refused
        ([1.05, 1.06], None),  # only the reflection: no door plane, and the look says why
    ],
)
def test_a_door_plane_must_be_where_the_handle_can_stand_on_it(
    perception, monkeypatch, planes, picked
):
    dp, fp = perception
    queue = [_plane(d) for d in planes]
    monkeypatch.setattr(
        dp.HandleDetector, "_door_plane", lambda self, *a: queue.pop(0) if queue else None
    )
    det = fp.HandleDetector(0.0381)
    depth, mask = _door_scene(planes[0])
    det._mask, det._base_T_cam, det._exclude = mask, CAM_T, None
    det._bar, det._bar_extent, det._bar_points = det._bar_front(mask, depth, CAM_K, CAM_T[2, :3])
    got = det._door_plane(None, CAM_K, depth, CAM_T, None, None)
    if picked is None:
        assert got is None
        assert "no door plane fits the handle" in det.last_miss and "+55 cm" in det.last_miss
    else:
        assert got[3] == picked


def test_a_door_plane_facing_sideways_is_not_the_door(perception, monkeypatch):
    dp, fp = perception
    side = (None, None, np.array([1.0, 0.0, 0.0]), -0.43, CAM_T[2, :3], CAM_T[:3, :3], CAM_T[:3, 3])
    queue = [side, _plane(0.55)]  # 11-04: a plane at the handle's standoff, facing sideways
    monkeypatch.setattr(
        dp.HandleDetector, "_door_plane", lambda self, *a: queue.pop(0) if queue else None
    )
    det = fp.HandleDetector()
    depth, mask = _door_scene(0.55)
    det._mask, det._base_T_cam, det._exclude = mask, CAM_T, None
    det._bar, det._bar_extent, det._bar_points = det._bar_front(mask, depth, CAM_K, CAM_T[2, :3])
    assert det._door_plane(None, CAM_K, depth, CAM_T, None, None)[3] == 0.55


def test_door_square_to_arm_takes_the_door_as_facing_the_arm_and_fits_no_plane(
    perception, monkeypatch, tmp_path
):
    dp, fp = perception

    def no_plane(self, *a):
        raise AssertionError("fit a door plane")

    monkeypatch.setattr(dp.HandleDetector, "_door_plane", no_plane)
    depth, mask = _door_scene(1.05)  # the door's depth only its reflection
    det = fp.HandleDetector(door_square_to_arm=True).detect(
        None, CAM_K, depth, CAM_T, handle_mask=mask
    )
    assert det["normal"] == pytest.approx([-1.0, 0.0, 0.0])
    assert det["handle"][0] == pytest.approx(
        0.50, abs=0.005
    )  # the bar's front, 0.5 m ahead of the camera
    assert det["hinge"] is None
    saved = dict(vars(dc)), dict(vars(shared)), dict(vars(constants))  # configure sets all three
    try:  # the node's parameter reaches the detector
        open_door.configure(tmp_path, 30, door_normal="square")
        assert shared.MAKE_DETECTOR().door_square_to_arm
        open_door.configure(tmp_path, 30)
        assert not shared.MAKE_DETECTOR().door_square_to_arm
    finally:
        vars(dc).update(saved[0])
        vars(shared).update(saved[1])
        vars(constants).update(saved[2])
        vars(shared).update(saved[1])
        vars(constants).update(saved[2])


def test_the_door_plane_is_fit_beside_the_handle_on_its_hinge_side(perception):
    _, fp = perception
    depth, mask = _door_scene(0.55)
    det = fp.HandleDetector()
    det._mask, det._base_T_cam, det._exclude = mask, CAM_T, None
    det._bar = det._bar_front(mask, depth, CAM_K, CAM_T[2, :3])[0]
    # looking along +x, the door's hinge (along it from the handle, toward the arm's +y) is to the
    # image's left
    region = det.door_region(np.zeros((400, 640, 3), np.uint8), None, None)
    us = np.nonzero(region)[1]
    assert us.max() < 396 and us.min() >= 396 - 4 - 3 * 24
    assert not (region & mask).any()


def _instances_scene():
    """A frame (door 0.55 m away, the 2 cm bar on it at 0.50 m, 25 cm tall) and SAM 3 instances
    on it: 11-04's wall sign (scoring best), the bench edge (big, lying down) and the bar."""
    depth, bar = _door_scene(0.55)
    sign, wide = np.zeros((400, 640), bool), np.zeros((400, 640), bool)
    sign[40:60, 20:45] = True
    wide[300:340, 100:400] = True
    frame = dict(depth_mm=depth, K=CAM_K, base_T_cam=CAM_T)
    return frame, sign, wide, bar


def test_the_handle_is_the_largest_handle_like_sam3_instance_not_the_best_scoring(perception):
    _, fp = perception
    frame, sign, wide, bar = _instances_scene()
    det = fp.HandleDetector()
    instances = [(0.80, sign, "handle"), (0.62, wide, "handle"), (0.45, bar, "handle")]
    assert det.pick_handle_mask(instances, frame) is bar
    assert det.pick_handle_mask(instances) is None  # no depth: none taken
    assert det.pick_handle_mask([], frame) is None


def test_the_detector_picks_again_after_a_detection(perception, monkeypatch):
    # every look after the first: detect() keeps the last bar's point count (11-05: a name clash)
    _, fp = perception
    frame, sign, wide, bar = _instances_scene()
    det = fp.HandleDetector()
    monkeypatch.setattr(
        fp.door_common.perception.HandleDetector, "detect", lambda *a, **k: {"hinge": 1}
    )
    det.detect(None, CAM_K, frame["depth_mm"], CAM_T, handle_mask=bar)
    assert det.pick_handle_mask([(0.8, sign, "handle"), (0.45, bar, "handle")], frame) is bar


def test_nothing_handle_like_is_a_miss_never_the_best_scoring(perception):
    _, fp = perception
    frame, sign, wide, _ = _instances_scene()
    det = fp.HandleDetector()
    assert det.pick_handle_mask([(0.8, sign, "handle"), (0.6, wide, "handle")], frame) is None
    assert "no SAM 3 instance of 2 is handle-like" in det.last_miss
    assert "2 cm tall" in det.last_miss  # the sign
    assert det.pick_handle_mask([(0.8, sign, "handle")]) is None  # without depth too


def _patch(rows, cols, depth_m, behind_m=None):
    """A frame with a rectangular patch (image rows, cols) `depth_m` away, a wall `behind_m`
    (default 5 cm) behind it, and the patch's mask: the outline SAM 3 would give it."""
    depth = np.full((400, 640), (behind_m or depth_m + 0.05) * 1000.0, np.float32)
    mask = np.zeros((400, 640), bool)
    mask[rows[0] : rows[1], cols[0] : cols[1]] = True
    depth[mask] = depth_m * 1000.0
    return dict(depth_mm=depth, K=CAM_K, base_T_cam=CAM_T), mask


def test_the_handle_limits_are_loose_and_have_no_distance_limit(perception):
    _, fp = perception
    det = fp.HandleDetector()
    # the printed handle (24.8 x 6.4 cm, 7.9 cm off the door) at 45 deg, 0.9 m away: ~24 x 10 cm
    frame, mask = _patch((90, 197), (300, 345), 0.9)
    height, width, _ = det._instance_size(mask, frame)  # 5th-95th percentiles of the outline
    assert (height, width) == (pytest.approx(0.22, abs=0.01), pytest.approx(0.09, abs=0.01))
    assert det.pick_handle_mask([(0.3, mask, "handle")], frame) is mask
    # handle-sized but 1.9 m away: the planner's reach decides, not the picker
    frame, mask = _patch((150, 200), (330, 343), 1.9)
    assert det.pick_handle_mask([(0.3, mask, "handle")], frame) is mask


@pytest.mark.parametrize(
    "rows, cols, depth_m, why",
    [
        ((100, 300), (320, 324), 0.5, "cm wide"),  # a sliver (a door edge, a pole): 25 cm x 5 mm
        ((100, 300), (0, 400), 0.5, "cm wide"),  # the bench edge: 25 cm tall, 50 cm across
        ((180, 200), (300, 340), 0.5, "cm tall"),  # a sign: 2.5 cm tall
        ((0, 400), (300, 340), 0.8, "cm tall"),  # 80 cm tall at 0.8 m: a door frame
        # within every size range, the wrong way up: handles mounted sideways, a door edge
        ((100, 167), (300, 433), 0.6, "sideways"),  # a horizontal bar: 20 cm wide, 10 cm tall
        ((100, 173), (300, 407), 0.6, "sideways"),  # a big lever: 16 x 11 cm
        ((20, 380), (300, 324), 0.5, "too narrow"),  # a door edge: 45 cm tall, 3 cm wide
    ],
)
def test_what_is_not_handle_sized_is_not_the_handle(perception, rows, cols, depth_m, why):
    _, fp = perception
    det = fp.HandleDetector()
    frame, mask = _patch(rows, cols, depth_m)
    assert det.pick_handle_mask([(0.9, mask, "handle")], frame) is None
    assert why in det.last_miss


@pytest.mark.parametrize(
    "height_px, width_px",
    # at 0.6 m: 25 x 6.4 cm (head-on), 25 x 8 (from the side), 19 x 10 (45 deg), the bar alone
    [(165, 43), (165, 53), (127, 67), (127, 40)],
)
def test_the_handle_passes_from_every_side(perception, height_px, width_px):
    _, fp = perception
    det = fp.HandleDetector()
    frame, mask = _patch((100, 100 + height_px), (300, 300 + width_px), 0.6)
    assert det.pick_handle_mask([(0.5, mask, "handle")], frame) is mask, det.last_miss


def test_an_instance_without_depth_is_not_the_handle(perception):
    # handle-shaped, no depth on it (out of range, like 11-04's door across the room)
    _, fp = perception
    det = fp.HandleDetector()
    frame, mask = _patch((100, 265), (300, 343), 0.6)
    frame["depth_mm"][mask] = 0.0
    assert det.pick_handle_mask([(0.9, mask, "handle")], frame) is None
    assert "no depth" in det.last_miss


def test_the_microwave_picks_sam3s_best_instance_as_before(perception):
    from microwave import perception as microwave_perception

    assert not hasattr(microwave_perception.HandleDetector, "pick_handle_mask")


def test_the_fridge_detector_reports_no_hinge(perception, monkeypatch):
    dp, fp = perception
    look = {
        "handle": np.array([0.5, -0.4, 0.3]),
        "normal": np.array([-1.0, 0.0, 0.0]),
        "hinge": np.zeros(3),
    }
    monkeypatch.setattr(dp.HandleDetector, "detect", lambda self, *a, **k: dict(look))
    assert fp.HandleDetector().detect(None, CAM_K, None, CAM_T)["hinge"] is None
    monkeypatch.setattr(dp.HandleDetector, "detect", lambda self, *a, **k: None)
    assert fp.HandleDetector().detect(None, CAM_K, None, CAM_T) is None  # a miss stays a miss


def test_a_look_uses_the_detectors_pick_keeps_the_mask_and_says_why_it_missed(
    tmp_path, monkeypatch
):
    saved_state = (
        dict(vars(dc)),
        dict(vars(shared)),
    )  # door_sim would stub detect_handle out: set up by hand
    open_door.configure(tmp_path)
    frame = dict(
        rgb=np.zeros((4, 4, 3), np.uint8),
        K=CAM_K,
        depth_mm=np.zeros((4, 4)),
        base_T_cam=CAM_T,
        box=None,
    )
    masks = [np.eye(4, dtype=bool), np.ones((4, 4), bool)]
    saved, picked = [], []

    class Cam:
        def wait_for_frames(self, timeout_s):
            return True

        def grab(self):
            return dict(frame)

        def sam3_handle_instances(self):
            return [(0.9, masks[0], "handle"), (0.5, masks[1], "handle")]

    class Detector:
        last_miss = "no door plane fits the handle (handle +55 cm in front)"

        def pick_handle_mask(self, instances, frame=None):
            picked.append(instances)
            return instances[1][1]

        def detect(self, *a, handle_mask=None, **k):
            return None

    monkeypatch.setattr(dc, "camera", lambda: Cam())
    monkeypatch.setattr(shared, "_save_frame", lambda f, tag, attempt, det: saved.append(f))
    monkeypatch.setattr(shared.time, "sleep", lambda s: None)
    shared.configure(MAKE_DETECTOR=Detector, HANDLE_SOURCE="sam3")
    try:
        with pytest.raises(dc.Refused, match=r"NO DETECTION \(1\).*handle \+55 cm in front"):
            shared.detect_handle()
    finally:
        vars(dc).update(saved_state[0])
        vars(shared).update(saved_state[1])
    assert len(picked[0]) == 2
    assert saved and all(f["handle_mask"] is masks[1] for f in saved)


def test_the_tee_logs_as_its_callers_lines(perception, tmp_path):
    import rclpy.logging
    from fridge import motion, node

    tee = node.TeeLogger(rclpy.logging.get_logger("fridge_tee_test"))
    motion._set_console(tmp_path / "console.log")
    try:
        # each from its own line, as callers make them: rclpy sees those lines, not the tee's
        tee.info("line")
        tee.error("line")  # 11-04: a 2nd severity on one tee line crashed every run
        tee.warning("line")
        tee.info("line")
        tee.fatal("line")
        # engine.py's throttled info (under the orchestrator): a filter on one caller's line only
        tee.info("waiting for the orchestrator", throttle_duration_sec=10.0)
        tee.info("line")
    finally:
        motion._set_console(None)
    assert [ln.split()[1] for ln in (tmp_path / "console.log").read_text().splitlines()] == [
        "INFO",
        "ERROR",
        "WARN",
        "INFO",
        "FATAL",
        "INFO",
        "INFO",
    ]


def test_a_sparse_door_plane_is_a_miss_not_a_crash(perception, monkeypatch):
    base, _ = perception
    # two door points 3 m apart along the door: no point near either percentile edge
    plane_pts = np.array([[-1.5, 0.0, 1.0], [1.5, 0.0, 1.0]])
    det = base.HandleDetector()
    monkeypatch.setattr(
        det,
        "_door_plane",
        lambda *a: (
            plane_pts,
            np.zeros((0, 3)),
            np.array([0.0, 0.0, -1.0]),
            1.0,
            np.array([0.0, -1.0, 0.0]),
            np.eye(3),
            np.zeros(3),
        ),
    )
    monkeypatch.setattr(det, "_handle_from_depth", lambda *a, **k: np.array([0.2, 0.0, 0.95]))
    assert det.detect(None, CAM_K, None, CAM_T) is None


def _stamped(sec, nanosec=0):
    from types import SimpleNamespace

    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)))


def _bare_camera(base, sam3):
    """A WristCamera without its ROS plumbing (no cv_bridge here): its buffer and pairing only."""
    import collections
    import threading

    cam = object.__new__(base.WristCamera)
    cam._lock, cam._latest, cam._recent, cam._sam3 = (
        threading.Lock(),
        None,
        collections.deque(),
        sam3,
    )
    cam._frame = lambda color, info, depth: {
        "color": color,
        "depth": depth,
        "rgb": np.zeros((2, 3, 3)),
    }
    return cam


def test_sam3_masks_the_frame_it_segmented_not_one_grabbed_before(perception):
    from types import SimpleNamespace

    base, _ = perception
    # SAM 3 segments the first image after the request reaches it: newer than one grabbed before
    reply = SimpleNamespace(
        success=True,
        message="1 instance",
        detections=SimpleNamespace(
            header=_stamped(10, 500).header,
            objects=[SimpleNamespace(mask_id=1, score=0.8, perception_class="handle")],
            label_image=SimpleNamespace(
                height=2, width=3, step=6, data=np.array([[0, 1, 1], [0, 0, 1]], "<u2").tobytes()
            ),
        ),
    )

    class Client:
        srv_name, srv_type = "/sam3", SimpleNamespace(Request=lambda **k: k)

        def wait_for_service(self, timeout_sec):
            return True

        def call_async(self, req):
            return SimpleNamespace(add_done_callback=lambda cb: cb(None), result=lambda: reply)

    cam = _bare_camera(base, Client())
    cam.sam3_prompts, cam.color_topic, cam.sam3_threshold = ["handle"], "/c", 0.3
    for s, ns in ((10, 0), (10, 500), (11, 0)):  # the grabbed frame, SAM 3's, a later one
        cam._on_frame(_stamped(s, ns), None, f"depth {s}.{ns}")
    instances, frame = cam.sam3_detect()
    assert frame["depth"] == "depth 10.500"
    assert instances[0][1].tolist() == [[False, True, True], [False, False, True]]
    cam._recent.clear()  # SAM 3's image no longer buffered: a miss, never another frame's depth
    assert cam.sam3_detect() == ([], None)


def test_only_a_camera_with_sam3_buffers_frames(perception):
    base, _ = perception
    depth_only = _bare_camera(base, None)  # the microwave's default
    for s in range(5):
        depth_only._on_frame(_stamped(s), None, None)
    assert not depth_only._recent and depth_only._latest[0].header.stamp.sec == 4
    with_sam3 = _bare_camera(base, object())
    for s in range(10):
        with_sam3._on_frame(_stamped(s), None, None)
    assert [m[0].header.stamp.sec for m in with_sam3._recent] == [6, 7, 8, 9]  # FRAME_KEEP_S = 3 s


def _yawed_handle(yaw_deg, reflection_m=1.05, noise_mm=1.0, seed=0):
    """Depth (mm) seen by CAM_T/CAM_K of the printed handle on a door yawed `yaw_deg` (outward
    normal (-cos, -sin, 0)): a 2.5 x 19 cm bar front 7.9 cm off the door, on a 6.4 x 24.8 cm plate
    1.2 cm off it, the bar's front centred 0.55 m ahead of the camera; the door itself reads as its
    reflection (`reflection_m`). Returns (depth, bar mask, plate mask): SAM 3 giving them apart."""
    y = np.radians(yaw_deg)
    n = np.array([-np.cos(y), -np.sin(y), 0.0])
    along = np.cross(n, [0.0, 0.0, 1.0])
    front = np.array([0.55, 0.0, 0.5])
    face = front - 0.079 * n
    vv, uu = np.mgrid[0:400, 0:640]
    rays = np.stack(
        [(uu - CAM_K[0, 2]) / CAM_K[0, 0], (vv - CAM_K[1, 2]) / CAM_K[1, 1], np.ones(uu.shape)], -1
    )
    rays = (
        rays @ CAM_T[:3, :3].T
    )  # base frame; the camera's depth is a hit's base x (it looks along +x)
    cam = CAM_T[:3, 3]
    depth = np.full(uu.shape, reflection_m * 1000.0)
    hit = {}
    for name, out, half_w, half_h in (
        ("plate", 0.012, 0.032, 0.124),
        ("bar", 0.079, 0.0125, 0.095),
    ):
        p0 = face + out * n
        t = ((p0 - cam) @ n) / (rays @ n)
        P = cam + rays * t[..., None]
        on = (np.abs((P - p0) @ along) <= half_w) & (np.abs(P[..., 2] - p0[2]) <= half_h) & (t > 0)
        hit[name] = on
        depth = np.where(on & (P[..., 0] * 1000 < depth), P[..., 0] * 1000, depth)
    depth = depth + np.random.default_rng(seed).normal(0, noise_mm, depth.shape)
    return depth.astype(np.float32), hit["bar"], hit["plate"] & ~hit["bar"]


@pytest.mark.parametrize("yaw_deg", [0.0, 15.0, -20.0, 30.0])
def test_door_normal_from_handle_finds_the_doors_yaw_from_the_bar_and_its_plate(
    perception, monkeypatch, yaw_deg
):
    dp, fp = perception

    def no_plane(self, *a):
        raise AssertionError("fit a door plane")

    monkeypatch.setattr(dp.HandleDetector, "_door_plane", no_plane)
    depth, bar, plate = _yawed_handle(yaw_deg)  # the door's own depth: only its reflection
    det = fp.HandleDetector(door_normal_from_handle=True)
    det._merged = (bar, fp._merge_inside(bar | plate, [bar, plate]))  # pick_handle_mask's merge
    got = det.detect(None, CAM_K, depth, CAM_T, handle_mask=bar)
    yaw = np.degrees(np.arctan2(-got["normal"][1], -got["normal"][0]))
    assert yaw == pytest.approx(yaw_deg, abs=1.0)
    assert got["normal"][2] == 0.0 and got["hinge"] is None
    assert got["handle"][0] == pytest.approx(0.55, abs=0.02)  # the bar's front


@pytest.mark.parametrize("door_normal_from_handle", [True, False])
def test_sam3_giving_the_bar_and_plate_apart_places_the_handle_on_the_bar(
    perception, door_normal_from_handle
):
    # the plate is the larger handle-like instance: from it, the grasp aimed ~6 cm too deep
    _, fp = perception
    depth, bar, plate = _yawed_handle(15.0)
    frame = dict(depth_mm=depth, K=CAM_K, base_T_cam=CAM_T)
    det = fp.HandleDetector(
        door_normal_from_handle=door_normal_from_handle,
        door_square_to_arm=not door_normal_from_handle,
    )
    picked = det.pick_handle_mask([(0.8, plate, "handle"), (0.6, bar, "handle")], frame)
    assert picked is bar
    got = det.detect(None, CAM_K, depth, CAM_T, handle_mask=picked)
    assert got["handle"][0] == pytest.approx(0.55, abs=0.01)  # the bar's front
    if door_normal_from_handle:  # the plate still in the fit
        assert np.degrees(np.arctan2(-got["normal"][1], -got["normal"][0])) == pytest.approx(
            15.0, abs=1.0
        )


def test_door_normal_from_handle_merges_the_bar_and_plate_and_misses_without_depth(perception):
    _, fp = perception
    _, bar, plate = _yawed_handle(15.0)
    far = np.zeros_like(bar)
    far[10:30, 10:30] = True  # elsewhere in the image: not merged
    merged = fp._merge_inside(bar | plate, [bar, plate, far])
    assert merged.sum() == (bar | plate).sum()
    det = fp.HandleDetector(door_normal_from_handle=True)
    no_depth = np.zeros((400, 640), np.float32)
    det._bar, det._bar_extent, det._bar_points = np.array([0.0, 0.0, 0.55]), (-0.1, 0.1), 999
    assert det._normal_from_handle(bar, no_depth, CAM_K, CAM_T) is None
    assert "handle points with depth" in det.last_miss


def test_door_normal_picks_how_each_look_measures_the_door(perception, tmp_path):
    """The node's door_normal: the handle model's fit (the default, the side grasp's), the bar and its plate, a plane, or
    square; how far two looks' doors may differ goes with it. The detector's own two flags stay exclusive."""
    _, fp = perception
    with pytest.raises(ValueError, match="pick one"):
        fp.HandleDetector(door_square_to_arm=True, door_normal_from_handle=True)
    saved = dict(vars(dc)), dict(vars(shared)), dict(vars(constants))
    try:
        open_door.configure(tmp_path, 30)
        det = shared.MAKE_DETECTOR()
        assert type(det).__name__ == "ModelNormalHandleDetector" and det.door_normal_from_handle
        assert shared.NORMAL_AGREE_DEG == 3.0
        open_door.configure(tmp_path, 30, door_normal="handle")
        det = shared.MAKE_DETECTOR()
        assert (
            type(det) is fp.HandleDetector
            and det.door_normal_from_handle
            and shared.NORMAL_AGREE_DEG == 5.0
        )
        open_door.configure(tmp_path, 30, door_normal="plane")
        det = shared.MAKE_DETECTOR()
        assert (
            not det.door_normal_from_handle
            and not det.door_square_to_arm
            and shared.NORMAL_AGREE_DEG == 10.0
        )
        with pytest.raises(ValueError, match="door_normal must be one of"):
            open_door.configure(tmp_path, 30, door_normal="wall")
    finally:
        vars(dc).update(saved[0])
        vars(shared).update(saved[1])
        vars(constants).update(saved[2])


def test_the_run_bag_records_the_wrist_and_scene_cameras_and_extra_topics(tmp_path):
    rosbag2_py = pytest.importorskip("rosbag2_py")
    pytest.importorskip("message_filters")
    from builtin_interfaces.msg import Time
    from fridge.bag import RunBag
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    clock = {"t": 0}
    subs = {}

    class Node:
        def create_action_subscription(self, msg_type, topic, cb, qos):
            subs[topic] = cb

        def get_clock(self):
            return MagicMock(now=lambda: MagicMock(nanoseconds=clock["t"]))

    wrist = ("/w/color", "/w/info", "/w/depth")
    scene = ("/s/color", "/s/info", "/s/depth")
    stereo = ("/w/infra1/image", "/w/infra1/info", "/w/infra2/image", "/w/infra2/info")
    bag = RunBag(
        Node(), tmp_path, [wrist, scene], 1.0, [("/boxes", "std_msgs/msg/String", String)], [stereo]
    )
    for i in range(30):  # 2 s at 15 Hz: 2 frames a camera at 1 Hz
        clock["t"] = int(i * 1e9 / 15)
        stamp = Time(sec=i // 15, nanosec=int((i % 15) * 1e9 / 15))
        for color, info, depth in (wrist, scene):
            for topic, msg in ((color, Image()), (info, CameraInfo()), (depth, Image())):
                msg.header.stamp = stamp
                subs[topic](msg)
        for topic, msg in zip(stereo, (Image(), CameraInfo(), Image(), CameraInfo()), strict=True):
            msg.header.stamp = stamp
            subs[topic](msg)
        subs["/boxes"](String(data=str(i)))
    # one image whose partner is a frame (67 ms) off: no pair, nothing recorded
    clock["t"] = int(10e9)
    for k, topic in enumerate(stereo):
        msg = (Image, CameraInfo)[k % 2]()
        msg.header.stamp = Time(sec=10, nanosec=0 if k < 2 else int(1e9 / 15))
        subs[topic](msg)
    assert bag.stereo_pairs == 2
    assert bag.close() == 2

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(tmp_path / "bag"), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    counts = {}
    while reader.has_next():
        topic = reader.read_next()[0]
        counts[topic] = counts.get(topic, 0) + 1
    assert counts == {t: 2 for t in (*wrist, *scene, *stereo, "/boxes")}


def test_the_run_bag_without_stereo_reports_no_pairs(tmp_path):
    pytest.importorskip("rosbag2_py")
    pytest.importorskip("message_filters")
    from fridge.bag import RunBag

    class Node:
        def create_action_subscription(self, msg_type, topic, cb, qos):
            pass

        def get_clock(self):
            return MagicMock(now=lambda: MagicMock(nanoseconds=0))

    bag = RunBag(Node(), tmp_path, [("/w/color", "/w/info", "/w/depth")], 1.0)
    assert bag.stereo_pairs is None
    assert bag.close() == 0


# --- the door's direction from the printed handle's model (door_normal.py, from the side grasp) ---


def _render_handle(yaw_deg, at, K, T, shape=(400, 640), samples=60000):
    """A depth image (mm) of the printed handle's model (its STL) seen by the camera (K, T: camera -> arm), the door
    yawed `yaw_deg` (door_normal's convention), the bar's front centre at `at` (arm frame); and the handle's mask. A
    z-buffer of the model's surface samples, each splatted over 2 x 2 pixels."""
    from fridge import door_normal as dn

    pts, _, front, _, _, _ = dn.load_handle_model(n=samples, seed=1)
    Rm = dn._model_rotation(yaw_deg)
    X = pts @ Rm.T + (np.asarray(at, float) - Rm @ front)
    cam = (np.linalg.inv(T) @ np.c_[X, np.ones(len(X))].T).T[:, :3]
    u = np.round(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(int)
    depth = np.full(shape, np.inf, np.float32)
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < shape[1]) & (vv >= 0) & (vv < shape[0])
            np.minimum.at(
                depth, (vv[ok], uu[ok]), (cam[ok, 2] * 1000.0).astype(np.float32)
            )  # the nearest wins
    depth[~np.isfinite(depth)] = 0.0
    return depth, depth > 0


@pytest.mark.parametrize("yaw", [0.0, 8.0, -15.0])
def test_the_handle_models_fit_finds_the_doors_yaw(yaw):
    """The side grasp's door direction (door_normal: the STL registered to the mask's depth) recovers a door's yaw
    from one look, as on 11-06's scene frames (+-1 deg there)."""
    from fridge import door_normal as dn

    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 200.0], [0.0, 0.0, 1.0]])
    T = CAM_T.copy()  # level, looking along the arm's +x, 0.5 m up
    depth, mask = _render_handle(yaw, at=[0.75, 0.0, 0.5], K=K, T=T)
    fit = dn.measure_door_normal(depth, K, T, mask)
    assert fit.yaw_deg == pytest.approx(yaw, abs=1.5)
    assert np.linalg.norm(fit.position[:2] - [0.75, 0.0]) < 0.01


def test_the_handle_models_fit_refuses_rather_than_guess():
    from fridge import door_normal as dn

    mask = np.zeros((400, 640), bool)
    mask[100:110, 300:305] = True  # a few pixels: too little of the handle
    with pytest.raises(dn.DoorNormalError, match="handle points with depth"):
        dn.measure_door_normal(np.full((400, 640), 700.0, np.float32), CAM_K, CAM_T, mask)
