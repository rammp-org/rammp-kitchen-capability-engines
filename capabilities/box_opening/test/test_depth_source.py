"""The depth pose source, proven against ray-cast synthetic scenes.

The renderer inverts the exact deprojection the detector performs: for
every pixel, the camera ray is intersected with the table plane and,
where the footprint covers it, the lid plane — so a recovered pose is a
round trip through real projective geometry, not an echo of the
detector's own math.
"""

import math
import time

import numpy as np
import pytest

from box_opening.models.container import ContainerModel
from box_opening.perception.depth_source import (
    container_pose_from_top,
    top_face_from_depth,
)

CFG = "capabilities/box_opening/config/containers/oxo_pop.yaml"

# the field-observed scan-pose camera: looking straight down from 0.575
ROT_DOWN = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
T_CAM = np.array([0.42, -0.075, 0.575])
K = np.array([[430.0, 0.0, 424.0], [0.0, 430.0, 240.0], [0.0, 0.0, 1.0]])
TABLE_Z = -0.027
H, W = 480, 848


def render_depth(boxes, rot_cam=ROT_DOWN, t_cam=T_CAM, table_z=TABLE_Z):
    """Ray-cast a depth image of the table plus rotated-rect box tops.

    boxes: [(cx, cy, top_z, w, h, yaw)] in base_link.
    """
    us, vs = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    rays = np.stack(
        [(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)],
        axis=-1,
    ).reshape(-1, 3)
    d_base = rays @ rot_cam.T  # ray directions in base
    dz = d_base[:, 2]
    depth = np.full(rays.shape[0], np.nan)
    ok = np.abs(dz) > 1e-9

    def lam_for(plane_z):
        lam = np.full_like(dz, np.nan)
        lam[ok] = (plane_z - t_cam[2]) / dz[ok]
        return lam

    lam_t = lam_for(table_z)
    hit_t = ok & (lam_t > 0)
    depth[hit_t] = lam_t[hit_t]
    for cx, cy, top_z, bw, bh, yaw in boxes:
        lam_b = lam_for(top_z)
        p = t_cam + d_base * lam_b[:, None]
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        inside = ok & (lam_b > 0) & (np.abs(lx) < bw / 2) & (np.abs(ly) < bh / 2)
        depth[inside] = lam_b[inside]
    return depth.reshape(H, W).astype(np.float32)


@pytest.fixture(scope="module")
def model():
    return ContainerModel.load(CFG)


def test_recovers_the_field_box_pose(model):
    """The box exactly where the 2026-09-01 bench run detected it."""
    truth = (0.458, -0.147, 0.080, model.dims[0], model.dims[1], math.radians(9.0))
    depth = render_depth([truth])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is not None, why
    assert fix.center[0] == pytest.approx(truth[0], abs=0.004)
    assert fix.center[1] == pytest.approx(truth[1], abs=0.004)
    assert fix.center[2] == pytest.approx(truth[2], abs=0.003)
    got = fix.yaw % (math.pi / 2)
    want = truth[5] % (math.pi / 2)
    err = min(abs(got - want), math.pi / 2 - abs(got - want))
    assert err < math.radians(3.0)
    # the measured footprint is the plateau CORE — the smoothness gate and
    # the core erosion shave a boundary ring by design, so it reads SMALLER
    # than the physical box (real captures read 0.056-0.104 for a 75 mm
    # lid). The gate's lower bound accounts for it; assert the core stays
    # inside the gate's window rather than at nominal size.
    for side in fix.footprint:
        assert 0.075 - 0.035 <= side <= 0.075 + 0.035

    cpose = container_pose_from_top(fix.center, fix.yaw, model)
    assert cpose.xyz[2] == pytest.approx(truth[2] - model.dims[2], abs=0.003)


def test_empty_table_yields_no_fix(model):
    depth = render_depth([])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None and "container height" in why


def test_wrong_size_plateau_is_rejected(model):
    """A tote-sized surface at box height must not be mistaken for the
    container — footprint sanity is the one defence the band has."""
    depth = render_depth([(0.45, 0.0, 0.080, 0.30, 0.22, 0.0)])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None and "footprint" in why


def test_truncated_box_at_the_border_is_partial_never_precise(model):
    """Half a box in view = a biased centroid: never good enough to press,
    and it used to be refused outright. But it IS a box, and since every
    search result gets a close-up aim at staging (2026-09-23), knowing
    roughly where it is is what the search is for: it comes back marked
    PARTIAL, and the watcher lets it into the coarse window only."""
    # camera footprint at table: ~0.9 x 0.5 m around [0.42, -0.075]; the
    # image's +x edge runs near base x -0.01: a model-sized box at -0.05
    # shows ~64 % of itself
    box = (-0.05, -0.075, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)
    fix, why = top_face_from_depth(render_depth([box]), K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is not None and fix.partial and "partial" in why
    assert abs(fix.center[1] + 0.075) < 0.02  # the cut runs across x: y is still true
    # a sliver (well under half the box) is still refused: not enough to be a box
    sliver = (-0.09, -0.075, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)
    fix, why = top_face_from_depth(render_depth([sliver]), K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None


def test_a_full_top_is_not_partial(model):
    truth = (0.45, -0.10, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.3)
    fix, why = top_face_from_depth(render_depth([truth]), K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is not None and not fix.partial


def test_a_lid_the_fingers_half_hide_is_a_partial_top():
    """Bench 2026-09-23, run 1: the box stood 36 cm from the base, at the
    bottom of the search's view where the gripper's fingers hang in every
    wrist frame. They hid part of the lid; the rest measured 0.06 x 0.10 m
    against the 0.105 model and was refused — "NO BOX" with the box in
    plain view. Now a partial top: roughly where the lid is."""
    from pathlib import Path

    pink = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    z = np.load(Path(__file__).parent / "data" / "wrist_search_pink_fingers_20260923.npz")
    fix, why = top_face_from_depth(z["depth"], z["k"], z["rot_cam"], z["trans_cam"], TABLE_Z, pink)
    assert fix is not None and fix.partial, why
    # the lid's pink pixels put its centre at [0.315, 0.179]: the partial
    # centroid is biased by the hidden part, but it is on the lid
    assert np.hypot(fix.center[0] - 0.315, fix.center[1] - 0.179) < 0.06


def test_a_partial_top_feeds_the_coarse_window_and_is_remembered(model, monkeypatch):
    """A partial top may stop the search and send the arm to staging; it
    never aims a press. And what the search saw is remembered after the
    coarse window has aged out — the box does not move while the mission
    tries its other ladders."""
    w = _watcher(model, require_circle=False)
    w.grab = _Grab()
    edge = [(-0.05, -0.075, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)]  # cut by the border
    for n in range(4):  # still frames: a full top would commit precisely
        _tick_at(w, T_CAM, edge, monkeypatch, n + 1)
    assert w.fix(now=time.monotonic()) is None  # nothing may aim a press
    assert w.coarse_fix(now=time.monotonic()) is not None
    assert "partial" in w.last_reject
    seen = w.last_coarse()
    assert seen is not None and abs(seen[0][1] + 0.075) < 0.02
    assert w.coarse_fix(now=time.monotonic() + 60.0) is None  # the window aged out ...
    assert w.last_coarse() == seen  # ... the memory did not
    w.reset()
    assert w.last_coarse() is None  # a new search forgets the old box


def test_wall_points_cannot_drag_the_centroid(model):
    """A second, lower plateau overlapping the band (stand-in for box
    side-wall returns) must not shift the fix: the plateau refinement
    re-tightens around the top face."""
    truth = (0.46, -0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    skirt = (0.46 + 0.050, -0.10, 0.035, 0.05, model.dims[1], 0.0)
    depth = render_depth([skirt, truth])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is not None, why
    assert fix.center[0] == pytest.approx(truth[0], abs=0.005)
    assert fix.center[2] == pytest.approx(truth[2], abs=0.003)


def test_missing_depth_frame_is_honest(model):
    fix, why = top_face_from_depth(None, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None and why == "no depth frame"


def test_two_boxes_are_ambiguous_without_an_roi(model):
    """Two container-sized tops: guessing which to press is refused."""
    a = (0.40, -0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    b = (0.55, 0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    depth = render_depth([a, b])
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None and "ambiguous" in why


def test_roi_disambiguates_two_boxes(model):
    """A pixel-space roi (the VLM's bbox) picks the intended box."""
    a = (0.40, -0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    b = (0.55, 0.10, 0.080, model.dims[0], model.dims[1], 0.0)
    depth = render_depth([a, b])
    # project box a's center into the image to build its roi
    p_cam = ROT_DOWN.T @ (np.array([a[0], a[1], a[2]]) - T_CAM)
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    roi = (u - 90, v - 90, u + 90, v + 90)
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model, roi=roi)
    assert fix is not None, why
    assert fix.center[0] == pytest.approx(a[0], abs=0.006)
    assert fix.center[1] == pytest.approx(a[1], abs=0.006)


def test_stereo_speckle_cannot_fake_a_box(model):
    """Locally-wild noise at container height (the blank-table failure of
    capture 20260901-130610) fails the smoothness gate: no false fix."""
    depth = render_depth([])
    rng = np.random.RandomState(7)
    # a large patch of the table speckled into the height band
    patch = depth[100:400, 200:700]
    depth[100:400, 200:700] = patch - rng.uniform(0.04, 0.11, patch.shape).astype(np.float32) * (
        rng.rand(*patch.shape) < 0.5
    )
    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is None


def test_camera_still_filter():
    """Only still-camera frames may vote: moving frames carry a
    systematic TF-vs-exposure bias that agreed with itself 10-15 mm off
    the truth and pressed the button's EDGE (field 2026-09-01)."""
    from box_opening.perception.depth_source import camera_is_still

    eye = np.eye(3)
    a = (eye, np.array([0.42, -0.07, 0.575]))
    assert camera_is_still(a, (eye, a[1] + [0.001, 0.0, 0.0]))
    assert not camera_is_still(a, (eye, a[1] + [0.01, 0.0, 0.0]))  # translating
    c, s = np.cos(0.05), np.sin(0.05)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert not camera_is_still(a, (rz @ eye, a[1]))  # rotating
    assert not camera_is_still(None, a)  # first frame never votes


def _project(pt, rot=ROT_DOWN, t=T_CAM):
    p = rot.T @ (np.asarray(pt, dtype=float) - t)
    return (
        K[0, 0] * p[0] / p[2] + K[0, 2],
        K[1, 1] * p[1] / p[2] + K[1, 2],
        p[2],
    )


def test_button_circle_refine_recovers_the_true_button(model):
    """The plateau centroid finds the LID; the circle finds the BUTTON.
    A dark disk painted 8 mm off the centroid must win the aim."""
    import cv2

    from box_opening.perception.depth_source import button_circle_refine

    truth = (0.458, -0.147, 0.080, model.dims[0], model.dims[1], 0.0)
    depth = render_depth([truth])
    color = np.full((H, W, 3), 235, dtype=np.uint8)
    button = (truth[0] + 0.008, truth[1] - 0.008, truth[2])
    u, v, z = _project(button)
    r_px = int(K[0, 0] * (model.button_diameter_m / 2) / z)
    cv2.circle(color, (int(u), int(v)), r_px, (60, 60, 60), -1)

    fix, why = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert fix is not None, why
    got = button_circle_refine(color, depth, K, ROT_DOWN, T_CAM, fix.center, model.button_diameter_m)
    assert got is not None
    assert got[0] == pytest.approx(button[0], abs=0.003)
    assert got[1] == pytest.approx(button[1], abs=0.003)


def test_button_circle_declines_honestly(model):
    """No circle in view, or a truncated crop -> None; the centroid
    fallback is always sane."""
    from box_opening.perception.depth_source import button_circle_refine

    truth = (0.458, -0.147, 0.080, model.dims[0], model.dims[1], 0.0)
    depth = render_depth([truth])
    blank = np.full((H, W, 3), 235, dtype=np.uint8)
    fix, _ = top_face_from_depth(depth, K, ROT_DOWN, T_CAM, TABLE_Z, model)
    assert button_circle_refine(blank, depth, K, ROT_DOWN, T_CAM, fix.center, model.button_diameter_m) is None
    assert button_circle_refine(None, depth, K, ROT_DOWN, T_CAM, fix.center, model.button_diameter_m) is None
    # centre projected at the image edge -> truncated crop -> decline
    edge_center = (0.85, -0.075, 0.080)
    assert button_circle_refine(blank, depth, K, ROT_DOWN, T_CAM, edge_center, model.button_diameter_m) is None


def test_origin_z_is_pinned_to_the_calibrated_table(model):
    """The plateau median wandered -3 mm one run (grip grazed the lid)
    and +8 mm the next (field 2026-09-01/02); the box rests on the
    surveyed table and its moulded height is constant, so the origin z
    comes from table_z whenever one is known."""
    from box_opening.perception.depth_source import container_pose_from_top

    top = (0.38, -0.17, 0.088)  # measured 8 mm above nominal
    pinned = container_pose_from_top(top, 0.0, model, table_z=-0.027)
    assert pinned.xyz[2] == -0.027
    legacy = container_pose_from_top(top, 0.0, model)
    assert abs(legacy.xyz[2] - (0.088 - model.dims[2])) < 1e-9


def test_watcher_wires_the_table_into_the_pose(model, capsys):
    """The 2026-09-02 regression: a 6-frame plateau median 8 mm high
    became origin z=-0.019 and shifted the press geometry. The wiring
    under test is BoxTopWatcher.to_container_pose passing its surveyed
    table_z through — dropping it must fail THIS test."""
    from box_opening.perception.depth_source import BoxTopWatcher

    w = BoxTopWatcher.__new__(BoxTopWatcher)  # wiring only, no node
    w.model = model
    w.table_z = TABLE_Z
    import threading

    w._lock = threading.Lock()
    top = (0.38, -0.17, TABLE_Z + model.dims[2] + 0.008)
    cp = w.to_container_pose((top, 0.3))
    assert cp.xyz[2] == TABLE_Z
    assert "8.0 mm" in capsys.readouterr().out  # residual stays visible


def test_far_from_nominal_top_is_refused_for_commit(model):
    """With origin z pinned, a top far from table+dims means not-our-box
    or a stale calibration; pressing nominal geometry would stroke air
    (below) or leave real lid unmodelled (above). The gate is tighter
    than travel_m so a committed fix always reaches material."""
    from box_opening.perception.depth_source import top_residual_reject

    nominal = TABLE_Z + model.dims[2]
    assert top_residual_reject(nominal + 0.008, TABLE_Z, model) is None
    assert top_residual_reject(nominal - 0.008, TABLE_Z, model) is None
    for bad in (nominal + 0.020, nominal - 0.020):
        why = top_residual_reject(bad, TABLE_Z, model)
        assert why and "recalibrat" in why


def test_fix_window_commits_median_then_goes_stale():
    from box_opening.perception.depth_source import FixWindow

    fw = FixWindow(min_hits=3, tol_m=0.03, window_s=2.0, fresh_s=1.0)
    assert fw.fix(now=0.0) is None
    for i, t in enumerate([0.0, 0.2, 0.4]):
        fw.add(np.array([0.5, 0.0, 0.130 + 0.001 * i]), 0.3, t)
    got = fw.fix(now=0.5)
    assert got is not None
    pos, yaw = got
    assert pos[2] == pytest.approx(0.131)
    assert float(yaw) == pytest.approx(0.3)
    assert fw.fix(now=2.0) is None  # nothing seen for > fresh_s


def test_fix_window_rejects_disagreeing_frames():
    from box_opening.perception.depth_source import FixWindow

    fw = FixWindow(min_hits=3, tol_m=0.03, window_s=2.0, fresh_s=1.0)
    fw.add(np.array([0.50, 0.0, 0.13]), 0.0, 0.0)
    fw.add(np.array([0.60, 0.0, 0.13]), 0.0, 0.2)  # 10 cm jump
    fw.add(np.array([0.50, 0.0, 0.13]), 0.0, 0.4)
    assert fw.fix(now=0.5) is None


def test_watcher_reset_zeroes_the_window_counters(model):
    """The re-look line printed scan-pose totals and hid its own reject
    reason (2026-09-02): reset() now zeroes the per-window counters and
    status() always names the last reject."""
    from box_opening.perception.depth_source import BoxTopWatcher, FixWindow

    w = BoxTopWatcher.__new__(BoxTopWatcher)
    import threading

    w._lock = threading.Lock()
    w.grab = type("G", (), {"missing": lambda self: []})()
    w.window = FixWindow(3, 0.015, 2.0, 1.0)
    w.coarse_window = FixWindow(2, 0.04, 1.5, 1.0)
    w.frames, w.hits, w.circle_hits = 40, 12, 11
    w.last_reject = "a top touches the image border"
    assert "border" in w.status()  # reported even though hits > 0
    w.reset()
    assert (w.frames, w.hits, w.circle_hits, w.last_reject) == (0, 0, 0, None)


def test_offered_roi_applies_only_from_the_still_epoch(model):
    """A bbox from a frame taken while the camera was still decelerating
    must not gate the parked frames; one from a still frame must (review
    2026-09-02)."""
    from box_opening.perception.depth_source import BoxTopWatcher

    w = BoxTopWatcher.__new__(BoxTopWatcher)
    w.roi = None
    w._still_since = None
    w._pending_roi = None
    w.offer_roi((1, 2, 3, 4), frame_t=10.0)  # seen while moving
    w._apply_pending_roi(still=False, frame_t=10.2)
    w._apply_pending_roi(still=True, frame_t=10.5)  # still epoch starts at 10.5
    assert w.roi is None  # the moving-frame bbox was dropped
    w.offer_roi((5, 6, 7, 8), frame_t=10.9)  # seen while still
    w._apply_pending_roi(still=True, frame_t=11.0)
    assert w.roi == (5, 6, 7, 8)
    w._apply_pending_roi(still=False, frame_t=12.0)  # moved again
    w.offer_roi((9, 9, 9, 9), frame_t=11.9)
    w._apply_pending_roi(still=True, frame_t=12.3)
    assert w.roi == (5, 6, 7, 8)  # the pre-epoch bbox did not apply


def _watcher(model, require_circle=True):
    """A BoxTopWatcher with the ROS parts left out: the tick's decision
    logic only."""
    from box_opening.perception.depth_source import BoxTopWatcher, FixWindow

    w = BoxTopWatcher.__new__(BoxTopWatcher)
    w.grab = type("G", (), {"missing": lambda self: []})()
    w.model = model
    w.table_z = TABLE_Z
    w.window = FixWindow(3, 0.015, 2.0, 1.0)
    w.coarse_window = FixWindow(2, 0.04, 1.5, 1.0)
    w.frames = w.hits = w.circle_hits = 0
    w.last_reject = None
    w.last_debug = None
    w.roi = None
    w.active = True
    w.require_circle = require_circle
    w.moving_ok = False
    import threading

    w._lock = threading.Lock()
    w._last_stamp = None
    w._last_depth_stamp = None
    w._last_cam = None
    w._still_since = None
    w._pending_roi = None
    w._last_coarse = None
    return w


def test_only_button_circle_sightings_vote(model):
    """The centroid finds the LID; the circle finds the BUTTON. Mixing a
    centroid sample into the window let it win the median and walk the
    press off the button (field 2026-09-03) — with the circle required,
    a frame without one votes not at all and says why."""
    import numpy as np

    from box_opening.perception.depth_source import TopFaceFix

    # a plateau at the lid centre, and the button circle 6 mm off it —
    # the offset the bench capture measured
    plateau = TopFaceFix(center=(0.45, -0.15, 0.08), yaw=0.0, footprint=(0.075, 0.075), n_px=250)
    circle = (0.456, -0.1505)

    w = _watcher(model)
    for _ in range(3):
        w.window.add(np.asarray(circle + (0.08,)), 0.0, 1.0)
    got = w.window.fix(1.2)
    assert got is not None and abs(got[0][0] - 0.456) < 1e-9  # the circle centre
    assert abs(got[0][0] - plateau.center[0]) > 0.005  # not the lid centre

    # a centroid-only frame must not enter the window when the circle is
    # required; the reject reason names it
    w2 = _watcher(model)
    w2.last_reject = "container top but no button circle"
    assert "no button circle" in w2.status()
    assert w2.window.fix(1.0) is None  # nothing voted


def test_shipped_config_requires_the_button_circle_and_trims_nothing():
    from box_opening.models.container import load_press_demo

    cfg = load_press_demo("capabilities/box_opening/config/containers/oxo_pop.yaml")
    assert cfg.require_button_circle is True
    # the press and the grip both aim at the circle centre itself. The
    # 6 mm radial miss of 2026-09-14..16 was the planner's descent bowing
    # (runtime/approach.py now flies a straight line) — NOT a pad offset,
    # so no trim: one here would double-correct.
    assert tuple(cfg.grip_offset_xy) == (0.0, 0.0)
    assert tuple(cfg.press_offset_xy) == (0.0, 0.0)


# --- the coarse path: sightings taken while the camera is moving ------------


class _Grab:
    """A grabber whose frames are set by the test, ROS parts left out."""

    def __init__(self):
        self.color = np.full((H, W, 3), 235, np.uint8)  # a lid, no button disc
        self.depth = None
        self.k = K
        self.color_stamp = type("S", (), {"sec": 0, "nanosec": 0})()

    def missing(self):
        return []

    def frame(self, boxes, t_cam, n):
        self.depth = render_depth(boxes, rot_cam=ROT_DOWN, t_cam=t_cam)
        self.color_stamp = type("S", (), {"sec": n, "nanosec": 0})()


def _tick_at(w, t_cam, boxes, monkeypatch, n):
    """One watcher tick with the camera at t_cam, looking at `boxes`."""
    from box_opening.perception import depth_source as ds

    w.grab.frame(boxes, t_cam, n)
    monkeypatch.setattr(ds, "camera_pose_at", lambda g, offset_s=None, stamp=None: (ROT_DOWN, t_cam))
    w._tick()


def test_a_moving_camera_feeds_the_coarse_window_only(model, monkeypatch):
    """Finding the box and aiming the press are different questions. While the
    arm sweeps, sightings carry the TF-lag bias that the stillness gate exists
    to keep out — good enough to stop the search, never good enough to press."""
    w = _watcher(model)
    w.grab = _Grab()
    box = [(0.45, -0.15, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)]
    for n, dx in enumerate((0.0, 0.02, 0.04)):  # the camera pans 2 cm a frame
        _tick_at(w, T_CAM + np.array([dx, 0.0, 0.0]), box, monkeypatch, n + 1)
    coarse = w.coarse_fix(now=time.monotonic())
    assert coarse is not None
    assert abs(coarse[0][0] - 0.45) < 0.02 and abs(coarse[0][1] + 0.15) < 0.02
    assert w.fix(now=time.monotonic()) is None  # nothing may aim a press


def test_a_still_camera_still_commits_the_precise_fix(model, monkeypatch):
    w = _watcher(model, require_circle=False)
    w.grab = _Grab()
    box = [(0.45, -0.15, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)]
    for n in range(4):  # same pose every frame: still
        _tick_at(w, T_CAM, box, monkeypatch, n + 1)
    assert w.fix(now=time.monotonic()) is not None


def test_coarse_sightings_that_disagree_do_not_commit(model, monkeypatch):
    """Two different boxes seen in two frames is not a fix, even coarsely."""
    w = _watcher(model)
    w.grab = _Grab()
    for n, (x, dx) in enumerate(((0.45, 0.0), (0.30, 0.02))):
        box = [(x, -0.15, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)]
        _tick_at(w, T_CAM + np.array([dx, 0.0, 0.0]), box, monkeypatch, n + 1)
    assert w.coarse_fix(now=time.monotonic()) is None


def test_reset_clears_the_coarse_window_too(model, monkeypatch):
    w = _watcher(model)
    w.grab = _Grab()
    box = [(0.45, -0.15, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)]
    for n, dx in enumerate((0.0, 0.02)):
        _tick_at(w, T_CAM + np.array([dx, 0.0, 0.0]), box, monkeypatch, n + 1)
    assert w.coarse_fix(now=time.monotonic()) is not None
    w.reset()
    assert w.coarse_fix(now=time.monotonic()) is None


def test_detection_does_not_follow_a_table_the_camera_reads_high(model, monkeypatch):
    """Regression, field 2026-09-16: the box sat right under the look and was
    not found in 438 frames across two runs.

    Passive stereo reads a blank bench as a broad smear, not a peak — on the
    1 September capture it spans -0.07 to +0.02 m around a true -0.027 — and
    the MODE of that smear sits 35 mm above the real surface. A table fitted
    from it moved the lid band off the lid: replaying that capture, the hit
    rate fell from 39/94 frames to 7/94, rejected as "no points at container
    height". The lid band must stay on the surveyed table no matter what
    flat surface dominates the frame."""
    w = _watcher(model, require_circle=False)
    w.grab = _Grab()
    box = (0.45, -0.15, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)
    # most of the view reads as a flat surface 35 mm above the true table
    smear = (0.45, -0.075, TABLE_Z + 0.035, 1.2, 0.9, 0.0)
    for n in range(6):
        _tick_at(w, T_CAM, [smear, box], monkeypatch, n + 1)
    got = w.fix(now=time.monotonic())
    assert got is not None, w.status()
    assert abs(got[0][2] - (TABLE_Z + model.dims[2])) < 0.01


def test_with_the_lag_measured_moving_frames_may_confirm_the_box(model, monkeypatch):
    """detect.confirm_in_flight: frames shot while the arm moves feed the
    precise window too (the circle is still required), which is what lets
    the wrist confirm the box during the approach instead of after it."""
    w = _watcher(model, require_circle=False)
    w.grab = _Grab()
    w.moving_ok = True
    box = [(0.45, -0.15, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)]
    for n, dx in enumerate((0.0, 0.02, 0.04, 0.06)):  # the camera pans 2 cm a frame
        _tick_at(w, T_CAM + np.array([dx, 0.0, 0.0]), box, monkeypatch, n + 1)
    got = w.fix(now=time.monotonic())
    assert got is not None
    assert abs(got[0][0] - 0.45) < 0.01 and abs(got[0][1] + 0.15) < 0.01


# --- a depth votes only with the colour frame it was shot with --------------


class _PairedGrab(_Grab):
    """_Grab with a depth stamp: frame() stamps both streams, colour_only()
    advances the colour while the depth stream stalls."""

    def __init__(self):
        super().__init__()
        self.depth_stamp = None

    def frame(self, boxes, t_cam, n):
        super().frame(boxes, t_cam, n)
        self.depth_stamp = self.color_stamp

    def colour_only(self, n):
        self.color_stamp = type("S", (), {"sec": n, "nanosec": 0})()


def test_a_paired_depth_still_commits_the_precise_fix(model, monkeypatch):
    w = _watcher(model, require_circle=False)
    w.grab = _PairedGrab()
    box = [(0.45, -0.15, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)]
    for n in range(4):
        _tick_at(w, T_CAM, box, monkeypatch, n + 1)
    assert w.frames == 4 and w.fix(now=time.monotonic()) is not None


def test_a_frozen_depth_adds_nothing_to_either_window(model, monkeypatch):
    """The depth stream stalls while colour keeps arriving: each new colour
    stamp used to re-vote the same old depth, and three agreeing votes of
    one frame made a fix of a moment long gone."""
    from box_opening.perception import depth_source as ds

    w = _watcher(model, require_circle=False)
    w.grab = _PairedGrab()
    box = [(0.45, -0.15, TABLE_Z + model.dims[2], model.dims[0], model.dims[1], 0.0)]
    _tick_at(w, T_CAM, box, monkeypatch, 1)
    n_fine, n_coarse = len(w.window.samples), len(w.coarse_window.samples)
    assert n_coarse == 1
    monkeypatch.setattr(ds, "camera_pose_at", lambda g, offset_s=None, stamp=None: (ROT_DOWN, T_CAM))
    for n in range(2, 8):
        w.grab.colour_only(n)
        w._tick()
    assert (len(w.window.samples), len(w.coarse_window.samples)) == (n_fine, n_coarse)
    assert w.frames == 1 and w.last_reject == "depth not new"
    assert w.fix(now=time.monotonic()) is None


def test_the_pose_is_looked_up_at_the_stamp_given_not_the_grabbers_newest():
    from types import SimpleNamespace

    from builtin_interfaces.msg import Time

    from box_opening.perception.depth_source import camera_pose_at

    asked = []

    class Buffer:
        def lookup_transform(self, root, parent, when):
            asked.append(when.nanoseconds)
            return SimpleNamespace(
                transform=SimpleNamespace(
                    rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                )
            )

    g = SimpleNamespace(
        tf_buffer=Buffer(),
        parent="tool",
        mount_xyz=[0.0, 0.0, 0.0],
        mount_quat=[0.0, 0.0, 0.0, 1.0],
        stamp_offset_s=0.0,
        color_stamp=Time(sec=5, nanosec=0),
    )
    assert camera_pose_at(g) is not None
    camera_pose_at(g, stamp=Time(sec=9, nanosec=500))
    camera_pose_at(g, offset_s=-0.2, stamp=Time(sec=9, nanosec=500))
    assert asked == [5_000_000_000, 9_000_000_500, 8_800_000_500]
