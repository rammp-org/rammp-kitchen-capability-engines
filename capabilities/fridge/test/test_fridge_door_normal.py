"""fridge.door_normal: the handle model's ICP, offline (no ROS, no camera)."""

from pathlib import Path

import numpy as np
import pytest

from fridge import door_normal as dn


def test_a_fit_that_loses_the_handle_returns_a_mask_over_every_point():
    """11-19 and 11-20: a warm start far off the handle lost the data in _icp, which returned the footprint mask of
    its ICP_MAX_POINTS subset; measure_door_normal indexed every point with it and the open raised IndexError
    (2224 vs 1200). A lost fit has to come back as cost inf with a mask the size of the data: the look fails as a
    poor fit and the open retries."""
    model = dn.load_handle_model()
    n = dn.ICP_MAX_POINTS + 1024
    P = np.random.default_rng(0).uniform(-0.05, 0.05, (n, 3)) + [3.0, 3.0, 0.0]  # nowhere near the model
    R0 = dn._model_rotation(0.0)
    t0 = np.array([0.8, -0.3, 0.4]) - R0 @ model[2]
    for footprint_from in (0, dn.ICP_ITERS):
        _, _, cost, on = dn._icp(P, np.zeros(3), R0, t0, model, 5, footprint_from)
        assert cost == np.inf
        assert on.shape == (n,) and not on.any()
        assert len(P[on]) == 0  # what measure_door_normal does with it


def _real_234904_frame(index):
    path = Path(__file__).parent / "fixtures" / f"nov16_234904_wrist_{index}.npz"
    with np.load(path) as frame:
        return frame["depth_mm"], frame["K"], frame["base_T_cam"], frame["mask"]


def test_real_wrist_door_surface_excludes_wrong_yaw_branch():
    wrong_branch = _real_234904_frame(3)
    original = dn.measure_door_normal(*wrong_branch)
    assert -9.0 < original.yaw_deg < -5.0
    # The wider SAM mask fits the wrong branch in isolation. At the measured door
    # yaw the handle inliers miss the established acceptance threshold.
    with pytest.raises(dn.DoorNormalError, match=r"yaw \+10.*within .* of the model"):
        dn.measure_door_normal(*wrong_branch, door_standoff_m=0.070)
    confirmed = dn.measure_door_normal(*_real_234904_frame(4), door_standoff_m=0.070)
    assert 9.0 < confirmed.yaw_deg < 11.0


def test_real_wrist_handle_with_obscured_door_requires_another_look():
    depth, K, T, mask = _real_234904_frame(4)
    depth[~mask] = 0
    with pytest.raises(dn.DoorNormalError, match="insufficient door surface depth"):
        dn.measure_door_normal(depth, K, T, mask, door_standoff_m=0.070)


def _v4_wrist(dock):
    path = Path(__file__).parent / "fixtures" / f"v4_dock_{dock}_1531_wrist.npz"
    with np.load(path) as frame:
        return frame["depth_mm"], frame["K"], frame["base_T_cam"], frame["mask"]


def test_v4_dock_02_narrow_mask_retries_clean_short_strip():
    depth, K, T, mask = _v4_wrist("02")
    points, rows, cols = dn._points(depth, K, T, mask)
    front = dn._bar_front_point(points, depth[rows, cols] / 1000, rows)
    # Removing the added columns recreates the original 96.8 mm trimmed span.
    original = depth.copy()
    original[:, :262] = 0
    assert dn.measure_door_surface(original, K, T, mask, front, 0.070) is None

    plane = dn.measure_door_surface(depth, K, T, mask, front, 0.070)
    assert plane is not None
    assert 0.10 <= plane.span_m < 0.18
    assert plane.inliers > 2400
    assert plane.rms_m < 0.002
    assert plane.yaw_deg == pytest.approx(11.0, abs=0.5)


def test_v4_cabinet_height_clipped_wrist_uses_registered_center():
    depth, K, T, mask = _v4_wrist("00")
    fit = dn.measure_door_normal(depth, K, T, mask, door_standoff_m=0.070)
    points, rows, cols = dn._points(depth, K, T, mask)
    front = points[depth[rows, cols] / 1000 <= _row_nearest(depth, rows, cols) + 0.020]
    bounds = np.percentile(front[:, 2], [5, 95])
    z, coverage, source = dn._handle_height(fit, bounds)
    assert coverage < dn.BAR_MIN_FULL_COVERAGE
    assert source == "registered center"
    assert z == pytest.approx(0.393, abs=0.003)
    assert z - np.mean(bounds) > 0.004


def _row_nearest(depth, rows, cols):
    nearest = np.full(depth.shape[0], np.inf)
    np.minimum.at(nearest, rows, depth[rows, cols] / 1000)
    return nearest[rows]


@pytest.mark.parametrize(
    ("support", "bounds", "source"),
    [
        ((0.248, 0.95, 40), (0.275, 0.495), "visible midpoint"),
        ((0.248, 0.90, 40), (0.300, 0.450), "visible midpoint (ambiguous registration)"),
        ((0.248, 0.95, 0), (0.300, 0.450), "visible midpoint (ambiguous registration)"),
    ],
)
def test_handle_height_only_uses_model_for_anchored_partial_bar(support, bounds, source):
    fit = dn.HandleFit(
        normal=np.array([-1, 0, 0]),
        yaw_deg=0,
        position=np.array([0.85, -0.2, 0.393]),
        R=np.eye(3),
        t=np.zeros(3),
        rms_m=0.002,
        inlier_fraction=0.95,
        points=600,
        footprint_fraction=1.0,
        debug={"height_support": support},
    )
    z, _, reason = dn._handle_height(fit, bounds)
    assert z == pytest.approx(np.mean(bounds))
    assert reason == source


def test_real_wrist_top_proposals_recover_a_rejected_primary(monkeypatch):
    # This fit uses NumPy and SciPy. Open3D is used by the generic plane detector.
    import sys
    from types import ModuleType

    try:
        import open3d  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "open3d", ModuleType("open3d"))

    path = Path(__file__).parent / "fixtures" / "nov16_wrist_3_proposals.npz"
    with np.load(path) as data:
        frame = {key: data[key] for key in ("K", "depth_mm", "base_T_cam")}
        frame["rgb"] = np.zeros((*frame["depth_mm"].shape, 3), np.uint8)
        instances = sorted(
            zip(data["scores"], np.unpackbits(data["masks"], axis=-1).astype(bool), data["labels"], strict=True),
            key=lambda item: -item[0],
        )
    detector = dn.make_detector()
    primary = detector.pick_handle_mask([i for i in instances if i[2] == "handle"], frame)
    assert primary is not None
    with pytest.raises(dn.DoorNormalError, match="of the model"):
        dn.measure_door_normal(frame["depth_mm"], frame["K"], frame["base_T_cam"], primary, door_standoff_m=0.070)
    detection, mask = detector.detect_instances(instances, frame)
    assert detection is not None
    assert not np.array_equal(mask, primary)
    assert detector.last_door_normal.yaw_deg == pytest.approx(10.3, abs=0.6)
    assert detection["handle"] == pytest.approx([0.860, -0.269, 0.337], abs=0.009)
