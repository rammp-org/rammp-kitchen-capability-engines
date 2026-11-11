"""Offline regression checks for container-only depth selection and frame agreement."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "container_detection", Path(__file__).parents[1] / "microwave_place" / "rim_detection.py"
)
detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection)


def test_nearest_50_valid_points_and_transform():
    depth = 0.4 + np.arange(400).reshape(20, 20) * 0.00001
    mask = np.ones_like(depth, dtype=bool)
    depth[5, 5] = np.nan
    depth[6, 6] = 0.1
    depth[7, 7] = 2.1
    mask[8, 8] = False
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    point, info = detection.container_point(depth, mask, np.eye(3), transform)
    zs = depth.ravel()[:50]
    us = np.arange(50) % 20
    vs = np.arange(50) // 20
    np.testing.assert_allclose(point, [1 + (us * zs).mean(), 2 + (vs * zs).mean(), 3 + zs.mean()])
    assert len(info["pixels_uv"]) == 50


def test_isolated_near_and_far_spikes_rejected_before_nearest_selection():
    depth = np.full((20, 20), 0.5)
    depth[8, 8], depth[10, 10] = 0.25, 1.5
    point, info = detection.container_point(depth, np.ones_like(depth, bool), np.eye(3), np.eye(4))
    assert point[2] == pytest.approx(0.5)
    assert info["depth_outliers_px"] == 2
    assert [8, 8] not in info["pixels_uv"]
    assert [10, 10] not in info["pixels_uv"]


def test_real_depth_slope_preserved():
    depth = np.tile(np.linspace(0.4, 0.6, 40), (40, 1))
    ys, xs, stats = detection.select_depth_pixels(depth, np.ones_like(depth, bool))
    assert len(xs) == 50
    assert stats["depth_outliers_px"] == 0
    assert np.mean(depth[ys, xs]) < 0.41


def test_sparse_mask_support_refused():
    depth = np.full((30, 30), 0.5)
    mask = np.zeros_like(depth, bool)
    mask[::4, ::4] = True
    point, why = detection.container_point(depth, mask, np.eye(3), np.eye(4))
    assert point is None
    assert "filtered" in why


def test_too_few_valid_points_refused():
    point, why = detection.container_point(np.ones((2, 2)), np.ones((2, 2), bool), np.eye(3), np.eye(4))
    assert point is None
    assert "need 50" in why


@pytest.mark.parametrize("disagree", [False, True])
def test_frame_averaging_and_agreement_reset(disagree):
    mask = np.ones((20, 20), bool)
    other = np.zeros_like(mask)
    other[:, :1] = True
    depths = [0.4, 0.5, 0.6] if not disagree else [0.3, 0.35, 0.4, 0.5, 0.6]
    masks = [mask] * len(depths)
    if disagree:
        masks[1] = other
    shots = iter(
        dict(container=m, depth_m=np.full((20, 20), z), depth_msg=None) for m, z in zip(masks, depths, strict=True)
    )
    camera = SimpleNamespace(shot=lambda *args: (next(shots), None), K=lambda: np.eye(3), _pose=lambda msg: np.eye(4))
    result = detection.detect_container(camera, "container", 0.5, 3, 0.85, 5, lambda msg: None, lambda: None)
    assert result["attempts"] == len(depths)
    assert result["point"][2] == pytest.approx(0.5)
    assert result["iou_min"] == 1.0
    assert len(result["info"]["depths_m"]) == 3
    assert all(len(d) == 50 for d in result["info"]["depths_m"])
    if disagree:
        assert result["rejected"]["masks disagree with the streak"] == 2
