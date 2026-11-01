"""The scene lift's fast paths return exactly what the slow ones did.

median_depth sorts instead of calling nanmedian, color_cloud projects with
K directly when there is no distortion, and robust_yaw scores every angle in
one percentile call. Each is pinned here against the reference it replaced.
"""

import warnings

import cv2
import numpy as np
import pytest

from box_opening.perception.scene import color_cloud, median_depth
from box_opening.perception.scene_source import robust_yaw


def _ref_median(depths):
    stack = np.stack(depths)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(stack > 0, stack, np.nan), axis=0)
    return np.nan_to_num(med, nan=0.0)


def _frames(n, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    f = [rng.uniform(0.3, 1.5, (40, 50)).astype(dtype) for _ in range(n)]
    for d in f:
        d[rng.random(d.shape) < 0.3] = 0  # random holes
    for d in f:
        d[0, :5] = 0  # no depth in any frame
    if n >= 3:
        f[0][1, 0], f[1][1, 0], f[2][1, 0] = 0.7, 0, 0  # valid in 1 of 3
        f[0][1, 1], f[1][1, 1], f[2][1, 1] = 0.7, 0.9, 0  # valid in 2 of 3
        for d in f[3:]:
            d[1, :2] = 0
    return f


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_median_depth_equals_nanmedian(n):
    f = _frames(n, seed=n)
    got = median_depth(f)
    np.testing.assert_array_equal(got, _ref_median(f).astype(np.float32))
    assert np.all(got[0, :5] == 0)
    if n >= 3:
        assert got[1, 0] == np.float32(0.7)
        assert got[1, 1] == (np.float32(0.7) + np.float32(0.9)) * np.float32(0.5)


def test_median_depth_dtype_and_empty():
    assert median_depth([]) is None
    assert median_depth(_frames(3, np.float32)).dtype == np.float32
    f64 = _frames(3, np.float64)
    got = median_depth(f64)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, _ref_median(f64))


def test_median_depth_nan_is_a_hole():
    f = _frames(3)
    f[1][5, 5] = np.nan
    np.testing.assert_array_equal(median_depth(f), _ref_median(f).astype(np.float32))


def _rot(rx, rz):
    cx, sx, cz, sz = np.cos(rx), np.sin(rx), np.cos(rz), np.sin(rz)
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])


def _scene():
    rng = np.random.default_rng(7)
    depth = rng.uniform(0.5, 1.2, (60, 80)).astype(np.float32)
    depth[rng.random(depth.shape) < 0.1] = 0
    kd = np.array([[300.0, 0, 41.0], [0, 302.0, 29.5], [0, 0, 1]])
    k = np.array([[610.0, 0, 322.0], [0, 612.0, 241.0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = _rot(0.02, -0.015)
    T[:3, 3] = [0.03, -0.004, 0.002]
    return depth, kd, k, T


def _ref_cloud(depth, kd, k, dist, T):
    from box_opening.perception.planes import cloud_from_depth

    pts, _u, _v = cloud_from_depth(depth, kd)
    pc = pts @ T[:3, :3].T + T[:3, 3]
    uv, _ = cv2.projectPoints(
        pc.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        k,
        None if dist is None or not len(np.atleast_1d(dist)) else np.asarray(dist, float),
    )
    return pc, uv.reshape(-1, 2)


@pytest.mark.parametrize("dist", [np.zeros(5), [], None, [0.1, 0.0, 0.001, 0.0, 0.0]])
def test_color_cloud_matches_project_points(dist):
    depth, kd, k, T = _scene()
    pc, uv = color_cloud(depth, kd, k, dist, T)
    pc_ref, uv_ref = _ref_cloud(depth, kd, k, dist, T)
    np.testing.assert_array_equal(pc, pc_ref)
    assert uv.shape == uv_ref.shape
    assert np.max(np.abs(uv - uv_ref)) < 1e-9


def _ref_yaw(xy):
    xy = np.asarray(xy, float)

    def area(theta):
        c, s = np.cos(-theta), np.sin(-theta)
        r = xy @ np.array([[c, -s], [s, c]]).T
        lo, hi = np.percentile(r, [3, 97], axis=0)
        return float(np.prod(hi - lo))

    coarse = np.radians(np.arange(0.0, 90.0, 1.0))
    best = coarse[int(np.argmin([area(t) for t in coarse]))]
    fine = best + np.radians(np.arange(-1.0, 1.0, 0.1))
    best = fine[int(np.argmin([area(t) for t in fine]))]
    return float(best % (np.pi / 2))


def _square(rng, n, yaw):
    p = rng.uniform(-0.05, 0.05, (n, 2)) + rng.normal(0, 0.002, (n, 2))
    c, s = np.cos(yaw), np.sin(yaw)
    return p @ np.array([[c, -s], [s, c]]).T + rng.uniform(-0.3, 0.3, 2)


@pytest.mark.parametrize("seed", range(20))
def test_robust_yaw_matches_loop(seed):
    rng = np.random.default_rng(100 + seed)
    xy = _square(rng, int(rng.integers(150, 2001)), rng.uniform(0, np.pi))
    assert abs(robust_yaw(xy) - _ref_yaw(xy)) < 1e-12


@pytest.mark.parametrize("deg", [0.0, 45.0])
def test_robust_yaw_axis_and_diagonal(deg):
    xy = _square(np.random.default_rng(3), 800, np.radians(deg))
    assert abs(robust_yaw(xy) - _ref_yaw(xy)) < 1e-12


def test_robust_yaw_ties_break_as_before():
    # a clean lattice ties mirror angles exactly (3 deg and 87 deg); the
    # stacked rotation must round as the old per-angle one did, or the tie
    # breaks the other way
    g = np.linspace(-0.05, 0.05, 30)
    xy = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    for deg in (0.0, 45.0):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        r = xy @ np.array([[c, -s], [s, c]]).T
        assert robust_yaw(r) == _ref_yaw(r)


def test_fast_paths_are_taken(monkeypatch):
    # the slow calls these replace must stay out of the common path: no
    # nanmedian, no projectPoints without distortion, two percentile calls
    calls = []
    real = np.percentile

    def boom(*_a, **_k):
        raise AssertionError("slow path taken")

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(np, "nanmedian", boom)
    monkeypatch.setattr(cv2, "projectPoints", boom)
    median_depth(_frames(3))
    depth, kd, k, T = _scene()
    color_cloud(depth, kd, k, np.zeros(5), T)
    monkeypatch.setattr(np, "percentile", counted)
    robust_yaw(_square(np.random.default_rng(1), 300, 0.3))
    assert len(calls) == 2
