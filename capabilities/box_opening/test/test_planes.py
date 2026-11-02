"""fit_plane's vectorised RANSAC against the loop it replaced.

The block-scored version must draw the same hypotheses in the same rng
order and keep the first best, so every caller (the scene locate's table
fits) gets the same plane it got from the loop."""

import numpy as np
import pytest

from box_opening.perception.planes import fit_plane


def _fit_plane_loop(points, tol_m=0.004, iters=300, max_points=40000, rng=None):
    """fit_plane as it was: one Python loop over the hypotheses."""
    P = np.asarray(points, dtype=float)
    if len(P) < 3:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    if len(P) > max_points:
        P = P[rng.choice(len(P), max_points, replace=False)]
    best_n, best_inl = 0, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        inl = np.abs((P - a) @ n) < tol_m
        cnt = int(inl.sum())
        if cnt > best_n:
            best_n, best_inl = cnt, inl
    if best_inl is None or best_n < 3:
        return None
    Q = P[best_inl]
    c = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    return n, c, best_n / float(len(P))


def _table_with_clutter(seed, n=20000, outliers=0.3):
    """A noisy horizontal table at z = 0.73 with a box's worth of points above it."""
    r = np.random.default_rng(seed)
    k = int(n * (1 - outliers))
    table = np.column_stack([r.uniform(-0.5, 0.5, k), r.uniform(-0.4, 0.4, k), 0.73 + r.normal(0, 0.002, k)])
    clutter = np.column_stack(
        [
            r.uniform(-0.1, 0.1, n - k),
            r.uniform(-0.1, 0.1, n - k),
            r.uniform(0.74, 0.90, n - k),
        ]
    )
    return np.vstack([table, clutter])


def _mm_quantised(seed, n=4000):
    """A cloud on a millimetre grid: many hypotheses tie on their count."""
    r = np.random.default_rng(seed)
    return np.round(r.uniform(-0.05, 0.05, (n, 3)) * 1000.0) / 1000.0


def _degenerate(seed, n=600):
    """Duplicated points and points on one line: most triples have no normal."""
    r = np.random.default_rng(seed)
    t = r.uniform(0, 1, n // 2)
    line = np.column_stack([t, 2 * t, -t])
    dup = np.repeat(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), n // 4, axis=0)
    return np.vstack([line, dup])


def _assert_same(got, want):
    assert (got is None) == (want is None)
    if want is None:
        return
    np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got[1], want[1], rtol=0, atol=1e-12)
    assert got[2] == want[2]


@pytest.mark.parametrize(
    "cloud, kwargs",
    [
        (_table_with_clutter(1), dict(tol_m=0.008, iters=150, max_points=3000)),
        (_table_with_clutter(2, outliers=0.6), dict(tol_m=0.008, iters=150, max_points=3000)),
        (_mm_quantised(3), dict(tol_m=0.004, iters=300)),
        (_mm_quantised(4, n=50), dict(tol_m=0.002, iters=200)),
        (_degenerate(5), dict(tol_m=0.004, iters=100)),
        (np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dict(iters=20)),
    ],
)
def test_the_vectorised_fit_matches_the_loop(cloud, kwargs):
    _assert_same(fit_plane(cloud, **kwargs), _fit_plane_loop(cloud, **kwargs))


def test_the_fit_still_finds_the_table():
    n, c, frac = fit_plane(_table_with_clutter(6), tol_m=0.008, iters=150, max_points=3000)
    assert abs(abs(n[2]) - 1.0) < 1e-3
    assert abs(c[2] - 0.73) < 2e-3
    assert 0.6 < frac < 0.8


def test_a_cloud_of_one_repeated_point_has_no_plane():
    assert fit_plane(np.tile([0.3, 0.1, 0.7], (500, 1))) is None


def test_no_hypotheses_means_no_plane():
    assert fit_plane(_table_with_clutter(7), iters=0) is None


def test_the_rng_ends_where_the_loop_left_it():
    cloud = _table_with_clutter(8)
    mine, ref = np.random.default_rng(42), np.random.default_rng(42)
    _assert_same(
        fit_plane(cloud, tol_m=0.008, iters=150, max_points=3000, rng=mine),
        _fit_plane_loop(cloud, tol_m=0.008, iters=150, max_points=3000, rng=ref),
    )
    assert mine.random() == ref.random()


def test_hypotheses_scored_across_several_blocks_match_the_loop():
    # 40000 points -> blocks of 100 hypotheses: 300 iters take three blocks
    cloud = _table_with_clutter(9, n=50000)
    kwargs = dict(tol_m=0.004, iters=300, max_points=40000)
    _assert_same(fit_plane(cloud, **kwargs), _fit_plane_loop(cloud, **kwargs))


class _ScriptedRng:
    """Hands out collinear (no-normal) triples except at the `good` draws."""

    def __init__(self, good):
        self.good, self.k = set(good), 0

    def choice(self, n, size, replace=True):
        k, self.k = self.k, self.k + 1
        return np.array([3, 4, 5]) if k in self.good else np.array([0, 1, 2])


@pytest.mark.parametrize("good", [0, 99, 100, 199, 299])
def test_the_one_good_hypothesis_is_scored_in_whichever_block_it_falls(good):
    # 40000 points -> blocks of 100; the only usable triple sits at a block's edge
    cloud = _table_with_clutter(10, n=40000, outliers=0.2)
    cloud[0:3] = [[0.0, 0.0, 0.73], [0.1, 0.0, 0.73], [0.2, 0.0, 0.73]]
    cloud[3:6] = [[0.0, 0.0, 0.73], [0.3, 0.0, 0.73], [0.0, 0.3, 0.73]]
    kwargs = dict(tol_m=0.008, iters=300, max_points=40000)
    got = fit_plane(cloud, rng=_ScriptedRng([good]), **kwargs)
    _assert_same(got, _fit_plane_loop(cloud, rng=_ScriptedRng([good]), **kwargs))
    assert got is not None and got[2] > 0.75
