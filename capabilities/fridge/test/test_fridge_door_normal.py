"""fridge.door_normal: the handle model's ICP, offline (no ROS, no camera)."""

import numpy as np

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
