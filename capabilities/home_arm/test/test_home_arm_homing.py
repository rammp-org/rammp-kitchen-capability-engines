"""The joint-space helpers of the move home."""

# isort: off
import home_arm_testlib  # noqa: F401  (first: puts home_arm and capability_engine on sys.path)
# isort: on

from types import SimpleNamespace

import numpy as np
import pytest
from home_arm import homing

HOME = [0.4425, -1.5718, -2.9033, -2.1048, -0.7412, -0.9194, 2.1631]


def test_a_continuous_joint_goes_the_short_way_round_and_a_limited_one_as_given():
    start = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    goal = homing.on_branch(HOME, start)
    # joint_3: HOME's -2.9033 is 0.38 rad the short way from +3.0, so +3.38, the same angle
    assert goal[2] == pytest.approx(-2.9033 + 2 * np.pi)
    assert goal[1] == HOME[1] and goal[3] == HOME[3] and goal[5] == HOME[5]


def test_a_branch_past_the_planners_bounds_keeps_the_goals_own_value():
    # from 5.9, -0.5 the short way round is 5.783: inside the planner's 6.0, taken
    assert homing.on_branch([-0.5, 0, 0, 0, 0, 0, 0], [5.9, 0, 0, 0, 0, 0, 0])[0] == pytest.approx(
        -0.5 + 2 * np.pi
    )
    # -0.2 the short way round is 6.083: past 6.0, the goal's own -0.2 stands
    assert homing.on_branch([-0.2, 0, 0, 0, 0, 0, 0], [5.9, 0, 0, 0, 0, 0, 0])[0] == -0.2


def test_end_error_compares_continuous_joints_the_short_way():
    q = np.array(HOME)
    q[2] += 2 * np.pi
    assert homing.end_error(q, HOME)[0] == pytest.approx(0.0, abs=1e-9)
    q = np.array(HOME)
    q[3] += 0.05
    assert homing.end_error(q, HOME) == (pytest.approx(0.05), 3)


def test_waypoints_lie_on_the_joint_line_on_the_starts_branch():
    start = np.array([0.3, -0.9, -2.6, -1.8, -0.4, -0.7, 1.9])
    w = homing.waypoints(start, HOME, (0.5,))[0]
    assert w == pytest.approx((start + np.array(HOME)) / 2)


def test_a_timed_path_rests_at_both_ends_and_no_joint_outruns_the_speed():
    path = np.array([[0.0] * 7, [0.1] + [0.0] * 6, [0.2, 0.05] + [0.0] * 5])
    pts = homing.timed_path(path, speed=0.1, min_s=0.5)
    t = [p[0] for p in pts]
    assert t == pytest.approx([0.0, 1.0, 2.0])
    assert np.all(pts[0][2] == 0) and np.all(pts[-1][2] == 0)
    assert np.abs(pts[1][2]).max() <= 0.1 + 1e-9


def test_a_short_timed_path_takes_at_least_its_minimum():
    pts = homing.timed_path(np.array([[0.0] * 7, [0.01] + [0.0] * 6]), speed=0.15, min_s=1.0)
    assert pts[-1][0] == pytest.approx(1.0)


def test_joined_plans_run_on_in_time_and_drop_the_shared_sample():
    a = SimpleNamespace(
        t=np.array([0.0, 1.0]), q=np.array([[0.0] * 7, [1.0] * 7]), qd=np.zeros((2, 7))
    )
    b = SimpleNamespace(
        t=np.array([0.0, 0.5]), q=np.array([[1.0] * 7, [2.0] * 7]), qd=np.zeros((2, 7))
    )
    pts = homing.joined([a, b])
    assert [p[0] for p in pts] == [0.0, 1.0, 1.5]
    assert [p[1][0] for p in pts] == [0.0, 1.0, 2.0]


LIMITS = [
    (None, None),
    (-2.24, 2.24),
    (None, None),
    (-2.57, 2.57),
    (None, None),
    (-2.09, 2.09),
    (None, None),
]


def test_near_limits_names_each_limited_joint_nearer_than_needed():
    q = np.array(HOME)
    q[5] = 2.09 - 0.094
    assert homing.near_limits(q, LIMITS) == [(5, pytest.approx(0.094))]
    assert homing.near_limits(np.array(HOME), LIMITS) == []


def test_the_first_limit_step_just_clears_the_margin_and_each_next_is_a_stride_longer():
    q = np.array(HOME)
    q[5] = 2.09 - 0.094
    steps = homing.limit_steps(q, HOME, LIMITS)
    f0, q1 = steps[0]
    assert 2.09 - q1[5] == pytest.approx(homing.LIMIT_FIRST_RAD)
    span = float(np.abs(homing.on_branch(HOME, q) - q).max())
    assert steps[1][0] - f0 == pytest.approx(homing.LIMIT_STRIDE_RAD / span)
    assert all(f * span <= homing.LIMIT_MAX_STEP_RAD + 1e-9 for f, _ in steps)


def test_a_home_too_near_a_limit_is_a_config_error():
    home = np.array(HOME)
    home[3] = 2.5
    with pytest.raises(ValueError, match="joint 4 is too near its limit"):
        homing.limit_steps(np.array(HOME), home, LIMITS)


def test_line_samples_a_joint_line_finely_from_end_to_end():
    pts = homing.line(np.zeros(7), np.r_[0.05, np.zeros(6)])
    assert len(pts) == 6 and pts[0][0] == 0.0 and pts[-1][0] == pytest.approx(0.05)
