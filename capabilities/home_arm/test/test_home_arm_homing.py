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


def test_a_stepped_path_rests_at_both_ends_and_no_joint_outruns_the_speed():
    path = np.array([[0.0] * 7, [0.1] + [0.0] * 6, [0.2, 0.05] + [0.0] * 5])
    pts = homing.stepped(path, speed=0.1)
    assert pts[0][0] == 0.0 and np.allclose(pts[0][1], path[0])
    assert np.allclose(pts[-1][1], path[-1])
    assert np.all(pts[0][2] == 0) and np.all(pts[-1][2] == 0)
    assert max(float(np.abs(p[2]).max()) for p in pts) <= 0.1 + 1e-3
    q = np.array([p[1] for p in pts])
    assert np.abs(np.diff(q, axis=0)).max() < 0.01  # smooth: no jump between samples


def test_a_short_stepped_path_takes_at_least_the_minimum_and_a_still_one_is_nothing():
    pts = homing.stepped(np.array([[0.0] * 7, [0.01] + [0.0] * 6]))
    assert pts[-1][0] >= 1.0
    assert homing.stepped(np.array([[0.2] * 7, [0.2] * 7])) == []
    assert homing.stepped(np.array([[0.2] * 7])) == []


def test_swing_is_how_much_wider_than_the_straight_way_a_plan_takes_a_joint():
    # flatbot 2026-11-10: joint_1 from 1.006 out to 1.87 and back to HOME's 0.4425
    start = np.array([1.006, -0.312, 2.699, -2.224, -2.790, -0.889, -0.623])
    goal = homing.on_branch(HOME, start)
    j1 = np.r_[np.linspace(1.006, 1.87, 20), np.linspace(1.87, 0.4425, 30)]
    path = start + np.linspace(0, 1, 50)[:, None] * (goal - start)
    path[:, 0] = j1
    extra, j = homing.swing(path, goal)
    assert j == 0 and extra == pytest.approx((1.87 - 0.4425) - (1.006 - 0.4425))
    assert extra > homing.SWING_MARGIN_RAD
    straight = start + np.linspace(0, 1, 50)[:, None] * (goal - start)
    assert homing.swing(straight, goal)[0] == pytest.approx(0.0, abs=1e-9)


def test_a_whole_turn_to_the_same_angle_swings_two_pi():
    start = np.zeros(7)
    path = np.zeros((65, 7))
    path[:, 6] = np.linspace(0.0, 2 * np.pi, 65)  # joint_7 round once, back at 0
    assert homing.swing(path, np.zeros(7)) == (pytest.approx(2 * np.pi), 6)
    # ... which its end alone cannot tell
    assert homing.end_error(path[-1], start)[0] == pytest.approx(0.0, abs=1e-9)


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


def test_plans_that_do_not_join_are_refused_a_whole_turn_apart_too():
    a = SimpleNamespace(
        t=np.array([0.0, 1.0]), q=np.array([[0.0] * 7, [1.0] * 7]), qd=np.zeros((2, 7))
    )
    off = SimpleNamespace(
        t=np.array([0.0, 0.5]), q=np.array([[1.005] + [1.0] * 6, [2.0] * 7]), qd=np.zeros((2, 7))
    )
    with pytest.raises(ValueError, match="plan 2 starts 0.0050 rad .joint 1."):
        homing.joined([a, off])
    turned = SimpleNamespace(
        t=np.array([0.0, 0.5]),
        q=np.array([[1.0] * 6 + [1.0 + 2 * np.pi], [2.0] * 7]),
        qd=np.zeros((2, 7)),
    )
    with pytest.raises(ValueError, match="joint 7"):
        homing.joined([a, turned])


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


def test_the_plan_free_steps_speed_cap_holds_whatever_arm_joint_speed_is_set_to(monkeypatch):
    """arm.joint_speed sets rammp_arm.JOINT_SPEED at startup, after homing is imported: the step is
    capped by the speed as it is then, not by a copy taken at import."""
    import rammp_arm

    path = np.array([[0.0] * 7, [0.3] + [0.0] * 6])
    for joint_speed in (0.85, 0.4, 1.5):
        monkeypatch.setattr(rammp_arm, "JOINT_SPEED", joint_speed)
        pts = homing.stepped(path)
        peak = max(float(np.abs(qd).max()) for _t, _q, qd in pts)
        assert peak == pytest.approx(homing.ESCAPE_SPEED_RAD_S, rel=0.05)
