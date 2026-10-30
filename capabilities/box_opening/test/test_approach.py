"""The vertical final approach: a planner waypoint, then the arm's own
straight line (the v1.0.0 planner's PlanToPose carries no approach_offset_m,
and a plan between two aligned poses bows sideways in between)."""

import math

import numpy as np
import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening.runtime.approach import plan_with_vertical_approach

QUAT = [1.0, 0.0, 0.0, 0.0]
NAMES = ["joint_%d" % i for i in range(1, 8)]
GOAL = [0.45, 0.0, 0.09]
START = [0.0] * 7


def _traj(rows, times):
    t = JointTrajectory()
    t.joint_names = list(NAMES)
    for row, tt in zip(rows, times, strict=False):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        p.time_from_start.sec = int(tt)
        p.time_from_start.nanosec = int(round((tt - int(tt)) * 1e9))
        t.points.append(p)
    return t


def _secs(p):
    return p.time_from_start.sec + p.time_from_start.nanosec * 1e-9


class Plan:
    """A PlanToPose result: success, message, trajectory, planning_time."""

    def __init__(self, success, trajectory=None, planning_time=0.0, message="ok"):
        self.success = success
        self.message = message
        self.trajectory = trajectory
        self.planning_time = planning_time


class Planner:
    """Stands in for the PlanToPose round trip: records every request and
    answers from a script, in order."""

    def __init__(self, answers):
        self.calls = []
        self.answers = list(answers)

    def __call__(self, xyz, quat, start_joints):
        self.calls.append((list(xyz), list(quat), list(start_joints)))
        return self.answers.pop(0)


def _above():
    return Plan(True, _traj([[0.0] * 7, [0.1] * 7, [0.2] * 7], [0.0, 1.0, 2.0]), 0.2)


def _down():
    return Plan(True, _traj([[0.2] * 7, [0.25] * 7, [0.3] * 7], [0.0, 0.5, 1.0]), 0.3)


class Line:
    """Stands in for the arm chain: records the request, answers a script."""

    def __init__(self, points=None, why=None):
        self.points, self.why, self.calls = points, why, []

    def straight_line(self, q, xyz, quat):
        self.calls.append((list(q), list(xyz), list(quat)))
        return (None, self.why) if self.points is None else (self.points, None)


LINE = [[0.2] * 7, [0.25] * 7, [0.3] * 7]


def test_without_an_offset_the_goal_is_planned_directly():
    only = Plan(True, _traj([[0.0] * 7, [0.1] * 7], [0.0, 1.0]))
    planner = Planner([only])
    assert plan_with_vertical_approach(planner, GOAL, QUAT, START, 0.0) is only
    assert planner.calls == [(GOAL, QUAT, START)]


def test_an_offset_plans_to_a_waypoint_above_then_flies_the_line_from_its_end():
    planner = Planner([_above()])
    line = Line(LINE)
    got = plan_with_vertical_approach(planner, GOAL, QUAT, START, 0.06, chain_model=line)
    assert planner.calls == [([0.45, 0.0, 0.15], QUAT, START)]  # ONE planner call
    # the line starts at the waypoint plan's end joints, aimed at the goal
    assert line.calls == [([0.2] * 7, GOAL, QUAT)]
    assert got.success and "straight descent" in got.message
    assert got.planning_time == pytest.approx(0.2)


def test_the_plan_and_the_line_fly_as_one_trajectory_with_strictly_increasing_time():
    got = plan_with_vertical_approach(Planner([_above()]), GOAL, QUAT, START, 0.06, chain_model=Line(LINE))
    pts = got.trajectory.points
    assert [p.positions[0] for p in pts] == pytest.approx([0.0, 0.1, 0.2, 0.25, 0.3])
    ts = [_secs(p) for p in pts]
    assert ts[:3] == pytest.approx([0.0, 1.0, 2.0]) and all(b > a for a, b in zip(ts, ts[1:], strict=False))
    assert list(got.trajectory.joint_names) == NAMES
    assert all(len(p.velocities) == 7 and len(p.accelerations) == 7 for p in pts)  # the driver's gate


def test_a_refused_line_falls_back_to_the_planners_descent_and_says_so():
    planner = Planner([_above(), _down()])
    got = plan_with_vertical_approach(
        planner, GOAL, QUAT, START, 0.06, chain_model=Line(None, "a joint limit 40 mm down")
    )
    assert planner.calls[1][0] == GOAL and planner.calls[1][2] == [0.2] * 7
    assert got.success and "PLANNER" in got.message and "joint limit" in got.message
    assert [p.positions[0] for p in got.trajectory.points] == pytest.approx([0.0, 0.1, 0.2, 0.25, 0.3])
    assert got.planning_time == pytest.approx(0.5)


def test_a_refused_waypoint_refuses_the_approach_without_planning_the_descent():
    planner = Planner([Plan(False, message="IK_FAIL")])
    got = plan_with_vertical_approach(planner, GOAL, QUAT, START, 0.06)
    assert not got.success and "IK_FAIL" in got.message
    assert len(planner.calls) == 1


def test_a_refused_descent_refuses_the_approach():
    planner = Planner([_above(), Plan(False, message="INVALID_START")])
    got = plan_with_vertical_approach(planner, GOAL, QUAT, START, 0.06, chain_model=Line(None, "singular"))
    assert not got.success and "INVALID_START" in got.message


def test_the_real_chain_flies_straight_below_a_real_waypoint():
    """The planner's own waypoint joints for the bearing -15 deg box (live
    probe 2026-09-16): below them the trajectory must stay on the vertical
    to 0.1 mm — the planner's plan for the same stretch bowed 7 mm."""
    from box_opening.models.container import attitude_quat
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain()
    q_ab = [0.1208, 0.5375, 3.3522, -1.8057, -0.1478, -0.8136, 0.1407]
    goal = [0.4264, -0.1151, 0.0815]
    quat = attitude_quat([180.0, 0.0, 0.0], math.atan2(goal[1], goal[0]))
    first = Plan(True, _traj([[0.0] * 7, q_ab], [0.0, 1.0]), 0.2)
    got = plan_with_vertical_approach(Planner([first]), goal, quat, [0.0] * 7, 0.06)  # the default chain
    assert got.success and "straight descent" in got.message
    tail = got.trajectory.points[2:]
    assert len(tail) >= 29
    lat = [np.hypot(*(chain.fk(p.positions)[1][:2] - np.array(goal[:2]))) for p in tail]
    assert max(lat) < 1e-4
    _R, t_end = chain.fk(tail[-1].positions)
    assert np.allclose(t_end, goal, atol=1e-4)


def test_no_answer_from_the_planner_is_no_plan():
    assert plan_with_vertical_approach(Planner([None]), GOAL, QUAT, START, 0.06) is None


def test_the_line_carries_a_real_velocity_profile_not_placeholder_stamps():
    """A guarded descent flown on its own reaches the driver as planned (the
    Runner re-times only unguarded and merged groups). With placeholder
    stamps and zero velocity at every 2 mm waypoint the arm stop-started
    down the whole line: 2.98 s for the 45 mm grip descent and 3 Nm of
    free-air torque ripple, live 2026-09-17. The line is a rest-to-rest
    profile now, and passes the gates in front of the driver as it is."""
    from box_opening.constants import JOINT_VMAX
    from box_opening.runtime import driver
    from box_opening.runtime.approach import line_trajectory

    pts = [[0.002 * 4.0 * k] * 7 for k in range(31)]  # 60 mm of line at ~4 rad/m
    line = line_trajectory(NAMES, pts)
    assert [list(p.positions) for p in line.points] == pts  # positions untouched
    speeds = [max(abs(v) for v in p.velocities) for p in line.points]
    assert speeds[0] == 0.0 and speeds[-1] == 0.0  # rest to rest
    assert all(v > 0.0 for v in speeds[1:-1])  # and nowhere in between
    assert driver.refusal(line, pts[0], JOINT_VMAX) is None
    # chained behind a planner trajectory, the whole still passes, slowed
    got = plan_with_vertical_approach(Planner([_above()]), GOAL, QUAT, START, 0.06, chain_model=Line(pts))
    flown = driver.dilate(got.trajectory, 0.35)
    ts = [_secs(p) for p in flown.points]
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))


def test_every_vertical_approach_says_where_its_descent_begins():
    """The guard of a descent flown on its own arms where the descent
    begins (runner._run_motion). Both ways of building one say where that
    is — the straight line, and the planner's own descent when the line is
    refused — and a re-fit moves it past the lateral jog it inserts."""
    got = plan_with_vertical_approach(Planner([_above()]), GOAL, QUAT, START, 0.06, chain_model=Line(LINE))
    assert got.waypoint == (2, [0.2] * 7)
    fallback = plan_with_vertical_approach(
        Planner([_above(), _down()]), GOAL, QUAT, START, 0.06, chain_model=Line(None, why="a joint limit")
    )
    assert fallback.success and "PLANNER descent" in fallback.message
    assert fallback.waypoint == (2, [0.2] * 7)


def test_a_plan_is_written_on_the_branch_of_the_line_it_continues():
    """The planner is handed planner_branch(start), HOME's side of the wrap:
    a plan-free line ending at joint_3 +0.450 gets a home plan starting at
    -5.833 (review 2026-10-27), and one goal through both turns joint_3 a
    full revolution at the junction. on_branch_of moves each continuous
    joint (1, 3, 5, 7) by the one whole number of turns that puts the
    plan's first point within half a turn of the line's end: every
    difference kept — the same motion to the arm, whose driver flies the
    differences — the other joints untouched, the plan itself as it was."""
    import copy

    from box_opening.runtime.approach import on_branch_of
    from box_opening.runtime.branches import CONTINUOUS_JOINTS

    turn = 2.0 * math.pi
    end = [0.2, -0.3, 0.450, 1.0, -0.4, 0.8, 1.7]  # the line's end, as the arm's joints were read
    start = [0.2 - turn, -0.3 + 1e-7, 0.450 - turn, 1.0, -0.4 + turn, 0.8, 1.7 - 2.0 * turn]
    rows = [start]
    for d in ((0.1, -0.2, 0.3, 0.05, -0.1, 0.2, -0.3), (0.3, -0.5, -2.0, 0.1, 0.4, -0.2, 0.6)):
        rows.append([v + dv for v, dv in zip(start, d, strict=False)])
    plan = _traj(rows, [0.0, 0.5, 1.2])
    for p, v in zip(plan.points, (0.0, 0.7, 0.0), strict=False):
        p.velocities = [v] * 7
    before = copy.deepcopy(plan)
    out = on_branch_of(plan, end)
    got, was = np.array([p.positions for p in out.points]), np.array(rows)
    turns = (got - was) / turn
    assert turns == pytest.approx(np.round(turns), abs=1e-9)  # whole turns ...
    assert [int(round(v)) for v in turns[0]] == [1, 0, 1, 0, -1, 0, 2]
    assert (np.round(turns) == np.round(turns[0])).all()  # ... the same for every point
    others = [j for j in range(7) if j not in CONTINUOUS_JOINTS]
    assert (got[:, others] == was[:, others]).all()
    assert np.diff(got, axis=0) == pytest.approx(np.diff(was, axis=0), abs=1e-12)  # every difference kept
    assert all(abs(got[0][j] - end[j]) <= math.pi for j in CONTINUOUS_JOINTS)
    assert got[0] == pytest.approx(end, abs=1e-6)
    assert [_secs(p) for p in out.points] == [_secs(p) for p in plan.points]
    assert [list(p.velocities) for p in out.points] == [list(p.velocities) for p in plan.points]
    assert [list(p.positions) for p in plan.points] == [list(p.positions) for p in before.points]  # a copy
    # a plan already on the line's side of the wrap is left as it is, and so is one half a turn less off
    assert [list(p.positions) for p in on_branch_of(out, end).points] == [list(p.positions) for p in out.points]
    near = _traj([[v + 3.0 for v in end], [v + 3.1 for v in end]], [0.0, 1.0])
    assert [list(p.positions) for p in on_branch_of(near, end).points] == [list(p.positions) for p in near.points]
