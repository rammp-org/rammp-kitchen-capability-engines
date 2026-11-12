"""Which way round a continuous joint is written (runtime/branches).

Bench 2026-09-25: the arm turned joint_3 a full circle (4.7 s) on its way
to a box on the left. At HOME joint_3 sits at 3.142, a hair past +pi, and
the transport reports it wrapped: -3.1411 that day. The planner, which
bounds the joint at +/-6.0 rad and plans between the numbers it is given,
went from -3.14 to the +3.x its IK solution sat on — the long way round."""

import math

import numpy as np
import pytest

from box_opening.constants import HOME
from box_opening.runtime.branches import on_branch, planner_branch

HOME_AS_REPORTED = [
    0.0671,
    -1.8969,
    3.1462,
    -2.2691,
    -0.2109,
    -0.6831,
    1.7251,
]  # /joint_states, flatbot 2026-10-21: joint_3 reported on +branch


def test_the_planner_is_handed_home_on_its_own_side_of_the_wrap():
    q = planner_branch(HOME_AS_REPORTED)
    assert q[2] == pytest.approx(HOME[2], abs=2e-3)  # +3.1462 -> -3.137: one angle, the planner's way round
    assert q[0] == pytest.approx(HOME_AS_REPORTED[0]) and q[6] == pytest.approx(HOME_AS_REPORTED[6])
    assert q[1] == HOME_AS_REPORTED[1] and q[3] == HOME_AS_REPORTED[3]  # bounded joints are never touched
    for a, b in zip(q, HOME_AS_REPORTED, strict=False):  # the same physical angles
        assert math.isclose(math.cos(a), math.cos(b), abs_tol=1e-9) and math.isclose(
            math.sin(a), math.sin(b), abs_tol=1e-9
        )
    # a reading already on HOME's side is left alone
    assert planner_branch(HOME) == pytest.approx(HOME)
    # a branch the planner's bounds (+/-6.0) cannot hold is left as reported
    far = list(HOME)
    far[2] = -0.1  # joint_3 near 0: nearest_branch gives -0.1 directly, within ±6.0
    assert planner_branch(far)[2] == pytest.approx(-0.1)


def test_a_live_reading_lines_up_with_a_planned_path_past_pi():
    path_q3 = 3.25  # a press that took joint_3 past +pi
    reported = path_q3 - 2 * math.pi  # -3.033, as the transport reports it
    ref = list(HOME)
    ref[2] = 3.2
    live = list(HOME)
    live[2] = reported
    assert on_branch(live, ref)[2] == pytest.approx(path_q3)


def _traj(rows):
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for k, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.time_from_start.sec, p.time_from_start.nanosec = 0, int(2e7 * (k + 1))
        t.points.append(p)
    return t


def test_the_push_and_the_recoil_start_where_the_arm_is_on_a_path_past_pi():
    """The push is cut from the touch's own path and the recoil retraces it,
    both from the NEAREST sample to the live stop. With joint_3 past +pi
    the reading is 2*pi from its own path: the push began with a full-turn
    jump in it (the driver's continuity gate refuses that), the recoil at
    the wrong end of the stroke."""
    from box_opening.runtime.retime import forward_tail, reverse_tail

    down = [list(HOME) for _ in range(50)]
    for k, q in enumerate(down):
        q[2] = 3.2 + 0.001 * k  # joint_3 past +pi all the way down
        q[5] = 0.9 + 0.01 * k  # the stroke itself
    stop = list(down[30])
    stop[2] -= 2 * math.pi  # as the transport reports it
    push = forward_tail(_traj(down), stop, arc_rad=0.05)
    assert push is not None
    assert push[0][2] == pytest.approx(down[30][2])  # on the path's side of the wrap
    assert max(np.abs(np.diff(push, axis=0)).max(axis=0)) < 0.05  # no jump anywhere
    back = reverse_tail(_traj(down), progress=0.62, live=stop, arc_rad=0.05)
    assert back is not None and back[0][2] == pytest.approx(down[30][2])
    assert max(np.abs(np.diff(back, axis=0)).max(axis=0)) < 0.05
    assert [q[5] for q in back] == sorted([q[5] for q in back], reverse=True)  # only back out


def test_the_client_hands_the_planner_its_start_on_homes_side():
    from box_opening.runtime.client import PlannerClient

    c = PlannerClient.__new__(PlannerClient)
    c._plan_pose = c._plan_joints = None
    c._call = lambda client, goal: goal  # the goal as it would go out
    g = c._plan_pose_once([0.42, 0.22, 0.21], [1.0, 0.0, 0.0, 0.0], HOME_AS_REPORTED)
    assert list(g.start_joints)[2] == pytest.approx(HOME[2], abs=2e-3)
    g = c.plan_to_joints(HOME, HOME_AS_REPORTED)
    assert list(g.start_joints)[2] == pytest.approx(HOME[2], abs=2e-3)


class _Plan:
    def __init__(self, rows, success=True):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        self.success, self.message, self.planning_time = success, "ok", 0.1
        self.trajectory = JointTrajectory()
        for row in rows:
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in row]
            self.trajectory.points.append(p)


def _client(answers):
    """A PlannerClient whose planner answers from `answers` (a joint goal ->
    a plan) and records every goal it was sent."""
    from box_opening.runtime.client import PlannerClient

    c = PlannerClient.__new__(PlannerClient)
    c._plan_pose = "pose"
    c._plan_joints = "joints"
    c.sent = []

    def call(kind, goal):
        c.sent.append((kind, goal))
        return answers(kind, goal)

    c._call = call
    return c


def test_a_joint_goal_is_never_a_winding():
    """HOME written +3.142 with the arm reading -3.1411, and a wrist goal
    written a whole turn away: the goal the planner is sent is the same
    configuration, every continuous joint the short way round."""
    c = _client(lambda kind, goal: goal)
    far = list(HOME)
    far[6] += 2 * math.pi + 0.1  # joint_7 written a turn and a bit away
    g = c.plan_to_joints(far, HOME_AS_REPORTED)
    start, goal = list(g.start_joints), list(g.target_joints)
    assert goal[6] == pytest.approx(HOME[6] + 0.1, abs=2e-3)
    assert all(abs(a - b) <= math.pi for a, b in zip(goal, start, strict=False))
    assert goal[1] == far[1] and goal[3] == far[3]  # bounded joints exactly as asked


def test_the_quickest_posture_to_a_box_on_the_left_only_pitches_the_wrist():
    """Flatbot 2026-10-21, box at [0.422, 0.223]: the planner's own pick
    turns joint_5 / joint_7 3.2-3.6 rad (a 4.7 s approach). The posture
    chosen now pitches the wrist down and barely turns it."""
    from box_opening.models.container import ContainerModel
    from box_opening.runtime.branches import joint_time
    from box_opening.runtime.kinematics import ArmChain

    m = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    start = planner_branch(HOME_AS_REPORTED)
    q = ArmChain().nearest_solution(start, [0.4216, 0.2232, 0.214], m.press_quat([0.4216, 0.2232, 0.094]))
    assert q is not None
    assert abs(q[4] - start[4]) < 0.6 and abs(q[6] - start[6]) < 1.0  # no wrist spin
    assert joint_time(start, q) < 1.8
    assert all(abs(a - b) <= math.pi for a, b in zip(q, start, strict=False))  # and no joint the long way round


def test_a_plan_that_ends_the_long_way_round_is_re_planned_to_the_short_way():
    """When no quicker posture was found up front, the planner's own pose
    plan is flown — unless its end is quicker to reach another way: then
    the arm is sent there instead (the same tool pose)."""
    start = planner_branch(HOME_AS_REPORTED)
    long_end = list(start)
    long_end[6] = start[6] - 2 * math.pi + 0.2  # joint_7 wound almost a whole turn
    wound = _Plan([start, long_end])

    def answers(kind, goal):
        return wound if kind == "pose" else _Plan([start, list(goal.target_joints)])

    c = _client(answers)
    res = c._no_long_way(wound, HOME_AS_REPORTED)
    assert res is not wound
    end = list(res.trajectory.points[-1].positions)
    assert end[6] == pytest.approx(start[6] + 0.2, abs=1e-6)  # the same angle, the short way
    assert [k for k, _g in c.sent] == ["joints"]
    # a plan already the quickest way is flown as it is, with no second plan
    short = _Plan([start, [v + (0.2 if i == 6 else 0.0) for i, v in enumerate(start)]])
    c2 = _client(answers)
    assert c2._no_long_way(short, HOME_AS_REPORTED) is short and c2.sent == []
    # and when the re-plan fails, the original still stands
    c3 = _client(lambda kind, goal: _Plan([start], success=False))
    assert c3._no_long_way(wound, HOME_AS_REPORTED) is wound
