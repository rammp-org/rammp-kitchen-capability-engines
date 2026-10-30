import math

import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening.constants import GAINS_MEDIUM, GAINS_STIFF, HOME
from box_opening.geometry import ang_diff
from box_opening.runtime.branches import planner_branch
from box_opening.runtime.guards import GuardSpec
from box_opening.runtime.legs import (
    MERGE_GAP_RAD,
    Kind,
    Leg,
    VerifyCtx,
    can_merge,
    merge_groups,
)


def _q(v):
    """A joint vector; a plain number is a one-joint arm."""
    return [float(x) for x in v] if isinstance(v, list | tuple) else [float(v)]


def _traj(rows, dt=1.0):
    t = JointTrajectory()
    rows = [_q(row) for row in rows]
    t.joint_names = ["joint_%d" % (i + 1) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = row
        p.velocities = [0.0] * len(row)
        p.accelerations = [0.0] * len(row)
        p.time_from_start.sec = int(i * dt)
        t.points.append(p)
    return t


def leg(name, chain=0, speed=0.25, guard=None, verify=None, kind=Kind.MOTION, start=0.0, end=0.1):
    """A leg from `start` to `end`. Two legs merge only where the second
    starts at the first's end (legs.MERGE_GAP_RAD): consecutive legs here
    are chained, so each test's split has the one reason it names."""
    return Leg(
        name=name,
        kind=kind,
        traj=_traj([start, end]),
        speed=speed,
        guard=guard,
        world="full",
        chain=chain,
        target=("joints", _q(end)),
        goal_joints=_q(end),
        verify=verify,
    )


def test_same_chain_same_speed_merges():
    groups = merge_groups([leg("a"), leg("b", start=0.1, end=0.2)])
    assert [len(g) for g in groups] == [2]


def test_chain_break_splits():
    groups = merge_groups([leg("a", chain=0), leg("b", chain=1, start=0.1, end=0.2)])
    assert [len(g) for g in groups] == [1, 1]


def test_speed_change_no_longer_splits():
    """One re-timed profile owns the group; each leg's speed is its cruise
    fraction inside it (retime.py, 2026-09-03) — a speed change is not a
    stop any more."""
    groups = merge_groups([leg("a", speed=0.25), leg("b", speed=0.15, start=0.1, end=0.2)])
    assert [len(g) for g in groups] == [2]


def test_a_guarded_leg_takes_its_approach_with_it():
    """The approach and the descent it leads into fly as ONE execution.

    They used to be two goals with a controller round trip and a full stop
    between them, which is the pause a person does not make when reaching
    for something. Nothing merges AFTER the guarded leg: a trip must never
    strand queued motion behind it."""
    g = GuardSpec(touch_nm=3.0, trip="press")
    groups = merge_groups([leg("a"), leg("b", guard=g, start=0.1, end=0.2), leg("c", start=0.2, end=0.3)])
    assert [len(x) for x in groups] == [2, 1]
    assert groups[0][-1].guard is g


def test_a_guarded_leg_owns_its_group_so_it_may_carry_a_verify():
    """A verify normally cannot be absorbed as a non-lead member — only the
    execution's owner is verified. A guarded tail IS that owner."""

    def v(ctx):
        return True, ""

    g = GuardSpec(touch_nm=3.0, trip="press")
    groups = merge_groups([leg("a"), leg("b", guard=g, verify=v, start=0.1, end=0.2)])
    assert [len(x) for x in groups] == [2]
    # ... but an unguarded verify-carrier still closes the group it ends
    groups = merge_groups([leg("a"), leg("b", verify=v, start=0.1, end=0.2)])
    assert [len(x) for x in groups] == [1, 1]


def test_a_warped_descent_keeps_its_own_execution():
    """Warping and the group profile are two ways to do one job. A warped
    leg's timing is already baked in (speed 1.0), so re-timing it as part of
    a group would throw that profile away and cruise it at full speed into
    contact — the set-down and the grip descent stay alone."""
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    down = leg("place:lid:down", guard=g, speed=1.0, start=0.1, end=0.2)
    down.warp = (0.5, 0.15, 0.3)
    groups = merge_groups([leg("place:lid:transit"), down])
    assert [len(x) for x in groups] == [1, 1]


def test_two_guarded_legs_never_share_an_execution():
    g = GuardSpec(touch_nm=3.0, trip="press")
    groups = merge_groups([leg("a", guard=g), leg("b", guard=g, start=0.1, end=0.2)])
    assert [len(x) for x in groups] == [1, 1]


def test_verify_closes_group():
    def v(ctx):
        return True, ""

    groups = merge_groups([leg("a", verify=v), leg("b", start=0.1, end=0.2), leg("c", start=0.2, end=0.3)])
    assert [len(g) for g in groups] == [1, 2]


def test_gripper_never_merges():
    gl = Leg(
        name="close",
        kind=Kind.GRIPPER,
        traj=None,
        speed=0.0,
        guard=None,
        world="full",
        chain=0,
        target=None,
        goal_joints=None,
        gripper_cmd=0.8,
    )
    groups = merge_groups([leg("a"), gl, leg("b", start=0.1, end=0.2)])
    assert [len(g) for g in groups] == [1, 1, 1]


def test_can_merge_is_symmetric_gate():
    assert can_merge(leg("a"), leg("b", start=0.1, end=0.2))
    assert not can_merge(leg("a", chain=0), leg("b", chain=1, start=0.1, end=0.2))


def test_verify_ctx_defaults():
    ctx = VerifyCtx(outcome="arrived")
    assert ctx.gripper_pos is None


def test_a_verify_carrying_leg_is_never_absorbed_into_a_group():
    """Only the owning member of an execution gets verified, so a leg with
    a verify must not be appended as a non-lead member."""
    from box_opening.runtime.legs import can_merge

    a = leg("a")
    assert can_merge(a, leg("b", start=0.1, end=0.2))
    assert not can_merge(a, leg("b", verify=lambda ctx: (True, ""), start=0.1, end=0.2))


# -- a group flies as ONE trajectory: its legs meet, under one gains spec --


def test_legs_that_do_not_meet_never_share_an_execution():
    """Two plan-free lines on one chain, the second starting 0.2 rad from
    where the first ends. Merged, the group joined them as written and flew
    the gap as one 0.2 rad step between two samples, and
    check_like_executor passed it (review 2026-10-27)."""
    a = leg("a", start=0.0, end=0.1)
    b = leg("b", start=0.3, end=0.4)
    a.plan_free = b.plan_free = True
    assert not can_merge(a, b)
    assert [len(g) for g in merge_groups([a, b])] == [1, 1]


def test_a_whole_turn_between_two_legs_is_a_gap():
    """One angle written a turn apart is still a gap: retime_group joins
    the paths as written and the driver's quintic interpolates the raw
    numbers, so the junction would turn joint_3 a full revolution. Review
    2026-10-27: a plan-free line read from the arm ended at joint_3 +0.450;
    the home planned from it started at -5.833, the client having handed
    the planner planner_branch of that end."""
    rise_end = [-1.0, -0.2, 0.45, -1.1, -0.3, -1.0, 1.8]
    home_start = planner_branch(rise_end)  # the start the client hands the planner
    assert home_start[2] == pytest.approx(rise_end[2] - 2.0 * math.pi)  # a whole turn as written
    assert ang_diff(home_start[2], rise_end[2]) == pytest.approx(0.0, abs=1e-12)  # to the arm, one angle
    rise = leg("rise", start=[-1.0, -0.2, 0.40, -1.0, -0.3, -1.1, 1.8], end=rise_end)
    rise.plan_free = True
    home = leg("home", start=home_start, end=HOME)
    assert not can_merge(rise, home)
    assert [len(g) for g in merge_groups([rise, home])] == [1, 1]


@pytest.mark.parametrize("gap, groups", [(1e-7, [2]), (1e-4, [2]), (-1e-4, [2]), (2 * MERGE_GAP_RAD, [1, 1])])
def test_a_leg_planned_from_the_one_before_it_still_meets_it(gap, groups):
    """The planner's float32 output starts ~1e-7 off the start it was
    handed: a gap well inside MERGE_GAP_RAD merges, either way round; twice
    it does not."""
    a, b = leg("a"), leg("b", start=0.1 + gap, end=0.2)
    assert [len(g) for g in merge_groups([a, b])] == groups


@pytest.mark.parametrize(
    "gains_a, gains_b, groups",
    [
        (None, GAINS_STIFF, [1, 1]),
        (GAINS_STIFF, None, [1, 1]),
        (GAINS_MEDIUM, GAINS_STIFF, [1, 1]),
        (GAINS_STIFF, GAINS_STIFF, [2]),
        (None, None, [2]),
    ],
)
def test_one_execution_flies_one_gains_spec(gains_a, gains_b, groups):
    """A goal carries one ImpedanceGains spec and the runner sends its
    group's (runner._run_motion): legs under different gains fly as goals
    of their own. None, the client's run default, is not a profile a leg
    names, STIFF included."""
    a, b = leg("a"), leg("b", start=0.1, end=0.2)
    a.gains, b.gains = gains_a, gains_b
    assert [len(g) for g in merge_groups([a, b])] == groups
