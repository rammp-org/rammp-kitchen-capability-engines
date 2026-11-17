"""Every home flies slow, in the driver's MEDIUM joint impedance
(constants.HOME_SPEED, HOME_GAINS; core.as_home): owner 2026-11-02, after a
full-speed home in POSITION control "almost hit the flat bot". A leg in a
control mode of its own (Leg.compliant) flies in it whatever the client's
(runner._mode), the arm waited still first where the mode changes: the
driver refuses a mode change while the arm moves."""

from types import SimpleNamespace

import pytest
from conftest import Q0, Q1, FakeClient
from conftest import leg as make_leg
from conftest import runner as make_runner

from box_opening.constants import COMPLIANT_QD_CEILING, GAINS_MEDIUM, HOME, HOME_GAINS, HOME_SPEED, TRANSIT_SPEED
from box_opening.primitives.core import PlanState, as_home
from box_opening.runtime import runner as runner_mod
from box_opening.runtime.legs import can_merge


def _home(start=Q1, end=Q0):
    return as_home(make_leg("home", start=start, end=end, speed=TRANSIT_SPEED))


def _watched(client):
    """The client's mode at each execution, and the re-timing's joint ceiling."""
    modes, real = [], client.execute

    def execute(*a, **k):
        modes.append(bool(getattr(client, "compliant", False)))
        return real(*a, **k)

    client.execute = execute
    return modes


def test_a_home_is_slow_under_medium_in_impedance_of_its_own():
    home = _home()
    assert (home.speed, home.compliant, home.gains) == (HOME_SPEED, True, GAINS_MEDIUM)
    assert HOME_GAINS == GAINS_MEDIUM and HOME_SPEED < TRANSIT_SPEED


def test_a_home_after_a_position_leg_flies_in_impedance_under_medium_the_arm_waited_still_first(tmp_path, monkeypatch):
    """home_arm's --lift-first, a guarded plan-free line in POSITION, then
    home: two goals (one mode each), the second in impedance under MEDIUM,
    re-timed (and previewed) under the impedance's wrist ceiling; the
    client's own mode put back after."""
    client = FakeClient()
    client.compliant = False
    modes = _watched(client)
    ceilings = []
    real_group = runner_mod.retime_group

    def retime_group(trajs, speeds, vmax, params, folds=None, step_caps=None):
        ceilings.append(params.qd_ceiling)
        return real_group(trajs, speeds, vmax, params, folds, step_caps)

    monkeypatch.setattr(runner_mod, "retime_group", retime_group)
    lift, home = make_leg("lift", start=Q0, end=Q1), _home()
    assert not can_merge(lift, home)  # one goal flies one mode
    runner = make_runner(client, tmp_path)
    res = runner.run([lift, home], execute=True)
    assert [r.ok for r in res] == [True, True]
    assert modes == [False, True] and client.gains_sent == [None, GAINS_MEDIUM]
    assert ceilings == [None, COMPLIANT_QD_CEILING] * 2  # the preview's timing is what flies
    assert client.stills == [1]  # once the lift had gone out, before home
    assert client.compliant is False


def test_a_home_in_a_compliant_run_waits_for_nothing(tmp_path):
    """The run's way home after the retreat: the same mode, so no wait; its
    own goal, under MEDIUM where the retreat flies the run's gains."""
    client = FakeClient()
    client.compliant = True
    modes = _watched(client)
    res = make_runner(client, tmp_path).run([make_leg("place:retreat", start=Q0, end=Q1), _home()], execute=True)
    assert [r.ok for r in res] == [True, True]
    assert modes == [True, True] and client.gains_sent == [None, GAINS_MEDIUM] and client.stills == []
    assert client.compliant is True


def test_a_position_leg_after_a_home_waits_the_arm_still(tmp_path):
    client = FakeClient()
    client.compliant = False
    modes = _watched(client)
    res = make_runner(client, tmp_path).run([_home(Q0, Q1), make_leg("look", start=Q1, end=Q0)], execute=True)
    assert [r.ok for r in res] == [True, True]
    assert modes == [True, False] and client.stills == [1]


def test_home_arm_plans_its_home_slow_under_medium(ctx):
    from box_opening.tasks import home_arm

    for table in (ctx.worlds.table_top_z, None):  # the guarded bench world, and the bare one
        ctx.worlds.table_top_z = table
        leg = home_arm._plan_home(ctx, PlanState(joints=list(Q1), chain=0))
        assert leg.goal_joints == pytest.approx(HOME)
        assert (leg.speed, leg.compliant, leg.gains) == (HOME_SPEED, True, GAINS_MEDIUM)


def test_every_home_a_run_plans_is_slow_under_medium(ctx, monkeypatch):
    """The no-box and recovery exit (press_demo.build_home_leg, record_
    detection_set's too) and the run's own homes (reach_press._home_legs:
    the way out, a stop's, the one after the set-down)."""
    from box_opening.primitives.core import ClearMarginFree
    from box_opening.tasks import press_demo, reach_press

    monkeypatch.setattr(ClearMarginFree, "near", lambda self, ctx_, q: [])  # no joint near its limit
    exit_home = press_demo.build_home_leg(ctx, list(Q1))
    (run_home,) = reach_press._home_legs(ctx, SimpleNamespace(park_tool_down=False), start=list(Q1))
    for leg in (exit_home, run_home):
        assert leg.name == "home" and leg.goal_joints == pytest.approx(HOME)
        assert (leg.speed, leg.compliant, leg.gains) == (HOME_SPEED, True, GAINS_MEDIUM)


def test_home_is_written_as_the_arm_reports_it_and_well_inside_the_joint_limits():
    """The temporary HOME (2026-11-09): its continuous joints in (-pi, pi], where the arm reports them, so no
    branch is needed to compare a live reading with it; its revolute joints 0.3 rad or more inside flatbot's
    URDF limits (joint_2 2.24, joint_4 2.57, joint_6 2.09)."""
    import math

    from box_opening.constants import HOME

    assert len(HOME) == 7
    for i in (0, 2, 4, 6):
        assert -math.pi < HOME[i] <= math.pi
    for i, limit in ((1, 2.24), (3, 2.57), (5, 2.09)):
        assert abs(HOME[i]) <= limit - 0.3
