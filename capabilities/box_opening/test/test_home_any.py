"""HOME from wherever the arm stands (primitives.home_any, EscapeBodyFree; 2026-11-09, the temporary HOME with
the arm 8 in forward on flatbot's chair): a start inside the chair and user's padding steps out of it first,
never deeper and the tool held; HOME is asked for again; a waypoint along the joint line is tried; and where
nothing plans, the refusal says so."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeClient, flatbot_urdf

from box_opening.constants import HOME
from box_opening.primitives import core
from box_opening.runtime import approach, body
from box_opening.runtime.kinematics import mat_to_quat_xyzw
from box_opening.tasks import cli_common

PKG = Path(__file__).resolve().parents[1]
FLATBOT_WORLD = str(PKG / "config" / "world_flatbot.yaml")
CONTAINER = str(PKG / "config" / "containers" / "ankou_pink.yaml")
DOWN = mat_to_quat_xyzw(np.diag([1.0, -1.0, -1.0]))
START = [0.3, -0.9, -2.6, -1.8, -0.4, -0.7, 1.9]
ROBOT = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_forearm_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.06}
      arm_bracelet_link:
      - {center: [0.0, 0.0, -0.045], radius: 0.045}
      arm_robotiq_85_base_link:
      - {center: [0.0, 0.0, 0.04], radius: 0.05}
      - {center: [0.0, 0.0, 0.10], radius: 0.035}
"""


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY)
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY = saved


def _no_chain_ctx(monkeypatch):
    """A ctx with no arm chain: no body model, no joint limits to step off (the planner's word alone)."""
    monkeypatch.setattr(approach, "chain_ready", lambda: (False, "no URDF in this test"))
    client = SimpleNamespace(joints=lambda: list(START), joint_names=["joint_%d" % i for i in range(1, 8)])
    return SimpleNamespace(client=client)


class Planner:
    """plan_to's stand-in: refuses HOME `home_fails` times from the start, and from anywhere when `from_start_only`
    is False; plans every waypoint. Records (name, start joints, goal) per ask."""

    def __init__(self, home_fails=0, from_start_only=True, waypoints_fail=False):
        self.home_fails, self.from_start_only, self.waypoints_fail = home_fails, from_start_only, waypoints_fail
        self.asked = []

    def __call__(self, at, name, q):
        self.asked.append((name, list(at.joints), list(q)))
        if name == "home":
            from_start = np.allclose(at.joints, START)
            if from_start and self.home_fails > 0:
                self.home_fails -= 1
                raise RuntimeError("planning failed for home: MotionGenStatus.FINETUNE_TRAJOPT_FAIL")
            if not from_start and not self.from_start_only:
                raise RuntimeError("planning failed for home: MotionGenStatus.FINETUNE_TRAJOPT_FAIL")
        elif self.waypoints_fail:
            raise RuntimeError("planning failed for %s" % name)
        return SimpleNamespace(name=name, goal_joints=list(q))


def test_home_is_asked_for_again_before_anything_else(monkeypatch):
    ctx = _no_chain_ctx(monkeypatch)
    plan = Planner(home_fails=core.HOME_TRIES - 1)
    legs = core.home_any(ctx, core.PlanState(joints=list(START), chain=0), HOME, plan, say=lambda s: None)
    assert [leg.name for leg in legs] == ["home"]
    assert [a[0] for a in plan.asked] == ["home"] * core.HOME_TRIES


def test_where_home_never_plans_from_the_start_it_goes_through_a_waypoint_half_way_along_the_joint_line(monkeypatch):
    ctx = _no_chain_ctx(monkeypatch)
    plan = Planner(home_fails=99)
    said = []
    legs = core.home_any(ctx, core.PlanState(joints=list(START), chain=0), HOME, plan, say=said.append)
    assert [leg.name for leg in legs] == ["home-waypoint", "home"]
    waypoint = legs[0].goal_joints
    # half way, each continuous joint the short way round (joint_3: -2.6 to HOME's -2.9033, not a turn)
    assert waypoint == pytest.approx([(a + b) / 2 for a, b in zip(START, HOME, strict=True)], abs=1e-9)
    assert plan.asked[-1] == ("home", waypoint, list(HOME))  # HOME planned from the waypoint
    assert any("through a waypoint 50%" in s for s in said)


def test_where_nothing_plans_the_refusal_names_what_was_tried(monkeypatch):
    ctx = _no_chain_ctx(monkeypatch)
    plan = Planner(home_fails=99, from_start_only=False)
    with pytest.raises(
        RuntimeError,
        match=r"did not plan from here, 3 times directly nor through a waypoint "
        r"50%/25%/75% of the way .*FINETUNE_TRAJOPT_FAIL",
    ):
        core.home_any(ctx, core.PlanState(joints=list(START), chain=0), HOME, plan, say=lambda s: None)


def _traj(start, goal, swing_j1=0.0):
    """A JointTrajectory stand-in from `start` to `goal`, joint_1 taken `swing_j1` rad out past its start midway."""
    mid = [(a + b) / 2 for a, b in zip(start, goal, strict=True)]
    mid[0] = start[0] + swing_j1
    pts = [SimpleNamespace(positions=list(q)) for q in (start, mid, goal)]
    return SimpleNamespace(joint_names=["arm_joint_%d" % i for i in range(1, 8)], points=pts)


class SwingingPlanner(Planner):
    """Plans every leg, but HOME from `swing_from` swings joint_1 0.9 rad out and back (flatbot 2026-11-10)."""

    def __init__(self, swing_from):
        super().__init__()
        self.swing_from = list(swing_from)

    def __call__(self, at, name, q):
        leg = super().__call__(at, name, q)
        swing = 0.9 if name == "home" and np.allclose(at.joints, self.swing_from) else 0.0
        leg.traj = _traj(list(at.joints), list(q), swing)
        return leg


def test_a_plan_home_that_swings_wide_is_refused_at_planning_and_home_goes_through_the_waypoint(monkeypatch):
    ctx = _no_chain_ctx(monkeypatch)
    plan = SwingingPlanner(START)
    said = []
    legs = core.home_any(ctx, core.PlanState(joints=list(START), chain=0), HOME, plan, say=said.append)
    assert [leg.name for leg in legs] == ["home-waypoint", "home"]
    assert [a[0] for a in plan.asked] == ["home", "home-waypoint", "home"]  # asked directly once: it answers alike
    assert any("through a waypoint 50%" in s and "swings wide" in s for s in said)


def test_where_every_plan_swings_wide_the_refusal_says_so(monkeypatch):
    ctx = _no_chain_ctx(monkeypatch)

    class Always(Planner):
        def __call__(self, at, name, q):
            leg = super().__call__(at, name, q)
            leg.traj = _traj(list(at.joints), list(q), 0.9)
            return leg

    with pytest.raises(RuntimeError, match=r"1 times directly .*swings wide .*arm_joint_1 excursion"):
        core.home_any(ctx, core.PlanState(joints=list(START), chain=0), HOME, Always(), say=lambda s: None)


def _body_ctx(monkeypatch, tmp_path, boxes):
    """flatbot's ctx (bench world, FakeClient) with a body model of `boxes` attached to its chain."""
    client = FakeClient()
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), client))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: flatbot_urdf())
    ctx, _runner = cli_common.build_ctx(Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False))
    lines = ["layer: test", "obstacles:"]
    lines += ["  - {name: %s, position: %s, dims: %s}" % (n, list(c), list(d)) for n, c, d in boxes]
    (tmp_path / "layer_test.yaml").write_text("\n".join(lines) + "\n")
    (tmp_path / "robot.yaml").write_text(ROBOT)
    (tmp_path / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_test.yaml]\n")
    approach.set_body(body.load(str(tmp_path / "planner.yaml")))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    return ctx, approach._chain(), client


def test_a_start_inside_the_padding_steps_out_first_never_deeper_the_tool_held_and_home_plans_from_its_end(
    monkeypatch, tmp_path
):
    # a slab whose padding the hand stands in, 0.45 over a point 0.6 out in front of the arm
    ctx, chain, client = _body_ctx(monkeypatch, tmp_path, [("slab", (0.6, 0.0, 0.40), (0.30, 0.30, 0.06))])
    q = chain.nearest_solution([0.0, 0.3, 0.0, 1.8, 0.0, 1.0, 1.57], [0.6, 0.0, 0.45], DOWN)
    assert q is not None
    client.live = list(q)
    start_depth = float(chain.body.sphere_depths([q])[0].max())
    assert start_depth > 0.0  # inside the padding: the planner would refuse this start
    plan = Planner()
    said = []
    legs = core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, plan, say=said.append)
    assert [leg.name for leg in legs] == ["escape-body", "home"]
    pts = np.array([p.positions for p in legs[0].traj.points])
    depth = chain.body.sphere_depths(pts)[0].max(axis=1)
    assert np.all(np.diff(depth) <= 1e-6)  # never deeper on the way out
    assert depth[-1] <= -core.ESCAPE_CLEAR_M  # out by the margin
    z = np.array([chain.fk(p)[1][2] for p in pts])
    assert z.min() >= z[0] - core.ESCAPE_DROP_M - 1e-9  # the tool held
    # slowly, in the driver's lightest joint impedance, unguarded (a compliant leg's guard only watches)
    assert legs[0].compliant is True and legs[0].gains == core.GAINS_SOFT and legs[0].guard is None
    assert legs[0].speed == core.ESCAPE_SPEED
    assert plan.asked == [("home", list(pts[-1]), list(HOME))]  # HOME planned from where it ends
    assert any("inside the chair and user's padding" in s for s in said)


def test_a_start_clear_of_the_padding_takes_no_step_out(monkeypatch, tmp_path):
    ctx, chain, client = _body_ctx(monkeypatch, tmp_path, [("far", (-0.8, 0.8, 0.2), (0.1, 0.1, 0.1))])
    q = chain.nearest_solution([0.0, 0.3, 0.0, 1.8, 0.0, 1.0, 1.57], [0.6, 0.0, 0.45], DOWN)
    client.live = list(q)
    assert not core.EscapeBodyFree().needed(q)
    legs = core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, Planner(), say=lambda s: None)
    assert [leg.name for leg in legs] == ["home"]


def test_a_start_just_outside_the_padding_is_close_enough_to_step_clear_first(monkeypatch, tmp_path):
    ctx, chain, client = _body_ctx(monkeypatch, tmp_path, [("slab", (0.6, 0.0, 0.40), (0.30, 0.30, 0.06))])
    q = chain.nearest_solution([0.0, 0.3, 0.0, 1.8, 0.0, 1.0, 1.57], [0.6, 0.0, 0.45], DOWN)
    esc = core.EscapeBodyFree()
    legs, state, _info = esc.plan(ctx, core.PlanState(joints=list(q), chain=0))
    out = np.array(legs[0].goal_joints)  # 10 mm clear: now walk back toward the slab until 5 mm clear
    d_out = float(chain.body.sphere_depths([out])[0].max())
    assert d_out <= -core.ESCAPE_CLEAR_M + 1e-6
    near = None
    for u in np.linspace(0.0, 1.0, 101):
        x = out + u * (np.array(q) - out)
        d = float(chain.body.sphere_depths([x])[0].max())
        if -core.ESCAPE_NEAR_M < d < 0.0:
            near = x
            break
    assert near is not None
    assert "within" in esc.needed(near)  # outside the padding, but close: it steps clear first
    assert not esc.needed(out)


def test_a_tool_low_over_a_known_table_is_lifted_clear_of_it(monkeypatch, tmp_path):
    ctx, chain, client = _body_ctx(monkeypatch, tmp_path, [("far", (-0.8, 0.8, 0.2), (0.1, 0.1, 0.1))])
    q = chain.nearest_solution([0.0, 0.3, 0.0, 1.8, 0.0, 1.0, 1.57], [0.6, 0.0, 0.30], DOWN)
    table = 0.28  # the tool 20 mm over it
    esc = core.EscapeBodyFree()
    assert "20 mm over the table" in esc.needed(q, table)
    legs, _state, info = esc.plan(ctx, core.PlanState(joints=list(q), chain=0), table)
    end = legs[0].goal_joints
    assert chain.fk(end)[1][2] >= table + core.ESCAPE_TABLE_M - 1e-6
    assert info["tool_drop_mm"] <= 1000 * core.ESCAPE_DROP_M + 1e-6
    assert legs[0].gains == core.GAINS_SOFT


def test_off_a_limit_where_the_line_home_would_drop_the_tool_it_steps_clear_softly_then_goes_home(
    monkeypatch, tmp_path
):
    # flatbot 2026-11-10 14:57: the fingers 2 cm over the table, joint_4 0.107 rad from its limit, joint_1 folded
    # 2.6 rad from HOME's: the clear-margin line toward HOME dropped the tool 46 mm, and home_arm refused
    ctx, chain, client = _body_ctx(monkeypatch, tmp_path, [("far", (-0.8, 0.8, 0.2), (0.1, 0.1, 0.1))])
    q = [3.0500, 0.0205, 0.4103, -2.4632, 0.1825, 1.3171, 2.9466]
    client.live = list(q)
    plan = Planner()
    said = []
    legs = core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, plan, say=said.append)
    assert [leg.name for leg in legs] == ["escape-body", "home"]
    step = legs[0]
    assert step.compliant is True and step.gains == core.GAINS_SOFT
    end = np.array(step.goal_joints)
    lo, hi = chain.limits[3]
    assert min(end[3] - lo, hi - end[3]) >= core.ESCAPE_ROOM_RAD - 1e-6  # off joint_4's limit
    pts = np.array([p.positions for p in step.traj.points])
    z = np.array([chain.fk(p)[1][2] for p in pts])
    assert z.min() >= z[0] - core.ESCAPE_DROP_M - 1e-9  # the tool never lower
    assert plan.asked == [("home", list(end), list(HOME))]  # HOME planned from where it ends
    assert any("drops the tool" in s for s in said) and any("lightest joint impedance" in s for s in said)


# --- a table the run knows: the planner's world holds none, so every planned leg is checked against it ---


class DippingPlanner(Planner):
    """Plans every leg, but HOME from `dip_from` goes through `low` on the way (the tool under the table)."""

    def __init__(self, dip_from, low):
        super().__init__()
        self.dip_from, self.low = list(dip_from), list(low)

    def __call__(self, at, name, q):
        leg = super().__call__(at, name, q)
        via = [self.low] if name == "home" and np.allclose(at.joints, self.dip_from) else []
        pts = [SimpleNamespace(positions=list(p)) for p in [list(at.joints), *via, list(q)]]
        leg.traj = SimpleNamespace(joint_names=["arm_joint_%d" % i for i in range(1, 8)], points=pts)
        return leg


def _over_a_table(monkeypatch, tmp_path):
    """The arm 0.6 out, the tool 0.40 up, over a table at 0.28 the scene camera saw from x 0.45 on; `low`: the
    same hand 30 mm under that table's top."""
    ctx, chain, client = _body_ctx(monkeypatch, tmp_path, [("far", (-0.8, 0.8, 0.2), (0.1, 0.1, 0.1))])
    seed = [0.0, 0.3, 0.0, 1.8, 0.0, 1.0, 1.57]
    q = chain.nearest_solution(seed, [0.6, 0.0, 0.40], DOWN)
    low = chain.nearest_solution(list(q), [0.6, 0.0, 0.25], DOWN)
    assert q is not None and low is not None
    client.live = list(q)
    ctx.worlds.table_top_z = 0.28
    ctx.worlds.table_extent = (0.45, 0.95, -0.40, 0.40)
    return ctx, chain, list(q), list(low)


def test_a_path_that_dips_under_the_table_is_named_and_one_that_stays_over_it_is_not(monkeypatch, tmp_path):
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)
    up = chain.nearest_solution(list(q), [0.6, 0.0, 0.48], DOWN)
    assert up is not None
    why = core.under_table(chain, chain.body, [q, low, up], 0.28, ctx.worlds.table_extent)
    assert why is not None and "under the table's top" in why
    assert core.under_table(chain, chain.body, [q, up], 0.28, ctx.worlds.table_extent) is None
    # with no extent seen, a dip under both ends is still refused
    assert core.under_table(chain, chain.body, [q, low, up], 0.28, None) is not None
    # a path that starts low may leave: no lower than its start
    assert core.under_table(chain, chain.body, [low, q, up], 0.28, ctx.worlds.table_extent) is None


def test_a_home_planned_under_a_known_table_is_planned_from_the_top_of_a_straight_lift(monkeypatch, tmp_path):
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)
    plan = DippingPlanner(q, low)
    said = []
    legs = core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, plan, say=said.append)
    assert [leg.name for leg in legs] == ["home-lift", "home"]
    lift = legs[0]
    top = chain.fk(lift.goal_joints)[1]
    assert top[2] == pytest.approx(0.40 + core.HOME_LIFTS_M[0], abs=2e-3)  # the lowest lift HOME plans from
    assert lift.guard is not None and lift.guard.trip == "obstruction"
    assert plan.asked[-1] == ("home", list(lift.goal_joints), list(HOME))
    assert any("went under the table" in s for s in said)


def test_where_every_home_goes_under_the_table_the_refusal_says_so(monkeypatch, tmp_path):
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)

    class Always(Planner):
        def __call__(self, at, name, q_goal):
            leg = super().__call__(at, name, q_goal)
            pts = [SimpleNamespace(positions=list(p)) for p in (list(at.joints), low, list(q_goal))]
            leg.traj = SimpleNamespace(joint_names=["arm_joint_%d" % i for i in range(1, 8)], points=pts)
            return leg

    with pytest.raises(RuntimeError, match=r"goes under the table"):
        core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, Always(), say=lambda s: None)


def test_with_no_table_known_a_low_plan_is_not_judged(monkeypatch, tmp_path):
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)
    ctx.worlds.table_top_z = None
    ctx.worlds.table_extent = None
    if ctx.worlds.table_top_z is not None:
        pytest.skip("this bench world carries its own table")
    legs = core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, DippingPlanner(q, low), say=lambda s: None)
    assert [leg.name for leg in legs] == ["home"]


def test_a_waypoint_that_is_itself_under_the_table_is_not_a_way_home(monkeypatch, tmp_path):
    # 2026-11-11, the robot's planner: judged leg by leg, a waypoint under the table is an end of both its legs
    # and passed; six poses homed under the table through one
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)

    class ThroughLow(Planner):
        """HOME planned directly dips under the table; every waypoint leg ends at `low` instead."""

        def __call__(self, at, name, q_goal):
            leg = super().__call__(at, name, q_goal)
            if name == "home-waypoint":
                seq = [list(at.joints), low]
                leg.goal_joints = list(low)
            elif np.allclose(at.joints, low):
                seq = [low, list(q_goal)]
            else:
                seq = [list(at.joints), low, list(q_goal)]
            pts = [SimpleNamespace(positions=list(p)) for p in seq]
            leg.traj = SimpleNamespace(joint_names=["arm_joint_%d" % i for i in range(1, 8)], points=pts)
            return leg

    with pytest.raises(RuntimeError, match=r"home through the waypoint \d+% of the way goes under the table"):
        core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, ThroughLow(), say=lambda s: None)


def test_where_no_straight_lift_plans_home_a_soft_step_higher_over_the_table_is_tried(monkeypatch, tmp_path):
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)

    def no_lift(self, ctx, state):
        raise RuntimeError("plan-free lift refused: a joint limit down the line")

    monkeypatch.setattr(core.LiftFree, "plan", no_lift)
    start = chain.nearest_solution(list(q), [0.6, 0.0, 0.36], DOWN)  # the tool 80 mm over the table
    assert start is not None
    ctx.client.live = list(start)
    plan = DippingPlanner(start, low)
    said = []
    legs = core.home_any(ctx, core.PlanState(joints=list(start), chain=0), HOME, plan, say=said.append)
    assert [leg.name for leg in legs] == ["home-rise", "home"]
    rise = legs[0]
    assert rise.compliant is True and rise.gains == core.GAINS_SOFT
    assert chain.fk(rise.goal_joints)[1][2] >= 0.28 + core.HOME_RISES_M[0] - 1e-6
    assert any("soft step up" in s for s in said)


def test_no_table_stands_where_home_itself_rests_under_its_top(monkeypatch, tmp_path):
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)
    C, r, _ = chain.body.spheres_at(HOME)
    C, r = np.asarray(C, float), np.asarray(r, float)
    i = int(np.argmin(C[:, 2] - r))  # HOME's lowest sphere, on the upper arm beside the base
    top = float(C[i, 2] - r[i]) + 0.03  # a table whose top is 30 mm over that sphere's bottom
    raised = list(HOME)
    raised[1] += 0.5 if chain.body.spheres_at(raised[:1] + [HOME[1] + 0.5] + raised[2:])[0][i][2] > C[i, 2] else -0.5
    Cr, rr, _ = chain.body.spheres_at(raised)
    assert float(np.asarray(Cr, float)[i, 2] - rr[i]) > top  # the same sphere over the top at both ends
    path = [raised, list(HOME), raised]
    assert core.under_table(chain, chain.body, path, top, None) is not None  # judged like any dip...
    assert core.under_table(chain, chain.body, path, top, None, HOME) is None  # ...but not where HOME rests
    # a dip out over the table is still one, HOME named or not
    up = chain.nearest_solution(list(q), [0.6, 0.0, 0.48], DOWN)
    assert core.under_table(chain, chain.body, [q, low, up], 0.28, None, HOME) is not None


def test_the_ways_up_stop_being_tried_once_they_have_taken_too_long_to_plan(monkeypatch, tmp_path):
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)
    monkeypatch.setattr(core, "HOME_RAISE_S", -1.0)

    class Always(Planner):
        def __call__(self, at, name, q_goal):
            leg = super().__call__(at, name, q_goal)
            pts = [SimpleNamespace(positions=list(p)) for p in (list(at.joints), low, list(q_goal))]
            leg.traj = SimpleNamespace(joint_names=["arm_joint_%d" % i for i in range(1, 8)], points=pts)
            return leg

    plan = Always()
    with pytest.raises(RuntimeError, match=r"lift of 80 mm and any after it were not tried"):
        core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, plan, say=lambda s: None)
    assert [a for _n, a, _g in plan.asked if not np.allclose(a, q)] == []  # nothing planned from the top of a way up


# --- the planner handed the run's measured table while HOME is planned (owner 2026-11-12, worlds.py) ---

START_IN_WORLD = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start state collides with the world"


def _shared_folder(monkeypatch, tmp_path):
    """_body_ctx's planner config given a world of its own, and a folder shared with the planner."""
    from box_opening import worlds

    (tmp_path / "planner.yaml").write_text("robot: robot.yaml\nworld: world.yaml\nlayers: [layer_test.yaml]\n")
    (tmp_path / "world.yaml").write_text("base_frame: arm_mount\nobstacles: []\n")
    table = worlds.PlannerTable(tmp_path / "out", "/box_opening", tmp_path / "planner.yaml")
    monkeypatch.setattr(worlds, "_PLANNER_TABLE", [table])
    return table


class Sees(Planner):
    """Plans every leg, noting the world a leg planned now would name."""

    def __init__(self, ctx):
        super().__init__()
        self.ctx, self.worlds = ctx, []

    def __call__(self, at, name, q):
        self.worlds.append(self.ctx.worlds.push_name("bench", tag="bare")[1])
        return super().__call__(at, name, q)


def test_the_planner_holds_the_measured_table_while_home_is_planned_and_its_own_world_after(monkeypatch, tmp_path):
    from box_opening.worlds import PLANNER_WORLD

    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)
    table = _shared_folder(monkeypatch, tmp_path)
    back = []
    ctx.client.planner_own_world = lambda: back.append(ctx.worlds.home_table_held)
    plan, said = Sees(ctx), []
    legs = core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, plan, say=said.append)
    assert [leg.name for leg in legs] == ["home"]
    assert plan.worlds and all(table.ours(w) for w in plan.worlds)  # every leg planned inside names the table world
    assert back == [False]  # the planner's own asked back once the plans are made, the table no longer named
    assert ctx.worlds.push_name("bench", tag="bare")[1] == PLANNER_WORLD
    assert any("the planner holds the measured table" in s for s in said)


def test_the_planners_own_world_is_put_back_where_no_home_plans(monkeypatch, tmp_path):
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)
    _shared_folder(monkeypatch, tmp_path)
    back = []
    ctx.client.planner_own_world = lambda: back.append(1)
    plan = Planner(home_fails=99, from_start_only=False, waypoints_fail=True)
    monkeypatch.setattr(core, "START", list(q), raising=False)
    with pytest.raises(RuntimeError):
        core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, plan, say=lambda s: None)
    assert back == [1] and not ctx.worlds.home_table_held


def test_with_no_folder_shared_home_is_planned_in_the_planners_own_world(monkeypatch, tmp_path):
    from box_opening.worlds import PLANNER_WORLD

    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)
    plan, said = Sees(ctx), []
    core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, plan, say=said.append)
    assert plan.worlds and set(plan.worlds) == {PLANNER_WORLD}
    assert not any("the planner holds" in s for s in said)


def test_a_start_the_planner_refuses_for_its_table_is_lifted_clear_first(monkeypatch, tmp_path):
    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)

    class InTable(Planner):
        def __call__(self, at, name, q_goal):
            if np.allclose(at.joints, q):
                raise RuntimeError("planning failed for %s: %s" % (name, START_IN_WORLD))
            return super().__call__(at, name, q_goal)

    # with no table in the planner's world that refusal is the planner's own word, and nothing is lifted
    with pytest.raises(RuntimeError, match="INVALID_START_STATE"):
        core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, InTable(), say=lambda s: None)
    _shared_folder(monkeypatch, tmp_path)
    said = []
    legs = core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, InTable(), say=said.append)
    assert [leg.name for leg in legs] == ["home-lift", "home"]
    assert any("starts in the table as the planner holds it" in s for s in said)


def test_a_goal_that_would_itself_stand_in_the_planners_table_keeps_the_planner_its_own_world(monkeypatch, tmp_path):
    from box_opening.worlds import PLANNER_WORLD

    ctx, chain, q, low = _over_a_table(monkeypatch, tmp_path)
    _shared_folder(monkeypatch, tmp_path)
    assert core._rests_in(HOME, ctx.worlds.home_table_block()) is None  # HOME rests clear of the table seen
    everywhere = (-2.0, 2.0, -2.0, 2.0, 5.0)  # a block that takes in the whole arm
    assert "under its top" in core._rests_in(HOME, everywhere)
    monkeypatch.setattr(core, "_rests_in", lambda goal, block: "arm_forearm_link 12 mm under its top at x 0.20, y 0.10")
    plan, said = Sees(ctx), []
    core.home_any(ctx, core.PlanState(joints=list(q), chain=0), HOME, plan, say=said.append)
    assert set(plan.worlds) == {PLANNER_WORLD}
    assert any("the planner is not handed the table" in s for s in said)
