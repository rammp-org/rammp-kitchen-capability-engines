"""home_arm from a start too near a joint limit for the planner.

rammp-curobo refuses a start within joint_limit_margin_rad (0.10) of a URDF
limit, and cannot retime some starts outside it while the wrist is folded,
so home_arm was refused in both worlds from where a run left the arm
(2026-10-23: arm_joint_4 0.047 rad from -2.57, the tool 268 mm above
arm_mount). It now steps plan-free along the joint line toward HOME first,
the shortest step HOME plans from, and plans HOME from where it ends.

rclpy is never initialised: the node and the client are stubs."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeClient

from box_opening.constants import HOME
from box_opening.primitives import core
from box_opening.primitives.core import ClearMarginFree, PlanState
from box_opening.runtime import approach
from box_opening.runtime.kinematics import ArmChain
from box_opening.tasks import cli_common, home_arm

PKG = Path(__file__).resolve().parents[1]
FLATBOT_URDF = str(PKG / "test" / "data" / "arm_flatbot_20261021.urdf")
FLATBOT_WORLD = str(PKG / "config" / "world_flatbot.yaml")
CONTAINER = str(PKG / "config" / "containers" / "ankou_pink.yaml")

# the live joints home_arm was refused from, 2026-10-23 11:19
STUCK = [0.0617, -2.0574, -3.1385, -2.5233, -0.2102, -1.2174, 1.7257]
LINE = np.array(HOME) - np.array(STUCK)


def _along(q):
    """How far along the STUCK -> HOME joint line q is (0..1)."""
    return float(np.dot(np.array(q) - STUCK, LINE) / np.dot(LINE, LINE))


class _Bad:
    success = False
    message = "MotionGenStatus.FINETUNE_TRAJOPT_FAIL"


class LivePlanner(FakeClient):
    """Plans HOME only from 45% or more of the way along the line, as the
    live planner did (2026-10-23: refused from 0.29-0.42, planned from 0.45)."""

    def plan_to_joints(self, q7, start_joints):
        ok = _along(start_joints) >= 0.45
        res = super().plan_to_joints(q7, start_joints)
        return res if ok else _Bad()


class NeverPlans(FakeClient):
    def plan_to_joints(self, q7, start_joints):
        super().plan_to_joints(q7, start_joints)
        return _Bad()


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY)
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY = saved


def _flatbot_ctx(monkeypatch, live, urdf=FLATBOT_URDF, client=None):
    client = client or FakeClient()
    client.live = list(live)
    monkeypatch.setattr(cli_common, "init_runtime", lambda execute: (object(), client))
    monkeypatch.setattr(cli_common, "_arm_urdf_path", lambda node, frame: urdf)
    ctx, _runner = cli_common.build_ctx(Namespace(container=CONTAINER, bench_world=FLATBOT_WORLD, execute=False))
    return ctx


def test_home_arm_steps_toward_home_until_the_planner_can_plan_it(monkeypatch, capsys):
    ctx = _flatbot_ctx(monkeypatch, STUCK, client=LivePlanner())
    legs = home_arm.build_legs(ctx)
    assert [x.name for x in legs] == ["clear-margin", "home"]
    step = legs[0]
    assert step.guard is not None and step.guard.trip == "obstruction"
    assert step.speed == core.CONTACT_SPEED
    pts = np.array([p.positions for p in step.traj.points])
    assert np.allclose(pts[0], STUCK)
    for q in pts:  # on the straight joint line to HOME
        assert np.allclose(q, np.array(STUCK) + _along(q) * LINE, atol=1e-9)
    # the shortest of the steps it tried that HOME plans from: 0.288 clears
    # the margin, 0.382 is 0.05 rad more, 0.475 is the first at 0.45 or more
    tried = sorted({round(_along(q), 3) for q in ctx.client.joint_starts})
    assert tried == pytest.approx([0.288, 0.382, 0.475], abs=1e-3)
    assert _along(pts[-1]) == pytest.approx(0.475, abs=1e-3)
    assert np.max(np.abs(pts[-1] - STUCK)) == pytest.approx(0.254, abs=1e-3)
    # the tool only rises: +119 mm
    fb = ArmChain(root="arm_mount", path=FLATBOT_URDF)
    z = [fb.fk(q)[1][2] for q in pts]
    assert min(z) == pytest.approx(z[0]) and z[-1] - z[0] == pytest.approx(0.119, abs=0.002)
    # HOME is planned from the step's END, once per step in the bare world:
    # world_flatbot.yaml has no table, so no band to guard
    assert ctx.client.joint_starts[-1] == pytest.approx(list(step.goal_joints))
    assert legs[1].world == "bench_bare" and len(ctx.client.joint_starts) == 3
    out = capsys.readouterr().out
    assert "arm_joint_4" not in out and "joint_4 is 0.047 rad from its limit" in out
    assert "a plan-free step 48% of the way to HOME first (the largest joint 0.254 rad, the tool +119 mm up" in out
    # the no-table caution once, up front; the refused tries say nothing
    assert out.count("CAUTION") == 1 and "CAUTION: no table is known" in out


# 2026-10-30: home_arm refused with "no plan-free step toward HOME of up to
# 0.35 rad per joint lets the planner plan HOME (last: None)": arm_joint_4 at
# its limit, and joint_3 at +3.097 — HOME's -3.137 the other side of the wrap
WRAPPED = [0.0617, -2.0919, 3.0970, -2.5701, -0.2357, -0.9522, 1.5790]


def test_a_continuous_joint_the_other_side_of_the_wrap_steps_the_short_way_round(monkeypatch, capsys):
    """joint_3 at +3.097 is 0.049 rad from HOME's -3.137 the short way and
    6.23 rad the long way. The joint line taken literally turned it the long
    way, so the first step that cleared joint_4's limit turned joint_3 2.49
    rad and none up to the cap was offered. Its goal is now the short way
    round (branches.on_branch): the first step, 40% of the way, clears the
    margin, its largest joint 0.12 rad, joint_3 0.02 of it, the tool only
    rising; HOME is planned from its end."""
    ctx = _flatbot_ctx(monkeypatch, WRAPPED)
    legs = home_arm.build_legs(ctx)
    assert [x.name for x in legs] == ["clear-margin", "home"]
    pts = np.array([p.positions for p in legs[0].traj.points])
    assert np.allclose(pts[0], WRAPPED)
    assert np.max(np.abs(pts[-1] - WRAPPED)) == pytest.approx(0.120, abs=2e-3)
    assert 0.0 < pts[-1][2] - WRAPPED[2] < 0.03  # joint_3 toward +pi, the short way
    assert min(pts[-1][3] - (-2.57), 2.57 - pts[-1][3]) == pytest.approx(core.CLEAR_MARGIN_FIRST_RAD, abs=1e-3)
    fb = ArmChain(root="arm_mount", path=FLATBOT_URDF)
    z = [fb.fk(q)[1][2] for q in pts]
    assert min(z) == pytest.approx(z[0]) and z[-1] > z[0]
    assert ctx.client.joint_starts == [pytest.approx(list(legs[0].goal_joints))]
    assert "a plan-free step 40% of the way to HOME first (the largest joint 0.120 rad" in capsys.readouterr().out


def test_a_joint_just_outside_the_margin_plans_from_the_start_first(monkeypatch):
    start = list(STUCK)
    start[3] = -2.42  # 0.15 rad of room: outside the margin, inside the trajopt's band
    ctx = _flatbot_ctx(monkeypatch, start)
    assert [x.name for x in home_arm.build_legs(ctx)] == ["home"]
    assert ctx.client.joint_starts == [pytest.approx(start)]


def test_a_start_clear_of_the_limits_is_planned_as_before(monkeypatch):
    ctx = _flatbot_ctx(monkeypatch, HOME)
    assert [x.name for x in home_arm.build_legs(ctx)] == ["home"]
    assert ctx.client.joint_starts == [pytest.approx(HOME)]


def test_without_the_arm_description_the_planner_has_the_last_word(monkeypatch):
    """No URDF, no limits to read: no step, and HOME is asked of the planner
    from the live joints as before (its own refusal names the joint)."""
    ctx = _flatbot_ctx(monkeypatch, STUCK, urdf=None)
    assert [x.name for x in home_arm.build_legs(ctx)] == ["home"]
    assert ctx.client.joint_starts == [pytest.approx(STUCK)]


def test_no_step_up_to_the_cap_that_plans_is_refused(monkeypatch):
    ctx = _flatbot_ctx(monkeypatch, STUCK, client=NeverPlans())
    with pytest.raises(RuntimeError, match="no plan-free step toward HOME of up to 0.35 rad per joint"):
        home_arm.build_legs(ctx)
    largest = max(np.max(np.abs(np.array(q) - STUCK)) for q in ctx.client.joint_starts)
    assert largest <= core.CLEAR_MARGIN_MAX_STEP_RAD + 1e-9


def test_a_line_that_would_lower_the_tool_is_refused(monkeypatch):
    start = list(HOME)
    start[3] = -2.50  # 0.07 rad from its limit
    goal = [0.0617, -1.9, -3.1385, -2.30, -0.2102, -1.4, 1.7257]
    fb = ArmChain(root="arm_mount", path=FLATBOT_URDF)
    q0, q1 = np.array(start), np.array(start) + 0.25 * (np.array(goal) - start)
    z = [fb.fk(q0 + u * (q1 - q0))[1][2] for u in np.linspace(0, 1, 26)]
    assert z[0] - min(z) > 0.02  # the line toward this goal goes down
    ctx = _flatbot_ctx(monkeypatch, start)
    with pytest.raises(RuntimeError, match="drops the tool 21 mm"):
        next(ClearMarginFree(goal).steps(ctx, PlanState(joints=start, chain=0)))


def test_a_goal_too_near_a_limit_is_refused(monkeypatch):
    goal = list(HOME)
    goal[5] = 2.05
    ctx = _flatbot_ctx(monkeypatch, STUCK)
    with pytest.raises(RuntimeError, match="the goal's joint_6 is 0.040 rad from its limit"):
        next(ClearMarginFree(goal).steps(ctx, PlanState(joints=STUCK, chain=0)))


def test_a_chain_rooted_off_the_planning_frame_is_refused(monkeypatch):
    """The drop is the chain root's z: on flatbot base_link leans 30 deg."""
    ctx = _flatbot_ctx(monkeypatch, STUCK)
    approach.set_chain_root("base_link", None)
    with pytest.raises(RuntimeError, match="chain root base_link != planning frame arm_mount"):
        next(ClearMarginFree(HOME).steps(ctx, PlanState(joints=STUCK, chain=0)))


# The field trip, 2026-10-23 12:23 (run-20261023-122340): the wrist joints'
# efforts from rest through the first 9% of the step, where the old guard
# struck at 4.28 Nm on arm_joint_4. (progress, efforts of arm_joint_4..7)
TRIP_RUN_UP = [
    (0.0003, [4.611, -0.397, 0.611, -0.609]),
    (0.0755, [6.044, -0.363, 1.436, -0.533]),
    (0.0811, [6.702, -0.357, 1.851, -0.515]),
    (0.0862, [7.876, -0.429, 2.460, -0.445]),
    (0.0918, [8.889, -0.434, 3.204, -0.379]),
]


def _step_replay(monkeypatch, push_from=None, push_nm=0.0):
    """The step's guard fed the field run-up, then the arm carrying its own
    weight (the gravity model's, plus the run-up's last offsets from it) to
    the end — and from `push_from` on, `push_nm` more on arm_joint_4."""
    from box_opening.runtime.guards import CompensatedGuard, TorqueGuard
    from box_opening.runtime.stamps import secs

    ctx = _flatbot_ctx(monkeypatch, STUCK, client=LivePlanner())
    step = home_arm.build_legs(ctx)[0]
    spec = step.guard
    assert spec.gravity is not None
    assert spec.arm_after == spec.rebaseline_after == core.CLEAR_MARGIN_SETTLE
    pts = step.traj.points
    total = secs(pts[-1].time_from_start)
    ts = np.array([secs(p.time_from_start) / total for p in pts])
    qs = np.array([p.positions for p in pts])

    def joints(f):
        return [float(np.interp(f, ts, qs[:, j])) for j in range(7)]

    hold = approach._gravity().wrist_holding
    offset = np.array(TRIP_RUN_UP[-1][1]) - hold(joints(TRIP_RUN_UP[-1][0]))
    feed = list(TRIP_RUN_UP)
    for f in np.linspace(0.1, 1.0, 91):
        e = np.array(hold(joints(f))) + offset
        if push_from is not None and f >= push_from:
            e[0] += push_nm
        feed.append((float(f), [float(v) for v in e]))
    guards = {
        "step": CompensatedGuard(
            spec.touch_nm, spec.gravity, rebaseline_after=spec.rebaseline_after, arm_after=spec.arm_after
        ),
        "from rest": TorqueGuard(spec.touch_nm),  # the guard the field run had
    }
    trips = {}
    for name, g in guards.items():
        for f, e in feed:
            g.on_progress(f)
            if g.on_efforts(e, joints(f)) if name == "step" else g.on_efforts(e):
                trips[name] = f
                break
    return trips


def test_the_weight_coming_back_onto_the_joints_is_not_a_strike(monkeypatch):
    trips = _step_replay(monkeypatch)
    assert trips == {"from rest": pytest.approx(0.0918)}  # the field's false trip, and only it


def test_a_push_once_the_step_is_under_way_still_trips(monkeypatch):
    trips = _step_replay(monkeypatch, push_from=0.5, push_nm=5.0)
    assert trips["step"] == pytest.approx(0.5, abs=0.011)
