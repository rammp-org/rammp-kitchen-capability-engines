"""PlannerClient against the planner's contract: what a plan the planner
calls a success must be before anything flies it, and what its "busy"
means (rammp-curobo 4fa044d, the build flatbot runs).

The planner says success for plans that do not end where they were asked:
PlanToJoints' FK-pose fallback, the same tool through other joints (3 of
72 plans on 2026-10-28), and cuRobo's previous solve handed back as this
one's (rammp-curobo #37). It says success, with a warning, for a pose plan
that winds a continuous joint because its replan the short way failed
(15:56:05). It turns a goal away, unplanned, while it plans another's.

The client is built without a ROS graph. The planner answers with its own
result messages (PlanToJoints.Result, PlanToPose.Result), from a double of
the client's round trip or, for the busy retry, of the action itself."""

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import Q0, Q1, FakeClient, runner
from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening.constants import HOME
from box_opening.runtime import approach
from box_opening.runtime import client as client_mod
from box_opening.runtime.branches import planner_branch
from box_opening.runtime.kinematics import mat_to_quat_xyzw

# What the planner said, word for word: 15:56:05 on 2026-10-28
# (~/.sheppy/logs/planner/20261028-142055-275110.log:300), and its answer
# to a goal sent while it plans another (planner_node.py _finish_plan)
WOUND = (
    "planned 219 points, 4.38 s at full speed — warning: plan ends on a 2*pi branch away from the start (it goes "
    "the long way round) and the replan onto the start's branch failed (MotionGenStatus.DT_EXCEPTION); returning "
    "it as planned"
)
BUSY = "planner busy — one plan at a time"
PLANNED = "planned 69 points, 1.36 s at full speed"
FAILED = "MotionGenStatus.TRAJOPT_FAIL: the goal is reachable but no collision-free PATH was found"

START = planner_branch(HOME)


def _traj(rows):
    t = JointTrajectory()
    t.joint_names = ["arm_joint_%d" % i for i in range(1, 8)]
    for k, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.time_from_start.nanosec = int(2e7 * (k + 1))
        t.points.append(p)
    return t


def _joints_result(rows=(), mismatch=0.0, message=PLANNED, success=True):
    """The planner's PlanToJoints answer: a failed plan's trajectory is empty."""
    return PlanToJoints.Result(
        success=success, message=message, trajectory=_traj(rows), planning_time=0.15, goal_mismatch_rad=float(mismatch)
    )


def _pose_result(rows=(), message=PLANNED, success=True):
    return PlanToPose.Result(success=success, message=message, trajectory=_traj(rows), planning_time=0.15)


def _client(answers):
    """A PlannerClient whose planner answers each goal from `answers(kind,
    goal)` and records every goal it was sent."""
    c = client_mod.PlannerClient.__new__(client_mod.PlannerClient)
    c._plan_pose, c._plan_joints = "pose", "joints"
    c.sent = []

    def call(kind, goal):
        c.sent.append((kind, goal))
        return answers(kind, goal)

    c._call = call
    c.set_world = lambda path: (True, "held")
    return c


@pytest.fixture
def chain(monkeypatch):
    """The arm's kinematics, the bench URDF's own, whatever another test set."""
    from box_opening.runtime.kinematics import ArmChain

    arm = ArmChain()
    monkeypatch.setattr(approach, "_chain", lambda: arm)
    return arm


def _rz(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# -- a joint plan: it ends on the joints asked for ------------------------------


@pytest.mark.parametrize("off, flies", [(0.0, True), (0.004, True), (0.0099, True), (0.0101, False), (0.05, False)])
def test_a_joint_plan_flies_only_when_it_ends_on_the_joints_asked_for(off, flies):
    """PlanToJoints falls back to planning to the goal's FK pose when its
    joint solve fails: the same tool to 5 mm, the elbow wherever its IK put
    it, said only in goal_mismatch_rad. The native solve ends on its goal
    (~5e-7 rad). Within PLAN_END_RAD (0.01) the plan is handed on as it
    came; past it the plan is a failed one, saying how far off it ends and
    what the planner itself said."""
    goal = list(START)
    goal[1] += 0.3
    end = list(goal)
    end[3] -= off  # the elbow
    res = _joints_result([START, end], mismatch=off)
    got = _client(lambda kind, g: res).plan_to_joints(goal, START)
    if flies:
        assert got is res
        return
    assert not got.success and got.trajectory is None  # nothing to fly by mistake
    assert "%.4f rad from its joint goal (joint 4; goal_mismatch_rad %.4f)" % (off, off) in got.message
    assert got.message.endswith("(planner: %s)" % PLANNED)


def test_a_joint_goal_written_a_whole_turn_away_is_the_goal_its_plan_ends_on():
    """The goal goes out on the start's branch (nearest_branch): asked for
    joint_7 a turn and 0.2 rad round from where it stands, the planner is
    sent 0.2 rad and its plan ends there, 2*pi from the joints asked for as
    written, the same angle to the arm. It flies. A bounded joint is never
    wrapped: joint_6 a turn off as written is not its goal."""
    asked = list(START)
    asked[6] += 2 * math.pi + 0.2
    c = _client(lambda kind, g: _joints_result([START, list(g.target_joints)]))
    got = c.plan_to_joints(asked, START)
    assert got.success
    end = list(got.trajectory.points[-1].positions)
    assert end[6] == pytest.approx(asked[6] - 2 * math.pi) and end[6] == pytest.approx(START[6] + 0.2)
    bent = list(START)
    bent[5] += 2 * math.pi
    got = _client(lambda kind, g: _joints_result([START, bent])).plan_to_joints(START, START)
    assert not got.success and "joint 6" in got.message


def test_the_planners_own_mismatch_measured_without_the_wrap_is_reported_not_judged():
    """goal_mismatch_rad is np.abs(final - q_goal).max() (planner.py): a goal
    half a turn from the start, reached the other way round to the same
    angle, reads 6.28 there. The arm ends on its goal: it flies."""
    goal = list(START)
    goal[0] = START[0] + math.pi - 0.002
    end = list(START)
    end[0] = START[0] - math.pi + 0.002  # the same angle, 4 mrad apart the short way
    res = _joints_result([START, end], mismatch=abs(end[0] - goal[0]))
    assert res.goal_mismatch_rad > 6.2
    assert _client(lambda kind, g: res).plan_to_joints(goal, START) is res


def test_a_joint_plan_the_planner_says_goes_the_long_way_round_is_refused():
    """Its fallback's pose plan wound joint_7 and its replan failed: the
    planner hands that back as a success, ending on the goal's angle a turn
    round. The goal it was sent already was the short way, so there is
    nothing to re-plan to: refused, and asked only the once."""
    goal = list(START)
    goal[6] += 0.2
    wound_end = list(goal)
    wound_end[6] -= 2 * math.pi
    c = _client(lambda kind, g: _joints_result([START, wound_end], mismatch=2 * math.pi, message=WOUND))
    got = c.plan_to_joints(goal, START)
    assert not got.success and "the plan goes the long way round" in got.message
    assert [k for k, _g in c.sent] == ["joints"]


def test_a_refused_plan_is_a_failed_plan_to_whoever_plans_a_leg(ctx):
    """core._plan_motion takes it as any failed plan (its callers' stops and
    fallbacks apply): a RuntimeError naming the leg and why — a PlanRefused,
    which home_arm tells apart from the planner's own failures."""
    from box_opening.primitives.core import PlanRefused, PlanState, _plan_motion

    def fallback(kind, g):  # the FK-pose fallback's other arm: the elbow 0.4 rad round
        end = list(g.target_joints)
        end[3] += 0.4
        return _joints_result([START, end], mismatch=0.4)

    ctx.client = _client(fallback)
    at = PlanState(joints=list(START), chain=0)
    with pytest.raises(PlanRefused, match=r"planning failed for home: the plan ends 0\.4000 rad from its joint goal"):
        _plan_motion(ctx, at, "home", ("joints", list(HOME)), ("full", "full.yaml"), 1.0)
    ctx.client = _client(lambda kind, g: _joints_result(success=False, message=FAILED))
    with pytest.raises(RuntimeError, match="TRAJOPT_FAIL") as failed:
        _plan_motion(ctx, at, "home", ("joints", list(HOME)), ("full", "full.yaml"), 1.0)
    assert not isinstance(failed.value, PlanRefused)  # the planner's own failure, as before


def test_a_reach_whose_joint_plan_is_refused_asks_the_planner_for_the_pose(chain):
    """_plan_pose_near hands the planner the arm's own IK answer as a joint
    goal. Refused, it asks for the pose instead, as when that plan fails."""
    q_look = list(START)
    q_look[1] += 0.3
    q_look[3] += 0.2
    R, t = chain.fk(q_look)

    def answers(kind, g):
        if kind == "joints":
            end = list(g.target_joints)
            end[3] += 0.4
            return _joints_result([START, end], mismatch=0.4)
        return _pose_result([START, q_look])

    c = _client(answers)
    got = c.plan_to_pose(list(t), mat_to_quat_xyzw(R), START)
    assert got.success and [k for k, _g in c.sent] == ["joints", "pose"]
    assert list(got.trajectory.points[-1].positions) == pytest.approx(q_look)


def _other_arm(kind, g):
    """The FK-pose fallback's answer to a joint goal: the same tool, the
    elbow 0.4 rad round."""
    end = list(g.target_joints)
    end[3] += 0.4
    return _joints_result([START, end], mismatch=0.4)


def _homing(ctx, answers):
    """ctx with a PlannerClient for home_arm: the arm at HOME's own joints."""
    c = _client(answers)
    c.joints = lambda: list(START)
    c._active_joints = None  # joint_names: the bench's (ClearMarginFree reads them)
    ctx.client = c
    return c


def test_home_arm_does_not_answer_a_refused_home_with_the_bare_world(ctx):
    """home_arm answers a home the guarded bench world refuses with the bare
    table world: its band makes the START invalid when the arm holds inside
    it. A home the client refused is not that. The planner did plan in the
    guarded world, to another arm at HOME's tool pose (its FK-pose fallback),
    and the bare world would plan the same home blind to where the box may
    be (review 2026-10-28): it stops, saying what is left to try, and the
    bare world is never asked."""
    from box_opening.primitives.core import PlanRefused
    from box_opening.tasks import home_arm

    c = _homing(ctx, _other_arm)
    with pytest.raises(PlanRefused, match=r"the plan ends 0\.4000 rad from its joint goal") as refused:
        home_arm.build_legs(ctx)
    assert "not re-planned in the bare world" in str(refused.value) and "--lift-first 0.08" in str(refused.value)
    assert [kw.get("model") is None for kind, kw in ctx.worlds.pushes if kind == "bench"] == [False]  # guarded only
    assert len(c.sent) == 1


def test_home_arm_with_no_table_known_stops_on_a_refused_home(ctx, capsys):
    """Flatbot: no table, so one plan in the bare world. Refused, it raises,
    with what is left to try; the planner's own failure raises as before."""
    from box_opening.primitives.core import PlanRefused
    from box_opening.tasks import home_arm

    ctx.worlds.table_top_z = None
    c = _homing(ctx, _other_arm)
    with pytest.raises(PlanRefused, match=r"0\.4000 rad from its joint goal.*--lift-first 0\.08") as refused:
        home_arm.build_legs(ctx)
    assert "bare world" not in str(refused.value)  # it planned in nothing else
    assert len(c.sent) == 1 and "[client] plan refused" in capsys.readouterr().out
    _homing(ctx, lambda kind, g: _joints_result(success=False, message=FAILED))
    with pytest.raises(RuntimeError, match="TRAJOPT_FAIL") as failed:
        home_arm.build_legs(ctx)
    assert not isinstance(failed.value, PlanRefused)


# -- a pose plan: it ends at its pose ---------------------------------------------


@pytest.mark.parametrize(
    "off_mm, turn_deg, flies",
    [
        (0.0, 0.0, True),
        (4.9, 0.0, True),
        (0.0, 5.7, True),
        (4.9, 5.7, True),  # inside cuRobo's own success test
        (7.0, 0.0, False),
        (0.0, 6.5, False),
        (41.0, 10.0, False),  # another goal's plan: #37's stale return
    ],
)
def test_a_pose_plan_flies_only_when_it_ends_at_its_pose(chain, off_mm, turn_deg, flies):
    """cuRobo calls a pose plan a success within 5 mm and 5.73 deg of its
    pose, and the arm's FK agrees with the planner's to 0.01 mm. cuRobo
    v0.7.8's finetune can hand back the previous solve's plan as a success
    (rammp-curobo #37): the start the same, the end another goal's — a look
    from 30 deg asked for and the 40 deg one's handed back. Inside 6 mm and
    6 deg the plan is handed on as it came; past either it is refused."""
    q_end = list(START)
    q_end[1] += 0.3
    q_end[3] += 0.2
    R, t = chain.fk(q_end)
    xyz = list(np.asarray(t, float) + np.array([0.0, off_mm / 1000.0, 0.0]))
    quat = mat_to_quat_xyzw(_rz(math.radians(turn_deg)) @ np.asarray(R, float))
    res = _pose_result([START, q_end])
    got = _client(lambda kind, g: res)._plan_pose_once(xyz, quat, START)
    if flies:
        assert got is res
        return
    assert not got.success and got.trajectory is None
    assert "the plan ends %.2f mm and %.2f deg from its pose" % (off_mm, turn_deg) in got.message


def test_a_pose_plan_ending_with_the_tool_half_a_turn_round_is_refused(chain):
    """A rotation vector loses its axis at half a turn: kinematics.rotvec
    reads a flip of exactly pi as no turn at all. The turn is read off the
    trace instead, so a tool flipped half a turn about its own axis is 180
    deg from its pose, and refused."""
    q_end = list(START)
    q_end[1] += 0.3
    q_end[3] += 0.2
    R, t = chain.fk(q_end)
    flipped = np.asarray(R, float) @ np.diag([-1.0, -1.0, 1.0])  # pi about the tool's own z
    got = _client(lambda kind, g: _pose_result([START, q_end]))._plan_pose_once(
        list(t), mat_to_quat_xyzw(flipped), START
    )
    assert not got.success and "the plan ends 0.00 mm and 180.00 deg from its pose" in got.message


def test_a_pose_plan_with_no_chain_to_measure_it_by_flies_as_it_always_did(monkeypatch):
    def no_chain():
        raise RuntimeError("no arm URDF rooted at arm_mount (is /arm/robot_description published?)")

    monkeypatch.setattr(approach, "_chain", no_chain)
    res = _pose_result([START, Q1])
    assert _client(lambda kind, g: res)._plan_pose_once([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], START) is res


def test_a_success_with_nothing_to_fly_is_refused():
    assert not _client(lambda kind, g: _joints_result([])).plan_to_joints(list(START), START).success
    pose = _client(lambda kind, g: _pose_result([]))._plan_pose_once([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], START)
    assert not pose.success


@pytest.mark.parametrize("joint", [0, 3])
def test_a_plan_whose_end_is_no_number_is_refused(chain, joint):
    """NaN is never past a limit (NaN > 0.01 is False), so a NaN in the
    plan's last point read as no gap at all and flew: it is no end to fly
    to, and refused, a joint plan's and a pose plan's alike. A joint goal
    with no number in it is as far from any end as a joint can be."""
    goal = list(START)
    goal[1] += 0.3
    end = list(goal)
    end[joint] = math.nan
    got = _client(lambda kind, g: _joints_result([START, end])).plan_to_joints(goal, START)
    assert not got.success and "the plan has no finite end to check against its joint goal" in got.message
    R, t = chain.fk(goal)
    got = _client(lambda kind, g: _pose_result([START, end]))._plan_pose_once(list(t), mat_to_quat_xyzw(R), START)
    assert not got.success and "the plan has no finite end to check against its pose" in got.message
    lost = list(goal)
    lost[joint + 1] = math.nan
    got = _client(lambda kind, g: _joints_result([START, goal])).plan_to_joints(lost, START)
    assert not got.success and "the plan ends inf rad from its joint goal (joint %d" % (joint + 2) in got.message


# -- the long way round: the planner's own word -----------------------------------


@pytest.mark.parametrize(
    "joint, turn, kept_unremarked",
    [
        (0, 2 * math.pi - 0.5, False),  # 15:56:05's: joint_1 most of a turn round
        (6, -(math.pi + 0.1), True),  # joint_7 just past half a turn: the short way saves 0.16 s
    ],
)
def test_a_pose_plan_the_planner_says_goes_the_long_way_round_is_re_planned_the_short_way(
    chain, joint, turn, kept_unremarked
):
    """_no_long_way re-plans a plan whose end is quicker to reach another way
    when that saves QUICKER_BY_S (0.3 s), and flies it as it is otherwise.
    One the planner itself says goes the long way round is re-planned to the
    short way whatever that saves."""
    end = list(START)
    end[joint] += turn
    c = _client(lambda kind, g: _joints_result([START, list(g.target_joints)]))
    if kept_unremarked:
        plain = _pose_result([START, end])
        assert c._no_long_way(plain, START) is plain and c.sent == []
    got = c._no_long_way(_pose_result([START, end], message=WOUND), START)
    assert got.success and [k for k, _g in c.sent] == ["joints"]
    short = list(got.trajectory.points[-1].positions)
    assert short[joint] == pytest.approx(end[joint] - math.copysign(2 * math.pi, turn))
    assert abs(short[joint] - START[joint]) < math.pi  # the short way round


def test_a_long_way_plan_is_never_flown_as_it_is(chain, monkeypatch):
    """The re-plan the short way fails: refused, where a plan nobody said
    winds still stands (test_branches). And with no chain to find another
    end by: refused, where a plain one goes on as it always did."""
    end = list(START)
    end[0] += 2 * math.pi - 0.5
    wound, plain = _pose_result([START, end], message=WOUND), _pose_result([START, end])
    c = _client(lambda kind, g: _joints_result(success=False, message=FAILED))
    got = c._no_long_way(wound, START)
    assert not got.success and got.trajectory is None
    assert "the plan goes the long way round, and its re-plan the short way failed (%s)" % FAILED in got.message
    assert c._no_long_way(plain, START) is plain

    def no_chain():
        raise RuntimeError("no arm URDF")

    monkeypatch.setattr(approach, "_chain", no_chain)
    got = c._no_long_way(wound, START)
    assert not got.success and "no kinematics to find the short way by (no arm URDF)" in got.message
    assert c._no_long_way(plain, START) is plain


# -- planner busy: asked again --------------------------------------------------------


class _Future:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class _Planner:
    """A plan_to_* action as the client sees it: takes every goal and answers
    each with the next of `answers`, recording when each goal was sent."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.goals, self.sent_at = [], []

    def wait_for_server(self, timeout_sec=None):
        return True

    def send_goal_async(self, goal):
        self.goals.append(goal)
        self.sent_at.append(time.monotonic())
        res = self.answers.pop(0)
        wrapped = SimpleNamespace(status=4 if res.success else 6, result=res)
        return _Future(SimpleNamespace(accepted=True, get_result_async=lambda: _Future(wrapped)))


@pytest.fixture
def live(monkeypatch):
    """A client around nothing but a node: spinning it waits out its timeout."""
    monkeypatch.setattr(client_mod.rclpy, "spin_once", lambda node, timeout_sec=0.0: time.sleep(timeout_sec))
    c = client_mod.PlannerClient.__new__(client_mod.PlannerClient)
    c.node = object()
    c._abort = None
    return c


def test_a_goal_the_planner_was_too_busy_for_is_asked_again(live, capsys):
    ok = _joints_result([START, START])
    planner = _Planner(_joints_result(success=False, message=BUSY), ok)
    assert live._call(planner, PlanToJoints.Goal()) is ok
    assert len(planner.goals) == 2 and planner.sent_at[1] - planner.sent_at[0] >= live.BUSY_WAIT_S
    out = capsys.readouterr().out
    assert out.count("[client] planner busy") == 1 and "(1/2)" in out


def test_a_planner_busy_three_times_over_is_a_failed_plan_as_before(live, capsys):
    planner = _Planner(*[_joints_result(success=False, message=BUSY) for _ in range(3)])
    got = live._call(planner, PlanToJoints.Goal())
    assert not got.success and got.message == BUSY
    assert len(planner.goals) == 1 + live.BUSY_RETRIES == 3
    assert planner.sent_at[-1] - planner.sent_at[0] >= 2 * live.BUSY_WAIT_S
    assert capsys.readouterr().out.count("[client] planner busy") == 2


def test_only_busy_is_asked_again(live):
    """A plan the planner tried and failed fails the same way twice: asked once."""
    planner = _Planner(_joints_result(success=False, message=FAILED))
    assert not live._call(planner, PlanToJoints.Goal()).success and len(planner.goals) == 1


def test_a_ctrl_c_while_waiting_on_a_busy_planner_asks_nothing_more(live, monkeypatch):
    live._abort = SimpleNamespace(requested=False)

    def spin(node, timeout_sec=0.0):
        live._abort.requested = True  # Ctrl+C lands while the client waits

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    planner = _Planner(_joints_result(success=False, message=BUSY), _joints_result([START, START]))
    got = live._call(planner, PlanToJoints.Goal())
    assert not got.success and len(planner.goals) == 1


# -- the run record keeps what the planner said ---------------------------------------


def _rows(tmp_path):
    import json

    return {r["leg"]: r for r in map(json.loads, next(tmp_path.glob("run-*.jsonl")).read_text().splitlines())}


def test_a_planned_legs_row_keeps_what_the_planner_said_and_how_its_plan_ends(ctx, tmp_path):
    from box_opening.primitives.core import PlanState, _plan_motion

    ctx.client.plans = [_joints_result([Q0, Q1], mismatch=4e-7)]
    at = PlanState(joints=list(Q0), chain=0)
    leg, _st = _plan_motion(ctx, at, "home", ("joints", list(Q1)), ("full", "full.yaml"), 1.0)
    assert leg.plan["message"] == PLANNED
    runner(ctx.client, tmp_path).run([leg], execute=True)
    plan = _rows(tmp_path)["home"]["plan"]
    assert plan["message"] == PLANNED and plan["solve_s"] == 0.15 and plan["goal_mismatch_rad"] == pytest.approx(4e-7)
    assert plan["end_rad"] == 0.0 and plan["plan_s"] >= 0.0 and "replanned" not in plan and "refused" not in plan


def test_a_pose_legs_row_keeps_how_far_its_plan_ends_from_the_pose(ctx, tmp_path, chain):
    from box_opening.primitives.core import PlanState, _plan_motion

    R, t = chain.fk(Q1)
    xyz = list(np.asarray(t, float) + np.array([0.0, 0.0, 0.002]))  # cuRobo's solve, 2 mm short of the pose
    ctx.client.plans = [_pose_result([Q0, Q1])]
    leg, _st = _plan_motion(
        ctx,
        PlanState(joints=list(Q0), chain=0),
        "reach:look",
        ("pose", xyz, mat_to_quat_xyzw(R)),
        ("full", "full.yaml"),
        1.0,
    )
    runner(ctx.client, tmp_path).run([leg], execute=True)
    plan = _rows(tmp_path)["reach:look"]["plan"]
    assert plan["end_mm"] == pytest.approx(2.0, abs=0.01) and plan["end_deg"] == pytest.approx(0.0, abs=1e-3)
    # the planner's own pose plan (PlanToPose) reports none; a pose leg
    # planned to the arm's IK answer is a PlanToJoints plan, and does
    assert "goal_mismatch_rad" not in plan


def test_a_leg_planned_at_flight_keeps_its_own_plan_in_the_run_record(tmp_path):
    from box_opening.runtime.legs import Kind, Leg

    lazy = Leg(
        name="retreat",
        kind=Kind.MOTION,
        traj=None,
        speed=1.0,
        guard=None,
        world="full",
        chain=0,
        target=("joints", list(Q1)),
        goal_joints=None,
    )
    res = runner(FakeClient(), tmp_path).run([lazy], execute=True)
    assert res[-1].ok
    plan = _rows(tmp_path)["retreat"]["plan"]
    assert plan["message"] == "ok" and plan["end_rad"] == 0.0 and plan["replanned"] is True


def test_a_refused_joint_plan_the_pose_plan_stood_in_for_is_in_the_legs_plan_record(ctx, chain):
    """The leg flies the pose plan, and its record is that plan's: the
    refused joint plan before it would leave no trace but a console line.
    The record names it (PlannerClient.refusals), and only for the leg it
    was refused for."""
    from box_opening.primitives.core import PlanState, _plan_motion

    q_look = list(START)
    q_look[1] += 0.3
    q_look[3] += 0.2
    R, t = chain.fk(q_look)

    def answers(kind, g):
        return _other_arm(kind, g) if kind == "joints" else _pose_result([START, q_look])

    ctx.client = _client(answers)
    at = PlanState(joints=list(START), chain=0)
    leg, _st = _plan_motion(ctx, at, "reach:look", ("pose", list(t), mat_to_quat_xyzw(R)), ("full", "full.yaml"), 1.0)
    plan = leg.plan
    assert plan["message"] == PLANNED and "goal_mismatch_rad" not in plan  # the pose plan's
    assert plan["end_mm"] == pytest.approx(0.0, abs=0.01)
    assert len(plan["refused"]) == 1
    assert plan["refused"][0].startswith("the plan ends 0.4000 rad from its joint goal (joint 4; goal_mismatch_rad")
    ctx.client._call = lambda kind, g: _joints_result([START, list(g.target_joints)])
    home, _st = _plan_motion(ctx, at, "home", ("joints", list(HOME)), ("full", "full.yaml"), 1.0)
    assert "refused" not in home.plan and home.plan["end_rad"] == 0.0


def test_a_re_plan_that_fails_at_flight_leaves_a_note_of_what_was_said(tmp_path):
    """The failure's row is the group lead's, with the plan it no longer
    flies, or none for a lazy leg: the failed re-plan, the client's refusal
    here, is a note of its own."""
    from box_opening.runtime.legs import Kind, Leg

    c = _client(_other_arm)
    c.joints = lambda: list(START)
    lazy = Leg(
        name="home",
        kind=Kind.MOTION,
        traj=None,
        speed=1.0,
        guard=None,
        world="full",
        chain=0,
        target=("joints", list(HOME)),
        goal_joints=None,
    )
    assert runner(c, tmp_path)._replan_group([lazy], 4) == (None, 4)
    note = _rows(tmp_path)["replan_failed"]
    assert note["kind"] == "note" and note["name"] == "home" and lazy.plan is None
    assert note["plan"]["message"].startswith("the plan ends 0.4000 rad from its joint goal (joint 4")
    assert note["plan"]["refused"] == [note["plan"]["message"]] and note["plan"]["end_rad"] is None
