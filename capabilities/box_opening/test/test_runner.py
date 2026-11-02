import math
import time

import pytest
from conftest import Q0, Q1, Q2, FakeClient, leg, runner, traj
from trajectory_msgs.msg import JointTrajectoryPoint

from box_opening.runtime.guards import GuardSpec
from box_opening.runtime.legs import Kind


def test_dry_run_executes_nothing(tmp_path):
    c = FakeClient()
    res = runner(c, tmp_path).run([leg("a")], execute=False)
    assert [r.outcome for r in res] == ["skipped"]
    assert c.executed == [] and c.worlds_pushed == []


def test_merged_group_is_one_execution(tmp_path):
    c = FakeClient()
    legs = [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=0)]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert all(r.ok for r in res)
    assert len(c.executed) == 1  # merged: one goal


def test_legs_that_do_not_meet_fly_as_two_executions(tmp_path):
    """b starts 0.1 rad from where a ends. One trajectory would fly that gap
    as a jump between two of its samples (legs.MERGE_GAP_RAD). Two goals
    instead, and the second, no longer starting where the arm is, is planned
    again from there."""
    c = FakeClient()
    legs = [leg("a", Q0, Q1, chain=0), leg("b", Q2, [0.3] * 7, chain=0)]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert all(r.ok for r in res)
    assert len(c.executed) == 2
    assert c.exec_starts[1] == Q1  # from where a left the arm, not from b's own start


def test_legs_under_different_gains_fly_as_two_executions(tmp_path):
    """A goal carries one ImpedanceGains spec. A merged group used to fly
    under the first gains any member named, so a leg under the run default
    (None) flew its neighbour's. Now each spec flies as its own goal, and a
    group under one spec sends it once."""
    from box_opening.constants import GAINS_STIFF

    c = FakeClient()
    a, b = leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=0)
    b.gains = GAINS_STIFF
    res = runner(c, tmp_path).run([a, b], execute=True)
    assert all(r.ok for r in res)
    assert len(c.executed) == 2
    assert c.gains_sent == [None, GAINS_STIFF]
    c = FakeClient()
    a, b = leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=0)
    a.gains = b.gains = GAINS_STIFF
    runner(c, tmp_path).run([a, b], execute=True)
    assert len(c.executed) == 1 and c.gains_sent == [GAINS_STIFF]


class _BranchClient(FakeClient):
    """Plans a joint goal the way PlannerClient hands it to the planner: the
    start on HOME's branch (planner_branch), the goal the short way round
    from there (nearest_branch). Keeps every path it flew."""

    def plan_to_joints(self, q7, start_joints):
        from box_opening.runtime.branches import nearest_branch, planner_branch

        self.joint_starts.append(list(start_joints))
        start = planner_branch(start_joints)
        return self._plan(nearest_branch(q7, start), start)

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None, gains=None):
        self.flown = getattr(self, "flown", []) + [[list(p.positions) for p in traj_.points]]
        return super().execute(traj_, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)


def _turned(path, j):
    """How far joint `j` turns along a flown path, the numbers as written."""
    return sum(abs(b[j] - a[j]) for a, b in zip(path, path[1:], strict=False))


# HOME's joint_3 is -3.137. The client hands the planner joint_3 0.30 as
# -5.983 and 0.45 as -5.833, but 0.20 as it is: the other way round, -6.083,
# is past the planner's -6.0. From a start under 0.283 a goal at 0.45 is
# written +0.45, a whole turn from the -5.833 the next leg starts at
_OVER = [-1.0, -0.15, 0.20, -1.05, -0.3, -1.0, 1.8]
_ON = [-1.0, -0.2, 0.20, -1.1, -0.3, -1.0, 1.8]
_ON_PAST = [-1.0, -0.2, 0.30, -1.1, -0.3, -1.0, 1.8]
_UP = [-1.0, -0.2, 0.45, -1.1, -0.3, -1.0, 1.8]


def test_a_lazy_group_planned_a_whole_turn_apart_flies_as_two_goals(tmp_path):
    """A lazy group merges before its legs have trajectories, so the merge
    cannot see where they meet. Planned at flight from live (joint_3 0.20,
    handed as it is), the retreat ends at +0.45, and the home planned from
    there starts at -5.833: one angle to the arm, a whole turn as written.
    Flown as one goal, joint_3 turned 9.23 rad (offline replay 2026-10-28).
    Two goals, and neither turns it half a turn."""
    from box_opening.constants import HOME

    c = _BranchClient()
    c.live = list(_OVER)
    g = GuardSpec(touch_nm=4.0, trip="setdown")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9, "torque_peak": 4.0})]
    down = leg("place:lid:down", _OVER, _ON, guard=g, world="interaction_place", chain=0)
    retreat = leg("retreat", _ON, _UP, chain=1, world="interaction_place")
    home = leg("home", _UP, HOME, chain=1)
    for x in (retreat, home):
        x.traj = x.goal_joints = None  # lazy: planned from live after the touch
    res = runner(c, tmp_path).run([down, retreat, home], execute=True)
    assert [r.leg_name for r in res] == ["place:lid:down", "retreat", "home"]
    assert all(r.ok for r in res)
    assert c.exec_starts[2][2] == pytest.approx(_UP[2] - 2.0 * math.pi)
    assert max(_turned(p, 2) for p in c.flown) < math.pi


def test_a_group_replanned_under_the_recoil_a_whole_turn_apart_flies_as_two_goals(tmp_path):
    """The retreat and the home were planned from the press's end, joint_3
    0.30, handed to the planner as -5.983: they met and merged. The recoil
    plans them again from its own end, 0.20, handed as it is: the retreat
    now ends at +0.45, and the home planned from there starts at -5.833.
    The runner checks a junction again before it flies it, not only when
    the legs are first planned."""
    from box_opening.constants import HOME
    from box_opening.runtime.branches import planner_branch
    from box_opening.runtime.legs import merge_groups

    c = _BranchClient()
    c.live = list(_OVER)
    press = leg(
        "press:down", _OVER, _ON_PAST, guard=GuardSpec(touch_nm=7.0, trip="press"), world="interaction_button", chain=0
    )
    up = planner_branch(_UP)
    retreat = leg("retreat", planner_branch(_ON_PAST), up, chain=1, world="interaction_button")
    home = leg("home", up, HOME, chain=1)
    assert [len(g) for g in merge_groups([press, retreat, home])] == [1, 2]  # as planned, they meet
    c.exec_script = [("touch", {"message": "contact", "progress": 0.8, "torque_peak": 7.1})]
    res = runner(c, tmp_path).run([press, retreat, home], execute=True)
    assert [r.leg_name for r in res] == ["press:down", "recoil", "retreat", "home"]
    assert all(r.ok for r in res)
    assert len(c.joint_starts) == 2  # both planned once, under the recoil
    assert c.exec_starts[3][2] == pytest.approx(_UP[2] - 2.0 * math.pi)
    assert max(_turned(p, 2) for p in c.flown) < math.pi


def test_unguarded_leg_requires_full_world(tmp_path):
    c = FakeClient()
    res = runner(c, tmp_path).run([leg("a", world="interaction_button")], execute=True)
    assert res[0].outcome == "refused" and c.executed == []


def test_nothing_moves_while_the_client_is_not_armed(tmp_path):
    """The driver has no dry-run gate of its own: the client's motion latch
    (set only by --execute) stands between a wiring bug and the arm."""
    c = FakeClient()
    c.armed = False
    res = runner(c, tmp_path).run([leg("a")], execute=True)
    assert res[0].outcome == "refused" and c.executed == []
    gl = leg("close", kind=Kind.GRIPPER, cmd=0.8)
    res = runner(c, tmp_path).run([gl], execute=True)
    assert res[0].outcome == "refused" and c.gripper_sent == []


def test_a_search_leg_that_stops_early_has_done_its_job(tmp_path):
    """The look and the sweep exist to find the box, not to reach their goal:
    stopping part-way because a fix committed is success, and the mission
    carries on from where the arm stopped."""
    c = FakeClient()
    c.exec_script = [("stopped", {"message": "a box came into view", "progress": 0.4})]
    lg = leg("look", world="bench")
    lg.stop_when = lambda progress=None: True
    res = runner(c, tmp_path).run([lg], execute=True)
    assert res[0].outcome == "stopped" and res[0].ok


def test_an_ordinary_leg_stopping_early_is_still_a_failure(tmp_path):
    c = FakeClient()
    c.exec_script = [("stopped", {"message": "?", "progress": 0.4})]
    res = runner(c, tmp_path).run([leg("a")], execute=True)
    assert res[0].outcome == "stopped" and not res[0].ok


def test_guarded_leg_refused_without_efforts(tmp_path):
    c = FakeClient()
    c.efforts = False
    g = GuardSpec(touch_nm=3.0, trip="press")
    res = runner(c, tmp_path).run([leg("press", guard=g, world="interaction_b")], execute=True)
    assert res[0].outcome == "refused" and c.executed == []


def test_start_drift_triggers_replan(tmp_path):
    c = FakeClient()
    c.live = [0.06] + [0.0] * 6  # 0.06 > 0.04 threshold
    r = runner(c, tmp_path)
    res = r.run([leg("a", Q0, Q1)], execute=True)
    assert res[0].ok
    # re-planned from live: the executed trajectory starts at the live joints
    assert c.exec_starts and c.exec_starts[0][0] == 0.06


def test_other_failure_stops_without_retry(tmp_path):
    c = FakeClient()
    c.exec_script = [
        (
            "failed",
            {"message": "controller rejected", "progress": 0.4, "torque_peak": None},
        )
    ]
    res = runner(c, tmp_path).run([leg("a"), leg("b", Q1, Q2)], execute=True)
    assert res[0].outcome == "failed"
    assert len(res) == 1 and len(c.executed) == 1  # stopped, b never ran


def test_contact_invalidates_downstream(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.5, "torque_peak": 4.0})]
    legs = [
        leg("descend", guard=g, world="interaction_x", chain=0),
        leg("after", Q1, Q2, chain=0),
    ]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert res[0].outcome == "touch" and res[0].ok  # setdown: trip = success
    assert res[1].ok
    assert len(c.executed) == 2  # never merged with contact


def test_jsonl_log_written(tmp_path):
    import json

    c = FakeClient()
    runner(c, tmp_path).run([leg("a")], execute=True)
    logs = list(tmp_path.glob("run-*.jsonl"))
    assert len(logs) == 1
    row = json.loads(logs[0].read_text().splitlines()[0])
    assert row["leg"] == "a" and row["outcome"] == "arrived"


def test_transit_gate_accepts_bench_world_pre_detection(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    assert r._refusal(leg("scan", world="bench")) is None
    assert "full or bench" in r._refusal(leg("weird", world="interaction_button"))


def test_replanned_trajectories_pass_the_sanity_gate(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    wandering = traj(Q0, Q1)
    mid = JointTrajectoryPoint()
    mid.positions = [1.5] + [0.05] * 6  # joint_1 wanders way out and back
    mid.time_from_start.sec = 1
    wandering.points.insert(1, mid)
    wandering.points[-1].time_from_start.sec = 2

    class R:
        success = True
        message = "ok"
        trajectory = wandering

    c.plans = [R]
    group, chain = r._replan_group([leg("a")], next_chain=5)
    assert group is None  # wandering replan refused, same gate as pre-built


def test_fast_retreat_allowed_in_interaction_world(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    ok_leg = leg("retreat", speed=0.35, world="interaction_button")
    assert r._refusal(ok_leg) is None  # ascends its own corridor
    other = leg("wander", speed=0.35, world="interaction_button")
    assert r._refusal(other) is not None  # only retreats get the pass


def test_no_replan_when_contact_left_arm_on_plan(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.97, "torque_peak": 4.0})]
    legs = [
        leg("descend", guard=g, world="interaction_x", chain=0),
        leg("retreat", Q1, Q2, chain=0, speed=0.15, world="interaction_x"),
    ]
    r = runner(c, tmp_path)
    res = r.run(legs, execute=True)
    assert all(x.ok for x in res)
    # live == planned start (FakeClient tracks to traj end): NO replan —
    # the pre-planned retreat executed as built (no pause at the bottom)
    assert c.exec_starts[1] == Q1


def test_lift_may_run_faster_than_contact_in_an_interaction_world(tmp_path):
    """Lift ascends out of the corridor it just descended, exactly like
    retreat — both are exempt from the transit-speed world gate."""
    c = FakeClient()
    res = runner(c, tmp_path).run(
        [leg("lift", speed=0.35, world="interaction_button")],
        execute=True,
    )
    assert res[0].ok and res[0].outcome != "refused"


def test_unguarded_fast_leg_in_an_interaction_world_is_still_refused(tmp_path):
    """The exemption is name-scoped — it must not open the gate widely."""
    c = FakeClient()
    res = runner(c, tmp_path).run(
        [leg("place:lid:transit", speed=0.35, world="interaction_place")],
        execute=True,
    )
    assert res[0].outcome == "refused"


def test_merged_group_takes_its_guard_from_the_guarded_member(tmp_path):
    """can_merge forbids this today; the lookup is what keeps a future
    relaxation from running a guarded stroke with the lead's (absent)
    guard, the lead's speed and the lead's verify."""
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    group = [leg("fast", Q0, Q1, chain=0), leg("descend", Q1, Q2, chain=0, guard=g)]
    r = runner(c, tmp_path)
    res = r._run_motion(group)
    # setdown semantics come from the guarded member: a trip is SUCCESS
    assert res.outcome == "touch" and res.ok


def test_merged_group_refuses_a_guard_that_is_not_last(tmp_path):
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    group = [leg("descend", Q0, Q1, chain=0, guard=g), leg("after", Q1, Q2, chain=0)]
    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="not last"):
        runner(c, tmp_path)._run_motion(group)


def test_replans_push_the_legs_own_world(tmp_path):
    """Worlds are pushed at plan time and by the replan path per leg —
    execution never consults the collision world, so the runner no longer
    pushes per group (one tracker lives in the client, audit 2026-09-02).
    A replanned leg must reach the planner with ITS world (content-hashed
    path), not whatever was loaded."""
    c = FakeClient()
    a = leg("a", Q0, Q1, world="full", chain=0)
    a.world_path = "/w/full-aaaa.yaml"
    b = leg("b", Q1, Q2, world="full", chain=1)
    b.world_path = "/w/full-bbbb.yaml"  # same name, new contents
    c.live = [0.06] + [0.0] * 6  # drift: a replans, then b chains clean
    runner(c, tmp_path).run([a, b], execute=True)
    assert c.worlds_pushed == ["/w/full-aaaa.yaml"]


def test_lazy_leg_is_planned_once_from_live_at_execution(tmp_path):
    """A lazy leg (traj None) after a touch is planned from live joints
    exactly when it executes, in its own world."""
    c = FakeClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9, "torque_peak": 4.0})]
    lazy = leg("retreat", Q1, Q2, chain=1, world="interaction_x")
    lazy.traj = None
    lazy.goal_joints = None
    lazy.world_path = "/w/interaction_x-1.yaml"
    legs = [leg("down", guard=g, world="interaction_x", chain=0), lazy]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert [r.outcome for r in res] == ["touch", "arrived"]
    assert c.worlds_pushed == ["/w/interaction_x-1.yaml"]
    assert len(c.executed) == 2


class _AsyncGripClient(FakeClient):
    """FakeClient that records send/join ordering against motion."""

    def __init__(self):
        super().__init__()
        self.events = []

    def gripper_send(self, position):
        self.events.append("send")
        return ("handle", position)

    def gripper_join(self, handle):
        self.events.append("join")
        return True, float(handle[1]), False

    def gripper_cmd(self, position):
        # the real client's blocking path is send + join; mirror it so the
        # ordering assertions mean the same thing on both paths
        if position is None:
            return super().gripper_cmd(position)
        return self.gripper_join(self.gripper_send(position))

    def execute(self, traj, speed, guard=None, while_running=None, stop_when=None):
        self.events.append("execute")
        return super().execute(traj, speed, guard=guard, while_running=while_running)


def test_deferred_gripper_close_overlaps_the_next_transit(tmp_path):
    """The owner's own example: fingers shut WHILE the arm moves."""
    c = _AsyncGripClient()
    close = leg("press:close", kind=Kind.GRIPPER, cmd=0.8)
    close.defer_join = True
    legs = [close, leg("approach", Q0, Q1, chain=0)]
    r = runner(c, tmp_path)
    res = r.run(legs, execute=True)
    assert all(r_.ok for r_ in res)
    # sent, THEN the transit ran; the join is lazy — it outlives the run
    # to overlap whatever planning follows, and finish() collects it
    assert c.events == ["send", "execute"]
    done = r.finish()
    assert done.ok and done.gripper_pos == pytest.approx(0.8)  # the knuckle the join read
    assert c.events == ["send", "execute", "join"]


def test_deferred_gripper_is_joined_before_any_guarded_leg(tmp_path):
    """A press descends with the fingers closed — the overlap must not
    let a guarded leg start while they are still moving."""
    c = _AsyncGripClient()
    g = GuardSpec(touch_nm=3.0, trip="press")
    close = leg("press:close", kind=Kind.GRIPPER, cmd=0.8)
    close.defer_join = True
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    legs = [close, leg("press:down", Q0, Q1, chain=0, guard=g, world="interaction_b")]
    runner(c, tmp_path).run(legs, execute=True)
    assert c.events.index("join") < c.events.index("execute")


def test_a_release_is_never_deferred(tmp_path):
    """defer_join is opt-in; an un-flagged gripper leg still blocks."""
    c = _AsyncGripClient()
    legs = [leg("place:lid:open", kind=Kind.GRIPPER, cmd=0.0), leg("retreat", Q0, Q1)]
    runner(c, tmp_path).run(legs, execute=True)
    assert c.events[:2] == ["send", "join"], "release settles before the arm moves"


def test_touch_forces_replan_even_under_the_drift_gate(tmp_path):
    """After a guard trip the predicted start is wrong BY DESIGN, and a
    sub-gate joint delta is still a multi-mm Cartesian shove into the
    thing just touched: the pre-planned retreat re-pressed the button at
    full speed, guardless (field 2026-09-02). Post-touch, the next
    motion replans from live unconditionally."""

    class TouchStopsShort(FakeClient):
        # a real trip halts the arm shy of the endpoint; the plain fake
        # teleports to it, which would hide exactly the hazard under test
        def execute(self, traj, speed, guard=None, while_running=None, stop_when=None):
            outcome, info = super().execute(traj, speed, guard, while_running=while_running)
            if outcome == "touch":
                self.live = [self.live[0] + 0.01] + self.live[1:]
            return outcome, info

    c = TouchStopsShort()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9, "torque_peak": 4.0})]
    legs = [
        leg("descend", guard=g, world="interaction_x", chain=0),
        leg("retreat", Q1, Q2, chain=1),
    ]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert res[0].outcome == "touch" and res[1].ok
    # the trip left the arm 0.01 rad from the predicted end — well under
    # the 0.04 free-air drift gate, which must NOT matter after a touch
    assert c.exec_starts[1][0] == 0.11  # replanned from live, not Q1


def test_arrived_leg_still_uses_the_drift_gate(tmp_path):
    """Free-air chaining keeps the owner's fewer-pauses rule: no touch,
    sub-gate drift, the pre-planned leg runs as planned. (The drift must
    appear AFTER leg a runs — the fake teleports live to each executed
    endpoint, so a pre-set offset only ever tests leg a's own gate.)"""

    class DriftsAfterArrive(FakeClient):
        def execute(self, traj, speed, guard=None, while_running=None, stop_when=None):
            outcome, info = super().execute(traj, speed, guard, while_running=while_running)
            if outcome == "arrived" and len(self.executed) == 1:
                self.live = [self.live[0] + 0.01] + self.live[1:]
            return outcome, info

    c = DriftsAfterArrive()
    legs = [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=1)]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert all(r.ok for r in res)
    # 0.01 under the 0.04 gate, no touch: pre-planned start kept — an
    # always-replan mutation would execute from live (0.11) instead
    assert c.exec_starts[1][0] == 0.1


def test_replanned_warped_leg_keeps_its_execution_profile(tmp_path):
    """_apply_warp bakes fast-then-slow into the trajectory and sets
    speed=1.0 as a do-not-dilate sentinel. A replan swaps in a fresh
    UNWARPED trajectory — inheriting the sentinel would descend into
    contact at full speed (review 2026-09-02). The profile is re-warped,
    or the leg honestly downgrades to the slow contact speed."""
    from box_opening.runtime.runner import _restore_execution_profile

    g = GuardSpec(touch_nm=3.0, trip="setdown", rebaseline_after=0.7)

    # a trajectory long enough to warp: profile re-applied, sentinel kept
    many = [[0.0 + 0.01 * i] * 7 for i in range(40)]
    t = traj(many[0], many[-1])
    t.points = []
    from trajectory_msgs.msg import JointTrajectoryPoint

    for i, row in enumerate(many):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in row]
        pt.velocities = [0.0] * 7
        pt.accelerations = [0.0] * 7
        pt.time_from_start.sec = i
        t.points.append(pt)
    lg = leg("down", guard=g, world="interaction_x")
    lg.speed = 1.0
    lg.warp = (0.5, 0.35, 0.3)
    lg.traj = t
    _restore_execution_profile(lg)
    assert lg.speed == 1.0  # profile baked in again
    assert lg.guard.rebaseline_after is not None
    end = lg.traj.points[-1].time_from_start
    assert end.sec + end.nanosec * 1e-9 > 39.0  # slower than the raw plan

    # a degenerate trajectory that cannot be warped: honest downgrade
    lg2 = leg("down2", guard=g, world="interaction_x")
    lg2.speed = 1.0
    lg2.warp = (0.35, 0.35, 0.3)  # fast==slow -> warp declines
    _restore_execution_profile(lg2)
    assert lg2.speed == 0.35  # the slow contact speed, never the sentinel
    assert lg2.guard.rebaseline_after is None

    # the retime hook follows the trajectory actually flown
    seen = {}
    lg3 = leg("press", guard=g, world="interaction_x")
    lg3.retime = lambda traj: seen.__setitem__("traj", traj)
    _restore_execution_profile(lg3)
    assert seen["traj"] is lg3.traj


def test_guard_observes_but_cannot_trip_before_arm_after():
    """The fast warp segment's dynamics tripped the gentle set-down
    threshold 64 ms into the descent and the lid was released 110 mm up
    (field 2026-09-02). Before arm_after the guard watches but never
    trips; after it, the same deviation trips."""
    from box_opening.runtime.guards import TorqueGuard

    g = TorqueGuard(4.0, arm_after=0.5)
    g.on_progress(0.05)
    assert g.on_efforts([0.0] * 4) is False  # baseline
    assert g.on_efforts([9.0, 0.0, 0.0, 0.0]) is False  # fast-zone jolt held
    assert g.peak == 9.0  # still observed
    g.on_progress(0.6)
    assert g.on_efforts([9.0, 0.0, 0.0, 0.0]) is True  # same dev now trips


def test_pending_gripper_outlives_the_run_and_joins_before_a_guarded_leg(tmp_path):
    """grip:open is dispatched on arrival at the hop (last leg of the press
    phase) and must overlap the NEXT phase's planning — so a run() ends
    without joining it, and the following run joins it before its guarded
    descent (audit 2026-09-02)."""
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    open_leg = leg("grip:open", kind=Kind.GRIPPER, cmd=0.0)
    open_leg.defer_join = True
    r.run([leg("retreat", Q0, Q1), open_leg], execute=True)
    assert c.events == ["execute", "send"]  # NOT joined at the end of run()
    g = GuardSpec(touch_nm=3.0, trip="obstruction")
    r.run([leg("grip:down", Q1, Q2, guard=g, world="interaction_b")], execute=True)
    assert c.events == ["execute", "send", "join", "execute"]


def test_start_gripper_dispatches_now_and_joins_lazily(tmp_path):
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    assert r.start_gripper("press:close", 0.8, execute=True)
    assert c.events == ["send"]
    g = GuardSpec(touch_nm=3.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    r.run([leg("press:down", Q0, Q1, guard=g, world="interaction_b")], execute=True)
    # the trailing execute is the reflex recoil off the button
    assert c.events == ["send", "join", "execute", "execute"]
    assert r.finish() is None  # nothing left pending


def test_release_overlaps_the_replan_but_never_the_motion(tmp_path):
    """place:lid:open is dispatched at once; the retreat's post-touch replan
    proceeds while the fingers open; the join lands before the retreat
    EXECUTES (a release completes before the arm moves away)."""
    c = _AsyncGripClient()
    g = GuardSpec(touch_nm=3.0, trip="setdown")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    release = leg("place:lid:open", kind=Kind.GRIPPER, cmd=0.0)
    release.defer_join = True
    release.join_before_motion = True

    class Spy(_AsyncGripClient):
        def plan_to_pose(self, *a, **k):
            self.events.append("plan")
            return super().plan_to_pose(*a, **k)

    c = Spy()
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    lazy = leg("retreat", Q1, Q2, chain=1, world="interaction_x")
    lazy.traj = None
    lazy.target = ("pose", [0.5, 0.0, 0.2], [0.0, 1.0, 0.0, 0.0])
    legs = [
        leg("down", guard=g, world="interaction_x", chain=0),
        release,
        lazy,
    ]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert all(r.ok for r in res)
    assert c.events == ["execute", "send", "plan", "join", "execute"]


def test_lookahead_runs_during_the_last_unguarded_motion(tmp_path):
    """The next phase is planned while this run's last unguarded motion
    flies, from that motion's predicted end joints; a guarded stroke never
    hosts it (audit 2026-09-02)."""
    c = FakeClient()
    r = runner(c, tmp_path)
    seen = []
    res = r.run(
        [leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=1)],
        execute=True,
        lookahead=lambda q: seen.append(list(q)) or ["next-legs"],
    )
    assert all(x.ok for x in res)
    assert seen == [Q2]  # hosted on b, from b's predicted end
    assert r.lookahead_result == ["next-legs"]

    # a failing lookahead is not a failed leg: the caller builds afterwards
    def boom(q):
        raise RuntimeError("planner said no")

    res = r.run([leg("c", Q2, Q1, chain=0)], execute=True, lookahead=boom)
    assert res[0].ok and r.lookahead_result is None

    # guarded strokes never host it
    g = GuardSpec(touch_nm=3.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    seen.clear()
    r.run([leg("p", Q1, Q2, guard=g, world="interaction_b")], execute=True, lookahead=lambda q: seen.append(q))
    assert seen == [] and r.lookahead_result is None


def test_planned_lead_never_merges_with_a_lazy_tail():
    from box_opening.runtime.legs import merge_groups

    a = leg("a", Q0, Q1, chain=1)
    b = leg("b", Q1, Q2, chain=1)
    b.traj = None
    assert [len(g) for g in merge_groups([a, b])] == [1, 1]
    c = leg("c", Q1, Q2, chain=1)
    c.traj = None
    assert [len(g) for g in merge_groups([b, c])] == [2]  # both lazy: one group


def test_restore_profile_moves_arm_after_with_the_rebaseline():
    """A replanned warped set-down gets a new time base; the guard must not
    be armed before the NEW slow zone or it judges slow-zone efforts
    against the fast baseline (review 2026-09-02)."""
    from trajectory_msgs.msg import JointTrajectoryPoint

    from box_opening.runtime.runner import _restore_execution_profile

    many = [[0.0 + 0.01 * i] * 7 for i in range(40)]
    t = traj(many[0], many[-1])
    t.points = []
    for i, row in enumerate(many):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in row]
        pt.velocities = [0.0] * 7
        pt.accelerations = [0.0] * 7
        pt.time_from_start.sec = i
        t.points.append(pt)
    g = GuardSpec(touch_nm=4.0, trip="setdown", rebaseline_after=0.3, arm_after=0.5)
    lg = leg("down", guard=g, world="interaction_x")
    lg.speed = 1.0
    lg.warp = (0.5, 0.35, 0.3)
    lg.traj = t
    _restore_execution_profile(lg)
    assert lg.guard.rebaseline_after is not None
    assert lg.guard.arm_after >= lg.guard.rebaseline_after
    assert lg.guard.arm_after >= 0.5


def test_ctrl_c_during_a_hosted_lookahead_still_cancels_the_goal():
    """The hosted plan runs INSIDE execute's cancel backstop: a Ctrl+C
    raised from it (second press) reaches cancel_goal_async, and a first
    press that only set the abort flag while the motion finished under
    the plan is delivered as a confirmed cancel, not swallowed (review
    2026-09-02)."""
    import pytest

    from box_opening.runtime import client as client_mod

    class Fut:
        def __init__(self, result=None, done=True):
            self._r, self._d = result, done

        def done(self):
            return self._d

        def result(self):
            return self._r

    class Send:
        accepted = True

        def __init__(self, events):
            self.events = events

        def get_result_async(self):
            return Fut(done=False)  # the motion is flying

        def cancel_goal_async(self):
            self.events.append("cancel")
            return Fut(result=object())

    class Exec:
        def __init__(self, events):
            self.events = events

        def wait_for_server(self, timeout_sec=None):
            return True

        def send_goal_async(self, goal, feedback_callback=None):
            return Fut(result=Send(self.events))

    class Abort:
        requested = False
        goal_in_flight = False

    class Node:
        pass

    events = []
    c = client_mod.PlannerClient.__new__(client_mod.PlannerClient)
    c.node = Node()
    c._execute = Exec(events)
    c._abort = Abort()
    # armed, with a fresh live state at the trajectory's start: the client
    # refuses anything else before a goal is ever sent
    c._armed = True
    c._eff, c._eff_at, c._q = [0.0] * 7, time.monotonic(), list(Q0)
    c.wrist_efforts = lambda: None
    c._cancel_confirm = lambda send, fut: events.append("cancel-confirm")
    # a planned trajectory's first point is at dt, never t=0 (cuRobo stamps
    # (k + 1) * dt) — the client refuses zero-stamped timing
    flown = traj(Q0, Q1)
    flown.points[0].time_from_start.nanosec = 20000000
    orig_spin = client_mod.rclpy.spin_once
    client_mod.rclpy.spin_once = lambda node, timeout_sec=0.0: None
    try:
        # second Ctrl+C: raised from inside the hosted plan
        def boom():
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            c.execute(flown, 0.5, guard=None, while_running=boom)
        assert "cancel" in events  # the backstop cancel was sent
        events.clear()

        # first Ctrl+C: the flag is set while the plan runs; the motion may
        # even have finished meanwhile — it must still be delivered
        def flag():
            c._abort.requested = True
            return "legs"

        with pytest.raises(KeyboardInterrupt):
            c.execute(flown, 0.5, guard=None, while_running=flag)
        assert events == ["cancel-confirm"]
        assert c._abort.goal_in_flight is False
    finally:
        client_mod.rclpy.spin_once = orig_spin


def test_unguarded_groups_fly_the_retimed_profile_at_the_sentinel(tmp_path):
    """Every unguarded motion group is re-timed (positions untouched) and
    executed at the 1.0 sentinel; guarded strokes keep their own speed."""
    c = FakeClient()
    r = runner(c, tmp_path)
    res = r.run([leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=0)], execute=True)
    assert all(x.ok for x in res)
    assert c.executed[-1][1] == 1.0  # the profile is baked in
    assert r.last_retime is not None and r.last_retime["n_points"] == 3  # junction de-duplicated
    g = GuardSpec(touch_nm=3.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    r.run([leg("p", Q1, Q2, guard=g, world="interaction_b", speed=0.35)], execute=True)
    assert c.executed[-1][1] == 0.35  # guarded: not re-timed here


def test_speed_scale_dilates_guarded_and_unguarded_alike(tmp_path):
    c = FakeClient()
    r = runner(c, tmp_path)
    r.time_scale = 0.5
    r.run([leg("a", Q0, Q1)], execute=True)
    assert c.executed[-1][1] == 1.0  # profile baked in (dilated inside it)
    assert r.last_retime is not None
    g = GuardSpec(touch_nm=3.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    r.run([leg("p", Q1, Q2, guard=g, world="interaction_b", speed=0.35)], execute=True)
    # [-1] is the recoil (re-timed, 1.0 sentinel); the guarded stroke is [-2]
    assert c.executed[-2][1] == pytest.approx(0.175)  # guarded: speed x scale
    assert c.executed[-1][1] == 1.0


def _press_leg(**kw):
    g = GuardSpec(touch_nm=7.0, trip="press")
    return leg("press:down", Q0, Q1, guard=g, world="interaction_button", **kw)


def test_press_trip_recoils_before_the_next_leg_is_planned(tmp_path):
    """A guard trip leaves the arm on the button; the recoil reverses the
    descent's own path at once and the considered leg is planned from the
    recoil's end while it flies (2026-09-03)."""
    c = FakeClient()
    c.exec_script = [("touch", {"message": "contact", "progress": 0.8, "torque_peak": 7.1})]
    res = runner(c, tmp_path).run(
        [_press_leg(chain=0), leg("retreat", Q1, Q2, chain=1, world="interaction_button")],
        execute=True,
    )
    assert [r.leg_name for r in res] == ["press:down", "recoil", "retreat"]
    assert all(r.ok for r in res)
    assert len(c.executed) == 3  # press, recoil, retreat
    # the retreat was planned ONCE, from the recoil's end — not replanned
    # again from live afterwards
    assert len(c.joint_starts) == 1


def test_a_set_down_trip_never_recoils(tmp_path):
    """The trip IS the success there and the lid must be released while it
    rests on the table — recoiling would drop it from 20 mm up."""
    c = FakeClient()
    g = GuardSpec(touch_nm=4.0, trip="setdown", arm_after=0.5)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9, "torque_peak": 4.2})]
    res = runner(c, tmp_path).run(
        [leg("place:lid:down", Q0, Q1, guard=g, world="interaction_place", chain=0)],
        execute=True,
    )
    assert [r.leg_name for r in res] == ["place:lid:down"]
    assert len(c.executed) == 1


def test_a_failed_press_trip_holds_where_it_struck(tmp_path):
    """An early trip means the stroke hit something above the button: the
    arm holds so the operator can see it, no recoil."""
    c = FakeClient()
    c.exec_script = [("touch", {"message": "contact", "progress": 0.1, "torque_peak": 7.0})]
    press = _press_leg(chain=0)
    press.verify = lambda v: (False, "guard tripped EARLY at 10% of the stroke")
    res = runner(c, tmp_path).run([press], execute=True)
    assert [r.leg_name for r in res] == ["press:down"] and not res[0].ok
    assert len(c.executed) == 1


def test_a_trip_records_where_the_fingertips_were(tmp_path):
    """The arm measures the surface it touched: at a trip the fingertip TF
    is the contact height, independent of the camera and of every model
    constant (2026-09-03)."""
    c = FakeClient()
    c.contact_at = [0.44, -0.16, 0.0837]
    g = GuardSpec(touch_nm=7.0, trip="press")
    c.exec_script = [("touch", {"message": "contact", "progress": 0.85})]
    res = runner(c, tmp_path).run(
        [leg("press:down", Q0, Q1, guard=g, world="interaction_button")],
        execute=True,
    )
    assert res[0].contact_xyz == [0.44, -0.16, 0.0837]
    # an untripped leg has nothing to report
    c.exec_script = []
    res2 = runner(c, tmp_path).run([leg("a", Q0, Q1)], execute=True)
    assert res2[0].contact_xyz is None


def test_flagged_gripper_leg_is_sent_under_the_previous_motion(tmp_path):
    """grip:open rides the retreat: sent the moment the retreat's goal is
    accepted, joined before the guarded grip:down as ever."""
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    retreat = leg("retreat", Q0, Q1, chain=1, world="interaction_b")
    open_leg = leg("grip:open", kind=Kind.GRIPPER, cmd=0.0, chain=1, world="interaction_b")
    open_leg.defer_join = True
    open_leg.send_with_previous_motion = True
    g = GuardSpec(touch_nm=7.0, trip="obstruction")
    down = leg("grip:down", Q1, Q2, guard=g, world="interaction_b", chain=2)
    res = r.run([retreat, open_leg, down], execute=True)
    assert [x.leg_name for x in res] == ["retreat", "grip:open", "grip:down"]
    # the send happened INSIDE the retreat's execute, and the join before grip:down
    assert c.events == ["execute", "send", "join", "execute"]


def test_flagged_gripper_leg_is_not_sent_under_a_guarded_motion(tmp_path):
    c = _AsyncGripClient()
    r = runner(c, tmp_path)
    g = GuardSpec(touch_nm=7.0, trip="obstruction")
    down = leg("grip:down", Q0, Q1, guard=g, world="interaction_b", chain=1)
    open_leg = leg("grip:open", kind=Kind.GRIPPER, cmd=0.0, chain=1, world="interaction_b")
    open_leg.defer_join = True
    open_leg.send_with_previous_motion = True
    r.run([down, open_leg], execute=True)
    assert c.events[:2] == ["execute", "send"]  # sent after, in order, not during
    r.finish()


class _StillGripClient(_AsyncGripClient):
    """_AsyncGripClient whose waits for a still arm are events too, each
    answering `settle` (s waited, timed out)."""

    def __init__(self, settle=(0.31, False)):
        super().__init__()
        self.settle = settle

    def wait_still(self):
        self.events.append("still")
        return self.settle


def _notes(tmp_path, kind):
    import json

    return [r for r in map(json.loads, next(tmp_path.glob("run-*.jsonl")).read_text().splitlines()) if r["leg"] == kind]


def test_a_gripper_leg_sent_when_still_goes_out_once_the_arm_stands_still_and_the_record_says_how_long(tmp_path):
    """reach_lid_off's grip:close, after the way down to the knob (the run
    before), the open that rode it still pending: in the joint impedance
    the way down "arrives" when its clock runs out, the arm still
    settling. The open is joined, then the arm waited still
    (Runner.wait_still, noted 'still'), then the fingers close. A gripper
    leg not flagged goes out at once."""
    c = _StillGripClient()
    r = runner(c, tmp_path)
    opened = leg("grip:open", kind=Kind.GRIPPER, cmd=0.0)
    opened.defer_join = True
    r.run([opened, leg("grip:down", Q0, Q1)], execute=True)
    close = leg("grip:close", kind=Kind.GRIPPER, cmd=0.8, verify=lambda ctx: (True, "on the knob"))
    close.send_when_still = True
    res = r.run([close, leg("lid:lift", Q1, Q2)], execute=True)
    assert all(x.ok for x in res) and [x.leg_name for x in res] == ["grip:open", "grip:close", "lid:lift"]
    assert c.events == ["send", "execute", "join", "still", "send", "join", "execute"]
    (note,) = _notes(tmp_path, "still")
    assert (note["before"], note["settle_s"], note["timed_out"]) == ("grip:close", 0.31, False)
    c2 = _StillGripClient()
    runner(c2, tmp_path).run([leg("grip:down", Q0, Q1), leg("grip:close", kind=Kind.GRIPPER, cmd=0.8)], execute=True)
    assert "still" not in c2.events


def test_a_wait_that_runs_out_is_noted_and_the_fingers_close_all_the_same(tmp_path):
    """A wait for a still arm never costs the run: past client.STILL_TIMEOUT_S
    it carries on, the record saying it timed out."""
    c = _StillGripClient(settle=(1.2, True))
    close = leg("grip:close", kind=Kind.GRIPPER, cmd=0.8)
    close.send_when_still = True
    res = runner(c, tmp_path).run([leg("grip:down", Q0, Q1), close], execute=True)
    assert all(x.ok for x in res) and c.events == ["execute", "still", "send", "join"]
    (note,) = _notes(tmp_path, "still")
    assert (note["settle_s"], note["timed_out"]) == (1.2, True)


def test_a_gripper_leg_sent_when_still_is_never_sent_under_the_previous_motion(tmp_path):
    """send_when_still wins over send_with_previous_motion: under the
    motion, the arm is anything but still."""
    c = _StillGripClient()
    r = runner(c, tmp_path)
    close = leg("grip:close", kind=Kind.GRIPPER, cmd=0.8)
    close.defer_join = close.send_with_previous_motion = close.send_when_still = True
    r.run([leg("grip:down", Q0, Q1), close], execute=True)
    assert c.events == ["execute", "still", "send"]  # after the motion, deferred: joined at the end
    r.finish()
    assert c.events[-1] == "join"


def test_a_client_that_keeps_no_stream_waits_for_nothing(tmp_path):
    class _NoStream(FakeClient):
        wait_still = None

    r = runner(_NoStream(), tmp_path)
    assert r.wait_still("reach-aim") is None
    assert not list(tmp_path.glob("run-*.jsonl"))  # nothing noted


def test_recoil_follows_a_push_that_ran_its_bound(tmp_path):
    """The arm is on the button whether the push met a stop or arrived at
    its bound: both recoil before the retreat."""
    c = FakeClient()
    g = GuardSpec(touch_nm=4.0, trip="press")
    c.exec_script = [("arrived", {"message": "ok", "progress": 1.0, "torque_peak": 1.2})]
    push = leg("press:push", Q0, Q1, guard=g, world="interaction_button", chain=0)
    push.verify = lambda v: (v.outcome in ("touch", "arrived"), "pressed")  # as press_push's
    res = runner(c, tmp_path).run(
        [push, leg("retreat", Q1, Q2, chain=1, world="interaction_button")],
        execute=True,
    )
    assert [x.leg_name for x in res] == ["press:push", "recoil", "retreat"]


def test_an_approach_and_its_guarded_descent_fly_as_one_goal(tmp_path):
    """The reach and the touch are ONE motion. Two goals meant a controller
    round trip and a full stop mid-reach — the pause a person does not make.
    One re-timed profile flows through the junction, and the torque guard
    arms where the descent begins instead of watching the approach's own
    dynamics swing the wrist."""
    c = FakeClient()
    c.exec_script = [("touch", {"message": "torque guard trip", "progress": 0.8})]
    g = GuardSpec(touch_nm=3.0, trip="touch")  # a merged PRESS tail also recoils: a second goal
    legs = [
        leg("approach", Q0, Q1, chain=0, speed=0.75),
        leg("press:down", Q1, Q2, chain=0, speed=0.15, guard=g),
    ]
    res = runner(c, tmp_path).run(legs, execute=True)
    assert all(r.ok for r in res)
    assert len(c.executed) == 1  # one goal
    assert c.executed[0][1] == 1.0  # the profile is baked in, not dilated
    armed = c.guards[0].arm_after
    assert armed is not None and 0.0 < armed < 1.0
    # the baseline is taken AT the moment the guard becomes able to act, not
    # earlier: a baseline captured at the junction and compared a tenth of a
    # second later tripped a 3 Nm touch threshold on nothing, at exactly the
    # first armed instant (field 2026-09-15, peak 5.9 Nm 87 mm above the
    # button). Taking both at the same point absorbs whatever steady offset
    # the corner left behind.
    assert c.guards[0].rebaseline_after == pytest.approx(armed)


def _line_leg(name, a, b, n=30, **kw):
    """A leg whose trajectory has enough samples to re-time realistically."""
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for k in range(n):
        f = k / (n - 1)
        pt = JointTrajectoryPoint()
        pt.positions = [x + (y - x) * f for x, y in zip(a, b, strict=False)]
        pt.velocities = [0.0] * 7
        pt.accelerations = [0.0] * 7
        pt.time_from_start.sec = int(k * 0.05)
        pt.time_from_start.nanosec = int(((k * 0.05) % 1.0) * 1e9)
        t.points.append(pt)
    lg = leg(name, a, b, **kw)
    lg.traj = t
    return lg


def test_a_merged_guard_waits_a_real_settle_after_the_junction(tmp_path):
    """The settle is a TIME, not a fraction. A fraction of a short trajectory
    is a shorter wait than the same fraction of a long one, and what the arm
    needs after a corner does not scale with how far it is going."""
    from box_opening.runtime.guards import GROUP_SETTLE_S

    c = FakeClient()
    c.exec_script = [("touch", {"message": "trip", "progress": 0.95})]
    g = GuardSpec(touch_nm=3.0, trip="press")
    far = [0.6] * 7
    legs = [
        _line_leg("approach", [0.0] * 7, far, chain=0, speed=0.75),
        _line_leg("press:down", far, [0.75] * 7, chain=0, speed=0.15, guard=g),
    ]
    r = runner(c, tmp_path)
    r.run(legs, execute=True)
    dur = r.last_retime["duration_s"]
    junction = r.last_retime["seg_start_fracs"][1]
    assert 0.0 < junction < 0.9  # the descent really is the tail of the group
    assert c.guards[0].arm_after == pytest.approx(junction + GROUP_SETTLE_S / dur)
    assert c.guards[0].rebaseline_after == pytest.approx(c.guards[0].arm_after)


def test_slow_mode_arms_a_merged_guard_at_the_same_place_along_the_path(tmp_path):
    """Bench 2026-09-21 13:10, --speed-scale 0.5: the chained approach +
    press tripped its 3 Nm touch guard at the first armed instant, 87 mm
    above the button, on nothing. The same motion had found the button four
    runs out of four at full speed (09-16, 09-17), where its guard arms
    0.25 s after the junction: 70 mm above the button. Slow mode stretches
    everything the settle exists to wait out, and the settle stayed 0.25 s
    of wall time — half as far along the path, back in the corner's wake,
    where the 09-15 false trip had been. The operator's slow mode is for
    watching the SAME run slowly: it must arm the guard at the same point
    of the path."""
    g = GuardSpec(touch_nm=3.0, trip="press")
    far = [0.6] * 7
    armed = {}
    for scale in (1.0, 0.5):
        c = FakeClient()
        c.exec_script = [("touch", {"message": "trip", "progress": 0.95})]
        legs = [
            _line_leg("approach", [0.0] * 7, far, chain=0, speed=0.75),
            _line_leg("press:down", far, [0.75] * 7, chain=0, speed=0.15, guard=g),
        ]
        r = runner(c, tmp_path)
        r.time_scale = scale
        r.run(legs, execute=True)
        armed[scale] = (c.guards[0].arm_after, r.last_retime["duration_s"])
    # twice as long (to the 20 ms lead-in before point 0, which is not motion) ...
    assert armed[0.5][1] == pytest.approx(2 * armed[1.0][1], rel=1e-2)
    assert armed[0.5][0] == pytest.approx(armed[1.0][0], abs=2e-3)  # ... armed at the same fraction of it


def test_a_lone_guarded_stroke_keeps_its_own_timing(tmp_path):
    """Nothing in front of it: the stroke flies exactly as planned and
    warped, dilated by its own speed — the contact profile is not the
    re-timer's business."""
    c = FakeClient()
    c.exec_script = [("touch", {"message": "trip", "progress": 0.9})]
    g = GuardSpec(touch_nm=3.0, trip="press")
    legs = [leg("press:down", Q0, Q1, chain=0, speed=0.15, guard=g)]
    runner(c, tmp_path).run(legs, execute=True)
    assert c.executed[0][1] == pytest.approx(0.15)
    assert c.guards[0].arm_after is None


def test_a_merged_stroke_expects_contact_in_the_groups_time_base(tmp_path):
    """press:down's expected-contact fraction is measured along its OWN
    stroke. Flown as the tail of a merged group the execution's progress
    counts the approach too, so the expectation is rescaled to the path
    actually flown — otherwise an honest contact reads as an early strike."""
    from box_opening.runtime.guards import time_fraction_at_path_fraction

    seen = {}
    c = FakeClient()
    c.exec_script = [("touch", {"message": "trip", "progress": 0.9})]
    g = GuardSpec(touch_nm=3.0, trip="press")
    approach = leg("approach", Q0, Q1, chain=0, speed=0.75)
    down = leg("press:down", Q1, Q2, chain=0, speed=0.15, guard=g)
    down.contact_path_frac = 0.8
    down.retime = lambda traj: seen.update(frac=time_fraction_at_path_fraction(traj, down.contact_path_frac))
    runner(c, tmp_path).run([approach, down], execute=True)
    # the approach is half the path, so 0.8 along the stroke is 0.9 along
    # the group — and later in time than 0.8 of it
    assert seen["frac"] > 0.8
    # the leg's own expectation is left as it was: a replan re-times it
    # against its own fresh trajectory before the group is re-scaled again
    assert down.contact_path_frac == 0.8


def _two_part_traj(n_head=11, n_line=31, head_s=1.5, line_s=2.0):
    """A press from staging as it reaches the driver: the planner's part to
    the waypoint (head_s), then the straight line down (line_s)."""
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from box_opening.runtime.stamps import set_stamp

    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, 8)]
    n = n_head + n_line - 1
    for k in range(n):
        p = JointTrajectoryPoint()
        p.positions = [0.004 * k] * 7
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        tt = head_s * k / (n_head - 1) if k < n_head else head_s + line_s * (k - n_head + 1) / (n_line - 1)
        set_stamp(p.time_from_start, tt)
        t.points.append(p)
    return t, n_head - 1


def test_a_standalone_descent_arms_where_its_straight_line_begins(tmp_path):
    """Bench 2026-09-21: a press flown from staging on its own tripped its
    3 Nm touch guard at 40 % of the stroke, three inches above the box, on
    nothing — 3.08 Nm while the arm was still braking into the waypoint.
    The guard armed at a fixed 25 % of the leg, which used to cover the
    whole free-air part only because the line below crawled. A standalone
    descent now arms, and takes its baseline, where its straight line
    begins plus a settle — the rule the merged press always had."""
    from box_opening.runtime.guards import GROUP_SETTLE_S

    c = FakeClient()
    traj_, wp = _two_part_traj()  # the line begins at 1.5 s of 3.5 s = 0.4286
    g = GuardSpec(touch_nm=3.0, trip="touch", arm_after=0.25)
    press = leg("press:down", guard=g, world="interaction_button", speed=1.0)
    press.traj = traj_
    press.goal_joints = list(traj_.points[-1].positions)
    press.guard_from = wp
    c.live = list(traj_.points[0].positions)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.88, "torque_peak": 3.2})]
    runner(c, tmp_path).run([press], execute=True)
    guard = c.guards[-1]
    at = 1.5 / 3.5
    assert guard.arm_after == pytest.approx(at + GROUP_SETTLE_S / 3.5, abs=1e-6)
    assert guard.rebaseline_after == pytest.approx(guard.arm_after)  # baseline AT the arming point
    # today's failure, replayed against that guard
    guard.on_progress(0.01)
    assert guard.on_efforts([1.0, 1.0, 1.0, 1.0]) is False  # first reading: baseline
    guard.on_progress(0.404)
    assert guard.on_efforts([4.08, 1.0, 1.0, 1.0]) is False  # braking into the waypoint: not a touch
    guard.on_progress(guard.arm_after + 0.01)
    assert guard.on_efforts([2.0, 1.0, 1.0, 1.0]) is False  # the baseline is re-taken here
    assert guard.on_efforts([2.5, 1.0, 1.0, 1.0]) is False
    assert guard.on_efforts([5.2, 1.0, 1.0, 1.0]) is True  # THIS is a touch


def test_slow_mode_arms_a_standalone_descent_at_the_same_place_along_the_path(tmp_path):
    """The settle is wall time AT FULL OPERATOR SPEED: --speed-scale dilates
    it with the motion, so the guard arms at the same point of the path
    however slowly the run is watched. (This test first asserted the
    opposite — "the same 0.25 s is half the fraction" — and the chained
    press false-tripped in slow mode the same day, 2026-09-21.)"""
    armed = {}
    for scale in (1.0, 0.5):
        c = FakeClient()
        traj_, wp = _two_part_traj()
        # 3.5 s at full operator speed: the 0.25 s settle (7 % of it) is
        # what arms the guard, not the 5 % floor
        press = leg(
            "press:down",
            guard=GuardSpec(touch_nm=3.0, trip="touch", arm_after=0.25),
            world="interaction_button",
            speed=1.0,
        )
        press.traj, press.guard_from = traj_, wp
        press.goal_joints = list(traj_.points[-1].positions)
        c.live = list(traj_.points[0].positions)
        c.exec_script = [("touch", {"message": "contact", "progress": 0.88})]
        r = runner(c, tmp_path)
        r.time_scale = scale
        r.run([press], execute=True)
        armed[scale] = c.guards[-1].arm_after
    from box_opening.runtime.guards import GROUP_SETTLE_S

    assert armed[1.0] == pytest.approx(
        1.5 / 3.5 + GROUP_SETTLE_S / 3.5, abs=1e-6
    )  # 3.5 s leg: the settle, not the floor
    assert armed[0.5] == pytest.approx(armed[1.0], abs=1e-6)


def test_a_replan_refreshes_where_the_descent_begins(tmp_path):
    """A replanned leg flies a NEW trajectory: an index into the old one
    would arm the guard at an arbitrary point of it."""
    c = FakeClient()
    c.live = [0.06] + [0.0] * 6  # drifted: forces the replan
    press = leg("press:down", guard=GuardSpec(touch_nm=3.0, trip="touch", arm_after=0.25), world="interaction_button")
    press.guard_from = 7
    press.waypoint = (7, [0.0] * 7)
    c.exec_script = [("touch", {"message": "contact", "progress": 0.9})]
    runner(c, tmp_path).run([press], execute=True)
    assert press.guard_from is None and press.waypoint is None  # the fake plan carries no waypoint


class _NoPlan:
    success = False
    message = "no plan"
    trajectory = None


def test_press_push_ran_its_bound_then_recoil_but_lookahead_failed_must_replan(tmp_path):
    """The push arrives at its bound, the recoil flies back along it and the
    retreat's re-plan hosted under the recoil FAILS. The retreat was
    planned from the PUSH END; the arm stands at the recoil's end, under
    DRIFT_REPLAN_RAD and the driver's start gate from it — sent as-is, the
    driver's first waypoint puts the arm back down on the button, unguarded.
    Any recoil that moved the arm replans what its hook did not."""
    c = FakeClient()
    push_end = [0.0, 0.03, 0.0, 0.03, 0.0, 0.0, 0.0]  # ~0.042 rad: a 10 mm push
    hop = [0.0, -0.1, 0.0, -0.1, 0.0, 0.0, 0.0]
    g = GuardSpec(touch_nm=4.0, trip="press")
    push = leg("press:push", Q0, push_end, guard=g, world="interaction_button", chain=0)
    push.verify = lambda v: (v.outcome in ("touch", "arrived"), "pressed")
    retreat = leg("retreat", push_end, hop, chain=1, world="interaction_button")
    c.exec_script = [("arrived", {"message": "ok", "progress": 1.0, "torque_peak": 1.2})]
    c.plans = [_NoPlan()]  # the replan hosted under the recoil fails
    res = runner(c, tmp_path).run([push, retreat], execute=True)
    assert [x.leg_name for x in res] == ["press:push", "recoil", "retreat"]
    # the recoil walked the push back to its start: the retreat flies from there
    assert c.exec_starts[2] == pytest.approx(Q0, abs=1e-6)


def test_a_press_trip_merged_as_a_groups_tail_still_holds_and_recoils(tmp_path, monkeypatch):
    """can_merge lets a guarded leg END a group and _run_motion flies it with
    the guarded member's guard — the hold and the recoil must follow that
    same owner, not group[0] (the unguarded approach), and reverse the path
    actually flown (the merged profile progress counts in)."""
    from box_opening.runtime import runner as runner_mod
    from box_opening.runtime.legs import merge_groups

    held = []
    monkeypatch.setattr(runner_mod, "_hold", lambda s: held.append(s))
    c = FakeClient()
    c.exec_script = [("touch", {"message": "torque guard trip", "progress": 0.8})]
    g = GuardSpec(touch_nm=3.0, trip="press")
    approach = leg("approach", Q0, Q1, chain=0, speed=0.75)
    press = leg("press:push", Q1, Q2, chain=0, speed=0.15, guard=g)
    press.hold_s = 0.3
    assert len(merge_groups([approach, press])) == 1
    res = runner(c, tmp_path).run([approach, press], execute=True)
    assert held == [0.3]
    assert [r.leg_name for r in res] == ["press:push", "recoil"]
    assert res[0].flown_traj is not None and len(res[0].flown_traj.points) > 2  # the merged profile, not press.traj


class _StopsWhereItTripped(FakeClient):
    """Executions keep their trajectories; a 'touch' leaves the arm at the
    scripted progress (in time) instead of at the goal."""

    def __init__(self):
        super().__init__()
        self.trajs = []

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None):
        from box_opening.runtime.stamps import secs

        self.trajs.append(traj_)
        outcome, info = super().execute(traj_, speed, guard=guard, while_running=while_running, stop_when=stop_when)
        if outcome == "touch":
            cut = secs(traj_.points[-1].time_from_start) * float(info["progress"])
            pts = [p for p in traj_.points if secs(p.time_from_start) <= cut]
            self.live = list(pts[-1].positions)
        return outcome, info


DIR = [0.0, 0.6, 0.0, 0.8, 0.0, 0.0, 0.0]  # unit length
TOP = [0.0, 0.3, 0.0, 1.2, 0.0, 0.5, 0.0]


def _down(a, n, step=0.002, offset=0.0):
    """n joint samples `step` rad apart along DIR from `a`: a straight
    joint-space stroke (a touch, and the push continuing it)."""
    return [
        [x + d * step * k + (offset if j == 0 else 0.0) for j, (x, d) in enumerate(zip(a, DIR, strict=False))]
        for k in range(n)
    ]


def _touch_and_push(progress, via_offset=0.0):
    from box_opening.runtime import approach

    names = ["joint_%d" % i for i in range(1, 8)]
    stop = _down(TOP, 61)[-1]  # the touch stopped 0.12 rad down its 0.2 rad stroke
    g = GuardSpec(touch_nm=7.0, trip="press")
    push = leg("press:push", stop, _down(stop, 26)[-1], guard=g, world="interaction_button", speed=1.0)
    push.traj = approach.line_trajectory(names, _down(stop, 26))  # 0.05 rad: ~12 mm
    push.plan_free = True
    push.recoil_via = approach.line_trajectory(names, _down(TOP, 101, offset=via_offset))
    c = _StopsWhereItTripped()
    c.live = list(stop)
    c.exec_script = [("touch", {"message": "stop met", "progress": progress, "torque_peak": 7.2})]
    return c, push, stop


@pytest.mark.parametrize("progress", [1.0, 0.6445])
def test_the_recoil_carries_on_back_up_the_touch_to_its_full_arc(tmp_path, progress):
    """A push is 12 mm and starts where the touch stopped: reversed alone it
    ends ON the touch's contact pose (run 20:04: 0.23 s back to where 6.1 Nm
    was read). The recoil carries on up the touch stroke to RECOIL_ARC_RAD."""
    import numpy as np

    from box_opening.constants import RECOIL_ARC_RAD

    c, push, stop = _touch_and_push(progress)
    res = runner(c, tmp_path).run([push], execute=True)
    assert [r.leg_name for r in res] == ["press:push", "recoil"]
    q = np.asarray([p.positions for p in c.trajs[-1].points])
    arc = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    assert arc == pytest.approx(RECOIL_ARC_RAD, abs=1e-6)
    # it ends ON the touch stroke, above where the touch stopped
    end = q[-1] - np.asarray(TOP)
    along = float(np.dot(end, DIR))
    assert np.linalg.norm(end - along * np.asarray(DIR)) < 1e-9
    pushed = float(np.dot(q[0] - np.asarray(stop), DIR))
    assert along == pytest.approx(0.12 + pushed - RECOIL_ARC_RAD, abs=1e-6) and along < 0.12
    # ... never a step back down on the way
    s = (q - np.asarray(TOP)) @ np.asarray(DIR)
    assert (np.diff(s) <= 1e-12).all()


def test_the_recoil_does_not_extend_along_a_stroke_this_push_did_not_continue(tmp_path):
    import numpy as np

    from box_opening.constants import START_GATE_RAD

    c, push, stop = _touch_and_push(1.0, via_offset=2 * START_GATE_RAD)
    runner(c, tmp_path).run([push], execute=True)
    q = np.asarray([p.positions for p in c.trajs[-1].points])
    assert q[-1] == pytest.approx(stop, abs=1e-6)  # the push reversed, nothing more


# -- what a run re-times with: a compliant one gentler and under the wrist's ceiling --


class _Flown(FakeClient):
    """Keeps every trajectory sent as the driver gets it. `compliant`: the
    client's flag (None: it has none, as a position client may not)."""

    def __init__(self, compliant):
        super().__init__()
        if compliant is not None:
            self.compliant = compliant
        self.flown = []

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None, gains=None):
        self.flown.append((traj_, speed))
        return super().execute(traj_, speed, guard, while_running, stop_when, gains)


def _wrist_home(n=40):
    """A transit at full speed whose wrist moves most: joint_6 1 rad."""
    return _line_leg("home", [0.0] * 7, [0.1, 0.8, 0.3, 0.5, 0.2, 1.0, 0.1], n=n, speed=1.0)


def _flown_s(traj_):
    from box_opening.runtime.stamps import secs

    return secs(traj_.points[-1].time_from_start)


def _wrist_peaks(traj_):
    """joints 5-7's fastest interval each, as flown (rad/s)."""
    import numpy as np

    from box_opening.runtime.stamps import secs

    q = np.asarray([p.positions for p in traj_.points], float)
    t = np.asarray([secs(p.time_from_start) for p in traj_.points])
    return (np.abs(np.diff(q, axis=0))[:, 4:] / np.diff(t)[:, None]).max(axis=0)


def test_a_compliant_run_retimes_gentler_and_under_the_wrist_ceiling_in_preview_and_flight(tmp_path):
    """15:45 2026-10-28, RIGID at zeta 0.7: out of rest the look asked
    joint_2 for ~4.2 rad/s^2 and the arm made ~3; on the way home joint_6
    topped out at 0.54 rad/s, its command at the clamp, where the re-timer
    allowed 1.10. A compliant client's runner re-times every unguarded
    motion at COMPLIANT_ACCEL / COMPLIANT_DECEL under COMPLIANT_QD_CEILING,
    its junctions no faster than COMPLIANT_JUNCTION_SPEED, and its preview
    shows the time that flies. In the mission's order: press_demo
    builds the runner on a position client, run_reach_press sets the flag
    on it and puts it back when the run ends — the runner reads it at every
    re-timing, so a flag kept from when it was built would fly a compliant
    run on the position profile."""
    import numpy as np

    from box_opening.constants import (
        COMPLIANT_ACCEL,
        COMPLIANT_DECEL,
        COMPLIANT_JUNCTION_SPEED,
        COMPLIANT_QD_CEILING,
        JOINT_VMAX,
    )
    from box_opening.runtime.retime import RetimeParams, retime_group

    c = _Flown(compliant=False)
    r = runner(c, tmp_path)
    assert r.retime_params() is r.retime
    c.compliant = True  # run_reach_press, on the runner press_demo built
    p = r.retime_params()
    assert (p.accel, p.decel, p.qd_ceiling, p.junction_cruise) == (
        COMPLIANT_ACCEL,
        COMPLIANT_DECEL,
        COMPLIANT_QD_CEILING,
        COMPLIANT_JUNCTION_SPEED,
    )
    # gentler than the position profile, and a wrist ceiling under the joint cap, so it binds below
    assert COMPLIANT_ACCEL < RetimeParams().accel and COMPLIANT_DECEL < RetimeParams().decel
    assert all(top < 0.9 * vmax for top, vmax in zip(COMPLIANT_QD_CEILING[4:], JOINT_VMAX[4:], strict=False))
    assert r.retime == RetimeParams()  # the runner's own profile is not changed by it
    home = _wrist_home()
    gentle = RetimeParams(accel=COMPLIANT_ACCEL, decel=COMPLIANT_DECEL, qd_ceiling=COMPLIANT_QD_CEILING)
    want = retime_group([home.traj], [1.0], JOINT_VMAX, gentle)[1]["duration_s"]
    table = r.preview([home])
    res = r.run([home], execute=True)
    assert res[-1].ok
    flown, speed = c.flown[-1]
    assert speed == 1.0 and _flown_s(flown) == pytest.approx(want, abs=1e-9)
    assert r.last_retime["duration_s"] == pytest.approx(want, abs=1e-9)
    # whole-nanosecond stamps: an interval at the ceiling reads a few parts in 1e8 over it
    assert (_wrist_peaks(flown) <= np.asarray(COMPLIANT_QD_CEILING[4:]) * (1 + 1e-6)).all()
    row = [line for line in table.splitlines() if line.startswith("home")][0]
    assert float(row.split()[5]) == pytest.approx(want, abs=0.006)
    # slow mode dilates the compliant profile as it dilates any other
    r.time_scale = 0.5
    c.live = [0.0] * 7
    r.run([_wrist_home()], execute=True)
    assert _flown_s(c.flown[-1][0]) == pytest.approx(2 * (want - 0.02) + 0.02, rel=1e-6)
    # the run over, its flag put back: the same runner flies the position profile again
    r.time_scale = 1.0
    c.compliant = False
    assert r.retime_params() is r.retime
    c.live = [0.0] * 7
    r.run([_wrist_home()], execute=True)
    plain = retime_group([home.traj], [1.0], JOINT_VMAX, RetimeParams())[1]["duration_s"]
    assert plain < want - 0.1 and _flown_s(c.flown[-1][0]) == pytest.approx(plain, abs=1e-9)


@pytest.mark.parametrize("compliant", [None, False])
def test_a_position_or_guarded_run_keeps_the_default_profile_and_no_ceiling(tmp_path, compliant):
    """A guarded reach-out flies in position (client.compliant False), as
    home_arm and every other task does: its transits and its merged guarded
    strokes keep RetimeParams' ease out 6 and in 3 and no wrist ceiling —
    the arm follows them there (15:41:58's position home: joint_6 at 0.738
    rad/s, 0.7 mm off on arrival)."""
    from box_opening.constants import JOINT_VMAX
    from box_opening.runtime.retime import RetimeParams, retime_group

    c = _Flown(compliant)
    r = runner(c, tmp_path)
    assert r.retime_params() is r.retime and r.retime == RetimeParams()
    home = _wrist_home()
    want = retime_group([home.traj], [1.0], JOINT_VMAX, RetimeParams())[1]["duration_s"]
    r.run([home], execute=True)
    flown, _speed = c.flown[-1]
    assert _flown_s(flown) == pytest.approx(want, abs=1e-9)
    assert _wrist_peaks(flown).max() > 0.8  # no ceiling: the 0.9 x 1.222 joint cap
    g = GuardSpec(touch_nm=3.0, trip="press")
    far = [0.6] * 7
    legs = [
        _line_leg("approach", [0.0] * 7, far, chain=0, speed=0.75),
        _line_leg("press:down", far, [0.75] * 7, chain=0, speed=0.15, guard=g),
    ]
    c.live = [0.0] * 7
    c.exec_script = [("touch", {"message": "trip", "progress": 0.95})]
    r.run(legs, execute=True)
    want = retime_group([lg.traj for lg in legs], [0.75, 0.15], JOINT_VMAX, RetimeParams())[1]["duration_s"]
    assert r.last_retime["duration_s"] == pytest.approx(want, abs=1e-9)


# -- an arrival's error: what the spring had not caught up, logged, never judged --


class _ArrivesOff(FakeClient):
    """A client whose arm ends `off` short of the trajectory's end on one
    joint (or reads `to` there): an impedance goal ends on time, its spring
    still converging. Compliant, it reports the gains it would send, as
    PlannerClient.execute does; in position it sends none."""

    def __init__(self, joint=1, off=0.01, to=None, compliant=True):
        super().__init__()
        self.compliant = compliant
        self._joint, self._off, self._to = joint, off, to

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None, gains=None):
        outcome, info = super().execute(traj_, speed, guard, while_running, stop_when, gains)
        if outcome == "arrived":
            self.live[self._joint] = self._to if self._to is not None else self.live[self._joint] - self._off
        info["gains"] = (3 if gains is None else gains) if self.compliant else None
        return outcome, info


class _TwoLink:
    """The arm's FK stood in for by one periodic in every joint, as the real
    one is: the tool 0.5 m out along joints 1 and 2, turned about z by
    joints 1 and 3 and tipped by joint 2."""

    def fk(self, q):
        import numpy as np

        a, b = q[0] + q[2], q[1]
        rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)], [0.0, np.sin(b), np.cos(b)]])
        t = 0.5 * np.array([np.cos(q[0]) * np.cos(b), np.sin(q[0]) * np.cos(b), np.sin(b)])
        return rz @ rx, t


def _rows(tmp_path):
    import json

    return {r["leg"]: r for r in map(json.loads, next(tmp_path.glob("run-*.jsonl")).read_text().splitlines())}


def test_an_arrival_logs_how_far_the_arm_stood_from_the_end_of_what_flew(tmp_path, monkeypatch):
    """2026-10-27 16:23, MEDIUM: TF put the tool 32 mm and 5.3 deg off its
    commanded look pose. Every arrival now says how far off it stood, so the
    log shows what STIFF buys. A merged group logs its one end for every
    member, as it logs its one t_wall."""
    import math

    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _TwoLink())
    c = _ArrivesOff(joint=1, off=0.01)
    res = runner(c, tmp_path).run([leg("a", Q0, Q1), leg("b", Q1, Q2)], execute=True)
    assert len(c.executed) == 1 and res[0].ok
    rows = _rows(tmp_path)
    for name in ("a", "b"):
        row = rows[name]
        assert row["end_err_rad"] == pytest.approx(0.01, abs=1e-9), name
        # 0.01 rad of joint 2 tips the 0.5 m tool 5 mm and turns it 0.57 deg
        assert row["end_err_mm"] == pytest.approx(5.0, rel=1e-3)
        assert row["end_err_deg"] == pytest.approx(math.degrees(0.01), rel=1e-6)
        assert row["mode"] == "impedance" and row["gains"] == 3


def test_an_arrival_with_no_chain_logs_the_joints_alone_and_the_run_carries_on(tmp_path, monkeypatch):
    from box_opening.runtime import approach

    def no_chain():
        raise RuntimeError("no arm URDF rooted at arm_mount (is /arm/robot_description published?)")

    monkeypatch.setattr(approach, "_chain", no_chain)
    c = _ArrivesOff(joint=1, off=0.01)
    res = runner(c, tmp_path).run([leg("a", Q0, Q1, chain=0), leg("b", Q1, Q2, chain=1)], execute=True)
    assert [r.ok for r in res] == [True, True] and len(c.executed) == 2
    row = _rows(tmp_path)["a"]
    assert row["end_err_rad"] == pytest.approx(0.01, abs=1e-9)
    assert row["end_err_mm"] is None and row["end_err_deg"] is None


def test_a_failed_execution_logs_no_arrival_error(tmp_path, monkeypatch):
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _TwoLink())
    c = _ArrivesOff()
    c.exec_script = [("failed", {"message": "halted: emergency stop", "progress": 0.4})]
    res = runner(c, tmp_path).run([leg("a", Q0, Q1)], execute=True)
    assert res[0].outcome == "failed"
    row = _rows(tmp_path)["a"]
    assert row["end_err_rad"] is None and row["end_err_mm"] is None and row["end_err_deg"] is None
    assert row["mode"] == "impedance" and row["gains"] == 3  # how it flew, still


def test_a_custom_specs_zeta_is_in_the_record_and_its_side_file_beside_its_name(tmp_path, monkeypatch):
    """RIGID flew at zeta 0.7 from 14:21 to 16:23 2026-10-28: a run under
    the same name at another zeta is told apart in the record by its zeta,
    not its name. A driver profile's zeta is the driver's own: None."""
    import json

    from box_opening.constants import CustomGains
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _TwoLink())
    spec = CustomGains("RIGID", (400.0,) * 4 + (200.0,) * 3, 1.0, (39.0,) * 4 + (9.0,) * 3)

    class Sends(_ArrivesOff):
        def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None, gains=None):
            outcome, info = super().execute(traj_, speed, guard, while_running, stop_when, gains)
            info.update(gains=spec, sent_at=time.monotonic(), sent_traj=traj_)
            return outcome, info

    res = runner(Sends(), tmp_path).run([leg("a", Q0, Q1)], execute=True)
    assert res[0].ok and (res[0].gains, res[0].gains_zeta) == ("RIGID", 1.0)
    assert (_rows(tmp_path)["a"]["gains"], _rows(tmp_path)["a"]["gains_zeta"]) == ("RIGID", 1.0)
    doc = json.loads((tmp_path / res[0].flight).read_text())
    assert (doc["gains"], doc["gains_zeta"]) == ("RIGID", 1.0)
    res = runner(_ArrivesOff(), tmp_path / "profile").run([leg("a", Q0, Q1)], execute=True)
    assert (res[0].gains, res[0].gains_zeta) == (3, None)


def test_an_arrival_across_the_wrap_is_the_short_way_round(tmp_path, monkeypatch):
    """HOME's joint_3 sits 4.6 mrad from -pi and reads either side of it: an
    end at -3.137 with the arm reading +3.140 is 6 mrad off, not 6.28 rad."""
    import math

    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _TwoLink())
    start = [0.0, 0.0, -3.10, 0.0, 0.0, 0.0, 0.0]
    end = [0.0, 0.0, -3.137, 0.0, 0.0, 0.0, 0.0]
    c = _ArrivesOff(joint=2, to=3.140)
    c.live = list(start)
    res = runner(c, tmp_path).run([leg("a", start, end)], execute=True)
    assert res[0].ok
    err = res[0].end_err_rad
    assert err == pytest.approx(2 * math.pi - 3.140 - 3.137, abs=1e-9) and err < 0.01
    assert res[0].end_err_deg == pytest.approx(math.degrees(err), rel=1e-6)
    assert res[0].end_err_mm == pytest.approx(0.0, abs=1e-9)


def test_a_position_arrival_logs_its_mode_and_no_gains(tmp_path, monkeypatch):
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _TwoLink())
    c = FakeClient()  # no compliant flag: a position run, its arm on the end
    runner(c, tmp_path).run([leg("a")], execute=True)
    row = _rows(tmp_path)["a"]
    assert row["mode"] == "position" and row["gains"] is None and row["end_err_rad"] == 0.0
    assert row["end_err_mm"] == pytest.approx(0.0, abs=1e-9) and row["end_err_deg"] == pytest.approx(0.0, abs=1e-6)


def test_the_recoil_logs_how_it_flew_and_where_it_ended_like_any_leg(tmp_path, monkeypatch):
    """The recoil is a row of its own in the run log: it says how it flew
    and how far from its end it arrived, as every motion does. The touch it
    backs off is no arrival, so it logs no end error."""
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _TwoLink())
    c = _ArrivesOff(joint=1, off=0.01, compliant=False)  # a guarded run: position
    c.exec_script = [("touch", {"message": "contact", "progress": 0.8, "torque_peak": 7.1})]
    res = runner(c, tmp_path).run([_press_leg(chain=0)], execute=True)
    assert [r.leg_name for r in res] == ["press:down", "recoil"] and all(r.ok for r in res)
    rows = _rows(tmp_path)
    recoil, press = rows["recoil"], rows["press:down"]
    assert recoil["mode"] == "position" and recoil["gains"] is None
    assert recoil["end_err_rad"] == pytest.approx(0.01, abs=1e-9)
    assert recoil["end_err_mm"] == pytest.approx(5.0, rel=1e-3)
    assert press["mode"] == "position" and press["end_err_rad"] is None and press["end_err_mm"] is None


# -- the flight log: each joint's signed arrival, the settle after it, what flew --


def test_an_arrival_signs_each_joints_error_by_its_travel_into_the_end(tmp_path, monkeypatch):
    """15:45 2026-10-28: the look arrived with joint_2 45 mrad PAST its end,
    the home with joint_6 253 mrad SHORT, and end_err_rad read both as one
    unsigned number. Each joint now has its own: + past the end along the
    way it travelled into it, - short of it, whichever way that is; and the
    worst is named."""
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_chain", lambda: _TwoLink())
    cases = [
        (Q0, Q1, 0.11, +10.0),  # up to 0.1, the arm on at 0.11: past
        (Q0, Q1, 0.09, -10.0),  # ... at 0.09: short
        (Q1, Q0, -0.01, +10.0),  # down to 0.0, the arm on at -0.01: past
        (Q1, Q0, 0.01, -10.0),  # ... at 0.01: short
    ]
    for i, (a, b, to, want) in enumerate(cases):
        d = tmp_path / str(i)
        c = _ArrivesOff(joint=1, to=to)
        c.live = list(a)
        res = runner(c, d).run([leg("a", a, b)], execute=True)
        assert res[0].end_err_mrad[1] == pytest.approx(want, abs=1e-9), (a, b, to)
        assert res[0].end_err_mrad[:1] + res[0].end_err_mrad[2:] == [0.0] * 6
        assert res[0].end_err_joint == "joint_2"
        row = _rows(d)["a"]
        assert row["end_err_mrad"] == res[0].end_err_mrad and row["end_err_joint"] == "joint_2"
        assert row["end_err_rad"] == pytest.approx(0.01, abs=1e-9)  # the unsigned worst, as before


def test_a_joint_is_signed_by_where_it_last_travelled_not_by_its_noise():
    """A joint that finished moving early holds to float noise (cuRobo's
    float32 samples); its last real travel into the end signs it. One that
    never moves reads along its own axis. One whose travel turned back
    before the end (a cuRobo path may) is signed by the way it came in,
    not by where it set out from."""
    from box_opening.runtime import runner as runner_mod

    rows = [
        [0.0, 0.5, 0.0, 0.0],
        [0.1, 0.3, 0.0, 0.3],
        [0.2, 0.2, 0.0, 0.2],
        [0.2 + 3e-7, 0.2, 0.0, 0.2],
        [0.2 - 2e-7, 0.2, 0.0, 0.2],
    ]
    t = traj(Q0, Q1)
    t.points = []
    for row in rows:
        p = JointTrajectoryPoint()
        p.positions = row
        t.points.append(p)
    assert list(runner_mod._travel_signs(t)) == [1.0, -1.0, 1.0, -1.0]
    # joint_4 came in down from 0.3: at 0.19 it is past its 0.2, though it set out from 0.0 below it
    mrad, worst = runner_mod._signed_end_error([0.21, 0.19, -0.004, 0.19], t)
    assert mrad == pytest.approx([10.0, 10.0, -4.0, 10.0], abs=0.05) and worst in ("joint_1", "joint_2", "joint_4")
    assert runner_mod._signed_end_error(None, t) == (None, None)


class _Clock:
    """runtime/runner's `time`, moved by hand: the settle note's window and
    the arrival are measured on it, the driver's stamps on its time()."""

    strftime = staticmethod(time.strftime)  # the record's name (bound before time() below hides the module)

    def __init__(self):
        self.mono = 100.0
        self.wall_off = 1793216726.0 - self.mono

    def monotonic(self):
        return self.mono

    def time(self):
        return self.mono + self.wall_off

    def sleep(self, s):
        self.mono += float(s)


# the look of 15:45 2026-10-28 in shape: the tool 14 mm off its end at the
# arrival, joint_2 past it, creeping back — out above 2 mm once more at
# +0.7 s — and in to 0.5 mm; (s after the arrival, mm off). The last is
# outside the 1.5 s window
CREEP = [
    (0.0, 14.0),
    (0.1, 12.0),
    (0.2, 8.0),
    (0.3, 6.0),
    (0.4, 4.0),
    (0.5, 3.0),
    (0.6, 1.5),
    (0.7, 2.5),
    (0.8, 1.8),
    (0.9, 0.9),
    (1.0, 0.6),
    (1.2, 0.5),
    (1.4, 0.5),
    (1.6, 0.5),
]


class _Creeps(FakeClient):
    """A compliant client whose every motion takes 1 s on `clock` and
    arrives as CREEP says: joint_2 past the end by what puts _TwoLink's tool
    `mm` off it, its /joint_states stamped on the driver's clock (`clock`'s
    time()) and heard 5 ms later — and only once the clock has passed
    them, as a real stream is. One more, measured 20 ms before the arrival
    30 mm off, is heard just after it, as the subscription's queue hands
    on what waited (SensorDataQoS keeps 5): by its stamp it is the motion's,
    not the settle's. goals_sent: every execution's send, and any a test
    adds."""

    def __init__(self, clock, script=CREEP):
        super().__init__()
        self.compliant = True
        self.clock, self.script = clock, script
        self.heard, self.sent = [], []

    @staticmethod
    def past(end, mm):
        q = list(end)
        q[1] += 2.0 * math.asin(mm / 1000.0)  # _TwoLink's tool turns 0.5 m about joint_2: a chord
        return q

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None, gains=None):
        from box_opening.runtime.client import JointSample

        sent_at = self.clock.mono
        self.sent.append(sent_at)
        self.clock.mono += 1.0
        outcome, info = super().execute(traj_, speed, guard, while_running, stop_when, gains)
        end = list(traj_.points[-1].positions)
        t0, w0 = self.clock.monotonic(), self.clock.time()
        self.live = self.past(end, self.script[0][1])
        qd = [0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.heard.append(JointSample(t0 + 0.002, w0 - 0.02, self.past(end, 30.0), [0.0, 0.3] + [0.0] * 5, None))
        self.heard += [JointSample(t0 + s + 0.005, w0 + s, self.past(end, mm), qd, None) for s, mm in self.script[1:]]
        info.update(gains=3, sent_at=sent_at, sent_traj=traj_, feedback_q=[(sent_at + 0.5, 0.5, list(end))])
        return outcome, info

    def joint_history(self, since=None):
        return [s for s in self.heard if s.t <= self.clock.mono and (since is None or s.t >= since)]

    def goals_sent(self):
        return list(self.sent)


def _creep(monkeypatch, tmp_path, script=CREEP):
    from box_opening.runtime import approach
    from box_opening.runtime import runner as runner_mod

    clock = _Clock()
    monkeypatch.setattr(runner_mod, "time", clock)
    monkeypatch.setattr(approach, "_chain", lambda: _TwoLink())
    c = _Creeps(clock, script)
    return clock, c, runner(c, tmp_path)


def _settles(tmp_path):
    import json

    return [
        r for r in map(json.loads, next(tmp_path.glob("run-*.jsonl")).read_text().splitlines()) if r["leg"] == "settle"
    ]


def test_an_arrivals_settle_is_noted_when_the_next_goal_goes_out(tmp_path, monkeypatch):
    """The look of 15:45 arrived 14 mm off and was within 2 mm only 0.55 s
    later; the record had the 14 mm. The settle is measured from the
    /joint_states the client kept, placed by their stamps, over the 1.5 s
    after the arrival, and noted when the next goal goes out — nothing
    waits for it: when the tool came within 5, 2 and 1 mm FOR GOOD (where
    the line between the last reading outside and the next meets each),
    how far each joint went past its end, how fast it moved, how far the
    tool strayed and where it was left."""
    clock, c, r = _creep(monkeypatch, tmp_path)
    res = r.run([leg("a", Q0, Q1)], execute=True)
    assert res[0].ok and res[0].end_err_mm == pytest.approx(14.0, abs=1e-6)
    assert _settles(tmp_path) == []  # nothing noted at the arrival
    clock.mono += 2.0  # the aim, the planning: past the window
    r.run([leg("b", Q1, Q2)], execute=True)
    (note,) = _settles(tmp_path)
    assert note["legs"] == ["a"] and note["seq"] == 1 and note["cut"] is None and note["window_s"] == 1.5
    assert note["arrived_t"] == pytest.approx(1793216727.0, abs=1e-3)
    # 1.0 to 1.2 s; +1.6 s is outside, and the reading measured before the arrival is not the settle's
    assert note["samples"] == 12 and note["last_s"] == 1.4 and note["gap_ms"] == pytest.approx(200.0)
    assert note["within_5mm_s"] == pytest.approx(0.3 + 0.1 * (6.0 - 5.0) / (6.0 - 4.0), abs=1e-3)
    # out above 2 mm again at +0.7 s: within it for good only after that
    assert note["within_2mm_s"] == pytest.approx(0.7 + 0.1 * (2.5 - 2.0) / (2.5 - 1.8), abs=1e-3)
    assert note["within_1mm_s"] == pytest.approx(0.8 + 0.1 * (1.8 - 1.0) / (1.8 - 0.9), abs=1e-3)
    assert note["max_mm"] == 14.0 and note["end_mm"] == 0.5
    past = note["past_mrad"]
    assert past[1] == pytest.approx(2000.0 * math.asin(0.014), abs=0.05) and past[1] > 0  # joint_2: past its end
    assert past[:1] + past[2:] == [0.0] * 6
    assert note["peak_qd"] == [0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_a_settle_window_ends_where_the_next_goal_begins(tmp_path, monkeypatch):
    """What the arm does once the next goal is out is that goal's: the
    window is cut at the runner's next execution, about to go out, or at
    any goal the client sent before it (the hold that primes the
    impedance, a release), and says which. Within 1 mm is not reached
    inside 0.65 s: None, not a guess."""
    clock, c, r = _creep(monkeypatch, tmp_path)
    r.run([leg("a", Q0, Q1)], execute=True)
    clock.mono += 0.65
    r.run([leg("b", Q1, Q2)], execute=True)
    (note,) = _settles(tmp_path)
    assert note["cut"] == "next execution" and note["window_s"] == pytest.approx(0.65)
    assert note["samples"] == 6 and note["last_s"] == 0.6 and note["end_mm"] == 1.5
    assert note["within_5mm_s"] == pytest.approx(0.35, abs=1e-3)
    assert note["within_2mm_s"] == pytest.approx(0.5 + 0.1 * (3.0 - 2.0) / (3.0 - 1.5), abs=1e-3)
    assert note["within_1mm_s"] is None
    # a release 0.35 s after b's arrival cuts b's window there
    clock.mono += 0.35
    c.sent.append(clock.mono)
    clock.mono += 1.0
    r.run([leg("c", Q2, Q1)], execute=True)
    note = _settles(tmp_path)[-1]
    assert note["legs"] == ["b"] and note["cut"] == "next goal" and note["window_s"] == pytest.approx(0.35)
    assert note["samples"] == 3 and note["within_5mm_s"] is None and note["end_mm"] == 6.0


def test_the_last_arrivals_settle_is_noted_at_the_run_end_and_only_once(tmp_path, monkeypatch):
    clock, c, r = _creep(monkeypatch, tmp_path)
    r.run([leg("a", Q0, Q1)], execute=True)
    clock.mono += 0.25
    r.flush()
    r.flush()
    (note,) = _settles(tmp_path)
    assert note["cut"] == "run end" and note["window_s"] == pytest.approx(0.25) and note["samples"] == 2
    assert note["last_s"] == 0.2 and note["within_5mm_s"] is None and note["max_mm"] == 14.0 and note["end_mm"] == 8.0


def test_a_settle_nothing_was_heard_after_claims_nothing(tmp_path, monkeypatch):
    """Nothing spins the node after a run's last motion, the home: its note
    at the run's end has the arrival's own reading alone, however long the
    exit took. It says so — cut where the run ended, no reading after the
    arrival — and claims neither where the tool went nor that it stayed
    within anything: None, never a 0.0 read off the arrival alone."""
    clock, c, r = _creep(monkeypatch, tmp_path, script=[(0.0, 0.8)])  # in at 0.8 mm, then nothing heard
    r.run([leg("home", Q0, Q1)], execute=True)
    clock.mono += 3.0  # the exit: the perception thread's join, its node destroyed
    r.flush()
    (note,) = _settles(tmp_path)
    assert note["cut"] == "run end" and note["window_s"] == 1.5 and note["samples"] == 0 and note["last_s"] == 0.0
    assert note["gap_ms"] == 1500.0 and note["peak_qd"] is None and note["max_mm"] == 0.8
    assert note["end_mm"] is None
    assert [note["within_%gmm_s" % mm] for mm in (5, 2, 1)] == [None, None, None]


def test_a_settle_heard_through_its_whole_window_is_whole_at_the_run_end(tmp_path, monkeypatch):
    """A window whose readings run on past its end was heard whole, whenever
    its note is written: at the run's end too it is not cut."""
    clock, c, r = _creep(monkeypatch, tmp_path)
    r.run([leg("a", Q0, Q1)], execute=True)
    clock.mono += 2.0  # the reading at +1.6 s is heard by now
    r.flush()
    (note,) = _settles(tmp_path)
    assert note["cut"] is None and note["window_s"] == 1.5 and note["samples"] == 12 and note["end_mm"] == 0.5


def test_no_settle_is_noted_for_an_execution_that_did_not_arrive(tmp_path, monkeypatch):
    clock, c, r = _creep(monkeypatch, tmp_path)
    c.exec_script = [("failed", {"message": "halted: emergency stop", "progress": 0.4})]
    r.run([leg("a", Q0, Q1)], execute=True)
    clock.mono += 2.0
    r.flush()
    assert _settles(tmp_path) == []


class _Flies(FakeClient):
    """A client that reports what it sent, as PlannerClient.execute does:
    when, the trajectory, and every 10th feedback; and lists, while each
    motion flies, the side files already written."""

    FEEDBACK_EVERY = 10

    def __init__(self, tmp_path):
        super().__init__()
        self.tmp_path, self.during, self.flown = tmp_path, [], []

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None, gains=None):
        self.during.append(sorted(p.name for p in self.tmp_path.glob("run-*.flights/*")))
        self.flown.append(traj_)
        outcome, info = super().execute(traj_, speed, guard, while_running, stop_when, gains)
        now = time.monotonic()
        info.update(
            sent_at=now - 1.0,
            sent_traj=traj_,
            feedback_q=[(now - 1.0 + 0.1 * k, 0.1 * k, [0.01 * k] * 7) for k in range(11)],
        )
        return outcome, info


def test_each_execution_keeps_what_it_sent_and_what_flew_beside_the_record_once_it_has_flown(tmp_path):
    """The runner dropped the trajectory each execution flew (flown_traj is
    not log material). Now every execution sent leaves a side file next to
    the run record, <record>.flights/<number>-<first leg>.json: each point
    sent (t, q, qd, qdd) and the driver's feedback of what flew — written
    once the motion is over, never while it flies — and every row of its
    legs names it."""
    import json

    import numpy as np

    c = _Flies(tmp_path)
    r = runner(c, tmp_path)
    r.run([leg("reach:look", Q0, Q1, chain=0)], execute=True)
    r.run([leg("b", Q1, Q2, chain=0), leg("c", Q2, [0.3] * 7, chain=0)], execute=True)
    assert c.during == [[], ["01-reach_look.json"]]  # nothing of a motion is written while it flies
    record = next(tmp_path.glob("run-*.jsonl"))
    folder = record.with_suffix(".flights")
    assert sorted(p.name for p in folder.iterdir()) == ["01-reach_look.json", "02-b.json"]
    doc = json.loads((folder / "02-b.json").read_text())
    assert doc["seq"] == 2 and doc["legs"] == ["b", "c"] and doc["outcome"] == "arrived"
    sent = c.flown[1]
    assert doc["trajectory"]["t"] == pytest.approx(
        [p.time_from_start.sec + 1e-9 * p.time_from_start.nanosec for p in sent.points], abs=1e-4
    )
    for key, attr in (("q", "positions"), ("qd", "velocities"), ("qdd", "accelerations")):
        np.testing.assert_allclose(doc["trajectory"][key], [list(getattr(p, attr)) for p in sent.points], atol=1e-4)
    assert doc["joint_names"] == list(sent.joint_names) and doc["feedback"]["every"] == 10
    fb = doc["feedback"]
    assert fb["t"] == pytest.approx([0.1 * k for k in range(11)], abs=1e-4)
    assert fb["fraction"] == pytest.approx([0.1 * k for k in range(11)])
    np.testing.assert_allclose(fb["q"], [[0.01 * k] * 7 for k in range(11)], atol=1e-6)
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    flights = {row["leg"]: row["flight"] for row in rows}
    assert flights == {
        "reach:look": record.stem + ".flights/01-reach_look.json",
        "b": record.stem + ".flights/02-b.json",
        "c": record.stem + ".flights/02-b.json",
    }


def test_an_execution_the_client_never_sent_keeps_no_flight_and_no_settle(tmp_path, monkeypatch):
    c = _Flies(tmp_path)
    r = runner(c, tmp_path)

    def refused(traj_, speed, **kw):
        return "failed", {"message": "refused before sending: velocity limit", "progress": 0.0}

    c.execute = refused
    res = r.run([leg("a")], execute=True)
    assert res[0].outcome == "failed" and res[0].flight is None
    assert not list(tmp_path.glob("run-*.flights")) and r._settling is None


def test_the_recoil_is_an_execution_like_any_other_in_the_flight_log(tmp_path, monkeypatch):
    """A push that ran its bound arrived: its settle is noted as the recoil
    goes out, its window cut there, after the hold on the button. The
    recoil keeps its side file, and its own settle is noted as the next
    execution goes out."""
    import json

    clock, c, r = _creep(monkeypatch, tmp_path)
    push = leg("press:push", Q0, Q1, guard=GuardSpec(touch_nm=4.0, trip="press"), world="interaction_button")
    push.verify = lambda v: (v.outcome in ("touch", "arrived"), "pressed")  # as press_push's
    push.hold_s = 0.35  # held on the button (the clock's sleep)
    res = r.run([push, leg("retreat", Q1, Q2, chain=1, world="interaction_button")], execute=True)
    assert [x.leg_name for x in res] == ["press:push", "recoil", "retreat"]
    record = next(tmp_path.glob("run-*.jsonl"))
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    assert [row["leg"] for row in rows] == ["press:push", "settle", "recoil", "settle", "retreat"]
    pushed, recoiled = rows[1], rows[3]
    assert pushed["legs"] == ["press:push"] and pushed["cut"] == "next execution"
    assert pushed["window_s"] == pytest.approx(0.35) and pushed["samples"] == 3
    assert rows[2]["flight"] == record.stem + ".flights/02-recoil.json"
    assert recoiled["legs"] == ["recoil"] and recoiled["seq"] == 2 and recoiled["cut"] == "next execution"


class _CtrlC(_Flies):
    """_Flies, its execution number `at` stopped by a Ctrl+C: as
    PlannerClient.execute does once the cancel and the release are out, it
    raises KeyboardInterrupt with nothing returned, what it kept of the call
    in last_execution — or, `sent` False, raises before anything went out,
    last_execution still the call before's."""

    def __init__(self, tmp_path, at, sent=True):
        super().__init__(tmp_path)
        self.at, self.sent, self.n = at, sent, 0

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None, gains=None):
        self.n += 1
        if self.n != self.at:
            outcome, info = super().execute(traj_, speed, guard, while_running, stop_when, gains)
            self.last_execution = info
            return outcome, info
        if self.sent:
            self.flown.append(traj_)
            now = time.monotonic()
            self.last_execution = {
                "message": "",
                "progress": 0.4,
                "torque_peak": None,
                "gains": None,
                "sent_at": now,
                "sent_traj": traj_,
                "feedback_q": [(now + 0.1, 0.4, [0.02] * 7)],
            }
        raise KeyboardInterrupt


@pytest.mark.parametrize("stopped", ["b", "recoil"])
def test_an_execution_a_ctrl_c_stopped_keeps_its_flight_and_the_ctrl_c_goes_on(tmp_path, stopped):
    """A Ctrl+C: the client cancels, lets the spring go and raises, handing
    nothing back. The motion it stopped is the one most worth reading: its
    side file is kept all the same, from what the client kept of the call,
    and the KeyboardInterrupt goes on to press_demo's exit."""
    import json

    import numpy as np

    c = _CtrlC(tmp_path, at=2)
    r = runner(c, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        if stopped == "b":
            r.run([leg("a", Q0, Q1)], execute=True)
            r.run([leg("b", Q1, Q2)], execute=True)
        else:
            c.exec_script = [("touch", {"message": "contact", "progress": 0.8, "torque_peak": 7.1})]
            r.run([_press_leg(chain=0), leg("retreat", Q1, Q2, chain=1, world="interaction_button")], execute=True)
    folder = next(tmp_path.glob("run-*.flights"))
    first = "01-a.json" if stopped == "b" else "01-press_down.json"
    assert sorted(p.name for p in folder.iterdir()) == [first, "02-%s.json" % stopped] and r._flights == 2
    doc = json.loads((folder / ("02-%s.json" % stopped)).read_text())
    assert doc["outcome"] == "aborted" and doc["legs"] == [stopped] and doc["feedback"]["fraction"] == [0.4]
    np.testing.assert_allclose(doc["trajectory"]["q"], [list(p.positions) for p in c.flown[-1].points], atol=1e-6)


def test_a_ctrl_c_before_anything_went_out_keeps_no_flight(tmp_path):
    """The client's last_execution is then the call before's: it is never
    kept again, under another leg's name."""
    c = _CtrlC(tmp_path, at=2, sent=False)
    r = runner(c, tmp_path)
    r.run([leg("a", Q0, Q1)], execute=True)
    with pytest.raises(KeyboardInterrupt):
        r.run([leg("b", Q1, Q2)], execute=True)
    folder = next(tmp_path.glob("run-*.flights"))
    assert sorted(p.name for p in folder.iterdir()) == ["01-a.json"] and r._flights == 1
