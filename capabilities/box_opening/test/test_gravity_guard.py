"""The arm's gravity model (runtime/kinematics.GravityModel) and the guard
that judges only what it does not explain (runtime/guards.CompensatedGuard),
pinned to flatbot's own URDF and /joint_states (2026-10-21, vsp_boxer_test)."""

import numpy as np
import pytest

from box_opening.runtime.guards import CompensatedGuard, GuardSpec, TorqueGuard
from box_opening.runtime.kinematics import ArmChain, GravityModel

DATA = "capabilities/box_opening/test/data/"
URDF = DATA + "arm_flatbot_20261021.urdf"
CONTACT_TOOL = (0.0, -0.009, 0.011)  # tasks/reach_press


@pytest.fixture(scope="module")
def model():
    return GravityModel(path=URDF, root="arm_mount")


@pytest.fixture(scope="module")
def recorded():
    d = np.load(DATA + "joint_states_flatbot_20261021.npz")
    return d["t"].astype(float), d["position"], d["effort"].astype(float)


def test_the_model_explains_what_flatbots_wrist_joints_read(model, recorded):
    t, q, eff = recorded
    v = np.r_[np.zeros((1, 7)), np.diff(q, axis=0) / np.diff(t)[:, None]]
    rest = np.abs(v).max(1) < 0.01
    assert rest.sum() > 50
    H = np.array([model.wrist_holding(row) for row in q[rest]])
    E = eff[rest, 3:]
    for j in range(3):  # joints 4-6 carry the wrist's weight; joint_7 none
        r = E[:, j] - H[:, j]
        assert np.std(r) < 0.2, "joint_%d: effort - model sd %.3f" % (j + 4, np.std(r))
        # the SIGN: the model tracks the effort, it does not mirror it
        assert np.corrcoef(E[:, j], H[:, j])[0, 1] > 0.9
    # the drift a guard sees on joint_4, raw and with the model's share taken
    # out — at rest, and through the whole recording, moving included
    raw = np.abs(E[:, 0] - E[0, 0]).max()
    left = np.abs((E[:, 0] - H[:, 0]) - (E[0, 0] - H[0, 0])).max()
    assert raw > 1.2 and left < 0.5
    Hall = np.array([model.wrist_holding(row) for row in q])
    left_all = np.abs((eff[:, 3:] - Hall) - (eff[0, 3:] - Hall[0])).max(0)
    assert np.abs(eff[:, 3] - eff[0, 3]).max() > 1.4 and left_all.max() < 0.75  # under DRAG_NM 1.5 by 2x


def test_the_model_uses_the_same_chain_as_the_kinematics(model):
    chain = ArmChain(path=URDF, root="arm_mount")
    assert model.joint_names == chain.joint_names
    q = [0.067, -1.897, -3.137, -2.269, -0.211, -0.683, 1.725]
    # a mass AT the tool frame is felt exactly as the chain places tool_frame
    _R, tool = chain.fk(q)
    k = model.tip_load(q, (0.0, 0.0, 0.0))
    T, axes = model._frames(q)
    for name, kj in zip(model.joint_names[3:], k, strict=False):
        p, a = axes[name]
        assert kj == pytest.approx(-float(a @ np.cross(tool - p, [0.0, 0.0, -1.0])), abs=1e-9)


def test_a_pull_down_at_the_fingertip_reads_mostly_on_joint_4(model):
    q = [0.067, -1.897, -3.137, -2.269, -0.211, -0.683, 1.725]
    k = model.tip_load(q, CONTACT_TOOL)
    assert abs(k[0]) > 0.3 and abs(k[3]) < 0.05  # joint_7 turns about the hand's own axis


def _installed_model(**kw):
    try:
        return GravityModel(root="base_link", **kw)
    except Exception as e:  # noqa: BLE001 - no installed description on this machine
        pytest.skip("no installed URDF: %s" % e)


def _assert_lumps_match_the_tree(model, n=200, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        q = rng.uniform(-3.0, 3.0, 7)
        pt = rng.uniform(-0.1, 0.1, 3)
        assert np.abs(model.holding(q) - model._holding_tree(q)).max() < 1e-9
        assert np.abs(model.holding(q, first=3) - model._holding_tree(q, first=3)).max() < 1e-9
        extra = [(0.3, pt)]
        assert np.abs(model.holding(q, extra=extra) - model._holding_tree(q, extra=extra)).max() < 1e-9
        tree_tip = model._holding_tree(q, extra=[(1.0 / 9.81, pt)], first=3) - model._holding_tree(q, first=3)
        assert np.abs(model.tip_load(q, pt) - tree_tip).max() < 1e-9


def test_the_lumped_model_matches_the_link_tree_walk(model):
    _assert_lumps_match_the_tree(model)  # flatbot, arm_mount
    _assert_lumps_match_the_tree(_installed_model(), seed=1)  # the bench chain, base_link


def test_the_lumps_follow_the_gripper_angle_they_were_built_at():
    open_ = GravityModel(path=URDF, root="arm_mount", gripper_rad=0.0)
    _assert_lumps_match_the_tree(open_, seed=2)
    _assert_lumps_match_the_tree(_installed_model(gripper_rad=0.0), seed=3)


def test_gripper_rad_is_fixed_at_construction(model):
    with pytest.raises(AttributeError):
        model.gripper_rad = 0.0


def _feed(guard, samples):
    """Readings into the guard until it trips (the client cancels there)."""
    guard.on_progress(0.5)
    out = []
    for eff, q in samples:
        out.append(guard.on_efforts(eff, q))
        if out[-1]:
            break
    return out


def test_a_reach_whose_gravity_drift_is_the_threshold_does_not_trip_the_compensated_guard():
    grav = lambda q: [2.0 * q[0], 0.0, 0.0, 0.0]  # noqa: E731 - joint_4 holds 2 Nm per unit of q[0]
    samples = [([2.0 * x, 0.0, 0.0, 0.0], [x] + [0.0] * 6) for x in np.linspace(0.0, 0.9, 10)]  # 1.8 Nm of drift
    raw = TorqueGuard(1.5)
    raw.on_progress(0.5)
    assert any(raw.on_efforts(e) for e, _q in samples)  # what the reach used to trip on
    comp = CompensatedGuard(1.5, grav)
    assert not any(_feed(comp, samples)) and comp.peak < 1e-9
    # ... and a strike on top of the same drift still trips it
    struck = samples[:5] + [([e[0] + 1.6, 0.0, 0.0, 0.0], q) for e, q in samples[5:]]
    assert any(_feed(CompensatedGuard(1.5, grav), struck))


def test_leaving_the_button_never_trips_the_pull_check_and_a_box_coming_along_does():
    grav = lambda q: [0.0] * 4  # noqa: E731
    k = [0.48, 0.0, 0.2, 0.0]
    ref = ([0.0] * 4, [0.0] * 7)
    q = [0.0] * 7
    # the leg starts ON the button: 8 N pushing up, let go of over the stroke
    leaving = [([-f * kj for kj in k], q) for f in np.linspace(8.0, 0.0, 12)]
    g = CompensatedGuard(10.0, grav, reference=ref, hang=(k, 2.5))
    assert not any(_feed(g, leaving)) and g.peak_down_n <= 1e-9
    # against its own first (loaded) sample that release read as a 3.8 Nm strike
    first = TorqueGuard(1.5)
    first.on_progress(0.5)
    assert any(first.on_efforts(e) for e, _q in leaving)
    # a 0.5 kg canister coming up with the finger
    caught = leaving + [([w * kj for kj in k], q) for w in np.linspace(0.0, 4.9, 15)]  # up to its whole weight
    g = CompensatedGuard(10.0, grav, reference=ref, hang=(k, 2.5))
    tripped = _feed(g, caught)
    assert any(tripped) and g.trip_report()["why"] == "hang" and g.trip_report()["compensated"]
    assert 2.5 < g.down_n < 2.9


def test_the_ringing_after_a_recoil_is_not_a_box_on_the_hand_and_a_steady_pull_still_is():
    """Field 2026-10-23 17:18 (run-20261023-171812): 0.1 s into the depth
    press's unload, 23 ms after the recoil stopped, joint_4 rang reading to
    reading and its last reading alone read as 3.26 N of pull: the arm
    stopped over the pressed box with nothing on the hand. The trip report's
    run-up, judged on the mean of HANG_SAMPLES readings."""
    from box_opening.runtime.guards import HANG_SAMPLES

    ref = ([7.645, 0.246, 0.581, -0.154], [0.0] * 7)
    q = [1.0] + [0.0] * 6
    grav = lambda q: [0.096 * q[0], 0.0, 0.0, 0.0]  # noqa: E731 - the 0.096 Nm the report's dev_nm took out
    k = [0.297, 0.0, 0.0, 0.0]  # what reads the report's 0.968 Nm on joint_4 as its 3.26 N
    ringing = [
        [7.706, 0.127, 0.098, -0.158],
        [8.37, -0.107, 0.583, -0.131],
        [7.768, -0.151, 0.572, -0.145],
        [6.597, 0.15, 0.32, -0.164],
        [8.192, 0.269, 0.916, -0.179],
        [7.289, -0.156, 0.443, -0.147],
        [8.709, 0.211, 0.729, -0.144],
    ]
    g = CompensatedGuard(10.0, grav, reference=ref, hang=(k, 2.5))
    assert not any(_feed(g, [(e, q) for e in ringing for _repeat in range(12)]))  # each message handed in 12 times
    assert abs(g.down_n) < 0.5
    one = CompensatedGuard(10.0, grav, reference=ref, hang=(k, 2.5))
    one.on_progress(0.5)
    one.on_efforts(ringing[-1], q)
    assert one.down_n == pytest.approx(3.26, abs=0.01)  # the reading that stopped the arm
    # a 0.3 kg canister coming along on top of the same ringing still trips it
    held = [[e[0] + 2.94 * k[0]] + e[1:] for e in ringing]
    g = CompensatedGuard(10.0, grav, reference=ref, hang=(k, 2.5))
    tripped = _feed(g, [(e, q) for e in ringing + held])
    assert tripped[-1] and len(tripped) <= len(ringing) + HANG_SAMPLES and g.trip_report()["why"] == "hang"


def test_a_reference_baseline_is_not_retaken_by_a_rebaseline():
    g = CompensatedGuard(1.0, lambda q: [0.0] * 4, reference=([5.0, 0.0, 0.0, 0.0], [0.0] * 7), rebaseline_after=0.2)
    g.on_progress(0.5)
    assert g.on_efforts([6.5, 0.0, 0.0, 0.0], [0.0] * 7)  # judged against the reference, not the first sample


def test_without_joints_the_compensated_guard_judges_raw():
    g = CompensatedGuard(1.0, lambda q: [100.0] * 4)
    g.on_progress(0.5)
    assert not g.on_efforts([0.0] * 4)
    assert g.on_efforts([1.5, 0.0, 0.0, 0.0])


def test_the_runner_builds_a_compensated_guard_and_never_replans_a_plan_free_line(ctx, tmp_path):
    from conftest import leg as make_leg
    from conftest import runner as make_runner

    grav = lambda q: [0.0] * 4  # noqa: E731
    r = make_runner(ctx.client, tmp_path)
    line = make_leg(
        "reach:slide-in",
        guard=GuardSpec(touch_nm=1.5, trip="obstruction", gravity=grav, reference=([0.0] * 4, [0.0] * 7)),
    )
    line.plan_free = True
    res = r.run([line], execute=True)
    assert res[-1].ok and isinstance(ctx.client.guards[-1], CompensatedGuard)
    # the arm is no longer where the line starts: refused, not re-planned
    ctx.client.live = [0.3] * 7
    line2 = make_leg("reach:slide-out", guard=GuardSpec(touch_nm=1.5, trip="obstruction", gravity=grav))
    line2.plan_free = True
    n = len(ctx.client.executed)
    res = r.run([line2], execute=True)
    assert not res[-1].ok and res[-1].outcome == "refused" and "plan-free" in res[-1].detail
    assert len(ctx.client.executed) == n and not ctx.client.plans


class _FeedingClient:
    """FakeClient whose execute() feeds a scripted (efforts, joints) stream
    through the guard the runner built, as the real client does."""

    def __init__(self, base, stream):
        self.base, self.stream = base, stream

    def __getattr__(self, name):
        return getattr(self.base, name)

    def execute(self, traj_, speed, guard=None, while_running=None, stop_when=None):
        self.base.guards.append(guard)
        guard.on_progress(0.1)
        for eff, q in self.stream:
            if guard.on_efforts(eff, q) if getattr(guard, "wants_joints", False) else guard.on_efforts(eff):
                return "touch", {"message": "torque guard trip", "progress": 0.5, "torque_peak": guard.peak}
        self.base.live = list(traj_.points[-1].positions)
        return "arrived", {"message": "ok", "progress": 1.0, "torque_peak": guard.peak}


def test_the_unload_built_by_the_runner_lets_the_button_go_and_stops_a_box_coming_along(ctx, tmp_path):
    """The headline fix, end to end through make_guard: a leg that starts
    loaded on the button is judged against the free-air reading, so the
    release never trips it — and a pull at the fingertip does."""
    from conftest import leg as make_leg

    from box_opening.models.container import ContainerModel
    from box_opening.runtime.runner import Runner
    from box_opening.tasks import reach_press

    k = [0.48, 0.0, 0.2, 0.0]
    q = [0.0] * 7
    m = ContainerModel.load("capabilities/box_opening/config/containers/ankou_pink.yaml")
    import box_opening.runtime.approach as approach

    approach._GRAVITY = type("G", (), {"wrist_holding": staticmethod(lambda q: [0.0] * 4)})()
    try:
        spec = reach_press.unload_guard(m, ([0.0] * 4, q), k)
    finally:
        approach._GRAVITY = None
    leaving = [([-f * kj for kj in k], q) for f in np.linspace(8.0, 0.0, 12)]
    for stream, ok in (
        (leaving, True),
        (leaving + [([w * kj for kj in k], q) for w in np.linspace(0, 4.9, 15)], False),
    ):
        ctx.client.live = [0.0] * 7  # each line starts where the arm is
        client = _FeedingClient(ctx.client, stream)
        line = make_leg("reach:unload", guard=spec)
        line.plan_free = True
        res = Runner(client, log_dir=tmp_path).run([line], execute=True)
        assert res[-1].ok is ok and res[-1].outcome == ("arrived" if ok else "touch")
        assert isinstance(ctx.client.guards[-1], CompensatedGuard)
        if not ok:
            assert res[-1].trip["why"] == "hang"


def test_a_plan_free_line_after_a_press_is_never_replanned_even_through_the_recoil(ctx, tmp_path):
    from conftest import leg as make_leg
    from conftest import runner as make_runner

    push = make_leg("press:push", guard=GuardSpec(touch_nm=7.0, trip="press"))
    line = make_leg("reach:unload", start=[0.1] * 7, end=[0.2] * 7, guard=GuardSpec(touch_nm=10.0, trip="obstruction"))
    line.plan_free = True
    ctx.client.exec_script = [("touch", {"message": "torque guard trip", "progress": 0.6})]
    n_plans = len(ctx.client.joint_targets)
    res = make_runner(ctx.client, tmp_path).run([push, line], execute=True)
    assert res[-1].leg_name == "reach:unload" and res[-1].outcome == "refused"
    assert len(ctx.client.joint_targets) == n_plans  # the planner was never asked for it
