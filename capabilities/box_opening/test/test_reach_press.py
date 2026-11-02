"""The reach-out press (tasks/reach_press): flatbot's box 0.80-0.89 m out on
a table 0.30 m above the arm's mount, where no straight-down pose reaches."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from box_opening.constants import TCP_OFFSET_M, TIP_TO_TOOL_M
from box_opening.models.container import ContainerModel, load_press_demo
from box_opening.perception.d405 import quat_to_mat
from box_opening.tasks import reach_press

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
BOX = (0.675, -0.578)  # where the scene camera put it, 2026-10-22


@pytest.fixture(scope="module")
def model():
    return ContainerModel.load(PINK)


def _axes(quat):
    R = np.asarray(quat_to_mat(*quat), float)
    return R[:, 0], R[:, 1], R[:, 2]


def test_the_reach_attitude_leans_back_with_the_camera_above_the_hand(model):
    u = np.array([BOX[0], BOX[1], 0.0]) / math.hypot(*BOX)
    for tilt in reach_press.REACH_TILTS_DEG:
        x, y, z = _axes(reach_press.reach_quat(model, BOX, tilt))
        assert math.degrees(math.acos(float(np.clip(-z[2], -1, 1)))) == pytest.approx(
            tilt, abs=1e-6
        )  # off straight down by tilt
        assert z @ u > 0  # the fingertips lead away from the arm: the hand leans back toward it
        assert (
            y[2] > 0 and abs(y @ np.cross([0, 0, 1], u)) < 1e-9
        )  # the wrist camera's side is up, in the bearing's plane
        assert abs(x[2]) < 1e-9 and abs(x @ u) < 1e-9  # the fingers side by side across the line to the box
    assert reach_press.REACH_TILTS_DEG[-1] <= 45  # past that the knuckles, then the palm, meet the flush lid


def test_the_fingertip_lands_where_it_is_aimed_and_is_read_back_from_the_tips(model):
    for tilt in (0, 30, 45):
        q = reach_press.reach_quat(model, BOX, tilt)
        R = np.asarray(quat_to_mat(*q), float)
        aim = np.array([0.66, -0.57, 0.40])
        tool = np.asarray(reach_press.tool_for_contact(aim, q))
        assert tool + R @ np.asarray(reach_press.CONTACT_TOOL) == pytest.approx(aim, abs=1e-12)
        tips = tool + R @ np.asarray(reach_press.TIP_MID_TOOL)  # what client.contact_xyz reads
        assert reach_press.contact_from_tips(tips, q) == pytest.approx(list(aim), abs=1e-12)
    # straight down, the same numbers the straight-down press uses
    q0 = reach_press.reach_quat(model, BOX, 0)
    tool = np.asarray(reach_press.tool_for_contact((0.5, -0.4, 0.4), q0))
    assert tool[2] - 0.4 == pytest.approx(TCP_OFFSET_M, abs=1e-9)
    tips = tool + np.asarray(quat_to_mat(*q0), float) @ np.asarray(reach_press.TIP_MID_TOOL)
    assert tips[2] - tool[2] == pytest.approx(TIP_TO_TOOL_M, abs=1e-9)


STRAIGHT_DOWN = np.diag([1.0, -1.0, -1.0])  # tool_frame's z along the world's -z


class _Chain:
    """Straight lines that always fly, through two waypoints; no joint
    limits (ClearMarginFree finds none near). fk stands still, the hand at
    R: np.eye(3) unless a test says otherwise — a lean of 180 deg, whose
    sine is ~0, so a test that means straight down sets STRAIGHT_DOWN."""

    limits = [(None, None)] * 7
    R = np.eye(3)

    def __init__(self, refuse=()):
        self.refuse = refuse
        self.lines = []

    def straight_line(self, q0, xyz, quat):
        self.lines.append((list(xyz), list(quat)))
        for what in self.refuse:
            if what(xyz, quat):
                return None, "IK did not converge"
        return [list(q0), [v + 0.01 for v in q0]], None

    def walk(self, q0, xyz, quat, step_m=None, step_rad=None, max_step_rad=None):
        return None, "no turning walk modelled"  # the depth press's slide in, then its turn

    def fk(self, q):
        return np.array(self.R, float), np.array([0.5, -0.4, 0.45])


def _tilt_of(quat):
    _x, _y, z = _axes(quat)
    return round(math.degrees(math.acos(float(np.clip(-z[2], -1, 1)))))


def test_the_reach_plan_takes_the_smallest_tilt_that_plans(ctx, monkeypatch):
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    chain = _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    asked = []

    def look_plans_from_40(ctx_, state, name, target, world, speed):
        asked.append(_tilt_of(target[2]))
        if _tilt_of(target[2]) < 40:
            raise RuntimeError("planning failed for reach:look: MotionGenStatus.IK_FAIL")
        return SimpleNamespace(goal_joints=[0.0] * 7), state

    monkeypatch.setattr(reach_press, "_plan_motion", look_plans_from_40)
    button = (0.66, -0.57, 0.40)
    plan, why = reach_press.plan_reach(ctx, cfg, button, [0.0] * 7)
    assert plan is not None and plan.tilt_deg == 40 and asked == [30, 35, 40] and why is None
    d = math.hypot(button[0], button[1])
    back = np.subtract(plan.start_xyz[:2], plan.look_xyz[:2])
    assert np.linalg.norm(back) == pytest.approx(reach_press.REACH_PULLBACK_M)  # pulled back toward the arm ...
    assert back @ (np.asarray(button[:2]) / d) == pytest.approx(reach_press.REACH_PULLBACK_M)
    # ... and up from the stroke's start: the wrist camera ~0.30 m off the lid, not ~0.245
    assert plan.look_xyz[2] == pytest.approx(plan.start_xyz[2] + reach_press.REACH_LOOK_RISE_M)
    # the start and the bottom are straight above/below each other: the stroke is vertical
    assert plan.start_xyz[:2] == pytest.approx(plan.bottom_xyz[:2], abs=1e-12)
    assert plan.start_xyz[2] - plan.bottom_xyz[2] == pytest.approx(reach_press.REACH_START_M + cfg.travel_m)
    # nothing plans: every tilt is named in the refusal
    monkeypatch.setattr(reach_press, "_plan_motion", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x: IK_FAIL")))
    plan, why = reach_press.plan_reach(ctx, cfg, button, [0.0] * 7)
    assert plan is None and all("%d deg" % t in why for t in reach_press.REACH_TILTS_DEG)


def test_a_tilt_whose_stroke_the_arm_cannot_fly_straight_is_passed_over(ctx, monkeypatch):
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    chain = _Chain(refuse=[lambda xyz, quat: _tilt_of(quat) == 30 and xyz[2] < 0.41])  # the 30 deg stroke down
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(
        reach_press, "_plan_motion", lambda c, s, n, t, w, sp: (SimpleNamespace(goal_joints=[0.0] * 7), s)
    )
    plan, why = reach_press.plan_reach(ctx, cfg, (0.66, -0.57, 0.40), [0.0] * 7)
    assert plan is not None and plan.tilt_deg == 35


DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]  # STRAIGHT_DOWN as a quaternion (x, y, z, w)


def _way_down(asked, solves=lambda tilt, tip: False):
    """reach_press.way_down for the fake chain: each ask kept as (tilt, the
    xy under the slid-in fingertip — the button's —, top_z); `solves(tilt,
    fingertip)` says where the hand turns down and strokes: the turn in
    place where the slide ends, then the press stroke from its end, each a
    step on from the last as _Chain's lines are, the spring 440 N/m. A
    refusal reads as 16:23's did."""
    from box_opening.tasks import depth_press as dp

    def way_down(ctx_, cfg_, q_look, slide, start, quat, top_z):
        tip = np.asarray(start, float) + np.asarray(quat_to_mat(*quat), float) @ np.asarray(reach_press.CONTACT_TOOL)
        asked.append((_tilt_of(quat), [float(tip[0]), float(tip[1])], float(top_z)))
        if solves(_tilt_of(quat), tip):
            turn = [[float(v) for v in slide[-1]], [float(v) + 0.01 for v in slide[-1]]]
            down = dp.WayDown(turn, list(DOWN_QUAT), 0.0, False)
            t0 = np.array([0.5, -0.4, 0.45])  # the fake fk's
            stroke = dp.StrokePlan(
                [("depth:press", [turn[-1], [v + 0.01 for v in turn[-1]]], dp.TOUCH_SPEED)],
                440.0,
                0.03 - cfg_.button_travel_m,
                float(top_z) - 0.03,
                t0,
                t0 + STRAIGHT_DOWN @ np.asarray(dp.DOWN_CONTACT_TOOL),
                turn[-1],
                list(DOWN_QUAT),
            )
            return down, stroke, None
        return (
            None,
            None,
            (
                "walk in: 0 deg short: a joint jump of 0.19 rad 44% of the way — near a singularity; "
                "turn after the slide: 0 deg short: IK did not converge 73% of the way"
            ),
        )

    return way_down


def test_a_compliant_depth_plan_passes_over_a_tilt_whose_way_down_refuses(ctx, monkeypatch):
    """The compliant depth press never flies the tilted stroke: from the
    look pose the hand turns down and strokes straight down (way_down).
    plan_reach(depth=True) asks that instead, at the scene button's top,
    passes over a tilt where it does not solve, and keeps that tilt's look
    leg and slide (`looks`). Without depth, the tilted stroke as before.
    (The fake chain has no IK of the arm's own: each look is the planner's
    pose plan, proven from where it ends, and kept with its leg.)"""
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] < 0.41])  # every tilt's stroke down (the fake fk stands at 0.45)
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(
        reach_press, "_plan_motion", lambda c, s, n, t, w, sp: (SimpleNamespace(goal_joints=[0.0] * 7), s)
    )
    asked = []
    monkeypatch.setattr(reach_press, "way_down", _way_down(asked, solves=lambda tilt, tip: tilt > 30))
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True, looks=looks)
    assert plan is not None and why is None and plan.tilt_deg == 35 and plan.stroke_pts is None
    assert plan.down.walked is False and plan.stroke.end_z == pytest.approx(BUTTON[2] - 0.03)
    assert [(t, z) for t, _xy, z in asked] == [(30, BUTTON[2]), (35, BUTTON[2])]
    assert all(xy == pytest.approx(list(BUTTON[:2])) for _t, xy, _z in asked)  # the slide ends over the button
    assert not any(xyz[2] < 0.41 for xyz, _q in chain.lines)  # no tilted stroke asked for
    (look,) = looks
    assert look.tilt_deg == 30 and look.down is None and look.look_leg is not None and look.slide_pts
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7)
    assert plan is None and all("%d deg: the stroke (" % t in why for t in reach_press.REACH_TILTS_DEG)


def test_where_no_tilt_turns_down_and_strokes_the_depth_plan_names_every_tilt_and_keeps_their_looks(ctx, monkeypatch):
    """A tilt whose look pose does not plan has no look to keep. (No IK of
    the arm's own here either: the looks kept are the planner's.)"""
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _Chain())

    def look_plans_from_35(ctx_, state, name, target, world, speed):
        if _tilt_of(target[2]) < 35:
            raise RuntimeError("planning failed for reach:look: MotionGenStatus.IK_FAIL")
        return SimpleNamespace(goal_joints=[0.0] * 7), state

    monkeypatch.setattr(reach_press, "_plan_motion", look_plans_from_35)
    monkeypatch.setattr(reach_press, "way_down", _way_down([]))
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True, looks=looks)
    assert plan is None and "30 deg: the look pose (MotionGenStatus.IK_FAIL)" in why
    for tilt in (35, 40, 45):
        assert (
            "%d deg: the way down (walk in: 0 deg short: a joint jump of 0.19 rad 44%% of the way — near a "
            "singularity; turn after the slide: 0 deg short: IK did not converge 73%% of the way)" % tilt
        ) in why
    assert [lk.tilt_deg for lk in looks] == [35, 40, 45]
    assert all(lk.down is None and lk.stroke is None and lk.look_leg is not None and lk.slide_pts for lk in looks)


Q_LOOK = [0.05] * 7  # where _flow's planner ends every look leg


class _IKChain(_Chain):
    """_Chain with the arm's own IK for a look pose, which
    client.pose_goal_joints asks: every look solves, at Q_LOOK, inside the
    arm's reach (kinematics.ArmChain.past_reach_m)."""

    def nearest_solution(self, q_start, xyz, quat, seeds=()):
        return np.array(Q_LOOK, float)

    def past_reach_m(self, xyz, quat):
        return -0.1


def _planner(planned, end=lambda tilt: Q_LOOK, refuse=()):
    """reach_press._plan_motion counting the planner's look plans: each
    (tilt, target) asked kept in `planned`, the leg ending at end(tilt), a
    tilt in `refuse` not planned."""

    def plan(ctx_, state, name, target, world, speed):
        tilt = _tilt_of(target[2])
        planned.append((tilt, target))
        if tilt in refuse:
            raise RuntimeError("planning failed for reach:look: MotionGenStatus.TRAJOPT_FAIL")
        return SimpleNamespace(goal_joints=[float(v) for v in end(tilt)]), state

    return plan


def test_each_tilt_is_proven_from_the_arms_own_ik_and_only_the_one_that_proves_is_planned(ctx, monkeypatch):
    """2026-10-28 14:28:13 planned four looks, 0.15-0.17 s each, and no
    way down flew from any of them. Each tilt is proven from the joints the
    client plans its look pose to (client.pose_goal_joints, the arm's own
    IK), and the planner is asked once — for the first tilt that proves,
    the pose goal its look always went with — or not at all. A tilt passed
    over is kept unplanned, at the joints it was proven from."""
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _IKChain())
    planned, asked = [], []
    monkeypatch.setattr(reach_press, "_plan_motion", _planner(planned))
    monkeypatch.setattr(reach_press, "way_down", _way_down(asked, solves=lambda tilt, tip: tilt > 30))
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True, looks=looks)
    assert why is None and plan.tilt_deg == 35 and plan.look_leg.goal_joints == Q_LOOK
    quat, _start, _bottom, look = reach_press.reach_geometry(ctx.model, BUTTON, 35, cfg.travel_m)
    assert planned == [(35, ("pose", look, quat))]  # one plan, asked as the look always was ...
    assert [t for t, _xy, _z in asked] == [30, 35]  # ... once its press was proven
    assert plan.look_joints == Q_LOOK and list(plan.slide_pts[0]) == Q_LOOK  # from the client's IK
    (kept,) = looks
    assert kept.tilt_deg == 30 and kept.look_leg is None and kept.look_joints == Q_LOOK and kept.slide_pts
    # no tilt proves: nothing planned
    planned.clear()
    monkeypatch.setattr(reach_press, "way_down", _way_down([]))
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True, looks=looks)
    assert plan is None and planned == []
    assert all("%d deg: the way down (walk in: " % t in why for t in reach_press.REACH_TILTS_DEG)
    assert [lk.tilt_deg for lk in looks] == list(reach_press.REACH_TILTS_DEG)
    assert all(lk.look_leg is None and lk.look_joints == Q_LOOK for lk in looks)
    # ... nor for the guarded press's tilted stroke where that does not fly
    monkeypatch.setattr(approach, "_chain", lambda: _IKChain(refuse=[lambda xyz, quat: xyz[2] < 0.41]))
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7)
    assert plan is None and planned == []
    assert all("%d deg: the stroke (" % t in why for t in reach_press.REACH_TILTS_DEG)


def test_a_tilt_proven_from_the_arms_own_ik_whose_look_does_not_plan_is_passed_over(ctx, monkeypatch):
    """The first tilt proves from the arm's own IK, but its look does not
    plan (a TRAJOPT_FAIL, or the posture inside something the cameras saw):
    that tilt is passed over, as a look that did not plan always was, and
    not kept — it has no look to fly — and the next tilt is proven and
    planned, with its leg. Where no look plans, each is asked once and every
    tilt is named."""
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _IKChain())
    planned = []
    monkeypatch.setattr(reach_press, "_plan_motion", _planner(planned, refuse=(30,)))
    monkeypatch.setattr(reach_press, "way_down", _way_down([], solves=lambda tilt, tip: True))
    for depth in (True, False):  # the compliant depth press, and the guarded press's tilted stroke
        planned.clear()
        looks = []
        plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=depth, looks=looks)
        assert why is None and plan.tilt_deg == 35 and plan.look_leg.goal_joints == Q_LOOK, depth
        assert [t for t, _target in planned] == [30, 35] and looks == [], depth
    # no look plans: nothing to fly, nothing kept
    planned.clear()
    monkeypatch.setattr(reach_press, "_plan_motion", _planner(planned, refuse=reach_press.REACH_TILTS_DEG))
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True, looks=looks)
    assert plan is None and looks == [] and [t for t, _target in planned] == list(reach_press.REACH_TILTS_DEG)
    assert all("%d deg: the look pose (MotionGenStatus.TRAJOPT_FAIL)" % t in why for t in reach_press.REACH_TILTS_DEG)


def test_a_look_planned_to_end_off_its_proof_is_proven_again_from_its_end(ctx, monkeypatch, capsys):
    """The planner's joint plan ends on its goal; its FK-pose fallback, and
    the client's own pose plan, end wherever their IK lands. A look planned
    to end more than LOOK_END_RAD from the joints its press was proven from
    is proven again from where it ends, as every look was before: there no
    way down flies, so that tilt is passed over and kept, with its planned
    leg, and the next proves and is planned. A continuous joint a whole
    turn round ends where it was proven."""
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _IKChain())
    moved = list(Q_LOOK)
    moved[1] += 1.5 * reach_press.LOOK_END_RAD  # joint_2
    froms = []
    solves = _way_down([], solves=lambda tilt, tip: True)

    def way_down(ctx_, cfg_, q, slide, start, quat, top_z):
        froms.append((_tilt_of(quat), [float(v) for v in q]))
        if froms[-1][1] == moved:
            return None, None, "turn after the slide: 0 deg short: IK did not converge 73% of the way"
        return solves(ctx_, cfg_, q, slide, start, quat, top_z)

    monkeypatch.setattr(reach_press, "way_down", way_down)
    planned = []
    monkeypatch.setattr(
        reach_press, "_plan_motion", _planner(planned, end=lambda tilt: moved if tilt == 30 else Q_LOOK)
    )
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True, looks=looks)
    assert why is None and plan.tilt_deg == 35 and [t for t, _target in planned] == [30, 35]
    assert froms == [(30, Q_LOOK), (30, moved), (35, Q_LOOK)]
    (kept,) = looks
    assert kept.tilt_deg == 30 and kept.look_leg.goal_joints == moved and kept.look_joints == moved
    assert "NOTE reach:look at 30 deg: planned to end 0.015 rad from the joints its press was proven from" in (
        capsys.readouterr().out
    )
    # joint_7 a whole turn round: the same joints to the arm, the proof stands
    wound = list(Q_LOOK)
    wound[6] += 2.0 * math.pi
    planned.clear()
    froms.clear()
    monkeypatch.setattr(reach_press, "_plan_motion", _planner(planned, end=lambda tilt: wound))
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True)
    assert why is None and plan.tilt_deg == 30 and len(planned) == 1 and froms == [(30, Q_LOOK)]
    assert plan.look_joints == Q_LOOK and plan.look_leg.goal_joints == wound


def test_a_shifted_press_plans_only_the_look_it_flies_from(ctx, monkeypatch):
    """plan_reach planned none of the looks it kept: plan_shifted_look plans
    the smallest tilt's once its shifted press flies, and none where it
    does not — that STOP costs the planner nothing, even where a larger
    tilt's press would fly and the smallest's look would not have planned
    (before, the next was asked then). A look whose press flies but which
    does not plan is passed over for the next, as plan_reach passes one
    over; one planned to end off the joints it was proven from is proven
    again from its end."""
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _IKChain())
    off_scene = lambda tilt, tip: np.hypot(tip[0] - BUTTON[0], tip[1] - BUTTON[1]) > 0.015  # noqa: E731
    froms = []
    shifted = _way_down([], off_scene)

    def way_down(ctx_, cfg_, q, *a):
        froms.append([float(v) for v in q])
        return shifted(ctx_, cfg_, q, *a)

    monkeypatch.setattr(reach_press, "way_down", way_down)
    planned = []
    monkeypatch.setattr(reach_press, "_plan_motion", _planner(planned))
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, BUTTON, [0.0] * 7, depth=True, looks=looks)
    assert plan is None and planned == [] and len(looks) == 4
    plan, near, why = reach_press.plan_shifted_look(ctx, cfg, looks, BUTTON, [0.0] * 7)
    assert why is None and near == "nearer the arm" and plan.tilt_deg == 30 and plan.look_leg.goal_joints == Q_LOOK
    quat, _start, _bottom, look = reach_press.reach_geometry(ctx.model, BUTTON, 30, cfg.travel_m)
    assert planned == [(30, ("pose", look, quat))]  # the one look that flies, as the look always was
    assert looks[0].look_leg is None  # the kept look itself is left as it was
    # no shifted press flies: nothing planned
    planned.clear()
    monkeypatch.setattr(reach_press, "way_down", _way_down([]))
    plan, near, why = reach_press.plan_shifted_look(ctx, cfg, looks, BUTTON, [0.0] * 7)
    assert plan is None and near is None and planned == []
    assert why.startswith("at 30 deg: 30 mm nearer the arm: the way down (walk in: ")
    # ... nor where it flies only from a larger tilt and the smallest's look would not plan: the search ends there
    monkeypatch.setattr(reach_press, "way_down", _way_down([], lambda tilt, tip: tilt > 30 and off_scene(tilt, tip)))
    monkeypatch.setattr(reach_press, "_plan_motion", _planner(planned, refuse=(30,)))
    plan, near, why = reach_press.plan_shifted_look(ctx, cfg, looks, BUTTON, [0.0] * 7)
    assert plan is None and near is None and planned == []
    assert why.startswith("at 30 deg: 30 mm nearer the arm: the way down (walk in: ")
    # the smallest tilt's look does not plan: the next one's
    monkeypatch.setattr(reach_press, "way_down", way_down)
    plan, near, why = reach_press.plan_shifted_look(ctx, cfg, looks, BUTTON, [0.0] * 7)
    assert why is None and plan.tilt_deg == 35 and [t for t, _target in planned] == [30, 35]
    # planned to end elsewhere: proven again from there
    moved = list(Q_LOOK)
    moved[3] -= 0.2
    planned.clear()
    froms.clear()
    monkeypatch.setattr(reach_press, "_plan_motion", _planner(planned, end=lambda tilt: moved))
    plan, near, why = reach_press.plan_shifted_look(ctx, cfg, looks, BUTTON, [0.0] * 7)
    assert why is None and plan.tilt_deg == 30 and plan.look_joints == moved and list(plan.slide_pts[0]) == moved
    assert froms == [Q_LOOK, moved] and len(planned) == 1


class _Gravity:
    """No weight anywhere; a pull at the fingertip reads like flatbot's."""

    K = [0.48, 0.0, 0.2, 0.0]

    def wrist_holding(self, q):
        return [0.0] * 4

    def tip_load(self, q, point_tool):
        return list(self.K)


def test_the_reach_touch_takes_every_trip_as_a_surface_and_leaves_the_height_to_the_verdict(ctx, monkeypatch):
    from box_opening.runtime import approach

    cfg = load_press_demo(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    q0 = [0.0, 0.3, 0.0, 1.9, 0.0, 0.9, 0.0]
    pts = [[a + 0.002 * k for a in q0] for k in range(30)]
    quat = reach_press.reach_quat(ContainerModel.load(PINK), BOX, 30)
    ref = ([0.0] * 4, q0)
    leg = reach_press.touch_leg(ctx, cfg, pts, quat, ("interaction", "w.yaml"), ref)
    assert leg.guard.trip == "touch" and leg.guard.touch_nm == cfg.contact_nm
    assert leg.guard.reference == ref and leg.guard.gravity is not None and leg.plan_free
    assert leg.guard.arm_after is None
    v = lambda outcome, progress: SimpleNamespace(outcome=outcome, progress=progress, torque_peak=3.2)  # noqa: E731
    ok, _why = leg.verify(v("touch", 0.7))
    assert ok
    ok, _why = leg.verify(v("touch", 0.05))  # high above the button: touch_verdict calls it foreign
    assert ok
    ok, why = leg.verify(v("arrived", 1.0))
    assert not ok and "no surface" in why
    # 40 mm above the aimed top is not this button: the run backs out instead of pressing
    from box_opening.tasks.press_demo import touch_verdict

    assert touch_verdict(0.040) == "foreign" and touch_verdict(0.012) == "popped" and touch_verdict(0.001) == "press"


def test_the_way_out_is_judged_on_what_gravity_does_not_explain(monkeypatch):
    from box_opening.runtime import approach

    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    ref = ([0.0] * 4, [0.0] * 7)
    g = reach_press.drag_guard(ref, _Gravity.K)
    assert g.trip == "obstruction" and g.touch_nm <= 1.5 and g.gravity is not None and g.reference == ref
    assert g.hang == (_Gravity.K, reach_press.HANG_N) and reach_press.HANG_N < 0.3 * 9.81  # under a 0.3 kg canister
    assert g.arm_after is None  # armed from the first sample: a catch is felt from the first millimetre
    u = reach_press.unload_guard(ContainerModel.load(PINK), ref, _Gravity.K)
    # it starts ON the button: only the push's backstop bounds the residual, the pull check does the rest
    assert u.touch_nm == ContainerModel.load(PINK).touch_nm and u.hang[1] == reach_press.HANG_N


def test_a_pull_at_the_fingertip_is_read_off_the_residual():
    k = _Gravity.K
    grav = lambda q: [2.0 * q[0], 0.0, 0.0, 0.0]  # noqa: E731
    ref = ([1.0, 0.0, 0.0, 0.0], [0.0] * 7)
    now = ([1.0 + 2.0 * 0.3 + 3.0 * k[0], 0.0, 3.0 * k[2], 0.0], [0.3] + [0.0] * 6)
    r = reach_press.residual(grav, ref, now)
    assert r == pytest.approx([3.0 * k[0], 0.0, 3.0 * k[2], 0.0])  # gravity's 0.6 Nm taken out
    assert reach_press.down_force(k, r) == pytest.approx(3.0)
    assert reach_press.down_force(k, [-v for v in r]) == pytest.approx(-3.0)  # a surface pushing up


def test_the_gravity_model_is_checked_against_the_arm_before_any_guarded_line():
    grav = lambda q: [3.0 * q[0], -1.0 * q[0], 0.0, 0.0]  # noqa: E731
    a = ([0.0, 0.0, 0.0, 0.0], [0.0] * 7)
    good = ([0.9 + 0.2, -0.3, 0.1, 0.0], [0.3] + [0.0] * 6)
    assert reach_press.gravity_mismatch(grav, a, good) is None
    flipped = ([-0.9, 0.3, 0.0, 0.0], [0.3] + [0.0] * 6)  # the driver's sign the other way round
    why = reach_press.gravity_mismatch(grav, a, flipped)
    assert why is not None and "joint_4" in why
    payload = ([0.9 + 1.5, -0.3, 0.0, 0.0], [0.3] + [0.0] * 6)  # something in the hand
    assert reach_press.gravity_mismatch(grav, a, payload) is not None


def _drift_of(monkeypatch, f, slide, stroke):
    calls = []

    def g(q):
        calls.append(tuple(q))
        return [f(q), 0.0, 0.0, 0.0]

    monkeypatch.setattr(reach_press, "_wrist_gravity", lambda: g)
    return reach_press._predicted_drift(SimpleNamespace(slide_pts=slide, stroke_pts=stroke)), calls


def _line(n, tag=0.0):
    return [[i / max(n - 1, 1), tag] + [0.0] * 5 for i in range(n)]


def test_the_predicted_drift_line_samples_the_gravity_model_and_never_misses_the_last_point(monkeypatch):
    # a display line computed with the arm at rest before the look leg: not
    # every point (232 gravity evaluations, ~0.2 s), the start once per line
    slide, stroke = _line(77), _line(39, tag=1.0)
    (s, t), calls = _drift_of(monkeypatch, lambda q: 3.0 * q[0], slide, stroke)
    every = reach_press.DRIFT_SAMPLE_EVERY
    for pts, tag in ((slide, 0.0), (stroke, 1.0)):
        mine = [q for q in calls if q[1] == tag]
        assert len(mine) <= 2 + math.ceil(len(pts) / every) < len(pts)
        assert mine.count(tuple(pts[0])) == 1 and len(set(mine)) == len(mine)  # the start once, nothing twice
        assert tuple(pts[-1]) in mine
    # rising along the path: the peak is at the last point, 77 and 39 not multiples of the stride
    assert s == pytest.approx(3.0) and t == pytest.approx(3.0)


def test_the_predicted_drift_on_a_smooth_path_is_the_per_point_peak(monkeypatch):
    f = lambda q: 2.0 * math.sin(math.pi * q[0])  # noqa: E731 - peaks mid-path
    slide, stroke = _line(77), _line(39)
    (s, t), _calls = _drift_of(monkeypatch, f, slide, stroke)
    full = [max(abs(f(q) - f(pts[0])) for q in pts) for pts in (slide, stroke)]
    assert s == pytest.approx(full[0], rel=0.02) and t == pytest.approx(full[1], rel=0.02)


def test_the_predicted_drift_of_a_one_point_or_empty_line_is_zero(monkeypatch):
    (s, t), _calls = _drift_of(monkeypatch, lambda q: 5.0 + q[0], _line(1), [])
    assert s == 0.0 and t == 0.0


def test_the_box_is_put_where_its_aimed_button_is(model):
    from box_opening.models.container import from_container

    for offset in ((0.0, 0.0, 0.121), (0.02, -0.01, 0.1)):
        for yaw in (0.0, 0.7, -2.0):
            pose = reach_press.cpose_for_button((0.66, -0.57, 0.40), yaw, offset)
            assert from_container(pose, offset) == pytest.approx([0.66, -0.57, 0.40], abs=1e-12)
            assert pose.yaw == yaw


def test_only_the_measured_attitude_is_pressed_reaching_out(model):
    assert reach_press.unsupported(model) is None
    other = SimpleNamespace(press_attitude_rpy_deg=(180.0, 0.0, 0.0))
    assert "measured for" in reach_press.unsupported(other)


def test_a_press_that_moved_the_box_is_measured_by_two_aims():
    assert reach_press.judge_box_moved((0.66, -0.57), (0.66, -0.57)) == 0.0
    assert reach_press.judge_box_moved((0.66, -0.57), (0.663, -0.574)) == pytest.approx(0.005)
    assert reach_press.judge_box_moved((0.66, -0.57, 0.4), (0.67, -0.57, 0.1)) == pytest.approx(0.01)  # heights ignored


def test_the_push_never_goes_deeper_than_the_gripper_clears_the_lid():
    cfg = load_press_demo(PINK)
    for tilt in reach_press.REACH_TILTS_DEG:
        depth = reach_press.reach_push_m(cfg, tilt)
        assert depth <= cfg.button_travel_m
        assert depth <= reach_press.REACH_LID_CLEAR_M[tilt] - reach_press.PUSH_MARGIN_M + 1e-12
        assert depth >= 0.004  # still past the button's own ~4 mm travel
    assert reach_press.reach_push_m(cfg, 45) < reach_press.reach_push_m(cfg, 30)


# ---- the whole reach-out press, the arm faked -------------------------------

BUTTON = (0.66, -0.57, 0.40)
POPPED = (BUTTON[0], BUTTON[1], BUTTON[2] + 0.012)  # the aim's press point on a popped knob: its top
Q = [0.0] * 7
# the rest readings, in the order the flow takes them; distinct so each
# guard's reference can be traced to the reading it must be
AT_REST = ([0.0, 0.0, 0.0, 0.0], Q)
AT_LOOK = ([0.01, 0.0, 0.0, 0.0], Q)
AT_LOOK2 = ([0.02, 0.0, 0.0, 0.0], Q)
AT_START = ([0.03, 0.0, 0.0, 0.0], Q)
FREE = [AT_REST, AT_LOOK, AT_LOOK2, AT_START, AT_START, AT_START]  # ... then 15 mm up and at the stroke's start


class _Rec:
    """What happened, in order: every leg the runner was handed, every aim."""

    def __init__(self):
        self.events = []
        self.legs = {}  # name -> the last leg of that name
        self.owl = []  # the scene OWL's ("on"/"off", events before it)
        self.reads = 0  # at-rest reads (client.settled_efforts)
        self.runs = []  # per runner.run: (its legs' names, the gains each execution it took was sent)


def _flow(
    ctx,
    monkeypatch,
    tmp_path,
    after=(POPPED, 12.0),
    first=(BUTTON, 0.4),
    scene_after="same",
    settled=None,
    touch=None,
    contact=BUTTON,
    home_refusals=0,
    run_args=None,
    chain=None,
    home_turns=0,
    closing=None,
):
    """Run run_reach_press with the arm, the cameras and the planner faked
    (`run_args`: press_demo's args, a guarded TF press by default; `chain`,
    a _Chain). Every plan starts at the joints it is handed (PlanState), as
    the planner's do; each home plan's joint_3 is written `home_turns`
    whole turns off them, as a plan handed planner_branch(start) can be.
    `closing`: press_demo's press:close, sent before the look and pending
    on the runner; joining it reads this (ok, knuckle, stalled), and the
    learned empty close is kept in tmp_path. Returns (exit code, the record, the chain, home plan's cpose and pad —
    and its start, the record's length and the live joints when it was
    planned)."""
    from conftest import leg as make_leg

    from box_opening.models.container import ContainerPose
    from box_opening.runtime import approach
    from box_opening.runtime.runner import Runner

    rec = _Rec()
    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    ctx.press_cfg = cfg
    chain = chain or _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    monkeypatch.setattr(reach_press, "oblique_aim_available", lambda: True)
    home = {}

    refusals = {"home": home_refusals}

    def plan(ctx_, state, name, target, world, speed):
        if name == "home" and refusals["home"] > 0:
            refusals["home"] -= 1
            raise RuntimeError(
                "planning failed for home: MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start"
            )
        start = list(state.joints)
        end = [0.0] * 7 if name == "home" else [0.05] * 7
        if name == "home":
            home.update(
                cpose=ctx_.cpose,
                pad=ctx_.contact_pad,
                start=list(start),
                at=len(rec.events),
                live=list(ctx_.client.joints()),
            )
            start[2] += 2.0 * math.pi * home_turns
            end = list(end)
            end[2] += 2.0 * math.pi * home_turns
        else:
            home.setdefault("look_cposes", []).append(ctx_.cpose)
        lg = make_leg(name, start=start, end=end, speed=speed, world=world[0])
        lg.target = target
        return lg, state

    monkeypatch.setattr(reach_press, "_plan_motion", plan)

    class R(Runner):
        def run(self, legs, execute, lookahead=None):
            for lg in legs:
                rec.events.append(lg.name)
                rec.legs[lg.name] = lg
            sent = len(getattr(self.client, "gains_sent", []))
            out = super().run(legs, execute, lookahead)
            rec.runs.append(([lg.name for lg in legs], getattr(self.client, "gains_sent", [])[sent:]))
            return out

    runner = R(ctx.client, log_dir=tmp_path)
    if closing is not None:
        from box_opening.constants import GRIPPER_CMD_CLOSED

        monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))  # never the robot's own gripper.json
        ctx.client.gripper_join = lambda handle: closing
        assert runner.start_gripper("press:close", GRIPPER_CMD_CLOSED, True)
    aims = [first, after]

    def aim(node, watcher, yaw, lid_z, prior_xy, runner_, tag):
        # each aim: (press point, knob mm[, the press point as the wrist
        # measured it, before the raise — by default the same])
        rec.events.append("aim:" + tag)
        got = aims.pop(0)
        if got is None:
            return None, "no circle", None, None
        return (got[0], 0.0), "3/3 frames", got[1], got[2] if len(got) > 2 else got[0]

    monkeypatch.setattr(reach_press, "_aim", aim)
    seq = list(FREE if settled is None else settled)

    def settled(*a, **k):
        rec.reads += 1
        return seq.pop(0) if seq else AT_START

    ctx.client.settled_efforts = settled
    # the touch trips with the fingertip exactly on `contact`
    quat = reach_press.reach_quat(ctx.model, BUTTON[:2], reach_press.REACH_TILTS_DEG[0])
    R_ = np.asarray(quat_to_mat(*quat), float)
    tips = np.asarray(contact) - R_ @ (np.asarray(reach_press.CONTACT_TOOL) - np.asarray(reach_press.TIP_MID_TOOL))
    ctx.client.contact_at = [float(v) for v in tips]
    ctx.client.exec_script = [
        ("arrived", {"message": "ok", "progress": 1.0}),  # look
        ("arrived", {"message": "ok", "progress": 1.0}),  # look, for the aimed button
        ("arrived", {"message": "ok", "progress": 1.0}),  # slide in
        touch or ("touch", {"message": "torque guard trip", "progress": 0.8, "torque_peak": 3.1}),
    ]
    fix = SimpleNamespace(pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0), top_xyz=BUTTON)
    moved_pose = ContainerPose(xyz=(BUTTON[0] + 0.05, BUTTON[1], BUTTON[2] - 0.121), yaw=0.0)
    seen = {
        "same": fix,
        "none": None,
        "moved": SimpleNamespace(pose=moved_pose, top_xyz=(BUTTON[0] + 0.05, BUTTON[1], BUTTON[2])),
    }
    # the scene OWL's enables/disables, each with how many events came before it
    owl = SimpleNamespace(
        enable=lambda: rec.owl.append(("on", len(rec.events))), disable=lambda: rec.owl.append(("off", len(rec.events)))
    )
    locator = SimpleNamespace(
        owl=owl, prime=owl.enable, locate=lambda s: (rec.events.append("scene"), seen[scene_after])[1]
    )
    args = SimpleNamespace(execute=True, press_only=True)
    code = reach_press.run_reach_press(None, ctx, cfg, runner, SimpleNamespace(), run_args or args, fix, locator)
    return code, rec, chain, home


def test_the_reach_out_press_flies_in_presses_unloads_judges_and_only_then_goes_home(ctx, monkeypatch, tmp_path):
    code, rec, chain, home = _flow(ctx, monkeypatch, tmp_path)
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        "reach:re-aim",
        "reach:slide-in",
        "press:down",
        "press:push",
        "reach:unload",
        "reach:rise",
        "reach:slide-out",
        "aim:reach-after",
        "scene",
        "home",
    ]
    assert home["pad"] > 0.0  # contact happened: home pads the box for a shove
    # each guard is judged against the free-air reading it must be: never a loaded first sample
    L = rec.legs
    assert L["reach:re-aim"].guard.reference == AT_LOOK and L["reach:re-aim"].guard.touch_nm == reach_press.DRAG_NM
    assert L["reach:slide-in"].guard.reference == AT_LOOK2 and L["reach:slide-in"].guard.hang is None
    assert L["press:down"].guard.reference == AT_START
    for name in ("reach:unload", "reach:rise", "reach:slide-out"):
        g = L[name].guard
        assert g.reference == AT_START, name  # the same pose, read before the touch
        assert g.gravity is not None and g.hang == (_Gravity.K, reach_press.HANG_N), name
    assert L["reach:unload"].guard.touch_nm == ctx.model.touch_nm  # starts on the button: the backstop only
    assert L["reach:slide-out"].guard.touch_nm == reach_press.DRAG_NM
    # the push's backstop bounds the TOTAL off the free-air reading at the
    # stroke's start, not what it adds to the touch's loaded first sample
    gp = L["press:push"].guard
    assert gp.trip == "press" and gp.touch_nm == ctx.model.touch_nm
    assert gp.gravity is not None and gp.reference == AT_START
    assert L["press:push"].recoil_via is L["press:down"].traj  # the recoil carries on up the touch
    # the push flown is the lid-clear depth, not button_travel_m (the fake fk stands at z 0.45)
    cfg = load_press_demo(PINK)
    depth = reach_press.reach_push_m(cfg, reach_press.REACH_TILTS_DEG[0])
    assert depth < cfg.button_travel_m
    assert any(xyz[2] == pytest.approx(0.45 - depth) for xyz, _q in chain.lines)
    assert not any(xyz[2] == pytest.approx(0.45 - cfg.button_travel_m) for xyz, _q in chain.lines)
    # the first stop on the way up is UNLOAD_CHECK_M above where the arm stood
    assert any(xyz[2] == pytest.approx(0.45 + reach_press.UNLOAD_CHECK_M) for xyz, _q in chain.lines)
    # every world from the aim on holds the box where the aim put it
    assert _reach_world_button(ctx) == pytest.approx(list(BUTTON))
    # toward the box, or loaded, at CONTACT_SPEED; up and away after an at-rest check, faster
    from box_opening.constants import CONTACT_SPEED, FREE_LINE_SPEED

    assert {n: L[n].speed for n in L if n.startswith(("reach:", "press:")) and n != "reach:look"} == {
        "reach:re-aim": CONTACT_SPEED,
        "reach:slide-in": CONTACT_SPEED,
        "press:down": cfg.press_speed,
        "press:push": cfg.press_speed,
        "reach:unload": CONTACT_SPEED,
        "reach:rise": FREE_LINE_SPEED,
        "reach:slide-out": FREE_LINE_SPEED,
    }
    assert CONTACT_SPEED < FREE_LINE_SPEED < cfg.press_speed
    # the slide and the stroke reach_from_here solved are flown as solved: the arm stood at their starts
    quat = reach_press.reach_quat(ctx.model, BUTTON[:2], reach_press.REACH_TILTS_DEG[0])
    start = reach_press.tool_for_contact((BUTTON[0], BUTTON[1], BUTTON[2] + reach_press.REACH_START_M), quat)
    bottom = reach_press.tool_for_contact((BUTTON[0], BUTTON[1], BUTTON[2] - cfg.travel_m), quat)
    # plan_reach's and reach_from_here's (and the rise's, back up to the start), none from the live joints
    assert sum(x == pytest.approx(start) for x, _q in chain.lines) == 3
    assert sum(x == pytest.approx(bottom) for x, _q in chain.lines) == 2


def test_a_solved_line_is_flown_as_solved_only_from_where_it_starts(ctx, monkeypatch):
    from box_opening.runtime import approach

    chain = _Chain()
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    pts = [[0.1] * 7, [0.2] * 7]
    ctx.client.live = [0.1 + 0.0009] + [0.1] * 6
    assert reach_press._line_from(ctx, pts, [0.9, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]) == (pts, None) and chain.lines == []
    ctx.client.live = [0.1 + 0.002] + [0.1] * 6  # drifted: solved again from where the arm is
    got, why = reach_press._line_from(ctx, pts, [0.9, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert got[0] == ctx.client.live and chain.lines == [([0.9, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0])]


def test_the_scene_owl_is_enabled_before_the_post_press_wrist_aim(ctx, monkeypatch, tmp_path):
    # its first bbox (~1.2 s) comes while the wrist re-aims, not after it
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path)
    assert code == 0
    assert rec.events == [
        "reach:look",
        "aim:reach-aim",
        "reach:re-aim",
        "reach:slide-in",
        "press:down",
        "press:push",
        "reach:unload",
        "reach:rise",
        "reach:slide-out",
        "aim:reach-after",
        "scene",
        "home",
    ]
    ons = [i for word, i in rec.owl if word == "on"]
    assert ons and min(ons) <= rec.events.index("aim:reach-after")


def test_an_aim_that_raises_after_the_early_enable_disables_the_scene_owl(ctx, monkeypatch):
    # the enable is latched: a Ctrl-C in the re-aim must not leave the scene OWL inferring
    ctx.model = ContainerModel.load(PINK)
    calls = []
    owl = SimpleNamespace(enable=lambda: calls.append("on"), disable=lambda: calls.append("off"))
    locator = SimpleNamespace(owl=owl, prime=owl.enable, locate=lambda s: calls.append("scene"))

    def aim(node, watcher, yaw, lid_z, prior_xy, runner_, tag):
        assert tag == "reach-after"
        raise KeyboardInterrupt

    monkeypatch.setattr(reach_press, "_aim", aim)
    fix = SimpleNamespace(top_xyz=BUTTON)
    with pytest.raises(KeyboardInterrupt):
        reach_press._after_check(None, ctx, SimpleNamespace(), SimpleNamespace(), fix, locator, BUTTON, None, 0.0)
    assert calls == ["on", "off"]


def test_both_reach_out_aims_take_their_own_budget_and_two_agreeing_hits(monkeypatch):
    """The look's aim and the guarded run's after-press one both go through
    _aim: REACH_AIM_S, not the straight-down STAGING_FIX_S whose 2.5 s left
    16:23 2026-10-27 0.72 s to spare, and a fix on REACH_AIM_HITS agreeing
    hits. The _flow harness replaces _aim outright, so this calls it."""
    from box_opening.tasks import press_demo

    asked = []

    def aim(node, watcher, yaw, **kw):
        asked.append(kw)
        kw["out"].update(button_up_mm=0.9, measured_xyz=(0.7213, -0.297, 0.2914))
        return ((0.7213, -0.297, 0.3095), yaw), "2 of 2 hits agree in 9 frames"

    monkeypatch.setattr(press_demo, "aim_button_at_staging", aim)
    for tag in ("reach-aim", "reach-after"):
        got, status, up, measured = reach_press._aim(None, None, 0.6, 0.3125, (0.7248, -0.286), None, tag)
        assert (
            got == (reach_press.aim_forward((0.7213, -0.297, 0.3095)), 0.6)
            and up == 0.9
            and measured == (0.7213, -0.297, 0.2914)
        )
    assert [(k["tag"], k["timeout_s"], k["min_hits"]) for k in asked] == [
        ("reach-aim", reach_press.REACH_AIM_S, reach_press.REACH_AIM_HITS),
        ("reach-after", reach_press.REACH_AIM_S, reach_press.REACH_AIM_HITS),
    ]
    assert all(k["lid_z"] == 0.3125 and k["prior_xy"] == (0.7248, -0.286) for k in asked)
    assert reach_press.REACH_AIM_S > press_demo.STAGING_FIX_S and reach_press.REACH_AIM_HITS == 2


def test_every_reach_out_aim_waits_for_a_still_arm_and_the_record_says_how_long(monkeypatch, tmp_path):
    """2026-10-28 16:23: the look's aim took its first two hits 0.32 and
    0.41 s in, the arm creeping back from past the look, the knob -7.2 and
    -5.5 mm above its lid where the pair that agreed after read +1.1 and
    +0.7; the aim's own still test (4 mm and 0.02 rad between frames)
    passed them. Each aim — the look's, the one after the press,
    knob_look's recheck alike — now starts once the arm stands still
    (runner.wait_still), noted 'still' before its tag."""
    import json

    from conftest import FakeClient

    from box_opening.runtime.runner import Runner
    from box_opening.tasks import press_demo

    order = []

    class _Settles(FakeClient):
        def wait_still(self):
            order.append("still")
            return 0.42, False

    def aim(node, watcher, yaw, **kw):
        order.append(kw["tag"])
        kw["out"].update(button_up_mm=0.9, measured_xyz=(0.7213, -0.297, 0.2914))
        return ((0.7213, -0.297, 0.3095), yaw), "2 of 2 hits agree in 9 frames"

    monkeypatch.setattr(press_demo, "aim_button_at_staging", aim)
    runner = Runner(_Settles(), log_dir=tmp_path)
    for tag in ("reach-aim", "reach-after"):
        got, _status, _up, _measured = reach_press._aim(None, None, 0.6, 0.3125, (0.7248, -0.286), runner, tag)
        assert got == (reach_press.aim_forward((0.7213, -0.297, 0.3095)), 0.6)
    assert order == ["still", "reach-aim", "still", "reach-after"]
    rows = [json.loads(line) for line in next(tmp_path.glob("run-*.jsonl")).read_text().splitlines()]
    assert [(r["leg"], r["before"], r["settle_s"], r["timed_out"]) for r in rows] == [
        ("still", "reach-aim", 0.42, False),
        ("still", "reach-after", 0.42, False),
    ]


def _reach_world_button(ctx):
    from box_opening.models.container import from_container

    kw = [kw for kind, kw in ctx.worlds.pushes if kind == "interaction" and kw.get("tag") == "reach"][-1]
    return from_container(kw["cpose"], ctx.model.button_offset)


def test_the_world_follows_the_aimed_button_not_the_scene_cameras(ctx, monkeypatch, tmp_path):
    aimed = (BUTTON[0] + 0.03, BUTTON[1], BUTTON[2] - 0.02)
    code, rec, _c, home = _flow(
        ctx,
        monkeypatch,
        tmp_path,
        first=(aimed, 0.4),
        after=((aimed[0], aimed[1], aimed[2] + 0.012), 12.0),
        contact=aimed,
    )
    assert _reach_world_button(ctx) == pytest.approx(list(aimed))
    assert code == 0


def test_when_no_straight_line_reaches_the_aimed_look_pose_the_planner_plans_it(ctx, monkeypatch, tmp_path):
    monkeypatch.setattr(reach_press, "reach_from_here", lambda *a, **k: (None, None, "IK did not converge"))
    code, rec, _c, home = _flow(ctx, monkeypatch, tmp_path)
    assert code == 0 and rec.events[:4] == ["reach:look", "aim:reach-aim", "reach:look", "reach:slide-in"]
    from box_opening.models.container import from_container

    assert from_container(home["look_cposes"][-1], ctx.model.button_offset) == pytest.approx(list(BUTTON))


def test_a_home_refused_from_inside_the_boxs_padded_volume_backs_off_toward_the_arm_first(
    ctx, monkeypatch, tmp_path, capsys
):
    """2026-10-22: the scene camera's box 85 mm off; the aim moved the box
    under the arm and every plan from the look pose was INVALID_START. The
    back-off says what the planner refused: INVALID_START_STATE_JOINT_LIMITS
    takes it too, and that start is not inside the box's padded volume."""
    code, rec, chain, _h = _flow(
        ctx, monkeypatch, tmp_path, home_refusals=1, settled=[AT_REST, ([3.0, 0.0, 0.0, 0.0], Q)]
    )  # a gravity mismatch: home from the look pose
    assert code == 1 and rec.events == ["reach:look", "reach:back-off", "home"]
    said = capsys.readouterr().out
    assert (
        "the planner refused home from here (MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start) — "
        "backing 15 cm straight off toward the arm first"
    ) in said
    assert "inside the box's padded volume" not in said
    back = rec.legs["reach:back-off"]
    assert back.plan_free and back.guard.trip == "obstruction" and back.guard.gravity is not None
    from box_opening.constants import FREE_LINE_SPEED

    assert back.speed == FREE_LINE_SPEED and back.guard.touch_nm == reach_press.DRAG_NM and back.guard.hang is None
    # straight back toward the arm, level: from the fake fk's (0.5, -0.4, 0.45), against the box's bearing
    xyz, _q = chain.lines[-1]
    b = np.asarray(ctx.cpose.xyz[:2]) / np.hypot(*ctx.cpose.xyz[:2])
    assert np.subtract([0.5, -0.4], xyz[:2]) == pytest.approx(reach_press.REACH_PULLBACK_M * b) and xyz[
        2
    ] == pytest.approx(0.45)


def test_a_home_refused_for_a_joint_on_its_limit_steps_off_it_first(ctx, tmp_path, capsys):
    """Flatbot 2026-10-27 14:33: the depth press ended over the box with
    arm_joint_6 0.043 rad from its limit, home was refused
    START_IN_LIMIT_MARGIN, and the arm held until home_arm stepped it off.
    The reach-out's own home steps off first, as home_arm and try_home do:
    plan-free along the joint line toward the rest pose, the shortest step
    home plans from, and home planned from where it ends."""
    from conftest import runner as make_runner

    from box_opening.constants import PLANNER_LIMIT_MARGIN_RAD
    from box_opening.primitives.core import CLEAR_MARGIN_FIRST_RAD
    from box_opening.runtime import approach
    from box_opening.tasks import press_demo

    cfg = load_press_demo(PINK)
    ctx.press_cfg = cfg
    rest = press_demo.rest_joints(cfg)
    lo, hi = approach._chain().limits[5]
    start = list(rest)
    start[5] = hi - 0.043  # arm_joint_6 where 14:33 left it
    ctx.client.live = list(start)

    class InMargin:
        success = False
        message = "START_IN_LIMIT_MARGIN: start: arm_joint_6 is 0.043 rad from its position limit"
        trajectory = None

    real = ctx.client.plan_to_joints

    def plan(q7, start_joints):  # refuses any start inside its margin, as rammp-curobo does
        if min(start_joints[5] - lo, hi - start_joints[5]) < PLANNER_LIMIT_MARGIN_RAD:
            return InMargin()
        return real(q7, start_joints)

    ctx.client.plan_to_joints = plan
    assert reach_press._go_home(ctx, cfg, make_runner(ctx.client, tmp_path), "STOP: test") is True
    said = capsys.readouterr().out
    assert "joint_6 is 0.043 rad from its limit" in said and "a plan-free step" in said and "holds" not in said
    # nothing planned from the refused start: home from the step's end, just clear of the margin
    end = ctx.client.joint_starts[0]
    assert hi - end[5] == pytest.approx(CLEAR_MARGIN_FIRST_RAD, abs=1e-6)
    assert len(ctx.client.executed) == 2 and ctx.client.live == pytest.approx(rest)


@pytest.mark.parametrize("off, flown", [(0.0, True), (0.039, True), (0.041, False)])
def test_a_home_planned_ahead_flies_as_planned_only_from_where_it_starts(ctx, tmp_path, capsys, off, flown):
    """_go_home handed home's legs planned before the arm stood here (the
    full run's, while the way back to the look pose flew): within
    DRIFT_REPLAN_RAD of where they start — joint_3 a whole turn round is
    where it starts, the short way round — they fly as planned and the
    planner is not asked; beyond it, home is planned from where the arm
    stands, as with none. Either way in the run log."""
    import json

    from conftest import leg as make_leg
    from conftest import runner as make_runner

    from box_opening.tasks import press_demo

    cfg = load_press_demo(PINK)
    ctx.press_cfg = cfg
    rest = list(press_demo.rest_joints(cfg))
    look = [0.3, -0.9, 0.4, -1.9, 0.2, -1.0, 1.5]
    ahead = [make_leg("home", start=look, end=rest, speed=1.0)]
    here = [v + (off if j == 1 else 2.0 * math.pi if j == 2 else 0.0) for j, v in enumerate(look)]
    ctx.client.live = list(here)
    assert reach_press._go_home(ctx, cfg, make_runner(ctx.client, tmp_path), "STOP: test", ahead=ahead) is True
    if flown:
        assert ctx.client.joint_starts == [] and ctx.client.exec_starts == [pytest.approx(look)]
    else:
        assert ctx.client.joint_starts == [pytest.approx(here)] and ctx.client.exec_starts == [pytest.approx(here)]
        assert "home is planned again from here" in capsys.readouterr().out
    assert ctx.client.live == pytest.approx(rest)
    rows = [json.loads(line) for f in sorted(tmp_path.glob("run-*.jsonl")) for line in open(f)]
    (note,) = (r for r in rows if r["leg"] == "home_ahead")
    assert note["flown"] is flown and note["off_rad"] == pytest.approx(off, abs=1e-6)


def test_an_open_box_is_not_reached_for(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, first=(POPPED, 14.0))
    assert code == 1 and "reach:slide-in" not in rec.events and rec.events[-1] == "home"


def test_a_press_that_does_not_pop_the_knob_is_a_failure(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, after=(BUTTON, 0.5))
    assert code == 1 and rec.events[-1] == "home"


def test_when_neither_camera_finds_the_box_after_the_press_the_arm_holds(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, after=None, scene_after="none")
    assert code == 1 and "home" not in rec.events and rec.events[-1] == "scene"


def test_a_moved_box_stops_the_run_and_home_is_planned_around_where_it_now_is(ctx, monkeypatch, tmp_path):
    moved = (POPPED[0] + 0.012, POPPED[1], POPPED[2])
    code, rec, _c, home = _flow(ctx, monkeypatch, tmp_path, after=(moved, 12.0))
    assert code == 1 and rec.events.index("aim:reach-after") < rec.events.index("home")
    from box_opening.models.container import from_container

    # the popped knob's top is not the box's height: it is placed by its xy, at the button's height
    assert from_container(home["cpose"], ctx.model.button_offset) == pytest.approx([moved[0], moved[1], BUTTON[2]])


def test_a_box_the_scene_camera_saw_move_stops_the_run_even_when_the_wrist_sees_it_in_place(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, scene_after="moved")
    assert code == 1 and rec.events[-1] == "home"


def test_home_is_planned_around_the_scene_cameras_box_when_the_wrist_cannot_find_it(ctx, monkeypatch, tmp_path):
    code, rec, _c, home = _flow(ctx, monkeypatch, tmp_path, after=None, scene_after="moved")
    assert code == 1 and home["cpose"].xyz[0] == pytest.approx(BUTTON[0] + 0.05)


def test_a_lid_that_came_up_or_went_down_is_a_moved_box(ctx, monkeypatch, tmp_path):
    tipped = (POPPED[0], POPPED[1], POPPED[2] + 0.015)  # the knob reads 12 mm up, its lid 15 mm higher
    code, _rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, after=(tipped, 12.0))
    assert code == 1


@pytest.mark.parametrize("first_raise,after_raise", [(0.021, 0.0), (0.0, 0.021), (0.021, 0.019)])
def test_the_lid_after_the_press_is_judged_by_what_the_wrist_measured(
    ctx, monkeypatch, tmp_path, first_raise, after_raise
):
    """Flatbot 2026-10-27: the wrist reads its table and lid ~20 mm low, and
    the aim's press point is raised by that table offset where its hits
    agree on one (button_aim.ButtonAimer.fix) — the touch is judged against
    it. Whether the box lifted is the wrist against itself: an aim raised
    before the press and one not raised after it (its offset past the xy
    bound) read no lift, nor two raised by offsets 2 mm apart."""
    import json

    first = (BUTTON, 0.4, (BUTTON[0], BUTTON[1], BUTTON[2] - first_raise))
    measured_after = (POPPED[0], POPPED[1], POPPED[2] - first_raise)  # the same box, the knob up
    after = ((POPPED[0], POPPED[1], measured_after[2] + after_raise), 12.0, measured_after)
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, first=first, after=after)
    assert code == 0 and rec.events[-3:] == ["aim:reach-after", "scene", "home"]
    rows = {r["leg"]: r for r in map(json.loads, open(next(tmp_path.glob("run-*.jsonl"))))}
    assert rows["touch"]["verdict"] == "press" and rows["touch"]["offset_mm"] == pytest.approx(0.0, abs=0.1)
    assert rows["reach_after"]["lid_dz_mm"] == pytest.approx(0.4, abs=0.05)  # the first aim's knob 0.4 mm up


def test_a_touch_that_finds_no_surface_backs_out_before_going_home(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, touch=("arrived", {"message": "ok", "progress": 1.0}))
    assert code == 1 and "press:push" not in rec.events
    assert rec.events[-4:] == [
        "reach:unload",
        "reach:rise",
        "reach:slide-out",
        "home",
    ]  # nothing touched: no after-check


def test_a_foreign_surface_is_not_pressed_and_the_box_is_judged_before_home(ctx, monkeypatch, tmp_path):
    high = (BUTTON[0], BUTTON[1], BUTTON[2] + 0.040)
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, contact=high, after=(BUTTON, 0.4))
    assert code == 1 and "press:push" not in rec.events
    assert rec.events[-6:] == ["reach:unload", "reach:rise", "reach:slide-out", "aim:reach-after", "scene", "home"]


def test_a_box_latched_on_the_finger_is_lifted_no_more_than_the_first_stop(ctx, monkeypatch, tmp_path):
    k = _Gravity.K
    hanging = ([0.03 + 3.0 * k[0], 0.0, 3.0 * k[2], 0.0], Q)  # 3 N pulling down at the fingertip
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, settled=FREE[:4] + [hanging])
    assert code == 1 and rec.events[-1] == "reach:unload"  # stopped 15 mm up: no rise, no slide out, no home


def test_anything_left_on_the_hand_at_the_strokes_start_holds_the_arm(ctx, monkeypatch, tmp_path):
    # no pull at all, but joint_6 is 1 Nm off its free-air reading
    off = ([0.03, 0.0, 0.0, 1.0], Q)
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, settled=FREE[:5] + [off])
    assert code == 1 and rec.events[-1] == "reach:rise"


def test_a_gravity_model_that_does_not_fit_the_arm_stops_before_any_guarded_line(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, settled=[AT_REST, ([3.0, 0.0, 0.0, 0.0], Q)])
    assert code == 1 and rec.events == ["reach:look", "home"]


@pytest.mark.parametrize("soft", [False, True])
def test_without_the_oblique_aim_nothing_moves(ctx, monkeypatch, tmp_path, soft):
    """Nor, in a compliant run, is the joint impedance primed: this refusal
    comes first, and its "nothing moved" stays true."""
    from box_opening.models.container import ContainerPose
    from box_opening.perception import button_aim
    from box_opening.runtime.runner import Runner

    class OldAimer:  # the straight-down aim's constructor
        def __init__(self, grab, model, table_z):
            pass

    monkeypatch.setattr(button_aim, "ButtonAimer", OldAimer)
    assert not reach_press.oblique_aim_available()
    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    fix = SimpleNamespace(pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0), top_xyz=BUTTON)
    args = SimpleNamespace(execute=True, press_only=True, compliant=soft, depth_press=soft)
    code = reach_press.run_reach_press(None, ctx, cfg, Runner(ctx.client, log_dir=tmp_path), None, args, fix, None)
    assert code == 1 and not ctx.client.executed and ctx.client.primes == []


@pytest.mark.parametrize("soft", [False, True])
def test_the_reach_out_preview_moves_nothing(ctx, monkeypatch, tmp_path, capsys, soft):
    """Guarded, or compliant with its press proven: the preview flies
    nothing, and primes no joint impedance either."""
    from box_opening.models.container import ContainerPose
    from box_opening.runtime import approach
    from box_opening.runtime.runner import Runner

    cfg = load_press_demo(PINK)
    ctx.model = ContainerModel.load(PINK)
    monkeypatch.setattr(approach, "_chain", lambda: _Chain())
    monkeypatch.setattr(approach, "_gravity", lambda: _Gravity())
    monkeypatch.setattr(reach_press, "oblique_aim_available", lambda: True)
    monkeypatch.setattr(
        reach_press,
        "_plan_motion",
        lambda c, s, n, t, w, sp: (
            SimpleNamespace(
                goal_joints=[0.0] * 7,
                name=n,
                traj=None,
                speed=sp,
                world="full",
                kind=None,
                guard=None,
                warp=None,
                plan_s=0.0,
                plan_server_s=None,
                waypoint=None,
                target=t,
                verify=None,
            ),
            s,
        ),
    )
    monkeypatch.setattr(Runner, "preview", lambda self, legs: "preview")
    if soft:
        monkeypatch.setattr(reach_press, "way_down", _way_down([], solves=lambda tilt, tip: True))
    fix = SimpleNamespace(pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0), top_xyz=BUTTON)
    args = SimpleNamespace(execute=False, press_only=True, compliant=soft, depth_press=soft)
    code = reach_press.run_reach_press(None, ctx, cfg, Runner(ctx.client, log_dir=tmp_path), None, args, fix, None)
    assert code == 0 and not ctx.client.executed and ctx.client.primes == []
    out = capsys.readouterr().out
    assert "gravity alone moves joint_4 0.00 Nm along the slide" in out
    assert ("COMPLIANT: every move" in out) is soft and "enters its joint impedance" not in out


def test_a_compliant_run_says_which_gains_every_move_flies(ctx, monkeypatch, capsys):
    """Owner 2026-10-27: "impedance mode needs to be firmer"; 2026-10-28:
    "extreamly rigid". The banner names the tuning in force — the client's
    compliant_gains, a custom spec (RIGID, with its own zeta) or a driver
    profile — and the client is back in position once the run is over; a
    guarded run has no banner."""
    from box_opening.constants import GAINS_MEDIUM

    during = []
    monkeypatch.setattr(reach_press, "_reach_press", lambda *a: during.append(ctx.client.compliant) or 0)
    args = SimpleNamespace(compliant=True, depth_press=True, execute=True)
    assert reach_press.run_reach_press(None, ctx, None, None, None, args, None, None) == 0
    out = capsys.readouterr().out
    assert (
        "COMPLIANT: every move in the arm's joint impedance, RIGID (Kq 400/400 N·m/rad on joints 1-4/5-7, "
        "zeta 1.0), and nothing the run reads stops it"
    ) in out
    ctx.client.compliant_gains = GAINS_MEDIUM
    reach_press.run_reach_press(None, ctx, None, None, None, args, None, None)
    assert "impedance, MEDIUM (Kq 80/30 N·m/rad on joints 1-4/5-7), and" in capsys.readouterr().out
    assert during == [True, True] and not ctx.client.compliant
    reach_press.run_reach_press(
        None, ctx, None, None, None, SimpleNamespace(compliant=False, depth_press=True), None, None
    )
    assert "COMPLIANT" not in capsys.readouterr().out and during[-1] is False


def test_a_stop_says_what_stopped_the_leg_not_always_something_met():
    """15:34 2026-10-27: the re-aim's guard lost its stream and the STOP
    line said the move met something; nothing had been touched."""
    from box_opening.runtime.runner import LegResult

    lost = LegResult("reach:re-aim", "failed", False, "effort stream lost during guarded leg — cancelled")
    assert reach_press._stopped_by([lost]) == "stopped (effort stream lost during guarded leg — cancelled)"
    assert reach_press._stopped_by([LegResult("reach:re-aim", "touch", False, "torque guard trip")]) == "met something"


# ---- the compliant depth press, proven before the arm moves toward the box ------------------


def _depth_press_flow(ctx, monkeypatch, tmp_path, way_down, compliant=True, **kw):
    """_flow with --depth-press — compliant, as press_demo runs by default
    since 2026-10-27, unless `compliant` is False (--guarded) — the way down
    `way_down`'s, the press itself faked: the guarded press, and a compliant
    one's hold at the end of its one-execution way in, which judges it at
    the aimed button the box was put at (ctx.cpose). Returns _flow's, and
    what the press was handed ({} when it was not called)."""
    from box_opening.models.container import from_container
    from box_opening.tasks import depth_press as dp

    got = {}

    def press(
        node,
        ctx_,
        cfg,
        runner,
        watcher,
        contact_tool,
        here=None,
        turned=None,
        knob_up_mm=None,
        aimed=None,
        compliant=False,
        top_z=None,
    ):
        got.update(aimed=aimed, top_z=top_z, compliant=compliant)
        return dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=14.0)

    def hold(ctx_, cfg_, runner_, stroke, top_z, seen):
        got.update(aimed=from_container(ctx_.cpose, ctx_.model.button_offset), top_z=top_z, compliant=True)
        return dp.DepthPress(free=True, contact=True, felt=True, judged=False, press_n=14.0)

    monkeypatch.setattr(dp, "press_down_from_here", press)
    monkeypatch.setattr(dp, "compliant_hold", hold)
    monkeypatch.setattr(reach_press, "way_down", way_down)
    kw.setdefault("touch", ("arrived", {"message": "ok", "progress": 1.0}))  # no TF stroke: the 4th motion is the clear
    args = SimpleNamespace(execute=True, press_only=True, depth_press=True, compliant=compliant)
    return (*_flow(ctx, monkeypatch, tmp_path, run_args=args, **kw), got)


@pytest.mark.parametrize("ik, look_plans", [(False, 4), (True, 0)], ids=["planner's looks", "own IK's looks"])
def test_a_scene_button_no_tilt_presses_even_30_mm_off_stops_before_anything_moves(
    ctx, monkeypatch, tmp_path, capsys, ik, look_plans
):
    """16:23 2026-10-27 flew the look, the re-aim and a 7.4 s slide in
    before it asked for the turn down, which no IK reaches there. Now the
    proof refuses at the rest pose: every tilt is named, then the button
    PROOF_SHIFT_M nearer the arm and to either side, at the smallest
    tilt's look, and nothing is sent to the arm but the hold where it
    stands that primes its joint impedance (client.prime_compliant). With
    the arm's own IK solving every look (`ik`) the proof comes before the
    planner and no look is planned (14:28:12 2026-10-28 planned four);
    without, each look is the planner's pose plan, as before."""
    asked = []
    code, rec, _chain, h, got = _depth_press_flow(
        ctx, monkeypatch, tmp_path, _way_down(asked), chain=_IKChain() if ik else None
    )
    assert code == 1 and rec.events == [] and ctx.client.executed == [] and got == {}
    assert len(h.get("look_cposes", [])) == look_plans
    # the joint impedance was primed while the proof was planned: a hold
    # where the arm stands, which the run leaves it in
    assert ctx.client.primes == [[0.0] * 7]
    out = capsys.readouterr().out
    assert (
        "STOP: no reach-out press for the scene camera's box — no tilt turns the hand down and strokes there, "
        "nor with the button 30 mm nearer the arm or to either side (30 deg: the way down (walk in: "
    ) in out
    assert out.rstrip().endswith("; nothing moved")
    for tilt in reach_press.REACH_TILTS_DEG:
        assert "%d deg: the way down (walk in: " % tilt in out
    for name in reach_press.SHIFTED:
        assert "30 mm %s: the way down (walk in: " % name in out
    assert [t for t, _xy, _z in asked] == list(reach_press.REACH_TILTS_DEG) + [30] * 3
    u = np.asarray(BUTTON[:2]) / np.hypot(*BUTTON[:2])
    across = np.array([-u[1], u[0]])  # to its left, seen from the arm
    for (_t, xy, z), want in zip(
        asked[4:], (BUTTON[:2] - 0.03 * u, BUTTON[:2] + 0.03 * across, BUTTON[:2] - 0.03 * across), strict=False
    ):
        assert xy == pytest.approx(list(want), abs=1e-9) and z == BUTTON[2]


def _refusals(tmp_path):
    import json

    return [r for f in tmp_path.glob("run-*.jsonl") for r in map(json.loads, open(f)) if r["leg"] == "reach_refused"]


def test_a_refused_proof_is_noted_in_the_record_word_for_word_as_its_stop_line(ctx, monkeypatch, tmp_path, capsys):
    """14:28:12, 14:29:19 and 14:29:54 2026-10-28 ended here, before the arm
    moved: the reason printed, the record ending at the scene fix. It is
    noted now, as the STOP line says it."""
    code, rec, _chain, _h, _got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down([]))
    assert code == 1 and rec.events == [] and ctx.client.executed == []
    (note,) = _refusals(tmp_path)
    assert note["why"].startswith("no reach-out press for the scene camera's box — no tilt turns the hand down")
    assert "[press_demo] STOP: %s; nothing moved" % note["why"] in capsys.readouterr().out


def test_a_box_no_look_plans_for_is_noted_in_the_record(ctx, monkeypatch, tmp_path, capsys):
    chain = _Chain(refuse=[lambda xyz, quat: True])  # no straight line anywhere: no tilt slides in
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, chain=chain)
    assert code == 1 and rec.events == [] and ctx.client.executed == []
    (note,) = _refusals(tmp_path)
    assert note["why"].startswith(
        "no reach-out press plans for the scene camera's box (30 deg: the slide in (IK did " "not converge)"
    )
    assert "[press_demo] STOP: %s; nothing moved" % note["why"] in capsys.readouterr().out


def test_a_guarded_reach_with_no_steady_reading_at_rest_is_noted_in_the_record(ctx, monkeypatch, tmp_path):
    code, rec, _c, _h = _flow(ctx, monkeypatch, tmp_path, settled=[None])
    assert code == 1 and rec.events == [] and ctx.client.executed == []
    (note,) = _refusals(tmp_path)
    assert note["why"].startswith("no steady joint efforts at the rest pose")


def test_a_reach_out_this_install_cannot_aim_is_noted_in_the_record(ctx, monkeypatch, tmp_path):
    from box_opening.models.container import ContainerPose
    from box_opening.runtime.runner import Runner

    monkeypatch.setattr(reach_press, "oblique_aim_available", lambda: False)
    ctx.model = ContainerModel.load(PINK)
    fix = SimpleNamespace(pose=ContainerPose(xyz=(BUTTON[0], BUTTON[1], BUTTON[2] - 0.121), yaw=0.0), top_xyz=BUTTON)
    args = SimpleNamespace(execute=True, press_only=False, compliant=True, depth_press=True)
    runner = Runner(ctx.client, log_dir=tmp_path)
    assert reach_press.run_reach_press(None, ctx, load_press_demo(PINK), runner, None, args, fix, None) == 1
    (note,) = _refusals(tmp_path)
    assert note["why"] == (
        "no reach-out press — this install's wrist aim cannot look at the lid obliquely "
        "(perception/button_aim takes no lid_z/prior_xy)"
    )
    assert not ctx.client.executed and ctx.client.primes == []


@pytest.mark.parametrize("ik, look_plans", [(False, 4), (True, 1)], ids=["planner's looks", "own IK's looks"])
def test_a_scene_button_the_press_reaches_30_mm_off_flies_the_look_and_the_aimed_button_decides(
    ctx, monkeypatch, tmp_path, capsys, ik, look_plans
):
    """As on flatbot's URDF for a scene button at r 0.766 m, az -15 deg
    (test_on_flatbot_a_scene_button_out_of_reach_by_a_little_still_flies_the_look):
    no tilt presses it, 30 mm nearer the arm the press reaches — and the
    aim, 9 mm nearer and 21 mm aside, too. The look flies, and the aimed
    button's own way down, solved after the aim, takes the press in — the
    re-aim, the slide in, the turn and the stroke as one execution. With
    the arm's own IK solving every look (`ik`) the one look planned is the
    one that flies (plan_shifted_look); without, each was planned."""
    import json

    from box_opening.tasks import depth_press as dp

    u = np.asarray(BUTTON[:2]) / np.hypot(*BUTTON[:2])
    aimed = (BUTTON[0] - 0.02 * u[0], BUTTON[1] - 0.02 * u[1], BUTTON[2])
    off_scene = lambda tilt, tip: np.hypot(tip[0] - BUTTON[0], tip[1] - BUTTON[1]) > 0.015  # noqa: E731
    asked = []
    code, rec, _chain, h, got = _depth_press_flow(
        ctx, monkeypatch, tmp_path, _way_down(asked, off_scene), first=(aimed, 0.4), chain=_IKChain() if ik else None
    )
    assert code == 0 and len(h["look_cposes"]) == look_plans
    way_in = ["reach:re-aim", "reach:slide-in", "depth:turn", "depth:press"]
    assert rec.events == ["reach:look", "aim:reach-aim", *way_in, "depth:clear", "home"]
    assert (way_in, [dp.COMPLIANT_GAINS]) in rec.runs
    assert got["aimed"] == pytest.approx(aimed) and got["top_z"] == pytest.approx(BUTTON[2]) and got["compliant"]
    out = capsys.readouterr().out
    assert ("the press does not reach the scene camera's button itself (30 deg: the way down (walk in: ") in out
    assert "but at 30 deg it does with the button 30 mm nearer the arm — inside what the aim corrects" in out
    # the scene button at each tilt, the first shifted one, then the aimed button after the aim
    assert [t for t, _xy, _z in asked] == list(reach_press.REACH_TILTS_DEG) + [30, 30]
    assert asked[-1][1] == pytest.approx(list(aimed[:2]), abs=1e-9)
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (plan,) = (r for r in rows if r["leg"] == "reach_plan")
    assert plan["proof"] == "shifted" and plan["down"] is None and plan["tilt_deg"] == 30
    (way,) = (r for r in rows if r["leg"] == "reach_way_in")
    assert (way["tilt_deg"], way["walked"], way["short_deg"], way["k"]) == (30, False, 0.0, 440.0)
    assert way["end_z"] == pytest.approx(BUTTON[2] - 0.03)


def test_where_no_straight_line_reaches_the_aimed_look_pose_the_planned_one_brings_its_own_proof(
    ctx, monkeypatch, tmp_path
):
    """The planner plans the move to the aimed button's look pose where no
    straight line flies there, as before; plan_reach proves the way down
    from that leg's end, which is where the arm then stands — no second
    solve, no re-aim after it: from there the slide in, the turn and the
    stroke fly as one execution."""
    import json

    from box_opening.tasks import depth_press as dp

    asked = []
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52])  # the move to the look pose (0.54), not the slide (0.47)
    code, rec, _c, _h, got = _depth_press_flow(
        ctx, monkeypatch, tmp_path, _way_down(asked, lambda tilt, tip: True), chain=chain
    )
    assert code == 0 and got["aimed"] == pytest.approx(BUTTON)
    way_in = ["reach:slide-in", "depth:turn", "depth:press"]
    assert rec.events == ["reach:look", "aim:reach-aim", "reach:look", *way_in, "depth:clear", "home"]
    assert (way_in, [dp.COMPLIANT_GAINS]) in rec.runs
    assert [t for t, _xy, _z in asked] == [30, 30]  # the scene button's, then the planned look's for the aimed one
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (way,) = (r for r in rows if r["leg"] == "reach_way_in")
    assert way["tilt_deg"] == 30 and way["walked"] is False


def test_a_planned_look_the_arm_stands_off_flies_its_way_in_leg_by_leg(ctx, monkeypatch, tmp_path, capsys):
    """After the planner's look leg the way in starts at that leg's planned
    end. An arrival a few mrad off it (impedance) is not jumped from: the
    way in flies leg by leg, each line solved again from where the arm
    stands, as before the one execution — the slide in from the live
    joints, then the press, which turns and strokes from there."""
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52])  # the move to the look pose (0.54), not the slide (0.47)
    real = ctx.client.execute

    def execute(traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        out = real(traj, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)
        if len(ctx.client.executed) == 2:  # the planned look for the aimed button: 5 mrad short on joint_2
            ctx.client.live[1] -= 0.005
        return out

    monkeypatch.setattr(ctx.client, "execute", execute)
    code, rec, _c, _h, got = _depth_press_flow(
        ctx, monkeypatch, tmp_path, _way_down([], lambda tilt, tip: True), chain=chain
    )
    assert code == 0 and got["compliant"] and got["aimed"] == pytest.approx(BUTTON)  # the press, handed the aim
    assert rec.events == ["reach:look", "aim:reach-aim", "reach:look", "reach:slide-in", "depth:clear", "home"]
    assert "the way in does not fly as one execution (the arm stands 0.005 rad from where reach:slide-in starts)" in (
        capsys.readouterr().out
    )
    slide = rec.legs["reach:slide-in"]
    assert list(slide.traj.points[0].positions) == pytest.approx([0.05, 0.045] + [0.05] * 5)  # from the live joints
    assert (["reach:slide-in"], [None]) in rec.runs and slide.gains is None  # a goal of its own, as before


@pytest.mark.parametrize(
    "soft, stops, code, asked_after",
    [
        (True, None, 0, []),  # home: nothing let go after it, nor between its goals
        (
            True,
            "depth:clear",
            1,
            [4],
        ),  # the look, the planned look, the slide in, then the clear: stopped 40% of the way
        (True, "home", 1, [5]),  # ... home from the clear's end stopped
        (False, "depth:clear", 1, []),  # --guarded: its clear stopped as before, nothing to let go of
    ],
    ids=["home", "clear stopped", "home stopped", "guarded"],
)
def test_a_reach_out_flown_leg_by_leg_lets_go_where_its_home_from_over_the_box_stopped(
    ctx, monkeypatch, tmp_path, soft, stops, code, asked_after
):
    """A compliant way in flown leg by leg (the planned look the arm stands
    off) presses and rises in press_compliant, then goes home from over the
    box (_home_from_over_box): the clear up and back, then home, goals of
    their own. Either stopped part-way leaves the springs pulling toward its
    next point, its feed-forward on: the run lets go once, after that goal,
    and the arm holds there. Home, nothing is let go. A guarded run's
    POSITION goals hold where they stopped: it lets nothing go."""
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52]) if soft else None  # the planned look, as above
    asked = []
    ctx.client.release = lambda: asked.append(len(ctx.client.executed)) or True
    nth = {"depth:clear": 4, "home": 5}.get(stops)
    real = ctx.client.execute

    def execute(traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        if len(ctx.client.executed) + 1 == nth:
            ctx.client.exec_script.insert(0, ("failed", {"message": "PATH_TOLERANCE_VIOLATED", "progress": 0.4}))
        out = real(traj, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)
        if soft and len(ctx.client.executed) == 2:  # the planned look for the aimed button: 5 mrad short on joint_2
            ctx.client.live[1] -= 0.005
        return out

    monkeypatch.setattr(ctx.client, "execute", execute)
    got, rec, _c, _h, pressed = _depth_press_flow(
        ctx, monkeypatch, tmp_path, _way_down([], lambda tilt, tip: True), compliant=soft, chain=chain
    )
    assert got == code and asked == asked_after and pressed["compliant"] == soft
    assert rec.events[3:] == ["reach:slide-in", "depth:clear", "home"][: 2 if stops == "depth:clear" else 3]


STOPPED = ("failed", {"message": "PATH_TOLERANCE_VIOLATED", "progress": 0.4})
REJECTED = ("failed", {"message": "goal rejected by the driver", "progress": 0.0})


def _goal_ends(ctx, monkeypatch, ends, drift_after=None):
    """The fake client with each `n`th goal sent (the look's the first)
    scripted to end as `ends[n]`; an arrival of goal `drift_after` stands 5
    mrad short of its end on joint_2, as an impedance arrival can."""
    real = ctx.client.execute

    def execute(traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        n = len(ctx.client.executed) + 1
        if n in ends:
            ctx.client.exec_script.insert(0, (ends[n][0], dict(ends[n][1])))
        out = real(traj, speed, guard=guard, while_running=while_running, stop_when=stop_when, gains=gains)
        if n == drift_after and out[0] == "arrived":
            ctx.client.live[1] -= 0.005
        return out

    monkeypatch.setattr(ctx.client, "execute", execute)


@pytest.mark.parametrize(
    "soft, way, nth, end, asked_after, flown",
    [
        (
            True,
            "planned look",
            1,
            STOPPED,
            [1],
            ["reach:look"],
        ),  # the look for the scene camera's button, 40% of the way
        (
            True,
            "planned look",
            1,
            REJECTED,
            [],
            ["reach:look"],
        ),  # ... turned down by the driver: at rest where it stood
        (True, "planned look", 2, STOPPED, [2], ["reach:look", "aim:reach-aim", "reach:look"]),  # the aimed button's
        (True, "planned look", 2, REJECTED, [], ["reach:look", "aim:reach-aim", "reach:look"]),
        (True, "planned look", 3, STOPPED, [3], ["reach:look", "aim:reach-aim", "reach:look", "reach:slide-in"]),
        (True, "planned look", 3, REJECTED, [], ["reach:look", "aim:reach-aim", "reach:look", "reach:slide-in"]),
        (True, "re-aim", 2, STOPPED, [2], ["reach:look", "aim:reach-aim", "reach:re-aim"]),
        (True, "re-aim", 2, REJECTED, [], ["reach:look", "aim:reach-aim", "reach:re-aim"]),
        (True, "re-aim", 3, STOPPED, [3], ["reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in"]),
        (False, "re-aim", 1, STOPPED, [], ["reach:look"]),  # --guarded: its POSITION goals hold where they stopped
        (False, "re-aim", 2, STOPPED, [], ["reach:look", "aim:reach-aim", "reach:re-aim"]),
        (False, "re-aim", 3, STOPPED, [], ["reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in"]),
    ],
    ids=[
        "look stopped",
        "look not started",
        "planned look stopped",
        "planned look not started",
        "slide in stopped",
        "slide in not started",
        "re-aim stopped",
        "re-aim not started",
        "slide in after the re-aim stopped",
        "guarded look",
        "guarded re-aim",
        "guarded slide in",
    ],
)
def test_a_reach_out_flown_leg_by_leg_lets_go_where_a_leg_before_the_press_stopped_part_way(
    ctx, monkeypatch, tmp_path, soft, way, nth, end, asked_after, flown
):
    """Before the press a compliant run flies the look for the scene
    camera's button, then — where its way in flies leg by leg (the planned
    look the arm stands off, or a re-aim that does not make one execution
    with the rest) — the planned look for the aimed button or the re-aim,
    then the slide in toward the box: each a goal of its own, unguarded.
    One stopped part-way leaves the springs pulling toward its next point,
    its feed-forward on, as the one execution's stop does: the run lets go
    once, after that goal, and the arm holds there, nothing pressed. One
    the driver turned down moved nothing: the arm stands at rest at a
    leg's end, and nothing is let go. A guarded run lets nothing go."""
    chain = None
    if way == "planned look":
        chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52])  # the planned look, as above
    elif soft:
        monkeypatch.setattr(reach_press, "_not_one_execution", lambda ctx_, legs: "its legs do not meet")
    asked = []
    ctx.client.release = lambda: asked.append(len(ctx.client.executed)) or True
    _goal_ends(ctx, monkeypatch, {nth: end}, drift_after=2 if way == "planned look" else None)
    got, rec, _c, _h, pressed = _depth_press_flow(
        ctx, monkeypatch, tmp_path, _way_down([], lambda tilt, tip: True), compliant=soft, chain=chain
    )
    assert got == 1 and asked == asked_after and pressed == {}  # nothing pressed
    assert rec.events == flown and len(ctx.client.executed) == nth  # the arm holds: no home


@pytest.mark.parametrize(
    "soft, refusals, ends, asked_after, flown",
    [
        (True, 0, {}, [], ["home"]),  # home: nothing let go
        (True, 0, {2: STOPPED}, [2], ["home"]),  # home from the look pose stopped 40% of the way
        (True, 0, {2: REJECTED}, [], ["home"]),  # ... turned down by the driver: at rest at the look pose
        (True, 1, {2: STOPPED}, [2], ["reach:back-off"]),  # home refused INVALID_START: the back-off stopped
        (True, 1, {3: STOPPED}, [3], ["reach:back-off", "home"]),  # ... it flew, then home stopped
        (False, 0, {2: STOPPED}, [], ["home"]),  # --guarded: its home stopped as before, nothing to let go of
        (False, 1, {2: STOPPED}, [], ["reach:back-off"]),
    ],
    ids=[
        "home",
        "home stopped",
        "home not started",
        "back-off stopped",
        "home after the back-off stopped",
        "guarded home",
        "guarded back-off",
    ],
)
def test_a_compliant_home_from_the_look_pose_lets_go_where_it_stopped_part_way(
    ctx, monkeypatch, tmp_path, soft, refusals, ends, asked_after, flown
):
    """A STOP at the look pose — here an open box, not reached for — sends
    the arm home from free air (_go_home): home planned, or straight back
    off toward the arm first where the planner refuses home INVALID_START.
    Either stopped part-way leaves the springs pulling toward its next
    point, its feed-forward on: the run lets go once, after that goal.
    Turned down by the driver, or home, the arm stands at rest at a leg's
    end, and nothing is let go. A guarded run lets nothing go."""
    asked = []
    ctx.client.release = lambda: asked.append(len(ctx.client.executed)) or True
    _goal_ends(ctx, monkeypatch, ends)
    got, rec, _c, _h, pressed = _depth_press_flow(
        ctx,
        monkeypatch,
        tmp_path,
        _way_down([], lambda tilt, tip: True),
        compliant=soft,
        first=(POPPED, 14.0),
        home_refusals=refusals,
    )
    assert got == 1 and asked == asked_after and pressed == {}
    assert rec.events == ["reach:look", "aim:reach-aim", *flown]


def test_where_neither_a_line_nor_a_planned_look_brings_the_aimed_button_a_way_down_the_arm_goes_home_from_the_look(
    ctx, monkeypatch, tmp_path, capsys
):
    """No straight line reaches the aimed button's look pose, and at no
    tilt the planner offers does the hand turn down and stroke there: the
    arm goes home from the look pose, in free air. No planned look leg for
    the aimed button flies, no slide in, nothing pressed; the console says
    why at every tilt."""
    import json

    asked = []
    scene_only = lambda tilt, tip: len(asked) == 1  # noqa: E731 - plan_reach's, for the scene camera's button
    chain = _Chain(refuse=[lambda xyz, quat: xyz[2] > 0.52])  # the move to the look pose (0.54), not the slide (0.47)
    code, rec, _c, _h, got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down(asked, scene_only), chain=chain)
    assert code == 1 and got == {}
    assert rec.events == ["reach:look", "aim:reach-aim", "home"]
    assert [t for t, _xy, _z in asked] == [30] + list(reach_press.REACH_TILTS_DEG)  # the scene's, then each planned
    out = capsys.readouterr().out
    assert (
        "STOP: no press reaches the aimed button (as straight lines: the move to its look pose (IK did not "
        "converge); planned: 30 deg: the way down (walk in: "
    ) in out
    assert all("%d deg: the way down (walk in: " % t in out for t in reach_press.REACH_TILTS_DEG)
    assert "— nothing was pressed — returning home" in out
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    assert not [r for r in rows if r["leg"] in ("reach_way_in", "home_refused")]


@pytest.mark.parametrize("closing", [(True, 0.79, False), (False, 0.636, False)])
def test_a_compliant_run_reads_the_fingers_close_before_its_way_in(ctx, monkeypatch, tmp_path, capsys, closing):
    """press_demo sends press:close before the look, and the runner joins
    it before a gripper leg or a guarded motion only. A compliant way in
    has neither, so its close was collected after home, unread. Now it is
    read once the aim has the button, before anything moves toward the
    box: the press:close row comes before the way in's. A close that never
    moved, its knuckle reading an empty close (0.636, after a driver
    restart, 2026-09-25, which press_demo did not take for shut), found the
    fingers shut: the press goes on, and 0.636 is kept as the empty close."""
    import json

    from box_opening.runtime import fingers
    from box_opening.tasks import depth_press as dp

    ok, knuckle, _stalled = closing
    code, rec, _c, _h, got = _depth_press_flow(
        ctx, monkeypatch, tmp_path, _way_down([], lambda tilt, tip: True), closing=closing
    )
    assert code == 0 and got["compliant"] and ctx.client.gripper_sent == [0.8]
    assert any("depth:press" in names and gains == [dp.COMPLIANT_GAINS] for names, gains in rec.runs)
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    legs = [r["leg"] for r in rows]
    assert legs.count("press:close") == 1 and legs.index("press:close") < legs.index("reach_way_in")
    (close,) = (r for r in rows if r["leg"] == "press:close")
    assert close["ok"] is ok and close["gripper_pos"] == pytest.approx(knuckle)
    unmoved = [r for r in rows if r["leg"] == "closed_unmoved"]
    assert fingers.learned_closed() == pytest.approx(fingers.NOMINAL_CLOSED if ok else knuckle)
    assert (unmoved == []) if ok else (unmoved[0]["knuckle"] == knuckle and unmoved[0]["learned"] is True)
    shut = (
        "press:close failed (gripper at 0.636 (overlapped)), but the knuckle reads an empty close: the fingers "
        "are shut (kept as this gripper's empty close)"
    )
    assert (shut in capsys.readouterr().out) is (not ok)


@pytest.mark.parametrize("soft", [True, False])
@pytest.mark.parametrize(
    "closing, read",
    [
        ((False, 0.0, False), "failed (gripper at 0.000 (overlapped))"),
        ((True, 0.41, True), "arrived (gripper at 0.410 (stalled) (overlapped))"),
    ],
)
def test_fingers_not_shut_send_the_arm_home_from_the_look_pose_with_nothing_pressed(
    ctx, monkeypatch, tmp_path, capsys, soft, closing, read
):
    """An open aperture strikes the lid and can still read "pressed";
    fingers closed on something (0.41, the knob's reading) press with it.
    Either is read after the aim, in free air at the look pose: the arm goes
    home from there, nothing moved toward the box, nothing learned. A
    guarded run's re-aim used to join a failed close and leave the arm
    holding at the look, and took a stalled one for closed."""
    from box_opening.runtime import fingers

    asked = []
    code, rec, _c, _h, got = _depth_press_flow(
        ctx, monkeypatch, tmp_path, _way_down(asked, lambda tilt, tip: True), compliant=soft, closing=closing
    )
    assert code == 1 and got == {} and rec.events == ["reach:look", "aim:reach-aim", "home"]
    if soft:
        assert len(asked) == 1  # the scene button's proof, before the look; none for the aimed one
    assert ("STOP: press:close %s — the fingers are not shut; nothing was pressed — returning home" % read) in (
        capsys.readouterr().out
    )
    assert fingers.learned_closed() == pytest.approx(fingers.NOMINAL_CLOSED)


@pytest.mark.parametrize("depth", [False, True])
def test_a_guarded_reach_out_asks_for_no_way_down(ctx, monkeypatch, tmp_path, depth):
    """--guarded, with or without the depth press, is planned as before:
    no way down is asked for, and the plan's note says it was not proven."""
    import json

    asked = []
    if depth:
        code, rec, _chain, _h, got = _depth_press_flow(ctx, monkeypatch, tmp_path, _way_down(asked), compliant=False)
        assert got["aimed"] == pytest.approx(BUTTON) and not got["compliant"]
    else:
        monkeypatch.setattr(reach_press, "way_down", _way_down(asked))
        code, rec, _chain, _h = _flow(ctx, monkeypatch, tmp_path)
        assert "press:down" in rec.events
    assert (
        code == 0
        and asked == []
        and rec.events[:4] == ["reach:look", "aim:reach-aim", "reach:re-aim", "reach:slide-in"]
    )
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (plan,) = (r for r in rows if r["leg"] == "reach_plan")
    assert plan["proof"] is None and plan["down"] is None
    assert not [r for r in rows if r["leg"] == "reach_way_in"]


# Flatbot's URDF, the scene camera's buttons of 2026-10-27 (run records' scene_fix top_xyz)
SCENE_1623 = (0.7248, -0.2860, 0.3125)
SCENE_1519 = (0.6754, -0.1812, 0.3111)


def _on_flatbot(ctx, monkeypatch):
    """Flatbot's own kinematics and the pink canister, each look leg
    'planned' to the joints client._plan_pose_near hands the planner. Pure
    kinematics: no planner, no arm."""
    from box_opening.constants import HOME
    from box_opening.primitives.look import look_joints
    from box_opening.runtime import approach
    from box_opening.runtime.branches import planner_branch
    from box_opening.runtime.kinematics import ArmChain

    chain = ArmChain(path="capabilities/box_opening/test/data/arm_flatbot_20261021.urdf", root="arm_mount")
    monkeypatch.setattr(approach, "_chain", lambda: chain)
    ctx.model = ContainerModel.load(PINK)

    def plan(ctx_, state, name, target, world, speed):
        if target[0] == "joints":  # a look posture chosen for its walks (quickest_look): planned to, as the planner's
            return SimpleNamespace(goal_joints=[float(v) for v in target[1]], target=target), state
        _kind, look, quat = target
        q = chain.nearest_solution(planner_branch(HOME), look, quat, seeds=(look_joints(HOME),))
        if q is None:
            raise RuntimeError("planning failed for %s: MotionGenStatus.IK_FAIL" % name)
        return SimpleNamespace(goal_joints=[float(v) for v in q], target=target), state

    monkeypatch.setattr(reach_press, "_plan_motion", plan)
    return chain, load_press_demo(PINK), list(HOME)


def test_on_flatbot_1623s_scene_button_is_refused_before_the_look(ctx, monkeypatch):
    """16:23 2026-10-27, replayed with the leans its turn tried then
    (TURN_SHORT_DEG, the guarded press's): the plan without the proof takes
    the tilted stroke at 30 deg, as the run did — 20.2 s from the look to
    home with nothing pressed, no straight-down hand at the slide's end. At
    every tilt the look pose and the slide solve, and neither the walk in
    nor the turn after the slide turns the hand down: refused, and 30 mm
    nearer the arm or to either side as well."""
    from box_opening.tasks import depth_press as dp

    monkeypatch.setattr(dp, "COMPLIANT_TURN_SHORT_DEG", dp.TURN_SHORT_DEG)
    _chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    plan, _why = reach_press.plan_reach(ctx, cfg, SCENE_1623, home)
    assert plan is not None and plan.tilt_deg == 30  # what 16:23 flew
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_1623, home, depth=True, looks=looks)
    assert plan is None
    assert why.count("the way down (walk in: ") == len(reach_press.REACH_TILTS_DEG), why
    assert all("%d deg: the way down (walk in: " % t in why for t in reach_press.REACH_TILTS_DEG)
    assert [lk.tilt_deg for lk in looks] == list(reach_press.REACH_TILTS_DEG)
    near, why = reach_press.shifted_way_down(ctx, cfg, looks[0], SCENE_1623)
    assert near is None and all("30 mm %s: the way down (walk in: " % n in why for n in reach_press.SHIFTED), why


# 15:19's aimed button (the touch met it at 0.312; the clear's replay, test_depth_press)
AIMED_1519 = (0.6617, -0.1972, 0.3111)


def _at(r_az, z=0.3125):
    """A button `r_az` (r m, az deg) from the arm, the lid at 16:23's height."""
    r, az = r_az
    return (r * math.cos(math.radians(az)), r * math.sin(math.radians(az)), z)


def test_on_flatbot_1623s_scene_button_is_proven_turning_short_after_the_slide(ctx, monkeypatch):
    """The compliant press's turn after the slide may stop up to 15 deg
    short of straight down (depth_press.COMPLIANT_TURN_SHORT_DEG): 16:23's
    scene button is proven at the first tilt, the slide in and a turn 10
    deg short, 1.77 rad of joint arc — no walk in reaches it — then the
    stroke from there, as under STIFF: RIGID's spring ~1270 N/m at the pads
    on the button's bottom (STIFF: ~505), the overshoot at its 15 mm least,
    bent 6 mm aside. 15:19's aimed button, which the guarded leans
    already reached, is as it was: one walk in, exactly down."""
    from box_opening.tasks import depth_press as dp

    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_1623, home, depth=True)
    assert why is None and plan.tilt_deg == 30 and plan.stroke_pts is None
    assert not plan.down.walked and max(dp.TURN_SHORT_DEG) < plan.down.short_deg <= 15.0
    assert plan.down.short_deg == 10.0  # as modelled before this was flown
    assert plan.down.pts[0] == pytest.approx(plan.slide_pts[-1])  # in place, where the slide ends
    arc = dp.joint_arc(plan.down.pts)
    assert arc <= dp.TURN_MAX_ARC_RAD and arc == pytest.approx(1.77, abs=0.02)
    R, _t = chain.fk(plan.down.pts[-1])
    assert math.degrees(reach_press.hand_lean(np.asarray(R, float))) == pytest.approx(plan.down.short_deg, abs=0.05)
    assert plan.stroke.q0 == pytest.approx([float(v) for v in plan.down.pts[-1]])
    assert [name for name, _pts, _speed in plan.stroke.legs] == ["depth:approach", "depth:press"]
    assert plan.stroke.k == pytest.approx(1270.0, abs=20.0) and plan.stroke.end_z < SCENE_1623[2] - cfg.button_travel_m
    assert plan.stroke.over == pytest.approx(dp.COMPLIANT_OVERSHOOT_M[0], abs=0.0005)
    assert plan.stroke.bent and plan.stroke.bend_m == pytest.approx(0.006, abs=0.001)
    plan, why = reach_press.plan_reach(ctx, cfg, AIMED_1519, home, depth=True)
    assert why is None and plan.tilt_deg == 30 and plan.down.walked and plan.down.short_deg == 0.0
    # a box the guarded leans reached only 30 mm off (r 0.766 m, az -15 deg) is reached as it stands
    plan, why = reach_press.plan_reach(ctx, cfg, _at((0.766, -15.0)), home, depth=True)
    assert why is None and plan.tilt_deg == 30 and not plan.down.walked and plan.down.short_deg == 8.0


def test_on_flatbot_1519s_scene_button_walks_in_exactly_down_at_30_deg(ctx, monkeypatch):
    """15:19 2026-10-27, the button pressed with ~13.5 N: proven at the
    first tilt, one walk in from the look pose to the hand exactly down,
    the stroke from its end past the button's bottom."""
    _chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_1519, home, depth=True)
    assert why is None and plan.tilt_deg == 30 and plan.stroke_pts is None
    assert plan.down.walked and plan.down.short_deg == 0.0
    assert plan.stroke.q0 == pytest.approx([float(v) for v in plan.down.pts[-1]])
    assert [name for name, _pts, _speed in plan.stroke.legs] == ["depth:approach", "depth:press"]
    assert plan.stroke.end_z < SCENE_1519[2] - cfg.button_travel_m


@pytest.mark.parametrize(
    "button, walked, short",
    [
        (_at((0.70, 15.0), 0.26), False, 15.0),
        (_at((0.70, -20.0)), False, 5.0),
        (SCENE_1519, True, 0.0),  # a walk in the runner flies: walked as before
    ],
)
def test_on_flatbot_the_proven_way_in_is_one_the_runner_flies(ctx, monkeypatch, tmp_path, button, walked, short):
    """The 2026-10-28 merge review of #53 + #55: the proof took any walk in
    the arm's IK solves, and the runner refuses a leg one of whose joints
    swings out and back further than its net change and SANITY_MARGIN_RAD
    (Runner._refusal). Under a lid at 0.26, r 0.70 m az +15 deg, the walk
    in 5 deg short swings joint_1 0.251 -> 0.687 -> 0.261, 0.437 rad
    against the gate's 0.361: proven, then refused once the look and the
    aim had flown ("STOP: the way in did not start (trajectory sanity gate:
    ...)"), as were 3 of the 119 walks in under the lid at 0.3125 (r
    0.68-0.72 m, az -20 deg; at r 0.70 the walk 2 deg short). The proofs
    before the look and before the way in now take only what the runner
    flies: there the slide and the turn, 15 and 5 deg short. Every leg of
    the way in passes the runner's gates."""
    from box_opening.runtime.runner import Runner

    _chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    plan, why = reach_press.plan_reach(ctx, cfg, button, home, depth=True)
    assert why is None and plan.tilt_deg == 30
    way, why, kind = reach_press.solve_way_in(ctx, cfg, button, plan.tilt_deg, plan.look_leg.goal_joints)
    assert way is not None, (why, kind)
    assert (way.down.walked, way.down.short_deg) == (plan.down.walked, plan.down.short_deg)  # as proven before the look
    assert way.down.walked is walked and way.down.short_deg == short and (way.slide is None) is walked
    runner = Runner(ctx.client, log_dir=tmp_path)
    legs = reach_press.way_in_legs(ctx, cfg, way)
    assert [(lg.name, runner._refusal(lg)) for lg in legs] == [(lg.name, None) for lg in legs]


@pytest.mark.parametrize(
    "scene, aimed, near, leans",
    [
        # to its left, the walk in 5 deg short swings joint_1 0.476 rad, past the runner's sanity gate's 0.417
        ((0.700, -22.0), (0.692, -23.6), "to its right", "guarded"),
        ((0.766, -15.0), (0.757, -16.6), "nearer the arm", "guarded"),
        ((0.700, -22.0), (0.692, -23.6), "nearer the arm", "compliant"),
    ],
)
def test_on_flatbot_a_scene_button_out_of_reach_by_a_little_still_flies_the_look(
    ctx, monkeypatch, scene, aimed, near, leans
):
    """Two boxes found offline (r m, az deg; the lid at 16:23's 0.3125)
    with the turn after the slide leaning no further than the guarded
    press's TURN_SHORT_DEG: no tilt presses the scene camera's button, but
    one PROOF_SHIFT_M off it the press reaches — and the aim's, 12-23 mm
    off as on 10-27, it does. With the compliant press's own leans the
    first is still such a box, reached 30 mm nearer the arm; the second is
    reached as it stands (test_on_flatbot_1623s_scene_button_is_proven_turning_short_after_the_slide)."""
    from box_opening.tasks import depth_press as dp

    if leans == "guarded":
        # and the turn only at the slide's end, as it was then: lower down
        # (TURN_LOW_ABOVE_M) both scene buttons are reached as they stand
        monkeypatch.setattr(dp, "COMPLIANT_TURN_SHORT_DEG", dp.TURN_SHORT_DEG)
        monkeypatch.setattr(reach_press, "TURN_LOW_ABOVE_M", ())
    _chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    looks = []
    plan, _why = reach_press.plan_reach(ctx, cfg, _at(scene), home, depth=True, looks=looks)
    assert plan is None and looks
    assert reach_press.shifted_way_down(ctx, cfg, looks[0], _at(scene)) == (near, None)
    way, why, kind = reach_press.solve_way_in(ctx, cfg, _at(aimed), looks[0].tilt_deg, looks[0].look_joints)
    assert way is not None, (why, kind)
    assert way.move and way.down.walked and way.slide is None and way.stroke.legs


# 2026-10-28 14:28:35: the scene camera's button, and the wrist's aim from the look its proof flew
SCENE_142835 = (0.6030, -0.4942, 0.3123)
AIMED_142835 = (0.6137, -0.4988, 0.3123)  # the aim's 0.307, pressed from the scene camera's higher top


def test_on_flatbot_a_turn_the_slides_end_cannot_reach_is_turned_lower_down(ctx, monkeypatch, tmp_path):
    """14:28:35 2026-10-28: the scene camera's button proven at 30 deg, the
    turn after the slide 15 deg short — COMPLIANT_TURN_SHORT_DEG's last —
    then the aim put the button 12 mm farther out, where no turn at the
    slide's end flies (joint_4 runs straight 60 mm above the button: the IK
    does not converge), and the arm went home from the look. The slide now
    carries on straight down and the hand turns 45 mm above the button:
    15 deg short, every leg of the way in one the runner flies, the stroke
    bent as before. Where the slide's own end turns, nothing changes."""
    from box_opening.runtime.runner import Runner

    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_142835, home, depth=True)
    assert why is None and plan.tilt_deg == 30 and not plan.down.walked and plan.down.short_deg == 15.0
    assert plan.down.slide is None and plan.down.above_m is None  # turned at the slide's end, as the run did
    q_look = plan.look_leg.goal_joints
    monkeypatch.setattr(reach_press, "TURN_LOW_ABOVE_M", ())
    way, why, kind = reach_press.solve_way_in(ctx, cfg, AIMED_142835, 30, q_look)
    assert way is None and kind == "down" and "15 deg short: IK did not converge 100% of the way" in why, why
    monkeypatch.undo()
    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    way, why, kind = reach_press.solve_way_in(ctx, cfg, AIMED_142835, 30, q_look)
    assert way is not None, (why, kind)
    assert not way.down.walked and way.down.short_deg == 15.0 and way.down.above_m == 0.045
    # the slide flown is the look's slide carried on down: it ends where the turn starts, the fingertip 45 mm up
    assert way.slide is way.down.slide and way.plan.slide_pts is way.slide
    assert way.down.pts[0] == pytest.approx(way.slide[-1])
    R, t = chain.fk(way.slide[-1])
    tip = np.asarray(t, float) + np.asarray(R, float) @ np.asarray(reach_press.CONTACT_TOOL)
    assert tip == pytest.approx([AIMED_142835[0], AIMED_142835[1], AIMED_142835[2] + 0.045], abs=5e-4)
    assert way.plan.start_xyz == pytest.approx([float(v) for v in t], abs=1e-4)  # the line's IK converges to 5e-5
    assert way.stroke.bent and way.stroke.end_z < AIMED_142835[2] - cfg.button_travel_m
    runner = Runner(ctx.client, log_dir=tmp_path)
    legs = reach_press.way_in_legs(ctx, cfg, way)
    assert [lg.name for lg in legs] == ["reach:re-aim", "reach:slide-in", "depth:turn", "depth:approach", "depth:press"]
    assert [(lg.name, runner._refusal(lg)) for lg in legs] == [(lg.name, None) for lg in legs]


# 2026-10-28's scene camera boxes that no look press reached (run records' scene_fix top_xyz)
SCENE_142812 = (0.5353, -0.6901, 0.3089)  # r 0.873 m: four looks planned, then a stop
SCENE_142919 = (1.1304, 0.0910, 0.3223)  # r 1.134 m: four IK_FAILs


def _counted(monkeypatch):
    """_on_flatbot's planner, each look plan's tilt kept in the list returned."""
    asked = []
    real = reach_press._plan_motion

    def plan(ctx_, state, name, target, world, speed):
        asked.append(_tilt_of(target[2]))
        return real(ctx_, state, name, target, world, speed)

    monkeypatch.setattr(reach_press, "_plan_motion", plan)
    return asked


def test_on_flatbot_the_look_is_planned_once_its_press_is_proven_and_as_it_always_was(ctx, monkeypatch):
    """15:19's scene button, proven at the first tilt: one plan, for the
    joints the client hands the planner for that look pose, and the same
    proof from them as before — the slide, the walk in and the stroke.
    14:28:12's box, 0.87 m out, had four looks planned before the run
    stopped: none now, nor for the shifted press, which no look reaches
    either."""
    from box_opening.constants import HOME
    from box_opening.primitives.look import look_joints
    from box_opening.runtime.branches import planner_branch

    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    asked = _counted(monkeypatch)
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_1519, home, depth=True)
    assert why is None and plan.tilt_deg == 30 and asked == [30]
    quat, start, _bottom, look = reach_press.reach_geometry(ctx.model, SCENE_1519, 30, cfg.travel_m)
    q = chain.nearest_solution(planner_branch(HOME), look, quat, seeds=(look_joints(HOME),))
    assert plan.look_leg.goal_joints == [float(v) for v in q] == plan.look_joints
    slide, _why = chain.straight_line(q, start, quat)
    down, stroke, _why = reach_press.way_down(ctx, cfg, [float(v) for v in q], slide, start, quat, SCENE_1519[2])
    assert np.array_equal(np.asarray(plan.slide_pts), np.asarray(slide))
    assert plan.down.walked and np.array_equal(np.asarray(plan.down.pts), np.asarray(down.pts))
    assert [(n, np.asarray(p).tolist()) for n, p, _s in plan.stroke.legs] == [
        (n, np.asarray(p).tolist()) for n, p, _s in stroke.legs
    ]
    asked.clear()
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_142812, home, depth=True, looks=looks)
    assert plan is None and asked == [] and "30 deg: the slide in (" in why
    assert all("%d deg: the way down (" % t in why for t in (35, 40, 45))
    assert [lk.tilt_deg for lk in looks] == [35, 40, 45] and all(lk.look_leg is None for lk in looks)
    plan, near, why = reach_press.plan_shifted_look(ctx, cfg, looks, SCENE_142812, home)
    assert plan is None and near is None and asked == [] and why.startswith("at 35 deg: 30 mm nearer the arm: ")


def test_on_flatbot_a_look_past_the_arms_reach_is_not_handed_to_the_planner(ctx, monkeypatch):
    """14:29:19 2026-10-28, a box seen 1.13 m out: no IK of the arm's
    solved a look, so the client handed the planner each look pose, and
    it answered IK_FAIL four times (~1.15 s). The wrist stands 99, 73 and
    47 mm past everything the arm reaches at 30-40 deg: neither IK is
    asked. At 45 deg it stands 21 mm past, inside what cuRobo still calls
    reached — 5 mm and 5.7 deg, the wrist 0.2875 m back from tool_frame —
    so that pose still goes to the planner."""
    from box_opening.constants import HOME

    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    asked = _counted(monkeypatch)
    looks = []
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_142919, home, depth=True, looks=looks)
    assert plan is None and looks == [] and asked == [45]
    for tilt, mm in ((30, 99), (35, 73), (40, 47)):
        assert "%d deg: the look pose (its wrist %d mm past the arm's reach)" % (tilt, mm) in why, why
    assert "45 deg: the look pose (MotionGenStatus.IK_FAIL)" in why
    # the margin over what cuRobo still calls reached: 5 mm, and 5.7 deg turning the wrist about tool_frame
    p = chain.points(list(HOME))
    lever = float(np.linalg.norm(p[-1] - p[5]))  # tool_frame to joint_6's origin
    assert reach_press.LOOK_PAST_REACH_M > 0.005 + 2.0 * lever * 0.05


def test_the_look_planned_is_the_request_the_client_always_sent_for_it(ctx, monkeypatch):
    """Through the client itself, its round trip to the planner stood in
    for: the proven look goes out as it always did — PlanToJoints to the
    joints the client's own IK picks for the pose (_plan_pose_near), from
    HOME's side of the wrap — and only once, with its world; for a box no
    look presses nothing goes to the planner, not even the world."""
    from test_branches import _Plan

    from box_opening.primitives import core
    from box_opening.runtime.branches import nearest_branch, planner_branch
    from box_opening.runtime.client import PlannerClient

    _chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    monkeypatch.setattr(reach_press, "_plan_motion", core._plan_motion)
    c = PlannerClient.__new__(PlannerClient)
    c._plan_pose, c._plan_joints = "pose", "joints"
    sent, worlds = [], []

    def call(kind, goal):
        sent.append((kind, goal))
        if kind == "joints":
            return _Plan([list(goal.start_joints), list(goal.target_joints)])
        return _Plan([], success=False)

    c._call = call
    c.set_world = lambda path: (worlds.append(path), (True, "ok"))[1]
    ctx.client = c
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_1519, home, depth=True)
    assert why is None and plan.tilt_deg == 30
    ((kind, goal),) = sent
    start = planner_branch(home)
    assert kind == "joints" and list(goal.start_joints) == pytest.approx(start, abs=1e-12)
    assert list(goal.target_joints) == pytest.approx(nearest_branch(plan.look_joints, start), abs=1e-12)
    assert plan.look_leg.goal_joints == pytest.approx(plan.look_joints, abs=1e-12) and worlds == ["full.yaml"]
    sent.clear()
    worlds.clear()
    plan, why = reach_press.plan_reach(ctx, cfg, SCENE_142812, home, depth=True, looks=[])
    assert plan is None and sent == [] and worlds == []


# ---- the look's posture for the shortest walks (plan_reach shortest_walk) ----------

# 2026-10-28/29's scene camera boxes and the table each was seen on (run records' scene_fix top_xyz, table_z_scene)
SCENE_162353 = ((0.5921, -0.2869, 0.3123), 0.2105)  # r 0.66 m, az -26 deg
SCENE_154524 = ((0.7214, -0.2167, 0.3126), 0.2146)  # r 0.75 m
SCENE_120459 = ((0.6135, -0.2267, 0.3115), 0.2129)  # 10-29 12:04, r 0.65 m
SCENE_142121 = ((0.7169, -0.1311, 0.3131), 0.2116)  # r 0.73 m
SCENE_133716 = ((0.7459, -0.1532, 0.3098), 0.2162)  # 10-29 13:37, r 0.76 m


def _seen(ctx, scene):
    """The scene camera's box as run_reach_press has it: ctx.cpose at the
    scene fix (its origin on the table), the table the scene camera
    measured (press_demo.adopt_scene_table). Returns its button."""
    from box_opening.models.container import ContainerPose

    top, table_z = scene
    ctx.cpose = ContainerPose(xyz=(top[0], top[1], table_z), yaw=0.0)
    ctx.worlds.table_top_z = table_z
    return top


def _walk(plan):
    """The look's walk to the hand turned down (reach_press.walk_legs) as one path."""
    return [q for pts in reach_press.walk_legs(plan) for q in pts]


def _guarded(plan):
    """What posture_refusal holds another posture's plan to: its walk, then its stroke."""
    return reach_press.walk_legs(plan) + [pts for _name, pts, _speed in plan.stroke.legs]


def _every_proof(monkeypatch):
    """quickest_look's choice whatever the machine's load: every one of its
    LOOK_POSTURE_PROOFS proven (the budget itself:
    test_the_posture_search_never_starts_a_proof_the_ik_time_left_would_not_hold)."""
    monkeypatch.setattr(reach_press, "LOOK_POSTURE_BUDGET_S", math.inf)


def _targets(monkeypatch):
    """reach_press._plan_motion as it is, each target it is asked kept in the list returned."""
    asked = []
    real = reach_press._plan_motion

    def plan(ctx_, state, name, target, world, speed):
        asked.append(target)
        return real(ctx_, state, name, target, world, speed)

    monkeypatch.setattr(reach_press, "_plan_motion", plan)
    return asked


@pytest.mark.parametrize(
    "scene", [SCENE_162353, SCENE_154524, SCENE_120459], ids=["10-28 16:23", "10-28 15:45", "10-29 12:04"]
)
def test_on_flatbot_the_look_goes_to_the_elbow_whose_walks_are_quickest(ctx, monkeypatch, tmp_path, scene):
    """2026-10-29 13:37: the walks between the tilted look and the hand
    turned down over the button were ~15 s of a 34.5 s run, each 1.9-2.4
    rad of joint travel for ~16 cm of the hand's, from the posture HOME's
    nearest IK picks for the look. Asked for the shortest walks (the proof
    before the look), the look holds the same pose — the same view — with
    the elbow swung elsewhere: at these boxes each walk is ~1 s quicker as
    a compliant run flies it, 2.7-2.9 s over the three walks with the
    look's longer transit taken off, its joint arc down by over 0.5 rad.
    The look is planned once, a joint goal to that posture; the way down
    is proven from it as it was from today's, it passed the guard, and the
    way in from where the look ends is one the runner flies."""
    from box_opening.constants import COMPLIANT_FREE_LINE_SPEED
    from box_opening.runtime.runner import Runner
    from box_opening.tasks import depth_press as dp

    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    _every_proof(monkeypatch)
    button = _seen(ctx, scene)
    today, why = reach_press.plan_reach(ctx, cfg, button, home, depth=True)
    assert why is None and today.posture is None  # not asked: as it always was
    asked = _targets(monkeypatch)
    plan, why = reach_press.plan_reach(ctx, cfg, button, home, depth=True, shortest_walk=True)
    assert why is None and plan.tilt_deg == today.tilt_deg == 30
    p = plan.posture
    assert p["chosen"] > 0 and p["saved_s"] >= 2.5 and p["proven"] >= 1, p
    assert p["walk_s_today"] == round(reach_press.flown_s(reach_press.walk_legs(today), COMPLIANT_FREE_LINE_SPEED), 2)
    assert p["walk_s"] == round(reach_press.flown_s(reach_press.walk_legs(plan), COMPLIANT_FREE_LINE_SPEED), 2)
    assert p["walk_s"] <= p["walk_s_today"] - 0.9 and p["transit_s"] > p["transit_s_today"]
    assert dp.joint_arc(_walk(plan)) <= dp.joint_arc(_walk(today)) - 0.5
    # the same look pose, another posture
    R0, t0 = chain.fk(today.look_joints)
    R1, t1 = chain.fk(plan.look_joints)
    assert np.asarray(t1) == pytest.approx(np.asarray(t0), abs=1e-4)
    assert np.asarray(R1) == pytest.approx(np.asarray(R0), abs=1e-4)
    gap = float(np.abs(np.asarray(plan.look_joints) - np.asarray(today.look_joints)).max())
    assert gap > 0.5 and p["joint_gap_rad"] == pytest.approx(gap, abs=1e-3)
    # planned once, a joint goal to it; proven from it and guarded
    assert asked == [("joints", plan.look_joints)] and plan.look_leg.goal_joints == plan.look_joints
    assert plan.slide_pts[0] == pytest.approx(plan.look_joints) and plan.down.pts and plan.stroke.legs
    assert reach_press.posture_refusal(ctx, chain, _guarded(plan), button) is None
    way, why, kind = reach_press.solve_way_in(ctx, cfg, button, plan.tilt_deg, plan.look_leg.goal_joints)
    assert way is not None, (why, kind)
    runner = Runner(ctx.client, log_dir=tmp_path)
    legs = reach_press.way_in_legs(ctx, cfg, way)
    assert [(lg.name, runner._refusal(lg)) for lg in legs] == [(lg.name, None) for lg in legs]


@pytest.mark.parametrize("scene", [SCENE_142121, SCENE_133716], ids=["10-28 14:21", "10-29 13:37"])
def test_on_flatbot_where_no_other_elbow_is_quicker_by_enough_the_look_is_todays_exactly(ctx, monkeypatch, scene):
    """10-28 14:21 (where the research pass of 10-28 found no gain) and
    10-29 13:37: the best other elbow proven saves 0.6-0.7 s over the look's
    transit and the three walks, under LOOK_POSTURE_GAIN_S — inside what an
    estimate timing each walk alone cannot tell. The look goes where it
    always did: today's posture exactly, planned as the pose goal it always
    was, the slide, the way down and the stroke the same proof. Only with the
    margin under that saving does the look move."""
    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    _every_proof(monkeypatch)
    button = _seen(ctx, scene)
    today, _why = reach_press.plan_reach(ctx, cfg, button, home, depth=True)
    asked = _targets(monkeypatch)
    plan, why = reach_press.plan_reach(ctx, cfg, button, home, depth=True, shortest_walk=True)
    assert why is None and plan.posture["chosen"] == 0 and plan.posture["proven"] >= 1, plan.posture
    assert plan.posture["walk_s"] == plan.posture["walk_s_today"] and plan.posture["saved_s"] == 0.0
    assert plan.look_joints == today.look_joints and plan.look_leg.goal_joints == today.look_leg.goal_joints
    quat, _start, _bottom, look = reach_press.reach_geometry(ctx.model, button, 30, cfg.travel_m)
    assert asked == [("pose", look, quat)]  # as _plan_look always asked it
    assert np.array_equal(np.asarray(plan.slide_pts), np.asarray(today.slide_pts))
    assert np.array_equal(np.asarray(_walk(plan)), np.asarray(_walk(today)))
    assert [(n, np.asarray(p).tolist()) for n, p, _s in plan.stroke.legs] == [
        (n, np.asarray(p).tolist()) for n, p, _s in today.stroke.legs
    ]
    monkeypatch.setattr(reach_press, "LOOK_POSTURE_GAIN_S", 0.3)
    plan, why = reach_press.plan_reach(ctx, cfg, button, home, depth=True, shortest_walk=True)
    assert why is None and plan.posture["chosen"] > 0 and 0.3 <= plan.posture["saved_s"] < 1.0, plan.posture


@pytest.mark.parametrize("near", ["table", "box"])
def test_on_flatbot_an_elbow_whose_centre_line_comes_near_the_table_or_the_box_is_refused(ctx, monkeypatch, near):
    """Nothing collision-checks the plan-free walks, and the planner's world
    is empty: another elbow is taken only where its centre line — joint_3's
    origin to joint_7's, along the walk and the stroke — stays
    POSTURE_CLEAR_M above the table and out of the box padded by it (a
    minimal guard, not a collision check; today's posture is not held to
    it, as it never was). 16:23's quickest elbow with the table measured 5
    mm too high for it, or a box standing 10 mm too tall under its elbow
    where that stands lowest: refused, and the look goes to a posture that
    passes, or to today's."""
    from box_opening.models.container import ContainerPose

    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    _every_proof(monkeypatch)
    button = _seen(ctx, SCENE_162353)
    plan, _why = reach_press.plan_reach(ctx, cfg, button, home, depth=True, shortest_walk=True)
    k = plan.posture["chosen"]
    assert k > 0
    rows = [chain.points(q)[2:7] for pts in _guarded(plan) for q in pts]  # joint_3's origin to joint_7's
    if near == "table":
        ctx.worlds.table_top_z = min(float(r[:, 2].min()) for r in rows) - reach_press.POSTURE_CLEAR_M + 0.005
        said = "above the table between"
    else:
        elbow = min((r[1] for r in rows), key=lambda e: float(e[2]))  # joint_4's origin, lowest
        z = float(elbow[2]) - float(ctx.model.dims[2]) - reach_press.POSTURE_CLEAR_M + 0.01
        ctx.cpose = ContainerPose(xyz=(float(elbow[0]), float(elbow[1]), z), yaw=0.0)
        said = "inside the box padded 90 mm"
    again, why = reach_press.plan_reach(ctx, cfg, button, home, depth=True, shortest_walk=True)
    assert why is None and said in again.posture["refused"][str(k)], again.posture
    assert again.posture["chosen"] != k
    if again.posture["chosen"]:
        assert reach_press.posture_refusal(ctx, chain, _guarded(again), button) is None


class _Line:
    """posture_refusal's chain: ArmChain.points' eight rows — joint_1 ..
    joint_7's origins, then the tool — the same at every waypoint."""

    def __init__(self, rows):
        self.rows = np.asarray(rows, float)

    def points(self, q):
        return self.rows


def test_the_posture_guard_judges_the_line_from_joint_3_to_joint_7_not_the_shoulder_nor_the_hand(ctx):
    """posture_refusal on flatbot's numbers: the table 0.21 m up, the pink
    canister's disc 105 mm across and 121 mm tall at (0.60, 0), the margin
    POSTURE_CLEAR_M. The shoulder stands below the table's top and the hand
    comes down onto the lid: neither is judged. Judged: the line between the
    joints — a forearm crossing over the box with both its ends outside the
    padded disc is in it — and every point of it over the table: an elbow 85
    mm above it is under the 90 kept. A cuboid box is judged at its yaw."""
    import dataclasses

    from box_opening.models.container import ContainerPose

    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = ContainerPose(xyz=(0.60, 0.0, 0.21), yaw=0.0)
    ctx.worlds.table_top_z = 0.21
    button = (0.60, 0.0, 0.331)
    arm = [
        [0.08, 0.0, 0.135],
        [0.14, 0.0, 0.245],  # the shoulder, under the table's top
        [0.15, 0.0, 0.45],
        [0.30, 0.0, 0.65],
        [0.45, 0.0, 0.66],
        [0.55, 0.0, 0.66],
        [0.60, 0.0, 0.56],
        [0.60, 0.0, 0.34],
    ]  # the hand, down onto the lid
    walk = [[Q, Q]]
    assert reach_press.posture_refusal(ctx, _Line(arm), walk, button) is None
    over = [list(r) for r in arm]
    over[4], over[5] = (
        [0.45, 0.0, 0.40],
        [0.75, 0.0, 0.40],
    )  # 150 mm either side of the disc's axis, 89 mm above its top
    assert reach_press.posture_refusal(ctx, _Line(over), walk, button).startswith(
        "the centre line between joint_5 and joint_6 inside the box padded 90 mm (path 1"
    )
    low = [list(r) for r in arm]
    low[3] = [0.30, 0.30, 0.21 + 0.085]
    assert reach_press.posture_refusal(ctx, _Line(low), walk, button).startswith(
        "the centre line 85 mm above the table between joint_3 and joint_4 (path 1, 0% of the way): under the 90 mm"
    )
    # a cuboid 200 x 100 mm along x: joint_5 170 mm to its side is out of it — turned 90 deg, in it
    ctx.model = dataclasses.replace(ctx.model, round=False, dims=(0.20, 0.10, 0.121))
    beside = [list(r) for r in arm]
    beside[3], beside[4] = [0.30, 0.17, 0.65], [0.60, 0.17, 0.40]
    assert reach_press.posture_refusal(ctx, _Line(beside), walk, button) is None
    ctx.cpose = ContainerPose(xyz=(0.60, 0.0, 0.21), yaw=math.pi / 2)
    assert reach_press.posture_refusal(ctx, _Line(beside), walk, button).startswith(
        "the centre line between joint_4 and joint_5 inside the box padded 90 mm"
    )
    ctx.cpose = None
    assert reach_press.posture_refusal(ctx, _Line(arm), walk, button) == "no box to keep the arm out of"


def test_the_posture_search_never_starts_a_proof_the_ik_time_left_would_not_hold(ctx, monkeypatch):
    """quickest_look spends at most about LOOK_POSTURE_BUDGET_S of IK at
    HOME: no proof of another posture starts that the time left would not
    hold at the slowest so far. Each proof taking 0.25 s by the clock, the
    first two of LOOK_POSTURE_PROOFS fit in 0.6 s, the third is not
    proven, and the search ends at 0.5 s with the quicker of the two."""
    chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    button = _seen(ctx, SCENE_162353)
    clock = [100.0]
    real = reach_press._proven

    def proven(*a, **k):
        clock[0] += 0.25
        return real(*a, **k)

    monkeypatch.setattr(reach_press, "_proven", proven)
    monkeypatch.setattr(reach_press, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    plan, why = reach_press.plan_reach(ctx, cfg, button, home, depth=True, shortest_walk=True)
    p = plan.posture
    assert why is None and p["ik_s"] == 0.5 and p["proven"] == 2 and p["chosen"] > 0, p
    assert list(p["refused"].values()) == ["not proven: the IK time left (LOOK_POSTURE_BUDGET_S)"]


@pytest.mark.parametrize(
    "scene",
    [SCENE_162353, SCENE_154524, SCENE_142121, SCENE_133716, SCENE_120459],
    ids=["10-28 16:23", "10-28 15:45", "10-28 14:21", "10-29 13:37", "10-29 12:04"],
)
def test_on_flatbot_the_posture_search_takes_a_bounded_ik_time(ctx, monkeypatch, scene):
    """The search's own cost at HOME, on the machine the test runs on: at
    most 2 x LOOK_POSTURE_SWING_RAD / LOOK_POSTURE_EVERY_RAD candidates,
    LOOK_POSTURE_PROOFS of them proven, the time it took in the record. On
    flatbot's Jetson (offline 2026-10-29) it took 0.42-0.60 s at these
    boxes: the bound here leaves room for a loaded machine."""
    _chain, cfg, home = _on_flatbot(ctx, monkeypatch)
    button = _seen(ctx, scene)
    plan, why = reach_press.plan_reach(ctx, cfg, button, home, depth=True, shortest_walk=True)
    p = plan.posture
    assert (
        why is None
        and 0 < p["candidates"] <= 2 * reach_press.LOOK_POSTURE_SWING_RAD / reach_press.LOOK_POSTURE_EVERY_RAD
    )
    assert p["proven"] + len(p["refused"]) <= reach_press.LOOK_POSTURE_PROOFS
    assert 0.0 < p["ik_s"] < 1.0


def test_the_walks_are_timed_on_the_profile_a_compliant_run_flies_them_on(ctx, tmp_path):
    from box_opening.runtime.runner import Runner

    ctx.client.compliant = True
    assert reach_press.compliant_retime() == Runner(ctx.client, log_dir=tmp_path).retime_params()


Q_ELBOW = [0.07] * 7  # the posture a faked quickest_look chooses


@pytest.mark.parametrize("case", ["chosen", "its look does not plan", "the search fails"])
def test_the_look_flies_to_the_posture_chosen_for_its_walks_and_the_record_says_so(
    ctx, monkeypatch, tmp_path, capsys, case
):
    """run_reach_press asks for the shortest walks once, at the proof before
    the look, from where the arm stands — never for the aimed button. The
    look flies to the posture chosen, a joint goal, and the record's
    reach_plan keeps quickest_look's note (look_posture). Where that
    posture's look does not plan, or the search fails, the look goes to
    today's posture as the pose goal it always was, and the note says
    why."""
    import dataclasses
    import json

    note = {
        "chosen": 2,
        "swing_rad": 1.2,
        "joint_gap_rad": 0.707,
        "walk_s": 1.79,
        "walk_s_today": 2.79,
        "transit_s": 2.16,
        "transit_s_today": 1.82,
        "saved_s": 2.67,
        "candidates": 9,
        "proven": 3,
        "refused": {},
        "ik_s": 0.43,
    }
    asked = []

    def quickest(ctx_, cfg_, chain_, tilt, button, geometry, q_from, plan):
        asked.append((tilt, [float(v) for v in q_from], list(plan.look_joints)))
        if case == "the search fails":
            raise RuntimeError("SVD did not converge")
        return dataclasses.replace(plan, look_joints=list(Q_ELBOW)), dict(note)

    monkeypatch.setattr(reach_press, "quickest_look", quickest)
    if case == "its look does not plan":
        monkeypatch.setattr(reach_press, "_plan_look_to", lambda ctx_, q_from, q: (None, "MotionGenStatus.IK_FAIL"))
    code, rec, _chain, _h, got = _depth_press_flow(
        ctx, monkeypatch, tmp_path, _way_down([], lambda t, tip: True), chain=_IKChain()
    )
    assert code == 0 and asked == [(30, [0.0] * 7, Q_LOOK)]
    rows = [json.loads(line) for line in open(next(tmp_path.glob("run-*.jsonl")))]
    (plan,) = (r for r in rows if r["leg"] == "reach_plan")
    target = rec.legs["reach:look"].target
    out = capsys.readouterr().out
    if case == "chosen":
        assert target == ("joints", Q_ELBOW) and plan["look_posture"] == note
        assert (
            "REACH-OUT: the look holds the same pose with the elbow swung 1.2 rad along its swing (joints up to "
            "0.71 rad from the nearest posture): each walk to the hand turned down 1.79 s against 2.79, the look "
            "2.16 s against 1.82 — 2.7 s quicker over 3 walks (0.43 s of IK)"
        ) in out
        return
    assert target[0] == "pose" and plan["look_posture"]["chosen"] == 0 and "the look holds the same pose" not in out
    if case == "its look does not plan":
        assert plan["look_posture"]["not_planned"] == {"2": "MotionGenStatus.IK_FAIL"}
        assert plan["look_posture"]["walk_s"] == 2.79 and plan["look_posture"]["saved_s"] == 0.0
        assert (
            "NOTE reach:look at 30 deg: the quicker posture's look does not plan (MotionGenStatus.IK_FAIL) — the "
            "look goes to today's"
        ) in out
    else:
        assert plan["look_posture"] == {"chosen": 0, "error": "SVD did not converge"}


def test_the_wrist_aim_is_moved_1_cm_out_from_the_arm_for_the_press_and_the_grip():
    """Owner 2026-10-29: "the camera calibration is slightly off, we need to
    make the arm move just 2 cm forward when pressing the box and gripping
    the lid so it grabs the middle, not the edge"; then, after flying it,
    "2 cm was too much, change it to 1". The aim's point goes AIM_FORWARD_M
    farther out along its bearing from arm_mount, its height kept; the
    wrist's own measured point (what its later readings are judged against)
    is not moved."""
    assert reach_press.AIM_FORWARD_M == pytest.approx(0.01)  # 2 cm was too much (owner, after flying it)
    for xyz in [(0.5475, -0.3106, 0.2941), (0.6339, -0.0094, 0.3117), (0.40, 0.30, 0.25)]:
        moved = reach_press.aim_forward(xyz)
        assert moved[2] == xyz[2]
        assert math.hypot(*moved[:2]) - math.hypot(*xyz[:2]) == pytest.approx(0.01)
        assert math.atan2(moved[1], moved[0]) == pytest.approx(math.atan2(xyz[1], xyz[0]))
