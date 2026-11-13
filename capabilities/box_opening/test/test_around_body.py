"""A way-in line the chair's body model refuses, re-planned round it (runtime/kinematics around_body): the hand on the
same path, the elbow swung out of the way along the arm's self-motion, every gate the plain line meets — and a line
still in the chair refused as before."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import flatbot_urdf

from box_opening.constants import HOME
from box_opening.models.container import ContainerModel, ContainerPose, load_press_demo
from box_opening.perception.d405 import quat_to_mat
from box_opening.runtime import approach, body, kinematics
from box_opening.runtime.client import pose_goal_joints
from box_opening.runtime.kinematics import CONVERGE_M, MAX_STEP_RAD, ArmChain, rotvec
from box_opening.tasks import depth_press as dp
from box_opening.tasks import reach_press as rp

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
BUTTON, TABLE_Z = (0.5921, -0.2869, 0.3123), 0.2105  # 2026-10-28 16:23's box, r 0.66 m, az -26 deg
# the elbow's one sphere (joint_4's origin is arm_forearm_link's), small: a box of 10 mm, unpadded, holds it
# within 30 mm of the box's centre
ROBOT = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_forearm_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.02}
"""
# ... or one on the bracelet, which the hand's pose holds where it is: no swing of the elbow moves it
ROBOT_HAND = ROBOT.replace("arm_forearm_link", "arm_bracelet_link")


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY)
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY = saved


def _body(tmp, boxes, robot=ROBOT):
    """flatbot's chain with a body model of `boxes` [(name, centre, dims)] in layer seat, unpadded."""
    lines = ["layer: seat", "obstacles:"]
    lines += ["  - {name: %s, position: %s, dims: %s}" % (n, [float(v) for v in c], list(d)) for n, c, d in boxes]
    (tmp / "layer_seat.yaml").write_text("\n".join(lines) + "\n")
    (tmp / "robot.yaml").write_text(robot)
    (tmp / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_seat.yaml]\nworld_padding: 0.0\n")
    approach.set_body(body.load(str(tmp / "planner.yaml"), robot_body=False))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    return approach._chain()


def _elbow(chain, q):
    return np.asarray(chain.points(list(q))[3], float)  # joint_4's origin


def _scene(ctx):
    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = ContainerPose(xyz=(BUTTON[0], BUTTON[1], TABLE_Z), yaw=0.0)
    ctx.worlds.table_top_z = TABLE_Z
    return load_press_demo(PINK)


def _slide(ctx, cfg, tilt=30):
    """16:23's look at `tilt`, at the arm's own IK posture, and its slide's end and attitude."""
    quat, start, _bottom, look = rp.reach_geometry(ctx.model, BUTTON, tilt, cfg.travel_m)
    return [float(v) for v in pose_goal_joints(look, quat, list(HOME))], start, quat


def _on_the_path(chain, plain, pts, quat):
    """Every waypoint of `pts` puts the tool where the plain line's same waypoint does, at its attitude."""
    assert len(pts) == len(plain)
    R_goal = np.asarray(quat_to_mat(*quat), float)
    for a, b in zip(plain, pts, strict=True):
        Ra, ta = chain.fk(a)
        Rb, tb = chain.fk(b)
        assert float(np.linalg.norm(np.asarray(tb) - np.asarray(ta))) < 2 * CONVERGE_M
        assert float(np.linalg.norm(rotvec(np.asarray(Rb) @ R_goal.T))) < 2e-4


def _box_mid_line(tmp_path, ctx, cfg):
    """A 10 mm box where the plain slide's elbow passes halfway: the slide enters it, from clear."""
    chain = _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.01, 0.01, 0.01))])
    q, start, quat = _slide(ctx, cfg)
    plain, why = chain.straight_line(q, start, quat)
    assert why is None
    mid = _elbow(chain, plain[len(plain) // 2])
    chain = _body(tmp_path, [("armrest", mid, (0.01, 0.01, 0.01))])
    return chain, q, start, quat, plain


def test_a_line_into_the_chair_is_swung_round_it_with_the_hand_on_the_same_path(ctx, tmp_path):
    """The slide in from 16:23's look, its elbow through a box halfway: refused as the plain line it is. Asked to
    go round (around_body), the same 2 mm waypoints put the tool on the same path at the same attitude, the
    elbow swung off the box, every step inside MAX_STEP_RAD and the limits, and the body check passes."""
    cfg = _scene(ctx)
    chain, q, start, quat, plain = _box_mid_line(tmp_path, ctx, cfg)
    pts, why = chain.straight_line(q, start, quat)
    assert pts is None and why.startswith("the chair and user: arm_forearm_link") and "into seat/armrest" in why
    pts, why = chain.straight_line(q, start, quat, **kinematics.around_body(chain))
    assert why is None and chain.body_refusal(pts) is None
    assert np.asarray(pts[0]) == pytest.approx(np.asarray(q), abs=1e-12)  # from where the arm stands
    _on_the_path(chain, plain, pts, quat)
    assert all(float(np.abs(np.subtract(b, a)).max()) <= MAX_STEP_RAD for a, b in zip(pts, pts[1:], strict=False))
    assert all(chain.within_limits(p) for p in pts[1:])
    assert dp.line_refusal(chain, pts) is None  # the runner's sanity gate on the trajectory it flies
    mid = len(pts) // 2
    assert float(np.linalg.norm(_elbow(chain, pts[mid]) - _elbow(chain, plain[mid]))) > 0.02


def test_a_walk_into_the_chair_is_swung_round_it_too(ctx, tmp_path):
    """The walk in from the look to the hand turned down (depth_press.walk_down_path's, TURN steps), its elbow
    through a box halfway: refused plain, flown round it when asked, the tool where the plain walk puts it."""
    cfg = _scene(ctx)
    chain = _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.01, 0.01, 0.01))])
    q, start, quat = _slide(ctx, cfg)
    tip = np.asarray(start) + rp._rot(quat) @ np.asarray(rp.CONTACT_TOOL)
    down, goal = dp.turn_target(ctx.model, tip, 0.0)
    kw = dict(step_m=dp.TURN_STEP_M, step_rad=dp.TURN_STEP_RAD, max_step_rad=dp.TURN_MAX_JUMP_RAD)
    plain, why = chain.walk(q, goal, down, **kw)
    assert why is None
    chain = _body(tmp_path, [("armrest", _elbow(chain, plain[len(plain) // 2]), (0.01, 0.01, 0.01))])
    assert chain.walk(q, goal, down, **kw)[0] is None
    pts, why = chain.walk(q, goal, down, **kw, **kinematics.around_body(chain))
    assert why is None and chain.body_refusal(pts) is None and len(pts) == len(plain)
    for a, b in zip(plain, pts, strict=True):  # the walk's poses turn as the plain walk's do
        Ra, ta = chain.fk(a)
        Rb, tb = chain.fk(b)
        assert float(np.linalg.norm(np.asarray(tb) - np.asarray(ta))) < 2 * CONVERGE_M
        assert float(np.linalg.norm(rotvec(np.asarray(Rb) @ np.asarray(Ra).T))) < 2e-4
    # through turn_walk, as the way in asks it
    got, quat_end, short, why = dp.turn_walk(chain, ctx.model, q, tip, (0.0,), **kinematics.around_body(chain))
    assert why is None and short == 0.0 and got[-1] == pytest.approx(pts[-1], abs=1e-9)


def test_a_line_no_swing_takes_out_of_the_chair_is_refused_as_before(ctx, tmp_path):
    """The slide with a box on the hand's own path (a sphere on the bracelet, which the hand's pose holds where it
    is): no swing of the elbow moves it, and the line is refused with the plain line's reason first. A guard of
    the caller's own refuses a line the swing did take clear (the elbow's box halfway), and says why."""
    cfg = _scene(ctx)
    chain = _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.01, 0.01, 0.01))], robot=ROBOT_HAND)
    q, start, quat = _slide(ctx, cfg)
    plain, _why = chain.straight_line(q, start, quat)
    wrist = approach.body_check().spheres_at(plain[len(plain) // 2])[0][0]
    chain = _body(tmp_path, [("armrest", wrist, (0.01, 0.01, 0.01))], robot=ROBOT_HAND)
    pts, plain_why = chain.straight_line(q, start, quat)
    assert pts is None and plain_why.startswith("the chair and user: arm_bracelet_link")
    pts, why = chain.straight_line(q, start, quat, **kinematics.around_body(chain))
    assert pts is None and why == "%s; with the elbow swung round it: %s" % (plain_why, plain_why)
    chain, q, start, quat, plain = _box_mid_line(tmp_path, ctx, cfg)
    plain_why = chain.straight_line(q, start, quat)[1]
    asked = []

    def guard(line):
        asked.append(len(line))
        return "the centre line too near the table (a test's guard)"

    pts, why = chain.straight_line(q, start, quat, **kinematics.around_body(chain, guard))
    assert pts is None and asked == [len(plain)]
    assert why == "%s; with the elbow swung round it: the centre line too near the table (a test's guard)" % plain_why


def test_only_a_chain_with_a_body_model_is_asked_to_go_round_and_a_line_that_passes_is_the_plain_one(ctx, tmp_path):
    """around_body() asks nothing of a chain with no body model (the bench) nor of a test's stand-in; a line the
    body model does not refuse is the plain one exactly, asked or not."""
    cfg = _scene(ctx)
    assert kinematics.around_body(SimpleNamespace()) == {}
    bench = ArmChain(path=flatbot_urdf(), root="arm_mount")
    assert kinematics.around_body(bench) == {}
    chain = _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.01, 0.01, 0.01))])
    assert kinematics.around_body(chain) == {"around_body": True}
    guard = object()
    assert kinematics.around_body(chain, guard) == {"around_body": guard}
    q, start, quat = _slide(ctx, cfg)
    plain, _why = bench.straight_line(q, start, quat)
    asked, _why = chain.straight_line(q, start, quat, **kinematics.around_body(chain))
    assert np.array_equal(np.asarray(asked), np.asarray(plain))


def test_sphere_depths_are_the_depths_the_line_check_refuses(ctx, tmp_path):
    """Body.sphere_depths: each sphere's depth in its deepest padded box, the number the line check (first_hit)
    calls a Hit's depth — what the re-plan drives out."""
    cfg = _scene(ctx)
    chain, q, start, quat, plain = _box_mid_line(tmp_path, ctx, cfg)
    check = approach.body_check()
    hit = check.first_hit(plain)
    D, links = check.sphere_depths(np.asarray(plain, float))
    assert D.shape == (len(plain), 1) and links == ["arm_forearm_link"]
    assert float(D[hit.index].max()) == pytest.approx(hit.depth_m, abs=1e-9) and hit.depth_m > 0
    assert float(D[0].max()) < 0.0  # the line starts clear


def test_the_way_in_from_a_look_whose_slide_enters_the_chair_proves_with_the_elbow_swung(ctx, monkeypatch, tmp_path):
    """plan_reach on 16:23's box with a box where the slide's elbow passes at 30 deg: the IK's look is clear, its
    slide is not. Re-planned round it, the 30 deg tilt proves from that look: the slide on the plain slide's path,
    clear, held to posture_refusal; the way down and the stroke from where it ends. The stroke itself is never
    re-planned: compliant_stroke's lines are the plain ones."""
    cfg = _scene(ctx)
    chain, q, start, quat, plain = _box_mid_line(tmp_path, ctx, cfg)
    assert approach.body_check().pose_clear(q)
    monkeypatch.setattr(
        rp, "_plan_motion", lambda c, s, n, t, w, sp: (SimpleNamespace(goal_joints=list(q), target=t), s)
    )
    seen = []
    real = chain.straight_line

    def line(q0, xyz, quat_, **kw):
        seen.append("around_body" in kw)
        return real(q0, xyz, quat_, **kw)

    monkeypatch.setattr(chain, "straight_line", line)
    plan, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True)
    assert why is None and plan.tilt_deg == 30
    assert plan.look_joints == pytest.approx(q, abs=1e-12)
    slide = plan.slide_pts[: len(plain)]
    _on_the_path(chain, plain, slide, quat)
    assert chain.body_refusal(plan.slide_pts) is None
    assert rp.posture_refusal(ctx, chain, [plan.slide_pts], BUTTON) is None
    assert seen[0] is True  # the slide asked to go round
    seen.clear()
    stroke, why = dp.compliant_stroke(chain, cfg, plan.stroke.q0, plan.stroke.quat, BUTTON[2])
    assert why is None and seen and not any(seen)  # the stroke's lines: plain, never asked to go round
    assert [np.asarray(p).tolist() for _n, p, _s in stroke.legs] == [
        np.asarray(p).tolist() for _n, p, _s in plan.stroke.legs
    ]


def test_a_slide_solved_again_where_the_arm_stands_goes_round_the_chair_as_the_proven_one_did(ctx, tmp_path):
    """The slide flown leg by leg (reach_press._line_from): with the arm off the proven slide's start past REUSE_RAD
    (an impedance arrival at the look), it is solved again from where the arm stands — re-planned round the chair
    when asked (the keyword the slide's call site hands it, held to way_in_guard), as the proof's was; refused as
    the plain line it is without. At its start the proven slide flies as solved."""
    cfg = _scene(ctx)
    chain, q, start, quat, _plain = _box_mid_line(tmp_path, ctx, cfg)
    around = kinematics.around_body(chain, rp.way_in_guard(ctx, chain, BUTTON[2]))
    proven, why = chain.straight_line(q, start, quat, **around)
    assert why is None
    ctx.client.live = [float(v) for v in q]
    ctx.client.live[0] += 2 * rp.REUSE_RAD
    pts, why = rp._line_from(ctx, proven, start, quat)
    assert pts is None and why.startswith("the chair and user: arm_forearm_link")
    pts, why = rp._line_from(ctx, proven, start, quat, **around)
    assert why is None and chain.body_refusal(pts) is None and rp.posture_refusal(ctx, chain, [pts], BUTTON) is None
    assert np.asarray(pts[0]) == pytest.approx(np.asarray(ctx.client.live), abs=1e-12)
    ctx.client.live = [float(v) for v in proven[0]]
    got, why = rp._line_from(ctx, proven, start, quat, **around)
    assert got is proven and why is None


def test_the_swing_minds_each_joints_room_to_its_limits():
    """The re-plan's step weighs, beside the spheres, every limited joint nearer a limit than AROUND_LIMIT_ROOM_RAD
    (_limit_shortfall): that much short, weighed AROUND_LIMIT_M_PER_RAD as a sphere's depth is in m, its rate along
    the null space toward the limit positive. Swung away from the chair without it, two slides beside it ran a joint
    into its limit. A joint with room, or with no limits, counts nothing."""
    chain = ArmChain(path=flatbot_urdf(), root="arm_mount")
    j = next(i for i, (lo, hi) in enumerate(chain.limits) if lo is not None and hi is not None)
    lo, hi = chain.limits[j]
    n = np.zeros(7)
    n[j] = 1.0
    q = np.zeros(7)
    q[j] = hi - 0.05
    d, dd = chain._limit_shortfall(q, n)
    w = kinematics.AROUND_LIMIT_M_PER_RAD
    assert d == pytest.approx([w * (kinematics.AROUND_LIMIT_ROOM_RAD - 0.05)]) and dd == pytest.approx([w])
    q[j] = lo + 0.05  # the other side: along +n the joint moves away from its limit
    d, dd = chain._limit_shortfall(q, n)
    assert d == pytest.approx([w * (kinematics.AROUND_LIMIT_ROOM_RAD - 0.05)]) and dd == pytest.approx([-w])
    q[j] = 0.5 * (lo + hi)
    assert [len(a) for a in chain._limit_shortfall(q, n)] == [0, 0]
