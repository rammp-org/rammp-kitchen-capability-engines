"""The look pose's posture clear of the chair and its user (tasks/reach_press clear_look, quickest_look with a body
model): where the arm's IK puts the look in a padded box, the same pose with the elbow swung clear of it."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import flatbot_urdf

from box_opening.constants import HOME
from box_opening.models.container import ContainerModel, ContainerPose, load_press_demo
from box_opening.runtime import approach, body
from box_opening.runtime.branches import planner_branch
from box_opening.runtime.client import pose_goal_joints
from box_opening.tasks import reach_press as rp

# The 2026-10-21 HOME (conftest.HOME_20261021): this file's recorded runs and poses were built on it.
pytestmark = pytest.mark.usefixtures("home_20261021")

PINK = "capabilities/box_opening/config/containers/ankou_pink.yaml"
# 2026-10-28 16:23's scene camera box (r 0.66 m, az -26 deg) and the table it stood on
BUTTON, TABLE_Z = (0.5921, -0.2869, 0.3123), 0.2105
# the elbow's sphere (joint_4's origin is arm_forearm_link's), the only one the body model checks here
ROBOT = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_forearm_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.05}
"""


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    monkeypatch.setenv("BOX_OPENING_STATE", str(tmp_path))
    saved = (approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY)
    yield
    approach._ROOT, approach._URDF_PATH, approach._CHAIN, approach._GRAVITY, approach._BODY = saved


def _body(tmp, boxes):
    """flatbot's chain with a body model of `boxes` [(name, centre, dims)] in layer seat, the planner's padding."""
    lines = ["layer: seat", "obstacles:"]
    lines += ["  - {name: %s, position: %s, dims: %s}" % (n, list(map(float, c)), list(d)) for n, c, d in boxes]
    (tmp / "layer_seat.yaml").write_text("\n".join(lines) + "\n")
    (tmp / "robot.yaml").write_text(ROBOT)
    (tmp / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_seat.yaml]\n")
    approach.set_body(body.load(str(tmp / "planner.yaml"), robot_body=False))
    approach.set_chain_root("arm_mount", flatbot_urdf())
    return approach._chain()


def _scene(ctx):
    ctx.model = ContainerModel.load(PINK)
    ctx.cpose = ContainerPose(xyz=(BUTTON[0], BUTTON[1], TABLE_Z), yaw=0.0)
    ctx.worlds.table_top_z = TABLE_Z
    return load_press_demo(PINK)


def _asked(monkeypatch, chain):
    """reach_press._plan_motion as the planner would answer each look, with the goal refused where box_opening
    refuses it (approach.goal_refusal): every target asked, in the list returned."""
    asked = []

    def plan(ctx_, state, name, target, world, speed):
        asked.append(target)
        why = approach.goal_refusal(target, state.joints)
        if why is not None:
            raise RuntimeError("%s refused before it was sent to the planner: %s" % (name, why))
        if target[0] == "joints":
            return SimpleNamespace(goal_joints=[float(v) for v in target[1]], target=target), state
        q = pose_goal_joints(target[1], target[2], state.joints)
        return SimpleNamespace(goal_joints=[float(v) for v in q], target=target), state

    monkeypatch.setattr(rp, "_plan_motion", plan)
    return asked


def _ik_look(chain, ctx, cfg, tilt=30):
    quat, _start, _bottom, look = rp.reach_geometry(ctx.model, BUTTON, tilt, cfg.travel_m)
    return np.asarray(pose_goal_joints(look, quat, list(HOME)), float), look, quat


def _elbow(chain, q):
    return np.asarray(chain.points(list(q))[3], float)  # joint_4's origin


@pytest.mark.parametrize("shortest_walk", [False, True], ids=["the look alone", "the posture search after it"])
def test_a_look_the_arms_ik_puts_in_the_chair_goes_to_the_same_pose_with_the_elbow_swung_clear(
    ctx, monkeypatch, tmp_path, shortest_walk
):
    """The IK's look posture with a 60 mm box round its elbow (padded 20 mm, the elbow's sphere 50 + 5): inside,
    a goal box_opening never sends. The look goes to the same tool pose with the elbow swung along its swing, clear
    of the box, planned as a JOINT goal to that posture (as a pose the client would send the IK's again), its press
    proven from there; the record says how far the IK's stood and where the chosen one stands."""
    cfg = _scene(ctx)
    monkeypatch.setattr(rp, "LOOK_POSTURE_BUDGET_S", float("inf"))
    chain = _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.1, 0.1, 0.1))])
    asked = _asked(monkeypatch, chain)
    q_ik, look, quat = _ik_look(chain, ctx, cfg)
    today, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True)
    assert why is None and today.tilt_deg == 30 and today.posture is None  # clear: as it always was
    assert today.look_joints == pytest.approx(list(q_ik), abs=1e-9) and asked == [("pose", look, quat)]
    chain = _body(tmp_path, [("elbow", _elbow(chain, q_ik), (0.06, 0.06, 0.06))])
    check = approach.body_check()
    assert check is not None and not check.pose_clear(list(q_ik))
    asked.clear()
    plan, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True, shortest_walk=shortest_walk)
    assert why is None and plan.tilt_deg == 30
    note = plan.posture["clear_look"]
    assert note["ik_mm"] < 0 and note["ik_box"] == "seat/elbow" and note["swing_rad"] != 0.0 and note["proven"] == 1
    assert note["clear"] >= 1 and note["clear_mm"] > 0
    # the same look pose — the same view — another elbow, clear of the box
    R0, t0 = chain.fk(list(q_ik))
    R1, t1 = chain.fk(plan.look_joints)
    assert np.asarray(t1) == pytest.approx(np.asarray(t0), abs=1e-4)
    assert np.asarray(R1) == pytest.approx(np.asarray(R0), abs=1e-4)
    assert check.pose_clear(plan.look_joints) and approach.goal_refusal(("joints", plan.look_joints)) is None
    assert np.linalg.norm(_elbow(chain, plan.look_joints) - _elbow(chain, q_ik)) > 0.1
    # planned once, a joint goal to it; proven from it: the slide starts there, the way down and the stroke follow
    assert asked == [("joints", plan.look_joints)] and plan.look_leg.goal_joints == plan.look_joints
    assert plan.slide_pts[0] == pytest.approx(plan.look_joints) and plan.down is not None and plan.stroke.legs
    if shortest_walk:
        p = plan.posture
        assert p["clear"] <= p["candidates"] and p["clear_mm"] > 0  # quickest_look weighed clear postures only
    else:
        assert plan.posture["chosen"] == 0  # no search asked: the clear posture itself


def test_the_tilted_stroke_reach_s_look_goes_to_a_clear_posture_too(ctx, monkeypatch, tmp_path):
    """The reach-out with its tilted stroke (plan_reach without depth: the re-aim's planned look), the same box round
    the IK look's elbow: the look goes to a clear posture of the same pose, a joint goal, its slide and its stroke
    proven from there and held to posture_refusal."""
    cfg = _scene(ctx)
    monkeypatch.setattr(rp, "LOOK_POSTURE_BUDGET_S", float("inf"))
    chain = _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.1, 0.1, 0.1))])
    q_ik, _look, _quat = _ik_look(chain, ctx, cfg)
    chain = _body(tmp_path, [("elbow", _elbow(chain, q_ik), (0.06, 0.06, 0.06))])
    asked = _asked(monkeypatch, chain)
    plan, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME))
    assert why is None and plan.tilt_deg == 30 and plan.down is None and plan.stroke_pts
    assert plan.posture["chosen"] == 0 and plan.posture["clear_look"]["proven"] == 1
    assert approach.body_check().pose_clear(plan.look_joints) and asked == [("joints", plan.look_joints)]
    assert plan.slide_pts[0] == pytest.approx(plan.look_joints)
    assert rp.posture_refusal(ctx, chain, [plan.slide_pts, plan.stroke_pts], BUTTON) is None


def test_the_posture_search_reads_its_swing_at_the_call(ctx, monkeypatch, tmp_path):
    """quickest_look's candidates follow LOOK_POSTURE_SWING_RAD as it stands when the look is planned (a study sets
    it), as they did before _swing_postures took them over."""
    cfg = _scene(ctx)
    monkeypatch.setattr(rp, "LOOK_POSTURE_BUDGET_S", float("inf"))
    approach.set_body(False)
    approach.set_chain_root("arm_mount", flatbot_urdf())
    _asked(monkeypatch, approach._chain())
    wide, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True, shortest_walk=True)
    assert why is None and wide.posture["candidates"] > 2
    monkeypatch.setattr(rp, "LOOK_POSTURE_SWING_RAD", rp.LOOK_POSTURE_EVERY_RAD)
    narrow, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True, shortest_walk=True)
    assert why is None and narrow.posture["candidates"] <= 2


def test_a_look_no_posture_of_which_is_clear_is_refused_before_the_planner_is_asked(ctx, monkeypatch, tmp_path):
    """A box round the whole elbow circle (0.9 m across, centred on the IK's elbow): no posture along the swing is
    clear at any tilt. Each tilt is refused, saying how far the IK's posture stands and that none is clear, and the
    planner is never asked for a look."""
    cfg = _scene(ctx)
    chain = _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.1, 0.1, 0.1))])
    q_ik, _look, _quat = _ik_look(chain, ctx, cfg)
    chain = _body(tmp_path, [("all", _elbow(chain, q_ik), (0.9, 0.9, 0.9))])
    asked = _asked(monkeypatch, chain)
    plan, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True, shortest_walk=True)
    assert plan is None and asked == []
    for tilt in rp.REACH_TILTS_DEG:
        assert "%d deg: the look pose (the arm's IK stands it -" % tilt in why
    assert why.count("no posture along its elbow's swing is clear of the chair and user") == len(rp.REACH_TILTS_DEG)


def test_with_the_body_model_off_the_look_and_its_posture_search_are_as_before(ctx, monkeypatch, tmp_path):
    """No body model (the bench): no posture is judged clear or not, nothing is ranked by clearance, and the look,
    its posture search and their notes are what they were — no `clear` in the note, no `clear_look`. A body model
    whose one box is far from the arm, ties left unbroken (LOOK_POSTURE_TIE_S 0), chooses the same posture."""
    cfg = _scene(ctx)
    monkeypatch.setattr(rp, "LOOK_POSTURE_BUDGET_S", float("inf"))
    approach.set_body(False)
    approach.set_chain_root("arm_mount", flatbot_urdf())
    chain = approach._chain()
    _asked(monkeypatch, chain)
    assert approach.body_check() is None
    plan, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True, shortest_walk=True)
    assert why is None and plan.posture["chosen"] > 0
    assert "clear" not in plan.posture and "clear_look" not in plan.posture
    _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.1, 0.1, 0.1))])
    monkeypatch.setattr(rp, "LOOK_POSTURE_TIE_S", 0.0)
    with_far, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True, shortest_walk=True)
    assert why is None and with_far.posture["clear"] == with_far.posture["candidates"]
    assert with_far.look_joints == pytest.approx(plan.look_joints, abs=1e-12)
    assert with_far.posture["chosen"] == plan.posture["chosen"] and "clear_look" not in with_far.posture


def test_quickest_look_weighs_only_clear_postures(ctx, monkeypatch, tmp_path):
    """16:23's quickest other elbow (the posture search's choice with no body model) with a box round its elbow:
    the search never weighs it, and whatever it chooses stands clear of the box."""
    cfg = _scene(ctx)
    monkeypatch.setattr(rp, "LOOK_POSTURE_BUDGET_S", float("inf"))
    chain = _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.1, 0.1, 0.1))])
    _asked(monkeypatch, chain)
    first, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True, shortest_walk=True)
    assert why is None and first.posture["chosen"] > 0, first.posture
    q_ik, _look, _quat = _ik_look(chain, ctx, cfg)
    elbow = _elbow(chain, first.look_joints)
    assert np.linalg.norm(elbow - _elbow(chain, q_ik)) > 0.11  # today's elbow is not in the box below
    _body(tmp_path, [("there", elbow, (0.06, 0.06, 0.06))])
    check = approach.body_check()
    assert check.pose_clear(list(q_ik)) and not check.pose_clear(first.look_joints)
    again, why = rp.plan_reach(ctx, cfg, BUTTON, list(HOME), depth=True, shortest_walk=True)
    p = again.posture
    assert why is None and p["clear"] < p["candidates"] and "clear_look" not in p
    assert check.pose_clear(again.look_joints) and p["clear_mm"] > 0
    assert again.look_joints != pytest.approx(first.look_joints, abs=1e-3)


def test_of_equally_quick_postures_the_one_farthest_from_the_chair_is_tried_first():
    """_rank_clear: by the bound, except that within LOOK_POSTURE_TIE_S of the quickest left the most clearance goes
    first, group by group; with no clearance (no body model) by the bound alone, a tie in self_motion's order."""
    q = np.zeros(7)
    tie = rp.LOOK_POSTURE_TIE_S
    rows = [(1.0, 1, 0.1, q), (1.0 + 0.5 * tie, 2, 0.2, q), (1.0 + 0.9 * tie, 3, 0.3, q), (1.0 + 3 * tie, 4, 0.4, q)]
    rows += [(1.0 + 3.5 * tie, 5, 0.5, q), (1.0 + 0.5 * tie, 6, 0.6, q)]
    clearance = {1: 0.020, 2: 0.045, 3: 0.031, 4: 0.010, 5: 0.090, 6: 0.045}
    assert [r[1] for r in rp._rank_clear(rows, clearance)] == [2, 6, 3, 1, 5, 4]
    assert [r[1] for r in rp._rank_clear(rows, None)] == [1, 2, 6, 3, 4, 5]
    assert [r[1] for r in rp._rank_clear(rows, {})] == [1, 2, 6, 3, 4, 5]


@pytest.mark.parametrize("budget_s", [float("inf"), 0.0], ids=["IK time to spare", "no IK time left"])
def test_clear_look_searches_its_swing_finely_and_proves_no_two_alike(ctx, monkeypatch, tmp_path, budget_s):
    """clear_look takes every SELF_MOTION_STEP_RAD of the swing (the clear windows beside the chair are 0.1-0.3 rad
    wide), not quickest_look's every LOOK_POSTURE_EVERY_RAD, and proves at most LOOK_CLEAR_PROOFS of them, none
    within LOOK_POSTURE_DEDUP_RAD of one already tried: a proof that fails stands for its neighbours. However long
    the IK took (quickest_look's LOOK_POSTURE_BUDGET_S spent): whether the look flies is not the computer's load's."""
    from box_opening.runtime.kinematics import SELF_MOTION_STEP_RAD

    cfg = _scene(ctx)
    monkeypatch.setattr(rp, "LOOK_POSTURE_BUDGET_S", budget_s)
    chain = _body(tmp_path, [("far", (2.0, 2.0, 2.0), (0.1, 0.1, 0.1))])
    q_ik, _look, _quat = _ik_look(chain, ctx, cfg)
    chain = _body(tmp_path, [("elbow", _elbow(chain, q_ik), (0.06, 0.06, 0.06))])
    check = approach.body_check()
    tried = []

    def never(ctx_, cfg_, chain_, tilt, button, geometry, q, depth):
        tried.append(np.asarray(q, float))
        return None, "the slide in (faked)", None

    monkeypatch.setattr(rp, "_proven", never)
    geometry = rp.reach_geometry(ctx.model, BUTTON, 30, cfg.travel_m)
    plan, why, kept, note = rp.clear_look(ctx, cfg, chain, check, 30, BUTTON, geometry, q_ik, list(HOME), True)
    assert plan is None and kept is None and "no clear posture along its elbow's swing proves" in why
    assert len(tried) == rp.LOOK_CLEAR_PROOFS == note["proven"] + len(note["refused"])
    assert all(float(np.abs(a - b).max()) >= rp.LOOK_POSTURE_DEDUP_RAD for i, a in enumerate(tried) for b in tried[:i])
    every = rp._swing_postures(chain, q_ik, planner_branch(list(HOME)))
    assert note["candidates"] >= 2 * len(every)  # far finer than quickest_look's
    assert note["candidates"] >= rp.LOOK_POSTURE_SWING_RAD / SELF_MOTION_STEP_RAD
