"""The fridge's planning on FakeCurobo (fridge_fake.py): what reaches cuRobo (the worlds, the legs, the joined
motions), what the arm is sent (one trajectory per motion, impedance, the gripper while it moves) and what the door file
says after. Collisions are cuRobo's and are not checked here (data/curobo_sim's GPU harness does)."""

import json
import time
import types

import numpy as np
import pytest
from door_common import common as dc
from door_common.common import DoorFrame
from fridge import close_door, open_door, planning, push_open, side_grasp
from fridge import constants as F
from fridge.kinematics import Chain
from fridge.world import FridgeModel, PlannerWorld, chair_stand_in, hand_reach
from fridge_fake import (  # noqa: F401 -- fridge: the fixture
    HANDLE,
    URDF,
    FakeCurobo,
    Plan,
    door_is_open,
    door_where_it_stands,
    fridge,
)
from scipy.spatial.transform import Rotation as R

from capability_engine import sim

IMPEDANCE = planning.IMPEDANCE


def _closed_door(tmp_path):
    """The door dict the open plans with, for the fixture's detection (configure has run)."""
    geom = dict(h_raw=HANDLE, orient=np.array([-0.5, 0.5, 0.5, -0.5]), n=np.array([-1.0, 0.0, 0.0]))
    from door_common import open_door as shared

    with dc.quiet():
        geom["hinge"], h, grasp, _ = shared.FIND_HINGE(geom["h_raw"], geom["orient"], geom["n"], None)
    return open_door.closed_door(geom, h, np.asarray(grasp.position))


# --- the arm's own model --------------------------------------------------------------------------------------------


def test_the_nodes_forward_kinematics_is_the_drivers_model():
    sim.reset_for_tests()
    scene, rb = sim.make_sim(str(URDF))
    chain = Chain(URDF.read_text())
    for q in np.random.default_rng(0).uniform(-1.5, 1.5, (6, 7)):
        p_sim, q_sim = sim.tool_pose(scene, rb, q)
        p, quat = chain.pose(q)
        assert np.allclose(p, p_sim, atol=1e-5)
        assert abs(float(np.dot(quat, q_sim))) == pytest.approx(1.0, abs=1e-6)
    sim.reset_for_tests()


def test_the_exact_jacobian_is_the_finite_difference_one():
    chain = Chain(URDF.read_text())
    for q in np.random.default_rng(1).uniform(-2.5, 2.5, (20, 7)):
        np.testing.assert_allclose(chain.jacobian(q), chain.jacobian_fd(q), atol=2e-5)


def test_local_ik_stays_on_its_branch_and_refuses_a_limit():
    chain = Chain(URDF.read_text())
    q0 = np.radians((98.8, 10.2, 179.5, -94.7, 98.7, 100.2, 16.1))
    p, quat = chain.pose(q0)
    q, why = chain.ik(p + [0.02, 0.0, -0.01], quat, q0)  # 2 cm along: the joints barely move
    assert why == "" and np.degrees(np.max(np.abs(q - q0))) < 10
    assert np.allclose(chain.pose(q)[0], p + [0.02, 0.0, -0.01], atol=5e-4)
    q_lim = q0.copy()
    q_lim[1] = chain.upper[1] - 0.01  # J2 at its limit already
    q, why = chain.ik(*chain.pose(q_lim), q_lim)
    assert q is None and "J2" in why


# --- the fridge in cuRobo's world -----------------------------------------------------------------------------------


def test_the_door_model_is_the_slab_the_cabinet_and_the_handle_where_the_door_file_says(fridge, tmp_path):
    door = _closed_door(tmp_path)
    model = FridgeModel(door)
    df = DoorFrame(door, door["open_sign"])
    (slab,), (body,), (handle,) = model.slab(0.0), model.body(), model.handle(0.0)
    face_x = HANDLE[0] + F.MEASURED["handle_centre_to_face_m"]  # the door square to the arm: its face in x
    assert slab["position"][0] - slab["dims"][0] / 2 == pytest.approx(face_x, abs=1e-3)
    assert slab["dims"][0] == pytest.approx(F.MEASURED["door_depth_m"]) and slab["dims"][1] == pytest.approx(df.length)
    assert body["position"][0] - body["dims"][0] / 2 == pytest.approx(slab["position"][0] + slab["dims"][0] / 2, abs=1e-3)
    assert handle["position"][0] + handle["dims"][0] / 2 == pytest.approx(face_x, abs=1e-3)  # on the face, toward the arm
    open_slab = model.slab(90.0)[0]  # open 90: turned a quarter, its middle swung toward the arm
    assert abs(open_slab["rpy_deg"][2] - slab["rpy_deg"][2]) == pytest.approx(90.0, abs=1e-6)
    assert open_slab["position"][0] < slab["position"][0] - 0.1
    # a window round the hand (the door runs on only ~9 cm past the handle): the slab in four pieces round it
    pieces = model.slab(0.0, window=(df.u_handle - 0.05, df.u_handle + 0.05, HANDLE[2] - 0.1, HANDLE[2] + 0.1))
    assert sorted(b["name"] for b in pieces) == ["door_above", "door_below", "door_free_side", "door_hinge_side"]
    assert sum(np.prod(b["dims"]) for b in pieces) == pytest.approx(np.prod(slab["dims"]) - 0.1 * 0.2 * df.t, rel=1e-6)


def test_the_planner_gets_the_scene_with_the_robots_layers_and_gets_its_own_world_back(tmp_path):
    arm = FakeCurobo()
    w = PlannerWorld(arm, tmp_path / "w", "/rammp_planner_worlds", ["/config/robot/curobo/layer_mebot_floor.yaml"],
                     extra=[chair_stand_in(0.1)])
    box = {"name": "b", "position": [1.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0], "dims": [0.1, 0.1, 0.1]}
    w.push([box], "test")
    assert arm.world_calls == [("/rammp_planner_worlds/fridge_world.yaml", False)]  # as the planner sees it
    scene = (tmp_path / "w" / "fridge_world.yaml").read_text()
    assert "layer_mebot_floor.yaml" in scene and "chair_stand_in" in scene and "base_frame: arm_mount" in scene
    w.push([box], "again")  # the planner holds it already: nothing sent
    assert len(arm.world_calls) == 1
    w.restore()
    assert arm.world_calls[-1] == ("world.yaml", True)  # put back even after a cancel
    assert chair_stand_in(0.381)["position"][0] + 0.5 == pytest.approx(-0.381)  # the level mount's box starts there


def test_a_refused_world_refuses(tmp_path):
    arm = FakeCurobo()
    arm.set_world = lambda name, even_if_cancelled=False: (False, "a layer box reaches into the arm")
    with pytest.raises(dc.Refused, match="refused the fridge's collision world"):
        PlannerWorld(arm, tmp_path).push([], "test")


# --- the planner ----------------------------------------------------------------------------------------------------


def test_a_leg_along_the_door_follows_its_path_and_cuRobo_checks_it(fridge):
    arm, planner, _ = fridge
    p, quat = arm.fk.pose(arm.q)
    goal = planning.line(p + [0.0, 0.0, 0.05], quat, "up", [], contact=True)
    plan, why = planner.leg(goal, arm.q)
    assert why is None and isinstance(plan, planning.LocalPlan)
    assert np.allclose(plan.q[0], arm.q) and not plan.qd[0].any() and not plan.qd[-1].any()  # from the start, at rest
    pos, _ = arm.fk.poses(plan.q)
    assert np.max(np.linalg.norm(pos[:, :2] - p[:2], axis=1)) < 1e-3  # straight up
    assert arm.blends[-1][0] == 1  # cuRobo's one-leg check
    assert not arm.pose_goals  # no pose goal for it


def test_the_timing_starts_on_the_arm_ends_still_and_slows_for_bends_and_caps():
    q0 = np.zeros(7)
    straight = np.array([[0.01 * k] + [0.0] * 6 for k in range(1, 101)])
    t, q, qd = planning.timed(q0, straight)
    assert np.allclose(q[0], q0) and not qd[0].any() and not qd[-1].any() and np.allclose(q[-1], straight[-1])
    assert np.abs(qd).max() <= planning.LOCAL_JOINT_SPEED * 1.05 and t[1] - t[0] == pytest.approx(0.02)
    t2, _, _ = planning.timed(q0, straight, caps=[1.0] * 50 + [0.3] * 50)
    assert t2[-1] > t[-1] * 1.3  # the capped half takes longer
    corner = np.vstack([straight[:50], straight[49] + np.array([[0.0, 0.01 * k] + [0.0] * 5 for k in range(1, 51)])])
    _, qc, qdc = planning.timed(q0, corner)
    k = int(np.argmin(np.linalg.norm(qc - straight[49], axis=1)))
    assert np.abs(qdc[k]).max() < 0.3 * np.abs(qdc).max()  # slowed into the corner


def test_a_start_inside_an_obstacle_refuses_at_once(fridge):
    arm, planner, _ = fridge
    arm.start_blocked = True
    p, quat = arm.fk.pose(arm.q)
    with pytest.raises(planning.StartNotClear, match="no plan can start from it"):
        planner.motion([planning.free(p + [0.1, 0, 0], quat, "x", [])], arm.q, from_arm=True)
    with pytest.raises(dc.Refused, match="start pose is not clear"):
        open_door.run(arm, planner, False)
    assert len(arm.pose_goals) == 2  # one plan each: no search


def test_a_search_stops_at_its_budget_and_repeats_are_planned_once(fridge):
    arm, planner, _ = fridge
    p, quat = arm.fk.pose(arm.q)
    goal = planning.free(p + [0.05, 0, 0], quat, "x", [])
    planner.motion([goal], arm.q)
    planner.motion([goal], arm.q)
    assert len(arm.pose_goals) == 1 and planner.cached == 1
    planner.budget_s = 0.0
    with pytest.raises(planning.OverBudget):
        planner.motion([planning.free(p + [0.06, 0, 0], quat, "y", [])], arm.q)


def test_join_keeps_a_stop_after_a_cut_corner_unless_curobo_refuses_it():
    """RAMMP-CuRobo 2.0.1's blend can reject a stop right after a blended junction (when the corner swallows the leg
    between): join asks for the true stop first, and only if that is refused makes it a BLEND_MIN_M corner."""
    arm = FakeCurobo()
    w = PlannerWorld(arm, "/tmp/fridge_join_test")
    planner = planning.Planner(arm, w, arm.fk)
    p, quat = arm.fk.pose(arm.q)
    goals = [planning.free(p + [0.02 * k, 0, 0], quat, f"g{k}", [], blend=b) for k, b in ((1, 0.03), (2, 0.0), (3, 0.0))]
    motion, _ = planner.motion(goals, arm.q)
    planner.join(motion, [])
    assert arm.blends[-1][1] == [0.03, 0.0] and not motion.notes  # the true stop
    blend = arm.blend

    def refuses_the_stop(legs, radii, **kw):
        if radii == [0.03, 0.0]:
            arm.blends.append((len(legs), list(radii), None, None))
            return Plan(False, "blend raised ValueError('overlap 0 at junction 1 is outside 0..-1')")
        return blend(legs, radii, **kw)

    arm.blend = refuses_the_stop
    motion, _ = planner.motion(goals, arm.q)
    planner.join(motion, [])
    assert [b[1] for b in arm.blends[-2:]] == [[0.03, 0.0], [0.03, planning.BLEND_MIN_M]] and motion.notes


# --- the open -------------------------------------------------------------------------------------------------------


def test_a_dry_open_plans_everything_and_moves_nothing(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False)
    assert arm.sent == [] and arm.grips == []
    assert "no wheelchair move" in push_open.NEXT_STEP
    plan = open_door.LAST_PLAN
    swing = plan["swing"].goals
    assert len(swing) == 1 and swing[0].path == "through"  # the pull + release: one leg
    assert [g.path for g in plan["approach"].goals] == ["free", "line"]


def test_the_open_runs_each_motion_as_one_trajectory_and_lets_go_while_moving(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, True)
    # the approach (position control), the swing + release, then off the handle + round the door + the push
    assert len(arm.sent) == 3
    assert "control_mode" not in arm.sent[0]
    for kw in arm.sent[1:]:
        assert kw["control_mode"] == IMPEDANCE and kw["gains"]["kq"] == F.PULL_GAINS["kq"]
    assert arm.sent[1]["path_tolerance"] == F.PULL_PATH_TOL_RAD
    assert arm.sent[2]["path_tolerance"] == F.AFTER_RELEASE_PATH_TOL_RAD
    # the gripper: closes as the swing starts, opens slowly while it runs on (then is seen open before what follows)
    assert arm.grips[0] == (1.0, F.GRASP_FORCE, None)
    assert arm.grips[1] == (0.0, None, F.GRIP_OPEN_SPEED)
    door = json.loads(dc.DOOR_FILE.read_text())
    assert door["door_open_deg"] == F.PUSH_OPEN_TO_DEG and door["open_sign"] in (-1.0, 1.0)
    assert door["closed_handle"][0] == pytest.approx(HANDLE[0], abs=1e-6)
    # the hand ends off the door's inner face, the close's start
    end = np.asarray(arm.get_state()["ee_pos"][:3])
    df = DoorFrame(door, door["open_sign"])
    v = float((df._rot(90.0).T @ (end[:2] - df.hinge[:2])) @ df.n) - df.mid
    assert v < -df.t / 2 - hand_reach(arm.get_state()["ee_pos"][3:7], np.r_[df._rot(90.0) @ df.n, 0], F.HAND_BOX_TOOL_M)


def test_the_swing_stops_at_the_angle_asked_for_then_lets_go(fridge):
    """User, 11-09: smooth, but the pull comes to a stop at the angle and only then do the fingers open -- the hand
    never pulls the door on past it (until 11-09 it ran on 8 deg while opening, and the door ended ~68 deg on a 60)."""
    arm, planner, _ = fridge
    order = []
    send, grip = arm.execute_joint_trajectory, arm.gripper

    def sent(names, points, **kw):
        order.append(("move", len(points)))
        return send(names, points, **kw)

    def gripped(position, *a, **kw):
        order.append(("grip", position))
        return grip(position, *a, **kw)

    arm.execute_joint_trajectory, arm.gripper = sent, gripped
    assert open_door.run(arm, planner, True, push=False, around=False, swing_deg=60)
    plan = open_door.LAST_PLAN
    goal = plan["swing"].goals[0]
    df = plan["model"].df
    if plan["cand"].get("side"):  # the bar carried round with the door, the hand turning about it
        end = side_grasp.swing_poses(plan["cand"], df, 60, plan["lag"], [60.0])[0]
    else:
        end = open_door.swung(plan["cand"]["pos"], plan["cand"]["quat"], df.hinge, df.sign, 60)
    np.testing.assert_allclose(goal.through[-1][0], end[0], atol=1e-9)  # the swing's last pose: exactly 60 deg
    assert all(abs(open_door.hand_angle(plan["door"], [*p, *q])) <= 60 + 1e-6
               for p, q in goal.through)
    # approach; close as the swing starts; the swing; THEN open (the arm still); then the back-off
    seq = order
    assert [k for k, _ in seq] == ["move", "grip", "move", "grip", "move"], order
    assert seq[1][1] == 1.0 and seq[3][1] == 0.0
    assert np.max(np.abs(plan["swing"].joined.qd[-1])) < 1e-6  # the swing trajectory ends at rest
    door = json.loads(dc.DOOR_FILE.read_text())
    assert door["door_open_deg"] == pytest.approx(60.0, abs=0.05)


def test_the_open_does_not_back_off_while_the_gripper_is_still_closed(fridge):
    arm, planner, _ = fridge
    grip = arm.gripper

    def stuck(position, *a, **kw):
        grip(position, *a, **kw)
        if position == 0.0 and len(arm.sent) >= 2:  # after the swing: the fingers don't open
            arm.gripper_pos = 0.8

    arm.gripper = stuck
    with pytest.raises(dc.Refused, match="did not open"):
        open_door.run(arm, planner, True, push=False, around=False, swing_deg=60)
    assert len(arm.sent) == 2  # the approach and the swing: no back-off with the hand on the handle


def test_the_open_refuses_a_pull_that_leaves_nothing_to_push(fridge):
    arm, planner, _ = fridge
    with pytest.raises(dc.Refused, match="too little to push"):
        open_door.run(arm, planner, False, swing_deg=F.PUSH_OPEN_TO_DEG)


def test_the_push_asks_for_the_smallest_wheelchair_move_that_lets_it_push(fridge):
    arm, planner, _ = fridge
    # cuRobo finds no arm configuration in behind the open door where it stands: only after a move
    arm.unreachable = lambda pos, quat, obstacles: isinstance(obstacles, list) and door_is_open(obstacles)
    assert open_door.run(arm, planner, False)
    assert "stops, asking you to" in push_open.NEXT_STEP and "BACK" in push_open.NEXT_STEP.upper()
    after = open_door.LAST_PLAN["after"]
    assert after.chair_move == F.CHAIR_MOVES_M[1]
    # the route for it is planned against the door all along the move (the arm held still)
    swept = [w for w in arm.worlds if isinstance(w, list) and any(b["name"].startswith("m2_") for b in w)]
    assert swept and {"m0_door", "m1_door", "m2_door", "m0_handle"} <= {b["name"] for b in swept[0]}
    assert open_door.run(arm, planner, True)  # executed: the pull, off the handle, then it asks
    assert len(arm.sent) == 3 and "chair_moved true" in push_open.NEXT_STEP
    assert json.loads(dc.DOOR_FILE.read_text())["push_after_chair_move"] == pytest.approx(after.chair_move)
    hinge = json.loads(dc.DOOR_FILE.read_text())["hinge"]
    arm.unreachable = lambda *a: False
    assert open_door.run(arm, planner, True, chair_moved=True)  # the push, against the door where it now is
    door = json.loads(dc.DOOR_FILE.read_text())
    assert door["push_after_chair_move"] is None and door["door_open_deg"] == F.PUSH_OPEN_TO_DEG
    assert door["hinge"][0] == pytest.approx(hinge[0] + after.chair_move)


def test_a_dry_open_with_save_door_leaves_the_door_as_the_planned_open_would(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    door = json.loads(dc.DOOR_FILE.read_text())
    assert door["door_open_deg"] == F.PUSH_OPEN_TO_DEG and "saved_by" in door


def test_release_only_backs_off_the_handle(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    arm.gripper_pos = 0.8
    assert open_door.run(arm, planner, True, release_only=True)
    assert arm.grips[-1][0] == 0.0 and len(arm.sent) == 1


# --- the close ------------------------------------------------------------------------------------------------------


def test_the_close_runs_as_one_trajectory_and_leaves_the_door_to_swing_shut(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, True)
    sent = len(arm.sent)
    assert close_door.run(arm, planner, True)  # an empty hand
    assert len(arm.sent) == sent + 1 and "control_mode" not in arm.sent[-1]
    assert all(h == planning.HOLD_NONE for h in arm.blends[-1][3])  # nothing held: no leg holds the tilt
    assert json.loads(dc.DOOR_FILE.read_text())["door_open_deg"] == F.CLOSE_TO_DEG
    assert close_door.LAST_CHAIR_BACK == 0.0


def test_the_container_is_carried_with_the_hand_and_the_fridge_is_never_shifted(fridge):
    """11-11: the close stood the container in by planning the whole fridge 17.8 cm nearer the arm, which put the
    push-open's end pose inside the door. Now every world the close plans in has the cabinet where the door file says
    (the door itself moves as the close pushes it), holding the container or not."""
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, True)
    door = json.loads(dc.DOOR_FILE.read_text())
    want = {b["name"]: b["position"] for b in FridgeModel(door).body()}  # the cabinet: the same at every door angle
    for grip in (0.8, 0.0):  # the gripper closed (holding the container), then open (an empty hand)
        arm.gripper_pos = grip
        n0 = len(arm.worlds)
        try:
            close_door.run(arm, planner, False, chair_back=0.0)
        except dc.Refused:
            pass
        worlds = [w for w in arm.worlds[n0:] if isinstance(w, list)]  # every world the close pushed to cuRobo
        assert any(b["name"] == "door" for w in worlds for b in w)
        for w in worlds:
            for b in w:
                if b["name"] in want:
                    assert b["position"] == pytest.approx(want[b["name"]], abs=1e-6), b["name"]


def _close_geometries(arm, planner, container=False):
    """The close's geometry for the fixture's door (the open leaves it at 90), the hand as it is now turned by each of
    CLOSE_YAWS_DEG."""
    assert open_door.run(arm, planner, False, save_door=True)
    door = json.loads(dc.DOOR_FILE.read_text())
    quat = np.asarray(arm.get_state()["ee_pos"][3:7], float)
    return [close_door.CloseGeometry(door, 90.0, (R.from_euler("z", yaw, degrees=True) * R.from_quat(quat)).as_quat(),
                                     0.30, dict(F.CLOSE_SETTINGS), F.CLOSE_TO_DEG, container)
            for yaw in F.CLOSE_YAWS_DEG]


def test_without_base_moves_the_push_plans_no_wheelchair_move_and_the_open_refuses(fridge, monkeypatch):
    arm, planner, _ = fridge
    monkeypatch.setattr(open_door, "PLACEMENT_SEARCH_S", 0.0)
    # as above: the push plans only after a wheelchair move -- which base_moves false never plans
    arm.unreachable = lambda pos, quat, obstacles: isinstance(obstacles, list) and door_is_open(obstacles)
    with pytest.raises(dc.Refused, match="base_moves false: no wheelchair move was planned"):
        open_door.run(arm, planner, False, base_moves=False)
    assert arm.sent == [] and push_open.NEXT_MOVE == 0.0
    with pytest.raises(dc.Refused, match="base_moves false"):
        open_door.run(arm, planner, True, chair_moved=True, base_moves=False)
    # the placement advice leaves out the cells that need the move
    monkeypatch.setattr(F, "PLACEMENT_CELLS", ((0.65, -0.10), (0.85, -0.10)))
    monkeypatch.setattr(F, "CELLS_NEEDING_BASE_MOVE", ((0.65, -0.10),))
    assert open_door._cells() == [(0.85, -0.10)]
    open_door.BASE_MOVES = True
    assert open_door._cells() == [(0.65, -0.10), (0.85, -0.10)]


def test_the_push_starts_behind_the_open_door_and_meets_it_on_its_arc(fridge):
    """User, 11-09: any part of the hand (or container) may push, at any angle, anywhere on the door (further out for
    leverage); only reaching ~45 deg matters. The push starts round the hinge past the open door, clear of it, tried
    furthest from the hinge first, and goes round the hinge at that distance until it meets the door at 45."""
    arm, planner, _ = fridge
    for container in (False, True):
        geos = _close_geometries(arm, planner, container)
        radii = geos[0].radii()
        assert radii == sorted(radii, reverse=True)
        assert radii[0] == pytest.approx(geos[0].df.length - min(F.CLOSE_RADII_FROM_EDGE_M))  # past the free edge first
        found = 0
        for geo, r in [(g, r) for g in geos for r in radii]:
            deg0, sweep = geo.push(r)
            if deg0 is None:
                continue
            found += 1
            assert deg0 > 90.0  # behind the open door
            assert geo.clear_of_door(geo.at(r, deg0)) >= F.CLOSE_START_GAP_M - 1e-9
            dists = np.array([np.linalg.norm((p - geo.h)[:2]) for _, p in sweep])
            r0 = np.linalg.norm((geo.at(r, deg0) - geo.h)[:2])
            assert np.all(np.diff(dists) <= 1e-9)  # round the hinge, drawing in (never out) ...
            assert dists[-1] >= min(r0, F.CLOSE_PUSH_END_RADIUS_M) - 1e-6  # ... no nearer than CLOSE_PUSH_END_RADIUS_M
            end = sweep[-1][1]
            gap = min(close_door._box_gap(geo.body(end), b) for b in geo.end_door)
            assert abs(gap) < 2e-3  # it ends touching the door at 45 deg: the hand, or the container
        assert found


def test_the_container_pushes_too_and_its_route_is_checked(fridge, monkeypatch):
    arm, planner, _ = fridge
    geo = _close_geometries(arm, planner, container=True)[0]
    pos, quat = geo.at(0.3, 170.0), geo.quat
    pts = close_door.body_points(pos, quat, True)
    assert len(pts) == len(close_door._hand_points(pos, quat)) + len(close_door._DIRS)
    c = close_door.container_centre(pos, quat)
    assert np.linalg.norm(c - pos) == pytest.approx(F.CONTAINER_CENTRE_TOOL_M)
    # cuRobo doesn't see the container: a route that would carry it into the door is refused here
    checked = []
    monkeypatch.setattr(close_door, "container_route_gap", lambda *a: checked.append(1) or 0.0)
    arm.gripper_pos = 0.8  # the gripper closed: it holds the container
    with pytest.raises(dc.Refused):
        close_door.run(arm, planner, False, chair_back=0.0)  # every route refused for the container
    assert checked  # cuRobo's routes were checked for the container
    checked.clear()
    arm.gripper_pos = 0.0  # open: an empty hand
    assert close_door.run(arm, planner, False, chair_back=0.0)  # no container: the same plans
    assert not checked


def test_push_starts_out_of_the_arms_reach_are_not_tried(fridge, monkeypatch):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    from door_common import open_door as shared

    monkeypatch.setattr(shared, "MAX_REACH", 0.3)
    asked = len(arm.pose_goals)
    with pytest.raises(dc.Refused, match="beyond the arm's reach"):
        close_door.run(arm, planner, False, chair_back=0.0)
    assert len(arm.pose_goals) == asked  # nothing sent to cuRobo


def test_the_close_refuses_without_an_opened_door_or_a_planned_move(fridge):
    arm, planner, _ = fridge
    with pytest.raises(dc.Refused, match="the open writes it"):
        close_door.run(arm, planner, False)
    dc.DOOR_FILE.write_text(json.dumps({"hinge": [0, 0, 0]}))
    with pytest.raises(dc.Refused, match="no opened door"):
        close_door.run(arm, planner, False)
    assert open_door.run(arm, planner, False, save_door=True)
    with pytest.raises(dc.Refused, match="no close has planned a wheelchair move"):
        close_door.run(arm, planner, True, chair_moved=True)


def test_the_close_asks_for_the_smallest_move_and_an_executed_close_uses_it(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    # the door where it stands now: no plan; moved with the chair: plans -- during the move AND after it (11-12: a move
    # is only offered if the after-move plan, the one an executed close makes, plans too)
    first = {}

    def unreachable(pos, quat, obstacles):
        if not (isinstance(obstacles, list) and door_is_open(obstacles)):
            return False
        door = np.round(next(b for b in obstacles if b["name"] == "door")["position"], 4).tolist()
        return first.setdefault("door", door) == door

    arm.unreachable = unreachable
    assert close_door.run(arm, planner, False)  # dry: picks and saves the move
    move = close_door.LAST_CHAIR_BACK
    assert move == F.CLOSE_CHAIR_MOVES_M[1] and json.loads(dc.DOOR_FILE.read_text())["close_chair_back"] == move
    with pytest.raises(dc.Refused, match="WHEELCHAIR"):  # executed without the move made: asks for it
        close_door.run(arm, planner, True)
    assert arm.sent == []
    assert close_door.run(arm, planner, False, chair_moved=True)  # a dry run with chair_moved plans what the execute will
    assert close_door.LAST_CHAIR_BACK == move and arm.sent == []
    arm.unreachable = lambda *a: False
    assert close_door.run(arm, planner, True, chair_moved=True)  # made: closes
    assert len(arm.sent) == 1


def test_holding_the_container_the_close_only_turns_the_hand_about_vertical(fridge, monkeypatch):
    """The container stays upright: holding it, every hand orientation the close tries keeps the tilt (and every leg
    holds it); with an empty hand it may also tip the hand down (any part may push, at any angle -- user, 11-09)."""
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    quat0 = np.asarray(arm.get_state()["ee_pos"][3:7])
    up0 = R.from_quat(quat0).inv().apply([0.0, 0.0, 1.0])
    seen = {}

    def capture(planner_, geos, *a, **k):
        seen["quats"] = [g.quat for g in geos.values()]
        return [], "stop here"

    monkeypatch.setattr(close_door, "search_starts", capture)
    tilts = {}
    for label, grip in (("container", 0.8), ("empty", 0.0)):
        arm.gripper_pos = grip
        with pytest.raises(dc.Refused):
            close_door.run(arm, planner, False, chair_back=0.0)
        tilts[label] = [np.degrees(np.arccos(np.clip(R.from_quat(q).inv().apply([0.0, 0.0, 1.0]) @ up0, -1, 1)))
                        for q in seen["quats"]]
    assert max(tilts["container"]) < 1e-3  # holding it: turned about vertical only
    assert max(tilts["empty"]) == pytest.approx(max(F.CLOSE_PITCHES_DEG), abs=1e-3)  # empty: tipped down too


# --- searches, timed events, retiming: edge cases ----------------------------------------------

REAL_TIMED_EVENTS = planning._timed_events  # the fixture replaces it with an instant one


class _Clock:
    """planning's clock, put forward by the time the arm takes to run each trajectory."""

    def __init__(self):
        self.offset = 0.0

    def monotonic(self):
        return time.monotonic() + self.offset

    def sleep(self, s):
        time.sleep(min(s, 0.01))


def test_each_search_gets_the_whole_budget_however_long_the_arm_moved(fridge, monkeypatch):
    """The budget times one search, not the action: a re-plan after the arm has moved (the push
    after the swing) gets all of it, however long the motions before it took."""
    arm, planner, world = fridge
    clock = _Clock()
    monkeypatch.setattr(planning, "time", clock)
    run = arm.execute_joint_trajectory

    def slow_arm(names, points, **kw):
        res = run(names, points, **kw)
        clock.offset += 1000.0  # every motion takes far longer than the budget
        arm.q = arm.q + np.radians(0.3)  # ends off the plan: what follows is planned afresh
        return res

    arm.execute_joint_trajectory = slow_arm
    planner = planning.Planner(arm, world, arm.fk, budget_s=F.PLAN_BUDGET_S)
    assert open_door.run(arm, planner, True)
    assert len(arm.sent) == 3
    # a search that does run past it still stops
    planner.search()
    clock.offset += F.PLAN_BUDGET_S + 1
    p, quat = arm.fk.pose(arm.q)
    with pytest.raises(planning.OverBudget):
        planner.motion([planning.free(p + [0.07, 0, 0], quat, "z", [])], arm.q)


def test_a_trajectory_that_stops_short_drops_its_later_events(fridge, monkeypatch):
    arm, planner, _ = fridge
    fired = []
    arm.execute_joint_trajectory = lambda names, points, **kw: types.SimpleNamespace(
        error_code=-1, error_string="e-stopped", final_error=np.zeros(7))
    q = np.asarray(arm.q, float)
    p, quat = arm.fk.pose(q)
    joined = Plan(True, "", np.array([0.02, 0.04]), np.vstack([q, q]), np.zeros((2, 7)))
    motion = planning.Motion([planning.free(p, quat, "here", [])], [], q, joined=joined)
    monkeypatch.setattr(planning, "_timed_events", REAL_TIMED_EVENTS)
    assert not planner.run(motion, "swing", events=[(0.0, lambda: fired.append("close")),
                                                   (0.5, lambda: fired.append("open"))])
    time.sleep(0.7)
    assert fired in ([], ["close"])  # the gripper never opens after the arm stopped


def _blocked_by(names):
    return Plan(
        False,
        "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start state collides with the "
        f"world [start collides with {', '.join(names)}] [world: 9 boxes]",
    )


def test_start_not_clear_names_what_blocks_it():
    e = planning.StartNotClear("x: " + _blocked_by(["door (29 mm)", "person/lap (0 mm)"]).message)
    assert e.blocking() == ["door", "person/lap"] and not e.only_fridge()
    door_and_handle = _blocked_by(["door (29 mm)", "handle (4 mm)"]).message
    assert planning.StartNotClear(door_and_handle).only_fridge()
    assert not planning.StartNotClear("INVALID_START_STATE_JOINT_LIMITS").only_fridge()


def test_the_push_open_comes_to_rest_at_exactly_its_angle_however_fast_it_runs(fridge, monkeypatch):
    """User, 11-09: the push must land on PUSH_OPEN_TO_DEG at any speed setting: the hand (the door isn't held) comes
    to rest there, its last PUSH_EASE_DEG no faster than PUSH_END_JOINT_SPEED, then steps off in a leg of its own."""
    import rammp_arm

    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False)
    goals = open_door.LAST_PLAN["after"].motion.goals
    labels = [g.label for g in goals]
    k = next(i for i, g in enumerate(goals) if g.label.startswith("onto the inner face"))
    push, off = goals[k], goals[k + 1]
    assert push.blend == 0.0, labels  # it stops at 90 ...
    assert goals[k - 1].blend > 0.0  # ... and nowhere before it: in behind the door, it blends onto the inner face
    arm2, planner2, _ = arm, planner, None
    motion = open_door.LAST_PLAN["after"].motion
    planner2.join(motion, [])
    radii = arm2.blends[-1][1]
    assert radii[k] == 0.0 and not motion.notes  # joined with the true stop at 90
    df = FridgeModel(json.loads(dc.DOOR_FILE.read_text()) if dc.DOOR_FILE.exists() else
                     open_door.opened(open_door.LAST_PLAN["door"], F.PUSH_OPEN_PULL_DEG)).df
    assert abs(df.angle_of(push.through[-1][0]) - df.angle_of(push.through[0][0])) == pytest.approx(
        F.PUSH_OPEN_TO_DEG - F.PUSH_OPEN_PULL_DEG, abs=0.5)
    assert "off the door" in off.label
    # the end of the push: the same joint speed limit at full speed and at 0.2
    ends = []
    for speed in (1.0, 0.2):
        monkeypatch.setattr(rammp_arm, "JOINT_SPEED", 0.85 * speed)
        monkeypatch.setattr(rammp_arm, "CART_SPEED", 0.16 * speed)
        caps = push_open.push_caps([F.PUSH_OPEN_PULL_DEG, F.PUSH_OPEN_TO_DEG])
        scale = rammp_arm.JOINT_SPEED * rammp_arm.contact_speed_scale() / planning.LOCAL_JOINT_SPEED
        ends.append(caps[-1] * planning.LOCAL_JOINT_SPEED * scale)
        assert caps[0] == 1.0
    assert max(ends) <= F.PUSH_END_JOINT_SPEED + 1e-9


def _refusal(arm, planner, handle, normal=(-1.0, 0.0, 0.0), **kw):
    from door_common import open_door as shared

    shared.detect_handle = lambda: (np.asarray(handle, float), np.array([-0.5, 0.5, 0.5, -0.5]),
                                    np.asarray(normal, float), None, None, None)
    with pytest.raises(dc.Refused) as e:
        open_door.run(arm, planner, False, **kw)
    return str(e.value)


def test_a_refusal_about_feasibility_leads_with_where_to_move_the_wheelchair(fridge, monkeypatch):
    _placement_region(monkeypatch)
    arm, planner, _ = fridge
    monkeypatch.setattr(open_door, "PLACEMENT_SEARCH_S", 0.0)
    arm.unreachable = lambda pos, quat, obstacles: True  # nothing plans anywhere
    msg = _refusal(arm, planner, [0.85, -0.04, 0.30])  # right of the region (y -0.04 > -0.15)
    assert "no grasp" in msg and "Wheelchair placement (" in msg and "to its left" in msg
    assert "NOT the cause" not in msg
    print("\n=== feasibility example ===\n" + msg)
    msg = _refusal(arm, planner, [0.85, -0.22, 0.30])  # inside it: nothing about placement
    assert "no grasp" in msg and "Wheelchair placement" not in msg and "wheelchair" not in msg.lower()


def test_a_refusal_about_something_else_mentions_placement_only_as_an_aside(fridge, monkeypatch):
    _placement_region(monkeypatch)
    arm, planner, _ = fridge
    arm.start_blocked = True  # the arm's start isn't clear: not about where the wheelchair is
    turned = (-0.96, 0.28, 0.0)  # the door 16 deg off square: outside the region's +-10 deg
    msg = _refusal(arm, planner, [0.85, -0.04, 0.30], normal=turned)  # right of the region, and turned
    assert msg.index("start pose is not clear") < msg.index("NOT the cause of this stop")
    assert "to its left" in msg and "front swings right" in msg and "Wheelchair placement (" not in msg
    msg = _refusal(arm, planner, [0.85, -0.22, 0.30])  # inside, square: no aside
    assert "start pose is not clear" in msg and "wheelchair" not in msg.lower()
    print("\n=== aside example ===\n" + _refusal(arm, planner, [0.85, -0.04, 0.30], normal=turned))


def _placement_region(monkeypatch, xs=(0.75, 0.95), ys=(-0.30, -0.15), yaw=(-10.0, 10.0)):
    """A made-up placement region for the advice tests, independent of the real table (constants.MOUNTS): every
    5 cm cell from xs[0] to xs[1] and ys[0] to ys[1] (handle x, y in arm_mount), and the door yaw window `yaw`. The
    defaults: x 0.75-0.95 (5 cells), y -0.30 to -0.15 (4 cells), so a handle at y -0.04 is right of it and one at
    x 0.85, y -0.22 is well inside it."""
    step = open_door.PLACEMENT_CELL_M
    cells = tuple((round(x, 3), round(y, 3)) for x in np.arange(xs[0], xs[1] + 1e-9, step)
                  for y in np.arange(ys[0], ys[1] + 1e-9, step))
    monkeypatch.setattr(F, "PLACEMENT_CELLS", cells)
    monkeypatch.setattr(F, "PLACEMENT_X", tuple(xs))
    monkeypatch.setattr(F, "PLACEMENT_Y", tuple(ys))
    monkeypatch.setattr(F, "PLACEMENT_YAW_DEG", tuple(yaw))


def test_the_placement_advice_does_not_flip_with_centimetres_of_detection_noise(fridge, monkeypatch):
    """11-09: x 0.797 got "1 cm back", x 0.856 "1 cm closer" (each snapped to its nearest cell's centre) when only y was
    off. The advice now moves to the region's nearest point and drops moves under PLACEMENT_DEADBAND_M."""
    _placement_region(monkeypatch)
    n = np.array([-1.0, 0.0, 0.0])
    # x between the region's cell centres (0.80, 0.85, 0.90), y off to its right: only y may move
    for x in (0.797, 0.81, 0.83, 0.856):
        advice = open_door.placement_advice([x, -0.06, 0.32], n)
        assert "closer to the fridge" not in advice and "back from the fridge" not in advice, advice
        assert "to its left" in advice  # y -0.06 is right of the region (y <= -0.15): the chair goes left
    dx, dy = open_door._placement_move(0.80, -0.06)
    assert dx == 0.0 and dy < -0.05
    # inside: no move at all
    assert open_door._placement_move(0.85, -0.22) == (0.0, 0.0)
    assert "inside that placement" in open_door.placement_advice([0.85, -0.22, 0.32], n)


def test_a_refused_open_searches_the_wheelchair_moves_that_make_it_plan(fridge, monkeypatch):
    """The advice the planner checked: the open is planned again with the fridge shifted as wheelchair moves would
    shift it, the smallest that plans reported."""
    arm, planner, _ = fridge
    # cuRobo can't reach the grasp unless the handle is 5 cm further left (+y) than detected
    ok_y = HANDLE[1] + 0.05
    real_ik = planning.Planner.leg

    def leg(self, goal, q_start, start_pose=None, from_arm=False):
        if "line up" in goal.label and goal.pos[1] < ok_y - 0.01:
            return None, f"{goal.label}: cuRobo found no plan (MotionGenStatus.IK_FAIL: no collision-free joint)"
        return real_ik(self, goal, q_start, start_pose, from_arm)

    monkeypatch.setattr(planning.Planner, "leg", leg)
    budget = planner.budget_s
    with pytest.raises(dc.Refused) as e:
        open_door.run(arm, planner, False)
    msg = str(e.value)
    assert "checked by planning it: the open plans" in msg and "to its right" in msg and "~5 cm" in msg
    assert planner.budget_s == budget  # the search's per-try budget put back
    monkeypatch.setattr(open_door, "PLACEMENT_SEARCH_S", 0.0)
    with pytest.raises(dc.Refused) as e:
        open_door.run(arm, planner, False)
    assert "checked by planning" not in str(e.value)


def test_a_close_starting_inside_the_door_tries_the_moves_that_take_the_door_away(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    plan_to_pose = arm.plan_to_pose

    def door_in_the_way(pos, quat, start_joints, hold=0):
        world = arm.worlds[-1] if arm.worlds else []
        # the door where it stands (not where a move takes it): the hand is within the planner's padding of it
        if isinstance(world, list) and door_where_it_stands(world):
            return _blocked_by(["door (29 mm)"])
        return plan_to_pose(pos, quat, start_joints, hold)

    arm.plan_to_pose = door_in_the_way
    assert close_door.run(arm, planner, False)
    assert close_door.LAST_CHAIR_BACK == F.CLOSE_CHAIR_MOVES_M[1]
    # the route for that move leaves out the door where it stands now
    swept = [
        w for w in arm.worlds if isinstance(w, list) and any(b["name"].startswith("m") for b in w)
    ]
    assert swept and all(not any(b["name"] == "door" for b in w) for w in swept)


def test_a_close_whose_start_is_blocked_by_unnamed_boxes_tries_the_other_moves(fridge):
    """The planner's explanation of a refused start can fail (no box list): the moves are tried."""
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    plan_to_pose = arm.plan_to_pose

    def unexplained(pos, quat, start_joints, hold=0):
        world = arm.worlds[-1] if arm.worlds else []
        if isinstance(world, list) and door_where_it_stands(world):  # the door where it stands, not once moved
            return Plan(
                False,
                "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start state collides "
                "(attribution unavailable: shape '[1, 1, 1, 3]' is invalid)",
            )
        return plan_to_pose(pos, quat, start_joints, hold)

    arm.plan_to_pose = unexplained
    assert close_door.run(arm, planner, False)
    assert close_door.LAST_CHAIR_BACK == F.CLOSE_CHAIR_MOVES_M[1]


def test_a_close_starting_inside_the_chair_refuses_at_once(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    arm.plan_to_pose = lambda *a, **k: _blocked_by(["person/lap (3 mm)"])
    with pytest.raises(dc.Refused, match="person/lap"):
        close_door.run(arm, planner, False)
    assert close_door.LAST_CHAIR_BACK is None


def test_a_move_whose_start_is_blocked_costs_one_plan(fridge):
    """A start blocked in a move's world rules that move out at once: every height and hand turn
    would start from the same pose in the same world."""
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    calls = []

    def blocked(pos, quat, start_joints, hold=0):
        calls.append(pos)
        return _blocked_by(["door (29 mm)"] if door_is_open(arm.worlds[-1]) else ["m1_door (5 mm)"])

    arm.plan_to_pose = blocked
    with pytest.raises(dc.Refused, match="no wheelchair move"):
        close_door.run(arm, planner, False)
    assert len(calls) == len(F.CLOSE_CHAIR_MOVES_M)


def test_a_goal_with_no_arm_configuration_is_not_asked_again_from_another_start(fridge):
    """IK_FAIL is about the goal pose alone: another start (the next grasp's) can't change it -- 11-09 sim, the same
    impossible push placements asked again for every grasp ran a search out of its budget."""
    arm, planner, _ = fridge
    p, quat = arm.fk.pose(arm.q)
    asked = []
    arm.unreachable = lambda pos, q, obstacles: asked.append(1) or True
    goal = planning.free(p + [0.05, 0, 0], quat, "in behind the door", [])
    assert planner.leg(goal, arm.q)[0] is None
    assert planner.leg(goal, np.asarray(arm.q) + 0.05)[0] is None  # another start: not asked again
    assert len(asked) == 1
    arm.unreachable = lambda *a: False  # a goal that failed otherwise (not unreachable) is still asked again
    other = planning.free(p + [0.06, 0, 0], quat, "y", [])
    arm.blend_fails = False
    assert planner.leg(other, arm.q)[0] is not None


def test_a_leg_whose_curobo_plan_raised_is_asked_again(fridge):
    """A raise is no planning answer (RAMMP-CuRobo reports it as EXCEPTION): one more try."""
    arm, planner, _ = fridge
    plan_to_pose = arm.plan_to_pose
    answers = []

    def raises_once(pos, quat, start_joints, hold=0):
        answers.append(pos)
        if len(answers) == 1:
            return Plan(False, "EXCEPTION: cuRobo plan_single raised: shape '[1, 1, 1, 3]' ...")
        return plan_to_pose(pos, quat, start_joints, hold)

    arm.plan_to_pose = raises_once
    p, quat = arm.fk.pose(arm.q)
    motion, why = planner.motion([planning.free(p + [0.05, 0, 0], quat, "x", [])], arm.q)
    assert motion is not None and len(answers) == 2
    # a planning answer (no plan) is not asked again
    arm.plan_to_pose = lambda *a, **k: answers.append(1) or Plan(False, "MotionGenStatus.IK_FAIL")
    n = len(answers)
    assert planner.motion([planning.free(p + [0.06, 0, 0], quat, "y", [])], arm.q)[0] is None
    assert len(answers) == n + 1


def test_the_push_open_ends_in_front_of_the_fridge_facing_it_in_the_same_leg(fridge):
    """After the push and its step-off, on to finish_pose in the push's one leg: no stop, no blend."""
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, True)
    door = json.loads(dc.DOOR_FILE.read_text())
    pos, quat = push_open.finish_pose(door)
    df = DoorFrame(door, door["open_sign"])
    # facing into the fridge (the closed door's inward normal), upright, mid-opening
    np.testing.assert_allclose(R.from_quat(quat).as_matrix()[:, 2], -np.r_[df.n, 0.0], atol=1e-9)
    assert R.from_quat(quat).as_matrix()[2, 1] == pytest.approx(1.0)
    u = float((pos[:2] - df.hinge[:2]) @ df.e)
    assert u == pytest.approx(df.length * F.PUSH_OPEN_FINISH_U, abs=1e-6)
    ee = np.asarray(arm.get_state()["ee_pos"], float)  # at one of the stand-offs (the first that plans)
    ends = [push_open.finish_pose(door, out) for out in F.PUSH_OPEN_FINISH_OUT_M]
    assert min(np.linalg.norm(ee[:3] - p) for p, _ in ends) < 0.01
    assert abs(float(np.dot(ee[3:7], quat))) == pytest.approx(1.0, abs=1e-3)
    assert len(arm.sent) == 3  # approach, swing, and the push-open: still one trajectory


def test_the_push_open_ends_at_its_step_off_if_the_hand_would_pass_near_the_door(fridge, monkeypatch):
    arm, planner, _ = fridge
    monkeypatch.setattr(F, "PUSH_OPEN_FINISH_CLEAR_M", 10.0)
    assert open_door.run(arm, planner, False)
    after = open_door.LAST_PLAN["after"]
    assert "off the door" in after.finish and "from the door" in after.finish
    assert "in front of the fridge" not in after.motion.goals[-1].label


def test_the_pushed_door_keeps_its_front_and_leaves_out_the_hands_window(fridge, tmp_path):
    door = {**_closed_door(tmp_path), "door_open_deg": 70.0}
    model = FridgeModel(door)
    df = model.df
    u, z = 0.35, float(door["handle_mid_z"])
    boxes = {b["name"]: b for b in model.pushed(70.0, u, z, 0.08, 0.25, 0.15, 0.025)}
    assert {"body", "handle", "door_panel"} <= set(boxes)
    panel = boxes["door_panel"]
    assert panel["dims"][0] == pytest.approx(0.025) and panel["dims"][1] == pytest.approx(df.length)
    # the panel sits in the door's front 2.5 cm, well clear of the inner face the hand presses
    off = (df._rot(70.0).T @ (np.asarray(panel["position"][:2]) - df.hinge[:2])) @ df.n - df.mid
    assert off == pytest.approx(df.t / 2 - 0.0125, abs=1e-4)
    inner = [b for n, b in boxes.items() if n.startswith("door_inner")]
    assert inner and all(b["dims"][0] == pytest.approx(df.t - 0.025) for b in inner)
    # nothing of the inner part where the hand is
    hand = df.pos(u, -df.t / 2, 70.0, z)
    assert all(close_door._box_gap(hand[None], b) > 0.0 for b in inner)


def test_a_through_leg_is_checked_segment_by_segment_against_its_own_worlds(fridge):
    arm, planner, _ = fridge
    p, quat = arm.fk.pose(arm.q)
    def box(name, x):
        return [{"name": name, "position": [x, 0, 0], "rpy_deg": [0, 0, 0], "dims": [0.1] * 3}]

    worlds = [("first", box("a", 5)), ("second", box("b", 6))]
    poses = [(p + [0.02, 0, 0], quat), (p + [0.04, 0, 0], quat)]
    goal = planning.through(poses, "push", [], segment_worlds=worlds)
    blends = len(arm.blends)
    motion, why = planner.motion([goal], arm.q)
    assert motion is not None and len(arm.blends) == blends + 3  # the whole leg, then each segment
    assert [b["name"] for b in arm.worlds[-2]] == ["a"]
    assert [b["name"] for b in arm.worlds[-1]] == ["b"]
    seen = []
    blend = arm.blend

    def refuse_second(legs, radii, **kw):
        seen.append(arm.worlds[-1])
        if arm.worlds[-1] and arm.worlds[-1][0]["name"] == "b":
            return Plan(False, "MotionGenStatus.COLLISION: b")
        return blend(legs, radii, **kw)

    arm.blend = refuse_second
    goal = planning.through([(p + [0.03, 0, 0], quat), (p + [0.05, 0, 0], quat)], "push2", [],
                            segment_worlds=worlds)
    motion, why = planner.motion([goal], arm.q)
    assert motion is None and "against second" in why


def test_the_push_is_checked_against_the_door_where_it_is_at_each_step(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False)
    goals = open_door.LAST_PLAN["after"].motion.goals
    push = next(g for g in goals if g.label.startswith("onto the inner face"))
    assert push.path == "through" and len(push.segment_worlds) == len(push.through)
    off = goals[next(i for i, g in enumerate(goals) if g is push) + 1]  # the step-off after the stop: against the door at 90
    assert all(label.startswith("the door at 90 deg") for label, _ in off.segment_worlds)
    labels = [label for label, _ in push.segment_worlds]
    assert labels[0].startswith(f"the door at {F.PUSH_OPEN_PULL_DEG} deg")  # where the pull stopped
    assert labels[-1].startswith("the door at 90 deg")
    assert all(any(b["name"] == "door_panel" for b in w) for _, w in push.segment_worlds)
