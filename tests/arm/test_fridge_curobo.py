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
from fridge import close_door, open_door, planning, push_open
from fridge import constants as F
from fridge.kinematics import Chain
from fridge.world import FridgeModel, PlannerWorld, chair_stand_in, hand_reach, scene_camera
from fridge_fake import (  # noqa: F401 -- fridge: the fixture
    HANDLE,
    URDF,
    FakeCurobo,
    Plan,
    door_is_open,
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
                     extra=[scene_camera(F.SCENE_CAMERA_XYZ, F.SCENE_CAMERA_BOX_M)])
    box = {"name": "b", "position": [1.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0], "dims": [0.1, 0.1, 0.1]}
    w.push([box], "test")
    assert arm.world_calls == [("/rammp_planner_worlds/fridge_world.yaml", False)]  # as the planner sees it
    scene = (tmp_path / "w" / "fridge_world.yaml").read_text()
    assert "layer_mebot_floor.yaml" in scene and "scene_camera" in scene and "base_frame: arm_mount" in scene
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


def test_join_works_round_blends_stop_after_a_cut_corner():
    """RAMMP-CuRobo 2.0.1's blend rejects a stop right after a blended junction: join gives it BLEND_MIN_M."""
    arm = FakeCurobo()
    w = PlannerWorld(arm, "/tmp/fridge_join_test")
    planner = planning.Planner(arm, w, arm.fk)
    p, quat = arm.fk.pose(arm.q)
    goals = [planning.free(p + [0.02 * k, 0, 0], quat, f"g{k}", [], blend=b) for k, b in ((1, 0.03), (2, 0.0), (3, 0.0))]
    motion, _ = planner.motion(goals, arm.q)
    planner.join(motion, [])
    assert arm.blends[-1][1] == [0.03, planning.BLEND_MIN_M]


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


def test_the_release_arc_lasts_long_enough_for_the_gripper_to_open(fridge):
    arm, planner, _ = fridge
    seen = {}
    run = planner.run

    def spy(motion, label, impedance=None, events=()):
        if label == "swing + release":
            seen["t_open"], seen["end"] = events[-1][0], float(motion.joined.t[-1])
        return run(motion, label, impedance, events)

    planner.run = spy
    assert open_door.run(arm, planner, True)
    assert seen["end"] - (seen["t_open"] + F.OPEN_LEAD_S) >= F.OPEN_DURING_S - 1e-6


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


def test_the_close_runs_as_one_trajectory_never_tilting_and_leaves_the_door_to_swing_shut(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, True)
    sent = len(arm.sent)
    assert close_door.run(arm, planner, True, hold_offset_x=0.0)
    assert len(arm.sent) == sent + 1 and "control_mode" not in arm.sent[-1]
    motion = arm.blends[-1]
    assert all(h == planning.HOLD_LEVEL for h in motion[3])  # every leg holds the hand's tilt
    assert json.loads(dc.DOOR_FILE.read_text())["door_open_deg"] == F.CLOSE_TO_DEG
    assert close_door.LAST_CHAIR_BACK == 0.0


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
    arm.unreachable = lambda pos, quat, obstacles: isinstance(obstacles, list) and door_is_open(obstacles)
    assert close_door.run(arm, planner, False, hold_offset_x=0.0)  # dry: picks and saves the move
    move = close_door.LAST_CHAIR_BACK
    assert move == F.CLOSE_CHAIR_MOVES_M[1] and json.loads(dc.DOOR_FILE.read_text())["close_chair_back"] == move
    with pytest.raises(dc.Refused, match="WHEELCHAIR"):  # executed without the move made: asks for it
        close_door.run(arm, planner, True, hold_offset_x=0.0)
    assert arm.sent == []
    arm.unreachable = lambda *a: False
    assert close_door.run(arm, planner, True, chair_moved=True, hold_offset_x=0.0)  # made: closes
    assert len(arm.sent) == 1


def test_the_close_may_turn_the_hand_about_vertical_but_never_tilts_it(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    quat0 = np.asarray(arm.get_state()["ee_pos"][3:7])
    first = []

    def only_turned(pos, quat, obstacles):  # cuRobo reaches the swing start only with the hand turned
        if not first:
            first.append(quat)
            return True
        return False

    arm.unreachable = only_turned
    assert close_door.run(arm, planner, True, hold_offset_x=0.0)
    up = [0.0, 0.0, 1.0]
    tilt = np.degrees(np.arccos(np.clip(R.from_quat(arm.get_state()["ee_pos"][3:7]).inv().apply(up)
                                         @ R.from_quat(quat0).inv().apply(up), -1, 1)))
    assert tilt < F.CLOSE_MAX_TILT_DEG


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


def test_a_swing_whose_release_never_lasts_long_enough_refuses_before_it(fridge, monkeypatch):
    arm, planner, _ = fridge
    monkeypatch.setattr(F, "OPEN_DURING_S", 1e6)
    with pytest.raises(dc.Refused, match="release lasts only"):
        open_door.run(arm, planner, True)
    assert len(arm.sent) == 1  # the approach only: the swing never sent


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


def test_a_close_starting_inside_the_door_tries_the_moves_that_take_the_door_away(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    plan_to_pose = arm.plan_to_pose

    def door_in_the_way(pos, quat, start_joints, hold=0):
        world = arm.worlds[-1] if arm.worlds else []
        # the door where it stands: the hand is within the planner's padding of it
        if isinstance(world, list) and door_is_open(world):
            return _blocked_by(["door (29 mm)"])
        return plan_to_pose(pos, quat, start_joints, hold)

    arm.plan_to_pose = door_in_the_way
    assert close_door.run(arm, planner, False, hold_offset_x=0.0)
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
        if isinstance(world, list) and door_is_open(world):
            return Plan(
                False,
                "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION: the start state collides "
                "(attribution unavailable: shape '[1, 1, 1, 3]' is invalid)",
            )
        return plan_to_pose(pos, quat, start_joints, hold)

    arm.plan_to_pose = unexplained
    assert close_door.run(arm, planner, False, hold_offset_x=0.0)
    assert close_door.LAST_CHAIR_BACK == F.CLOSE_CHAIR_MOVES_M[1]


def test_a_close_starting_inside_the_chair_refuses_at_once(fridge):
    arm, planner, _ = fridge
    assert open_door.run(arm, planner, False, save_door=True)
    arm.plan_to_pose = lambda *a, **k: _blocked_by(["person/lap (3 mm)"])
    with pytest.raises(dc.Refused, match="person/lap"):
        close_door.run(arm, planner, False, hold_offset_x=0.0)
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
        close_door.run(arm, planner, False, hold_offset_x=0.0)
    assert len(calls) == len(F.CLOSE_CHAIR_MOVES_M)


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
    push = open_door.LAST_PLAN["after"].motion.goals[-1]
    assert push.path == "through" and len(push.segment_worlds) == len(push.through)
    labels = [label for label, _ in push.segment_worlds]
    assert labels[0].startswith("the door at 68 deg")
    assert labels[-1].startswith("the door at 90 deg")
    assert all(any(b["name"] == "door_panel" for b in w) for _, w in push.segment_worlds)
