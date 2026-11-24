"""Planning regressions from the two November 16 fridge captures."""

import json
import time
from pathlib import Path

import numpy as np
import pybullet as bullet
import pytest
from door_common import common
from door_common import open_door as shared
from fridge import constants, open_door, planning, world
from fridge.door_sweep import DoorSweep
from fridge.world import FridgeModel, PlannerWorld
from fridge_fake import HANDLE_QUAT, URDF, FakeCurobo, Plan

from capability_engine import sim

FIXTURES = Path(__file__).parent / "fixtures" / "ws11"
# Padded by the sweep, as in the v2 pinned planner world.
V2_LEGS = dict(
    name="person/legs", position=[-0.00150, -0.35075, -0.18200], dims=[0.8668, 0.4827, 0.6776]
)


@pytest.fixture(autouse=True)
def restore_configuration():
    modules = (common, shared, constants, world)
    saved = [dict(vars(module)) for module in modules]
    yield
    for module, state in zip(modules, saved, strict=True):
        vars(module).update(state)


def recorded(index, tmp_path):
    data = json.loads(sorted(FIXTURES.glob("*.json"))[index].read_text())
    q = np.asarray(data["joints_rad"])
    normal = np.asarray(data["door_normal"])
    handle = np.asarray(data["handle_m"])
    open_door.configure(tmp_path, 30)
    hinge, _, _, fields = shared.FIND_HINGE(handle, HANDLE_QUAT, normal, None)
    geom = dict(
        h_raw=handle,
        n=normal,
        orient=HANDLE_QUAT,
        hinge=hinge,
        hinge_seen=None,
        door_mid=None,
        door_z=None,
        hinge_fields=fields,
    )
    arm = FakeCurobo(q)
    seat = [dict(box, name=f"seat/{box['name']}") for box in data["seat_boxes"]]
    planner = planning.Planner(
        arm, PlannerWorld(arm, tmp_path / "worlds", extra=[*seat, V2_LEGS]), arm.fk
    )
    return data, geom, planner


def start_clearance(q, boxes):
    sim.reset_for_tests()
    try:
        _, robot = sim.make_sim(str(URDF))
        client = robot.physics_client_id
        bodies = []
        for box in boxes:
            shape = bullet.createCollisionShape(
                bullet.GEOM_BOX,
                halfExtents=np.array(box["dims"]) / 2 + 0.020,
                physicsClientId=client,
            )
            bid = bullet.createMultiBody(
                0,
                shape,
                basePosition=box["position"],
                baseOrientation=bullet.getQuaternionFromEuler(
                    np.radians(box.get("rpy_deg", [0, 0, 0]))
                ),
                physicsClientId=client,
            )
            bodies.append((bid, box["name"]))
        return sim.clearance(robot, q, bodies)
    finally:
        sim.reset_for_tests()


@pytest.mark.parametrize("supports_repair", [True, False])
def test_nov16_armrest_collision_plans_bounded_staging_only_if_backend_repairs(
    tmp_path, monkeypatch, supports_repair
):
    data, geom, planner = recorded(0, tmp_path)
    arm = planner.arm
    distance, contact = start_clearance(arm.q, data["seat_boxes"])
    assert distance == pytest.approx(-0.0194, abs=0.002)
    assert "armrest_arm" in contact
    probes = []

    def plan(target, measured, *, deadline):
        target, measured = np.asarray(target), np.asarray(measured)
        probes.append(target)
        assert deadline > time.monotonic()
        np.testing.assert_array_equal(measured, arm.q)
        if len(probes) == 1 or not supports_repair:
            return Plan(
                False,
                "INVALID_START_STATE_WORLD_COLLISION "
                "[start collides with seat/armrest_arm (18 mm)]",
            )
        # Model the response contract of a planner that supports start-state repair.
        return arm._to(measured, target)

    monkeypatch.setattr(arm, "plan_to_joints", plan)
    with pytest.raises(open_door.StagingRequired) as failure:
        open_door.plan_open(
            planner, geom, arm.q, constants.PUSH_OPEN_PULL_DEG, open_door.PushOpen()
        )

    assert len(probes) == 2
    assert np.max(np.abs(probes[1] - arm.q)) == pytest.approx(np.radians(12.0))
    result = failure.value.payload
    assert result["blocking_bodies"] == [{"body": "seat/armrest_arm", "penetration_mm": 18.0}]
    assert result["staging_planned"] is supports_repair
    assert result["executed"] is False
    assert not arm.sent and not arm.grips


def test_nov16_live_placement_has_bounded_result_and_yaw_advice(tmp_path):
    _, geom, planner = recorded(1, tmp_path)
    arm = planner.arm
    start = time.monotonic()
    try:
        plan = open_door.plan_open(
            planner, geom, arm.q, constants.PUSH_OPEN_PULL_DEG, open_door.PushOpen()
        )
    except open_door.Infeasible as failure:
        result = failure.payload
        assert result["executed"] is False
        assert result["base_rotation_deg"] < 0
        assert "Rotate base by" in result["suggestion"]
        assert result["failure_counts"] and result["candidates"]
        assert any(
            item["reason"] and "limit_margin_deg" in " ".join(item) for item in result["candidates"]
        )
    else:
        assert plan["approach"].joined.success
    assert time.monotonic() - start < 10
    assert not arm.sent and not arm.grips


def test_wrist_wrap_equivalents_keep_pose_and_limits(tmp_path):
    _, _, planner = recorded(1, tmp_path)
    chain = planner.fk
    q = np.zeros(7)
    q[[4, 6]] = np.radians([350, -350])
    alternates = chain.wrist_equivalents(q)
    assert len(alternates) == 4
    pos, _ = chain.poses(alternates)
    np.testing.assert_allclose(pos, np.broadcast_to(pos[0], pos.shape), atol=1e-10)
    assert planner.limit_margin(alternates[-1]) > planner.limit_margin(q)
    assert not chain.wrist_equivalents(np.zeros(7))[1:]


def test_executing_goal_does_not_search_hypothetical_placements(tmp_path, monkeypatch):
    _, geom, planner = recorded(1, tmp_path)
    arm = planner.arm
    monkeypatch.setattr(open_door, "detect", lambda _: (geom, arm.q))
    monkeypatch.setattr(arm, "unreachable", lambda *_: True)
    monkeypatch.setattr(
        open_door, "search_placement", lambda *_: pytest.fail("live goal searched placements")
    )
    with pytest.raises(open_door.Infeasible) as failure:
        open_door.run(arm, planner, True, search_alternate_placements=True)
    assert failure.value.payload["failure_counts"]["line-up"] > 0
    assert not arm.sent and not arm.grips


def test_nov16_side_open_uses_a_shorter_checked_push_handoff(tmp_path):
    _, geom, planner = recorded(1, tmp_path)
    arm = planner.arm
    open_door.configure(tmp_path, 30, grasp="side")
    after = open_door.PushOpen(base_moves=False)

    plan = open_door.plan_open(
        planner, geom, arm.q, constants.PUSH_OPEN_PULL_DEG, after, adaptive_pull=True
    )

    assert 30 <= plan["swing_deg"] < constants.PUSH_OPEN_PULL_DEG
    assert plan["approach"].joined.success
    assert after.chair_move == 0.0
    assert after.end_deg == constants.PUSH_OPEN_TO_DEG
    assert any(goal.label == "in behind the door" for goal in after.motion.goals)
    assert not arm.sent and not arm.grips


@pytest.mark.parametrize("index", [0, 1])
def test_nov16_door_sweep_is_clear_of_seat_and_legs(tmp_path, index):
    _, geom, planner = recorded(index, tmp_path)
    candidate = open_door.grasp_candidates(geom)[0]
    model = FridgeModel(
        open_door.closed_door(
            geom, candidate["h"], candidate.get("ref", candidate["pos"]), candidate
        )
    )
    boxes, padding = planner.world.door_sweep_boxes()
    assert DoorSweep(model, boxes, 0, constants.PUSH_OPEN_TO_DEG, padding).obstacle() is None
    after = open_door.PushOpen(base_moves=False)
    assert after.safe_routes(model, planner.world, 60, in_place=True) == ((0.0, 90.0),)


def test_c2_search_016_door_leg_contact_is_blocked(tmp_path):
    # v2/fake/reliability/search, search-016 hit person/legs with the slab
    # at 50.33 degrees. Recreate the fit using the frozen search observation.
    handle = np.array([0.733491862039143, -0.4067476244761701, 0.33882659325054254])
    yaw = np.radians(13.23552302431958)
    normal = -np.array([np.cos(yaw), np.sin(yaw), 0])
    open_door.configure(tmp_path, 30, grasp="side")
    hinge, _, _, fields = shared.FIND_HINGE(handle, HANDLE_QUAT, normal, None)
    geom = dict(
        h_raw=handle,
        n=normal,
        orient=HANDLE_QUAT,
        hinge=hinge,
        hinge_seen=None,
        door_mid=None,
        door_z=None,
        hinge_fields=fields,
    )
    q = np.radians([103.7074, -32.1909, -170.7741, -123.1256, 102.6052, 110.6805, 2.1714])
    arm = FakeCurobo(q)
    planner = planning.Planner(arm, PlannerWorld(arm, tmp_path / "worlds", extra=[V2_LEGS]), arm.fk)
    with pytest.raises(open_door.Infeasible, match="Door sweep blocked by person/legs") as exc:
        open_door.plan_open(
            planner,
            geom,
            q,
            constants.PUSH_OPEN_PULL_DEG,
            open_door.PushOpen(base_moves=False),
            adaptive_pull=True,
        )
    assert exc.value.payload["executed"] is False
    assert "cm" in exc.value.payload["operator_hint"]
    assert planner.plans == 0
    assert not arm.sent and not arm.grips


@pytest.mark.parametrize("side_succeeds", [False, True])
def test_auto_open_shares_deadline_and_reports_progress(tmp_path, monkeypatch, side_succeeds):
    from types import SimpleNamespace

    _, geom, planner = recorded(1, tmp_path)
    after = open_door.PushOpen(base_moves=False)
    now = [100.0]
    planner.deadline = 119.5
    tried, messages = [], []

    def attempt(active, geom, q, swing, push, diagnostics, staging, adaptive, grasp):
        assert push is after
        tried.append((grasp, adaptive, active.deadline))
        if len(tried) == 1:
            now[0] = active.deadline
            raise planning.OverBudget("front limit")
        if len(tried) == 2 and not side_succeeds:
            now[0] = active.deadline
            raise planning.OverBudget("side limit")
        return dict(approach=None, model=SimpleNamespace(cabinet=lambda: []), name=grasp)

    monkeypatch.setattr(open_door.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(open_door, "_plan_open", attempt)
    monkeypatch.setattr(planner, "join", lambda *_: None)
    result = open_door.plan_open(
        planner,
        geom,
        planner.arm.q,
        60,
        after,
        adaptive_pull=True,
        progress=lambda percent, message: messages.append((percent, message)),
    )
    assert result["name"] == "side"
    assert planner.deadline == 119.5
    assert tried == [("front", True, 112), ("side", False, 119)] + (
        [] if side_succeeds else [("side", True, 119.5)]
    )
    assert [message for _, message in messages] == [
        "Planning front grasp, adaptive pull",
        "Planning side grasp, 60° pull",
    ] + ([] if side_succeeds else ["Planning side grasp, adaptive pull"])
