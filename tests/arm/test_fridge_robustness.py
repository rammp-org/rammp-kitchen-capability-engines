"""Safety checks for the fridge's collision budget and grasp/release handoffs."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from door_common.common import Refused
from fridge import open_door, push_open, world_budget
from fridge_fake import fridge  # noqa: F401 -- parametrized arm and planner fixture
from scipy.spatial.transform import Rotation

FIXTURES = Path(__file__).parent / "fixtures"


def box(name, x, y=0.0):
    return dict(name=name, position=[x, y, 0], dims=[0.1, 0.1, 0.1], rpy_deg=[0, 0, 0])


def test_budget_keeps_near_and_fixed_boxes_and_logs_far_drop(capsys):
    fixed, near, far = box("handle", 0.7), box("room_near", 0.9), box("room_far", 2)
    assert world_budget.fit_boxes(
        [fixed], [near, far], budget=60, reserved=58, padding=0.02, label="insert"
    ) == [near, fixed]
    assert "dropped room_far" in capsys.readouterr().out


def test_budget_merge_contains_original_rotated_boxes():
    a, b = box("a", 0.8), box("b", 0.8)
    yaw = Rotation.from_euler("z", 31, degrees=True)
    b["position"] = (np.array(a["position"]) + yaw.apply([0.1, 0, 0])).tolist()
    a["rpy_deg"] = b["rpy_deg"] = [0, 0, 31]
    (merged,) = world_budget.fit_boxes(
        [], [a, b], budget=1, reserved=0, padding=0.02, label="route"
    )
    for original in (a, b):
        center = yaw.inv().apply(np.subtract(original["position"], merged["position"]))
        assert np.all(
            abs(center) + np.asarray(original["dims"]) / 2 <= np.asarray(merged["dims"]) / 2
        )


def test_budget_refuses_unmergeable_near_boxes():
    with pytest.raises(Refused, match=r"61 boxes still needed with collision_cache_obb=60"):
        world_budget.fit_boxes(
            [],
            [box("a", 0.7), box("b", 0.7, 0.3)],
            budget=60,
            reserved=59,
            padding=0.02,
            label="route",
        )


def test_empty_close_gets_one_new_view_then_refuses(fridge):
    arm, planner, _ = fridge
    close = arm.gripper
    looks = []
    miss = json.loads((FIXTURES / "fridge_ws11_s.json").read_text())["dock-00-2027"]
    assert not miss["contact_events"] and miss["angle_deg"] == 0.0

    def empty(position, **kwargs):
        close(position, **kwargs)
        if position > 0.5:
            arm.gripper_pos = 1.0

    def retry():
        looks.append(len(arm.sent))
        return open_door.detect(arm)

    arm.gripper = empty
    with pytest.raises(open_door.GraspUnconfirmed, match="empty close"):
        open_door.run(
            arm, planner, True, push=False, around=False, swing_deg=60, detect_retry=retry
        )
    assert looks == [2]
    assert [position for position, _, _ in arm.grips] == [1, 0, 1, 0]
    assert len(arm.sent) == 3  # approach, collision-checked retreat, approach


def test_close_uses_measured_feedback_in_preference_to_legacy_state():
    arm = SimpleNamespace(
        gripper=lambda *args, **kwargs: True,
        gripper_state=lambda: SimpleNamespace(position=0.68, current=0.0, present=True),
        get_state=lambda: {"gripper_pos": 1.0},
    )
    open_door.confirm_grasp(arm)


def test_calibrated_tracking_miss_can_replan_the_push():
    data = json.loads((FIXTURES / "fridge_ws11_s.json").read_text())["dock-20-1531"]
    plan = dict(swing_deg=data["target_deg"], after=push_open.PushOpen((1, 0)))
    assert open_door.can_replan_swing(plan, data["estimated_deg"], data["end_error_m"])
    assert not open_door.can_replan_swing(plan, data["estimated_deg"], 0.06)
    assert not open_door.can_replan_swing(plan, 35, data["end_error_m"])


def test_kitchen_031_fits_the_deployed_cache():
    path = FIXTURES / "fridge_ws11_s_kitchen_031.json"
    data = json.loads(path.read_text())
    before = data["reserved"] + len(data["required"]) + len(data["surroundings"])
    assert before == 61
    fitted = world_budget.fit_boxes(**data)
    assert len(fitted) + data["reserved"] == 60
    assert fitted[-len(data["required"]) :] == data["required"]


def test_world_collision_at_release_plans_a_checked_clearance(monkeypatch):
    from fridge import planning

    normal = np.array([0.0, 1.0])
    source = json.loads((FIXTURES / "fridge_ws11_s.json").read_text())["dock-23-2027"]
    assert "INVALID_START_STATE_WORLD_COLLISION" in source["detail"]
    door = {"door_open_deg": 45.4}  # estimate used by the original planner
    model = SimpleNamespace(
        df=SimpleNamespace(_rot=lambda deg: np.eye(2), n=normal),
        full=lambda deg: [box("body", 1), box("handle", 0.7)],
    )
    retreat = planning.line(
        np.array([0.6, 0.2, 0.4]), np.array([0.0, 0.0, 0.0, 1.0]), "retreat", [box("body", 1)]
    )
    monkeypatch.setattr(push_open, "FridgeModel", lambda door: model)
    monkeypatch.setattr(push_open, "back_off_goals", lambda *args: [retreat])
    monkeypatch.setattr(push_open, "release_geometry", lambda *args: np.zeros((1, 3)))
    after = push_open.PushOpen((1, 0), base_moves=False)
    monkeypatch.setattr(after, "safe_routes", lambda *args, **kwargs: [(0, 90)])
    checked = []

    def motion(goals, q, **kwargs):
        checked.append(goals[0])
        return planning.Motion(goals, [SimpleNamespace(q=np.array([q]))], q), None

    def route(planner, door, q, pose, world, model, **kwargs):
        assert {b["name"] for b in world} == {"body", "handle"}
        if len(checked) == 1:
            return None, source["detail"], ""
        assert pose[0][1] == pytest.approx(retreat.pos[1] + push_open.RELEASE_CLEARANCE_M)
        return planning.Motion([], [], q), (0.35, 0, 1, False), ""

    monkeypatch.setattr(push_open, "plan_finished_push", route)
    planner = SimpleNamespace(motion=motion, world=None)
    ok, _ = after.plan(planner, door, np.zeros(7), (retreat.pos, retreat.quat), from_arm=True)
    assert ok
    assert len(checked) == 2
    assert checked[1].world == retreat.world
    assert len(after.motion.goals) == 2
    assert all(goal.blend == 0.0 for goal in after.motion.goals)


def test_retry_stationary_check_requires_full_world_validation(fridge):
    from fridge import planning

    arm, planner, world = fridge
    q = np.asarray(arm.get_state()["position"])
    obstacles = [box("handle", 0.8)]
    arm.plan_to_joints = lambda *args, **kwargs: SimpleNamespace(
        success=False, message="LIBRARY_VALIDATION_FAILED: arm_joint_6 velocity 1.250 > limit"
    )

    def validate(legs, radii, **kwargs):
        assert world.loaded[-1] == obstacles[0]
        np.testing.assert_allclose(legs[0].q, np.tile(q, (3, 1)))
        assert not legs[0].qd.any() and not radii
        return SimpleNamespace(success=True)

    arm.blend = validate
    assert planner.preflight_start(obstacles, q, q, grasp_retry=True) is None

    arm.blend = lambda *args, **kwargs: SimpleNamespace(success=False, message="infeasible state")
    with pytest.raises(planning.StartNotClear, match="stationary check: infeasible state"):
        planner.preflight_start(obstacles, q, q, grasp_retry=True)


@pytest.mark.parametrize("margin", [8.9, 9.0])
def test_final_joined_hook_withdrawal_clearance(monkeypatch, margin):
    from unittest.mock import Mock

    joints = np.zeros((5, 7))
    after = open_door.HookBackOff()
    after.motion = SimpleNamespace(joined=None)
    planner = Mock()
    planner.join.side_effect = lambda motion, world: setattr(
        motion, "joined", SimpleNamespace(q=joints)
    )
    planner.limit_margin.return_value = margin
    monkeypatch.setattr(open_door, "load_door", lambda: {})
    monkeypatch.setattr(open_door, "FridgeModel", Mock())
    monkeypatch.setattr(push_open, "after_release", Mock(return_value="planned"))

    if margin < open_door.HOOK_EXIT_LIMIT_MARGIN_DEG:
        with pytest.raises(Refused, match="joint clearance"):
            after.execute(Mock(get_state=lambda: {"velocity": [0.0]}), planner)
        planner.run.assert_not_called()
    else:
        assert after.execute(Mock(get_state=lambda: {"velocity": [0.0]}), planner)
        planner.run.assert_called_once()
    assert planner.limit_margin.call_args.args[0] is joints


@pytest.mark.parametrize("speed", [0.0, 0.1])
def test_pull_release_uses_settle_speed_and_allows_loaded_offset(monkeypatch, speed):
    from unittest.mock import Mock

    from fridge import planning

    # A successful driver result, a valid tool endpoint and a 2.5° loaded joint offset.
    zeros = np.zeros(7)
    plan = SimpleNamespace(t=[0.0, 1.0], q=[zeros, zeros], qd=[zeros, zeros])
    motion = SimpleNamespace(q0=zeros, joined=plan, goals=[SimpleNamespace(pos=np.zeros(3))])
    idle = {"position": zeros, "ee_pos": np.zeros(7), "velocity": zeros}
    loaded = dict(idle, position=np.full(7, np.radians(2.5)), velocity=np.full(7, speed))
    arm = Mock()
    arm.get_state.side_effect = [idle, loaded]
    arm.execute_joint_trajectory.return_value = SimpleNamespace(error_code=0)
    monkeypatch.setattr(
        planning, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda s: None)
    )
    planner = planning.Planner(arm, None, None)

    if speed:
        with pytest.raises(Refused, match="still moving"):
            planner.run(motion, "swing", settle_for_release=True)
    else:
        assert planner.run(motion, "swing", settle_for_release=True)


def test_release_only_wait_timeout_does_not_open_the_gripper(monkeypatch):
    from unittest.mock import Mock

    arm = Mock(get_state=lambda: {"velocity": [0.1]})
    stamps = iter([0.0, 0.31])
    monkeypatch.setattr(push_open, "time", SimpleNamespace(monotonic=lambda: next(stamps)))
    with pytest.raises(Refused, match="no release or withdrawal"):
        push_open._release(arm)
    arm.gripper.assert_not_called()
