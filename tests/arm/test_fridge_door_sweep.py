"""The slab can hit the chair even when the arm's world checks would pass."""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fridge import constants, push_open
from fridge.door_sweep import DoorSweep, SweepBlocked, check_open, deployment_boxes
from fridge.world import FridgeModel, PlannerWorld


@pytest.fixture
def model():
    return FridgeModel(
        {
            "hinge": [0, 0, 0],
            "closed_normal": [0, 1, 0],
            "closed_grasp_pos": [1, 0, 0],
            "closed_handle": [1, 0, 0],
            "door_z": [-0.3, 0.3],
            "door_t": 0.04,
            "face_behind_grasp": 0,
            "door_past_handle": 0,
            "body_depth": 0.5,
        }
    )


def box(position, name="person/legs", dims=(0.02, 0.02, 0.02), **kwargs):
    return dict(name=name, position=position, dims=dims, **kwargs)


@pytest.mark.parametrize("sign", [-1, 1])
def test_intermediate_collision_with_clear_endpoints(model, sign):
    model.df.sign = sign
    boxes = [box((0.63, sign * 0.63, 0), rpy_deg=[20, 10, 40])]
    assert DoorSweep(model, boxes, 0, 0).obstacle() is None
    assert DoorSweep(model, boxes, 90, 90).obstacle() is None
    assert DoorSweep(model, boxes, 0, 90).obstacle() == "person/legs"


def test_vertical_separation_and_hinge_margin(model):
    assert DoorSweep(model, [box((0.7, 0.055, 0))], 0, 0).obstacle() == "person/legs"
    assert DoorSweep(model, [box((0.7, 0.055, 0.5))], 0, 0).obstacle() is None
    # After chair padding, 36 mm remains to the slab, within the 42 mm allowance at 90 degrees.
    assert DoorSweep(model, [box((-0.066, 0.7, 0))], 90, 90).obstacle() == "person/legs"
    assert DoorSweep(model, [box((-0.080, 0.7, 0))], 90, 90).obstacle() is None


def test_refusal_reports_a_shift_that_clears_the_sweep(model, tmp_path):
    boxes = [box((0.6, 0.6, 0))]
    world = PlannerWorld(None, tmp_path, extra=boxes)
    with pytest.raises(SweepBlocked) as exc:
        check_open(model, world, (90,), None)
    result = exc.value.payload
    assert result["executed"] is False
    assert result["reason"] == "door_sweep_blocked"
    assert "cm" in result["operator_hint"]
    shift = np.r_[result["fridge_shift_m"], 0]
    assert DoorSweep(model, boxes, 0, 90).obstacle(shift) is None


def test_final_push_angle_and_between_dock_travel(model, tmp_path):
    legs = box((0.04, 0.9, 0))
    world = PlannerWorld(None, tmp_path, extra=[legs])
    after = push_open.PushOpen(base_moves=False)
    with pytest.raises(SweepBlocked, match="person/legs"):
        check_open(model, world, push_open.pull_angles(60), after)
    assert after.safe_routes(model, world, 60, in_place=True) == ()
    assert push_open.PushOpen().safe_routes(model, world, 60, in_place=True) == ()
    assert push_open.PushOpen().safe_routes(model, world, 60, in_place=False)[0][1] == 90
    short_handoff = push_open.PushOpen(allow_short_handoff=True)
    assert short_handoff.safe_routes(model, world, 60, in_place=True)[0][0] != 0.0
    assert short_handoff.safe_routes(model, world, 60, in_place=True)[0][1] == 90
    # Endpoints of a chair translation are clear, but the slab would pass through the seat.
    seat = box((0.5, 0.3, 0), name="seat/armrest")
    sweep = DoorSweep(model, [seat], 0, 0)
    shift = np.array([0, 0.6, 0])
    assert sweep.obstacle() is None
    assert sweep.obstacle(shift) is None
    assert DoorSweep(model, [seat], 0, 0, translation=shift).obstacle() == "seat/armrest"


def test_deployment_layers_match_planner_config(tmp_path):
    config = tmp_path / "planner.yaml"
    layers = [f"{name}.yaml" for name in ("person", "seat", "platform")]
    config.write_text(yaml.safe_dump({"layers": layers, "planner": {"world_padding": 0.024}}))
    for layer in layers:
        name = layer.removesuffix(".yaml")
        (tmp_path / layer).write_text(
            yaml.safe_dump({"layer": name, "obstacles": [box([0, 0, 0], name="part")]})
        )
    boxes, padding = deployment_boxes(config)
    assert [b["name"] for b in boxes] == ["person/part", "seat/part", "platform/part"]
    assert padding == 0.024
    config.write_text(yaml.safe_dump({"layers": layers[:-1]}))
    boxes, _ = deployment_boxes(config)
    assert [b["name"] for b in boxes] == ["person/part", "seat/part"]
    config.write_text(yaml.safe_dump({"layers": layers[:1]}))
    with pytest.raises(ValueError, match="must supply nonempty person and seat"):
        deployment_boxes(config)


def test_resumed_push_refuses_before_planning_at_blocked_dock(model, tmp_path, monkeypatch):
    world = PlannerWorld(None, tmp_path, extra=[box((0.55, 0.75, 0))])
    ai = SimpleNamespace(get_state=lambda: {"gripper_pos": 0})
    door = dict(model.door, door_open_deg=40, push_after_chair_move=0.05)
    monkeypatch.setattr(push_open, "load_door", lambda: door)
    with pytest.raises(SweepBlocked, match="person/legs") as exc:
        push_open.push_after_chair_move(ai, SimpleNamespace(world=world), True, (1, 0))
    assert exc.value.payload["executed"] is False


def test_push_floor_block_keeps_reposition_hint(model, tmp_path):
    angle = np.deg2rad(constants.PUSH_OPEN_MIN_FINAL_DEG)
    legs = box((0.9 * np.cos(angle), 0.9 * np.sin(angle), 0))
    world = PlannerWorld(None, tmp_path, extra=[legs])
    with pytest.raises(SweepBlocked, match="person/legs") as exc:
        check_open(model, world, push_open.pull_angles(60), push_open.PushOpen(base_moves=False))
    shift = np.r_[exc.value.payload["fridge_shift_m"], 0]
    sweep = DoorSweep(model, [legs], 30, 90, push_grid=True)
    assert sweep.clear_push_end(90) is None
    assert sweep.clear_push_end(90, shift) == 90


def test_resumed_push_uses_and_records_the_checked_endpoint(model, tmp_path, monkeypatch):
    world = PlannerWorld(None, tmp_path, extra=[box((-0.11, 0.9, 0))])
    ai = SimpleNamespace(
        get_state=lambda: {
            "gripper_pos": 0,
            "position": np.zeros(7),
            "ee_pos": [0, 0, 0, 0, 0, 0, 1],
        }
    )
    door = dict(model.door, door_open_deg=40, push_after_chair_move=0.05)
    planner = SimpleNamespace(
        world=world, search=lambda: None, join=lambda *_: None, run=lambda *_, **__: True
    )
    saved = {}
    monkeypatch.setattr(push_open, "load_door", lambda: door)
    monkeypatch.setattr(push_open, "save_door_geometry", lambda **fields: saved.update(fields))
    monkeypatch.setattr(push_open, "FINAL_ANGLE_DEG", None)

    def plan(*args, end_deg, **kwargs):
        moved = args[1]
        moved_model = FridgeModel(moved)
        boxes, padding = world.door_sweep_boxes()
        sweep = DoorSweep(moved_model, boxes, 40, 90, padding, push_grid=True)
        assert end_deg == 90
        assert end_deg == sweep.clear_push_end(90)
        return SimpleNamespace(goals=[]), (0.35,), ""

    monkeypatch.setattr(push_open, "plan_finished_push", plan)
    assert push_open.push_after_chair_move(ai, planner, True, (1, 0))
    assert saved["door_open_deg"] == 90
    assert saved["push_after_chair_move"] is None
    assert saved["door_open_deg"] == push_open.FINAL_ANGLE_DEG


def test_support_cabinet_blocks_the_dock_and_its_translation(model, tmp_path):
    model.door.update(support_floor_z=-0.5, support_height=0.5)
    (support,) = model.support()
    assert support in model.full(30)
    assert support in model.cabinet()
    np.testing.assert_allclose(support["position"], [0.5, -0.25, -0.25])

    # The feet are below the door and can still hit the stationary support.
    feet = box((0.5, -0.25, -0.42))
    clear_model = FridgeModel({**model.door, "support_height": 0.0})
    assert DoorSweep(clear_model, [feet], 0, 90).obstacle() is None
    sweep = DoorSweep(model, [feet], 0, 90)
    assert sweep.obstacle() == "person/legs"
    assert sweep.clear_push_end(90) is None
    with pytest.raises(SweepBlocked):
        check_open(model, PlannerWorld(None, tmp_path, extra=[feet]), (60,), push_open.PushOpen())

    near = box((1.20, -0.25, -0.42))
    assert DoorSweep(model, [near], 0, 0).obstacle() is None
    assert (
        DoorSweep(model, [near], 0, 0, translation=np.array([0.3, 0, 0])).obstacle()
        == "person/legs"
    )


@pytest.mark.parametrize(
    ("hinge", "angle", "reuse"),
    [([0, 0, 0], 40, True), ([0.02, 0, 0], 40, False), ([0, 0, 0], 42, False)],
)
def test_push_reuse_requires_agreement_with_the_release_geometry(model, hinge, angle, reuse, monkeypatch):
    after = push_open.PushOpen(base_moves=False)
    after.chair_move, after.end_deg = 0.0, 90.0
    after.release_geometry = push_open.release_geometry(model, 40)
    after.safe_routes = lambda *args, **kwargs: ((0.0, 90.0),)
    monkeypatch.setattr(push_open, "load_door", lambda: dict(model.door, hinge=hinge, door_open_deg=angle))
    observed = []
    monkeypatch.setattr(push_open, "after_release", lambda *args, reuse: observed.append(reuse))
    planner = SimpleNamespace(world=None, join=lambda *args: None, run=lambda *args, **kwargs: False)

    assert not after.execute(None, planner)
    assert observed == [reuse]


def test_parallel_push_avoids_marginal_free_edge_contact(model, monkeypatch):
    door = dict(model.door, door_open_deg=40)
    placements = ((0.97, 0, 1), (0.80, 0, 1))
    monkeypatch.setattr(push_open, "placements", lambda: placements)
    planned = object()
    planner = SimpleNamespace(motion=lambda *args, **kwargs: (planned, None))
    pose = (np.zeros(3), np.array([0, 0, 0, 1]))

    motion, placement = push_open.plan_push(planner, door, np.zeros(7), pose, [])
    assert motion is planned
    assert placement[:3] == placements[1]
