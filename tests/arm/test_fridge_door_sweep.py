"""The slab can hit the chair even when the arm's world checks would pass."""

import numpy as np
import pytest
import yaml
from fridge import push_open
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
    assert DoorSweep(model, [box((0.7, 0.08, 0))], 0, 0).obstacle() == "person/legs"
    assert DoorSweep(model, [box((0.7, 0.08, 0.5))], 0, 0).obstacle() is None


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
    with pytest.raises(SweepBlocked, match="person/legs"):
        check_open(model, world, push_open.pull_angles(60), push_open.PushOpen(base_moves=False))
    assert check_open(model, world, push_open.pull_angles(60), push_open.PushOpen()) == (60,)
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
    with pytest.raises(ValueError, match="must supply person, seat and platform"):
        deployment_boxes(config)
