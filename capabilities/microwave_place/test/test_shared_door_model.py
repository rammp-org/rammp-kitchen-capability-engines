"""Focused tests for wiring the recorded microwave door into shared collision helpers."""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / "capabilities/microwave_place/microwave_place/microwave_place_node.py"
COMMON = ROOT / "shared/door_common/door_common/common.py"


class Refused(Exception):
    pass


def function(path, name, scope):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), scope)
    return scope[name]


def test_recorded_open_angle_drives_shared_door_boxes(tmp_path):
    record = {
        "door_open_deg": 51.0,
        "open_sign": -1.0,
        "closed_normal": [1.0, 0.0, 0.0],
        "closed_grasp_pos": [0.2, 0.3, 0.5],
        "hinge": [0.1, 0.3, 0.5],
    }
    path = tmp_path / "door.json"
    path.write_text(json.dumps(record))
    calls = {}

    class DoorFrame:
        def __init__(self, door, sign):
            calls["frame"] = (door, sign)

        def bodies_point(self, angle):
            calls["angle"] = angle
            return [0.4, 0.2, 0.5]

    boxes = [
        ("door", (0.1, 0.2, 0.3), (0.02, 0.3, 0.25), 0.5),
        ("microwave body", (0.1, 0.2, 0.3), (0.2, 0.3, 0.25), 0.0),
    ]

    def door_model_boxes(door, open_grasp_pos):
        calls["model"] = (door, open_grasp_pos)
        return boxes, {"open_deg": 51.0}

    common = SimpleNamespace(DoorFrame=DoorFrame, door_model_boxes=door_model_boxes)
    scope = {"json": json, "place_motion": SimpleNamespace(Refused=Refused)}
    shared_model = function(NODE, "_shared_door_model", scope)
    _, open_grasp_pos, actual_boxes, info = shared_model(path, common)

    assert calls["frame"] == (record, -1.0)
    assert calls["angle"] == 51.0
    assert calls["model"] == (record, open_grasp_pos)
    assert actual_boxes == boxes
    assert info["open_deg"] == 51.0


def test_scene_box_and_shared_door_boxes_form_door_world():
    scene_boxes = [
        {
            "name": "microwave",
            "position": [0.4, 0.5, 0.6],
            "dims": [0.8, 0.4, 0.5],
            "rpy_deg": [0.0, 0.0, 30.0],
        }
    ]
    door_boxes = [
        ("door", (0.1, 0.2, 0.3), (0.02, 0.3, 0.25), 0.5),
        ("microwave body", (0.2, 0.3, 0.4), (0.2, 0.3, 0.25), 0.0),
    ]
    scope = {"np": np, "json": json, "place_motion": SimpleNamespace(Refused=Refused)}
    scene_converter = function(NODE, "_shared_curobo_boxes", scope)
    world_fn = function(
        COMMON,
        "door_world",
        {
            "np": np,
            "re": re,
            "CUROBO_FLOOR_LAYER": "/config/robot/curobo/layer_mebot_floor.yaml",
            "DOOR_WORLD_PAD_M": 0.015,
        },
    )
    world = world_fn(scene_converter(scene_boxes, door_boxes))

    assert world["base_frame"] == "arm_mount"
    assert world["layers"] == ["/config/robot/curobo/layer_mebot_floor.yaml"]
    assert [box["name"] for box in world["obstacles"]] == [
        "door_model_microwave",
        "door_model_door",
        "door_model_microwave_body",
    ]
    np.testing.assert_allclose(world["obstacles"][0]["dims"], [0.83, 0.43, 0.53])
    assert world["obstacles"][1]["dims"] == [0.07, 0.63, 0.53]


def test_missing_open_angle_refuses_shared_model(tmp_path):
    path = tmp_path / "door.json"
    path.write_text(json.dumps({"closed_normal": [1, 0, 0]}))
    scope = {"json": json, "place_motion": SimpleNamespace(Refused=Refused)}
    shared_model = function(NODE, "_shared_door_model", scope)

    with pytest.raises(Refused, match="no door_open_deg"):
        shared_model(path, SimpleNamespace())


def test_place_uses_shared_pybullet_and_curobo_helpers():
    tree = ast.parse(NODE.read_text())
    capability = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Capability")
    place = next(item for item in capability.body if isinstance(item, ast.FunctionDef) and item.name == "_place")
    calls = {
        node.func.attr
        for node in ast.walk(place)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert {"add_door_model", "door_in_curobo", "search_place"} <= calls
    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "door_world_loaded"
        for node in ast.walk(place)
    )
