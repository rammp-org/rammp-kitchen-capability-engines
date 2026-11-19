"""Offline geometry and lifecycle regressions; no ROS graph or robot motion."""

import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

PACKAGE = Path(__file__).parents[1] / "microwave_grab"
spec = importlib.util.spec_from_file_location("grab_collision_world", PACKAGE / "collision_world.py")
world = importlib.util.module_from_spec(spec)
spec.loader.exec_module(world)

# Use the production cavity coordinate transforms without importing ROS/PyBullet.
tree = ast.parse((PACKAGE / "place_motion.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Oven")
scope = {"np": np}
exec(compile(ast.Module(body=[cls], type_ignores=[]), "Oven", "exec"), scope)


@pytest.fixture
def oven():
    return scope["Oven"](
        dict(
            insertion_axis=[1, 0, 0],
            lateral_axis_left=[0, 1, 0],
            opening_corners=[[0, -0.1524, 0], [0, 0.1524, 0], [0, 0.1524, 0.1905], [0, -0.1524, 0.1905]],
            point=[0.1524, 0, 0.0159],
            floor_z=0.0159,
        )
    )


@pytest.fixture
def door():
    return dict(
        closed_normal=[-1, 0, 0],
        hinge=[0, 0.17, 0.1],
        closed_grasp_pos=[-0.067, -0.17, 0.1],
        door_z=[-0.03, 0.24],
        door_open_deg=90,
    )


def inside(box, p):
    a = np.radians(box["rpy_deg"][2])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return bool(np.all(np.abs((np.asarray(p) - box["position"]) @ R) <= np.asarray(box["dims"]) / 2))


def test_shell_leaves_cavity_open_and_blocks_control_panel(oven):
    boxes = world.shell_boxes(oven, [-0.28, 0.21, -0.04, 0.25], [0.3048, 0.1905], 0.3048, 0.0286, 0.02)
    for x in [-0.02, 0, 0.10, 0.30]:
        assert not any(inside(b, [x, 0, 0.10]) for b in boxes)
    for point in [[-0.01, -0.22, 0.1], [0.10, 0.19, 0.1], [0.10, 0, 0.23], [0.10, 0, -0.02], [0.315, 0, 0.1]]:
        assert any(inside(b, point) for b in boxes), point
    assert len(boxes) == 5


@pytest.mark.parametrize("outer", [[-0.1, 0.21, -0.04, 0.25], [-0.28, 0.21, 0.01, 0.25], [np.nan, 0.21, -0.04, 0.25]])
def test_partial_or_invalid_outer_bounds_refused(oven, outer):
    with pytest.raises(world.InvalidGeometry):
        world.shell_boxes(oven, outer, [0.3048, 0.1905], 0.3048, 0.0286, 0.02)


def test_left_door_opens_outward_and_uses_handle_width(oven, door):
    box = world.door_box(oven, door, 90, 0.08, 10, (0.3048, 0.1905))
    assert box["dims"] == pytest.approx([0.04, 0.40, 0.31])
    assert box["position"][0] == pytest.approx(-0.20)
    assert box["position"][1] == pytest.approx(0.15)
    assert inside(box, [-0.2, 0.15, 0.1])
    assert not inside(box, [0.2, 0.15, 0.1])  # never swing the slab into the oven


def test_geometry_rotates_and_translates_with_the_microwave(oven, door):
    first = world.door_box(oven, door, 90, 0.08, 10, (0.3048, 0.1905))
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([0.7, -0.3, 0.2])
    oven.n, oven.w, oven.b0 = R @ oven.n, R @ oven.w, R @ oven.b0 + shift
    transformed = {
        k: (R @ np.asarray(v) + shift).tolist() for k, v in door.items() if k not in ("door_z", "door_open_deg")
    }
    transformed["door_open_deg"] = 90
    transformed["closed_normal"] = (R @ np.asarray(door["closed_normal"])).tolist()
    transformed["door_z"] = (np.asarray(door["door_z"]) + shift[2]).tolist()
    second = world.door_box(oven, transformed, 90, 0.08, 10, (0.3048, 0.1905))
    np.testing.assert_allclose(second["position"], R @ first["position"] + shift)
    assert second["dims"] == pytest.approx(first["dims"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("hinge", [0.5, 0.17, 0.1]),
        ("closed_normal", [0, 1, 0]),
        ("door_z", None),
        ("door_z", [0.5, 0.8]),
        ("door_open_deg", 50),
        ("closed_grasp_pos", [np.nan, 0, 0]),
    ],
)
def test_stale_or_incomplete_registration_refused(oven, door, key, value):
    door[key] = value
    with pytest.raises(world.InvalidGeometry):
        world.door_box(oven, door, 90, 0.08, 10, (0.3048, 0.1905))


def test_door_file_must_be_recent_and_not_a_demo_without_geometry(tmp_path):
    path = tmp_path / "door.json"
    path.write_text(json.dumps({"hinge": [0, 0, 0]}))
    os.utime(path, (1, 1))
    with pytest.raises(world.InvalidGeometry, match="old"):
        world.read_door(path, 600)


@pytest.mark.parametrize("fail_at", ["load", "motion", None])
def test_restore_on_load_failure_cancel_or_success(fail_at):
    arm = SimpleNamespace(set_world=Mock(return_value=(True, "restored")))
    loader = Mock(return_value="loaded")
    if fail_at == "load":
        loader.side_effect = RuntimeError("load failed")
    try:
        with world.loaded_world(arm, "/shared/scene.yaml", "/config/world.yaml", loader, Mock()):
            if fail_at == "motion":
                raise RuntimeError("cancelled")
    except RuntimeError:
        assert fail_at is not None
    arm.set_world.assert_called_once_with("/config/world.yaml", even_if_cancelled=True)


def test_restore_failure_is_reported():
    arm = SimpleNamespace(set_world=Mock(return_value=(False, "planner unavailable")))
    with pytest.raises(world.InvalidGeometry, match="restore"):
        with world.loaded_world(arm, "/shared/scene.yaml", "/config/world.yaml", Mock(), Mock()):
            pass


def test_scene_keeps_floor_and_writes_geometry_for_review(tmp_path):
    path = world.write_scene(tmp_path, "arm_mount", ["/config/floor.yaml"], [], {"angle": 90})
    import yaml

    scene = yaml.safe_load(path.read_text())
    assert scene == dict(base_frame="arm_mount", layers=["/config/floor.yaml"], obstacles=[])
    assert json.loads(path.with_suffix(".json").read_text()) == {"angle": 90}


def test_local_path_refuses_door_contact_before_execution():
    checker = SimpleNamespace(clearance=Mock(side_effect=[(0.2, ""), (0.2, ""), (-0.01, "wrist vs door")]))
    checkpoint = Mock()
    with pytest.raises(world.InvalidGeometry, match="wrist vs door"):
        world.check_local_path(checker, [[0] * 7, [1] * 7], lambda qs, step: qs, 0.5, 0.01, checkpoint, "insertion")
    assert checkpoint.call_count == 2


def test_outer_rectangle_excludes_open_door_and_interior_points(oven):
    # Camera looks along world +X: optical (u,v,d) -> world (d,u,v).
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    T[:3, 3] = [-1, 0, 0.1]
    depth = np.full((30, 40), 1 - 0.0286)
    K = np.array([[70, 0, 20], [0, 70, 15], [0, 0, 1]])
    depth[:, :3] = 0.7  # open door in front
    depth[5:25, 8:32] = 1.2  # cavity behind the front
    shot = dict(depth_m=depth, mask=np.ones_like(depth, bool), K=K, world_T_cam=T)
    outer = world.outer_bounds(oven, shot, 0.0286)
    assert outer[0] > -0.25
    assert outer[1] < 0.3
    depth[:] = 1.2
    with pytest.raises(world.InvalidGeometry, match="exterior-front"):
        world.outer_bounds(oven, shot, 0.0286)
