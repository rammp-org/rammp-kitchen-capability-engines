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
    for point in [[-0.01, -0.22, 0.1], [0.10, 0.19, 0.1], [0.10, 0, 0.23]]:
        assert any(inside(b, point) for b in boxes), point
    assert {b["name"] for b in boxes} == {"microwave_control_panel", "microwave_left_wall", "microwave_top"}
    for point in [[0.10, 0, -0.02], [0.315, 0, 0.1]]:
        assert not any(inside(b, point) for b in boxes)


@pytest.mark.parametrize("outer", [[-0.1, 0.21, -0.04, 0.25], [-0.28, 0.21, 0.01, 0.25], [np.nan, 0.21, -0.04, 0.25]])
def test_partial_or_invalid_outer_bounds_refused(oven, outer):
    with pytest.raises(world.InvalidGeometry):
        world.shell_boxes(oven, outer, [0.3048, 0.1905], 0.3048, 0.0286, 0.02)


def test_left_door_opens_outward_and_uses_handle_width(oven, door):
    box = world.door_box(oven, door, 90, 0.08, 10, (0.3048, 0.1905), [-0.28, 0.21, -0.04, 0.25], 0.0286)
    assert box["dims"] == pytest.approx([0.04, 0.40, 0.31])
    assert box["position"][0] == pytest.approx(-0.2286)
    assert box["position"][1] == pytest.approx(0.1724)
    assert inside(box, [-0.2, 0.1724, 0.1])
    assert not inside(box, [0.2, 0.15, 0.1])  # never swing the slab into the oven


def test_geometry_rotates_and_translates_with_the_microwave(oven, door):
    first = world.door_box(oven, door, 90, 0.08, 10, (0.3048, 0.1905), [-0.28, 0.21, -0.04, 0.25], 0.0286)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([0.7, -0.3, 0.2])
    oven.n, oven.w, oven.b0 = R @ oven.n, R @ oven.w, R @ oven.b0 + shift
    transformed = {
        k: (R @ np.asarray(v) + shift).tolist() for k, v in door.items() if k not in ("door_z", "door_open_deg")
    }
    transformed["door_open_deg"] = 90
    transformed["closed_normal"] = (R @ np.asarray(door["closed_normal"])).tolist()
    transformed["door_z"] = (np.asarray(door["door_z"]) + shift[2]).tolist()
    second = world.door_box(oven, transformed, 90, 0.08, 10, (0.3048, 0.1905), [-0.28, 0.21, -0.04, 0.25], 0.0286)
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
        world.door_box(oven, door, 90, 0.08, 10, (0.3048, 0.1905), [-0.28, 0.21, -0.04, 0.25], 0.0286)


def test_old_door_file_is_accepted_without_changing_timestamp(tmp_path):
    path = tmp_path / "door.json"
    path.write_text(json.dumps({"hinge": [0, 0, 0]}))
    os.utime(path, (1, 1))
    assert world.read_door(path) == {"hinge": [0, 0, 0]}
    assert path.stat().st_mtime == 1
    path.write_text("[]")
    with pytest.raises(world.InvalidGeometry, match="object"):
        world.read_door(path)


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


def test_expected_layers_follow_active_config_not_flatbot(tmp_path):
    import yaml

    for name, count in [("seat", 17), ("person", 3), ("platform", 16)]:
        (tmp_path / f"{name}.yaml").write_text(
            yaml.safe_dump(dict(layer=name, meta=dict(frame="arm_mount"), obstacles=[{}] * count))
        )
    config = tmp_path / "planner.yaml"
    config.write_text(yaml.safe_dump(dict(layers=["seat.yaml", "person.yaml"])))
    assert world.planner_layers(config) == {"seat": 17, "person": 3}
    config.write_text("layers: []\n")
    with pytest.raises(world.InvalidGeometry):
        world.planner_layers(config)


def test_door_registration_ignores_allowed_saved_pose_drift(oven, door):
    outer = [-0.28, 0.21, -0.04, 0.25]
    original = world.door_box(oven, door, 90, 0.08, 10, (0.3048, 0.1905), outer, 0.0286)
    # Move both hinge and grasp without changing the saved door dimensions.
    for key in ("hinge", "closed_grasp_pos"):
        door[key] = (np.asarray(door[key]) + [0.04, -0.02, 0.01]).tolist()
    door["door_z"] = (np.asarray(door["door_z"]) + 0.01).tolist()
    registered = world.door_box(oven, door, 90, 0.08, 10, (0.3048, 0.1905), outer, 0.0286)
    np.testing.assert_allclose(original["position"], registered["position"])
    np.testing.assert_allclose(original["dims"], registered["dims"])


def test_submillimetre_wall_fit_is_refused(oven):
    with pytest.raises(world.InvalidGeometry, match="thinner than 5 mm"):
        world.shell_boxes(oven, [-0.28, 0.1531, -0.04, 0.25], [0.3048, 0.1905], 0.3048, 0.0286, 0.02)


def test_fixed_control_panel_and_borders_preserve_open_cavity(oven):
    outer = world.fixed_outer_bounds((0.3048, 0.1905), 0.127, 0.0254, 0.02)
    assert outer == pytest.approx([-0.2794, 0.1724, -0.0254, 0.2159])
    boxes = world.shell_boxes(oven, outer, (0.3048, 0.1905), 0.3048, 0.0286, 0.02)
    panel = next(b for b in boxes if b["name"] == "microwave_control_panel")
    assert panel["dims"][1] == pytest.approx(0.127)
    assert panel["position"][1] + panel["dims"][1] / 2 == pytest.approx(-0.1524)
    for name in ("microwave_top",):
        assert next(b for b in boxes if b["name"] == name)["dims"][2] == pytest.approx(0.0254)
    for x in [-0.02, 0.1, 0.30]:
        assert not any(inside(b, [x, 0, 0.1]) for b in boxes)


@pytest.mark.parametrize("panel,border,left", [(0, 0.0254, 0.02), (0.127, -1, 0.02), (float("nan"), 0.0254, 0.02)])
def test_invalid_fixed_dimensions_refused(panel, border, left):
    with pytest.raises(world.InvalidGeometry):
        world.fixed_outer_bounds((0.3048, 0.1905), panel, border, left)


@pytest.mark.parametrize("distance,passes", [(0.009, True), (0.005, True), (0.004, False), (-0.001, False)])
def test_five_mm_grab_clearance_threshold(distance, passes):
    checker = SimpleNamespace(clearance=Mock(return_value=(distance, "forearm vs door")))

    def run():
        world.check_local_path(checker, [[0] * 7], lambda qs, step: qs, 0.5, 0.005, Mock(), "insertion")

    if passes:
        run()
    else:
        with pytest.raises(world.InvalidGeometry):
            run()


def test_open_door_right_surface_is_cavity_left_edge(oven, door):
    box = world.door_box(oven, door, 90, 0.08, 10, (0.3048, 0.1905), [-0.2794, 0.1724, -0.0254, 0.2159], 0.0286)
    yaw = np.radians(box["rpy_deg"][2])
    normal = np.array([np.cos(yaw), np.sin(yaw), 0])
    along = np.array([-np.sin(yaw), np.cos(yaw), 0])
    half_across = abs(normal @ oven.w) * box["dims"][0] / 2 + abs(along @ oven.w) * box["dims"][1] / 2
    minimum_a = float((np.array(box["position"]) - oven.b0) @ oven.w) - half_across
    assert minimum_a == pytest.approx(0.3048 / 2)
    assert not inside(box, [-0.2, 0.15, 0.1])  # immediately inside the cavity's left edge


def test_precise_clearance_report_preserves_failed_sample():
    report = Mock()
    checker = SimpleNamespace(clearance=Mock(return_value=(0.004876, "forearm vs door")))
    with pytest.raises(world.InvalidGeometry, match="4.876 mm below 5.000 mm"):
        world.check_local_path(checker, [[0.1] * 7], lambda qs, step: qs, 0.5, 0.005, Mock(), "insertion", report)
    diagnostic = report.call_args.args[0]
    assert diagnostic["distance_mm"] == pytest.approx(4.876)
    assert diagnostic["joints"] == [0.1] * 7
    assert diagnostic["required_mm"] == 5


def test_successful_path_reports_minimum_clearance():
    report = Mock()
    checker = SimpleNamespace(clearance=Mock(side_effect=[(0.012, "arm"), (0.009, "finger")]))
    world.check_local_path(checker, [[0] * 7], lambda qs, step: qs, 0.5, 0.005, Mock(), "lift", report)
    assert report.call_args.args[0]["distance_mm"] == pytest.approx(9)
