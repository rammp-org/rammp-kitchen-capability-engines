"""Open microwave shell and door, in arm_mount. Geometry helpers have no ROS dependency.

The cavity stays empty. The saved closed-door handle/hinge estimate supplies the
finite door width and height; live cavity/body detection checks its registration.
"""

import json
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml


class InvalidGeometry(ValueError):
    pass


def _vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise InvalidGeometry(f"{name} must contain three finite coordinates")
    return a


def shell_boxes(oven, outer, opening, depth, recess, wall):
    """Top and two sides only; right side includes the control panel.

    outer is (left/right horizontal bounds, bottom/top height bounds) in the
    cavity's A/S/Z frame. A increases toward the viewer's left; S into the oven.
    """
    a0, a1, z0, z1 = np.asarray(outer, dtype=float)
    width, height = opening
    if not np.isfinite([a0, a1, z0, z1, width, height, depth, recess, wall]).all():
        raise InvalidGeometry("non-finite microwave dimensions")
    if min(width, height, depth, recess, wall) <= 0 or not (a0 < -width / 2 < width / 2 < a1 and z0 < 0 < height < z1):
        raise InvalidGeometry("outer microwave rectangle must surround the cavity")
    surrounds = [-width / 2 - a0, a1 - width / 2, -z0, z1 - height]
    if min(surrounds) < 0.005:
        raise InvalidGeometry("exterior fit leaves a wall thinner than 5 mm; incomplete body segmentation")
    return [
        oven.box("microwave_control_panel", (a0, -width / 2), (-recess, depth + wall), (z0, z1)),
        oven.box("microwave_left_wall", (width / 2, a1), (-recess, depth + wall), (z0, z1)),
        oven.box("microwave_top", (-width / 2, width / 2), (-recess, depth + wall), (height, z1)),
    ]


def door_box(oven, door, angle_deg, position_tol, yaw_tol_deg, opening, outer, recess):
    """Keep saved door dimensions, registering its inner edge to the live cavity left/front edge.

    A left-hinged door opens outward at the configured angle (90 degrees by
    default). The angle is assumed, not newly observed. Saved coordinates
    supply dimensions only; placement and heading come from the live cavity.
    """
    opening_width, opening_height = opening
    n = _vector(door["closed_normal"], "closed_normal")
    if np.linalg.norm(n[:2]) < 1e-6:
        raise InvalidGeometry("invalid closed door normal")
    n = np.r_[n[:2] / np.linalg.norm(n[:2]), 0.0]
    hinge = _vector(door["hinge"], "hinge")
    grasp = _vector(door["closed_grasp_pos"], "closed_grasp_pos")
    face_off = float(door.get("face_behind_grasp", 0.067))
    thickness = float(door.get("door_t", 0.04))
    past = float(door.get("door_past_handle", 0.06))
    heights = np.asarray(door["door_z"], dtype=float)
    if heights.shape != (2,) or not np.isfinite(heights).all() or heights[1] <= heights[0]:
        raise InvalidGeometry("door_z must contain measured bottom and top heights")
    if not np.isfinite([face_off, thickness, past, angle_deg, position_tol, yaw_tol_deg]).all():
        raise InvalidGeometry("invalid door settings")
    if not (0 < angle_deg <= 120 and 0 < thickness < 0.15 and 0 <= past < 0.2 and 0 < face_off < 0.3):
        raise InvalidGeometry("invalid door dimensions or opening angle")
    if position_tol <= 0 or not 0 < yaw_tol_deg < 45:
        raise InvalidGeometry("invalid saved-door alignment tolerances")
    # Use the saved relative span only to size the door. Its absolute hinge,
    # normal and heights are not registration constraints on the live cavity.
    face = grasp - n * face_off
    saved_angle = float(door["door_open_deg"])
    if not np.isfinite(saved_angle) or abs(saved_angle - angle_deg) > 10:
        raise InvalidGeometry("saved door opening does not match configured angle; finish opening/re-detect")
    span = hinge - face
    along = span - n * float(span @ n)
    width = float(np.linalg.norm(along[:2])) + past
    if not 0.2 < width < 0.7:
        raise InvalidGeometry("implausible handle-to-hinge door width")
    # One live frame for the body, cavity and door: the saved hinge locates
    # no collision box. Its handle-to-hinge span supplies the door width only.
    a0, a1, z0, z1 = np.asarray(outer, dtype=float)
    if not np.isfinite([a0, a1, z0, z1, recess]).all() or recess <= 0:
        raise InvalidGeometry("invalid live exterior bounds")
    if not a0 < -opening_width / 2 < opening_width / 2 < a1 or not z0 < 0 < opening_height < z1:
        raise InvalidGeometry("live exterior must surround cavity before registering door")
    # At 90 degrees, the slab's rightmost surface is exactly the cavity's
    # left edge. Its whole thickness extends left, away from the opening.
    hinge = oven.b0 + opening_width / 2 * oven.w - recess * oven.n
    n, toward_hinge = -oven.n, oven.w
    # A left hinge opens outward with negative yaw about world up.
    a = -np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    centre = hinge + rot @ (-toward_hinge * width / 2 + n * thickness / 2)
    centre[2] = float(oven.b0[2] + (z0 + z1) / 2)
    normal = rot @ n
    return dict(
        name="microwave_open_door",
        position=centre.tolist(),
        dims=[thickness, width, float(heights[1] - heights[0]) + 0.04],
        rpy_deg=[0.0, 0.0, float(np.degrees(np.arctan2(normal[1], normal[0])))],
    )


def read_door(path):
    """Read saved dimensions; timestamps do not determine geometric validity."""
    path = Path(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise InvalidGeometry("door.json must contain an object")
    return data


@contextmanager
def loaded_world(arm, path, restore, load, log):
    """Restore even if loading/acknowledgement fails or the action is cancelled."""
    try:
        message = load(arm, str(path), log)
        yield message
    finally:
        ok, message = arm.set_world(restore, even_if_cancelled=True)
        if not ok:
            raise InvalidGeometry(f"STOPPED: could not restore cuRobo world {restore}: {message}")


def write_scene(directory, frame, layers, boxes, metadata):
    directory = Path(directory)
    if not directory.is_absolute() or not layers or not all(Path(p).is_absolute() for p in layers):
        raise InvalidGeometry("planner world directory and floor layers must be absolute paths")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"microwave_grab_{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(dict(base_frame=frame, layers=list(layers), obstacles=boxes), sort_keys=False))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return path


def fixed_outer_bounds(opening, panel_width, border, left_wall):
    """A increases left: the control panel starts at the cavity's right edge."""
    width, height = opening
    values = [width, height, panel_width, border, left_wall]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise InvalidGeometry("fixed microwave dimensions must be positive and finite")
    return [-width / 2 - panel_width, width / 2 + left_wall, -border, height + border]


def check_local_path(checker, samples, dense, step_deg, clearance_m, checkpoint, phase, report=None):
    """Reject local IK routes that touch the same shell/door boxes used by cuRobo.

    Check both finger extremes; the held container itself is not represented.
    """
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise InvalidGeometry("clear_min must be finite and nonnegative")
    worst = None
    for index, q in enumerate(dense(samples, step_deg)):
        checkpoint()
        for closed in (False, True):
            distance, pair = checker.clearance(q, closed=closed)
            record = dict(
                phase=phase,
                sample=index,
                joints=np.asarray(q).tolist(),
                gripper_closed=closed,
                distance_mm=float(distance) * 1000,
                required_mm=clearance_m * 1000,
                pair=pair,
            )
            if worst is None or distance * 1000 < worst["distance_mm"]:
                worst = record
            if not np.isfinite(distance) or distance < clearance_m:
                if report is not None:
                    report(record)
                raise InvalidGeometry(
                    f"{phase}: {pair} clearance {distance * 1000:.3f} mm below {clearance_m * 1000:.3f} mm"
                )
    if report is not None and worst is not None:
        report(worst)


def planner_layers(config_path):
    """Expected persistent layers/counts from the same active config cuRobo loads."""
    path = Path(config_path)
    config = yaml.safe_load(path.read_text())
    layers = config.get("layers") if isinstance(config, dict) else None
    if not isinstance(layers, list) or not layers:
        raise InvalidGeometry("active planner config has no persistent collision layers")
    expected = {}
    for ref in layers:
        if not isinstance(ref, str) or not ref:
            raise InvalidGeometry("invalid planner collision layer path")
        layer_path = Path(ref)
        if not layer_path.is_absolute():
            layer_path = path.parent / layer_path
        layer = yaml.safe_load(layer_path.read_text())
        name, obstacles = layer.get("layer"), layer.get("obstacles")
        if (
            not isinstance(name, str)
            or not name
            or name in expected
            or not isinstance(obstacles, list)
            or not obstacles
        ):
            raise InvalidGeometry("invalid or empty persistent collision layer")
        if layer.get("meta", {}).get("frame") != "arm_mount":
            raise InvalidGeometry("persistent collision layers must use arm_mount")
        expected[name] = len(obstacles)
    return expected
