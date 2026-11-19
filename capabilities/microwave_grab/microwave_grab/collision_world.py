"""Open microwave shell and door, in arm_mount. Geometry helpers have no ROS dependency.

The cavity stays empty. The saved closed-door handle/hinge estimate supplies the
finite door width and height; live cavity/body detection checks its registration.
"""

import json
import time
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
    """Four surrounding walls and a back; right wall includes the control panel.

    outer is (left/right horizontal bounds, bottom/top height bounds) in the
    cavity's A/S/Z frame. A increases toward the viewer's left; S into the oven.
    """
    a0, a1, z0, z1 = np.asarray(outer, dtype=float)
    width, height = opening
    if not np.isfinite([a0, a1, z0, z1, width, height, depth, recess, wall]).all():
        raise InvalidGeometry("non-finite microwave dimensions")
    if min(width, height, depth, recess, wall) <= 0 or not (a0 < -width / 2 < width / 2 < a1 and z0 < 0 < height < z1):
        raise InvalidGeometry("outer microwave rectangle must surround the cavity")
    return [
        oven.box("microwave_control_panel", (a0, -width / 2), (-recess, depth + wall), (z0, z1)),
        oven.box("microwave_left_wall", (width / 2, a1), (-recess, depth + wall), (z0, z1)),
        oven.box("microwave_bottom", (-width / 2, width / 2), (-recess, depth + wall), (z0, 0)),
        oven.box("microwave_top", (-width / 2, width / 2), (-recess, depth + wall), (height, z1)),
        oven.box("microwave_back", (-width / 2, width / 2), (depth, depth + wall), (0, height)),
    ]


def door_box(oven, door, angle_deg, position_tol, yaw_tol_deg, opening):
    """Use the opener's detected hinge edge and closed grasp, as door_model_boxes does.

    A left-hinged door opens outward at the configured angle (90 degrees by
    default). The angle is assumed, not newly observed. Stored geometry must
    still agree with the live cavity plane and its left edge.
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
    yaw_error = np.degrees(np.arccos(np.clip(n @ -oven.n, -1, 1)))
    h_local = oven.to_oven([hinge])[0]
    # The closed door lies in front of the recessed cavity frame.
    face = grasp - n * face_off
    plane_error = abs(float((face - hinge) @ n))
    if yaw_error > yaw_tol_deg or plane_error > position_tol or abs(h_local[1]) > position_tol:
        raise InvalidGeometry("saved door no longer agrees with the live microwave front; reopen/re-detect the door")
    if abs(h_local[0] - opening_width / 2) > position_tol:
        raise InvalidGeometry("saved hinge does not match the live cavity left edge; reopen/re-detect the door")
    live_mid_z = oven.b0[2] + opening_height / 2
    if abs(float(heights.mean()) - live_mid_z) > position_tol:
        raise InvalidGeometry("saved door height no longer agrees with the live cavity")
    saved_angle = float(door["door_open_deg"])
    if not np.isfinite(saved_angle) or abs(saved_angle - angle_deg) > 10:
        raise InvalidGeometry("saved door opening does not match configured angle; finish opening/re-detect")
    span = hinge - face
    along = span - n * float(span @ n)
    width = float(np.linalg.norm(along[:2])) + past
    if not 0.2 < width < 0.7:
        raise InvalidGeometry("implausible handle-to-hinge door width")
    toward_hinge = along / np.linalg.norm(along[:2])
    if toward_hinge @ oven.w < 0.9:
        raise InvalidGeometry("saved geometry is not a left-hinged microwave")
    # A left hinge opens outward with negative yaw about world up.
    a = -np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    centre = hinge + rot @ (-toward_hinge * width / 2 - n * thickness / 2)
    centre[2] = float(heights.mean())
    normal = rot @ n
    return dict(
        name="microwave_open_door",
        position=centre.tolist(),
        dims=[thickness, width, float(heights[1] - heights[0]) + 0.04],
        rpy_deg=[0.0, 0.0, float(np.degrees(np.arctan2(normal[1], normal[0])))],
    )


def read_door(path, max_age_s):
    path = Path(path)
    if not np.isfinite(max_age_s) or max_age_s <= 0:
        raise InvalidGeometry("door_max_age_s must be positive")
    age = time.time() - path.stat().st_mtime
    if not -5 <= age <= max_age_s:
        raise InvalidGeometry(f"saved door is {age:.0f} s old; reopen/re-detect before grabbing")
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


def outer_bounds(oven, shot, recess):
    """Robust outer bounds from body-mask depths at the front, excluding the open door."""
    depth = np.asarray(shot["depth_m"], dtype=float)
    mask = np.asarray(shot["mask"], dtype=bool)
    if mask.shape != depth.shape:
        raise InvalidGeometry("body mask and depth shapes differ")
    valid = mask & np.isfinite(depth) & (depth > 0.15) & (depth < 2.5)
    v, u = np.nonzero(valid)
    d = depth[v, u]
    K, T = np.asarray(shot["K"], dtype=float), np.asarray(shot["world_T_cam"], dtype=float)
    if K.shape != (3, 3) or T.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(T).all():
        raise InvalidGeometry("invalid body camera calibration or transform")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise InvalidGeometry("invalid body camera focal length")
    points = np.column_stack(((u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d))
    # Explicit small dot products avoid BLAS dispatch for three-coordinate points.
    relative = np.einsum("ij,kj->ik", points, T[:3, :3]) + T[:3, 3] - oven.b0
    local = np.column_stack(
        (np.einsum("ij,j->i", relative, oven.w), np.einsum("ij,j->i", relative, oven.n), relative[:, 2])
    )
    # Restrict to the exterior front plane; interior and door depths cannot
    # enlarge the estimated control panel and front surround.
    local = local[
        (np.abs(local[:, 1] + recess) < 0.025)
        & (np.abs(local[:, 0]) < 0.5)
        & (local[:, 2] > -0.15)
        & (local[:, 2] < 0.5)
    ]
    if len(local) < 200:
        raise InvalidGeometry("fewer than 200 exterior-front depth points; cannot model control panel")
    lo, hi = np.percentile(local[:, [0, 2]], [1, 99], axis=0)
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]


def check_local_path(checker, samples, dense, step_deg, clearance_m, checkpoint, phase):
    """Reject local IK routes that touch the same shell/door boxes used by cuRobo.

    Check both finger extremes; the held container itself is not represented.
    """
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise InvalidGeometry("clear_min must be finite and nonnegative")
    for q in dense(samples, step_deg):
        checkpoint()
        for closed in (False, True):
            distance, pair = checker.clearance(q, closed=closed)
            if not np.isfinite(distance) or distance < clearance_m:
                raise InvalidGeometry(f"{phase}: {pair} clearance {distance:.3f} m below {clearance_m:.3f} m")
