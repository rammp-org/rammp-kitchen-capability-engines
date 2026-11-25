"""Stationary RGB-D checks for a raw arrival before handle detection."""

import math

import numpy as np
from door_common.common import Refused
from scipy.spatial.transform import Rotation

from .constants import HAND_BOX_TOOL_M
from .door_sweep import DoorSweep

CLOSE_M = 0.34
LOOK_CLEARANCE_M = 0.43
MAX_REVERSE_M = 0.29
# Includes the handle protrusion, fitted-door uncertainty and the raw route margin.
CLOSE_ARM_M = 0.16
LOOK_ARM_CLEARANCE_M = 0.20
REPARK = "With the route clear, back straight away from the fridge until your footrest is about 45 cm away, then retry."


def depth_points(frame):
    """Unproject a subsample of the aligned scene image into arm_mount."""
    transform = frame["base_T_cam"]
    if transform is None or not np.isfinite(transform).all():
        raise Refused(f"Scene depth has no valid arm transform. {REPARK}")
    depth = np.asarray(frame["depth_mm"])[::4, ::4] / 1000
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0.10) & (depth < 2.0))
    rays = np.linalg.solve(frame["K"], np.array([4 * cols, 4 * rows, np.ones_like(rows)]))
    return (transform[:3, :3] @ (rays * depth[rows, cols])).T + transform[:3, 3]


def front_plane(points, model=None):
    """Return the inward XY normal and face intercept of a supported vertical surface."""
    if model is not None:
        inward = -np.asarray(model.df.n[:2])
        slab = model.slab(0)[0]
        return inward, float(np.dot(inward, slab["position"][:2]) - slab["dims"][0] / 2)

    # Look at the interior height of the fixture. Reject narrow returns such
    # as the handle or an isolated foreground object before accepting a plane.
    view = points[
        (points[:, 0] > 0.30)
        & (points[:, 0] < 1.50)
        & (np.abs(points[:, 1]) < 0.80)
        & (points[:, 2] > 0.42)
        & (points[:, 2] < 0.76)
    ]
    best = None
    edges = np.arange(0.43, 1.81, 0.014)
    for angle in np.arange(-40, 23, 2):
        yaw = math.radians(angle)
        normal = np.array([math.cos(yaw), math.sin(yaw)])
        offsets = view[:, :2] @ normal
        counts, _ = np.histogram(offsets, bins=edges)
        for index in np.argpartition(counts, -3)[-3:]:
            hits = np.abs(offsets - (edges[index] + 0.007)) < 0.014
            patch = view[hits]
            if len(patch) < 35 or np.ptp(patch[:, 2]) < 0.12:
                continue
            across = patch[:, :2] @ np.array([-normal[1], normal[0]])
            if np.ptp(across) < 0.12:
                continue
            intercept = float(np.median(offsets[hits]))
            score = len(patch) / intercept
            if best is None or score > best[0]:
                best = score, normal, intercept
    if best is None:
        return None
    return best[1:]


def _inside(points, centre, axes, half_sizes):
    return (np.abs((points - centre) @ axes.T) <= half_sizes).all(axis=1)


def arm_envelopes(fk, joints):
    """Enclose each parked link along its own axis, including both endpoints."""
    points = np.vstack((np.zeros(3), fk.joint_points(joints)))
    tool, quat = fk.pose(joints)
    rotation = Rotation.from_quat(quat)
    # The measured hand envelope includes the bracelet and open fingers.
    # End the last link inside it instead of extending a thick box to the tool.
    points[-1] = tool + rotation.apply([0, 0, -0.17])
    boxes = []
    for index, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True)):
        dx, dy, dz = b - a
        length = np.linalg.norm(b - a)
        boxes.append(
            {
                "name": f"arm link {index}",
                "position": (a + b) / 2,
                "dims": [length + 0.16, 0.16, 0.16],
                "rpy_deg": [0, math.degrees(math.atan2(-dz, math.hypot(dx, dy))), math.degrees(math.atan2(dy, dx))],
            }
        )
    limits = np.asarray(HAND_BOX_TOOL_M)
    boxes.append(
        {
            "name": "parked gripper",
            "position": tool + rotation.apply(limits.mean(axis=1)),
            "dims": np.ptp(limits, axis=1),
            "rpy_deg": rotation.as_euler("xyz", degrees=True),
        }
    )
    return boxes


def reverse_distance(frame, chair, arm_boxes, padding, model=None):
    """Allow only straight separation from the observed front and check visible obstacles."""
    if not chair or not arm_boxes:
        raise Refused(f"Arrival recovery needs the verified chair, occupant and arm geometry. {REPARK}")
    points = depth_points(frame)
    if len(points) < 35:
        raise Refused(f"Scene depth is incomplete. {REPARK}")
    half_sizes, centres, axes = DoorSweep._boxes([*chair, *arm_boxes])
    scene = points
    for half, centre, axis in zip(half_sizes, centres, axes, strict=True):
        scene = scene[~_inside(scene, centre, axis, half + padding)]
    plane = front_plane(scene, model)
    if plane is None:
        return 0.0
    normal, front = plane
    leading = centres[:, :2] @ normal + (np.abs(axes[:, :, :2] @ normal) * half_sizes).sum(axis=1)
    clearance = front - max(leading[: len(chair)])
    arm_clearance = front - max(leading[len(chair) :])
    if clearance >= CLOSE_M and arm_clearance >= CLOSE_ARM_M:
        return 0.0
    chair_back = LOOK_CLEARANCE_M - clearance if clearance < CLOSE_M else 0.0
    arm_back = LOOK_ARM_CLEARANCE_M - arm_clearance if arm_clearance < CLOSE_ARM_M else 0.0
    distance = max(chair_back, arm_back) / normal[0]
    if clearance <= padding or distance > MAX_REVERSE_M:
        raise Refused(f"The chair has insufficient clearance for automatic backing. {REPARK}")

    # Reverse carries the arm and occupant as a rigid unit, so their relative
    # separation stays fixed. Check the newly occupied volume against scene
    # returns. Keep the initial arm envelope at full size for the self mask,
    # then check the actual link path at each outward step.
    scene = scene[scene[:, 2] > -0.46]
    for half, centre, axis in zip(half_sizes, centres, axes, strict=True):
        for x in np.linspace(0, distance + 0.01, math.ceil(distance / 0.01) + 2):
            # Require spatial support so an isolated stereo return on a
            # masked robot edge does not represent a stationary obstruction.
            if np.count_nonzero(_inside(scene, centre - [x, 0, 0], axis, half + padding)) >= 2:
                raise Refused(f"The scene depth shows an obstacle in the reverse path. {REPARK}")
    return distance


def relook_obstacle(frame):
    """Conservatively block arm motion through the observed fixture front."""
    plane = front_plane(depth_points(frame))
    if plane is None:
        raise Refused(f"Scene depth could not place the fridge for a checked wrist look. {REPARK}")
    normal, front = plane
    return [
        {
            "name": "scene fridge front",
            "position": [*((front + 0.45) * normal), 0.25],
            "dims": [1.0, 3.0, 1.8],
            "rpy_deg": [0, 0, math.degrees(math.atan2(normal[1], normal[0]))],
        }
    ]
