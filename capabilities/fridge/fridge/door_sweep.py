"""Test the moving slab against the chair and occupant in arm_mount."""

from pathlib import Path

import numpy as np
import yaml
from door_common.common import Refused
from scipy.spatial.transform import Rotation

from . import constants as F

CHAIR_LAYERS = {"person", "seat", "platform"}
SAMPLE_DEG = 2.0
LAYER_PADDING_M = 0.020


class SweepBlocked(Refused):
    def __init__(self, obstacle, sweep):
        shift = sweep.clear_shift()
        hint = "Increase clearance around the door and retry."
        if shift is not None:
            x, y = shift[:2]
            if x:
                hint = f"Back the chair ~{round(x * 100)} cm away from the fridge and retry."
            else:
                direction = "right" if y > 0 else "left"
                hint = f"Shift the chair ~{round(abs(y) * 100)} cm to its {direction} and retry."
        super().__init__(f"Door sweep blocked by {obstacle}. {hint}")
        self.payload = {
            "executed": False,
            "reason": "door_sweep_blocked",
            "obstacle": obstacle,
            "operator_hint": hint,
            "fridge_shift_m": None if shift is None else shift[:2].tolist(),
        }


def deployment_boxes(config_path):
    """Read the planner’s unpadded occupant and chair layers, including a separate platform if configured."""
    path = Path(config_path)
    config = yaml.safe_load(path.read_text())
    boxes = {}
    for filename in config["layers"]:
        layer = yaml.safe_load((path.parent / filename).read_text())
        name = layer["layer"]
        if name in CHAIR_LAYERS:
            boxes.setdefault(name, []).extend(dict(box, name=f"{name}/{box['name']}") for box in layer["obstacles"])
    if any(not boxes.get(name) for name in ("person", "seat")) or not all(boxes.values()):
        raise ValueError("planner config must supply nonempty person and seat layers and any configured platform")
    padding = float(config.get("planner", {}).get("world_padding", LAYER_PADDING_M))
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("invalid planner world_padding")
    return [box for group in boxes.values() for box in group], padding


class DoorSweep:
    """Separating-axis test for padded oriented boxes along a door arc.

    Sample both endpoints. The nearest sample is at most half a step away,
    so the radius of the outermost slab corner bounds all between-sample travel.
    """

    def __init__(self, model, boxes, start, end, padding=LAYER_PADDING_M, *, translation=None, push_grid=False):
        self.names = [box["name"] for box in boxes]
        if not boxes:
            return
        count = max(1, int(np.ceil(abs(end - start) / SAMPLE_DEG)))
        angles = np.linspace(start, end, count + 1)
        half_step = abs(end - start) / (2 * count)
        if push_grid:
            # Fixed samples let every candidate use the same conservative prefix.
            angles = np.r_[start, np.arange(np.floor(start) + 1, end + 1)]
            half_step = 0.5
        self.angles = angles
        slab = [model.slab(deg)[0] for deg in angles]
        dims, centres, axes = self._boxes(slab + model.support())
        count = len(slab)
        radius = np.max(np.linalg.norm(centres[:count, :2] - model.df.hinge[:2], axis=1))
        radius += np.linalg.norm(dims[0, :2])
        # If the closed slab is fixed and the pivot is off by e, the
        # rotated slab is off by (I - R)e. Bound that displacement over
        # each sample's interval, with a floor for closed-pose translation.
        angle_bound = np.radians(np.minimum(180, np.abs(angles) + half_step))
        hinge_pad = F.DOOR_SWEEP_HINGE_MARGIN_M * np.maximum(1, 2 * np.sin(angle_bound / 2))
        dims[:count, :2] += hinge_pad[:, None] + radius * np.radians(half_step) + 0.00002
        dims[count:] += padding  # the stationary support is padded in the arm's collision world too
        if translation is not None:
            shift = np.asarray(translation) / 2
            centres += shift
            dims += np.abs(axes @ shift)
        chair_dims, chair_centres, chair_axes = self._boxes(boxes)
        chair_dims += padding

        # For each slab and chair box pair, test their six face normals and
        # the nine cross products of their edges. Zero cross products impose
        # no constraint. Keeping the axes unnormalized avoids dividing by zero.
        a, b = np.broadcast_arrays(axes[:, None], chair_axes[None, :])
        cross = np.cross(a[..., :, None, :], b[..., None, :, :])
        self.axes = np.concatenate((a, b, cross.reshape(*a.shape[:2], 9, 3)), axis=2)
        self.radii = (np.abs(self.axes @ a.swapaxes(-1, -2)) * dims[:, None, None]).sum(axis=-1) + (
            np.abs(self.axes @ b.swapaxes(-1, -2)) * chair_dims[None, :, None]
        ).sum(axis=-1)
        delta = centres[:, None] - chair_centres
        self.gaps = (self.axes @ delta[..., None]).squeeze(-1)

    @staticmethod
    def _boxes(boxes):
        dims = np.array([box["dims"] for box in boxes], float) / 2
        centres = np.array([box["position"] for box in boxes], float)
        axes = Rotation.from_euler("xyz", [box.get("rpy_deg", (0, 0, 0)) for box in boxes], degrees=True)
        if not np.all(np.isfinite(centres)) or not np.all(np.isfinite(dims) & (dims > 0)):
            raise ValueError("invalid door or chair box geometry")
        return dims, centres, axes.as_matrix().swapaxes(-1, -2)

    def _hits(self, shift):
        gaps = self.gaps if shift is None else self.gaps + self.axes @ shift
        hits = (np.abs(gaps) <= self.radii + 1e-10).all(axis=-1)
        count = len(self.angles)
        return hits[:count] | hits[count:].any(axis=0)

    def obstacle(self, shift=None):
        if not self.names:
            return None
        hit = self._hits(shift).any(axis=0)
        return next((name for name, intersects in zip(self.names, hit, strict=True) if intersects), None)

    def clear_push_end(self, minimum, shift=None):
        """Largest whole-degree endpoint with an entirely clear sampled prefix."""
        if not self.names:
            return float(F.PUSH_OPEN_TO_DEG) if minimum <= F.PUSH_OPEN_TO_DEG else None
        blocked = np.flatnonzero(self._hits(shift).any(axis=1))
        index = int(blocked[0]) - 1 if blocked.size else len(self.angles) - 1
        if index >= 0 and self.angles[index] >= minimum:
            return float(self.angles[index])
        return None

    def clear_shift(self):
        """Smallest tested chair move along a single axis that clears this sweep at the new dock."""
        for distance in np.arange(0.02, 1.01, 0.02):
            for direction in ((1, 0, 0), (0, 1, 0), (0, -1, 0)):
                shift = distance * np.array(direction)
                if self.obstacle(shift) is None:
                    return shift
        return None


def check_open(model, world, angles, after):
    """Return pull angles with a clear sweep and at least one clear push route."""
    boxes, padding = world.door_sweep_boxes()
    safe = []
    for angle in angles:
        if DoorSweep(model, boxes, 0.0, angle, padding).obstacle() is not None:
            continue
        if after is None or after.safe_routes(model, world, angle, in_place=angle < angles[0]):
            safe.append(angle)
    if not safe:
        end = min(angles) if after is None else push_minimum(min(angles))
        sweep = DoorSweep(model, boxes, 0.0, min(end, F.PUSH_OPEN_TO_DEG), padding)
        raise SweepBlocked(sweep.obstacle() or "the chair keep-out", sweep)
    return tuple(safe)


def push_minimum(start):
    """The demo opening floor and the minimum travel beyond the pull handoff."""
    return max(F.PUSH_OPEN_MIN_FINAL_DEG, float(np.ceil(start + F.PUSH_OPEN_MIN_PUSH_DEG)))
