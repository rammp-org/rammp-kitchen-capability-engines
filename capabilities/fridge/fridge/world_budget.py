"""Fit optional surroundings into the planner's box cache without clipping geometry."""

import numpy as np
from door_common.common import Refused, say
from scipy.spatial.transform import Rotation

DEFAULT_BOX_BUDGET = 60
# Gen3's 0.902 m nominal reach, plus the hand, collision spheres and margin.
# Use the whole reachable volume because free-space plans may leave the TCP corridor.
ARM_SWEEP_RADIUS_M = 1.25
MERGE_GAP_M = 0.04
MERGE_EXTRA_VOLUME = 0.10


def _distance(box, padding):
    rotation = Rotation.from_euler("xyz", box["rpy_deg"], degrees=True)
    local_origin = rotation.inv().apply(-np.asarray(box["position"]))
    return float(np.linalg.norm(np.maximum(np.abs(local_origin) - np.asarray(box["dims"]) / 2 - padding, 0)))


def _merge(a, b, padding):
    if not np.allclose(a["rpy_deg"], b["rpy_deg"], rtol=0, atol=1e-10):
        return None
    rotation = Rotation.from_euler("xyz", a["rpy_deg"], degrees=True)
    centers = rotation.inv().apply([a["position"], b["position"]])
    halves = np.asarray([a["dims"], b["dims"]]) / 2
    lo, hi = centers - halves, centers + halves
    overlap = np.minimum(*hi) - np.maximum(*lo)
    if np.count_nonzero(overlap < -1e-8) > 1 or np.min(overlap) < -min(MERGE_GAP_M, 2 * padding):
        return None
    lower, upper = lo.min(axis=0), hi.max(axis=0)
    dims = upper - lower
    union = np.prod(2 * halves, axis=1).sum() - np.prod(np.maximum(overlap, 0))
    if np.prod(dims) > union * (1 + MERGE_EXTRA_VOLUME):
        return None
    return dict(
        name=f"{a['name']}+{b['name']}",
        position=rotation.apply((lower + upper) / 2).tolist(),
        rpy_deg=a["rpy_deg"],
        dims=(dims + 1e-9).tolist(),
    )


def fit_boxes(required, surroundings, *, budget, reserved, padding, label):
    optional = list(surroundings)
    available = budget - reserved - len(required)
    if len(optional) <= available:
        return optional + list(required)
    optional.sort(key=lambda box: -_distance(box, padding))
    changes = []
    while len(optional) > available and optional:
        distance = _distance(optional[0], padding)
        if distance <= ARM_SWEEP_RADIUS_M:
            break
        changes.append(f"dropped {optional.pop(0)['name']} ({distance:.2f} m from arm mount, outside arm sweep)")
    while len(optional) > available:
        for i, a in enumerate(optional):
            for j in range(i + 1, len(optional)):
                merged = _merge(a, optional[j], padding)
                if merged is not None:
                    break
            else:
                continue
            changes.append(f"merged {merged['name']}")
            optional[i] = merged
            del optional[j]
            break
        else:
            raise Refused(
                f"cuRobo box budget for {label}: {reserved + len(required) + len(optional)} boxes still needed "
                f"with collision_cache_obb={budget} ({reserved} layer, {len(required)} required, "
                f"{len(optional)} nearby)."
            )
    say(f"  cuRobo box budget: {', '.join(changes)}")
    return optional + list(required)
