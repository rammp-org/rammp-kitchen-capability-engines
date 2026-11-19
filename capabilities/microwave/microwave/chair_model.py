"""Load the microwave's local keep-out boxes from the active cuRobo config.

No ROS or arm calls. Only persistent config layers are loaded, not the replaceable
scene world. Missing or unsupported geometry refuses startup rather than silently
using the bundled Flatbot chair. The caller retains door_common's mesh margin.
"""

import math
from pathlib import Path

import yaml


def _yaml(path):
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def _triple(value, label, positive=False):
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{label}: expected three coordinates")
    v = tuple(float(x) for x in value)
    if not all(math.isfinite(x) and (not positive or x > 0) for x in v):
        raise ValueError(f"{label}: invalid coordinates")
    return v


def load_planner_chair(planner_config):
    """Return (padded boxes, layer names) for enable_chair_collision(boxes).

    Uses planner.world_padding (cuRobo default: 0.02 m) after checking the
    layers' certified minimum. Rotated boxes and non-arm_mount layers are refused: door_common's chair
    checker currently accepts axis-aligned boxes only.
    """
    config_path = Path(planner_config).expanduser()
    config = _yaml(config_path)
    paths = config.get("layers")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{config_path}: no persistent collision layers")
    planning = config.get("planner", {})
    pad = float(planning.get("world_padding", 0.02))
    if not math.isfinite(pad) or pad < 0:
        raise ValueError(f"{config_path}: invalid world_padding")
    no_pad = planning.get("no_pad_names", ["pedestal"])
    if not isinstance(no_pad, list) or not all(isinstance(name, str) for name in no_pad):
        raise ValueError(f"{config_path}: invalid no_pad_names")
    boxes, names = [], []
    for layer in paths:
        if not isinstance(layer, str) or not layer:
            raise ValueError(f"{config_path}: layer paths must be nonempty strings")
        path = Path(layer)
        if not path.is_absolute():
            path = config_path.parent / path
        data = _yaml(path)
        name = data.get("layer")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"{path}: missing or duplicate layer name")
        meta = data.get("meta", {})
        if meta.get("frame") != "arm_mount":
            raise ValueError(f"{path}: collision layer must use arm_mount")
        certified = float(meta["padding_assumed_m"])
        if not math.isfinite(certified) or certified < 0 or pad < certified:
            raise ValueError(f"{path}: world_padding below the layer's certified minimum")
        obstacles = data.get("obstacles")
        if not isinstance(obstacles, list) or not obstacles:
            raise ValueError(f"{path}: collision layer has no boxes")
        for obstacle in obstacles:
            label = f"{name}/{obstacle['name']}"
            centre = _triple(obstacle["position"], label)
            dims = _triple(obstacle["dims"], label, positive=True)
            rotation = _triple(obstacle.get("rpy_deg", [0, 0, 0]), label)
            if any(abs(x) > 1e-9 for x in rotation):
                raise ValueError(f"{label}: local chair checker cannot represent rotated boxes")
            if any(existing[0] == label for existing in boxes):
                raise ValueError(f"{label}: duplicate box name")
            padding = 0.0 if obstacle["name"] in no_pad else pad
            boxes.append((label, centre, tuple(d / 2 + padding for d in dims)))
        names.append(name)
    return boxes, names
