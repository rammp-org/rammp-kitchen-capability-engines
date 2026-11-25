"""The raised wrist re-look's scene obstacle must reach the planner as plain YAML (mobile v5.3, raw search 005)."""

import numpy as np
import yaml

from fridge import arrival


def test_relook_obstacle_serializes_with_numpy_plane(monkeypatch):
    monkeypatch.setattr(arrival, "depth_points", lambda frame: np.zeros((1, 3)))
    monkeypatch.setattr(arrival, "front_plane", lambda points: (np.array([0.96, -0.28]), np.float64(0.71)))
    (obstacle,) = arrival.relook_obstacle(frame={})
    text = yaml.safe_dump(obstacle)  # the planner's world writer refused NumPy scalars (RepresenterError)
    assert yaml.safe_load(text)["position"] == obstacle["position"]
    assert not any(isinstance(x, np.generic) for x in obstacle["position"] + obstacle["rpy_deg"])
