"""Parked arm geometry tests and WS11-X mobile v5.3 search regression."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from door_common.common import Refused

from fridge import constants, docking
from fridge.door_sweep import deployment_boxes
from fridge.kinematics import Chain
from fridge.world import FridgeModel

FIXTURES = Path(__file__).parent / "fixtures" / "parked_route"


@pytest.mark.parametrize(("x", "blocked"), [(0.245, False), (0.12, True)])
def test_sphere_route_distinguishes_near_miss_and_collision(x, blocked):
    model = SimpleNamespace(
        door={"closed_handle": [x, 0, 0]},
        prefix="",
        full=lambda _: [dict(name="door", position=[x, 0, 0], dims=[0.02, 0.10, 0.10])],
    )
    spheres = [[0, -0.12, 0, 0.04]]
    route = [(0, np.pi)]
    if blocked:
        with pytest.raises(Refused, match="parked arm near door"):
            docking.check_sphere_route(model, spheres, route, [0, 0])
    else:
        docking.check_sphere_route(model, spheres, route, [0, 0])


# The archived fits are from X's dock/001, 021, 028 and 031 at 0cbc1cd.
# Rebuild the sweep with the archived robot's URDF and deployment config.
@pytest.mark.parametrize("case", json.loads((FIXTURES / "routes.json").read_text()), ids=lambda case: str(case["id"]))
def test_x_refused_routes_use_spheres_or_legacy_if_missing(case):
    planner = FIXTURES / "planner.yaml"
    urdf = (FIXTURES / "robot.urdf").read_text()
    q = np.radians(constants.START_JOINTS_DEG)
    mount = docking.AXLE_TO_MOUNT[:2]
    model = FridgeModel(case["door"])
    chair, padding = deployment_boxes(planner)
    route = case["route"]
    warnings = []

    fallback = docking.ParkedRouteGuard(
        planner.with_name("missing.yaml"), lambda: urdf, Chain(urdf), mount, False, warnings.append
    )
    for _ in range(2):
        with pytest.raises(Refused, match="arm link"):
            fallback.check(model, q, chair, padding, route)
    assert len(warnings) == 1

    guard = docking.ParkedRouteGuard(planner, lambda: urdf, Chain(urdf), mount, False, warnings.append)
    guard.check(model, q, chair, padding, route)
    assert len(warnings) == 1


def test_parked_arm_cannot_overlap_an_occupant_layer():
    planner = FIXTURES / "planner.yaml"
    urdf = (FIXTURES / "robot.urdf").read_text()
    q = np.radians(constants.START_JOINTS_DEG)
    warnings = []
    guard = docking.ParkedRouteGuard(
        planner, lambda: urdf, Chain(urdf), docking.AXLE_TO_MOUNT[:2], True, warnings.append
    )
    arm_mount_sphere = [0.03, 0.0, 0.055]
    chair = [dict(name="person/test", position=arm_mount_sphere, dims=[0.01, 0.01, 0.01])]
    with pytest.raises(Refused, match="parked arm near person/test"):
        guard.check(None, q, chair, 0.02, ())
    assert warnings == []
