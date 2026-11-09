"""home_arm's tests: this package and capability_engine from the checkout. The URDF is flatbot's,
rendered from its mount.yaml as the driver renders it (the test image's /fixtures/urdf); a test that
needs it skips where it is not there (HOME_ARM_TEST_URDF names another)."""

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[2] / "capability_engine")]

URDF = Path(os.environ.get("HOME_ARM_TEST_URDF", "/fixtures/urdf/flatbot.urdf"))

# flatbot's planner spheres for the links the tests use (a subset of its robot.yaml) and buffer.
ROBOT = """robot_cfg:
  kinematics:
    base_link: arm_mount
    collision_sphere_buffer: 0.005
    collision_spheres:
      arm_base_link:
      - {center: [0.0, 0.0, 0.05], radius: 0.06}
      arm_half_arm_2_link:
      - {center: [0.0, 0.0, -0.05], radius: 0.05}
      - {center: [0.0, 0.0, -0.15], radius: 0.05}
      arm_forearm_link:
      - {center: [0.0, 0.0, 0.0], radius: 0.06}
      - {center: [0.0, -0.1, 0.0], radius: 0.05}
      arm_bracelet_link:
      - {center: [0.0, 0.0, -0.045], radius: 0.045}
      arm_robotiq_85_base_link:
      - {center: [0.0, 0.0, 0.04], radius: 0.05}
      - {center: [0.0, 0.0, 0.10], radius: 0.035}
"""


@pytest.fixture
def urdf_text():
    if not URDF.is_file():
        pytest.skip(f"no URDF at {URDF} (the test image's /fixtures/urdf, or HOME_ARM_TEST_URDF)")
    return URDF.read_text()


def write_model(tmp, boxes, padding=None, robot=ROBOT):
    """A planner config in `tmp` with one layer of `boxes` [(name, centre, dims)]: its path."""
    lines = ["layer: test", "obstacles:"]
    lines += [f"  - {{name: {n}, position: {list(c)}, dims: {list(d)}}}" for n, c, d in boxes]
    (tmp / "layer_test.yaml").write_text("\n".join(lines) + "\n")
    (tmp / "robot.yaml").write_text(robot)
    pad = "" if padding is None else f"world_padding: {padding}\n"
    (tmp / "planner.yaml").write_text("robot: robot.yaml\nlayers: [layer_test.yaml]\n" + pad)
    return tmp / "planner.yaml"
