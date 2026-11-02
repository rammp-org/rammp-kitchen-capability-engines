"""The microwave's door planned through door_common, dry, in the sim: a shared change that harms
it shows up here.

CI starts capabilities in mock mode only, so nothing else runs this planning code. The door is a
real recorded door file (10-24).
"""

import json

import numpy as np
import pytest
from door_common import close_door, open_door
from door_common import common as dc
from microwave import open_door as microwave_open

from capability_engine import sim

HANDLE_QUAT = np.array([-0.5, 0.5, 0.5, -0.5])
MICROWAVE_DOOR = {
    "closed_normal": [-0.9998242162892161, 0.01874930723133269, 0.0],
    "closed_grasp_pos": [0.795407417267638, -0.19243639528962211, 0.26055629510592393],
    "closed_handle": [0.8433989796495204, -0.19333636203672608, 0.26055629510592393],
    "door_z": [0.13539826826136991, 0.40185965793508877],
    "hinge": [0.879997694735208, 0.13497556172597747, 0.2593880762812125],
    "hinge_pivot": [0.8983214551036536, 0.15554629533022274, 0.2593880762812125],
    "open_sign": -1.0,
}
KINOVA_HOME = np.radians((0.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0))


def _detection(door):
    return {
        "h_raw": np.array(door["closed_handle"]),
        "orient": HANDLE_QUAT,
        "n": np.array(door["closed_normal"]),
        "hinge": np.array(door["hinge_pivot"]),
        "hinge_seen": np.array(door["hinge"]),
        "door_z": np.array(door["door_z"]),
    }


class _Hand:
    """A resting arm at joints `q`, for a dry run: nothing here can move."""

    def __init__(self, scene, rb, q):
        pos, quat = sim.tool_pose(scene, rb, q)
        self.state = {"position": list(q), "ee_pos": [*pos, *quat], "gripper_pos": 0.0}

    def get_state(self):
        return self.state


@pytest.fixture
def door_sim(tmp_path):
    """The sim on the driver's model, door_common configured for the microwave, every constant
    restored afterwards."""
    saved = dict(vars(dc)), dict(vars(open_door))
    sim.reset_for_tests()
    scene, rb = sim.make_sim("/fixtures/urdf/dojo.urdf")
    microwave_open.configure(tmp_path)
    yield scene, rb
    sim.reset_for_tests()
    vars(dc).update(saved[0])
    vars(open_door).update(saved[1])


def test_microwave_open_plans(door_sim):
    scene, rb = door_sim
    # the push-open is all cuRobo's (asked of the planner; the test image has none): unchecked here
    plan, why = open_door.plan_open(
        scene, rb, _detection(MICROWAVE_DOOR), KINOVA_HOME, 50, 45, microwave_open.PushOpen(None)
    )
    assert why is None
    assert plan["swing"] == 50
    assert plan["after"] == "push-open: not checked (no cuRobo)"


def test_microwave_close_plans(door_sim, tmp_path):
    scene, rb = door_sim
    dc.DOOR_FILE.write_text(json.dumps({**MICROWAVE_DOOR, "door_open_deg": 90.0}))
    hand = _Hand(scene, rb, dc.load_park())
    args = close_door.close_args(False, route_over=True, keep_orientation=True)
    assert close_door.push_close(hand, args)


def test_keep_out_is_off_for_the_microwave_and_refuses_when_enabled(door_sim):
    scene, rb = door_sim
    assert not sim.keep_out_enabled()
    behind = sim.tool_pose(scene, rb, np.radians((-135.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0)))
    assert sim.solve_ik(scene, rb, *behind, KINOVA_HOME)[1] != sim.KEEP_OUT_ERR
    dc.enable_keep_out(0.0)
    assert sim.keep_out_enabled()
    assert sim.solve_ik(scene, rb, *behind, KINOVA_HOME)[1] == sim.KEEP_OUT_ERR


def test_configure_refuses_what_is_not_a_constant_and_what_is_missing(door_sim):
    with pytest.raises(ValueError, match="unknown"):
        open_door.configure(GRIP_EXTT=0.1)
    with pytest.raises(ValueError, match="missing"):
        open_door.configure(GRIP_EXT=None)


def test_a_missing_capability_value_fails_rather_than_borrowing_the_microwaves():
    assert dc.PARK_JOINTS is None
    with pytest.raises(dc.Refused, match="PARK_JOINTS"):
        dc.load_park()
