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
from capability_engine.arm_messages import Plan

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


def test_chair_boxes_are_the_planners_with_its_padding():
    boxes = {name: (centre, half) for name, centre, half in dc.load_chair_boxes()}
    # 16 platform, 10 seat (11-07: the arm-side armrest is no longer carved), 4 person
    assert len(boxes) == 30
    centre, half = boxes["person/torso"]
    # 11-07: 0.2032 m back, the arm moved 8 in forward on the chair
    assert centre == (-0.31900, -0.35075, 0.70485)
    assert half == pytest.approx([0.2318 / 2 + 0.02, 0.4827 / 2 + 0.02, 0.7749 / 2 + 0.02])


def test_chair_refuses_the_arm_but_not_its_base_and_shoulder():
    sim.reset_for_tests()
    try:
        scene, rb = sim.make_sim("/fixtures/urdf/flatbot.urdf")
        q = KINOVA_HOME
        tool = sim.tool_pose(scene, rb, q)[0]
        half = [0.05, 0.05, 0.05]
        dc.enable_chair_collision([("at the tool", tuple(tool), half)])
        assert "at the tool" in sim.keep_out_message(rb, q)
        sim.reset_for_tests()
        scene, rb = sim.make_sim("/fixtures/urdf/flatbot.urdf")
        dc.enable_chair_collision([("at the base", (0.0, 0.0, 0.05), half)])
        assert sim.keep_out_message(rb, q) is None
    finally:
        sim.reset_for_tests()


class _Planner:
    """plan_to_joints as a fake: every leg's plan is `samples(start, goal)`."""

    def __init__(self, samples=None, fail_leg=None):
        self.samples, self.fail_leg, self.legs = samples, fail_leg, 0

    def plan_to_joints(self, goal, start):
        self.legs += 1
        if self.legs == self.fail_leg:
            return Plan(False, "no path")
        q = np.linspace(start, goal, 5) if self.samples is None else self.samples(start, goal)
        return Plan(True, "ok", q=np.asarray(q))


PATH = [np.full(7, 0.05 * k) for k in range(1, 10)]


@pytest.fixture
def curobo_paths(monkeypatch):
    monkeypatch.setattr(open_door, "CUROBO_PATHS", True)


def test_only_the_microwave_runs_its_paths_through_curobo(door_sim):
    assert open_door.CUROBO_PATHS
    assert open_door._sent_keep_out(np.zeros(7), PATH, None) is None


def test_curobo_paths_are_off_by_default(monkeypatch):
    monkeypatch.setattr(open_door, "CUROBO_PATHS", False)
    planner = _Planner(fail_leg=1)
    assert open_door._check_curobo_path(planner, np.zeros(7), PATH, "the swing") is None


def test_curobo_gets_every_fourth_waypoint_and_the_last():
    knots = open_door.curobo_knots(PATH)
    assert [k[0] for k in knots] == pytest.approx([0.2, 0.4, 0.45])


def test_curobo_passes_every_target_but_the_last_without_stopping():
    sent = {}

    class _Arm:
        def go_to_joint_configs(self, targets, label, **kw):
            sent["targets"] = targets
            return True

    assert dc.through_curobo_knots(_Arm(), PATH[:3], "three")
    assert [t.blend_radius for t in sent["targets"]] == [dc.CUROBO_BLEND_M, dc.CUROBO_BLEND_M, 0.0]


def test_curobo_path_passes_a_direct_plan(curobo_paths):
    assert open_door._check_curobo_path(_Planner(), np.zeros(7), PATH, "the swing") is None


def test_curobo_path_refuses_a_leg_it_cant_plan(curobo_paths):
    why = open_door._check_curobo_path(_Planner(fail_leg=2), np.zeros(7), PATH, "the swing")
    assert "leg 2" in why and "no path" in why


def test_curobo_path_refuses_a_detour(curobo_paths):
    loop = lambda start, goal: [start, start + 1.0, goal]  # noqa: E731
    why = open_door._check_curobo_path(_Planner(loop), np.zeros(7), PATH, "the swing")
    assert "detour" in why


def test_curobo_path_is_not_checked_without_a_planner(curobo_paths):
    assert open_door._check_curobo_path(object(), np.zeros(7), PATH, "the swing") is None


def test_microwave_close_plans(door_sim, tmp_path):
    scene, rb = door_sim
    dc.DOOR_FILE.write_text(json.dumps({**MICROWAVE_DOOR, "door_open_deg": 90.0}))
    hand = _Hand(scene, rb, dc.load_park())
    args = close_door.close_args(False, route_over=True, keep_orientation=True)
    assert close_door.push_close(hand, args)


def test_keep_out_is_off_until_enabled_and_refuses_when_enabled(door_sim):
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
