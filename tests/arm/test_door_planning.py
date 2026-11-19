"""The microwave's door planned through door_common, dry, on the arm model (numpy, the driver's
URDF): a shared change that harms it shows up here.

CI starts capabilities in mock mode only, so nothing else runs this planning code. The door is a
real recorded door file (10-24).
"""

import json

import numpy as np
import pytest
from door_common import arm_model, close_door, open_door
from door_common import common as dc
from microwave import open_door as microwave_open

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
        pos, quat = rb.pose(q)
        self.state = {"position": list(q), "ee_pos": [*pos, *quat], "gripper_pos": 0.0}

    def get_state(self):
        return self.state


@pytest.fixture
def door_sim(tmp_path):
    """The arm model on the driver's URDF, door_common configured for the microwave, every constant
    restored afterwards."""
    saved = dict(vars(dc)), dict(vars(open_door))
    arm_model.load("/fixtures/urdf/dojo.urdf")
    scene, rb = dc.make_sim()
    microwave_open.configure(tmp_path)
    yield scene, rb
    arm_model.set_model_provider(None)
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


def test_the_model_ik_returns_to_the_pose_it_was_given(door_sim):
    _, rb = door_sim
    q = KINOVA_HOME + np.radians([10.0, 5.0, -10.0, 8.0, 5.0, -6.0, 4.0])
    pos, quat = rb.pose(q)
    sol, err = rb.ik(pos, quat, KINOVA_HOME)
    assert err < 1e-3
    assert np.allclose(rb.pose(sol)[0], pos, atol=1e-3)


def test_clearance_sees_a_box_at_the_tool_and_inside_the_hand(door_sim):
    _, rb = door_sim
    tool = rb.pose(KINOVA_HOME)[0]
    # smaller than the hand: none of its surface points is in it
    inside = dc.add_sim_box(None, rb, tool, (0.005, 0.005, 0.005), 0.0)
    gap, what = dc.clearance(rb, KINOVA_HOME, [(inside, "a small box")])
    assert gap < 0 and "gripper" in what
    far = dc.add_sim_box(None, rb, tool + [0.0, 0.0, 1.0], (0.05, 0.05, 0.05), 0.0)
    assert dc.clearance(rb, KINOVA_HOME, [(far, "far")])[0] > 0.5


def test_a_held_container_is_checked_too(door_sim):
    _, rb = door_sim
    pos, quat = rb.pose(KINOVA_HOME)
    held = close_door._container_held(rb, 0.16)
    centre = pos + arm_model.R.from_quat(quat).apply(held.centre_tool)
    over = centre + [0.0, 0.0, 0.10]  # just over the container
    box = dc.add_sim_box(None, rb, over, (0.03, 0.03, 0.01), 0.0)
    assert dc.clearance(rb, KINOVA_HOME, [(box, "lid")], held=held)[1].startswith("container")


class _Validator:
    """self.arm.blend as a fake: every one-leg check answers `answer`; what it was given is kept."""

    def __init__(self, answer):
        self.answer, self.legs = answer, []

    def blend(self, legs, radii):
        self.legs.append(legs[0])
        return self.answer


def test_curobo_validates_the_path_as_move_joints_sends_it():
    ok = _Validator(Plan(True, "ok"))
    assert dc.curobo_validates(ok, np.zeros(7), PATH, "the swing") == ""
    leg = ok.legs[0]
    # from the start, through every knot
    assert np.allclose(leg.q[0], 0.0) and np.allclose(leg.q[-1], PATH[-1])
    hit = _Validator(Plan(False, "COLLISION with person/lap"))
    assert "person/lap" in dc.curobo_validates(hit, np.zeros(7), PATH, "the swing")


def test_curobo_validation_passes_without_a_planner():
    assert dc.curobo_validates(object(), np.zeros(7), PATH, "the swing") == ""


def test_the_exact_swing_is_validated_by_curobo(door_sim):
    assert open_door.SWING_EXACT and open_door.APPROACH_EXACT  # the microwave's
    plan, _ = open_door.plan_open(
        *door_sim, _detection(MICROWAVE_DOOR), KINOVA_HOME, 50, 45, microwave_open.PushOpen(None)
    )
    hit = _Validator(Plan(False, "COLLISION with person/lap"))
    why = open_door._exact_problem(hit, plan)
    assert "person/lap" in why
    assert len(hit.legs) == 1  # the first exact leg refused: nothing after it asked


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


class _CuroboHand(_Hand):
    """_Hand with a stand-in cuRobo (the test image has none): plan_to_pose is the arm model's local
    IK from the start (a joint line there), plan_to_joints a joint line, set_world accepted."""

    def __init__(self, scene, rb, q):
        super().__init__(scene, rb, q)
        self.rb, self.q = rb, np.asarray(q, float)

    def raw_joints(self):
        return self.q.copy()

    def set_world(self, world):
        return True, "ok"

    @staticmethod
    def _line(start, goal):
        t, q = np.linspace(0, 1, 8), np.linspace(start, goal, 8)
        return Plan(True, "ok", t=t, q=q, qd=np.zeros((8, 7)))

    def plan_to_joints(self, goal, start):
        return self._line(start, goal)

    def plan_to_pose(self, pos, quat, start, hold=0):
        q, err = self.rb.ik(pos, quat, start)
        return self._line(start, q) if err < 0.005 else Plan(False, "IK_FAIL")


def test_microwave_close_plans(door_sim, tmp_path, monkeypatch):
    scene, rb = door_sim
    (tmp_path / "curobo").mkdir()  # where the planner reads the door world: it must exist
    monkeypatch.setattr(dc, "DOOR_WORLD_DIR", tmp_path / "curobo" / "door_world")
    dc.DOOR_FILE.write_text(json.dumps({**MICROWAVE_DOOR, "door_open_deg": 90.0}))
    hand = _CuroboHand(scene, rb, dc.load_park())
    # the default close (curobo_ik): cuRobo's configuration at the swing start, the arc by local IK
    args = close_door.close_args(False, keep_orientation=True)
    assert close_door.push_close(hand, args)


def test_the_close_lets_curobo_choose_the_configuration_by_default():
    assert close_door.CLOSE_DEFAULTS["curobo_ik"]


def test_configure_refuses_what_is_not_a_constant_and_what_is_missing(door_sim):
    with pytest.raises(ValueError, match="unknown"):
        open_door.configure(GRIP_EXTT=0.1)
    with pytest.raises(ValueError, match="missing"):
        open_door.configure(GRIP_EXT=None)


def test_a_missing_capability_value_fails_rather_than_borrowing_the_microwaves():
    assert dc.PARK_JOINTS is None
    with pytest.raises(dc.Refused, match="PARK_JOINTS"):
        dc.load_park()
