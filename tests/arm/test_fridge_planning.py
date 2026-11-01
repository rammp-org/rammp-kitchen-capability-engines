"""The fridge's door planned through door_common, dry, in the sim, and its run on a fake arm.

CI starts capabilities in mock mode only, so nothing else runs this planning code. The door is the
real recorded one of 10-24's successful open, on the arm model it was recorded with; the height
pinned below is what feeding-deployment's own planner gives for the same inputs.
"""

import json

import numpy as np
import pytest
import rammp_arm
from conftest import cornell_sim
from door_common import close_door as shared_close
from door_common import common as dc
from door_common import open_door as shared
from fridge import close_door, constants, open_door, push_open
from microwave import open_door as microwave_open
from scipy.spatial.transform import Rotation as R

from capability_engine import sim

HANDLE_QUAT = np.array([-0.5, 0.5, 0.5, -0.5])
FRIDGE_DOOR = {
    "closed_normal": [-0.9958624675323723, -0.09087324006733077, 0.0],
    "closed_grasp_pos": [0.48977151695907806, -0.42941837656177184, 0.6332621961894848],
    "closed_handle": [0.5375729154006319, -0.42505646103853995, 0.6332621961894848],
    "door_z": [0.17606219618948477, 0.9507621961894848],
    "door_mid": [0.5776990101471697, -0.30644391665061677, 0.6561220367153857],
    "hinge": [0.5556401837786342, -0.05357762407288291, 0.6332621961894848],
    "hinge_pivot": [0.5556401837786342, -0.05357762407288291, 0.6332621961894848],
    "detected_edge": [0.5665943200626593, -0.1326923682422464, 0.5839634412476657],
    "door_open_deg": 90.0,
    "open_sign": -1.0,
    "face_behind_grasp": 0.053,
    "door_t": 0.10668,
    "door_past_handle": 0.0889,
    "body_depth": 0.5,
}
# read from the arm, 10-24
TELEOP_START = np.radians((-40.6, -74.8, -134.9, -104.7, 25.4, -41.7, 112.2))
# the web app's Fridge start pose
FRIDGE_START = np.radians((98.8, 10.2, 179.5, -94.7, 98.7, 100.2, 16.1))
END_OF_PULL = (  # the real end of the 12:13 pull: tool pose, joints
    [0.17979409112004632, 0.01231713312600509, 0.6330638266165626]
    + [0.7871527327029976, 0.035830726972973584, 0.02809743162107746, 0.615075010660562],
    [
        1.3806298470483718,
        -1.029408700379304,
        2.415947569254344,
        -1.5271258112397774,
        0.26937631412702023,
        -1.4587065396745515,
        0.7973411093059086,
    ],
)


def _detection():
    return {
        "h_raw": np.array(FRIDGE_DOOR["closed_handle"]),
        "orient": HANDLE_QUAT,
        "n": np.array(FRIDGE_DOOR["closed_normal"]),
        "hinge": np.array(FRIDGE_DOOR["hinge_pivot"]),
        "hinge_seen": np.array(FRIDGE_DOOR["detected_edge"]),
        "door_z": np.array(FRIDGE_DOOR["door_z"]),
    }


class _Arm:
    """self.arm's calls on the sim: a move ends where it was told, a move into the keep-out
    raises as self.arm's does. Nothing here can reach a real arm. get_state never touches the sim,
    as self.arm's doesn't: RunLog calls it from its own thread, and FK there would race the
    planning thread's joints."""

    def __init__(self, scene, rb, q):
        self.scene, self.rb, self.gripper_pos, self.calls = scene, rb, 0.0, []
        self._arrive(q)

    def _arrive(self, q):
        self.q = np.array(q, float)
        self.ee = [*np.concatenate(sim.tool_pose(self.scene, self.rb, self.q))]

    def get_state(self):
        return {
            "position": list(self.q),
            "ee_pos": self.ee,
            "gripper_pos": self.gripper_pos,
            "effort": [0.0] * 7,
            "velocity": [0.0] * 7,
        }

    def move_joints(self, knots, cart_lengths=None, speed_scale=1.0, impedance=None, monitor=None):
        _, q, _ = rammp_arm.joint_trajectory(self.q, knots, cart_lengths, speed_scale)
        for qi in list(q[::2]) + [q[-1]]:
            assert sim.keep_out_message(self.rb, qi) is None
        self.calls.append(("joints", cart_lengths is not None, speed_scale))
        self._arrive(knots[-1])
        return True

    def go_to_joint_config(self, q, label=""):
        self.calls.append(("curobo", label))
        self._arrive(q)
        return True

    def gripper(self, position, force=None):
        self.calls.append(("gripper", position, force))
        self.gripper_pos = 0.8 if position > 0.5 else 0.0
        return True

    def execute_command(self, command):
        return self.gripper(0.0)


@pytest.fixture
def door_sim(tmp_path, monkeypatch):
    """The sim on the arm the door was recorded with, door_common configured for the fridge,
    every constant restored afterwards."""
    saved = dict(vars(dc)), dict(vars(shared))
    scene, rb = cornell_sim()
    open_door.configure(tmp_path)
    monkeypatch.setattr(
        shared,
        "detect_handle",
        lambda: (
            np.array(FRIDGE_DOOR["closed_handle"]),
            HANDLE_QUAT,
            np.array(FRIDGE_DOOR["closed_normal"]),
            None,
            None,
            np.array(FRIDGE_DOOR["detected_edge"]),
        ),
    )
    dc.set_camera(type("NoFrames", (), {"grab": lambda self: None})())
    yield scene, rb
    dc.set_camera(None)
    sim.reset_for_tests()
    vars(dc).update(saved[0])
    vars(shared).update(saved[1])


def test_the_open_plans_the_full_pull_and_the_route_round_the_door(door_sim):
    scene, rb = door_sim
    after = open_door.ReleaseAround()
    plan, why = shared.plan_open(scene, rb, _detection(), TELEOP_START, 90, 90, after)
    assert why is None
    assert plan["swing"] == 90
    assert "round the door OK" in plan["after"]


def test_the_open_refuses_rather_than_swing_less(door_sim):
    scene, rb = door_sim
    after = push_open.PushOpen()
    plan, why = shared.plan_open(scene, rb, _detection(), TELEOP_START, 50, 50, after)
    assert plan is None
    assert "push-open" in why


def test_the_route_round_the_door_ends_upright_beside_it(door_sim):
    scene, rb = door_sim
    pts, legs, msg = open_door.plan_release_around(scene, rb, FRIDGE_DOOR, *END_OF_PULL)
    assert "10 cm past its free edge" in msg
    assert len(legs) == len(pts) - 1
    assert R.from_quat(pts[-1][1]).as_matrix()[2, 1] == pytest.approx(1.0)
    assert sim.tool_pose(scene, rb, legs[-1][-1])[0] == pytest.approx(pts[-1][0], abs=0.005)


def test_the_push_open_plans_from_the_fridge_start_pose(door_sim):
    scene, rb = door_sim
    after = push_open.PushOpen()
    plan, why = shared.plan_open(scene, rb, _detection(), FRIDGE_START, 50, 50, after)
    assert why is None
    assert plan["after"] == "push-open OK"


def test_a_30_degree_mount_plans_the_open_and_close_with_the_fridge_further_out(door_sim, tmp_path):
    sim.reset_for_tests()
    scene, rb = sim.make_sim("/fixtures/urdf/flatbot.urdf")
    open_door.configure(tmp_path, 30)
    shift = np.array([0.14, 0.0, 0.0])
    geom = _detection()
    geom = {k: geom[k] + shift if k in ("h_raw", "hinge", "hinge_seen") else geom[k] for k in geom}
    after = open_door.ReleaseAround()
    plan, why = shared.plan_open(scene, rb, geom, TELEOP_START, 90, 90, after)
    assert why is None
    assert plan["swing"] == 90
    door = {
        k: (np.array(v) + shift).tolist()
        if k in ("closed_handle", "closed_grasp_pos", "hinge")
        else v
        for k, v in FRIDGE_DOOR.items()
    }
    dc.DOOR_FILE.write_text(json.dumps(door))
    assert close_door.run(_Arm(scene, rb, TELEOP_START), False)


def test_the_mount_must_be_a_known_one(door_sim, tmp_path):
    with pytest.raises(ValueError, match="mount_tilt_deg"):
        open_door.configure(tmp_path, 15)


def test_a_door_not_measured_here_changes_the_model(door_sim, tmp_path):
    open_door.configure(tmp_path, door_width_m=0.5, door_depth_m=0.2, hinge_along_door_m=0.4)
    assert shared.DOOR_MODEL["door_past_handle"] == pytest.approx(0.1)
    assert shared.DOOR_MODEL["door_t"] == 0.2
    assert shared.DOOR_Z_FROM_HANDLE == pytest.approx((-18 * 0.0254, 12.5 * 0.0254))


def test_the_close_finds_feeding_deployments_height_and_never_tilts(door_sim):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    shift = np.asarray(constants.CHAIR_BACK_DIR) * constants.CHAIR_BACK_M
    dc.DOOR_FILE.write_text(json.dumps(FRIDGE_DOOR))
    z = close_door.close_height(arm, FRIDGE_DOOR, {}, shift, constants.CLOSE_TO_DEG)
    assert z == pytest.approx(0.85606, abs=1e-4)


def test_an_executed_close_needs_the_chair_moved(door_sim):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    dc.DOOR_FILE.write_text(json.dumps(FRIDGE_DOOR))
    with pytest.raises(dc.Refused, match="WHEELCHAIR"):
        close_door.run(arm, True)
    assert arm.calls == []
    assert close_door.run(arm, False)  # a dry run plans as if it had
    assert arm.calls == []
    assert close_door.run(arm, True, chair_moved=True)
    # the gripper is never touched, no cuRobo near the door
    assert all(c[0] == "joints" for c in arm.calls)
    assert json.loads(dc.DOOR_FILE.read_text())["door_open_deg"] == constants.CLOSE_TO_DEG


def test_a_close_without_an_opened_door_refuses(door_sim):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    with pytest.raises(dc.Refused, match="the open writes it"):
        close_door.run(arm, True, chair_moved=True)


def test_the_whole_open_sends_checked_joints_and_leaves_the_door_for_the_close(door_sim):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    assert open_door.run(arm, True)
    kinds = [c[0] for c in arm.calls]
    assert kinds == ["curobo", "joints", "gripper", "joints", "gripper", "joints", "joints"]
    assert arm.calls[2] == ("gripper", 1.0, constants.GRASP_FORCE)
    door = json.loads(dc.DOOR_FILE.read_text())
    assert door["door_open_deg"] == pytest.approx(90.0, abs=0.1)
    assert door["hinge"] == door["hinge_pivot"]  # measured, not the detected edge
    assert door["door_t"] == pytest.approx(constants.MEASURED["door_depth_m"])
    assert list(dc.RUNS_DIR.glob("*_open/run.json"))
    assert dc.LAST_GRASP_FILE.exists()


def test_a_dry_open_moves_nothing(door_sim):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    assert open_door.run(arm, False)
    assert arm.calls == []
    assert not dc.DOOR_FILE.exists()


def test_release_only_lets_go_and_backs_off(door_sim):
    scene, rb = door_sim
    arm = _Arm(scene, rb, END_OF_PULL[1])
    arm.gripper_pos = 0.8
    assert open_door.run(arm, True, release_only=True)
    assert [c[0] for c in arm.calls] == ["gripper", "joints"]
    assert arm.gripper_pos == 0.0


def test_the_keep_out_is_the_fridges_alone(door_sim, tmp_path):
    assert sim.keep_out_enabled()
    sim.set_keep_out(None)
    microwave_open.configure(tmp_path)
    assert not sim.keep_out_enabled()


def test_the_close_settings_are_door_commons(door_sim):
    known = set(shared_close.CLOSE_DEFAULTS)
    assert set(constants.CLOSE_SETTINGS) <= known
    assert set(close_door.GOAL_SETTINGS) - {"chair_back", "close_to_deg", "chair_moved"} <= known
