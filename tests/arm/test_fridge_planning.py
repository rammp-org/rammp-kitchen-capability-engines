"""The fridge's door planned through door_common, dry, in the sim, and its run on a fake arm.

CI starts capabilities in mock mode only, so nothing else runs this planning code. The door is the
real recorded one of 10-24's successful open, on the arm model it was recorded with; the height
pinned below is what feeding-deployment's own planner gives for the same inputs.
"""

import importlib.util
import json
import sys
from unittest.mock import MagicMock

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
# handle_centre_to_face_m these tests were written with (de5d52f, 11-01)
BLACK_HANDLE_TO_FACE_M = 1.5 * 0.0254
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
    """The sim on the arm the door was recorded with, door_common configured for the fridge as it
    these tests were written with (1.5 in; the defaults are now the STL's),
    every constant restored afterwards."""
    saved = dict(vars(dc)), dict(vars(shared)), dict(vars(constants))
    scene, rb = cornell_sim()
    open_door.configure(tmp_path, handle_centre_to_face_m=BLACK_HANDLE_TO_FACE_M)
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
    vars(constants).update(saved[2])


def test_the_open_plans_the_full_pull_and_the_route_round_the_door(door_sim):
    scene, rb = door_sim
    after = open_door.ReleaseAround()
    plan, why = shared.plan_open(scene, rb, _detection(), TELEOP_START, 90, 90, after)
    assert why is None
    assert plan["swing"] == 90
    assert "round the door OK" in plan["after"]


def test_the_open_refuses_rather_than_swing_less(door_sim, monkeypatch):
    scene, rb = door_sim
    monkeypatch.setattr(constants, "CHAIR_MOVES_M", (0.0,))  # the wheelchair stays where it is
    plan, why = shared.plan_open(
        scene, rb, _detection(), TELEOP_START, 50, 50, push_open.PushOpen()
    )
    assert plan is None
    assert "push-open" in why


def test_the_push_open_asks_for_the_smallest_wheelchair_move_that_lets_it_push(door_sim):
    scene, rb = door_sim
    after = push_open.PushOpen()
    plan, why = shared.plan_open(scene, rb, _detection(), TELEOP_START, 50, 50, after)
    assert why is None
    assert after.chair_move == pytest.approx(0.10)  # tried 0 (refused above), +5, -5 cm first
    assert plan["after"] == "push-open OK, after the wheelchair moves ~10 cm back"


def test_the_route_round_the_door_ends_upright_beside_it(door_sim):
    scene, rb = door_sim
    pts, legs, msg = open_door.plan_release_around(scene, rb, FRIDGE_DOOR, *END_OF_PULL)
    assert "10 cm past its free edge" in msg
    assert len(legs) == len(pts) - 1
    assert R.from_quat(pts[-1][1]).as_matrix()[2, 1] == pytest.approx(1.0)
    assert sim.tool_pose(scene, rb, legs[-1][-1])[0] == pytest.approx(pts[-1][0], abs=0.005)


def test_the_hinge_and_the_door_face_are_measured_from_the_handle(door_sim):
    m = constants.MEASURED
    h, n = np.array(FRIDGE_DOOR["closed_handle"]), np.array(FRIDGE_DOOR["closed_normal"])
    hinge, h2, grasp, _ = shared.FIND_HINGE(h, HANDLE_QUAT, n, None)
    into, along = -n, np.cross(n, [0.0, 0.0, 1.0])
    want = (
        h2
        + into * (BLACK_HANDLE_TO_FACE_M + m["pin_behind_face_m"])
        + along * m["hinge_along_door_m"]
    )
    assert hinge == pytest.approx(
        want
    )  # from the handle, not the grasp's tool point ~5 cm in front of it
    model = shared.door_model(n, grasp.position, h2)
    face = np.asarray(grasp.position) - n * model["face_behind_grasp"]
    assert np.dot(h2 - face, n) == pytest.approx(
        BLACK_HANDLE_TO_FACE_M
    )  # the face behind the handle


def test_the_wheelchair_moves_are_said_both_ways_and_move_the_door(door_sim):
    assert "back ~10 cm" in close_door.move_chair(0.10)
    assert "forward ~5 cm" in close_door.move_chair(-0.05)
    door = push_open.shifted(
        FRIDGE_DOOR, 0.10, (1.0, 0.0)
    )  # back 10 cm: the fridge 10 cm further out
    assert door["hinge"][0] == pytest.approx(FRIDGE_DOOR["hinge"][0] + 0.10)
    assert door["closed_normal"] == FRIDGE_DOOR["closed_normal"]


def _asked_cm(text):
    """The wheelchair move (cm, + back) a message asks for."""
    import re

    m = re.search(r"straight (back|forward) ~(\d+) cm", text)
    return int(m.group(2)) * (1 if m.group(1) == "back" else -1)


def test_the_open_pulls_backs_off_asks_for_the_wheelchair_then_pushes(door_sim):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    assert open_door.run(arm, False, swing_deg=50)  # the dry run says which move it will ask for
    assert arm.calls == [] and push_open.NEXT_STEP.startswith("Plan OK")
    planned = _asked_cm(push_open.NEXT_STEP)
    assert planned != 0

    assert open_door.run(arm, True, swing_deg=50)
    kinds = [c[0] for c in arm.calls]
    assert kinds == [
        "curobo",
        "joints",
        "gripper",
        "joints",
        "gripper",
        "joints",
    ]  # ... let go, back off: stop
    assert arm.gripper_pos == 0.0
    # asked from where the pull really ended (the door's angle as reached): the dry run's to a step
    asked = _asked_cm(push_open.NEXT_STEP)
    assert abs(asked - planned) <= 5 and "chair_moved true" in push_open.NEXT_STEP
    before = json.loads(dc.DOOR_FILE.read_text())
    assert before["push_after_chair_move"] == pytest.approx(asked / 100)

    arm.calls.clear()  # the wheelchair moved: the arm with it, the door further out in its frame
    assert open_door.run(arm, True, swing_deg=50, chair_moved=True)
    assert [c[0] for c in arm.calls] == ["joints"]  # the push, nothing else
    after = json.loads(dc.DOOR_FILE.read_text())
    assert after["door_open_deg"] == constants.PUSH_OPEN_TO_DEG
    assert after["push_after_chair_move"] is None
    # where the door now is, for the close
    assert after["hinge"][0] == pytest.approx(before["hinge"][0] + asked / 100)
    with pytest.raises(dc.Refused, match="no push waiting"):  # once only
        open_door.run(arm, True, chair_moved=True)


def test_the_push_open_plans_from_the_fridge_start_pose(door_sim):
    scene, rb = door_sim
    after = push_open.PushOpen()
    plan, why = shared.plan_open(scene, rb, _detection(), FRIDGE_START, 50, 50, after)
    assert why is None
    assert plan["after"].startswith("push-open OK")


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


def test_a_30_degree_mount_pulls_60_and_pushes_open_at_the_11_01_handle_height(door_sim, tmp_path):
    sim.reset_for_tests()
    scene, rb = sim.make_sim("/fixtures/urdf/flatbot.urdf")
    open_door.configure(tmp_path, 30)
    assert constants.PUSH_OPEN_DZ_M == pytest.approx(0.10)
    assert max(shared.GRASP_PITCHES_DEG) == 45
    # the 11-01 flatbot handle, ~0.31 m above the arm base, door square to the chair
    h, n = np.array([0.84, -0.28, 0.31]), np.array([-1.0, 0.0, 0.0])
    hinge = shared.FIND_HINGE(h, HANDLE_QUAT, n, None)[0]
    geom = {
        "h_raw": h,
        "orient": HANDLE_QUAT,
        "n": n,
        "hinge": hinge,
        "hinge_seen": None,
        "door_z": None,
    }
    pull = constants.PUSH_OPEN_PULL_DEG
    plan, why = shared.plan_open(scene, rb, geom, TELEOP_START, pull, pull, push_open.PushOpen())
    assert why is None
    assert plan["swing"] == 60
    assert plan["after"].startswith("push-open OK")


def test_the_level_arm_keeps_door_commons_tilts_and_pushes_below_the_handle(door_sim, tmp_path):
    open_door.configure(tmp_path, 30)
    open_door.configure(tmp_path, 0)
    assert constants.PUSH_OPEN_DZ_M == pytest.approx(-0.10)
    assert shared.GRASP_PITCHES_DEG == (0, 10, 14, 17, 25)


def test_the_mount_must_be_a_known_one(door_sim, tmp_path):
    with pytest.raises(ValueError, match="mount_tilt_deg"):
        open_door.configure(tmp_path, 15)


def test_a_door_not_measured_here_changes_the_model(door_sim, tmp_path):
    open_door.configure(tmp_path, door_width_m=0.5, door_depth_m=0.2, hinge_along_door_m=0.4)
    assert shared.DOOR_MODEL["door_past_handle"] == pytest.approx(0.1)
    assert shared.DOOR_MODEL["door_t"] == 0.2
    m = constants.MEASURED
    expected = (-m["door_below_handle_m"], m["door_above_handle_m"])
    assert shared.DOOR_Z_FROM_HANDLE == pytest.approx(expected)


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
    assert open_door.run(arm, True, push=False)
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
    assert open_door.run(arm, False, push=False)
    assert arm.calls == []
    assert not dc.DOOR_FILE.exists()


def test_a_dry_open_with_save_door_leaves_the_door_an_executed_open_does(door_sim):
    scene, rb = door_sim
    assert open_door.run(_Arm(scene, rb, TELEOP_START), True, push=False)
    executed = json.loads(dc.DOOR_FILE.read_text())
    dc.DOOR_FILE.unlink()
    arm = _Arm(scene, rb, TELEOP_START)
    assert open_door.run(arm, False, push=False, save_door=True)
    assert arm.calls == []
    saved = json.loads(dc.DOOR_FILE.read_text())
    placed = ("closed_normal", "closed_grasp_pos", "closed_handle", "door_z", "hinge")
    for k in (*placed, "hinge_pivot", "detected_edge"):
        assert saved[k] == pytest.approx(executed[k], abs=1e-6), k
    for k in ("face_behind_grasp", "door_t", "door_past_handle", "body_depth", "open_sign"):
        assert saved[k] == executed[k], k
    assert saved["door_open_deg"] == pytest.approx(executed["door_open_deg"], abs=0.5)


def test_save_door_after_the_push_open_leaves_the_door_at_the_push_angle(door_sim, monkeypatch):
    scene, rb = door_sim
    geom = _detection()
    plan, why = shared.plan_open(scene, rb, geom, FRIDGE_START, 50, 50, push_open.PushOpen())
    assert why is None

    def open_task(ai, execute, max_swing, min_swing, after_swing):
        shared.LAST_PLAN = dict(q0=FRIDGE_START, plan=plan, geom=geom)
        return True

    monkeypatch.setattr(shared, "open_task", open_task)
    dc.DOOR_FILE.write_text(json.dumps({"door_mid": [9.0, 9.0, 9.0]}))  # an older door's
    assert open_door.run(_Arm(scene, rb, FRIDGE_START), False, save_door=True)
    door = json.loads(dc.DOOR_FILE.read_text())
    assert door["door_open_deg"] == constants.PUSH_OPEN_TO_DEG
    assert door["hinge"] == pytest.approx(geom["hinge"])
    assert "door_mid" not in door  # replaced, not merged


def test_save_door_never_saves_an_earlier_runs_plan(door_sim, monkeypatch):
    scene, rb = door_sim
    shared.LAST_PLAN = dict(q0=TELEOP_START, plan={}, geom={})

    def open_task(ai, execute, max_swing, min_swing, after_swing):
        return False  # this run plans nothing

    monkeypatch.setattr(shared, "open_task", open_task)
    assert not open_door.run(_Arm(scene, rb, TELEOP_START), False, save_door=True)
    assert not dc.DOOR_FILE.exists()


def test_the_open_defaults_to_the_60_degree_pull_and_push(door_sim, monkeypatch):
    scene, rb = door_sim
    asked = {}

    def open_task(ai, execute, max_swing, min_swing, after_swing):
        asked.update(swing=(max_swing, min_swing), after=after_swing)
        return True

    monkeypatch.setattr(shared, "open_task", open_task)
    monkeypatch.setattr(open_door, "push_open", lambda ai, execute, after: True)
    assert open_door.run(_Arm(scene, rb, TELEOP_START), False)
    assert asked["swing"] == (60, 60)
    assert isinstance(asked["after"], push_open.PushOpen)


def test_the_grasp_offset_moves_the_grasp_and_the_door_with_it(door_sim):
    lo, hi = shared.DOOR_Z_FROM_HANDLE
    open_door.set_grasp_offset(0.03)
    assert shared.VERTICAL_CORR == pytest.approx(constants.VERTICAL_CORR + 0.03)
    # the door stays where it is
    assert shared.DOOR_Z_FROM_HANDLE == pytest.approx((lo - 0.03, hi - 0.03))
    handle = np.array(FRIDGE_DOOR["closed_handle"])
    h, grasp = shared.grasp_pose(handle, HANDLE_QUAT, np.array(FRIDGE_DOOR["closed_normal"]))
    assert grasp.position[2] == pytest.approx(handle[2] + 0.03)
    assert h[2] + shared.DOOR_Z_FROM_HANDLE[1] == pytest.approx(handle[2] + hi)
    open_door.set_grasp_offset(0.0)  # each run sets its own: none left over
    assert shared.VERTICAL_CORR == constants.VERTICAL_CORR
    assert shared.DOOR_Z_FROM_HANDLE == pytest.approx((lo, hi))


def test_the_grasp_offset_stays_on_the_measured_bar(door_sim, tmp_path):
    open_door.configure(tmp_path, handle_length_m=0.20)
    limit = 0.10 - constants.GRASP_END_MARGIN_M
    assert open_door.grasp_offset_limit() == pytest.approx(limit)
    open_door.set_grasp_offset(-limit)
    with pytest.raises(dc.Refused, match="off the handle"):
        open_door.set_grasp_offset(limit + 0.005)
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    with pytest.raises(dc.Refused, match="off the handle"):
        open_door.run(arm, True, grasp_offset=0.2)
    assert arm.calls == []


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


@pytest.fixture
def perception(monkeypatch):
    """door_common.perception and the fridge's, without open3d / message_filters (not in these
    tests' image)."""
    for name in ("open3d", "message_filters"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    import door_common.perception
    from fridge import perception as fridge_perception

    return door_common.perception, fridge_perception


# a level camera 0.5 m up, looking along the arm's +x: image right is the arm's -y, image down is -z
CAM_K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 200.0], [0.0, 0.0, 1.0]])
CAM_T = np.eye(4)
CAM_T[:3, :3] = np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
CAM_T[2, 3] = 0.5


def _door_scene(door_m):
    """Depth (mm) of a door `door_m` from the camera, a 2 cm wide vertical bar on it at 0.50 m
    (columns 400-415, rows 100-300), and SAM 3's mask of the bar, a little wider than it (so it
    takes in some door)."""
    depth = np.full((400, 640), door_m * 1000.0, np.float32)
    depth[100:300, 400:416] = 500.0
    mask = np.zeros((400, 640), bool)
    mask[100:300, 396:420] = True
    return depth, mask


def test_the_handle_is_the_bar_front_from_its_own_depth_whatever_the_door_reads(perception):
    _, fp = perception
    for door_m in (0.55, 1.05):  # a true door; a glossy one returning its reflection
        depth, mask = _door_scene(door_m)
        h, _, n = fp.HandleDetector()._bar_front(mask, depth, CAM_K, CAM_T[2, :3])
        assert h[2] == pytest.approx(0.50, abs=0.005)  # on the bar, not the door behind it
        assert h[0] == pytest.approx((407.5 - 320) / 400 * 0.5, abs=0.005)
        assert n == 16 * 200


def _plane(door_m):
    """door_common's _door_plane result: a plane facing the camera `door_m` away (camera frame)."""
    return (
        None,
        None,
        np.array([0.0, 0.0, -1.0]),
        door_m,
        CAM_T[2, :3],
        CAM_T[:3, :3],
        CAM_T[:3, 3],
    )


@pytest.mark.parametrize(
    ("planes", "picked"),
    [
        ([0.55], 0.55),  # a true door: the first plane, as before
        ([1.05, 0.55], 0.55),  # its reflection first: refused, the door found next
        ([0.503, 0.55], 0.55),  # flush with the handle's front (the handle, its mount): refused
        ([1.05, 1.06], None),  # only the reflection: no door plane, and the look says why
    ],
)
def test_a_door_plane_must_be_where_the_handle_can_stand_on_it(
    perception, monkeypatch, planes, picked
):
    dp, fp = perception
    queue = [_plane(d) for d in planes]
    monkeypatch.setattr(
        dp.HandleDetector, "_door_plane", lambda self, *a: queue.pop(0) if queue else None
    )
    det = fp.HandleDetector(0.0381)
    depth, mask = _door_scene(planes[0])
    det._mask, det._base_T_cam, det._exclude = mask, CAM_T, None
    det._bar, det._bar_extent, det._bar_points = det._bar_front(mask, depth, CAM_K, CAM_T[2, :3])
    got = det._door_plane(None, CAM_K, depth, CAM_T, None, None)
    if picked is None:
        assert got is None
        assert "no door plane fits the handle" in det.last_miss and "+55 cm" in det.last_miss
    else:
        assert got[3] == picked


def test_a_door_plane_facing_sideways_is_not_the_door(perception, monkeypatch):
    dp, fp = perception
    side = (None, None, np.array([1.0, 0.0, 0.0]), -0.43, CAM_T[2, :3], CAM_T[:3, :3], CAM_T[:3, 3])
    queue = [side, _plane(0.55)]  # 11-04: a plane at the handle's standoff, facing sideways
    monkeypatch.setattr(
        dp.HandleDetector, "_door_plane", lambda self, *a: queue.pop(0) if queue else None
    )
    det = fp.HandleDetector()
    depth, mask = _door_scene(0.55)
    det._mask, det._base_T_cam, det._exclude = mask, CAM_T, None
    det._bar, det._bar_extent, det._bar_points = det._bar_front(mask, depth, CAM_K, CAM_T[2, :3])
    assert det._door_plane(None, CAM_K, depth, CAM_T, None, None)[3] == 0.55


def test_two_looks_must_agree_on_the_door_too(door_sim, monkeypatch):
    def look(yaw_deg):
        n = np.array([-np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg)), 0.0])
        return np.array([0.65, -0.35, 0.32]), HANDLE_QUAT, n, None, None, None

    a, b, c = look(-49.0), look(-17.0), look(-20.0)  # 11-04: one handle, doors 31 deg apart
    assert not shared._looks_agree(a, b) and shared._looks_agree(b, c)
    shared.configure(NORMAL_AGREE_DEG=None)  # door_common's own: not compared
    assert shared._looks_agree(a, b)


def test_door_square_to_arm_takes_the_door_as_facing_the_arm_and_fits_no_plane(
    perception, monkeypatch, tmp_path
):
    dp, fp = perception

    def no_plane(self, *a):
        raise AssertionError("fit a door plane")

    monkeypatch.setattr(dp.HandleDetector, "_door_plane", no_plane)
    depth, mask = _door_scene(1.05)  # the door's depth only its reflection
    det = fp.HandleDetector(door_square_to_arm=True).detect(
        None, CAM_K, depth, CAM_T, handle_mask=mask
    )
    assert det["normal"] == pytest.approx([-1.0, 0.0, 0.0])
    assert det["handle"][0] == pytest.approx(
        0.50, abs=0.005
    )  # the bar's front, 0.5 m ahead of the camera
    assert det["hinge"] is None
    saved = dict(vars(dc)), dict(vars(shared)), dict(vars(constants))  # configure sets all three
    try:  # the node's parameter reaches the detector
        open_door.configure(tmp_path, 30, door_square_to_arm=True)
        assert shared.MAKE_DETECTOR().door_square_to_arm
        open_door.configure(tmp_path, 30)
        assert not shared.MAKE_DETECTOR().door_square_to_arm
    finally:
        vars(dc).update(saved[0])
        vars(shared).update(saved[1])
        vars(constants).update(saved[2])


def test_the_door_plane_is_fit_beside_the_handle_on_its_hinge_side(perception):
    _, fp = perception
    depth, mask = _door_scene(0.55)
    det = fp.HandleDetector()
    det._mask, det._base_T_cam, det._exclude = mask, CAM_T, None
    det._bar = det._bar_front(mask, depth, CAM_K, CAM_T[2, :3])[0]
    # looking along +x, the door's hinge (along it from the handle, toward the arm's +y) is to the
    # image's left
    region = det.door_region(np.zeros((400, 640, 3), np.uint8), None, None)
    us = np.nonzero(region)[1]
    assert us.max() < 396 and us.min() >= 396 - 4 - 3 * 24
    assert not (region & mask).any()


def _instances_scene():
    """A frame (door 0.55 m away, the 2 cm bar on it at 0.50 m, 25 cm tall) and SAM 3 instances
    on it: 11-04's wall sign (scoring best), the bench edge (big, lying down) and the bar."""
    depth, bar = _door_scene(0.55)
    sign, wide = np.zeros((400, 640), bool), np.zeros((400, 640), bool)
    sign[40:60, 20:45] = True
    wide[300:340, 100:400] = True
    frame = dict(depth_mm=depth, K=CAM_K, base_T_cam=CAM_T)
    return frame, sign, wide, bar


def test_the_handle_is_the_largest_handle_like_sam3_instance_not_the_best_scoring(perception):
    _, fp = perception
    frame, sign, wide, bar = _instances_scene()
    det = fp.HandleDetector()
    instances = [(0.80, sign, "handle"), (0.62, wide, "handle"), (0.45, bar, "handle")]
    assert det.pick_handle_mask(instances, frame) is bar
    assert det.pick_handle_mask(instances) is None  # no depth: none taken
    assert det.pick_handle_mask([], frame) is None


def test_the_detector_picks_again_after_a_detection(perception, monkeypatch):
    # every look after the first: detect() keeps the last bar's point count (11-05: a name clash)
    _, fp = perception
    frame, sign, wide, bar = _instances_scene()
    det = fp.HandleDetector()
    monkeypatch.setattr(
        fp.door_common.perception.HandleDetector, "detect", lambda *a, **k: {"hinge": 1}
    )
    det.detect(None, CAM_K, frame["depth_mm"], CAM_T, handle_mask=bar)
    assert det.pick_handle_mask([(0.8, sign, "handle"), (0.45, bar, "handle")], frame) is bar


def test_nothing_handle_like_is_a_miss_never_the_best_scoring(perception):
    _, fp = perception
    frame, sign, wide, _ = _instances_scene()
    det = fp.HandleDetector()
    assert det.pick_handle_mask([(0.8, sign, "handle"), (0.6, wide, "handle")], frame) is None
    assert "no SAM 3 instance of 2 is handle-like" in det.last_miss
    assert "2 cm tall" in det.last_miss  # the sign
    assert det.pick_handle_mask([(0.8, sign, "handle")]) is None  # without depth too


def _patch(rows, cols, depth_m, behind_m=None):
    """A frame with a rectangular patch (image rows, cols) `depth_m` away, a wall `behind_m`
    (default 5 cm) behind it, and the patch's mask: the outline SAM 3 would give it."""
    depth = np.full((400, 640), (behind_m or depth_m + 0.05) * 1000.0, np.float32)
    mask = np.zeros((400, 640), bool)
    mask[rows[0] : rows[1], cols[0] : cols[1]] = True
    depth[mask] = depth_m * 1000.0
    return dict(depth_mm=depth, K=CAM_K, base_T_cam=CAM_T), mask


def test_the_handle_limits_are_loose_and_have_no_distance_limit(perception):
    _, fp = perception
    det = fp.HandleDetector()
    # the printed handle (24.8 x 6.4 cm, 7.9 cm off the door) at 45 deg, 0.9 m away: ~24 x 10 cm
    frame, mask = _patch((90, 197), (300, 345), 0.9)
    height, width, _ = det._instance_size(mask, frame)  # 5th-95th percentiles of the outline
    assert (height, width) == (pytest.approx(0.22, abs=0.01), pytest.approx(0.09, abs=0.01))
    assert det.pick_handle_mask([(0.3, mask, "handle")], frame) is mask
    # handle-sized but 1.9 m away: the planner's reach decides, not the picker
    frame, mask = _patch((150, 200), (330, 343), 1.9)
    assert det.pick_handle_mask([(0.3, mask, "handle")], frame) is mask


@pytest.mark.parametrize(
    "rows, cols, depth_m, why",
    [
        ((100, 300), (320, 324), 0.5, "cm wide"),  # a sliver (a door edge, a pole): 25 cm x 5 mm
        ((100, 300), (0, 400), 0.5, "cm wide"),  # the bench edge: 25 cm tall, 50 cm across
        ((180, 200), (300, 340), 0.5, "cm tall"),  # a sign: 2.5 cm tall
        ((0, 400), (300, 340), 0.8, "cm tall"),  # 80 cm tall at 0.8 m: a door frame
        # within every size range, the wrong way up: handles mounted sideways, a door edge
        ((100, 167), (300, 433), 0.6, "sideways"),  # a horizontal bar: 20 cm wide, 10 cm tall
        ((100, 173), (300, 407), 0.6, "sideways"),  # a big lever: 16 x 11 cm
        ((20, 380), (300, 324), 0.5, "too narrow"),  # a door edge: 45 cm tall, 3 cm wide
    ],
)
def test_what_is_not_handle_sized_is_not_the_handle(perception, rows, cols, depth_m, why):
    _, fp = perception
    det = fp.HandleDetector()
    frame, mask = _patch(rows, cols, depth_m)
    assert det.pick_handle_mask([(0.9, mask, "handle")], frame) is None
    assert why in det.last_miss


@pytest.mark.parametrize(
    "height_px, width_px",
    # at 0.6 m: 25 x 6.4 cm (head-on), 25 x 8 (from the side), 19 x 10 (45 deg), the bar alone
    [(165, 43), (165, 53), (127, 67), (127, 40)],
)
def test_the_handle_passes_from_every_side(perception, height_px, width_px):
    _, fp = perception
    det = fp.HandleDetector()
    frame, mask = _patch((100, 100 + height_px), (300, 300 + width_px), 0.6)
    assert det.pick_handle_mask([(0.5, mask, "handle")], frame) is mask, det.last_miss


def test_an_instance_without_depth_is_not_the_handle(perception):
    # handle-shaped, no depth on it (out of range, like 11-04's door across the room)
    _, fp = perception
    det = fp.HandleDetector()
    frame, mask = _patch((100, 265), (300, 343), 0.6)
    frame["depth_mm"][mask] = 0.0
    assert det.pick_handle_mask([(0.9, mask, "handle")], frame) is None
    assert "no depth" in det.last_miss


def test_the_microwave_picks_sam3s_best_instance_as_before(perception):
    from microwave import perception as microwave_perception

    assert not hasattr(microwave_perception.HandleDetector, "pick_handle_mask")


def test_the_fridge_detector_reports_no_hinge(perception, monkeypatch):
    dp, fp = perception
    look = {
        "handle": np.array([0.5, -0.4, 0.3]),
        "normal": np.array([-1.0, 0.0, 0.0]),
        "hinge": np.zeros(3),
    }
    monkeypatch.setattr(dp.HandleDetector, "detect", lambda self, *a, **k: dict(look))
    assert fp.HandleDetector().detect(None, CAM_K, None, CAM_T)["hinge"] is None
    monkeypatch.setattr(dp.HandleDetector, "detect", lambda self, *a, **k: None)
    assert fp.HandleDetector().detect(None, CAM_K, None, CAM_T) is None  # a miss stays a miss


def test_a_look_uses_the_detectors_pick_keeps_the_mask_and_says_why_it_missed(
    tmp_path, monkeypatch
):
    saved_state = (
        dict(vars(dc)),
        dict(vars(shared)),
    )  # door_sim would stub detect_handle out: set up by hand
    open_door.configure(tmp_path)
    frame = dict(
        rgb=np.zeros((4, 4, 3), np.uint8),
        K=CAM_K,
        depth_mm=np.zeros((4, 4)),
        base_T_cam=CAM_T,
        box=None,
    )
    masks = [np.eye(4, dtype=bool), np.ones((4, 4), bool)]
    saved, picked = [], []

    class Cam:
        def wait_for_frames(self, timeout_s):
            return True

        def grab(self):
            return dict(frame)

        def sam3_handle_instances(self):
            return [(0.9, masks[0], "handle"), (0.5, masks[1], "handle")]

    class Detector:
        last_miss = "no door plane fits the handle (handle +55 cm in front)"

        def pick_handle_mask(self, instances, frame=None):
            picked.append(instances)
            return instances[1][1]

        def detect(self, *a, handle_mask=None, **k):
            return None

    monkeypatch.setattr(dc, "camera", lambda: Cam())
    monkeypatch.setattr(shared, "_save_frame", lambda f, tag, attempt, det: saved.append(f))
    monkeypatch.setattr(shared.time, "sleep", lambda s: None)
    shared.configure(MAKE_DETECTOR=Detector, HANDLE_SOURCE="sam3")
    try:
        with pytest.raises(dc.Refused, match=r"NO DETECTION \(1\).*handle \+55 cm in front"):
            shared.detect_handle()
    finally:
        vars(dc).update(saved_state[0])
        vars(shared).update(saved_state[1])
    assert len(picked[0]) == 2
    assert saved and all(f["handle_mask"] is masks[1] for f in saved)


def test_the_fridge_has_no_gates_on_where_the_handle_is(door_sim, tmp_path):
    assert (shared.PLAUSIBLE_X, shared.PLAUSIBLE_Y, shared.PLAUSIBLE_Z) == (None, None, None)
    assert (shared.MIN_Z, shared.MAX_Z, shared.HINGE_RADIUS_RANGE) == (None, None, None)
    assert shared.NORMAL_AGREE_DEG == 10.0
    h, n = (
        np.array([0.6, -0.3, 0.05]),
        np.array([-1.0, 0.0, 0.0]),
    )  # a handle 5 cm above the arm base
    shared.grasp_pose(h, HANDLE_QUAT, n)  # no height gate: reach, IK and collisions decide
    spec = importlib.util.find_spec(
        "door_common.open_door"
    )  # door_common's own values are still the microwave's
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert (
        fresh.PLAUSIBLE_X == (0.30, 0.90)
        and fresh.MIN_Z == 0.15
        and fresh.HINGE_RADIUS_RANGE == (0.25, 0.45)
    )
    assert fresh.NORMAL_AGREE_DEG is None and not fresh.PREFER_NEAR_START


def test_the_open_prefers_the_plan_nearest_the_start(door_sim):
    scene, rb = door_sim
    tried = []

    class Record(shared.AfterSwing):
        def check(self, scene, rb, geom, cand, explain):
            tried.append(cand)
            return False, "recorded"

    shared.plan_open(scene, rb, _detection(), TELEOP_START, 90, 90, Record())
    assert len(tried) > 1
    near = [
        shared._joint_dist_deg(TELEOP_START, c["q_pre"]) // shared.NEAR_START_BUCKET_DEG
        for c in tried
    ]
    keys = [(-c["swing"], d, -c["margin"]) for c, d in zip(tried, near, strict=True)]
    assert keys == sorted(keys)


def test_a_cancel_stops_the_planning(door_sim):
    scene, rb = door_sim

    def cancelled():
        raise KeyboardInterrupt  # stands in for the engine's Cancelled

    shared.configure(CHECKPOINT=cancelled)
    with pytest.raises(KeyboardInterrupt):
        shared.plan_open(scene, rb, _detection(), TELEOP_START, 90, 90, shared.AfterSwing())


def test_the_tee_logs_as_its_callers_lines(perception, tmp_path):
    import rclpy.logging
    from fridge import motion, node

    tee = node.TeeLogger(rclpy.logging.get_logger("fridge_tee_test"))
    motion._set_console(tmp_path / "console.log")
    try:
        # each from its own line, as callers make them: rclpy sees those lines, not the tee's
        tee.info("line")
        tee.error("line")  # 11-04: a 2nd severity on one tee line crashed every run
        tee.warning("line")
        tee.info("line")
        tee.fatal("line")
        # engine.py's throttled info (under the orchestrator): a filter on one caller's line only
        tee.info("waiting for the orchestrator", throttle_duration_sec=10.0)
        tee.info("line")
    finally:
        motion._set_console(None)
    assert [ln.split()[1] for ln in (tmp_path / "console.log").read_text().splitlines()] == [
        "INFO",
        "ERROR",
        "WARN",
        "INFO",
        "FATAL",
        "INFO",
        "INFO",
    ]


def test_a_close_plan_says_where_it_was_sent_to_move_the_chair_next(door_sim, monkeypatch):
    # door_common.perception's imports, which these tests don't have
    for name in ("open3d", "message_filters"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    from fridge import node

    monkeypatch.setattr(close_door, "run", lambda *a, **k: True)
    monkeypatch.setattr(node, "WristCamera", MagicMock())
    engine = MagicMock(mock=False, cfg={**node.PARAMS, "allow_execute": True}, instance_id="")
    ctx = MagicMock()

    out = node.Capability.execute(engine, "close", {"execute": False}, ctx)
    assert out["note"].startswith("Plan OK. Move the WHEELCHAIR straight back ~30 cm")
    assert "chair_moved true" in out["note"]
    assert ctx.progress.call_args.args[2] == out["note"]  # the feedback T3 prints

    ctx.reset_mock()
    out = node.Capability.execute(engine, "close", {"execute": True, "chair_moved": True}, ctx)
    assert out["note"] == ""  # the close itself: nothing more to do
    assert all("WHEELCHAIR" not in str(c) for c in ctx.progress.call_args_list)


def test_an_open_asking_for_a_wheelchair_move_says_so_where_it_was_sent(door_sim, monkeypatch):
    for name in ("open3d", "message_filters"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    from fridge import node

    def run(*a, **k):
        push_open.NEXT_STEP = "Pull done and the hand is off the handle. Move the WHEELCHAIR ..."
        return True

    monkeypatch.setattr(open_door, "run", run)
    monkeypatch.setattr(node, "WristCamera", MagicMock())
    engine = MagicMock(mock=False, cfg={**node.PARAMS, "allow_execute": True}, instance_id="")
    ctx = MagicMock()
    out = node.Capability.execute(engine, "open", {"execute": True}, ctx)
    assert out["note"].startswith("Pull done")
    assert ctx.progress.call_args.args[2] == out["note"]


def test_a_run_keeps_what_the_node_logged_in_its_console_log(door_sim, monkeypatch):
    for name in ("open3d", "message_filters"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    from fridge import motion, node

    scene, rb = door_sim
    rcl = MagicMock()
    logger = node.TeeLogger(rcl)
    dc.set_output(logger.info)  # as the node sets it: the planner's lines
    try:
        logger.info("before the run")  # between runs: the terminal only
        arm = _Arm(scene, rb, TELEOP_START)
        assert open_door.run(arm, False, push=False)
        logger.warning("after the run")
    finally:
        dc.set_output(None)
    (run,) = dc.RUNS_DIR.glob("*_open")
    console = (run / "console.log").read_text()
    assert "INFO run log:" in console
    assert "DRY RUN -- would grasp, then swing 90 deg" in console
    assert "before the run" not in console and "after the run" not in console
    assert (
        rcl.info.call_count > 10 and rcl.warning.call_count == 1
    )  # the terminal still gets every line
    assert {p.name for p in run.iterdir()} >= {"run.json", "arm.npz", "console.log"}

    def refuse():
        logger.error("from the arm, mid-run")
        raise dc.Refused("no plan")

    with pytest.raises(dc.Refused):
        motion.logged_run("close", _Arm(scene, rb, TELEOP_START), {}, 1.0, refuse)
    (run,) = dc.RUNS_DIR.glob("*_close")
    console = (run / "console.log").read_text()
    assert "ERROR from the arm, mid-run" in console and "ERROR stopped: no plan" in console


def test_a_run_records_its_rosbag_in_its_folder_and_closes_it(door_sim, monkeypatch):
    from fridge import motion

    made = []

    class Recorder:
        def __init__(self, run_dir):
            made.append(self)
            self.run_dir, self.closed = run_dir, False

        def close(self):
            self.closed = True
            return 7

    def refuse():
        assert made and not made[-1].closed  # recording while the run runs
        raise dc.Refused("no plan")

    monkeypatch.setattr(motion, "BAG", Recorder)
    scene, rb = door_sim
    with pytest.raises(dc.Refused):  # a refusal still closes the bag
        motion.logged_run("open", _Arm(scene, rb, TELEOP_START), {}, 1.0, refuse)
    (run,) = dc.RUNS_DIR.glob("*_open")
    (rec,) = made
    assert rec.run_dir == run and rec.closed

    def broken(run_dir):
        raise RuntimeError("no rosbag2")

    monkeypatch.setattr(motion, "BAG", broken)  # a recorder that fails: the run goes on
    assert motion.logged_run("open", _Arm(scene, rb, TELEOP_START), {}, 1.0, lambda: True)

    monkeypatch.setattr(motion, "BAG", Recorder)
    monkeypatch.setattr(motion, "BAG_MIN_FREE_GB", 1e9)  # a full disk: no bag, the run goes on
    assert motion.logged_run("open", _Arm(scene, rb, TELEOP_START), {}, 1.0, lambda: True)
    assert len(made) == 1


def test_the_close_plans_after_an_executed_open_that_saw_no_door_face(door_sim):
    # door_sim's detection, like door_square_to_arm's, has no door_mid: the open writes it null
    # (11-05 replay: the close then crashed shifting it for the wheelchair)
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    assert open_door.run(arm, True, push=False)
    assert json.loads(dc.DOOR_FILE.read_text())["door_mid"] is None
    arm._arrive(TELEOP_START)
    assert close_door.run(arm, False)


_REAL_DETECT_HANDLE = shared.detect_handle  # door_sim stubs it out


def test_two_looks_average_the_door_only_when_both_saw_it(door_sim, monkeypatch):
    # door_square_to_arm: a look whose bar has depth fits no door (door_z / door_mid None),
    # one whose bar hasn't does
    looks = iter(
        [
            dict(door_z=(0.1, 0.5), door_mid=np.array([0.8, -0.2, 0.3])),
            dict(door_z=None, door_mid=None),
        ]
    )

    class Cam:
        def wait_for_frames(self, timeout_s):
            return True

        def grab(self):
            return dict(rgb=None, K=CAM_K, depth_mm=None, base_T_cam=CAM_T, box=None)

    class Detector:
        def detect(self, *a, **k):
            return dict(
                handle=np.array([0.8, -0.3, 0.3]),
                quat=HANDLE_QUAT,
                normal=np.array([-1.0, 0.0, 0.0]),
                hinge=None,
                **next(looks),
            )

    monkeypatch.setattr(dc, "camera", lambda: Cam())
    monkeypatch.setattr(shared, "_save_frame", lambda *a: None)
    shared.configure(MAKE_DETECTOR=Detector, HANDLE_SOURCE="depth")
    _, _, _, door_z, door_mid, _ = _REAL_DETECT_HANDLE()
    assert door_z is None and door_mid is None


def test_a_sparse_door_plane_is_a_miss_not_a_crash(perception, monkeypatch):
    base, _ = perception
    # two door points 3 m apart along the door: no point near either percentile edge
    plane_pts = np.array([[-1.5, 0.0, 1.0], [1.5, 0.0, 1.0]])
    det = base.HandleDetector()
    monkeypatch.setattr(
        det,
        "_door_plane",
        lambda *a: (
            plane_pts,
            np.zeros((0, 3)),
            np.array([0.0, 0.0, -1.0]),
            1.0,
            np.array([0.0, -1.0, 0.0]),
            np.eye(3),
            np.zeros(3),
        ),
    )
    monkeypatch.setattr(det, "_handle_from_depth", lambda *a, **k: np.array([0.2, 0.0, 0.95]))
    assert det.detect(None, CAM_K, None, CAM_T) is None


def test_the_push_open_refuses_a_pull_that_leaves_nothing_to_push(door_sim):
    scene, rb = door_sim
    arm = _Arm(scene, rb, TELEOP_START)
    max_pull = constants.PUSH_OPEN_TO_DEG - constants.PUSH_OPEN_MIN_PUSH_DEG
    assert max_pull == 85
    with pytest.raises(dc.Refused, match="too little to push"):
        open_door.run(arm, False, swing_deg=max_pull + 1)
    assert arm.calls == []
    try:  # 85 itself is allowed: anything it refuses is the plan's, not this gate's
        open_door.run(arm, False, swing_deg=max_pull)
    except dc.Refused as e:
        assert "too little to push" not in str(e)


def test_a_new_executed_open_voids_a_push_still_waiting_for_the_wheelchair(door_sim, monkeypatch):
    scene, rb = door_sim
    # an earlier open asked for a wheelchair move
    dc.save_door_geometry(**FRIDGE_DOOR, push_after_chair_move=0.10)

    def refused(*a, **k):
        raise dc.Refused("NO DETECTION (1) after 8 retries")

    monkeypatch.setattr(shared, "open_task", refused)
    with pytest.raises(dc.Refused):
        open_door.run(_Arm(scene, rb, TELEOP_START), True)
    with pytest.raises(dc.Refused, match="no push waiting"):
        open_door.run(_Arm(scene, rb, TELEOP_START), True, chair_moved=True)


def _stamped(sec, nanosec=0):
    from types import SimpleNamespace

    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)))


def _bare_camera(base, sam3):
    """A WristCamera without its ROS plumbing (no cv_bridge here): its buffer and pairing only."""
    import collections
    import threading

    cam = object.__new__(base.WristCamera)
    cam._lock, cam._latest, cam._recent, cam._sam3 = (
        threading.Lock(),
        None,
        collections.deque(),
        sam3,
    )
    cam._frame = lambda color, info, depth: {
        "color": color,
        "depth": depth,
        "rgb": np.zeros((2, 3, 3)),
    }
    return cam


def test_sam3_masks_the_frame_it_segmented_not_one_grabbed_before(perception):
    from types import SimpleNamespace

    base, _ = perception
    # SAM 3 segments the first image after the request reaches it: newer than one grabbed before
    reply = SimpleNamespace(
        success=True,
        message="1 instance",
        detections=SimpleNamespace(
            header=_stamped(10, 500).header,
            objects=[SimpleNamespace(mask_id=1, score=0.8, perception_class="handle")],
            label_image=SimpleNamespace(
                height=2, width=3, step=6, data=np.array([[0, 1, 1], [0, 0, 1]], "<u2").tobytes()
            ),
        ),
    )

    class Client:
        srv_name, srv_type = "/sam3", SimpleNamespace(Request=lambda **k: k)

        def wait_for_service(self, timeout_sec):
            return True

        def call_async(self, req):
            return SimpleNamespace(add_done_callback=lambda cb: cb(None), result=lambda: reply)

    cam = _bare_camera(base, Client())
    cam.sam3_prompts, cam.color_topic, cam.sam3_threshold = ["handle"], "/c", 0.3
    for s, ns in ((10, 0), (10, 500), (11, 0)):  # the grabbed frame, SAM 3's, a later one
        cam._on_frame(_stamped(s, ns), None, f"depth {s}.{ns}")
    instances, frame = cam.sam3_detect()
    assert frame["depth"] == "depth 10.500"
    assert instances[0][1].tolist() == [[False, True, True], [False, False, True]]
    cam._recent.clear()  # SAM 3's image no longer buffered: a miss, never another frame's depth
    assert cam.sam3_detect() == ([], None)


def test_only_a_camera_with_sam3_buffers_frames(perception):
    base, _ = perception
    depth_only = _bare_camera(base, None)  # the microwave's default
    for s in range(5):
        depth_only._on_frame(_stamped(s), None, None)
    assert not depth_only._recent and depth_only._latest[0].header.stamp.sec == 4
    with_sam3 = _bare_camera(base, object())
    for s in range(10):
        with_sam3._on_frame(_stamped(s), None, None)
    assert [m[0].header.stamp.sec for m in with_sam3._recent] == [6, 7, 8, 9]  # FRAME_KEEP_S = 3 s


def test_a_look_detects_on_sam3s_own_frame(door_sim, monkeypatch):
    grabbed = dict(rgb="grabbed", K=CAM_K, depth_mm=None, base_T_cam=CAM_T, box=None)
    segmented = dict(grabbed, rgb="segmented")
    mask = np.ones((2, 2), bool)
    seen = []

    class Cam:
        def wait_for_frames(self, timeout_s):
            return True

        def grab(self):
            return dict(grabbed)

        def sam3_detect(self):
            return [(0.8, mask, "handle")], dict(segmented)

    class Detector:
        def detect(self, rgb, *a, handle_mask=None, **k):
            seen.append((rgb, handle_mask is mask))
            return dict(
                handle=np.array([0.8, -0.3, 0.3]),
                quat=HANDLE_QUAT,
                normal=np.array([-1.0, 0.0, 0.0]),
                hinge=None,
                door_z=None,
                door_mid=None,
            )

    monkeypatch.setattr(dc, "camera", lambda: Cam())
    monkeypatch.setattr(shared, "_save_frame", lambda *a: None)
    shared.configure(MAKE_DETECTOR=Detector, HANDLE_SOURCE="sam3")
    _REAL_DETECT_HANDLE()
    assert seen == [("segmented", True), ("segmented", True)]


def test_the_trajectory_check_finds_what_the_arm_would_refuse(door_sim):
    scene, rb = door_sim
    # the start pose turned about J1 (and J2) until the arm is in the wheelchair's keep-out
    turned = (
        np.array(TELEOP_START) + np.radians([d1, d2, 0, 0, 0, 0, 0])
        for d2 in (0, 30, -30)
        for d1 in range(0, 360, 15)
    )
    behind = next(q for q in turned if sim.keep_out_message(rb, q) is not None)
    assert dc.trajectory_keep_out(TELEOP_START, [behind]) is not None
    assert dc.trajectory_keep_out(TELEOP_START, [TELEOP_START]) is None
    sim.set_keep_out(None)  # the microwave: no keep-out, nothing to check
    assert dc.trajectory_keep_out(TELEOP_START, [behind]) is None


def test_the_open_plans_no_swing_or_approach_the_arm_would_refuse(door_sim, monkeypatch):
    scene, rb = door_sim
    monkeypatch.setattr(
        shared, "_swing_keep_out", lambda q0, arc: "forearm 0.4 cm into the keep-out"
    )
    plan, why = shared.plan_open(scene, rb, _detection(), TELEOP_START, 90, 90, shared.AfterSwing())
    assert plan is None
    monkeypatch.undo()
    calls = []

    def approach_refused(q0, knots, cart_lengths=None, speed_scale=1.0):
        calls.append(len(knots))
        return "wrist 0.1 cm into the keep-out"

    monkeypatch.setattr(shared, "trajectory_keep_out", approach_refused)
    plan, why = shared.plan_open(scene, rb, _detection(), TELEOP_START, 90, 90, shared.AfterSwing())
    assert plan is None and calls


def test_an_open_waiting_for_the_wheelchair_has_not_opened_the_door(door_sim, monkeypatch):
    for name in ("open3d", "message_filters"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    from fridge import node

    monkeypatch.setattr(node, "WristCamera", MagicMock())
    engine = MagicMock(
        mock=False, cfg={**node.PARAMS, "allow_execute": True}, instance_id="fridge_1"
    )

    def opened_part_way(*a, **k):
        push_open.NEXT_STEP = "Pull done ... Move the WHEELCHAIR straight back ~10 cm ..."
        return True

    monkeypatch.setattr(open_door, "run", opened_part_way)
    ctx = MagicMock()
    node.Capability.execute(engine, "open", {"execute": True}, ctx)
    ctx.update_object.assert_not_called()

    def pushed(*a, **k):
        push_open.NEXT_STEP = ""
        return True

    monkeypatch.setattr(open_door, "run", pushed)
    node.Capability.execute(engine, "open", {"execute": True, "chair_moved": True}, ctx)
    ctx.update_object.assert_called_once_with("fridge_1", opened=True)


def test_a_grasp_out_of_reach_says_there_is_no_height_limit(door_sim):
    shared.configure(MAX_REACH=0.1)
    with pytest.raises(dc.Refused) as e:
        shared.grasp_pose(np.array([0.8, -0.3, 0.3]), HANDLE_QUAT, np.array([-1.0, 0.0, 0.0]))
    assert "no height limit" in str(e.value) and "None" not in str(e.value)
