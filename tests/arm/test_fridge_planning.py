"""The fridge's door planned through door_common, dry, in the sim, and its run on a fake arm.

CI starts capabilities in mock mode only, so nothing else runs this planning code. The door is the
real recorded one of 10-24's successful open, on the arm model it was recorded with; the height
pinned below is what feeding-deployment's own planner gives for the same inputs.
"""

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
    saved = dict(vars(dc)), dict(vars(shared)), dict(vars(constants))
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
    vars(constants).update(saved[2])


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


def test_a_30_degree_mount_pulls_70_and_pushes_open_at_the_11_01_handle_height(door_sim, tmp_path):
    sim.reset_for_tests()
    scene, rb = sim.make_sim("/fixtures/urdf/flatbot.urdf")
    open_door.configure(tmp_path, 30)
    assert constants.PUSH_OPEN_DZ_M == pytest.approx(0.10)
    assert max(shared.GRASP_PITCHES_DEG) == 35
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
    plan, why = shared.plan_open(scene, rb, geom, TELEOP_START, 70, 70, push_open.PushOpen())
    assert why is None
    assert plan["swing"] == 70
    assert plan["after"] == "push-open OK"


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


def test_the_open_defaults_to_the_70_degree_pull_and_push(door_sim, monkeypatch):
    scene, rb = door_sim
    asked = {}

    def open_task(ai, execute, max_swing, min_swing, after_swing):
        asked.update(swing=(max_swing, min_swing), after=after_swing)
        return True

    monkeypatch.setattr(shared, "open_task", open_task)
    monkeypatch.setattr(open_door, "push_open", lambda ai, execute: True)
    assert open_door.run(_Arm(scene, rb, TELEOP_START), False)
    assert asked["swing"] == (70, 70)
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


def test_the_two_looks_compare_the_measured_hinge_not_a_door_edge(monkeypatch):
    # door_common.perception's imports, which these tests don't have
    for name in ("open3d", "message_filters"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    import door_common.perception
    from fridge.perception import HandleDetector

    base = door_common.perception.HandleDetector
    look = {
        "handle": np.array([0.5, -0.4, 0.3]),
        "normal": np.array([-1.0, 0.0, 0.0]),
        "hinge": np.zeros(3),
    }
    monkeypatch.setattr(base, "detect", lambda self, *a, **k: dict(look))
    det = HandleDetector(np.array([0.1, 0.37, 0.0])).detect()
    assert det["hinge"] == pytest.approx([0.6, -0.03, 0.3])

    monkeypatch.setattr(base, "detect", lambda self, *a, **k: None)
    assert HandleDetector(np.zeros(3)).detect() is None  # a miss stays a miss


def test_a_close_plan_says_where_it_was_sent_to_move_the_chair_next(monkeypatch):
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
