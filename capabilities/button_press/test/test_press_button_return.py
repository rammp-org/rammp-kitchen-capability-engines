"""Leaving the panel: back straight out before any cuRobo move, and never cuRobo right at the panel.
Going home (Run.go_home): to the start JOINTS, after the run's own failures too; a driver failure or a
cancel holds. A cuRobo move refused at a joint limit turns that joint off it and tries once more.

2026-10-14: the return to start handed the whole move to cuRobo with the fingertips 2 cm from the
button; cuRobo's world has no microwave, and its path swept the gripper into it. Runs Run's moves
against a stand-in arm that records what would be sent (no ROS, nothing moves).
"""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

pb = pytest.importorskip("button_press.press_button")  # needs rclpy + rammp interfaces
from capability_engine import StreamClosed  # noqa: E402

from button_press import Abort, MotionFailed  # noqa: E402

N_OUT = np.array([-0.993, 0.104, 0.053]) / np.linalg.norm([-0.993, 0.104, 0.053])  # out of the panel, toward the arm
SPOT = np.array([0.675, -0.317, 0.077])  # the pre-press spot
HOME = np.array([0.577, 0.001, 0.433])
BUTTON = SPOT - 0.06 * N_OUT  # the pre-press spot is 6 cm out of the panel
Q_HOME = np.array([0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57])  # the start joints
LIMIT_REFUSAL = (
    "back to start: go_to_joint_config failed: RammpArm: back to start ended PLANNING_FAILED START_IN_LIMIT_MARGIN: "
    "start: arm_joint_6 is 0.000 rad from its position limit [-2.090, 2.090], inside joint_limit_margin_rad 0.100"
)


class FakeArm:
    def __init__(self, p):
        self.p, self.R = np.asarray(p, float), np.eye(3)
        self.q = Q_HOME + 0.5  # away from the start
        self.refuse = []  # go_to_joints raises these first, one per call

    def ee_pose(self):
        return self.p, self.R

    def joints(self):
        return self.q


def run_at(p, execute=True, fail=()):
    """A Run whose straight moves and cuRobo moves are recorded; straight legs named in `fail` abort."""
    run = pb.Run.__new__(pb.Run)
    run.args = argparse.Namespace(execute=execute, presses=1, push_mm=1.0, stop_nm=4.0)
    run.arm = FakeArm(p)
    run.sent, run.speeds = [], {}

    def move(path, name, track_abort_m=pb.TRACK_ABORT_M, cart_speed=pb.CART_SPEED, collide=True):
        if not execute:
            return
        run.sent.append(("straight", name, path))
        run.speeds[name] = cart_speed
        run.collide[name] = collide
        if name in fail:
            run.arm.p = np.asarray(path[0][0], float)  # stopped partway along the leg
            raise Abort(f"{name}: ended 1.2 cm from the target -- HOLDING HERE")
        run.arm.p = np.asarray(path[-1][0], float)

    def long_move(p, Rm, name):
        run.sent.append(("curobo", name, p))
        run.arm.p = np.asarray(p, float)

    def go_to_joints(q, name, log):
        if run.arm.refuse:
            raise MotionFailed(run.arm.refuse.pop(0))
        run.sent.append(("curobo joints", name, q))
        run.arm.p, run.arm.q = HOME, np.asarray(q, float)

    def turn_off_limit(error, name, log):
        if "START_IN_LIMIT_MARGIN" not in error:
            return False
        run.sent.append(("turn off limit", name, None))
        return True

    run.arm.go_to_joints, run.arm.turn_off_limit = go_to_joints, turn_off_limit
    run.q0, run.panel, run.world = Q_HOME, (BUTTON, N_OUT), SimpleNamespace(loaded=[{"name": "microwave"}])
    run.move, run.long_move, run.log = move, long_move, lambda rec: None
    run.collide = {}  # leg name -> whether run.move was asked to clearance-check it
    run.check_leg = lambda name, q, knots: (1.0, "")  # collision.py has its own tests
    run.reanchor_seed = lambda why: run.sent.append(("reseed", why, None))
    return run


def test_go_home_backs_straight_out_first_then_goes_to_the_start_joints():
    run = run_at(SPOT)
    run.go_home()
    kinds = [(k, n) for k, n, _ in run.sent]
    assert kinds == [("straight", "back out from the panel"), ("curobo joints", "back to start")]
    path = run.sent[0][2]
    end = path[-1][0]
    assert np.allclose(end, SPOT + pb.RETREAT_OUT_M * N_OUT)  # straight out along the normal
    assert all(np.allclose(Rm, np.eye(3)) for _, Rm in path)  # without turning the wrist
    assert all(abs(float((p - SPOT) @ N_OUT) - np.linalg.norm(p - SPOT)) < 1e-9 for p, _ in path)
    assert np.allclose(run.sent[1][2], Q_HOME)  # the start JOINTS, not just the start pose


def test_the_retreat_clears_the_panel_like_staging():
    assert pb.RETREAT_OUT_M >= 0.10  # cuRobo starts well clear of it


def test_go_home_far_from_the_panel_goes_straight_to_curobo():
    run = run_at(BUTTON + 0.30 * N_OUT)
    run.go_home("move refused")
    assert [(k, n) for k, n, _ in run.sent] == [("curobo joints", "back to start")]


def test_go_home_at_the_start_does_nothing():
    run = run_at(HOME)
    run.arm.q = Q_HOME + np.radians(0.5)
    run.go_home()
    assert run.sent == []


def test_go_home_turns_a_joint_off_its_limit_and_tries_again():
    run = run_at(BUTTON + 0.30 * N_OUT)
    run.arm.refuse = [LIMIT_REFUSAL]
    run.go_home()
    assert [(k, n) for k, n, _ in run.sent] == [("turn off limit", "back to start"), ("curobo joints", "back to start")]


def test_go_home_holds_on_any_other_refusal():
    run = run_at(BUTTON + 0.30 * N_OUT)
    run.arm.refuse = ["back to start: go_to_joint_config failed: INVALID_GOAL supervisor rejected planned trajectory"]
    with pytest.raises(MotionFailed, match="supervisor"):
        run.go_home()
    assert run.sent == []


def _failing_run(error, monkeypatch):
    run = run_at(SPOT)

    def press(run_, a, pub):
        raise error

    monkeypatch.setattr(pb, "_press_button", press)
    return run


@pytest.mark.parametrize(
    "error",
    [
        Abort("dome detector: only 4/7 frames agree on the start_30s pixel"),
        Abort("press 1 (contact): no touch within 30 mm"),
        MotionFailed("to staging: go_to_ee_pose failed: RammpArm: to staging ended PLANNING_FAILED START_IN_COLLISION"),
    ],
)
def test_a_run_that_fails_its_own_checks_goes_home(error, monkeypatch):
    run = _failing_run(error, monkeypatch)
    with pytest.raises(type(error)):
        pb.press_button(run, run.args, None)
    assert [k for k, _, _ in run.sent] == ["straight", "curobo joints"]


@pytest.mark.parametrize(
    "error",
    [
        MotionFailed("to pre-press: trajectory ended PATH_TOLERANCE_VIOLATED -- HOLDING HERE"),
        MotionFailed("back to start: go_to_joint_config failed: E-stop engaged"),
        RuntimeError("Cancelled"),  # a Stop or cancel, ArmError: not an Abort
    ],
)
def test_a_driver_failure_or_a_cancel_holds(error, monkeypatch):
    run = _failing_run(error, monkeypatch)
    with pytest.raises(type(error)):
        pb.press_button(run, run.args, None)
    assert run.sent == []


def test_a_failed_way_home_keeps_the_runs_own_failure(monkeypatch):
    run = _failing_run(Abort("press 1 (contact): no touch within 30 mm"), monkeypatch)
    run.arm.refuse = ["back to start: go_to_joint_config failed: E-stop engaged"]
    with pytest.raises(Abort, match="no touch"):
        pb.press_button(run, run.args, None)


def test_turn_off_limit_turns_only_that_joint_clear_of_the_margin():
    from button_press import press_arm as pa

    sent = []
    q0 = np.array([1.28, 0.42, -3.01, -2.09, -1.94, -2.0997, -2.21])
    pa_arm = pa.PressArm.__new__(pa.PressArm)
    pa_arm.execute, pa_arm.sent_goals = True, 0
    pa_arm.raw_joints = lambda: q0
    pa_arm._wait_still = lambda: HOME
    pa_arm.arm = SimpleNamespace(
        execute_joint_trajectory=lambda names, points, **k: sent.append((names, points))
        or SimpleNamespace(error_code=0, error_string="")
    )
    assert pa_arm.turn_off_limit(LIMIT_REFUSAL, "back to start", lambda rec: None)
    (names, points) = sent[0]
    assert list(names) == list(pa.rammp_arm.JOINTS)
    q_end = np.asarray(points[-1][1])
    assert np.allclose(points[0][1], q0) and np.allclose(points[0][2], 0) and np.allclose(points[-1][2], 0)
    assert np.allclose(np.delete(q_end, 5), np.delete(q0, 5))  # only joint 6
    assert abs(q_end[5] - (-2.090 + 0.100 + pa.LIMIT_CLEAR_RAD)) < 1e-9
    assert not pa_arm.turn_off_limit("INVALID_GOAL supervisor rejected planned trajectory", "x", lambda rec: None)
    assert len(sent) == 1


def test_an_aborted_press_returns_to_the_spot_in_a_straight_line():
    run = run_at(SPOT)

    def press_contact(p_spot, n_out, name, log, push, stop_nm):
        run.arm.p = SPOT - 0.008 * N_OUT  # stopped 8 mm in
        raise Abort(f"{name}: no touch within 30 mm -- session closed, HOLDING HERE")

    run.arm.press_contact = press_contact
    with pytest.raises(Abort, match="press 1"):
        run.press(SPOT, np.eye(3), N_OUT)
    assert [k for k, _, _ in run.sent if k != "reseed"] == ["straight"]  # back to the spot; never cuRobo at the panel
    assert run.sent[-2][0] == "reseed"  # the leg back is checked from the joints the press left
    assert np.allclose(run.sent[-1][2][-1][0], SPOT)


@pytest.mark.parametrize("error", [MotionFailed("the arm ended off the line"), StreamClosed("closed by the driver")])
def test_a_press_the_driver_failed_holds_where_it_is(error):
    """REVIEW_RULES C5: a motion the driver failed (an e-stop, control lost, the stream closed) ends
    the run; nothing moves the arm again."""
    run = run_at(SPOT)

    def press_contact(p_spot, n_out, name, log, push, stop_nm):
        run.arm.p = SPOT - 0.004 * N_OUT
        raise error

    run.arm.press_contact = press_contact
    with pytest.raises(type(error)):
        run.press(SPOT, np.eye(3), N_OUT)
    assert run.sent == []


def test_a_dry_run_leaves_without_moving():
    run = run_at(SPOT, execute=False)
    run.retreat(N_OUT)
    assert run.sent == []


# ---- the way in: cuRobo only to a lead-in point, then straight in (2026-10-15: cuRobo's path to the
# staging pose hit the top of the microwave on the raised counter)


def test_staging_goes_to_the_lead_in_point_then_straight_in():
    p_st = SPOT + 0.15 * N_OUT
    run = run_at(HOME)
    pb.go_to_staging(run, p_st, np.eye(3), N_OUT, 0.20)
    assert [(k, n) for k, n, _ in run.sent] == [
        ("curobo", "to the lead-in point"),
        ("reseed", "at the lead-in point"),
        ("straight", "straight in to staging"),
    ]  # checked from cuRobo's end
    assert np.allclose(run.sent[0][2], p_st + 0.20 * N_OUT)  # 20 cm further out than staging
    path = run.sent[2][2]
    assert np.allclose(path[-1][0], p_st)
    assert all(abs(float((p - p_st) @ N_OUT) - np.linalg.norm(p - p_st)) < 1e-9 for p, _ in path)  # on the normal
    assert run.speeds["straight in to staging"] == pb.LEAD_IN_SPEED_MPS > pb.CART_SPEED  # the quick leg


def test_lead_in_zero_is_one_curobo_move():
    run = run_at(HOME)
    pb.go_to_staging(run, SPOT + 0.15 * N_OUT, np.eye(3), N_OUT, 0.0)
    assert [(k, n) for k, n, _ in run.sent] == [("curobo", "to staging")]


def test_lead_in_is_off_by_default():
    assert pb.Args().lead_in == pb.STAGE_LEAD_IN_M == 0.0


class LeadArm:
    """IK reaches a lead-in point only up to `reach` m out from staging (2026-10-15: 10 cm put the
    point beside the arm's base, 7 mm short)."""

    def __init__(self, p_st, reach):
        self.p_st, self.reach = p_st, reach
        self.kin = self

    def joints(self):
        return np.zeros(7)

    def ee_pose(self):
        return HOME, np.eye(3)

    def solve_translation(self, q, d, posture=None, rot=None):
        out = float((HOME + d - self.p_st) @ N_OUT)
        return np.zeros(7), (0.0 if out <= self.reach + 1e-9 else 0.007), None, 0.0, []

    def precheck_poses(self, *a, **k):
        return 0.0, 0.0, None, []


@pytest.mark.parametrize("reach,used", [(0.20, 0.10), (0.06, 0.05), (0.02, 0.0)])
def test_the_lead_in_shrinks_to_what_the_arm_can_reach(reach, used):
    p_st = SPOT + 0.27 * N_OUT
    run = run_at(HOME)
    run.arm, run.seed_posture = LeadArm(p_st, reach), None
    lead, p_lead, leg = pb.pick_lead_in(run, p_st, np.eye(3), N_OUT, 0.10)
    assert abs(lead - used) < 1e-9 and np.allclose(p_lead, p_st + used * N_OUT)
    assert (leg is None) == (used == 0.0)


def test_preflight_waits_for_the_wrist_colour_image():
    """Its subscriptions are the action's, made just before preflight: with camera_frame "" the tf check
    needs the image's frame_id, so a late first frame must be waited for, not fail the run."""
    waited = []

    class Per:
        def wait(self, name, timeout, max_age=None):
            waited.append(name)
            return object()

        def cam_rotation_in_base(self):
            assert "color" in waited  # the frame comes from the image
            return np.eye(3)

    class Arm:
        def state(self):
            return {"gripper_pos": 1.0, "ee_pos": np.r_[HOME, 0, 0, 0, 1], "position": np.zeros(7)}

        def status(self):
            return [], []

        def gripper_present(self):
            return True

        def joints(self):
            return np.zeros(7)

    run = pb.Run.__new__(pb.Run)
    run.args, run.arm, run.per, run.log = pb.Args(camera_frame=""), Arm(), Per(), lambda rec: None
    run.preflight()
    assert waited[:2] == ["info", "color"]


def test_only_the_legs_toward_the_panel_are_clearance_checked():
    """Straight out along the normal and back from the button: not checked (they touch the panel or only move
    away from it); the straight lead-in to staging: checked."""
    run = run_at(SPOT)
    run.go_home()
    assert run.collide == {"back out from the panel": False}
    run = run_at(HOME)
    pb.go_to_staging(run, SPOT + 0.15 * N_OUT, np.eye(3), N_OUT, 0.20)
    assert run.collide == {"straight in to staging": True}


def test_a_long_move_is_refused_until_curobo_has_the_microwave():
    run = pb.Run.__new__(pb.Run)
    run.args = argparse.Namespace(execute=True)
    run.world = SimpleNamespace(loaded=None)
    sent = []
    run.arm = SimpleNamespace(go_to_ee_pose=lambda *a, **k: sent.append(a))
    run.log = lambda rec: None
    with pytest.raises(Abort, match="no microwave"):
        run.long_move(HOME, np.eye(3), "to staging")
    assert sent == []
    run.world.loaded = [{"name": "microwave"}]
    run.long_move(HOME, np.eye(3), "to staging")
    assert len(sent) == 1


# ---- the shared start pose (Run.go_to_start, start_joints_deg)
START_DEG = (21.4, -43.8, -92.2, -110.6, -58.8, -93.8, -165.7)


def start_run(execute=True, start=START_DEG):
    run = run_at(HOME, execute=execute)
    run.args.start_joints_deg = start
    run.q0 = run.arm.q.copy()  # preflight's: where the arm happened to be
    return run


def test_go_to_start_moves_to_the_start_joints_and_makes_them_home():
    run = start_run()
    run.go_to_start()
    assert [(k, n) for k, n, _ in run.sent] == [("curobo joints", "to the start")]
    assert np.allclose(run.sent[0][2], np.radians(START_DEG))
    assert np.allclose(run.q0, np.radians(START_DEG))  # go_home now ends the run there
    assert np.allclose(run.seed_posture, np.radians(START_DEG))


def test_go_to_start_already_there_moves_nothing():
    run = start_run()
    run.arm.q = np.radians(START_DEG) + np.radians(1.0)
    run.go_to_start()
    assert run.sent == []
    assert np.allclose(run.q0, np.radians(START_DEG))


def test_go_to_start_unset_keeps_the_arm_and_its_start():
    run = start_run(start=())
    q_before = run.q0.copy()
    run.go_to_start()
    assert run.sent == [] and np.allclose(run.q0, q_before)


def test_go_to_start_in_a_dry_run_away_from_it_refuses():
    run = start_run(execute=False)
    with pytest.raises(Abort, match="start_joints_deg"):
        run.go_to_start()
    assert run.sent == []


def test_go_to_start_refused_by_cuRobo_holds_and_keeps_the_old_start():
    run = start_run()
    q_before = run.q0.copy()
    run.arm.refuse = ["to the start: go_to_joint_config failed: PLANNING_FAILED START_STATE_IN_COLLISION"]
    with pytest.raises(MotionFailed):
        run.go_to_start()
    assert run.sent == [] and np.allclose(run.q0, q_before)  # go_home after it: already home, nothing moves


def test_go_to_start_turns_a_joint_off_its_limit_and_tries_again():
    run = start_run()
    run.arm.refuse = [LIMIT_REFUSAL]
    run.go_to_start()
    assert [k for k, _, _ in run.sent] == ["turn off limit", "curobo joints"]


def test_go_home_after_go_to_start_returns_to_the_start_joints():
    run = start_run()
    run.go_to_start()
    run.arm.p, run.arm.q = SPOT, np.radians(START_DEG) + 0.4  # pressed, at the panel
    run.sent.clear()
    run.go_home()
    assert [(k, n) for k, n, _ in run.sent] == [
        ("straight", "back out from the panel"),
        ("curobo joints", "back to start"),
    ]
    assert np.allclose(run.sent[1][2], np.radians(START_DEG))
