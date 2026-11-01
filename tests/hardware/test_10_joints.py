"""Stage 1, joint-space motions: move_joints (RammpArm's), go_to_joint_config, go_to_preset,
execute_command(JointCommand) and the driver's raw execute_joint_trajectory.

Every move starts at home and stays within a few degrees of it, except the wrist turns."""

import time

import numpy as np
import pytest
import rammp_arm
from conftest import STOP_S, joint_err

from capability_engine import ArmError
from capability_engine.arm import SUCCESSFUL
from capability_engine.cornell_commands import JointCommand

pytestmark = pytest.mark.motion

FINAL_DEG = 1.0  # move_joints: every joint this close to the last knot once still
GO_TO_DEG = rammp_arm.GO_TO_TRACK_DEG
TIMING_S = 0.5  # a move's measured duration vs its planned one: this or TIMING_FRAC, the larger
TIMING_FRAC = 0.3
KNOT_DEG = 2.0  # a move through knots passes this close to each one
SMALL = np.array([0.1, 0.0, 0.0, 0.15, 0.0, -0.15, 0.2])  # a few degrees on J1, J4, J6, J7
WRIST = np.array([0, 0, 0, 0, 0, 0, 1.0])  # a long move that only turns the wrist
WRAP = np.array([0, 0, 0.15, 0, 0, 0, 0])  # J3 starts at ~pi: its target wraps to ~-pi
MONITOR_AFTER_S = 0.6  # the monitor test stops the move this long after it starts


def finished_at(cell, target, deg):
    cell.observer.wait_still()
    err = np.degrees(joint_err(cell.observer.q(), target))
    assert err.max() < deg, f"joints off by {np.round(err, 2)} deg"


def planned_s(q0, knots, speed_scale=1.0):
    return float(rammp_arm.joint_trajectory(q0, knots, None, speed_scale)[0][-1])


def assert_duration(measured, planned):
    assert abs(measured - planned) < max(TIMING_S, TIMING_FRAC * planned), (measured, planned)


def test_go_to_joint_config(cell, home):
    target = home + SMALL
    assert cell.act(lambda arm, ctx: arm.go_to_joint_config(target))
    finished_at(cell, target, GO_TO_DEG)
    assert cell.act(lambda arm, ctx: arm.go_to_joint_config(home))
    finished_at(cell, home, GO_TO_DEG)


def test_go_to_preset(cell, home):
    assert cell.act(lambda arm, ctx: arm.move_joints([home + SMALL]))
    assert cell.act(lambda arm, ctx: arm.go_to_preset("home"))
    finished_at(cell, home, GO_TO_DEG)


def test_an_unknown_preset_is_refused(cell, home):
    t0 = time.monotonic()
    ok, last_error = cell.act(lambda arm, ctx: (arm.go_to_preset("no_such_preset"), arm.last_error))
    assert not ok and last_error
    assert cell.observer.motion_span(t0) is None


def test_move_joints(cell, home):
    target = home + SMALL
    planned = planned_s(cell.observer.q(), [target])
    t0 = time.monotonic()

    def body(arm, ctx):
        return arm.move_joints([target]), list(arm.last_path), arm.monitor_stopped

    ok, path, stopped = cell.act(body)
    assert ok and not stopped
    finished_at(cell, target, FINAL_DEG)
    first, last = cell.observer.motion_span(t0)
    assert_duration(last - first, planned)
    assert path and np.all(np.diff([p[0] for p in path]) >= 0)
    assert np.linalg.norm(np.array(path[-1][1:]) - cell.observer.tool()[0]) < 0.005


def test_move_joints_takes_a_wrapped_joint_the_short_way(cell, home):
    """Knots may be given wrapped to +-pi (get_state()'s joints): J3 at ~pi going +0.15 is given
    as ~-3.0 and must turn 0.15 rad, not spin the other way round."""
    target = home + WRAP
    wrapped = rammp_arm._wrap(target)
    assert wrapped[2] < 0  # the test means something only if J3's target does wrap
    t0 = time.monotonic()
    assert cell.act(lambda arm, ctx: arm.move_joints([wrapped]))
    finished_at(cell, target, FINAL_DEG)
    _, q, _ = cell.observer.joints_since(t0)
    assert joint_err(q, home)[:, 2].max() < WRAP[2] + 0.05  # the driver wraps J3 too


def test_move_joints_through_knots(cell, home):
    # No knot reverses a joint: min-jerk timing does not stop at knots, so a reversal jolts.
    knots = [home + SMALL * [1, 0, 0, 1, 0, 0, 0], home + SMALL * [1, 0, 0, 1, 0, 1, 0]]
    knots.append(home + SMALL)
    t0 = time.monotonic()
    assert cell.act(lambda arm, ctx: arm.move_joints(knots))
    finished_at(cell, knots[-1], FINAL_DEG)
    _, q, _ = cell.observer.joints_since(t0)
    for knot in knots[:-1]:
        closest = min(joint_err(row, knot).max() for row in q)
        assert np.degrees(closest) < KNOT_DEG, f"passed {np.degrees(closest):.2f} deg from a knot"


def test_move_joints_speed_scale(cell, home):
    durations = {}
    for scale, target in ((1.0, home + WRIST), (0.5, home)):
        t0 = time.monotonic()
        assert cell.act(lambda arm, ctx, s=scale, q=target: arm.move_joints([q], speed_scale=s))
        cell.observer.wait_still()
        first, last = cell.observer.motion_span(t0)
        durations[scale] = last - first
    assert 1.6 < durations[0.5] / durations[1.0] < 2.4, durations


def test_move_joints_monitor_stops_it(cell, home):
    target = home + WRIST

    def body(arm, ctx):
        t0 = time.monotonic()
        ok = arm.move_joints([target], monitor=lambda: time.monotonic() - t0 > MONITOR_AFTER_S)
        return ok, arm.monitor_stopped, time.monotonic()

    ok, stopped, returned = cell.act(body)
    assert ok and stopped
    rest = cell.observer.wait_still()
    assert rest - returned < STOP_S
    assert np.degrees(joint_err(cell.observer.q(), target).max()) > 10  # stopped well short


def test_a_monitor_that_raises_stops_the_arm_and_propagates(cell, home):
    target = home + WRIST

    def body(arm, ctx):
        t0 = time.monotonic()

        def monitor():
            if time.monotonic() - t0 > MONITOR_AFTER_S:
                raise ValueError("the student's monitor failed")
            return False

        try:
            arm.move_joints([target], monitor=monitor)
        except ValueError:
            return time.monotonic()
        raise AssertionError("the monitor's exception did not reach the caller")

    raised = cell.act(body)
    assert cell.observer.wait_still() - raised < STOP_S
    assert np.degrees(joint_err(cell.observer.q(), target).max()) > 10


def test_move_joints_with_impedance(cell, home):
    target = home + SMALL
    assert cell.act(lambda arm, ctx: arm.move_joints([target], impedance=rammp_arm.IMPEDANCE))
    finished_at(cell, target, 3 * FINAL_DEG)  # compliant: it may sag a little


def test_execute_command_joint(cell, home):
    assert cell.act(lambda arm, ctx: arm.execute_command(JointCommand(pos=list(home + SMALL))))
    finished_at(cell, home + SMALL, FINAL_DEG)


def test_execute_joint_trajectory(cell, home):
    target = home + SMALL
    feedback = []

    def body(arm, ctx):
        t, q, qd = rammp_arm.joint_trajectory(arm.raw_joints(), [target])
        points = list(zip(t, q, qd, strict=True))
        return arm.execute_joint_trajectory(rammp_arm.JOINTS, points, feedback=feedback.append)

    res = cell.act(body)
    assert res.error_code == SUCCESSFUL, (res.error_code, res.error_string)
    assert np.abs(res.final_error).max() < np.radians(FINAL_DEG)
    finished_at(cell, target, FINAL_DEG)
    assert feedback, "no feedback"
    assert feedback[-1].actual.shape == (7,)


def test_execute_joint_trajectory_refusal_raises(cell, home):
    """The driver API raises where RammpArm's methods return False, and nothing moves."""
    t0 = time.monotonic()

    def body(arm, ctx):
        names = [f"joint_{i}" for i in range(1, 8)]  # the driver takes only arm_joint_*
        t, q, qd = rammp_arm.joint_trajectory(arm.raw_joints(), [home + SMALL])
        return arm.execute_joint_trajectory(names, list(zip(t, q, qd, strict=True)))

    with pytest.raises(ArmError):
        cell.act(body)
    assert cell.observer.motion_span(t0) is None


def test_the_next_action_starts_from_where_the_arm_is(cell, home):
    assert cell.act(lambda arm, ctx: arm.move_joints([home + SMALL]))
    cell.observer.wait_still()
    raw = cell.act(lambda arm, ctx: arm.raw_joints())
    assert np.degrees(joint_err(raw, cell.observer.q()).max()) < 0.1
