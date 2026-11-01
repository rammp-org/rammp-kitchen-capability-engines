"""Stage 5, streaming: stream() on each kind of controller, its hold on exit, the driver closing
an idle session, refusals, and set_gains.

Each stream moves the arm a little (a few cm or a few degrees) and is left at a known rest."""

import time

import numpy as np
import pytest
from conftest import STOP_S, WAIT_S, joint_err, wait_until

from capability_engine import ArmError, ModeConflict, NotInAction, StreamClosed

pytestmark = pytest.mark.motion

RATE_S = 0.01  # send period: 100 Hz
RUN_S = 1.0  # each streamed move lasts this long
WRIST_RAD_S = 0.15  # joint velocity on J7
TOOL_M_S = 0.02  # twist: up
J7_STEP = 0.1  # joint position ramp on J7
RISE_M = 0.02  # pose ramp: up
RATE_TOL = 0.35  # a streamed move covers its commanded distance within this fraction
DRIFT_M = 0.005  # the tool's sideways drift during a straight streamed move
HOLD_DEG = 1.0  # an impedance hold drifts no more than this
IDLE_S = 0.5  # longer than the session's timeout_s: the driver closes it


def controller(arm, channel, impedance=False):
    """The name of an available controller that reads `channel` (impedance or not)."""
    for c in arm.list_controllers():
        if c.available and c.channels == [channel] and ("impedance" in c.name) == impedance:
            return c.name
    pytest.skip(f"no available {'impedance ' if impedance else ''}controller reads {channel}")


def stream_for(arm, name, setpoint_at, run_s=RUN_S, **kwargs):
    """Stream setpoint_at(t) on controller `name` for run_s at 100 Hz, then leave the block."""
    with arm.stream(name, **kwargs) as s:
        t0 = time.monotonic()
        while (t := time.monotonic() - t0) < run_s:
            s.send(setpoint_at(t))
            time.sleep(RATE_S)


def closed_and_still(cell, left_at):
    """After the block: the session is closed and the arm still within STOP_S."""
    wait_until(lambda: cell.observer.stream and not cell.observer.stream.open, WAIT_S, "closed")
    assert cell.observer.wait_still() - left_at < STOP_S


def test_joint_velocity(cell, home):
    def body(arm, ctx):
        name = controller(arm, "joint_velocity")
        v = np.zeros(7)
        v[6] = WRIST_RAD_S
        stream_for(arm, name, lambda t: v)
        return time.monotonic()

    left = cell.act(body)
    closed_and_still(cell, left)
    moved = cell.observer.q()[6] - home[6]
    assert abs(moved - WRIST_RAD_S * RUN_S) < RATE_TOL * WRIST_RAD_S * RUN_S, moved


def test_twist(cell, home):
    p0, _ = cell.observer.tool()

    def body(arm, ctx):
        twist = np.array([0.0, 0.0, TOOL_M_S, 0.0, 0.0, 0.0])
        stream_for(arm, controller(arm, "twist"), lambda t: twist)
        return time.monotonic()

    left = cell.act(body)
    closed_and_still(cell, left)
    d = cell.observer.tool()[0] - p0
    assert abs(d[2] - TOOL_M_S * RUN_S) < RATE_TOL * TOOL_M_S * RUN_S, d
    assert np.linalg.norm(d[:2]) < DRIFT_M, d


@pytest.mark.parametrize("impedance", [False, True], ids=["position", "impedance"])
def test_joint_position(cell, home, impedance):
    def body(arm, ctx):
        q0 = arm.raw_joints()
        step = np.zeros(7)
        step[6] = J7_STEP
        name = controller(arm, "joint_position", impedance)
        gains = "medium" if impedance else None
        stream_for(arm, name, lambda t: q0 + step * min(t / RUN_S, 1.0), RUN_S + 0.3, gains=gains)
        return time.monotonic()

    left = cell.act(body)
    closed_and_still(cell, left)
    target = home + np.eye(7)[6] * J7_STEP
    assert np.degrees(joint_err(cell.observer.q(), target).max()) < 2 * HOLD_DEG


@pytest.mark.parametrize("impedance", [False, True], ids=["position", "impedance"])
def test_pose(cell, home, impedance):
    p0, _ = cell.observer.tool()

    def body(arm, ctx):
        name = controller(arm, "pose", impedance)
        up = np.array([0.0, 0.0, RISE_M])
        ee = arm.ee_state()  # pose setpoints are the flange's (/ee_state's frame), not the tool's
        stream_for(
            arm,
            name,
            lambda t: (ee.position + up * min(t / RUN_S, 1.0), ee.orientation),
            RUN_S + 0.3,
        )
        return time.monotonic()

    left = cell.act(body)
    closed_and_still(cell, left)
    d = cell.observer.tool()[0] - p0
    assert abs(d[2] - RISE_M) < RATE_TOL * RISE_M and np.linalg.norm(d[:2]) < DRIFT_M, d


def test_an_impedance_hold_holds(cell, home):
    def body(arm, ctx):
        q0 = arm.raw_joints()
        stream_for(arm, controller(arm, "joint_position", True), lambda t: q0, gains="medium")

    cell.act(body)
    cell.observer.wait_still()
    assert np.degrees(joint_err(cell.observer.q(), home).max()) < HOLD_DEG


def test_an_idle_session_is_closed_by_the_driver(cell, home):
    """Stop sending for longer than timeout_s: the driver closes the session, the arm holds,
    and the next send() raises StreamClosed."""

    def body(arm, ctx):
        name = controller(arm, "joint_position")
        with arm.stream(name, timeout_s=0.1) as s:
            s.send(arm.raw_joints())
            time.sleep(IDLE_S)
            with pytest.raises(StreamClosed):
                s.send(arm.raw_joints())

    cell.act(body)
    wait_until(lambda: not cell.observer.stream.open, WAIT_S, "the stream to close")
    cell.observer.wait_still()
    assert np.degrees(joint_err(cell.observer.q(), home).max()) < HOLD_DEG


def test_a_bad_setpoint_raises_and_closes(cell, home):
    def body(arm, ctx):
        with arm.stream(controller(arm, "joint_velocity")) as s:
            s.send([0.0, 0.0, 0.0])

    with pytest.raises(ArmError):
        cell.act(body)
    wait_until(lambda: not cell.observer.stream.open, WAIT_S, "the stream to close")
    assert cell.observer.still_now()


def test_a_second_stream_is_a_mode_conflict(cell, home):
    """One motion at a time: a second stream, a trajectory or a planned move inside an open
    stream raises ModeConflict and sends nothing; the first stream is still open after."""
    t0 = time.monotonic()

    def body(arm, ctx):
        name = controller(arm, "joint_velocity")
        q = arm.raw_joints()
        with arm.stream(name, timeout_s=1.0) as s:
            s.send(np.zeros(7))
            with pytest.raises(ModeConflict), arm.stream(name):
                pass
            with pytest.raises(ModeConflict):
                arm.move_joints([q + np.eye(7)[6] * 0.2])
            with pytest.raises(ModeConflict):
                arm.go_to_joint_config(q + np.eye(7)[6] * 0.2)
            s.send(np.zeros(7))  # the first one is still open

    cell.act(body)
    wait_until(lambda: not cell.observer.stream.open, WAIT_S, "the stream to close")
    assert cell.observer.motion_span(t0) is None


def test_send_after_the_block_raises(cell, home):
    def body(arm, ctx):
        with arm.stream(controller(arm, "joint_velocity")) as s:
            s.send(np.zeros(7))
        with pytest.raises(NotInAction):
            s.send(np.zeros(7))

    cell.act(body)


def test_set_gains(cell):
    accepted, message = cell.act(lambda arm, ctx: arm.set_gains("medium"))
    assert accepted, message
