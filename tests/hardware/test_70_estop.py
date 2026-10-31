"""Stage 7, the software e-stop (/estop, published here as the operator or orchestrator would;
a capability never publishes it).

Students are told: RammpArm's methods return False with the reason in last_error, an e-stop
included; the driver API (execute_joint_trajectory, stream, send) raises EStopped. Every test
clears the e-stop it engaged and checks the arm moves again."""

import time

import numpy as np
import pytest
import rammp_arm
from conftest import STOP_S, WAIT_S, diagnostic, wait_until

from capability_engine import EStopped, StreamClosed
from capability_engine.driver_state import ESTOPPED_REASON

pytestmark = pytest.mark.motion

SMALL = np.array([0.1, 0.0, 0.0, 0.15, 0.0, -0.15, 0.2])
LONG = np.array([0, 0, 0, 0, 0, 0, 1.2])
SLOW = 0.3
UP = np.array([0.0, 0.0, 0.05])
INTERRUPT_AFTER_S = 0.5
DIAGNOSTICS_S = 1.5
RUNAWAY_S = 5.0


def says_estop(message):
    """not_ready()'s reason ("arm is E-STOPPED") or the driver's ("halted: emergency stop")."""
    return "e-stop" in message.lower() or "emergency stop" in message.lower()


def recover(cell, home):
    """Clear the e-stop (and a fault it latched), then check the arm moves again."""
    cell.observer.set_estop(False)
    time.sleep(DIAGNOSTICS_S)  # /diagnostics is 1 Hz: let it show a latched fault

    def link(arm, ctx):
        d = arm.diagnostics()
        return (diagnostic(d, "Arm link") or diagnostic(d, "Arm")).level

    if cell.act(link) != 0:
        cell.observer.clear_faults()
    assert cell.act(lambda arm, ctx: arm.go_to_joint_config(home)), "the arm did not recover"


def test_an_estopped_arm_refuses_every_command(cell, home):
    t0 = time.monotonic()
    try:
        cell.observer.set_estop(True)

        def body(arm, ctx):
            out = {"not_ready": arm.not_ready()}
            ee = arm.get_state()["ee_pos"]
            for name, call in {
                "move_joints": lambda: arm.move_joints([home + SMALL]),
                "move_cartesian": lambda: arm.move_cartesian([(ee[:3] + UP, ee[3:])]),
                "go_to_joint_config": lambda: arm.go_to_joint_config(home + SMALL),
                "go_to_ee_pose": lambda: arm.go_to_ee_pose(ee[:3] + UP, ee[3:]),
                "gripper": lambda: arm.gripper(1.0),
            }.items():
                arm.last_error = ""
                out[name] = (call(), arm.last_error)
            t, q, qd = rammp_arm.joint_trajectory(arm.raw_joints(), [home + SMALL])
            for name, call in {
                "execute_joint_trajectory": lambda: arm.execute_joint_trajectory(
                    rammp_arm.JOINTS, list(zip(t, q, qd, strict=True))
                ),
                "stream": lambda: arm.stream("joint_position").__enter__(),
            }.items():
                try:
                    call()
                    out[name] = None
                except EStopped as error:
                    out[name] = error
            return out

        out = cell.act(body)
    finally:
        motion = cell.observer.motion_span(t0)
        recover(cell, home)
    assert motion is None, "the arm moved while e-stopped"
    assert out["not_ready"] == ESTOPPED_REASON
    for name in ("move_joints", "move_cartesian", "go_to_joint_config", "go_to_ee_pose", "gripper"):
        ok, last_error = out[name]
        assert ok is False and says_estop(last_error), (name, last_error)
    for name in ("execute_joint_trajectory", "stream"):
        assert isinstance(out[name], EStopped), (name, out[name])


def mid_motion(home):
    def move_joints(arm, ctx):
        return arm.move_joints([home + LONG], speed_scale=SLOW), arm.last_error

    def go_to_joint_config(arm, ctx):
        return arm.go_to_joint_config(home + LONG, speed_scale=0.1), arm.last_error

    def move_cartesian(arm, ctx):
        ee = arm.get_state()["ee_pos"]
        return arm.move_cartesian([(ee[:3] + UP, ee[3:])], speed_scale=SLOW), arm.last_error

    def gripper(arm, ctx):
        return arm.gripper(1.0, speed=0.1), arm.last_error

    return {f.__name__: f for f in (move_joints, go_to_joint_config, move_cartesian, gripper)}


@pytest.mark.parametrize("motion", list(mid_motion(None)))
def test_an_estop_mid_motion_returns_false(cell, home, motion):
    if motion == "gripper":
        assert cell.act(lambda arm, ctx: arm.gripper(0.0))
    g0 = cell.observer.gripper()
    handle = cell.start(mid_motion(home)[motion])
    if motion == "gripper":  # mid-motion: once the fingers move
        wait_until(lambda: abs(cell.observer.gripper() - g0) > 0.02, WAIT_S, "the fingers")
    else:
        cell.observer.wait_moving(timeout_s=10.0)
        time.sleep(INTERRUPT_AFTER_S)
    t_estop = time.monotonic()
    try:
        cell.observer.set_estop(True)
        _, outcome = cell.finish(handle)
        returned = time.monotonic() - t_estop
        rest = cell.observer.wait_still() - t_estop
    finally:
        recover(cell, home)
    assert outcome.raised is None, f"it raised {outcome.raised!r}: RammpArm returns False"
    ok, last_error = outcome.value
    assert ok is False and says_estop(last_error), last_error
    assert returned < WAIT_S and rest < STOP_S, (returned, rest)


def test_an_estop_mid_stream_raises(cell, home):
    v = np.zeros(7)
    v[6] = 0.2

    def body(arm, ctx):
        name = next(c.name for c in arm.list_controllers() if c.channels == ["joint_velocity"])
        with arm.stream(name) as s:
            t0 = time.monotonic()
            while time.monotonic() - t0 < RUNAWAY_S:
                s.send(v)
                time.sleep(0.01)
        raise AssertionError("the e-stop never reached the stream")

    handle = cell.start(body)
    cell.observer.wait_moving()
    t_estop = time.monotonic()
    try:
        cell.observer.set_estop(True)
        _, outcome = cell.finish(handle)
        rest = cell.observer.wait_still() - t_estop
        wait_until(lambda: not cell.observer.stream.open, WAIT_S, "the stream to close")
    finally:
        recover(cell, home)
    assert isinstance(outcome.raised, EStopped | StreamClosed), outcome.raised
    assert rest < STOP_S
