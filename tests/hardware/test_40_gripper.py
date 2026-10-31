"""Stage 4, the gripper: gripper(position, force, speed) and the open/close commands.

Nothing in the fingers. gripper() returns once the fingers stop: when it returns they are where
they will stay, and that is where it was told."""

import time

import pytest
from conftest import POLL_S

from capability_engine.cornell_commands import CloseGripperCommand, OpenGripperCommand

pytestmark = pytest.mark.motion

OPEN_MAX = 0.05  # gripper_state position once opened
CLOSED_MIN = 0.6  # ... once closed on nothing
HALF_TOL = 0.1  # gripper(0.5) lands this close to 0.5
SETTLED = 0.01  # after gripper() returns, the fingers move no more than this
SETTLE_CHECK_S = 0.5
RETURN_S = 4.0  # gripper() returns within this (its default timeout_s)
SLOW, FAST = 0.2, 1.0  # speeds of the speed test


def grip(cell, position, **kwargs):
    """gripper(position) in one action: how long it took, and the fingers once it returned."""
    t0 = time.monotonic()
    assert cell.act(lambda arm, ctx: arm.gripper(position, **kwargs))
    took = time.monotonic() - t0
    at_return = cell.observer.gripper()
    time.sleep(SETTLE_CHECK_S)
    assert abs(cell.observer.gripper() - at_return) < SETTLED, "the fingers kept moving"
    return took, at_return


def test_close_then_open(cell):
    took, closed = grip(cell, 1.0)
    assert closed > CLOSED_MIN and took < RETURN_S, (closed, took)
    took, opened = grip(cell, 0.0)
    assert opened < OPEN_MAX and took < RETURN_S, (opened, took)


def test_half_closed(cell):
    _, half = grip(cell, 0.5)
    assert abs(half - 0.5) < HALF_TOL, half
    grip(cell, 0.0)


def test_speed(cell):
    grip(cell, 0.0)
    took = {}
    for speed in (SLOW, FAST):
        t0 = time.monotonic()
        grip(cell, 1.0, speed=speed)
        t, pos = cell.observer.gripper_since(t0)  # from the fingers' first move to the last
        moving = t[(abs(pos - pos[0]) > SETTLED) & (abs(pos - pos[-1]) > SETTLED)]
        took[speed] = (moving[-1] - moving[0]) if len(moving) else 0.0
        grip(cell, 0.0)
    assert took[SLOW] > 1.5 * took[FAST], took


def test_force_is_accepted(cell):
    _, closed = grip(cell, 1.0, force=0.2)
    assert closed > CLOSED_MIN
    grip(cell, 0.0)


def test_execute_command_open_close(cell):
    assert cell.act(lambda arm, ctx: arm.execute_command(CloseGripperCommand()))
    time.sleep(POLL_S)
    assert cell.observer.gripper() > CLOSED_MIN
    assert cell.act(lambda arm, ctx: arm.execute_command(OpenGripperCommand()))
    assert cell.observer.gripper() < OPEN_MAX
