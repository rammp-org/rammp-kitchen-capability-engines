"""Stage 2, straight-line tool moves: move_cartesian and execute_command(Cartesian...Command).

Students expect the tool to go in a straight line, keep its orientation unless told otherwise,
and stop where they said. Every move is a few cm from home's tool pose."""

import time

import numpy as np
import pytest
import rammp_arm
from conftest import angle_deg
from scipy.spatial.transform import Rotation as R

from capability_engine.cornell_commands import CartesianTrajectoryCommand

pytestmark = pytest.mark.motion

END_M = 0.005  # the tool this close to the last pose once still
LINE_M = 0.005  # every tool sample this close to the straight segment
TURN_DEG = 2.0  # orientation drift on a pure translation; the end error of a turn
SPEED_MARGIN = 1.2  # the tool's peak speed may exceed CART_SPEED by this factor
UP = np.array([0.0, 0.0, 0.05])
SIDE = np.array([0.0, 0.04, 0.0])
FAR = np.array([1.0, 0.0, 0.0])  # 1 m further out: out of reach
TWIST_DEG = 15.0  # the in-place turn, about the tool's z


def segment_distance(points, a, b):
    """Each point's distance from the segment a-b."""
    ab = b - a
    s = np.clip((points - a) @ ab / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(points - (a + s[:, None] * ab), axis=1)


def settle(cell):
    cell.observer.wait_still()
    return cell.observer.tool()


def test_move_cartesian_goes_straight(cell, home):
    p0, q0 = cell.observer.tool()
    target = p0 + UP
    t0 = time.monotonic()
    assert cell.act(lambda arm, ctx: arm.move_cartesian([(target, q0)]))
    p1, q1 = settle(cell)
    assert np.linalg.norm(p1 - target) < END_M, p1 - target
    _, pos, quat = cell.observer.tool_since(t0)
    assert segment_distance(pos, p0, target).max() < LINE_M
    assert max(angle_deg(q, q0) for q in quat) < TURN_DEG
    _, speed = cell.observer.tool_speed(t0)
    assert speed.max() < SPEED_MARGIN * rammp_arm.CART_SPEED, speed.max()


def test_move_cartesian_through_two_segments(cell, home):
    p0, q0 = cell.observer.tool()
    corner, end = p0 + UP, p0 + UP + SIDE
    t0 = time.monotonic()
    assert cell.act(lambda arm, ctx: arm.move_cartesian([(corner, q0), (end, q0)]))
    p1, _ = settle(cell)
    assert np.linalg.norm(p1 - end) < END_M
    _, pos, _ = cell.observer.tool_since(t0)
    near = np.minimum(segment_distance(pos, p0, corner), segment_distance(pos, corner, end))
    assert near.max() < LINE_M
    assert np.linalg.norm(pos - corner, axis=1).min() < END_M  # it reaches the corner


def test_move_cartesian_turns_in_place(cell, home):
    p0, q0 = cell.observer.tool()
    q_target = (R.from_quat(q0) * R.from_euler("z", TWIST_DEG, degrees=True)).as_quat()
    t0 = time.monotonic()
    assert cell.act(lambda arm, ctx: arm.move_cartesian([(p0, q_target)]))
    p1, q1 = settle(cell)
    assert angle_deg(q1, q_target) < TURN_DEG
    _, pos, _ = cell.observer.tool_since(t0)
    assert np.linalg.norm(pos - p0, axis=1).max() < LINE_M


def test_an_unreachable_line_is_refused_before_moving(cell, home):
    p0, q0 = cell.observer.tool()
    t0 = time.monotonic()
    ok, last_error = cell.act(
        lambda arm, ctx: (arm.move_cartesian([(p0 + FAR, q0)]), arm.last_error)
    )
    assert not ok and last_error
    assert cell.observer.motion_span(t0) is None


def test_execute_command_cartesian(cell, home):
    p0, q0 = cell.observer.tool()
    command = CartesianTrajectoryCommand([(p0 + UP, q0)])
    assert cell.act(lambda arm, ctx: arm.execute_command(command))
    p1, _ = settle(cell)
    assert np.linalg.norm(p1 - (p0 + UP)) < END_M
