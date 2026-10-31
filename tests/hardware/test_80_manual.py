"""Stage 8, needs a person (--manual): the red e-stop and an unexpected contact, mid-motion.

Each test says what to do on the console, then waits for it to happen. Students' code must see
the motion return False with a reason in last_error (not hang, not raise), and the arm must move
again once an operator clears the fault (the harness calls /clear_faults, as the operator)."""

import time

import numpy as np
import pytest
from conftest import ACTION_WAIT_S, WAIT_S, diagnostic, say, wait_until

pytestmark = [pytest.mark.motion, pytest.mark.manual]

LONG = np.array([0, 0, 0, 0, 0, 0, 3.0])  # a slow full wrist turn: time for a person to act
SLOW = 0.15
ACT_S = 20.0  # how long the person has to act
RELEASE_S = 60.0  # ... and to release the e-stop
CLOSED_ON_NOTHING = 0.6  # a grip on an object stops short of this


def slow_turn(home):
    def body(arm, ctx):
        return arm.move_joints([home + LONG], speed_scale=SLOW), arm.last_error

    return body


def clear_until_ready(cell, timeout_s):
    """Call /clear_faults until the arm is commandable again (the person releases first)."""
    deadline = time.monotonic() + timeout_s
    while True:
        reply = cell.observer.clear_faults()
        if reply.success:
            return
        assert time.monotonic() < deadline, f"/clear_faults: {reply.message}"
        time.sleep(2.0)


def test_the_red_estop_mid_motion(cell, home):
    handle = cell.start(slow_turn(home))
    cell.observer.wait_moving(timeout_s=10.0)
    say(f"PRESS THE RED E-STOP within {ACT_S:g} s, while the wrist turns.")
    _, outcome = cell.finish(handle, ACT_S + WAIT_S)
    say(f"Release the e-stop (within {RELEASE_S:g} s); the test then clears the fault.")
    clear_until_ready(cell, RELEASE_S)
    assert cell.act(lambda arm, ctx: arm.go_to_joint_config(home)), "no motion after the clear"
    assert outcome.raised is None, f"it raised {outcome.raised!r}: RammpArm returns False"
    ok, last_error = outcome.value
    assert ok is False and last_error, "the motion did not report the e-stop"


def test_an_unexpected_contact_mid_motion(cell, home):
    """With the driver's contact detection on (kinova-gen3-ros2 #112), a push stops the goal
    (CONTACT_STOP) and the driver refuses every goal until /clear_faults. self.arm has no way to
    clear it, so a capability's next motion must fail with a reason that says so."""
    if cell.act(lambda arm, ctx: diagnostic(arm.diagnostics(), "Contact detection")) is None:
        pytest.skip("this driver has no contact detection")
    handle = cell.start(slow_turn(home))
    cell.observer.wait_moving(timeout_s=10.0)
    say(
        f"PUSH THE FOREARM firmly sideways within {ACT_S:g} s, while the wrist turns. "
        "A contact stop drops torque for ~160 ms: the arm may droop (driver #92)."
    )
    _, outcome = cell.finish(handle, ACT_S + WAIT_S)
    wait_until(cell.observer.still_now, WAIT_S, "the arm to stop")

    def next_motion(arm, ctx):
        return arm.move_joints([home]), arm.last_error

    after = cell.act(next_motion, ACTION_WAIT_S)
    clear_until_ready(cell, WAIT_S * 3)
    assert cell.act(lambda arm, ctx: arm.go_to_joint_config(home)), "no motion after the clear"
    assert outcome.raised is None, f"it raised {outcome.raised!r}"
    ok, last_error = outcome.value
    assert ok is False and "contact" in last_error.lower(), last_error
    # The driver gives its reason ("call /clear_faults") only in its own log, and self.arm has
    # no way to clear a contact stop: the student sees a bare "goal REJECTED".
    ok, last_error = after
    assert ok is False and last_error, last_error


def test_a_grasp_on_an_object_returns_and_holds(cell, home):
    """The fingers stall on an object: gripper() still returns (True) once they stop, and the
    grip holds after it returns."""
    assert cell.act(lambda arm, ctx: arm.gripper(0.0))
    say(f"Hold the foam block between the open fingers; the gripper closes in {ACT_S / 2:g} s.")
    time.sleep(ACT_S / 2)
    t0 = time.monotonic()
    assert cell.act(lambda arm, ctx: arm.gripper(1.0))
    took = time.monotonic() - t0
    held = cell.observer.gripper()
    say("Let go of the block; the test checks the grip for 3 s.")
    time.sleep(3.0)
    assert took < 4.0 and held < CLOSED_ON_NOTHING, (took, held)
    assert abs(cell.observer.gripper() - held) < 0.02, "the grip slipped"
    assert cell.act(lambda arm, ctx: arm.gripper(0.0))
