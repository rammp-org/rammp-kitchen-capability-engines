"""Stage 6, an action cut short mid-motion: the orchestrator's cancel (Control and the goal's
own), the engine's stop(), pause, and a signal to the process (docker stop / sheppy woof).

Students are told: on a cancel or a stop self.arm stops the motion at once, the motion raises
Cancelled, and the engine takes the next action. Here the arm must be still within STOP_S of
the interrupt, short of its target, with no stream left open, and the next action must move."""

import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import rammp_arm
from conftest import ACTION_WAIT_S, STOP_S, WAIT_S, joint_err, wait_until
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import Result
from rammp_orchestrator_interfaces.srv import Control
from rclpy.action import ActionClient

from capability_engine import Cancelled
from capability_engine.engine import stop_wait_s

pytestmark = pytest.mark.motion

LONG = np.array([0, 0, 0, 0, 0, 0, 1.2])  # a slow wrist turn: several seconds, hits nothing
SLOW = 0.3  # speed_scale of the long moves
SLOW_GO_TO = 0.1
UP = np.array([0.0, 0.0, 0.05])
WRIST_RAD_S = 0.2  # the streamed motion: J7 velocity
INTERRUPT_AFTER_S = 0.5  # the interrupt comes this long after the arm starts moving
RUNAWAY_S = 5.0  # a streamed motion nobody stopped fails after this long
SHORT_DEG = 5.0  # the arm stopped at least this far short of the target
PAUSE_HOLD_S = 1.5  # how long the pause test holds the action paused after its motion


def long_motions(home):
    """name -> body(arm, ctx): a motion of several seconds from home."""

    def move_joints(arm, ctx):
        return arm.move_joints([home + LONG], speed_scale=SLOW)

    def go_to_joint_config(arm, ctx):
        return arm.go_to_joint_config(home + LONG, speed_scale=SLOW_GO_TO)

    def go_to_ee_pose(arm, ctx):
        ee = arm.get_state()["ee_pos"]
        return arm.go_to_ee_pose(ee[:3] + UP, ee[3:], speed_scale=SLOW_GO_TO)

    def move_cartesian(arm, ctx):
        ee = arm.get_state()["ee_pos"]
        return arm.move_cartesian([(ee[:3] + UP, ee[3:])], speed_scale=SLOW)

    def execute_joint_trajectory(arm, ctx):
        t, q, qd = rammp_arm.joint_trajectory(arm.raw_joints(), [home + LONG], None, SLOW)
        return arm.execute_joint_trajectory(rammp_arm.JOINTS, list(zip(t, q, qd, strict=True)))

    def stream(arm, ctx):
        name = next(c.name for c in arm.list_controllers() if c.channels == ["joint_velocity"])
        v = np.zeros(7)
        v[6] = WRIST_RAD_S
        with arm.stream(name) as s:
            t0 = time.monotonic()
            while time.monotonic() - t0 < RUNAWAY_S:
                s.send(v)
                time.sleep(0.01)
        raise AssertionError("the interrupt never reached the stream")

    return {f.__name__: f for f in (
        move_joints, go_to_joint_config, go_to_ee_pose, move_cartesian,
        execute_joint_trajectory, stream,
    )}  # fmt: skip


def assert_short_of_target(cell, home, motion, p0):
    """It stopped where the interrupt caught it, not at the end of the motion."""
    if motion in ("go_to_ee_pose", "move_cartesian"):
        rise = cell.observer.tool()[0][2] - p0[2]
        assert rise < UP[2] - 0.01, f"the tool rose {rise:.3f} m: it finished"
    else:
        err = np.degrees(joint_err(cell.observer.q(), home + LONG).max())
        assert err > SHORT_DEG, f"{err:.1f} deg from the target: it finished"


def control_cancel(cell, handle):
    cell.control(Control.Request.CANCEL)


def goal_cancel(cell, handle):
    handle.cancel_goal_async()


def pause_then_cancel(cell, handle):
    cell.control(Control.Request.PAUSE)
    time.sleep(0.2)
    cell.control(Control.Request.CANCEL)


def engine_stop(cell, handle):
    cell.engine.stop()


INTERRUPTS = {f.__name__: f for f in (control_cancel, goal_cancel, pause_then_cancel, engine_stop)}
CASES = [(m, "control_cancel") for m in long_motions(None)]
CASES += [(m, i) for m in ("move_joints", "go_to_ee_pose", "stream") for i in INTERRUPTS]
CASES = sorted(set(CASES))


def test_a_cancel_while_planning_never_moves_the_arm(cell, home):
    """Cancelled before cuRobo has answered: the planned motion must never start."""

    def body(arm, ctx):
        ee = arm.get_state()["ee_pos"]
        return arm.go_to_ee_pose(ee[:3] + UP, ee[3:], speed_scale=SLOW_GO_TO)

    t0 = time.monotonic()
    handle = cell.start(body)
    cell.control(Control.Request.CANCEL)
    result, outcome = cell.finish(handle)
    assert result.status == Result.STATUS_CANCELLED, (result.status, result.error_code)
    assert isinstance(outcome.raised, Cancelled), outcome
    time.sleep(3.0)  # a goal sent after all would be moving by now
    assert cell.observer.motion_span(t0) is None


def test_a_cancelled_gripper_returns_at_once(cell):
    """The 2F-85 latches its last setpoint, so the fingers may finish their move; what the
    student sees is Cancelled, promptly."""
    assert cell.act(lambda arm, ctx: arm.gripper(0.0))
    handle = cell.start(lambda arm, ctx: arm.gripper(1.0, speed=0.1))
    g0 = cell.observer.gripper()
    wait_until(lambda: abs(cell.observer.gripper() - g0) > 0.02, WAIT_S, "the fingers to move")
    t_cancel = time.monotonic()
    cell.control(Control.Request.CANCEL)
    result, outcome = cell.finish(handle)
    assert time.monotonic() - t_cancel < STOP_S
    assert result.status == Result.STATUS_CANCELLED and isinstance(outcome.raised, Cancelled)
    assert cell.act(lambda arm, ctx: arm.gripper(0.0))


@pytest.mark.parametrize("motion,interrupt", CASES)
def test_an_interrupted_motion_stops_the_arm(cell, home, motion, interrupt):
    p0, _ = cell.observer.tool()
    handle = cell.start(long_motions(home)[motion])
    cell.observer.wait_moving(timeout_s=10.0)
    time.sleep(INTERRUPT_AFTER_S)
    t_interrupt = time.monotonic()
    INTERRUPTS[interrupt](cell, handle)
    result, outcome = cell.finish(handle)
    assert time.monotonic() - t_interrupt < stop_wait_s()
    assert result.status == Result.STATUS_CANCELLED, (result.status, result.error_code)
    assert isinstance(outcome.raised, Cancelled), outcome
    assert cell.observer.wait_still() - t_interrupt < STOP_S
    assert_short_of_target(cell, home, motion, p0)
    assert not cell.observer.stream.open
    # The engine (a new one after a stop) takes the next action, and it moves.
    assert cell.act(lambda arm, ctx: arm.go_to_joint_config(home))


def test_a_pause_lets_the_motion_finish_then_holds_the_action(cell, home):
    """Pause is honoured between steps (ctx.checkpoint()), not inside a motion."""
    target = home + LONG

    def body(arm, ctx):
        ok = arm.move_joints([target], speed_scale=SLOW)
        ctx.checkpoint()
        return ok, time.monotonic()

    handle = cell.start(body)
    cell.observer.wait_moving()
    cell.control(Control.Request.PAUSE)
    cell.observer.wait_still(timeout_s=ACTION_WAIT_S)
    assert np.degrees(joint_err(cell.observer.q(), target).max()) < 1.0  # it finished
    time.sleep(PAUSE_HOLD_S)
    future = handle.get_result_async()
    assert not future.done(), "the action did not wait at its checkpoint"
    resumed = time.monotonic()
    cell.control(Control.Request.RESUME)
    result, outcome = cell.finish(handle)
    assert result.status == Result.STATUS_SUCCESS
    ok, past_checkpoint = outcome.value
    assert ok and past_checkpoint >= resumed


SIGNAL_SCRIPT = """
import numpy as np
from capability_engine import Action, CapabilityEngine, run


class Long(CapabilityEngine):
    actions = [Action("run", "Run")]

    def __init__(self):
        super().__init__("{engine_id}")

    def execute(self, action_id, params, ctx):
        return {{"moved": self.arm.move_joints([np.array({target})], speed_scale={speed})}}


run(Long)
"""


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_a_signal_mid_motion_stops_the_arm_and_exits(cell, home, tmp_path, sig):
    """docker stop sends SIGINT (the images' STOPSIGNAL); sheppy may send SIGTERM."""
    engine_id = f"hw_test_signal_{sig.name.lower()}"
    script = tmp_path / "long.py"
    target = (home + LONG).tolist()
    script.write_text(SIGNAL_SCRIPT.format(engine_id=engine_id, target=target, speed=SLOW))
    proc = subprocess.Popen(
        [sys.executable, str(script), "--ros-args", "-p", "use_orchestrator:=false"],
        stderr=subprocess.PIPE,
        text=True,
    )
    client = ActionClient(cell.client, ExecuteAction, f"/{engine_id}/execute_action")
    logs = []
    try:
        assert client.wait_for_server(timeout_sec=30.0)
        client.send_goal_async(ExecuteAction.Goal(action_id="run"))
        cell.observer.wait_moving(timeout_s=10.0)
        time.sleep(INTERRUPT_AFTER_S)
        t_signal = time.monotonic()
        proc.send_signal(sig)
        reader = threading.Thread(target=lambda: logs.append(proc.communicate()[1]))
        reader.start()
        assert cell.observer.wait_still() - t_signal < STOP_S
        reader.join(timeout=stop_wait_s() + WAIT_S)
        assert proc.poll() is not None, "the engine did not exit"
        assert proc.returncode == 0, logs
        assert "Traceback" not in logs[0] and "Error" not in logs[0], logs[0]
    finally:
        client.destroy()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    wait_until(lambda: not cell.observer.stream or not cell.observer.stream.open)
    assert np.degrees(joint_err(cell.observer.q(), home + LONG).max()) > SHORT_DEG
