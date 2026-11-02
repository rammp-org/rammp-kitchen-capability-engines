"""self.base: twists to the drive base, its odometry, drive mode requests, per-action entities, the
stop at the end of an action, cancel, and mock mode. A BaseInterface on a fresh node beside a
FakeBase."""

import math
import threading
import time

import pytest
from fake_driver import FakeBase, FakeCtx, counts
from rammp_base_interfaces.msg import DriveCommand, MibStatus
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from capability_engine import base as base_module
from capability_engine.base import BaseError, BaseInterface
from capability_engine.engine import Cancelled

WAIT_S = 5.0
POLL_S = 0.01
NO_TOKEN = [0] * 16
ZERO = (0.0, 0.0, 0.0, NO_TOKEN)


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(POLL_S)


@pytest.fixture
def base_env(ros):
    """(make, fake): make(mock=False) -> a BaseInterface on a fresh spun node; a FakeBase."""
    fake = FakeBase()
    executor = MultiThreadedExecutor()
    executor.add_node(fake)
    made = []

    def make(mock=False):
        node = Node(f"base_test_engine_{len(made)}")
        executor.add_node(node)
        made.append((BaseInterface(node, ReentrantCallbackGroup(), mock=mock), node))
        return made[-1]

    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    yield make, fake
    for base, _ in made:
        base.end_action()
    executor.shutdown()
    spinner.join()
    for _, node in made:
        node.destroy_node()
    fake.destroy_node()


def test_calls_outside_an_action_raise(base_env):
    make, _ = base_env
    base, _ = make()
    for call in (lambda: base.send_twist(0.1, 0.0), base.stop, base.odometry):
        with pytest.raises(BaseError, match="only available inside an action"):
            call()


def test_twists_reach_the_base_and_the_end_stops_it(base_env):
    make, fake = base_env
    base, node = make()
    idle = counts(node)
    base.begin_action(FakeCtx())
    assert counts(node) == idle  # nothing is made until it is used
    wait_until(lambda: (base.send_twist(0.2, -0.3, 0.05), fake.received())[1])
    assert fake.received()[-1] == (0.2, 0.05, -0.3, NO_TOKEN)
    assert counts(node) != idle
    sent = len(fake.received())
    base.end_action()
    wait_until(lambda: len(fake.received()) >= sent + 1)
    time.sleep(0.1)  # every repeat has landed
    assert fake.received()[sent:] == [ZERO] * base_module.STOP_REPEATS
    assert counts(node) == idle


def test_an_action_that_never_drove_sends_no_stop(base_env):
    make, fake = base_env
    base, _ = make()
    base.begin_action(FakeCtx())
    base.end_action()
    time.sleep(0.2)
    assert fake.received() == []


def test_cancel_refuses_twists_but_not_stop(base_env):
    make, fake = base_env
    base, _ = make()
    ctx = FakeCtx()
    base.begin_action(ctx)
    ctx.cancel()
    with pytest.raises(Cancelled):
        base.send_twist(0.2, 0.0)
    wait_until(lambda: (base.stop(), fake.received())[1])
    assert all(r == ZERO for r in fake.received())


def test_odometry_is_the_latest_message(base_env):
    make, fake = base_env
    base, _ = make()
    fake.set_odom(1.5, -0.5, 0.3)
    fake.start()
    base.begin_action(FakeCtx())
    odom = base.odometry()
    assert (odom.pose.pose.position.x, odom.pose.pose.position.y) == (1.5, -0.5)
    assert odom.header.frame_id == "odom"
    fake.set_odom(2.0, 0.0, 0.0)
    wait_until(lambda: base.odometry().pose.pose.position.x == 2.0)


def test_odometry_raises_when_none_comes(base_env, monkeypatch):
    monkeypatch.setattr(base_module, "ODOM_WAIT_S", 0.3)
    make, _ = base_env
    base, _ = make()
    base.begin_action(FakeCtx())
    with pytest.raises(BaseError, match="no /base/odom"):
        base.odometry()


def test_stale_odometry_raises(base_env):
    make, fake = base_env
    base, _ = make()
    fake.start()
    base.begin_action(FakeCtx())
    base.odometry()
    fake.stop_publishing()
    time.sleep(base_module.ODOM_STALE_S + 0.2)
    with pytest.raises(BaseError, match="s old"):
        base.odometry()


def test_waiting_for_odometry_follows_a_cancel(base_env):
    make, _ = base_env
    base, _ = make()
    ctx = FakeCtx()
    base.begin_action(ctx)
    threading.Timer(0.2, ctx.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        base.odometry()
    assert time.monotonic() - t0 < 1.0


def test_an_ended_action_forgets_its_odometry(base_env, monkeypatch):
    make, fake = base_env
    base, _ = make()
    fake.start()
    base.begin_action(FakeCtx())
    base.odometry()
    base.end_action()
    fake.stop_publishing()
    time.sleep(0.1)
    base.begin_action(FakeCtx())
    monkeypatch.setattr(base_module, "ODOM_WAIT_S", 0.3)
    with pytest.raises(BaseError, match="no /base/odom"):
        base.odometry()  # the last action's message is not returned


def test_mock_sends_nothing_and_dead_reckons(base_env):
    make, fake = base_env
    base, node = make(mock=True)
    idle = counts(node)
    base.begin_action(FakeCtx())
    base.send_twist(0.5, 0.0)
    time.sleep(0.2)
    base.send_twist(0.0, math.pi)  # turn on the spot
    time.sleep(0.1)
    odom = base.odometry()
    base.end_action()
    assert fake.received() == []
    assert counts(node) == idle
    assert odom.pose.pose.position.x == pytest.approx(0.1, abs=0.02)
    yaw = 2 * math.atan2(odom.pose.pose.orientation.z, odom.pose.pose.orientation.w)
    assert yaw == pytest.approx(0.1 * math.pi, abs=0.05)
    assert odom.twist.twist.angular.z == math.pi
    base.begin_action(FakeCtx())  # the pose is kept, the base no longer moves
    x = base.odometry().pose.pose.position.x
    time.sleep(0.1)
    assert base.odometry().pose.pose.position.x == x


AUTONOMOUS, SHARED = MibStatus.SYSTEM_STATE_AUTONOMOUS, MibStatus.SYSTEM_STATE_SHARED
IDLE, ESTOP = MibStatus.SYSTEM_STATE_IDLE, MibStatus.SYSTEM_STATE_ESTOP
LOW = MibStatus.PROFILE_LOW  # FakeBase's profile in force


def test_request_mode_enters_it_keeping_the_profile(base_env):
    make, fake = base_env
    base, _ = make()
    fake.start_mib(IDLE)
    base.begin_action(FakeCtx())
    base.request_mode("autonomous")
    assert fake.state() == AUTONOMOUS
    assert fake.requests()[-1] == (DriveCommand.REQUEST_AUTONOMOUS, LOW)


def test_request_mode_already_in_it_sends_nothing(base_env):
    make, fake = base_env
    base, _ = make()
    fake.start_mib(SHARED)
    base.begin_action(FakeCtx())
    base.request_mode("shared")
    assert fake.requests() == []


def test_switching_modes_goes_through_disable(base_env):
    make, fake = base_env
    base, _ = make()
    fake.start_mib(IDLE)
    base.begin_action(FakeCtx())
    base.request_mode("shared")
    base.request_mode("autonomous")
    assert fake.state() == AUTONOMOUS
    requests = [r for r, _ in fake.requests()]
    deduped = [r for i, r in enumerate(requests) if i == 0 or r != requests[i - 1]]
    # A request repeats until MibStatus answers it.
    assert deduped == [
        DriveCommand.REQUEST_SHARED,
        DriveCommand.REQUEST_DISABLE,
        DriveCommand.REQUEST_AUTONOMOUS,
    ]


def test_request_mode_refused_in_estop_sends_nothing(base_env):
    make, fake = base_env
    base, _ = make()
    fake.start_mib(ESTOP)
    base.begin_action(FakeCtx())
    with pytest.raises(BaseError, match="ESTOP"):
        base.request_mode("autonomous")
    assert fake.requests() == []


def test_an_unanswered_request_is_resent_then_fails(base_env, monkeypatch):
    monkeypatch.setattr(base_module, "MODE_WAIT_S", 0.6)
    monkeypatch.setattr(base_module, "MODE_RESEND_S", 0.1)
    make, fake = base_env
    base, _ = make()
    fake.start_mib(IDLE)
    fake.ignore_commands()
    base.begin_action(FakeCtx())
    with pytest.raises(BaseError, match="did not enter AUTONOMOUS"):
        base.request_mode("autonomous")
    assert len(fake.requests()) >= 3


def test_no_mib_status_fails(base_env, monkeypatch):
    monkeypatch.setattr(base_module, "MODE_WAIT_S", 0.3)
    make, _ = base_env
    base, _ = make()
    base.begin_action(FakeCtx())
    with pytest.raises(BaseError, match="no /rammp/mib/status"):
        base.request_mode("autonomous")


def test_unknown_mode_raises(base_env):
    make, _ = base_env
    base, _ = make()
    base.begin_action(FakeCtx())
    with pytest.raises(BaseError, match="not one of"):
        base.request_mode("enable")


def test_request_mode_follows_a_cancel(base_env):
    make, fake = base_env
    base, _ = make()
    fake.start_mib(IDLE)
    fake.ignore_commands()
    ctx = FakeCtx()
    base.begin_action(ctx)
    threading.Timer(0.3, ctx.cancel).start()
    with pytest.raises(Cancelled):
        base.request_mode("autonomous")


def test_the_user_taking_over_stops_the_twists_and_is_not_undone(base_env):
    """A joystick touch ends AUTONOMOUS (IDLE): send_twist() raises, and the end of the action
    neither re-requests the mode nor sends DISABLE."""
    make, fake = base_env
    base, _ = make()
    fake.start_mib(IDLE)
    base.begin_action(FakeCtx())
    base.request_mode("autonomous")
    base.send_twist(0.1, 0.0)
    fake.set_state(IDLE)
    wait_until(lambda: _raises(lambda: base.send_twist(0.1, 0.0)))
    with pytest.raises(BaseError, match="left AUTONOMOUS for IDLE"):
        base.send_twist(0.1, 0.0)
    sent = len(fake.requests())
    base.end_action()
    time.sleep(0.2)
    assert len(fake.requests()) == sent


def test_the_end_of_the_action_leaves_the_mode_it_entered(base_env):
    make, fake = base_env
    base, _ = make()
    fake.start_mib(IDLE)
    base.begin_action(FakeCtx())
    base.request_mode("shared")
    base.send_twist(0.1, 0.0)
    base.end_action()
    wait_until(lambda: fake.state() == IDLE)
    assert fake.requests()[-1] == (DriveCommand.REQUEST_DISABLE, LOW)


def test_mock_request_mode_sends_nothing(base_env):
    make, fake = base_env
    base, node = make(mock=True)
    idle = counts(node)
    fake.start_mib(IDLE)
    base.begin_action(FakeCtx())
    base.request_mode("autonomous")
    base.send_twist(0.1, 0.0)
    base.end_action()
    time.sleep(0.1)
    assert fake.requests() == []
    assert counts(node) == idle


def _raises(call):
    try:
        call()
    except BaseError:
        return True
    return False


def test_a_mode_it_did_not_switch_into_is_left_alone(base_env):
    make, fake = base_env
    base, _ = make()
    fake.start_mib(AUTONOMOUS)
    base.begin_action(FakeCtx())
    base.request_mode("autonomous")
    base.end_action()
    time.sleep(0.2)
    assert fake.requests() == []
    assert fake.state() == AUTONOMOUS
