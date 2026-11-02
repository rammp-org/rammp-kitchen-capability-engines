"""PlannerClient against the kinova-gen3-ros2 driver's contract: what reaches
/execute_joint_trajectory and /setpoint/gripper, and what never does.

The client is built without a ROS graph (its __init__ opens real action
clients); the doubles below stand in for rclpy's ActionClient and
publisher, shaped like the real ones (a goal handle with .accepted and
get_result_async(), a result wrapper with .result)."""

import time
from types import SimpleNamespace

import pytest
from action_msgs.srv import CancelGoal
from rammp_arm_interfaces.action import ExecuteJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening.runtime import client as client_mod

NAMES = ["joint_%d" % i for i in range(1, 8)]
LIVE = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]


def _traj(start, end, times=(0.02, 1.0)):
    t = JointTrajectory()
    t.joint_names = list(NAMES)
    for row, tt in zip((start, end), times, strict=False):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        p.time_from_start.sec = int(tt)
        p.time_from_start.nanosec = int(round((tt - int(tt)) * 1e9))
        t.points.append(p)
    return t


def _secs(p):
    return p.time_from_start.sec + p.time_from_start.nanosec * 1e-9


class _Future:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class _GoalHandle:
    accepted = True

    def __init__(self, error_code, error_string):
        self._wrapped = SimpleNamespace(
            status=4,
            result=ExecuteJointTrajectory.Result(error_code=error_code, error_string=error_string),
        )

    def get_result_async(self):
        return _Future(self._wrapped)

    def cancel_goal_async(self):
        return _Future(SimpleNamespace(return_code=0))  # ERROR_NONE: taken


class _Driver:
    """The execute_joint_trajectory action as the client sees it: records
    each goal, replays scripted feedback, answers with one result."""

    def __init__(self, error_code=0, error_string="", feedback=()):
        self.goals = []
        self._answer = (error_code, error_string)
        self._feedback = list(feedback)

    def wait_for_server(self, timeout_sec=None):
        return True

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        for fraction in self._feedback:
            fb = ExecuteJointTrajectory.Feedback(fraction_complete=fraction)
            feedback_callback(SimpleNamespace(feedback=fb))
        return _Future(_GoalHandle(*self._answer))


class _RunningFuture:
    """A goal result that is outstanding until the driver settles it."""

    def __init__(self, result):
        self._result = result
        self._done = False

    def settle(self):
        self._done = True

    def done(self):
        return self._done

    def result(self):
        return self._result if self._done else None


class _RunningHandle:
    """A goal that is still flying: its result settles when cancelled."""

    accepted = True

    def __init__(self, error_code=-6):
        self._fut = _RunningFuture(
            SimpleNamespace(
                status=5,
                result=ExecuteJointTrajectory.Result(error_code=error_code),
            )
        )
        self.cancels = 0

    def get_result_async(self):
        return self._fut

    def cancel_goal_async(self):
        self.cancels += 1
        self._fut.settle()  # the driver settles the goal on a cancel
        return _Future(SimpleNamespace(return_code=0))  # ERROR_NONE: taken


class _FlyingDriver(_Driver):
    """Accepts a goal that keeps running until something cancels it."""

    def __init__(self):
        super().__init__()
        self.handle = _RunningHandle()

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        return _Future(self.handle)


class _Publisher:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


class _Guard:
    def __init__(self):
        self.progress = []
        self.peak = 0.0

    def on_progress(self, p):
        self.progress.append(p)

    def on_efforts(self, eff):
        return False


@pytest.fixture(autouse=True)
def _no_spin(monkeypatch):
    monkeypatch.setattr(client_mod.rclpy, "spin_once", lambda node, timeout_sec=0.0: None)


def _client(driver=None, armed=True):
    c = client_mod.PlannerClient.__new__(client_mod.PlannerClient)
    c.node = object()
    c._abort = None
    c._armed = armed
    c._execute = driver if driver is not None else _Driver()
    c._active_joints = None  # no /joint_states names yet: joint_names is the bench's
    c._q = list(LIVE)
    c._eff = [0.0] * 7
    c._eff_at = time.monotonic()
    c._gripper_pos = 0.0
    c._gripper_target = None
    c._gripper_pub = _Publisher()
    return c


def test_nothing_reaches_the_driver_while_the_client_is_not_armed():
    driver = _Driver()
    outcome, info = _client(driver, armed=False).execute(_traj(LIVE, LIVE), 1.0)
    assert outcome == "failed" and driver.goals == []


def test_a_trajectory_that_starts_away_from_the_arm_is_never_sent():
    driver = _Driver()
    away = [LIVE[0] + 0.1] + LIVE[1:]
    outcome, info = _client(driver).execute(_traj(away, away), 1.0)
    assert outcome == "failed" and driver.goals == []


def test_a_slowed_leg_reaches_the_driver_dilated_in_position_mode():
    driver = _Driver()
    end = [v + 0.05 for v in LIVE]
    outcome, _info = _client(driver).execute(_traj(LIVE, end), 0.5)
    assert outcome == "arrived"
    (goal,) = driver.goals
    assert _secs(goal.trajectory.points[-1]) == pytest.approx(2.0)
    assert goal.control_mode == 0  # POSITION: impedance with default gains is not a press
    assert goal.preemption == 0  # QUEUE: a new goal never displaces one still settling


def test_a_driver_failure_fails_the_leg_with_its_reason():
    driver = _Driver(error_code=-9, error_string="halted: emergency stop")
    end = [v + 0.05 for v in LIVE]
    outcome, info = _client(driver).execute(_traj(LIVE, end), 1.0)
    assert outcome == "failed"
    assert "HALTED" in info["message"] and "emergency stop" in info["message"]


def test_execution_progress_arms_the_guard():
    driver = _Driver(feedback=(0.0, 0.3))
    guard = _Guard()
    end = [v + 0.05 for v in LIVE]
    outcome, info = _client(driver).execute(_traj(LIVE, end), 1.0, guard=guard)
    assert guard.progress == [0.0, pytest.approx(0.3)]
    assert info["progress"] == pytest.approx(0.3)


def test_a_gripper_command_goes_out_as_the_drivers_normalized_setpoint():
    c = _client()
    handle = c.gripper_send(0.8)
    assert handle is not None
    (msg,) = c._gripper_pub.sent
    assert msg.position == pytest.approx(1.0)


def test_a_gripper_command_pending_through_a_long_motion_is_still_joined():
    """press:close is dispatched early and joined after the scan flight, so the
    fingers close while the arm moves. In slow mode that flight alone outlasts
    the join timeout — the join must still read the gripper, not declare a
    failure without ever looking at it (field 2026-09-14, --speed-scale 0.3:
    "STOP: leg press:close -> failed (gripper at 0.793)", a gripper that had
    in fact closed)."""
    c = _client()
    handle = c.gripper_send(0.8)
    # the arm flew for longer than the whole gripper timeout
    stale = handle._replace(t0=handle.t0 - (client_mod.PlannerClient.GRIPPER_TIMEOUT_S + 3.0))
    c._gripper_pos = 0.793  # the fingers closed while it flew
    c._eff_at = time.monotonic()
    assert c.gripper_join(stale) == (True, 0.793, False)


def test_a_gripper_that_never_moved_still_fails_after_a_long_motion():
    """The other half: pending a long time is not itself success."""
    c = _client()
    handle = c.gripper_send(0.8)
    stale = handle._replace(t0=handle.t0 - (client_mod.PlannerClient.GRIPPER_TIMEOUT_S + 3.0))
    c._eff_at = time.monotonic()  # still reading 0.0: nothing moved
    ok, pos, _stalled = c.gripper_join(stale)
    assert ok is False and pos == pytest.approx(0.0)


def test_a_motion_stops_the_moment_the_search_finds_something():
    """The sweep flies until the detector commits a coarse fix. The goal is
    cancelled then and there, and the arm holds where it stopped — which is
    where the precise fix is taken from."""
    driver = _FlyingDriver()
    c = _client(driver)
    polls = {"n": 0}

    def found_it(progress=None):
        polls["n"] += 1
        return polls["n"] >= 2  # nothing on the first look, then a fix

    end = [v + 0.05 for v in LIVE]
    outcome, _info = c.execute(_traj(LIVE, end), 1.0, stop_when=found_it)
    assert outcome == "stopped"
    assert driver.handle.cancels == 1


def test_a_motion_that_finds_nothing_runs_to_its_end():
    driver = _Driver()
    end = [v + 0.05 for v in LIVE]
    outcome, _info = _client(driver).execute(_traj(LIVE, end), 1.0, stop_when=lambda progress=None: False)
    assert outcome == "arrived"


def test_no_gripper_command_goes_out_while_the_client_is_not_armed():
    c = _client(armed=False)
    assert c.gripper_send(0.8) is None
    assert c._gripper_pub.sent == []


def test_a_compensated_guard_is_fed_the_joints_of_the_same_joint_states_message():
    """The reach-out's guards take gravity out at the joints the efforts
    were read at (runtime/guards.CompensatedGuard): the client hands both."""
    driver = _FlyingDriver()
    c = _client(driver)
    c._eff = [9.0, 9.0, 9.0, 1.0, 2.0, 3.0, 4.0]
    seen = []

    class Guard(_Guard):
        wants_joints = True

        def on_efforts(self, eff, joints=None):
            seen.append((eff, joints))
            return True  # a trip cancels the goal

    end = [v + 0.05 for v in LIVE]
    outcome, _info = c.execute(_traj(LIVE, end), 1.0, guard=Guard())
    assert outcome == "touch" and driver.handle.cancels == 1
    assert seen == [([1.0, 2.0, 3.0, 4.0], list(LIVE))]
    # a plain guard is called as it always was
    c2 = _client(_FlyingDriver())
    calls = []

    class Plain(_Guard):
        def on_efforts(self, eff):
            calls.append(eff)
            return True

    assert c2.execute(_traj(LIVE, end), 1.0, guard=Plain())[0] == "touch" and calls == [[0.0] * 4]


def test_settled_efforts_average_fresh_readings_and_refuse_a_frozen_stream(monkeypatch):
    c = _client()
    ticks = {"n": 0}

    def spin(node, timeout_sec=0.0):  # a /joint_states message per spin
        ticks["n"] += 1
        c._eff = [0.0, 0.0, 0.0, float(ticks["n"] % 2), 0.0, 0.0, 0.0]
        c._eff_at = time.monotonic()

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    eff, q = c.settled_efforts(duration_s=0.05, settle_s=0.0)
    assert eff[0] == pytest.approx(0.5, abs=0.2) and q == pytest.approx(LIVE)
    monkeypatch.setattr(client_mod.rclpy, "spin_once", lambda node, timeout_sec=0.0: None)
    c._eff_at = time.monotonic() - 5.0  # the stream stopped
    assert c.settled_efforts(duration_s=0.05, settle_s=0.0, timeout_s=0.2) is None


def test_settled_joints_are_averaged_on_one_branch_across_the_wrap(monkeypatch):
    """joint_3 at HOME sits 4.6 mrad from +-pi and is reported either side
    of it: the average is pi, not 0."""
    import math

    c = _client()
    flip = {"n": 0}

    def spin(node, timeout_sec=0.0):
        flip["n"] += 1
        q = list(LIVE)
        q[2] = math.pi - 0.002 if flip["n"] % 2 else -math.pi + 0.002
        c._q = q
        c._eff_at = time.monotonic()

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    _eff, q = c.settled_efforts(duration_s=0.05, settle_s=0.0)
    assert abs(abs(q[2]) - math.pi) < 0.003


# -- a stop is claimed only when the driver confirmed it (review 2026-10-22) --


class _Pending:
    """A future the scripted driver settles (or never does)."""

    def __init__(self, result=None, done=False):
        self._r, self._d = result, done

    def done(self):
        return self._d

    def result(self):
        return self._r if self._d else None

    def settle(self, r=None):
        if r is not None:
            self._r = r
        self._d = True


def _ended(status, error_code=0):
    return SimpleNamespace(status=status, result=ExecuteJointTrajectory.Result(error_code=error_code))


class _ScriptedHandle:
    """A flying goal whose cancel answer and result are scripted: `cancel`
    is the CancelGoal response (None: it never comes), `ended` the result
    the driver settles on the cancel (None: never)."""

    accepted = True

    def __init__(self, cancel, ended):
        self._cancel = cancel
        self._ended = ended
        self.result_fut = _Pending()
        self.cancels = 0

    def get_result_async(self):
        return self.result_fut

    def cancel_goal_async(self):
        self.cancels += 1
        if self._ended is not None:
            self.result_fut.settle(self._ended)
        return _Pending(self._cancel, done=self._cancel is not None)


class _ScriptedDriver(_Driver):
    def __init__(self, handle):
        super().__init__()
        self.handle = handle

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        return _Future(self.handle)


class _TripGuard(_Guard):
    def on_efforts(self, eff):
        return eff is not None  # trips on the first reading


def _cresp(code):
    return CancelGoal.Response(return_code=code)


def _no_waiting(monkeypatch):
    # the confirmation waits (3 s cancel, 10 s result) shortened to a tick
    real = client_mod.spin_until_done
    monkeypatch.setattr(
        client_mod, "spin_until_done", lambda node, fut, timeout_s, abort=None: real(node, fut, 0.01, abort)
    )


def _run(handle, **kw):
    end = [v + 0.05 for v in LIVE]
    return _client(_ScriptedDriver(handle)).execute(_traj(LIVE, end), 1.0, **kw)


def test_a_trip_nobody_confirmed_stopping_is_a_failure_not_a_touch(monkeypatch):
    """The driver neither answers the cancel nor ends the goal (it dies when
    one cyclic exchange goes 3 s unanswered): reading a contact height off
    that arm and pushing on top of it is how a box gets pressed through."""
    _no_waiting(monkeypatch)
    outcome, info = _run(_ScriptedHandle(None, None), guard=_TripGuard())
    assert outcome == "failed" and "did not confirm" in info["message"]
    assert info["torque_peak"] == 0.0


def test_a_trip_whose_cancel_was_refused_and_stroke_completed_is_a_failure(monkeypatch):
    _no_waiting(monkeypatch)
    h = _ScriptedHandle(_cresp(CancelGoal.Response.ERROR_REJECTED), _ended(4, 0))  # SUCCESSFUL
    outcome, info = _run(h, guard=_TripGuard())
    assert outcome == "failed" and "did not confirm" in info["message"]


@pytest.mark.parametrize(
    "cancel, ended",
    [
        (_cresp(CancelGoal.Response.ERROR_NONE), _ended(5, -6)),  # taken, CANCELED
        (None, _ended(5, -6)),  # the answer lost, but the goal reads CANCELED
        (_cresp(CancelGoal.Response.ERROR_GOAL_TERMINATED), _ended(4, 0)),  # ended on its own just then
    ],
)
def test_a_trip_the_driver_confirmed_stopping_is_a_touch(monkeypatch, cancel, ended):
    _no_waiting(monkeypatch)
    outcome, info = _run(_ScriptedHandle(cancel, ended), guard=_TripGuard())
    assert outcome == "touch" and info["message"] == "torque guard trip"


@pytest.mark.parametrize(
    "cancel, ended",
    [(None, None), (_cresp(CancelGoal.Response.ERROR_REJECTED), _ended(4, 0))],
)
def test_a_stop_when_nobody_confirmed_is_a_failure_not_a_stop(monkeypatch, cancel, ended):
    _no_waiting(monkeypatch)
    outcome, info = _run(_ScriptedHandle(cancel, ended), stop_when=lambda progress=None: True)
    assert outcome == "failed" and "did not confirm" in info["message"]


def test_a_guarded_stroke_is_cancelled_within_a_fifth_of_a_second_of_the_stream_stopping(monkeypatch):
    """/joint_states stops as the stroke starts while the driver keeps
    flying it: 0.5 s staleness plus a 1 s grace was 1.5 s blind, ~126 mm
    of the depth touch's 84 mm/s stroke."""
    monkeypatch.setattr(client_mod.rclpy, "spin_once", lambda node, timeout_sec=0.0: time.sleep(0.005))
    h = _ScriptedHandle(_cresp(CancelGoal.Response.ERROR_NONE), _ended(5, -6))
    c = _client(_ScriptedDriver(h))
    t0 = time.monotonic()
    c._eff_at = t0  # the last message, now
    outcome, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0, guard=_Guard())
    blind = time.monotonic() - t0
    assert outcome == "failed" and "effort stream lost" in info["message"]
    assert h.cancels == 1
    assert blind < client_mod.PlannerClient.GUARD_STALE_S + 0.05 < 0.3


def test_a_guarded_leg_after_some_computing_starts_on_a_fresh_reading(monkeypatch):
    """Flatbot 2026-10-27 15:34: the re-aim started 0.3 s after the last
    /joint_states this node had served — still "flowing" for the start gate
    (EFFORT_STALE_S) — with camera frames queued ahead of the next one. Its
    guard called the stream lost 10 ms in, the arm never having moved. The
    gate now waits for a sample its guard would take."""
    h = _ScriptedHandle(_cresp(CancelGoal.Response.ERROR_NONE), _ended(5, -6))
    c = _client(_ScriptedDriver(h))
    c._eff_at = time.monotonic() - 0.3
    spins = {"n": 0, "sent_at": None}

    def spin(node, timeout_sec=0.0):  # three camera frames, then a /joint_states, over and over
        spins["n"] += 1
        if spins["n"] % 4 == 0:
            c._eff_at = time.monotonic()
        if spins["n"] == 40:
            h.result_fut.settle(_ended(4, 0))

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    outcome, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0, guard=_Guard())
    assert outcome == "arrived" and h.cancels == 0, info["message"]


def test_a_slow_stop_when_does_not_read_as_a_lost_stream(monkeypatch):
    """An in-flight confirm runs the wrist's depth fix in stop_when on the
    guarded stroke: its own time must not age the stream the guard judges."""
    h = _ScriptedHandle(_cresp(CancelGoal.Response.ERROR_NONE), _ended(5, -6))
    c = _client(_ScriptedDriver(h))
    spins = {"n": 0}

    def spin(node, timeout_sec=0.0):  # a /joint_states message per spin
        spins["n"] += 1
        c._eff_at = time.monotonic()
        if spins["n"] == 3:
            h.result_fut.settle(_ended(4, 0))

    def slow(progress=None):
        time.sleep(client_mod.PlannerClient.GUARD_STALE_S + 0.05)
        return False

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    outcome, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0, guard=_Guard(), stop_when=slow)
    assert outcome == "arrived" and h.cancels == 0


def _abort_client(handle):
    from box_opening.runtime.abort import AbortFlag, sigint_handler

    flag = AbortFlag()
    c = _client(_ScriptedDriver(handle))
    c._abort = flag
    return c, flag, sigint_handler(flag)


def test_a_ctrl_c_in_the_spin_that_delivers_the_result_is_not_swallowed(monkeypatch):
    h = _ScriptedHandle(None, None)
    c, flag, handler = _abort_client(h)

    def spin(node, timeout_sec=0.0):
        handler(2, None)  # Ctrl+C ...
        h.result_fut.settle(_ended(4, 0))  # ... while the result callback runs

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    with pytest.raises(KeyboardInterrupt):
        c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)
    assert flag.goal_in_flight is False and h.cancels == 0  # over already: nothing to cancel
    with pytest.raises(KeyboardInterrupt):
        c.gripper_send(0.8)  # what the runner does next never goes out
    assert c._gripper_pub.sent == []


def test_a_ctrl_c_while_the_goal_is_rejected_is_not_swallowed(monkeypatch):
    class Rejected:
        accepted = False

    c, flag, handler = _abort_client(None)
    c._execute.send_goal_async = lambda goal, feedback_callback=None: (handler(2, None), _Future(Rejected()))[1]
    with pytest.raises(KeyboardInterrupt):
        c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)


def _live_stream(c, monkeypatch, fresh_after_settle=None, settle_s=0.15):
    """A /joint_states message every ~20 ms (49 Hz live); with
    `fresh_after_settle` the stream freezes after that many fresh messages
    past the settle. Returns the count of fresh messages delivered after it."""
    t0 = time.monotonic()
    seen = {"after": 0}

    def spin(node, timeout_sec=0.0):
        time.sleep(0.02)
        late = time.monotonic() - t0 >= settle_s
        if late and fresh_after_settle is not None and seen["after"] >= fresh_after_settle:
            return  # frozen: _eff_at stops moving, then goes stale
        c._eff = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4]
        c._eff_at = time.monotonic()
        if late:
            seen["after"] += 1

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    return seen


def test_settled_efforts_default_read_holds_the_arm_about_0_3_s(monkeypatch):
    """Review 2026-10-22: the default 0.15 s settle + 0.30 s mean held the
    arm ~0.47 s per read, ~9 reads a depth run. 0.15 s + 0.15 s (~7
    messages at 49 Hz) is the whole read."""
    c = _client()
    seen = _live_stream(c, monkeypatch)
    t0 = time.monotonic()
    got = c.settled_efforts()
    elapsed = time.monotonic() - t0
    assert got is not None
    eff, q = got
    assert eff == pytest.approx([0.1, 0.2, 0.3, 0.4]) and q == pytest.approx(LIVE)
    assert 0.30 <= elapsed < 0.40
    assert seen["after"] >= 3


def test_settled_efforts_still_refuse_fewer_than_three_fresh_readings(monkeypatch):
    """Two fresh messages after the settle, then a frozen stream: no
    reference, at the default window as at any other."""
    c = _client()
    _live_stream(c, monkeypatch, fresh_after_settle=2)
    t0 = time.monotonic()
    assert c.settled_efforts(timeout_s=0.8) is None
    assert time.monotonic() - t0 < 1.0


def test_settled_efforts_defaults_keep_the_settle():
    import inspect

    params = inspect.signature(client_mod.PlannerClient.settled_efforts).parameters
    assert params["settle_s"].default == 0.15
    assert params["duration_s"].default == 0.15


def _compliant_run(monkeypatch, guard, fresh=True, gains=None, client_gains=None):
    """A compliant client flying one guarded leg; the result settles after a
    dozen spins. `fresh`: each spin serves a /joint_states. `client_gains`:
    the client's own compliant_gains (else it has none set, as _client
    builds it, and falls back on constants.COMPLIANT_RUN_GAINS)."""
    h = _ScriptedHandle(_cresp(CancelGoal.Response.ERROR_NONE), _ended(5, -6))
    d = _ScriptedDriver(h)
    c = _client(d)
    c.compliant = True
    if client_gains is not None:
        c.compliant_gains = client_gains
    spins = {"n": 0}

    def spin(node, timeout_sec=0.0):
        spins["n"] += 1
        if fresh:
            c._eff_at = time.monotonic()
        if spins["n"] == 12:
            h.result_fut.settle(_ended(4, 0))
        time.sleep(0.03 if not fresh else 0.0)

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    outcome, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0, guard=guard, gains=gains)
    return outcome, info, h, d


def _sent(gains):
    """What ImpedanceGains `gains` asks the driver for: a named profile's
    number, or PROFILE_CUSTOM's (4, kq, zeta, torque_limit) — the driver
    reads `custom` only then."""
    if gains.profile != 4:
        return gains.profile
    return (4, list(gains.custom.kq), gains.custom.zeta, list(gains.custom.torque_limit))


# constants.GAINS_RIGID as the driver must receive it, the numbers copied here
RIGID_SENT = (4, [400.0] * 7, 1.0, [39.0] * 4 + [9.0] * 3)


def test_a_compliant_leg_flies_in_impedance_under_its_gains(monkeypatch):
    """Owner 2026-10-27: "impedance mode needs to be firmer"; 2026-10-28,
    after 13:43's press under STIFF: "extreamly rigid"; 2026-10-29: "double
    the spring coefficient". A compliant goal that names no gains flies the
    run's, RIGID — PROFILE_CUSTOM with its own Kq 400 N·m/rad on every joint
    (the driver's ceiling), zeta 1.0 and torque limits 39/9 N·m — never the
    driver's session default, MEDIUM, which every goal of 16:23 flew; one
    that names a profile flies that one, as its number, and so does a
    client told another run default."""
    from rammp_arm_interfaces.msg import ImpedanceGains

    outcome, info, h, d = _compliant_run(monkeypatch, None)
    (goal,) = d.goals
    assert outcome == "arrived" and goal.control_mode == client_mod.IMPEDANCE
    assert client_mod.COMPLIANT_RUN_GAINS is client_mod.GAINS_RIGID
    assert goal.gains.profile == ImpedanceGains.PROFILE_CUSTOM and _sent(goal.gains) == RIGID_SENT
    assert info["gains"] is client_mod.GAINS_RIGID
    _o, info, _h, d = _compliant_run(monkeypatch, None, gains=client_mod.GAINS_MEDIUM)
    assert d.goals[0].gains.profile == 2 and info["gains"] == 2
    _o, info, _h, d = _compliant_run(monkeypatch, None, client_gains=client_mod.GAINS_MEDIUM)
    assert d.goals[0].gains.profile == 2 and info["gains"] == 2
    _o, info, _h, d = _compliant_run(
        monkeypatch, None, gains=client_mod.GAINS_STIFF, client_gains=client_mod.GAINS_SOFT
    )
    assert d.goals[0].gains.profile == 3 and info["gains"] == 3  # the leg's own wins
    _o, info, _h, d = _compliant_run(monkeypatch, None, gains=client_mod.GAINS_STIFF)
    assert _sent(d.goals[0].gains) == 3 and info["gains"] == 3  # STIFF named under the RIGID default: STIFF


def test_a_client_built_for_a_run_flies_rigid(monkeypatch):
    """PlannerClient.__init__ (the rclpy surface stubbed): position until a
    run sets compliant, RIGID once it does."""

    class _Node:
        def __getattr__(self, name):  # create_subscription, create_client, ...
            return lambda *a, **k: None

    for name in ("ActionClient", "Buffer", "TransformListener"):
        monkeypatch.setattr(client_mod, name, lambda *a, **k: None)
    c = client_mod.PlannerClient(_Node())
    assert c.compliant is False and c.compliant_gains is client_mod.GAINS_RIGID


class _Rejected:
    accepted = False


class _RejectingDriver(_Driver):
    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        return _Future(_Rejected())


def test_a_compliant_rejection_names_the_gains_it_sent():
    d = _RejectingDriver()
    c = _client(d)
    c.compliant = True
    outcome, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)
    assert outcome == "failed" and "its RIGID gains refused" in info["message"]
    c.compliant_gains = client_mod.GAINS_MEDIUM
    _o, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)
    assert "its MEDIUM gains refused" in info["message"]
    c.compliant = False  # a position goal sends none, so none is named
    _o, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)
    assert info["message"] == "goal rejected by the driver (a stream session open, or a mode change while moving)"
    assert info["gains"] is None and d.goals[-1].gains.profile == 0


def test_a_position_leg_sends_no_gains():
    d = _Driver()
    c = _client(d)
    c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0, gains=client_mod.GAINS_STIFF)
    assert d.goals[0].control_mode == client_mod.POSITION and d.goals[0].gains.profile == 0


def test_the_gains_named_here_are_the_drivers_own_profiles():
    """constants.GAINS_* are robot-body constants copied from the driver:
    they must be the profile numbers its interface defines, or a run would
    fly a tuning other than the one its banner and press model name. A goal
    that names none is PROFILE_SESSION_DEFAULT, which the driver resolves
    to MEDIUM. A custom spec (GAINS_RIGID) goes out as PROFILE_CUSTOM, and
    the same tables key it as they key a profile: its Kq, its name."""
    from rammp_arm_interfaces.msg import ImpedanceGains

    from box_opening import constants

    assert (constants.GAINS_SOFT, constants.GAINS_MEDIUM, constants.GAINS_STIFF, constants.GAINS_CUSTOM) == (
        ImpedanceGains.PROFILE_SOFT,
        ImpedanceGains.PROFILE_MEDIUM,
        ImpedanceGains.PROFILE_STIFF,
        ImpedanceGains.PROFILE_CUSTOM,
    )
    assert ExecuteJointTrajectory.Goal().gains.profile == ImpedanceGains.PROFILE_SESSION_DEFAULT
    rigid = constants.GAINS_RIGID
    assert set(constants.IMPEDANCE_KQ) == set(constants.IMPEDANCE_NAMES) == {1, 2, 3, rigid}
    assert constants.IMPEDANCE_KQ[rigid] == rigid.kq == (400.0,) * 7
    assert constants.IMPEDANCE_NAMES[rigid] == rigid.name == "RIGID"
    assert all(len(kq) == len(LIVE) for kq in constants.IMPEDANCE_KQ.values())
    assert constants.COMPLIANT_RUN_GAINS in constants.IMPEDANCE_KQ
    assert client_mod.GAINS_STIFF is constants.GAINS_STIFF  # client.GAINS_* still name them
    assert client_mod.GAINS_RIGID is rigid and client_mod.GAINS_CUSTOM == constants.GAINS_CUSTOM


def test_fill_gains_writes_a_custom_spec_as_profile_custom_and_a_profile_as_its_number():
    """client.fill_gains writes every goal's gains (the primer's, execute's;
    a release copies the last goal's): GAINS_RIGID as PROFILE_CUSTOM with
    its own kq, zeta and torque_limit, a named profile as its number with
    `custom` untouched — the driver reads `custom` only under
    PROFILE_CUSTOM."""
    from rammp_arm_interfaces.msg import ImpedanceGains

    msg = ImpedanceGains()
    client_mod.fill_gains(msg, client_mod.GAINS_RIGID)
    assert msg.profile == ImpedanceGains.PROFILE_CUSTOM == 4
    assert list(msg.custom.kq) == [400.0] * 7 and msg.custom.zeta == 1.0
    assert list(msg.custom.torque_limit) == [39.0] * 4 + [9.0] * 3
    for profile in (client_mod.GAINS_SOFT, client_mod.GAINS_MEDIUM, client_mod.GAINS_STIFF):
        msg = ImpedanceGains()
        client_mod.fill_gains(msg, profile)
        assert msg.profile == profile and msg.custom == ImpedanceGains().custom, profile
        assert list(msg.custom.kq) == [0.0] * 7 and msg.custom.zeta == 0.0, profile


# kinova-gen3-driver's PROFILE_CUSTOM bounds, copied: include/kinova_lowlevel/interface/gains.h
# (kKqMin/kKqMax, kZetaMin/kZetaMax) and src/interface/gains.cpp (kTorqueLimitFloor, the worst-case
# gravity on either mount x1.05; kTorqueLimitCeil, the URDF's effort limits), which validate_custom
# checks when the driver accepts a goal
DRIVER_KQ_RANGE = (1.0, 400.0)
DRIVER_ZETA_RANGE = (0.05, 2.0)
DRIVER_TORQUE_FLOOR = (18.8, 37.6, 14.6, 14.5, 3.7, 3.7, 0.5)
DRIVER_TORQUE_CEIL = (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0)


def test_every_custom_gains_spec_lies_within_the_drivers_profile_custom_bounds():
    """The driver refuses a PROFILE_CUSTOM goal with any value out of its
    bounds, with a reason, and never clamps (validate_custom, at accept):
    under the run's default, every compliant goal of the run, its primer
    first, would be refused. GAINS_RIGID: Kq at the 400 ceiling on joints
    1-4 and 200 on 5-7, zeta 1.0, each torque limit at its joint's ceiling,
    above the floor that holds gravity (joint 2: 39 against 37.6)."""
    from box_opening import constants

    custom = [spec for spec in constants.IMPEDANCE_KQ if isinstance(spec, constants.CustomGains)]
    assert constants.GAINS_RIGID in custom and constants.COMPLIANT_RUN_GAINS in custom
    for spec in custom:
        assert len(spec.kq) == len(spec.torque_limit) == len(LIVE), spec.name
        assert all(DRIVER_KQ_RANGE[0] <= kq <= DRIVER_KQ_RANGE[1] for kq in spec.kq), spec.name
        assert DRIVER_ZETA_RANGE[0] <= spec.zeta <= DRIVER_ZETA_RANGE[1], spec.name
        assert all(
            lo <= t <= hi for t, lo, hi in zip(spec.torque_limit, DRIVER_TORQUE_FLOOR, DRIVER_TORQUE_CEIL, strict=False)
        ), spec.name


def test_a_compliant_guard_watches_and_never_stops_the_leg(monkeypatch):
    """Owner 2026-10-27: nothing it reads stops the arm. A trip is noted,
    the leg flies to its end."""
    outcome, info, h, _d = _compliant_run(monkeypatch, _TripGuard())
    assert outcome == "arrived" and h.cancels == 0 and "would_trip" in info
    assert "would have stopped it" in info["message"]


def test_a_compliant_leg_flies_on_through_a_silent_stream(monkeypatch):
    outcome, _info, h, _d = _compliant_run(monkeypatch, _Guard(), fresh=False)
    assert outcome == "arrived" and h.cancels == 0


# -- the hold that primes the joint impedance (prime_compliant) ----------------


class _SilentDriver(_Driver):
    """Takes every goal and answers none of them, until `answering`: then
    each is accepted and arrives at once, as _Driver's. A primer that waited
    on its goal would hang here."""

    def __init__(self):
        super().__init__()
        self.futures = []
        self.answering = False

    def send_goal_async(self, goal, feedback_callback=None):
        if self.answering:
            return super().send_goal_async(goal, feedback_callback)
        self.goals.append(goal)
        self.futures.append(_Pending())
        return self.futures[-1]


def _priming(driver, armed=True):
    c = _client(driver, armed=armed)
    c.compliant = True
    return c


def test_the_impedance_is_primed_by_a_hold_where_the_arm_stands_and_nothing_waits_on_it(monkeypatch):
    """Flatbot 2026-10-27 16:23: the look, the run's first compliant goal,
    took 1.857 s against its 1.60 s preview — the driver's 0.25 s switch out
    of POSITION, then its spring fading in under the look. The primer is ONE
    impedance goal, queued as every goal is, holding the arm where it stands
    under the run's gains; the call spins nothing and waits for neither the
    goal's acceptance nor its result: the planning that follows serves it."""
    from box_opening.constants import JOINT_VMAX

    d = _SilentDriver()
    c = _priming(d)
    c._active_joints = ["arm_joint_%d" % i for i in range(1, 8)]  # flatbot's /joint_states names
    spins = []
    monkeypatch.setattr(client_mod.rclpy, "spin_once", lambda node, timeout_sec=0.0: spins.append(timeout_sec))
    monkeypatch.setattr(client_mod, "spin_until_done", lambda *a, **k: pytest.fail("the primer waited on its goal"))
    assert c.prime_compliant() is True
    (goal,) = d.goals
    assert spins == [] and c._primer is d.futures[0] and not d.futures[0].done()
    assert goal.control_mode == client_mod.IMPEDANCE and goal.preemption == client_mod.QUEUE
    assert _sent(goal.gains) == RIGID_SENT and goal.sender_id == "box_opening"
    hold = goal.trajectory
    assert list(hold.joint_names) == c._active_joints
    assert [_secs(p) for p in hold.points] == pytest.approx(list(client_mod.PlannerClient.PRIME_TIMES_S))
    assert client_mod.PlannerClient.PRIME_TIMES_S == (0.02, 0.05)
    for p in hold.points:  # two identical points at the live joints, at rest
        assert list(p.positions) == pytest.approx(LIVE)
        assert list(p.velocities) == [0.0] * 7 and list(p.accelerations) == [0.0] * 7
    assert client_mod.driver.refusal(hold, LIVE, JOINT_VMAX) is None
    # why the first point is not at t=0: the gates every goal passes refuse it
    assert "non-monotonic" in client_mod.driver.refusal(_traj(LIVE, LIVE, times=(0.0, 0.05)), LIVE, JOINT_VMAX)
    # the caller's own read is the one held, under whichever gains the run flies
    q = [v + 0.01 for v in LIVE]
    c.compliant_gains = client_mod.GAINS_MEDIUM
    assert c.prime_compliant(q) is True
    for p in d.goals[-1].trajectory.points:
        assert list(p.positions) == pytest.approx(q)
    assert d.goals[-1].gains.profile == client_mod.GAINS_MEDIUM
    # ... within the start gate of the live arm, as any goal's first waypoint
    assert c.prime_compliant([LIVE[0] + 0.1] + LIVE[1:]) is False and len(d.goals) == 2
    assert spins == []


def test_only_an_armed_compliant_client_with_no_abort_requested_primes():
    from box_opening.runtime.abort import AbortFlag

    d = _SilentDriver()
    assert _priming(d, armed=False).prime_compliant() is False  # no --execute
    assert _client(d).prime_compliant() is False  # a position client: --guarded, home_arm
    aborted = _priming(d)
    aborted._abort = AbortFlag()
    aborted._abort.requested = True  # Ctrl+C while the scene camera looked
    assert aborted.prime_compliant() is False
    assert d.goals == []


def test_no_hold_goes_out_on_a_frozen_joint_stream_or_without_the_driver(monkeypatch):
    """The hold is judged against the live arm, as every goal is
    (driver.refusal): where /joint_states has gone quiet (no sample newer
    than GUARD_STALE_S, and the spins serve none) none goes out, though it
    would hold the caller's own read; nor where the driver's action does not
    answer. The look then pays the switch itself, as before the primer."""
    monkeypatch.setattr(client_mod.rclpy, "spin_once", lambda node, timeout_sec=0.0: time.sleep(0.01))
    d = _SilentDriver()
    frozen = _priming(d)
    frozen._eff_at = time.monotonic() - 1.0  # the last sample a second old
    assert frozen.prime_compliant(list(LIVE)) is False
    gone = _SilentDriver()
    gone.wait_for_server = lambda timeout_sec=None: False
    assert _priming(gone).prime_compliant() is False
    assert d.goals == [] and gone.goals == []


def test_an_execute_after_the_primer_returns_its_own_result():
    """The primer's goal is never answered here; the look after it is, and
    that answer is the look's."""
    d = _SilentDriver()
    c = _priming(d)
    assert c.prime_compliant()
    d.answering = True
    end = [v + 0.05 for v in LIVE]
    outcome, info = c.execute(_traj(LIVE, end), 1.0)
    assert outcome == "arrived" and info["message"] == "arrived" and info["gains"] is client_mod.GAINS_RIGID
    primer, look = d.goals
    assert list(primer.trajectory.points[-1].positions) == pytest.approx(LIVE)
    assert list(look.trajectory.points[-1].positions) == pytest.approx(end)
    assert not d.futures[0].done()  # the primer's own answer: never come, never waited on


# -- a compliant stop lets go: the impedance reference moved to where the arm stands ----------

MID = [v + 0.02 for v in LIVE]  # where the arm stands when the Ctrl+C lands, part-way


def _ctrl_c_mid_goal(monkeypatch, compliant=True, flag=True, fresh=True, gains=None):
    """One leg flown by a client (compliant, else in position) under
    `gains`, Ctrl+C landing on the third spin with the arm at MID: through
    the abort flag (`flag`; SIGINT mid-goal only sets it), else raised from
    the spin, as rclpy's own handler does in a CLI that keeps it. With
    `fresh` False, /joint_states stopped a second before. The cancel is
    taken and the goal ends CANCELED. Returns (client, driver, handle) once
    execute has raised the KeyboardInterrupt."""
    h = _ScriptedHandle(_cresp(CancelGoal.Response.ERROR_NONE), _ended(5, -6))
    if flag:
        c, _flag, handler = _abort_client(h)
    else:
        c, handler = _client(_ScriptedDriver(h)), None
    c.compliant = compliant
    spins = {"n": 0}

    def spin(node, timeout_sec=0.0):  # a /joint_states message per spin, unless it stopped
        spins["n"] += 1
        c._eff_at = time.monotonic() if fresh or spins["n"] < 3 else min(c._eff_at, time.monotonic() - 1.0)
        if not fresh:
            time.sleep(0.005)
        if spins["n"] == 3:
            c._q = list(MID)
            if handler is None:
                raise KeyboardInterrupt
            handler(2, None)

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    with pytest.raises(KeyboardInterrupt):
        c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0, gains=gains)
    return c, c._execute, h


@pytest.mark.parametrize(
    "flag, gains, held_under",
    [
        (True, None, RIGID_SENT),  # the client's run default
        (False, client_mod.GAINS_SOFT, client_mod.GAINS_SOFT),  # the backstop's cancel, a leg under its own gains
    ],
    ids=["abort flag", "backstop"],
)
def test_a_compliant_ctrl_c_lets_the_spring_go_where_the_arm_stands(monkeypatch, flag, gains, held_under):
    """With the stop-the-arm checks off, Ctrl+C and the e-stop are the only
    stops, and a cancel moves nothing in the driver's joint impedance: it
    pulls on toward the cancelled goal's last reference, its velocity and
    acceleration feed-forward still on (kinova-gen3-driver 94736cd
    supervisor.cpp:278-296), the arm overshooting by centimetres at speed
    or pushing on into the box. After the cancel — the abort flag's,
    confirmed, or the backstop's, unconfirmed — exactly one more goal goes
    out: a hold at the joints the arm stands at, at rest, LATEST_WINS (a
    queued one would wait behind a goal whose cancel did not land), under
    the cancelled goal's own gains, PROFILE_CUSTOM's values included
    (none would be MEDIUM, five times RIGID's sag on joints 1-4)."""
    from box_opening.constants import JOINT_VMAX

    c, d, h = _ctrl_c_mid_goal(monkeypatch, flag=flag, gains=gains)
    assert h.cancels == 1
    leg, rel = d.goals
    assert leg.control_mode == client_mod.IMPEDANCE and leg.preemption == client_mod.QUEUE
    assert rel.control_mode == client_mod.IMPEDANCE == 1 and rel.preemption == client_mod.LATEST_WINS == 1
    assert rel.gains == leg.gains and rel.gains is not leg.gains and _sent(rel.gains) == held_under
    assert rel.sender_id == "box_opening" and list(rel.trajectory.joint_names) == NAMES
    points = rel.trajectory.points
    assert [list(p.positions) for p in points] == [MID, MID]  # where the arm stands, not where the leg went
    for p in points:
        assert list(p.velocities) == [0.0] * 7 and list(p.accelerations) == [0.0] * 7
    assert [_secs(p) for p in points] == pytest.approx(list(client_mod.PlannerClient.RELEASE_TIMES_S))
    assert _secs(points[0]) > 0.0 and client_mod.driver.refusal(rel.trajectory, MID, JOINT_VMAX) is None


def test_no_release_goes_out_on_a_frozen_joint_stream(monkeypatch, capsys):
    """/joint_states stopped a second before the Ctrl+C: a LATEST_WINS hold
    at that sample would pull the arm back to where it stood then. None
    goes out, and the console says the spring was not let go."""
    _c, d, h = _ctrl_c_mid_goal(monkeypatch, fresh=False)
    assert h.cancels == 1 and len(d.goals) == 1
    assert "no fresh /joint_states: not released" in capsys.readouterr().out


def test_a_position_ctrl_c_sends_no_release(monkeypatch):
    """Never for a POSITION goal: there is no spring to let go, and an
    IMPEDANCE hold would switch the driver's mode under the arm."""
    c, d, h = _ctrl_c_mid_goal(monkeypatch, compliant=False)
    assert h.cancels == 1 and [g.control_mode for g in d.goals] == [client_mod.POSITION]
    assert c.release() is False and len(d.goals) == 1  # nor when asked: the last goal sent was POSITION


def _flowing(monkeypatch, c):
    """/joint_states flowing: every spin serves `c` a fresh message."""

    def spin(node, timeout_sec=0.0):
        c._eff_at = time.monotonic()

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)


def test_one_release_per_goal_sent(monkeypatch):
    """The Ctrl+C's release is followed by the run's own (reach_press.let_go,
    unwinding): nothing more goes out. A goal sent after it, the primer
    too, is one a release may follow again."""
    c, d, _h = _ctrl_c_mid_goal(monkeypatch)
    assert len(d.goals) == 2
    assert c.release() is False and len(d.goals) == 2
    d = _Driver()
    c = _client(d)
    c.compliant = True
    _flowing(monkeypatch, c)
    c.compliant_gains = client_mod.GAINS_MEDIUM
    assert c.prime_compliant() is True
    assert c.release() is True and c.release() is False
    assert c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0, gains=client_mod.GAINS_STIFF)[0] == "arrived"
    assert c.release() is True and c.release() is False
    modes = [(g.control_mode, g.preemption, g.gains.profile) for g in d.goals]
    assert modes == [(1, 0, 2), (1, 1, 2), (1, 0, 3), (1, 1, 3)]  # primer, its release, the leg, its release


def test_an_unarmed_client_sends_no_release(monkeypatch):
    """The motion latch (--execute) gates the release too: an unarmed client
    sends nothing, whatever it sent before it was disarmed."""
    d = _Driver()
    c = _client(d)
    c.compliant = True
    _flowing(monkeypatch, c)
    assert c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)[0] == "arrived" and len(d.goals) == 1
    c._armed = False
    assert c.release() is False and len(d.goals) == 1
    c._armed = True  # the latch was all that held it
    assert c.release() is True and len(d.goals) == 2


def test_a_release_that_fails_never_costs_the_ctrl_c_it_rides_on(monkeypatch, capsys):
    """A release runs while a Ctrl+C unwinds: whatever goes wrong in it, here
    its send raising, is printed and swallowed, never raised over the
    KeyboardInterrupt press_demo exits 130 on, and it latches nothing."""

    def send(self, goal, feedback_callback=None):
        if self.goals:  # the leg went out; its release does not
            raise RuntimeError("the context is not valid")
        self.goals.append(goal)
        return _Future(self.handle)

    monkeypatch.setattr(_ScriptedDriver, "send_goal_async", send)
    c, d, h = _ctrl_c_mid_goal(monkeypatch)  # the KeyboardInterrupt all the same
    assert h.cancels == 1 and len(d.goals) == 1 and c._released is False
    assert "[client] not released (the context is not valid) — check the arm" in capsys.readouterr().out


class _TakesOneDriver(_Driver):
    """Takes the first goal and turns down every one after it."""

    def send_goal_async(self, goal, feedback_callback=None):
        if not self.goals:
            return super().send_goal_async(goal, feedback_callback)
        self.goals.append(goal)
        return _Future(_Rejected())


def test_a_release_the_driver_did_not_take_is_tried_again(monkeypatch, capsys):
    """A hold the driver turned down let nothing go: the next release (the
    run's own, reach_press.let_go, after the client's) sends it again."""
    d = _TakesOneDriver()
    c = _client(d)
    c.compliant = True
    _flowing(monkeypatch, c)
    assert c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)[0] == "arrived"
    assert c.release() is False and c.release() is False
    assert [g.preemption for g in d.goals] == [client_mod.QUEUE, client_mod.LATEST_WINS, client_mod.LATEST_WINS]
    assert capsys.readouterr().out.count("not released: the driver did not take the hold") == 2


# -- the flight log: what the arm said, kept at the price of an append ------------


def _joint_states(k, velocity=True):
    """/joint_states message `k` as the flatbot driver sends it: the arm's
    seven joints and the gripper's knuckle, stamped on its clock at 49 Hz."""
    from sensor_msgs.msg import JointState

    m = JointState()
    t = 1793216726.0 + k / 49.0
    m.header.stamp.sec = int(t)
    m.header.stamp.nanosec = int(round((t - int(t)) * 1e9))
    m.name = ["arm_joint_%d" % i for i in range(1, 8)] + ["arm_robotiq_85_left_knuckle_joint"]
    m.position = [0.001 * k + 0.1 * i for i in range(7)] + [0.8]
    m.velocity = [0.01 * k] * 7 + [float("nan")] if velocity else []
    m.effort = [1.0] * 7 + [float("nan")]
    return m


def _listening_client(monkeypatch):
    """PlannerClient.__init__ with the rclpy surface stubbed, as
    test_a_client_built_for_a_run_flies_rigid builds it."""

    class _Node:
        def __getattr__(self, name):  # create_subscription, create_client, ...
            return lambda *a, **k: None

    for name in ("ActionClient", "Buffer", "TransformListener"):
        monkeypatch.setattr(client_mod, name, lambda *a, **k: None)
    return client_mod.PlannerClient(_Node())


def test_the_joint_states_history_keeps_the_newest_in_order_and_hands_out_a_copy(monkeypatch):
    """What the arm does after an arrival is read from what the client heard
    (runner._note_settle): the newest JOINT_HISTORY_N messages — 3 s and
    more of a ~98 Hz stream, a state_publish_divisor of 1 — each with its
    stamp, the arm's joints, velocities and efforts in order, the knuckle
    left out."""
    c = _listening_client(monkeypatch)
    n = client_mod.PlannerClient.JOINT_HISTORY_N
    assert n >= 3 * 100
    for k in range(n + 50):
        c._js_cb(_joint_states(k))
    kept = c.joint_history()
    assert len(kept) == n
    stamps = [s.stamp for s in kept]
    assert stamps == sorted(stamps) and stamps[0] == pytest.approx(1793216726.0 + 50 / 49.0, abs=1e-6)  # 50 dropped
    assert all(a.t <= b.t for a, b in zip(kept, kept[1:], strict=False))
    k = n + 49
    assert kept[-1].q == pytest.approx([0.001 * k + 0.1 * i for i in range(7)]) and kept[-1].q == c.joints()
    assert kept[-1].qd == pytest.approx([0.01 * k] * 7) and kept[-1].effort == [1.0] * 7
    # a copy: what is heard after it changes nothing in it
    c._js_cb(_joint_states(n + 50))
    assert kept[-1].stamp == stamps[-1] and c.joint_history()[-1].stamp > stamps[-1]
    # since: only what was heard at or after it; a message without velocities keeps none
    time.sleep(0.002)
    mid = time.monotonic()
    c._js_cb(_joint_states(n + 51, velocity=False))
    (late,) = c.joint_history(since=mid)
    assert late.qd is None and late.stamp == pytest.approx(1793216726.0 + (n + 51) / 49.0, abs=1e-6)


class _ReportingDriver(_Driver):
    """Replays `n` feedbacks, each with its fraction and the joints it
    measured, as the driver's sampler does every tick."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        for k in range(self.n):
            fb = ExecuteJointTrajectory.Feedback(
                fraction_complete=k / self.n, actual=JointTrajectoryPoint(positions=[k * 1e-3] * 7)
            )
            feedback_callback(SimpleNamespace(feedback=fb))
        return _Future(_GoalHandle(0, ""))


def test_every_tenth_feedback_is_kept_with_its_fraction_and_when_it_was_heard():
    """The driver's 1 kHz feedback was heard and deserialized for its
    fraction alone: every 10th now keeps the joints it measured too, for the
    run record's side file — and every one still moves the progress."""
    driver = _ReportingDriver(95)
    c = _client(driver)
    t0 = time.monotonic()
    end = [v + 0.05 for v in LIVE]
    outcome, info = c.execute(_traj(LIVE, end), 1.0)
    assert outcome == "arrived"
    kept = list(info["feedback_q"])
    assert [round(f * 95) for _t, f, _q in kept] == list(range(0, 95, 10))  # the 1st, the 11th, ... the 91st
    for (_t, _f, q), k in zip(kept, range(0, 95, 10), strict=False):
        assert q == pytest.approx([k * 1e-3] * 7)
    assert t0 <= info["sent_at"] <= kept[0][0] <= kept[-1][0] <= time.monotonic()
    assert list(info["sent_traj"].points[-1].positions) == pytest.approx(end)
    assert info["progress"] == pytest.approx(94 / 95)


def test_the_feedback_kept_is_bounded_to_the_newest(monkeypatch):
    monkeypatch.setattr(client_mod.PlannerClient, "FEEDBACK_KEEP", 5)
    _outcome, info = _client(_ReportingDriver(200)).execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)
    assert [round(f * 200) for _t, f, _q in info["feedback_q"]] == [150, 160, 170, 180, 190]


def test_a_refused_trajectory_reports_nothing_sent():
    away = [LIVE[0] + 0.1] + LIVE[1:]
    outcome, info = _client().execute(_traj(away, away), 1.0)
    assert outcome == "failed" and "sent_at" not in info and "feedback_q" not in info


def test_every_goal_that_goes_to_the_driver_is_timed():
    """An arrival's settle window ends where the next goal begins
    (runner._note_settle): every execution's send and a release's are
    kept, oldest first."""
    from collections import deque

    c = _client(_Driver())
    c._sent_at = deque(maxlen=client_mod.PlannerClient.GOALS_KEPT)
    c.compliant = True
    t0 = time.monotonic()
    _outcome, info = c.execute(_traj(LIVE, [v + 0.05 for v in LIVE]), 1.0)
    assert c.release()
    sent = c.goals_sent()
    assert len(sent) == 2 and t0 <= sent[0] <= info["sent_at"] <= sent[1] <= time.monotonic()


def test_a_ctrl_c_leaves_what_went_out_where_the_runner_keeps_it_from(monkeypatch):
    """A Ctrl+C raises through execute once the cancel and the release are
    out, with nothing returned: what the call kept is last_execution, from
    which the runner keeps the stopped motion's side file
    (runner._keep_aborted). A call stopped before anything went out leaves
    its own there, nothing sent — never the call before's."""
    t0 = time.monotonic()
    c, d, _h = _ctrl_c_mid_goal(monkeypatch)
    info = c.last_execution
    assert t0 <= info["sent_at"] <= time.monotonic() and len(d.goals) == 2  # the leg, its release
    assert list(info["sent_traj"].points[-1].positions) == pytest.approx([v + 0.05 for v in LIVE])
    assert list(info["feedback_q"]) == []  # this driver sends no feedback
    with pytest.raises(KeyboardInterrupt):  # the abort flag still up: the next leg never starts
        c.execute(_traj(MID, [v + 0.05 for v in LIVE]), 1.0)
    assert c.last_execution is not info and "sent_at" not in c.last_execution


# -- a still arm before a camera read at rest, or a grip's close ---------------

# 15:45's second look (2026-10-28, /joint_states at 49 Hz): the fastest |qd|
# of any joint at each reading from its arrival on (rad/s) — under 0.02 once
# where joint_2 turns back (+0.07 s), joint_2 creeping back from 45 mrad past
# its end at up to 0.13, then joint_6 — still from +0.72 s for good
LOOK_1545 = [
    0.070,
    0.038,
    0.024,
    0.018,
    0.050,
    0.065,
    0.087,
    0.093,
    0.105,
    0.116,
    0.119,
    0.129,
    0.125,
    0.124,
    0.123,
    0.122,
    0.121,
    0.104,
    0.101,
    0.088,
    0.086,
    0.080,
    0.075,
    0.071,
    0.066,
    0.060,
    0.065,
    0.054,
    0.052,
    0.043,
    0.045,
    0.033,
    0.032,
    0.030,
    0.027,
    0.017,
    0.015,
    0.012,
    0.008,
    0.003,
    0.002,
    0.000,
    0.000,
    0.000,
]


def _reading(k, speed, q2, velocity=True):
    """/joint_states message `k` of a 49 Hz stream, stamped on the
    driver's clock: joint_2 at `q2`, moving at `speed` (rad/s), the other
    joints at rest; without `velocity`, positions only."""
    from sensor_msgs.msg import JointState

    m = JointState()
    t = 1793216799.6768 + k / 49.0
    m.header.stamp.sec = int(t)
    m.header.stamp.nanosec = int(round((t - int(t)) * 1e9))
    m.name = ["arm_joint_%d" % i for i in range(1, 8)] + ["arm_robotiq_85_left_knuckle_joint"]
    m.position = [0.1, q2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    m.velocity = [0.0, -speed] + [0.0] * 5 + [float("nan")] if velocity else []
    m.effort = [1.0] * 7 + [float("nan")]
    return m


def _still_stream(monkeypatch, speeds, velocity=True):
    """A listening client on a scripted clock: each spin moves it one period
    of a 49 Hz stream and hears the next of `speeds` — joint_2 moving at it
    (_reading), None a spin that hears nothing — and past them the arm at
    rest. Returns (the client, the clock: its time and readings so far)."""
    c = _listening_client(monkeypatch)
    clock = {"t": 500.0, "k": 0, "q2": 0.3}
    monkeypatch.setattr(client_mod, "time", SimpleNamespace(monotonic=lambda: clock["t"]))

    def spin(node, timeout_sec=0.0):
        clock["t"] += 1.0 / 49.0
        k = clock["k"]
        clock["k"] += 1
        speed = speeds[k] if k < len(speeds) else 0.0
        if speed is not None:
            clock["q2"] -= speed / 49.0
            c._js_cb(_reading(k, speed, clock["q2"], velocity))

    monkeypatch.setattr(client_mod.rclpy, "spin_once", spin)
    return c, clock


def test_the_creep_after_the_look_is_not_still_and_the_arm_at_rest_after_it_is(monkeypatch):
    """15:45's second look, its own readings: the camera's still test passed
    frames while joint_2 crept back. Every joint under STILL_QD_RAD_S for
    STILL_HOLD_S is still — not the one reading under it where joint_2
    turns back, but the fifth in a row from +0.72 s, the 40th reading (the
    bag's +0.80 s: the creep over, joint_6 between two swings, the tool
    1.2 mm from where it stood when the way in left). Nothing goes to the
    driver: the wait only spins the node."""
    c, clock = _still_stream(monkeypatch, LOOK_1545)
    waited, timed_out = c.wait_still()
    assert not timed_out
    assert clock["k"] == 40 and waited == pytest.approx(40 / 49.0)
    assert c.goals_sent() == []
    # the same readings without velocities: their positions over their stamps
    c, clock = _still_stream(monkeypatch, LOOK_1545, velocity=False)
    assert c.wait_still() == (pytest.approx(40 / 49.0), False) and clock["k"] == 40


def test_a_joint_turning_back_through_zero_is_not_still(monkeypatch):
    """The way homes of 15:45 turned back through zero at +0.70-0.73 s, two
    readings under STILL_QD_RAD_S, and swung on 17 mm. Still is five
    readings in a row at 49 Hz (STILL_HOLD_S): four are not."""
    turn = [0.15, 0.06, 0.002, 0.009, 0.04, 0.10] + [0.0] * 5
    c, clock = _still_stream(monkeypatch, turn)
    assert not c.wait_still()[1] and clock["k"] == 11  # the fifth at rest, not the turn
    four = [0.10] + [0.01] * 4 + [0.10] + [0.0] * 5
    c, clock = _still_stream(monkeypatch, four)
    assert not c.wait_still()[1] and clock["k"] == 11


def test_a_wait_right_after_another_finds_the_arm_still_at_once_never_on_what_was_heard_before_a_goal(monkeypatch):
    """knob_look's recheck aims right after its aim, nothing flown between:
    the readings heard since say the arm stands still, and one fresh
    reading is enough. Nothing heard before the call, as after a long
    plan, takes five; so does what was heard before the last goal went
    out, which says nothing of the arm since, and a stream silent longer
    than STILL_GAP_S (the aim drops messages between frames) — shorter,
    the readings either side of it are one still stretch."""
    gap = int(client_mod.PlannerClient.STILL_GAP_S * 49) + 1
    c, clock = _still_stream(monkeypatch, [0.0] * 11 + [None] * 5 + [0.0] + [None] * gap + [0.0] * 20)
    assert c.wait_still() == (pytest.approx(5 / 49.0), False)  # nothing heard before it: five readings
    assert c.wait_still() == (pytest.approx(1 / 49.0), False) and clock["k"] == 6  # right after: one
    clock["t"] += 0.001
    c._mark_sent()  # a goal went out after the last reading was heard (the arm at rest all the same)
    assert c.wait_still() == (pytest.approx(5 / 49.0), False) and clock["k"] == 11
    assert c.wait_still() == (pytest.approx(6 / 49.0), False)  # silent 0.10 s: still since before it
    assert c.wait_still() == (pytest.approx((gap + 5) / 49.0), False)  # silent 0.27 s: five more


def test_the_wait_runs_out_carries_on_and_says_so_in_one_line(monkeypatch, capsys):
    """Nothing waits longer than STILL_TIMEOUT_S: the run carries on, one
    line saying which joint was still moving and how fast — or that no
    reading said."""
    c, _clock = _still_stream(monkeypatch, [0.05] * 200)
    waited, timed_out = c.wait_still()
    assert timed_out and waited == pytest.approx(client_mod.PlannerClient.STILL_TIMEOUT_S, abs=1.0 / 49.0)
    assert capsys.readouterr().out.splitlines() == [
        "[client] the arm is not still after 1.2 s (arm_joint_2 at 0.050 rad/s at the last reading) — carrying on"
    ]
    c, _clock = _still_stream(monkeypatch, [None] * 200)  # the stream silent
    assert c.wait_still()[1]
    assert capsys.readouterr().out.splitlines() == [
        "[client] the arm is not still after 1.2 s (no /joint_states reading says how fast it moves) — carrying on"
    ]


def test_a_ctrl_c_while_the_arm_settles_ends_the_run_as_a_ctrl_c(monkeypatch):
    """No goal is in flight while the wait spins, so a Ctrl+C raises at once
    (abort.sigint_handler), and raised inside a subscription's take it
    surfaces as RuntimeError (press_demo._spin_detect): the wait hands it
    on as the KeyboardInterrupt it is — main's exit 130, not a traceback.
    A RuntimeError with no Ctrl+C behind it goes on as it is."""
    from box_opening.runtime.abort import AbortFlag, sigint_handler

    c, _clock = _still_stream(monkeypatch, [0.05] * 200)
    c._abort = AbortFlag()
    handler = sigint_handler(c._abort)

    def ctrl_c_in_a_take(node, timeout_sec=0.0):
        try:
            handler(2, None)
        except KeyboardInterrupt as e:
            raise RuntimeError("the take failed") from e

    monkeypatch.setattr(client_mod.rclpy, "spin_once", ctrl_c_in_a_take)
    with pytest.raises(KeyboardInterrupt):
        c.wait_still()
    c._abort = AbortFlag()  # nobody pressed Ctrl+C

    def broken(node, timeout_sec=0.0):
        raise RuntimeError("the context is gone")

    monkeypatch.setattr(client_mod.rclpy, "spin_once", broken)
    with pytest.raises(RuntimeError, match="the context is gone"):
        c.wait_still()


def test_still_is_the_newest_readings_each_joint_under_the_limit_none_further_apart_than_the_gap():
    """still_for: from the first of the trailing run of readings under the
    limit, by their stamps (a message sent unstamped: when it was heard),
    to the newest; 0.0 when the newest moves; a reading further from the
    one before it than the gap begins the run again; the count stops at
    `enough`."""
    S = client_mod.JointSample
    rest, q = [0.0] * 7, [0.0] * 7
    gap = client_mod.PlannerClient.STILL_GAP_S
    run = [S(0.0, 10.0 + 0.02 * k, q, rest, None) for k in range(6)]
    assert client_mod.still_for(run, 0.02, gap) == pytest.approx(0.10)
    assert client_mod.still_for(run, 0.02, gap, enough=0.03) == pytest.approx(0.04)  # the first past 0.03
    moved = run[:3] + [S(0.0, 10.06, q, [0.0, 0.0, 0.0, 0.0, 0.0, 0.021, 0.0], None)] + run[4:]
    assert client_mod.still_for(moved, 0.02, gap) == pytest.approx(0.02)  # since the reading that moved
    assert client_mod.still_for(moved[:4], 0.02, gap) == 0.0
    late = run + [S(0.0, 10.10 + gap + 0.01, q, rest, None)]
    assert client_mod.still_for(late, 0.02, gap) == 0.0  # alone after the gap
    unstamped = [S(3.0 + 0.02 * k, 0.0, q, rest, None) for k in range(5)]
    assert client_mod.still_for(unstamped, 0.02, gap) == pytest.approx(0.08)
    assert client_mod.still_for([], 0.02, gap) == 0.0
