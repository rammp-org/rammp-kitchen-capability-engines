"""self.arm on the real arm: the cell's driver and cuRobo planner, checked against what a
capability's author is told to expect (README "API", CONTRIBUTING "Using self.arm").

Every check runs inside a real action of a standalone CapabilityEngine, the way a capability's
execute() runs, and is judged by an Observer that reads the driver's topics itself, never
through self.arm.

Gates: a test that moves the arm or changes the cell (marked `motion`) skips without --motion;
one that needs a person to act (marked `manual`) skips without --manual. A person stays at the
e-stop for every --motion run. Run it on the cell with `make test-hw`; see README.md here.
"""

import itertools
import os
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import rammp_arm
import rclpy
from rammp_arm_interfaces.msg import EeState, GripperState, StreamStatus
from rammp_common_interfaces.msg import ControlStatus, EStop
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import Result
from rammp_orchestrator_interfaces.srv import Control
from rclpy.action import ActionClient, get_action_names_and_types
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from capability_engine.engine import Action, CapabilityEngine

# The driver's "home" preset (Kinova's factory Home): every motion test starts here.
HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.96, 1.571]
WAIT_S = 5.0  # a server, a goal's acceptance, a status change
ACTION_WAIT_S = 90.0  # one action's result
POLL_S = 0.01
STATE_FRESH_S = 0.5  # a state sample older than this is not "now"
STILL_RAD_S = 0.01  # every joint slower than this ...
STILL_FOR_S = 0.2  # ... for this long: the arm is still
MOVING_RAD_S = 0.02  # a joint faster than this: the arm is moving
STOP_S = 1.0  # an interrupt (cancel, stop, e-stop) to the arm still
AT_HOME_DEG = 1.0  # within this of HOME: no reset move needed
# The envelope: once a motion test has started at home, the Observer e-stops the arm (and the
# session ends) if the tool leaves GUARD_TOOL_M of its home position or any of J1..J6 leaves
# GUARD_JOINT_DEG of home. J7 only turns the tool about its own axis, so it is free.
GUARD_TOOL_M = 0.15
GUARD_JOINT_DEG = 25.0
SPEED_SPAN_S = 0.1  # tool speeds are measured over this span, not between bunched samples
HISTORY = 50 * 300  # samples kept per topic: five minutes at 50 Hz
LATCHED = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)
ENGINE_IDS = itertools.count()


def pytest_addoption(parser):
    parser.addoption("--motion", action="store_true", help="allow tests that move the arm")
    parser.addoption("--manual", action="store_true", help="allow tests that need a person")
    parser.addoption(
        "--home",
        default=",".join(map(str, HOME)),
        help="the joints (rad, comma-separated) every motion test starts from",
    )
    parser.addoption(
        "--world",
        default="world_real_bench.yaml",
        help="the cell's cuRobo world: loaded before any --motion test, and left loaded",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "motion: moves the arm or changes the cell (--motion)")
    config.addinivalue_line("markers", "manual: needs a person at the cell (--manual)")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "motion" in item.keywords and not config.getoption("--motion"):
            item.add_marker(pytest.mark.skip(reason="moves the arm: needs --motion"))
        if "manual" in item.keywords and not config.getoption("--manual"):
            item.add_marker(pytest.mark.skip(reason="needs a person: needs --manual"))


def say(message):
    """To the person at the cell, past pytest's capture."""
    sys.__stderr__.write(f"\n>>> {message}\n")
    sys.__stderr__.flush()


def wait_until(condition, timeout_s=WAIT_S, what="condition"):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, f"timed out after {timeout_s:g} s waiting for {what}"
        time.sleep(POLL_S)


def joint_err(a, b):
    """|a - b| per joint (rad), the short way round for the continuous joints."""
    d = np.asarray(a, float) - np.asarray(b, float)
    for i in rammp_arm.CONTINUOUS:
        d[i] = (d[i] + np.pi) % (2 * np.pi) - np.pi
    return np.abs(d)


def angle_deg(qa, qb):
    """The rotation (deg) between two xyzw quaternions."""
    return float(np.degrees((R.from_quat(qa).inv() * R.from_quat(qb)).magnitude()))


def diagnostic(diagnostics, task):
    """The /diagnostics status of the driver's task `task` ("Arm", "Arm link", ...), or None. The
    driver names each "<node>: <task>"."""
    for d in diagnostics.status:
        if d.name == task or d.name.endswith(f": {task}"):
            return d
    return None


def tool_of(ee):
    """The tool frame of an /ee_state message (the flange + TOOL_Z along the flange's z):
    (position, quat xyzw) in arm_mount."""
    p, o = ee.pose.position, ee.pose.orientation
    quat = np.array([o.x, o.y, o.z, o.w])
    pos = np.array([p.x, p.y, p.z]) + R.from_quat(quat).apply([0.0, 0.0, rammp_arm.TOOL_Z])
    return pos, quat


class Observer(Node):
    """The cell as the driver reports it, read independently of self.arm. It also stands in for
    the operator: it publishes /estop and calls /clear_faults, which a capability never does."""

    def __init__(self):
        super().__init__("hw_test_observer")
        self._lock = threading.Lock()
        self._joints = deque(maxlen=HISTORY)  # (t, q, qd): arm_joint_1..7, as reported
        self._tool = deque(maxlen=HISTORY)  # (t, position, quat xyzw): the tool frame
        self._grip = deque(maxlen=HISTORY)  # (t, position 0..1)
        self.control = None  # the latest /control_status
        self.stream = None  # the latest /stream_status
        self.create_subscription(
            JointState, "/joint_states", self._on_joints, qos_profile_sensor_data
        )
        self.create_subscription(EeState, "/ee_state", self._on_ee, qos_profile_sensor_data)
        self.create_subscription(
            GripperState, "/gripper_state", self._on_grip, qos_profile_sensor_data
        )
        self.create_subscription(
            ControlStatus, "/control_status", lambda m: setattr(self, "control", m), LATCHED
        )
        self.create_subscription(
            StreamStatus, "/stream_status", lambda m: setattr(self, "stream", m), LATCHED
        )
        self._estop = self.create_publisher(EStop, "/estop", 10)
        self._clear = self.create_client(Trigger, "/clear_faults")
        self.estop_engaged = False  # this session engaged it: the session's end clears it
        self.guard = None  # (home joints, home tool position) while the envelope is armed
        self.tripped = None  # why the envelope guard e-stopped the arm

    def _on_joints(self, m):
        idx = {n: i for i, n in enumerate(m.name)}
        if not all(n in idx for n in rammp_arm.JOINTS):
            return
        pick = [idx[n] for n in rammp_arm.JOINTS]
        q = np.array([m.position[i] for i in pick])
        qd = np.array([m.velocity[i] for i in pick]) if m.velocity else np.zeros(7)
        with self._lock:
            self._joints.append((time.monotonic(), q, qd))
        guard = self.guard
        if guard is not None:
            err = np.degrees(joint_err(q, guard[0])[:6])
            if err.max() > GUARD_JOINT_DEG:
                self._trip(f"J{int(err.argmax()) + 1} is {err.max():.0f} deg from home")

    def _on_ee(self, m):
        pos, quat = tool_of(m)
        with self._lock:
            self._tool.append((time.monotonic(), pos, quat))
        guard = self.guard
        if guard is not None and np.linalg.norm(pos - guard[1]) > GUARD_TOOL_M:
            self._trip(f"the tool is {np.linalg.norm(pos - guard[1]):.2f} m from home")

    def _trip(self, why):
        """Leave the envelope: e-stop the arm and keep it e-stopped (the session ends)."""
        self.guard = None
        m = EStop(engaged=True, source="hw_test", reason=f"envelope guard: {why}")
        m.header.stamp = self.get_clock().now().to_msg()
        self._estop.publish(m)
        self.estop_engaged = True
        self.tripped = why

    def _on_grip(self, m):
        with self._lock:
            self._grip.append((time.monotonic(), float(m.position)))

    # -- now ------------------------------------------------------------------------------------

    def _latest(self, buf, what):
        wait_until(lambda: len(buf) > 0, WAIT_S, f"the first {what}")
        with self._lock:
            sample = buf[-1]
        age = time.monotonic() - sample[0]
        assert age < STATE_FRESH_S, f"{what} is {age:.2f} s old"
        return sample

    def q(self):
        return self._latest(self._joints, "/joint_states")[1]

    def tool(self):
        """(position, quat xyzw) of the tool frame in arm_mount, now."""
        return self._latest(self._tool, "/ee_state")[1:]

    def gripper(self):
        return self._latest(self._grip, "/gripper_state")[1]

    # -- history --------------------------------------------------------------------------------

    def joints_since(self, t0, t1=None):
        """(t, q, qd) arrays of the /joint_states received from t0 (monotonic) to t1."""
        return self._since(self._joints, t0, t1)

    def tool_since(self, t0, t1=None):
        """(t, position, quat) arrays of the tool frame from t0 to t1."""
        return self._since(self._tool, t0, t1)

    def gripper_since(self, t0, t1=None):
        return self._since(self._grip, t0, t1)

    def _since(self, buf, t0, t1, required=True):
        t1 = time.monotonic() if t1 is None else t1
        with self._lock:
            rows = [s for s in buf if t0 <= s[0] <= t1]
        if not rows and not required:
            return None
        assert rows, "no samples in the window"
        return tuple(np.array(col) for col in zip(*rows, strict=True))

    def tool_speed(self, t0, t1=None):
        """(t, speed m/s) of the tool from t0 to t1, each over SPEED_SPAN_S: samples arrive
        bunched, so neighbouring ones give false spikes."""
        t, pos, _ = self.tool_since(t0, t1)
        j = np.searchsorted(t, t + SPEED_SPAN_S)
        ok = j < len(t)
        i, j = np.nonzero(ok)[0], j[ok]
        return t[i], np.linalg.norm(pos[j] - pos[i], axis=1) / (t[j] - t[i])

    def motion_span(self, t0, t1=None):
        """(first, last) monotonic time a joint moved faster than MOVING_RAD_S after t0, or None
        if the arm never moved."""
        t, _, qd = self.joints_since(t0, t1)
        moving = np.abs(qd).max(axis=1) > MOVING_RAD_S
        if not moving.any():
            return None
        return float(t[moving][0]), float(t[moving][-1])

    def still_now(self):
        rows = self._since(self._joints, time.monotonic() - STILL_FOR_S, None, required=False)
        return rows is not None and len(rows[0]) > 1 and bool(np.abs(rows[2]).max() < STILL_RAD_S)

    def wait_still(self, timeout_s=WAIT_S):
        """The monotonic time the arm came to rest (STILL_FOR_S of stillness), waiting for it."""
        wait_until(self.still_now, timeout_s, "the arm to be still")
        return time.monotonic() - STILL_FOR_S

    def wait_moving(self, timeout_s=WAIT_S):
        def moving():
            return np.abs(self._latest(self._joints, "/joint_states")[2]).max() > MOVING_RAD_S

        wait_until(moving, timeout_s, "the arm to move")
        return time.monotonic()

    # -- the operator's controls ----------------------------------------------------------------

    def set_estop(self, engaged):
        assert self.tripped is None, f"the envelope guard tripped: {self.tripped}"
        m = EStop(engaged=engaged, source="hw_test", reason="self.arm hardware test")
        m.header.stamp = self.get_clock().now().to_msg()
        self._estop.publish(m)
        self.estop_engaged = engaged
        wait_until(
            lambda: self.control is not None and self.control.estopped == engaged,
            WAIT_S,
            f"/control_status estopped={engaged}",
        )

    def clear_faults(self):
        assert self._clear.wait_for_service(timeout_sec=WAIT_S), "no /clear_faults service"
        future = self._clear.call_async(Trigger.Request())
        wait_until(future.done, 15.0, "/clear_faults")
        return future.result()

    def serves(self, action):
        return any(name == action for name, _ in get_action_names_and_types(self))


class HwEngine(CapabilityEngine):
    """A standalone capability whose one action runs `body(arm, ctx)` and keeps its outcome."""

    display_name = "self.arm hardware test"
    required_resources = ["arm", "gripper"]
    actions = [Action("run", "Run one test body")]

    def __init__(self, engine_id):
        super().__init__(
            engine_id, parameter_overrides=[Parameter("use_orchestrator", value=False)]
        )
        self.body = None
        self.outcome = None  # SimpleNamespace(value, raised) of the last body

    def execute(self, action_id, params, ctx):
        self.outcome = None
        try:
            value = self.body(self.arm, ctx)
        except Exception as error:  # Cancelled too: the engine reports it as usual
            self.outcome = SimpleNamespace(value=None, raised=error)
            raise
        except BaseException as error:  # pytest's skip or fail: kept for act(), and the
            # engine, which catches only Exception, sees a plain failure
            self.outcome = SimpleNamespace(value=None, raised=error)
            raise RuntimeError(repr(error)) from None
        self.outcome = SimpleNamespace(value=value, raised=None)
        return {}


class Cell:
    """The engine under test, a client that drives it like the orchestrator, and the Observer."""

    def __init__(self, home):
        self.home_q = home
        self.observer = Observer()
        self.client = Node("hw_test_client")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.observer)
        self.executor.add_node(self.client)
        self.spin_error = None  # an exception out of the executor: every later test is void
        self._spinner = threading.Thread(target=self._spin, daemon=True)
        self._spinner.start()
        self._engine = None
        self._action = self._control = None

    def _spin(self):
        try:
            self.executor.spin()
        except BaseException as error:
            self.spin_error = error
            raise

    @property
    def engine(self):
        """The engine; a new one after the last was stopped (a stopped engine takes no goal)."""
        if self._engine is None or self._engine._stopping is not None:
            # A stopped engine stays in the executor (it takes no goal): destroying a node, or
            # its clients, while the executor spins is unsafe on Humble.
            if self._engine is not None:
                self._wait_idle(self._engine)
            engine_id = f"hw_test_{os.getpid()}_{next(ENGINE_IDS)}"
            self._engine = HwEngine(engine_id)
            self.executor.add_node(self._engine)
            self._action = ActionClient(self.client, ExecuteAction, f"/{engine_id}/execute_action")
            self._control = self.client.create_client(Control, f"/{engine_id}/control")
            assert self._action.wait_for_server(timeout_sec=WAIT_S)
            assert self._control.wait_for_service(timeout_sec=WAIT_S)
        return self._engine

    @staticmethod
    def _wait_idle(engine):
        wait_until(lambda: engine._current is None, ACTION_WAIT_S, "the engine to idle")

    def abandon(self):
        """After a test: cancel an action it left running and wait for the arm to rest."""
        engine = self._engine
        if engine is None or engine._current is None:
            return
        engine.stop()
        self._wait_idle(engine)
        self.observer.wait_still(ACTION_WAIT_S)

    def start(self, body):
        """Send the engine a "run" goal that runs `body(arm, ctx)`: its accepted goal handle."""
        engine = self.engine
        engine.body = body
        send = self._action.send_goal_async(ExecuteAction.Goal(action_id="run"))
        wait_until(send.done, WAIT_S, "the goal's acceptance")
        handle = send.result()
        assert handle.accepted, "the engine refused the goal"
        return handle

    def finish(self, handle, timeout_s=ACTION_WAIT_S):
        """(the action's Result, the body's outcome)."""
        future = handle.get_result_async()
        wait_until(future.done, timeout_s, "the action's result")
        return future.result().result.result, self._engine.outcome

    def act(self, body, timeout_s=ACTION_WAIT_S):
        """Run `body(arm, ctx)` as one action: its return value, or what it raised, re-raised."""
        result, outcome = self.finish(self.start(body), timeout_s)
        assert outcome is not None, f"the body never ran: {result.status} {result.error_code}"
        if outcome.raised is not None:
            raise outcome.raised
        assert result.status == Result.STATUS_SUCCESS, (result.status, result.error_code)
        return outcome.value

    def control(self, command):
        future = self._control.call_async(Control.Request(command=command))
        wait_until(future.done, WAIT_S, "/control")
        return future.result()

    def go_home(self):
        """Bring the arm to the home joints (a planned go_to_joint_config) unless it is there,
        then arm the envelope guard around it."""
        self.observer.guard = None
        if joint_err(self.observer.q(), self.home_q).max() > np.radians(AT_HOME_DEG):
            ok = self.act(lambda arm, ctx: arm.go_to_joint_config(self.home_q))
            assert ok, "could not go home"
        self.observer.wait_still()
        self.observer.guard = (np.array(self.home_q), self.observer.tool()[0])

    def close(self):
        try:
            self.observer.guard = None
            if self._engine is not None:
                self._engine.stop()
                self._wait_idle(self._engine)
            # A tripped guard's e-stop stays: a person looks at the arm and clears it.
            if self.observer.estop_engaged and self.observer.tripped is None:
                self.observer.set_estop(False)
        finally:
            self.executor.shutdown()
            self._spinner.join(timeout=WAIT_S)
            self.observer.destroy_node()
            self.client.destroy_node()


@pytest.fixture(scope="session")
def cell(request):
    rclpy.init()
    home = [float(x) for x in request.config.getoption("--home").split(",")]
    assert len(home) == 7, "--home takes 7 joint angles"
    c = Cell(home)
    try:
        if request.config.getoption("--motion"):  # never move with an unknown planner world
            world = request.config.getoption("--world")
            ok, message = c.act(lambda arm, ctx: arm.set_world(world))
            assert ok, f"set_world({world}): {message}"
        yield c
    finally:
        c.close()
        rclpy.try_shutdown()


@pytest.fixture(autouse=True)
def tidy(request):
    """After each test: no action left running, the envelope disarmed. A tripped guard or a dead
    executor ends the session."""
    yield
    if "cell" not in request.fixturenames:
        return
    cell = request.getfixturevalue("cell")
    cell.observer.guard = None
    if cell.observer.tripped is not None:
        pytest.exit(f"envelope guard e-stopped the arm: {cell.observer.tripped}", returncode=3)
    if cell.spin_error is not None:
        pytest.exit(f"the executor died: {cell.spin_error!r}", returncode=3)
    cell.abandon()
    if cell.observer.estop_engaged:  # a test that failed before clearing its e-stop
        cell.observer.set_estop(False)


@pytest.fixture
def home(cell):
    """Start the test with the arm at rest at the home joints."""
    cell.go_home()
    return np.array(cell.home_q)
