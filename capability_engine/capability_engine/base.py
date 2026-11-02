"""`self.base`: the drive base (the MIB), for a capability.

One velocity command: a body-frame twist, sent as rammp_base_interfaces/TwistSetpoint on
TWIST_TOPIC (best-effort, latest wins: rammp-rtps messages/control_message.hpp). One reading: the
base's nav_msgs/Odometry on ODOM_TOPIC. The MIB drives on TwistSetpoint only in its AUTONOMOUS or
SHARED state: request_mode() asks for one with a DriveCommand on DRIVE_COMMAND_TOPIC, resending it
until MibStatus on MIB_STATUS_TOPIC shows it (DriveCommand.msg's protocol). After that, a twist
raises BaseError once the MIB has left that state (a joystick touch ends AUTONOMOUS): the user's
decision, which nothing here re-requests. If the action switched the MIB into a mode and the MIB
is still in it, end_action() asks for DISABLE (IDLE), as that joystick touch would.

Like self.arm, it lives on the engine's node and callback group and makes its ROS entities inside
an action only: the setpoint publisher on the first send_twist() or stop(), the odometry
subscription on the first odometry(), the DriveCommand publisher and MibStatus subscription on
the first request_mode(), all in an ActionScope that end_action() closes. If the action published
a setpoint, end_action() first sends a zero twist STOP_REPEATS times (the topics are best-effort),
then the DISABLE, so a cancel, a failure or an exception leaves the base commanded still. Every
publish is under _lock, where end_action() marks the action ended before it stops the base, so
nothing is sent after that stop.

In mock mode it makes no ROS entity: send_twist() and request_mode() send nothing, and odometry()
is the commanded twist integrated from the origin of odom (dead reckoning), kept for the node's
lifetime.
"""

import math
import threading
import time

from nav_msgs.msg import Odometry
from rammp_base_interfaces.msg import DriveCommand, MibStatus, TwistSetpoint
from rclpy.qos import qos_profile_sensor_data

from capability_engine.action_scope import ActionScope
from capability_engine.arm import SETPOINT_QOS, wait_until

TWIST_TOPIC = "/base/setpoint/twist"  # the MIB's rt/base/setpoint/twist
ODOM_TOPIC = "/base/odom"  # nav_msgs/Odometry: pose in odom, twist in base_link
ODOM_WAIT_S = 10.0  # how long odometry() waits for the first message of the action
ODOM_STALE_S = 0.5  # odometry older than this is an error
STOP_REPEATS = 3  # end_action() sends the zero twist this many times...
STOP_PERIOD_S = 0.02  # ...this far apart (and then the DISABLE, as often)
DRIVE_COMMAND_TOPIC = "/rammp/drive_command"  # the MIB's rt/rammp/drive_command
MIB_STATUS_TOPIC = "/rammp/mib/status"  # the MIB's rt/rammp/mib/status, every 500 ms
MIB_STATUS_STALE_S = 1.5  # three MibStatus periods: older than this is an error
MODE_WAIT_S = 3.0  # request_mode(): how long the MIB has to show each requested state
MODE_RESEND_S = 0.5  # request_mode(): resend an unanswered DriveCommand this often
# request_mode()'s modes: (DriveCommand request, the MibStatus system_state that answers it)
MODES = {
    "autonomous": (DriveCommand.REQUEST_AUTONOMOUS, MibStatus.SYSTEM_STATE_AUTONOMOUS),
    "shared": (DriveCommand.REQUEST_SHARED, MibStatus.SYSTEM_STATE_SHARED),
}
# The states the MIB accepts AUTONOMOUS and SHARED from (DriveCommand.msg).
ENTERABLE_FROM = {
    MibStatus.SYSTEM_STATE_IDLE,
    MibStatus.SYSTEM_STATE_ENABLED,
    MibStatus.SYSTEM_STATE_PUSH_ASSIST,
}
STATE_NAMES = {
    getattr(MibStatus, n): n.removeprefix("SYSTEM_STATE_")
    for n in dir(MibStatus)
    if n.startswith("SYSTEM_STATE_")
}
NOT_IN_ACTION_MESSAGE = "the base is only available inside an action"
NO_ODOM_MESSAGE = f"no {ODOM_TOPIC} -- is the base up, and ROS_DOMAIN_ID exported?"
NO_STATUS_MESSAGE = f"no {MIB_STATUS_TOPIC} -- is the MIB up, and ROS_DOMAIN_ID exported?"
MOCK_NOT_SENT = "mock: base twists not sent"


class BaseError(Exception):
    """A base call failed: outside an action, no fresh odometry or MibStatus, the MIB refused or
    left the requested mode."""


class BaseInterface:
    """`self.base`: the drive base's mode, velocity commands and odometry, on the engine's
    node."""

    def __init__(self, node, callback_group, *, mock: bool = False):
        self._node = node
        self._group = callback_group
        self._mock = mock
        self._lock = threading.Lock()  # everything below
        self._ctx = None  # the running action's; None: no action
        self._scope = None
        self._pub = None  # made by the first send_twist() or stop() of the action
        self._cmd_pub = None  # the DriveCommand publisher, made by the first request_mode()
        self._subs = set()  # the topics this action subscribed to
        self._latest = {}  # topic -> (message, time.monotonic() received), this action's
        self._mode = None  # the system_state request_mode() entered this action, if any
        self._mock_logged = False
        self._mock_pose = [0.0, 0.0, 0.0]  # mock mode: x, y, yaw in odom
        self._mock_twist = (0.0, 0.0, 0.0)  # mock mode: the commanded vx, vy, wz in base_link
        self._mock_t = time.monotonic()  # mock mode: when _mock_pose was last advanced

    # -- lifecycle ----------------------------------------------------------------------------

    def begin_action(self, ctx):
        """Start an action: base calls are allowed until end_action()."""
        self.end_action()  # the last action never ended: carry nothing
        with self._lock:
            self._ctx, self._scope = ctx, ActionScope(self._node, self._group)

    def end_action(self):
        """End the action: if it published a setpoint, send the zero twist STOP_REPEATS times;
        if it switched the MIB into a mode it is still in, then DISABLE as often; then destroy its
        entities and forget what they received. Takes at most worst_settle_s()."""
        with self._lock:
            sends = []  # (publisher, message), in order
            if self._pub is not None:
                sends += [(self._pub, _setpoint(0.0, 0.0, 0.0))] * STOP_REPEATS
            status = self._latest.get(MIB_STATUS_TOPIC, (None,))[0]
            # Only a mode this action switched the MIB into, and only while the MIB is in it.
            entered = self._cmd_pub is not None and status is not None
            if entered and self._mode == status.system_state:
                disable = DriveCommand(
                    request=DriveCommand.REQUEST_DISABLE, profile=status.active_profile
                )
                sends += [(self._cmd_pub, disable)] * STOP_REPEATS
            scope = self._scope
            self._ctx = self._scope = self._pub = self._cmd_pub = self._mode = None
            self._subs, self._latest = set(), {}
            self._mock_logged = False
            if self._mock:
                self._mock_command((0.0, 0.0, 0.0))
        try:
            for i, (pub, msg) in enumerate(sends):
                if i:
                    time.sleep(STOP_PERIOD_S)
                pub.publish(msg)
        finally:
            if scope is not None:
                scope.close()

    @staticmethod
    def worst_settle_s() -> float:
        """How long end_action() may take: the spacing of its zero twists and DISABLEs."""
        return (2 * STOP_REPEATS - 1) * STOP_PERIOD_S

    # -- commands -----------------------------------------------------------------------------

    def send_twist(self, linear_x: float, angular_z: float, linear_y: float = 0.0):
        """Command the base's velocity in base_link (REP 103): `linear_x` forward (m/s),
        `angular_z` counter-clockwise seen from above (rad/s), `linear_y` left (m/s; a
        differential drive ignores it). One message: resend it at a steady rate while you drive,
        and stop() (or leave the action) to halt. Never blocks. Raises Cancelled once the action
        is cancelled, BaseError outside an action or once the MIB has left the mode
        request_mode() entered."""
        with self._lock:
            ctx = self._check_in_action()
            if ctx.cancelled():
                ctx.checkpoint()  # raises the engine's Cancelled; end_action() stops the base
            if self._mode is not None:
                state = self._latest[MIB_STATUS_TOPIC][0].system_state
                if state != self._mode:
                    self._fail(
                        f"the MIB left {STATE_NAMES[self._mode]} for {_state_name(state)} "
                        "(a joystick touch, an e-stop or a fault): not driving"
                    )
            self._publish((float(linear_x), float(linear_y), float(angular_z)))

    def stop(self):
        """Command a zero twist, even once the action is cancelled. BaseError outside an action."""
        with self._lock:
            self._check_in_action()
            self._publish((0.0, 0.0, 0.0))

    def _publish(self, twist):
        """Send (vx, vy, wz), under _lock in an action."""
        if self._mock:
            if not self._mock_logged:
                self._node.get_logger().info(MOCK_NOT_SENT)
                self._mock_logged = True
            self._mock_command(twist)
            return
        if self._pub is None:
            self._pub = self._scope.create_publisher(TwistSetpoint, TWIST_TOPIC, SETPOINT_QOS)
        self._pub.publish(_setpoint(*twist))

    def request_mode(self, mode: str):
        """Ask the MIB to drive on this action's twists: "autonomous" (the Jetson alone; a
        joystick touch ends it) or "shared" (the Jetson and the joystick together). Returns once
        MibStatus shows it, keeping the drive profile in force. From the other of the two it
        first asks for DISABLE (IDLE), as the MIB accepts neither from the other. Honours a pause
        before it starts. Raises Cancelled once the action is cancelled, BaseError outside an
        action, for another mode, when the MIB cannot enter it now (INITIALIZING, ESTOP, ERROR),
        or when MibStatus does not show it within MODE_WAIT_S."""
        if mode not in MODES:
            raise BaseError(f"request_mode: {mode!r} is not one of {sorted(MODES)}")
        request, target = MODES[mode]
        with self._lock:
            ctx = self._check_in_action()
        ctx.checkpoint()
        if self._mock:
            with self._lock:
                self._check_in_action()
                self._node.get_logger().info(f"mock: drive mode {mode!r} not requested")
            return
        status = self._read(
            MIB_STATUS_TOPIC, MibStatus, MODE_WAIT_S, MIB_STATUS_STALE_S, NO_STATUS_MESSAGE, ctx
        )
        state = status.system_state
        if state != target:
            if state in (MibStatus.SYSTEM_STATE_AUTONOMOUS, MibStatus.SYSTEM_STATE_SHARED):
                idle = MibStatus.SYSTEM_STATE_IDLE
                state = self._drive(DriveCommand.REQUEST_DISABLE, status, idle, ctx)
            if state not in ENTERABLE_FROM:
                self._fail(
                    f"request_mode {mode!r}: the MIB is in {_state_name(state)} "
                    f"({status.error_message})"
                )
            self._drive(request, status, target, ctx)
        with self._lock:
            self._check_in_action()
            self._mode = target

    def _drive(self, request, status, target, ctx):
        """Send DriveCommand `request` (with `status`'s profile) every MODE_RESEND_S until
        MibStatus shows `target`, for up to MODE_WAIT_S; returns `target`."""
        msg = DriveCommand(request=request, profile=status.active_profile)
        deadline = time.monotonic() + MODE_WAIT_S
        while True:
            with self._lock:
                self._check_in_action()
                if self._cmd_pub is None:
                    self._cmd_pub = self._scope.create_publisher(
                        DriveCommand, DRIVE_COMMAND_TOPIC, SETPOINT_QOS
                    )
                self._cmd_pub.publish(msg)
            wait_s = min(MODE_RESEND_S, max(0.0, deadline - time.monotonic()))
            if wait_until(lambda: self._latest_state() == target, wait_s, ctx):
                return target
            if time.monotonic() >= deadline:
                self._fail(
                    f"the MIB did not enter {STATE_NAMES[target]} within {MODE_WAIT_S} s: "
                    f"it is in {_state_name(self._latest_state())}"
                )

    # -- state --------------------------------------------------------------------------------

    def odometry(self) -> Odometry:
        """The base's latest nav_msgs/Odometry: pose in odom, twist in base_link. The first call
        of an action subscribes to ODOM_TOPIC and waits up to ODOM_WAIT_S for a message. Raises
        Cancelled once the action is cancelled, BaseError outside an action, when no odometry
        comes, or when the latest is older than ODOM_STALE_S."""
        with self._lock:
            ctx = self._check_in_action()
            if self._mock:
                return self._mock_odometry()
        return self._read(ODOM_TOPIC, Odometry, ODOM_WAIT_S, ODOM_STALE_S, NO_ODOM_MESSAGE, ctx)

    def _read(self, topic, msg_type, wait_s, stale_s, missing, ctx):
        """The latest `msg_type` on `topic`: the action's first read subscribes and waits up to
        `wait_s` for one. BaseError (`missing`) when none comes, or when it is older than
        `stale_s`; Cancelled once the action is cancelled."""
        with self._lock:
            self._check_in_action()
            if topic not in self._subs:
                self._scope.create_subscription(
                    msg_type, topic, lambda m: self._store(topic, m), qos_profile_sensor_data
                )
                self._subs.add(topic)
        if not wait_until(lambda: self._received(topic) is not None, wait_s, ctx):
            self._fail(missing)
        msg, received = self._received(topic)
        age = time.monotonic() - received
        if age > stale_s:
            self._fail(f"{topic} is {age:.2f} s old (limit {stale_s} s)")
        return msg

    def _store(self, topic, msg):
        with self._lock:
            if self._ctx is not None:
                self._latest[topic] = (msg, time.monotonic())

    def _received(self, topic):
        """This action's latest (message, received) on `topic`, or None; BaseError once it has
        ended."""
        with self._lock:
            self._check_in_action()
            return self._latest.get(topic)

    def _latest_state(self):
        """The latest MibStatus system_state of this action, or None."""
        entry = self._received(MIB_STATUS_TOPIC)
        return None if entry is None else entry[0].system_state

    def _check_in_action(self):
        """The running action's ctx; BaseError when none runs."""
        if self._ctx is None:
            self._fail(NOT_IN_ACTION_MESSAGE)
        return self._ctx

    def _fail(self, message):
        self._node.get_logger().error(message)
        raise BaseError(message)

    # -- mock mode ----------------------------------------------------------------------------

    def _mock_command(self, twist):
        """Advance the mock pose under the twist commanded so far, then command `twist`."""
        now = time.monotonic()
        dt, self._mock_t = now - self._mock_t, now
        vx, vy, wz = self._mock_twist
        x, y, yaw = self._mock_pose
        self._mock_pose = [
            x + (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt,
            y + (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt,
            yaw + wz * dt,
        ]
        self._mock_twist = twist

    def _mock_odometry(self) -> Odometry:
        self._mock_command(self._mock_twist)
        x, y, yaw = self._mock_pose
        m = Odometry()
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = "odom", "base_link"
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        vx, vy, wz = self._mock_twist
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z = vx, vy, wz
        return m


def _state_name(state) -> str:
    return "no MibStatus" if state is None else STATE_NAMES.get(state, f"state {state}")


def _setpoint(vx, vy, wz) -> TwistSetpoint:
    """A TwistSetpoint with no token (all zeros): ExecuteAction carries none yet, as for the arm."""
    m = TwistSetpoint()
    m.twist.linear.x, m.twist.linear.y, m.twist.angular.z = vx, vy, wz
    return m
