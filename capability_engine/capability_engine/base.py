"""`self.base`: the drive base (the MIB), for a capability.

One command: a body-frame velocity, sent as rammp_base_interfaces/TwistSetpoint on TWIST_TOPIC
(best-effort, latest wins: rammp-rtps messages/control_message.hpp). One reading: the base's
nav_msgs/Odometry on ODOM_TOPIC. The MIB drives on TwistSetpoint only in its AUTONOMOUS or SHARED
state; entering one (DriveCommand) is not this interface's job, and it does not check the state.

Like self.arm, it lives on the engine's node and callback group and makes its ROS entities inside
an action only: the setpoint publisher on the first send_twist() or stop(), the odometry
subscription on the first odometry(), both in an ActionScope that end_action() closes. If the
action published a setpoint, end_action() first sends a zero twist STOP_REPEATS times (the topic
is best-effort), so a cancel, a failure or an exception leaves the base commanded still. Every
publish is under _lock, where end_action() marks the action ended before it stops the base, so
nothing is sent after that stop.

In mock mode it makes no ROS entity: send_twist() sends nothing, and odometry() is the commanded
twist integrated from the origin of odom (dead reckoning), kept for the node's lifetime.
"""

import math
import threading
import time

from nav_msgs.msg import Odometry
from rammp_base_interfaces.msg import TwistSetpoint
from rclpy.qos import qos_profile_sensor_data

from capability_engine.action_scope import ActionScope
from capability_engine.arm import SETPOINT_QOS, wait_until

TWIST_TOPIC = "/base/setpoint/twist"  # the MIB's rt/base/setpoint/twist
ODOM_TOPIC = "/base/odom"  # nav_msgs/Odometry: pose in odom, twist in base_link
ODOM_WAIT_S = 10.0  # how long odometry() waits for the first message of the action
ODOM_STALE_S = 0.5  # odometry older than this is an error
STOP_REPEATS = 3  # end_action() sends the zero twist this many times...
STOP_PERIOD_S = 0.02  # ...this far apart
NOT_IN_ACTION_MESSAGE = "the base is only available inside an action"
NO_ODOM_MESSAGE = f"no {ODOM_TOPIC} -- is the base up, and ROS_DOMAIN_ID exported?"
MOCK_NOT_SENT = "mock: base twists not sent"


class BaseError(Exception):
    """A base call failed: outside an action, or no fresh odometry."""


class BaseInterface:
    """`self.base`: velocity commands to the drive base and its odometry, on the engine's node."""

    def __init__(self, node, callback_group, *, mock: bool = False):
        self._node = node
        self._group = callback_group
        self._mock = mock
        self._lock = threading.Lock()  # everything below
        self._ctx = None  # the running action's; None: no action
        self._scope = None
        self._pub = None  # made by the first send_twist() or stop() of the action
        self._subscribed = False
        self._odom = None  # (Odometry, time.monotonic() received), this action's
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
        then destroy its entities and forget its odometry. Takes at most worst_settle_s()."""
        with self._lock:
            pub, scope = self._pub, self._scope
            self._ctx = self._scope = self._pub = self._odom = None
            self._subscribed = self._mock_logged = False
            if self._mock:
                self._mock_command((0.0, 0.0, 0.0))
        try:
            if pub is not None:
                for i in range(STOP_REPEATS):
                    if i:
                        time.sleep(STOP_PERIOD_S)
                    pub.publish(_setpoint(0.0, 0.0, 0.0))
        finally:
            if scope is not None:
                scope.close()

    @staticmethod
    def worst_settle_s() -> float:
        """How long end_action() may take: the spacing of its zero twists."""
        return (STOP_REPEATS - 1) * STOP_PERIOD_S

    # -- commands -----------------------------------------------------------------------------

    def send_twist(self, linear_x: float, angular_z: float, linear_y: float = 0.0):
        """Command the base's velocity in base_link (REP 103): `linear_x` forward (m/s),
        `angular_z` counter-clockwise seen from above (rad/s), `linear_y` left (m/s; a
        differential drive ignores it). One message: resend it at a steady rate while you drive,
        and stop() (or leave the action) to halt. Never blocks. Raises Cancelled once the action
        is cancelled, BaseError outside an action."""
        with self._lock:
            ctx = self._check_in_action()
            if ctx.cancelled():
                ctx.checkpoint()  # raises the engine's Cancelled; end_action() stops the base
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
            if not self._subscribed:
                self._scope.create_subscription(
                    Odometry, ODOM_TOPIC, self._on_odom, qos_profile_sensor_data
                )
                self._subscribed = True
        if not wait_until(lambda: self._latest() is not None, ODOM_WAIT_S, ctx):
            self._fail(NO_ODOM_MESSAGE)
        msg, received = self._latest()
        age = time.monotonic() - received
        if age > ODOM_STALE_S:
            self._fail(f"{ODOM_TOPIC} is {age:.2f} s old (limit {ODOM_STALE_S} s)")
        return msg

    def _on_odom(self, msg):
        with self._lock:
            if self._ctx is not None:
                self._odom = (msg, time.monotonic())

    def _latest(self):
        """This action's latest (Odometry, received), or None; BaseError once it has ended."""
        with self._lock:
            self._check_in_action()
            return self._odom

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


def _setpoint(vx, vy, wz) -> TwistSetpoint:
    """A TwistSetpoint with no token (all zeros): ExecuteAction carries none yet, as for the arm."""
    m = TwistSetpoint()
    m.twist.linear.x, m.twist.linear.y, m.twist.angular.z = vx, vy, wz
    return m
