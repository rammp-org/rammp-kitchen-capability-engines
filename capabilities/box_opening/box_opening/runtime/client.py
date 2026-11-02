"""The single ROS surface: the arm driver, the planner, the gripper, TF.

Two containers answer it, sheppy's arm module. The kinova-gen3-ros2 driver
executes (/execute_joint_trajectory), reports (/joint_states, the gripper
knuckle included) and takes gripper commands on /setpoint/gripper. The
RAMMP-CuRobo v1.0.0 planner plans (/rammp_curobo/plan_to_pose,
plan_to_joints, set_world) and never moves anything. TF comes from
kinova_gen3_description's robot_state_publisher, which the launch starts.

The spec §6 hardening stands: the guard is ARMED by execution feedback
progress > 0 (never at goal-accept) and efforts come from the /joint_states
stream. What the driver does not check before it moves — the start state,
velocity limits, timing — is checked here (runtime/driver.py), and nothing
moves at all unless this client was armed by --execute. What the planner
calls a success is checked too (runtime/planner.py): a plan is handed on
only when it ends where it was asked (PlannerClient.PLAN_END_*), and never
as it came when the planner says it goes the long way round.
"""

import copy
import math
import sys
import threading
import time
from collections import deque, namedtuple

import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening import LOGGER as log
from box_opening.constants import (  # noqa: F401 - GAINS_* are re-exported: callers name them client.GAINS_*
    COMPLIANT_RUN_GAINS,
    EXECUTE_ACTION,
    FINGERTIP_FRAMES,
    GAINS_CUSTOM,
    GAINS_MEDIUM,
    GAINS_RIGID,
    GAINS_SOFT,
    GAINS_STIFF,
    GRIPPER_FORCE,
    GRIPPER_SETPOINT_TOPIC,
    GRIPPER_SPEED,
    IMPEDANCE_NAMES,
    JOINT_VMAX,
    JOINTS,
    NODE_NAMESPACE,
    CustomGains,
)
from box_opening.runtime import driver, planner
from box_opening.runtime.approach import plan_with_vertical_approach
from box_opening.runtime.branches import joint_time, nearest_branch, planner_branch
from box_opening.runtime.stamps import set_stamp

# Interface packages are installed in the driver/planner containers and in
# ros2_ws on the bench, but may not be available on a fresh flatbot host.
# Wrap in try/except so that *importing* this module (launch file, CLI arg
# parsing) does not fail — instantiating PlannerClient still requires them.
try:
    from rammp_arm_interfaces.action import ExecuteJointTrajectory
    from rammp_arm_interfaces.msg import GripperSetpoint
    from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
    from rammp_curobo_interfaces.srv import SetWorld
except ImportError:
    ExecuteJointTrajectory = GripperSetpoint = PlanToJoints = PlanToPose = SetWorld = None


_GRIPPER_JOINT_HINTS = ("robotiq", "knuckle", "finger")

# ExecuteJointTrajectory.Goal.control_mode / .preemption (rammp_arm_interfaces)
POSITION = 0
IMPEDANCE = 1
QUEUE = 0
LATEST_WINS = 1  # replaces the goal running, and any queued behind it, at once (release only)
# ImpedanceGains.profile: the driver's named joint-impedance tunings,
# GAINS_SOFT / GAINS_MEDIUM / GAINS_STIFF (constants.py, with their Kq), or
# GAINS_CUSTOM with the spec's own kq / zeta / torque_limit (GAINS_RIGID)

# a gripper command in flight: knuckle-radian target, the knuckle at send, when
GripperHandle = namedtuple("GripperHandle", "target start t0")
# one /joint_states message as PlannerClient.joint_history keeps it: when it
# was heard (time.monotonic()), its header stamp (s, the driver's clock: when
# it was measured), and the arm's joints in order — positions, velocities and
# efforts, None where the message carried none
JointSample = namedtuple("JointSample", "t stamp q qd effort")


def _measured_at(s):
    """When the JointSample `s` was measured: its stamp, else (a message
    sent unstamped) when it was heard."""
    return s.stamp if s.stamp else s.t


def _speeds(samples, i):
    """Each joint's speed (rad/s) at samples[i]: its velocities, else its
    positions differenced, wrap-aware, over when it and the reading before
    it were measured; None where neither says."""
    s = samples[i]
    if s.qd is not None:
        return [abs(float(v)) for v in s.qd]
    if i == 0:
        return None
    dt = _measured_at(s) - _measured_at(samples[i - 1])
    if not dt > 0.0:
        return None
    from box_opening.geometry import ang_diff

    return [abs(ang_diff(a, b)) / dt for a, b in zip(s.q, samples[i - 1].q, strict=False)]


def still_for(samples, qd_max, gap_s, enough=None):
    """How long the newest of `samples` (JointSample, oldest first) have read
    the arm still, by when they were measured: from the first of the
    trailing run in which every reading has each joint under `qd_max`
    (_speeds) and was measured at most `gap_s` after the reading before it,
    to the newest. 0.0 when the newest is not still; the count stops once
    it reaches `enough`."""
    newest = first = None
    for i in range(len(samples) - 1, -1, -1):
        v = _speeds(samples, i)
        if v is None or not all(x < qd_max for x in v):
            break
        t = _measured_at(samples[i])
        if first is not None and not 0.0 <= first - t <= gap_s:
            break
        newest = t if newest is None else newest
        first = t
        if enough is not None and newest - first >= enough:
            break
    return 0.0 if newest is None else newest - first


def fill_gains(msg, spec):
    """Write gains `spec` into ImpedanceGains `msg`: a CustomGains as
    PROFILE_CUSTOM with its kq, zeta and torque_limit, a profile as its
    number."""
    if isinstance(spec, CustomGains):
        msg.profile = GAINS_CUSTOM
        msg.custom.kq = [float(v) for v in spec.kq]
        msg.custom.zeta = float(spec.zeta)
        msg.custom.torque_limit = [float(v) for v in spec.torque_limit]
    else:
        msg.profile = int(spec)


def spin_until_done(node, future, timeout_s, abort=None):
    """Spin `node` until `future` resolves; None on timeout — or at once
    when `abort` has been requested, so a plan hosted inside an execution
    (see PlannerClient.execute while_running) cannot delay the cancel."""
    t0 = time.monotonic()
    while not future.done():
        if abort is not None and abort.requested:
            return None
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - t0 > timeout_s:
            return None
    return future.result()


def pose_goal_joints(xyz, quat_xyzw, start_joints):
    """The joints a pose goal from `start_joints` is planned to
    (PlannerClient._plan_pose_near): of the arm's own IK solutions for
    tool_frame at (xyz, quat_xyzw), the one quickest to fly to from the
    start (kinematics.ArmChain.nearest_solution), walked from there or else
    from the look above HOME — or None, where it finds none or the
    kinematics fail, and the pose goes to the planner as a pose. The
    reach-out asks first, to prove its press from these joints before the
    planner is asked for the look (tasks/reach_press.plan_reach); asked
    again with the same numbers the chain answers from its last answer
    (nearest_solution), so the plan request that follows is the one it
    always was."""
    if not start_joints:
        return None
    try:
        from box_opening.constants import HOME
        from box_opening.primitives.look import look_joints
        from box_opening.runtime.approach import _chain

        return _chain().nearest_solution(planner_branch(start_joints), xyz, quat_xyzw, seeds=(look_joints(HOME),))
    except Exception:  # a kinematics problem must never cost the plan
        return None


class PlannerClient:
    # a pending gripper setpoint is re-sent this often: the topic is
    # best-effort and latest-wins, so one lost message must not strand a
    # command, and re-sending the same absolute setpoint changes nothing
    GRIPPER_RESEND_S = 0.1
    GRIPPER_TIMEOUT_S = 10.0
    # a knuckle reading older than this is a stream gap, never "settled"
    GRIPPER_STREAM_GAP_S = 0.2

    def __init__(self, node, abort=None, motion_enabled=False):
        self.node = node
        self._abort = abort  # AbortFlag when the CLI owns SIGINT (abort.py)
        # The motion latch, set only by --execute. The driver executes
        # whatever it receives and has no dry-run parameter, so this is the
        # software gate on arm AND gripper motion.
        self._armed = bool(motion_enabled)
        self._base_frame = "base_link"  # overridden by cli_common/preflight from WorldStore
        self._q = None
        self._eff = None
        self._eff_at = 0.0  # monotonic stamp of the last joint_states message
        # the newest JOINT_HISTORY_N /joint_states messages, oldest first
        # (joint_history), appended under the lock by whichever thread
        # spins this node
        self._history = deque(maxlen=self.JOINT_HISTORY_N)
        self._history_lock = threading.Lock()
        # when each goal went to the driver (time.monotonic()), newest last:
        # what the arm does after an arrival stops being that arrival's
        # settle when the next goal goes out (goals_sent)
        self._sent_at = deque(maxlen=self.GOALS_KEPT)
        # the last execute call's info, filled as it runs: a Ctrl+C's
        # unwinding hands none back, and the runner keeps the stopped
        # motion's side file from this (runner._keep_aborted)
        self.last_execution = None
        # Set on first /joint_states message: 'joint_1..7' on bench,
        # 'arm_joint_1..7' on flatbot (nightly driver renders arm_mount URDF).
        self._active_joints = None
        # Fingertip TF frame names — match the URDF's link prefix.
        # Overridden to "arm_robotiq_*" on flatbot when joint prefix is detected.
        self._tip_frames = list(FINGERTIP_FRAMES)
        self._gripper_pos = None
        self._gripper_target = None  # driver setpoint (0..1) re-sent while pending
        self.world_held = None  # path of the world the planner holds
        self.compliant = False  # joint impedance, nothing read stops a leg (execute); a run sets it
        # the ImpedanceGains spec a compliant goal that names none flies
        # (constants.COMPLIANT_RUN_GAINS, RIGID); the driver's own is MEDIUM
        self.compliant_gains = COMPLIANT_RUN_GAINS
        self._primer = None  # prime_compliant's goal, sent and never waited on
        # the last goal sent, as release() needs it: its control_mode and a
        # copy of its gains; whether a release has gone out since
        self._last_mode = None
        self._last_gains = None
        self._released = False
        self._release_goal = None  # release()'s goal, kept referenced, never waited on
        # the driver publishes /joint_states best-effort (SensorDataQoS): a
        # reliable subscription would match nothing and hear nothing
        node.create_subscription(JointState, "/joint_states", self._js_cb, qos_profile_sensor_data)
        self._plan_pose = ActionClient(node, PlanToPose, NODE_NAMESPACE + "/plan_to_pose")
        self._plan_joints = ActionClient(node, PlanToJoints, NODE_NAMESPACE + "/plan_to_joints")
        self._execute = ActionClient(node, ExecuteJointTrajectory, EXECUTE_ACTION)
        self._set_world = node.create_client(SetWorld, NODE_NAMESPACE + "/set_world")
        # latest-wins, best-effort depth 1: the driver subscribes exactly so
        self._gripper_pub = node.create_publisher(
            GripperSetpoint,
            GRIPPER_SETPOINT_TOPIC,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        node.create_timer(self.GRIPPER_RESEND_S, self._publish_gripper)
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, node)

    @property
    def joint_names(self):
        """The 7 arm joint names as seen in /joint_states.

        'joint_1..7' on bench; 'arm_joint_1..7' on flatbot with the nightly
        driver (which renders the URDF with arm_mount as root and prefixes
        every joint). Determined from the first /joint_states message; returns
        the bench default before any message arrives."""
        return self._active_joints if self._active_joints is not None else list(JOINTS)

    # -- state streams -----------------------------------------------------
    # What the arm does after the driver calls a goal arrived is read from
    # here (runner._note_settle). /joint_states runs 49 Hz on flatbot (the
    # driver's state_publish_divisor 2; 1 would be ~98 Hz), so 600 messages
    # are 6-12 s: an arrival's settle window (runner.SETTLE_WINDOW_S, 1.5 s)
    # is noted when the next goal goes out, the aim or the planning between.
    # Its stamp, not when it was heard, says when the arm was where it says:
    # the aim spins this node once a frame and the subscription keeps 5
    # (SensorDataQoS), so between frames messages wait, and some are dropped
    JOINT_HISTORY_N = 600
    GOALS_KEPT = 8

    def _js_cb(self, msg):
        if self._active_joints is None:
            if "arm_joint_1" in msg.name:
                self._active_joints = ["arm_joint_%d" % i for i in range(1, 8)]
                self._tip_frames = [
                    "arm_robotiq_85_left_finger_tip_link",
                    "arm_robotiq_85_right_finger_tip_link",
                ]
            elif "joint_1" in msg.name:
                self._active_joints = list(JOINTS)
            else:
                return  # no recognisable arm joints yet — wait
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q = [float(msg.position[idx[n]]) for n in self._active_joints]
            eff = [float(msg.effort[idx[n]]) for n in self._active_joints] if len(msg.effort) == len(msg.name) else None
            qd = (
                [float(msg.velocity[idx[n]]) for n in self._active_joints]
                if len(msg.velocity) == len(msg.name)
                else None
            )
        except (KeyError, IndexError):
            return
        self._q = q
        self._eff = eff
        self._eff_at = time.monotonic()
        stamp = msg.header.stamp
        with self._history_lock:
            self._history.append(JointSample(self._eff_at, stamp.sec + 1e-9 * stamp.nanosec, q, qd, eff))
        for name in msg.name:
            if any(h in name for h in _GRIPPER_JOINT_HINTS):
                self._gripper_pos = float(msg.position[idx[name]])
                break

    def motion_enabled(self):
        """Whether this client may move the arm or the gripper (--execute)."""
        return self._armed

    def joints(self):
        t0 = time.monotonic()
        while self._q is None:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if time.monotonic() - t0 > 10:
                sys.exit(
                    "no /joint_states — is sheppy's `arm` node up, and does this "
                    "shell export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp?"
                )
        return list(self._q)

    def joint_history(self, since=None):
        """The /joint_states messages kept (JointSample, the newest
        JOINT_HISTORY_N), oldest first: a copy, only those heard at or after
        `since` (time.monotonic()) when it is given. Spins nothing, waits
        for nothing: what this node has heard is what there is."""
        lock = getattr(self, "_history_lock", None)
        if lock is None:
            return []
        with lock:
            kept = list(self._history)
        return kept if since is None else [s for s in kept if s.t >= since]

    def goals_sent(self):
        """When each of the last GOALS_KEPT goals went to the driver
        (time.monotonic()), oldest first: every execution, the hold that
        primes the joint impedance and the release."""
        return list(getattr(self, "_sent_at", ()))

    def _mark_sent(self):
        """A goal is about to go to the driver: when (goals_sent)."""
        sent = getattr(self, "_sent_at", None)
        if sent is not None:
            sent.append(time.monotonic())

    # A guarded leg polls wrist_efforts() and trips on deviation from a
    # baseline. If the /joint_states subscription STALLS, self._eff keeps
    # its last value forever: the deviation stays flat, the guard can never
    # trip, and the existing "efforts lost" watchdog never fires either —
    # it only catches messages that arrive WITHOUT effort fields. Stale
    # readings must therefore read as no readings (review 2026-08-28).
    EFFORT_STALE_S = 0.5
    # A guarded stroke's own limit, with no grace on top: ten periods of
    # the ~49 Hz stream, which never went 83 ms without a message on the
    # flatbot (2026-10-21). 0.5 s plus a 1 s grace was 1.5 s flown blind,
    # ~126 mm of the depth touch's stroke (review 2026-10-22).
    GUARD_STALE_S = 0.2

    def wrist_efforts(self, max_age=None):
        if self._eff is None:
            return None
        if time.monotonic() - self._eff_at > (self.EFFORT_STALE_S if max_age is None else max_age):
            return None  # frozen stream: the guard has lost its senses
        return list(self._eff[3:])

    def settled_efforts(self, duration_s=0.15, settle_s=0.15, timeout_s=2.0):
        """(wrist efforts, joints), each averaged over `duration_s` of fresh
        /joint_states once `settle_s` has passed, or None: the arm at rest,
        read as a free-air reference (tasks/reach_press). The arm must be
        standing still — this only reads.

        0.15 s is ~7 messages at 49 Hz: at rest one message is within
        ~0.03 Nm, so the mean sits far below every threshold it feeds
        (STATIC_NM 0.75 and up) and the gravity model's own ~0.12 Nm
        residual. A longer window only holds the arm still (review
        2026-10-22: 0.30 s cost ~1.3 s a depth run for nothing measurable)."""
        effs, qs = [], []
        t0 = time.monotonic()
        last = None
        while time.monotonic() - t0 < timeout_s:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if time.monotonic() - t0 < settle_s:
                continue
            if self._eff is None or self._q is None or time.monotonic() - self._eff_at > self.EFFORT_STALE_S:
                continue
            if self._eff_at == last:
                continue
            last = self._eff_at
            effs.append(list(self._eff[3:]))
            qs.append(list(self._q))
            if time.monotonic() - t0 >= settle_s + duration_s and len(effs) >= 3:
                break
        if len(effs) < 3:
            return None
        # continuous joints are reported wrapped into (-pi, pi]: HOME's
        # joint_3 sits 4.6 mrad from the wrap, and a plain mean of readings
        # either side of it is a different arm (branches.py)
        from box_opening.geometry import ang_diff

        n = float(len(effs))
        q0 = qs[0]
        q = [q0[j] + sum(ang_diff(row[j], q0[j]) for row in qs) / n for j in range(len(q0))]
        return [sum(c) / n for c in zip(*effs, strict=False)], q

    # A camera read at rest, and a grip's close, wait here for the arm to
    # stand still (wait_still; the runner notes it, runner.wait_still).
    # "Arrived" is the driver's trajectory clock running out, and the
    # camera's own still test (perception/depth_source STILL_TRANS_M 4 mm,
    # STILL_ROT_RAD 0.02 between processed frames 75-90 ms apart) passes an
    # arm creeping 22-25 mm/s: on 2026-10-28 15:45 both looks arrived with
    # joint_2 45-47 mrad past their end and crept back at up to 0.12-0.13
    # rad/s, and at 16:23 the look's aim took its first two hits 0.32 and
    # 0.41 s in, the knob -7.2 and -5.5 mm above its lid where the two that
    # agreed after them read +1.1 and +0.7. On 15:45's /joint_states
    # (49 Hz) the arm at rest read at most 0.0032 rad/s on any joint in
    # RIGID at zeta 0.7 (50 s at HOME; at 1.0 unmeasured: the first run's
    # 'still' notes show it) and 0.0010 in POSITION (200 s), and the looks'
    # creep read over STILL_QD_RAD_S to +0.70 s. A settling joint turns
    # back through zero, one or two readings under it (the look's turn at
    # +0.07 s, the way homes' at +0.70-0.73), where STILL_HOLD_S takes five
    # in a row at 49 Hz: the looks passed at +0.81 s, the POSITION home at
    # +0.08 (+0.10 by positions). The looks' pass is the creep over, not
    # the arm at its last rest: joint_6 stopped between two swings, the
    # tool 1.2-1.5 mm from where it stood when the way in left (+1.4-1.6 s),
    # drifting back at up to 5 mm/s (joint_6 at up to 0.012-0.017 rad/s),
    # within 1 mm of it for good from +1.03-1.07 s — far inside the
    # camera's still test, and each frame's pose is TF's at its stamp.
    # Readings further apart than STILL_GAP_S make no one still stretch
    # (the aim spins this node once a frame, and between frames messages
    # are dropped): the arm swings at 0.7-1.1 Hz, its turns 0.45 s and more
    # apart. The way homes, 86-88 mm short when they ended, passed only at
    # +1.49-1.50 s: past STILL_TIMEOUT_S, and the wait carries on (no
    # camera reads there).
    STILL_QD_RAD_S = 0.02
    STILL_HOLD_S = 0.08
    STILL_GAP_S = 0.25
    STILL_TIMEOUT_S = 1.2

    def wait_still(self):
        """Wait, the arm at rest after a motion, until it stands still: every
        joint under STILL_QD_RAD_S in each /joint_states reading over
        STILL_HOLD_S (still_for), of those heard since the last goal went to
        the driver (goals_sent), the newest heard during this call — so a
        call right after another, nothing sent between, finds it still at
        once. Returns (s waited, timed out): after STILL_TIMEOUT_S it
        carries on, saying so in one line. Sends nothing and stops nothing:
        it spins this node while the arm settles (or the stream is
        silent), no more.

        No goal is in flight here, so a Ctrl+C raises at once
        (abort.sigint_handler); raised inside a subscription's take it
        surfaces as RuntimeError (press_demo._spin_detect), and goes on as
        the KeyboardInterrupt it is: the run ends as a Ctrl+C (exit 130),
        not a traceback."""
        t0 = time.monotonic()
        sent = self.goals_sent()
        since = sent[-1] if sent else None
        last = None
        while True:
            try:
                rclpy.spin_once(self.node, timeout_sec=0.02)
            except RuntimeError as e:
                if self._abort is not None and self._abort.requested:
                    raise KeyboardInterrupt from e
                raise
            waited = time.monotonic() - t0
            if self._eff_at != last:  # a new reading to judge
                last = self._eff_at
                kept = self.joint_history(since=since)
                if (
                    kept
                    and kept[-1].t > t0
                    and still_for(kept, self.STILL_QD_RAD_S, self.STILL_GAP_S, self.STILL_HOLD_S) >= self.STILL_HOLD_S
                ):
                    return waited, False
            if waited >= self.STILL_TIMEOUT_S:
                kept = self.joint_history(since=since)
                v = _speeds(kept, len(kept) - 1) if kept and kept[-1].t > t0 else None
                if v is None:
                    said = "no /joint_states reading says how fast it moves"
                else:
                    j = max(range(len(v)), key=lambda k: v[k])
                    said = "%s at %.3f rad/s at the last reading" % (self.joint_names[j], v[j])
                log.info("[client] the arm is not still after %.1f s (%s) — carrying on" % (waited, said))
                return waited, True

    def efforts_present(self):
        self.joints()  # ensure at least one message arrived
        return self._eff is not None

    # Callbacks already waiting in this node's queue (the cameras' frames)
    # are served before a guard calls the stream lost: a backlog here is not
    # a silent arm. Capped in time, so a stream that has really stopped is
    # still cancelled within GUARD_STALE_S and this.
    GUARD_DRAIN_S = 0.02

    def _fresh_joints(self, timeout_s=1.0, max_age=None):
        """The live joints while /joint_states is flowing — the newest sample
        served within `max_age` (default EFFORT_STALE_S) — else None: a start
        gate judged against a frozen sample is no gate."""
        age = self.EFFORT_STALE_S if max_age is None else float(max_age)
        t0 = time.monotonic()
        while True:
            if self._q is not None and time.monotonic() - self._eff_at <= age:
                return list(self._q)
            if time.monotonic() - t0 > timeout_s:
                return None
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def contact_xyz(self, timeout_s=0.25):
        """Where the FINGERTIPS are right now, from TF: the midpoint of the
        two finger-tip links, or None.

        Called the instant a guard trip lands, this is the arm measuring the
        surface it just touched with its own kinematics — no camera, no
        model constant. Unlike tool_frame these frames exist in the live
        tree, so the lookup resolves instead of stalling."""
        if getattr(self, "_tip_frames_missing", False):
            return None
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            try:
                pts = []
                for frame in self._tip_frames:
                    tf = self._tf.lookup_transform(self._base_frame, frame, Time())
                    tr = tf.transform.translation
                    pts.append([tr.x, tr.y, tr.z])
                return [sum(c) / len(pts) for c in zip(*pts, strict=False)]
            except Exception:
                rclpy.spin_once(self.node, timeout_sec=0.05)
        self._tip_frames_missing = True
        log.info("[client] no %s in TF — contact heights unavailable" % (self._tip_frames[0],))
        return None

    def planner_reachable(self, timeout_s=5.0):
        """True when both planner actions AND its set_world service respond.

        set_world is advertised only once cuRobo has finished loading, so
        an action that answers before it does is not a planner that can plan
        yet (box opening never calls it: worlds.py)."""
        return (
            self._plan_pose.wait_for_server(timeout_sec=timeout_s)
            and self._plan_joints.wait_for_server(timeout_sec=2.0)
            and self._set_world.wait_for_service(timeout_sec=2.0)
        )

    def driver_reachable(self, timeout_s=5.0):
        """True when the driver's trajectory action responds."""
        return self._execute.wait_for_server(timeout_sec=timeout_s)

    def tf_ready(self, target, source, timeout_s=2.0):
        """True once TF can give `target` <- `source` (the latest): the arm's
        TF comes from robot_state_publisher in sheppy's box_opening node."""
        from rclpy.time import Time

        t0 = time.monotonic()
        while True:
            if self._tf.can_transform(target, source, Time()):
                return True
            if time.monotonic() - t0 > timeout_s:
                return False
            rclpy.spin_once(self.node, timeout_sec=0.05)

    # Programs that move this arm on their own: one of these running beside a
    # mission is two controllers on one arm.
    OTHER_ARM_CLIENTS = ("rammp_adl_runtime",)

    def other_arm_clients(self):
        """The OTHER_ARM_CLIENTS on the ROS graph now."""
        return sorted({n for n, _ns in self.node.get_node_names_and_namespaces() if n in self.OTHER_ARM_CLIENTS})

    # -- planning ----------------------------------------------------------
    # The planner plans one goal at a time and turns any other away at once,
    # unplanned: "planner busy — one plan at a time" (planner_node.py
    # _finish_plan, which leaves no line of it in its own log). Teleop's
    # home, the microwave engine and the driver's GoTo actions plan on the
    # same node, 0.14-0.34 s a plan on 2026-10-28 (p10-p90 of 72): a goal
    # that lands in one of theirs is asked again, BUSY_RETRIES times
    # BUSY_WAIT_S apart, before it fails as it always did.
    BUSY_RETRIES = 2
    BUSY_WAIT_S = 0.2

    # A plan's END is checked against what was asked before anything uses
    # it: the arm flies what comes back, and the leg after it starts there
    # (core._plan_motion's goal_joints). A joint plan ends ON its goal — the
    # native solve to ~5e-7 rad (rammp-curobo CLAUDE.md, 2026-08-11; its
    # smoke test bounds it at 1e-3). Where that solve fails, PlanToJoints
    # falls back to planning to the goal's FK pose (3 of 72 plans on
    # 2026-10-28, 13:32-13:34: TRAJOPT_FAIL on joint goals from a client
    # the planner report never identified): the same tool to cuRobo's
    # 5 mm / 5.7 deg, through whatever joints its IK finds, another elbow
    # or wrist among them, said only in goal_mismatch_rad, which nothing
    # here read. cuRobo v0.7.8's finetune can also hand back the previous
    # solve's plan as a success (rammp-curobo #37: 3-7 of 200 in its
    # probe), which the start gate catches only when the two starts differ.
    # 0.01 rad is ten times that smoke test's bound and at most 10.2 mm at
    # the tool (one joint off, 2000 postures on flatbot's URDF, offline
    # 2026-10-28).
    PLAN_END_RAD = 0.01
    # A pose plan meets cuRobo's own success test: the tool within
    # position_threshold 0.005 m and rotation_threshold 0.05 of its pose
    # (motion_gen.py's defaults, which the rammp config leaves alone) — the
    # error quaternion's vector part, sin(turn/2) < 0.05, 5.73 deg, as the
    # planner's own planner.py and PlanToPose.action read it. The arm's FK
    # here agrees with the planner's to 0.01 mm, so a plan that met the
    # test for THIS pose is inside these, and one outside them met it for
    # some other.
    PLAN_END_M = 0.006
    PLAN_END_DEG = 6.0

    # Every plan this client refused (_refuse), oldest first, a tuple that
    # each refusal adds one to. The run record keeps the ones made while a
    # leg was planned (core._plan_motion, Runner._replan_group): a refused
    # joint plan the pose plan then stood in for leaves no other trace.
    refusals = ()

    def _refuse(self, res, why):
        """planner.refuse(res, why): the failed plan a caller gets in place
        of `res`, its message kept in `refusals`."""
        out = planner.refuse(res, why)
        self.refusals += (out.message,)
        return out

    def _call(self, client, goal, timeout_s=120.0):
        """The planner's result for `goal` (None: not taken, or no answer in
        `timeout_s`), asked again while the planner answers that it is busy
        with another plan (BUSY_RETRIES): then its last answer, a failed
        plan as before. A Ctrl+C ends the wait at once."""
        if not client.wait_for_server(timeout_sec=5.0):
            sys.exit("planner not reachable — is sheppy's `planner` node up?")
        res = self._ask(client, goal, timeout_s)
        for n in range(1, self.BUSY_RETRIES + 1):
            if not planner.busy(res):
                break
            log.info(
                "[client] planner busy (another goal is planning) — asking again in %.1f s (%d/%d)"
                % (self.BUSY_WAIT_S, n, self.BUSY_RETRIES)
            )
            if not self._spin_for(self.BUSY_WAIT_S):
                break  # aborted: the busy answer stands, a failed plan
            res = self._ask(client, goal, timeout_s)
        return res

    def _ask(self, client, goal, timeout_s):
        """One round trip of `goal` to the planner action `client`."""
        send = spin_until_done(self.node, client.send_goal_async(goal), 10.0, abort=self._abort)
        if send is None or not send.accepted:
            return None
        wrapped = spin_until_done(self.node, send.get_result_async(), timeout_s, abort=self._abort)
        return None if wrapped is None else wrapped.result

    def _spin_for(self, seconds):
        """Serve this node for `seconds` — a plan hosted inside an execution
        (execute's while_running) keeps its feedback flowing — or False at
        once when an abort is requested."""
        t_end = time.monotonic() + float(seconds)
        while True:
            if self._abort is not None and self._abort.requested:
                return False
            left = t_end - time.monotonic()
            if left <= 0.0:
                return True
            rclpy.spin_once(self.node, timeout_sec=min(left, 0.05))

    def plan_to_pose(self, xyz, quat_xyzw, start_joints, approach_offset_m=0.0):
        """>0 approach_offset_m: arrive from a waypoint that far straight above
        the goal — a diagonal descent touches a surface before its lateral
        convergence finishes (edge presses, field 2026-09-01). The v1.0.0
        planner cannot constrain that, so it is two plans flown as one
        trajectory (runtime/approach.py)."""
        return plan_with_vertical_approach(self._plan_pose_near, xyz, quat_xyzw, start_joints, float(approach_offset_m))

    def _plan_pose_near(self, xyz, quat_xyzw, start_joints):
        """A pose goal planned as a JOINT goal: the IK solution quickest to
        fly to from the start (kinematics.ArmChain.nearest_solution), so the
        planner cannot pick a far one of the redundant arm's solutions —
        handed the pose, it turned the wrist 3.2-3.6 rad on some plans (a
        4.7 s approach to a box on the left, bench 2026-09-25). The planner
        still plans and collision-checks the path; when no solution is found
        or its plan fails, the pose goes to it as before."""
        q_goal = pose_goal_joints(xyz, quat_xyzw, start_joints)
        if q_goal is not None:
            res = self.plan_to_joints(q_goal, start_joints)
            if res is not None and res.success:
                return res
        return self._no_long_way(self._plan_pose_once(xyz, quat_xyzw, start_joints), start_joints)

    # A plan is re-planned when an equivalent end is at least this much quicker
    # to fly to: a continuous joint the long way round is up to ~5 s of it.
    QUICKER_BY_S = 0.3

    def _no_long_way(self, res, start_joints):
        """The planner's own pose plan — flown when no quicker solution was
        found up front (the table's far edge) — unless it ends somewhere the
        same tool pose is quicker to reach: a continuous joint turned the long
        way round, or the wrist folded the far way (kinematics.equivalents).
        Then it is re-planned to that end (plan_to_joints); the original stands
        if that plan fails. The arm does not turn a joint round when it does
        not need to (owner, 2026-09-25).

        Except a plan the planner itself says goes the long way round
        (planner.LONG_WAY_ROUND): its own replan onto the start's branch
        failed, and it handed the winding plan back as a success (15:56:05
        2026-10-28, 4.38 s). That one is never flown as it is: it is
        re-planned to the quickest end whatever that saves, and refused when
        no end is found, there is no chain to find one by, or that plan
        fails too."""
        if res is None or not res.success or not start_joints or res.trajectory is None or not res.trajectory.points:
            return res
        wound = planner.long_way(res)
        try:
            from box_opening.runtime.approach import _chain

            start = planner_branch(start_joints)
            end = [float(v) for v in res.trajectory.points[-1].positions]
            alts = _chain().equivalents(start, end)
        except Exception as e:  # noqa: BLE001 - a kinematics problem never costs a plan the planner did not flag
            return (
                self._refuse(
                    res,
                    "the plan goes the long way round, and there are no kinematics to find the "
                    "short way by (%s) — not flown" % e,
                )
                if wound
                else res
            )
        if not alts:
            return (
                self._refuse(
                    res, "the plan goes the long way round, and no other way to its end was found — " "not flown"
                )
                if wound
                else res
            )
        best = min(alts, key=lambda q: joint_time(start, q))
        if not wound and joint_time(start, best) > joint_time(start, end) - self.QUICKER_BY_S:
            return res
        better = self.plan_to_joints(best, start_joints)
        if better is not None and better.success:
            return better
        if wound:
            return self._refuse(
                res,
                "the plan goes the long way round, and its re-plan the short way failed (%s) — "
                "not flown" % getattr(better, "message", "no answer"),
            )
        return res

    def _plan_pose_once(self, xyz, quat_xyzw, start_joints):
        g = PlanToPose.Goal()
        g.target.position.x, g.target.position.y, g.target.position.z = (float(v) for v in xyz)
        (
            g.target.orientation.x,
            g.target.orientation.y,
            g.target.orientation.z,
            g.target.orientation.w,
        ) = (float(v) for v in quat_xyzw)
        # on HOME's side of the wrap: the planner plans between the numbers
        # it is given, and a start written the other way round is a full turn
        g.start_joints = planner_branch(start_joints) if start_joints else []
        return self._checked_pose(self._call(self._plan_pose, g), xyz, quat_xyzw)

    def _checked_pose(self, res, xyz, quat_xyzw):
        """`res`, the planner's plan to the tool pose (xyz, quat_xyzw), when
        it ends there: within PLAN_END_M and PLAN_END_DEG by the arm's own
        FK, which every plan that met cuRobo's success test for this pose
        does. One that ends further off is another goal's plan, and refused,
        as is one with no finite end to measure. With no chain to measure
        by, it goes on as it always did (a kinematics problem never costs
        the plan); how far off it ends is in the run record
        (planner.plan_record)."""
        if res is None or not getattr(res, "success", False):
            return res
        if planner.plan_end(getattr(res, "trajectory", None)) is None:
            return self._refuse(res, "the plan has no finite end to check against its pose — not flown")
        err = planner.pose_end_error(res.trajectory, xyz, quat_xyzw)
        if err is None:
            return res
        off, turn = err
        if off > self.PLAN_END_M or math.degrees(turn) > self.PLAN_END_DEG:
            return self._refuse(
                res,
                "the plan ends %.2f mm and %.2f deg from its pose, past cuRobo's own success "
                "test (5 mm, 5.7 deg) — another goal's plan; not flown" % (1000.0 * off, math.degrees(turn)),
            )
        return res

    def plan_to_joints(self, q7, start_joints):
        """A joint goal, each continuous joint taken the short way round from
        the start (branches.nearest_branch): the same arm configuration,
        never a winding. The planner does this for itself since 1.0.0
        (_nearest_branch); done here too, it does not depend on that.

        The plan comes back only if it ends at `q7` (_checked_joints)."""
        start = planner_branch(start_joints) if start_joints else None
        goal = nearest_branch(q7, start) if start is not None else [float(v) for v in q7]
        g = PlanToJoints.Goal(target_joints=goal)
        g.start_joints = start or []  # on HOME's side of the wrap (plan_to_pose)
        return self._checked_joints(self._call(self._plan_joints, g), q7)

    def _checked_joints(self, res, q7):
        """`res`, the planner's plan to the joints `q7`, when it ends there:
        every joint within PLAN_END_RAD, each continuous joint the short way
        round — the goal went out on the start's branch, a whole turn from
        `q7` where that is the short way. A plan that ends elsewhere (the
        FK-pose fallback's other arm, a previous goal's plan) is refused, as
        is one with no finite end to measure, and one the planner says goes
        the long way round: the goal it was sent already was the short way,
        so the long way is all its fallback found. The planner's
        goal_mismatch_rad is reported with it, not judged: the same gap
        measured without the wrap (rammp-curobo's planner.py,
        np.abs(final - q_goal).max()), it reads 6.28 for a joint a turn
        from its goal."""
        if res is None or not getattr(res, "success", False):
            return res
        rad, j = planner.joint_end_error(getattr(res, "trajectory", None), q7)
        if rad is None:
            return self._refuse(res, "the plan has no finite end to check against its joint goal — not flown")
        if rad > self.PLAN_END_RAD:
            mismatch = planner.finite(getattr(res, "goal_mismatch_rad", None))
            return self._refuse(
                res,
                "the plan ends %.4f rad from its joint goal (joint %d%s) — the planner's "
                "FK-pose fallback, another arm at the same tool pose, or another goal's plan; "
                "not flown" % (rad, j + 1, "" if mismatch is None else "; goal_mismatch_rad %.4f" % mismatch),
            )
        if planner.long_way(res):
            return self._refuse(
                res, "the plan goes the long way round, where its goal went out the short way — " "not flown"
            )
        return res

    # -- execution ---------------------------------------------------------
    def _cancel_confirm(self, send, result_future):
        """Ctrl+C path: cancel on a live context and report what the driver
        actually confirmed — never claim a stop that wasn't answered."""
        spin_until_done(self.node, send.cancel_goal_async(), 5.0)
        wrapped = spin_until_done(self.node, result_future, 10.0)
        if wrapped is not None:
            log.info("\nCtrl+C — cancel delivered; the driver stops and holds")
        else:
            log.info("\nCtrl+C — cancel sent; no result confirmation in 10 s — " "check the arm")

    # a cancel the driver took: it stopped the goal, or the goal had
    # already ended (within the cancel's latency of the request)
    _CANCEL_TAKEN = (CancelGoal.Response.ERROR_NONE, CancelGoal.Response.ERROR_GOAL_TERMINATED)

    def _stop_confirmed(self, send, result_future, cancel_s=3.0, result_s=10.0):
        """Cancel a goal in flight and say whether the driver confirmed the
        stop: the result arrived, and either it reads CANCELED or the cancel
        was taken. A SUCCESSFUL result after a cancel nobody took is a
        stroke flown to its end past the request — not a stop."""
        cresp = spin_until_done(self.node, send.cancel_goal_async(), cancel_s)
        wrapped = spin_until_done(self.node, result_future, result_s)
        if wrapped is None:
            return False
        if getattr(wrapped, "status", None) == GoalStatus.STATUS_CANCELED:
            return True
        return getattr(cresp, "return_code", None) in self._CANCEL_TAKEN

    # prime_compliant's hold: two points at the arm's joints, the first past
    # t=0 (check_like_executor refuses a first point at 0), the last 50 ms
    # in. It is done ~0.3 s after it is sent (mode_settle_s 0.25 s, then
    # 0.05). The reach-out's look goes out ~0.5 s after it: the look's plan
    # (0.34 s at 15:34 and 16:23 2026-10-27), the compliant proof's slide and
    # way down (0.09-0.16 s for those two boxes, timed offline on flatbot's
    # URDF), then 48 ms from the plan's note to the look (16:23). So the hold
    # never holds the look back; a 0.75 s one would keep it waiting ~0.5 s.
    PRIME_TIMES_S = (0.02, 0.05)

    def prime_compliant(self, q=None):
        """Put the driver in its joint impedance now, holding the arm where it
        stands, so the first compliant goal after a POSITION one neither waits
        for the switch nor starts on a half-faded spring. True when the hold
        went out.

        A goal in another control_mode than the one the driver runs waits
        mode_settle_s (0.25 s) before it is submitted, and on entry the mode
        fades its spring, its damper and its M·qdd feedforward in over
        gain_ramp_s (0.5 s), whether or not a trajectory runs
        (kinova-gen3-driver supervisor.cpp, joint_impedance_mode.cpp on_enter
        and compute). Flatbot 2026-10-27 16:23: the look, that run's first
        compliant goal, took 1.857 s against its 1.60 s preview, and no
        compliant leg after it ran more than 3 ms over its own (15:34's
        position look: 1.451 s against 1.45). Sent while the reach is
        planned, the hold has paid both by the time the look goes out. After
        it the mode holds its last point; a look sent while it still runs is
        queued behind it (QUEUE) and promoted gaplessly.

        Sent only by an armed, compliant client with no abort requested,
        under its compliant_gains, from a fresh /joint_states sample
        (GUARD_STALE_S) and through the gates every goal passes
        (driver.refusal): it holds `q`, the read the caller's plan starts
        from, else that sample. Nothing waits for the goal's acceptance or
        its result: the planning that follows spins the node, and the next
        execute() waits on its own goal only."""
        if not self._armed or not bool(getattr(self, "compliant", False)):
            return False
        if self._abort is not None and self._abort.requested:
            return False
        live = self._fresh_joints(max_age=self.GUARD_STALE_S)
        if live is None:
            return False
        held = list(live) if q is None else [float(v) for v in q]
        hold = JointTrajectory()
        hold.joint_names = list(self.joint_names)
        for t in self.PRIME_TIMES_S:
            p = JointTrajectoryPoint()
            p.positions = list(held)
            p.velocities = [0.0] * len(held)
            p.accelerations = [0.0] * len(held)
            set_stamp(p.time_from_start, t)
            hold.points.append(p)
        if driver.refusal(hold, live, JOINT_VMAX):
            return False
        if not self._execute.wait_for_server(timeout_sec=0.2):
            return False
        goal = ExecuteJointTrajectory.Goal()
        goal.trajectory = hold
        goal.control_mode = IMPEDANCE
        fill_gains(goal.gains, getattr(self, "compliant_gains", COMPLIANT_RUN_GAINS))
        goal.preemption = QUEUE
        goal.sender_id = "box_opening"
        self._note_sent(goal)
        self._primer = self._execute.send_goal_async(goal)  # kept referenced; never waited on
        return True

    def _note_sent(self, goal):
        """Keep what release() needs of `goal`, about to be sent: its
        control_mode and a copy of its gains. Nothing has let it go yet.
        And when it went (goals_sent)."""
        self._last_mode = int(goal.control_mode)
        self._last_gains = copy.deepcopy(goal.gains)
        self._released = False
        self._mark_sent()

    # release()'s hold: as prime_compliant's, two points at the arm's joints,
    # the first past t=0 (check_like_executor refuses a first point at 0)
    RELEASE_TIMES_S = (0.02, 0.05)

    def release(self):
        """Let the joint spring go: the driver's impedance reference moved to
        where the arm stands, at rest, so it pushes on nothing. True when the
        driver took it.

        A cancel resets the driver's executor and writes no target: its
        joint impedance goes on commanding the last reference it was given,
        with that target's velocity and acceleration feed-forward
        (kinova-gen3-driver 94736cd, supervisor.cpp's cancel and
        joint_impedance_mode.cpp's compute). Nothing there lets go by
        itself: the mode has no watchdog (cmd_timeout_s 0), arbitration is
        disabled, and only a halt (the e-stop) holds the measured joints.
        So a compliant press stopped on the button pushes on into the box
        toward its reference 27-72 mm below the button's top (the pink box's
        12 mm travel and COMPLIANT_OVERSHOOT_M) for as long as the driver
        runs, and a goal cancelled at speed overshoots: the damper's stale
        velocity alone stands joint 1 2·zeta·sqrt(M/Kq) past the reference
        per rad/s it was moving at, 0.085 rad elbow up under MEDIUM (Kq 80,
        zeta 0.5, M 0.573 kg·m²; 0.084 under STIFF), centimetres at the hand.

        Sent only by an armed client after an IMPEDANCE goal that no release
        has followed, from a fresh /joint_states sample (GUARD_STALE_S): a
        stale one would pull the arm back to where it was. Not gated by the
        abort flag, which a Ctrl+C has already set: this moves nothing, it
        only replaces the reference. LATEST_WINS: a goal whose cancel did
        not land is displaced at once, where a queued release would be
        promoted after it and pull the arm back to this snapshot. Under the
        last goal's own gains: without them the driver's session default,
        MEDIUM, would take over from RIGID and sag 5-7x as far."""
        if not self._armed or getattr(self, "_last_mode", None) != IMPEDANCE or getattr(self, "_released", False):
            return False
        try:
            live = self._fresh_joints(max_age=self.GUARD_STALE_S)
            if live is None:
                log.info(
                    "[client] no fresh /joint_states: not released — the joint spring still pulls toward the "
                    "reference it was stopped at; check the arm"
                )
                return False
            hold = JointTrajectory()
            hold.joint_names = list(self.joint_names)
            for t in self.RELEASE_TIMES_S:
                p = JointTrajectoryPoint()
                p.positions = list(live)
                p.velocities = [0.0] * len(live)
                p.accelerations = [0.0] * len(live)
                set_stamp(p.time_from_start, t)
                hold.points.append(p)
            why = driver.refusal(hold, live, JOINT_VMAX)
            if why:
                log.info("[client] not released (%s) — check the arm" % why)
                return False
            if not self._execute.wait_for_server(timeout_sec=0.5):
                log.info("[client] not released: execute_joint_trajectory does not answer — check the arm")
                return False
            goal = ExecuteJointTrajectory.Goal()
            goal.trajectory = hold
            goal.control_mode = IMPEDANCE
            goal.gains = self._last_gains
            goal.preemption = LATEST_WINS
            goal.sender_id = "box_opening"
            self._mark_sent()
            # no abort: after a Ctrl+C the flag is already set
            send = spin_until_done(self.node, self._execute.send_goal_async(goal), 2.0)
        except Exception as e:  # noqa: BLE001 - a release must never cost the unwinding it rides on
            log.info("[client] not released (%s) — check the arm" % e)
            return False
        if send is None or not send.accepted:
            log.info("[client] not released: the driver did not take the hold — check the arm")
            return False
        self._release_goal = send
        self._released = True
        log.info("[client] released: the joint spring now holds where the arm stands")
        return True

    # The driver sends a feedback every sampler tick, 1 kHz by code, its
    # `actual` the measured joints (kinova-gen3-driver 94736cd
    # supervisor.cpp), and what this node hears of it is deserialized here
    # anyway: every FEEDBACK_EVERY-th heard is kept (~100 a second, if all
    # are; how many are has never been measured — the side file's times
    # say) with its fraction (the driver's own trajectory clock) and when it
    # was heard — the newest FEEDBACK_KEEP, a minute of motion (the longest
    # of 2026-10-28 flew 9.2 s, 15:46:37's way in) — for the run record's
    # side file (runner._keep_flight)
    FEEDBACK_EVERY = 10
    FEEDBACK_KEEP = 6000

    def execute(self, traj, speed, guard=None, while_running=None, stop_when=None, gains=None):
        """Run one trajectory on the driver; outcome 'arrived' | 'touch' | 'failed'.

        A COMPLIANT client (self.compliant, press_demo's default since
        2026-10-27) flies it in the driver's joint impedance, under `gains`
        (an ImpedanceGains profile; None: this client's compliant_gains,
        STIFF), and nothing it reads stops it: a guard only watches — its
        would-be trip is in info["would_trip"], a silent stream is skipped —
        and the leg runs to its end. The arm yields instead; the e-stop and
        Ctrl+C remain, and a Ctrl+C's cancel is followed by a release: the
        spring then holds where the arm stands, not toward the cancelled
        goal's reference (release). info["gains"] is the profile sent, None
        in position.

        `speed` dilates the trajectory's time base (the driver has no speed
        scale of its own); 1.0 flies it as planned or re-timed. Nothing is
        sent unless the client is armed and the trajectory passes the gates
        the driver lacks (runtime/driver.py): fresh live joints at its start,
        velocities under the limits, time running forward.

        With a guard: feedback progress arms it, /joint_states efforts feed
        it; a trip cancels the goal — the driver stops and holds its last
        reference, so the contact load stays on.

        `stop_when(progress)`: polled while the motion flies with the
        driver's progress fraction; the first True cancels the goal and
        returns 'stopped'. This is how a search ends the moment it finds
        what it was looking for, leaving the arm where it saw it — and how
        a reach that carries its own descent stops at the junction when the
        box has not been confirmed by then.

        `while_running` (UNGUARDED legs only): a callable run once right
        after the goal is accepted — the next phase's planning, hidden under
        this motion instead of after it (the planner is its own container).
        Its blocking plans spin this same node, so feedback keeps flowing;
        its result lands in info["while_running"], an exception in
        info["while_running_error"] — never past the cancel path.

        Once the goal goes out, info["sent_at"] is when (time.monotonic()),
        info["sent_traj"] the trajectory sent (dilated where `speed` asked)
        and info["feedback_q"] what flew: every FEEDBACK_EVERY-th feedback
        as (when it was heard, its fraction, the measured joints). The same
        info is self.last_execution from the call's start: a Ctrl+C raises
        through here, once the cancel and the release are out, with
        nothing returned."""
        info = {"message": "", "progress": 0.0, "torque_peak": None, "gains": None}
        self.last_execution = info
        if self._abort is not None and self._abort.requested:
            raise KeyboardInterrupt  # aborted before this leg ever started
        if not self._armed:
            info["message"] = "motion disabled: the client was not armed (--execute)"
            return "failed", info
        if not self._execute.wait_for_server(timeout_sec=5.0):
            info["message"] = "execute_joint_trajectory not available — is the arm node up?"
            return "failed", info
        flown = traj if float(speed) == 1.0 else driver.dilate(traj, speed)
        # a guarded leg starts on a sample fresher than its own limit, not
        # one merely flowing (EFFORT_STALE_S): flatbot 2026-10-27 15:34, the
        # re-aim after the aim's computing started on a 0.3 s old one and its
        # guard called the stream lost 10 ms in, the arm never having moved
        compliant = bool(getattr(self, "compliant", False))
        live = self._fresh_joints(max_age=None if guard is None or compliant else 0.5 * self.GUARD_STALE_S)
        if live is None:
            info["message"] = "no fresh /joint_states — refusing to send a trajectory"
            return "failed", info
        why = driver.refusal(flown, live, JOINT_VMAX)
        if why:
            info["message"] = "refused before sending: " + why
            return "failed", info
        goal = ExecuteJointTrajectory.Goal()
        goal.trajectory = flown
        goal.control_mode = IMPEDANCE if compliant else POSITION
        if compliant:
            # every compliant goal names its profile: a bare one flies the
            # driver's session default, MEDIUM (Kq 80/30), as all six goals of
            # 2026-10-27 16:23 did. A POSITION goal names none: the driver
            # refuses gains it cannot apply
            spec = gains if gains is not None else getattr(self, "compliant_gains", COMPLIANT_RUN_GAINS)
            fill_gains(goal.gains, spec)
            info["gains"] = spec if isinstance(spec, CustomGains) else int(spec)
        goal.preemption = QUEUE  # never displace a goal that is still settling
        goal.sender_id = "box_opening"
        self._note_sent(goal)
        # what flew, at the price of a count and an append a feedback
        kept, every, heard = deque(maxlen=self.FEEDBACK_KEEP), self.FEEDBACK_EVERY, 0

        def _fb(msg):
            nonlocal heard
            info["progress"] = float(msg.feedback.fraction_complete)
            if heard % every == 0:
                kept.append((time.monotonic(), info["progress"], list(msg.feedback.actual.positions)))
            heard += 1
            if guard is not None:
                guard.on_progress(info["progress"])

        # goal_in_flight makes SIGINT set the flag instead of raising: the
        # loop below then delivers the cancel on a LIVE context (abort.py)
        if self._abort is not None:
            self._abort.goal_in_flight = True
        unwinding = False
        try:
            info.update(sent_at=time.monotonic(), sent_traj=flown, feedback_q=kept)
            send = spin_until_done(
                self.node,
                self._execute.send_goal_async(goal, feedback_callback=_fb),
                10.0,
            )
            if send is None or not send.accepted:
                if compliant:
                    why = "a mode change while moving, or its %s gains refused" % IMPEDANCE_NAMES.get(
                        info["gains"], "profile %s" % (info["gains"],)
                    )
                else:
                    why = "or a mode change while moving"
                info["message"] = "goal rejected by the driver (a stream session open, %s)" % why
                return "failed", info
            result_future = send.get_result_async()
            contact = False
            stopped = False
            aborted = False
            t0 = time.monotonic()
            try:
                if while_running is not None and guard is None:
                    # INSIDE the cancel backstop: a second Ctrl+C raised from
                    # the hosted plan must still reach the cancel below, and
                    # a first Ctrl+C that only set the flag while the motion
                    # finished under the plan must not be swallowed (review
                    # 2026-09-02)
                    try:
                        info["while_running"] = while_running()
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # a failed lookahead is not a failed leg
                        info["while_running_error"] = str(exc)
                    if self._abort is not None and self._abort.requested:
                        self._cancel_confirm(send, result_future)
                        aborted = True
                while not aborted and not result_future.done():
                    if self._abort is not None and self._abort.requested:
                        self._cancel_confirm(send, result_future)
                        aborted = True
                        break
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                    if guard is not None:
                        eff = self.wrist_efforts(max_age=self.GUARD_STALE_S)
                        t_drain = time.monotonic()
                        while eff is None and time.monotonic() - t_drain < self.GUARD_DRAIN_S:
                            rclpy.spin_once(self.node, timeout_sec=0.0)
                            eff = self.wrist_efforts(max_age=self.GUARD_STALE_S)
                        if eff is None and compliant:
                            pass  # it only watches: a silent stream stops nothing
                        elif eff is None:
                            # a guard that has lost its senses is no guard:
                            # stop the stroke instead of pressing blind
                            ok = self._stop_confirmed(send, result_future)
                            info["message"] = (
                                "effort stream lost during guarded leg — "
                                "cancelled (guard blind > %.1f s)%s"
                                % (self.GUARD_STALE_S, "" if ok else " (stop unconfirmed)")
                            )
                            return "failed", info
                        if eff is None:
                            tripped = False
                        elif getattr(guard, "wants_joints", False):
                            # the same /joint_states message as the efforts:
                            # the guard takes gravity out at these joints
                            tripped = guard.on_efforts(eff, list(self._q))
                        else:
                            tripped = guard.on_efforts(eff)
                        if tripped and compliant:
                            # watched, not obeyed: the leg flies on
                            info.setdefault("would_trip", round(float(guard.peak or 0.0), 2))
                        elif tripped:
                            if not self._stop_confirmed(send, result_future):
                                # never a contact height off an arm that may
                                # have flown on, nor a push on top of it
                                info["torque_peak"] = guard.peak
                                info["message"] = "guard tripped; the driver did not confirm the stop — check the arm"
                                return "failed", info
                            contact = True
                            break
                    # after the guard: the age it judges is the stream's, not
                    # a stop_when's own time (an in-flight confirm runs the
                    # wrist's depth fix here)
                    if stop_when is not None and stop_when(info["progress"]):
                        if not self._stop_confirmed(send, result_future):
                            info["message"] = "stop_when: the driver did not confirm the stop — check the arm"
                            return "failed", info
                        stopped = True
                        break
                    if time.monotonic() - t0 > 240:
                        spin_until_done(self.node, send.cancel_goal_async(), 3.0)
                        info["message"] = "execution watchdog timeout (240 s)"
                        return "failed", info
            except KeyboardInterrupt:
                # backstop: default-handler CLIs and the second-Ctrl+C
                # escalation land here; try the cancel, promise nothing
                spin_until_done(self.node, send.cancel_goal_async(), 3.0)
                log.info("\nCtrl+C — cancel sent (unconfirmed; check the arm)")
                if compliant:
                    self.release()  # LATEST_WINS: it displaces the goal too, if the cancel did not land
                raise
            if aborted:
                if compliant:
                    # the cancel left the spring pulling toward the goal's
                    # last reference, its feed-forward still on: let it go
                    self.release()
                raise KeyboardInterrupt  # unwind AFTER the confirmed cancel
        except BaseException:
            unwinding = True
            raise
        finally:
            if self._abort is not None:
                self._abort.goal_in_flight = False
                if self._abort.requested and not unwinding:
                    # a Ctrl+C that landed in the spin delivering the result
                    # (or a rejection) only set the flag: the goal is over,
                    # so nothing to cancel — but it must not be swallowed
                    raise KeyboardInterrupt
        if guard is not None:
            info["torque_peak"] = guard.peak
        if contact:
            info["message"] = "torque guard trip"
            return "touch", info
        if stopped:
            info["message"] = "stopped part-way: what it was watching for happened"
            return "stopped", info
        wrapped = result_future.result()
        if wrapped is None:
            info["message"] = "no result"
            return "failed", info
        result = wrapped.result
        if result.error_code == driver.SUCCESSFUL:
            info["message"] = "arrived"
            if "would_trip" in info:
                info["message"] = "arrived (compliant: the guard would have stopped it at %.1f Nm)" % info["would_trip"]
            return "arrived", info
        info["message"] = driver.result_message(result.error_code, result.error_string)
        return "failed", info

    # -- services / gripper ------------------------------------------------
    def set_world(self, path_or_name):
        """Box opening sends the planner no world (worlds.py): it plans in
        the one its deployment started it with, world_empty.yaml on flatbot,
        and refuses an empty scene sent at runtime anyway. Kept as the one
        place every plan and replan asks, so nothing sneaks a SetWorld in;
        `path_or_name` (worlds.PLANNER_WORLD) is what it records as held."""
        self.world_held = str(path_or_name)
        return True, "box opening sends no world: the planner plans in the one it was started with"

    def gripper_cmd(self, position):
        """Command the gripper and wait (knuckle rad: 0.0 open .. 0.8 closed);
        position None = query only (the live knuckle, no motion).
        Returns (ok, position, stalled)."""
        if position is None:
            return self._gripper_pos is not None, self._gripper_pos or 0.0, False
        handle = self.gripper_send(position)
        if handle is None:
            return False, 0.0, False
        return self.gripper_join(handle)

    def gripper_send(self, position):
        """Start a gripper command (knuckle rad) and return a handle WITHOUT
        waiting, or None when it cannot be sent.

        Lets a close overlap the motion that follows it — the fingers shut
        while the arm transits instead of the arm standing still for the
        whole close; the driver's gripper is not a control mode, so it rides
        alongside a trajectory by design. The caller MUST join before
        anything that depends on the fingers having arrived."""
        if self._abort is not None and self._abort.requested:
            raise KeyboardInterrupt  # the operator stopped the run: no more motion
        if not self._armed:
            return None
        if self._gripper_pos is None:
            t0 = time.monotonic()
            while self._gripper_pos is None and time.monotonic() - t0 < 2.0:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            if self._gripper_pos is None:
                return None  # no knuckle reading: nothing could confirm the command
        self._gripper_target = driver.setpoint_position(position)
        self._publish_gripper()
        return GripperHandle(float(position), float(self._gripper_pos), time.monotonic())

    def _publish_gripper(self):
        """Send the pending setpoint; also the re-send timer's callback."""
        if self._gripper_target is None:
            return
        msg = GripperSetpoint()
        msg.position = float(self._gripper_target)
        # sent on every message: the driver keeps neither speed nor force
        msg.speed = GRIPPER_SPEED
        msg.force = GRIPPER_FORCE
        self._gripper_pub.publish(msg)

    def gripper_join(self, handle):
        """Wait out a gripper_send: (ok, position, stalled), the same verdict
        as gripper_cmd — the overlap must not change what callers verify.

        The driver's gripper reports no result, so completion is read off the
        knuckle in /joint_states (driver.GripperWait): at the target; moved
        and then still (closed on the knob, a stall the grip band judges);
        or never moved at all (a failure). Re-sending stops once it is in.

        The wait is timed from HERE, not from the dispatch: a deferred command
        is meant to sit pending through the motion it overlaps, and that motion
        can outlast the timeout on its own — in slow mode the scan flight did,
        and the join reported a closed gripper as a failure without ever
        reading it (field 2026-09-14). GripperWait still counts "never moved"
        from the dispatch, where that question belongs."""
        wait = driver.GripperWait(handle.target, handle.start, handle.t0)
        deadline = time.monotonic() + self.GRIPPER_TIMEOUT_S
        try:
            while time.monotonic() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.02)
                now = time.monotonic()
                pos = self._gripper_pos
                fresh = pos is not None and now - self._eff_at <= self.GRIPPER_STREAM_GAP_S
                done = wait.update(handle.start if pos is None else pos, now, fresh)
                if done is not None:
                    return done
            return False, float(self._gripper_pos or 0.0), False
        finally:
            self._gripper_target = None
