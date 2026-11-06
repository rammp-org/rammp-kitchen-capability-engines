"""One `peer` action: plan to a standoff in front of the fridge's back wall, then teleop.

The phases, in order:

  scene homing   (scene_homing) the scene camera's fridge box and depth place a point in the
                 fridge (scene_homing.SceneFridge); go_to_ee_pose puts the wrist camera
                 scene_standoff_m in front of it, looking along scene_approach. This only gets
                 the wrist camera a view of the back wall; nothing is fine-tuned from it. No fridge
                 in the scene camera, or no plan: it is skipped, with a warning.
  fridge TF      wait up to fridge_tf_wait_s for the plane detector's fridge_frame (measured with
                 the arm still). Found: home again on it, then wall teleop. Not found: search
                 teleop until it is, then home on it and wall teleop.

This is the old FridgeTeleopNode's logic, run inside one execute() instead of a node's timer.
Everything the old node did to talk to the driver is now self.arm's:

  homing   self.arm.go_to_ee_pose(): cuRobo plans the tool frame (flange + arm.tool_z) to the
           standoff pose in arm.frame and the driver runs it. Blocking; False on a refusal or
           failure, with the reason in self.arm.last_error.
  teleop   `with self.arm.stream("ee_twist") as s: s.send(twist)`: the session is opened (after
           /list_controllers and discovery), and leaving the block, however it is left, sends a
           zero twist and closes it. /ee_state comes from self.arm.ee_state().

The joystick, the TF buffer and the debug publishers are this action's own entities
(create_action_subscription / create_action_tf_buffer / create_action_publisher): they exist
from the start of execute() to its end, and the engine destroys them safely.

Two teleop modes, one servo:

  wall     after homing: the stick slides the camera over the fridge's back wall (the followed
           plane), the camera facing it.
  search   no usable fridge TF within fridge_tf_wait_s (and search_when_no_fridge): the stick
           moves the camera in a box of +-search_range_m around where it is, its orientation
           held, so the plane detector gets another view. u = left, v = up, standoff = forward /
           back level with the floor. The first fridge TF measured with the arm still (let go of
           the stick) ends the search: the stream closes and it homes by itself, then wall teleop
           starts. No fridge after all goes back to searching.

The joystick only moves the camera: no buttons are read. Teleop ends on max_teleop_s (the action
succeeds), on the orchestrator's cancel (Cancelled: the block exits and the stream closes), or on
an arm error.

Homing uses only a fridge TF stamped follow_settle_s after the arm last moved (measured from
/ee_state's twist, and the end of every stream): an estimate from a moving camera is not a goal.

Fridge-frame convention: +z points INTO the back wall, away from the arm. The camera looks along
+z, the standoff sits at z = -standoff_distance, and the arm base is always at z < 0 in it.

Frames: B base_frame (parent of the fridge TF), F fridge_frame, S arm.frame (/ee_state and the
goals), W driver_ee_frame (what /ee_state reports, the flange), E ee_link (placed, limited
and oriented), G the tool frame go_to_ee_pose places (W + arm.tool_z along W's z), T
twist_frame (the axes of the twist setpoint).
"""

import math
import re
import time
from typing import NamedTuple

import numpy as np
import rammp_arm
from capability_engine import ArmError, EngineError, StateTimeout
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Joy
from tf2_ros import TransformException

from . import geometry as geo
from . import plane_follow as pf
from .scene_homing import SceneFridge

CONTROLLER = "ee_twist"
WAIT_STEP_S = 0.1  # how often the waits for TF poll (and honour pause and cancel)
SEARCH_CHECK_S = 0.2  # how often a search looks for the fridge TF
# /ee_state's twist above either of these means the arm is moving (homing waits for a fridge TF
# measured follow_settle_s after it last was)
STILL_LIN = 0.005  # m/s
STILL_ANG = 0.03  # rad/s
MIN_SPEED_SCALE = 0.01  # the driver refuses (not clamps) a go_to speed_scale below this
# cuRobo's refusal to plan from a start inside its joint-limit margin, one match per joint
LIMIT_MARGIN = re.compile(
    r"arm_joint_(\d+) is [-\d.]+ rad from its position limit \[\s*([-\d.]+),\s*([-\d.]+)\]"
    r",? inside joint_limit_margin_rad ([\d.]+)"
)
UNWIND_SPEED = 0.1  # rad/s, average joint speed of an unwind (min-jerk: peak ~1.9x)
UNWIND_POINTS = 25
LATCHED = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)

WALL, SEARCH = "wall", "search"

_FOLLOW_REJECT = {
    pf.TILT: "more than follow_max_tilt_deg from the homing plane",
    pf.SHIFT: "more than follow_max_shift_m from the homing wall offset",
    pf.STEP: "a jump of more than follow_max_step_deg / follow_max_step_m from the "
    "followed plane (did the detector lock onto another surface?)",
}


class _Latch(NamedTuple):
    """The plane homing used, all in S. The follower starts from it."""

    n: np.ndarray  # wall normal, INTO the wall
    d: float  # n . x = d
    anchor: np.ndarray  # fridge TF origin at homing (on the wall)
    up: np.ndarray  # world up
    # rotation from detector_rotation(n, up) to the fridge TF's own axes at homing (about z;
    # identity when up_frame / up_axis match the detector's fixed frame): keeps the homing roll.
    R_off: np.ndarray
    off_deg: float  # R_off's angle (warned about at homing when > 1 deg)
    stamp_ns: int


def _tf_matrix(tf) -> np.ndarray:
    t, r = tf.transform.translation, tf.transform.rotation
    return geo.transform_from_pq((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))


def facing(cfg) -> np.ndarray:
    """R_FE, the camera's target orientation in the fridge frame. ValueError on bad axes."""
    return geo.facing_rotation(
        geo.axis_vector(cfg["optical_axis"]),
        geo.axis_vector(cfg["roll_reference_ee_axis"]),
        geo.axis_vector(cfg["roll_reference_fridge_axis"]),
        math.radians(cfg["roll_deg"]),
    )


def follow_params(cfg) -> pf.FollowParams:
    return pf.FollowParams(
        w0=float(cfg["follow_w0"]),
        max_rate=math.radians(cfg["follow_max_rate_deg_s"]),
        max_shift_rate=float(cfg["follow_max_shift_rate"]),
        deadband_deg=float(cfg["follow_deadband_deg"]),
        deadband_m=float(cfg["follow_deadband_m"]),
        max_tilt_deg=float(cfg["follow_max_tilt_deg"]),
        max_shift_m=float(cfg["follow_max_shift_m"]),
        max_step_deg=float(cfg["follow_max_step_deg"]),
        max_step_m=float(cfg["follow_max_step_m"]),
    )


def search_frame(R_SE: np.ndarray, p_SE: np.ndarray, optical_E, up_S) -> np.ndarray:
    """T_SG for a search: origin at E, z = the optical axis made level (a virtual wall straight
    ahead), so u is left, v up and the standoff axis horizontal. Looking within ~6 deg of straight
    up or down, z is the optical axis itself. ValueError when it is exactly vertical."""
    opt = R_SE @ np.asarray(optical_E, dtype=float)
    up = np.asarray(up_S, dtype=float)
    n = opt - float(opt @ up) * up
    if float(np.linalg.norm(n)) < 0.1:
        n = opt
    return geo.make_transform(geo.gravity_aligned_rotation(n, up), p_SE)


class PeerSession:
    """The state of one `peer` action. Make it inside execute() (its entities are the action's)."""

    def __init__(self, engine, ctx, cfg, *, execute: bool, speed: float):
        self._node = engine
        self._arm = engine.arm
        self._ctx = ctx
        self._log = engine.get_logger()
        self.execute = execute
        self.speed = speed

        c = cfg
        self.rate_hz = float(c["rate_hz"])
        self.stream_timeout_s = float(c["stream_timeout_s"])
        self.base_frame = c["base_frame"]
        self.fridge_frame = c["fridge_frame"]
        self.verify_fridge_z = bool(c["verify_fridge_z"])
        self.driver_ee_frame = c["driver_ee_frame"]
        self.ee_link = c["ee_link"]
        self.up_frame = c["up_frame"]
        self.static_tf_timeout_s = float(c["static_tf_timeout_s"])
        self.fridge_tf_wait_s = float(c["fridge_tf_wait_s"])
        self.standoff = float(c["standoff_distance"])
        self.goal_offset = (float(c["offset_x"]), float(c["offset_y"]))
        self.twist_frame = c["twist_frame"]
        self.max_speed = float(c["max_speed"])
        self.standoff_speed = float(c["standoff_speed"])
        self.kp_lin = float(c["kp_lin"])
        self.kp_ang = float(c["kp_ang"])
        self.max_lin = float(c["max_lin"])
        self.max_ang = float(c["max_ang"])
        self.max_lead_m = float(c["max_lead_m"])
        self.lead_slowdown_m = float(c["lead_slowdown_m"])
        self.u_limits = [float(v) for v in c["u_limits"]]
        self.v_limits = [float(v) for v in c["v_limits"]]
        self.standoff_limits = [float(v) for v in c["standoff_limits"]]
        self.follow_mode = c["follow_mode"]
        self.follow_settle_s = float(c["follow_settle_s"])
        self.follow_stale_s = float(c["follow_stale_s"])
        self.search_when_no_fridge = bool(c["search_when_no_fridge"])
        self.search_range_m = float(c["search_range_m"])
        self.scene_homing = bool(c["scene_homing"])
        self.scene_wait_s = float(c["scene_wait_s"])
        self.scene_frames = int(c["scene_frames"])
        self.scene_standoff = float(c["scene_standoff_m"])
        self.scene_approach = c["scene_approach"]
        self._scene_forward_B = geo.axis_vector(c["scene_forward_axis"])
        self.unwind_limits = bool(c["unwind_limits"])
        self.unwind_extra_rad = float(c["unwind_extra_rad"])
        self.unwind_max_rad = float(c["unwind_max_rad"])
        self.joy_topic = c["joy_topic"]
        self.axis_u = int(c["axis_u"])
        self.axis_v = int(c["axis_v"])
        self.axis_standoff = int(c["axis_standoff"])
        self.axis_signs = [float(v) for v in c["axis_signs"]]
        self.deadzone = float(c["deadzone"])
        self.expo = float(c["expo"])
        self.joy_timeout_s = float(c["joy_timeout_s"])
        self.fridge_tf_timeout_s = float(c["fridge_tf_timeout_s"])
        self.ee_state_timeout_s = float(c["ee_state_timeout_s"])

        # Target orientation of E in F, as in the original homing; fixed, because F itself moves
        # with the TF. (node.py's _problems() checked the axes at startup.)
        self._R_FE = facing(cfg)
        self._optical_E = geo.axis_vector(c["optical_axis"])
        self._up_U = geo.axis_vector(c["up_axis"])
        self._follow_params = follow_params(cfg)

        # --- state --------------------------------------------------------------------------
        self._T_WE = None  # driver ee -> ee_link (rigid, from TF)
        self._T_EG = None  # ee_link -> the tool frame go_to_ee_pose places
        self._fridge_static_logged = False
        self._frame_checked = False
        self._joy = None
        self._joy_rx = 0.0
        self._bad_indices = set()
        self._faults = set()
        self._input_armed = False  # the stick has been seen at neutral since teleop began
        self._cmd = (0.0, 0.0, 0.0)  # stick (u, v, standoff), -1..1
        self._mode = WALL
        self._pending_latch = None  # the plane the last homing goal was computed with
        self._latch = None
        self._follower = None
        self._search = None  # (T_SG, R_SE) of the running search, fixed when it starts
        self._found_announced = False
        self._next_check = 0.0
        self._fridge_seen_ns = None
        self._follow_fresh_at = 0.0
        self._motion_ns = 0  # following's gate (follow_mode idle: while the stick moves)
        self._moved_ns = 0  # homing's gate: when the arm last moved (node clock, ns)
        self._was_engaged = False
        self._target_G = None  # servo target of E in the plane frame
        self._ff_prev = None  # (p_SW, R_SW) target on the previous tick, for feedforward

        # --- this action's ROS entities (the engine destroys them when it ends) --------------
        self._tf = engine.create_action_tf_buffer()
        engine.create_action_subscription(Joy, self.joy_topic, self._on_joy, 10)
        pub = engine.create_action_publisher
        self._goal_pub = pub(PoseStamped, "~/goal_pose", LATCHED)
        self._twist_pub = pub(TwistStamped, "~/commanded_twist", 10)
        self._servo_target_pub = pub(PoseStamped, "~/servo_target", 10)
        self._followed_plane_pub = pub(PoseStamped, "~/followed_plane", 10)
        self._scene = SceneFridge(engine, cfg) if self.scene_homing else None

    @property
    def _arm_frame(self) -> str:
        # Read when used, never at import: the engine sets it from the arm.frame parameter.
        return rammp_arm.FRAME

    def _now_ns(self) -> int:
        return self._node.get_clock().now().nanoseconds

    # ------------------------------------------------------------------ the action

    def run(self, *, teleop: bool, max_teleop_s) -> dict:
        """Scene-home, then home on the fridge TF and (with `teleop`) servo until a cancel or
        `max_teleop_s`. With no fridge TF to home on, search (with `teleop` and
        search_when_no_fridge) until one appears, then home on it by itself."""
        self._wait_arm_tf()
        scene_homed = self.scene_homing and self._scene_home()
        homes, searches, teleop_s, deadline = 0, 0, 0.0, None
        while True:
            found, why = self._wait_for_fridge()
            if found is None:
                if not teleop and scene_homed and homes == 0:
                    self._log.warning(f"scene-homed, but no usable fridge TF ({why})")
                    return {"ended": "scene_homed", "scene_homed": True, "homes": 0,
                            "searches": 0, "teleop_s": 0.0, "fridge_tf": why}
                if not (teleop and self.search_when_no_fridge):
                    raise EngineError(
                        "no_fridge",
                        f"no usable fridge TF within {self.fridge_tf_wait_s:.1f}s: {why} -- the "
                        f"arm holds where it is",
                        recoverable=True,
                    )
                mode = SEARCH
                searches += 1
                self._log.warning(
                    f"no usable fridge TF ({why}): searching -- move the stick to give the "
                    f"detector another view"
                )
            else:
                self._home(found)
                homes += 1
                if not teleop:
                    return {"ended": "homed", "scene_homed": scene_homed, "homes": homes,
                            "searches": 0, "teleop_s": 0.0}
                mode = WALL
            if deadline is None and max_teleop_s is not None:
                deadline = time.monotonic() + float(max_teleop_s)
            started = time.monotonic()
            why = self._teleop(mode, deadline)
            # The stream has closed: the arm comes to rest now. No fridge TF from before is a goal.
            self._moved_ns = self._now_ns()
            teleop_s += time.monotonic() - started
            if why != "found":  # only a search ends with "found": home on it
                return {"ended": why, "scene_homed": scene_homed, "homes": homes,
                        "searches": searches, "teleop_s": round(teleop_s, 1)}

    def _wait_arm_tf(self) -> None:
        deadline = time.monotonic() + self.static_tf_timeout_s
        while True:
            try:
                self._T_WE = self._lookup(self.driver_ee_frame, self.ee_link)
                break
            except TransformException as exc:
                if time.monotonic() > deadline:
                    raise EngineError(
                        "no_tf",
                        f"arm TF {self.driver_ee_frame} -> {self.ee_link} not available ({exc}). "
                        f"Is robot_state_publisher up, and do driver_ee_frame / ee_link "
                        f"exist in it?",
                        recoverable=True,
                    ) from exc
                self._ctx.checkpoint()
                time.sleep(WAIT_STEP_S)
        # go_to_ee_pose places the tool frame: the flange (W) + arm.tool_z along its z.
        T_WG = geo.make_transform(np.eye(3), (0.0, 0.0, rammp_arm.TOOL_Z))
        self._T_EG = geo.invert_transform(self._T_WE) @ T_WG
        self._log.info(
            f"{self.ee_link} is {np.linalg.norm(self._T_WE[:3, 3]):.3f} m from "
            f"{self.driver_ee_frame}; the tool frame is {np.linalg.norm(self._T_EG[:3, 3]):.3f} m "
            f"from {self.ee_link}; follow_mode = {self.follow_mode}"
        )

    # ------------------------------------------------------------------ homing

    def _settled(self, stamp_ns: int) -> bool:
        """Whether a fridge TF stamped `stamp_ns` was measured with the arm still (a static TF
        always was)."""
        return stamp_ns == 0 or stamp_ns >= self._moved_ns + int(self.follow_settle_s * 1e9)

    def _find_fridge(self, *, gated: bool = True):
        """(((position, rotation) of the tool frame goal in arm.frame, latch), '') from ONE fresh
        fridge TF, or (None, why). `gated`: only a TF measured with the arm still."""
        T_SF, why, stamp_ns = self._fridge_in(self._arm_frame)
        if T_SF is None:
            return None, why
        if gated and not self._settled(stamp_ns):
            return None, "the latest fridge TF was measured while the arm moved; waiting for one"
        latch, why = self._latch_from(T_SF, stamp_ns)
        if latch is None:
            return None, why
        return (self._goal_from(T_SF), latch), ""

    def _wait_for_fridge(self):
        """_find_fridge(), polled for up to fridge_tf_wait_s once the arm has settled: (found, '')
        or (None, why)."""
        self._ctx.progress("homing", 0.0, "waiting for the fridge TF")
        # from when an estimate can first pass the settle gate, not from now
        settle_left = (self._moved_ns - self._now_ns()) * 1e-9 + self.follow_settle_s
        deadline = time.monotonic() + max(0.0, settle_left) + self.fridge_tf_wait_s
        while True:
            found, why = self._find_fridge()
            if found is not None:
                return found, ""
            if time.monotonic() > deadline:
                return None, why
            self._log.warning(
                f"homing: waiting for the fridge TF ({why})", throttle_duration_sec=1.0
            )
            self._ctx.checkpoint()
            time.sleep(WAIT_STEP_S)

    def _home(self, found) -> None:
        (pos, R), latch = found
        if latch.off_deg > 1.0:
            self._log.warning(
                f"{self.fridge_frame}'s in-plane axes are {latch.off_deg:.1f} deg from world down "
                f"as up_frame / up_axis define it: the detector's fixed_frame +z is not this "
                f"up (or it used its near-horizontal fallback). Teleop keeps the homing roll"
            )
        self._pending_latch = latch
        quat = geo.matrix_to_quat(R)
        self._goal_pub.publish(self._pose_msg(pos, R))
        self._log.info(
            f"homing goal (tool frame in {self._arm_frame}): p=({pos[0]:.3f}, {pos[1]:.3f}, "
            f"{pos[2]:.3f}) q=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}) "
            f"[{self.ee_link} at standoff {self.standoff:.3f} m, offsets "
            f"({self.goal_offset[0]:.3f}, {self.goal_offset[1]:.3f}) in {self.fridge_frame}]"
        )
        ok, why = self._move_to("homing", (pos, R), "to the fridge standoff")
        if not ok:
            raise EngineError("refused", f"homing: {why} -- the arm holds where it is",
                              recoverable=True)

    def _move_to(self, phase: str, goal, label: str):
        """go_to_ee_pose the tool frame to `goal` = (position, rotation) in arm.frame; a dry run
        only plans it. (True, '') or (False, why). A start inside cuRobo's joint-limit margin
        (START_IN_LIMIT_MARGIN) is unwound first (unwind_limits) and the move tried once more."""
        ok, why = self._try_move(phase, goal, label, None)
        if ok or not self.unwind_limits or "START_IN_LIMIT_MARGIN" not in why:
            return ok, why
        ok, q_start, why_u = self._unwind(why)
        if not ok:
            return False, f"{why} (not unwound: {why_u})"
        return self._try_move(phase, goal, label, q_start)

    def _unwind(self, why: str):
        """Turn each joint the planner said is inside its limit margin back to unwind_extra_rad
        past the margin: one slow min-jerk joint trajectory, executed directly by the driver (no
        cuRobo: it would refuse the same start), only those joints moving. (True, joints the next
        plan starts from -- None: the measured ones --, '') or (False, None, why)."""
        hits = LIMIT_MARGIN.findall(why)
        if not hits:
            return False, None, "could not read the joint and its limits from the planner's message"
        q0 = np.asarray(self._arm.raw_joints(), dtype=float)
        q1 = q0.copy()
        for idx, lo, hi, margin in hits:
            j, m = int(idx) - 1, float(margin) + self.unwind_extra_rad
            q1[j] = min(max(q0[j], float(lo) + m), float(hi) - m)
        delta = q1 - q0
        big = float(np.max(np.abs(delta)))
        moves = ", ".join(
            f"{rammp_arm.JOINTS[j]} {math.degrees(delta[j]):+.1f} deg"
            for j in range(len(delta)) if abs(delta[j]) > 1e-6
        )
        if big > self.unwind_max_rad:
            return False, None, f"it needs {moves}, more than unwind_max_rad {self.unwind_max_rad}"
        if big < 1e-6:
            return False, None, "the joints are already outside the margin"
        if not self.execute:
            self._log.warning(f"dry run: would first turn {moves} away from the limit")
            return True, q1, ""
        self._log.warning(f"start inside cuRobo's joint-limit margin: turning {moves} first")
        self._ctx.progress("unwind", 0.0, f"turning {moves} away from its limit")
        T = max(1.0, big / UNWIND_SPEED)
        points = []
        for k in range(UNWIND_POINTS + 1):
            tau = k / UNWIND_POINTS
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
            points.append((tau * T, q0 + s * delta, ds * delta))
        res = self._arm.execute_joint_trajectory(list(rammp_arm.JOINTS), points)
        self._moved_ns = self._now_ns()
        if res.error_code != 0:
            return False, None, f"the unwind trajectory ended {res.error_code} {res.error_string}"
        time.sleep(0.3)  # let /joint_states catch up before the next plan reads them
        return True, None, ""

    def _try_move(self, phase: str, goal, label: str, q_start):
        """One go_to_ee_pose (dry run: plan_to_pose from `q_start`, else the measured joints)."""
        ctx, arm = self._ctx, self._arm
        pos, R = goal
        quat = geo.matrix_to_quat(R)
        if not self.execute:
            start = arm.raw_joints() if q_start is None else q_start
            plan = arm.plan_to_pose(pos, quat, start)
            if not plan.success:
                return False, f"dry run: no plan ({plan.message})"
            self._log.info(f"dry run: cuRobo planned {label} in {plan.planning_time:.2f}s")
            ctx.progress(phase, 100.0, f"dry run: planned {label}, not moved")
            return True, ""
        ctx.progress(phase, 10.0, f"planning and moving {label}")
        scale = max(MIN_SPEED_SCALE, rammp_arm.GO_TO_SPEED_SCALE * self.speed)
        ok = arm.go_to_ee_pose(pos, quat, label, scale)
        self._moved_ns = self._now_ns()  # it moved (or tried): no fridge TF from before is a goal
        if not ok:
            return False, arm.last_error
        ctx.progress(phase, 100.0, f"moved {label}")
        return True, ""

    # ------------------------------------------------------------------ scene homing

    def _scene_home(self) -> bool:
        """Put the wrist camera in front of the fridge the scene camera sees. True if it moved
        there (a dry run: planned it); False, with a warning, if it was skipped."""
        self._ctx.progress("scene_homing", 0.0, "locating the fridge in the scene camera")
        est, why = self._scene.estimate(self._ctx, self.scene_wait_s, self.scene_frames)
        if est is None:
            self._log.warning(f"scene homing skipped: {why}")
            return False
        p_cam, frame_id, _, summary = est
        try:
            T_SC = self._lookup(self._arm_frame, frame_id)
            up_S = self._up_S()
            d, why = self._scene_direction(T_SC, (T_SC @ np.append(p_cam, 1.0))[:3], up_S)
        except TransformException as exc:
            self._log.warning(f"scene homing skipped: no TF to {self._arm_frame} ({exc})")
            return False
        if d is None:
            self._log.warning(f"scene homing skipped: {why}")
            return False
        p_S = (T_SC @ np.append(p_cam, 1.0))[:3]
        # a stand-in fridge frame at the point, +z along the approach, the detector's in-plane
        # axes: the same facing (R_FE) and tool offset as homing on the real one
        T_SF = geo.make_transform(pf.detector_rotation(d, up_S), p_S)
        goal = self._goal_from(T_SF, standoff=self.scene_standoff, offset=(0.0, 0.0))
        self._goal_pub.publish(self._pose_msg(*goal))
        self._log.info(
            f"scene homing: fridge point ({p_S[0]:.3f}, {p_S[1]:.3f}, {p_S[2]:.3f}) in "
            f"{self._arm_frame} [{summary}], approach ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}) "
            f"({self.scene_approach}); {self.ee_link} {self.scene_standoff:.2f} m in front of it"
        )
        ok, why = self._move_to("scene_homing", goal, "in front of the fridge (scene camera)")
        if not ok:
            self._log.warning(f"scene homing skipped: {why}")
        return ok

    def _scene_direction(self, T_SC: np.ndarray, p_S: np.ndarray, up_S: np.ndarray):
        """(the approach, level, unit, in S, '') or (None, why). Raises TransformException."""
        if self.scene_approach == "robot_forward":
            d = self._lookup(self._arm_frame, self.base_frame)[:3, :3] @ self._scene_forward_B
        elif self.scene_approach == "camera_ray":
            d = p_S - T_SC[:3, 3]
        else:  # "from_arm": from the arm's base, S's origin
            d = p_S.copy()
        d = d - float(d @ up_S) * up_S
        n = float(np.linalg.norm(d))
        if n < 1e-3:
            return None, f"scene_approach {self.scene_approach} is vertical"
        d = d / n
        if float(p_S @ d) <= 0.0:
            return None, (
                f"the fridge point is behind the arm along scene_approach {self.scene_approach}"
            )
        return d, ""

    def _goal_from(self, T_SF: np.ndarray, standoff=None, offset=None):
        """(position, rotation) of the tool frame in arm.frame, for E at the standoff (default
        standoff_distance, offset_x / offset_y) in front of fridge frame T_SF."""
        standoff = self.standoff if standoff is None else standoff
        ox, oy = self.goal_offset if offset is None else offset
        p_FE = np.array([ox, oy, -standoff])
        T_SG = T_SF @ geo.make_transform(self._R_FE, p_FE) @ self._T_EG
        return T_SG[:3, 3], T_SG[:3, :3]

    def _up_S(self) -> np.ndarray:
        """World up in S. Raises TransformException."""
        return self._lookup(self._arm_frame, self.up_frame)[:3, :3] @ self._up_U

    def _latch_from(self, T_SF: np.ndarray, stamp_ns: int):
        """The plane of fridge TF `T_SF` (in S), or (None, why)."""
        try:
            up_S = self._up_S()
        except TransformException as exc:
            return None, f"no TF {self._arm_frame} -> {self.up_frame} ({exc})"
        n, origin = T_SF[:3, 2].copy(), T_SF[:3, 3].copy()
        try:
            geo.gravity_aligned_rotation(n, up_S)  # the stick's u/v must exist
            R_rebuilt = pf.detector_rotation(n, up_S)
        except ValueError as exc:
            return None, f"{self.fridge_frame}: {exc}"
        R_off = R_rebuilt.T @ T_SF[:3, :3]
        off_deg = math.degrees(float(np.linalg.norm(geo.rotvec_from_matrix(R_off))))
        return _Latch(n, float(n @ origin), origin, up_S, R_off, off_deg, stamp_ns), ""

    # ------------------------------------------------------------------ TF

    def _lookup(self, target: str, source: str) -> np.ndarray:
        """T_target_source from the latest TF. Raises TransformException."""
        if target == source:
            return np.eye(4)
        return _tf_matrix(self._tf.lookup_transform(target, source, Time()))

    def _fridge_in(self, frame: str):
        """(T_frame_F, '', stamp ns) from a fresh fridge TF, or (None, why, 0). A stamp of 0
        means the TF is static."""
        try:
            tf_BF = self._tf.lookup_transform(self.base_frame, self.fridge_frame, Time())
        except TransformException as exc:
            return None, f"no TF {self.base_frame} -> {self.fridge_frame} ({exc})", 0
        stamp_ns = Time.from_msg(tf_BF.header.stamp).nanoseconds
        if stamp_ns == 0:
            if not self._fridge_static_logged:
                self._fridge_static_logged = True
                self._log.info(
                    f"{self.base_frame} -> {self.fridge_frame} is static; fridge_tf_timeout_s "
                    f"does not apply (and there is nothing to follow)"
                )
        elif self.fridge_tf_timeout_s > 0.0:
            age = (self._now_ns() - stamp_ns) * 1e-9
            if age > self.fridge_tf_timeout_s:
                return None, (
                    f"fridge TF is {age:.2f}s old (> fridge_tf_timeout_s = "
                    f"{self.fridge_tf_timeout_s}s)"
                ), 0
        T_BF = _tf_matrix(tf_BF)
        try:
            T_XB = self._lookup(frame, self.base_frame)
            if self.verify_fridge_z:
                why = self._check_fridge_z(T_BF)
                if why:
                    return None, why, 0
        except TransformException as exc:
            return None, f"no TF {frame} / {self._arm_frame} -> {self.base_frame} ({exc})", 0
        return T_XB @ T_BF, "", stamp_ns

    def _check_fridge_z(self, T_BF: np.ndarray) -> str:
        """'' if the fridge frame follows the convention (the arm base at z < 0 in it), else why
        not. A frame that fails would put the goal behind the wall: refused, never flipped."""
        p_arm_B = self._lookup(self.base_frame, self._arm_frame)[:3, 3]
        z_arm = float((geo.invert_transform(T_BF) @ np.append(p_arm_B, 1.0))[2])
        if z_arm < 0.0:
            return ""
        return (
            f"{self.fridge_frame} +z points toward {self._arm_frame} (arm base at z = "
            f"{z_arm:+.2f} m), but this capability assumes +z points INTO the back wall. Fix the "
            f"RANSAC normal sign"
        )

    # ------------------------------------------------------------------ joystick

    def _on_joy(self, msg: Joy) -> None:  # on the executor: must never block
        self._joy = msg
        self._joy_rx = time.monotonic()

    def _warn_index(self, kind: str, idx: int, n: int) -> None:
        if (kind, idx) not in self._bad_indices:
            self._bad_indices.add((kind, idx))
            self._log.error(
                f"joy {kind} index {idx} is out of range (the joystick reports {n}); treating "
                f"it as released. Fix the axis_* indices"
            )

    def _axis(self, joy, idx: int) -> float:
        if idx < 0:
            return 0.0
        if idx >= len(joy.axes):
            self._warn_index("axis", idx, len(joy.axes))
            return 0.0
        v = float(joy.axes[idx])
        return v if math.isfinite(v) else 0.0

    def _read_input(self, joy):
        """Whether `joy` is fresh. Also sets self._cmd (the stick; no buttons are read)."""
        fresh = joy is not None and time.monotonic() - self._joy_rx <= self.joy_timeout_s
        if not fresh:
            self._input_armed = False
            self._cmd = (0.0, 0.0, 0.0)
            self._fault(
                "joy",
                f"no {self.joy_topic} message for > {self.joy_timeout_s}s; holding position. Is "
                f"joy_node running with autorepeat_rate > 0?",
            )
            return False
        self._clear_fault("joy")
        self._cmd = self._read_stick(joy)
        return True

    def _read_stick(self, joy):
        """(u, v, standoff) in -1..1: deadzone, sign, then expo. u + = LEFT, v + = UP,
        standoff + = closer."""

        def axis(idx: int, sign: float) -> float:
            if idx < 0:
                return 0.0
            a = geo.scalar_deadzone(self._axis(joy, idx), self.deadzone) * sign
            return math.copysign(abs(a) ** self.expo, a) if a else 0.0

        return (
            axis(self.axis_u, self.axis_signs[0]),
            axis(self.axis_v, self.axis_signs[1]),
            axis(self.axis_standoff, self.axis_signs[2]),
        )

    def _engaged(self, fresh: bool) -> bool:
        """Whether the stick moves the target this tick: deflected, and seen at neutral since
        teleop began (no deadman)."""
        if not fresh:
            return False
        active = any(c != 0.0 for c in self._cmd)
        if not active:
            self._input_armed = True
        elif not self._input_armed:
            self._log.warning(
                "joystick was already deflected when teleop started: let go of it (back to "
                "neutral) before it moves the arm",
                throttle_duration_sec=2.0,
            )
        return active and self._input_armed

    # ------------------------------------------------------------------ teleop

    def _teleop(self, mode: str, deadline) -> str:
        """Servo in `mode` (WALL or SEARCH) until a search finds the fridge ("found") or
        `deadline` ("max_teleop_s"). A cancel raises Cancelled out of it."""
        self._mode = mode
        if mode == WALL:
            self._latch = self._pending_latch  # the plane homing just used
            self._start_follower()
        else:
            self._latch = self._follower = None
            self._search = None  # fixed on the first tick, from where the arm is
            self._found_announced = False
            self._next_check = 0.0
        self._target_G = None  # seeded from the measured pose on the first tick
        self._ff_prev = None
        self._input_armed = False
        self._cmd = (0.0, 0.0, 0.0)
        if mode == WALL:
            phase = "teleop"
            message = "teleop: move the stick to move the camera, let go to hold"
        else:
            phase = "searching"
            message = (
                f"no fridge frame: move the stick to look around (within "
                f"{self.search_range_m:.2f} m); once the fridge is detected, let go of the stick "
                f"and it goes to the standoff by itself"
            )
        self._log.info(message)
        # A stream ignores pause (holding still in a paused send() would expire the session): say
        # so. A dry run sends nothing and pauses at its checkpoint.
        self._ctx.progress(phase, 50.0 if mode == WALL else 0.0, message,
                           pausable=not self.execute)
        if not self.execute:
            return self._loop(None, deadline)
        with self._arm.stream(CONTROLLER, timeout_s=self.stream_timeout_s) as s:
            return self._loop(s.send, deadline)

    def _loop(self, send, deadline) -> str:
        period = 1.0 / self.rate_hz
        last = next_t = time.monotonic()
        while True:
            now = time.monotonic()
            dt = min(max(now - last, 0.0), 0.1)
            last = now
            fresh = self._read_input(self._joy)
            if deadline is not None and now > deadline:
                self._log.info("teleop: max_teleop_s reached")
                return "max_teleop_s"
            if self._mode == SEARCH and now >= self._next_check:
                self._next_check = now + SEARCH_CHECK_S
                if self._search_check():
                    return "found"
            v, w = self._command(dt, self._engaged(fresh))
            if send is not None:
                send([*v, *w])  # raises Cancelled once cancelled; the block's exit closes
            else:
                self._ctx.checkpoint()
            next_t += period
            sleep = next_t - time.monotonic()
            if sleep > 0.0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # fell behind: don't burst to catch up

    def _search_check(self) -> bool:
        """Announce a fridge TF appearing (or going) while searching; True to home on it now (one
        measured with the arm still)."""
        found, why = self._find_fridge(gated=False)
        if found is None:
            if self._found_announced:
                self._found_announced = False
                self._log.warning(f"search: lost the fridge TF ({why})")
            return False
        if not self._found_announced:
            self._found_announced = True
            message = "fridge detected: let go of the stick, it goes to the standoff once still"
            self._log.info(f"search: {message}")
            self._ctx.progress("searching", 50.0, message, pausable=not self.execute)
        if self._settled(found[1].stamp_ns):
            self._log.info("search: fridge TF measured with the arm still, homing")
            return True
        return False

    # ------------------------------------------------------------------ faults

    def _fault(self, key: str, message: str) -> None:
        """Warn when a fault starts, then every 5 s while it lasts."""
        if key not in self._faults:
            self._faults.add(key)
            self._log.warning(message)
        else:
            self._log.warning(message, throttle_duration_sec=5.0)

    def _clear_fault(self, key: str) -> None:
        if key in self._faults:
            self._faults.discard(key)
            self._log.info(f"resumed ({key} fault cleared)")

    def _emit(self, v: np.ndarray, w: np.ndarray):
        dbg = TwistStamped()
        dbg.header.stamp = self._node.get_clock().now().to_msg()
        dbg.header.frame_id = self.twist_frame
        dbg.twist.linear.x, dbg.twist.linear.y, dbg.twist.linear.z = (float(x) for x in v)
        dbg.twist.angular.x, dbg.twist.angular.y, dbg.twist.angular.z = (float(x) for x in w)
        self._twist_pub.publish(dbg)
        return v, w

    def _hold(self, key: str, message: str):
        self._fault(key, message)
        self._ff_prev = None  # the next target is not continuous with the last
        return self._emit(np.zeros(3), np.zeros(3))

    # ------------------------------------------------------------------ following

    def _start_follower(self) -> None:
        latch = self._latch
        self._follower = pf.PlaneFollower(latch.n, latch.d, latch.anchor, self._follow_params)
        self._fridge_seen_ns = latch.stamp_ns  # already used: it IS the latch
        self._follow_fresh_at = time.monotonic()
        # the arm has just been moving (homing): estimates from then are suspect
        self._motion_ns = self._now_ns()
        self._was_engaged = False

    def _update_follow(self, dt: float, engaged: bool) -> None:
        """Feed new fridge estimates to the follower and advance it one tick."""
        f = self._follower
        if self.follow_mode == "off":
            return
        if self.follow_mode == "idle" and engaged:
            if not self._was_engaged:
                f.brake()  # come to a smooth stop; no new estimates while moving
            self._motion_ns = self._now_ns()
        else:
            self._ingest_fridge()
        self._was_engaged = engaged
        f.step(dt)

    def _ingest_fridge(self) -> None:
        T_SF, why, stamp_ns = self._fridge_in(self._arm_frame)
        if T_SF is not None and stamp_ns != self._fridge_seen_ns:
            self._fridge_seen_ns = stamp_ns
            self._follow_fresh_at = time.monotonic()
            if stamp_ns < self._motion_ns + int(self.follow_settle_s * 1e9):
                return  # measured while (or just after) the camera was moving
            n = T_SF[:3, 2]
            res = self._follower.offer(n, float(n @ T_SF[:3, 3]))
            if res == pf.ACCEPTED:
                self._log.info(
                    f"following: new {self.fridge_frame} estimate, "
                    f"{self._follower.ref_tilt_deg:.1f} deg / "
                    f"{1000 * self._follower.ref_shift_m:+.0f} mm from homing",
                    throttle_duration_sec=1.0,
                )
            elif res in _FOLLOW_REJECT:
                self._log.warning(
                    f"following: ignored a {self.fridge_frame} estimate, {_FOLLOW_REJECT[res]}; "
                    f"holding the followed plane. If the wall really is there, cancel and run "
                    f"peer again",
                    throttle_duration_sec=2.0,
                )
            return
        idle_s = time.monotonic() - self._follow_fresh_at
        if idle_s > self.follow_stale_s and self._fridge_seen_ns != 0:
            reason = f" ({why})" if T_SF is None else ""
            self._log.warning(
                f"following: no new {self.fridge_frame} for {idle_s:.1f}s{reason}; holding the "
                f"followed plane. Is the wall within the camera's depth range?",
                throttle_duration_sec=5.0,
            )

    def _plane_now(self):
        """(T_SG, R_S of the plane frame to show, R_SE target) this tick. G is the plane frame,
        gravity-aligned (u left, v up). WALL: the latched anchor projected onto the followed
        plane, facing it. SEARCH: where the search started, its orientation held."""
        if self._mode == SEARCH:
            T_SG, R_SE = self._search
            return T_SG, T_SG[:3, :3], R_SE
        latch, f = self._latch, self._follower
        n = f.normal
        T_SG = geo.make_transform(geo.gravity_aligned_rotation(n, latch.up), f.origin)
        R_SF = pf.detector_rotation(n, latch.up) @ latch.R_off
        return T_SG, R_SF, R_SF @ self._R_FE

    def _start_search(self, T_SE: np.ndarray) -> None:
        """Fix the search's frame at E's measured pose. TransformException without up."""
        try:
            T_SG = search_frame(T_SE[:3, :3], T_SE[:3, 3], self._optical_E, self._up_S())
        except ValueError as exc:
            raise EngineError(
                "no_fridge",
                f"no fridge TF, and the camera looks straight up or down ({exc}): cannot "
                f"search -- the arm holds where it is",
                recoverable=True,
            ) from exc
        self._search = (T_SG, T_SE[:3, :3].copy())

    # ------------------------------------------------------------------ the servo

    def _limits(self):
        """(u, v, standoff) limits of the target: the wall's, or a search's box."""
        if self._mode == SEARCH:
            r = self.search_range_m
            return [-r, r], [-r, r], [-r, r]
        return self.u_limits, self.v_limits, self.standoff_limits

    def _slide(self, cu: float, cv: float, cs: float, dt: float, scale: float):
        """The target after one tick of stick input at `scale` x full speed."""
        u, v, z = self._target_G
        s = -z
        k = scale * dt
        u_lim, v_lim, s_lim = self._limits()
        return np.array(
            [
                self._limited(u, u + cu * self.max_speed * k, u_lim),
                self._limited(v, v + cv * self.max_speed * k, v_lim),
                -self._limited(s, s - cs * self.standoff_speed * k, s_lim),
            ]
        )

    def _lead_scale(self, lead: float) -> float:
        """1 up to lead_slowdown_m, falling linearly to 0 at max_lead_m."""
        if lead <= self.lead_slowdown_m:
            return 1.0
        if lead >= self.max_lead_m:
            return 0.0
        return (self.max_lead_m - lead) / (self.max_lead_m - self.lead_slowdown_m)

    @staticmethod
    def _limited(cur: float, new: float, lim) -> float:
        """Clamp to lim, but never pull a target that already sits outside it."""
        lo, hi = min(lim[0], cur), max(lim[1], cur)
        return min(max(new, lo), hi)

    def _seed_target(self, T_SG: np.ndarray, p_SE: np.ndarray) -> None:
        self._target_G = (geo.invert_transform(T_SG) @ np.append(p_SE, 1.0))[:3]
        u, v, s = self._target_G[0], self._target_G[1], -self._target_G[2]
        where = "the plane origin" if self._mode == WALL else "where the search started"
        self._log.info(
            f"servo target at u={u:+.3f} v={v:+.3f} standoff={s:.3f} m (u = left, v = up from "
            f"{where})"
        )
        u_lim, v_lim, s_lim = self._limits()
        if not (
            u_lim[0] <= u <= u_lim[1] and v_lim[0] <= v <= v_lim[1] and s_lim[0] <= s <= s_lim[1]
        ):
            self._log.warning(
                "the arm starts outside u_limits / v_limits / standoff_limits; the stick only "
                "moves it back toward them"
            )

    def _ee_state(self):
        """/ee_state, or None when it is missing or older than ee_state_timeout_s (the loop then
        cannot be closed). Errors other than staleness (control taken, ...) propagate."""
        try:
            ee = self._arm.ee_state()
        except ArmError as exc:
            if type(exc) not in (ArmError, StateTimeout):  # a stale read is a plain ArmError
                raise
            self._fault("ee_state", f"/ee_state: {exc}; zero twist (cannot close the loop)")
            return None
        if ee.age_s > self.ee_state_timeout_s:
            self._fault(
                "ee_state",
                f"/ee_state is {ee.age_s:.2f}s old (> ee_state_timeout_s); zero twist",
            )
            return None
        if not self._frame_checked:
            self._frame_checked = True
            if ee.frame_id.lstrip("/") != self._arm_frame.lstrip("/"):
                self._log.error(
                    f"/ee_state is in {ee.frame_id!r}, not arm.frame {self._arm_frame!r}: set "
                    f"arm.frame to the driver's frame"
                )
        self._clear_fault("ee_state")
        if (
            float(np.linalg.norm(ee.linear)) > STILL_LIN
            or float(np.linalg.norm(ee.angular)) > STILL_ANG
        ):
            self._moved_ns = self._now_ns()
        return ee

    def _command(self, dt: float, engaged: bool):
        """(linear, angular) twist for this tick, in twist_frame axes.

        The stick slides a target (u, v, standoff) for E in the plane frame; the plane and the
        target orientation come from the FOLLOWED plane (WALL) or the search's start (SEARCH). W
        is servoed onto the matching W pose,
            v = v_ff + kp_lin * (p_target - p),  w = w_ff + kp_ang * rot_err,
        both capped, (v_ff, w_ff) being the rate of change of the W target. With no input and no
        new estimate the target stays put and the servo holds it (a zero twist would drift:
        ee_twist's null-space posture term moves the arm under a zero setpoint).
        """
        if self._mode == WALL:
            self._update_follow(dt, engaged)
        ee = self._ee_state()
        if ee is None:
            self._ff_prev = None
            return self._emit(np.zeros(3), np.zeros(3))
        try:
            R_TS = self._lookup(self.twist_frame, self._arm_frame)[:3, :3]
        except TransformException as exc:
            return self._hold(
                "twist_frame", f"no TF {self.twist_frame} -> {self._arm_frame} ({exc}); zero twist"
            )
        self._clear_fault("twist_frame")

        T_SW = geo.transform_from_pq(ee.position, ee.orientation)
        T_SE = T_SW @ self._T_WE
        p_SE = T_SE[:3, 3]
        if self._mode == SEARCH and self._search is None:
            try:
                self._start_search(T_SE)
            except TransformException as exc:
                return self._hold(
                    "up_frame", f"no TF {self._arm_frame} -> {self.up_frame} ({exc}); zero twist"
                )
            self._clear_fault("up_frame")
        T_SG, R_S_plane, R_SE_t = self._plane_now()
        if self._target_G is None:
            self._seed_target(T_SG, p_SE)
            self._ff_prev = None

        def in_S(p_G: np.ndarray) -> np.ndarray:
            return (T_SG @ np.append(p_G, 1.0))[:3]

        # 1. slide the target with the stick (only while engaged)
        cu, cv, cs = self._cmd if engaged else (0.0, 0.0, 0.0)
        if dt > 0.0 and (cu or cv or cs):
            mag = math.hypot(cu, cv)
            if mag > 1.0:  # diagonal no faster than straight
                cu, cv = cu / mag, cv / mag
            lead_now = float(np.linalg.norm(in_S(self._target_G) - p_SE))
            nxt = self._slide(cu, cv, cs, dt, 1.0)
            if float(np.linalg.norm(in_S(nxt) - p_SE)) >= lead_now:
                # Moving away from the arm: slow down smoothly as it falls behind, instead of
                # stop-go. Moves back toward the arm always run at full speed.
                scale = self._lead_scale(lead_now)
                nxt = self._slide(cu, cv, cs, dt, scale) if scale > 0.0 else None
                if scale < 1.0:
                    self._log.warning(
                        f"arm is {1000 * lead_now:.0f} mm behind the target: slowing the target "
                        f"to {100 * scale:.0f}% (lower max_speed if this persists)",
                        throttle_duration_sec=2.0,
                    )
            if nxt is not None:
                self._target_G = nxt

        # 2. target for E -> target for W
        p_SE_t = in_S(self._target_G)
        R_SW_t = R_SE_t @ self._T_WE[:3, :3].T
        p_SW_t = p_SE_t - R_SW_t @ self._T_WE[:3, 3]

        # 3. feedforward: how fast the W target itself moves (stick, following, lever arm)
        v_ff, w_ff = np.zeros(3), np.zeros(3)
        if self._ff_prev is not None and dt > 0.0:
            p_prev, R_prev = self._ff_prev
            v_ff = (p_SW_t - p_prev) / dt
            w_ff = geo.rotvec_from_matrix(R_SW_t @ R_prev.T) / dt
        self._ff_prev = (p_SW_t, R_SW_t)

        # 4. servo W onto it
        v = geo.clamp_norm(v_ff + self.kp_lin * (p_SW_t - T_SW[:3, 3]), self.max_lin)
        w = geo.clamp_norm(
            w_ff + self.kp_ang * geo.rotvec_from_matrix(R_SW_t @ T_SW[:3, :3].T), self.max_ang
        )
        self._servo_target_pub.publish(self._pose_msg(p_SE_t, R_SE_t))
        self._followed_plane_pub.publish(self._pose_msg(T_SG[:3, 3], R_S_plane))
        return self._emit(R_TS @ v, R_TS @ w)

    def _pose_msg(self, p, R) -> PoseStamped:
        """A pose in arm.frame (RViz: Axes)."""
        msg = PoseStamped()
        msg.header.frame_id = self._arm_frame
        msg.header.stamp = self._node.get_clock().now().to_msg()
        q = geo.matrix_to_quat(R)
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(x) for x in p)
        o = msg.pose.orientation
        o.x, o.y, o.z, o.w = (float(x) for x in q)
        return msg