"""PressArm: the button press's view of the arm, on the engine's `self.arm` (RAMMP's arm driver).

press_button reads the arm's state and sends every motion through this class. It owns no ROS
entity: `self.arm` makes them for the running action and settles them when it ends.

Reads:
  joints(), state()   self.arm.joint_states() (J1/J3/J5/J7 wrapped to +-pi); stale state raises
  ee_pose()           self.arm.ee_state() (the flange) + TOOL_Z along its z = Kinova's tool_frame
  gripper             self.arm.gripper_state() (0 open .. 1 closed)
  status()            self.arm.not_ready() (e-stop, control), diagnostics, tool-frame check

Moves:
  smooth_poses()      short straight legs (approach, press): sends exactly the joint path that
                      kinematics.precheck_poses checked, as one execute_joint_trajectory goal,
                      timed by a minimum-jerk profile capped at CART_SPEED / JOINT_SPEED
  go_to_ee_pose()     long moves (staging, back to start, bail): the driver plans with cuRobo and
                      executes at the arm.go_to_speed_scale parameter. The path cannot be previewed
                      and only the goal pose is gated here; cuRobo's world has the microwave
                      and counter in it (collision.PlannerWorld, pushed by the run)
  press_contact()     the press: in along the panel normal until the wrist torques feel the button, on
                      the driver's ee_twist stream
  close_gripper()     once, at preflight, on execute runs only; the gripper is never opened

Safety: a dry run sends nothing; a leg is refused if the arm moved since it was planned
(START_DRIFT_DEG); the driver aborts if a joint lags its reference by PATH_TOL_RAD. A sent motion
that fails raises MotionFailed and the arm holds where it is: nothing here moves it again. A cancel
of the action stops the motion in flight (self.arm raises Cancelled).
"""

from __future__ import annotations

import time

import numpy as np
import rammp_arm
from capability_engine import ArmError
from capability_engine.arm import QUEUE
from scipy.spatial.transform import Rotation

from button_press import Abort, MotionFailed, say
from button_press.geometry import max_joint_delta_deg
from button_press.kinematics import MAX_REACH_M, Z_RANGE_M

# /ee_state is the flange (end_effector_link): checked 2026-10-07, tf2 base_link -> end_effector_link
# == /ee_state, and sim tool_frame FK at the real joints == /ee_state + 0.120 m along the flange z
# to 0.1 mm. The button press's PREPRESS_EE_OFFSET_M was measured in Kinova's tool_frame (= the
# Kortex tool pose = flange + 0.120 m), so ee_pose() reports that. preflight re-checks it live
# (TOOL_FRAME_CHECK_M): if the driver's /ee_state ever changes frame, the run refuses.
TOOL_Z = 0.120
TOOL_FRAME_CHECK_M = 0.005
CONTINUOUS = [0, 2, 4, 6]  # J1/J3/J5/J7 spin freely
DT = 0.02  # trajectory sample spacing (cuRobo's interpolation_dt)
# Speed: the old stack ran with Kortex's low/medium preset. These are peaks of the min-jerk profile,
# the values microwave/sheppy_arm.py ran attended on this arm before "a little faster" (2026-10-07).
CART_SPEED = 0.06  # m/s
JOINT_SPEED = 0.35  # rad/s, on any joint, whatever the Cartesian length says
ROT_EQUIV_M = 0.15  # a 1 rad wrist turn counts as 15 cm of travel when timing a leg
MIN_DURATION_S = 1.0
PATH_TOL_RAD = 0.15  # ~8.6 deg tracking guard per joint (driver-side abort), = sheppy_arm.py
TRACK_ABORT_M = 0.01  # final tool position this far from the target -> hold (= old arm.py)
START_DRIFT_DEG = 0.5  # the arm must still be where a leg was planned from when it is sent
DIAG_LEVELS = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}
# ---- long moves through the driver's go_to_ee_pose (staging, back to start, bail): the driver
# asks RAMMP's cuRobo for a plan from the measured joints and executes it. Its target is the
# Kinova tool_frame in rammp_arm.FRAME, the same frame as ee_pose(). What this side cannot do: see
# or gate the planned path before it runs, or dry-run it. Only the goal pose is gated here.
# cuRobo's world has the microwave and its counter as boxes (collision.py; the run pushes them before any
# long move), and on flatbot the wheelchair and user as the planner config's own layers.
# Speed: self.arm's arm.go_to_speed_scale parameter (0.3 by default). 2026-10-14: at 1.0 a 51 cm
# staging move from HOME took 1.16 s and hit the microwave cuRobo cannot see, faster than anyone
# could reach the e-stop; 0.15 made it ~8 s.
# Preflight's close (close_gripper). /gripper_state.position: 0 open .. 1 closed; the press needs
# the closed fingertips.
GRIPPER_CLOSED_ABOVE = 0.7
GRIPPER_CLOSE_SPEED = 0.5
GRIPPER_CLOSE_FORCE = 0.5
GRIPPER_CLOSE_TIMEOUT_S = 6.0
# ---- the contact press's streaming session (self.arm.stream). Leaving it sends a zero twist and
# closes the session, which latches a hold at the measured q (no slew back to the last setpoint).
STREAM_RATE_HZ = 100.0  # setpoint rate (10x self.arm's stream timeout, 0.1 s)
STREAM_START_CHECK_M = 0.002  # the arm must be at the pre-press spot before a session opens
# ---- contact press (press_contact): box-opening's two-stage press
# (RAMMP-box-opening primitives/core.py press_stroke + press_push, runtime/guards.py TorqueGuard),
# along the panel normal on the driver's ee_twist controller. TOUCH: push until a wrist joint's
# torque moves CONTACT_TOUCH_NM off its baseline -- the fingertip met the button, located by the
# arm, not the camera. PUSH: CONTACT_PUSH_M further from there, position-bounded, the button's
# stop (CONTACT_STOP_NM) ending it early. One stroke until the torque builds up is what shoved
# the microwave and the table under it back 2026-10-15 (as box-opening's single hard stroke
# compressed the whole container, 2026-09-03): a force only the button's bottoming out can give
# must not be what ends the stroke. ee_twist integrates the twist into a STIFF held reference, so
# both bounds apply to the COMMANDED travel, and every stop CLOSES the session (the driver then
# holds at the measured q, which drops the push; a streamed zero would keep pushing).
TWIST_CONTROLLER = "ee_twist"
CONTACT_SPEED_MPS = 0.01  # along the normal, both stages
CONTACT_BASELINE_S = 0.2  # hold still first; the baseline is the wrist torques' mean over it (GUARD_MEAN_S)
CONTACT_WRIST = slice(3, 7)  # joints 4-7: the torques box-opening's guard watches (client.wrist_efforts)
CONTACT_TOUCH_NM = 3.0  # touch = any wrist torque this far off its baseline (box-opening contact_nm)
CONTACT_MIN_TOUCH_M = 0.004  # a touch before this struck something in front of the panel (box-opening
# EARLY_TRIP_FRAC). Was 0.008: 2026-10-15 real contact starts 7-10 mm in and a
# good press that clicked registered at 7.6 mm, refused as "struck something else"
CONTACT_MAX_TOUCH_M = 0.03  # no touch by this: no surface -- the panel is not where the camera put it
CONTACT_PUSH_M = 0.001  # the press: at most this far past the touch. Was 0.008 (box-opening's
# button_travel_m), then 0.004. 2026-10-15: the wrist torques only register the
# touch 2-4 mm after real contact (1-2 N.m of free-air noise keeps the threshold
# at 3), so the finger is already in; pushing on bent it (held 15.5 deg off the
# push line) off the dome and it sprang back over it on the retract: a double
# press. 1 mm / 4 N.m clicked once.
CONTACT_STOP_NM = 4.0  # the button's stop felt during the push: also the press (the backstop). Was 7.0
# (peaks 6.9-7.2 N.m), then 5.0 (2026-10-15): less force, less bend, less spring-back
CONTACT_START_S = 0.2  # launch transient: deviations observed, never judged, before this
CONTACT_LATERAL_ABORT_M = 0.01  # sideways off the press line -> close and hold
CONTACT_TIMEOUT_S = (CONTACT_MAX_TOUCH_M + CONTACT_PUSH_M) / CONTACT_SPEED_MPS + 1.0


class ContactPress:
    """The two stages of the contact press, fed one sample per tick (pure: unit-tested).

    update(t, commanded, along, lateral, dev_nm) -> None (keep pushing) or (kind, message).
    t: s since the push started; commanded / along: m into the panel from the pre-press spot,
    commanded and measured; lateral: m off the press line; dev_nm: the largest wrist torque
    deviation from the baseline. Kinds "pressed" (the push bound) and "stop" (the button's stop
    felt during the push) are presses; "early", "no_surface", "lateral", "timeout" are not.
    touch_m / touch_cmd: where the touch tripped (None before)."""

    def __init__(
        self,
        touch_nm=CONTACT_TOUCH_NM,
        min_touch=CONTACT_MIN_TOUCH_M,
        max_touch=CONTACT_MAX_TOUCH_M,
        push=CONTACT_PUSH_M,
        stop_nm=CONTACT_STOP_NM,
        start_s=CONTACT_START_S,
        lateral_abort=CONTACT_LATERAL_ABORT_M,
        timeout_s=CONTACT_TIMEOUT_S,
    ):
        self.touch_nm, self.min_touch, self.max_touch, self.push = touch_nm, min_touch, max_touch, push
        self.stop_nm, self.start_s, self.lateral_abort, self.timeout_s = stop_nm, start_s, lateral_abort, timeout_s
        self.touch_m = self.touch_cmd = None

    @property
    def phase(self):
        return "touch" if self.touch_m is None else "push"

    def update(self, t, commanded, along, lateral, dev_nm):
        if lateral > self.lateral_abort:
            return "lateral", (
                f"arm moved {lateral*1000:.1f} mm sideways off the press line " f"(limit {self.lateral_abort*1000:.0f})"
            )
        if t > self.timeout_s:
            return "timeout", f"still pushing after {t:.1f} s ({along*1000:.1f} mm in)"
        judged = t >= self.start_s
        if self.touch_m is None:
            if judged and dev_nm > self.touch_nm:
                if along < self.min_touch:
                    return "early", (
                        f"touched at {along*1000:.1f} mm in ({dev_nm:.1f} N.m), before the panel "
                        f"(>= {self.min_touch*1000:.0f} mm) -- struck something else"
                    )
                self.touch_m, self.touch_cmd = along, commanded
                return None
            if commanded >= self.max_touch:
                return "no_surface", (
                    f"no touch within {self.max_touch*1000:.0f} mm -- the panel is not where "
                    "the camera put it, or the microwave is sliding"
                )
            return None
        if dev_nm > self.stop_nm:
            return (
                "stop",
                f"the button's stop felt {(along - self.touch_m)*1000:.1f} mm past the touch ({dev_nm:.1f} N.m)",
            )
        if commanded - self.touch_cmd >= self.push:
            return "pressed", f"{self.push*1000:.0f} mm past the touch (at {self.touch_m*1000:.1f} mm in)"
        return None


def wrap(q):
    """Joint vector with the continuous joints (J1/J3/J5/J7) wrapped to [-pi, pi)."""
    q = np.asarray(q, dtype=float).copy()
    q[CONTINUOUS] = (q[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
    return q


def unwrap_from(q_start, knots):
    """Knots made continuous from `q_start`: each free joint steps the short way from the previous knot."""
    out, prev = [], np.asarray(q_start, dtype=float)
    for k in knots:
        d = np.asarray(k, dtype=float) - prev
        d[CONTINUOUS] = (d[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
        prev = prev + d
        out.append(prev.copy())
    return out


def min_jerk_samples(path, s_knots, duration):
    """Positions + velocities every DT along a joint path (knots at path parameter `s_knots`, 0..L),
    timed by a minimum-jerk s(t) over `duration`, piecewise-linear between knots."""
    path, length = np.asarray(path), float(s_knots[-1])
    n = max(2, int(np.ceil(duration / DT)) + 1)
    t = np.linspace(0.0, duration, n)
    u = t / duration
    s = length * (10 * u**3 - 15 * u**4 + 6 * u**5)
    q = np.stack([np.interp(s, s_knots, path[:, j]) for j in range(path.shape[1])], axis=1)
    qd = np.gradient(q, t, axis=0)
    qd[0] = qd[-1] = 0.0
    return t, q, qd


def leg_timing(q0, knots, poses, p0, R0, cart_speed=CART_SPEED, joint_speed=JOINT_SPEED):
    """Time a leg of tool `poses` that starts at joints q0 / tool pose (p0, R0).

    Returns (joint path from q0, path length at each knot, duration). The duration keeps the
    min-jerk peak speed (1.875 x the average) under `cart_speed` (CART_SPEED unless a caller asks
    for more) in tool travel (a wrist turn counts via ROT_EQUIV_M) and under `joint_speed` (JOINT_SPEED)
    on every joint."""
    path = [np.asarray(q0, dtype=float)] + unwrap_from(q0, knots)
    seg, prev_p, prev_R = [], np.asarray(p0, dtype=float), np.asarray(R0, dtype=float)
    for pt, Rt in poses:
        turn = Rotation.from_matrix(np.asarray(Rt) @ prev_R.T).magnitude()
        seg.append(max(1e-4, float(np.linalg.norm(np.asarray(pt) - prev_p)), turn * ROT_EQUIV_M))
        prev_p, prev_R = np.asarray(pt, dtype=float), np.asarray(Rt, dtype=float)
    s_knots = np.concatenate([[0.0], np.cumsum(seg)])
    joint_travel = float(np.sum([np.max(np.abs(b - a)) for a, b in zip(path[:-1], path[1:], strict=False)]))
    duration = max(MIN_DURATION_S, 1.875 * s_knots[-1] / cart_speed, 1.875 * joint_travel / joint_speed)
    return np.array(path), s_knots, duration


class PressArm:
    """The live arm, through the engine's self.arm (`arm`), for one action.

    With execute=False every motion is gated and logged but nothing is sent. `kin` is the
    capability's Kinematics (the sim the legs are gated in); `checkpoint` is the action's
    ctx.checkpoint, polled in this class's own waits so a cancel ends them.
    """

    def __init__(self, arm, kin, execute: bool, checkpoint):
        self.arm = arm
        self.kin = kin
        self.execute = execute
        self.checkpoint = checkpoint
        self.sent_goals = 0  # motions sent to the driver (the result's "moved")

    # -- state --------------------------------------------------------------------------------
    def state(self):
        """Latest arm state: {position (joints, wrapped), ee_pos (tool xyz + quat xyzw), gripper_pos}.

        Raises StateTimeout (an ArmError) if the driver's topics are stale (driver down)."""
        js, ee, grip = self.arm.joint_states(), self.arm.ee_state(), self.arm.gripper_state()
        tool = ee.position + Rotation.from_quat(ee.orientation).apply([0.0, 0.0, TOOL_Z])
        return {
            "position": wrap(js.position),
            "ee_pos": np.concatenate([tool, ee.orientation]),
            "gripper_pos": grip.position,
        }

    def joints(self):
        return np.asarray(self.state()["position"], dtype=float)

    def joint_torques(self):
        """Measured arm joint torques (N.m, joints 1-7) from /joint_states.effort; Abort if there are none."""
        tau = np.asarray(self.arm.joint_states().effort, dtype=float)
        if tau.shape != (7,) or not np.all(np.isfinite(tau)):
            raise Abort("/joint_states carries no arm efforts -- the contact press needs joint torques")
        return tau

    def raw_joints(self):
        """Measured joints as the driver reports them (not wrapped): the trajectory start."""
        return np.asarray(self.arm.joint_states().position, dtype=float)

    def ee_pos(self):
        return np.asarray(self.state()["ee_pos"][:3], dtype=float)

    def ee_pose(self):
        """(xyz, 3x3 rotation) of the Kinova tool frame in the arm base frame."""
        ee = self.state()["ee_pos"]
        return np.asarray(ee[:3], dtype=float), Rotation.from_quat(ee[3:7]).as_matrix()

    def gripper_present(self):
        return self.arm.gripper_state().present

    def close_gripper(self):
        """Close the gripper (the press uses the closed fingertips) and check it reads closed.

        Execute runs only; keep everything clear of the fingers. Returns the position reached;
        raises MotionFailed if the driver refuses it, Abort if it stops short of closed."""
        if not self.execute:
            raise Abort("close_gripper called in a dry run")
        self.sent_goals += 1
        if not self.arm.gripper(
            1.0, timeout_s=GRIPPER_CLOSE_TIMEOUT_S, force=GRIPPER_CLOSE_FORCE, speed=GRIPPER_CLOSE_SPEED
        ):
            raise MotionFailed(f"gripper close failed: {self.arm.last_error}")
        pos = float(self.arm.gripper_state().position)
        if pos <= GRIPPER_CLOSED_ABOVE:
            raise Abort(
                f"gripper did not close in {GRIPPER_CLOSE_TIMEOUT_S:.0f} s (position {pos:.2f}) -- "
                "something between the fingers?"
            )
        return pos

    def status(self):
        """Pre-run health check. Returns (problems, info lines); no problems = ready.

        Checks that the arm may be commanded (self.arm.not_ready(): e-stop, control), /diagnostics,
        and that the sim FK at the live joints matches /ee_state + TOOL_Z. If that last check fails,
        the stored pre-press offset would be in the wrong frame, so the run must refuse."""
        problems, info = [], []
        st = self.arm.control_status()
        info.append(
            f"control_status : estopped={st.estopped} "
            f"arbitration={'enforced' if st.arbitration_enabled else 'disabled'}"
            f" owned={st.owned}{' by ' + repr(st.owner_id) if st.owned else ''} rejected={st.rejected_count}"
        )
        why = self.arm.not_ready()
        if why is not None:
            problems.append(f"the arm must not be commanded now: {why}")
        try:
            diag = self.arm.diagnostics().status
        except ArmError as e:  # none, or stale: health information only, as before the port
            diag = []
            info.append(f"diagnostics    : {e}")
        for s in sorted(diag, key=lambda s: s.name):
            info.append(f"diagnostics    : {s.name}: {DIAG_LEVELS.get(s.level, s.level)} {s.message}")
            if s.level >= 2 and "Arm" in s.name:
                problems.append(f"diagnostics {s.name} {DIAG_LEVELS.get(s.level, s.level)}: {s.message}")
        # Live tool-frame check: our sim FK at the measured joints vs the driver's pose + TOOL_Z.
        s = self.state()
        p_fk, q_fk = self.kin.fk(s["position"])
        dp = float(np.linalg.norm(p_fk - s["ee_pos"][:3]))
        dr = float(np.degrees((Rotation.from_quat(q_fk) * Rotation.from_quat(s["ee_pos"][3:7]).inv()).magnitude()))
        info.append(f"tool frame     : sim FK vs /ee_state+{TOOL_Z} m: {dp*1000:.1f} mm, {dr:.2f} deg")
        if dp > TOOL_FRAME_CHECK_M or dr > 1.0:
            problems.append(
                f"tool frame mismatch: sim tool_frame FK is {dp*1000:.1f} mm / {dr:.1f} deg from "
                f"/ee_state + {TOOL_Z} m -- wrong mount_rpy_deg, or /ee_state is not the flange any more? "
                "The pre-press offset would be wrong; refusing"
            )
        return problems, info

    # -- planning (pure sim) ------------------------------------------------------------------
    def precheck_poses(self, poses, name, log, posture=None, start=None):
        """Gate a path of (xyz, R) tool poses in the sim (kinematics.precheck_poses), chained from
        `start` = (q, xyz, R) -- the arm's current joints and tool pose by default. Returns (worst IK
        err m, worst joint step deg, end state, joint solution of every point)."""
        if start is None:
            p, R = self.ee_pose()
            start = (self.joints(), p, R)
        return self.kin.precheck_poses(poses, name, log, start, posture=posture)

    # -- execution ----------------------------------------------------------------------------
    def smooth_poses(
        self,
        poses,
        knots,
        start,
        name,
        log,
        track_abort_m=TRACK_ABORT_M,
        cart_speed=CART_SPEED,
    ):
        """Execute a pre-checked straight leg and verify where it ended.

        Sends `knots` (the joint solutions precheck_poses found, planned from `start` = (joints,
        tool xyz, R)) as one timed joint trajectory, then waits for the arm to stop. Refuses (Abort,
        nothing sent) if the arm has moved more than START_DRIFT_DEG since planning. Raises
        MotionFailed (arm holds) if the driver reports an error or the tool ends more than
        `track_abort_m` from the target. Returns the final tool xyz (dry run: logs only and
        returns the current xyz)."""
        target = np.asarray(poses[-1][0], dtype=float)
        rec = {"step": name, "smooth": True, "target": target.tolist(), "n_points": len(poses), "executed": False}
        if not self.execute:
            log(rec)
            return self.ee_pos()
        q_now = self.raw_joints()
        drift = max_joint_delta_deg(q_now, start[0])
        if drift > START_DRIFT_DEG:
            raise Abort(f"{name}: the arm moved {drift:.1f} deg since this leg was planned -- nothing sent")
        path, s_knots, duration = leg_timing(q_now, knots, poses, start[1], start[2], cart_speed)
        t, q, qd = min_jerk_samples(path, s_knots, duration)
        self.sent_goals += 1
        res = self.arm.execute_joint_trajectory(
            rammp_arm.JOINTS, list(zip(t, q, qd, strict=True)), path_tolerance=PATH_TOL_RAD, preemption=QUEUE
        )
        fin = self._wait_still()
        err = float(np.linalg.norm(fin - target))
        rec.update(
            executed=True,
            returned=res.error_code,
            error=res.error_string,
            duration_s=duration,
            ee_after=fin.tolist(),
            final_err_m=err,
        )
        log(rec)
        say(f"  {name:22s} reached {np.round(fin, 3)}  ({err*1000:.1f} mm from target, {duration:.1f} s)")
        if res.error_code != 0:
            code = rammp_arm.ERRORS.get(res.error_code, res.error_code)
            raise MotionFailed(f"{name}: trajectory ended {code} {res.error_string} -- HOLDING HERE")
        if err > track_abort_m:
            raise MotionFailed(f"{name}: ended {err*100:.1f} cm from the target -- HOLDING HERE")
        return fin

    def _wait_still(self, timeout_s=5.0, settle_reads=4, tol_m=5e-4):
        """The goal's result arrives when the reference ends; wait for the tool to stop."""
        deadline, prev, steady = time.monotonic() + timeout_s, None, 0
        cur = self.ee_pos()
        while time.monotonic() < deadline and steady < settle_reads:
            self.checkpoint()
            time.sleep(0.05)
            cur = self.ee_pos()
            steady = steady + 1 if prev is not None and np.linalg.norm(cur - prev) < tol_m else 0
            prev = cur
        return cur

    # -- long moves through the driver's go_to_ee_pose -----------------------------------------
    def check_goal_pose(self, p_goal, name):
        """Gate a goal whose path the driver plans: reach and height band only. Raises Abort."""
        tb = np.asarray(p_goal, dtype=float) - self.kin.base_pos
        bad = []
        if np.linalg.norm(tb) > MAX_REACH_M:
            bad.append(f"reach {np.linalg.norm(tb):.3f} > {MAX_REACH_M}")
        if not Z_RANGE_M[0] <= tb[2] <= Z_RANGE_M[1]:
            bad.append(f"z {tb[2]:.3f} outside {Z_RANGE_M}")
        if bad:
            raise Abort(f"{name}: goal fails the gates: {'; '.join(bad)} -- arm not commanded")

    def go_to_ee_pose(self, p_goal, R_goal, name, log, track_abort_m=TRACK_ABORT_M):
        """Long move: send the tool to (p_goal, R_goal) via self.arm.go_to_ee_pose.

        cuRobo plans from the measured joints and the driver executes at arm.go_to_speed_scale; the
        path is not straight and is never previewed. Only the goal is gated (dry run: that is all
        it does). Raises MotionFailed (arm holds) on a driver failure (self.arm's own end check is
        arm.go_to_track_m) or if the tool ends more than `track_abort_m` from the goal. Returns the
        tool xyz reached (dry run: the current one)."""
        self.check_goal_pose(p_goal, name)
        target = np.asarray(p_goal, dtype=float)
        travel = float(np.linalg.norm(target - self.ee_pos()))
        say(
            f"  {name:16s} go_to_ee_pose  goal ok ({travel*100:.0f} cm away; path planned by cuRobo "
            "in the driver, not previewed)"
        )
        rec = {"step": name, "planner": "go_to_ee_pose", "target": target.tolist(), "executed": False}
        if not self.execute:
            log(rec)
            return self.ee_pos()
        phases = []

        def on_feedback(fb):  # on the executor: must not block
            if not phases or phases[-1] != fb.phase:
                phases.append(fb.phase)

        self.sent_goals += 1
        ok = self.arm.go_to_ee_pose(target, Rotation.from_matrix(R_goal).as_quat(), label=name, feedback=on_feedback)
        fin = self._wait_still()
        e = float(np.linalg.norm(fin - target))
        rec.update(executed=True, ok=ok, error=self.arm.last_error if not ok else "", phases=phases)
        rec.update(ee_after=fin.tolist(), final_err_m=e)
        log(rec)
        say(f"  {name:22s} reached {np.round(fin, 3)}  ({e*1000:.1f} mm from target)")
        if not ok:
            raise MotionFailed(f"{name}: go_to_ee_pose failed: {self.arm.last_error.strip()} -- HOLDING HERE")
        if e > track_abort_m:
            raise MotionFailed(f"{name}: ended {e*100:.1f} cm from the target -- HOLDING HERE")
        return fin

    def press_contact(self, p_spot, n_out, name, log, push=CONTACT_PUSH_M, stop_nm=CONTACT_STOP_NM):
        """Contact press (ContactPress): on ee_twist, hold still CONTACT_BASELINE_S (the wrist torque
        baseline), then push along -n_out at CONTACT_SPEED_MPS until the touch, and `push` past it
        (or until a wrist torque is `stop_nm` off its baseline); then leave the stream at once (it
        closes, and the driver holds at the measured q). Does not come back out: the caller moves
        back to the spot. Every tick goes into the log (`trace`).

        Returns {stop, touch_m, depth_m, peak_nm, t_s}; raises Abort (session closed, holding)
        unless it pressed. A cancel, or the driver ending the session, raises out of the stream
        (closed, holding). Dry run: logs only."""
        rec = {
            "step": name,
            "contact": True,
            "speed": CONTACT_SPEED_MPS,
            "touch_nm": CONTACT_TOUCH_NM,
            "push_m": push,
            "stop_nm": stop_nm,
            "executed": False,
        }
        if not self.execute:
            log(rec)
            return None
        n_out = np.asarray(n_out, dtype=float) / np.linalg.norm(n_out)
        p0 = self.ee_pos()
        if np.linalg.norm(p0 - np.asarray(p_spot)) > STREAM_START_CHECK_M:
            raise Abort(
                f"{name}: arm is {np.linalg.norm(p0 - p_spot)*1000:.1f} mm from the pre-press spot "
                "-- no session opened"
            )
        self.joint_torques()  # Abort before anything moves if there are none
        zero = np.zeros(6)
        v_cmd = np.concatenate([-CONTACT_SPEED_MPS * n_out, np.zeros(3)])
        press = ContactPress(push=push, stop_nm=stop_nm)
        peak = t = 0.0
        stop = None
        trace = []
        self.sent_goals += 1
        with self.arm.stream(TWIST_CONTROLLER) as s:
            taus, t_b = [], time.monotonic()
            while time.monotonic() - t_b < CONTACT_BASELINE_S:  # hold still: the baseline
                s.send(zero)
                taus.append(self.joint_torques()[CONTACT_WRIST])
                time.sleep(1.0 / STREAM_RATE_HZ)
            tau0 = np.mean(taus, axis=0)
            t_start = time.monotonic()
            while stop is None:
                t = time.monotonic() - t_start
                p = self.ee_pos()
                dev = np.abs(self.joint_torques()[CONTACT_WRIST] - tau0)
                rel = p - p0
                along = -float(rel @ n_out)  # + = into the panel
                lat = float(np.linalg.norm(rel + along * n_out))
                peak = max(peak, float(dev.max()))
                trace.append([round(t, 3), round(along * 1000, 2), *np.round(dev, 2).tolist()])
                stop = press.update(t, CONTACT_SPEED_MPS * t, along, lat, float(dev.max()))
                if stop is None:
                    s.send(v_cmd)
                    time.sleep(1.0 / STREAM_RATE_HZ)
        # the stream is closed: the driver holds at the measured q, so the push stops here
        fin = self.ee_pos()
        depth = -float((fin - p0) @ n_out)
        rec.update(
            executed=True,
            stop=stop[0],
            why=stop[1],
            touch_m=press.touch_m,
            depth_m=depth,
            peak_nm=peak,
            t_s=t,
            ee_after=fin.tolist(),
            trace_cols=["t_s", "in_mm", "dev_j4", "dev_j5", "dev_j6", "dev_j7"],
            trace=trace,
        )
        log(rec)
        touch = "no touch" if press.touch_m is None else f"touch at {press.touch_m*1000:.1f} mm"
        say(
            f"  {name:22s} contact: {stop[0]} -- {stop[1]}; {touch}, peak {peak:.1f} N.m; "
            f"holding {depth*1000:.1f} mm in after {t:.2f} s"
        )
        if stop[0] not in ("pressed", "stop"):
            raise Abort(f"{name}: {stop[1]} -- session closed, HOLDING HERE")
        return {"stop": stop[0], "touch_m": press.touch_m, "depth_m": depth, "peak_nm": peak, "t_s": t}
