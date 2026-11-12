"""Taking the container out of the microwave, from the take-out start pose (the wrist camera looking at it, the
gripper open).

  1. detect (rim_detection): the averaged nearest filtered points in the container mask, in arm_mount;
  2. reach: cuRobo plans to the pre-grasp with the hand turned 90 deg about its own axis (the fingers then
     above and below the brim), `reach_in` back from the grasp along the hand's level heading;
     then the straight level line in along that heading to the grasp: the tool frame `grasp_depth` along the heading
     from the detected container point and `grasp_up` above it. Position control, timed by pose_lengths:
     the wrist turns no faster than the hand travels), from pre-grasp up to `impedance_in` short of the grasp; there a
     short stop, and the rest of the line in in the driver's joint impedance (take_gains, at impedance_speed);
  3. impedance off (the position setpoint on the current joints), then the gripper closed on the brim (grab_container's
     force), checked: the fingers stopped and closed sufficiently;
  4. straight up `take_lift`;
  5. local fixed-heading level withdrawal to pre-grasp x/y at lifted height.
Steps 3-5 use position control. HOME and approach use cuRobo collision checking;
local insertion, lift and withdrawal retain IK and are not collision-checked.
"""

import re
import time

import numpy as np
import rammp_arm
from capability_engine.arm import HOLD_FIXED
from scipy.spatial.transform import Rotation

from . import place_motion as pm

GRIP_CLOSE, GRIP_FORCE, GRIP_SPEED = 1.0, 0.5, 1.0  # grab_container's close (grasp.py)
GRIP_WAIT_S = 4.0  # the close streams until the fingers stop, at most this long
STILL_CHECK_S, STILL_TOL = 0.3, 0.005  # before lifting: the fingers unmoved (within this) over this long ...
CLOSED_MIN = 0.6  # ... and closed at least this far (on the brim, not stuck open)
SET_TOL = 0.05  # the gripper may end at most this much more closed than asked ...
OPEN_EXTRA = 0.15  # ... and at most this much more open
LIFT_SPEED = 0.05  # m/s peak, the straight lift off the glass
IMP_END_TOL = 0.03  # m: in impedance the hand lags its setpoint; this far off the grasp at most, or it does not close


def prepare_home(arm, home_joints, execute, speed_scale, ctx, log):
    """Reach the configured viewing configuration before acquiring wrist images.

    A dry run cannot observe the future home camera view, so refuses off-home.
    The gripper is unchanged here; normal take-out opens it after planning.
    """
    home = np.asarray(home_joints, dtype=float)
    if home.shape != (7,) or not np.isfinite(home).all():
        raise pm.Refused("take_home_joints must contain seven finite joint angles in radians")
    ctx.checkpoint()
    q = np.asarray(arm.raw_joints(), dtype=float)
    off = float(np.max(np.abs(pm.wrap_deg(q - home))))
    if off <= pm.START_TOL_DEG:
        log(f"take-out home: already within {off:.2f} deg of configured viewing pose")
        return q
    if not execute:
        raise pm.Refused(f"dry run: arm is {off:.1f} deg from take_home_joints; move home before detecting")
    ctx.progress("home", 0.0)
    log("-> cuRobo to configured take-out home, gripper unchanged")
    if not arm.go_to_joint_config(home, "take-out home", speed_scale=speed_scale):
        raise pm.Refused(f"take-out home failed: {arm.last_error}")
    ctx.checkpoint()
    q = np.asarray(arm.raw_joints(), dtype=float)
    off = float(np.max(np.abs(pm.wrap_deg(q - home))))
    if off > pm.START_TOL_DEG:
        raise pm.Refused(f"take-out home ended {off:.1f} deg off target -- not detecting")
    return q


def grasp_pose(model, q0, point, reach_in, grasp_depth, grasp_up):
    """The grasp from the start q0 and the rim point: dict(heading (level, the hand's at q0: the line in runs along
    it), quat (q0's hand turned 90 deg about its own axis, the way that points the fingers' line, tool x, up, and
    levelled), target (`grasp_depth` along the heading and `grasp_up` above the rim point), pre (reach_in back along
    it))."""
    T = model.fk(np.asarray(q0, dtype=float))
    h = np.array([T[0, 2], T[1, 2], 0.0])
    h /= np.linalg.norm(h)
    turns = [T[:3, :3] @ Rotation.from_euler("z", s * 90, degrees=True).as_matrix() for s in (1, -1)]
    # the line between the fingers is the tool's x axis: it points straight up where the place held the brim, so
    # the turn that points it up
    R = max(turns, key=lambda Rg: Rg[2, 0])
    # then levelled: the start's own small pitch taken out about the hand's level left axis
    pitch = float(np.arcsin(np.clip(-R[2, 2], -1.0, 1.0)))
    R = Rotation.from_rotvec(-pitch * np.cross([0.0, 0.0, 1.0], h)).as_matrix() @ R
    target = np.asarray(point, dtype=float) + grasp_depth * h + np.array([0.0, 0.0, grasp_up])
    return dict(heading=h, quat=Rotation.from_matrix(R).as_quat(), target=target, pre=target - reach_in * h)


def set_approach_world(arm, world, log):
    """Load the floor scene and require the planner's persistent flatbot layers.

    The path is resolved by the planner, not by this capability's container.
    Older planners without layer acknowledgements fail closed.
    """
    ok, message = arm.set_world(world)
    if not ok:
        raise pm.Refused(f"take-out: set_world({world}) failed: {message}")
    config_layers = re.search(r"config layers: ([^|)]*)", message)
    scene_layers = re.search(r"scene layers: ([^|)]*)", message)
    for name in ("platform", "seat", "person"):
        if config_layers is None or re.search(rf"\b{name} [1-9]\d*\b", config_layers[1]) is None:
            raise pm.Refused(f"take-out: planner did not confirm config layer {name}: {message}")
    if scene_layers is None or re.search(r"\bfloor [1-9]\d*\b", scene_layers[1]) is None:
        raise pm.Refused(f"take-out: planner did not confirm the floor scene layer: {message}")
    log(f"take-out approach world: {message}")
    return message


def plan_from_pregrasp(model, q_pre, cand, cfg):
    """Keep the original straight insertion and lift, seeded at pre-grasp.

    Legacy IK remains here to preserve Cartesian lines and the impedance phase.
    These paths are not cuRobo collision-checked.
    """
    T = model.fk(np.asarray(q_pre, dtype=float))
    distance = float(np.linalg.norm(T[:3, 3] - cand["pre"]))
    angle = float(np.degrees((Rotation.from_matrix(T[:3, :3]).inv() * Rotation.from_quat(cand["quat"])).magnitude()))
    if distance > pm.END_TOL or angle > pm.END_TOL_DEG:
        raise pm.Refused(f"pre-grasp is {distance * 100:.1f} cm / {angle:.1f} deg off target -- not inserting")
    line = pm.plan_line(model, q_pre, cand["target"] - T[:3, 3], "line in", step=pm.LINE_STEP)
    pm.check_line(model, line, cand["pre"], cand["target"], cand["heading"], "line in")
    end = pm.end_pose_error(model, line[-1], cand["target"], cand["heading"])
    if end[0] > pm.END_TOL or end[1] > pm.END_TOL_DEG:
        raise pm.Refused(f"the line in ends {end[0] * 100:.1f} cm / {end[1]:.1f} deg off the grasp")
    lift = pm.plan_line(model, line[-1], np.r_[0.0, 0.0, cfg["take_lift"]], "lift", step=pm.LINE_STEP)
    return dict(line=line, lift=lift, end=end)


def plan_exit(arm, model, q_start, pre, cfg):
    """cuRobo withdrawal to pre-grasp x/y at the lifted height, orientation fixed.

    A path leaving the existing 1 cm straight corridor is refused: the planner
    knows the chair/person/floor, not the microwave interior or held container.
    Never fall back to the old unchecked local-IK retreat.
    """
    speed = float(cfg["take_exit_speed_scale"])
    if not np.isfinite(speed) or not 0.0 < speed <= 1.0:
        raise pm.Refused("take_exit_speed_scale must be finite and in (0, 1]")
    T = model.fk(np.asarray(q_start, dtype=float))
    a = T[:3, 3].copy()
    target = np.r_[np.asarray(pre, dtype=float)[:2], a[2]]
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    plan = arm.plan_to_pose(target, quat, q_start, hold=HOLD_FIXED)
    if not plan.success:
        raise pm.Refused(f"take-out exit: cuRobo refused: {plan.message}")
    if len(plan.q) < 2:
        raise pm.Refused("take-out exit: cuRobo returned no executable trajectory")
    if np.max(np.abs(pm.wrap_deg(np.asarray(plan.q[0]) - q_start))) > pm.START_TOL_DEG:
        raise pm.Refused("take-out exit: trajectory does not start at the requested joints")
    # Include the measured start and densely check the path between returned samples.
    qs = [np.asarray(q_start, dtype=float), *np.asarray(plan.q, dtype=float)]
    samples = pm.dense(qs, pm.LINE_STEP_DEG)
    pm.check_line(model, samples, a, target, None, "cuRobo exit")
    for q in samples:
        R = model.fk(q)[:3, :3]
        angle = float(np.degrees(Rotation.from_matrix(T[:3, :3].T @ R).magnitude()))
        if angle > pm.END_TOL_DEG:
            raise pm.Refused(f"cuRobo exit turns the held container {angle:.1f} deg -- refusing")
    if np.linalg.norm(model.fk(plan.q[-1])[:3, 3] - target) > pm.END_TOL:
        raise pm.Refused("cuRobo exit does not reach the retreat endpoint")
    # Preserve planner timing/velocities; slow down rather than reinterpolate its path.
    dt = np.diff(np.asarray(plan.t, dtype=float))
    if len(dt) != len(plan.q) - 1 or not np.isfinite(dt).all() or np.any(dt <= 0):
        raise pm.Refused("cuRobo exit returned invalid sample timing")
    peak = float(np.max(np.asarray(pm.pose_lengths(model, plan.q)) / dt))
    speed = min(speed, float(cfg["insert_speed"]) / max(peak, 1e-9))
    if not 0.01 <= speed <= 1.0:
        raise pm.Refused("cuRobo exit needs a speed scale below the driver minimum -- refusing")
    return dict(out=plan.q, out_t=plan.t, out_qd=plan.qd, out_speed_scale=speed, out_target=target)


def execute_exit(arm, model, pre, cfg, ctx, log):
    """Replan from actual lifted joints and send only the validated cuRobo samples."""
    ctx.checkpoint()
    q = np.asarray(arm.raw_joints(), dtype=float)
    exit_plan = plan_exit(arm, model, q, pre, cfg)
    return _execute_checked_exit(arm, model, q, exit_plan, ctx, log)


def _execute_checked_exit(arm, model, q, exit_plan, ctx, log):
    """Send the checked samples only if their measured start is still current."""
    ctx.checkpoint()
    now = np.asarray(arm.raw_joints(), dtype=float)
    if np.max(np.abs(pm.wrap_deg(now - q))) > pm.START_TOL_DEG:
        raise pm.Refused("arm moved while planning the exit -- nothing sent")
    log("-> cuRobo exit: fixed orientation, checked level withdrawal to above pre-grasp")
    result = arm.execute_joint_trajectory(
        rammp_arm.JOINTS,
        zip(exit_plan["out_t"], exit_plan["out"], exit_plan["out_qd"], strict=True),
        speed_scale=exit_plan["out_speed_scale"],
    )
    if result.error_code != pm.SUCCESSFUL:
        raise pm.Refused(f"STOPPED holding the container: cuRobo exit failed ({result.error_string})")
    time.sleep(pm.SETTLE_AFTER_S)
    q_end = np.asarray(arm.raw_joints(), dtype=float)
    T_end, T_start = model.fk(q_end), model.fk(q)
    distance = float(np.linalg.norm(T_end[:3, 3] - exit_plan["out_target"]))
    angle = float(np.degrees(Rotation.from_matrix(T_start[:3, :3].T @ T_end[:3, :3]).magnitude()))
    if distance > pm.END_TOL or angle > pm.END_TOL_DEG:
        raise pm.Refused(
            f"STOPPED holding the container: exit ended {distance * 100:.1f} cm / {angle:.1f} deg off target"
        )
    return exit_plan


def plan_withdrawal(arm, model, q_start, pre, cfg):
    """Local straight withdrawal; deliberately not checked against the cuRobo world."""
    start = model.fk(np.asarray(q_start, dtype=float))[:3, 3]
    target = np.r_[np.asarray(pre)[:2], start[2]]
    qs = pm.plan_line(model, q_start, target - start, "local withdrawal", step=pm.LINE_STEP)
    pm.check_line(model, qs, start, target, None, "local withdrawal")
    duration = sum(pm.pose_lengths(model, qs)) / cfg["insert_speed"]
    empty = dict(out_t=[0.0], out_speed_scale=1.0, out_target=start)
    return dict(out=qs, out_t=[0.0, duration], out_speed_scale=1.0, out_target=target, pullback=empty)


def execute_withdrawal(arm, model, pre, cfg, ctx, log):
    """Preserve fixed-heading local retreat to above pre-grasp, then stop."""
    ctx.checkpoint()
    q = np.asarray(arm.raw_joints(), dtype=float)
    route = plan_withdrawal(arm, model, q, pre, cfg)
    ctx.progress("retract", 0.0)
    log("-> local straight retreat to pre-grasp x/y at lifted height (no cuRobo collision check)")
    move_timed(arm, model, route["out"], "local withdrawal", cfg["insert_speed"], "holding the container")
    actual = model.fk(np.asarray(arm.raw_joints(), dtype=float))[:3, 3]
    if np.linalg.norm(actual - route["out_target"]) > pm.END_TOL:
        raise pm.Refused("STOPPED holding container: local retreat endpoint tracking error")
    return route


def plan_take_out(arm, model, q0, cand, cfg):
    """Plan cuRobo approach and local insertion/lift/withdrawal before motion."""
    exit_speed = cfg["take_exit_speed_scale"]
    if not np.isfinite(exit_speed) or not 0.0 < exit_speed <= 1.0:
        raise pm.Refused("take_exit_speed_scale must be finite and in (0, 1]")
    speed = cfg["take_approach_speed_scale"]
    if not np.isfinite(speed) or not 0.0 < speed <= 1.0:
        raise pm.Refused("take_approach_speed_scale must be finite and in (0, 1]")
    approach = arm.plan_to_pose(cand["pre"], cand["quat"], q0)
    if not approach.success:
        raise pm.Refused(f"take-out approach: cuRobo refused: {approach.message}")
    if len(approach.q) < 2:
        raise pm.Refused("take-out approach: cuRobo returned no executable trajectory")
    legs = plan_from_pregrasp(model, approach.q[-1], cand, cfg)
    exit_plan = plan_withdrawal(arm, model, legs["lift"][-1], cand["pre"], cfg)
    return dict(approach=approach.q, approach_t=approach.t, approach_qd=approach.qd, **legs, **exit_plan)


def set_gripper(arm, position, log, what):
    """Open the gripper to `position` (0 open .. 1 closed) and wait for it: Refused unless it ends at least as open
    as asked and at most OPEN_EXTRA more open (this 2F-85 reports ~0.21 for a 0.30 command), fingers still."""
    arm.gripper(position)  # streams until the fingers stop
    t0, pos = time.monotonic(), float(arm.gripper_state().position)
    while time.monotonic() - t0 < pm.OPEN_WAIT_S:
        time.sleep(STILL_CHECK_S)
        nxt = float(arm.gripper_state().position)
        if abs(nxt - pos) <= STILL_TOL:
            break
        pos = nxt
    log(f"gripper {what}: {pos:.2f} (asked {position:.2f})")
    if not position - OPEN_EXTRA <= pos <= position + SET_TOL:
        raise pm.Refused(
            f"STOPPED before moving: the gripper is at {pos:.2f} for {position:.2f} asked "
            f"(allowed {position - OPEN_EXTRA:.2f}..{position + SET_TOL:.2f})"
        )


def move_timed(arm, model, qs, what, speed, state, impedance=None):
    """move_joints through qs[1:], timed like the approach (pose_lengths: no joint, and no turn of the hand, faster
    than `speed` / ROT_LENGTH), not by the tool's travel alone: near the wrist singularity a straight line swings J5
    and J7 a long way for little tool travel. `impedance`: move_joints' impedance dict (None: position control).
    Refused if it fails."""
    ok = arm.move_joints(
        qs[1:],
        cart_lengths=pm.pose_lengths(model, qs),
        speed_scale=speed / rammp_arm.CART_SPEED,
        **({} if impedance is None else {"impedance": impedance}),
    )
    if not ok:
        raise pm.Refused(f"STOPPED, {state}: {what} did not complete ({arm.last_error})")
    time.sleep(pm.SETTLE_AFTER_S)


def close_on_brim(arm, log):
    """Close the gripper (grab_container's force) and check it: the fingers stopped, closed past CLOSED_MIN and not
    sufficiently. Fully closed readings do not reject the grasp. Returns its position."""
    arm.gripper(GRIP_CLOSE, timeout_s=GRIP_WAIT_S, force=GRIP_FORCE, speed=GRIP_SPEED)  # blocks: fingers stopped
    pos = float(arm.gripper_state().position)
    time.sleep(STILL_CHECK_S)
    pos2 = float(arm.gripper_state().position)
    log(f"gripper closed to {pos2:.3f} (1.0 = fully closed; does not establish grasp success)")
    if abs(pos2 - pos) > STILL_TOL:
        raise pm.Refused(f"STOPPED in the oven: the fingers are still moving ({pos:.3f} -> {pos2:.3f}) -- not lifting")
    if pos2 < CLOSED_MIN:
        raise pm.Refused(f"STOPPED in the oven: the gripper only closed to {pos2:.3f} -- not lifting")
    return pos2


def run_take_out(arm, model, q0, cand, legs, cfg, ctx, log):
    """The planned take-out on the arm: position control, except the last `impedance_in` of the line in, in the
    driver's joint impedance (cfg take_gains; a short stop at the switch). The arm leaves impedance (exit_impedance:
    the position setpoint on the current joints) before the gripper closes, whatever happened; the close, the raise
    and the retract are in position control."""
    c = cfg
    pm.ready(arm)
    q_now = np.asarray(arm.raw_joints(), dtype=float)
    off = float(np.max(np.abs(pm.wrap_deg(q_now - q0))))
    if off > pm.START_TOL_DEG:
        raise pm.Refused(f"the arm is {off:.1f} deg from where the plan starts (max {pm.START_TOL_DEG:g}): re-plan")

    ctx.progress("gripper", 0.0)
    set_gripper(arm, c["grip_open"], log, "opened for the brim")

    # Execute the exact collision-checked samples, preserving cuRobo's timing and velocities.
    ctx.checkpoint()
    ctx.progress("approach", 0.0)
    log("-> cuRobo approach to pre-grasp (position control, stops at pre-grasp)")
    result = arm.execute_joint_trajectory(
        rammp_arm.JOINTS,
        zip(legs["approach_t"], legs["approach"], legs["approach_qd"], strict=True),
        speed_scale=c["take_approach_speed_scale"],
    )
    if result.error_code != pm.SUCCESSFUL:
        raise pm.Refused(f"STOPPED before insertion: cuRobo approach failed ({result.error_string})")

    # Re-seed from the measured endpoint rather than bridging into a path from assumed joints.
    ctx.checkpoint()
    q_pre = np.asarray(arm.raw_joints(), dtype=float)
    insertion = plan_from_pregrasp(model, q_pre, cand, c)
    # Refuse before insertion if the new local lift endpoint has no valid exit.
    ctx.checkpoint()
    plan_withdrawal(arm, model, insertion["lift"][-1], cand["pre"], c)
    line = [np.asarray(q, dtype=float) for q in insertion["line"]]
    k = max(1, len(line) - 1 - int(np.ceil(c["impedance_in"] / pm.LINE_STEP)))
    ctx.progress("reach", 0.0)
    log(f"-> straight line in to {c['impedance_in'] * 100:.0f} cm short of the grasp (position control)")
    move_timed(arm, model, line[: k + 1], "the line in", c["insert_speed"], "at the opening, gripper open")

    # 2b. joint impedance for the last `impedance_in` of the line in only (cfg take_gains); it ends before the close
    ctx.checkpoint()
    gains = c["take_gains"]
    try:
        log(f"impedance on (kq {gains['kq'][0]:g}/{gains['kq'][4]:g}): settle {c['settle_s']:.1f} s")
        res = pm.hold_goal(arm, line[k], gains, hold_s=c["settle_s"])
        if res.error_code != pm.SUCCESSFUL:
            raise pm.Refused(f"STOPPED at the opening: the impedance hold ended {res.error_string}")
        pm.wait_still(arm, pm.SETTLE_WAIT_S, pm.SETTLE_STILL_S)
        ctx.progress("reach", 50.0)
        log(
            f"-> line in the last {c['impedance_in'] * 100:.0f} cm "
            f"(impedance, {c['impedance_speed'] * 100:.0f} cm/s peak)"
        )
        move_timed(
            arm, model, line[k:], "the line in", c["impedance_speed"], "in the oven, gripper open", pm.compliant(gains)
        )
        pm.wait_still(arm, pm.STILL_AFTER_S)
    finally:
        pm.exit_impedance(arm, log)  # position control before anything else, whatever happened
    q_in = np.asarray(arm.raw_joints(), dtype=float)
    d, turn = pm.end_pose_error(model, q_in, cand["target"], cand["heading"])
    log(f"at the grasp: {d * 100:.1f} cm, {turn:.1f} deg from it")
    if d > IMP_END_TOL:
        raise pm.Refused(f"STOPPED in the oven, gripper open: {d * 100:.1f} cm off the grasp -- not closing")

    # 3. close on the brim, 4. raise, 5. the level retract: position control
    ctx.checkpoint()
    ctx.progress("close", 0.0)
    pos = close_on_brim(arm, log)
    ctx.checkpoint()
    ctx.progress("lift", 0.0)
    log(f"-> raise {c['take_lift'] * 100:.0f} cm (position control)")
    q = np.asarray(arm.raw_joints(), dtype=float)
    lift = pm.plan_line(model, q, np.r_[0.0, 0.0, c["take_lift"]], "lift", step=pm.LINE_STEP)
    # Account for impedance tracking error before committing to the lift.
    ctx.checkpoint()
    plan_withdrawal(arm, model, lift[-1], cand["pre"], c)
    move_timed(arm, model, lift, "the raise", LIFT_SPEED, "holding the container in the oven")
    ctx.checkpoint()
    ctx.progress("retract", 0.0)
    exit_plan = execute_withdrawal(arm, model, cand["pre"], c, ctx, log)
    T = model.fk(np.asarray(arm.raw_joints(), dtype=float))
    log(f"Done: holding the container outside the oven, tool at {np.round(T[:3, 3], 3).tolist()}")
    return {
        "executed": True,
        "gripper": pos,
        "exit_collision_checked": False,
        "exit_speed_scale": exit_plan["out_speed_scale"],
        "exit_target": exit_plan["out_target"].tolist(),
        "pullback_target": exit_plan["pullback"]["out_target"].tolist(),
    }
