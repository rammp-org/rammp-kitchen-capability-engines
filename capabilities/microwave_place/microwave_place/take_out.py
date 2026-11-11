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
     force), checked: the fingers stopped, closed on something;
  4. straight up `take_lift`;
  5. the level line back out to above the pre-grasp pose (the place's plan_level_back_out).
Steps 3-5 in position control. Only the cuRobo approach is collision-checked against the configured world;
the straight insertion onward retains the local IK and is not collision-checked.
"""

import re
import time

import numpy as np
import rammp_arm
from scipy.spatial.transform import Rotation

from . import place_motion as pm

GRIP_CLOSE, GRIP_FORCE, GRIP_SPEED = 1.0, 0.5, 1.0  # grab_container's close (grasp.py)
GRIP_WAIT_S = 4.0  # the close streams until the fingers stop, at most this long
STILL_CHECK_S, STILL_TOL = 0.3, 0.005  # before lifting: the fingers unmoved (within this) over this long ...
CLOSED_MIN = 0.6  # ... and closed at least this far (on the brim, not stuck open)
EMPTY_ABOVE = 0.998  # the gripper closed past this = nothing between the fingers (the brim stops it ~0.98-0.996)
SET_TOL = 0.05  # the gripper may end at most this much more closed than asked ...
OPEN_EXTRA = 0.15  # ... and at most this much more open
LIFT_SPEED = 0.05  # m/s peak, the straight lift off the glass
IMP_END_TOL = 0.03  # m: in impedance the hand lags its setpoint; this far off the grasp at most, or it does not close


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
    """Keep the original straight insertion, lift and retreat, seeded at pre-grasp.

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
    out = pm.plan_level_back_out(model, lift[-1], cand["pre"])
    return dict(line=line, lift=lift, out=out, end=end)


def plan_take_out(arm, model, q0, cand, cfg):
    """cuRobo to pre-grasp, then the existing local Cartesian paths; nothing moves."""
    speed = cfg["take_approach_speed_scale"]
    if not np.isfinite(speed) or not 0.0 < speed <= 1.0:
        raise pm.Refused("take_approach_speed_scale must be finite and in (0, 1]")
    approach = arm.plan_to_pose(cand["pre"], cand["quat"], q0)
    if not approach.success:
        raise pm.Refused(f"take-out approach: cuRobo refused: {approach.message}")
    if len(approach.q) < 2:
        raise pm.Refused("take-out approach: cuRobo returned no executable trajectory")
    legs = plan_from_pregrasp(model, approach.q[-1], cand, cfg)
    return dict(approach=approach.q, approach_t=approach.t, approach_qd=approach.qd, **legs)


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
    on nothing. Returns its position; Refused otherwise."""
    arm.gripper(GRIP_CLOSE, timeout_s=GRIP_WAIT_S, force=GRIP_FORCE, speed=GRIP_SPEED)  # blocks: fingers stopped
    pos = float(arm.gripper_state().position)
    time.sleep(STILL_CHECK_S)
    pos2 = float(arm.gripper_state().position)
    log(f"gripper closed to {pos2:.3f} (1.0 = closed on nothing)")
    if abs(pos2 - pos) > STILL_TOL:
        raise pm.Refused(f"STOPPED in the oven: the fingers are still moving ({pos:.3f} -> {pos2:.3f}) -- not lifting")
    if pos2 < CLOSED_MIN:
        raise pm.Refused(f"STOPPED in the oven: the gripper only closed to {pos2:.3f} -- not lifting")
    if pos2 > EMPTY_ABOVE:
        raise pm.Refused(f"STOPPED in the oven: the gripper closed to {pos2:.3f}, on nothing -- not lifting")
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
    move_timed(arm, model, lift, "the raise", LIFT_SPEED, "holding the container in the oven")
    ctx.checkpoint()
    ctx.progress("retract", 0.0)
    log("-> retract: the level line out to above the pre-grasp pose (position control)")
    q = np.asarray(arm.raw_joints(), dtype=float)
    out = pm.plan_level_back_out(model, q, cand["pre"])
    move_timed(arm, model, out, "the retract", c["insert_speed"], "holding the container")
    T = model.fk(np.asarray(arm.raw_joints(), dtype=float))
    log(f"Done: holding the container outside the oven, tool at {np.round(T[:3, 3], 3).tolist()}")
    return {"executed": True, "gripper": pos}
