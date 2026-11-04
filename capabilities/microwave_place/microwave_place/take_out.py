"""Taking the container out of the microwave, from the take-out start pose (the wrist camera looking at it, the
gripper open).

  1. detect (rim_detection): the nearest point on the container's outer rim, in arm_mount;
  2. reach: from the start, the hand turned 90 deg about its own axis (the fingers then above and below the brim)
     while it moves straight to the pre-grasp pose, `reach_in` back from the grasp along the hand's level heading;
     then the straight level line in along that heading to the grasp: the tool frame `grasp_depth` along the heading
     from the rim point and `grasp_up` above it. Position control, timed like the place's approach (pose_lengths:
     the wrist turns no faster than the hand travels), as one motion up to `impedance_in` short of the grasp; there a
     short stop, and the rest of the line in in the driver's joint impedance (take_gains, at impedance_speed);
  3. impedance off (the position setpoint on the current joints), then the gripper closed on the brim (grab_container's
     force), checked: the fingers stopped, closed on something;
  4. straight up `take_lift`;
  5. the level line back out to above the pre-grasp pose (the place's plan_level_back_out).
Steps 3-5 in position control. Nothing is collision-checked (the start is in front of the opening, the line in runs
straight along the hand's heading into it).
"""

import time

import numpy as np
import rammp_arm
from scipy.spatial.transform import Rotation, Slerp

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
TURN_STEP_DEG = 2.0  # the turning move's waypoints: at most this much turn ...
JUMP_DEG = 15.0  # ... and no joint more than this between two of them (more = the IK flipped the arm)


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


def plan_turning_line(model, q_start, p_end, quat_end, step=pm.LINE_STEP):
    """Joint waypoints along the straight line from q_start's tool position to p_end while the hand turns
    (slerp) to quat_end; the first q_start. Refused where the IK can't follow."""
    T = model.fk(np.asarray(q_start, dtype=float))
    a, b = T[:3, 3], np.asarray(p_end, dtype=float)
    rots = Rotation.from_matrix([T[:3, :3], Rotation.from_quat(quat_end).as_matrix()])
    angle = np.degrees((rots[0].inv() * rots[1]).magnitude())
    n = max(1, int(np.ceil(max(np.linalg.norm(b - a) / step, angle / TURN_STEP_DEG))))
    slerp = Slerp([0.0, 1.0], rots)
    qs, seed = [np.asarray(q_start, dtype=float)], np.asarray(q_start, dtype=float)
    for k in range(1, n + 1):
        u = k / n
        q = model.ik(a + u * (b - a), slerp([u]).as_quat()[0], seed, restarts=0, stay=0.01)
        if q is None or np.max(np.abs(pm.wrap_deg(q - seed))) > JUMP_DEG:
            raise pm.Refused(f"reach: the arm can't turn and move there ({u * 100:.0f}% of the way)")
        qs.append(q)
        seed = q
    return qs


def plan_take_out(model, q0, cand, cfg):
    """Every leg on the arm model: the turning move to the pre-grasp, the line in (on its line, along the heading,
    ending on the grasp), the lift, the line out. Raises Refused."""
    c = cfg
    turn = plan_turning_line(model, q0, cand["pre"], cand["quat"])
    line = pm.plan_line(model, turn[-1], cand["target"] - cand["pre"], "line in", step=pm.LINE_STEP)
    pm.check_line(model, line, cand["pre"], cand["target"], cand["heading"], "line in")
    end = pm.end_pose_error(model, line[-1], cand["target"], cand["heading"])
    if end[0] > pm.END_TOL or end[1] > pm.END_TOL_DEG:
        raise pm.Refused(f"the line in ends {end[0] * 100:.1f} cm / {end[1]:.1f} deg off the grasp")
    lift = pm.plan_line(model, line[-1], np.r_[0.0, 0.0, c["take_lift"]], "lift", step=pm.LINE_STEP)
    out = pm.plan_level_back_out(model, lift[-1], cand["pre"])
    return dict(turn=turn, line=line, lift=lift, out=out, end=end)


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

    # 2a. turn and reach to the switch point, `impedance_in` short of the grasp along the line in: ONE motion, position
    # control (the arm library runs one trajectory goal at a time, so the switch to impedance is a short stop there)
    line = [np.asarray(q, dtype=float) for q in legs["line"]]
    k = max(1, len(line) - 1 - int(np.ceil(c["impedance_in"] / pm.LINE_STEP)))  # the switch point's waypoint
    knots = [np.asarray(q, dtype=float) for q in legs["turn"][1:]] + line[1 : k + 1]
    lengths = pm.pose_lengths(model, [q_now, *knots])
    ctx.progress("reach", 0.0)
    log(
        f"-> reach: turn the hand 90 deg on the way to the pre-grasp, line in to {c['impedance_in'] * 100:.0f} cm "
        f"short of the grasp; {sum(lengths) * 100:.0f} cm of timed path at {c['insert_speed'] * 100:.0f} cm/s peak "
        "(position control)"
    )
    if not arm.move_joints(knots, cart_lengths=lengths, speed_scale=c["insert_speed"] / rammp_arm.CART_SPEED):
        raise pm.Refused(f"STOPPED: the reach did not complete ({arm.last_error})")

    # 2b. joint impedance for the last `impedance_in` of the line in only (cfg take_gains); it ends before the close
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
    ctx.progress("close", 0.0)
    pos = close_on_brim(arm, log)
    ctx.progress("lift", 0.0)
    log(f"-> raise {c['take_lift'] * 100:.0f} cm (position control)")
    q = np.asarray(arm.raw_joints(), dtype=float)
    lift = pm.plan_line(model, q, np.r_[0.0, 0.0, c["take_lift"]], "lift", step=pm.LINE_STEP)
    move_timed(arm, model, lift, "the raise", LIFT_SPEED, "holding the container in the oven")
    ctx.progress("retract", 0.0)
    log("-> retract: the level line out to above the pre-grasp pose (position control)")
    q = np.asarray(arm.raw_joints(), dtype=float)
    out = pm.plan_level_back_out(model, q, cand["pre"])
    move_timed(arm, model, out, "the retract", c["insert_speed"], "holding the container")
    T = model.fk(np.asarray(arm.raw_joints(), dtype=float))
    log(f"Done: holding the container outside the oven, tool at {np.round(T[:3, 3], 3).tolist()}")
    return {"executed": True, "gripper": pos}
