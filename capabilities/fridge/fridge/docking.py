"""Find the fixture-A dock using stationary camera looks and differential-drive twists."""

import math
import time
from itertools import accumulate

import numpy as np
from capability_engine import BaseError
from door_common.common import Refused

from . import base_drive
from .door_sweep import DoorSweep

# H and inward normal relative to arm_mount at the WS11-E floor mark.
TARGET_HANDLE = np.array([0.825, -0.235])
TARGET_YAW = math.radians(-14)
POSITION_TOLERANCE = 0.025
YAW_TOLERANCE = math.radians(2)
MAX_YAW_TRAVEL = math.radians(180)
MAX_YAW_SPEED = 0.12
SPEED_PARAMS = {
    "docking_max_linear_speed": 0.05,
    "docking_max_yaw_speed": MAX_YAW_SPEED,
}
MAX_TIME = 75.0
MAX_LOOKS = 4
LINEAR_SETTLED = 0.005
ANGULAR_SETTLED = 0.01
TRANSLATION_TOL = 0.008
TURN_TOL = math.radians(0.7)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def correction(handle, normal, axle_to_mount, target_handle=TARGET_HANDLE, target_yaw=TARGET_YAW):
    """Return desired axle displacement and yaw in the current chair frame."""
    yaw = wrap(math.atan2(-normal[1], -normal[0]) - target_yaw)
    axle = np.asarray(axle_to_mount)
    return handle[:2] + axle - rotation(yaw) @ (target_handle + axle), yaw


def legs(displacement, yaw, *, other_direction=False):
    """Shortest initial turn, signed straight travel, and terminal turn."""
    advance = float(np.linalg.norm(displacement))
    if advance <= POSITION_TOLERANCE / 2:
        return ((0.0, yaw),)
    heading = math.atan2(displacement[1], displacement[0])
    reverse = wrap(heading + math.pi)
    prefer_reverse = abs(reverse) + abs(wrap(yaw - reverse)) < abs(heading) + abs(wrap(yaw - heading))
    if prefer_reverse != other_direction:
        heading, advance = reverse, -advance
    return ((0.0, heading), (advance, 0.0), (0.0, wrap(yaw - heading)))


def route_options(delta, yaw):
    yield legs(delta, yaw)
    if np.linalg.norm(delta) <= POSITION_TOLERANCE / 2:
        return
    yield legs(delta, yaw, other_direction=True)
    # A short retreat can swing the footrest and parked arm clear before turning.
    for retreat in (0.08, 0.15):
        remaining = delta + np.array([retreat, 0])
        for other_direction in (False, True):
            yield ((-retreat, 0.0), *legs(remaining, yaw, other_direction=other_direction))


def checked_route(delta, yaw, remaining_d, remaining_yaw, geom, guard):
    blocked = None
    for route in route_options(delta, yaw):
        travel = sum(abs(d) + TRANSLATION_TOL for d, _ in route if abs(d) > TRANSLATION_TOL)
        turning = sum(abs(a) + TURN_TOL for _, a in route if abs(a) > TURN_TOL)
        if travel > remaining_d or turning > remaining_yaw:
            continue
        try:
            guard(geom, route)
        except Refused as error:
            blocked = blocked or error
        else:
            return route, travel, turning
    if blocked is not None:
        raise blocked
    raise Refused("Docking correction exceeds the remaining distance or heading budget. Repark the chair.")


def relative_axle(start, end, odom_to_axle):
    """Convert odometric base_link poses to displacement of the axle about which MIB turns."""
    yaw = wrap(end[2] - start[2])
    d = rotation(-start[2]) @ (np.asarray(end[:2]) - start[:2])
    return d + (rotation(yaw) - np.eye(2)) @ odom_to_axle, yaw


def _sample(base):
    odom = base.odometry()
    twist = odom.twist.twist
    pose = base_drive._pose(odom)
    if not np.isfinite([*pose, twist.linear.x, twist.angular.z]).all():
        raise BaseError("Nonfinite docking odometry")
    return pose, abs(twist.linear.x), abs(twist.angular.z)


def _drive_leg(base, arm, ctx, distance, angle, speed, yaw_speed, odom_to_axle, deadline, clock, sleep):
    turning = angle != 0.0
    goal = angle if turning else distance
    tolerance = TURN_TOL if turning else TRANSLATION_TOL
    if abs(goal) <= tolerance:
        return
    start, v, w = _sample(base)
    if v > LINEAR_SETTLED or w > ANGULAR_SETTLED:
        raise BaseError("The wheelchair was moving at the start of a docking segment")
    sign = math.copysign(1, goal)
    top_speed = yaw_speed if turning else speed
    deadline = min(deadline, clock() + abs(goal) / top_speed + 5.0)

    def sample():
        if ctx.paused():
            raise BaseError("Docking paused during a move")
        if arm.control_status().estopped:
            raise BaseError("Arm e-stop during docking")
        if clock() >= deadline:
            raise BaseError("Docking exceeded its motion deadline")
        pose, v, w = _sample(base)
        d, yaw = relative_axle(start, pose, odom_to_axle)
        progress = sign * (yaw if turning else d[0])
        if (
            (turning and np.linalg.norm(d) > 0.02)
            or (not turning and (abs(d[1]) > 0.02 or abs(yaw) > math.radians(2)))
            or progress < -tolerance
            or progress > abs(goal) + tolerance
        ):
            raise BaseError("Wheelchair departed the checked docking segment")
        return progress, v, w

    sent_at = clock()
    try:
        while True:
            progress, _, _ = sample()
            now = clock()
            if now - sent_at > 0.25:
                raise BaseError("Docking twist refresh was delayed")
            sent_at = now
            remaining = abs(goal) - progress
            if remaining <= tolerance / 2:
                break
            command = sign * min(top_speed, max(top_speed / 4, 2 * remaining))
            vx, wz = (0.0, command) if turning else (command, 0.0)
            base.send_twist(vx, wz)
            sleep(base_drive.PERIOD_S)
    finally:
        base.stop()
    until = clock() + base_drive.SETTLE_S
    while True:
        ctx.checkpoint()
        progress, v, w = sample()
        if clock() >= until and v <= LINEAR_SETTLED and w <= ANGULAR_SETTLED:
            break
        base.stop()
        sleep(base_drive.PERIOD_S)
    if abs(progress - abs(goal)) > tolerance:
        raise BaseError("Wheelchair settled outside the checked segment endpoint")


def stationary_look(base, arm, look, odom_to_axle, *, mode_entered):
    before, v, w = _sample(base)
    if v > LINEAR_SETTLED or w > ANGULAR_SETTLED:
        raise BaseError("The wheelchair must be stopped for a docking look")
    if arm.control_status().estopped:
        raise BaseError("Arm e-stop before docking look")
    observation = look()
    after, v, w = _sample(base)
    delta, turn = relative_axle(before, after, odom_to_axle)
    if v > LINEAR_SETTLED or w > ANGULAR_SETTLED or np.linalg.norm(delta) > 0.01 or abs(turn) > TURN_TOL:
        raise BaseError("The wheelchair moved during a docking look")
    if arm.control_status().estopped:
        raise BaseError("Arm e-stop during docking look")
    if mode_entered:
        base.send_twist(0.0, 0.0)  # verify the MIB still accepts AUTONOMOUS even at the final dock
    return observation


def dock(
    base,
    arm,
    ctx,
    look,
    guard,
    *,
    speed,
    cfg,
    axle_to_mount,
    odom_to_axle,
    on_motion,
    target_handle=TARGET_HANDLE,
    target_yaw=TARGET_YAW,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Return the last camera observation and number of corrections executed."""
    deadline = clock() + MAX_TIME
    remaining_d = min(0.35, cfg["base_max_distance"])
    linear_limit = cfg.get("docking_max_linear_speed", SPEED_PARAMS["docking_max_linear_speed"])
    yaw_speed = cfg.get("docking_max_yaw_speed", SPEED_PARAMS["docking_max_yaw_speed"])
    if not 0 < linear_limit <= 0.15 or not 0 < yaw_speed <= 0.30:
        raise Refused("Docking speed limits must be in (0, 0.15] m/s and (0, 0.30] rad/s")
    if not math.isfinite(speed) or not speed > 0:
        raise Refused("Docking requires a positive drive speed")
    speed = min(linear_limit, speed, cfg["base_max_speed"])
    remaining_yaw = MAX_YAW_TRAVEL
    entered = False
    for index in range(MAX_LOOKS):
        ctx.checkpoint()
        geom, q = stationary_look(base, arm, look, odom_to_axle, mode_entered=entered)
        if clock() >= deadline:
            raise Refused("Docking timed out waiting for a stationary camera observation")
        handle, normal = geom["h_raw"], geom["n"]
        if not np.isfinite([*handle, *normal]).all() or np.linalg.norm(normal[:2]) < 0.9:
            raise Refused("Docking requires a measured fixture heading and finite handle position")
        delta, yaw = correction(handle, normal, axle_to_mount, target_handle, target_yaw)
        if np.linalg.norm(handle[:2] - target_handle) <= POSITION_TOLERANCE and abs(yaw) <= YAW_TOLERANCE:
            return (geom, q), index
        if index == MAX_LOOKS - 1:
            break
        route, distance, rotation_travel = checked_route(delta, yaw, remaining_d, remaining_yaw, geom, guard)
        if not entered:
            on_motion()
            base.request_mode("autonomous", profile=cfg["base_profile"])
            entered = True
        for d, a in route:
            # Only check pause while stopped. Motor commands contain their own cancel check.
            ctx.checkpoint()
            if clock() >= deadline:
                raise BaseError("Docking exceeded its motion deadline")
            _drive_leg(base, arm, ctx, d, a, speed, yaw_speed, odom_to_axle, deadline, clock, sleep)
        remaining_d -= distance
        remaining_yaw -= rotation_travel
    raise Refused("Docking did not converge within three corrections. Repark the chair.")


GUIDED_STEP_M = 0.10
GUIDED_TURN = math.radians(15)
GUIDED_LOOK_HEADING = math.radians(28)
GUIDED_POSITION_MARGIN = 0.025
GUIDED_YAW_MARGIN = math.radians(4)
GUIDED_SETTLE_S = 1.0
GUIDED_MAX_TIME = 150.0
GUIDED_MAX_STEPS = 24


def guidance_text(distance, turn):
    if distance:
        movement = f"Drive {'forward' if distance > 0 else 'back'} about {round(abs(distance) * 100)} cm"
    else:
        movement = f"Turn {'left' if turn > 0 else 'right'} about {round(math.degrees(abs(turn)))}°"
    return f"{movement}, then stop. Keep the route clear. Wait here while I check again."


def guided_routes(delta, yaw):
    yield from route_options(delta, yaw)
    # A shallow diagonal followed by a straight correction can make a
    # lateral move without turning the handle out of the cameras.
    along, across = rotation(-yaw) @ delta
    if abs(across) <= TRANSLATION_TOL:
        return
    first_direction = 1 if along > 0 else -1
    for direction in (first_direction, -first_direction):
        diagonal = math.copysign(GUIDED_LOOK_HEADING, across * direction)
        travel = across / math.sin(diagonal)
        return_straight = along - travel * math.cos(diagonal)
        yield (
            (0.0, wrap(yaw + diagonal)),
            (travel, 0.0),
            (0.0, -diagonal),
            (return_straight, 0.0),
        )


def guided_step(delta, yaw, geom, guard):
    """Ask for only one short segment, with an allowance for human overshoot."""

    def view_excursion(route):
        # Prefer a retreat for a near-sideways correction. Facing far away
        # from the dock can put the handle outside both cameras at a re-look.
        headings = accumulate((turn for _, turn in route), initial=-yaw)
        return max(0.0, max(abs(wrap(angle)) for angle in headings) - GUIDED_LOOK_HEADING)

    blocked = None
    for route in sorted(guided_routes(delta, yaw), key=view_excursion):
        for distance, turn in route:
            if abs(distance) > TRANSLATION_TOL or abs(turn) > TURN_TOL:
                break
        else:
            continue
        distance = math.copysign(min(abs(distance), GUIDED_STEP_M), distance) if distance else 0.0
        turn = math.copysign(min(abs(turn), GUIDED_TURN), turn) if turn else 0.0
        far_distance = distance + math.copysign(GUIDED_POSITION_MARGIN, distance) if distance else 0.0
        far_turn = turn + math.copysign(GUIDED_YAW_MARGIN, turn) if turn else 0.0
        try:
            guard(geom, ((far_distance, far_turn),))
        except Refused as error:
            blocked = blocked or error
        else:
            return distance, turn
    raise Refused("No checked small wheelchair move is available. Stop and repark the chair.") from blocked


def _await_guided_step(base, arm, ctx, start, step, odom_to_axle, deadline, clock, sleep):
    distance, turn = step
    stopped_since = None
    while clock() < deadline:
        ctx.checkpoint()
        if arm.control_status().estopped:
            raise BaseError("Arm e-stop during guided docking")
        pose, v, w = _sample(base)
        (x, y), yaw = relative_axle(start, pose, odom_to_axle)
        if distance:
            progress = math.copysign(1, distance) * x
            target, margin = abs(distance), GUIDED_POSITION_MARGIN
            departed = abs(y) > margin or abs(yaw) > GUIDED_YAW_MARGIN
        else:
            progress = math.copysign(1, turn) * yaw
            target, margin = abs(turn), GUIDED_YAW_MARGIN
            departed = math.hypot(x, y) > GUIDED_POSITION_MARGIN
        if departed or not -margin <= progress <= target + margin:
            raise Refused("Stop the wheelchair. It moved outside the checked step.")
        moved = progress >= target * 0.5
        if moved and v <= LINEAR_SETTLED and w <= ANGULAR_SETTLED:
            stopped_since = clock() if stopped_since is None else stopped_since
            if clock() - stopped_since >= GUIDED_SETTLE_S:
                return
        else:
            stopped_since = None
        sleep(base_drive.PERIOD_S)
    raise Refused("Stop the wheelchair. Guided docking timed out.")


def guide(
    base,
    arm,
    ctx,
    look,
    guard,
    *,
    axle_to_mount,
    odom_to_axle,
    show,
    coarse=False,
    target_handle=TARGET_HANDLE,
    target_yaw=TARGET_YAW,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Guide a manual driver using fresh looks between bounded steps."""
    deadline = clock() + GUIDED_MAX_TIME
    for count in range(GUIDED_MAX_STEPS + 1):
        ctx.checkpoint()
        geom, q = stationary_look(base, arm, look, odom_to_axle, mode_entered=False)
        handle, normal = geom["h_raw"], geom["n"]
        if not np.isfinite([*handle, *normal]).all() or np.linalg.norm(normal[:2]) < 0.9:
            raise Refused("Guidance requires a measured fixture heading and finite handle position")
        delta, yaw = correction(handle, normal, axle_to_mount, target_handle, target_yaw)
        if coarse:
            ready = np.linalg.norm(delta) <= 0.08 and abs(yaw) <= math.radians(6)
        else:
            ready = np.linalg.norm(handle[:2] - target_handle) <= POSITION_TOLERANCE and abs(yaw) <= YAW_TOLERANCE
        if clock() >= deadline or (count == GUIDED_MAX_STEPS and not ready):
            break
        if ready:
            return (geom, q), count
        step = guided_step(delta, yaw, geom, guard)
        start, v, w = _sample(base)
        if v > LINEAR_SETTLED or w > ANGULAR_SETTLED:
            raise Refused("Stop the wheelchair before continuing guided docking.")
        show(*step)
        _await_guided_step(base, arm, ctx, start, step, odom_to_axle, min(deadline, clock() + 18), clock, sleep)
    raise Refused("Stop the wheelchair. Guided docking did not converge.")


# Drive-wheel midpoint and floor from the Gen1.5 generated platform/floor layers.
AXLE_TO_MOUNT = (0.11595, 0.284325, 0.50190)
# The moving Nov 16 bag's base_link odometry turns about its own origin, at the axle.
ODOM_TO_AXLE = (0.0, 0.0)


def route_samples(route):
    """Sample the axle trajectory including segment joins and both endpoints."""
    axle, heading = np.zeros(2), 0.0
    yield axle, heading
    for distance, turn in route:
        steps = max(1, math.ceil(abs(distance) / 0.01), math.ceil(abs(turn) / math.radians(2)))
        delta = rotation(heading) @ np.array([distance, 0.0])
        for fraction in np.linspace(0, 1, steps + 1)[1:]:
            yield axle + fraction * delta, heading + fraction * turn
        axle, heading = axle + delta, heading + turn


def check_route(model, boxes, padding, route, axle_to_mount, *, door_angle=0.0):
    """Sweep the moving chair boxes against the fitted, stationary appliance."""
    if not boxes:
        raise Refused("Docking requires the measured chair and occupant layers")
    mount = np.asarray(axle_to_mount)
    swept = []
    for axle, heading in route_samples(route):
        transform = rotation(heading)
        for box in boxes:
            xy = axle - mount + transform @ (np.asarray(box["position"][:2]) + mount)
            swept.append(
                {
                    **box,
                    "position": [*xy, box["position"][2]],
                    "rpy_deg": [
                        *box.get("rpy_deg", (0, 0, 0))[:2],
                        box.get("rpy_deg", (0, 0, 0))[2] + math.degrees(heading),
                    ],
                }
            )
    # Allow for between-sample travel, pose fit, tracking error and axle calibration.
    obstacle = DoorSweep(model, swept, door_angle, door_angle, padding + 0.08, appliance=True).obstacle()
    if obstacle is not None:
        raise Refused(f"Docking route blocked by {obstacle}. Repark the chair.")
