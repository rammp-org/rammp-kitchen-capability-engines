"""Drive the wheelchair straight a fixed distance with self.base: the moves the open and the close plan, made by the
base when the goal sets `drive_base` (otherwise the goal asks the person, as before).

self.base (capability_engine/base.py) sends one velocity per send_twist() and has no "drive this far", so this is that
loop: AUTONOMOUS on the MIB, then a twist every PERIOD_S until odometry shows the distance, then zero twists. It stops,
raising BaseError, if the base goes the wrong way or further than asked, if the time runs out, or if the arm is
e-stopped; self.base itself raises on a cancel, on the MIB leaving AUTONOMOUS and on stale odometry. Every way out sends
a zero twist; the engine sends more, and asks the MIB for IDLE, when the action ends.
"""

import math
import time

from capability_engine import ArmError, BaseError
from door_common.common import Refused, say

PERIOD_S = 0.05  # a twist every 50 ms: the MIB stops the base 300 ms after the last one
SETTLE_S = 0.5  # zero twists for this long before the end is measured
PROGRESS_S = 1.0  # while driving, the distance so far is logged this often

# node ROS parameters, per robot: the goal's base_speed limit, the longest move, how close to the distance it must
# end, how long past distance / speed it may take, and the MIB's drive profile while it drives
PARAMS = {
    "base_max_speed": 0.15,  # m/s
    "base_max_distance": 0.35,  # m
    "base_tolerance": 0.01,  # m
    "base_timeout_margin": 5.0,  # s
    "base_profile": "low",
}
SPEED_DEFAULT = 0.05  # the goal's base_speed default (m/s)


def _pose(odom):
    p, q = odom.pose.pose.position, odom.pose.pose.orientation
    return p.x, p.y, math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def check(base, arm):
    """Before the arm moves: odometry is there and the arm isn't e-stopped. Moves nothing. Raises Refused."""
    try:
        base.odometry()
        estopped = arm.control_status().estopped  # the e-stop the drive watches: there before anything moves
    except (BaseError, ArmError) as e:
        raise Refused(f"drive_base: {e} -- nothing moved.") from e
    if estopped:
        raise Refused("drive_base: the arm is e-stopped -- nothing moved.")


def move_chair(base, arm, ctx, move, *, speed, cfg, execute):
    """Drive the wheelchair straight `move` m (+ back, away from the fridge; - forward). Returns how far it really moved
    (m, + back), within cfg's base_tolerance of `move`. Refuses (nothing moved) unless `execute`, over
    base_max_distance or base_max_speed, or with the arm e-stopped."""
    dist, tol = abs(move), cfg["base_tolerance"]
    sign = -1.0 if move > 0 else 1.0  # back is -x in base_link
    if not execute:
        raise Refused("drive_base: a dry run never drives the base.")
    if dist > cfg["base_max_distance"]:
        raise Refused(f"drive_base: {dist * 100:.0f} cm is over base_max_distance -- nothing moved.")
    if not 0.0 < speed <= cfg["base_max_speed"]:
        raise Refused(f"drive_base: base_speed {speed} is not in (0, {cfg['base_max_speed']}] -- nothing moved.")
    if arm.control_status().estopped:
        raise Refused("drive_base: the arm is e-stopped -- nothing moved.")
    if dist < tol:
        return 0.0
    ctx.checkpoint()
    base.request_mode("autonomous", profile=cfg["base_profile"])
    x0, y0, yaw0 = _pose(base.odometry())

    def moved():
        """(along the starting heading, + forward; straight-line distance from the start)."""
        x, y, _ = _pose(base.odometry())
        return (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0), math.hypot(x - x0, y - y0)

    timeout_s = dist / speed + cfg["base_timeout_margin"]
    say(f"drive_base: driving the wheelchair {dist * 100:.0f} cm {'back' if move > 0 else 'forward'} at {speed} m/s")
    t0 = time.monotonic()
    next_log, sent = t0 + PROGRESS_S, 0
    try:
        while True:
            if arm.control_status().estopped:
                raise BaseError("the arm is e-stopped: base stopped")
            along, away = moved()
            now = time.monotonic()
            if now >= next_log:  # the run log says how the drive went even when it stalls (11-11: 0 cm, no trace)
                say(
                    f"drive_base: {now - t0:.1f} s, {sent} twists sent, moved {along * 100:+.1f} cm along the start "
                    f"heading ({away * 100:.1f} cm from the start)"
                )
                next_log += PROGRESS_S
            if sign * along < -tol:
                raise BaseError(f"the base moved the wrong way ({along * 100:+.1f} cm): stopped")
            if away > dist + tol:
                raise BaseError(f"the base moved {away * 100:.1f} cm, more than asked: stopped")
            if now - t0 > timeout_s:
                raise BaseError(
                    f"the base did not cover {dist * 100:.0f} cm in {timeout_s:.0f} s: stopped after moving "
                    f"{along * 100:+.1f} cm ({sent} twists sent; the run's bag has the MIB's status and arbitration)"
                )
            if dist - sign * along <= speed * PERIOD_S:  # would pass the end before the next twist
                break
            base.send_twist(sign * speed, 0.0)  # Cancelled once cancelled; BaseError once the MIB left AUTONOMOUS
            sent += 1
            time.sleep(PERIOD_S)
    finally:
        base.stop()
    t_end = time.monotonic() + SETTLE_S
    while time.monotonic() < t_end:
        base.stop()
        time.sleep(PERIOD_S)
    along, _ = moved()
    say(f"drive_base: stopped, moved {abs(along) * 100:.1f} cm {'forward' if along > 0 else 'back'}")
    if abs(sign * along - dist) > tol:
        off = (sign * along - dist) * 100
        raise BaseError(f"the base ended {off:+.1f} cm from the distance: the arm doesn't go on")
    return -along
