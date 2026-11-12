"""Joint-space helpers for the move home: which way round a continuous joint goes, whether a plan
ends where it was asked to and goes there without swinging wide, the waypoints along the joint
line, and a timed trajectory from a path or from several plans. Pure numpy and rammp_arm's path
helpers: no ROS."""

import numpy as np
import rammp_arm
from rammp_arm import CONTINUOUS, _wrap, joint_trajectory

# cuRobo bounds the continuous joints here, as plain revolute ones
PLANNER_CONTINUOUS_LIMIT_RAD = 6.0
PLAN_END_RAD = 0.01  # a plan ending further than this from its goal (any joint) is not the goal's
# A plan may take a joint at most this much further than the straight way to its goal needs (its
# range along the plan, against |goal - start|): box_opening's trajectory sanity gate
# (SANITY_MARGIN_RAD). On flatbot (2026-11-10) cuRobo's every plan home from [1.006, -0.312, 2.699,
# -2.224, -2.790, -0.889, -0.623] took joint_1 out to 1.87 rad and back to 0.44 (1.45 rad for a
# 0.56 change), the hand toward the seated user; through the joint line's midpoint both legs
# stayed inside this.
SWING_MARGIN_RAD = 0.35
# One plan's first sample is where the last one ended (each is planned from the last one's end):
# further apart than this (any joint, not wrapped) is two plans that do not join
JOIN_RAD = 1e-3
WAYPOINTS = (0.5, 0.25, 0.75)  # fractions of the joint line toward HOME, tried in this order
# A waypoint's plan may end this far (any joint) from the waypoint it was sent, the home planned
# from where it really ends: cuRobo's FK-pose fallback lands on the same arm a little along its
# self-motion (box_opening's planner rig, 2026-11-12: 0.08-0.32 rad off on joints 1, 5 and 7 from
# a pose 25% into a recorded home, each refused and no waypoint left), while another arm at the
# same tool pose ends 0.9-4.3 rad off and stays refused. The plan that ends at the home keeps
# PLAN_END_RAD.
WAYPOINT_END_RAD = 0.35
# Where no single waypoint plans, the joint line in this many equal legs, fewest first. cuRobo
# swings wide on the long legs of an arm whose elbow must straighten and fold the other way
# (joint 4's sign against the home's, left so by a jog or a recovery) and follows the line on
# short ones: on that rig, from the 12 such poses over a 29 in table that were refused or homed
# only sometimes, of 24 direct plans 20 swung wide and the other 4 ended on another arm, and 29
# of the 62 single-waypoint legs that planned swung (joint 1 0.43-2.12 rad out and back); in 4
# legs 7 poses planned clean, in 6 legs one more.
LINE_LEGS = (4, 6)
# The plan-free steps (out of the padding, off a limit): the joint that moves most at most this
# fast at the peak of a minimum-jerk profile (rammp_arm.joint_trajectory), never in less than
# rammp_arm.MIN_DURATION_S
ESCAPE_SPEED_RAD_S = 0.15
# A joint too near its limit: rammp-curobo refuses a start inside joint_limit_margin_rad (0.10 on
# flatbot: "START_IN_LIMIT_MARGIN ... jog it away from the limit first"), and its finetune failed
# from up to 0.154 rad of room (box_opening, 2026-10-23, HOME planned from 0.161). So a start with a
# joint nearer than LIMIT_NEED_RAD takes a plan-free step along the joint line toward the home
# first: the shortest that leaves every joint LIMIT_FIRST_RAD clear, then LIMIT_STRIDE_RAD longer
# each time, up to LIMIT_MAX_STEP_RAD (the joint that moves most), the home planned from each in
# turn.
PLANNER_LIMIT_MARGIN_RAD = 0.10
LIMIT_NEED_RAD = PLANNER_LIMIT_MARGIN_RAD + 0.08
LIMIT_FIRST_RAD = PLANNER_LIMIT_MARGIN_RAD + 0.02
LIMIT_STRIDE_RAD = 0.05
LIMIT_MAX_STEP_RAD = 0.35
LINE_STEP_RAD = 0.01  # a plan-free line is sampled this finely for its checks


def wrap(d):
    """Angles `d` into [-pi, pi)."""
    return (np.asarray(d, float) + np.pi) % (2 * np.pi) - np.pi


def on_branch(goal, start):
    """`goal` with each continuous joint moved by whole turns to the value nearest `start`'s: the
    same arm configuration, reached the short way round, never a winding. A value that would leave
    the planner's bounds keeps the goal's own."""
    goal = np.asarray(goal, float).copy()
    start = np.asarray(start, float)
    for i in CONTINUOUS:
        near = start[i] + float(wrap(goal[i] - start[i]))
        if abs(near) <= PLANNER_CONTINUOUS_LIMIT_RAD:
            goal[i] = near
    return goal


def end_error(q, goal):
    """(rad, joint index): how far `q` is from `goal`, each continuous joint the short way round."""
    d = _wrap(np.asarray(q, float) - np.asarray(goal, float))
    j = int(np.argmax(np.abs(d)))
    return float(abs(d[j])), j


def swing(path, goal):
    """(rad, joint index): how much further than the straight way the plan `path` (k, J) takes its
    widest-swinging joint: the joint's range along the path (continuous joints followed round, not
    wrapped) less |goal - start| (`goal` on the start's branch). A plan that spins a continuous
    joint a whole turn to land on the same angle swings 2 pi."""
    P = np.asarray(path, float)
    if len(P) < 2:
        return 0.0, 0
    d = np.diff(P, axis=0)
    d[:, CONTINUOUS] = wrap(d[:, CONTINUOUS])
    U = np.vstack([P[:1], P[0] + np.cumsum(d, axis=0)])
    straight = np.abs(on_branch(goal, P[0]) - P[0])
    extra = (U.max(axis=0) - U.min(axis=0)) - straight
    j = int(np.argmax(extra))
    return float(extra[j]), j


def waypoints(start, goal, fractions=WAYPOINTS):
    """The joint states `fractions` of the way from `start` to `goal` (on the start's branch)."""
    start = np.asarray(start, float)
    goal = on_branch(goal, start)
    return [start + f * (goal - start) for f in fractions]


def stepped(path, speed=ESCAPE_SPEED_RAD_S):
    """Trajectory points [(t s, q, qd)] through the joint states `path` (k, J), from rest to rest
    on a minimum-jerk profile (rammp_arm.joint_trajectory), the joint that moves most at most
    `speed` at its peak; [] where the path does not move."""
    P = np.asarray(path, float)
    if len(P) < 2:
        return []
    # rammp_arm.JOINT_SPEED as it is now: arm.joint_speed sets it at startup, after this module is
    # imported, and a copy taken then would scale the step's cap by the two speeds' ratio
    out = joint_trajectory(P[0], list(P[1:]), speed_scale=speed / rammp_arm.JOINT_SPEED)
    if out is None:
        return []
    t, q, qd = out
    return [(float(ti), qi, vi) for ti, qi, vi in zip(t, q, qd, strict=True)]


def joined(plans):
    """One trajectory's points [(t, q, qd)] through several plans (each with t, q, qd), one after
    the other, each starting where the last ended (both at rest, as the planner plans them): the
    second's first sample, the first's last, is dropped and its times run on from the first's end.
    ValueError where a plan does not start within JOIN_RAD of the last one's end (any joint, not
    wrapped: a whole turn apart is a jump too)."""
    out, t0 = [], 0.0
    for k, plan in enumerate(plans):
        rows = list(zip(plan.t, plan.q, plan.qd, strict=True))
        if k:
            gap = np.abs(np.asarray(rows[0][1], float) - np.asarray(out[-1][1], float))
            if float(gap.max()) > JOIN_RAD:
                j = int(gap.argmax())
                raise ValueError(
                    f"plan {k + 1} starts {float(gap[j]):.4f} rad (joint {j + 1}) from where plan "
                    f"{k} ends"
                )
            rows = rows[1:]
        out += [(t0 + float(t), q, qd) for t, q, qd in rows]
        t0 = out[-1][0] if out else 0.0
    return out


def near_limits(q, limits, need=LIMIT_NEED_RAD):
    """[(joint index, room rad)] of the joints of `q` nearer than `need` to a limit; `limits`
    [(lower, upper) or (None, None)] per joint."""
    out = []
    for j, (lo, hi) in enumerate(limits):
        if lo is not None:
            room = float(min(q[j] - lo, hi - q[j]))
            if room < need:
                out.append((j, room))
    return out


def limit_steps(start, goal, limits):
    """The plan-free steps along the joint line from `start` toward `goal` (on the start's branch)
    that take every joint at least LIMIT_FIRST_RAD from its limits, shortest first: [(fraction of
    the line, joint state)], none longer than LIMIT_MAX_STEP_RAD. ValueError where the goal itself
    stands nearer than LIMIT_NEED_RAD to a limit."""
    start = np.asarray(start, float)
    goal = on_branch(goal, start)
    first = 0.0
    for j, (lo, hi) in enumerate(limits):
        if lo is None:
            continue
        if min(goal[j] - lo, hi - goal[j]) < LIMIT_NEED_RAD:
            raise ValueError(f"the home's joint {j + 1} is too near its limit for the planner")
        # the room to each limit changes linearly along the line and is clear at its end
        for r0, r1 in ((start[j] - lo, goal[j] - lo), (hi - start[j], hi - goal[j])):
            if r0 < LIMIT_FIRST_RAD:
                first = max(first, (LIMIT_FIRST_RAD - r0) / (r1 - r0))
    span = float(np.max(np.abs(goal - start)))
    out, k = [], 0
    while span > 0.0:
        f = min(1.0, first + k * LIMIT_STRIDE_RAD / span)
        k += 1
        if f * span > LIMIT_MAX_STEP_RAD + 1e-9:
            break
        if f > 0.0:
            out.append((f, start + f * (goal - start)))
        if f >= 1.0:
            break
    return out


def line(a, b, step=LINE_STEP_RAD):
    """The straight joint line from `a` to `b`, sampled every `step` (the joint that moves most)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.ceil(float(np.abs(b - a).max()) / step)))
    return a + (np.arange(n + 1) / n)[:, None] * (b - a)
