"""Joint-space helpers for the move home: which way round a continuous joint goes, whether a plan
ends where it was asked to, the waypoints along the joint line, and a timed trajectory from a path
or from several plans. Pure numpy: no ROS."""

import numpy as np

CONTINUOUS = (0, 2, 4, 6)  # joint_1, joint_3, joint_5, joint_7 turn without end
# cuRobo bounds the continuous joints here, as plain revolute ones
PLANNER_CONTINUOUS_LIMIT_RAD = 6.0
PLAN_END_RAD = 0.01  # a plan ending further than this from its goal (any joint) is not the goal's
WAYPOINTS = (0.5, 0.25, 0.75)  # fractions of the joint line toward HOME, tried in this order
# the step out of the padding: the joint that moves most, at most this fast
ESCAPE_SPEED_RAD_S = 0.15
ESCAPE_MIN_S = 1.0  # and never in less than this
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
    d = np.asarray(q, float) - np.asarray(goal, float)
    d[list(CONTINUOUS)] = wrap(d[list(CONTINUOUS)])
    j = int(np.argmax(np.abs(d)))
    return float(abs(d[j])), j


def waypoints(start, goal, fractions=WAYPOINTS):
    """The joint states `fractions` of the way from `start` to `goal` (on the start's branch)."""
    start = np.asarray(start, float)
    goal = on_branch(goal, start)
    return [start + f * (goal - start) for f in fractions]


def timed_path(path, speed=ESCAPE_SPEED_RAD_S, min_s=ESCAPE_MIN_S):
    """Trajectory points [(t s, q, qd)] through the joint states `path` (k, J): each segment timed
    so the joint that moves most goes at `speed`, the whole at least `min_s`, at rest at both ends,
    the velocities between them the path's slope."""
    P = np.asarray(path, float)
    seg = np.abs(np.diff(P, axis=0)).max(axis=1) / speed
    total = float(seg.sum())
    if total < min_s:
        seg = seg * (min_s / total)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    V = np.zeros_like(P)
    if len(P) > 2:
        V[1:-1] = (P[2:] - P[:-2]) / (t[2:] - t[:-2])[:, None]
    return [(float(ti), q, v) for ti, q, v in zip(t, P, V, strict=True)]


def joined(plans):
    """One trajectory's points [(t, q, qd)] through several plans (each with t, q, qd), one after
    the other, each starting where the last ended (both at rest, as the planner plans them): the
    second's first sample, the first's last, is dropped and its times run on from the first's end.
    """
    out, t0 = [], 0.0
    for k, plan in enumerate(plans):
        rows = list(zip(plan.t, plan.q, plan.qd, strict=True))
        if k:
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
