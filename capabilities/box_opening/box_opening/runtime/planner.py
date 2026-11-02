"""What box-opening holds true about the rammp-curobo planner.

The planner flatbot runs (rammp-curobo dev 4fa044d, cuRobo v0.7.8) calls
some plans successes that do not end where they were asked, and the
mission flies whatever comes back: core._plan_motion makes a plan's last
point the next leg's start. PlanToJoints falls back to planning to the
goal's FK pose when its native joint solve fails (rammp-curobo's
planner.py plan_to_joints, method 'auto'): the same tool to cuRobo's
5 mm / 5.7 deg, through whatever joints its IK finds, another elbow or
wrist among them, said only in goal_mismatch_rad. cuRobo's finetune can
hand back the previous solve's plan as a success (rammp-curobo #37). A
pose plan that winds a continuous joint is replanned onto the start's
branch, and when that replan fails the winding plan comes back as a
success, said only in its message (LONG_WAY_ROUND). And the node plans
one goal at a time and turns any other away, unplanned (PLANNER_BUSY).

So the client measures each plan the planner calls a success against what
it asked for (PlannerClient._checked_joints, _checked_pose) by what is
here, and hands one that ends elsewhere back as a failed plan (Refused);
the run record keeps what the planner said of a leg's plan and how its end
met the leg's target (plan_record). Nothing here talks to ROS: it reads
the planner's results, so core and the runner import it as they import
legs.
"""

import math
from collections import namedtuple

from box_opening import LOGGER as log
from box_opening.geometry import ang_diff
from box_opening.runtime.branches import CONTINUOUS_JOINTS

# The planner's own words (rammp-curobo 4fa044d): planner_node.py's answer
# to a goal that arrives while it plans another, and its planner.py's
# warning on a pose plan it hands back as a success although it winds a
# continuous joint, its replan onto the start's branch having failed —
# 2026-10-28 15:56:05, word for word: "planned 219 points, 4.38 s at full
# speed — warning: plan ends on a 2*pi branch away from the start (it goes
# the long way round) and the replan onto the start's branch failed
# (MotionGenStatus.DT_EXCEPTION); returning it as planned"
PLANNER_BUSY = "planner busy"
LONG_WAY_ROUND = "goes the long way round"

# A plan the planner called a success and the client will not hand on
# (PlannerClient._checked_joints, _checked_pose, _no_long_way): shaped like
# the planner's result, so a caller does with it whatever it does with a
# failed plan. No trajectory: nothing can fly it by mistake.
Refused = namedtuple("Refused", "success message trajectory planning_time")


def finite(v):
    """`v` as a float, or None when it is no finite number."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def busy(res):
    """Whether planner result `res` is its "busy with another plan": not
    planned at all, only turned away."""
    return (
        res is not None
        and not getattr(res, "success", False)
        and str(getattr(res, "message", "") or "").startswith(PLANNER_BUSY)
    )


def long_way(res):
    """Whether planner result `res` is a success its message says goes the
    long way round."""
    return bool(getattr(res, "success", False)) and LONG_WAY_ROUND in str(getattr(res, "message", "") or "")


def refuse(res, why):
    """`res`, a plan the planner called a success, refused because `why`:
    one console line, and a failed result whose message is `why` and then
    what the planner said of it."""
    log.info("[client] plan refused: %s" % why)
    said = str(getattr(res, "message", "") or "")
    return Refused(
        False,
        "%s (planner: %s)" % (why, said) if said else why,
        None,
        finite(getattr(res, "planning_time", None)) or 0.0,
    )


def plan_end(traj):
    """The joints at the last point of trajectory `traj`, or None with no
    last point, or one with a number in it that is not finite: no end to
    fly to, nor to check."""
    try:
        end = [float(v) for v in traj.points[-1].positions]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return end if end and all(math.isfinite(v) for v in end) else None


def joint_end_error(traj, q_goal):
    """(rad, joint): how far the last point of `traj` is from the joints
    `q_goal` at its worst joint (0-based) — each continuous joint
    (branches.CONTINUOUS_JOINTS) the short way round, so a goal written a
    whole turn from where the plan ends is the same goal; a bounded joint
    as the numbers are written; a goal that is no finite number as far off
    as a joint can be (inf) — or (None, None) with no finite last point of
    `q_goal`'s length to measure."""
    end = plan_end(traj)
    try:
        goal = [float(v) for v in q_goal]
    except (TypeError, ValueError):
        return None, None
    if end is None or len(end) != len(goal):
        return None, None
    errs = [
        abs(ang_diff(e, g)) if j in CONTINUOUS_JOINTS else abs(e - g)
        for j, (e, g) in enumerate(zip(end, goal, strict=False))
    ]
    errs = [e if math.isfinite(e) else math.inf for e in errs]
    j = max(range(len(errs)), key=errs.__getitem__)
    return float(errs[j]), j


def pose_end_error(traj, xyz, quat_xyzw):
    """(m, rad): how far the tool frame at the last point of `traj` stands
    from `xyz` at the attitude `quat_xyzw` — its offset and its turn, by the
    arm's own kinematics (approach._chain: the URDF cuRobo plans with, its
    FK 0.01 mm from the planner's), a number that is not finite as inf — or
    None with no finite last point, or no chain to measure with. The turn
    is read off the trace, which holds at half a turn too: a rotation
    vector loses its axis there (kinematics.rotvec reads a flip of exactly
    pi as none). A measurement only: it never raises."""
    end = plan_end(traj)
    if end is None:
        return None
    try:
        import numpy as np

        from box_opening.perception.d405 import quat_to_mat
        from box_opening.runtime.approach import _chain

        R, t = _chain().fk(end)
        off = float(np.linalg.norm(np.asarray(xyz, float) - np.asarray(t, float)))
        R_goal = quat_to_mat(*(float(v) for v in quat_xyzw))
        cos_turn = (float(np.trace(R_goal @ np.asarray(R, float).T)) - 1.0) / 2.0
    except Exception:  # noqa: BLE001 - no chain (no URDF for its root, ...): nothing measured
        return None
    turn = math.acos(min(1.0, max(-1.0, cos_turn))) if math.isfinite(cos_turn) else math.inf
    return (off if math.isfinite(off) else math.inf), turn


def plan_record(plan, target, plan_s=None, refused=()):
    """What the run record keeps of a leg's plan (Leg.plan, runner._log):
    the planner's message and solve time; its goal_mismatch_rad where it
    reports one, which a PlanToJoints plan does — a pose leg planned to the
    arm's own IK answer is one (PlannerClient._plan_pose_near), measured
    against those joints; how far the plan ends from `target` (Leg.target)
    as the client's checks measure it, end_rad for ("joints", q7), end_mm
    and end_deg for ("pose", xyz, quat_xyzw, ...); how long the asking took
    (`plan_s`); and the plans the client refused while it asked
    (`refused`, PlannerClient.refusals), where there were any: a refused
    joint plan the pose plan then stood in for leaves no other trace. A
    measurement only."""
    out = {"message": "no response" if plan is None else str(getattr(plan, "message", "") or "")}
    solve_s = finite(getattr(plan, "planning_time", None))
    out["solve_s"] = None if solve_s is None else round(solve_s, 3)
    # radians to 1e-7: the native solve ends ~5e-7 from its goal
    mismatch = finite(getattr(plan, "goal_mismatch_rad", None))
    if mismatch is not None:
        out["goal_mismatch_rad"] = round(mismatch, 7)
    traj = getattr(plan, "trajectory", None)
    kind = target[0] if target else None
    if kind == "joints":
        rad = finite(joint_end_error(traj, target[1])[0])
        out["end_rad"] = None if rad is None else round(rad, 7)
    elif kind == "pose":
        err = pose_end_error(traj, target[1], target[2])
        off, turn = (None, None) if err is None else (finite(err[0]), finite(err[1]))
        out["end_mm"] = None if off is None else round(1000.0 * off, 2)
        out["end_deg"] = None if turn is None else round(math.degrees(turn), 3)
    if plan_s is not None:
        out["plan_s"] = round(float(plan_s), 3)
    if refused:
        out["refused"] = list(refused)
    return out
