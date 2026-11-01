"""The vertical final approach: a planner waypoint, then the arm's own line.

cuRobo's approach_offset_m (RAMMP-CuRobo feat/perceived-world 21b1ba0)
never reached a release: the v1.0.0 PlanToPose carries a target and
start_joints and nothing else. Its job stands — the last approach_offset_m
metres of every contact descent must come straight down onto the goal,
because a diagonal descent touches the surface before its lateral
convergence finishes (edge presses, field 2026-09-01).

Two plans were the first answer (one to a waypoint approach_offset_m above
the goal, one from there to the goal), and they missed: a plan between
two vertically aligned poses is vertical only at its ends. Its joint-space
path bowed 5-7 mm radially outward, widest 26-29 mm above the goal, which
is exactly where the guard trips (travel_m + TCP_OFFSET_M above it); every
press since the migration landed 5.6-6.6 mm off the button, and the
2026-09-16 probe on the live planner reproduced each miss to a millimetre.

So the waypoint stays a planner result (collision-checked, exact to
0.01 mm) and the stretch below it is built from the arm's own kinematics
(runtime/kinematics.py): a straight line in 2 mm steps, the attitude held,
0.01 mm from vertical. cuRobo accepts the line's end joints as a valid
target (checked live). Should the line be refused (a joint limit, a
singularity), the planner's own plan flies instead and the message says so.

They fly as ONE trajectory with a rest at the waypoint (every cuRobo plan
starts and ends at rest). An approach action in cuRobo v1.1.0 would
remove the stop.

The line carries a REAL velocity profile (line_trajectory). It used to
carry placeholder stamps — 0.05 s a step, zero velocity at every waypoint
— on the belief that the Runner re-times every leg. It re-times unguarded
and merged groups; a guarded descent flown on its own (the press from
staging, grip:down, the plan-free lift) went to the driver as it was,
and the arm stop-started every 2 mm: 2.98 s for the 45 mm grip descent
and 3 Nm of torque ripple in free air on the live run of 2026-09-17.
"""

import math

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening.constants import JOINT_VMAX
from box_opening.runtime.retime import RetimeParams, positions_to_traj, retime_group
from box_opening.runtime.stamps import secs, set_stamp

_CHAIN = None
_ROOT = "base_link"
_URDF_PATH = None  # None = use kinematics.urdf_path() (the installed bench URDF)


def set_chain_root(root, urdf_path=None):
    """Switch the URDF chain's root frame ('base_link' on bench, 'arm_mount'
    on flatbot). `urdf_path` overrides the default installed URDF when the
    active robot's description differs (e.g. the nightly arm on flatbot, whose
    URDF is rendered with arm_mount as root and published on
    /arm/robot_description). Resets the singleton so the next _chain() call
    rebuilds with the new root and path."""
    global _ROOT, _URDF_PATH, _CHAIN, _GRAVITY
    _ROOT = str(root)
    _URDF_PATH = urdf_path
    _CHAIN = None
    _GRAVITY = None


def chain_ready():
    """(ok, why): whether _chain() has a URDF for its root. Off the bench
    root there is no installed URDF to fall back to — the bench one has no
    arm_mount — so a root set without the robot's own description has no
    kinematics at all, and every plan-free line and FK must refuse."""
    if _ROOT != "base_link" and _URDF_PATH is None:
        return False, "no arm URDF rooted at %s (is /arm/robot_description published?)" % _ROOT
    return True, ""


def _chain():
    """The arm's kinematic chain, parsed once from the driver's URDF."""
    global _CHAIN
    if _CHAIN is None:
        ok, why = chain_ready()
        if not ok:
            raise RuntimeError(why)
        from box_opening.runtime.kinematics import ArmChain

        _CHAIN = ArmChain(root=_ROOT, path=_URDF_PATH)
    return _CHAIN


_GRAVITY = None


def _gravity():
    """The arm's gravity model, from the same URDF and root as _chain()."""
    global _GRAVITY
    if _GRAVITY is None:
        ok, why = chain_ready()
        if not ok:
            raise RuntimeError(why)
        from box_opening.runtime.kinematics import GravityModel

        _GRAVITY = GravityModel(root=_ROOT, path=_URDF_PATH)
    return _GRAVITY


def line_trajectory(joint_names, points):
    """A JointTrajectory through joint-space `points` with a rest-to-rest
    velocity profile at full cruise (positions untouched, runtime/retime):
    exactly what a planner trajectory is, so a leg's speed dilates it, a
    warp reshapes it and a merged group re-profiles it like any other.
    Only a guarded leg flown on its own flies this timing (dilated by its
    speed, or warped): the runner re-times every other motion's positions
    itself (Runner.retime_params, a compliant run's own profile)."""
    traj, _info = retime_group([positions_to_traj(joint_names, points)], [1.0], JOINT_VMAX, RetimeParams())
    return traj


class ChainedPlan:
    """A PlanToPose-shaped result for the two plans together. `waypoint` is
    (index into trajectory.points, joints) where the straight line begins."""

    def __init__(self, success, message, trajectory=None, planning_time=0.0, waypoint=None):
        self.success = success
        self.message = message
        self.trajectory = trajectory
        self.planning_time = planning_time
        self.waypoint = waypoint


def _point(p, t):
    q = JointTrajectoryPoint()
    q.positions = list(p.positions)
    q.velocities = list(p.velocities)
    q.accelerations = list(p.accelerations)
    set_stamp(q.time_from_start, t)
    return q


def on_branch_of(traj, q_ref):
    """A copy of planned trajectory `traj` with each continuous joint
    (branches.CONTINUOUS_JOINTS) moved by the one whole number of turns that
    puts its first point within half a turn of `q_ref`: every difference
    kept, so the same motion to the arm, whose driver compares every target
    with the measurement the short way round (runtime/branches) — written
    to continue a path that ends at `q_ref`. The planner is handed
    planner_branch(start), HOME's side of the wrap, which can sit a turn from
    a plan-free line's raw end (review 2026-10-27: joint_3 +0.450 against
    -5.833), and a junction between them is a full revolution in one goal
    (legs.MERGE_GAP_RAD)."""
    from box_opening.runtime.branches import CONTINUOUS_JOINTS

    first = [float(v) for v in traj.points[0].positions]
    shift = [0.0] * len(first)
    for j in CONTINUOUS_JOINTS:
        if j < len(first):
            shift[j] = 2.0 * math.pi * round((float(q_ref[j]) - first[j]) / (2.0 * math.pi))
    out = JointTrajectory()
    out.header = traj.header
    out.joint_names = list(traj.joint_names)
    for p in traj.points:
        q = _point(p, secs(p.time_from_start))
        q.positions = [float(v) + s for v, s in zip(p.positions, shift, strict=False)]
        out.points.append(q)
    return out


def chain(first, second):
    """`first` then `second` as one trajectory. `second` opens at `first`'s
    final configuration, so its opening point is dropped: time stays
    strictly increasing through the waypoint."""
    out = JointTrajectory()
    out.joint_names = list(first.joint_names)
    for p in first.points:
        out.points.append(_point(p, secs(p.time_from_start)))
    offset = secs(first.points[-1].time_from_start)
    for p in second.points[1:]:
        out.points.append(_point(p, secs(p.time_from_start) + offset))
    return out


def plan_with_vertical_approach(plan_pose, xyz, quat_xyzw, start_joints, offset_m, chain_model=None):
    """Plan to `xyz`/`quat_xyzw` from `start_joints`, arriving from a
    waypoint `offset_m` straight above the goal along a straight line.

    `plan_pose(xyz, quat_xyzw, start_joints)` is one PlanToPose round trip
    returning a result (success, message, trajectory, planning_time) or
    None when the planner did not answer. `chain_model` (an ArmChain, or
    anything with straight_line(q, xyz, quat) -> (points, why)) defaults
    to the driver URDF's chain."""
    if offset_m <= 0.0:
        return plan_pose(xyz, quat_xyzw, start_joints)
    above = [xyz[0], xyz[1], xyz[2] + float(offset_m)]
    first = plan_pose(above, quat_xyzw, start_joints)
    if first is None:
        return None
    if not first.success:
        return ChainedPlan(
            False,
            "approach waypoint %.3f m above the goal: %s" % (offset_m, first.message),
        )
    end = list(first.trajectory.points[-1].positions)
    kin = chain_model if chain_model is not None else _chain()
    pts, why = kin.straight_line(end, xyz, quat_xyzw)
    if pts is not None:
        return ChainedPlan(
            True,
            "straight descent: %d points over %.0f mm" % (len(pts), 1000.0 * offset_m),
            chain(first.trajectory, line_trajectory(first.trajectory.joint_names, pts)),
            float(first.planning_time or 0.0),
            waypoint=(len(first.trajectory.points) - 1, list(end)),
        )
    # the planner's own plan below the waypoint — it bows (see above), so
    # the message carries the reason the line was refused
    second = plan_pose(xyz, quat_xyzw, end)
    if second is None:
        return None
    if not second.success:
        return ChainedPlan(False, "descent from the approach waypoint: %s" % second.message)
    return ChainedPlan(
        True,
        "PLANNER descent (may bow — the straight line was refused: %s)" % why,
        chain(first.trajectory, second.trajectory),
        float(first.planning_time or 0.0) + float(second.planning_time or 0.0),
        # the descent begins at the same junction either way: a guard flown
        # with this leg arms there, and a re-fit may still replace the
        # planner's bowing descent with the straight line
        waypoint=(len(first.trajectory.points) - 1, list(end)),
    )
