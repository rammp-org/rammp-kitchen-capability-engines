"""The primitives (spec §5): plan/execute split, guarded descents shared.

Each primitive chain-plans from `state.joints` (tour_demo chaining). A
contact leg invalidates downstream pre-plans: it increments the chain, so
nothing merges across it, and what follows it is LAZY — planned by the
Runner from the live arm once the guard has stopped it.

Ctx.last_pose / last_world track the most recent commanded tool pose and
the world it was planned against.
"""

import time
from dataclasses import dataclass

from box_opening import LOGGER as log
from box_opening.constants import (
    CONTACT_SPEED,
    GAINS_SOFT,
    GRIPPER_CMD_OPEN,
    HOME_GAINS,
    HOME_SPEED,
    IMPEDANCE_NAMES,
    PLANNER_LIMIT_MARGIN_RAD,
    SANITY_MARGIN_RAD,
    TCP_OFFSET_M,
    TIP_TO_TOOL_M,
    TRANSIT_SPEED,
)
from box_opening.runtime.guards import (
    GuardSpec,
    in_band,
    sanity_violations,
)
from box_opening.runtime.legs import Kind, Leg
from box_opening.runtime.planner import Refused, plan_record

# a set-down's success IS the guard trip — overdrive the commanded depth
# past nominal surface contact so the table is always felt (err-TALL world
# modeling can otherwise leave an exact-height target arriving untouched)
SETDOWN_OVERDRIVE_M = 0.005

# a set-down trip with the fingertips more than this above where the lid
# meets the table struck something on the way down. The descent's progress
# cannot say so once it flies merged behind the carry: the guard reads the
# whole group's, and flatbot 2026-10-23 17:57 tripped at 70 % of it — the
# wrist camera on the microwave beside the drop spot, 134 mm up — and let
# the lid go there. The slack covers the table under the spot differing
# from the one measured at the box and the lid thickness's estimate.
SETDOWN_HIGH_M = 0.04

# clearance between the CARRIED lid's underside and the container top
# during the place transit — the planner cannot model a held object
CARRY_CLEAR_M = 0.04

# extra container xy half-extent in FULL worlds planned after a contact
# leg: a press can scoot the box off its detected pose (field 2026-09-01,
# ~2 cm at 8.1 Nm) and a transit must not thread the needle beside a
# cuboid the box may no longer be inside
CONTACT_SHIFT_PAD_M = 0.03


@dataclass
class Ctx:
    model: object
    cpose: object
    client: object
    worlds: object
    lid_at: object = None  # set after Place(lid): later worlds carry the lid
    lid_drop: object = None  # runtime-resolved drop spot (adapts to the box)
    lid_drops: list = None  # every spot the cameras saw room at, best first (press_demo.clear_lid_drops)
    config_path: str = None
    last_pose: tuple = None  # (xyz, quat_xyzw) of the last commanded pose
    last_world: tuple = None  # (name, path) of the last interaction world
    contact_pad: float = 0.0  # container xy padding once contact has happened
    mission_frames: object = None  # this run's detection record (detection_set.MissionFrames), --execute missions
    # home's legs planned from the look pose while the way back to it flew
    # (reach_lid_off), for a stop there; dropped once the arm moves on
    home_ahead: list = None
    # the lift that took the lid off the box (reach_lid_off.LiftOff), while the lid is in the
    # fingers: where it goes back to on a STOP before the carry (reach_lid_off.put_lid_back)
    lid_lifted: object = None
    # the scene camera's SceneLocator (its grabber streams the whole run) and
    # the lid-off's video of both cameras (perception/camera_video), --execute
    # full runs (reach_press._reach_press)
    scene_locator: object = None
    lid_video: object = None
    # where the press held the pads on the knob (x, y; depth_press.compliant_hold's
    # held_pad_xyz, felt), the box's button as the arm measured it: 'nearest'
    # set-down rings centre on the box there (reach_lid_off._nearest_points)
    held_xy: tuple = None
    # the counter's edges each map of this run showed, as the body check's keep-out boxes (source ->
    # [box]: press_demo.keep_out_counter); every source's are in force at once
    counter_boxes: dict = None
    # the wrist survey's frames as points (tasks/survey: one {"points" (N, 3) float32 in the base
    # frame, "cam_xyz", "tag"} per frame), where the scene camera saw no box and the survey found
    # it: the counter round the box as the wrist saw it, for what else needs the counter (the
    # cabinet edge's keep-out); the lid's spot gets the same points through the fix's table_points
    survey_clouds: list = None


@dataclass
class PlanState:
    joints: list  # predicted joints the next leg plans from; None = lazy
    chain: int  # bumped by every contact leg: nothing merges across it


def tcp_z(z):
    """Commanded tool_frame z for a FINGERTIP target z.

    The fingertips reach TCP_OFFSET_M beyond the frame the planner is
    commanded in, so a target meant for them is commanded that much
    higher. Everything fingertip-referenced goes through here: the press
    stroke, the grip descent, the lid set-down and the carry hover — they
    must shift TOGETHER, or the geometry between grabbing the lid and
    putting it down stops matching."""
    return float(z) + TCP_OFFSET_M


def band_verify(band):
    """Grip-band check with an honest fallback when no gripper state exists."""

    def verify(ctx):
        if ctx.gripper_pos is None:
            return True, "band unchecked — no gripper state"
        ok = in_band(ctx.gripper_pos, band)
        return ok, "grip %.3f vs band %s" % (ctx.gripper_pos, list(band))

    return verify


def _full_world(ctx, tag=""):
    tag = tag or ("lid" if ctx.lid_at is not None else "")
    return ctx.worlds.push_name(
        "full",
        model=ctx.model,
        cpose=ctx.cpose,
        lid_at=ctx.lid_at,
        tag=tag,
        container_pad_xy=ctx.contact_pad,
    )


def _interaction_world(ctx, contact_z, depth_max, tag):
    return ctx.worlds.push_name(
        "interaction",
        model=ctx.model,
        cpose=ctx.cpose,
        contact_z=contact_z,
        depth_max=depth_max,
        lid_at=ctx.lid_at,
        tag=tag,
    )


def descent_index(plan):
    """Index into a plan's trajectory where its vertical final descent
    begins (its approach waypoint), or None for a plan without one."""
    wp = getattr(plan, "waypoint", None)
    return None if wp is None else int(wp[0])


class PlanRefused(RuntimeError):
    """_plan_motion's failure for a plan the planner called a success and
    the client refused (runtime/planner.Refused): it ends elsewhere than it
    was asked, or goes the long way round. A RuntimeError like any failed
    plan's, so every caller's stop or fallback applies; home_arm, whose
    fallback for a failed plan is a world that knows less, tells it apart."""


def _plan_motion(
    ctx,
    state,
    name,
    target,
    world,
    speed,
    guard=None,
    invalidates=False,
    verify=None,
    lazy=False,
):
    world_name, world_path = world
    kind, *rest = target
    _refuse_goal_inside(name, target, state.joints)
    if lazy or state.joints is None:
        # LAZY: this leg follows an expected touch, so its start is unknown
        # until the guard stops the arm. Pre-planning it was pure waste —
        # the post-touch replan discarded it every run (audit 2026-09-02).
        # The Runner plans it from live joints, once, in the leg's world.
        # Everything chained after it is lazy too (its end is unknown).
        if kind == "pose":
            ctx.last_pose = (list(rest[0]), list(rest[1]))
        leg = Leg(
            name=name,
            kind=Kind.MOTION,
            traj=None,
            speed=speed,
            guard=guard,
            world=world_name,
            world_path=str(world_path),
            chain=state.chain,
            target=target,
            goal_joints=None,
            verify=verify,
        )
        next_chain = state.chain + 1 if invalidates else state.chain
        return leg, PlanState(joints=None, chain=next_chain)
    # Worlds are a PLAN-time concern (spec §6): the planner must hold this
    # leg's world BEFORE the plan is requested — SetWorld only at execution
    # time means every trajectory was actually planned against the previous
    # world (2026-08-24 review, critical). The client deduplicates pushes of
    # the world it already holds (one tracker, review 2026-09-02).
    ok, msg = ctx.client.set_world(world_path)
    if not ok:
        raise RuntimeError("set_world before planning %s failed: %s" % (name, msg))
    n_refused = len(getattr(ctx.client, "refusals", ()))
    n_busy = getattr(ctx.client, "busy_retries", 0)
    t_plan = time.monotonic()

    if kind == "pose":
        offset = float(rest[2]) if len(rest) > 2 else 0.0
        plan = ctx.client.plan_to_pose(rest[0], rest[1], state.joints, approach_offset_m=offset)
        ctx.last_pose = (list(rest[0]), list(rest[1]))
    else:
        plan = ctx.client.plan_to_joints(rest[0], state.joints)
    plan_s = time.monotonic() - t_plan
    if plan is None or not plan.success:
        raise (PlanRefused if isinstance(plan, Refused) else RuntimeError)(
            "planning failed for %s: %s" % (name, getattr(plan, "message", "no response"))
        )
    if "PLANNER descent" in str(getattr(plan, "message", "")):
        # the straight final stretch was refused and the planner's own
        # (bowing) plan flies instead — say so, a press may land off-centre
        log.info("  NOTE %s: %s" % (name, plan.message))
    end = list(plan.trajectory.points[-1].positions)
    _refuse_goal_inside(name, target, state.joints, end)
    leg = Leg(
        name=name,
        kind=Kind.MOTION,
        traj=plan.trajectory,
        speed=speed,
        guard=guard,
        world=world_name,
        world_path=str(world_path),
        chain=state.chain,
        target=target,
        goal_joints=end,
        verify=verify,
        plan_s=plan_s,
        plan_server_s=getattr(plan, "planning_time", None),
        # the client's refusals while it planned this leg: none, or the
        # joint plan the pose plan stood in for (PlannerClient.refusals)
        plan=plan_record(
            plan,
            target,
            plan_s,
            getattr(ctx.client, "refusals", ())[n_refused:],
            busy=getattr(ctx.client, "busy_retries", 0) - n_busy,  # the planner busy with another goal
        ),
        waypoint=getattr(plan, "waypoint", None),
        guard_from=descent_index(plan),
    )
    if guard is not None:
        # from here on the box may not be exactly where it was detected —
        # every later FULL world allows for a contact-shifted container
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
    next_chain = state.chain + 1 if invalidates else state.chain
    return leg, PlanState(joints=end, chain=next_chain)


def _refuse_goal_inside(name, target, start_joints, end=None):
    """Raise where the goal of planned leg `name` would put the arm inside the body model
    (approach.goal_refusal: strictly, the rest pose excepted) — before the planner is asked, and
    again at the plan's end (`end`): flatbot's planner may plan in an empty world."""
    from box_opening.runtime.approach import goal_refusal

    why = goal_refusal(target, start_joints, end)
    if why is not None:
        raise RuntimeError(
            "%s refused before it was %s: its goal puts the arm inside %s"
            % (name, "flown" if end is not None else "sent to the planner", why)
        )


def as_home(leg):
    """`leg`, planned to a home, flown as every home flies (constants
    .HOME_SPEED, HOME_GAINS): cruising at HOME_SPEED, in the driver's joint
    impedance under MEDIUM whatever the run's own mode (Leg.compliant;
    runner._mode puts the client in it). Returns it."""
    leg.speed, leg.compliant, leg.gains = HOME_SPEED, True, HOME_GAINS
    return leg


def _gripper_leg(
    ctx,
    state,
    name,
    cmd,
    world,
    verify=None,
    defer_join=False,
    join_before_motion=False,
    send_with_previous_motion=False,
    send_when_still=False,
):
    world_name, world_path = world
    return Leg(
        name=name,
        kind=Kind.GRIPPER,
        traj=None,
        speed=0.0,
        guard=None,
        world=world_name,
        world_path=str(world_path),
        chain=state.chain,
        target=None,
        goal_joints=None,
        gripper_cmd=cmd,
        verify=verify,
        defer_join=defer_join,
        join_before_motion=join_before_motion,
        send_with_previous_motion=send_with_previous_motion,
        send_when_still=send_when_still,
    )


class Place:
    """Transit above the pose (+ lid-height margin — the planner cannot
    model a held object), guarded descent where a trip = set-down, then
    open the gripper (spec §5, §6)."""

    def __init__(self, target_xyz, quat, name="place", speed=None, touch_nm=None, band=None):
        self.target_xyz = list(target_xyz)
        self.quat = list(quat)
        self.name = name
        # grip band to re-check at the bottom, right before the release:
        # the LAST moment at which still holding the thing is checkable,
        # and the moment it matters. The lift used to carry this check,
        # which cost a dead stop between the lift and the carry every run
        # (they are otherwise one continuous motion).
        self.band = None if band is None else tuple(band)
        # descent speed; None keeps the conservative contact default for
        # callers that predate the config knob (tests, isolated CLI use)
        self.speed = CONTACT_SPEED if speed is None else float(speed)
        # set-down trip threshold; None keeps the model's press threshold.
        # A lid touching a table loads the wrist far less than a press
        # pops a seal — at 7.0 the guard stayed blind through a 10 mm
        # crunch at the slid drop spot (field 2026-09-02).
        self.touch_nm = touch_nm

    @staticmethod
    def hover_for(ctx, target_xyz):
        """The carry pose above a set-down target: hover standoff plus a
        lid height, raised to the carry floor — the carried lid hangs a
        lid-height below the fingertips and the planner cannot see it, so
        the carry must clear the container body even directly overhead."""
        m = ctx.model
        carry_floor = ctx.cpose.xyz[2] + m.dims[2] + m.lid_dims[2] + CARRY_CLEAR_M
        z = max(target_xyz[2] + m.hover_standoff + m.lid_dims[2], tcp_z(carry_floor))
        return [target_xyz[0], target_xyz[1], z]

    def plan(self, ctx, state):
        m = ctx.model
        hover = self.hover_for(ctx, self.target_xyz)
        full = _full_world(ctx)
        transit, state = _plan_motion(
            ctx,
            state,
            self.name + ":transit",
            ("pose", hover, self.quat),
            full,
            TRANSIT_SPEED,
        )
        world = _interaction_world(ctx, self.target_xyz[2], SETDOWN_OVERDRIVE_M, self.name)
        ctx.last_world = world
        guard = GuardSpec(
            touch_nm=m.touch_nm if self.touch_nm is None else float(self.touch_nm),
            trip="setdown",
            # contact is only possible at the stroke's very end — the
            # fast segment's dynamics must not trip the gentler set-down
            # threshold (lid released 110 mm up, field 2026-09-02)
            arm_after=0.5,
        )

        def verify(v):
            slipped = self.band is not None and v.gripper_pos is not None and not in_band(v.gripper_pos, self.band)
            if v.outcome == "touch":
                high = None if v.contact_xyz is None else float(v.contact_xyz[2]) - TIP_TO_TOOL_M - self.target_xyz[2]
                if high is not None and high > SETDOWN_HIGH_M:
                    return False, (
                        "struck something %.0f mm above where the lid meets "
                        "the table — set-down NOT confirmed, the lid is held" % (high * 1000)
                    )
                if v.progress is not None and v.progress < 0.5:
                    return False, (
                        "guard tripped at %.0f%% of the descent — struck "
                        "something on the way down, set-down NOT confirmed" % (v.progress * 100)
                    )
                if slipped:
                    return False, (
                        "felt a surface but the fingers read %.3f, outside "
                        "%s — the lid slipped on the way here and this is "
                        "the table, not the lid" % (v.gripper_pos, list(self.band))
                    )
                peak = "" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak
                return True, "surface felt%s — set down" % peak
            if v.outcome == "arrived":
                if slipped:
                    return False, (
                        "never felt the surface and the fingers read %.3f, "
                        "outside %s — the lid slipped" % (v.gripper_pos, list(self.band))
                    )
                return False, ("full stroke with no trip — never felt the surface, set-down NOT confirmed")
            return False, "set-down %s" % v.outcome

        # a set-down SUCCEEDS only on the touch: target exactly at surface
        # height can 'arrive' without ever feeling the table — command a
        # hair below so the guard verdict is deterministic
        down_xyz = [
            self.target_xyz[0],
            self.target_xyz[1],
            self.target_xyz[2] - SETDOWN_OVERDRIVE_M,
        ]
        descend, state = _plan_motion(
            ctx,
            state,
            self.name + ":down",
            # vertical final 50 mm: every free plan bows a little (the
            # "small arch", field 2026-09-01); a set-down comes straight
            # down onto its spot
            ("pose", down_xyz, self.quat, 0.05),
            world,
            self.speed,
            guard=guard,
            invalidates=True,
            verify=verify,
        )
        release = _gripper_leg(
            ctx,
            state,
            self.name + ":open",
            GRIPPER_CMD_OPEN,
            world,
            # dispatched at once; the retreat's post-touch replan runs while
            # the fingers open and the join lands right before that retreat
            # executes — a release still completes before the arm moves
            # away (audit 2026-09-02)
            defer_join=True,
            join_before_motion=True,
        )
        return [transit, descend, release], state


class LiftFree:
    """A plan-free STRAIGHT vertical move from the live joints: the arm's own
    kinematics (runtime/kinematics), no planner. The recovery the bringup
    guide used to call "jog clear by hand first": a pose the planner
    refuses as a start (fingers at the table after an abort or a jog) is
    lifted straight up — away from the only thing it can be in — until the
    planner accepts it. Guarded as an obstruction trip: anything met on
    the way UP is a failure, never pushed through. The attitude is held."""

    def __init__(self, dz, name="lift-free", speed=None, touch_nm=4.0):
        self.dz = float(dz)
        self.name = name
        self.speed = CONTACT_SPEED if speed is None else float(speed)
        self.touch_nm = float(touch_nm)

    def plan(self, ctx, state):
        from box_opening.runtime import approach
        from box_opening.runtime.approach import _chain, line_trajectory
        from box_opening.runtime.kinematics import mat_to_quat_xyzw

        # "vertical" is the chain root's z: on flatbot base_link leans 30 deg
        # from the gravity-aligned arm_mount the planner works in, and a lift
        # solved there moved the fingers 40 mm sideways for 69 mm up
        if approach._ROOT != ctx.worlds.base_frame:
            raise RuntimeError(
                "plan-free lift refused: chain root %s != planning frame %s" % (approach._ROOT, ctx.worlds.base_frame)
            )
        chain = _chain()
        q0 = list(state.joints if state.joints is not None else ctx.client.joints())
        R, t = chain.fk(q0)
        quat = mat_to_quat_xyzw(R)
        end_xyz = [float(t[0]), float(t[1]), float(t[2]) + self.dz]
        pts, why = chain.straight_line(q0, end_xyz, quat)
        if pts is None:
            raise RuntimeError("plan-free lift refused: %s" % why)
        world_name, world_path = ctx.worlds.push_name("bench", model=None, tag="bare")
        guard = GuardSpec(touch_nm=self.touch_nm, trip="obstruction")
        leg = Leg(
            name=self.name,
            kind=Kind.MOTION,
            traj=line_trajectory(ctx.client.joint_names, pts),
            speed=self.speed,
            guard=guard,
            world=world_name,
            world_path=str(world_path),
            chain=state.chain,
            target=("pose", end_xyz, list(quat)),
            goal_joints=[float(v) for v in pts[-1]],
            plan_s=0.0,
        )
        ctx.last_pose = (end_xyz, list(quat))
        return [leg], PlanState(joints=list(leg.goal_joints), chain=state.chain + 1)


# ClearMarginFree: a joint this near its limit is too near for the planner
# to start from. It refuses a start inside its margin, and its finetune
# trajopt (bound activation_distance 0.05 beyond that) failed every HOME from
# 0.120-0.154 rad of room and planned it from 0.161 (2026-10-23)
CLEAR_MARGIN_ROOM_RAD = PLANNER_LIMIT_MARGIN_RAD + 0.08
# the shortest step leaves every joint this far from its limits, just
# outside the margin; each next step moves the largest joint this much more
CLEAR_MARGIN_FIRST_RAD = PLANNER_LIMIT_MARGIN_RAD + 0.02
CLEAR_MARGIN_STRIDE_RAD = 0.05
# it is plan-free, so it stays a step: no joint moves more than this
CLEAR_MARGIN_MAX_STEP_RAD = 0.35
# and the tool never goes lower than its start by more than this on the way
# (the table and the box are below it)
CLEAR_MARGIN_DROP_M = 0.005
# The step's guard watches its first this-much of the step without tripping
# and takes its baseline there. A pose held long at rest reads far under
# the arm's weight (2026-10-23: arm_joint_4 4.6 of the 9.3 Nm the gravity
# model holds there, arm_joint_6 0.6 of 2.2: a support or the joints'
# friction carrying the rest), and the weight comes back onto the joints as
# the step starts (arm_joint_4 at 8.9 Nm by 9% of it): judged from rest that
# read as a 4.3 Nm strike, 1 mm into a lift.
CLEAR_MARGIN_SETTLE = 0.2


class ClearMarginFree:
    """Plan-free steps along the straight joint line toward `goal`, for a
    start with a joint too near its limit for the planner to plan from.

    rammp-curobo refuses a start inside its joint-limit margin ("jog it away
    from the limit first"), and its retiming fails from starts just outside
    it, and from some farther out while the wrist is still folded:
    2026-10-23, arm_joint_4 0.047 rad from -2.57 and the tool 268 mm above
    arm_mount, HOME was refused from that joint jogged alone to 0.18-0.26
    rad of room, and planned from 45% of the way along the joint line. So
    steps() offers steps shortest first, from the one that just clears the
    margin, and the caller plans from each until one plans. Along the line
    each joint's room never falls below the smaller of its two ends (the
    goal's is clear); a step is refused when the tool would drop on the way.
    Guarded as an obstruction trip, compensated for the arm's weight, from a
    baseline taken once the step is under way (CLEAR_MARGIN_SETTLE)."""

    def __init__(self, goal, name="clear-margin", speed=None, touch_nm=4.0):
        self.goal = [float(v) for v in goal]
        self.name = name
        self.speed = CONTACT_SPEED if speed is None else float(speed)
        self.touch_nm = float(touch_nm)

    @staticmethod
    def _limited():
        from box_opening.runtime.approach import _chain

        # continuous joints are left out: the planner's +-2 pi is never near
        return [(j, lo, hi) for j, (lo, hi) in enumerate(_chain().limits) if lo is not None and hi is not None]

    def near(self, ctx, q):
        """[(joint name, room)] for each joint too near its limit for the
        planner; [] when none is, or when there is no URDF to read the
        limits from (the planner's own refusal stands, as before)."""
        from box_opening.runtime import approach

        if not approach.chain_ready()[0]:
            return []
        names = list(ctx.client.joint_names)
        rooms = [(names[j], float(min(q[j] - lo, hi - q[j]))) for j, lo, hi in self._limited()]
        return [(n, r) for n, r in rooms if r < CLEAR_MARGIN_ROOM_RAD]

    def steps(self, ctx, state):
        """Yield (legs, PlanState, info) per step, shortest first: no legs
        for a start already outside the margin, then one plan-free leg each.
        info: {"fraction", "largest_rad", "tool_move" (m, planning frame)}."""
        import math

        import numpy as np

        from box_opening.runtime import approach
        from box_opening.runtime.approach import _chain, line_trajectory

        # the drop is the chain root's z: it must be the planning frame's
        # (on flatbot base_link leans 30 deg from the gravity-aligned arm_mount)
        if approach._ROOT != ctx.worlds.base_frame:
            raise RuntimeError(
                "%s refused: chain root %s != planning frame %s" % (self.name, approach._ROOT, ctx.worlds.base_frame)
            )
        from box_opening.runtime.branches import on_branch
        from box_opening.runtime.kinematics import body_refusal

        chain = _chain()
        names = list(ctx.client.joint_names)
        q0 = np.asarray(state.joints if state.joints is not None else ctx.client.joints(), float)
        # each continuous joint's goal the short way round from where it
        # stands: 2026-10-30 joint_3 stood at +3.097 and HOME's is -3.137,
        # 0.049 rad apart, and the line taken literally turned it 6.23 rad,
        # so the first step that cleared joint_4's limit was a 2.49 rad one
        # and none up to the cap was offered (the driver too goes the short
        # way round, and the planner is handed HOME's branch: planner_branch)
        goal = np.asarray(on_branch(self.goal, q0), float)
        limited = self._limited()
        first = 0.0
        for j, lo, hi in limited:
            if min(goal[j] - lo, hi - goal[j]) < CLEAR_MARGIN_ROOM_RAD:
                raise RuntimeError(
                    "%s refused: the goal's %s is %.3f rad from its limit, too near for the planner"
                    % (self.name, names[j], min(goal[j] - lo, hi - goal[j]))
                )
            # the room to each limit is linear along the line and clear at its end
            for f0, f1 in ((q0[j] - lo, goal[j] - lo), (hi - q0[j], hi - goal[j])):
                if f0 < CLEAR_MARGIN_FIRST_RAD:
                    first = max(first, (CLEAR_MARGIN_FIRST_RAD - f0) / (f1 - f0))
        span = float(np.max(np.abs(goal - q0)))
        if first == 0.0:
            yield [], state, {"fraction": 0.0, "largest_rad": 0.0, "tool_move": np.zeros(3)}
            if span == 0.0:
                return
        k = 0
        while True:
            frac = min(1.0, first + k * CLEAR_MARGIN_STRIDE_RAD / span)
            k += 1
            if frac == 0.0:
                continue
            largest = frac * span
            if largest > CLEAR_MARGIN_MAX_STEP_RAD + 1e-9:
                return
            q1 = q0 + frac * (goal - q0)
            n = max(2, int(math.ceil(largest / 0.01)))
            pts = [q0 + (i / n) * (q1 - q0) for i in range(n + 1)]
            tool = [chain.fk(q)[1] for q in pts]
            drop = float(tool[0][2] - min(t[2] for t in tool))
            if drop > CLEAR_MARGIN_DROP_M:
                # a longer step runs the same line: it drops as much or more
                raise RuntimeError(
                    "%s refused: the joint line toward the goal drops the tool %.0f mm — jog the "
                    "arm clear by hand first" % (self.name, 1000 * drop)
                )
            body = body_refusal(chain, pts)
            if body is not None:
                # a longer step runs the same line on: it goes in as far or further
                raise RuntimeError(
                    "%s refused: the joint line toward the goal enters %s — jog the arm clear by hand first"
                    % (self.name, body)
                )
            world_name, world_path = ctx.worlds.push_name("bench", model=None, tag="bare")
            leg = Leg(
                name=self.name,
                kind=Kind.MOTION,
                traj=line_trajectory(ctx.client.joint_names, [[float(v) for v in q] for q in pts]),
                speed=self.speed,
                # the arm's weight changes along the step: compensated, from
                # a baseline past its start (CLEAR_MARGIN_SETTLE)
                guard=GuardSpec(
                    touch_nm=self.touch_nm,
                    trip="obstruction",
                    gravity=approach._gravity().wrist_holding,
                    rebaseline_after=CLEAR_MARGIN_SETTLE,
                    arm_after=CLEAR_MARGIN_SETTLE,
                ),
                world=world_name,
                world_path=str(world_path),
                chain=state.chain,
                target=("joints", [float(v) for v in q1]),
                goal_joints=[float(v) for v in q1],
                plan_s=0.0,
            )
            info = {"fraction": float(frac), "largest_rad": float(largest), "tool_move": tool[-1] - tool[0]}
            yield [leg], PlanState(joints=list(leg.goal_joints), chain=state.chain + 1), info
            if frac >= 1.0:
                return


# EscapeBodyFree: a start the planner refuses because the arm stands inside a padded box of the chair or its user
# (rammp-curobo: INVALID_START_STATE_WORLD_COLLISION) though it is clear of the boxes themselves: left near the user
# by a stop, a jog or the other arm configurations a recovery may find. 2026-11-09, the arm 8 in forward on
# flatbot's chair (its model's boxes moved back 0.2032 m): of 101 such starts, the planner refused all 101; this
# step got 82 of them out by 10 mm, the hand held within the drop below, and HOME planned from where it ended.
ESCAPE_CLEAR_M = 0.010  # out of every padded box by this much: HOME is planned from there
ESCAPE_STEP_RAD = 0.02  # the step along the way out, its largest joint
ESCAPE_MAX_RAD = 0.6  # plan-free, so it stays a step: no joint moves more than this
ESCAPE_DROP_M = CLEAR_MARGIN_DROP_M  # nor the tool lower than its start by more than this (the table is below)
ESCAPE_MAX_STEPS = 200
# where no step gets further and the arm is already out of the padding by this much, the way out ends there
# (2026-11-09 live check: 11 of the 23 refused had come out by 0-9 mm)
ESCAPE_OUT_M = 0.001
# Owner 2026-11-10: "if the arm is close or in an obstacle, it slowly, and on max impedance ['the lightest it can
# be'], moves to the closest valid point outside of the obstacle, then goes home". That afternoon home_arm refused
# with the fingers 2 cm over the table, joint_4 0.107 rad from its limit and joint_1 folded 2.6 rad from HOME's: the
# clear-margin line toward HOME dropped the tool 46 mm (to 7.5 cm, through the table), and nothing else was tried.
# So the way out is the nearest point it can walk to that is clear of everything known, flown slowly and soft:
ESCAPE_NEAR_M = ESCAPE_CLEAR_M  # a start within this of the padding (or in it) counts as close: stepped clear first
ESCAPE_TABLE_M = 0.05  # with the table known (a run's measured one), the tool at least this far above it
ESCAPE_ROOM_RAD = CLEAR_MARGIN_ROOM_RAD  # and every revolute joint this far from its limits
ESCAPE_M_PER_RAD = 0.1  # a joint's missing room weighs this much against the tool's or a sphere's missing metres
ESCAPE_SPEED = 0.10  # slowly: a cruise fraction (Leg.speed), below CONTACT_SPEED
# the lightest the driver offers: its SOFT profile (Kq 40/15 N.m/rad, zeta 0.4, a 0.45 rad leash), a fifth of
# MEDIUM's stiffness on joints 1-4, where a home flies. A custom spec could go to Kq 1, but nothing that soft has
# flown, and an arm that soft may not follow the step at all. Flown compliant, nothing the run reads stops it (the
# client's guards only watch): the arm gives way where it meets something; the e-stop and Ctrl+C remain.
ESCAPE_GAINS = GAINS_SOFT


class EscapeBodyFree:
    """A plan-free step to the nearest point clear of everything known, for an arm in or close to an obstacle:
    every sphere ESCAPE_CLEAR_M outside the chair and user's PADDED boxes (Body.sphere_depths), the tool
    ESCAPE_TABLE_M above the table where one is known, and every revolute joint ESCAPE_ROOM_RAD from its limits.
    Walked in joint space ESCAPE_STEP_RAD at a time down what is still missing (the deepest sphere's depth, the
    tool's height under the table's clearance, each short joint's room), never deeper in the padding than it
    stands (a path that starts inside may leave, not go in further: Body.first_hit's start rule) nor into it from
    outside, the tool never lower than ESCAPE_DROP_M under its start (its downward pull projected out where it
    would go lower), no joint past its limit. Flown slowly (ESCAPE_SPEED) in the driver's joint impedance at its
    lightest (ESCAPE_GAINS, SOFT): a joint line through the way out that gives way where it meets something.
    Refused (RuntimeError) where no way of up to ESCAPE_MAX_RAD a joint gets there: the operator jogs the arm
    clear by hand."""

    def __init__(self, name="escape-body", speed=None, touch_nm=4.0):
        self.name = name
        self.speed = ESCAPE_SPEED if speed is None else float(speed)
        self.touch_nm = float(touch_nm)

    @staticmethod
    def _body():
        from box_opening.runtime import approach

        if not approach.chain_ready()[0]:
            return None, None
        chain = approach._chain()
        return chain, getattr(chain, "body", None)

    def needed(self, q, table_z=None):
        """Why the arm at `q` is in or close to an obstacle: inside the body model's padding or within
        ESCAPE_NEAR_M of it, or the tool within ESCAPE_TABLE_M of the table `table_z` (None: none known); '' where
        it is clear of both (or no arm chain is known)."""
        import numpy as np

        chain, body = self._body()
        if chain is None:
            return ""
        why = []
        if body is not None:
            d = float(body.sphere_depths(np.atleast_2d(np.asarray(q, float)))[0].max())
            if d > 0.0:
                why.append("stands %.0f mm inside the chair and user's padding" % (1000 * d))
            elif d > -ESCAPE_NEAR_M:
                why.append("stands within %.0f mm of the chair and user's padding" % (1000 * -d))
        if table_z is not None:
            over = float(chain.fk(list(q))[1][2]) - float(table_z)
            if over < ESCAPE_TABLE_M:
                why.append("holds the tool %.0f mm over the table" % (1000 * over))
        return " and ".join(why)

    def plan(self, ctx, state, table_z=None, over_m=None):
        """(legs, PlanState, info) of the way to the nearest point clear of everything known from `state`, `table_z`
        the table's height (None: none known), the tool `over_m` above it (None: ESCAPE_TABLE_M); no legs where
        it is clear already. info: {"steps", "largest_rad",
        "depth_mm" (the start's, > 0 in the padding), "tool_drop_mm", "tool_rise_mm"}. Raises RuntimeError with
        why where there is none."""
        import numpy as np

        from box_opening.runtime.approach import line_trajectory
        from box_opening.runtime.kinematics import body_refusal

        chain, body = self._body()
        if chain is None:
            raise RuntimeError("%s refused: no arm chain (no URDF): nothing to step clear with" % self.name)
        q0 = np.asarray(state.joints if state.joints is not None else ctx.client.joints(), float)
        n = len(q0)
        lo = np.full(n, -np.inf)  # the hard limits, never crossed
        hi = np.full(n, np.inf)
        limited = []
        for j, (a, b) in enumerate(chain.limits):
            if a is not None and b is not None:
                lo[j], hi[j] = a + 0.005, b - 0.005
                limited.append((j, a, b))

        def depth(Q):
            Q = np.atleast_2d(Q)
            if body is None:
                return np.full(len(Q), -np.inf)
            return body.sphere_depths(Q)[0].max(axis=1)

        def tool_z(q):
            return float(chain.fk(list(q))[1][2])

        def missing(q):
            """(merit, [sphere m, table m, [joint rad ...]]): what the point `q` still lacks."""
            vp = max(0.0, float(depth(q)[0]) + ESCAPE_CLEAR_M) if body is not None else 0.0
            vt = max(0.0, float(table_z) + over - tool_z(q)) if table_z is not None else 0.0
            vl = [max(0.0, ESCAPE_ROOM_RAD - min(q[j] - a, b - q[j])) for j, a, b in limited]
            return vp + vt + ESCAPE_M_PER_RAD * sum(vl), (vp, vt, vl)

        over = ESCAPE_TABLE_M if over_m is None else float(over_m)
        z0 = tool_z(q0)
        q, d = q0.copy(), float(depth(q0)[0])
        start_mm = 1000 * d if body is not None else float("-inf")
        pts = [q0.copy()]
        eye = np.eye(n) * 1e-4

        def grad(f):
            return np.array([(f(q + eye[i]) - f(q - eye[i])) / 2e-4 for i in range(n)])

        for _k in range(ESCAPE_MAX_STEPS):
            m, (vp, vt, vl) = missing(q)
            if m <= 0.0:
                break
            if float(np.max(np.abs(q - q0))) > ESCAPE_MAX_RAD:
                raise RuntimeError(
                    "%s refused: no way clear within %.2f rad a joint (still short: %s) — jog the arm clear by hand"
                    % (self.name, ESCAPE_MAX_RAD, _short_of(vp, vt, vl, limited))
                )
            way = np.zeros(n)
            if vp > 0.0:
                g = grad(lambda x: float(depth(x)[0]))
                way -= g / max(float(np.max(np.abs(g))), 1e-12)
            zg = grad(tool_z)
            if vt > 0.0:
                way += zg / max(float(np.max(np.abs(zg))), 1e-12)
            for (j, a, b), v in zip(limited, vl, strict=True):
                if v > 0.0:
                    way[j] += 1.0 if q[j] - a < b - q[j] else -1.0
            if tool_z(q) - z0 <= -ESCAPE_DROP_M + 0.001 and way @ zg < 0.0 and zg @ zg > 0.0:
                way = way - (way @ zg) / (zg @ zg) * zg  # at the floor of the drop: nothing downward
            if float(np.max(np.abs(way))) < 1e-9:
                raise RuntimeError(
                    "%s refused: no direction clear (short: %s) — jog the arm clear by hand"
                    % (self.name, _short_of(vp, vt, vl, limited))
                )
            dq = way / float(np.max(np.abs(way))) * ESCAPE_STEP_RAD
            for _half in range(6):
                qn = np.clip(q + dq, lo, hi)
                dn = float(depth(qn)[0])
                if (
                    missing(qn)[0] < m - 1e-9
                    and dn <= max(d, -ESCAPE_CLEAR_M) + 1e-6  # never deeper, nor in from outside
                    and tool_z(qn) >= z0 - ESCAPE_DROP_M
                ):
                    break
                dq = dq / 2.0
            else:
                rooms = [min(q[j] - a, b - q[j]) for j, a, b in limited]
                if d <= -ESCAPE_OUT_M and min(rooms, default=np.inf) >= CLEAR_MARGIN_FIRST_RAD:
                    break  # out of the padding, off the limits, short of the margins: HOME is asked for from here
                raise RuntimeError(
                    "%s refused: every way clear goes deeper into the padding, lowers the tool more than %.0f mm or "
                    "meets a joint limit (short: %s) — jog the arm clear by hand"
                    % (self.name, 1000 * ESCAPE_DROP_M, _short_of(vp, vt, vl, limited))
                )
            q, d = qn, dn
            pts.append(q.copy())
        else:
            raise RuntimeError("%s refused: not clear in %d steps" % (self.name, ESCAPE_MAX_STEPS))
        z_all = [tool_z(p) for p in pts]
        info = {
            "steps": len(pts) - 1,
            "largest_rad": float(np.max(np.abs(q - q0))),
            "depth_mm": start_mm,
            "tool_drop_mm": 1000 * max(0.0, z0 - min(z_all)),
            "tool_rise_mm": 1000 * (z_all[-1] - z0),
        }
        if len(pts) == 1:
            return [], state, info
        why = body_refusal(chain, pts) if body is not None else None
        if why is not None:
            raise RuntimeError("%s refused: the way clear enters %s — jog the arm clear by hand" % (self.name, why))
        world_name, world_path = ctx.worlds.push_name("bench", model=None, tag="bare")
        leg = Leg(
            name=self.name,
            kind=Kind.MOTION,
            traj=line_trajectory(ctx.client.joint_names, [[float(v) for v in p] for p in pts]),
            speed=self.speed,
            guard=None,  # flown compliant: no guard would stop it (the client's guards only watch); the arm gives way
            world=world_name,
            world_path=str(world_path),
            chain=state.chain,
            target=("joints", [float(v) for v in q]),
            goal_joints=[float(v) for v in q],
            plan_s=0.0,
            gains=ESCAPE_GAINS,
            compliant=True,
        )
        return [leg], PlanState(joints=list(leg.goal_joints), chain=state.chain + 1), info


def _short_of(vp, vt, vl, limited):
    """What a point still lacks, in words: the padding's margin, the table's clearance, each short joint's room."""
    out = []
    if vp > 0:
        out.append("%.0f mm of the padding's margin" % (1000 * vp))
    if vt > 0:
        out.append("%.0f mm over the table" % (1000 * vt))
    out += ["%.3f rad of joint_%d's room" % (v, j + 1) for (j, _a, _b), v in zip(limited, vl, strict=True) if v > 0]
    return ", ".join(out) or "nothing"


# home_any: HOME from wherever the arm stands. 2026-11-09 (the temporary HOME, the arm 8 in forward), the live
# planner asked for HOME from 221 random starts clear of the chair and user's padding planned 203 at once; of the
# 18 it did not (16 a plan the client refused for ending at another arm with HOME's tool pose, 2 FINETUNE_TRAJOPT_FAIL),
# 3 planned when asked again and 10 through a waypoint along the joint line toward HOME.
HOME_TRIES = 3  # HOME asked for this many times from one start before the waypoints: the planner is stochastic
HOME_WAYPOINTS = (0.5, 0.25, 0.75)  # fractions of the joint line toward HOME, each continuous joint the short way


class PlanSwung(RuntimeError):
    """A plan that the runner's trajectory sanity gate would refuse at execution (guards.sanity_violations,
    SANITY_MARGIN_RAD): a joint swings wider than the straight way needs. 2026-11-10 on flatbot every plan HOME
    from [1.006, -0.312, 2.699, -2.224, -2.790, -0.889, -0.623], and from a 4% clear-margin step near there, took
    joint_1 out to 1.87 rad and back (1.15-1.45 rad for a 0.56 change, the hand toward the user): home_arm was
    REFUSED at execution 3 times. Asked again, the planner answered alike; through the joint line's 50% waypoint both
    legs passed the gate (plans only)."""


class PlanUnderTable(RuntimeError):
    """A plan that takes the arm under a table the run knows about: the planner's world holds no table
    (worlds.py), so a HOME it plans may sweep through one. 2026-11-11, plans only on the robot's planner: 24 of
    344 homes from poses over a 29 in table went under its top (the tool up to 143 mm, the arm 211 mm), every
    one from joint_1 >= 1.2 rad off HOME, and both poses the arm was left in on 2026-11-10 did (the tool to
    z 0.211-0.215, that day's table at 0.2656)."""


# a planned leg keeps every sphere's bottom (the tool frame, with no body model) this far over the table
HOME_TABLE_M = 0.0  # the spheres are buffered already (robot.yaml's collision_sphere_buffer)
HOME_TABLE_DIP_M = 0.010  # ... or no more than this under where that sphere starts / the lower of its two ends
HOME_TABLE_STEP_RAD = 0.03  # the leg is checked at joint steps no longer than this
HOME_LIFTS_M = (0.08, 0.12, 0.16)  # straight lifts tried, lowest first, where HOME plans under the table
# ... then the soft step (EscapeBodyFree) up to the tool this far over the table: where the straight lift meets a
# joint limit or has no IK down its line (2026-11-11: 45 of 67 refused plans), a walk in joint space still rises
HOME_RISES_M = (0.15, 0.25)
# no further way up is started once the ways up have taken this long to plan (2026-11-11: a refusal after all five,
# each with its waypoints, took 10.7 s at the median and 32 s at worst; every plan that homed took under 15 s)
HOME_RAISE_S = 12.0


def under_table(chain, body, pts, table_z, extent=None, rest=None):
    """Why the joint path `pts` goes under the table at `table_z`, or None. Each of the arm's checked spheres
    (Body.spheres_at; the tool frame alone with no body model) is followed at steps of HOME_TABLE_STEP_RAD. A
    sphere whose bottom is under table_z + HOME_TABLE_M is under the table where
      * it stands over the table the scene camera saw (`extent`, (x0, x1, y0, y1)) and more than
        HOME_TABLE_DIP_M lower than it started (a start that low already may leave, not go lower) — unless the
        path's END has a sphere that low inside the extent too: the extent is then not the table's (HOME stands
        behind the table's edge) and only the next rule holds;
      * or it is more than HOME_TABLE_DIP_M under the lower of its heights at the path's two ends: the table's
        edge is not known beyond what the camera saw, and a path that dips under both its ends over a table it
        does not know is not flown.
    No table stands where the arm rests: with `rest` (HOME's joints), a sphere no farther from the arm's base
    axis than the farthest edge of a sphere `rest` itself holds under the table's top is not judged (2026-11-11:
    HOME's own upper arm stands 35 mm under a 29 in table's top 0.14 m from the base, and five clear poses were
    refused for that link 2-8 mm under the top beside it)."""
    import numpy as np

    P = [np.asarray(p, float) for p in pts]
    if len(P) < 2:
        return None
    Q = [P[0]]
    for a, b in zip(P, P[1:], strict=False):
        n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / HOME_TABLE_STEP_RAD)))
        Q += [a + (b - a) * (k / n) for k in range(1, n + 1)]

    def at(q):
        if body is not None:
            C, r, links = body.spheres_at(q)
            C, r = np.asarray(C, float), np.asarray(r, float)
            return C[:, :2], C[:, 2] - r, list(links), r
        t = np.asarray(chain.fk(list(q))[1], float)
        return t[None, :2], t[None, 2], ["the tool"], np.zeros(1)

    floor = float(table_z) + HOME_TABLE_M
    xy0, b0, links, _ = at(Q[0])
    xy1, b1, _, _ = at(Q[-1])
    r_rest = 0.0  # no table within this of the arm's base axis: the rest pose stands under its top there
    if rest is not None:
        xyr, br, _, rr = at(np.asarray(rest, float))
        if bool(np.any(br < floor)):
            r_rest = float(np.max((np.hypot(xyr[:, 0], xyr[:, 1]) + rr)[br < floor]))

    def over(xy):
        if extent is None:
            return np.zeros(len(xy), bool)
        x0, x1, y0, y1 = extent
        return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)

    seen = extent is not None and not bool(np.any(over(xy1) & (b1 < floor)))
    ends = np.minimum(b0, b1)
    worst = None
    for q in Q[1:-1]:
        xy, b, _, _ = at(q)
        bad = (b < floor) & (b < ends - HOME_TABLE_DIP_M)
        if seen:
            bad |= (b < floor) & over(xy) & (b < b0 - HOME_TABLE_DIP_M)
        bad &= np.hypot(xy[:, 0], xy[:, 1]) > r_rest
        if bool(np.any(bad)):
            i = int(np.argmin(np.where(bad, b, np.inf)))
            if worst is None or b[i] < worst[0]:
                worst = (float(b[i]), links[i], float(xy[i][0]), float(xy[i][1]))
    if worst is None:
        return None
    return "%s %.0f mm under the table's top (z %.3f) at x %.2f, y %.2f" % (
        worst[1],
        1000 * (float(table_z) - worst[0]),
        float(table_z),
        worst[2],
        worst[3],
    )


def home_any(
    ctx,
    state,
    goal,
    plan_to,
    say=log.info,
    who="[home]",
    goal_words="HOME",
    plan_words="HOME",
    step_tail="",
    ask_again=True,
):
    """The legs from `state` to the joint goal `goal` (HOME): plan_to(state, name, q) plans one leg (raising
    RuntimeError, or PlanRefused, where the planner or the client refuses). In order:
      1. an arm in or close to an obstacle (inside the body model's padding or within ESCAPE_NEAR_M of it, the
         tool within ESCAPE_TABLE_M of a known table) steps slowly, in the driver's lightest joint impedance, to
         the nearest point clear of everything known, off the joint limits too (EscapeBodyFree);
      2. a joint too near its limit for the planner steps along the joint line first (ClearMarginFree), the
         shortest step the goal plans from, each step's plan asked for once (as since 2026-10-23), or through a
         waypoint from the step's end where that plan swings wide; where none does (the line drops the tool, say,
         or no step up to the cap plans), the soft step of 1. instead, and the goal asked for from its end;
      3. otherwise the goal asked for up to HOME_TRIES times, stopping at the first plan that swings wide;
      4. then a waypoint HOME_WAYPOINTS of the way along the joint line, clear of the padding, planned to, and the
         goal planned from it.
    Every leg passes the runner's trajectory sanity gate here, at planning (PlanSwung), not at execution, and
    is kept over a table the run knows (under_table, PlanUnderTable: the planner's world holds none): where the
    goal planned directly goes under it, the goal is planned from the top of a straight lift (HOME_LIFTS_M,
    LiftFree) or of the soft step higher over the table (HOME_RISES_M), then through the waypoints, each way
    judged as one path from the start (a waypoint under the table is under the table); no further way up is
    started after HOME_RAISE_S of planning them. No table is taken to stand where the goal itself rests under
    its top (under_table's `rest`).
    The caller's words name the goal in what is said (`goal_words`: "HOME", "the rest pose"; `plan_words`: what
    is planned, "HOME", "home"; `step_tail` after a clear-margin step's note). ask_again False (a run's own
    home, whose caller has its own answer to a home the planner refuses: a back-off, the next way out): the goal
    is asked for once and the planner's refusal raised as it is; a plan that swings wide or goes under the table
    still goes through the lift and the waypoints. Raises RuntimeError with the last
    refusal where none plans."""
    import numpy as np

    from box_opening.runtime.branches import on_branch

    legs = []
    cut = []  # said in the refusal where the ways up ran out of planning time (HOME_RAISE_S)
    esc = EscapeBodyFree()
    table_z = getattr(getattr(ctx, "worlds", None), "table_top_z", None)
    table_xy = getattr(getattr(ctx, "worlds", None), "table_extent", None)
    tchain, tbody = esc._body()

    def planned(at, name, q):
        """plan_to's leg, the runner's sanity gate applied now (PlanSwung) rather than at execution, and the leg
        kept over a table the run knows (PlanUnderTable: the planner's world holds none)."""
        leg = plan_to(at, name, q)
        traj = getattr(leg, "traj", None)
        bad = sanity_violations(traj, SANITY_MARGIN_RAD) if traj is not None else []
        if bad:
            raise PlanSwung("%s swings wide (%s): the runner's sanity gate would refuse it" % (name, "; ".join(bad)))
        if traj is not None and table_z is not None and tchain is not None:
            low = under_table(tchain, tbody, [list(p.positions) for p in traj.points], table_z, table_xy, goal)
            if low:
                raise PlanUnderTable("%s goes under the table (%s): the planner's world holds no table" % (name, low))
        return leg

    def over_the_table(q_from, some, what):
        """`some` legs flown one after another from `q_from`, kept over the table as ONE path: a waypoint between
        two legs is an end of both, and judged leg by leg a waypoint that is itself under the table passed
        (2026-11-11: six poses homed under the table through one, the tool up to 163 mm under)."""
        if table_z is None or tchain is None:
            return
        pts = [list(q_from)]
        for leg in some:
            traj = getattr(leg, "traj", None)
            pts += [list(p.positions) for p in traj.points] if traj is not None else [list(leg.goal_joints)]
        low = under_table(tchain, tbody, pts, table_z, table_xy, goal)
        if low:
            raise PlanUnderTable("%s goes under the table (%s): the planner's world holds no table" % (what, low))

    def via_waypoint(at, last):
        """([waypoint leg, goal leg], f, last) from `at` through the first waypoint that plans, or (None, None,
        last)."""
        q0 = np.asarray(at.joints, float)
        g = np.asarray(on_branch(goal, q0), float)
        for f in HOME_WAYPOINTS:
            w = [float(v) for v in q0 + f * (g - q0)]
            if tbody is not None and not tbody.pose_clear(w):
                continue
            try:
                a = planned(at, "home-waypoint", w)
                b = planned(PlanState(joints=list(a.goal_joints), chain=at.chain), "home", list(goal))
                over_the_table(q0, [a, b], "home through the waypoint %.0f%% of the way" % (100 * f))
            except RuntimeError as e:
                last = e
                continue
            return [a, b], f, last
        return None, None, last

    def raised(at, last):
        """(legs, words, last) from `at`, for a goal that planned from here goes under the table: up first, by a
        straight lift (LiftFree, obstruction-guarded, HOME_LIFTS_M) or, where none does, the soft step up to the
        tool HOME_RISES_M over the table (EscapeBodyFree), the lowest the goal plans from the top of — directly
        or through a waypoint — clear of the padding and over the table all the way; or (None, None, last)."""
        from box_opening.runtime.kinematics import body_refusal

        was = getattr(ctx, "last_pose", None)
        t0 = time.monotonic()
        for kind, m in [("lift", dz) for dz in HOME_LIFTS_M] + [("rise", m) for m in HOME_RISES_M]:
            ctx.last_pose = was  # LiftFree sets it: a lift not taken leaves no trace
            if time.monotonic() - t0 > HOME_RAISE_S:
                cut.append(
                    " (the ways up had taken %.0f s to plan: the %s %.0f mm and any after it were not tried)"
                    % (HOME_RAISE_S, "lift of" if kind == "lift" else "soft step up to", 1000 * m)
                )
                break
            try:
                if kind == "lift":
                    up, top = LiftFree(m, name="home-lift").plan(ctx, at)
                    if tbody is not None:
                        hit = body_refusal(tchain, [list(p.positions) for p in up[0].traj.points])
                        if hit is not None:
                            raise RuntimeError("the lift of %.0f mm enters %s" % (1000 * m, hit))
                    words = "a straight lift of %.0f mm (obstruction-guarded)" % (1000 * m)
                else:
                    up, top, _info = EscapeBodyFree(name="home-rise").plan(ctx, at, table_z, over_m=m)
                    if not up:
                        raise RuntimeError("the tool stands %.0f mm over the table already" % (1000 * m))
                    words = "a slow soft step up to the tool %.0f mm over the table (%s; it gives way)" % (
                        1000 * m,
                        IMPEDANCE_NAMES.get(ESCAPE_GAINS, ESCAPE_GAINS),
                    )
            except RuntimeError as e:
                last = e
                continue
            try:
                rest = [planned(top, "home", list(goal))]
                over_the_table(at.joints, up + rest, "home from the top of %s" % words)
            except RuntimeError as e:
                rest, f, last = via_waypoint(top, e)
                if rest is None:
                    continue
                try:
                    over_the_table(at.joints, up + rest, "home from the top of %s" % words)
                except RuntimeError as e2:
                    last = e2
                    continue
                words += ", then through a waypoint %.0f%% of the way along the joint line" % (100 * f)
            return up + rest, words, last
        ctx.last_pose = was
        return None, None, last

    def step_clear(st, why):
        """The soft step to the nearest point clear of everything known from `st`, said: (legs, PlanState)."""
        step, st, info = esc.plan(ctx, st, table_z)
        if step:
            say(
                "%s the arm %s: slowly, in the driver's lightest joint impedance (%s), to the nearest point clear of "
                "everything known first (%d steps, the largest joint %.3f rad, the tool %+.0f mm, down %.0f mm at "
                "most; it gives way where it meets something) — %s is planned from where it ends"
                % (
                    who,
                    why,
                    IMPEDANCE_NAMES.get(ESCAPE_GAINS, ESCAPE_GAINS),
                    info["steps"],
                    info["largest_rad"],
                    info["tool_rise_mm"],
                    info["tool_drop_mm"],
                    plan_words,
                )
            )
        return step, st

    def off_the_limit():
        """The clear-margin step off a joint limit toward the goal and the legs from its end (ClearMarginFree), or
        RuntimeError."""
        last = None
        say(
            "%s %s: too near for the planner to start from — trying plan-free steps along the joint line toward "
            "%s, shortest first" % (who, ", ".join("%s is %.3f rad from its limit" % jr for jr in near), goal_words)
        )
        for step, at, info in clear.steps(ctx, state):
            try:
                rest, via = [planned(at, "home", list(goal))], None
            except PlanSwung as e:  # asked again, the planner answers alike: through a waypoint from the step's end
                rest, via, last = via_waypoint(at, e)
                if rest is None:
                    continue
            except RuntimeError as e:
                last = e
                continue
            if step:
                d = info["tool_move"]
                say(
                    "%s a plan-free step %.0f%% of the way to %s first (the largest joint %.3f rad, the tool "
                    "%+.0f mm up and %.0f mm across; obstruction-guarded)%s"
                    % (
                        who,
                        100 * info["fraction"],
                        goal_words,
                        info["largest_rad"],
                        1000 * d[2],
                        1000 * (d[0] ** 2 + d[1] ** 2) ** 0.5,
                        step_tail,
                    )
                )
            if via is not None:
                say(
                    "%s %s planned through a waypoint %.0f%% of the way along the joint line (the direct plan swung "
                    "wide: %s)" % (who, plan_words, 100 * via, last)
                )
            return step + rest
        raise RuntimeError(
            "no plan-free step toward %s of up to %.2f rad per joint lets the planner plan %s (last: %s) — "
            "jog the arm clear by hand" % (goal_words, CLEAR_MARGIN_MAX_STEP_RAD, plan_words, last)
        )

    why = esc.needed(state.joints if state.joints is not None else ctx.client.joints(), table_z)
    if why:
        step, state = step_clear(state, why)
        legs += step
    q = state.joints if state.joints is not None else ctx.client.joints()
    clear = ClearMarginFree(goal)
    near = clear.near(ctx, q)
    last = fell_back = None
    if near:
        try:
            return legs + off_the_limit()
        except RuntimeError as e:
            # 2026-11-10: the line toward the goal dropped the tool through the table: the soft step clear instead,
            # off the limits too, and the goal asked for from where it ends
            try:
                step, state = step_clear(
                    state, "%s, and %s" % (", ".join("%s is %.3f rad from its limit" % jr for jr in near), e)
                )
            except RuntimeError as e2:
                raise RuntimeError("%s; and %s" % (e, e2)) from e
            if not step:
                raise
            legs += step
            q = state.joints
            last = fell_back = e
    asked = 0
    low = None
    for _attempt in range(HOME_TRIES if ask_again else 1):
        asked += 1
        try:
            return legs + [planned(state, "home", list(goal))]
        except PlanSwung as e:  # asked again, the planner answers alike
            last = e
            break
        except PlanUnderTable as e:  # from this start HOME sweeps low: from the top of a straight lift instead
            last = low = e
            rest, words, last = raised(PlanState(joints=list(q), chain=state.chain), last)
            if rest is not None:
                say(
                    "%s %s planned from the top of %s: planned from here it went under the table (%s)"
                    % (who, plan_words, words, e)
                )
                return legs + rest
            break
        except RuntimeError as e:
            if not ask_again:
                raise
            last = e
    rest, via, last = via_waypoint(PlanState(joints=list(q), chain=state.chain), last)
    if rest is not None:
        say(
            "%s %s planned through a waypoint %.0f%% of the way along the joint line (asked for %d times directly; "
            "last: %s)" % (who, plan_words, 100 * via, asked, last)
        )
        return legs + rest
    raise RuntimeError(
        "%s did not plan from %s, %d times directly nor through a waypoint %s of the way along the joint line "
        "(last: %s)%s — jog the arm clear by hand"
        % (
            plan_words,
            "here" if fell_back is None else "the end of the soft step clear (taken because %s)" % fell_back,
            asked,
            "/".join("%.0f%%" % (100 * f) for f in HOME_WAYPOINTS),
            last,
            ""
            if low is None
            else "; planned directly it %s, and it planned clear from the top of no straight lift of %s mm nor "
            "soft step up to %s mm over the table%s"
            % (
                str(low).split(" ", 1)[1],
                "/".join("%.0f" % (1000 * d) for d in HOME_LIFTS_M),
                "/".join("%.0f" % (1000 * d) for d in HOME_RISES_M),
                "".join(cut),
            ),
        )
    )


def _push_guard_and_verify(m, cfg, depth_m=None):
    """The push's backstop and its verdict; `depth_m` the push's own bound
    when it is not button_travel_m (the reach-out's shallower push)."""
    guard = GuardSpec(touch_nm=max(0.5, m.touch_nm - cfg.contact_nm), trip="press")

    def verify(v):
        mm = (cfg.button_travel_m if depth_m is None else float(depth_m)) * 1000
        if v.outcome == "touch":
            peak = "" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak
            return True, "pressed — met a stop%s inside the %.0f mm push" % (peak, mm)
        if v.outcome == "arrived":
            return True, "pressed — full %.0f mm push from the measured contact" % mm
        return False, "push %s" % v.outcome

    return guard, verify
