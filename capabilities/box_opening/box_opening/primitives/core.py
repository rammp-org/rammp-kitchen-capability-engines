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
    GRIPPER_CMD_OPEN,
    PLANNER_LIMIT_MARGIN_RAD,
    TCP_OFFSET_M,
    TIP_TO_TOOL_M,
    TRANSIT_SPEED,
)
from box_opening.runtime.guards import (
    GuardSpec,
    in_band,
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
    # the scene camera's SceneLocator (its grabber streams the whole run) and
    # the lid-off's video of both cameras (perception/camera_video), --execute
    # full runs (reach_press._reach_press)
    scene_locator: object = None
    lid_video: object = None


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
        plan=plan_record(plan, target, plan_s, getattr(ctx.client, "refusals", ())[n_refused:]),
        waypoint=getattr(plan, "waypoint", None),
        guard_from=descent_index(plan),
    )
    if guard is not None:
        # from here on the box may not be exactly where it was detected —
        # every later FULL world allows for a contact-shifted container
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
    next_chain = state.chain + 1 if invalidates else state.chain
    return leg, PlanState(joints=end, chain=next_chain)


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
                return False, ("full stroke with no trip — never felt the surface, " "set-down NOT confirmed")
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
