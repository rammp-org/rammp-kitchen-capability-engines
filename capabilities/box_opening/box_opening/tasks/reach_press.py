"""The reach-out press: how every box is pressed.

Flatbot, 2026-10-22: a table 0.30 m above the arm's mount and the box
0.80-0.89 m out — no straight-down pose reaches the button there at any
wrist yaw (planner probe, 12 yaws x 3 heights). It is pressed with the hand
tilted back toward the arm, ONE FINGERTIP coming STRAIGHT DOWN onto the
button. It was the press for a box out of straight-down reach; since
2026-10-28 it presses every box, however near (owner: "make the arm always
go to that 45 degree looking pose even if it's close enough to go straight
up and down"; press_demo's straight-down staging press is gone):

  LOOK    the hand pulled back toward the arm (REACH_PULLBACK_M) and up
          (REACH_LOOK_RISE_M) clear of the box's padded cuboid, already at
          the press attitude. The wrist
          camera looks down at the lid from there and aims (the oblique aim,
          perception/button_aim). The box is put where the aim found it and
          the hand moves to THAT button's look pose along a straight line,
          guarded in a guarded run — the scene camera's box can be 8-10 cm
          off.
  SLIDE   a plan-free straight line in and down, holding the attitude, to
          REACH_START_M above the aimed button. Guarded at DRAG_NM in a
          guarded run.
  TOUCH   a plan-free straight line DOWN in world z to travel_m below it,
          guarded at contact_nm like the straight-down touch: the force is
          vertical, nothing shoves the box sideways.
  PUSH    reach_push_m further down from the contact — no deeper than the
          gripper clears the lid at this tilt — backstopped, held, then the
          runner's recoil reverses it.
  UNLOAD  straight back up the stroke, judged for a PULL at the fingertip
          (HANG_N) against the free-air reading taken there before the
          touch: leaving the button reads as a push let go, never as a pull.
          It stops UNLOAD_CHECK_M up and checks at rest that nothing came
          with the finger — a latched box rises that far at most — then
          rises to the stroke's start, above the highest popped knob.
  CHECK   at rest at the stroke's start, against the same pose read before
          the touch: anything on the hand and the arm holds.
  SLIDE   back out to the look pose, guarded at DRAG_NM and HANG_N against
          that same free-air reading. No guard here can feel a box slid
          along the table (a 0.3-0.5 kg canister slides at 1-2.5 N, and the
          recorded residual reads as 2.4-2.6 N sideways): what keeps the
          box on the table is the hand starting this leg 35 mm or more above
          anything on the lid, with nothing on it.
  AFTER   the wrist re-aims from the look pose and the scene camera looks
          again: did the knob pop, did the box move? Judged BEFORE the arm
          goes anywhere; neither camera finding the box holds the arm.

Every guard here judges what the arm's own weight does not explain
(runtime/guards.CompensatedGuard with kinematics.GravityModel): along a
15 cm reach at 0.8-0.9 m joint_4's gravity torque moves 1.1-1.9 Nm — as
much as a canister hanging off a finger — so a raw guard either tripped on
the reach itself or missed the canister (review 2026-10-22). The model is
checked against the arm before any guarded line flies (gravity_mismatch).

The tilt is the smallest of REACH_TILTS_DEG that plans. It stops at 45 deg
because of the gripper's shape: past that its knuckles, then its palm, come
down onto the flush lid before the fingertip reaches the button (the URDF's
collision meshes with the measured 131 mm fingers: 8 mm of lid clearance at
30 deg, 5.6 at 45, under 3 past 60). A guarded press ends the run there. A
compliant depth press, the default, goes on to take the lid off — the look
at the knob from the look pose, the press once more where it is not up, the
grip, the lift, the carry and the set-down (tasks/reach_lid_off) — unless
--press-only.

Each tilt is proven from its look pose's joints before the planner is asked
for anything, and only the first that proves has its look planned
(plan_reach). A compliant depth press's look goes to the posture — the same
pose, the elbow swung elsewhere — whose walks to the hand turned down over
the button are quickest, where that is LOOK_POSTURE_GAIN_S quicker than the
posture nearest HOME and its centre line keeps clear of the table and the
box (quickest_look, posture_refusal: a minimal guard, not a collision
check).
"""

import dataclasses
import math
import time
from dataclasses import dataclass

import numpy as np

from box_opening import LOGGER as log
from box_opening.constants import (
    COMPLIANT_ACCEL,
    COMPLIANT_DECEL,
    COMPLIANT_FREE_LINE_SPEED,
    COMPLIANT_JUNCTION_SPEED,
    COMPLIANT_QD_CEILING,
    COMPLIANT_RUN_GAINS,
    CONTACT_SPEED,
    DRIFT_REPLAN_RAD,
    FREE_LINE_SPEED,
    IMPEDANCE_KQ,
    IMPEDANCE_NAMES,
    JOINT_VMAX,
    PLANNER_LIMIT_MARGIN_RAD,
    TIP_TO_TOOL_M,
    TRANSIT_SPEED,
    CustomGains,
)
from box_opening.geometry import rotate_about_world_axis
from box_opening.models.container import ContainerPose, attitude_quat
from box_opening.primitives.core import (
    CLEAR_MARGIN_MAX_STEP_RAD,
    CONTACT_SHIFT_PAD_M,
    ClearMarginFree,
    PlanState,
    _full_world,
    _interaction_world,
    _plan_motion,
    _push_guard_and_verify,
)
from box_opening.runtime import fingers
from box_opening.runtime.guards import GuardSpec
from box_opening.runtime.legs import Kind, Leg, can_merge, merge_groups
from box_opening.runtime.retime import RetimeParams, positions_to_traj, retime_group

REACH_TILTS_DEG = (30, 35, 40, 45)
# How far the rest of the gripper stands above the flush lid when the
# fingertip touches the button's centre, by tilt (URDF collision meshes with
# the measured 131 mm fingers; 30 and 45 computed, 35 and 40 interpolated).
# The push must never go deeper than this, or the knuckles press the lid.
# Off centre across the bearing it is less — 3.2 mm at 45 deg 4 mm off (the
# other fingertip's corner): such a push meets the lid, trips its backstop
# and the knob does not pop, which the run reports as a failure (exit 1).
REACH_LID_CLEAR_M = {30: 0.0082, 35: 0.0073, 40: 0.0064, 45: 0.0056}
PUSH_MARGIN_M = 0.0015
# CONTACT_TOOL and REACH_LID_CLEAR_M were measured for this attitude (the
# fingers side by side across the line to the box): another roll puts a
# different part of the hand lowest
REACH_ATTITUDE_RPY_DEG = (180.0, 0.0, 90.0)


def reach_push_m(cfg, tilt_deg):
    """The push's depth at this tilt: the configured bound, or less, so the
    gripper stays PUSH_MARGIN_M clear of the lid."""
    return min(float(cfg.button_travel_m), REACH_LID_CLEAR_M[int(tilt_deg)] - PUSH_MARGIN_M)


# the stroke starts this far above the button (a straight-down press's final
# approach) — and the unload ends there: above a popped knob's 10-25 mm
REACH_START_M = 0.06
# Where a compliant depth press's turn down at the slide's end does not fly,
# the slide carries on straight down at its attitude and the hand turns
# there, the fingertip this far above the button's top, highest first; the
# lowest is depth_press.TOUCH_SLOW_M, where its stroke slows anyway. Out at
# the edge of reach it is the turn's END that fails, not its path: with the
# pads 60 mm above the button the elbow runs out of bend (flatbot 2026-10-28
# 14:28:35, the aimed button r 0.791 az -39: the 15 deg turn reaches
# joint_4 -0.30 at 88% and its last step does not converge; the arm went
# home from the look). 15 mm lower the same turn ends at joint_4 -0.30 and
# the stroke bottoms at -0.68: the stroke goes DOWN, toward the shoulder
# (0.25 above arm_mount), so only the turn's height bounds the reach. The
# turn pivots on the fingertip, the lowest point of the hand at these
# tilts, so nothing comes lower than it (32.7 mm over the lid at 35 mm,
# flatbot's URDF); the box is shut (refuse_open_box), its lid flush.
TURN_LOW_ABOVE_M = (0.045, 0.035)
# Before planning home from over the box, the tool this far above the box's
# padded cuboid (worlds.planned_top_z). The planner's lowest collision
# spheres belong to the gripper's base link, not the fingers (rammp_curobo
# configs/robot_gen3_2f85.yaml: arm_robotiq_85_base_link centres
# (+-0.045, 0, 0.145) and (0, +-0.045, 0.145), r 0.018 + its
# collision_sphere_buffer 0.005), 25 mm past tool_frame and
# HAND_SPHERE_OFF_AXIS_M off its axis. Their lowest point is
# 0.025 cos(lean) + 0.045 sin(lean) + 0.023 below tool_frame: 48 mm with the
# hand straight down, which this leaves 12 mm clear; 67 mm at 30 deg, which
# it leaves 7 mm INSIDE the padded cuboid (2026-10-27 16:23: home refused
# INVALID_START_STATE_WORLD_COLLISION after the clear, then a 15 cm
# back-off). The four are symmetric, so the sin term bounds every lean.
HOME_CLEAR_M = 0.06
# A compliant run's clear (clear_targets) stands that, plus what the hand's
# lean hangs those spheres lower (HAND_SPHERE_OFF_AXIS_M: the robot's body,
# not the scene's), plus CLEAR_EXTRA_M over the padded cuboid, and pulls
# back toward the arm CLEAR_BACK_M as it rises, else REACH_PULLBACK_M.
# Straight up, a hand turned down drives joint_6 into its limit (15:19:
# 0.104 rad, a 1.44 s clear-margin step; 14:33: no line up flew, home was
# refused START_IN_LIMIT_MARGIN and the arm held over the box).
HAND_SPHERE_OFF_AXIS_M = 0.045
CLEAR_EXTRA_M = 0.005
CLEAR_BACK_M = 0.05
# The look pose, pulled back toward the arm from the stroke's start: 15-20 cm
# plans in the full world at both spots, 10 cm does not — the gripper meets
# the padded container cuboid (planner probe 2026-10-22).
REACH_PULLBACK_M = 0.15
# ... and this far above the stroke's start. Level with it the wrist camera
# stood ~0.245 m from the lid (flatbot 2026-10-27 14:04, the scene camera
# placed by its calibration) and the glossy lid and button held depth in
# patches; the looks that had depth on the lid stood ~0.30 m off it (12:58,
# when the scene camera put the lid 9 cm high). 6.5 cm up is ~0.30 m, and
# the slide in, from above the old one, comes no nearer the box.
REACH_LOOK_RISE_M = 0.065
# The wrist's aim puts the button this much nearer the arm than it is, along
# the box's bearing from arm_mount: every aim's point is moved out by it
# (aim_forward), so the press and the grip come down that much farther out,
# and an aim after the press is judged against the first on the same footing
# (owner 2026-10-29: "the camera calibration is slightly off, we need to make
# the arm move just 2 cm forward when pressing the box and gripping the lid so
# it grabs the middle, not the edge"). On 2026-10-29's full runs the wrist's
# look after the press put the button 6-11 mm farther out than the aim the
# press and the grip went by (12:02, 12:03, 12:04; 10:27's 0 mm), and 12:03's
# grip closed on air 11 mm short of it. The look pose and the proof before it
# go by the scene camera's button, as before. 2 cm was too much (owner
# 2026-10-29, after flying it: "2 cm was too much, change it to 1"): 1 cm, as
# the after-looks' 6-11 mm said.
AIM_FORWARD_M = 0.01
# The closed 2F-85's lowest point at these tilts, in tool_frame, with the
# reach attitude's roll (fingers side by side across the line to the box):
# one fingertip's end, 11 mm past tool_frame like TCP_OFFSET_M straight down.
CONTACT_TOOL = (0.0, -0.009, 0.011)
TIP_MID_TOOL = (0.0, 0.0, -TIP_TO_TOOL_M)  # the finger-tip links' midpoint (client.contact_xyz)
# Every guard below judges the gravity-compensated residual. On flatbot's
# recording (kinematics.GravityModel) it stayed under 0.76 Nm on every wrist
# joint through 22 s of motion; the reach lines' own model error is ~0.15.
DRAG_NM = 1.5
# A pull straight down at the fingertip, read off the residual (guards
# CompensatedGuard `hang`): a 0.3-0.5 kg canister (estimated — weigh it) is
# 2.9-4.9 N; the recording's worst residual reads as 1.6-1.7 N moving, and
# 0.5-0.6 N between two readings at rest
HANG_N = 2.5
HANG_STATIC_N = 1.0
STATIC_NM = 0.75  # ... and at rest at the stroke's start, any wrist joint this far off its free-air reading
UNLOAD_CHECK_M = 0.015  # the first stop on the way up: past the button's travel
# The model against the arm, between the rest pose and the look pose: off by
# more than this on any wrist joint (and a sign error is off by twice the
# change) and no guard here can be trusted
GRAVITY_CHECK_NM = 1.0
GRAVITY_CHECK_FRAC = 0.3
LINE_MIN_M = 0.001  # a line shorter than this is not flown: the arm is there
BOX_MOVED_M = 0.005  # the re-aimed button this far from the first aim: the press moved the box
BOX_LIFTED_M = 0.008  # ... or its lid this much higher or lower
SCENE_MOVED_M = 0.02  # the scene camera's box this far from where it was before the press


def reach_quat(model, xy, tilt_deg):
    """The press attitude turned to face `xy` — the wrist camera rides above
    the hand, the fingers side by side across the line to the box — with its
    down axis tilted `tilt_deg` away from the arm: the hand leans back."""
    d = math.hypot(xy[0], xy[1])
    q = attitude_quat(model.press_attitude_rpy_deg, math.atan2(xy[1], xy[0]))
    return list(rotate_about_world_axis(q, (xy[1] / d, -xy[0] / d, 0.0), math.radians(tilt_deg)))


def _rot(quat):
    from box_opening.perception.d405 import quat_to_mat

    return np.asarray(quat_to_mat(*quat), float)


def tool_for_contact(contact_xyz, quat):
    """The tool_frame position that puts the fingertip's contact point at
    `contact_xyz` with the hand at `quat`."""
    return [float(v) for v in np.asarray(contact_xyz, float) - _rot(quat) @ np.asarray(CONTACT_TOOL)]


def contact_from_tips(tip_mid_xyz, quat):
    """Where the fingertip's contact point is, from the finger-tip links'
    midpoint the arm measures at a trip (client.contact_xyz)."""
    v = _rot(quat) @ (np.asarray(CONTACT_TOOL) - np.asarray(TIP_MID_TOOL))
    return [float(c) for c in np.asarray(tip_mid_xyz, float) + v]


def cpose_for_button(button, yaw, offset):
    """The container pose whose button (from_container with `offset`) is at
    `button`: every world from the aim on holds the box there."""
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy, oz = (float(v) for v in offset)
    return ContainerPose(
        xyz=(float(button[0]) - (c * ox - s * oy), float(button[1]) - (s * ox + c * oy), float(button[2]) - oz), yaw=yaw
    )


def unsupported(model):
    """Why this container cannot be pressed reaching out, or None."""
    rpy = tuple(float(v) for v in model.press_attitude_rpy_deg)
    if any(abs(a - b) > 1e-6 for a, b in zip(rpy, REACH_ATTITUDE_RPY_DEG, strict=False)):
        return "its press attitude is %s; the reach-out's fingertip contact and lid clearance were measured for %s" % (
            list(rpy),
            list(REACH_ATTITUDE_RPY_DEG),
        )
    return None


@dataclass
class ReachPlan:
    tilt_deg: int
    quat: list
    button: tuple  # the button top it is aimed at (x, y, z)
    look_xyz: list
    start_xyz: list
    bottom_xyz: list
    look_leg: object  # planned from the joints asked, full world; None for a look plan_reach kept unplanned
    slide_pts: list  # look -> start, the attitude held
    stroke_pts: list  # start -> bottom, straight down; None for a depth plan, which never flies it
    # a depth plan's (plan_reach depth=True, solve_way_in): how the hand turns
    # down from the look pose (depth_press.WayDown) and the compliant press
    # stroke from there (depth_press.StrokePlan)
    down: object = None
    stroke: object = None
    # plan_reach's: the joints its slide and the rest were proven from — the
    # arm's own IK's for the look pose (client.pose_goal_joints), or the
    # planned look's end where that IK found none or the plan ended elsewhere
    # — or, with shortest_walk, the posture quickest_look chose
    look_joints: list = None
    # plan_reach shortest_walk's: how the look's posture was chosen
    # (quickest_look's note, for the run record's reach_plan); None where it
    # was not asked
    posture: dict = None


def _short(e):
    s = str(e)
    return s.split(": ", 1)[-1][:90]


def reach_geometry(model, button, tilt, travel_m):
    """(quat, start, bottom, look) for pressing `button` at `tilt`: the
    hand's attitude, the stroke's start REACH_START_M above the button and
    its bottom travel_m below it (tool_frame, for the fingertip's contact
    point), and the look pose REACH_PULLBACK_M back toward the arm from the
    start and REACH_LOOK_RISE_M above it."""
    bx, by, bz = (float(v) for v in button)
    d = math.hypot(bx, by)
    quat = reach_quat(model, (bx, by), tilt)
    start = tool_for_contact((bx, by, bz + REACH_START_M), quat)
    bottom = tool_for_contact((bx, by, bz - travel_m), quat)
    look = [start[0] - REACH_PULLBACK_M * bx / d, start[1] - REACH_PULLBACK_M * by / d, start[2] + REACH_LOOK_RISE_M]
    return quat, start, bottom, look


def walk_or_turn(ctx, q_look, slide_pts, start_xyz, quat, top_z=None):
    """way_down's way to the hand turned down over the slide's end
    `start_xyz`, no stroke: one walk in from the look pose's joints
    `q_look` (depth_press.walk_down_path), else the turn about the
    fingertip where the slide `slide_pts` ends (depth_press.turn_down_path,
    up to 15 deg short: COMPLIANT_TURN_SHORT_DEG) — where that turn does not
    fly and `top_z` (the button's top, or the lid's) is given, the slide
    carried on straight down to TURN_LOW_ABOVE_M above it and the turn
    there (WayDown slide, start_xyz and above_m: the callers fly that
    slide). The reach-out's lid off comes down to the knob the same way
    (reach_lid_off.solve_grip_way). Returns (depth_press.WayDown, None) or
    (None, why)."""
    from box_opening.runtime import approach
    from box_opening.tasks import depth_press as dp  # imports this module: not at the top

    chain = approach._chain()
    pts, down, short, why_walk = dp.walk_down_path(chain, ctx.model, q_look, start_xyz, quat)
    if pts is not None:
        return dp.WayDown(pts, down, short, True), None
    pts, down, short, why_turn = dp.turn_down_path(
        chain, ctx.model, slide_pts[-1], CONTACT_TOOL, dp.COMPLIANT_TURN_SHORT_DEG
    )
    if pts is not None:
        return dp.WayDown(pts, down, short, False), None
    whys = ["turn after the slide: %s" % why_turn]
    if top_z is not None:
        tip = np.asarray(start_xyz, float) + _rot(quat) @ np.asarray(CONTACT_TOOL)
        for above in TURN_LOW_ABOVE_M:
            if float(tip[2]) <= float(top_z) + above + LINE_MIN_M:
                continue  # the slide already ends this low
            low = tool_for_contact((float(tip[0]), float(tip[1]), float(top_z) + above), quat)
            drop, why = chain.straight_line(slide_pts[-1], low, quat)
            if drop is None:
                whys.append("slid on down to %.0f mm above the button: %s" % (1000 * above, why))
                continue
            pts, down, short, why_turn = dp.turn_down_path(
                chain, ctx.model, drop[-1], CONTACT_TOOL, dp.COMPLIANT_TURN_SHORT_DEG
            )
            if pts is None:
                whys.append("turned %.0f mm above the button: %s" % (1000 * above, why_turn))
                continue
            slide = [list(q) for q in slide_pts] + [list(q) for q in drop[1:]]
            return dp.WayDown(pts, down, short, False, slide, low, above), None
    return None, "walk in: %s; %s" % (why_walk, "; ".join(whys))


def way_down(ctx, cfg, q_look, slide_pts, start_xyz, quat, top_z):
    """How the compliant depth press gets from the look pose's joints
    `q_look` to pressing the button whose top is at `top_z`, solved by the
    arm's own kinematics before anything flies: one walk in from the look
    pose to the hand turned straight down over the slide's end `start_xyz`
    (depth_press.walk_in_down; WayDown walked, its end the stroke's start),
    else the slide `slide_pts` held at `quat`, then the TURN about the
    fingertip (CONTACT_TOOL) where it ends (depth_press.turn_down), up to
    15 deg short of straight down as the compliant press turns
    (depth_press.COMPLIANT_TURN_SHORT_DEG: way_down serves only that
    press) — where that turn does not fly, the slide carried on straight
    down to TURN_LOW_ABOVE_M above the button and the turn there (WayDown
    slide and start_xyz: the callers fly that slide); then the press stroke
    from the hand turned down
    (depth_press.compliant_stroke). Each of those lines is one the runner
    flies (depth_press.line_refusal): a walk it refuses is no walk, and the
    slide and the turn are tried. The aimed button's walk or turn and
    stroke fly as solved, in one execution with the move and the slide
    before them (way_in_legs); only where those legs do not make one
    (_not_one_execution) is each solved again from where the arm stands,
    trying the same leans. Returns (WayDown, StrokePlan, None) or (None,
    None, why)."""
    from box_opening.runtime import approach
    from box_opening.tasks import depth_press as dp  # imports this module: not at the top

    chain = approach._chain()
    way, why = walk_or_turn(ctx, q_look, slide_pts, start_xyz, quat, top_z)
    if way is None:
        return None, None, why
    slide = slide_pts if way.slide is None else way.slide  # it turned lower down: the slide carries on to there
    stroke, why = dp.compliant_stroke(chain, cfg, way.pts[-1] if way.pts else slide[-1], way.quat, top_z)
    if stroke is None:
        return None, None, why
    # the walk or the turn passed the runner's gate in turn_walk; the slide
    # before a turn and the stroke fly as legs of their own too
    lines = ([] if way.walked else [("the slide in", slide)]) + [(name, p) for name, p, _speed in stroke.legs]
    for name, line in lines:
        why = dp.line_refusal(chain, line)
        if why is not None:
            return None, None, "%s: %s" % (name, why)
    return way, stroke, None


# A look pose whose wrist stands further than this past everything the arm
# reaches (kinematics.ArmChain.past_reach_m) is refused without asking the
# planner. Where the arm's own IK finds no look the client hands the planner
# the pose, and cuRobo calls a pose reached within 5 mm and 5.7 deg
# (sin(err/2) < 0.05: its MotionGen defaults; rammp-curobo 4fa044d
# planner.py _verify_hold). 5.7 deg swings flatbot's wrist, 0.2875 m back
# from tool_frame, 29 mm: 34 mm in all, so no look it could still call
# reached is refused. 2026-10-28 14:29:19, a box seen at r 1.13 m: the four
# looks' wrists stood 99, 73, 47 and 21 mm past, and the planner answered
# IK_FAIL to each (~1.15 s); 14:29:54 (r 1.04 m): the 30 deg look's 22 mm.
LOOK_PAST_REACH_M = 0.04
# A planned look that ends further than this from the joints its press was
# proven from (any joint; a continuous one the short way round) is proven
# again from where it ends, as every look was until 2026-10-28. The
# planner's joint plan ends on its goal (rammp-curobo's smoke test:
# goal_mismatch_rad under 1e-3); its FK-pose fallback inside PlanToJoints
# (3 of 10-28's 72 plans) and the client's PlanToPose end wherever their IK
# lands.
LOOK_END_RAD = 0.01


def _joint_gap(q_end, q):
    """The largest joint difference between `q_end` and `q`, a continuous
    joint the short way round (branches.on_branch)."""
    from box_opening.runtime.branches import on_branch

    return max(abs(a - float(b)) for a, b in zip(on_branch(q_end, q), q, strict=False))


def _plan_look(ctx, q_from, look_xyz, quat):
    """The look leg from joints `q_from`, planned as it always has been —
    the pose goal, full world, TRANSIT_SPEED — which the client plans to its
    own IK's joints (client.pose_goal_joints), else hands the planner as a
    pose. Returns (leg, None) or (None, why)."""
    try:
        leg, _st = _plan_motion(
            ctx,
            PlanState(joints=list(q_from), chain=0),
            "reach:look",
            ("pose", look_xyz, quat),
            _full_world(ctx),
            TRANSIT_SPEED,
        )
    except RuntimeError as e:
        return None, _short(e)
    return leg, None


def _proven(ctx, cfg, chain, tilt, button, geometry, q_look, depth):
    """plan_reach's proof of `tilt` (`geometry`: reach_geometry's quat,
    start, bottom and look for `button`) from the look pose's joints
    `q_look`: (ReachPlan with no look leg yet, None, None) where the slide
    and the stroke — with `depth`, the way down — fly from there; else
    (None, why, None), or, where the slide flies and the way down does not,
    (None, why, the ReachPlan plan_reach keeps in its `looks`)."""
    quat, start, bottom, look = geometry
    q_look = [float(v) for v in q_look]
    slide, why = chain.straight_line(q_look, start, quat)
    if slide is None:
        return None, "the slide in (%s)" % why, None
    if depth:
        down, stroke, why = way_down(ctx, cfg, q_look, slide, start, quat, button[2])
        if down is None:
            kept = ReachPlan(tilt, quat, button, look, start, bottom, None, slide, None, look_joints=q_look)
            return None, "the way down (%s)" % why, kept
        if down.slide is not None:  # it turns lower down: the slide carries on to there
            slide, start = down.slide, down.start_xyz
        return ReachPlan(tilt, quat, button, look, start, bottom, None, slide, None, down, stroke, q_look), None, None
    stroke, why = chain.straight_line(slide[-1], bottom, quat)
    if stroke is None:
        return None, "the stroke (%s)" % why, None
    return ReachPlan(tilt, quat, button, look, start, bottom, None, slide, stroke, look_joints=q_look), None, None


# The look pose's POSTURE (plan_reach shortest_walk, the compliant depth
# press). The 7-joint arm holds the look pose — the same tool pose, the
# same wrist camera view — with its elbow anywhere along a swing about the
# line from its shoulder to its wrist (kinematics.ArmChain.self_motion). The
# client plans the look to the posture quickest to fly to from HOME
# (client.pose_goal_joints: ArmChain.nearest_solution), which minimises the
# one transit to the look and nothing after it. The run then walks between
# the look and the hand turned down over the button 3-4 times — the way in,
# the way back, the grip's way in, a back out — and on 2026-10-29 13:37
# those walks were ~15 s of a 34.5 s run: from that posture each is 1.9-2.4
# rad of joint travel for ~16 cm of the hand's. The elbow swung elsewhere,
# the same walk is shorter. Offline (flatbot's URDF, the compliant run's own
# timing: flown_s, transit_s), the posture quickest_look chooses:
#   10-28 16:23 (r 0.66 m, az -26 deg) walk 2.79 -> 1.79 s, look +0.34 s:
#               2.7 s quicker over the three walks
#   10-28 15:45 (r 0.75)  walk 3.15 -> 2.16 s, +0.15 s: 2.8 s
#   10-29 12:04 (r 0.65)  walk 2.79 -> 1.70 s, +0.38 s: 2.9 s
#   10-29 13:37 (r 0.76), 10-28 14:21 (r 0.73): today's kept — the best
#               other elbow there saves 0.6-0.7 s, under LOOK_POSTURE_GAIN_S
# The candidates: the self-motion walked LOOK_POSTURE_SWING_RAD each way
# from today's posture, one every LOOK_POSTURE_EVERY_RAD, inside the
# planner's joint limits (a goal within PLANNER_LIMIT_MARGIN_RAD of one is
# refused: GOAL_IN_LIMIT_MARGIN), none within LOOK_POSTURE_DEDUP_RAD of
# today's or of one before it. They are ranked by a bound — the transit and
# the walk as straight joint lines at their joint speed caps — and the
# LOOK_POSTURE_PROOFS best are proven as today's is (_proven: the slide and
# the way down, every line one the runner flies), then GUARDED
# (posture_refusal) and timed as flown. The best is taken only where it is
# LOOK_POSTURE_GAIN_S quicker than today's over the transit and
# LOOK_POSTURE_WALKS walks: elsewhere today's posture stands, exactly as it
# was planned before.
LOOK_POSTURE_SWING_RAD = 2.4  # past ~2 rad the look's transit costs more than the walks save
LOOK_POSTURE_EVERY_RAD = 0.3
LOOK_POSTURE_DEDUP_RAD = 0.2  # any joint: closer than this is the same posture
LOOK_POSTURE_PROOFS = 3
# IK time the search may spend (s): no proof is started that the time left
# would not hold at the slowest one so far. Flatbot's Jetson (offline,
# 2026-10-29): the sweep ~0.04 s, the ranking ~0.05, each proof 0.09-0.20
LOOK_POSTURE_BUDGET_S = 0.6
# How much quicker another posture must be, over the transit and the walks,
# for the look to go there (s). The estimate times each walk alone; the
# runner flies it merged into a longer execution (way_in_legs), so a few
# tenths are inside what the estimate cannot tell: at 14:21 and 13:37 the
# best other elbow saved 0.6-0.7 s by it, and the research pass of 10-28
# found no gain at 14:21
LOOK_POSTURE_GAIN_S = 1.0
LOOK_POSTURE_WALKS = 3  # the way in, the way back, the grip's way in (a back out only where the grip misses)
# posture_refusal: the arm's centre line kept this far off the table and
# the box — the links' radius (the Gen3's are ~0.06 m from their joints'
# axes) and 0.03 m more
POSTURE_LINK_RADIUS_M = 0.06
POSTURE_CLEAR_M = POSTURE_LINK_RADIUS_M + 0.03
POSTURE_LINE_STEP_M = 0.02  # the centre line sampled this finely between joint origins
# ArmChain.points' rows: joint_1 .. joint_7's origins, then the tool. The
# shoulder (joint_1 and joint_2) stands where the arm is mounted, below the
# table's top on flatbot (joint_2 0.244 m above arm_mount, the table 0.21),
# whatever the posture: the line is judged from joint_3's origin to
# joint_7's. The hand past joint_7 is not judged: the tool's pose alone puts
# it — the walk's and the stroke's poses are the same from every posture —
# and it must come down over the box, to the lid ~0.10 m above the table.
CENTRE_LINE = (2, 7)  # ArmChain.points rows [2, 7): joint_3 .. joint_7
CENTRE_LINE_NAMES = ("joint_3", "joint_4", "joint_5", "joint_6", "joint_7")


def compliant_retime():
    """The profile a compliant run re-times every motion with
    (runtime/runner.py Runner.retime_params for a compliant client — the
    depth press flies only in one): the compliant ease out of rest and into
    it, the wrist's ceiling and the junction ceiling."""
    return dataclasses.replace(
        RetimeParams(),
        accel=COMPLIANT_ACCEL,
        decel=COMPLIANT_DECEL,
        qd_ceiling=COMPLIANT_QD_CEILING,
        junction_cruise=COMPLIANT_JUNCTION_SPEED,
    )


def flown_s(legs, speed):
    """How long the chained joint paths `legs` take at the cruise fraction
    `speed` as ONE execution under a compliant run's profile (retime_group,
    as Runner._run_motion re-times a merged group): an estimate, rest to
    rest; the runner flies a walk merged into a longer execution."""
    names = ["joint_%d" % (i + 1) for i in range(len(JOINT_VMAX))]
    trajs = [positions_to_traj(names, [[float(v) for v in q] for q in pts]) for pts in legs if len(pts) > 1]
    if not trajs:
        return 0.0
    _traj, info = retime_group(trajs, [float(speed)] * len(trajs), JOINT_VMAX, compliant_retime())
    return float(info["duration_s"])


def _joint_line(a, b, step_rad=0.02):
    """The straight joint-space line from `a` to `b`, a waypoint every
    `step_rad` of the joint that moves most."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(math.ceil(float(np.abs(b - a).max()) / step_rad)))
    return [a + (b - a) * (k / n) for k in range(n + 1)]


def transit_s(q_from, q):
    """The look's transit from `q_from` to `q` (each continuous joint the
    short way round from q_from's planner branch, as the client sends the
    goal: plan_to_joints) as a straight joint line at TRANSIT_SPEED under a
    compliant run's profile — a stand-in for the planner's path, which in an
    empty world is near it, not it."""
    from box_opening.runtime.branches import nearest_branch, planner_branch

    start = planner_branch(q_from)
    return flown_s([_joint_line(start, nearest_branch(q, start))], TRANSIT_SPEED)


def _line_bound_s(a, b, speed):
    """The slowest joint's travel from `a` to `b` over its cruise cap at
    `speed` under a compliant run's profile (RetimeParams.vmax_margin x the
    limit x speed, no faster than COMPLIANT_QD_CEILING): quickest_look's
    ranking, no ease in or out."""
    p = compliant_retime()
    caps = [min(p.vmax_margin * v * float(speed), c) for v, c in zip(JOINT_VMAX, COMPLIANT_QD_CEILING, strict=False)]
    return max(abs(float(x) - float(y)) / c for x, y, c in zip(a, b, caps, strict=False))


def walk_legs(plan):
    """A depth ReachPlan's WALK: the joint paths from its look pose to the
    hand turned down over the button — the walk in, or the slide and the
    turn after it (way_in_legs flies them so)."""
    if plan.down.walked:
        return [plan.down.pts]
    return [plan.slide_pts, plan.down.pts]


def posture_refusal(ctx, chain, legs, button):
    """Why the arm's centre line along the joint paths `legs` comes nearer
    than POSTURE_CLEAR_M to the table or into the box padded by it, or None.
    A MINIMAL GUARD, NOT A COLLISION CHECK: nothing collision-checks the
    plan-free walks, and the planner's world is empty (worlds.py), so an
    elbow swung elsewhere (quickest_look) is held to this before its walk is
    taken. The centre line is ArmChain.points' joint origins from joint_3's
    to joint_7's (CENTRE_LINE) joined by straight lines, sampled every
    POSTURE_LINE_STEP_M at every waypoint; the links' bulk is the margin's
    POSTURE_LINK_RADIUS_M. Not judged: the shoulder, which stands where the
    arm is mounted at every posture, and the hand past joint_7 — the
    gripper, its fingers and the wrist camera — which the tool's poses put
    where they put it from every posture, down over the box. The table:
    ctx.worlds.table_top_z, the scene camera's
    (press_demo.adopt_scene_table), else the base of the scene camera's box
    (ctx.cpose). The box: ctx.cpose with the model's dims — the round
    model's disc a vertical cylinder, any other a cuboid at its yaw — up to
    its top or `button`'s, whichever is higher."""
    worlds, cpose, m = getattr(ctx, "worlds", None), getattr(ctx, "cpose", None), ctx.model
    if cpose is None:
        return "no box to keep the arm out of"
    table_z = getattr(worlds, "table_top_z", None)
    if table_z is None or not math.isfinite(float(table_z)):
        table_z = float(cpose.xyz[2])
    floor = float(table_z) + POSTURE_CLEAR_M
    top = max(float(cpose.xyz[2]) + float(m.dims[2]), float(button[2])) + POSTURE_CLEAR_M
    cx, cy = float(cpose.xyz[0]), float(cpose.xyz[1])
    c, s = math.cos(float(cpose.yaw)), math.sin(float(cpose.yaw))
    for k_leg, pts in enumerate(legs):
        if pts is None or len(pts) == 0:
            continue
        P = np.array([chain.points(q)[CENTRE_LINE[0] : CENTRE_LINE[1]] for q in pts])  # (n, 5, 3)
        n = len(P)
        for j in range(P.shape[1] - 1):  # the segment from row j to row j + 1
            a, b = P[:, j], P[:, j + 1]
            steps = max(1, int(math.ceil(float(np.linalg.norm(b - a, axis=1).max()) / POSTURE_LINE_STEP_M)))
            f = np.linspace(0.0, 1.0, steps + 1)
            S = a[:, None, :] + f[None, :, None] * (b - a)[:, None, :]  # (n, steps + 1, 3)
            low = S[..., 2] < floor
            if low.any():
                i = int(np.argwhere(low)[0][0])
                return (
                    "the centre line %.0f mm above the table between %s and %s (path %d, %.0f%% of the way): "
                    "under the %.0f mm kept"
                    % (
                        1000 * (float(S[i, :, 2].min()) - float(table_z)),
                        CENTRE_LINE_NAMES[j],
                        CENTRE_LINE_NAMES[j + 1],
                        k_leg + 1,
                        100.0 * i / max(1, n - 1),
                        1000 * POSTURE_CLEAR_M,
                    )
                )
            dx, dy = S[..., 0] - cx, S[..., 1] - cy
            if m.round:
                foot = np.hypot(dx, dy) < float(m.dims[0]) / 2.0 + POSTURE_CLEAR_M
            else:
                u, v = c * dx + s * dy, -s * dx + c * dy
                foot = (np.abs(u) < float(m.dims[0]) / 2.0 + POSTURE_CLEAR_M) & (
                    np.abs(v) < float(m.dims[1]) / 2.0 + POSTURE_CLEAR_M
                )
            inside = foot & (S[..., 2] < top)
            if inside.any():
                i = int(np.argwhere(inside)[0][0])
                return (
                    "the centre line between %s and %s inside the box padded %.0f mm (path %d, %.0f%% of the way)"
                    % (
                        CENTRE_LINE_NAMES[j],
                        CENTRE_LINE_NAMES[j + 1],
                        1000 * POSTURE_CLEAR_M,
                        k_leg + 1,
                        100.0 * i / max(1, n - 1),
                    )
                )
    return None


def quickest_look(ctx, cfg, chain, tilt, button, geometry, q_from, plan):
    """plan_reach shortest_walk's posture for the look pose: of the arm's
    postures holding it (the LOOK POSTURE comment above), the one whose
    look and walks are quickest, where that is LOOK_POSTURE_GAIN_S quicker
    than today's — `plan`, the depth ReachPlan proven from the joints the
    client plans the look to (client.pose_goal_joints) — else `plan`
    itself. Every other posture is proven as `plan` was (_proven at
    `tilt`, `geometry`: reach_geometry's for `button`, from the posture)
    and must pass posture_refusal along its walk and its stroke; `plan`
    is not held to it, as it never was. Returns (the ReachPlan, the note:
    `chosen` 0 for today's posture, else its number among the candidates
    in self_motion's order, `swing_rad` how far along the elbow's swing it
    stands, `joint_gap_rad` its largest joint difference from today's;
    `walk_s` / `walk_s_today` and `transit_s` / `transit_s_today` as
    flown_s and transit_s time them, `saved_s` over the transit and
    LOOK_POSTURE_WALKS walks; `candidates`, `proven` and `refused` (why,
    by number); `ik_s` the time it took)."""
    from box_opening.runtime.branches import (
        CONTINUOUS_JOINTS,
        PLANNER_CONTINUOUS_LIMIT_RAD,
        nearest_branch,
        planner_branch,
    )
    from box_opening.runtime.kinematics import SELF_MOTION_STEP_RAD

    t0 = time.monotonic()
    start = planner_branch(q_from)
    q_today = np.asarray(plan.look_joints, float)
    walk0, transit0 = flown_s(walk_legs(plan), COMPLIANT_FREE_LINE_SPEED), transit_s(start, q_today)
    best = (transit0 + LOOK_POSTURE_WALKS * walk0, 0, 0.0, plan, walk0, transit0)
    every = max(1, int(round(LOOK_POSTURE_EVERY_RAD / SELF_MOTION_STEP_RAD)))
    cands, kept = [], [q_today]
    for s, q in chain.self_motion(q_today, LOOK_POSTURE_SWING_RAD):
        if int(round(abs(s) / SELF_MOTION_STEP_RAD)) % every:
            continue
        q = np.array(nearest_branch(q, start), float)
        if not chain.within_limits(q, PLANNER_LIMIT_MARGIN_RAD) or any(
            abs(q[i]) > PLANNER_CONTINUOUS_LIMIT_RAD for i in CONTINUOUS_JOINTS
        ):
            continue
        if any(float(np.abs(q - u).max()) < LOOK_POSTURE_DEDUP_RAD for u in kept):
            continue
        kept.append(q)
        cands.append((len(cands) + 1, s, q))
    # ranked by the bound, each posture's walk ending where today's turns the hand down
    R_end, t_end = chain.fk(next(p[-1] for p in reversed(walk_legs(plan)) if len(p)))
    ranked = []
    for number, s, q in cands:
        end = chain.solve_pose(q, t_end, R_end)
        if end is not None:
            bound = _line_bound_s(start, q, TRANSIT_SPEED) + LOOK_POSTURE_WALKS * _line_bound_s(
                q, end, COMPLIANT_FREE_LINE_SPEED
            )
            ranked.append((bound, number, s, q))
    ranked.sort(key=lambda r: r[0])  # stable: a tie keeps self_motion's order
    proven, refused, slowest = 0, {}, 0.0
    for _bound, number, s, q in ranked[:LOOK_POSTURE_PROOFS]:
        if time.monotonic() - t0 + slowest > LOOK_POSTURE_BUDGET_S:
            refused[number] = "not proven: the IK time left (LOOK_POSTURE_BUDGET_S)"
            continue
        t1 = time.monotonic()
        alt, why, _kept = _proven(ctx, cfg, chain, tilt, button, geometry, [float(v) for v in q], True)
        if alt is not None:
            why = posture_refusal(ctx, chain, walk_legs(alt) + [p for _name, p, _speed in alt.stroke.legs], button)
        slowest = max(slowest, time.monotonic() - t1)
        if why is not None:
            refused[number] = why
            continue
        proven += 1
        walk, transit = flown_s(walk_legs(alt), COMPLIANT_FREE_LINE_SPEED), transit_s(start, q)
        est = transit + LOOK_POSTURE_WALKS * walk
        if est < best[0]:
            best = (est, number, s, alt, walk, transit)
    _est, number, s, chosen, walk, transit = best
    if number and best[0] > transit0 + LOOK_POSTURE_WALKS * walk0 - LOOK_POSTURE_GAIN_S:
        number, s, chosen, walk, transit = 0, 0.0, plan, walk0, transit0
    note = {
        "chosen": number,
        "swing_rad": round(s, 2),
        "joint_gap_rad": round(float(np.abs(np.asarray(chosen.look_joints, float) - q_today).max()), 3),
        "walk_s": round(walk, 2),
        "walk_s_today": round(walk0, 2),
        "transit_s": round(transit, 2),
        "transit_s_today": round(transit0, 2),
        "saved_s": round((transit0 + LOOK_POSTURE_WALKS * walk0) - (transit + LOOK_POSTURE_WALKS * walk), 2),
        "candidates": len(cands),
        "proven": proven,
        "refused": {str(k): v for k, v in refused.items()},
        "ik_s": round(time.monotonic() - t0, 3),
    }
    return chosen, note


def _plan_look_to(ctx, q_from, q_look):
    """The look leg from joints `q_from` to the posture `q_look`
    (quickest_look's), planned as a joint goal — full world, TRANSIT_SPEED,
    as _plan_look's — which the client sends on the start's branch and
    takes back only where it ends there (plan_to_joints). Returns (leg,
    None) or (None, why)."""
    try:
        leg, _st = _plan_motion(
            ctx,
            PlanState(joints=list(q_from), chain=0),
            "reach:look",
            ("joints", [float(v) for v in q_look]),
            _full_world(ctx),
            TRANSIT_SPEED,
        )
    except RuntimeError as e:
        return None, _short(e)
    return leg, None


def plan_reach(ctx, cfg, button, q_from, tilts=REACH_TILTS_DEG, depth=False, looks=None, shortest_walk=False):
    """The smallest tilt whose slide and stroke the arm's own kinematics
    can fly as straight lines from its look pose, and whose look pose plans
    (full world, from `q_from`). The way back out retraces both. `depth`
    (the compliant depth press): no tilted stroke — that press never flies
    it — but the hand must turn down from the look pose over the slide and
    stroke from there (way_down); a tilt whose look pose and slide solve but
    whose way down does not is passed over, and appended to `looks` (a
    list) with no `down`. `depth` and `shortest_walk` (the proof before the
    look, from HOME): the tilt proven, its look goes to the posture whose
    walks are quickest where that is LOOK_POSTURE_GAIN_S quicker
    (quickest_look; ReachPlan.posture says which), planned as a joint goal
    (_plan_look_to) — and to today's, as always, where that does not plan.

    Each tilt is proven from the joints the client plans its look pose to
    (client.pose_goal_joints: the arm's own IK) before the planner is asked
    for anything, and only the first that proves is planned, with the
    request its look has always gone with (_plan_look). On 2026-10-28
    14:28:13 four looks were planned, 0.15-0.17 s each, and no way down flew
    from any of them. A look the arm's IK solved is kept in `looks`
    unplanned (look_leg None; plan_shifted_look plans the one that flies).
    A planned look that ends off the joints it was proven from
    (LOOK_END_RAD) is proven again from its end. Where the arm's IK solves
    no look the pose goes to the planner, and its plan is proven from where
    it ends, as always — unless the look stands past everything the arm
    reaches (LOOK_PAST_REACH_M). Returns (ReachPlan, None) or (None,
    why)."""
    from box_opening.runtime import approach
    from box_opening.runtime.client import pose_goal_joints

    chain = approach._chain()
    bx, by, bz = (float(v) for v in button)
    refused = []
    for tilt in tilts:
        geometry = reach_geometry(ctx.model, button, tilt, cfg.travel_m)
        quat, _start, _bottom, look = geometry
        try:
            past = chain.past_reach_m(look, quat)
        except Exception:  # noqa: BLE001 - a kinematics problem must never refuse a look: it is planned as always
            past = None
        if past is not None and past > LOOK_PAST_REACH_M:  # no IK solves it, the arm's own or the planner's
            refused.append("%d deg: the look pose (its wrist %.0f mm past the arm's reach)" % (tilt, 1000 * past))
            continue
        q_look, leg = pose_goal_joints(look, quat, list(q_from)), None  # as _plan_look's PlanState hands it
        if q_look is None:  # the pose to the planner, as the client hands it: proven from where its plan ends
            leg, why = _plan_look(ctx, q_from, look, quat)
            if leg is None:
                refused.append("%d deg: the look pose (%s)" % (tilt, why))
                continue
            q_look = leg.goal_joints
        plan, why, kept = _proven(ctx, cfg, chain, tilt, (bx, by, bz), geometry, q_look, depth)
        posture = None
        if plan is not None and leg is None:  # the first tilt that proves: the one look planned
            if depth and shortest_walk:
                today = plan
                try:
                    plan, posture = quickest_look(ctx, cfg, chain, tilt, (bx, by, bz), geometry, q_from, plan)
                except Exception as e:  # noqa: BLE001 - a kinematics problem never costs the look: today's posture
                    plan, posture = today, {"chosen": 0, "error": _short(e)}
                if posture["chosen"]:
                    # the client takes a joint plan back only where it ends within PLAN_END_RAD (LOOK_END_RAD)
                    # of its goal, so the proof below is never made again from elsewhere unguarded
                    leg, why_leg = _plan_look_to(ctx, q_from, plan.look_joints)
                    if leg is None:  # today's posture, planned as it always was
                        log.info(
                            "  NOTE reach:look at %d deg: the quicker posture's look does not plan (%s) — the look "
                            "goes to today's" % (tilt, why_leg)
                        )
                        posture = dict(
                            posture,
                            chosen=0,
                            swing_rad=0.0,
                            joint_gap_rad=0.0,
                            walk_s=posture["walk_s_today"],
                            transit_s=posture["transit_s_today"],
                            saved_s=0.0,
                            not_planned={str(posture["chosen"]): why_leg},
                        )
                        plan = today
                    q_look = plan.look_joints
            if leg is None:
                leg, why_leg = _plan_look(ctx, q_from, look, quat)
            if leg is None:
                plan, why = None, "the look pose (%s)" % why_leg
            else:
                gap = _joint_gap(leg.goal_joints, q_look)
                if gap > LOOK_END_RAD:
                    log.info(
                        "  NOTE reach:look at %d deg: planned to end %.3f rad from the joints its press was proven "
                        "from — proven again from there" % (tilt, gap)
                    )
                    plan, why, kept = _proven(ctx, cfg, chain, tilt, (bx, by, bz), geometry, leg.goal_joints, depth)
        if plan is None:
            refused.append("%d deg: %s" % (tilt, why))
            if kept is not None and looks is not None:
                kept.look_leg = leg  # None where it was not planned
                looks.append(kept)
            continue
        plan.look_leg = leg
        plan.posture = posture
        return plan, None
    return None, "; ".join(refused)


# Where the compliant depth press does not reach the scene camera's button,
# the look still flies when it reaches the button this far off it — toward
# the arm, or to either side (_shifted_buttons): the aim, not the scene
# camera, puts the button the press goes to. On 2026-10-27 the four aims
# put the button 12-23 mm from the scene camera's (14:33, 15:19, 15:34,
# 16:23), each 11-22 mm to its right and up to 11 mm nearer the arm.
# Refusing on the scene button alone at the edge of reach refuses boxes the
# aim brings into reach; two such were found offline (flatbot's URDF): the
# scene's button at az -22.0 deg, r 0.700 m with the aim's at -23.6, 0.692,
# and at r 0.766, az -15 with the aim's at 0.757, -16.6.
PROOF_SHIFT_M = 0.03
SHIFTED = ("nearer the arm", "to its left", "to its right")  # _shifted_buttons' order, seen from the arm


def _shifted_buttons(button, m):
    """Three buttons at `button`'s height: `m` toward the arm along its
    bearing, and `m` to either side across it (SHIFTED's order)."""
    bx, by, bz = (float(v) for v in button)
    d = math.hypot(bx, by)
    ux, uy = bx / d, by / d
    return [(bx - m * ux, by - m * uy, bz), (bx - m * uy, by + m * ux, bz), (bx + m * uy, by - m * ux, bz)]


def shifted_way_down(ctx, cfg, look, button, m=PROOF_SHIFT_M):
    """For a scene button no tilt presses (plan_reach depth=True): whether
    the press reaches one of _shifted_buttons from `look` — the smallest
    tilt whose look pose and slide solved (plan_reach's `looks`) — the slide
    from its look joints (look_joints: those it was proven from, planned or
    not) straight to that button's start at its tilt, then way_down.
    Returns (SHIFTED's name, None) for the first that does, or (None,
    why)."""
    from box_opening.runtime import approach

    chain = approach._chain()
    q = look.look_joints
    whys = []
    for name, b in zip(SHIFTED, _shifted_buttons(button, m), strict=False):
        quat, start, _bottom, _look = reach_geometry(ctx.model, b, look.tilt_deg, cfg.travel_m)
        slide, why = chain.straight_line(q, start, quat)
        if slide is None:
            whys.append("%.0f mm %s: the slide in (%s)" % (1000 * m, name, why))
            continue
        down, _stroke, why = way_down(ctx, cfg, q, slide, start, quat, b[2])
        if down is None:
            whys.append("%.0f mm %s: the way down (%s)" % (1000 * m, name, why))
            continue
        return name, None
    return None, "; ".join(whys)


def plan_shifted_look(ctx, cfg, looks, button, q_from, m=PROOF_SHIFT_M):
    """For a scene button no tilt presses (plan_reach depth=True): the press
    to one of _shifted_buttons from the smallest tilt of its `looks`
    (shifted_way_down, as it always was asked), and that look's leg,
    planned from `q_from` only once the press reaches — plan_reach plans
    none of them, so a box nothing presses costs the planner nothing
    (14:28:12 2026-10-28: four looks planned, none pressed from). Where the
    press reaches but the look does not plan, the next look is asked in its
    place, as plan_reach passes over a look that does not plan; where the
    press does not reach, the search ends there, unplanned. Until
    2026-10-28 a look that did not plan was never kept, so the next one was
    asked: a kept look that would not have planned now stops the run where
    a larger tilt's press might have flown — knowing would cost a plan on
    every such stop. One planned to end off the joints it was proven from
    (LOOK_END_RAD) is proven again from where it ends. Returns (the
    ReachPlan with its look leg planned, SHIFTED's name, None) or (None,
    None, why)."""
    from box_opening.runtime import approach

    whys = []
    for look in looks:
        near, why = shifted_way_down(ctx, cfg, look, button, m)
        if near is None:
            return None, None, "; ".join(whys + ["at %d deg: %s" % (look.tilt_deg, why)])
        leg = look.look_leg
        if leg is None:
            leg, why = _plan_look(ctx, q_from, look.look_xyz, look.quat)
            if leg is None:
                whys.append(
                    "at %d deg the press reaches it %.0f mm %s, but the look pose does not plan (%s)"
                    % (look.tilt_deg, 1000 * m, near, why)
                )
                continue
        gap = _joint_gap(leg.goal_joints, look.look_joints)
        if gap > LOOK_END_RAD:
            log.info(
                "  NOTE reach:look at %d deg: planned to end %.3f rad from the joints its press was proven from — "
                "proven again from there" % (look.tilt_deg, gap)
            )
            q = [float(v) for v in leg.goal_joints]
            slide, why = approach._chain().straight_line(q, look.start_xyz, look.quat)
            if slide is None:
                whys.append("at %d deg, planned %.3f rad off: the slide in (%s)" % (look.tilt_deg, gap, why))
                continue
            look = dataclasses.replace(look, slide_pts=slide, look_joints=q)
            near, why = shifted_way_down(ctx, cfg, look, button, m)
            if near is None:
                return None, None, "; ".join(whys + ["at %d deg, planned %.3f rad off: %s" % (look.tilt_deg, gap, why)])
        return dataclasses.replace(look, look_leg=leg), near, None
    return None, None, "; ".join(whys)


@dataclass
class WayIn:
    """The compliant depth press's way in for the aimed button, from where
    the arm stands (solve_way_in): the MOVE to its look pose ([] when there),
    reach_from_here's ReachPlan, how the hand turns down (WayDown), the
    slide before the turn (None when the way down is walked) and the press
    stroke (StrokePlan)."""

    move: list
    plan: ReachPlan
    down: object
    slide: list
    stroke: object


def solve_way_in(ctx, cfg, button, tilt, q_from):
    """The aimed button's whole way in at `tilt` from joints `q_from`, before
    the arm moves toward it: reach_from_here's lines (the move to the look
    pose — one the runner flies, depth_press.line_refusal — and the slide),
    then way_down from the look pose. Returns (WayIn, None, None) or (None,
    why, kind): kind 'lines' when the lines do not fly (the planner may plan
    the move instead), 'down' when the way down does not."""
    from box_opening.runtime import approach
    from box_opening.tasks import depth_press as dp

    move, plan, why = reach_from_here(ctx, cfg, button, tilt, need_stroke=False, q=q_from)
    if plan is None:
        return None, why, "lines"
    why = dp.line_refusal(approach._chain(), move)
    if why is not None:
        return None, "the move to its look pose (%s)" % why, "lines"
    down, stroke, why = way_down(
        ctx, cfg, move[-1] if move else q_from, plan.slide_pts, plan.start_xyz, plan.quat, float(button[2])
    )
    if down is None:
        return None, "the way down (%s)" % why, "down"
    plan.down, plan.stroke = down, stroke
    if down.slide is not None:  # it turns lower down: the slide carries on to there
        plan.slide_pts, plan.start_xyz = down.slide, down.start_xyz
    return WayIn(move, plan, down, None if down.walked else plan.slide_pts, stroke), None, None


def reach_from_here(ctx, cfg, button, tilt, need_stroke=True, q=None):
    """The reach for `button` at `tilt` as plan-free lines from where the arm
    stands (`q`, else the arm's joints): a MOVE to its look pose at its
    attitude, then the slide and the stroke (not solved without
    `need_stroke`: the depth press flies its own). Returns (move points —
    [] when already there —, ReachPlan with no look_leg, None) or (None,
    None, why)."""
    from box_opening.runtime import approach

    chain = approach._chain()
    quat, start, bottom, look = reach_geometry(ctx.model, button, tilt, cfg.travel_m)
    q = ctx.client.joints() if q is None else [float(v) for v in q]
    _R, t = chain.fk(q)
    if float(np.linalg.norm(np.asarray(look, float) - np.asarray(t, float))) < LINE_MIN_M:
        move, q_look = [], q
    else:
        move, why = chain.straight_line(q, look, quat)
        if move is None:
            return None, None, "the move to its look pose (%s)" % why
        q_look = move[-1]
    slide, why = chain.straight_line(q_look, start, quat)
    if slide is None:
        return None, None, "the slide in (%s)" % why
    stroke = None
    if need_stroke:
        stroke, why = chain.straight_line(slide[-1], bottom, quat)
        if stroke is None:
            return None, None, "the stroke (%s)" % why
    return (
        move,
        ReachPlan(int(tilt), quat, tuple(float(v) for v in button), look, start, bottom, None, slide, stroke),
        None,
    )


def line_leg(ctx, name, pts, quat, speed, guard, world, verify=None, chain=0):
    """A plan-free straight-line leg through joint waypoints `pts` (the arm's
    own kinematics, runtime/kinematics.straight_line). The runner never
    re-plans it (Leg.plan_free)."""
    from box_opening.runtime import approach

    _R, t = approach._chain().fk(pts[-1])
    world_name, world_path = world
    return Leg(
        name=name,
        kind=Kind.MOTION,
        traj=approach.line_trajectory(ctx.client.joint_names, pts),
        speed=float(speed),
        guard=guard,
        world=world_name,
        world_path=str(world_path),
        chain=chain,
        target=("pose", [float(v) for v in t], list(quat)),
        goal_joints=[float(v) for v in pts[-1]],
        verify=verify,
        plan_s=0.0,
        plan_free=True,
    )


def way_in_legs(ctx, cfg, way):
    """The compliant depth press's way in (`way`, solve_way_in's) as the
    legs of ONE execution, in this order, the empty ones left out:
    'reach:re-aim' (way.move, at the plan's attitude); 'reach:slide-in', the
    walk turning down when the way down is walked, else the slide at the
    plan's attitude, then 'depth:turn' in place; then the press stroke
    (way.stroke.legs: 'depth:approach' at press_speed, 'depth:press' at
    TOUCH_SPEED). The free-air lines fly at free_line_speed (a compliant
    run's COMPLIANT_FREE_LINE_SPEED). Each starts at the last waypoint of
    the one before — solve_way_in chains them — and every one is unguarded
    under depth_press.COMPLIANT_GAINS (compliant_line): the one gains spec
    legs.can_merge asks of one execution, and the spring the press is
    modelled on. `cfg` is the press config the stroke was solved with (its
    press_speed is in way.stroke)."""
    from box_opening.tasks import depth_press as dp

    # Their world is a label: a plan-free line is never planned, re-planned
    # or collision-checked against it (line_leg builds it, Runner._replan_group
    # refuses plan-free legs, push_name only writes a YAML). "full" only
    # satisfies Runner._refusal's spec §6 rule that an unguarded leg faster
    # than CONTACT_SPEED be labelled full or bench.
    world = _full_world(ctx)
    fast = free_line_speed(ctx)
    lines = [("reach:re-aim", way.move, way.plan.quat, fast)]
    if way.down.walked:
        lines.append(("reach:slide-in", way.down.pts, way.down.quat, fast))
    else:
        lines += [("reach:slide-in", way.slide, way.plan.quat, fast), ("depth:turn", way.down.pts, way.down.quat, fast)]
    lines += [(name, pts, way.stroke.quat, speed) for name, pts, speed in way.stroke.legs]
    return [
        dp.compliant_line(ctx, name, pts, quat, speed, world)
        for name, pts, quat, speed in lines
        if pts is not None and len(pts) > 0
    ]


def _wrist_gravity():
    from box_opening.runtime import approach

    return approach._gravity().wrist_holding


def tip_load_at(q):
    """The wrist joints' holding torque per newton pulled straight down at
    the fingertip's contact point, with the arm at `q`."""
    from box_opening.runtime import approach

    return [float(v) for v in approach._gravity().tip_load(q, CONTACT_TOOL)]


def touch_leg(ctx, cfg, pts, quat, world, reference):
    """The TOUCH as a straight line down: stops at first contact
    (cfg.contact_nm on the compensated residual, against the free-air
    `reference` read at the stroke's start). Every trip is a surface found
    — how far above the button is touch_verdict's to judge from where the
    fingertip stopped; no trip found no surface."""
    guard = GuardSpec(touch_nm=cfg.contact_nm, trip="touch", gravity=_wrist_gravity(), reference=reference)

    def verify(v):
        if v.outcome == "touch":
            peak = "" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak
            return True, "surface found%s" % peak
        if v.outcome == "arrived":
            return False, "no surface within %.0f mm below the aimed button" % (cfg.travel_m * 1000)
        return False, "touch %s" % v.outcome

    return line_leg(ctx, "press:down", pts, quat, cfg.press_speed, guard, world, verify)


def drag_guard(reference, hang_k=None):
    """Free air, nothing to meet: DRAG_NM of compensated residual on any
    wrist joint is a strike, and (with `hang_k`) HANG_N of pull at the
    fingertip is the box coming along."""
    return GuardSpec(
        touch_nm=DRAG_NM,
        trip="obstruction",
        gravity=_wrist_gravity(),
        reference=reference,
        hang=None if hang_k is None else (list(hang_k), HANG_N),
    )


def unload_guard(model, reference, hang_k):
    """Leaving the button: the leg starts loaded (the touch's contact), so
    only the push's own backstop bounds the residual — but any PULL at the
    fingertip past HANG_N, against the free-air `reference`, stops it."""
    return GuardSpec(
        touch_nm=float(model.touch_nm),
        trip="obstruction",
        gravity=_wrist_gravity(),
        reference=reference,
        hang=(list(hang_k), HANG_N),
    )


def compliant(ctx):
    """A compliant run (PlannerClient.compliant): impedance, nothing stops it."""
    return bool(getattr(ctx.client, "compliant", False))


def let_go(ctx):
    """A compliant stop lets go: the driver's joint spring holds where the
    arm stands (PlannerClient.release), no longer pulling it toward the
    reference it was stopped at — below the button, or the next point of a
    path cut short. Only where the arm is loaded or was stopped part-way:
    between legs in free air the reference is a leg's end, the arm at rest
    there, and a release would only add its own sag. A way home from the
    box that did not get home lets go: a leg of it stopped part-way, or,
    where a plan was refused instead, that sag is all it costs. A client
    with no release lets nothing go."""
    getattr(ctx.client, "release", lambda: None)()


def let_go_part_way(ctx, res):
    """let_go after a compliant runner.run (`res`, its results) that failed
    part-way: the runner flew its last execution and the driver reported
    progress on it, so the springs pull on toward its next point, its
    feed-forward on. One that did not start — the runner's gates, the
    client's own or the driver refused it, no progress — left the arm at
    rest at a leg's end: nothing to let go of, as _fly_way_in's way in that
    did not start. Nor in a guarded run, whose POSITION goals hold where
    they stopped. Called after the run returns, never with a goal in
    flight."""
    last = res[-1] if res else None  # the runner returns at the first leg that fails
    if compliant(ctx) and last is not None and not last.ok and last.flown_traj is not None and bool(last.progress):
        let_go(ctx)


def gains_text(profile):
    """A driver gains profile, or a custom spec (constants.CustomGains), as
    the console names it, e.g. "RIGID (Kq 400/400 N·m/rad on joints 1-4/5-7,
    zeta 1.0)": a custom spec's zeta too, its own, since one name has flown
    more than one (RIGID at 0.7 until 2026-10-29)."""
    p = profile if isinstance(profile, CustomGains) else int(profile)
    kq = IMPEDANCE_KQ.get(p)
    name = IMPEDANCE_NAMES.get(p, "gains profile %s" % (p,))
    if kq is None:
        return name
    zeta = ", zeta %s" % (p.zeta,) if isinstance(p, CustomGains) else ""
    return "%s (Kq %g/%g N·m/rad on joints 1-4/5-7%s)" % (name, kq[0], kq[4], zeta)


def free_guard(ctx, reference, hang_k=None):
    """drag_guard for a free-air leg — none in a compliant run, which stops
    for nothing it reads."""
    return None if compliant(ctx) else drag_guard(reference, hang_k)


def free_line_speed(ctx):
    """The cruise fraction of a plan-free line in free air, nothing under
    the hand but the lid it may carry: COMPLIANT_FREE_LINE_SPEED in a
    compliant run, which no guard has to stop in time and whose every
    motion the runner re-times under the wrist's ceiling and meets at
    COMPLIANT_JUNCTION_SPEED — read from the same flag (runtime/runner.py
    Runner.retime_params), so no line flies at it without them; else
    FREE_LINE_SPEED, a guarded run's."""
    return COMPLIANT_FREE_LINE_SPEED if compliant(ctx) else FREE_LINE_SPEED


def residual(gravity, ref, now):
    """What gravity does not explain between two (efforts, joints) readings."""
    return [
        (float(e1) - float(e0)) - (float(h1) - float(h0))
        for e0, e1, h0, h1 in zip(ref[0], now[0], gravity(ref[1]), gravity(now[1]), strict=False)
    ]


def down_force(k, r):
    """The residual `r` read as a pull straight down at the fingertip (N)."""
    kk = sum(float(v) * float(v) for v in k)
    return 0.0 if kk <= 0.0 else sum(float(a) * float(b) for a, b in zip(k, r, strict=False)) / kk


def gravity_mismatch(gravity, a, b):
    """Why the gravity model does not describe this arm, or None: the wrist
    joints' measured change between two rest readings `a` -> `b` against the
    model's. A sign error is off by twice the change, a wrong URDF or a
    payload by its own size."""
    r = residual(gravity, a, b)
    dh = [float(h1) - float(h0) for h0, h1 in zip(gravity(a[1]), gravity(b[1]), strict=False)]
    for j, (rj, dj) in enumerate(zip(r, dh, strict=False)):
        if abs(rj) > GRAVITY_CHECK_NM + GRAVITY_CHECK_FRAC * abs(dj):
            return "joint_%d's effort changed %+.2f Nm where the arm's gravity model says %+.2f" % (j + 4, dj + rj, dj)
    return None


def judge_box_moved(first_xy, again_xy):
    """The press moved the box by this much (m), from two aims of its button."""
    return math.hypot(float(again_xy[0]) - float(first_xy[0]), float(again_xy[1]) - float(first_xy[1]))


def oblique_aim_available():
    """Whether this install's wrist aim takes the reach-out's priors (the
    oblique aim, perception/button_aim): without it the look pose is flown
    for nothing."""
    import inspect

    from box_opening.perception.button_aim import ButtonAimer

    params = inspect.signature(ButtonAimer).parameters
    return "lid_z" in params and "prior_xy" in params


# The tilted look's own budget and fix (press_demo.STAGING_FIX_S and
# button_aim.MIN_HITS are the straight-down aim's, where most frames hit).
# Flatbot 2026-10-27: three agreeing hits took 14, 19 and 25 frames
# (1.38-1.78 s), and at 15:19 the first hit did not agree, so three more
# were needed. A compliant run aimed the moment the look leg arrived, with no
# at-rest read before it: 16:23's 1.78 s included the settle and left 0.72 s
# of the 2.5, and a timeout ends the run with nothing pressed. The aim
# returns on its fix: the longer budget is spent only where it would have
# timed out. The wait for a still arm before it (_aim) is its own, at most
# client.STILL_TIMEOUT_S.
REACH_AIM_S = 4.0
# two agreeing hits among the last four (button_aim.agreeing_fix), as
# depth_press.look's LOOK_HITS: one outlier no longer costs three hits
REACH_AIM_HITS = 2


def _aim(node, watcher, yaw, lid_z, prior_xy, runner, tag):
    """(got, status, the knob's height above its lid in mm, the press point
    as the wrist measured it) from aim_button_at_staging, within REACH_AIM_S
    and on REACH_AIM_HITS agreeing hits, got's point moved AIM_FORWARD_M
    out from the arm (aim_forward). got's z is raised by the wrist's
    table offset (button_aim.ButtonAimer.fix): the height the arm and the
    scene camera are compared with. The measured point is the one the
    wrist's own readings are judged against — the lid after the press, the
    depth press's depth from over the box — so the wrist's own error
    cancels, as it did before the raise (flatbot 2026-10-27 15:19: the aim
    14 mm off the wrist's depth under the fingers measured, 36 mm raised).

    The arm stands still first (runner.wait_still, noted 'still' before
    `tag`): the aim's own still test passes frames from an arm creeping
    after the look (16:23 2026-10-28: its first two hits read the knob
    -7.2 and -5.5 mm above its lid, the pair that agreed +1.1 and +0.7).
    Its first frame may be one shot just before the wait passed (on 15:45's
    looks, in the creep's last 0.1-0.15 s, the tool under 15 mm/s). A
    recheck right after an aim, nothing flown between, finds it still at
    once (client.wait_still)."""
    from box_opening.tasks.press_demo import aim_button_at_staging

    if runner is not None:
        runner.wait_still(tag)
    seen = {}
    got, status = aim_button_at_staging(
        node,
        watcher,
        yaw,
        timeout_s=REACH_AIM_S,
        runner=runner,
        tag=tag,
        out=seen,
        lid_z=lid_z,
        prior_xy=prior_xy,
        min_hits=REACH_AIM_HITS,
    )
    measured = seen.get("measured_xyz", None if got is None else got[0])
    if got is not None:
        got = (aim_forward(got[0]), got[1])  # what the hand goes by; `measured` stays the wrist's own
    return got, status, seen.get("button_up_mm"), measured


def aim_forward(xyz):
    """The wrist's aim `xyz` moved AIM_FORWARD_M farther from the arm along
    its bearing from arm_mount, its height kept: the cameras' error."""
    x, y, z = (float(v) for v in xyz)
    d = math.hypot(x, y)
    return (x + AIM_FORWARD_M * x / d, y + AIM_FORWARD_M * y / d, z)


def _lid_z(button_z, up_mm):
    return None if up_mm is None else float(button_z) - float(up_mm) / 1000.0


def _stopped_by(res):
    """What stopped a guarded free-air leg, for its STOP line: a trip is
    something met; anything else says itself (15:34 2026-10-27: "met
    something" printed for a guard that only lost its stream)."""
    last = res[-1]
    if last.outcome == "touch":
        return "met something"
    return "stopped (%s)" % (last.detail or last.outcome)


def _settled(ctx, where):
    got = ctx.client.settled_efforts()
    if got is None:
        log.info(
            "[press_demo] STOP: no steady joint efforts at %s — the reach-out's guards cannot be referenced" % where
        )
    return got


def _line(ctx, xyz, quat):
    """Joint waypoints for a straight line from where the arm is, or
    ([], None) when it is already there."""
    from box_opening.runtime import approach

    chain = approach._chain()
    q = ctx.client.joints()
    _R, t = chain.fk(q)
    if float(np.linalg.norm(np.asarray(xyz, float) - np.asarray(t, float))) < LINE_MIN_M:
        return [], None
    return chain.straight_line(q, xyz, quat)


REUSE_RAD = 1e-3  # the arm this close (every joint) to where a solved line starts: it is flown as solved


def _line_from(ctx, pts, xyz, quat):
    """`pts` (a line solved earlier, ending at `xyz`) when the arm stands at
    its start, else _line from where it stands. The runner refuses a
    plan-free line that no longer starts where the arm is all the same."""
    if pts and float(np.abs(np.asarray(ctx.client.joints(), float) - np.asarray(pts[0], float)).max()) < REUSE_RAD:
        return pts, None
    return _line(ctx, xyz, quat)


def _home_legs(ctx, cfg, start=None):
    """Home to the rest pose from where the arm stands — or from joints
    `start`, where it will stand (plan_way_out: the clear's end, planned
    while the way in flies) — in the full world.
    A joint too near its limit for the planner to start from gets home_arm's
    plan-free step along the joint line toward the rest pose first, the
    shortest one home plans from (primitives.ClearMarginFree): flatbot
    2026-10-27 14:33, the depth press ended with arm_joint_6 0.043 rad from
    its limit, home was refused START_IN_LIMIT_MARGIN, and the arm held over
    the box until home_arm stepped it clear. Raises the last refusal."""
    from box_opening.tasks.press_demo import rest_joints

    rest = list(rest_joints(cfg))
    q = ctx.client.joints() if start is None else start
    state = PlanState(joints=[float(v) for v in q], chain=0)
    clear = ClearMarginFree(rest)
    near = clear.near(ctx, state.joints)
    if not near:
        leg, _ = _plan_motion(ctx, state, "home", ("joints", rest), _full_world(ctx), TRANSIT_SPEED)
        return [leg]
    log.info(
        "[press_demo] %s: too near for the planner to start from — trying plan-free steps along the joint line "
        "toward the rest pose, shortest first" % ", ".join("%s is %.3f rad from its limit" % jr for jr in near)
    )
    last = None
    for step, at, info in clear.steps(ctx, state):
        try:
            leg, _ = _plan_motion(ctx, at, "home", ("joints", rest), _full_world(ctx), TRANSIT_SPEED)
        except RuntimeError as e:
            last = e
            continue
        if step:
            d = info["tool_move"]
            log.info(
                "[press_demo] a plan-free step %.0f%% of the way to the rest pose first (the largest joint %.3f rad, "
                "the tool %+.0f mm up and %.0f mm across; obstruction-guarded)"
                % (100 * info["fraction"], info["largest_rad"], 1000 * d[2], 1000 * math.hypot(d[0], d[1]))
            )
        return step + [leg]
    raise RuntimeError(
        "no plan-free step toward the rest pose of up to %.2f rad per joint lets the planner plan home "
        "(last: %s) — jog the arm clear by hand" % (CLEAR_MARGIN_MAX_STEP_RAD, last)
    )


def _back_off(ctx, runner, ref=None, why=None, lets_go=True):
    """Straight back toward the arm by REACH_PULLBACK_M, level, at the
    hand's own attitude — guarded, like the slide out: out of the
    padded cuboid of a box that is not where the arm's pose was planned
    around, so the planner takes the start again. Guarded against `ref`, a
    free-air reading the caller already holds, else one read here. `why`:
    the planner's refusal of home from here, printed as it said it — not
    always the box's padded volume (INVALID_START_STATE_JOINT_LIMITS takes
    the back-off too). False when it cannot be flown or met something; a
    compliant one stopped part-way lets go there (let_go_part_way) unless
    `lets_go` is False (_go_home's)."""
    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import mat_to_quat_xyzw

    chain = approach._chain()
    q = ctx.client.joints()
    R, t = chain.fk(q)
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    d = math.hypot(bx, by)
    target = [float(t[0]) - REACH_PULLBACK_M * bx / d, float(t[1]) - REACH_PULLBACK_M * by / d, float(t[2])]
    quat = mat_to_quat_xyzw(R)
    pts, why_line = chain.straight_line(q, target, quat)
    if pts is None:
        log.info("[press_demo] the way back off toward the arm is not a straight line the arm can fly (%s)" % why_line)
        return False
    if not compliant(ctx):
        ref = ref or _settled(ctx, "where the arm stands")
        if ref is None:
            return False
    said = (
        "the planner sees the arm inside the box's padded volume"
        if why is None
        else "the planner refused home from here (%s)" % _short(why)
    )
    log.info("[press_demo] %s — backing %.0f cm straight off toward the arm first" % (said, 100 * REACH_PULLBACK_M))
    # from free air only (_go_home's callers), level and away from the box —
    # at FREE_LINE_SPEED in a compliant run too, not free_line_speed: it flies
    # only where the planner refused the start, the arm perhaps inside the
    # box's padded volume
    res = runner.run(
        [line_leg(ctx, "reach:back-off", pts, quat, FREE_LINE_SPEED, free_guard(ctx, ref), _full_world(ctx))],
        execute=True,
    )
    if lets_go:
        let_go_part_way(ctx, res)
    return all(r.ok for r in res)


def _planned_ahead(ctx, runner, legs):
    """`legs`, home planned before the arm stood here (_go_home's `ahead`),
    where the arm stands within DRIFT_REPLAN_RAD of where they start — the
    gap the short way round, as the start gate in front of the driver
    measures it (runtime/driver.start_gap_rad) — as the runner flies any
    planned leg from there. Else None, said on the console, and _go_home
    plans home from where the arm stands: a plan from somewhere else is
    never flown, and the runner's own re-plan of one would skip _home_legs'
    clear-margin step and _go_home's back-off. In the run log either way
    (home_ahead). None for none."""
    from box_opening.runtime.driver import start_gap_rad

    if not legs:
        return None
    off = start_gap_rad(ctx.client.joints(), legs[0].traj)
    flown = off <= DRIFT_REPLAN_RAD
    runner.note("home_ahead", flown=flown, off_rad=round(off, 4))
    if not flown:
        log.info(
            "[press_demo] the arm stands %.3f rad from where home was planned from (gate %.2f) — home is "
            "planned again from here" % (off, DRIFT_REPLAN_RAD)
        )
    return legs if flown else None


def _go_home(ctx, cfg, runner, why, ref=None, lets_go=True, ahead=None):
    """Home from free air (the look pose, or where the run started), planned
    in the full world around the box where it now is. A start the planner
    refuses INVALID_START — inside the box's padded cuboid (the box not
    where the pose was planned around: the scene camera 85 mm off,
    2026-10-22), or INVALID_START_STATE_JOINT_LIMITS, which the same test
    takes — backs straight off toward the arm first (guarded against `ref`
    when given), the refusal printed. Every refusal is in the run log.
    False — the arm holds — when the plan is refused or a move fails. A
    compliant run's back-off or home that stopped part-way lets go there
    (let_go_part_way), unless `lets_go` is False: _home_from_over_box's
    callers let go of a home that did not get there themselves. `ahead`:
    home's legs planned before the arm stood here — the full run's, from
    the look pose while the way back to it flew (reach_lid_off._stop_home)
    — flown as planned where the arm stands where they start
    (_planned_ahead), no planner call at rest; else home is planned here."""
    if why:
        log.info("[press_demo] %s — returning home" % why)
    try:
        legs = _planned_ahead(ctx, runner, ahead) or _home_legs(ctx, cfg)
    except RuntimeError as e:
        runner.note("home_refused", why=str(e)[:300])
        if "INVALID_START" not in str(e) or not _back_off(ctx, runner, ref, why=str(e), lets_go=lets_go):
            log.info("[press_demo] home plan refused (%s) — the arm holds; home it with sheppy's home_arm" % _short(e))
            return False
        try:
            legs = _home_legs(ctx, cfg)
        except RuntimeError as e2:
            runner.note("home_refused", why=str(e2)[:300], after="reach:back-off")
            log.info("[press_demo] home plan refused (%s) — the arm holds; home it with sheppy's home_arm" % _short(e2))
            return False
    res = runner.run(legs, execute=True)
    if lets_go:
        let_go_part_way(ctx, res)
    return all(r.ok for r in res)


def hand_lean(R):
    """How far the hand at attitude `R` (tool_frame in the world) leans off
    straight down (rad), as depth_press.turn_down measures it."""
    return math.acos(float(np.clip(-float(np.asarray(R, float)[2, 2]), -1.0, 1.0)))


def clear_targets(ctx, R, t):
    """Where a compliant run's clear takes the tool from (R, t) over the
    box, in the order they are tried: HOME_CLEAR_M + CLEAR_EXTRA_M over the
    box's padded cuboid, higher by what the hand's lean hangs the planner's
    lowest gripper spheres lower (HAND_SPHERE_OFF_AXIS_M sin(lean)) — so
    they clear it by 17 mm or more at any lean — and never lower than the
    hand; and back toward the arm along the box's bearing, CLEAR_BACK_M,
    then REACH_PULLBACK_M: where 16:23's planner took home after its 15 cm
    back-off."""
    from box_opening.worlds import planned_top_z

    want = (
        planned_top_z(ctx.model, ctx.cpose)
        + HOME_CLEAR_M
        + HAND_SPHERE_OFF_AXIS_M * math.sin(hand_lean(R))
        + CLEAR_EXTRA_M
    )
    z = max(float(t[2]), want)
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    d = math.hypot(bx, by)
    return [[float(t[0]) - b * bx / d, float(t[1]) - b * by / d, z] for b in (CLEAR_BACK_M, REACH_PULLBACK_M)]


def _near_a_limit(ctx, q):
    """The joints `q` holds too near a limit for home to plan from without
    _home_legs' clear-margin step (primitives.ClearMarginFree.near, whose
    goal does not enter it), as text; "" when none is."""
    return ", ".join("%s %.3f rad from its limit" % jr for jr in ClearMarginFree(q).near(ctx, q))


def _clear_up_and_back(ctx, runner, R, t):
    """_clear_over_box for a compliant run: up and back toward the arm to
    the first of clear_targets whose straight line flies and leaves every
    joint room for home to plan from (_near_a_limit), else the first whose
    line flies — unguarded, at free_line_speed. Flatbot's URDF, 15:19's box
    pressed from the scene camera's top (0.3111), turned down: the 50 mm
    line ends with arm_joint_6 0.160 rad from its limit, the 150 mm one
    0.368. True when there (or already high enough), None when no line
    flies (the back-off is the fallback), False — the arm holds — when the
    leg failed."""
    from box_opening.runtime.kinematics import mat_to_quat_xyzw

    targets = clear_targets(ctx, R, t)
    if float(t[2]) >= targets[0][2]:
        return True
    quat = mat_to_quat_xyzw(R)
    up, lean = targets[0][2] - float(t[2]), math.degrees(hand_lean(R))
    chosen, whys = None, []
    for target in targets:
        back = math.hypot(target[0] - float(t[0]), target[1] - float(t[1]))
        pts, why = _line(ctx, target, quat)
        if pts is None:
            whys.append("%.0f mm back: %s" % (1000 * back, why))
            continue
        if not pts:
            return True
        near = _near_a_limit(ctx, pts[-1])
        if chosen is None or not near:
            chosen = (pts, back)
        if not near:
            break
        whys.append("%.0f mm back: %s at its end" % (1000 * back, near))
    if chosen is None:
        runner.note("depth_clear", flown=False, up_mm=round(1000 * up, 1), lean_deg=round(lean, 1), why="; ".join(whys))
        return None
    pts, back = chosen
    # the console says why a nearer target was passed over, as the run log does
    log.info(
        "[press_demo] up %.0f mm and %.0f mm back toward the arm (the hand leaning %.0f deg), clear of the box's "
        "padded volume, before home%s" % (1000 * up, 1000 * back, lean, " (%s)" % "; ".join(whys) if whys else "")
    )
    # up and away from the box, toward the look pose the hand came in from
    res = runner.run(
        [line_leg(ctx, "depth:clear", pts, quat, free_line_speed(ctx), None, _full_world(ctx))], execute=True
    )
    if any(not r.ok for r in res):
        log.info("[press_demo] STOP: the way up from over the box %s — the arm holds; check the box" % _stopped_by(res))
        return False
    runner.note(
        "depth_clear",
        flown=True,
        up_mm=round(1000 * up, 1),
        back_mm=round(1000 * back, 1),
        lean_deg=round(lean, 1),
        why="; ".join(whys) or None,
    )
    return True


@dataclass
class WayOut:
    """A compliant reach-out's way home from the stroke's start, planned
    while its way in flies (plan_way_out): the clear — joint waypoints up
    and back toward the arm to one of clear_targets, `back_m` back, the
    hand's attitude `quat` held — and the home legs planned from its end
    (_home_legs). `passed`: why a nearer target was passed over, or None."""

    clear: list
    quat: list
    home: list
    back_m: float = 0.0
    passed: str = None


def plan_way_out(ctx, cfg, q_from, quat):
    """The way home from joints `q_from` — the stroke's start, where the
    rise back up from the hold ends (depth_press.rise_back_up) — planned
    before the arm is there: _fly_way_in hands it to the runner as the way
    in's lookahead, so the planner is asked while the way in flies and
    never after the hold (16:23 2026-10-27: home refused INVALID_START after
    the clear, then a 3.54 s back-off). The clear goes to the first of
    clear_targets(fk(q_from)) whose straight line flies and leaves every
    joint room for home to plan from, else the first whose line flies, as
    _clear_up_and_back picks (15:19's box turned down: the 50 mm line ends
    with arm_joint_6 0.160 rad from its limit, the 150 mm one 0.368); home
    is planned from its end (_home_legs, its clear-margin step included),
    and a target whose home does not plan is passed over. Returns (WayOut,
    None) or (None, why)."""
    from box_opening.runtime import approach

    chain = approach._chain()
    q_from = [float(v) for v in q_from]
    R, t = chain.fk(q_from)
    whys, near_limit = [], []

    def home_from(pts, back):
        try:
            home = _home_legs(ctx, cfg, start=pts[-1])
        except RuntimeError as e:
            whys.append("%.0f mm back: home refused (%s)" % (1000 * back, _short(e)))
            return None
        return WayOut(pts, list(quat), home, back, "; ".join(whys) or None)

    for target in clear_targets(ctx, R, t):
        back = math.hypot(target[0] - float(t[0]), target[1] - float(t[1]))
        pts, why = chain.straight_line(q_from, target, quat)
        if pts is None:
            whys.append("%.0f mm back: %s" % (1000 * back, why))
            continue
        pts = [[float(v) for v in q] for q in pts]
        near = _near_a_limit(ctx, pts[-1])
        if near:
            whys.append("%.0f mm back: %s at its end" % (1000 * back, near))
            near_limit.append((pts, back))
            continue
        way = home_from(pts, back)
        if way is not None:
            return way, None
    for pts, back in near_limit:
        way = home_from(pts, back)
        if way is not None:
            return way, None
    return None, "; ".join(whys) or "no clear target"


def _clear_over_box(ctx, runner, ref):
    """From over the box, before its home is planned: straight up until the
    tool stands HOME_CLEAR_M above the box's cuboid as the planner pads it
    (worlds.planned_top_z) — the gripper base's collision spheres reach 48
    mm below tool_frame straight down, and a start among them is refused
    (20:04: INVALID_START twice, the arm held over the box). Slow, drag- and
    hang-guarded against the free-air `ref`: the hang's reference,
    tip_load_at the line's start, holds only along a short vertical line.
    A compliant run goes up as far as the hand's lean needs and back toward
    the arm instead (_clear_up_and_back). True when there (or already
    clear), None when no straight line flies from here (the back-off is the
    fallback), False — the arm holds — when it met something."""
    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import mat_to_quat_xyzw
    from box_opening.worlds import planned_top_z

    R, t = approach._chain().fk(ctx.client.joints())
    if compliant(ctx):
        return _clear_up_and_back(ctx, runner, R, t)
    want = planned_top_z(ctx.model, ctx.cpose) + HOME_CLEAR_M
    if float(t[2]) >= want:
        return True
    quat = mat_to_quat_xyzw(R)
    pts, why = _line(ctx, [float(t[0]), float(t[1]), want], quat)
    if pts is None:
        runner.note("depth_clear", flown=False, up_mm=round(1000 * (want - float(t[2])), 1), why=why)
        return None
    if not pts:
        return True
    log.info("[press_demo] up %.0f mm, clear of the box's padded volume, before home" % (1000 * (want - float(t[2]))))
    res = runner.run(
        [
            line_leg(
                ctx, "depth:clear", pts, quat, CONTACT_SPEED, drag_guard(ref, tip_load_at(pts[0])), _full_world(ctx)
            )
        ],
        execute=True,
    )
    if any(not r.ok for r in res):
        log.info("[press_demo] STOP: the way up from over the box %s — the arm holds; check the box" % _stopped_by(res))
        return False
    return True


def _home_from_over_box(ctx, cfg, runner, result):
    """Home after a free depth press (DepthPress.free): clear of the box's
    padded volume first, then planned once, the back-off left as the
    fallback. False — the arm holds, and a compliant run's caller lets go
    (let_go) — when it did not get there."""
    ref = result.ref or (None if compliant(ctx) else _settled(ctx, "over the box"))
    if ref is not None or compliant(ctx):
        clear = _clear_over_box(ctx, runner, ref)
        if clear is False:
            return False
    from box_opening.tasks.depth_press import home_why

    return _go_home(ctx, cfg, runner, home_why(result), ref=ref, lets_go=False)


def _unload_stopped(res):
    """What a stopped unload means, from its guard's report."""
    trip = res[-1].trip or {}
    if trip.get("why") == "hang":
        return (
            "something pulled down on the hand (%.1f N — the box caught on the fingers?) — the arm holds; check the box"
            % (trip.get("down_n") or 0.0)
        )
    if trip.get("why") == "torque":
        return "the hand is still pressing on the box (%.1f Nm on the way up) — the arm holds ON it" % (
            trip.get("dev_nm") or 0.0
        )
    return "the way up stopped (%s) — the arm holds" % (res[-1].detail or res[-1].outcome)


def _still_free(ctx, runner, ref_start, where, strict):
    """At rest, nothing on the hand against the free-air reading at the
    stroke's start: a pull under HANG_STATIC_N, and (`strict`, above the
    knob) no wrist joint off by STATIC_NM. None when free, else why."""
    now = _settled(ctx, where)
    if now is None:
        return "no steady efforts at %s" % where
    r = residual(_wrist_gravity(), ref_start, now)
    pull = down_force(tip_load_at(now[1]), r)
    worst = max(range(len(r)), key=lambda j: abs(r[j]))
    runner.note("reach_free_check", where=where, residual_nm=[round(v, 3) for v in r], pull_n=round(pull, 2))
    if pull > HANG_STATIC_N:
        return "the hand carries %.1f N pulling down at the fingertip at %s — the box came with the hand?" % (
            pull,
            where,
        )
    if strict and abs(r[worst]) > STATIC_NM:
        return "joint_%d is %.1f Nm off its free-air reading at %s — something is on the hand" % (
            worst + 4,
            abs(r[worst]),
            where,
        )
    return None


def _back_out(ctx, runner, plan, world, ref_start):
    """Off the button and back to the look pose, never pulling the box:
    UNLOAD UNLOAD_CHECK_M straight up the stroke, CHECK at rest that nothing
    came with the finger, RISE to the stroke's start, CHECK again, SLIDE back
    out along the slide in. Every leg and check is judged against the
    free-air reading at the stroke's start (`ref_start`). False — the arm
    holds where it stopped, the reason printed — when anything was felt or
    a line cannot be flown."""
    from box_opening.runtime import approach

    m = ctx.model
    _R, t = approach._chain().fk(ctx.client.joints())
    first = [float(t[0]), float(t[1]), min(float(t[2]) + UNLOAD_CHECK_M, float(plan.start_xyz[2]))]
    # the rise only after the at-rest check 15 mm up found nothing on the hand
    for name, target, strict, speed in (
        ("reach:unload", first, False, CONTACT_SPEED),
        ("reach:rise", plan.start_xyz, True, FREE_LINE_SPEED),
    ):
        up, why = _line(ctx, target, plan.quat)
        if up is None:
            log.info(
                "[press_demo] STOP: the way up is not a straight line the arm can fly (%s) — the arm holds%s"
                % (why, " with the fingertip still on the button" if name == "reach:unload" else "")
            )
            return False
        if up:
            res = runner.run(
                [line_leg(ctx, name, up, plan.quat, speed, unload_guard(m, ref_start, tip_load_at(up[0])), world)],
                execute=True,
            )
            if any(not r.ok for r in res):
                log.info("[press_demo] STOP: %s" % _unload_stopped(res))
                return False
        why = _still_free(
            ctx,
            runner,
            ref_start,
            "%.0f mm up" % (1000 * UNLOAD_CHECK_M) if not strict else "the stroke's start",
            strict,
        )
        if why is not None:
            log.info("[press_demo] STOP: %s The arm holds; check the box" % why)
            return False
    out, why = _line(ctx, plan.look_xyz, plan.quat)
    if out is None:
        log.info("[press_demo] STOP: the slide out is not a straight line the arm can fly (%s) — the arm holds" % why)
        return False
    if out:
        # after the strict check at the stroke's start: away from the box, above it
        res = runner.run(
            [
                line_leg(
                    ctx,
                    "reach:slide-out",
                    out,
                    plan.quat,
                    FREE_LINE_SPEED,
                    drag_guard(ref_start, tip_load_at(out[0])),
                    world,
                )
            ],
            execute=True,
        )
        if any(not r.ok for r in res):
            log.info(
                "[press_demo] STOP: something pulled on the way out (the box caught on the fingers?) — the arm "
                "holds; check the box"
            )
            return False
    return True


def _after_check(node, ctx, runner, watcher, scene_fix, locator, button, lid_before, yaw):
    """From the look pose, before the arm goes anywhere: the wrist re-aims
    (did the knob pop, did the box move or lift?) and the scene camera looks
    again. Moves ctx.cpose to where the box is now seen. Returns
    (found, moved, popped, up_after)."""
    from box_opening.tasks.press_demo import SCENE_LOCATE_S, knob_is_up

    m = ctx.model
    if locator is not None:
        # where the scene OWL must answer (no scene segmenter, or it is
        # down), its first bbox takes ~1.2 s (tick + inference): let it come
        # while the wrist re-aims (no OWL, no GPU, the arm still at the look
        # pose); locate() enables again (idempotent) and disables in its
        # finally. A live segmenter keeps the OWL off (SceneLocator.prime)
        locator.prime()
    try:
        again, status, up_after, measured_after = _aim(node, watcher, yaw, button[2], button[:2], runner, "reach-after")
    except BaseException:
        if locator is not None:
            locator.owl.disable()  # never leave the latched enable on
        raise
    moved = None if again is None else judge_box_moved(button, again[0])
    # the wrist's lid against the wrist's (lid_before too): one of two aims
    # raised by its table offset and the other not would read a lift
    lid_after = None if again is None else _lid_z(measured_after[2], up_after)
    lifted = None if lid_before is None or lid_after is None else lid_after - lid_before
    popped = knob_is_up(up_after)
    seen_again = locator.locate(SCENE_LOCATE_S) if locator is not None else None
    scene_moved = None if seen_again is None else judge_box_moved(scene_fix.top_xyz, seen_again.top_xyz)
    scene_dz = None if seen_again is None else float(seen_again.top_xyz[2]) - float(scene_fix.top_xyz[2])
    runner.note(
        "reach_after",
        knob_up_mm=None if up_after is None else round(up_after, 1),
        moved_mm=None if moved is None else round(1000 * moved, 1),
        lid_dz_mm=None if lifted is None else round(1000 * lifted, 1),
        scene_moved_mm=None if scene_moved is None else round(1000 * scene_moved, 1),
        scene_dz_mm=None if scene_dz is None else round(1000 * scene_dz, 1),
    )
    log.info(
        "[press_demo] AFTER THE PRESS: knob %s; the button %s; the scene camera %s"
        % (
            "UP %.1f mm — the box is open" % up_after
            if popped
            else ("%.1f mm above the lid" % up_after if up_after is not None else "not judged"),
            "not re-found (%s)" % status
            if moved is None
            else "%.0f mm from where it was%s"
            % (1000 * moved, "" if lifted is None else ", its lid %+.0f mm" % (1000 * lifted)),
            "did not find the box"
            if scene_moved is None
            else "puts the box %.0f mm from where it was, %+.0f mm in height" % (1000 * scene_moved, 1000 * scene_dz),
        )
    )
    found = again is not None or seen_again is not None
    box_moved = (
        (moved is not None and moved > BOX_MOVED_M)
        or (lifted is not None and abs(lifted) > BOX_LIFTED_M)
        or (scene_moved is not None and (scene_moved > SCENE_MOVED_M or abs(scene_dz) > SCENE_MOVED_M))
    )
    # home is planned around the box where it is seen now; a popped knob's
    # top is not the box's height
    if again is not None:
        x, y, z = (float(v) for v in again[0])
        ctx.cpose = cpose_for_button((x, y, button[2] if popped else z), yaw, m.button_offset)
    elif seen_again is not None:
        ctx.cpose = seen_again.pose
    return found, box_moved, popped, up_after


def _home_after_contact(node, ctx, cfg, runner, watcher, scene_fix, locator, button, lid_before, yaw):
    """The look pose reached after a contact: judge the box, then home —
    or hold when neither camera finds it. Returns (homed, popped, up_after)."""
    found, box_moved, popped, up_after = _after_check(
        node, ctx, runner, watcher, scene_fix, locator, button, lid_before, yaw
    )
    if not found:
        log.info(
            "[press_demo] STOP: neither camera finds the box after the press — it may have left the table. The arm "
            "holds at the look pose; check the box, then home the arm with sheppy's home_arm"
        )
        return False, popped, up_after
    if box_moved:
        _go_home(ctx, cfg, runner, "STOP: the press MOVED the box — check it is still standing on the table")
        return False, popped, up_after
    return _go_home(ctx, cfg, runner, ""), popped, up_after


# _predicted_drift samples every this many points plus the last: it is a
# printed estimate, and evaluating the gravity model at every point (and the
# start again for each) cost ~0.2 s at rest before the look leg (2026-10-22)
DRIFT_SAMPLE_EVERY = 4


def _predicted_drift(plan):
    """joint_4's gravity torque change along the slide and the stroke (Nm):
    what a raw guard would have read as a strike. A display estimate only
    (gravity_mismatch is the gate): sampled every DRIFT_SAMPLE_EVERY points
    plus the last. A depth plan has no tilted stroke: 0 down it."""
    g = _wrist_gravity()

    def drift(pts):
        if pts is None or len(pts) == 0:
            return 0.0
        g0 = float(g(pts[0])[0])
        idx = sorted(set(range(DRIFT_SAMPLE_EVERY, len(pts), DRIFT_SAMPLE_EVERY)) | {len(pts) - 1})
        return max(abs(float(g(pts[i])[0]) - g0) for i in idx)

    return drift(plan.slide_pts), drift(plan.stroke_pts)


def _not_one_execution(ctx, legs):
    """Why the way in's legs (way_in_legs) do not fly as one execution, or
    None: they do not merge into one group (legs.merge_groups — solve_way_in
    chains them and way_in_legs names one gains spec on every one, so it
    should not happen), or the first does not start where the arm stands
    (REUSE_RAD, as _line_from reuses a solved line). After the planner's
    look leg the way in starts at the joints that leg was planned to
    (plan_reach's look_joints; the leg ends within LOOK_END_RAD of them),
    which an impedance arrival leaves a few mm off; the driver would jump
    to them, and the runner refuses it from DRIFT_REPLAN_RAD on. Leg by
    leg, each is solved again from where the arm stands."""
    if not legs:
        return "no line to fly"
    if len(merge_groups(legs)) != 1:
        return "; ".join(
            "%s does not continue %s as one goal" % (b.name, a.name)
            for a, b in zip(legs, legs[1:], strict=False)
            if not can_merge(a, b)
        )
    start = np.asarray(legs[0].traj.points[0].positions, float)
    off = float(np.abs(np.asarray(ctx.client.joints(), float) - start).max())
    if off >= REUSE_RAD:
        return "the arm stands %.3f rad from where %s starts" % (off, legs[0].name)
    return None


def _phase_at(names, fracs, progress):
    """The member of a merged execution flying at `progress`, a time
    fraction of it: the last of `names` whose start fraction (retime_group's
    seg_start_fracs) is at or before it."""
    at = names[0]
    for name, frac in zip(names, fracs, strict=False):
        if float(frac) <= float(progress):
            at = name
    return at


def _note_flight(runner, kind, names, last):
    """The run record's row `kind` for one execution the runner re-timed
    and flew, its members `names` and its result `last`: a merged group
    logs every leg with the group's t_wall, and where each began, as a
    fraction of the execution's time, keeps them apart — with how fast it
    took each junction and how many it stopped at. Returns the runner's
    report of it (last_retime)."""
    retimed = runner.last_retime or {}
    runner.note(
        kind,
        legs=list(names),
        outcome=last.outcome,
        t_wall=None if last.t_wall is None else round(float(last.t_wall), 3),
        duration_s=round(float(retimed.get("duration_s") or 0.0), 3),
        seg_start_fracs=[round(float(v), 4) for v in retimed.get("seg_start_fracs") or []],
        junction_speeds=[round(float(v), 3) for v in retimed.get("junction_speeds") or []],
        stops=retimed.get("stops"),
    )
    return retimed


def _fly_way_in(ctx, cfg, runner, way, legs, top_z, scene_fix, locator):
    """A compliant depth press's way in (`way`, flown as `legs`,
    way_in_legs') from the aimed look pose to the stroke's end below the
    button's bottom: ONE runner.run, re-timed into one execution, nothing
    read between its legs — the way home planned while it flies, from the
    stroke's start (plan_way_out, the runner's lookahead). Then the hold on
    the button whose top is at `top_z` (depth_press.compliant_hold) and the
    way home from it (_way_home): the rise back up the stroke, the clear
    and home, one execution where they meet. A way in that did not start
    goes home from the look pose; one that stopped part-way holds the arm
    there, its STOP naming the leg in flight. From the way in's start to
    where the way home ends, a stop lets go (let_go): the way in stopped
    part-way, the way home stopped or did not get home (its fallback's
    clear or home, _home_from_over_box), anything raised. Returns the
    process exit code."""
    from box_opening.tasks import depth_press as dp
    from box_opening.tasks.press_demo import viz_hook

    names = [lg.name for lg in legs]
    if way.down.short_deg:
        log.info(
            "[press_demo] TURN: %.0f deg short of straight down (no hand nearer straight down flies here)"
            % way.down.short_deg
        )
    log.info("[press_demo] ONE MOTION from the look pose to below the button: %s" % ", ".join(names))
    log.info("[press_demo] %s" % dp.press_text(way.stroke, cfg))
    viz_hook("press", way.stroke, top_z, cfg)  # the stroke, drawn off this thread (perception/viz)
    # The stroke meets the box by design: from here every world pads it for
    # a shove, as depth_press.press_compliant's does — home's too, planned
    # while the way in flies, so it is padded before it flies
    pad = ctx.contact_pad
    ctx.contact_pad = CONTACT_SHIFT_PAD_M
    try:
        # The way home from the stroke's start, where the rise back up ends.
        # The end the lookahead is handed, the stroke's bottom inside the
        # padded box, is no start for it, and is left unused
        res = runner.run(
            legs, execute=True, lookahead=lambda _end: plan_way_out(ctx, cfg, way.stroke.q0, way.stroke.quat)
        )
        ahead = runner.lookahead_result  # every run() resets it
        last = res[-1]
        flown = all(r.ok for r in res)
        # The runner's report of the group it re-timed (last_retime) is this
        # execution's only where the runner went on to fly it, and so handed
        # back what flew (LegResult.flown_traj): a refusal before that — its
        # gates, a plan-free start the arm has left — leaves the report of
        # the last group that flew, the look's.
        tried = last.flown_traj is not None
        retimed = _note_flight(runner, "reach_way_in_flight", names, last) if tried else {}
        # nothing moved where the runner refused the execution, nor where the
        # client reports no progress for it: the driver rejected the goal, or
        # the client's own gates refused it before sending
        moved = flown or (tried and bool(last.progress))
        if flown:
            runner.note("reach_slid_in", turned_down=way.down.walked, residual_nm=None)
            seen = {}
            held = dp.compliant_hold(ctx, cfg, runner, way.stroke, top_z, seen)
            result, homed = _way_home(
                ctx,
                cfg,
                runner,
                way.stroke,
                held,
                seen.get("held_joints") or ctx.client.joints(),
                ahead,
                scene_fix,
                locator,
            )
    except BaseException:
        # A Ctrl+C in the hold's read lands with no goal in flight (abort.py
        # raises at once) and press_demo exits 130: the springs would push on
        # into the box toward the stroke's end, 27-72 mm below the button's
        # top, for as long as the driver runs
        let_go(ctx)
        raise
    if not moved:
        # from the look pose, in free air: nothing came near the box — no
        # shove to pad it for, and nothing to let go of but a home that
        # stops part-way (_go_home's own)
        ctx.contact_pad = pad
        _go_home(
            ctx,
            cfg,
            runner,
            "STOP: the way in did not start (%s) — nothing was pressed" % (last.detail or last.outcome),
        )
        return 1
    if not flown:
        log.info(
            "[press_demo] STOP: the way in stopped during %s (%s) — the arm holds"
            % (_phase_at(names, retimed.get("seg_start_fracs") or [0.0], last.progress), last.detail or last.outcome)
        )
        let_go(ctx)
        return 1
    if not result.free or not homed:
        # the way home stopped (_way_home): the arm holds on the button — no
        # way up flies — or where the rise or the one execution stopped. Free
        # but not home, its fallback home from over the box did not get there
        # (_home_from_over_box): its clear or home stopped part-way, as the one
        # execution can, or a plan was refused, the arm at rest at a leg's end
        let_go(ctx)
    runner.finish()
    return dp.finish(ctx, dataclasses.replace(result, touch_xyz=seen.get("touch_xyz")), homed)


def _in_flight(legs, last, retimed):
    """The leg of `legs` (one runner.run's) in flight when the execution
    whose result is `last` stopped: by where each member of its merge group
    began (_phase_at), where the runner re-timed and flew that group
    (`retimed`, its last_retime), else the group's first. A merged group's
    result carries its last member's name."""
    group = next((g for g in merge_groups(legs) if g[-1].name == last.leg_name), None)
    names = [lg.name for lg in group] if group else [last.leg_name]
    fracs = (retimed or {}).get("seg_start_fracs") or []
    if len(names) == 1 or last.flown_traj is None or len(fracs) != len(names):
        return names[0]
    return _phase_at(names, fracs, last.progress or 0.0)


def _rise_legs(ctx, stroke, q, world):
    """The way up from a compliant hold, the arm held at joints `q` by the
    button under `stroke`, as the legs that fly it, labelled `world`:
    'depth:rise' back up the stroke to its start (depth_press.rise_back_up)
    — else a straight line from where the arm stands to there; none when it
    stands there — at free_line_speed. Returns (legs, True where it goes
    back up the stroke, None), or (None, False, why) where no way up flies."""
    from box_opening.runtime import approach
    from box_opening.tasks import depth_press as dp

    pts = dp.rise_back_up(approach._chain(), stroke, q)
    on_stroke = pts is not None
    if pts is None:
        pts, why = _line(ctx, [float(v) for v in stroke.t0], stroke.quat)
        if pts is None:
            return None, False, why
    legs = [dp.compliant_line(ctx, "depth:rise", pts, stroke.quat, free_line_speed(ctx), world)] if pts else []
    return legs, on_stroke, None


def _no_way_up(held, why):
    """(DepthPress, homed) where no way up from the hold flies: the hold's
    verdict `held`, free False — the arm holds."""
    why = "the way up is not a straight line the arm can fly (%s) — the arm holds" % why
    return dataclasses.replace(held, free=False, why=why), False


def _rise_then_home(ctx, cfg, runner, up, held, scene_fix, locator):
    """The way home from the hold as before the one execution: the rise
    `up` (its legs; none where the arm stands at the stroke's start) on its
    own, then home from over the box (_home_from_over_box: the clear up and
    back, then home planned, the back-off its fallback). Returns
    (DepthPress, homed): free False with why where the rise did not fly."""
    from box_opening.tasks import depth_press as dp

    if up:
        res = runner.run(up, execute=True)
        if any(not r.ok for r in res):
            return dataclasses.replace(
                held, free=False, why="the way up stopped (%s) — the arm holds" % res[-1].detail
            ), False
    result = dp.after_scene_check(ctx, runner, held, scene_fix, locator)
    return result, result.free and _home_from_over_box(ctx, cfg, runner, result)


def _way_home(ctx, cfg, runner, stroke, held, q_held, ahead, scene_fix, locator):
    """From a compliant hold to home, the arm held at `q_held` by the
    button under `stroke` (the hold's verdict `held`): the rise back up the
    stroke to its start (_rise_legs: depth_press.rise_back_up, else a
    straight line from where the arm stands), then the clear and
    home that plan_way_out planned while the way in flew (`ahead`, its
    (WayOut, why) as the runner's lookahead result), each home leg written
    on the branch of the end of the leg before it (approach.on_branch_of) —
    ONE runner.run, no planner call after the hold. All under
    COMPLIANT_GAINS, so they merge into one execution: the home transit
    flies STIFF too. Legs that do not meet (a home plan off the clear's end,
    home's clear-margin step, guarded) the runner flies as goals of their
    own. Where nothing was planned, or the runner refused the one execution
    before anything moved: the rise alone — never the pads left on the
    button — then home from over the box (_rise_then_home); after a refusal
    that rise is solved again from where the arm stands. Stopped part-way:
    the arm holds, the STOP naming the leg in flight. One execution — or
    the first, where it was not one and that one stopped — has its row in
    the run record (reach_way_out_flight), as the way in's has. Returns
    (DepthPress, homed): free False with why where the arm holds."""
    from box_opening.runtime import approach
    from box_opening.tasks import depth_press as dp

    world = _full_world(ctx)
    up, on_stroke, why = _rise_legs(ctx, stroke, q_held, world)
    if up is None:
        return _no_way_up(held, why)
    if not held.free:
        return _rise_then_home(ctx, cfg, runner, up, held, scene_fix, locator)  # off the button, and no further
    # None: the lookahead raised (the runner's "lookahead plan failed" line says what)
    way, why = ahead if ahead is not None else (None, "the lookahead plan failed")
    if way is None:
        runner.note(
            "reach_way_out",
            planned=False,
            why=why,
            clear_back_mm=None,
            legs=[lg.name for lg in up],
            rise_on_stroke=on_stroke,
        )
        log.info(
            "[press_demo] no way home was planned while the way in flew (%s) — up off the button, then home from "
            "over the box" % why
        )
        return _rise_then_home(ctx, cfg, runner, up, held, scene_fix, locator)
    legs = up + [dp.compliant_line(ctx, "depth:clear", way.clear, way.quat, free_line_speed(ctx), world)]
    for leg in way.home:
        if leg.traj is not None:
            leg.traj = approach.on_branch_of(leg.traj, legs[-1].goal_joints)
            leg.goal_joints = [float(v) for v in leg.traj.points[-1].positions]
        if leg.guard is None:
            leg.gains = dp.COMPLIANT_GAINS  # one gains spec: the home transit merges, and flies RIGID
        legs.append(leg)
    names = [lg.name for lg in legs]
    groups = merge_groups(legs)
    runner.note(
        "reach_way_out",
        planned=True,
        why=way.passed,
        clear_back_mm=round(1000 * way.back_m, 1),
        legs=names,
        rise_on_stroke=on_stroke,
    )
    log.info(
        "[press_demo] %s from the button to home: %s — up to the stroke's start, then up and %.0f mm back toward "
        "the arm, clear of the box's padded volume"
        % ("ONE MOTION" if len(groups) == 1 else "in %d goals" % len(groups), ", ".join(names), 1000 * way.back_m)
    )
    res = runner.run(legs, execute=True)
    last = res[-1]
    if len(res) == 1 and last.flown_traj is not None:
        # the first execution, the rise in it, re-timed and sent: the
        # runner's report is its own (a later goal's would replace it)
        _note_flight(runner, "reach_way_out_flight", [lg.name for lg in groups[0]], last)
    if all(r.ok for r in res):
        return held, True
    started = last.flown_traj is not None and bool(last.progress)
    if len(res) == 1 and not started:
        # nothing moved: the runner's gates refused it, or the driver its
        # first goal. The pads are still on the button. One gate is the
        # arm no longer standing where a plan-free line starts, and the
        # held reading's rise would be refused again: it is solved anew
        # from where the arm stands
        log.info(
            "[press_demo] the way home did not start (%s) — up off the button on its own, then home from over the box"
            % (last.detail or last.outcome)
        )
        up, _on_stroke, why = _rise_legs(ctx, stroke, ctx.client.joints(), world)
        if up is None:
            return _no_way_up(held, why)
        return _rise_then_home(ctx, cfg, runner, up, held, scene_fix, locator)
    if not started:
        # a goal after the first did not start: the first, the rise in it,
        # flew, and the arm stands at its end in free air off the button
        log.info(
            "[press_demo] the way home's %s did not start (%s) — home from where the arm stands, over the box"
            % (_in_flight(legs, last, runner.last_retime), last.detail or last.outcome)
        )
        return held, _home_from_over_box(ctx, cfg, runner, held)
    return dataclasses.replace(
        held,
        free=False,
        why="the way home stopped during %s (%s) — the arm holds"
        % (_in_flight(legs, last, runner.last_retime), last.detail or last.outcome),
    ), False


def run_reach_press(node, ctx, cfg, runner, watcher, args, scene_fix, locator):
    """The whole reach-out press from the scene camera's fix. Returns the
    process exit code: 0 pressed and the knob popped (--press-only: pressed),
    1 a failure, the box moved or the knob not up, 2 no button.

    THE FULL RUN (a compliant run without --press-only, the default since
    2026-10-28): from the one-execution way in and the hold
    on, tasks/reach_lid_off.run_full — back to the look pose, the knob
    judged there, the press once more where it is not up, the lid off, its
    spot chosen with the lid lifted (the owner: never at the start, never a
    failure), set down, home: 0 only with the lid set down and the arm home.
    A box the aim sees already open has its lid taken off unpressed.

    COMPLIANT (args.compliant with args.depth_press, press_demo's default
    since 2026-10-27; press_demo sets depth_press on every run since
    2026-10-28): every leg in the driver's joint impedance, under the
    client's compliant_gains (RIGID) where a leg names none, and
    nothing the run reads stops the arm — no torque guard, no at-rest read
    or check, no look after the press (depth_press.press_compliant); a
    camera read at rest and the grip's close wait for the arm to stand
    still (runner.wait_still), between motions, never stopping one. Its
    free-air lines (the re-aim, the slide in, the clear before home) fly at
    COMPLIANT_FREE_LINE_SPEED (free_line_speed), the turn and the rise too,
    under the wrist's ceiling: CONTACT_SPEED is there so a guard can stop a
    line in time, and these carry none (16:23 crawled 10.5 s of them at 0.15).
    The box's height is the higher of the aim's and the scene camera's (15:19:
    the touch met the button at 0.312, the scene said 0.311, the wrist
    camera's aim 0.288). 0 then means pressed, by the spring's hold. Its
    press is proven by the arm's own kinematics before the arm moves
    toward the box: from the look pose the hand turns down and strokes —
    for the scene camera's button, or one PROOF_SHIFT_M off it, before the
    look (plan_reach depth=True, plan_shifted_look), and for the aimed
    button before the re-aim (solve_way_in). Where it does not, nothing
    moves, or the arm goes home from the look pose. Where it does, the way
    in flies as proven, as ONE execution from the aimed look pose to the
    stroke's end — the re-aim, the slide in or the walk, the turn and the
    stroke (way_in_legs, _fly_way_in) — then the hold, and from it the rise,
    the clear and home as ONE execution, planned while the way in flew
    (plan_way_out, _way_home); leg by leg, each solved again from where the
    arm stands, only where the way in's legs do not make one execution
    (_not_one_execution)."""
    on = bool(getattr(args, "compliant", False)) and bool(getattr(args, "depth_press", False))
    was = getattr(ctx.client, "compliant", False)
    if on:
        ctx.client.compliant = True
        log.info(
            "[press_demo] COMPLIANT: every move in the arm's joint impedance, %s, and nothing the run reads stops "
            "it — no torque guard, no at-rest check, no look after the press. The e-stop and Ctrl+C stop it"
            % gains_text(getattr(ctx.client, "compliant_gains", COMPLIANT_RUN_GAINS))
        )
    try:
        return _reach_press(node, ctx, cfg, runner, watcher, args, scene_fix, locator)
    finally:
        ctx.client.compliant = was


def _reach_press(node, ctx, cfg, runner, watcher, args, scene_fix, locator):
    from box_opening.models.container import from_container
    from box_opening.perception import camera_video
    from box_opening.runtime import approach
    from box_opening.tasks.press_demo import CAPTURES_DIR, dump_wrist_frame, refuse_open_box, touch_verdict

    m = ctx.model
    why = unsupported(m)
    if why is None and not oblique_aim_available():
        why = (
            "this install's wrist aim cannot look at the lid obliquely (perception/button_aim takes no lid_z/prior_xy)"
        )
    # Every refusal before the arm moves is noted as well as printed
    # (reach_refused, here and below): 14:28:12, 14:29:19 and 14:29:54
    # 2026-10-28 stopped at the proof or the look's plans, and their records
    # end at the scene fix with no reason
    if why is not None:
        log.info("[press_demo] STOP: no reach-out press — %s; nothing moved" % why)
        runner.note("reach_refused", why="no reach-out press — %s" % why)
        return 1
    yaw = scene_fix.pose.yaw
    prior = from_container(scene_fix.pose, m.button_offset)
    # the scene camera's own lid top: on flatbot it read 99-102 mm above the
    # table against the model's 121 (2026-10-22), so the aim measures it
    prior_button = (float(prior[0]), float(prior[1]), float(scene_fix.top_xyz[2]))
    log.info(
        "[press_demo] REACH-OUT: pressing with the hand tilted back toward the arm, one fingertip straight down onto "
        "the button"
    )
    soft = compliant(ctx)  # no guard references to read, no checks to make
    depth = bool(getattr(args, "depth_press", False))
    # A compliant depth press is proven before the arm moves toward the box:
    # from the look pose the hand turns down and strokes (way_down). 16:23
    # 2026-10-27 flew the look, aimed, flew the re-aim and a 7.4 s slide in
    # to a turn no IK reaches ("no straight-down hand reaches here"), then
    # the clear, a back-off and home: 20.2 s from the look to home, nothing
    # pressed. This is planning before the arm moves, not a check that stops it.
    proof = soft and depth
    # The full run (owner 2026-10-28): a compliant depth press goes on to
    # take the lid off (tasks/reach_lid_off); --press-only stops at the
    # press. Where the lid goes is chosen only once it is lifted, from what
    # the wrist camera sees then — never here, never a refusal (owner
    # 2026-10-28: "it should go no matter what").
    lid_off = proof and not bool(getattr(args, "press_only", False))
    looks = [] if proof else None
    # One read: the hold that primes the joint impedance and the look's plan
    # start from the same joints. The first compliant goal after a POSITION
    # one waits out the driver's switch (mode_settle_s 0.25 s) and starts on
    # a spring still fading in (gain_ramp_s 0.5 s): 16:23 2026-10-27's look
    # took 1.857 s against its 1.60 s preview, and no compliant leg after it
    # ran more than 3 ms over its own. The hold (client.prime_compliant) pays
    # both while the reach is planned. A preview and a guarded run send none.
    q0 = ctx.client.joints()
    # where the arm stood when the scene camera found the box: its points
    # there are the arm, not the table (reach_lid_off.spot_map)
    eyes = (node, watcher, scene_fix, list(q0))
    if args.execute and lid_off and camera_video.enabled():
        # what the wrist and the scene cameras see from the grip's way down
        # on, side by side (reach_lid_off.run_full starts and writes it)
        ctx.scene_locator = locator
        ctx.lid_video = camera_video.CameraVideo(
            [("wrist", getattr(watcher, "grab", None)), ("scene", getattr(locator, "grab", None))], CAPTURES_DIR
        )
    if args.execute and soft:
        prime = getattr(ctx.client, "prime_compliant", None)
        if prime is not None and prime(q0):
            log.info("[press_demo] the driver enters its joint impedance now, while the reach is planned")
    # the proof also picks the look's posture for the shortest walks (quickest_look): from here, not from the
    # look pose once the aim moved the button — the look's re-plan below keeps the elbow it flew there with
    plan, why = plan_reach(ctx, cfg, prior_button, q0, depth=proof, looks=looks, shortest_walk=True)
    proven = "scene" if proof and plan is not None else None
    if plan is None and proof and looks:
        # no look was planned for it: the shifted press's is, once it flies
        plan, near, why_near = plan_shifted_look(ctx, cfg, looks, prior_button, q0)
        if plan is None:
            stop = (
                "no reach-out press for the scene camera's box — no tilt turns the hand down and strokes there, "
                "nor with the button %.0f mm nearer the arm or to either side (%s; %s)"
                % (1000 * PROOF_SHIFT_M, why, why_near)
            )
            log.info("[press_demo] STOP: %s; nothing moved" % stop)
            runner.note("reach_refused", why=stop)
            return 1
        proven = "shifted"
        log.info(
            "[press_demo] REACH-OUT: the press does not reach the scene camera's button itself (%s), but at %d deg "
            "it does with the button %.0f mm %s — inside what the aim corrects — so the look flies and the aimed "
            "button decides" % (why, plan.tilt_deg, 1000 * PROOF_SHIFT_M, near)
        )
    if plan is None:
        log.info("[press_demo] STOP: no reach-out press plans for the scene camera's box (%s); nothing moved" % why)
        runner.note("reach_refused", why="no reach-out press plans for the scene camera's box (%s)" % why)
        return 1
    runner.note(
        "reach_plan",
        tilt_deg=plan.tilt_deg,
        look_xyz=[round(v, 4) for v in plan.look_xyz],
        button=list(prior_button),
        proof=proven,
        down=None
        if plan.down is None
        else dict(
            {"walked": plan.down.walked, "short_deg": plan.down.short_deg},
            **({} if plan.down.above_m is None else {"turn_above_mm": round(1000 * plan.down.above_m)}),
        ),
        look_posture=plan.posture,
    )
    if plan.posture is not None and plan.posture["chosen"]:
        p = plan.posture
        log.info(
            "[press_demo] REACH-OUT: the look holds the same pose with the elbow swung %.1f rad along its swing "
            "(joints up to %.2f rad from the nearest posture): each walk to the hand turned down %.2f s against "
            "%.2f, the look %.2f s against %.2f — %.1f s quicker over %d walks (%.2f s of IK)"
            % (
                p["swing_rad"],
                p["joint_gap_rad"],
                p["walk_s"],
                p["walk_s_today"],
                p["transit_s"],
                p["transit_s_today"],
                p["saved_s"],
                LOOK_POSTURE_WALKS,
                p["ik_s"],
            )
        )
    try:
        slide_nm, stroke_nm = _predicted_drift(plan)
        drift = "; gravity alone moves joint_4 %.2f Nm along the slide and %.2f down the stroke (taken out)" % (
            slide_nm,
            stroke_nm,
        )
    except Exception as e:  # noqa: BLE001 - a display line only; the check below is the gate
        drift = " (no gravity model: %s)" % _short(e)
    log.info(
        "[press_demo] REACH-OUT: hand tilted %d deg; looking from %.2f m back toward the arm%s"
        % (plan.tilt_deg, REACH_PULLBACK_M, drift)
    )
    if not args.execute:
        runner.run([plan.look_leg], execute=False)
        log.info(
            "[press_demo] preview ends at the look pose: the aim, the slide, the press and the checks need --execute"
        )
        return 0
    at_rest = None if soft else _settled(ctx, "the rest pose")
    if at_rest is None and not soft:
        runner.note(
            "reach_refused",
            why="no steady joint efforts at the rest pose — the reach-out's guards cannot be " "referenced",
        )
        return 1
    res = runner.run([plan.look_leg], execute=True)
    if any(not r.ok for r in res):
        let_go_part_way(ctx, res)
        return 1
    at_look = None
    if not soft:
        at_look = _settled(ctx, "the look pose")
        if at_look is None:
            return 1
        mismatch = gravity_mismatch(_wrist_gravity(), at_rest, at_look)
        runner.note(
            "reach_gravity_check",
            ok=mismatch is None,
            residual_nm=[round(v, 3) for v in residual(_wrist_gravity(), at_rest, at_look)],
        )
        if mismatch is not None:
            _go_home(
                ctx,
                cfg,
                runner,
                "STOP: %s — the reach-out's guards would be judging the wrong arm; " "nothing was pressed" % mismatch,
            )
            return 1

    got, status, up_mm, measured = _aim(node, watcher, yaw, prior_button[2], prior_button[:2], runner, "reach-aim")
    if got is None:
        dump_wrist_frame(watcher, "reach-look")
        _go_home(ctx, cfg, runner, "STOP: no button from the look pose (%s) — nothing was pressed" % status)
        return 2
    open_already = refuse_open_box(up_mm)
    if open_already is not None:
        runner.note("open_box", button_above_lid_mm=round(up_mm, 1), opened=lid_off)
        if not lid_off:
            _go_home(ctx, cfg, runner, "STOP: " + open_already)
            return 1
        # the full run takes an open box's lid off: nothing presses the knob
        log.info(
            "[press_demo] ALREADY OPEN: the knob stands %.0f mm above its lid — not pressing it, which would shut "
            "the box; taking the lid off" % up_mm
        )
    # The fingers are shut before the arm moves toward the box. press_demo
    # sent press:close before the look, and the runner joins it before a
    # gripper leg or a guarded motion only: a compliant run has neither, so
    # its close was collected after home and its result never read. An open
    # aperture strikes the lid and can still read "pressed", and fingers
    # closed on something press with it. The knuckle says which: shut is
    # what an empty close reads. The close has had the look and the aim
    # (about 3 s), so a normal run reads what it already did
    closed = runner.join_pending()
    if closed is not None and not fingers.reads_empty_close(closed.gripper_pos):
        _go_home(
            ctx,
            cfg,
            runner,
            "STOP: %s %s (%s) — the fingers are not shut; nothing was pressed"
            % (closed.leg_name, closed.outcome, closed.detail),
        )
        return 1
    if closed is not None and not closed.ok:
        # "never moved": the close found the fingers already shut, at a
        # reading press_demo did not take for shut. An empty close's reading
        # changes (0.793 to 0.636 at a driver restart, 2026-09-25) and was
        # learned at the straight-down touch only: learned here too, the
        # next run finds them shut and sends no close
        kept = fingers.learn_closed(closed.gripper_pos)
        runner.note("closed_unmoved", knuckle=round(closed.gripper_pos, 4), learned=kept)
        log.info(
            "[press_demo] %s %s (%s), but the knuckle reads an empty close: the fingers are shut%s"
            % (closed.leg_name, closed.outcome, closed.detail, " (kept as this gripper's empty close)" if kept else "")
        )
    button = tuple(float(v) for v in got[0])
    measured = tuple(float(v) for v in measured)  # the wrist's own (_aim)
    lid_before = _lid_z(measured[2], up_mm)
    log.info(
        "[press_demo] AIMED from the look pose: button at [%.3f, %.3f, %.3f], %.0f mm from the scene camera's"
        % (button[0], button[1], button[2], 1000 * judge_box_moved(prior_button, button))
    )
    if soft and float(prior_button[2]) > button[2]:
        # the press goes by the higher top, the stroke reaching on past it
        log.info(
            "[press_demo] the scene camera puts the lid %.0f mm higher than the wrist's aim: pressing from that "
            "height" % (1000 * (float(prior_button[2]) - button[2]))
        )
        button = (button[0], button[1], float(prior_button[2]))
    # the box is where its button is, not where the scene camera put it: every
    # world from here on holds its cuboid there
    ctx.cpose = cpose_for_button(button, yaw, m.button_offset)
    if ctx.mission_frames is not None:
        # a "fix", not an "aim": the oblique aim goes through the wrist
        # camera's mount in TF, which is known to be off (76 mm high, 20:04)
        ctx.mission_frames.located(button, "fix")
    world = _interaction_world(ctx, button[2], cfg.travel_m, "reach")
    if open_already is not None:  # the full run (above): the knob up already, its lid off without a press
        from box_opening.tasks import reach_lid_off

        # the lid by the higher of the scene camera's top and the aim's knob
        # less its height: an estimate the grip errs low from (grip_lid_z)
        lid_z = max(float(prior_button[2]), float(got[0][2]) - float(up_mm) / 1000.0)
        try:
            return reach_lid_off.take_lid_off(
                ctx, cfg, runner, eyes, button, plan.tilt_deg, lid_z, "the open box's lid at %.3f" % lid_z, yaw=yaw
            )
        except BaseException:
            let_go(ctx)
            raise
    # A compliant run flies the re-aim and the slide in at
    # COMPLIANT_FREE_LINE_SPEED, as its one-execution way in does
    # (way_in_legs): CONTACT_SPEED is there so a guard can stop a line in
    # time, and these carry none (16:23 2026-10-27 crawled them 9.9 s at
    # 0.15). Their world is a label: a plan-free line is never planned,
    # re-planned or collision-checked against it (line_leg builds it,
    # Runner._replan_group refuses plan-free legs, push_name only writes a
    # YAML). "full" only satisfies Runner._refusal's spec §6 rule that an
    # unguarded leg faster than CONTACT_SPEED be labelled full or bench, as
    # depth:turn, approach, press and rise already are; the speed without it
    # is refused, and the arm holds at the look pose with nothing pressed.
    line_speed, line_world = (COMPLIANT_FREE_LINE_SPEED, _full_world(ctx)) if soft else (CONTACT_SPEED, world)
    # To the look pose for the AIMED button. Where the scene camera's box was
    # wrong, the arm may already stand inside the padded cuboid of the box
    # where it really is, and the planner refuses every start there
    # (2026-10-22: the box 85 mm nearer the arm). So the arm moves there the
    # way it slides — a straight line at the tilt it holds, slow and guarded
    # in a guarded run — and only when no such line flies does the planner
    # plan the move.
    if proof:
        # the aimed button's whole way in — the move, the slide or the walk,
        # the turn and the stroke — proven from the look pose before the arm
        # moves toward it, then flown as it was proven, as one execution
        way, why_way, kind = solve_way_in(ctx, cfg, button, plan.tilt_deg, ctx.client.joints())
        if way is None and kind == "lines":
            plan2, why = plan_reach(ctx, cfg, button, ctx.client.joints(), depth=True)
            if plan2 is None:
                why_way = "as straight lines: %s; planned: %s" % (why_way, why)
            else:
                res = runner.run([plan2.look_leg], execute=True)
                if any(not r.ok for r in res):
                    let_go_part_way(ctx, res)
                    return 1
                # plan_reach proved the way down from the joints the leg was
                # planned to (look_joints; the leg ends within LOOK_END_RAD of
                # them): that proof is the way in. Solved again from the live
                # joints, an impedance arrival a few mm off the look pose would
                # only add a re-aim, which the planned move never had. It
                # starts at those joints, though, and an impedance arrival read
                # the moment its goal ends stands mm off them, past REUSE_RAD:
                # this way in then flies leg by leg (_not_one_execution)
                way = WayIn([], plan2, plan2.down, None if plan2.down.walked else plan2.slide_pts, plan2.stroke)
        if way is None:
            # from the look pose, in free air: nothing slid into the box's
            # padded volume, so no clear and no back-off before home
            _go_home(ctx, cfg, runner, "STOP: no press reaches the aimed button (%s) — nothing was pressed" % why_way)
            return 1
        move, plan2 = way.move, way.plan
        runner.note(
            "reach_way_in",
            tilt_deg=plan2.tilt_deg,
            walked=way.down.walked,
            short_deg=way.down.short_deg,
            k=round(way.stroke.k, 1),
            end_z=round(way.stroke.end_z, 4),
            **({} if way.down.above_m is None else {"turn_above_mm": round(1000 * way.down.above_m)}),
        )
        # Nothing is read between the re-aim, the slide in, the turn and the
        # stroke here, yet each flew as a goal of its own, ending at rest
        # (16:23 2026-10-27: 0.18 s between its re-aim and its slide in), and
        # each new goal's first waypoint — the joints measured where the last
        # one stopped — became the driver's impedance reference. As one
        # execution the retimer flows through every junction but a reversal
        # in joint space (into the stroke), taken at rest inside the goal.
        legs = way_in_legs(ctx, cfg, way)
        why_not = _not_one_execution(ctx, legs)
        if why_not is not None and lid_off:
            # the full run's way in is one execution: solved again from where
            # the arm stands (a planned look it arrived a few mrad off), as
            # its press once more and its grip are (tasks/reach_lid_off)
            way2, _why2, _kind2 = solve_way_in(ctx, cfg, button, plan2.tilt_deg, ctx.client.joints())
            legs2 = None if way2 is None else way_in_legs(ctx, cfg, way2)
            if legs2 is not None and _not_one_execution(ctx, legs2) is None:
                way, legs, why_not = way2, legs2, None
        if why_not is None:
            if lid_off:
                from box_opening.tasks import reach_lid_off

                return reach_lid_off.run_full(ctx, cfg, runner, eyes, way, legs, button, yaw, lid_before)
            return _fly_way_in(ctx, cfg, runner, way, legs, button[2], scene_fix, locator)
        log.info(
            "[press_demo] the way in does not fly as one execution (%s) — leg by leg, each solved again from where "
            "the arm stands" % why_not
        )
    else:
        move, plan2, why_line = reach_from_here(ctx, cfg, button, plan.tilt_deg, need_stroke=not depth)
    if plan2 is not None:
        if move:
            res = runner.run(
                [line_leg(ctx, "reach:re-aim", move, plan2.quat, line_speed, free_guard(ctx, at_look), line_world)],
                execute=True,
            )
            if any(not r.ok for r in res):
                log.info(
                    "[press_demo] STOP: the move to the aimed button's look pose %s — the arm holds; "
                    "nothing was pressed" % _stopped_by(res)
                )
                let_go_part_way(ctx, res)
                return 1
    else:
        plan2, why = plan_reach(ctx, cfg, button, ctx.client.joints())
        if plan2 is None:
            _go_home(
                ctx,
                cfg,
                runner,
                "STOP: no reach-out press for the aimed button (as straight lines: %s; planned: %s)" % (why_line, why),
            )
            return 1
        res = runner.run([plan2.look_leg], execute=True)
        if any(not r.ok for r in res):
            let_go_part_way(ctx, res)
            return 1
    if not soft:
        at_look = _settled(ctx, "the look pose")
        if at_look is None:
            return 1
    # With the depth press the slide in also turns the hand straight down,
    # its pads ending where the tilted fingertip would: one walk, flown as
    # the slide is (line_speed, the slide's guard), never lower than it
    # (tasks/depth_press walk_in_down). Where no such walk flies, the slide,
    # then the TURN.
    walk, turned = None, None
    if depth:
        from box_opening.tasks.depth_press import walk_in_down

        walk, turned, why_walk = walk_in_down(ctx, plan2.start_xyz, plan2.quat)
        if walk is None:
            log.info(
                "[press_demo] the slide in does not turn the hand down on the way (%s) — sliding, then turning"
                % why_walk
            )
    if walk is not None:
        slide, quat_in = walk, turned
    else:
        slide, why = _line_from(ctx, plan2.slide_pts, plan2.start_xyz, plan2.quat)
        if not slide:
            _go_home(
                ctx,
                cfg,
                runner,
                "STOP: the slide in to the button is not a straight line the arm can fly (%s)" % (why or "no length"),
            )
            return 1
        quat_in = plan2.quat
    res = runner.run(
        [line_leg(ctx, "reach:slide-in", slide, quat_in, line_speed, free_guard(ctx, at_look), line_world)],
        execute=True,
    )
    if any(not r.ok for r in res):
        log.info(
            "[press_demo] STOP: the slide in %s before the button — the arm holds; nothing was pressed"
            % _stopped_by(res)
        )
        let_go_part_way(ctx, res)  # toward the slide's next point, toward the box
        return 1
    at_start = (
        None if soft else _settled(ctx, "the stroke's start" if walk is None else "over the button, pointing down")
    )
    if at_start is None and not soft:
        return 1
    runner.note(
        "reach_slid_in",
        turned_down=walk is not None,
        residual_nm=None if soft else [round(v, 3) for v in residual(_wrist_gravity(), at_look, at_start)],
    )
    if depth:
        # over the box: from here the height is the wrist camera's depth and
        # the press the torques' (tasks/depth_press), not TF's lid — the
        # look's aim where the close-up does not confirm the button, as the
        # wrist measured it: the guarded press checks it against the wrist's
        # own depth under the fingers (_aim). The compliant one goes by top_z.
        from box_opening.tasks import depth_press

        result = depth_press.press_down_from_here(
            node,
            ctx,
            cfg,
            runner,
            watcher,
            CONTACT_TOOL,
            here=at_start,
            turned=turned,
            knob_up_mm=up_mm,
            aimed=measured,
            compliant=soft,
            top_z=button[2],
        )
        result = depth_press.after_scene_check(ctx, runner, result, scene_fix, locator)
        homed = result.free and _home_from_over_box(ctx, cfg, runner, result)
        if soft and result.free and not homed:
            # free but not home, its home from over the box did not get there
            # (_home_from_over_box): its clear or home stopped part-way, as
            # _fly_way_in's fallback can, or a plan was refused, the arm at
            # rest at a leg's end. Not free, press_compliant let go where it stopped
            let_go(ctx)
        runner.finish()
        code = depth_press.finish(ctx, result, homed)
        if lid_off:
            # flown leg by leg, the press is all a full run does: never a success
            log.info(
                "[press_demo] STOP: the way in flew leg by leg, which ends at the press — the lid is not taken off"
            )
            code = max(code, 1)
        return code

    def stop_after_back_out(why, code=1):
        log.info("[press_demo] STOP: %s" % why)
        if not _back_out(ctx, runner, plan2, world, at_start):
            return code
        if ctx.contact_pad > 0.0:  # something was touched: judge the box before going anywhere
            _home_after_contact(node, ctx, cfg, runner, watcher, scene_fix, locator, button, lid_before, yaw)
        else:
            _go_home(ctx, cfg, runner, "")
        return code

    stroke, why = _line_from(ctx, plan2.stroke_pts, plan2.bottom_xyz, plan2.quat)
    if not stroke:
        return stop_after_back_out(
            "the stroke down is not a straight line the arm can fly (%s) — nothing was pressed" % (why or "no length")
        )
    touch = touch_leg(ctx, cfg, stroke, plan2.quat, world, at_start)
    touched = runner.run([touch], execute=True)
    if touched[-1].outcome == "touch":
        # contact happened: from here every world pads the box for a shove
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
    if any(not r.ok for r in touched) or touched[-1].contact_xyz is None:
        return stop_after_back_out("the touch did not find the button (%s)" % touched[-1].detail)
    pad = contact_from_tips(touched[-1].contact_xyz, plan2.quat)
    off = pad[2] - button[2]
    verdict = touch_verdict(off)
    runner.note(
        "touch",
        offset_mm=round(1000 * off, 1),
        verdict=verdict,
        reach_tilt_deg=plan2.tilt_deg,
        landed_mm=round(1000 * judge_box_moved(button, pad), 1),
    )
    log.info(
        "[press_demo] TOUCH: the fingertip met the surface %+.1f mm from the aimed button top, %.1f mm from its centre"
        % (1000 * off, 1000 * judge_box_moved(button, pad))
    )
    if verdict == "foreign":
        return stop_after_back_out(
            "the fingertip met a surface %.0f mm above the button — not this box's button" % (1000 * off)
        )
    pressed = verdict == "press"
    if pressed:
        depth = reach_push_m(cfg, plan2.tilt_deg)
        log.info(
            "[press_demo] PUSH %.1f mm (the gripper clears the lid by %.1f mm at %d deg)"
            % (1000 * depth, 1000 * REACH_LID_CLEAR_M[plan2.tilt_deg], plan2.tilt_deg)
        )
        _R, t = approach._chain().fk(ctx.client.joints())
        down, why = _line(ctx, [float(t[0]), float(t[1]), float(t[2]) - depth], plan2.quat)
        if not down:
            return stop_after_back_out(
                "the push is not a straight line the arm can fly (%s) — nothing was pressed" % (why or "no length")
            )
        # the backstop judged against the free-air reading at the stroke's
        # start: the push's first sample already carries the touch's
        # overshoot, and a backstop riding on it is no bound on the total
        _g, verify = _push_guard_and_verify(m, cfg, depth_m=depth)
        guard = GuardSpec(touch_nm=float(m.touch_nm), trip="press", gravity=_wrist_gravity(), reference=at_start)
        push = line_leg(ctx, "press:push", down, plan2.quat, cfg.press_speed, guard, world, verify)
        push.hold_s = float(cfg.push_hold_s)
        push.recoil_via = touch.traj  # the recoil carries on back up the touch
        res = runner.run([push], execute=True)  # held, then the runner's recoil reverses it
        if any(not r.ok for r in res):
            return stop_after_back_out("the push failed (%s)" % res[-1].detail)
        log.info("[press_demo] PRESSED — %s" % res[0].detail)
        if ctx.mission_frames is not None:
            ctx.mission_frames.note(stage="pressed")
    else:
        log.info(
            "[press_demo] the knob is ALREADY up — the fingertip met it %.1f mm above the lid; not pushing it down"
            % (1000 * off)
        )
    if not _back_out(ctx, runner, plan2, world, at_start):
        return 1

    # did the knob pop, did the box move — judged before the arm goes anywhere
    homed, popped, up_after = _home_after_contact(
        node, ctx, cfg, runner, watcher, scene_fix, locator, button, lid_before, yaw
    )
    if ctx.mission_frames is not None and pressed:
        ctx.mission_frames.note(popped=None if up_after is None else bool(popped))
    if not homed:
        return 1
    runner.finish()
    if not pressed:
        log.info("[press_demo] nothing was pressed: the knob was already up when the fingertip met it")
        return 1
    if not popped:
        log.info(
            "[press_demo] STOP: pressed, but the knob did not pop%s"
            % ("" if up_after is not None else " (the wrist could not judge it)")
        )
        return 1
    if not args.press_only:
        log.info(
            "[press_demo] a guarded reach-out run ends after the press: the lid comes off in the compliant run "
            "(without --guarded)"
        )
    return 0
