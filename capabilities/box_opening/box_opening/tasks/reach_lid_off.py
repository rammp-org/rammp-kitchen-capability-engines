"""The reach-out's lid off: the compliant reach-out press (tasks/reach_press)
as the whole run — press, see the knob up, grip it, lift the lid off, set it
down where the cameras see room, home. Owner 2026-10-28: every box goes
through the reach-out, and the full run is its default; --press-only stops
after the press as before (reach_press._fly_way_in).

  PRESS   the way in as one execution and the hold (reach_press.way_in_legs,
          depth_press.compliant_hold), as the press-only run flies them.
  BACK    the rise back up the stroke, then the way down backwards — the
          turn and the slide, or the walk, each reversed — to the aimed look
          pose: ONE execution, every line one the proof flew forward, home
          planned from the look pose while it flies (a STOP there flies it).
  LOOK    from there, where the wrist aimed before the press: the oblique
          aim again (reach_press._aim) reads the knob's height above its lid,
          and once more where the first read says it is not up. Flatbot's
          URDF: over boxes 0.60-0.78 m out no turned-down hand rises the 56-69
          mm above the stroke's start that would put the wrist the 245 mm off
          the lid the straight-down depth look took at staging (it rises 0-30
          mm), and at the stroke's start the button sits at the frame's top
          edge 176-189 mm from the lens, inside the OAK's stereo minimum. From
          the look pose the button is at the frame's centre, 299 mm off, 29
          deg from the lens axis — the view the press was aimed from.
  AGAIN   not up: the same press once more, its way in solved again from
          where the arm stands, and the look again. Still not up: STOP, home.
          Not up on a lid that rose (released, LID_POP_M): STOP, home, the
          box left open — a flush knob gives the fingers nothing to hold,
          and a released lid is not pressed again.
  DIRECT  first, straight from the first press's hold (GRIP_FROM_PRESS):
          up GRIP_RISE_M, the fingers opening to the knob's width as the hand
          rises, down the stroke's line to the grip's height (grip_lid_z),
          the close and the LIFT below — no way back, no look. On air, back
          to the look pose, and the LOOK as below, the GRIP from there.
  GRIP    up: the fingers open as the hand comes down from the look pose —
          the way in once more, its stroke the grip's: straight down until the
          closed pads' reference stands GRIP_BELOW_LID_M under the lid — ONE
          execution. While it flies (grip_ahead): the lift, and from where it
          ends the lid's spot, the carry there and home after the set-down.
          The fingers close: on
          air (past the grip band's closed end) they open again and the hand
          backs out the way it came to the look pose, home planned while it
          does; there the wrist aims at the knob again, and where it finds
          it no farther than a slid box is gripped (depth_press.SLIDE_OK_M),
          the hand comes down to it once more (GRIP_TRIES). On air again,
          or no knob, or no way down to it: home.
  SPOT    where the lid goes, from the scene camera (owner 2026-10-29: "in
          parallel will pressing/opening the box so we can seemlesly grab
          the lid and move it to the safe location without stopping or
          interuptions"): its map of the table, the arm it saw taken out
          (spot_map), the room on it judged while the press flies
          (room_ahead, press_demo.clear_lid_drops), ranked by how far the
          hand turns from where the clear ends; the first spot the hand
          reaches straight down (drop_reachable) whose carry and home after
          the set-down both plan. Never a failure (owner 2026-10-28: "it
          should go no matter what"): nothing clear, the most room seen; no
          map, beside the box; else the nearest table point the hand
          reaches, off the box (spot_candidates) — one line says which, and
          why.
  LIFT    the close, then the lid straight up off the canister
          (lift_off_m), up and back toward the arm clear of the box's padded
          volume (as the way home's clear), the carry to the spot and down
          onto the table: ONE execution. Where no spot's carry and home
          planned ahead, as before: the lift and the clear alone, the carry
          to the scene camera's best spot planned while they fly
          (carry_ahead), and there the lid's one stop, the spot look: the
          wrist camera's frame — the held lid, the hand and the box taken
          out of it — pooled with the scene camera's map (choose_and_carry);
          the carry planned ahead flies where the spot chosen there is its
          own, else the carry to it is planned there.
  PLACE   the fingers open as the retreat back up to the carry's height
          starts; then home, planned ahead, slow and in the driver's MEDIUM
          joint impedance, as every home flies (core.plan_home).

Every move up to the release flies in the driver's joint impedance under the
run's one gains spec (depth_press.COMPLIANT_GAINS, RIGID), unguarded, merged
where its legs meet: nothing the run reads stops the arm. The way home after
the release flies slow under MEDIUM, as home_arm's does. Its free-air lines — the rise and
the way back, the grip's way up, the clear, the retreat — fly at
reach_press.free_line_speed; the legs that meet the box, the knob or the
table at their own. It decides only what it must:
the knob up or not, the box where it was, the fingers on the knob or on
air, where the lid goes, the lid still in them at the table. Contact is bounded by the spring,
not stopped by a guard: the grip's pads ride on the lid where it stands
higher than estimated. The set-down is the one move the arm's own feel
ends: it stops where the lid meets the table (watch_setdown, cut there) and
lets go, and goes no lower than SETDOWN_RELEASE_M over where the scene
camera's table and the model's lid put that. Where it looks at rest — the
knob, and the spot where nothing was planned ahead — and before the fingers
close on the knob, it waits for the arm to stand still (runner.wait_still):
between motions, never a stop.
"""

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from box_opening import LOGGER as log
from box_opening.constants import DRIFT_REPLAN_RAD, GRIPPER_CMD_CLOSED, GRIPPER_CMD_OPEN, TRANSIT_SPEED
from box_opening.models.container import ContainerPose
from box_opening.perception import clear_table
from box_opening.primitives.core import (
    CONTACT_SHIFT_PAD_M,
    Place,
    PlanState,
    _full_world,
    _gripper_leg,
    band_verify,
    tcp_z,
)
from box_opening.runtime import abort, fingers
from box_opening.runtime.driver import GripperWait
from box_opening.runtime.guards import in_band
from box_opening.runtime.legs import merge_groups
from box_opening.tasks import depth_press as dp
from box_opening.tasks import reach_press as rp

# The press, then once more where the look does not see the knob up (owner
# 2026-10-28) and the lid did not rise (LID_POP_M: a released lid is not
# pressed again), then STOP.
PRESSES = 2
# A knob the look reads down is looked at once more before it is judged
# (as the staging run's after-look was): pressing a knob that popped shuts
# the box again, and the oblique aim read a flush button 0.9-3.7 mm up on
# 2026-10-27/28's looks, hits apart by up to 7.5 mm.
KNOB_RECHECK_AIMS = 1
# The pink button's travel from flush to its stop: the guarded pushes of
# 2026-10-28 14:26 and 14:30 met it 7.6 and 7.4 mm below their touches. A
# compliant press that popped the knob held the pads on it there.
KNOB_STROKE_M = 0.0075
# Where the hold held the pads, the lid the grip goes by stands this much
# above them — not the scene camera's top (owner 2026-10-30: the scene camera
# gives the arm a general location). The grip went by that top, kept between
# the held pads and KNOB_STROKE_M above them: of 2026-10-28..30's grips the
# ten that held the knob stood 0.1-7.5 mm above (0.1-5.5 where the lid rose
# as well), and 16:23, 10:40 and 10:58's three on air all at the 7.4-7.5 mm
# cap with a rise added, the scene camera's top 9-19 mm above the touch.
GRIP_LID_ABOVE_HELD_M = 0.004
# A released lid RISES: on 2026-10-28 16:15:19 and 16:15:52 the press let
# the pink canister's lid go and the wrist read it 9.6 and 9.4 mm higher
# than before the press, the button 6 and 3 mm from where it was aimed, its
# knob 7.8 and then 0.9 mm above the lid. Both runs stopped "the press moved
# the box" (BOX_LIFTED_M) and went home with the box open. A lid this much
# higher with the button where it was (depth_press.SLIDE_OK_M) is the box
# open; lower, or higher than this, is still the box moved. Open is not
# something to hold, though: the fingers close on the knob at its base
# (GRIP_BELOW_LID_M), and a flush one gives them nothing. At 16:23:53 the
# lid rose 8.2 mm, the button 4.6 mm from where it was aimed, its knob
# -0.6 mm above it: the grip closed on air (0.768 against the band
# [0.30, 0.55]), opened and backed out — 11.6 s more over the box after
# the look, then home, exit 1. A risen lid is gripped only with its knob up
# (KnobLook.popped); with it flush the run STOPs, home, the box left open,
# and does not press it again: pushing a released lid back down may latch
# it again (an open OXO pressed was shut, 2026-09-21:
# press_demo.refuse_open_box; not tried on this canister). Nor does the
# stroke fit it: the lid stands on its spring 8-10 mm above the top the
# stroke was built for, and on a lid that does not give the same stroke
# pushes up to 24-29 N more than the first press's 52-57 N (those three
# boxes' 2876-3034 N/m; a lid its spring lets back down ends the pads where
# the first hold did).
LID_POP_M = (0.004, 0.030)
# The grip: the closed pads' reference this far under the lid as estimated
# (grip_lid_z). The 2F-85's pads stand 13.5 mm higher open than shut, and
# 2.3 mm higher closed on the 44 mm knob (flatbot's URDF: the knuckle at
# 0.41 rad of 0.8): the open fingers come down 11.5 mm above the estimate,
# and closing, the pads reach the knob at its base — riding on the lid,
# the spring giving, where it stands higher than estimated.
GRIP_BELOW_LID_M = 0.002
# The fingers come down to the knob opened only to its width and
# GRIP_SIDE_CLEAR_M either side of it (grip_open_cmd), not to the 2F-85's full
# ~85 mm. 2026-10-29 14:31:59 (the box 0.74 m out, the hand leaning 15 deg):
# fully open, the fingertips came down 46-49 mm from the knob's axis, across
# the line to the box — on the rim of a lid only 52.5 mm in radius — and
# closed on it (0.147: "on the box ~0.1", runtime/fingers), not on the 44 mm
# knob; the owner saw "the gripper went down too far while open and grabbed
# the edge of the box". Opened to 64 mm they stand 32 mm off the knob's axis,
# 20 mm inside the lid's edge, and the knob may be 10 mm off the aim either
# side. The open pads stand ~10 mm over the shut ones then (13.5 fully open):
# they still come down above the lid.
GRIP_SIDE_CLEAR_M = 0.010
# The 2F-85's aperture fully open (GRIPPER_CMD_OPEN, knuckle 0) and, as the
# knuckle closes to GRIPPER_CMD_CLOSED, about linear on it: the 44 mm knob
# closes it to 0.39-0.42 (the grip band's middle), the line's 0.39.
GRIPPER_OPEN_WIDTH_M = 0.085
GRIP_SLOW_M = dp.TOUCH_SLOW_M  # the grip's last stretch down flies at TOUCH_SPEED, as the press's
# A grip on air is tried once more, at the knob aimed at again. 2026-10-29
# 12:03:52: the press let the lid go (+6.2 mm), the look found its knob up
# 9.1 mm and the button 12.3 mm from where the press was aimed; the grip came
# down where the press was aimed and closed on air (0.789 against the band
# [0.30, 0.55]) 11 mm short of the knob, backed out and went home, exit 1.
# Now the fingers open, the hand backs out to the look pose and the wrist
# aims there again (_regrip): the second grip goes to where it sees the knob
# now, at the first grip's tilt. Only a close on air — past the band's closed
# end, nothing between the pads — is tried again: fingers stopped wide on
# something larger than the knob, or a gripper that failed, are not a miss a
# re-aim mends. Never more than once.
GRIP_TRIES = 2
# The grip straight from the press (owner 2026-10-30): after the first press's
# hold the hand rises GRIP_RISE_M straight up, the fingers opening to the
# knob's width as it does, and comes down again on the stroke's line to the
# grip's height (grip_lid_z: where the hold held the pads), and closes — no
# way back to the look pose, no look, no way down from it: 10:08's took 2.6
# s back, 0.7 at the look and 3.2 down again. On air it goes back to the look
# pose, and the look decides as before: the knob not up, the press once more;
# up, the grip from there (attempt 1). Its own attempt is 0. With no look
# there is no rise measured (KnobLook.lid_dz_m): a lid that rose meets the
# fingertips before the way down ends, the spring on it, and they close round
# the knob's base.
GRIP_FROM_PRESS = True
GRIP_RISE_M = 0.025  # over the popped knob (~7.5 mm proud) before the fingers spread
_MISSED = "the grip straight from the press missed"  # _grip_from_hold's: back at the look pose
# The grip's close-up record (owner 2026-10-30, "a recorder first"): at each
# close, once the arm stands still, the wrist's frame — the fingers open
# round the knob — kept, with where the knob's circle sits in it and whether
# the grip held. A centring before the close needs where a held grip has the
# knob: with the grip centred the knob's centre is at or past the frame's
# bottom edge, and the camera's TF puts the hand's centre line ~13 mm from
# where the one held grip of 10-30 (10:08) had it; its five on air had it
# 10-23 mm off along the frame's height and up to 18 mm across. The knob's
# radius in those frames, px:
GRIP_SHOT_R_PX = (45, 95)
# ... and GRIP_RETRY_LOWER_M lower than the grip that closed on air (owner
# 2026-10-30: "if it doesn't grab it, it tries again but this time going ~1
# cm lower"). 10:40: the grip went by the scene camera's top (0.346, its
# heights high that day) less the knob's stroke and a 6 mm "the lid rose",
# its pads at 0.338 where the press had held them at 0.327, and closed on
# air over the knob; 1 cm lower is 0.328. Where the re-aim finds no knob,
# the grip goes where the first one went, the press's and the look's aim.
GRIP_RETRY_LOWER_M = 0.010
# ... and where it finds the knob, there, this far at most from where the
# first grip went: fingers that closed over it can nudge the lid (10:40 and
# 10:58 2026-10-30, the knob re-aimed 14 and 19 mm off once the aim no longer
# took the lid's 60 mm ring for it, button_aim.TABLE_RECTIFY_M); farther is
# the box moved, and the hand goes home
REGRIP_MAX_M = 0.040
# ... and found there on this many agreeing hits, not the reach-out's 2
# (reach_press.REACH_AIM_HITS): a popped knob aims worse than a flush one —
# on 14 real popped-knob frames and 42 variants the re-aim landed 4.1 mm
# median, 11.6 mm p90 and 21 mm at most from the same run's flush first aim,
# and the 3 recorded regrips at re-aimed points (10-30, before #103/#109)
# closed on air (sweep aim-popped). The agreeing hits' mean or median then
REGRIP_AIM_HITS = 3
# Straight up with the knob held, this much past the cap's own height: off
# the canister's rim. Flatbot's URDF lifts the hand turned down only 60-140
# mm straight up over boxes 0.60-0.78 m out (150 mm only at 0.58 m), so the
# rest of the way out is up and back toward the arm (reach_press
# .clear_targets), as the way home's clear goes.
LIFT_OFF_CLEAR_M = 0.015
# The set-down: at setdown_speed or faster (PLACE_TOUCH_SPEED below) to
# SETDOWN_SLOW_M above where the plan has the lid meet the table — the scene
# camera's table, the model's lid hanging from the grip (setdown_poses) —
# then at PLACE_TOUCH_SPEED, watched (watch_setdown), down to
# SETDOWN_RELEASE_M above it: stopped where the arm feels the lid meet the
# table, and let go there. The plan is no measure of it: the 17 recorded
# set-downs met the table 8-74 mm above where it put that (the lid hangs
# lower from the knob than lid_dims.z says), and flown to the plan's end
# the spring pressed on down 2-34 mm past the touch — 2026-11-02 16:20 met
# it 72-77 mm up, during the fast way down, and ended 37 mm under the hand,
# ~60 N (owner: "presses the lid down onto the table waaaaaaay too hard ...
# the lid should just be let go of when it's ~1 cm above the table"). The
# slow stretch begins above all of them, and the touch's own end — a lid the
# plan puts right — lets it go 1 cm over the table, to drop onto it.
SETDOWN_SLOW_M = 0.08
SETDOWN_RELEASE_M = 0.010
# Owner 2026-10-29: "we also need to speed up the lid placement". The way down
# to SETDOWN_SLOW_M over the table is free air, so it flies at the free-air
# lines' speed (reach_press.free_line_speed: 0.55 in a compliant run; it was
# setdown_speed, 0.35); the last stretch onto the table at PLACE_TOUCH_SPEED,
# the press's TOUCH_SPEED. It was twice that from #89 (12:02's and 12:04's way
# down and touch took 1.8 s of the carry), and the owner found the lid set
# down "too hard" (2026-10-29 14:45): the faster touch met the table at twice
# the speed, the damper (zeta 1.0) pushing on while the reference ran on.
PLACE_TOUCH_SPEED = dp.TOUCH_SPEED


def lift_off_m(model):
    """How far the knob is lifted straight up: the cap off the rim."""
    return float(model.lid_dims[2]) + LIFT_OFF_CLEAR_M


def grip_open_cmd(model):
    """The knuckle the fingers come down to the knob at: open to its width
    (model.button_diameter_m) and GRIP_SIDE_CLEAR_M either side, on the
    2F-85's about-linear aperture (GRIPPER_OPEN_WIDTH_M at GRIPPER_CMD_OPEN,
    nothing at GRIPPER_CMD_CLOSED); never wider than fully open."""
    width = float(model.button_diameter_m) + 2.0 * GRIP_SIDE_CLEAR_M
    frac = min(max(width / GRIPPER_OPEN_WIDTH_M, 0.0), 1.0)
    return GRIPPER_CMD_OPEN + (GRIPPER_CMD_CLOSED - GRIPPER_CMD_OPEN) * (1.0 - frac)


def grip_lid_z(top_z, held_z=None):
    """The lid's height the grip goes by: GRIP_LID_ABOVE_HELD_M above where
    the compliant hold held the pads on the button (`held_z`), the arm's own
    touch; without a hold, the press's top (`top_z`, the higher of the
    aim's and the scene camera's)."""
    if held_z is None:
        return float(top_z)
    return float(held_z) + GRIP_LID_ABOVE_HELD_M


def setdown_poses(ctx, drop):
    """(attitude, tool where the lid meets the table, the carry's hover)
    for the spot `drop`: the hand straight down, turned by the spot's yaw
    (press_demo.place_quat, the attitude clear_lid_drops judged the room
    for); the lid hangs where the grip took it, GRIP_BELOW_LID_M under the
    pads' reference; the hover above it as Place.hover_for has it, high
    enough for the lid to clear the box."""
    from box_opening.tasks.press_demo import place_quat

    quat = list(place_quat(ctx.model, drop))
    contact = [
        float(drop.xyz[0]),
        float(drop.xyz[1]),
        tcp_z(float(drop.xyz[2]) + float(ctx.model.lid_dims[2]) - GRIP_BELOW_LID_M),
    ]
    return quat, contact, [float(v) for v in Place.hover_for(ctx, contact)]


def _abort_check(ctx):
    """runtime.abort.check on this run's client: LookaheadAborted where a Ctrl+C has come."""
    abort.check(getattr(ctx, "client", None))


def drop_reachable(ctx, drop, q_start, seeds=()):
    """Why the hand does not set the lid down at `drop`, or None: no
    straight-down hand at the carry's hover (the arm's own IK, from
    `q_start`, else from `seeds`), or no straight line down from it to the
    set-down's end."""
    from box_opening.runtime import approach

    chain = approach._chain()
    quat, contact, hover = setdown_poses(ctx, drop)
    q = chain.nearest_solution(list(q_start), hover, quat, seeds=tuple(seeds))
    if q is None:
        return "no straight-down hand reaches its carry height %.3f" % hover[2]
    pts, why = chain.straight_line(q, [contact[0], contact[1], contact[2] + SETDOWN_RELEASE_M], quat)
    if pts is None:
        return "the set-down is not a straight line the arm can fly (%s)" % why
    return None


# ---- where the lid goes, chosen with the lid lifted -----------------------

# The spot look: the first new frame shot once the arm stands still
# (runner.wait_still, which took over depth_press.look's blind 0.3 s
# settle), within SPOT_FRAME_S
SPOT_FRAME_S = 1.0
# What of the wrist's frame is not the table: nearer the lens than this —
# the fingers and the lid in them, 116-190 mm off (the OAK's stereo drops
# most of it) — and inside the held lid's column, its radius plus this
# (the lid, and what it hides), and the box's footprint (clear_table's
# BOX_CLEAR_M past it: the box, which every spot keeps clear of anyway)
WRIST_HAND_M = 0.20
HELD_LID_CLEAR_M = 0.03
# The wrist's frame at the look pose (the reach-aim's, before the press)
# sees the table round the box from above, where the scene camera, 14 cm
# over it, sees it edge-on: pooled into the room map (aim_cloud), offline
# on 20 aimed missions the spots verified clear went from 10 to 18 and the
# map-blind tiers ('beside the box', 'nearest') from 8 to 0. False: shadow
# mode — the pooled map is built while the press flies and kept beside the
# run record (the lid_spot row's 'pooled'), the scene camera's alone
# chooses. True once attended runs show every object placed in the scene
# camera's blind band standing in it
POOL_AIM_FRAME = False
# A spot tier's candidates checked for the hand's reach at most this many,
# and the carry planned to at most CARRY_TRIES of those that pass
TIER_TRIES = {"clear": 8, "most room": 10, "beside the box": 8, "nearest": 24}
CARRY_TRIES = 4
# The carry planned while the lift flies (carry_ahead) flies where the spot
# chosen after the lift is its own — this near, the hand turned the same —
# and the arm stands within DRIFT_REPLAN_RAD of where it starts. 2026-10-29
# 13:37: the arm stood 3.2 s with the lid in the fingers between the lift's
# arrival and the carry's start — 0.63 s for the arm to stand still, 0.10 s
# for the wrist's frame, 2.38 s choosing the spot and planning the carry,
# 0.06 s sending it. Replayed off the robot on flatbot's URDF with that
# run's wrist and scene frames, the choice took 1.6 s: 0.07 s of map, 0.82 s
# judging its room (press_demo.clear_lid_drops) and 0.47 s of the hand's
# reach (8 spots' drop_reachable: IK and the set-down's line), then the
# carry: the planner's 0.19 s (its plan_s in the record) and 0.07 s of
# set-down lines. Only the room needs the wrist's frame: the reach of the
# spots, judged from the same joints, and the carry to the best are done
# while the lift flies (lid_spot notes each part's time at rest). The same
# replay through carry_ahead: 1.64 s of it under 13:37's 2.54 s lift, the
# same spot chosen after it, 0.89 s at rest instead of 1.67.
AHEAD_SAME_SPOT_M = 0.01
# The last resort: table points on rings round the box, from the lid's
# disc NEAREST_CLEAR_M off the box's outward NEAREST_STEP_M apart to
# NEAREST_MAX_M, every NEAREST_DEG, nearest the box first, then nearest
# the arm (the hand reaches nearer in). The rings centre on the box where
# the press held its button (Ctx.held_xy), and pass over every spot whose
# disc covers something a camera saw standing. 2026-10-30 13:35 set the
# lid down 0.120 m from the box (its rim LID_CLEAR_M off the canister,
# the press 18 mm off the aim) on the blue holder's flange, and the holder
# tipped; 10:08 left it touching the holder. The first ring is the lid's
# rim 50 mm off the box (0.155 m for the pink canister): lid_place_min_clear's
# 0.1985 m has no spot the hand reaches on 10:08's run
NEAREST_CLEAR_M = 0.05
# ... a held button farther than this from ctx.cpose's is not this box's
# where it is now (the box moved since, and was aimed at again): the rings
# centre on ctx.cpose. The presses since #109 held the pads 2-18 mm off it
NEAREST_HELD_MAX_M = 0.03
NEAREST_STEP_M = 0.025
NEAREST_MAX_M = 0.45
NEAREST_DEG = 20


def _table_z(watcher, scene_fix, cpose):
    """The table the lid goes down on: the scene camera's (main's
    adopt_scene_table puts it on the watcher), else the box's base."""
    for z in (getattr(watcher, "table_z", None), getattr(scene_fix, "table_z_scene", None)):
        if z is not None and math.isfinite(float(z)):
            return float(z)
    return float(cpose.xyz[2])


def spot_look(node, watcher, runner):
    """The wrist camera's points in the base frame (press_demo.wrist_points)
    from its first frame shot once the arm stands still (runner.wait_still,
    noted 'still' before the spot look), or None: no camera, no node, no
    new frame within SPOT_FRAME_S, or no pose for it. Noted 'spot_look':
    how many points, else why none (spot_look_why), and the frame kept
    (captures/wrist-spot-look-*): 2026-10-29 10:28's look gave the room map
    no points, and nothing said whether it had no frame, no pose for it or
    no depth in range with the lid in the fingers."""
    from box_opening.perception.depth_source import camera_pose_at
    from box_opening.tasks.press_demo import _spin_detect, wrist_points

    g = getattr(watcher, "grab", None)
    if node is None or g is None:
        runner.note("spot_look", points=0, why="no wrist camera" if g is None else "no node to spin")
        return None
    runner.wait_still("spot-look")
    after = node.get_clock().now().nanoseconds
    t_end = time.monotonic() + SPOT_FRAME_S
    posed = True  # every new frame so far had its pose
    while time.monotonic() < t_end:
        _spin_detect(node)
        if g.color_stamp is not None and g.depth is not None and dp._stamp_ns(g.color_stamp) > after:
            pts = wrist_points(watcher)
            if pts is None and camera_pose_at(g) is None:
                # TF does not hold the frame's stamp yet: it lands after the frame (/tf at
                # 16 Hz, a frame ~36 ms late), so ask again, for this frame or the next, until
                # SPOT_FRAME_S. 2 of 6 logged spot looks ended here with ~0.9 s left
                posed = False
                continue
            n = 0 if pts is None else len(pts)
            runner.note("spot_look", points=n, why=None if n else spot_look_why(g), frame=_keep_spot_frame(g, n))
            return pts
    why = "no new wrist frame" if posed else "no camera pose at the new frames' stamps (TF)"
    runner.note("spot_look", points=0, why="%s within %.1f s of the arm standing still" % (why, SPOT_FRAME_S))
    return None


def spot_look_why(g):
    """Why the wrist frame `g` (press_demo.wrist_points') gave no points: no
    pose for it at its stamp (camera_pose_at), or what its depth held —
    how many pixels read nothing, how many stood nearer the lens than
    clear_table.WRIST_NEAREST_M (the fingers and the lid in them) or past
    its 2 m."""
    from box_opening.perception.depth_source import camera_pose_at

    if getattr(g, "k", None) is None:
        return "no camera intrinsics"
    try:
        if camera_pose_at(g) is None:
            return "no camera pose at the frame's stamp (TF)"
    except Exception as e:  # what the note says, never what stops the run
        return "the camera pose failed (%s)" % e
    d = np.asarray(g.depth, np.float32)
    seen = np.isfinite(d) & (d > 0)
    near = seen & (d <= clear_table.WRIST_NEAREST_M)
    far = seen & (d >= 2.0)
    return "no depth in range: of %d pixels %d read nothing, %d nearer than %.2f m, %d past 2 m" % (
        d.size,
        int(d.size - seen.sum()),
        int(near.sum()),
        clear_table.WRIST_NEAREST_M,
        int(far.sum()),
    )


def _keep_spot_frame(g, n):
    """The spot look's frame into captures/wrist-spot-look-<stamp>/
    (press_demo._save_wrist_frame, written on a thread off the carry's
    planning; the newest AIM_CAPTURES_KEPT kept). Its folder, or None
    where there is no frame or no pose to keep it with."""
    from box_opening.tasks.press_demo import CAPTURES_DIR, _prune_captures, _save_wrist_frame, capture_stamp

    if getattr(g, "color", None) is None or getattr(g, "k", None) is None:
        return None
    try:
        out = _save_wrist_frame(
            g,
            CAPTURES_DIR / ("wrist-spot-look-%s" % capture_stamp()),
            "the spot look, the lid lifted: %d table points" % n,
            background=True,
            after=lambda: _prune_captures("spot-look"),
        )
    except Exception:  # a frame that cannot be kept costs the note its folder, never the run
        return None
    return None if out is None else str(out)


def _off_line(pts, line, clear_m):
    """`pts` without those within `clear_m` of the polyline `line`."""
    keep = np.ones(len(pts), bool)
    line = np.asarray(line, float)
    for a, b in zip(line[:-1], line[1:], strict=False):
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        keep &= np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) > clear_m
    return pts[keep]


def wrist_table(ctx, pts, q, lens_in_hand=None):
    """The wrist's points `pts` with what is not the table taken out, the
    arm at joints `q` holding the lid: nearer the lens than WRIST_HAND_M
    (the hand, the fingers, the lid in them), inside the held lid's column
    (its radius and HELD_LID_CLEAR_M round the pads: the lid, and what it
    hides) and over the box's footprint. `lens_in_hand`: the camera
    config's measured wrist_lens_in_hand_m."""
    from box_opening.runtime import approach
    from box_opening.tasks.press_demo import container_half

    pts = np.asarray(pts, float)
    if not len(pts):
        return pts
    R, t = approach._chain().fk(list(q))
    R, t = np.asarray(R, float), np.asarray(t, float)
    lens = t + R @ np.asarray(lens_in_hand if lens_in_hand is not None else (0.0, 0.065, -0.105), float)
    pads = t + R @ np.asarray(dp.DOWN_CONTACT_TOOL, float)
    lid_r = 0.5 * max(float(ctx.model.lid_dims[0]), float(ctx.model.lid_dims[1]))
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    keep = np.linalg.norm(pts - lens, axis=1) > WRIST_HAND_M
    keep &= np.hypot(pts[:, 0] - pads[0], pts[:, 1] - pads[1]) > lid_r + HELD_LID_CLEAR_M
    keep &= np.hypot(pts[:, 0] - bx, pts[:, 1] - by) > container_half(ctx.model) + clear_table.BOX_CLEAR_M
    return pts[keep]


def spot_map(ctx, wrist, q, scene_fix, scene_q, table_z, lens_in_hand=None):
    """The table map the lid's spot is chosen on (clear_table.table_map, as
    press_demo.room_map builds it): the wrist's frame taken with the lid
    lifted (`wrist`, wrist_table'd at the arm's joints `q`), pooled with the
    scene camera's points of the table — moved by its error at the box (the
    aim measured where the box is), what was the arm when it looked (its
    centre line at `scene_q`, clear_table.ARM_CLEAR_M round it: flatbot
    stood at HOME) left out. None where neither shows a table."""
    clouds = []
    scene = _scene_cloud(ctx, scene_fix, scene_q, table_z)
    if scene is not None:
        clouds.append(scene)
    tz = scene[1] if scene is not None else float(table_z)
    if wrist is not None and len(wrist):
        clouds.append((wrist_table(ctx, wrist, q, lens_in_hand), tz))
    return clear_table.table_map(clouds, ctx.cpose.xyz[:2]) if clouds else None


def _scene_cloud(ctx, scene_fix, scene_q, table_z):
    """spot_map's scene camera cloud: (its table points moved by its error
    at the box, the arm when it looked left out; the table they were
    measured on), or None where it kept none."""
    from box_opening.runtime import approach

    tz = getattr(scene_fix, "table_z_scene", None)
    tz = float(tz) if tz is not None and math.isfinite(float(tz)) else float(table_z)
    pts = getattr(scene_fix, "table_points", None)
    if pts is None or not len(pts):
        return None
    pts = np.asarray(pts, float) + [
        float(ctx.cpose.xyz[0]) - float(scene_fix.pose.xyz[0]),
        float(ctx.cpose.xyz[1]) - float(scene_fix.pose.xyz[1]),
        0.0,
    ]
    chain = approach._chain()
    if scene_q is not None and hasattr(chain, "points"):
        pts = _off_line(pts, chain.points(list(scene_q)), clear_table.ARM_CLEAR_M)
    return pts, tz


def aim_cloud(ctx, watcher, table_z):
    """The wrist's frame at the look pose — the aim's, kept by press_demo
    (watcher.last_aim_frame: its tag, the judged frame and the table offset
    the wrist read in it) — as a room map cloud: ((points, its own table),
    None) or (None, why). What is not the table is taken out: nearer the
    lens than WRIST_HAND_M (the fingers, shut for the press) and over the
    box's footprint and BOX_CLEAR_M. The points move AIM_FORWARD_M out along
    the box's bearing, as the box was put from the aim (reach_press.
    aim_forward); the scene camera's moved by its error at the box are as
    far moved. Its table is its own: the scene's (`table_z`) and the table
    offset the wrist read (its table sits 7-68 mm below the scene's)."""
    from box_opening.tasks.press_demo import container_half

    got = getattr(watcher, "last_aim_frame", None)
    if got is None:
        return None, "no aim kept its frame"
    _tag, frame, off_raw = got
    if frame is None:
        return None, "the aim's frame had no pose"
    if off_raw is None or not math.isfinite(float(off_raw)):
        return None, "the wrist read no table of its own at the aim"
    f, (rot, trans), _extra = frame
    if getattr(f, "depth", None) is None or getattr(f, "k", None) is None:
        return None, "the aim's frame has no depth"
    pts = clear_table.depth_points(f.depth, f.k, rot, trans)
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    keep = np.linalg.norm(pts - np.asarray(trans, float), axis=1) > WRIST_HAND_M
    keep &= np.hypot(pts[:, 0] - bx, pts[:, 1] - by) > container_half(ctx.model) + clear_table.BOX_CLEAR_M
    pts = pts[keep]
    d = math.hypot(bx, by)
    if d > 0:
        pts[:, 0] += rp.AIM_FORWARD_M * bx / d
        pts[:, 1] += rp.AIM_FORWARD_M * by / d
    return (pts, float(table_z) + float(off_raw)), None


def nearest_centre(ctx):
    """Where the 'nearest' rings centre (x, y): the box whose button is
    where the press held the pads (Ctx.held_xy), within NEAREST_HELD_MAX_M
    of ctx.cpose's button; else ctx.cpose's."""
    from box_opening.models.container import from_container

    held = getattr(ctx, "held_xy", None)
    here = (float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1]))
    if held is None:
        return here
    button = from_container(ctx.cpose, ctx.model.button_offset)
    if math.hypot(float(held[0]) - float(button[0]), float(held[1]) - float(button[1])) > NEAREST_HELD_MAX_M:
        return here
    c = rp.cpose_for_button((float(held[0]), float(held[1]), float(button[2])), ctx.cpose.yaw, ctx.model.button_offset)
    return float(c.xyz[0]), float(c.xyz[1])


def _nearest_points(ctx, table_z, tmap=None):
    """spot_candidates' last resort: table points round the box (centred
    where the press held its button: nearest_centre), the lid's disc
    NEAREST_CLEAR_M off the box's, nearest the box first, then nearest the
    arm — inside the table the scene camera saw, where it saw one, and none
    whose disc (LID_CLEAR_M round it) covers a cell of `tmap` something
    stands in: the box's holder, where a camera saw it."""
    from box_opening.tasks.press_demo import container_half

    m = ctx.model
    lid_r = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1]))
    bx, by = nearest_centre(ctx)
    ext = getattr(ctx.worlds, "table_extent", None)
    stands = np.zeros((0, 3)) if tmap is None else tmap.standing()
    first = container_half(m) + lid_r + NEAREST_CLEAR_M
    out = []
    for r in np.arange(first, NEAREST_MAX_M + 1e-9, NEAREST_STEP_M):
        ring = []
        for a in np.radians(np.arange(0.0, 360.0, NEAREST_DEG)):
            x, y = bx + r * math.cos(a), by + r * math.sin(a)
            if ext is not None and not (
                ext[0] + lid_r <= x <= ext[1] - lid_r and ext[2] + lid_r <= y <= ext[3] - lid_r
            ):
                continue
            if len(stands) and np.hypot(stands[:, 0] - x, stands[:, 1] - y).min() <= lid_r + clear_table.LID_CLEAR_M:
                continue  # the lid would come down on what stands there
            ring.append((math.hypot(x, y), x, y))
        out += [ContainerPose(xyz=(x, y, float(table_z)), yaw=0.0) for _d, x, y in sorted(ring)]
    return out


def spot_candidates(ctx, tmap, table_z, q, seeds=(), reach=None, cache=None):
    """Where the lid may go, best first, each one the hand reaches straight
    down from joints `q` (`reach`, drop_reachable's stand-in — _Reach, the
    verdicts kept —, else drop_reachable): (spot, tier, what it says).
    Tiers, each tried at most TIER_TRIES: 'clear' — where `tmap` shows
    room for the lid and the hand (press_demo.clear_lid_drops, best first,
    the hand's turn to each counted from where the arm's FK has it face at
    `q`); 'most room' — nothing clear, the spots it judged, the most room
    first; 'beside the box' — the spots beside it in the set-down zone
    (press_demo.lid_drop_candidates, drop_zone); 'nearest' — the table
    points round the box (_nearest_points). A generator: only as many as
    are asked for are checked. `cache`: the room judged on `tmap` before
    (room_ahead's, clear_table.clear_spots')."""
    from box_opening.tasks import press_demo as pd

    m = ctx.model
    tried = []

    def fresh(d):
        return all(math.hypot(d.xyz[0] - t.xyz[0], d.xyz[1] - t.xyz[1]) > 1e-3 for t in tried)

    tiers = []
    if tmap is not None:
        from box_opening.runtime import approach
        from box_opening.runtime.kinematics import mat_to_quat_xyzw

        R, _t = approach._chain().fk([float(v) for v in q])
        facing = clear_table.hand_yaw(mat_to_quat_xyzw(np.asarray(R, float)))  # where the hand holding the lid faces
        judged = []
        drops, spots, why = pd.clear_lid_drops(
            m, ctx.cpose, tmap, table_z, judged_out=judged, hand_at=facing, cache=cache
        )
        tiers.append(
            (
                "clear",
                [
                    (
                        d,
                        "where the cameras see room for it and the hand (%.0f mm to spare%s; the hand "
                        "turns %+.0f deg)"
                        % (
                            1000 * s.room_m,
                            "" if s.what == "nothing" else ", %s nearest anything" % s.what,
                            math.degrees(s.rotation),
                        ),
                    )
                    for d, s in zip(drops, spots, strict=False)
                ],
            )
        )
        most = sorted(judged, key=lambda s: -s.room_m)
        tiers.append(
            (
                "most room",
                [
                    (
                        ContainerPose(xyz=(s.xy[0], s.xy[1], float(table_z)), yaw=s.turn),
                        "nothing reads clear (%s): the most room seen, %s %.0f mm short"
                        % (why or "no spot judged", s.what, -1000 * s.room_m),
                    )
                    for s in most
                ],
            )
        )
    zone = pd.drop_zone(m, ctx.worlds)
    said = "no camera shows the table" if tmap is None else "the hand reaches no spot the cameras judged"
    tiers.append(
        (
            "beside the box",
            [
                (
                    ContainerPose(xyz=tuple(float(v) for v in xyz), yaw=0.0),
                    "%s: beside the box, in the set-down zone" % said,
                )
                for xyz, _side in pd.lid_drop_candidates(m, ctx.cpose, table_z, zone=zone)
            ],
        )
    )
    tiers.append(
        (
            "nearest",
            [
                (
                    d,
                    "%s, nor beside the box in the set-down zone: the nearest table point the hand "
                    "reaches, off the box" % said,
                )
                for d in _nearest_points(ctx, table_z, tmap)
            ],
        )
    )
    for tier, cands in tiers:
        n = 0
        for d, says in cands:
            if n >= TIER_TRIES[tier]:
                break
            if not fresh(d):
                continue
            n += 1
            tried.append(d)
            _abort_check(ctx)  # each reach check is an IK walk: a Ctrl+C goes out between two
            why = (reach or drop_reachable)(ctx, d, q, seeds)
            if why is None:
                yield d, tier, says


def _spot_key(d):
    """A spot's identity for _Reach: where, to 0.1 mm, and the hand's turn."""
    return (round(float(d.xyz[0]), 4), round(float(d.xyz[1]), 4), round(float(d.yaw) % (2 * math.pi), 3))


def _same_spot(a, b):
    """Whether spots `a` and `b` are the one spot for a carry planned to
    either: AHEAD_SAME_SPOT_M apart at most, the hand turned the same."""
    turn = abs(math.remainder(float(a.yaw) - float(b.yaw), 2 * math.pi))
    return math.hypot(a.xyz[0] - b.xyz[0], a.xyz[1] - b.xyz[1]) <= AHEAD_SAME_SPOT_M and turn < 1e-3


class _Reach:
    """drop_reachable kept by spot (_spot_key): (why, how long it took).
    `known`, the verdicts of another one from (nearly) the same joints —
    carry_ahead's, the arm within DRIFT_REPLAN_RAD of where it judged them
    — are taken as they are, their time saved (saved_s); every spot not
    among them is judged here (ik_s) and kept (`found`)."""

    def __init__(self, known=None):
        self.known = dict(known or {})
        self.found = {}
        self.ik_s = 0.0
        self.saved_s = 0.0
        self.reused = 0

    def __call__(self, ctx, drop, q, seeds=()):
        key = _spot_key(drop)
        if key in self.known:
            why, took = self.known[key]
            self.saved_s += took
            self.reused += 1
            return why
        t = time.monotonic()
        why = drop_reachable(ctx, drop, q, seeds)
        took = time.monotonic() - t
        self.ik_s += took
        self.found[key] = (why, took)
        return why


# ---- the way back to the look pose, and the look ----------------------------


def _look_joints(way):
    """Where `way` (a reach_press.WayIn) starts its way down: the aimed
    look pose."""
    if way.move:
        return [float(v) for v in way.move[-1]]
    first = way.down.pts if way.down.walked else way.slide
    return [float(v) for v in first[0]]


def _down_backwards(ctx, way, world):
    """The way down backwards, to the look pose it starts at: the turn,
    then the slide — or the walk — each reversed (a proven line flown
    back is the same joints, the runner's gates alike)."""
    lines = []
    if way.down.pts is not None and len(way.down.pts) > 1:
        lines.append(("reach:slide-out" if way.down.walked else "depth:turn-back", list(way.down.pts)[::-1]))
    if not way.down.walked and way.slide is not None and len(way.slide) > 1:
        lines.append(("reach:slide-out", list(way.slide)[::-1]))
    fast = rp.free_line_speed(ctx)
    return [
        dp.compliant_line(ctx, name, [[float(v) for v in q] for q in pts], way.plan.quat, fast, world)
        for name, pts in lines
    ]


def way_back_legs(ctx, way, q_held):
    """From the hold, the arm held at joints `q_held`, back to the aimed
    look pose: the rise up the stroke (reach_press._rise_legs), then the way
    down backwards (_down_backwards) — one execution where they meet. A rise
    that is not the stroke's own (a straight line to its start) is followed
    by a walk from its end to the look pose, the one the runner flies.
    Returns (legs, None, None); (None, 'up', why) where no way up flies; or
    (the rise's legs, 'look', why) where no way from its end to the look
    pose does."""
    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import mat_to_quat_xyzw

    world = _full_world(ctx)
    up, on_stroke, why = rp._rise_legs(ctx, way.stroke, q_held, world)
    if up is None:
        return None, "up", why
    if on_stroke:
        return up + _down_backwards(ctx, way, world), None, None
    chain = approach._chain()
    R, t = chain.fk(_look_joints(way))
    quat = mat_to_quat_xyzw(np.asarray(R, float))
    q = up[-1].goal_joints if up else [float(v) for v in q_held]
    pts, why = chain.walk(
        q,
        [float(v) for v in t],
        quat,
        step_m=dp.TURN_STEP_M,
        step_rad=dp.TURN_STEP_RAD,
        max_step_rad=dp.TURN_MAX_JUMP_RAD,
    )
    if pts is not None:
        why = dp.line_refusal(chain, pts)
    if pts is None or why is not None:
        return up, "look", "no way from the stroke's start back to the look pose flies (%s)" % why
    return up + [dp.compliant_line(ctx, "reach:slide-out", pts, quat, rp.free_line_speed(ctx), world)], None, None


@dataclass
class KnobLook:
    """What the look after a press saw (knob_look). The grip needs the knob
    up (popped): the fingers close on it at its base. A lid that rose
    (lid_rose) is the box open, but with its knob flush it gives them
    nothing to hold, and knob_look stops the run (`stop`, LID_POP_M)."""

    found: bool
    up_mm: float = None  # the knob above its lid: the highest any look read
    shift_m: float = None  # the button from where the press was aimed
    lid_dz_m: float = None  # the lid, by the wrist's own two readings
    status: str = ""
    stop: str = None  # why the run stops here: the box not found, or moved, or its lid released with the knob flush
    lid_rose: bool = False  # the lid rose LID_POP_M with the button where it was: released

    @property
    def popped(self):
        """The knob up (press_demo.knob_is_up), whatever the lid did: what
        the grip closes on. A risen lid with its knob flush is not."""
        from box_opening.tasks.press_demo import knob_is_up

        return self.found and knob_is_up(self.up_mm)


def knob_look(node, ctx, runner, watcher, button, yaw, lid_before, press):
    """From the look pose after press number `press`: the oblique aim at
    the aimed `button` (reach_press._aim, the knob's height above its own
    lid), once more where it reads the knob down or finds no button
    (KNOB_RECHECK_AIMS) — the highest reading the verdict, as the staging
    run's after-look's was. A box the press slid farther than the staging
    grip took (depth_press.SLIDE_OK_M) or whose lid the wrist
    reads BOX_LIFTED_M off its reading before (`lid_before`) moved: STOP.
    A lid LID_POP_M higher, the button where it was, was released
    (lid_rose): its knob up, the grip follows it up; its knob flush, STOP —
    nothing to hold, and a released lid is not pressed again. The reach_knob
    note says which, and why the run stops."""
    from box_opening.tasks.press_demo import knob_is_up

    best, last, status, looks = None, None, "", 0
    for _k in range(1 + KNOB_RECHECK_AIMS):
        got, status, up, measured = rp._aim(node, watcher, yaw, button[2], button[:2], runner, "reach-after")
        looks += 1
        if got is None:
            continue
        last = (got, up, measured)
        if up is not None and (best is None or up > best):
            best = float(up)
        if knob_is_up(up):
            break
    if last is None:
        runner.note("reach_knob", press=press, found=False, looks=looks, why=status)
        log.info(
            "[press_demo] AFTER PRESS %d: the wrist does not find the button from the look pose (%s)" % (press, status)
        )
        return KnobLook(
            False,
            status=status,
            stop="the wrist does not find the box after the press (%s) — the knob is " "not judged" % status,
        )
    got, up, measured = last
    shift = rp.judge_box_moved(button, got[0])
    lid_after = rp._lid_z(measured[2], up)
    dz = None if lid_before is None or lid_after is None else lid_after - lid_before
    rose = dz is not None and LID_POP_M[0] <= dz <= LID_POP_M[1] and shift <= dp.SLIDE_OK_M
    look = KnobLook(True, best, shift, dz, status, lid_rose=rose)
    if shift > dp.SLIDE_OK_M:
        look.stop = "the press MOVED the box %.0f mm — check it is still standing on the table" % (1000 * shift)
    elif dz is not None and not rose and abs(dz) > rp.BOX_LIFTED_M:
        look.stop = "the press moved the box: its lid %+.0f mm — check it is still standing on the table" % (1000 * dz)
    elif rose and not look.popped:
        # the next run cannot tell: its aim judges the box by the knob alone
        # (press_demo.refuse_open_box), and to it this one is shut
        look.stop = (
            "the lid rose %+.0f mm (released) but its knob is not up (%.1f mm above the lid) — the fingers "
            "have nothing to hold, so the lid is not taken off; the box is left open: lift the lid off by "
            "hand, or shut it, before the next run (which would take it for shut and press it)" % (1000 * dz, best)
        )
    runner.note(
        "reach_knob",
        press=press,
        found=True,
        looks=looks,
        knob_up_mm=None if best is None else round(best, 1),
        popped=look.popped,
        shift_mm=round(1000 * shift, 1),
        lid_dz_mm=None if dz is None else round(1000 * dz, 1),
        lid_rose=rose,
        stop=look.stop,
    )
    if rose:
        verdict = "%s, and its LID ROSE %+.0f mm — the box is open" % (
            "UP %.1f mm" % best if look.popped else "%.1f mm above the lid" % best,
            1000 * dz,
        )
    else:
        verdict = (
            "UP %.1f mm — the box is open" % best
            if look.popped
            else ("%.1f mm above the lid" % best if best is not None else "not judged")
        )
    log.info(
        "[press_demo] AFTER PRESS %d: knob %s (%d look%s); the button %.0f mm from where the press was aimed%s"
        % (
            press,
            verdict,
            looks,
            "" if looks == 1 else "s",
            1000 * shift,
            "" if dz is None or rose else ", its lid %+.0f mm" % (1000 * dz),
        )
    )
    return look


# ---- the grip ---------------------------------------------------------------


@dataclass
class GripDescent:
    """The grip's stroke from the hand turned down at `q0` (grip_descent):
    `legs` [(name, joint points, speed)] as a press stroke's
    (reach_press.way_in_legs flies them), 'grip:approach' at grip_speed to
    GRIP_SLOW_M above where it ends, then 'grip:down' at TOUCH_SPEED, the
    closed pads' reference ending at `pad_z`."""

    legs: list
    quat: list
    q0: list
    pad_z: float


def grip_descent(chain, cfg, q0, quat, lid_z):
    """Straight down from the hand turned down at joints `q0`, holding
    `quat`, until the closed pads' reference (depth_press.DOWN_CONTACT_TOOL)
    stands GRIP_BELOW_LID_M under the lid at `lid_z`. Returns (GripDescent,
    None) or (None, why)."""
    q0 = [float(v) for v in q0]
    R0, t0 = chain.fk(q0)
    R0, t0 = np.asarray(R0, float), np.asarray(t0, float)
    pad0 = t0 + R0 @ np.asarray(dp.DOWN_CONTACT_TOOL, float)
    end = float(lid_z) - GRIP_BELOW_LID_M
    if float(pad0[2]) < end + GRIP_SLOW_M:
        return None, "the pads already stand %.0f mm above where the grip ends" % (1000 * (float(pad0[2]) - end))

    def down_to(q, z):
        tool_z = float(t0[2]) + (z - float(pad0[2]))  # the attitude held: the tool drops as the pads do
        pts, why = chain.straight_line(q, [float(t0[0]), float(t0[1]), tool_z], quat)
        return pts, None if pts is not None else "the grip's way down is not a straight line the arm can fly (%s)" % why

    approach, why = down_to(q0, end + GRIP_SLOW_M)
    if approach is None:
        return None, why
    down, why = down_to(approach[-1], end)
    if down is None:
        return None, why
    return GripDescent(
        [("grip:approach", approach, cfg.grip_speed), ("grip:down", down, dp.TOUCH_SPEED)], list(quat), q0, end
    ), None


def solve_grip_way(ctx, cfg, button, tilt, q_from, lid_z):
    """The way from joints `q_from` (the look pose) down to the knob of the
    `button` the press was aimed at, at `tilt`: the press's way in
    (reach_press.reach_from_here's move to the look pose and slide, then
    walk_or_turn) with the grip's stroke (grip_descent onto the lid at
    `lid_z`) where the press's was — every line one the runner flies
    (depth_press.line_refusal). Returns (reach_press.WayIn, None) or (None,
    why)."""
    from box_opening.runtime import approach

    chain = approach._chain()
    move, plan, why = rp.reach_from_here(ctx, cfg, button, tilt, need_stroke=False, q=q_from)
    if plan is None:
        return None, why
    why = dp.line_refusal(chain, move)
    if why is not None:
        return None, "the move to its look pose (%s)" % why
    down, why = rp.walk_or_turn(ctx, move[-1] if move else q_from, plan.slide_pts, plan.start_xyz, plan.quat, lid_z)
    if down is None:
        return None, "the way down (%s)" % why
    if down.slide is not None:  # it turns lower down, as the press did: the slide carries on to there
        plan.slide_pts, plan.start_xyz = down.slide, down.start_xyz
    grip, why = grip_descent(chain, cfg, down.pts[-1] if down.pts else plan.slide_pts[-1], down.quat, lid_z)
    if grip is None:
        return None, why
    lines = ([] if down.walked else [("the slide in", plan.slide_pts)]) + [(n, p) for n, p, _s in grip.legs]
    for name, line in lines:
        why = dp.line_refusal(chain, line)
        if why is not None:
            return None, "%s: %s" % (name, why)
    return rp.WayIn(move, plan, down, None if down.walked else plan.slide_pts, grip), None


def _grip_backwards(ctx, way, world):
    """The grip's way down backwards, from the knob to the look pose: up
    its stroke, then _down_backwards."""
    pts = []
    for _name, p, _speed in way.stroke.legs:
        pts += [[float(v) for v in q] for q in (p[1:] if pts else p)]
    legs = (
        [dp.compliant_line(ctx, "grip:up", pts[::-1], way.stroke.quat, rp.free_line_speed(ctx), world)]
        if len(pts) > 1
        else []
    )
    return legs + _down_backwards(ctx, way, world)


# ---- the lid's way off the box, and from its spot to the table -------------


@dataclass
class LiftOff:
    """From the knob held to the spot look (plan_lift): 'lid:lift'
    straight up and 'lid:clear' up and back toward the arm, `back_m`
    back — one execution after the close."""

    legs: list
    back_m: float = 0.0
    q_grip: list = None  # the joints the knob was held at, the lid on the box
    q_top: list = None  # ... and at the top of the lift, straight above
    quat: list = None  # the hand's attitude all the way


def _clear_after_lift(ctx, chain, q, quat):
    """The clear from the lifted lid at joints `q`: to the first of
    reach_press.clear_targets whose straight line flies and leaves every
    joint room for the planner to start from (reach_press._near_a_limit),
    else the first that flies, as plan_way_out picks. Returns (points, how
    far back, None) or (None, 0, why)."""
    R, t = chain.fk(q)
    chosen, whys = None, []
    for target in rp.clear_targets(ctx, np.asarray(R, float), t):
        back = math.hypot(target[0] - float(t[0]), target[1] - float(t[1]))
        pts, why = chain.straight_line(q, target, quat)
        if pts is not None:
            why = dp.line_refusal(chain, pts)
        if pts is None or why is not None:
            whys.append("%.0f mm back: %s" % (1000 * back, why))
            continue
        pts = [[float(v) for v in p] for p in pts]
        near = rp._near_a_limit(ctx, pts[-1])
        if not near:
            return pts, back, None
        whys.append("%.0f mm back: %s at its end" % (1000 * back, near))
        chosen = chosen or (pts, back)
    if chosen is not None:
        return chosen[0], chosen[1], None
    return None, 0.0, "no way up and back from the lifted lid flies (%s)" % "; ".join(whys)


def plan_lift(ctx, cfg, q_grip, quat):
    """The lid's way off the box from the knob held at joints `q_grip`, the
    hand at `quat` (LiftOff): straight up lift_off_m, then the clear
    (_clear_after_lift). The grip's lookahead, while the hand comes down to
    the knob. Returns (LiftOff, None) or (None, why)."""
    from box_opening.runtime import approach

    chain = approach._chain()
    world = _full_world(ctx)
    q_grip = [float(v) for v in q_grip]
    _R, t = chain.fk(q_grip)
    lift, why = chain.straight_line(q_grip, [float(t[0]), float(t[1]), float(t[2]) + lift_off_m(ctx.model)], quat)
    if lift is not None:
        why = dp.line_refusal(chain, lift)
    if lift is None or why is not None:
        return None, "the lift off the canister is not a straight line the arm can fly (%s)" % why
    lift = [[float(v) for v in p] for p in lift]
    clear, back, why = _clear_after_lift(ctx, chain, lift[-1], quat)
    if clear is None:
        return None, why
    return LiftOff(
        [
            dp.compliant_line(ctx, "lid:lift", lift, quat, cfg.lift_speed, world),
            dp.compliant_line(ctx, "lid:clear", clear, quat, rp.free_line_speed(ctx), world),
        ],
        back,
        q_grip=q_grip,
        q_top=lift[-1],
        quat=[float(v) for v in quat],
    ), None


@dataclass
class Carry:
    """From the spot look to the lid on the table (plan_carry): `legs`
    'lid:carry' (the planner's) to the hover over `drop`, 'place:down' and
    'place:touch' — one execution — and `retreat`, the joint points back up
    to the hover after the release, at `quat`."""

    legs: list
    retreat: list
    quat: list
    drop: object
    over_m: float = None  # how far past where the plan has the lid meet the table the touch ends: -SETDOWN_RELEASE_M
    k_z: float = None  # the hand's spring straight down there, N/m
    contact: object = None  # the set-down watched as it flies (watch_setdown): where the lid met the table


# The set-down watched for where the lid meets the table (watch_setdown,
# tasks/press_contact's PressContact) from where its slow stretch begins,
# SETDOWN_SLOW_M above where the plan puts that, on the execution's last
# stretch below there (PressContact final=True: never the lift off a box that
# stands lower). Replayed on the 16 recorded set-downs of 10-29 and 10-30, a
# watch over the fast way down (from 0.07 m) met one 88 mm early, from 0.05 m
# none; at PLACE_TOUCH_SPEED the slow stretch is what the press's slow leg
# is, which met none of the 25 presses early. Cut where it meets
# (SETDOWN_FROM_CONTACT; past 0: the reference held where the pads met the
# table, the lag of a detection under it), and the release opens there: the
# first touch is the lid let go — on its rim if it leans, to settle flat (in
# 10 of the 16 the hand went on down 8-20 mm after it, the lid settling or
# the knob slipping in the fingers, the spring pressing on).
SETDOWN_WATCH_M = SETDOWN_SLOW_M
SETDOWN_FROM_CONTACT = True


def watch_setdown(ctx, legs, carry):
    """The set-down of `carry` (plan_carry's) watched as it flies in the
    execution `legs` begins (the first motion leg's stop_when): a
    PressContact on where the lid meets the table — SETDOWN_WATCH_M above
    where the plan puts that on, the execution's last stretch below there —
    cut where it meets (past 0) with SETDOWN_FROM_CONTACT. Kept on the carry
    (carry.contact) for the reach_setdown row. Returns it, or None: nothing
    to watch."""
    from box_opening.runtime import approach
    from box_opening.runtime.legs import Kind
    from box_opening.tasks.press_contact import PressContact, pads_z

    lead = next((lg for lg in legs if lg.kind is Kind.MOTION), None)
    touch = carry.legs[-1] if carry.legs else None
    if lead is None or touch is None or touch.goal_joints is None:
        return None
    chain = approach._chain()
    end = pads_z(chain, touch.goal_joints)
    over = float(carry.over_m or 0.0)
    expect = end + over  # where the plan has the lid meet the table
    carry.contact = PressContact(
        chain, ctx.client, expect, expect + SETDOWN_WATCH_M, 0.0, end, cut=SETDOWN_FROM_CONTACT, final=True
    )
    lead.stop_when = carry.contact
    return carry.contact


def plan_carry(ctx, cfg, q_from, drop):
    """The carry from joints `q_from` (where the spot look stands) to the
    spot `drop` (planned in the planner's own world, worlds.PLANNER_WORLD:
    it is told nothing of the box or the table) — then the set-down's straight lines
    from its end and the retreat's back up. Returns (Carry, None) or (None,
    why)."""
    from box_opening.runtime import approach

    chain = approach._chain()
    world = _full_world(ctx)
    q_from = [float(v) for v in q_from]
    quat, contact, hover = setdown_poses(ctx, drop)
    try:
        carry, _st = rp._plan_motion(
            ctx, PlanState(joints=q_from, chain=0), "lid:carry", ("pose", hover, quat), world, TRANSIT_SPEED
        )
    except RuntimeError as e:
        return None, "the carry (%s)" % rp._short(e)
    if carry.traj is not None:
        carry.traj = approach.on_branch_of(carry.traj, q_from)  # continues the clear, not a turn away
        carry.goal_joints = [float(v) for v in carry.traj.points[-1].positions]
    carry.gains = dp.COMPLIANT_GAINS
    lines = []
    q = carry.goal_joints
    k_z = None
    z_at = {"place:down": contact[2] + SETDOWN_SLOW_M, "place:touch": contact[2] + SETDOWN_RELEASE_M}
    for name in ("place:down", "place:touch", "place:retreat"):
        if name == "place:touch":  # the hand's spring where the slow stretch begins, for the record
            k_z = float(dp.fingertip_stiffness_z(chain, q))
        z = z_at.get(name, hover[2])
        pts, why = chain.straight_line(q, [hover[0], hover[1], z], quat)
        if pts is not None:
            why = dp.line_refusal(chain, pts)
        if pts is None or why is not None:
            return None, "%s (%s)" % (name, why)
        lines.append([[float(v) for v in p] for p in pts])
        q = lines[-1][-1]
    down_speed = max(float(cfg.setdown_speed), rp.free_line_speed(ctx))
    legs = [
        carry,
        dp.compliant_line(ctx, "place:down", lines[0], quat, down_speed, world),
        dp.compliant_line(ctx, "place:touch", lines[1], quat, PLACE_TOUCH_SPEED, world),
    ]
    return Carry(legs, lines[2], quat, drop, over_m=-SETDOWN_RELEASE_M, k_z=k_z), None


def plan_home_after(ctx, cfg, carry):
    """The retreat back up from the set-down and home planned from its
    end, with the lid lying at the spot (ctx.lid_at, for this plan only):
    the legs after the release, home under its own gains (MEDIUM), or
    (None, why). The carry's lookahead."""
    from box_opening.runtime import approach

    world = _full_world(ctx)
    q = carry.retreat[-1]
    was = ctx.lid_at
    ctx.lid_at = carry.drop  # home is planned around the lid where it will lie
    try:
        home = rp._home_legs(ctx, cfg, start=q)
    except RuntimeError as e:
        return None, "home after the set-down (%s)" % rp._short(e)
    finally:
        ctx.lid_at = was
    for leg in home:
        if leg.traj is not None:
            leg.traj = approach.on_branch_of(leg.traj, q)
            leg.goal_joints = [float(v) for v in leg.traj.points[-1].positions]
            q = leg.goal_joints
        if leg.guard is None and getattr(leg, "compliant", None) is None:
            # one gains spec: a step off a joint limit merges with the retreat; home itself flies its own,
            # MEDIUM (core.as_home), a goal of its own after the retreat
            leg.gains = dp.COMPLIANT_GAINS
    retreat = dp.compliant_line(ctx, "place:retreat", carry.retreat, carry.quat, rp.free_line_speed(ctx), world)
    return [retreat, *home], None


# ---- the runs ---------------------------------------------------------------


def _started(last):
    return last.flown_traj is not None and bool(last.progress)


def _stop_home(ctx, cfg, runner, why):
    """STOP from the look pose, in free air: home (reach_press._go_home,
    which lets go of a home stopped part-way) — the one planned from there
    while the way back to it, or the back-out, flew (ctx.home_ahead), where
    the arm stands where it starts; else planned here. The run log says
    which (home_ahead). Returns 1."""
    ahead, ctx.home_ahead = ctx.home_ahead, None
    if not ahead:
        # none in hand (off_rad None): that lookahead failed — the runner's
        # line says why — or a way in, or the grip's way down, did not start
        # since; home is planned here, at rest, as before
        runner.note("home_ahead", flown=False, off_rad=None)
    rp._go_home(ctx, cfg, runner, "STOP: %s" % why, ahead=ahead)
    return 1


def _note_frames(ctx, **facts):
    frames = getattr(ctx, "mission_frames", None)
    if frames is not None:
        frames.note(**facts)


def _press_and_back(ctx, cfg, runner, way, legs, top_z, press, direct=None):
    """Press number `press`: the way in `legs` (way_in_legs of `way`) as
    one execution, the hold on the button whose top is at `top_z`, and the
    way back to the look pose (_way_back) as one more. Returns (None,
    where the hold held the pads when it felt the button, else None) at the
    look pose, or (the exit code, None) where the run stopped: home from the
    look pose where the way in did not start, the arm let go and held where
    a move stopped part-way, up off the button and home from over the box
    where the way back did not start. Home is planned from the look pose
    while the way back flies: ctx.home_ahead, for a stop there (_stop_home).
    `direct(seen)` (the first press's, GRIP_FROM_PRESS): after a hold that
    felt the button, the grip straight from it (_grip_from_hold) in place of
    the way back — its exit code where it ended the run, else (_MISSED) the
    arm back at the look pose, as after the way back."""
    from box_opening.tasks.press_demo import viz_hook

    names = [lg.name for lg in legs]
    if way.down.short_deg:
        log.info(
            "[press_demo] TURN: %.0f deg short of straight down (no hand nearer straight down flies here)"
            % way.down.short_deg
        )
    log.info(
        "[press_demo] %sONE MOTION from the look pose to below the button: %s"
        % ("PRESS %d of %d: " % (press, PRESSES) if press > 1 else "", ", ".join(names))
    )
    log.info("[press_demo] %s" % dp.press_text(way.stroke, cfg))
    ctx.home_ahead = None  # the arm leaves the look pose: no home planned before is flown from where it comes back
    viz_hook("press", way.stroke, top_z, cfg)  # the stroke, drawn off this thread (perception/viz)
    # where the pads meet the box ends the first press's stroke past it; a press after it (the
    # knob not up) flies its full stroke, watched
    contact = dp.watch_press(ctx, legs, way.stroke, top_z, cut=press == 1)
    res = runner.run(legs, execute=True)
    last = res[-1]
    flown = all(r.ok for r in res)
    retimed = rp._note_flight(runner, "reach_way_in_flight", names, last) if last.flown_traj is not None else {}
    if not flown and not _started(last):
        return _stop_home(
            ctx,
            cfg,
            runner,
            "the way in did not start (%s) — nothing was pressed%s"
            % (last.detail or last.outcome, "" if press == 1 else " again"),
        ), None
    if not flown:
        log.info(
            "[press_demo] STOP: the way in stopped during %s (%s) — the arm holds"
            % (rp._phase_at(names, retimed.get("seg_start_fracs") or [0.0], last.progress), last.detail or last.outcome)
        )
        rp.let_go(ctx)
        return 1, None
    runner.note("reach_slid_in", turned_down=way.down.walked, residual_nm=None)
    seen = {}
    held = dp.compliant_hold(ctx, cfg, runner, way.stroke, top_z, seen, contact=contact)
    if held.felt:
        _note_frames(ctx, stage="pressed", touch_xyz=list(seen["touch_xyz"]) if seen.get("touch_xyz") else None)
    held_z = float(seen["touch_xyz"][2]) if held.felt and seen.get("touch_xyz") else None
    if held_z is not None:
        ctx.held_xy = (float(seen["touch_xyz"][0]), float(seen["touch_xyz"][1]))  # where the button was held
    if direct is not None and held_z is not None and seen.get("held_joints"):
        got = direct(seen)
        return (None, held_z) if got == _MISSED else (got, None)
    code = _way_back(ctx, cfg, runner, way, seen.get("held_joints") or ctx.client.joints(), held)
    return (None, held_z) if code is None else (code, None)


def _way_back(ctx, cfg, runner, way, q, held):
    """From the button under `way`'s stroke, the arm at joints `q`, back to
    the look pose (way_back_legs), home planned from there while it flies
    (ctx.home_ahead). None there; else the exit code where it stopped: the
    arm let go and held where a move stopped part-way, up off the button
    and home from over the box where the way back did not start. `held`:
    the hold's (depth_press.DepthPress)."""
    back, kind, why = way_back_legs(ctx, way, q)
    if kind == "up":
        log.info("[press_demo] STOP: the way up is not a straight line the arm can fly (%s) — the arm holds" % why)
        rp.let_go(ctx)
        return 1
    if kind == "look":
        log.info("[press_demo] STOP: %s — up off the button, then home from over the box; the knob is not judged" % why)
        result, homed = rp._rise_then_home(ctx, cfg, runner, back, held, None, None)
        if not result.free or not homed:
            rp.let_go(ctx)
        return 1
    groups = merge_groups(back)
    log.info(
        "[press_demo] %s from the button back to the look pose: %s"
        % ("ONE MOTION" if len(groups) == 1 else "in %d goals" % len(groups), ", ".join(lg.name for lg in back))
    )
    # Home from the look pose, in the full world, planned while the way back
    # flies (the runner's lookahead), for a stop there: the full runs of
    # 2026-10-28 planned it with the arm standing at the look pose, 0.21 and
    # 0.23 s at 16:15:35 and 16:16:09, where the press-only runs planned
    # their home while the way in flew, 7 of 7. Nothing else asks the
    # planner, which takes one goal at a time, in that flight: its legs are
    # plan-free lines, never re-planned. The lid coming off leaves it unused:
    # a plan of ~0.17 s (the day's median) under a ~5 s motion.
    res = runner.run(back, execute=True, lookahead=lambda end: rp._home_legs(ctx, cfg, start=end))
    last = res[-1]
    if len(res) == 1 and last.flown_traj is not None:
        rp._note_flight(runner, "reach_way_back_flight", [lg.name for lg in groups[0]], last)
    if all(r.ok for r in res):
        ctx.home_ahead = runner.lookahead_result  # None where it failed: a stop plans home where the arm stands
        return None
    if _started(last):
        log.info(
            "[press_demo] STOP: the way back to the look pose stopped during %s (%s) — the arm holds"
            % (rp._in_flight(back, last, runner.last_retime), last.detail or last.outcome)
        )
        rp.let_go(ctx)
        return 1
    if len(res) == 1:
        # nothing moved, the pads still on the button: up on its own,
        # solved again from where the arm stands, then home from over the box
        up, _on_stroke, why = rp._rise_legs(ctx, way.stroke, ctx.client.joints(), _full_world(ctx))
        if up is None:
            log.info("[press_demo] STOP: the way up is not a straight line the arm can fly (%s) — the arm holds" % why)
            rp.let_go(ctx)
            return 1
        log.info(
            "[press_demo] STOP: the way back to the look pose did not start (%s) — up off the button, then home "
            "from over the box" % (last.detail or last.outcome)
        )
        result, homed = rp._rise_then_home(ctx, cfg, runner, up, held, None, None)
    else:
        log.info(
            "[press_demo] STOP: the way back's %s did not start (%s) — home from over the box"
            % (rp._in_flight(back, last, runner.last_retime), last.detail or last.outcome)
        )
        result, homed = held, rp._home_from_over_box(ctx, cfg, runner, held)
    if not result.free or not homed:
        rp.let_go(ctx)
    return 1


def grip_from_hold_lines(ctx, way, q_held):
    """The grip straight from the hold, the arm at joints `q_held` on the
    button under `way`'s stroke: (rise, down, None) — GRIP_RISE_M straight
    up onto the stroke's line, then down it to where the pads stand
    GRIP_LID_ABOVE_HELD_M - GRIP_BELOW_LID_M over where the hold held them
    (grip_lid_z), each as joint points — or (None, None, why)."""
    from box_opening.runtime import approach

    chain = approach._chain()
    q_held = [float(v) for v in q_held]
    _R, t_held = chain.fk(q_held)
    _R, t_line = chain.fk([float(v) for v in way.stroke.q0])  # the stroke's line: the aimed knob
    quat = way.stroke.quat
    top = [float(t_line[0]), float(t_line[1]), float(t_held[2]) + GRIP_RISE_M]
    rise, why = chain.straight_line(q_held, top, quat)
    if rise is not None:
        why = dp.line_refusal(chain, rise)
    if rise is None or why is not None:
        return None, None, "the rise off the button (%s)" % why
    bottom = [top[0], top[1], float(t_held[2]) + GRIP_LID_ABOVE_HELD_M - GRIP_BELOW_LID_M]
    down, why = chain.straight_line(rise[-1], bottom, quat)
    if down is not None:
        why = dp.line_refusal(chain, down)
    if down is None or why is not None:
        return None, None, "the way down to the knob (%s)" % why
    return [[float(v) for v in p] for p in rise], [[float(v) for v in p] for p in down], None


def _grip_from_hold(ctx, cfg, runner, eyes, way, seen, room):
    """The grip straight from the first press's hold (GRIP_FROM_PRESS): the
    fingers opening to the knob's width as the hand rises off the button,
    the lift and the first spot the hand reaches planned while it does
    (grip_ahead_rise, from the way down's end), the way down to the knob
    once they are open, the carry and home planned while it comes down
    (grip_ahead_down), then the close and the lift to the set-down as from
    the look pose (_close_and_carry). Returns its exit code where it ended the
    run (0: the lid off, the arm home); _MISSED where the hand went back to
    the look pose instead — the lines did not fly, the close did not hold
    — for the look to judge the knob; the arm holding where it stopped part
    way, 1."""
    world = _full_world(ctx)
    held_z = float(seen["touch_xyz"][2])
    lid_z = grip_lid_z(None, held_z)
    rise, down, why = grip_from_hold_lines(ctx, way, seen["held_joints"])
    if rise is None:
        runner.note("reach_grip_refused", attempt=0, why=why, from_press=True)
        log.info("[press_demo] no grip straight from the press (%s): back to the look pose" % why)
        return _MISSED if _way_back(ctx, cfg, runner, way, seen["held_joints"], dp.DepthPress(free=True)) is None else 1
    quat = way.stroke.quat
    runner.note(
        "reach_grip",
        lid_z=round(lid_z, 4),
        pads_z=round(lid_z - GRIP_BELOW_LID_M, 4),
        tilt_deg=way.plan.tilt_deg,
        how="straight from the press: %.0f mm over where the hold held the pads" % (1000 * GRIP_LID_ABOVE_HELD_M),
        attempt=0,
        from_press=True,
    )
    log.info(
        "[press_demo] GRIP STRAIGHT FROM THE PRESS: up %.0f mm, the fingers opening to the knob's width as it "
        "rises, then down to the knob, the closed pads %.0f mm under the lid at %.3f (%.0f mm over where the press "
        "held them)" % (1000 * GRIP_RISE_M, 1000 * GRIP_BELOW_LID_M, lid_z, 1000 * GRIP_LID_ABOVE_HELD_M)
    )
    _video(ctx, "GRIP straight from the press: up, the fingers open, down to the knob")
    opened = _gripper_leg(
        ctx, PlanState(joints=None, chain=0), "grip:open", grip_open_cmd(ctx.model), world, defer_join=True
    )
    up = dp.compliant_line(ctx, "grip:rise", rise, quat, rp.free_line_speed(ctx), world)
    lower = dp.compliant_line(ctx, "grip:down", down, quat, dp.TOUCH_SPEED, world)
    # the lookahead split over both (grip_ahead_rise, grip_ahead_down): the lift and the
    # first spot while the hand rises, the carry and home while it comes down
    res = runner.run(
        [opened, up],
        execute=True,
        lookahead=lambda _end: grip_ahead_rise(ctx, cfg, eyes, lower.goal_joints, quat, room),
    )
    first, rise_stood = runner.lookahead_result, runner.lookahead_stood_s
    spread = runner.join_pending()  # the fingers open before the hand comes down round the knob
    if not all(r.ok for r in res) or (spread is not None and not spread.ok):
        bad = res[-1] if not all(r.ok for r in res) else spread
        if _started(res[-1]) and not all(r.ok for r in res):
            log.info(
                "[press_demo] STOP: the rise off the button stopped (%s) — the arm holds" % (bad.detail or bad.outcome)
            )
            rp.let_go(ctx)
            return 1
        log.info(
            "[press_demo] the grip straight from the press did not start (%s): back to the look pose"
            % (bad.detail or bad.outcome)
        )
        return _MISSED if _way_back(ctx, cfg, runner, way, ctx.client.joints(), dp.DepthPress(free=True)) is None else 1
    res = runner.run([lower], execute=True, lookahead=None if first is None else (lambda _end: grip_ahead_down(first)))
    ahead = _lookahead(runner)
    if ahead is not None and isinstance(ahead[2], CarryAhead):
        # how long the arm stood for the lookahead, at the rise's top and at the way down's bottom
        stood = [v for v in (rise_stood, ahead[2].host_stood_s) if v is not None]
        ahead[2].host_stood_s = round(sum(stood), 4) if stood else None
    if not all(r.ok for r in res):
        if _started(res[-1]):
            log.info("[press_demo] STOP: the way down to the knob stopped (%s) — the arm holds" % res[-1].detail)
            rp.let_go(ctx)
            return 1
        return _MISSED if _way_back(ctx, cfg, runner, way, ctx.client.joints(), dp.DepthPress(free=True)) is None else 1

    def back_out(why, opened=True):
        log.info("[press_demo] the grip straight from the press: %s — back to the look pose to look at the knob" % why)
        if opened:
            res = runner.run(
                [_gripper_leg(ctx, PlanState(joints=None, chain=0), "grip:open", GRIPPER_CMD_OPEN, world)], execute=True
            )
            if not res[-1].ok:
                log.info("[press_demo] STOP: the fingers did not open (%s) — the arm holds" % res[-1].detail)
                rp.let_go(ctx)
                return 1
        code = _way_back(ctx, cfg, runner, way, ctx.client.joints(), dp.DepthPress(free=True))
        return _MISSED if code is None else code

    def miss(detail, air, empty=False):
        runner.note("reach_grip_result", ok=False, detail=detail, attempt=0, on_air=air, retry=True, from_press=True)
        log.info("[press_demo] the grip straight from the press did not hold the knob (%s)" % detail)
        _video(ctx, "the grip straight from the press missed (%s): back to the look pose" % detail)
        # shut on nothing, they back out shut; stopped on something, they open first, or the
        # hand would carry it back up the stroke
        return back_out(
            "the fingers closed on air" if air else "the fingers closed on something else (%s)" % detail,
            opened=not empty,
        )

    return _close_and_carry(ctx, cfg, runner, eyes, ahead, quat, 0, room, back_out, miss)


def _fingers_shut(ctx, cfg, runner):
    """Before a press after the first, at the look pose: the fingers shut on
    nothing, as the first press's are (reach_press reads them before it). A
    grip straight from the press that missed can leave them open, or at the
    pre-shape ("the lid has no way off the box", "the lift did not start"),
    and an open aperture strikes the lid and can still read "pressed";
    fingers closed on something press with it. Shut already, nothing to do;
    else a blocking press:close here, which must read shut. Returns None to
    press, or the exit code: home, nothing pressed."""
    ok, pos, _stalled = ctx.client.gripper_cmd(None)  # query only
    if ok and fingers.reads_empty_close(pos):
        return None
    log.info(
        "[press_demo] the fingers read %s before the press once more: shutting them first"
        % ("nothing" if pos is None else "%.3f" % pos)
    )
    world = _full_world(ctx)
    res = runner.run(
        [_gripper_leg(ctx, PlanState(joints=None, chain=0), "press:close", GRIPPER_CMD_CLOSED, world)], execute=True
    )
    closed = res[-1]
    if fingers.reads_empty_close(closed.gripper_pos):
        return None
    return _stop_home(
        ctx,
        cfg,
        runner,
        "%s %s (%s) before the press once more — the fingers are not shut; nothing was pressed"
        % (closed.leg_name, closed.outcome, closed.detail),
    )


def run_full(ctx, cfg, runner, eyes, way, legs, button, yaw, lid_before):
    """The full reach-out run from the aimed look pose, its way in `way`
    proven and flown as `legs` (reach_press._reach_press's, one execution):
    the press and the way back (_press_and_back), the look (knob_look), the
    press once more where the knob is not up and the lid did not rise (risen
    with its knob flush: knob_look's STOP), then the lid off
    (take_lid_off) at the aimed `button` (its top the press's), the grip
    going by the lid where the last hold held the pads (grip_lid_z). With
    GRIP_FROM_PRESS the first press's hold goes straight on to the grip
    (_grip_from_hold), the lid video recording from the press on; where
    that grip misses, back at the look pose, the look and all after it as
    above.
    `eyes` (node, watcher, scene_fix, the joints the scene camera looked
    at the arm in): the cameras. Returns the process exit code: 0 only with
    the lid set down and the arm home. From the way in's start on, anything raised lets go (as
    _fly_way_in): a Ctrl+C in the hold's read lands with no goal in flight,
    the springs pushing on into the box."""
    node, watcher = eyes[0], eyes[1]
    tilt = way.plan.tilt_deg
    top_z = float(button[2])
    ctx.contact_pad = CONTACT_SHIFT_PAD_M  # the stroke meets the box by design: every world pads it from here
    held_z = None
    video = None
    room = room_ahead(ctx, eyes)  # the lid's spot judged on the scene camera's map while the press flies

    def end(code):
        _finish_video(video, "the end: %s" % ("the lid set down, the arm home" if code == 0 else "exit %d" % code))
        return code

    try:
        if GRIP_FROM_PRESS:
            video = ctx.lid_video.start() if getattr(ctx, "lid_video", None) is not None else None
        for press in range(1, PRESSES + 1):
            if press > 1:
                code = _fingers_shut(ctx, cfg, runner)
                if code is not None:
                    return end(code)
            direct = None
            if GRIP_FROM_PRESS and press == 1:

                def direct(seen, way=way):
                    return _grip_from_hold(ctx, cfg, runner, eyes, way, seen, room)

            code, z = _press_and_back(ctx, cfg, runner, way, legs, top_z, press, direct=direct)
            if code is not None:
                return end(code)
            held_z = z if z is not None else held_z
            knob = knob_look(node, ctx, runner, watcher, button, yaw, lid_before, press)
            _note_frames(ctx, popped=bool(knob.popped) if knob.found else None)
            if knob.stop is not None:
                return end(_stop_home(ctx, cfg, runner, knob.stop))
            if knob.popped:
                break
            if press == PRESSES:
                return end(
                    _stop_home(
                        ctx,
                        cfg,
                        runner,
                        "pressed %d times and the knob is not up (%s) — the lid is "
                        "not taken off"
                        % (PRESSES, "%.1f mm above the lid" % knob.up_mm if knob.up_mm is not None else "not judged"),
                    )
                )
            log.info(
                "[press_demo] the knob is not up (%s): pressing once more — the same press, from the look pose"
                % ("%.1f mm above the lid" % knob.up_mm if knob.up_mm is not None else "not judged")
            )
            way, why, _kind = rp.solve_way_in(ctx, cfg, button, tilt, ctx.client.joints())
            if way is not None:
                legs = rp.way_in_legs(ctx, cfg, way)
                why = rp._not_one_execution(ctx, legs)
            if why is not None:
                return end(_stop_home(ctx, cfg, runner, "the press once more does not fly (%s)" % why))
            runner.note(
                "reach_way_in",
                press=press + 1,
                tilt_deg=way.plan.tilt_deg,
                walked=way.down.walked,
                short_deg=way.down.short_deg,
                k=round(way.stroke.k, 1),
                end_z=round(way.stroke.end_z, 4),
            )
        lid_z = grip_lid_z(top_z, held_z)
        rose = ""
        if knob.lid_rose:
            # the knob up on a released lid (with it flush, knob_look
            # stopped the run): the lid stands as much higher as the
            # wrist's two readings from the look pose differ (their bias
            # cancels)
            lid_z += float(knob.lid_dz_m)
            rose = ", %+.0f mm: the lid rose" % (1000 * knob.lid_dz_m)
        if video is None:
            video = ctx.lid_video.start() if getattr(ctx, "lid_video", None) is not None else None
        return end(
            take_lid_off(
                ctx,
                cfg,
                runner,
                eyes,
                button,
                tilt,
                lid_z,
                "the lid at %.3f (the press's top %.3f%s%s)"
                % (lid_z, top_z, "" if held_z is None else ", the pads held at %.3f" % held_z, rose),
                yaw=yaw,
                room=room,
            )
        )
    except BaseException as e:
        rp.let_go(ctx)
        if isinstance(e, KeyboardInterrupt):
            # a cancel (the engine's Ctrl+C): end at once, with no video encoded on the way out
            # (seconds of ffmpeg before the cancelled result); the pictures taken are dropped
            if video is not None:
                video.stop()
        else:
            _finish_video(video, "stopped: %s" % (type(e).__name__))  # after the let-go: the arm first
        raise


def _video(ctx, text):
    """The lid video's caption from now on (perception/camera_video), where
    this run records one."""
    v = getattr(ctx, "lid_video", None)
    if v is not None:
        v.mark(text)


def _finish_video(video, caption):
    """The lid video written, `caption` its last picture's, and where it is
    said; None, no video. Never what stops the run's ending."""
    if video is None:
        return
    try:
        path = video.finish(caption)
    except Exception as e:  # a video that cannot be written is said, never raised
        log.info("[press_demo] the cameras' video was not written (%s)" % e)
        return
    if path is not None:
        log.info("[press_demo] VIDEO: what the wrist and scene cameras saw from the grip on: %s" % path)


def _video_spot(ctx, eyes, drop):
    """The lid's rim at the spot `drop` drawn on the lid video's scene panel,
    where the scene camera itself sees that place: spot_map moved the scene's
    points by its error at the box, so the rim is moved back by as much."""
    v, loc = getattr(ctx, "lid_video", None), getattr(ctx, "scene_locator", None)
    if v is None or loc is None:
        return
    try:
        T, K = loc.base_to_color(), getattr(loc.grab, "k", None)
        if T is None or K is None:
            return
        fix = eyes[2]
        dx, dy = (
            (0.0, 0.0)
            if fix is None
            else (float(fix.pose.xyz[0]) - float(ctx.cpose.xyz[0]), float(fix.pose.xyz[1]) - float(ctx.cpose.xyz[1]))
        )
        r = 0.5 * max(float(ctx.model.lid_dims[0]), float(ctx.model.lid_dims[1]))
        a = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
        pts = np.stack(
            [
                float(drop.xyz[0]) + dx + r * np.cos(a),
                float(drop.xyz[1]) + dy + r * np.sin(a),
                np.full_like(a, float(drop.xyz[2])),
            ],
            axis=1,
        )
        T = np.asarray(T, float)
        cam = (pts - T[:3, 3]) @ T[:3, :3]  # into the colour frame
        if (cam[:, 2] <= 0.05).any():
            return
        uv = cam @ np.asarray(K, float).T
        v.overlay("scene", uv[:, :2] / uv[:, 2:3], "the lid goes here")
    except Exception:  # the drawing is the video's, never the run's
        return


def _back_out(ctx, cfg, runner, way, why, opened=True):
    """From the knob, the fingers on nothing they keep: back to the look
    pose (_back_to_look), then home. Returns 1."""
    code = _back_to_look(ctx, cfg, runner, way, why, opened)
    return code if code is not None else _stop_home(ctx, cfg, runner, why)


def _back_to_look(ctx, cfg, runner, way, why, opened=True):
    """From the knob, the fingers on nothing they keep: open (`opened`
    False: left as they are: open already, or shut on nothing, a close that
    reached its target: _close_reached), the grip's way down backwards to the look
    pose (_grip_backwards: one execution), home planned from there while it
    flies (ctx.home_ahead, for a stop there: _stop_home). Returns None at
    the look pose, or 1 where the run stops, the arm let go: the fingers did
    not open, or the way back out did not fly. `why` says what the STOP
    line says first."""
    world = _full_world(ctx)
    if opened:
        res = runner.run(
            [_gripper_leg(ctx, PlanState(joints=None, chain=0), "grip:open", GRIPPER_CMD_OPEN, world)], execute=True
        )
        if not all(r.ok for r in res):
            log.info("[press_demo] STOP: %s, and the fingers do not open (%s) — the arm holds" % (why, res[-1].detail))
            rp.let_go(ctx)
            return 1
    back = _grip_backwards(ctx, way, world)
    log.info("[press_demo] backing out the way the hand came down: %s" % ", ".join(lg.name for lg in back))
    # home from where the back-out ends, planned as it flies, as the way
    # back's (_press_and_back): 2026-10-28 16:24:21 stood 0.21 s there for it
    res = runner.run(back, execute=True, lookahead=lambda end: rp._home_legs(ctx, cfg, start=end))
    if not all(r.ok for r in res):
        log.info(
            "[press_demo] STOP: %s, and the way back out %s (%s) — the arm holds"
            % (why, "stopped" if _started(res[-1]) else "did not start", res[-1].detail or res[-1].outcome)
        )
        rp.let_go(ctx)
        return 1
    ctx.home_ahead = runner.lookahead_result
    return None


def _on_air(r, band):
    """Whether the close `r` (grip:close's LegResult) closed on air: the
    gripper did what it was told and the knuckle stands past the grip
    `band`'s closed end (0.8 rad fully shut), nothing between the pads —
    12:03:52 2026-10-29's 0.789. Not a gripper that failed, nor fingers
    stopped below the band, wider than the knob: on something else."""
    return (
        r.leg_name == "grip:close"
        and r.outcome == "arrived"
        and r.gripper_pos is not None
        and float(r.gripper_pos) > float(band[1])
    )


def _close_reached(r):
    """Whether the close `r` (grip:close's LegResult) reached its target: the
    fingers shut with nothing between the pads, within GripperWait's
    AT_TARGET_TOL of GRIPPER_CMD_CLOSED (every logged close on air read
    0.768-0.790). _on_air also counts a close that stalled between the grip
    band's top and that, on something 4-26 mm thick (an off-centre chord of
    the knob, the lid's lip): those fingers open before the hand backs out,
    or they would carry it up."""
    return (
        r.leg_name == "grip:close"
        and r.outcome == "arrived"
        and r.gripper_pos is not None
        and float(r.gripper_pos) >= GRIPPER_CMD_CLOSED - GripperWait.AT_TARGET_TOL
    )


def grip_knob_circle(color):
    """The knob's circle in the wrist's close-up at the grip (BGR): (u, v,
    r) px — of HoughCircles' within GRIP_SHOT_R_PX, the one centred lowest
    in the frame's bottom middle, where the knob sits — or None."""
    import cv2

    g = cv2.medianBlur(cv2.cvtColor(np.ascontiguousarray(color), cv2.COLOR_BGR2GRAY), 5)
    h, w = g.shape[:2]
    cs = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=40,
        param1=80,
        param2=18,
        minRadius=GRIP_SHOT_R_PX[0],
        maxRadius=GRIP_SHOT_R_PX[1],
    )
    if cs is None:
        return None
    near = [c for c in cs[0] if 0.34 * w < c[0] < 0.72 * w and c[1] > 0.75 * h]
    if not near:
        return None
    u, v, r = max(near, key=lambda c: c[1])
    return float(u), float(v), float(r)


class _GripShot:
    """The grip's close-up (GRIP_SHOT_R_PX's record), taken once the arm
    stands still before the close (Leg.when_still): the wrist's frame kept
    (captures/wrist-grip-close-*, press_demo._save_wrist_frame on a thread)
    and the knob's circle in it measured on another while the fingers
    close; note() records both with whether the grip held."""

    def __init__(self, watcher, model, attempt):
        from box_opening.tasks.press_demo import CAPTURES_DIR, _prune_captures, _save_wrist_frame, capture_stamp

        g = getattr(watcher, "grab", None)
        self.result, self.thread, self.frame = None, None, None
        color = None if g is None else getattr(g, "color", None)
        if color is None:
            return
        k = getattr(g, "k", None)
        color, k = np.array(color, copy=True), None if k is None else np.asarray(k, float)
        try:
            out = _save_wrist_frame(
                g,
                CAPTURES_DIR / ("wrist-grip-close-%s-a%d" % (capture_stamp(), attempt)),
                "the grip's close-up, attempt %d" % attempt,
                background=True,
                after=lambda: _prune_captures("grip-close"),
            )
            self.frame = None if out is None else str(out)
        except Exception:  # a frame not kept costs the record its folder, never the grip
            self.frame = None
        self.thread = threading.Thread(
            target=self._measure, args=(color, k, float(model.button_diameter_m)), name="grip-shot", daemon=True
        )
        self.thread.start()

    def _measure(self, color, k, knob_d):
        circle = grip_knob_circle(color)
        cam = None
        if circle is not None and k is not None:
            u, v, r = circle
            z = k[0, 0] * 0.5 * knob_d / r  # its range from its size
            cam = [
                round(1000 * z * (u - k[0, 2]) / k[0, 0], 1),
                round(1000 * z * (v - k[1, 2]) / k[1, 1], 1),
                round(1000 * z, 1),
            ]
        self.result = dict(knob_px=None if circle is None else [round(c, 1) for c in circle], knob_cam_mm=cam)

    def note(self, runner, attempt, held, detail):
        if self.thread is not None:
            self.thread.join(1.0)
        runner.note(
            "grip_close_look",
            attempt=attempt,
            held=held,
            detail=detail,
            frame=self.frame,
            **(self.result or dict(knob_px=None, knob_cam_mm=None, why="no wrist frame")),
        )


def _regrip(ctx, cfg, runner, eyes, way, button, yaw, detail, opened=True):
    """After a first close on air (`detail`, what it read): the fingers
    open and the hand backs out the grip's way down to the look pose
    (_back_to_look, home planned while it flies), and from there the wrist
    aims at the knob again (reach_press._aim, 'reach-regrip', on
    REGRIP_AIM_HITS agreeing hits), as knob_look does after a press. Returns (the button the grip goes to
    once more, None): the re-aim's x and y where it finds the knob no
    farther than REGRIP_MAX_M from the aimed `button`, else — no knob
    found — the aimed `button`'s, where the first grip went; its height the
    aimed one's (take_lid_off lowers the grip's lid: GRIP_RETRY_LOWER_M).
    Or (None, the exit code) where the run stops: the back-out did not fly
    (the arm let go), or the knob is farther than that — the box moved —:
    home from the look pose, flying the home planned on the way back.
    Noted 'reach_regrip': the re-aim's point, its shift and what follows."""
    code = _back_to_look(ctx, cfg, runner, way, "the grip closed on air (%s)" % detail, opened)
    if code is not None:
        runner.note("reach_regrip", found=None, outcome="the back-out did not fly")
        return None, code
    node, watcher = eyes[0], eyes[1]
    got, status, up, _measured = rp._aim(
        node, watcher, yaw, button[2], button[:2], runner, "reach-regrip", min_hits=REGRIP_AIM_HITS
    )
    if got is None:
        runner.note("reach_regrip", found=False, why=status, outcome="grip again where the first went")
        log.info(
            "[press_demo] the wrist does not find the button again from the look pose (%s): gripping once more "
            "where the first grip went" % status
        )
        return tuple(float(v) for v in button), None
    shift = rp.judge_box_moved(button, got[0])
    moved = shift > REGRIP_MAX_M
    runner.note(
        "reach_regrip",
        found=True,
        xyz=[round(v, 4) for v in got[0]],
        shift_mm=round(1000 * shift, 1),
        knob_up_mm=None if up is None else round(float(up), 1),
        outcome="home" if moved else "grip again",
    )
    if moved:
        return None, _stop_home(
            ctx,
            cfg,
            runner,
            "the grip closed on air (%s) and the wrist now sees the button %.0f "
            "mm from where the first grip went — the box moved; the lid is not "
            "taken off" % (detail, 1000 * shift),
        )
    again = (float(got[0][0]), float(got[0][1]), float(button[2]))
    log.info(
        "[press_demo] RE-AIMED from the look pose: the knob at [%.3f, %.3f], %.0f mm from where the first grip "
        "went%s — gripping once more there"
        % (again[0], again[1], 1000 * shift, "" if up is None else ", %.1f mm above its lid" % up)
    )
    return again, None


def take_lid_off(ctx, cfg, runner, eyes, button, tilt, lid_z, how, yaw=None, attempt=1, room=None):
    """From the look pose, the knob up: the lid off, set down, home. The
    grip's way down (solve_grip_way, from where the arm stands), the
    fingers opening as it flies, ONE execution; while it flies, the lift,
    the lid's spot from the scene camera's map (`room`, judged while the
    press flew: room_ahead), the carry there and home after the set-down
    (grip_ahead). The close, judged by the grip band, then the lift, the
    clear, the carry and the set-down: ONE execution; the release; the
    retreat and home, ONE execution (_release_and_home). Where no carry
    and home planned ahead, as before: the lift and the clear alone, the
    carry to the scene camera's best spot planned while they fly
    (carry_ahead), the spot look there (choose_and_carry). `eyes` (node,
    watcher, scene_fix, the joints the scene camera looked at the arm in)
    the cameras; `how` says what the grip's lid height `lid_z` is. A first
    close on air (_on_air), `yaw` given (the box's, for the aim): back out
    to the look pose, aim at the knob again (_regrip) and take_lid_off once
    more, `attempt` 2, at the re-aimed button — the same tilt and lid
    height. Returns the exit code: 0 with the lid set down and the arm
    home."""
    from box_opening.tasks.press_demo import viz_hook

    world = _full_world(ctx)
    ctx.contact_pad = CONTACT_SHIFT_PAD_M  # the grip meets the box by design
    way, why = solve_grip_way(ctx, cfg, button, tilt, ctx.client.joints(), lid_z)
    if way is None:
        runner.note("reach_grip_refused", attempt=attempt, why=why)
        return _stop_home(
            ctx,
            cfg,
            runner,
            "the hand does not come down to the %sknob (%s) — the lid is not taken "
            "off" % ("re-aimed " if attempt > 1 else "", why),
        )
    legs = rp.way_in_legs(ctx, cfg, way)
    names = [lg.name for lg in legs]
    runner.note(
        "reach_grip",
        lid_z=round(lid_z, 4),
        pads_z=round(way.stroke.pad_z, 4),
        tilt_deg=tilt,
        walked=way.down.walked,
        short_deg=way.down.short_deg,
        how=how,
        button=[round(float(v), 4) for v in button],
        attempt=attempt,
    )
    log.info(
        "[press_demo] GRIP%s: the fingers open to the knob's width on the way down; ONE MOTION from the look pose "
        "to the knob: %s — "
        "the closed pads %.0f mm under %s%s"
        % (
            " ONCE MORE, lower" if attempt > 1 else "",
            ", ".join(names),
            1000 * GRIP_BELOW_LID_M,
            how,
            ", the hand %.0f deg short of straight down" % way.down.short_deg if way.down.short_deg else "",
        )
    )
    viz_hook("grip", way.stroke, lid_z)  # the grip's way down and the lift, drawn off this thread (perception/viz)
    # opened to the knob's width and a margin, not fully: fully open the
    # fingertips came down on the lid's rim (14:31:59, GRIP_SIDE_CLEAR_M)
    opened = _gripper_leg(
        ctx, PlanState(joints=None, chain=0), "grip:open", grip_open_cmd(ctx.model), world, defer_join=True
    )
    # the hand leaves the look pose for the knob: the way back's home is not
    # flown from where it comes back (a back-out plans its own, _back_out)
    ctx.home_ahead = None
    _video(ctx, "GRIP%s: the fingers open on the way down to the knob" % (" once more" if attempt > 1 else ""))
    res = runner.run(
        [opened, *legs], execute=True, lookahead=lambda end: grip_ahead(ctx, cfg, eyes, end, way.stroke.quat, room)
    )
    ahead = _lookahead(runner)
    last = res[-1]
    if not all(r.ok for r in res):
        if not _started(last):
            return _stop_home(
                ctx,
                cfg,
                runner,
                "the way down to the knob did not start (%s) — the lid is not taken "
                "off" % (last.detail or last.outcome),
            )
        log.info(
            "[press_demo] STOP: the way down to the knob stopped during %s (%s) — the arm holds"
            % (rp._in_flight(legs, last, runner.last_retime), last.detail or last.outcome)
        )
        rp.let_go(ctx)
        return 1

    def back_out(why, opened=True):
        return _back_out(ctx, cfg, runner, way, why, opened=opened)

    def miss(detail, air, empty=False):
        retry = air and attempt < GRIP_TRIES and yaw is not None
        runner.note("reach_grip_result", ok=False, detail=detail, attempt=attempt, on_air=air, retry=retry)
        if retry:
            log.info(
                "[press_demo] GRIP MISSED — %s: the fingers closed on air; %sbacking out to the look pose, aiming "
                "at the knob once more and coming down %.0f mm lower"
                % (detail, "" if empty else "opening them, ", 1000 * GRIP_RETRY_LOWER_M)
            )
            _video(
                ctx,
                "GRIP MISSED (%s): back out to the look pose, aim again, %.0f mm lower"
                % (detail, 1000 * GRIP_RETRY_LOWER_M),
            )
            regrip, code = _regrip(ctx, cfg, runner, eyes, way, button, yaw, detail, opened=not empty)
            if regrip is None:
                return code
            lower = lid_z - GRIP_RETRY_LOWER_M
            return take_lid_off(
                ctx,
                cfg,
                runner,
                eyes,
                regrip,
                tilt,
                lower,
                "%s, %.0f mm lower than the grip that " "closed on air: %.3f" % (how, 1000 * GRIP_RETRY_LOWER_M, lower),
                yaw=yaw,
                attempt=attempt + 1,
                room=room,
            )
        log.info(
            "[press_demo] GRIP FAILED%s — %s: the fingers closed on air or on something else, so the knob is not "
            "up or the grip missed it"
            % (" AGAIN, %.0f mm lower" % (1000 * GRIP_RETRY_LOWER_M) if attempt > 1 else "", detail)
        )
        return _back_out(
            ctx,
            cfg,
            runner,
            way,
            "the grip failed%s (%s) — the lid is not taken off"
            % (" twice, the second time %.0f mm lower" % (1000 * GRIP_RETRY_LOWER_M) if attempt > 1 else "", detail),
            opened=not empty,
        )

    return _close_and_carry(ctx, cfg, runner, eyes, ahead, way.stroke.quat, attempt, room, back_out, miss)


def _close_and_carry(ctx, cfg, runner, eyes, ahead, quat, attempt, room, back_out, miss):
    """The hand down round the knob, the fingers open, the hand at `quat`:
    the close, judged by the grip band, then the lift, the clear, the carry
    and the set-down — ONE execution, all planned while the hand came down
    (`ahead`, grip_ahead's) — and the release and home (_release_and_home);
    where no carry and home planned ahead, the lift and the clear alone and
    the spot look there (choose_and_carry). `back_out(why, opened=True)`: the
    hand's way off the knob, nothing lifted — no lift to fly, or one that
    did not start; `miss(detail, on_air)`: a close that did not hold. Each
    returns what this does: the exit code, or its caller's sentinel."""
    world = _full_world(ctx)
    lift, why, placed = ahead if ahead is not None else (None, "the lookahead plan failed", None)
    if lift is None:
        lift, why = plan_lift(ctx, cfg, ctx.client.joints(), quat)
    if lift is None:
        return back_out("the lid has no way off the box (%s) — not gripping it" % why, opened=False)
    # closed once the arm stands still: the hand where it came down to, not
    # where it is settling past (Leg.send_when_still)
    close = _gripper_leg(
        ctx,
        PlanState(joints=None, chain=0),
        "grip:close",
        GRIPPER_CMD_CLOSED,
        world,
        verify=band_verify(cfg.grip_band),
        send_when_still=True,
    )
    shots = []  # the close-up, taken once the arm stands still (GRIP_SHOT_R_PX)
    close.when_still = lambda: shots.append(_GripShot(eyes[1], ctx.model, attempt))
    # the spot, its carry and home after it planned while the hand came
    # down: the close, then the lid lifted, carried and set down without a
    # stop. A close that fails stops before anything lifts
    one = placed is not None and placed.carry is not None
    carried = None
    ctx.lid_lifted = lift  # the lid's way back onto the box, for a STOP before the carry (put_lid_back)
    if one:
        carry, drop = placed.carry, placed.drop
        flying = [close, *lift.legs, *carry.legs]
        runner.note(
            "lid_spot",
            xyz=[round(v, 4) for v in drop.xyz],
            tier=placed.tier,
            wrist_points=None,
            mapped=placed.tmap is not None,
            refused="; ".join(placed.refused) or None,
            chosen="ahead, with the scene camera",
            room_s=None if placed.room_s is None else round(placed.room_s, 3),
            # ahead_s is the lookahead's spot work and its wait for the room (room_wait_s);
            # host_stood_s, how long the hosting leg stood at its end for the lookahead
            ahead_s=round(placed.spent_s, 3),
            room_wait_s=None if placed.room_wait_s is None else round(placed.room_wait_s, 3),
            spot_checks=placed.checks,
            spot_ik_s=round(placed.ik_s, 3),
            plan_s=round(placed.plan_s, 3),
            host_stood_s=placed.host_stood_s,
            pooled=pooled_note(ctx, runner, placed.room, drop),  # the wrist's look-pose frame pooled in: shadow
        )
        _say_spot(ctx, eyes, (drop, placed.tier, placed.says), placed.refused, placed.tmap, placed.table_z)
        log.info(
            "[press_demo] CLOSE, then ONE MOTION: %s — %.0f mm straight up, %.0f mm back toward the arm, to "
            "[%.2f, %.2f] and down onto the table, %s"
            % (
                ", ".join(lg.name for lg in flying[1:]),
                1000 * lift_off_m(ctx.model),
                1000 * lift.back_m,
                drop.xyz[0],
                drop.xyz[1],
                _touch_says(carry),
            )
        )
        _video(
            ctx,
            "CLOSE on the knob, then ONE MOTION: LIFT, CLEAR, CARRY to [%.2f, %.2f] (%s) and SET DOWN, "
            "let go where it meets the table" % (drop.xyz[0], drop.xyz[1], placed.tier),
        )
        watch_setdown(ctx, flying, carry)  # where the lid meets the table, watched
        res = runner.run(flying, execute=True)
    else:
        if placed is not None:
            log.info(
                "[press_demo] no spot whose carry and home both plan was found while the hand came down (%s): "
                "the lift alone, then the spot look" % ("; ".join(placed.refused) or "no spot the hand reaches")
            )
        flying = [close, *lift.legs]
        log.info(
            "[press_demo] CLOSE, then ONE MOTION: %s — %.0f mm straight up, then %.0f mm back toward the arm"
            % (", ".join(lg.name for lg in lift.legs), 1000 * lift_off_m(ctx.model), 1000 * lift.back_m)
        )
        # while the lift and the clear fly, the carry to the scene camera's
        # best spot from where the clear ends (carry_ahead): flown after the
        # lift where the spot chosen there is the same
        _video(ctx, "CLOSE on the knob, then LIFT %.0f mm and CLEAR the box" % (1000 * lift_off_m(ctx.model)))
        res = runner.run(flying, execute=True, lookahead=lambda end: carry_ahead(ctx, cfg, eyes, end, room=room))
        carried = _lookahead(runner)
    grip = [r for r in res if r.leg_name in ("grip:open", "grip:close")]
    for shot in shots:
        shot.note(
            runner,
            attempt,
            bool(grip) and grip[-1].ok and grip[-1].leg_name == "grip:close",
            grip[-1].detail if grip else None,
        )
    if not grip or not grip[-1].ok or grip[-1].leg_name != "grip:close":
        detail = grip[-1].detail if grip else res[-1].detail
        return miss(detail, bool(grip) and _on_air(grip[-1], cfg.grip_band), bool(grip) and _close_reached(grip[-1]))
    runner.note("reach_grip_result", ok=True, detail=grip[-1].detail, attempt=attempt)
    log.info("[press_demo] LID GRIPPED — %s" % grip[-1].detail)
    moves = [r for r in res if r.leg_name not in ("grip:open", "grip:close")]
    last = moves[-1] if moves else res[-1]
    if not moves or not all(r.ok for r in moves):
        if len(moves) <= 1 and not _started(last):
            # nothing moved: the knob in the fingers, the lid still on the canister
            return back_out(
                "the lift did not start (%s) — the lid is let go where it lies, on the box"
                % (last.detail or last.outcome)
            )
        log.info(
            "[press_demo] STOP: the %s stopped during %s (%s) — the arm holds the lid"
            % (
                "lid's way from the box to the table" if one else "lift",
                rp._in_flight(flying, last, runner.last_retime),
                last.detail or last.outcome,
            )
        )
        rp.let_go(ctx)
        return 1
    if not one:
        return choose_and_carry(ctx, cfg, runner, eyes, ahead=carried)
    if last.flown_traj is not None:
        rp._note_flight(runner, "reach_carry_flight", [lg.name for lg in merge_groups(flying)[-1]], last)
    return _release_and_home(ctx, cfg, runner, carry, drop, placed.tier, placed.home, None)


@dataclass
class CarryAhead:
    """What carry_ahead made while the lift and the clear flew, from the
    joints `q` the clear ends at: the scene camera's map's best spot the
    carry plans to (`drop`, its tier and what it says) and that carry
    (Carry; None where none planned, `refused` saying why), the reach
    verdicts it judged (`known`, _Reach.found), how long it all took
    (`spent_s`) and the carry's own plan (`plan_s`); with home asked for,
    the legs after the release (`home`, plan_home_after's). The map it chose
    on (`tmap`, on the table at `table_z`), and how long the room on it took
    while the press flew (`room_s`, None: judged here), how long this waited
    for that (`room_wait_s`: spent_s mixed the wait with the work), the
    spots whose reach it judged (`checks`) and their IK's time (`ik_s`); and
    how long the leg that hosted it stood at its end for it (`host_stood_s`,
    Runner.lookahead_stood_s, set by the caller)."""

    q: list
    drop: object = None
    tier: str = None
    says: str = None
    carry: object = None
    refused: list = field(default_factory=list)
    known: dict = field(default_factory=dict)
    spent_s: float = 0.0
    plan_s: float = 0.0
    home: list = None
    tmap: object = None
    table_z: float = None
    room_s: float = None
    room_wait_s: float = None
    checks: int = 0
    ik_s: float = 0.0
    host_stood_s: float = None
    room: object = None  # the Room it chose on (room_for's), None: judged here


def _lookahead(runner):
    """runner.lookahead_result, the CarryAhead in it (grip_ahead's third,
    or carry_ahead's own) told how long the leg that hosted it stood at its
    end for it (host_stood_s, Runner.lookahead_stood_s): the lookahead's
    overrun, which a lid_spot row keeps."""
    got = runner.lookahead_result
    placed = got[2] if isinstance(got, tuple) and len(got) == 3 else got
    if isinstance(placed, CarryAhead):
        placed.host_stood_s = getattr(runner, "lookahead_stood_s", None)
    return got


def carry_ahead(ctx, cfg, eyes, q_end, room=None, home=False):
    """The lid's spot from the scene camera's map alone (spot_map, no wrist
    frame; `room`'s, the room on it judged while the press flew, where it
    is the box's still: room_for), its spot_candidates the hand reaches from
    `q_end`, where the clear ends, and the carry from there to the first
    that plans (plan_carry, at most CARRY_TRIES) — with `home`, and home
    after its set-down too (plan_home_after): a spot home does not plan
    from is passed over. The grip's lookahead (grip_ahead, with home): the
    close to the set-down then fly as one. Else the lift's: the spot is
    chosen after the lift, the wrist's frame in its map (choose_and_carry),
    and this carry flies only where that spot is its own. Returns a
    CarryAhead. CarrySearch's two parts, one after the other."""
    return CarrySearch(ctx, cfg, eyes, q_end, room=room, home=home).start().finish()


class CarrySearch:
    """carry_ahead in two parts, each of which a motion can host (the grip
    straight from the press: grip_ahead_rise, grip_ahead_down), both on the
    main thread — the planner is called from nowhere else. start(): the
    room (room_for), the map and the first spot the hand reaches, no plan;
    finish(): the carry (and home after it) planned to that spot, the next
    ones tried where it is refused, and the CarryAhead — what carry_ahead
    returns, its spent_s both parts' time."""

    def __init__(self, ctx, cfg, eyes, q_end, room=None, home=False):
        self.ctx, self.cfg, self.eyes, self.room, self.home = ctx, cfg, eyes, room, home
        self.q = [float(v) for v in q_end]
        self.ahead = None
        self._spots = self._next = None
        self._reach = None

    def start(self):
        from box_opening.primitives.look import look_joints
        from box_opening.tasks import press_demo as pd

        t0 = time.monotonic()
        ctx = self.ctx
        _node, watcher, scene_fix, scene_q = self.eyes
        t_room = time.monotonic()
        judged = room_for(ctx, self.room)
        room_wait_s = None if self.room is None else time.monotonic() - t_room
        table_z = _table_z(watcher, scene_fix, ctx.cpose)
        tmap = judged.tmap if judged is not None else spot_map(ctx, None, self.q, scene_fix, scene_q, table_z)
        rest = list(pd.rest_joints(self.cfg))
        self._reach = _Reach()
        self.ahead = CarryAhead(
            self.q,
            tmap=tmap,
            table_z=table_z,
            room_s=None if judged is None else judged.spent_s,
            room_wait_s=room_wait_s,
            room=judged,
        )
        self._spots = spot_candidates(
            ctx,
            tmap,
            table_z,
            self.q,
            seeds=(look_joints(rest), rest),
            reach=self._reach,
            cache=None if judged is None else judged.cache,
        )
        self._next = next(self._spots, None)
        self.ahead.spent_s += time.monotonic() - t0
        return self

    def finish(self):
        t0 = time.monotonic()
        ctx, ahead = self.ctx, self.ahead
        while self._next is not None:
            if len(ahead.refused) >= CARRY_TRIES:
                break
            drop, tier, says = self._next
            _abort_check(ctx)
            t = time.monotonic()
            carry, why = plan_carry(ctx, self.cfg, self.q, drop)
            after = None
            if carry is not None and self.home:
                _abort_check(ctx)  # a plan cut short by the Ctrl+C is no refusal: never the next spot
                after, why = plan_home_after(ctx, self.cfg, carry)
                carry = carry if after is not None else None
            _abort_check(ctx)
            if carry is not None:
                ahead.drop, ahead.tier, ahead.says, ahead.carry, ahead.home = drop, tier, says, carry, after
                ahead.plan_s = time.monotonic() - t
                break
            ahead.refused.append("[%.2f, %.2f]: %s" % (drop.xyz[0], drop.xyz[1], why))
            self._next = next(self._spots, None)
        ahead.known = self._reach.found
        ahead.checks, ahead.ik_s = len(self._reach.found), self._reach.ik_s
        ahead.spent_s += time.monotonic() - t0
        return ahead


@dataclass
class Room:
    """The scene camera's map round the box (spot_map, no wrist frame) and
    the room judged on it at every spot the lid may go (`cache`,
    clear_table.clear_spots'), for the box at `at` (ctx.cpose's xyz; None:
    judging it failed): room_ahead's, made while the press flies."""

    at: tuple
    tmap: object = None
    cache: dict = field(default_factory=dict)
    spent_s: float = 0.0
    thread: object = None
    # the map with the wrist's look-pose frame pooled in (aim_cloud): what the
    # room is judged on with POOL_AIM_FRAME, else built beside it (shadow mode);
    # why there is none; the scene camera's alone, kept for the row's comparison
    pooled: object = None
    pooled_why: str = None
    scene_map: object = None
    aim_points: int = 0


def room_ahead(ctx, eyes):
    """Owner 2026-10-29: the lid's spot from the scene camera "in parallel
    will pressing/opening the box so we can seemlesly grab the lid and move
    it to the safe location without stopping or interuptions". The room at
    every spot the lid may go, judged on the scene camera's map
    (press_demo.clear_lid_drops: ~1.3 s of the 1.7-2.0 s the scene
    camera's spot took on 10-29's runs), on a thread of its own while the
    press flies (~8.5 s). Only the ranking needs the way the hand faces
    where the clear ends, known once the grip is planned: grip_ahead ranks
    from the cache. Returns the Room, its thread started."""
    from box_opening.tasks import press_demo as pd

    _node, watcher, scene_fix, scene_q = eyes
    model, cpose = ctx.model, ctx.cpose
    table_z = _table_z(watcher, scene_fix, cpose)
    room = Room(at=tuple(float(v) for v in cpose.xyz))

    def judge():
        t0 = time.monotonic()
        try:
            room.tmap = room.scene_map = spot_map(ctx, None, None, scene_fix, scene_q, table_z)
            pool_aim_frame(ctx, room, watcher, scene_fix, scene_q, table_z)
            if POOL_AIM_FRAME and room.pooled is not None:
                room.tmap = room.pooled
            if room.tmap is not None:
                pd.clear_lid_drops(model, cpose, room.tmap, table_z, cache=room.cache)
        except Exception as e:  # not judged here, it is judged in the grip's lookahead
            room.at = None
            log.info("[press_demo] the room for the lid was not judged while the box was pressed (%s)" % e)
        room.spent_s = time.monotonic() - t0

    room.thread = threading.Thread(target=judge, name="lid-room", daemon=True)
    room.thread.start()
    return room


def pool_aim_frame(ctx, room, watcher, scene_fix, scene_q, table_z):
    """room.pooled: the scene camera's cloud and the wrist's at the look
    pose (aim_cloud) in one map, or None with room.pooled_why. Never what
    stops the room being judged."""
    try:
        aim, why = aim_cloud(ctx, watcher, table_z)
        if aim is None:
            room.pooled_why = why
            return
        room.aim_points = len(aim[0])
        scene = _scene_cloud(ctx, scene_fix, scene_q, table_z)
        clouds = ([scene] if scene is not None else []) + [aim]
        room.pooled = clear_table.table_map(clouds, ctx.cpose.xyz[:2])
        if room.pooled is None:
            room.pooled_why = "no table plane in the pooled clouds"
    except Exception as e:  # noqa: BLE001 - the shadow map must never cost the run
        room.pooled, room.pooled_why = None, "pooling failed (%s)" % e


def pooled_note(ctx, runner, room, drop):
    """The lid_spot row's 'pooled': the map with the wrist's look-pose
    frame in it against the scene camera's alone — the cells seen and
    standing in each, the spot chosen (`drop`) — and the two maps kept
    beside the run record (<record>.side/room-maps.npz), where an attended
    run checks that every object it placed stands in the pooled one and
    the spot's room on each; whether the pooled map chose (POOL_AIM_FRAME).
    Nothing is judged here: the room is the lid-room thread's work. None
    without a room judged ahead."""
    from box_opening.tasks import press_demo as pd

    if room is None:
        return None
    out = {"chose": bool(POOL_AIM_FRAME and room.pooled is not None), "why": room.pooled_why}
    maps = {"scene": room.scene_map, "pooled": room.pooled}
    for key, tmap in maps.items():
        if tmap is not None:
            out["%s_seen" % key] = int(tmap.seen.sum())
            out["%s_standing" % key] = int(np.isfinite(tmap.tall).sum())
    out["aim_points"] = room.aim_points
    out["spot"] = None if drop is None else [round(float(v), 4) for v in drop.xyz[:2]] + [round(float(drop.yaw), 4)]
    arrays = {}
    for key, tmap in maps.items():
        if tmap is not None:
            arrays.update(
                {
                    "%s_tall" % key: tmap.tall.astype(np.float32),
                    "%s_seen" % key: tmap.seen,
                    "%s_off" % key: tmap.off,
                    "%s_origin" % key: np.array([tmap.x0, tmap.y0, tmap.cell]),
                }
            )
    path = runner.side_file("room-maps.npz") if arrays and hasattr(runner, "side_file") else None
    if path is not None:
        box_xy = np.asarray(ctx.cpose.xyz[:2], float)

        def write():
            try:
                np.savez_compressed(path, box_xy=box_xy, **arrays)
            except Exception as e:  # noqa: BLE001 - a side file must never cost the run
                log.info("[press_demo] (the room maps are not kept: %s)" % e)

        # compressed off the motion's way (the close is the next thing sent), with
        # the captures' writes (press_demo.wait_capture_writes waits for it)
        writer = threading.Thread(target=write, name="room-maps", daemon=False)
        pd._PENDING_WRITES[:] = [t for t in pd._PENDING_WRITES if t.is_alive()] + [writer]
        writer.start()
        out["maps"] = str(path.relative_to(runner.log_dir))
    return out


def room_for(ctx, room):
    """`room` (room_ahead's, its thread waited for) where it was judged for
    the box where ctx.cpose has it now; else None, and the room is judged
    again."""
    if room is None:
        return None
    if room.thread is not None:
        while room.thread.is_alive():  # judging can outlast the press (4.9-8.2 s on the robot)
            room.thread.join(0.05)
            _abort_check(ctx)
    return room if room.at == tuple(float(v) for v in ctx.cpose.xyz) else None


def grip_ahead(ctx, cfg, eyes, q_grip, quat, room=None):
    """The grip's lookahead, while the hand comes down to the knob: the
    lift from the knob held at `q_grip` (plan_lift) and, from where its
    clear ends, the lid's spot, the carry there and home after its
    set-down (carry_ahead, `room` judged while the press flew): the close,
    the lift, the clear, the carry and the set-down then fly as one
    (take_lid_off). Returns (LiftOff, None, CarryAhead) or (None, why,
    None)."""
    lift, why = plan_lift(ctx, cfg, q_grip, quat)
    if lift is None:
        return None, why, None
    _abort_check(ctx)
    return lift, None, carry_ahead(ctx, cfg, eyes, lift.legs[-1].goal_joints, room=room, home=True)


def grip_ahead_rise(ctx, cfg, eyes, q_grip, quat, room=None):
    """grip_ahead's first part, while the hand rises off the press (the grip
    straight from it): the lift from the knob held at `q_grip` and, from
    where its clear ends, the room and the first spot the hand reaches
    (CarrySearch.start). The arm stood at the rise's top 0.01-1.50 s for
    the whole of grip_ahead (mean 0.48 s on 7 runs, the rise 0.77-1.17 s
    against 0.71-2.29 s of it); the carry and home plans go to the way
    down (grip_ahead_down), which hosted nothing. Returns (LiftOff, None,
    CarrySearch) or (None, why, None)."""
    lift, why = plan_lift(ctx, cfg, q_grip, quat)
    if lift is None:
        return None, why, None
    _abort_check(ctx)
    return lift, None, CarrySearch(ctx, cfg, eyes, lift.legs[-1].goal_joints, room=room, home=True).start()


def grip_ahead_down(first):
    """grip_ahead's second part, while the hand comes down to the knob: the
    carry to the spot grip_ahead_rise found and home after its set-down
    (CarrySearch.finish) — `first` its result. Returns grip_ahead's:
    (LiftOff, None, CarryAhead) or (None, why, None)."""
    lift, why, search = first
    if lift is None or search is None:
        return lift, why, None
    return lift, None, search.finish()


def _ahead_note(ahead, off, flew, chosen, reach):
    """lid_spot's 'ahead': whether the carry carry_ahead planned flew, why
    not, how far the arm stood from where it was planned from (`off`, rad)
    and the time it saved at rest — its plan where it flew, and the reach
    verdicts taken from it (`reach`, _Reach). None without one."""
    if ahead is None:
        return None
    why = None
    if not flew:
        if ahead.carry is None:
            why = "none planned (%s)" % ("; ".join(ahead.refused) or "no spot the hand reaches")
        elif off > DRIFT_REPLAN_RAD:
            why = "the arm stands %.3f rad from where it was planned from (gate %.2f)" % (off, DRIFT_REPLAN_RAD)
        elif chosen is None:
            why = "no carry plans here"
        else:
            d = chosen[0]
            why = "the spot chosen with the lid lifted is %.0f mm from it%s" % (
                1000 * math.hypot(d.xyz[0] - ahead.drop.xyz[0], d.xyz[1] - ahead.drop.xyz[1]),
                ""
                if abs(math.remainder(float(d.yaw) - float(ahead.drop.yaw), 2 * math.pi)) < 1e-3
                else ", the hand turned otherwise",
            )
    return dict(
        flown=flew,
        why=why,
        off_rad=round(off, 4),
        xyz=None if ahead.drop is None else [round(v, 4) for v in ahead.drop.xyz],
        spent_s=round(ahead.spent_s, 3),
        # of spent_s, the wait for the room judged while the press flew; the spots
        # it judged the reach of, and how long the lift stood at its end for it
        room_wait_s=None if ahead.room_wait_s is None else round(ahead.room_wait_s, 3),
        spot_checks=ahead.checks,
        host_stood_s=ahead.host_stood_s,
        reused=reach.reused,
        saved_s=round(reach.saved_s + (ahead.plan_s if flew else 0.0), 3),
    )


def choose_and_carry(ctx, cfg, runner, eyes, ahead=None):
    """The lid lifted and clear of the box, in the fingers: its one stop.
    The wrist's frame (spot_look) and the scene camera's table make the map
    (spot_map) the spot is chosen on; the first of spot_candidates the carry plans to
    (plan_carry, at most CARRY_TRIES) is where the lid goes, one line saying
    which tier and why. `ahead` (carry_ahead's, made while the lift flew):
    the arm within DRIFT_REPLAN_RAD of where it was made, its reach
    verdicts are taken (_Reach), and its carry flies where the spot chosen
    is its own (_same_spot) — no plan at rest; else the carry is planned
    here, as without one. Then the carry and the set-down, ONE execution,
    home planned while it flies (plan_home_after); the knuckle read (the
    lid slipped?); the release; the retreat and home, ONE execution. The
    lid_spot note says how long each part took here, and whether the carry
    planned ahead flew (_ahead_note). Returns the exit code: 0 with the lid
    set down and the arm home."""
    from box_opening.geometry import ang_diff
    from box_opening.primitives.look import look_joints
    from box_opening.tasks import press_demo as pd

    node, watcher, scene_fix, scene_q = eyes
    table_z = _table_z(watcher, scene_fix, ctx.cpose)
    t0 = time.monotonic()
    _video(ctx, "SPOT LOOK: the arm stands still for the wrist's frame, the lid lifted")
    wrist = spot_look(node, watcher, runner)
    t_look = time.monotonic()
    q = [float(v) for v in ctx.client.joints()]  # where the arm stood still for the frame
    tmap = spot_map(
        ctx,
        wrist,
        q,
        scene_fix,
        scene_q,
        table_z,
        getattr(getattr(watcher, "grab", None), "wrist_lens_in_hand_m", None),
    )
    t_map = time.monotonic()
    # the reach of a spot judged from where the clear was to end is the
    # reach from here where the arm stands as near as the runner flies a
    # planned leg from (its drift gate): ahead's verdicts are taken, and its
    # carry, planned from there, flies from here
    off = None if ahead is None else max(abs(ang_diff(a, b)) for a, b in zip(q, ahead.q, strict=False))
    near = off is not None and off <= DRIFT_REPLAN_RAD
    reach = _Reach(ahead.known if near else None)
    rest = list(pd.rest_joints(cfg))
    carry, refused, chosen, carry_s, flew = None, [], None, 0.0, False
    for drop, tier, says in spot_candidates(ctx, tmap, table_z, q, seeds=(look_joints(rest), rest), reach=reach):
        if len(refused) >= CARRY_TRIES:
            break
        if near and ahead.carry is not None and _same_spot(drop, ahead.drop):
            carry, chosen, flew = ahead.carry, (ahead.drop, tier, says), True
            break
        t = time.monotonic()
        carry, why = plan_carry(ctx, cfg, q, drop)
        carry_s += time.monotonic() - t
        if carry is not None:
            chosen = (drop, tier, says)
            break
        refused.append("[%.2f, %.2f]: %s" % (drop.xyz[0], drop.xyz[1], why))
    t_end = time.monotonic()
    note = _ahead_note(ahead, off, flew, chosen, reach)
    runner.note(
        "lid_spot",
        xyz=None if chosen is None else [round(v, 4) for v in chosen[0].xyz],
        tier=None if chosen is None else chosen[1],
        wrist_points=0 if wrist is None else len(wrist),
        mapped=tmap is not None,
        refused="; ".join(refused) or None,
        look_s=round(t_look - t0, 3),
        map_s=round(t_map - t_look, 3),
        candidates_s=round(t_end - t_map - carry_s, 3),
        ik_s=round(reach.ik_s, 3),
        spot_checks=len(reach.found),  # the spots whose reach was judged here (the rest: ahead's)
        carry_s=round(carry_s, 3),
        ahead=note,
        pooled=pooled_note(ctx, runner, None if ahead is None else ahead.room, None if chosen is None else chosen[0]),
    )
    if note is not None:
        log.info(
            "[press_demo] %s"
            % (
                "the carry planned while the lift flew goes (%.2f s not spent here)" % note["saved_s"]
                if flew
                else "the carry planned while the lift flew is not flown: %s" % note["why"]
            )
        )
    if carry is None:
        return put_lid_back(
            ctx,
            cfg,
            runner,
            "no carry to a spot the lid could go to plans (%s)" % ("; ".join(refused) or "no spot the hand reaches"),
        )
    drop, tier, says = chosen
    _say_spot(ctx, eyes, chosen, refused, tmap, table_z)
    _video(
        ctx,
        "CARRY to [%.2f, %.2f] (%s) and SET DOWN, let go where it meets the table" % (drop.xyz[0], drop.xyz[1], tier),
    )
    log.info(
        "[press_demo] ONE MOTION: %s — to [%.2f, %.2f] and down onto the table, %s"
        % (", ".join(lg.name for lg in carry.legs), drop.xyz[0], drop.xyz[1], _touch_says(carry))
    )
    watch_setdown(ctx, carry.legs, carry)  # where the lid meets the table, watched
    res = runner.run(carry.legs, execute=True, lookahead=lambda _end: plan_home_after(ctx, cfg, carry))
    ahead = runner.lookahead_result
    last = res[-1]
    if len(res) == 1 and last.flown_traj is not None:
        rp._note_flight(runner, "reach_carry_flight", [lg.name for lg in merge_groups(carry.legs)[0]], last)
    if not all(r.ok for r in res):
        if len(res) == 1 and not _started(last) and last.outcome == "refused":
            # the run's own checks refused it before anything went out (the driver never saw
            # it): the arm stands where the clear left it, holding the lid, which goes back on
            # the box the way it came off
            return put_lid_back(
                ctx,
                cfg,
                runner,
                "the lid's way to the table was refused before it started (%s)" % (last.detail or last.outcome),
            )
        log.info(
            "[press_demo] STOP: the lid's way to the table %s during %s (%s) — the arm holds the lid"
            % (
                "stopped" if _started(last) else "did not start",
                rp._in_flight(carry.legs, last, runner.last_retime),
                last.detail or last.outcome,
            )
        )
        if _started(last):
            rp.let_go(ctx)
        return 1
    home, why = ahead if ahead is not None else (None, "the lookahead plan failed")
    if home is None:
        home, why = plan_home_after(ctx, cfg, carry)
    return _release_and_home(ctx, cfg, runner, carry, drop, tier, home, why)


def _say_spot(ctx, eyes, chosen, refused, tmap, table_z):
    """Where the lid goes (`chosen`: the spot, its tier, what it says):
    kept (ctx.lid_drop), said with the spots the planner refused, and drawn
    (perception/viz; the lid video's scene panel)."""
    from box_opening.tasks import press_demo as pd

    drop, tier, says = chosen
    ctx.lid_drop = drop
    log.info(
        "[press_demo] lid goes to [%.2f, %.2f, %.3f] — %.0f mm from the box, %s%s"
        % (
            drop.xyz[0],
            drop.xyz[1],
            drop.xyz[2],
            1000 * math.hypot(drop.xyz[0] - ctx.cpose.xyz[0], drop.xyz[1] - ctx.cpose.xyz[1]),
            says,
            "; the planner refused %s" % "; ".join(refused) if refused else "",
        )
    )
    pd.viz_hook("spot", tmap, drop, tier, says, table_z)  # the map and the spot, drawn off this thread (perception/viz)
    _video_spot(ctx, eyes, drop)


def _touch_says(carry):
    """Where `carry`'s set-down lets the lid go."""
    return "slowly from %.0f mm over where the plan has the lid meet it, let go where it %s %.0f mm over that" % (
        1000 * SETDOWN_SLOW_M,
        "meets it, or" if SETDOWN_FROM_CONTACT else "ends,",
        -1000 * (carry.over_m or 0.0),
    )


def put_lid_back(ctx, cfg, runner, why):
    """A STOP with the lid in the fingers and the arm standing (no carry to any spot plans, or
    the carry was refused before it went out): the lid goes back where it came from, the one
    place known to take it, rather than staying in the fingers over the table. Built from where
    the arm stands (so no drift gate can refuse it): a straight line back over the box to the top
    of the lift, and straight down to where the knob was held (ctx.lid_lifted) in one execution;
    the fingers open there; the hand rises straight up off the knob as the lid came up; home.

    Never after a cancel, an e-stop or a motion stopped part way: those leave the arm where it
    stopped (the caller's STOP says it holds the lid). Returns 1: the lid did not come off. Its
    last line says the lid is back on the box and the arm home, or where it stopped holding."""
    from box_opening.runtime import approach

    lifted = ctx.lid_lifted
    if lifted is None or lifted.q_grip is None:
        log.info("[press_demo] STOP: %s — the arm holds the lid (no lift to retrace)" % why)
        return 1
    log.info("[press_demo] PUT BACK: %s — the lid back on the box where it was gripped, then home" % why)
    _video(ctx, "PUT BACK: %s" % why)
    chain = approach._chain()
    world = _full_world(ctx)
    quat = lifted.quat
    top_xyz = [float(v) for v in chain.fk(lifted.q_top)[1]]
    grip_xyz = [float(v) for v in chain.fk(lifted.q_grip)[1]]
    lines, q = [], [float(v) for v in ctx.client.joints()]
    for name, target, speed in (
        ("lid:back-over", top_xyz, rp.free_line_speed(ctx)),
        ("lid:put-back", grip_xyz, cfg.lift_speed),
    ):
        pts, refusal = chain.straight_line(q, target, quat)
        if pts is not None:
            refusal = dp.line_refusal(chain, pts)
        if pts is None or refusal is not None:
            runner.note("lid_put_back", ok=False, why=why, refused="%s: %s" % (name, refusal))
            log.info(
                "[press_demo] STOP: %s, and the lid's way back onto the box is not a straight line the arm can "
                "fly (%s: %s) — the arm holds the lid" % (why, name, refusal)
            )
            return 1
        pts = [[float(v) for v in p] for p in pts]
        lines.append(dp.compliant_line(ctx, name, pts, quat, speed, world))
        q = pts[-1]
    res = runner.run(lines, execute=True)
    if not all(r.ok for r in res):
        runner.note("lid_put_back", ok=False, why=why, stopped=res[-1].leg_name)
        log.info(
            "[press_demo] STOP: %s, and the lid's way back onto the box %s (%s) — the arm holds the lid"
            % (why, "stopped" if _started(res[-1]) else "did not start", res[-1].detail or res[-1].outcome)
        )
        if _started(res[-1]):
            rp.let_go(ctx)
        return 1
    release = _gripper_leg(ctx, PlanState(joints=None, chain=0), "lid:release", GRIPPER_CMD_OPEN, world)
    res = runner.run([release], execute=True)
    if not all(r.ok for r in res):
        runner.note("lid_put_back", ok=False, why=why, stopped="lid:release")
        log.info(
            "[press_demo] STOP: %s, the lid is back on the box, and the fingers do not open (%s) — the arm holds"
            % (why, res[-1].detail or res[-1].outcome)
        )
        rp.let_go(ctx)  # the spring pressing the lid onto the box
        return 1
    ctx.lid_lifted = None  # the lid is on the box again, out of the fingers
    runner.note("lid_put_back", ok=True, why=why)
    rise, refusal = chain.straight_line(
        [float(v) for v in ctx.client.joints()], [grip_xyz[0], grip_xyz[1], grip_xyz[2] + lift_off_m(ctx.model)], quat
    )
    if rise is not None:
        refusal = dp.line_refusal(chain, rise)
    if rise is None or refusal is not None:
        log.info(
            "[press_demo] STOP: %s, the lid is back on the box, and the way up off the knob is not a straight "
            "line the arm can fly (%s) — the arm holds, its fingers open" % (why, refusal)
        )
        return 1
    rise = [[float(v) for v in p] for p in rise]
    res = runner.run([dp.compliant_line(ctx, "lid:rise-off", rise, quat, cfg.lift_speed, world)], execute=True)
    if not all(r.ok for r in res):
        log.info(
            "[press_demo] STOP: %s, the lid is back on the box, and the way up off the knob %s (%s) — the arm holds"
            % (why, "stopped" if _started(res[-1]) else "did not start", res[-1].detail or res[-1].outcome)
        )
        if _started(res[-1]):
            rp.let_go(ctx)
        return 1
    homed = rp._go_home(ctx, cfg, runner, "home after putting the lid back (%s)" % why)
    if not homed:
        return 1
    runner.finish()
    log.info("[press_demo] STOP: %s — the lid is back on the box, and the arm is home" % why)
    return 1


# The retreat back up from the lid on the table flies from where the arm
# stands at the release (retreat_from_live), its first RETREAT_HOLD_S
# standing there: the fingers open on the knob before the hand rises
RETREAT_HOLD_S = 0.25


def retreat_from_live(ctx, carry, world):
    """The retreat back up to the hover over the lid, solved from where the
    arm stands at the release — the joints read, on the planned line's
    branch — not from where the touch was sent: the carry arrives 8-43 mrad
    off that (the table holds the hand up, the spring lags), and against the
    runner's 40 mrad drift gate the planned retreat was refused on 2 of 7
    runs (39.5, 42.8 mrad; 13:35 and 14:26 on 10-30), 4 more within 3-8
    mrad of it. Its first RETREAT_HOLD_S stand still where the arm is
    (Leg.start_hold_s): the reference set where the hand is, the open hand
    sinks nowhere (a refused retreat's rebuild let it drop 15.6 and 26.6 mm
    toward the lid), and the fingers open before it rises. Returns (the
    leg, how far its end is from the planned retreat's, rad) — the home
    planned ahead flies from it within DRIFT_REPLAN_RAD — or (None, why)
    where no line from here solves: the planned one goes, as before."""
    from box_opening.geometry import ang_diff
    from box_opening.runtime import approach
    from box_opening.runtime.branches import on_branch

    chain = approach._chain()
    planned = [list(map(float, p)) for p in carry.retreat]
    live = [float(v) for v in on_branch(ctx.client.joints(), planned[0])]
    _R, t_end = chain.fk(planned[-1])
    pts, why = chain.straight_line(live, [float(v) for v in t_end], carry.quat)
    if pts is not None:
        why = dp.line_refusal(chain, pts)
    if pts is None or why is not None:
        return None, why
    pts = [[float(v) for v in p] for p in pts]
    leg = dp.compliant_line(ctx, "place:retreat", pts, carry.quat, rp.free_line_speed(ctx), world)
    leg.start_hold_s = RETREAT_HOLD_S
    return leg, max(abs(ang_diff(a, b)) for a, b in zip(pts[-1], planned[-1], strict=False))


def _release_and_home(ctx, cfg, runner, carry, drop, tier, home, why):
    """The lid on the table at `drop` (`carry` flown): the knuckle read (the
    lid slipped?); the release, the retreat starting as the fingers open
    (defer_join: the pads leave the knob in the first millimetres, the
    retreat's first ones come slower than that); then `home` — the legs
    after the release planned ahead, None: home planned after the retreat
    (`why` there was none), slow and under MEDIUM as every home flies
    (core.plan_home). Returns the exit code: 0 with the lid set down and the
    arm home."""
    world = _full_world(ctx)
    ok_q, pos, _stalled = ctx.client.gripper_cmd(None)  # query only: still on the knob?
    kept = not ok_q or pos is None or in_band(pos, cfg.grip_band)
    watched = None if getattr(carry, "contact", None) is None else carry.contact.record()
    runner.note("reach_setdown", knuckle=None if pos is None else round(float(pos), 3), kept=kept, contact=watched)
    if watched and watched.get("met"):
        log.info(
            "[press_demo] SET-DOWN CONTACT: the lid met the table %+.0f mm from where the plan put it; %s"
            % (
                watched["above_top_mm"],
                (
                    "the touch stopped there, %.0f mm short of its planned end, and lets go"
                    if watched.get("cut")
                    else "a cut would have ended the touch %.0f mm short of its planned end"
                )
                % watched["cut_mm"]
                if watched.get("cut_mm") is not None
                else "the touch's planned end stands",
            )
        )
    if not kept:
        log.info(
            "[press_demo] the fingers read %.3f at the table, outside the grip band %s: the lid slipped on the way"
            % (pos, list(cfg.grip_band))
        )
    release = _gripper_leg(ctx, PlanState(joints=None, chain=0), "place:open", GRIPPER_CMD_OPEN, world, defer_join=True)
    ctx.lid_at = drop  # every world from here holds the lid where it was set down
    retreat = (
        home
        if home is not None
        else [dp.compliant_line(ctx, "place:retreat", carry.retreat, carry.quat, rp.free_line_speed(ctx), world)]
    )
    up, home_legs = retreat[:1], retreat[1:]
    live_up, off = retreat_from_live(ctx, carry, world)
    if live_up is None:
        log.info("[press_demo] no retreat solves from where the arm stands (%s): the one planned flies" % off)
    else:
        up = [live_up]
        ahead = bool(home_legs)
        if home_legs and off > DRIFT_REPLAN_RAD:
            # it ends too far from where home was planned from for the runner to fly that home
            home_legs = []
            why = "the retreat from where the arm stood ends %.3f rad from the one home was planned from" % off
        runner.note(
            "retreat_from_live",
            end_off_rad=round(off, 4),
            hold_s=RETREAT_HOLD_S,
            home_ahead=ahead,  # a home was planned ahead ...
            home_flies=bool(home_legs),  # ... and flies from where this retreat ends
        )
    log.info(
        "[press_demo] RELEASE as the retreat starts (%s), then HOME, slow, in MEDIUM impedance%s"
        % (", ".join(lg.name for lg in up), ": %s" % ", ".join(lg.name for lg in home_legs) if home_legs else "")
    )
    _video(ctx, "RELEASE as the retreat starts, then HOME")
    res = runner.run([release, *up], execute=True)
    opened = runner.join_pending()
    if opened is not None and not opened.ok:
        log.info("[press_demo] STOP: the fingers did not open at the table (%s) — the arm holds" % opened.detail)
        rp.let_go(ctx)  # the spring pressing the lid onto the table, or holding it over
        return 1
    if not all(r.ok for r in res):
        last = res[-1]
        if _started(last):
            log.info(
                "[press_demo] STOP: the way up from the lid stopped during %s (%s) — the arm holds"
                % (rp._in_flight(up, last, runner.last_retime), last.detail or last.outcome)
            )
            rp.let_go(ctx)
            return 1
        line, _why = rp._line(ctx, up[0].target[1], carry.quat)
        if line:
            res = runner.run(
                [dp.compliant_line(ctx, "place:retreat", line, carry.quat, rp.free_line_speed(ctx), world)],
                execute=True,
            )
            if not all(r.ok for r in res):
                log.info("[press_demo] STOP: the way up from the lid stopped (%s) — the arm holds" % res[-1].detail)
                rp.let_go_part_way(ctx, res)
                return 1
        home_legs = []  # planned from where the arm stands now
    if home_legs:
        res = runner.run(home_legs, execute=True)
        if not all(r.ok for r in res):
            log.info(
                "[press_demo] STOP: the way home stopped during %s (%s) — the arm holds"
                % (rp._in_flight(home_legs, res[-1], runner.last_retime), res[-1].detail or res[-1].outcome)
            )
            if _started(res[-1]):
                rp.let_go(ctx)
            return 1
    else:
        ok = rp._go_home(ctx, cfg, runner, "home from the lid (%s)" % why if why else "")
        if not ok:
            return 1
    runner.finish()
    if not kept:
        log.info(
            "[press_demo] STOP: the arm is home, but the lid slipped out of the fingers on the way to [%.2f, %.2f]"
            % (drop.xyz[0], drop.xyz[1])
        )
        return 1
    _note_frames(ctx, stage="done", popped=True)
    log.info(
        "[press_demo] DONE — box open, lid set down at [%.2f, %.2f] (%s), arm home" % (drop.xyz[0], drop.xyz[1], tier)
    )
    return 0
