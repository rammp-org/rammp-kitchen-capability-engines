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
  AGAIN   flush (below press_demo.KNOB_SHUT_BELOW_MM): the same press once
          more, its way in solved again from where the arm stands, and the
          look again. Still flush: STOP, home. Neither flush nor up: not
          pressed again (a popped knob pressed latches the box again), the
          GRIP as for a knob up; no height read: STOP, home.
          Not up on a lid that rose (released, LID_POP_M): STOP, home, the
          box left open — a flush knob gives the fingers nothing to hold,
          and a released lid is not pressed again.
  DIRECT  first, straight from the first press's hold (GRIP_FROM_PRESS):
          up GRIP_RISE_M, the fingers opening to the knob's width once the
          pads have left the knob (GRIP_OPEN_ABOVE_HELD_M), down the stroke's
          line to the grip's height (grip_lid_z),
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
          (room_ahead, lid_spots), ranked by how far the
          hand turns from where the clear ends; the first spot the hand
          reaches straight down (drop_reachable) whose carry and home after
          the set-down both plan. Nothing clear: the most room seen, then
          beside the box, then the nearest table point the hand reaches,
          off the box (spot_candidates) — one line says which, and why.
          Every one of them on table the cameras measured this run, the
          whole lid and the hand (clear_table.off_table; owner 2026-11-17,
          after a lid let go over the floor: the "it should go no matter
          what" of 2026-10-28 ends there). No such spot among those: more
          are looked for before anything else (wider_candidates: the whole
          of the measured table, then the hand leaning as far as the
          press's own look does, SETDOWN_LEAN_MAX_DEG; and once more on a
          map with a fresh wrist frame in it where the one in hand was not
          made where the arm stands: SpotSearch). Still none: the lid goes
          back on the box (put_lid_back), said.
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
  PLACE   the fingers open only where the set-down's watch felt the lid
          meet the table, at about the height the run has for it
          (SetdownContact): then as the retreat back up to the carry's
          height starts; then home — planned with the carry while the hand
          came down to the knob, or while the retreat flies where the spot
          was chosen after the lift, never while the set-down flies (its
          watch is polled from its first sample) — slow and in the driver's
          MEDIUM joint impedance, as every home flies (core.plan_home). No
          such contact by the touch's end: the lid is not let go, and not
          taken back to the box either (owner 2026-11-19: "it should keep
          trying spots, assume tere will always be a spot for the lid"):
          up with it, that spot and the lid's own footprint round it set
          aside for the run, and on to the next spot by the same rules,
          set down the same watched way (another_spot) — spot after spot,
          for as long as the run's time leaves room for one more and the
          way back to the box after it (RunClock, SPOT_S and BOX_S), and
          never more than SETDOWN_SPOTS. The box is the last resort
          (carry_back, put_lid_back): no spot left, no time for another,
          or a set-down that could not be watched at all.

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
lets go there, and only there; it goes no lower than SETDOWN_PAST_M under
where the run has the lid meet the table (the cameras' table, moved by what
the fingers found of the lid's top: setdown_table_z). Where it looks at rest — the
knob, and the spot where nothing was planned ahead — and before the fingers
close on the knob, it waits for the arm to stand still (runner.wait_still):
between motions, never a stop.
"""

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from box_opening import LOGGER as log
from box_opening.constants import (
    DRIFT_REPLAN_RAD,
    GRIPPER_CMD_CLOSED,
    GRIPPER_CMD_OPEN,
    PLANNER_LIMIT_MARGIN_RAD,
    SANITY_MARGIN_RAD,
    TRANSIT_SPEED,
)
from box_opening.models.container import ContainerPose
from box_opening.perception import clear_table
from box_opening.primitives import core
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
from box_opening.runtime.guards import in_band, sanity_violations
from box_opening.runtime.legs import merge_groups
from box_opening.tasks import depth_press as dp
from box_opening.tasks import reach_press as rp
from box_opening.tasks.press_contact import CONTACT_ONSET_M, HISTORY_S, PressContact

# The press, then once more where the look does not see the knob up (owner
# 2026-10-28) and the lid did not rise (LID_POP_M: a released lid is not
# pressed again), then STOP. Only a knob the look reads flush is pressed
# again (press_demo.knob_call, below KNOB_SHUT_BELOW_MM): the tilted look
# reads popped knobs down to 5.0 mm (the wrist study's I-Y2), and in the
# physics study a second press on a popped knob latched the box again
# (E-C20, 1 of 1 flown).
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
# ... and wider than that where the lid leaves room: the pads midway between the knob's edge and the
# lid's (grip_open_cmd), 74.5 mm for the pink canister's 44 mm knob on its 105 mm lid — 15 mm clear
# of the knob either side and 15 mm inside the lid's edge. Flatbot 2026-11-12 19:12:49: the press
# opened the box, its aim 7.5 mm from the knob's centre (the look after it), and the fingers, open
# to 64 mm, came down 8 mm clear of a knob the wrist measured 47-48 mm across — one of them at its
# edge — and shut the box again (the owner watched it; the next look read the knob flush).
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
# pose, and the look decides as before: the knob flush, the press once more;
# up or in between (KnobLook.shut), the grip from there (attempt 1). Its own attempt is 0. With no look
# there is no rise measured (KnobLook.lid_dz_m): a lid that rose meets the
# fingertips before the way down ends, the spring on it, and they close round
# the knob's base.
GRIP_FROM_PRESS = True
# How far the hand rises off the hold before the fingers spread and it comes down again. It was 25 mm
# ("over the popped knob (~7.5 mm proud)"), which left out what the button and the lid do when the
# press lets them go: the button comes back up its stroke (KNOB_STROKE_M), the released lid rises on
# its seal (9.4-9.6 mm on 2026-10-28, LID_POP_M up to 30) and the knob stands 7.8 mm over that — its
# top ~25 mm over where the hold held the pads. Flatbot 2026-11-12 19:12:49 (the owner: "it just
# didnt raise high enough before it came back down to grab it, reclosing the box"): told 25 mm, the
# pads reached 22.9 and stood at 20 when the hand came down. Now the stroke, the most a released lid
# rises and a knob's height over it; where that line does not fly, the same with the hand leaning
# further back by its top (LIFT_LEAN_DEG), then GRIP_RISE_MIN_M each way.
GRIP_RISE_M = KNOB_STROKE_M + LID_POP_M[1] + KNOB_STROKE_M
GRIP_RISE_MIN_M = 0.035
# ... and the hand comes down again only where the arm measured its pads this far over the hold at
# the rise's top (PadsAbove.highest): the popped knob's top and a margin. Lower, the fingers may not
# have left the knob, and the way down would press it shut: back to the look pose instead, the look
# to judge it.
GRIP_RISE_LEAST_M = 0.030
# The way down is watched (press_contact.PressContact, armed once it has come down GRIP_TOUCH_ARM_M:
# the pads lag a reference that starts from rest, and that reads as a rise). Where the open
# fingertips meet something — the lid, standing higher released than the hold says, or a knob one of
# them came down on — the reference stops there, nothing pressed on down, and the hand comes back up
# touch_rise_m before the fingers close: their pads end GRIP_TOUCH_CLEAR_M over what they met, round
# the knob's base on a lid, over the top of a knob (a close on air, the look to aim again, the box
# still open). Nothing met, the way down ends where it always did.
GRIP_TOUCH = True
GRIP_TOUCH_ARM_M = 0.008
GRIP_TOUCH_CLEAR_M = 0.0015
# ... and the fingers open on the way up once the pads, as the arm measures
# them, stand this far over where the hold held them: off the popped knob,
# its stroke and a margin over it (press_contact.PadsAbove, Leg.send_when).
# They opened as the rise began (owner 2026-11-02: "it opens the grippers too
# fast and makes the fingers slide along the lid. i need it to open after the
# fingers stop touching the lid"): 16:20's spread from shut to the knob's
# width in the rise's first 0.35 s, the pads 6 mm over the hold sinking to
# 1.5 as they spread, the hand rising only after
# ... which was the stroke and 5 mm, 12.5 mm over the hold: still on a popped knob, which follows the
# pads up to ~25 mm (GRIP_RISE_M's note). Now the stroke and the most a released lid rises; a hand
# that does not get that high opens at the rise's top, as before.
GRIP_OPEN_ABOVE_HELD_M = KNOB_STROKE_M + LID_POP_M[1]
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
# ... and where that lift, or every clear from its top, does not fly, this
# much past it (plan_lift): 35 mm for the pink canister's 25 mm cap. On the
# chair study's 0.86 / 0.91 m side counters (study/fixes/highcounter, 11-06)
# the 40 mm lift stopped at joint_6's limit 24-40 mm up (median 30) in 67 of
# 225 rows; at 35 mm in 51, the lid coming off in 143 rows against 127 at 40
# mm (with the clear to the carry's floor on the cameras' top, the leaning
# set-down and the walk carry), at every contact height in 10 cells against
# 0; that floor over the lid's top as gripped (lift_clear_targets), 113, 9 of
# them at 35 mm. The cap's skirt then clears the rim by 9.5 mm at the lift's
# top (14.5 at 40 mm) and its lowest point stays 9.4 mm or more over the
# canister on the way (14.4). 30 mm (0.005) got 158 rows, the skirt 4.4 mm
# over the rim: not until the cap's skirt is measured at 22 mm or less
# (lid_dims.z is an estimate).
LIFT_OFF_MIN_CLEAR_M = 0.010
# ... and where no lift straight up flies at the hand's own lean, the hand
# leans this much further back toward the arm as it rises (plan_lift), the
# least of these that flies. Flatbot 2026-11-12 18:12: the box 0.71 m out on
# a 29 in table, the knob held with the hand leaning 30 deg and the elbow 0.5
# rad from straight — the arm at its reach, so the lift ran out 20-28 mm up
# whatever the elbow's swing (every self-motion of that grip, offline), and
# the run backed out of an open box ("the lid has no way off the box"). The
# wrist rides 0.29 m up the hand's axis: each degree the hand leans back
# lowers it 2.5-3.5 mm there and brings it in, and from that grip and 18:11's
# (45 deg, 30 mm of lift) the full lift flies leaning 5 deg more. The lid
# turns with the hand about the knob, its edge nearest the arm that much
# lower: a lean is flown only where that edge still ends over the rim
# (_lean_keeps_clear), 4 mm of the 15 at 5 deg for the pink canister's lid.
LIFT_LEAN_DEG = (5.0, 10.0)
# The set-down: at setdown_speed or faster (PLACE_TOUCH_SPEED below) to
# SETDOWN_SLOW_M above where the plan has the lid meet the table — the
# table the set-down goes by (setdown_table_z), the model's lid hanging from
# the grip (setdown_poses) — then at PLACE_TOUCH_SPEED, watched
# (watch_setdown), down to SETDOWN_PAST_M UNDER it: stopped where the arm
# feels the lid meet the table, and let go there and nowhere else. Flown to
# the plan's end the spring pressed on down 2-34 mm past the touch —
# 2026-11-02 16:20 met it 72-77 mm up, during the fast way down, and ended
# 37 mm under the hand, ~60 N (owner: "presses the lid down onto the table
# waaaaaaay too hard ... the lid should just be let go of when it's ~1 cm
# above the table"). Until 2026-11-17 the touch ended 10 mm OVER the plan's
# table and the lid was let go there whatever was felt; that day's first run
# let it go over the floor, 64 mm over the plan, on a contact that was none
# (SETDOWN_SETTLE_S). A lid is now let go only on a contact (owner: "the arm
# dropped the lid off the table"), so the reference must go on past where
# the table is for the arm to feel it: the watch needs 3-5 mm of the
# reference past the touch (press_contact's rises). 8 mm under the plan: the
# 19 set-downs flown to their plan's end (10-29 to 11-04) first met the table
# -0.4..+26 mm from where setdown_table_z has it, never lower. That is the
# first touch, not what the watch needs (review 2026-11-17): the pads lag the
# reference, and a lid that lands on its rim settles as the hand comes on, so
# the rise is seen 4-10 mm of the reference after it — more where the lid
# settles slowly. Those flights replayed through the watch with the touch
# ending here (test_reach_lid_off, setdown_flown.SETDOWNS_OLDER): of the 18
# that kept the lid, the 12 flown at today's touch pace (53-61 mm/s) are all
# felt, the reference +14.7..-3.4 mm from the plan (the least 4.6 mm before
# this end: 10-29 12:04, its gap rising 0.26 mm a mm of descent, met 16 mm
# over the plan); of the 6 flown with the touch at twice that (104-125 mm/s,
# #89) five are felt at -2.0..-6.4 mm and one, 10-29 14:45 — met 2 mm over
# the plan with the pads 6 mm behind the reference, the gap rising 0.27 mm a
# mm — only at -14.0 mm: past this end, it would have gone back on the box.
# Not made deeper for it. At today's pace nothing on record needs it; a rise
# that slow met that near the plan would want some 18 mm (the slow rule's 5
# mm of it), and a touch that deep presses a table it does not feel twice as
# hard and takes the counter's keep-out down with it (clear_table
# .COUNTER_TOP_CLEAR_M stands 25 mm under the counter for THIS end, the
# hand's spheres 17 mm under its top). A lid such a touch does not feel goes
# back on the box; it is not dropped.
SETDOWN_SLOW_M = 0.08
SETDOWN_PAST_M = 0.008
# What the watch takes for the lid meeting the table (SetdownContact):
#
# SETDOWN_SETTLE_S: no rise is judged for this long after the slow stretch
# begins. The three set-downs flown with the watch (2026-11-13 01:52, 01:53,
# 11-17 16:40; their flight files replayed) all "met the table" 74-165 ms
# after it armed, 62.3-64.2 mm over the plan, the last with the lid over the
# floor. What rose was the arm catching up with its reference: the way down
# (free_line_speed, 0.55) flies arm_joint_2 27-47 mrad and arm_joint_4 25-42
# mrad off their reference, the pads 8-12 mm UNDER theirs as the slow stretch
# begins, and as the reference slows (135 -> 55 mm/s) those joints close up:
# the pads came up on the reference 3.8-4.7 mm in 0.18-0.27 s (14-22 mm/s),
# over the press's fast rule (3 mm over the least of 0.2 s), still descending
# at 52-66 % of the reference's pace. The flights were cut there, so how long
# the catching up lasts is not on record: 0.6 s is the press's own settle
# (press_contact.CONTACT_SETTLE_S), 33 mm of the slow stretch.
#
# SETDOWN_CREDIBLE_M: a rise that BEGAN higher than this over where the plan
# has the lid meet the table is not the table: it is noted, the watch settles
# again, and the descent goes on, slowly. The 19 real ones stood -0.4..+26.3
# mm from the plan (setdown_table_z's).
#
# SETDOWN_FREE_REF_M, SETDOWN_FREE_RISE_M: where a rise began is where the
# pads last came FREELY down — the least gap since they last kept the
# reference's pace: the gap no more than SETDOWN_FREE_RISE_M higher at the end
# of SETDOWN_FREE_REF_M of the reference's descent than at its start, on
# three kept samples in a row (a gap that falls back takes it down with it).
# What holds the pads up from there on is ONE thing, whichever rule sees it
# and however far they have sunk into it since: the table only where that
# began inside the band. The second review of 2026-11-17: a rise was dated by
# the rule that saw it (the fast one by the least of its last 0.2 s) and
# "held up" was counted from where a settle ended, so something that YIELDS,
# first met above the band — a bag, a cloth, a hand: the pads sinking half of
# the way their free path goes under its top — was noted "not the table",
# then taken for the table once the pads had sunk inside the band (met at +48
# mm: let go at +30, the reference 37 mm past the first touch; at +56: after
# 67 mm), and the arm's catching up, where it was more than SETDOWN_HELD_M
# and straddled the settle's end, was let go on in free air (24 mm at 26
# mm/s: "the table" at +35). The catching up is such a rise too, begun where
# the slow stretch begins (the three recorded stood lowest 9-62 ms after it,
# the pads 67-68 mm over the plan): never the table, whatever its size and
# pace, and over once the pads come freely down again — the table met after
# that is the table (the first review's cases: at +10..+26 mm after a 5 mm
# rise above the band that fell back; after 13 mm of catching up; after one
# that outlasts the settle). Of the 19 set-downs flown onto the table, the 17
# the watch feels began +1.1..+27.2 mm over the plan by this; the heights the
# rules date them to, +0.6..+24.0, are what the record says, as before. 10 mm
# and 1.5 mm: a lid that settles on its rim raises the gap 0.26 mm a mm of
# descent or more (SETDOWN_PAST_M's note), the catching up 0.3-0.45 (the
# three recorded), free air once it has settled 0.1 (press_contact: the dip
# recovers at 5-6 mm/s).
#
# SETDOWN_STALL_REF_M, SETDOWN_STALL_PADS_M, SETDOWN_HELD_M: the pads held up
# are held up by something, whatever the settle says — the table where that
# began inside the band (met while the watch settled again); above it,
# something else, and the descent stops there with the lid not let go. Held
# up, by the pads' own descent: STALLED, they came down less than
# SETDOWN_STALL_PADS_M while the reference came down SETDOWN_STALL_REF_M
# (three kept samples in a row), which the catching up never shows — in the
# three recorded flights the pads came down 6.9-8.3 mm in every 12 mm of the
# reference's (55-69 % of its pace), in the 19 flown onto the table 6.4 mm or
# more before the table and still 3.9 mm where the rise rules felt it;
# something rigid stops the descent 12.4 mm of the reference past where the
# pads met it at most (some 20 N at the hand's 1.6 N/mm), wherever between 36
# and 60 mm over the plan it stands and whatever the arm's lag. Or RISEN: the
# gap SETDOWN_HELD_M over where the pads last came freely down — what yields
# without stalling them, stopped with the spring 15-16 mm deflected (some 25
# N). Past what the catching up can make: what had risen by the three cuts and
# what the joints' errors left there are worth, 8-11 mm at the pads in all;
# one of the whole SETDOWN_HELD_M still reads as held (the descent stops in
# free air and the lid goes back on the box, not let go): no flight on record
# came near it. The history, both reviews of 2026-11-17: "held up" was the gap
# SETDOWN_HELD_M over its least since the watch ARMED, dated to that least's
# sample — the pads 9 mm under the reference coming up 5 mm as the slow
# stretch begins, then one rise above the band (the rules blind for the next
# 33 mm, +40 down to +7 mm over the plan, where the 19 recorded tables stood),
# a table at +10, +13, +20 and +26 mm read "something held the lid up 69 mm
# over the plan", the height the watch armed at, and the lid went back on the
# box from a table it stood on; the same with no rise above the band once the
# catching up reached 13 mm (2026-11-17's gap stood 13.7 mm under the
# reference where this watch would arm, had risen 6.0 mm by its cut and was
# still rising 25 mm/s). Then it was counted from where a settle ended and
# dated where the pads stood: the count dropped just short of its 15 mm where
# the settle ended and begun again, something rigid 36-54 mm over the plan —
# or a table the plan stands that far under: setdown_table_z does not move it
# for a rough fix, or with no grip height — was pressed up to 33 mm, some 52 N
# (it had been 17 mm; the owner, 2026-11-02, of ~60 N: "presses the lid down
# onto the table waaaaaaay too hard").
#
# Not covered, said plainly (on synthetic descents: the tests' and the second
# review's). A "catching up" that begins with the pads within ~3 mm of the
# band or inside it — 0.55 s or later after the slow stretch begins with the
# lags on record (8.5-13.8 mm, lowest 9-62 ms after it), from 0.25 s with one
# of 28 mm — reads as the table, as a lid settling on its rim does. So does
# something that yields met 36-38 mm over the plan (dated up to 3 mm lower,
# where its rise has reached CONTACT_ONSET_M), and something so soft that the
# pads sink nine tenths of the way (the gap rising no faster than free air's
# own; four fifths, where it stands within 38 mm) — these as before both
# reviews. And a table met before the pads have come freely down again — the
# catching up still under way less than ~10 mm of the descent before it, or a
# rise above the band that has not fallen back — is one rise with that, and
# the lid goes back on the box: a table 34 mm over the plan under such a rise,
# one 20 mm or more over it under 13.8 mm of catching up that takes 0.75 s.
SETDOWN_SETTLE_S = 0.6
SETDOWN_CREDIBLE_M = 0.035
SETDOWN_FREE_REF_M = 0.010
SETDOWN_FREE_RISE_M = 0.0015
SETDOWN_STALL_REF_M = 0.012
SETDOWN_STALL_PADS_M = 0.002
SETDOWN_HELD_M = 0.015
# Owner 2026-10-29: "we also need to speed up the lid placement". The way down
# to SETDOWN_SLOW_M over the table is free air, so it flies at the free-air
# lines' speed (reach_press.free_line_speed: 0.55 in a compliant run; it was
# setdown_speed, 0.35); the last stretch onto the table at PLACE_TOUCH_SPEED,
# the press's TOUCH_SPEED. It was twice that from #89 (12:02's and 12:04's way
# down and touch took 1.8 s of the carry), and the owner found the lid set
# down "too hard" (2026-10-29 14:45): the faster touch met the table at twice
# the speed, the damper (zeta 1.0) pushing on while the reference ran on.
PLACE_TOUCH_SPEED = dp.TOUCH_SPEED
# Past every spot straight down (spot_candidates' second pass: none
# reached, or the carries to those refused), the hand sets the lid down
# leaning this far back toward the arm, facing
# the spot (reach_press.reach_quat; the round lid's yaw is free), the lid
# where the grip took it (setdown_poses). On the 0.86 / 0.91 m side counters
# of the chair study (study/fixes/highcounter, 11-06) no straight-down hand
# reaches its carry height (0.514 m, 0.564 on 0.91 m) over spots 10-112 mm
# inside the arm's reach: joint_6's range is what fails, in 39 of the 225
# press-planned rows outright and in every row once the clear goes no
# higher than the carry's floor (lift_clear_targets). Leaning 10 / 20 / 30
# deg the lid came off in 77 / 100 / 98 rows (30 deg no more, and it put
# the lid below the rim over the canister in 6); the lean matched to the
# grip's own, the lid level, in 98. At 20 deg the lid lands tilted 8 deg
# median, 20 at most, the edge nearest the arm first: let go where that edge
# meets the table (SetdownContact), its far edge falls 15-36 mm and it settles
# flat — as today's straight-down set-down after a grip leaning 0-15 deg
# lands it 0-15 deg tilted. A bench run whose hand reaches a spot straight
# down never leans.
SETDOWN_LEAN_DEG = 20.0
# Past every spot at that lean, and only then (wider_candidates' last pass), the hand leans as far as
# the press's own look pose does: reach_press.REACH_TILTS_DEG's first, 30 deg. Owner 2026-11-19, asked
# whether a set-down that feels no table should put the lid back on the box or try another spot first:
# "it should keep trying spots, assume tere will always be a spot for the lid" — and on the one far
# table on record there are next to none at 20 deg. 2026-11-17 16:40's map (the box 0.78 m out, the
# table measured from x 0.64), flatbot's bare arm from the joints that run's carry began at, the
# measured table gone through 4 cm apart: of 28 places with table under the whole lid the hand
# reaches 0 straight down, 2 leaning 20 deg — 4 cm from each other, one lid's footprint: a set-down
# that feels no table at the one leaves no second spot — and 7 leaning 30, three of them on the box's
# far side (40 deg: 14; not taken: nothing flew a set-down past 20).
# What 20 was chosen for (the note above) is kept by two checks of their own, asked only of a lean
# past it:
#  - how the lid lands. Leaning 20 deg it lands tilted 20 deg at most, its far edge falling 36 mm. A
#    hand leaning further is taken only where the lid, level as the grip took it, lands tilted no
#    more than that (landing_tilt_deg): where the grip itself leaned, as it does over the far boxes
#    that need this.
#  - the canister. At 30 deg the study's carry "put the lid below the rim over the canister" in 6
#    rows. A carry to such a spot is refused where the lid's lowest point, anywhere over the
#    canister, would come under the top the lid came off (lid_low_over_box) — and so is every carry
#    from one spot to the next, whose way may cross the box, which stands open by then.
SETDOWN_LEAN_MAX_DEG = float(rp.REACH_TILTS_DEG[0])

# ---- the lid's other spots (owner 2026-11-19: "it should keep trying spots") ------------------
#
# SETDOWN_SPOTS: the most spots one run takes the lid to. The run's time ends the tries first
# wherever the engine runs it (below: four to six spots in a clean run); this bounds a run started
# by hand, which nothing times. Never an unbounded loop: each spot left also takes its own footprint
# out of what is left to try (SpotSearch.aside).
SETDOWN_SPOTS = 6
# The run's time. The engine stops a run that is still going at its budget with a Ctrl+C, wherever
# it is (node.RUN_BUDGET_S: 90 s, from the run's start — less a line-up made before it — under the
# orchestrator, whose own cancel comes at 120 s; from its last go for a goal sent by hand), and a
# run cut with the lid in its fingers ends as the arm holding it, for a person to clear. So the run
# is told its budget as it is started (RUN_BUDGET_ENV, node.budget_env; RunClock reads it) and
# starts no further spot it cannot finish and still end properly: after it, where that one feels no
# table either, the lid back on the box, the release and home.
#
# How long those take, from the run records (~/.ros/box_opening/runs and ~/.ros/rammp_box_opening/
# runs, 295 runs, the 22 that set a lid down, 2026-10-29 to 11-17; measured 2026-11-19):
#
# SPOT_S, a spot flown to its end: the carry, the way down and the touch as planned took 2.4-3.5 s
# in 20 of the 22 (median 3.1; 5.1 s on 10-30 14:26, and 7.8 s on 10-29 12:02, the hand turning 91
# deg), the touch 0.3 s more since it ends 18 mm lower; the way back up is the retreat's line,
# 1.3-1.7 s flown alone (14 runs, a 0.25 s hold in the newest), and 0.1-0.5 s go by between two
# goals. 8 s holds 21 of the 22; the one it does not takes 2 s of what BOX_S and END_S have to spare.
# CHOICE_S, the next spot chosen with the arm standing: the wrist's frame, the map, the reach of
# the spots and the carry's plan took 1.3-1.8 s at rest (4 runs of 10-29); the same work under a
# motion 0.7-4.6 s (22 runs, 3.2 s at nine in ten: the Jetson shares its cores with the driver).
# BOX_S, from a touch that felt nothing to the arm home with the lid back on the box. Never flown
# on the robot (no run on record put a lid back), so it is the sum of its parts as each was flown:
# back up 1.3-1.7 s; back over the box a carry's line, 0.7-2.2 s at nine in ten (6.0 at most) and
# down the lift's, 0.3-0.9; the fingers open 0.4-1.6; up off the knob 0.3-0.9; home planned at rest
# 0.1-2.8 s (14 runs) and flown 3.0 s median, 6.9 at nine in ten, 8.6 at most (the 55 homes since
# 2026-11-10, slow under MEDIUM); the arm standing still before a goal 0.5 s median (170 waits).
# The medians add up to ~10 s, the nine-in-ten values to ~18.
# END_S: what the two clocks may differ by (the run reads its go up to ~0.5 s after the engine
# counts from it) and the run's own ending (its record, its threads).
#
# A further spot is started only with CHOICE_S + SPOT_S + BOX_S + END_S (35 s) left as the one
# before it is left, and its carry is sent only with SPOT_S + BOX_S + END_S (31 s) left. A clean
# run on flatbot reaches its first set-down 27-39 s after it starts (2026-11-13, 11-17), so of 90 s
# some 51-63 are left there: three to five more spots at ~6.5 s each (up 1.3 s, still 0.5, chosen in
# 1, flown 3.5: the medians). A run that had a full line-up inside its budget (node.LINE_UP_S, 35 s)
# has none to spare, and its first no-table goes to the box.
# The first set-down itself is never refused for time: it is the run's own plan, merged with the
# lift where it was chosen ahead.
SPOT_S = 8.0
CHOICE_S = 4.0
BOX_S = 20.0
END_S = 3.0
RUN_BUDGET_ENV = "BOX_OPENING_RUN_BUDGET"
# wider_candidates' own tier, the whole of the measured table: the reach of at most this many places
# asked in each pass — twice the 'nearest' tier's 24; a reach check took 6-40 ms offline, 85 ms on
# the robot under a motion (2026-11-17 16:40's 40 checks in 3.4 s), so some 4 s a pass with the arm
# standing where the hand reaches none. The places stand clear_table.SPOT_APART_M apart (4 cm, the
# least two offered spots stand apart); every one with table under the whole lid and the hand is
# asked where there are no more of them than this — 16:40's map has 28 — and on a bigger table this
# many of them, evenly from the nearest the hand to the farthest, so that it is still the whole
# table that is gone through (_offers, _thinned)
WHOLE_TABLE_TRIES = 48
WHOLE_TABLE = "the whole table"
# What a carry flown with the lid is asked along the path it actually flies, once it is not the first
# spot's own (lid_way_refusal; review of 2026-11-19). plan_carry asked a planned carry only of the
# joint gate, the chair and its user and the box's footprint; the map only of the straight chord at
# carry height (_offers' way); and the planner's world holds no table. On flatbot's URDF a joint-space
# line between two hovers 0.35-0.85 m apart left that chord by 36-174 mm, and in two of seven pairs
# sank 111-117 mm, to 80-89 mm over the table (the review's probes). So:
#  - LID_HANG_M: the lid hangs this much lower than the model's cap puts it — the 19 set-downs flown
#    to their plan's end first met the table 13.2 mm over the plan in the mean (setdown_table_z's
#    note: lid_dims.z is an estimate) — and every height asked of the lid here is taken that much
#    lower (the canister's top too: lid_low_over_box had no margin at all);
#  - CARRY_SAG_M: the lid's lowest point may pass this far under the lower of the path's two ends
#    (each a hover, the lid CARRY_CLEAR_M over the box's top at the least), no further: the arm's
#    own walk between two hovers keeps their height, a path that sinks under it is no carry;
#  - CARRY_CHECK_RAD: the path is asked between its waypoints too, no joint moving more than this
#    from one state asked to the next (runtime/body's CHECK_STEP_RAD).
LID_HANG_M = 0.013
CARRY_SAG_M = 0.010
CARRY_CHECK_RAD = 0.02
# A leaning hand's real body and what stands on the table (hand_into_standing; the same review). The
# gate's hand (clear_table.HAND_PARTS) is a hand straight down: two rectangles over the spot. Leaning
# back toward the arm, the hand's body stands arm-side of the spot — at 30 deg a point 250 mm up the
# hand is 125 mm arm-side and 217 mm up — and the rectangles stay where they were: a spot was taken
# with a 15 cm thing 7 cm arm-side of the lid's middle, the arm's centre line passing through it; at
# 20 deg that line passed 18 mm from its top. A leaning set-down of the wider search, or of a spot
# after the first, is taken only where the body model's own spheres of the hand and wrist, at the
# set-down's own joints all the way down, stand this far clear of everything the map shows standing:
# clear_table.UP_SLACK_M, what the hand straight down keeps of a thing's height, on top of the
# spheres' own buffer (5 mm) and the map's cell (a thing is its whole cell).
HAND_STANDS_CLEAR_M = clear_table.UP_SLACK_M
# The way back up from a touch that felt nothing, and what goes by before the next goal: the part of
# SPOT_S that is not the carry's own flight (the retreat's line 1.3-1.7 s, 0.1-0.5 s between goals).
# Once a carry is planned its own flight is known (flight_s), and the time is asked for that and this
SPOT_UP_S = 2.5
# Two spots where something held the lid up at heights no further apart than this are one thing: the
# table itself, standing higher than the run has it (the plan is the same at both; the 19 flown met
# the table within 27 mm of each other across runs whose plans stood up to 24 mm apart). No further
# spot is pressed on: each "held" presses 12-16 mm into what it met, some 20-25 N (SETDOWN_STALL_REF_M)
SAME_HELD_M = 0.015


@dataclass(frozen=True)
class LeanSpot(ContainerPose):
    """A spot for the lid that the hand reaches only leaning `lean_deg`
    back toward the arm (spot_candidates' second pass): setdown_poses sets
    the lid down there leaning; `yaw` is the spot's own, unused there."""

    lean_deg: float = SETDOWN_LEAN_DEG


def lift_off_m(model, clear_m=None):
    """How far the knob is lifted straight up: the cap off the rim,
    LIFT_OFF_CLEAR_M (else `clear_m`) past it."""
    return float(model.lid_dims[2]) + (LIFT_OFF_CLEAR_M if clear_m is None else float(clear_m))


def _lift_m(ctx, lift):
    """How far straight up LiftOff `lift` takes the lid (its lift_m; lift_off_m where it says none)."""
    up = getattr(lift, "lift_m", None)
    return lift_off_m(ctx.model) if up is None else float(up)


def grip_open_cmd(model):
    """The knuckle the fingers come down to the knob at: the pads midway
    between the knob's edge (model.button_diameter_m) and the lid's
    (model.lid_dims), never nearer the knob than GRIP_SIDE_CLEAR_M either
    side, on the 2F-85's about-linear aperture (_width_cmd); never wider
    than fully open."""
    knob = float(model.button_diameter_m)
    lid = float(min(model.lid_dims[0], model.lid_dims[1]))
    return _width_cmd(max(knob + 2.0 * GRIP_SIDE_CLEAR_M, 0.5 * (knob + lid)))


def _width_cmd(width):
    """The knuckle that opens the 2F-85's pads `width` apart: about linear from GRIPPER_OPEN_WIDTH_M
    at GRIPPER_CMD_OPEN to nothing at GRIPPER_CMD_CLOSED."""
    frac = min(max(float(width) / GRIPPER_OPEN_WIDTH_M, 0.0), 1.0)
    return GRIPPER_CMD_OPEN + (GRIPPER_CMD_CLOSED - GRIPPER_CMD_OPEN) * (1.0 - frac)


# A leaning hand's pads slant (fingers.pad_corner_below_m): with their middle over the knob's centre,
# the corner nearest the arm comes down on the lid first and the pads meet the knob's side that much
# higher — 2.8 mm at 15 deg, the most flown until 2026-11-12; 4.6 at 25, 5.5 at 30, of a knob 7.8 mm
# proud. So a leaning grip comes down this far further out from the arm (grip_shift_m): the knob under
# the pads' low half, their ends there no more than GRIP_PAD_OVER_M over the corner, and never nearer
# their edge than GRIP_PAD_EDGE_M.
GRIP_PAD_OVER_M = 0.002
GRIP_PAD_EDGE_M = 0.003


def grip_shift_m(quat):
    """How far out from the arm, past the knob's centre, the grip puts the pads' middle with the
    hand at `quat` (GRIP_PAD_OVER_M's note): nothing for a hand nearly straight down."""
    lean = math.radians(dp._lean_deg(quat))
    half = 0.5 * fingers.PAD_WIDTH_M
    if math.sin(lean) * half <= GRIP_PAD_OVER_M:
        return 0.0
    return max(0.0, min(half - GRIP_PAD_OVER_M / math.sin(lean), half - GRIP_PAD_EDGE_M))


def touch_rise_m(model):
    """How far the hand comes back up once its open fingertips have met the lid (GRIP_TOUCH), before
    they close: closing brings the pads further out along the hand (fingers.pads_above_shut), from
    where they stand open (grip_open_cmd) to where they stand on the knob's width, and they are to
    end GRIP_TOUCH_CLEAR_M over the lid there, round the knob's base, not pressed into it."""
    onto = fingers.pads_above_shut(grip_open_cmd(model)) - fingers.pads_above_shut(
        _width_cmd(float(model.button_diameter_m))
    )
    return max(onto, 0.0) + GRIP_TOUCH_CLEAR_M


def grip_lid_z(top_z, held_z=None):
    """The lid's height the grip goes by: GRIP_LID_ABOVE_HELD_M above where
    the compliant hold held the pads on the button (`held_z`), the arm's own
    touch; without a hold, the press's top (`top_z`, the higher of the
    aim's and the scene camera's)."""
    if held_z is None:
        return float(top_z)
    return float(held_z) + GRIP_LID_ABOVE_HELD_M


def setdown_poses(ctx, drop, held=None):
    """(attitude, tool where the lid meets the table, the carry's hover)
    for the spot `drop`: the hand straight down, turned by the spot's yaw
    (press_demo.place_quat, the attitude lid_spots judged the room
    for); the lid hangs where the grip took it, GRIP_BELOW_LID_M under the
    pads' reference; the hover above it as Place.hover_for has it, high
    enough for the lid to clear the box. A LeanSpot: the hand leaning
    (_lean_setdown_poses), the lid as gripped by the hand at joints `held`."""
    from box_opening.tasks.press_demo import place_quat

    if float(getattr(drop, "lean_deg", 0.0) or 0.0) != 0.0:
        return _lean_setdown_poses(ctx, drop, held)
    quat = list(place_quat(ctx.model, drop))
    contact = [
        float(drop.xyz[0]),
        float(drop.xyz[1]),
        tcp_z(float(drop.xyz[2]) + float(ctx.model.lid_dims[2]) - GRIP_BELOW_LID_M),
    ]
    return quat, contact, [float(v) for v in Place.hover_for(ctx, contact)]


def _lean_setdown_poses(ctx, drop, held):
    """setdown_poses for a LeanSpot: the hand leaning drop.lean_deg back
    toward the arm, facing the spot (reach_press.reach_quat). The lid as
    the grip took it, the hand at the joints `held` — where the carry
    starts: from the close on the lift and the clear hold the grip's
    attitude — level there, its top GRIP_BELOW_LID_M over the pads'
    reference (depth_press.DOWN_CONTACT_TOOL), held so in the hand: at the
    lean its bottom's centre over the spot, its lowest point on the table
    where the tool meets it; the hover over that as Place.hover_for has it,
    its carry floor (the lid's underside CARRY_CLEAR_M over the box) taken
    under the lid's lowest point. Straight down, held straight down, this
    is setdown_poses' own."""
    from box_opening.perception.d405 import quat_to_mat
    from box_opening.runtime import approach

    if held is None:
        raise ValueError("a leaning set-down needs the joints the hand holds the lid at")
    m = ctx.model
    Rg, _t = approach._chain().fk([float(v) for v in held])
    Rg = np.asarray(Rg, float)
    up = np.array([0.0, 0.0, 1.0])
    top = np.asarray(dp.DOWN_CONTACT_TOOL, float) + Rg.T @ (GRIP_BELOW_LID_M * up)  # in the tool frame
    normal = Rg.T @ up
    quat = [float(v) for v in rp.reach_quat(m, drop.xyz[:2], float(drop.lean_deg))]
    R = np.asarray(quat_to_mat(*quat), float)
    n = R @ normal
    bottom = R @ top - float(m.lid_dims[2]) * n  # the lid's bottom centre, from the tool
    lid_r = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1]))
    low = float(bottom[2]) - lid_r * math.sqrt(max(0.0, 1.0 - float(n[2]) ** 2))  # its lowest point, from the tool
    contact = [float(drop.xyz[0]) - float(bottom[0]), float(drop.xyz[1]) - float(bottom[1]), float(drop.xyz[2]) - low]
    floor = float(ctx.cpose.xyz[2]) + float(m.dims[2]) + core.CARRY_CLEAR_M + GRIP_BELOW_LID_M - low
    hover = [contact[0], contact[1], max(contact[2] + float(m.hover_standoff) + float(m.lid_dims[2]), floor)]
    return quat, contact, hover


def _abort_check(ctx):
    """runtime.abort.check on this run's client: LookaheadAborted where a Ctrl+C has come."""
    abort.check(getattr(ctx, "client", None))


def landing_tilt_deg(ctx, drop, held):
    """How far from level the lid lands at the LeanSpot `drop` (deg): level as the grip took it, the
    hand then at joints `held` (_lean_setdown_poses' lid), and the hand at the set-down leaning
    drop.lean_deg back toward the arm, facing the spot. What SETDOWN_LEAN_DEG's note says of a lean
    of 20 deg — tilted 20 deg at most — is asked with this of a lean past it (SETDOWN_LEAN_MAX_DEG)."""
    from box_opening.perception.d405 import quat_to_mat
    from box_opening.runtime import approach

    Rg, _t = approach._chain().fk([float(v) for v in held])
    normal = np.asarray(Rg, float).T @ np.array([0.0, 0.0, 1.0])  # the lid's, in the tool frame
    quat = rp.reach_quat(ctx.model, drop.xyz[:2], float(drop.lean_deg))
    n = np.asarray(quat_to_mat(*quat), float) @ normal
    return math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))


def _path_states(pts, step_rad=CARRY_CHECK_RAD):
    """The joint path `pts` as the arm flies it between its waypoints (straight joint lines): its
    states, no joint moving more than `step_rad` from one to the next."""
    out = [[float(v) for v in pts[0]]]
    for a, b in zip(pts[:-1], pts[1:], strict=False):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(math.ceil(float(np.abs(b - a).max()) / step_rad)))
        out += [[float(v) for v in a + (b - a) * k / n] for k in range(1, n + 1)]
    return out


def _lid_along(ctx, pts, held):
    """The held lid at each joint state of `pts`: ((n, 2) its bottom centre's x and y, (n,) the
    height of its lowest point) — level as the grip took it, the hand then at joints `held`
    (_lean_setdown_poses' lid), and hanging LID_HANG_M lower than the model's cap puts it."""
    from box_opening.runtime import approach

    chain = approach._chain()
    m = ctx.model
    up = np.array([0.0, 0.0, 1.0])
    Rg = np.asarray(chain.fk([float(v) for v in held])[0], float)
    top = np.asarray(dp.DOWN_CONTACT_TOOL, float) + Rg.T @ (GRIP_BELOW_LID_M * up)  # the lid's top, in the tool frame
    normal = Rg.T @ up
    lid_r = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1]))
    xy, low = [], []
    for p in pts:
        R, t = chain.fk([float(v) for v in p])
        R, t = np.asarray(R, float), np.asarray(t, float)
        n = R @ normal
        bottom = t + R @ top - float(m.lid_dims[2]) * n
        xy.append([float(bottom[0]), float(bottom[1])])
        low.append(float(bottom[2]) - lid_r * math.sqrt(max(0.0, 1.0 - float(n[2]) ** 2)) - LID_HANG_M)
    return np.asarray(xy, float).reshape(-1, 2), np.asarray(low, float)


def lid_low_over_box(ctx, pts, held):
    """Why the joint path `pts` would bring the held lid too low over the canister, or None:
    somewhere along it (between its waypoints too: _path_states) the lid's disc — level as the grip
    took it, the hand then at joints `held`, hanging LID_HANG_M lower than the model's cap
    (_lid_along) — stands over the box's footprint (round every place the run has the box:
    box_centres) with its lowest point under the top the lid came off: the cameras' top of the
    box, or the lid's top as the fingers took it where that stands higher (_gripped_lid_top). The
    hover of every spot holds the lid CARRY_CLEAR_M over that top (Place.hover_for, _lean_setdown_poses);
    this asks it of the way between two hovers, and of a hand that leans further than any flown
    (SETDOWN_LEAN_MAX_DEG's note: the canister stands open under it)."""
    from box_opening.runtime import approach
    from box_opening.tasks.press_demo import container_half

    m = ctx.model
    over = container_half(m) + 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1]))
    floor = float(ctx.cpose.xyz[2]) + float(m.dims[2])
    q_grip = getattr(getattr(ctx, "lid_lifted", None), "q_grip", None)
    if q_grip is not None:
        floor = max(floor, _gripped_lid_top(approach._chain(), q_grip))
    centres = box_centres(ctx)
    xy, low = _lid_along(ctx, _path_states(pts), held)
    under = 0.0  # the most the lid passes under that top, anywhere over the box (not the first state that does)
    for (x, y), z in zip(xy, low, strict=True):
        if min(math.hypot(float(x) - cx, float(y) - cy) for cx, cy in centres) < over:
            under = max(under, floor - float(z))
    if under > 1e-4:
        return "the lid's lowest point would pass %.0f mm under the top it came off, over the open canister" % (
            1000 * under
        )
    return None


def lid_way_refusal(ctx, tmap, pts, held, table_z):
    """Why the joint path `pts`, flown with the lid in the fingers, is no carry at carry height over
    the table the cameras measured, or None (LID_HANG_M's note). Along the whole of it as the arm
    flies it (_path_states), the lid as _lid_along has it:

     - its lowest point never more than CARRY_SAG_M under the lower of the path's two ends: a path
       that sinks between two hovers is refused, whatever it would or would not meet;
     - nothing the map `tmap` shows standing under the path it sweeps, taller than the lid passes
       there: the lid's disc and the hand over it, as clear_spots asks the straight way from the
       box (clear_table.carry_clearance_fast: LID_CLEAR_M, HAND_PARTS' own margins, UP_SLACK_M),
       asked here of the flown path stretch by stretch at the height the lid has on each, over the
       table at `table_z`.

    The canister is lid_low_over_box's (plan_carry asks both)."""
    states = _path_states(pts)
    xy, low = _lid_along(ctx, states, held)
    ends = float(min(low[0], low[-1]))
    k = int(np.argmin(low))
    if float(low[k]) < ends - CARRY_SAG_M:
        return "the lid would sink %.0f mm under the height it is carried at on the way, to %.0f mm over the table" % (
            1000 * (ends - float(low[k])),
            1000 * (float(low[k]) - float(table_z)),
        )
    if tmap is None or not len(tmap.standing()):
        return None
    m = ctx.model
    lid_r, lid_h = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1])), float(m.lid_dims[2])
    last = 0
    for i in range(1, len(states)):
        if i < len(states) - 1 and math.hypot(*(xy[i] - xy[last])) < tmap.cell:
            continue
        over = float(low[last : i + 1].min()) - float(table_z)  # the lid's lowest point over the table on this stretch
        room, what, at = clear_table.carry_clearance_fast(tmap, xy[last], xy[i], lid_r, over + lid_h, lid_h)
        if room < 0.0:
            return (
                "%s would pass %.0f mm inside what it must keep from what stands near [%.2f, %.2f], the lid %.0f mm "
                "over the table there"
                % (
                    what,
                    -1000 * room,
                    at[0],
                    at[1],
                    1000 * over,
                )
            )
        last = i
    return None


def _hand_links(body):
    """(links, missing): the links of `body` (runtime/body.Body) that are the hand and the wrist — every
    link the arm's last three joints move: the wrist's two links, and the bracelet with the gripper and
    the camera on it — and which of those three the model gives NO sphere to (`missing`, in words; all
    three for a body that does not say what moves its links). A Body knows only the links its sphere
    file names, and a file with none on the wrist is no model of the wrist: asked as it stood, the check
    of a leaning hand passed whatever the wrist's own links came into (the tests' stand-in file had
    none, and a 30 cm thing 250 mm arm-side of a 30 deg spot was taken with the wrist's centre line 3 mm
    from it; flatbot's own robot file, which has them, refused it — the review of 2026-11-19)."""
    paths, n = getattr(body, "paths", None) or {}, len(getattr(body, "index", None) or ())
    moved = {link: sum(1 for step in path if step[2] is not None) for link, path in paths.items()}
    parts = ((n - 2, "the wrist's first link"), (n - 1, "the wrist's second link"), (n, "the hand"))
    if n < 3:
        return set(), [name for _k, name in parts]
    links = {link for link, k in moved.items() if k >= n - 2}
    return links, [name for k, name in parts if k not in moved.values()]


_HAND_MODEL_SAID = set()  # what was said of a body model that is none of the hand and wrist: once a run


def _hand_model():
    """(the body model, its hand and wrist links) for what is asked of the hand's own body past the
    first spot — a leaning set-down's way down, a later carry's path — or (None, why there is none
    for them): no body model at all (the bench, body_layers off), or one whose sphere file gives the
    hand, or either link of the wrist, no sphere (_hand_links). The second is said once in the run's
    lines: its other checks (the chair, the user) still stand, and nothing else would tell that these
    do not."""
    from box_opening.runtime import approach

    body = approach.body_check()
    if body is None:
        return None, "no body model"
    links, missing = _hand_links(body)
    if not missing:
        return body, links
    why = "the body model has no spheres on %s" % " or ".join(missing)
    if why not in _HAND_MODEL_SAID:
        _HAND_MODEL_SAID.add(why)
        log.info(
            "[press_demo] %s: it is no model of the hand and wrist for the lid's further spots — no leaning "
            "set-down is taken past the first spot's own, and a carry after the first is asked of the lid and "
            "a hand straight down over it alone" % why
        )
    return None, why


def _hand_room(body, links, tmap, pts, table_z, most):
    """The least room the spheres of `body` on `links` (the hand's and the wrist's: _hand_model) have
    from what `tmap` shows standing over the table at `table_z`, at no more than `most` states evenly
    along the joint states `pts`, its ends among them: (metres, (x, y) of the thing, its height, the
    link), or None where nothing stands (clear_table.spheres_clearance)."""
    pts = [[float(v) for v in p] for p in pts]
    every = max(1, int(math.ceil((len(pts) - 1) / max(1, most - 1))))
    states = pts[::every] + ([pts[-1]] if (len(pts) - 1) % every else [])
    worst = None
    for q in states:
        centres, radii, at = body.spheres_at(q)
        keep = np.array([link in links for link in at], bool)
        got = clear_table.spheres_clearance(
            tmap, np.asarray(centres, float)[keep], np.asarray(radii, float)[keep], table_z
        )
        if got is not None and (worst is None or got[0] < worst[0]):
            worst = (got[0], got[1], got[2], [link for link, k in zip(at, keep, strict=True) if k][got[3]])
    return worst


def _into_what_stands(worst):
    """The words for a _hand_room that is too little: how near, what, where, which link."""
    room, at, tall, link = worst
    return "%s what stands %.0f mm tall at [%.2f, %.2f] (%s; %.0f mm is kept)" % (
        "%.0f mm into" % (-1000 * room) if room < 0.0 else "within %.0f mm of" % (1000 * room),
        1000 * tall,
        at[0],
        at[1],
        link,
        1000 * HAND_STANDS_CLEAR_M,
    )


def hand_into_standing(ctx, tmap, pts, table_z):
    """Why the hand, coming down the joint line `pts` to a leaning set-down, would come into what
    the map shows standing on the table, or None (HAND_STANDS_CLEAR_M's note): the body model's
    spheres of the hand and wrist (_hand_model) at states all the way down the line, each
    HAND_STANDS_CLEAR_M clear of every standing thing's column over the table at `table_z`
    (clear_table.spheres_clearance). No body model of them (none at all: the bench, body_layers off;
    or one with no spheres on the hand or on a link of the wrist): nothing tells the leaning hand's
    body from what stands there, and that is why — a leaning set-down is then not taken past the
    first spot's own."""
    body, links = _hand_model()
    if body is None:  # (`links` then says why: none at all, or what its sphere file lacks)
        return "%s of the hand to hold it, leaning, out of what stands by the spot" % (
            links if links == "no body model" else "%s: no model" % links
        )
    if tmap is None or not len(tmap.standing()):
        return None
    worst = _hand_room(body, links, tmap, pts, table_z, 9)
    if worst is None or worst[0] >= HAND_STANDS_CLEAR_M:
        return None
    return "the hand leaning there would come %s" % _into_what_stands(worst)


def hand_way_refusal(ctx, tmap, pts, table_z):
    """Why the hand, flying the joint path `pts` with the lid, would come into what the map shows
    standing on the table, or None: the same spheres by the same margin (hand_into_standing), at
    states all along the path as the arm flies it (_path_states). lid_way_refusal asks the lid's
    disc and a hand straight down over it (HAND_PARTS); from a set-down's attitude to the next the
    hand leans and turns on the way, and its body is not over the lid. None too where there is no
    body model of the hand and wrist (_hand_model): the lid's own check then stands alone, as for a
    hand straight down it may — and no leaning set-down is taken on such a run (hand_into_standing)."""
    body, links = _hand_model()
    if body is None or tmap is None or not len(tmap.standing()):
        return None
    worst = _hand_room(body, links, tmap, _path_states(pts), table_z, 40)
    if worst is None or worst[0] >= HAND_STANDS_CLEAR_M:
        return None
    return "the hand would pass %s on the way" % _into_what_stands(worst)


class LaterWay:
    """What the carry to a spot and its set-down are asked beyond what the first spot's are, on the
    map `tmap` over the table at `table_z`, the lid as the hand at joints `held` holds it — for a
    spot of the wider search and for every spot after the first (SpotSearch):
      path(pts)           the carry's own path: lid_way_refusal, then the hand's own body along it,
                          hand_way_refusal (plan_carry asks the canister of it too,
                          lid_low_over_box);
      descent(drop, pts)  a leaning set-down's way down: hand_into_standing (a hand straight down
                          is the gate's own, clear_table.off_table's)."""

    def __init__(self, ctx, tmap, table_z, held):
        self.ctx, self.tmap, self.table_z = ctx, tmap, float(table_z)
        self.held = [float(v) for v in held]

    def path(self, pts):
        why = lid_way_refusal(self.ctx, self.tmap, pts, self.held, self.table_z)
        return why if why is not None else hand_way_refusal(self.ctx, self.tmap, pts, self.table_z)

    def descent(self, drop, pts):
        if _spot_lean(drop) == 0.0:
            return None
        return hand_into_standing(self.ctx, self.tmap, pts, self.table_z)


def drop_reachable(ctx, drop, q_start, seeds=(), held=None, stands=None):
    """Why the hand does not set the lid down at `drop`, or None: no hand
    at the carry's hover at the set-down's attitude (straight down; for a
    LeanSpot leaning, the lid as gripped at `q_start`: setdown_poses) — the
    arm's own IK, from `q_start`, else from `seeds` — or no straight line
    down from it to the set-down's end (SETDOWN_PAST_M under where the plan
    has the lid meet the table). `held`: the joints the lid hangs level at,
    where those are not `q_start` (a carry from one spot to the next starts
    with the hand as the set-down left it, not as the grip had it:
    SpotSearch.held). `stands(drop, line)`: what else the way down is asked
    (LaterWay.descent: a leaning hand's body and what stands on the table)."""
    from box_opening.runtime import approach
    from box_opening.runtime import kinematics as kin

    chain = approach._chain()
    quat, contact, hover = setdown_poses(ctx, drop, held=q_start if held is None else held)
    q = chain.nearest_solution(list(q_start), hover, quat, seeds=tuple(seeds))
    if q is None:
        lean = float(getattr(drop, "lean_deg", 0.0) or 0.0)
        hand = "straight-down hand" if lean == 0.0 else "hand leaning %.0f deg" % lean
        return "no %s reaches its carry height %.3f" % (hand, hover[2])
    why = kin.body_refusal(chain, [q], start_exempt=False)  # the carry's end: the planner's world may be empty
    if why is not None:
        return "the hand at its carry height is in the way (%s)" % why
    pts, why = chain.straight_line(q, [contact[0], contact[1], contact[2] - SETDOWN_PAST_M], quat)
    if pts is None:
        return "the set-down is not a straight line the arm can fly (%s)" % why
    return None if stands is None else stands(drop, pts)


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
TIER_TRIES = {"clear": 8, "most room": 10, "beside the box": 8, "nearest": 24, WHOLE_TABLE: WHOLE_TABLE_TRIES}
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
# The no-land radius (owner 2026-11-04: "in run 2 the lid accidentally landed on the rim. we need
# to make a no land radius around the box"). 11-04 12:40 set the lid down 114 mm from where the
# press held the button (the old host build's 0.120 m ring) and it came down on the box's rim.
# Whatever tier chose it — clear, most room, beside the box, nearest — no spot puts the lid's
# disc within NO_LAND_CLEAR_M of the box's footprint (no_land_radius: 0.155 m centre to centre
# for the pink canister, the first nearest ring's), round every place the run has the box: where
# the press held its button and where ctx.cpose puts it (box_centres). The tiers measured from
# one of them only, and the cameras put them 37-57 mm apart on 11-04 (the aims' lines)
NO_LAND_CLEAR_M = NEAREST_CLEAR_M


def _table_z(watcher, scene_fix, cpose):
    """The table the lid goes down on: the scene camera's (main's
    adopt_scene_table puts it on the watcher), else the box's base."""
    for z in (getattr(watcher, "table_z", None), getattr(scene_fix, "table_z_scene", None)):
        if z is not None and math.isfinite(float(z)):
            return float(z)
    return float(cpose.xyz[2])


# The table the set-down goes by (setdown_table_z): the cameras' table, moved by what the fingers
# found of the lid's top. The cameras read the lid's top and the table it stands on in one frame;
# the fingers then hold the knob where the lid's top really is (the grip goes by the press's hold:
# grip_lid_z). What the two tops differ by is the cameras' height error at the box, and the table
# is that far off too: it is taken as far under the lid's top as the cameras saw it (SceneFix
# .height_m), the top where the fingers hold it. 2026-11-13 01:52, 01:53 and 11-17 16:40 (the wrist
# camera's fix, front table): the press met the lid 39.8, 41.4 and 44.7 mm over the cameras' top,
# the knob was gripped 45.6, 53.6 and 54.8 mm over where the fix had it, and the set-down's plan,
# built on the cameras' table, stood that far too low: the lid would have met the table (where one
# was) some 5 cm above "where the plan put it". On the 19 set-downs flown to their plan's end
# (10-29 to 11-04, the scene camera's fix: 0-24 mm moved) the hand came to rest on the table
# +0.2..+29.7 mm from the cameras' plan (mean +13.4) and -15.9..+16.6 mm from this one (mean
# +1.7); the lid first met it -0.4..+26.3 mm over this one (mean +13.2: lid_dims.z, the cap's
# thickness, is an estimate, and the lid hangs a little lower than it says).
# Not moved further than TOUCH_TABLE_MAX_M: the press's own band lets the cameras' top be 25 mm
# over and 45 mm under what it meets (press_contact.CONTACT_*), the aim's top up to 15 mm over the
# fix's; a difference past that is no measure of the table.
TOUCH_TABLE_MAX_M = 0.08


def setdown_table_z(ctx, table_z, fix, q_grip=None):
    """(the table's height the lid is set down by, how far that is from the cameras' `table_z`):
    the cameras' table moved by the fingers' hold of the lid's top (the comment above) — the pads
    at the joints `q_grip` the knob is gripped at (None: the lift's, ctx.lid_lifted),
    GRIP_BELOW_LID_M under the lid's top, and `fix` (the SceneFix whose table it is) the lid's top
    over its table. (`table_z`, None) where there is no grip yet, no fix that measured both (a
    rough one's place is no measure), or the two differ by more than TOUCH_TABLE_MAX_M."""
    from box_opening.runtime import approach
    from box_opening.tasks.press_contact import pads_z

    table_z = float(table_z)
    if q_grip is None:
        q_grip = getattr(getattr(ctx, "lid_lifted", None), "q_grip", None)
    top, seen = getattr(fix, "top_xyz", None), getattr(fix, "table_z_scene", None)
    if q_grip is None or top is None or seen is None or getattr(fix, "rough", False):
        return table_z, None
    height = float(top[2]) - float(seen)
    if not math.isfinite(height) or height <= 0.0:
        return table_z, None
    by_touch = pads_z(approach._chain(), q_grip) + GRIP_BELOW_LID_M - height
    moved = by_touch - table_z
    if not math.isfinite(moved) or abs(moved) > TOUCH_TABLE_MAX_M:
        log.info(
            "[press_demo] the fingers hold the lid's top %.0f mm from where the cameras' table and the box's height "
            "put it (more than %.0f): no measure of the table — the lid is set down by the cameras' table, %.3f"
            % (1000 * moved, 1000 * TOUCH_TABLE_MAX_M, table_z)
        )
        return table_z, None
    return by_touch, moved


def _say_setdown_table(spot_z, moved, table_z):
    """One line where the set-down's table is not the cameras' (setdown_table_z moved it 5 mm or more)."""
    if moved is not None and abs(moved) >= 0.005:
        log.info(
            "[press_demo] the table the lid is set down by: %.3f, %.0f mm %s the cameras' %.3f — the fingers hold "
            "the lid's top that much %s than the cameras put it, and the table is as far under it as they saw it"
            % (
                spot_z,
                1000 * abs(moved),
                "over" if moved > 0 else "under",
                table_z,
                "higher" if moved > 0 else "lower",
            )
        )


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


def spot_map(ctx, wrist, q, scene_fix, scene_q, table_z, lens_in_hand=None, more=()):
    """The table map the lid's spot is chosen on (clear_table.table_map, as
    press_demo.room_map builds it): the wrist's frame taken with the lid
    lifted (`wrist`, wrist_table'd at the arm's joints `q`), pooled with the
    scene camera's points of the table — moved by its error at the box (the
    aim measured where the box is), what was the arm when it looked (its
    centre line at `scene_q`, clear_table.ARM_CLEAR_M round it: flatbot
    stood at HOME) left out. `more`: [(points, joints)], the wrist's frames
    of this run's earlier looks with the lid in the fingers, pooled in as
    that one is (SpotSearch.look_again: each look sees another part of the
    table). None where none shows a table."""
    clouds = []
    scene = _scene_cloud(ctx, scene_fix, scene_q, table_z)
    if scene is not None:
        clouds.append(scene)
    tz = scene[1] if scene is not None else float(table_z)
    for pts, at in [(wrist, q), *more]:
        if pts is not None and len(pts):
            clouds.append((wrist_table(ctx, pts, at, lens_in_hand), tz))
    if not clouds:
        log.info("[press_demo] no map of the table for the lid: no camera kept points of it")
        return None
    why = []
    tmap = clear_table.table_map(clouds, ctx.cpose.xyz[:2], why=why)
    if tmap is None:
        log.info("[press_demo] no map of the table for the lid: %s" % "; ".join(why))
    return tmap


def counter_map(ctx, watcher, scene_fix, scene_q, table_z, aim=False):
    """The table map the counter's edges are read from before the lid is off
    (press_demo.keep_out_counter): the scene camera's cloud as spot_map takes
    it (moved by its error at the box, the arm when it looked left out) and,
    `aim`, the wrist's frame at the look pose (aim_cloud): parked side-on to
    a cabinet, the scene camera may not see the counter at all. A cloud that
    cannot be read is left out (said), the other still maps; None where
    neither shows a table, or where the map cannot be made (said: no map is
    no news of the counter, never the run's end)."""
    reads = [("the scene camera's cloud", lambda: _scene_cloud(ctx, scene_fix, scene_q, table_z))]
    if aim:
        reads.append(("the wrist's frame at the look", lambda: aim_cloud(ctx, watcher, table_z)[0]))
    clouds = []
    for name, read in reads:
        try:
            got = read()
        except Exception as e:  # noqa: BLE001 - a cloud for the counter's edges only
            log.info("[press_demo] the counter's edges: %s could not be read (%s) — left out" % (name, e))
            continue
        if got is not None:
            clouds.append(got)
    try:
        return clear_table.table_map(clouds, ctx.cpose.xyz[:2]) if clouds else None
    except Exception as e:  # noqa: BLE001 - a map for the counter's edges only
        log.info("[press_demo] no map of the counter for its edges: its clouds could not be mapped (%s)" % e)
        return None


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
    pts = np.asarray(pts, float)
    if not getattr(scene_fix, "rough", False):
        # the scene camera's error at the box, taken out of all of it. A ROUGH fix (a box the frame
        # cut, scene_source.SCENE_ROUGH) is off by its own fit, not by the camera's error: the cloud
        # is kept as it was measured
        pts = pts + [
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


def no_land_radius(m):
    """How near the box's centre the lid's centre may never be set down (m): the box's footprint
    (container_half), the lid's disc and NO_LAND_CLEAR_M between their rims."""
    from box_opening.tasks.press_demo import container_half

    return container_half(m) + 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1])) + NO_LAND_CLEAR_M


def box_centres(ctx):
    """Every place the run has the box's centre (x, y): ctx.cpose's (the aim's, the scene
    camera's before it), and, once the press held the pads on its button (Ctx.held_xy), the box
    whose button is there — wherever that is, unlike nearest_centre: a keep-out errs wide."""
    out = [(float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1]))]
    held = getattr(ctx, "held_xy", None)
    if held is not None:
        from box_opening.models.container import from_container

        button = from_container(ctx.cpose, ctx.model.button_offset)
        c = rp.cpose_for_button(
            (float(held[0]), float(held[1]), float(button[2])), ctx.cpose.yaw, ctx.model.button_offset
        )
        out.append((float(c.xyz[0]), float(c.xyz[1])))
    return out


def box_clearance(ctx, drop):
    """How far outside the no-land radius the lid set down at `drop` stands (m): the least over
    box_centres, negative inside it (the spot is never taken: spot_candidates)."""
    r = no_land_radius(ctx.model)
    return min(math.hypot(float(drop.xyz[0]) - x, float(drop.xyz[1]) - y) for x, y in box_centres(ctx)) - r


def no_land_note(ctx, drop):
    """The lid_spot row's 'no_land': the radius, how far outside it the chosen spot stands, and the
    box's centres it was measured from."""
    if drop is None:
        return None
    return {
        "radius_mm": round(1000 * no_land_radius(ctx.model), 1),
        "clear_mm": round(1000 * box_clearance(ctx, drop), 1),
        "centres": [[round(x, 4), round(y, 4)] for x, y in box_centres(ctx)],
    }


def spot_zone(m, cpose, worlds, tmap):
    """(x0, x1, y0, y1) a spot with room for the lid is looked for in: as far round the box as a
    spot may lie (clear_table.clear_spots' ring), on the table the cameras measured this run —
    what the map saw of it (clear_table.table_bounds) and, where the fix measured its extent,
    that (press_demo.drop_zone) — each the lid's own half-width and margin inside; and no farther
    out from the arm than its hand comes down: the straight-down hand's reach band of the bench
    (press_demo.DROP_X_M, the robot's), or as far out as the box itself where that stands past it
    (the arm pressed it there, leaning; a hand straight down or leaning SETDOWN_LEAN_DEG reaches
    no farther). It was that band alone, x 0.28..0.65, whatever table the cameras saw: 2026-11-17
    16:40 measured the table over x 0.64..1.15, the box 0.78 m out, and looked for a spot in x
    0.28..0.65. A box inside the band is searched round as before. The corner stands on the
    lattice the old zone's points stood on (from DROP_X_M[0], and CLEAR_SPOTS_Y_M from the box),
    so a spot judged there before is the same spot now."""
    from box_opening.tasks import press_demo as pd

    bx, by = float(cpose.xyz[0]), float(cpose.xyz[1])
    lid_r = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1]))
    reach = pd.lid_place_min_clear(m) + clear_table.SPOT_REACH_M
    zone = [max(bx - reach, pd.DROP_X_M[0]), min(bx + reach, max(pd.DROP_X_M[1], bx)), by - reach, by + reach]
    seen = clear_table.table_bounds(tmap)
    measured = []
    if seen is not None:
        inset = lid_r + clear_table.LID_CLEAR_M
        measured.append((seen[0] + inset, seen[1] - inset, seen[2] + inset, seen[3] - inset))
    if getattr(worlds, "table_extent", None) is not None:
        measured.append(pd.drop_zone(m, worlds))
    for x0, x1, y0, y1 in measured:
        zone = [max(zone[0], x0), min(zone[1], x1), max(zone[2], y0), min(zone[3], y1)]
    step = clear_table.SPOT_STEP_M
    ox, oy = pd.DROP_X_M[0], by - pd.CLEAR_SPOTS_Y_M
    zone[0] = ox + step * math.ceil((zone[0] - ox) / step - 1e-6)
    zone[2] = oy + step * math.ceil((zone[2] - oy) / step - 1e-6)
    return tuple(zone)


def lid_spots(m, cpose, worlds, tmap, table_z, judged_out=None, hand_at=None, cache=None):
    """(spots, why): press_demo.clear_lid_drops' search — where `tmap` shows room for the lid
    and the hand (clear_table.clear_spots), best first, and why none — over the table the cameras
    measured (spot_zone) instead of the bench's reach band. `hand_at`, `cache`, `judged_out`:
    clear_spots'. The spots on the table at `table_z` are the caller's to make."""
    from box_opening.models.container import from_container
    from box_opening.tasks import press_demo as pd

    lid_h = float(m.lid_dims[2])
    if hand_at is None:
        hand_at = clear_table.hand_yaw(rp.reach_quat(m, from_container(cpose, m.button_offset)[:2], 0.0))
    return clear_table.clear_spots(
        tmap,
        (float(cpose.xyz[0]), float(cpose.xyz[1])),
        spot_zone(m, cpose, worlds, tmap),
        pd.lid_place_min_clear(m),
        lambda xy: m.press_quat((xy[0], xy[1], float(table_z))),
        0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1])),
        lid_h,  # the fingertips at the release: the lid's height up, at the least
        float(m.dims[2]) + lid_h + core.CARRY_CLEAR_M,  # ... and on the carry (Place.hover_for's floor)
        lid_h,
        judged_out=judged_out,
        hand_at=hand_at,
        cache=cache,
    )


def _clear_first(m, cpose, judged):
    """The spots of `judged` (clear_spots') with room, in the order they are tried: those inside
    the straight-down hand's reach band of the bench (press_demo.DROP_X_M) first, as clear_spots
    ranks them (the roomiest, the least turn, the nearest the box: the order every run before
    2026-11-17 tried them in, where the band was all that was searched); then the rest of the
    measured table, ranked alike, the nearest the arm first — the hand reaches nearer in, and each
    spot tried costs an IK. SPOT_APART_M apart."""
    from box_opening.tasks import press_demo as pd

    bx, by = float(cpose.xyz[0]), float(cpose.xyz[1])

    def key(s):
        beyond = not pd.DROP_X_M[0] - 1e-9 <= s.xy[0] <= pd.DROP_X_M[1] + 1e-9
        how_far = math.hypot(s.xy[0], s.xy[1]) if beyond else math.hypot(s.xy[0] - bx, s.xy[1] - by)
        return (beyond, *clear_table._rank(s), how_far)

    picked = []
    for s in sorted((s for s in judged if s.room_m >= 0.0), key=key):
        if all(math.hypot(s.xy[0] - p.xy[0], s.xy[1] - p.xy[1]) >= clear_table.SPOT_APART_M for p in picked):
            picked.append(s)
    return picked


def spot_off_table(ctx, tmap, drop, held=None):
    """Why the lid let go at the spot `drop` would not rest on table the cameras measured this run
    (clear_table.off_table: the map's cells under the whole lid and the hand, and the fix's extent
    where it measured one), or None. The hand as the set-down has it there (setdown_poses; a
    LeanSpot's leaning, the lid as gripped at joints `held`)."""
    m = ctx.model
    quat, contact, _hover = setdown_poses(ctx, drop, held=held)
    return clear_table.off_table(
        tmap,
        drop.xyz[:2],
        quat,
        0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1])),
        float(m.lid_dims[2]),
        extent=getattr(ctx.worlds, "table_extent", None),
        hand_xy=contact[:2],
    )


@dataclass
class SpotsPassed:
    """What spot_candidates passed over, for the line that says why the lid has no spot: how many
    spots lay off the table the cameras measured (and the first one's why), how many inside the
    box's no-land radius, how many the hand does not reach."""

    off_table: int = 0
    first_off: str = None
    no_land: int = 0
    unreachable: int = 0
    aside: int = 0  # within the footprint left at a spot it was taken to and felt no table at (SpotSearch.aside)
    tilted: int = 0  # the hand leaning past SETDOWN_LEAN_DEG would land the lid tilted more than that lean lands it
    blocked: int = 0  # something the cameras saw stands in the lid's way there, taller than the lid is carried
    tight: int = 0  # a place of the whole table with less room than the hand straight down asks (clearance)

    def says(self):
        if not (
            self.off_table
            or self.no_land
            or self.unreachable
            or self.aside
            or self.tilted
            or self.blocked
            or self.tight
        ):
            return "no spot to try: no camera measured the table round the box"
        parts = []
        if self.off_table:
            parts.append("%d not on the table the cameras measured (the first: %s)" % (self.off_table, self.first_off))
        if self.no_land:
            parts.append("%d inside the box's no-land radius" % self.no_land)
        if self.aside:
            parts.append("%d within the footprint left at a spot where no table was felt" % self.aside)
        if self.tilted:
            parts.append(
                "%d where the hand leaning %.0f deg would land the lid tilted more than %.0f"
                % (self.tilted, SETDOWN_LEAN_MAX_DEG, SETDOWN_LEAN_DEG)
            )
        if self.blocked:
            parts.append("%d with something standing in the lid's way there" % self.blocked)
        if self.tight:
            parts.append("%d with less room for the lid and the hand than a spot must have" % self.tight)
        if self.unreachable:
            parts.append("%d the hand does not reach" % self.unreachable)
        return "of the spots looked at, " + ", ".join(parts)


def table_note(ctx, table_z, spot_z, passed):
    """The lid_spot row's 'table': the table as the cameras measured it (its height, the fix's
    extent), the height the set-down goes by (setdown_table_z) and what the spot search passed
    over for not lying on it."""
    ext = getattr(ctx.worlds, "table_extent", None)
    return {
        "z": None if table_z is None else round(float(table_z), 4),
        "setdown_z": None if spot_z is None else round(float(spot_z), 4),
        "extent": None if ext is None else [round(float(v), 3) for v in ext],
        "off_table": None if passed is None else passed.off_table,
        "first_off": None if passed is None else passed.first_off,
        "unreachable": None if passed is None else passed.unreachable,
        "aside": None if passed is None else passed.aside,
        "blocked": None if passed is None else passed.blocked,
        "tight": None if passed is None else passed.tight,
    }


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


def spot_candidates(
    ctx,
    tmap,
    table_z,
    q,
    seeds=(),
    reach=None,
    cache=None,
    spot_z=None,
    passed=None,
    aside=None,
    held=None,
    way=False,
    stop=None,
):
    """Where the lid may go, best first, each one the hand reaches straight
    down from joints `q` (`reach`, drop_reachable's stand-in — _Reach, the
    verdicts kept —, else drop_reachable): (spot, tier, what it says).
    Tiers, each tried at most TIER_TRIES: 'clear' — where `tmap` shows
    room for the lid and the hand on the table the cameras measured
    (lid_spots, in _clear_first's order, the hand's turn to each counted
    from where the arm's FK has it face at `q`); 'most room' — nothing
    clear, the spots it judged, the most room
    first; 'beside the box' — the spots beside it in the set-down zone
    (press_demo.lid_drop_candidates, drop_zone); 'nearest' — the table
    points round the box (_nearest_points). In every tier a spot inside the
    no-land radius round the box (box_clearance) is passed over, and so is
    one where the lid would not rest on table the cameras measured this run
    (spot_off_table: the whole lid and the hand on cells the map saw at the
    table's height, inside the extent the fix measured; no map, no spot) —
    neither counted as a try. Where every spot is passed over the lid has
    nowhere to go, and its caller looks wider (wider_candidates) before it
    puts it back on the box. A generator: only
    as many as are asked for are
    checked. Past every tier, the same tiers once more as LeanSpots, the
    hand leaning SETDOWN_LEAN_DEG (setdown_poses): a lid goes down leaning
    only once every spot straight down is past (none reached, or the
    carries the caller tried to those refused; its CARRY_TRIES counts both
    passes). `cache`: the room judged on `tmap` before (room_ahead's,
    clear_table.clear_spots'). `spot_z`: the table's height the spots are
    made at (setdown_table_z's; None: `table_z`, the height `tmap` was
    made on). `passed`, a SpotsPassed, counts what was passed over and
    why. `aside(spot)`: whether a spot is set aside for this run (one the
    lid was taken to and felt no table at, and what lies within the lid's
    own footprint of it: SpotSearch.aside) — passed over, never a try.
    `held`: the joints the lid hangs level at, where those are not `q`
    (drop_reachable's). `way`: the lid's way there is asked of every spot
    too (_offers: a carry that does not start over the box). `stop()`:
    _offers' (the run's clock, asked all through a later spot's choice)."""
    passed = passed if passed is not None else SpotsPassed()
    if stop is not None and stop():
        return  # (before the room on the map is judged: that alone can take a second)
    tiers = _spot_tiers(ctx, tmap, table_z, q, cache, spot_z)
    # every tier straight down first, as the bench's runs chose; only past
    # them all, the same again with the hand leaning (SETDOWN_LEAN_DEG)
    passes = [(lean, tiers) for lean in (0.0, SETDOWN_LEAN_DEG)]
    yield from _offers(ctx, tmap, q, seeds, reach, passed, passes, aside=aside, held=held, way=way, stop=stop)


def _spot_tiers(ctx, tmap, table_z, q, cache=None, spot_z=None):
    """spot_candidates' tiers, in its order: [(tier, [(spot, what it says), ...]), ...], no spot of
    them checked for anything yet (_offers does that, one at a time)."""
    from box_opening.tasks import press_demo as pd

    m = ctx.model
    spot_z = float(table_z if spot_z is None else spot_z)
    tiers = []
    if tmap is not None:
        from box_opening.runtime import approach
        from box_opening.runtime.kinematics import mat_to_quat_xyzw

        R, _t = approach._chain().fk([float(v) for v in q])
        facing = clear_table.hand_yaw(mat_to_quat_xyzw(np.asarray(R, float)))  # where the hand holding the lid faces
        judged = []
        _spots, why = lid_spots(m, ctx.cpose, ctx.worlds, tmap, table_z, judged_out=judged, hand_at=facing, cache=cache)
        tiers.append(
            (
                "clear",
                [
                    (
                        ContainerPose(xyz=(s.xy[0], s.xy[1], spot_z), yaw=s.turn),
                        "where the cameras see room for it and the hand (%.0f mm to spare%s; the hand "
                        "turns %+.0f deg)"
                        % (
                            1000 * s.room_m,
                            "" if s.what == "nothing" else ", %s nearest anything" % s.what,
                            math.degrees(s.rotation),
                        ),
                    )
                    for s in _clear_first(m, ctx.cpose, judged)
                ],
            )
        )
        most = sorted(judged, key=lambda s: -s.room_m)
        tiers.append(
            (
                "most room",
                [
                    (
                        ContainerPose(xyz=(s.xy[0], s.xy[1], spot_z), yaw=s.turn),
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
                for xyz, _side in pd.lid_drop_candidates(m, ctx.cpose, spot_z, zone=zone)
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
                for d in _nearest_points(ctx, spot_z, tmap)
            ],
        )
    )
    return tiers


def _offers(ctx, tmap, q, seeds, reach, passed, passes, aside=None, held=None, skip=(), stop=None, way=False):
    """What spot_candidates and wider_candidates yield: `passes`, [(the hand's lean, tiers)], gone
    through in order, each tier's spots one at a time through what every spot must pass (its
    docstring: the no-land radius, `aside`, the measured table under the whole lid and the hand; for
    a lean past SETDOWN_LEAN_DEG how the lid lands, landing_tilt_deg) — none of those a try — then
    the hand's reach from `q`, at most TIER_TRIES of a tier in a pass. `skip`: the spots (_spot_key)
    offered before, not offered again. `stop()`: asked of every spot looked at, before anything is
    worked out for it, and before each reach check; true ends it all (the run's time:
    SpotSearch.out_of_time — the wider search goes through hundreds of places, and the clock was
    asked only once a place had passed them all: 1-3 s later on a bare table, the review of
    2026-11-19). A place of the whole table, the hand straight down, must also have the room a
    'clear' spot has (clear_table.clearance: the lid's and the hand's own margins from what stands
    and from what was not seen) — the tier is ranked by nothing, and asked no room at all before.

    `way`: the lid's way to a spot is asked too — the straight line from where the hand holds it at
    `q` to the spot, against what the map shows standing taller than the lid is carried
    (clear_table.carry_clearance_fast, as clear_spots asks it of the way from the box). The room a
    'clear' spot was judged with holds the way from the BOX; the planner is told nothing of what
    stands on the table; and the other tiers never asked. So of a carry that starts elsewhere (from
    a spot the lid was left at) and of the whole table's places, far from the box, it is asked here:
    a spot with something standing in the lid's way is passed over, never a try."""
    tried, ended = [], []
    lid_from = q if held is None else held
    start = lid_r = carry_h = lid_h = None
    if way and tmap is not None:
        from box_opening.runtime import approach

        m = ctx.model
        start = [float(v) for v in approach._chain().fk([float(v) for v in q])[1][:2]]
        lid_r, lid_h = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1])), float(m.lid_dims[2])
        carry_h = float(m.dims[2]) + lid_h + core.CARRY_CLEAR_M  # the fingertips on the carry (Place.hover_for's floor)

    def fresh(d):
        return all(math.hypot(d.xyz[0] - t.xyz[0], d.xyz[1] - t.xyz[1]) > 1e-3 for t in tried)

    def passing(cands, lean):
        """`cands` through what a spot must pass before the hand's reach is asked, one at a time: (the
        spot to try, a LeanSpot where `lean`; what it says). What is passed over is counted, never a try.
        `roomy` (the loop's): the tier whose places must have a clear spot's room."""
        for d, says in cands:
            if stop is not None and stop():
                ended.append(True)  # (the loop below ends with it: no tier after this one is begun)
                return
            if not fresh(d):
                continue
            tried.append(d)
            if box_clearance(ctx, d) < 0.0:
                passed.no_land += 1  # inside the no-land radius: the lid would come down on the box's rim
                continue
            if lean:
                d = LeanSpot(xyz=d.xyz, yaw=d.yaw, lean_deg=lean)
                says = "%s; past every spot %s: the hand leaning %.0f deg" % (
                    says,
                    "straight down" if lean <= SETDOWN_LEAN_DEG else "at %.0f deg too" % SETDOWN_LEAN_DEG,
                    lean,
                )
            if aside is not None and aside(d):
                passed.aside += 1  # where no table was felt, or within the footprint that was left there
                continue
            if _spot_key(d) in skip:
                continue
            if lean > SETDOWN_LEAN_DEG and landing_tilt_deg(ctx, d, lid_from) > SETDOWN_LEAN_DEG + 0.5:
                # the lid would land tilted more than a lean of SETDOWN_LEAN_DEG lands it (half a degree: the
                # grip's own IK, and a lean of exactly that much more than the grip's)
                passed.tilted += 1
                continue
            off = spot_off_table(ctx, tmap, d, lid_from)
            if off is not None:
                # not table the cameras measured under the whole lid and the hand: never let go there
                passed.off_table += 1
                if passed.first_off is None:
                    passed.first_off = "[%.2f, %.2f], %s" % (d.xyz[0], d.xyz[1], off)
                continue
            if start is not None:
                room, _what, _at = clear_table.carry_clearance_fast(tmap, start, d.xyz[:2], lid_r, carry_h, lid_h)
                if room < 0.0:
                    passed.blocked += 1  # something stands in the lid's way there, taller than it is carried
                    continue
            if roomy and not lean:
                quat, _contact, _hover = setdown_poses(ctx, d)
                if clear_table.clearance(tmap, d.xyz[:2], quat, lid_r, lid_h)[0] < 0.0:
                    passed.tight += 1  # nearer something than the lid or the hand straight down may come
                    continue
            yield d, says

    for lean, tiers in passes:
        tried.clear()
        for tier, cands in tiers:
            roomy = tier == WHOLE_TABLE and tmap is not None
            if roomy and lid_r is None:
                lid_r = 0.5 * max(float(ctx.model.lid_dims[0]), float(ctx.model.lid_dims[1]))
                lid_h = float(ctx.model.lid_dims[2])
            todo = passing(cands, lean)
            if tier == WHOLE_TABLE:
                # all of the table: every place that passes, and where those are more than a pass asks the
                # reach of, that many of them evenly from the nearest to the farthest (_thinned)
                todo = iter(_thinned(list(todo), TIER_TRIES[tier]))
            n = 0
            while n < TIER_TRIES[tier]:
                got = next(todo, None)
                if ended:
                    return
                if got is None:
                    break
                d, says = got
                if stop is not None and stop():
                    return
                n += 1
                _abort_check(ctx)  # each reach check is an IK walk: a Ctrl+C goes out between two
                if reach is not None:
                    why = reach(ctx, d, q, seeds)
                else:
                    why = drop_reachable(ctx, d, q, seeds, held=held)
                if why is None:
                    yield d, tier, says
                else:
                    passed.unreachable += 1


def _thinned(items, keep):
    """`items` where there are no more than `keep`; else `keep` of them, evenly through their order
    (the first always among them)."""
    if len(items) <= keep:
        return items
    return [items[(i * len(items)) // keep] for i in range(keep)]


def _table_points(ctx, tmap, table_z, near=(0.0, 0.0)):
    """wider_candidates' own tier: every place on the table the cameras measured this run
    (clear_table.table_places: the map's own cells, the lid's half-width and LID_CLEAR_M inside what
    it saw; inside the fix's extent too where it measured one, press_demo.drop_zone), SPOT_APART_M
    apart, the nearest `near` first (where the hand holds the lid: the shortest carry, and the
    likeliest reached — the hand reaches where it stands). Not only round the box (spot_zone's ring,
    _nearest_points' rings) and not ranked by room: whether the lid may lie there is asked of each
    as of every spot (spot_off_table)."""
    from box_opening.tasks import press_demo as pd

    if tmap is None:
        return []
    m = ctx.model
    inset = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1])) + clear_table.LID_CLEAR_M
    within = pd.drop_zone(m, ctx.worlds) if getattr(ctx.worlds, "table_extent", None) is not None else None
    places = clear_table.table_places(tmap, clear_table.SPOT_APART_M, inset, within=within)
    ox, oy = float(near[0]), float(near[1])
    places.sort(key=lambda p: math.hypot(p[0] - ox, p[1] - oy))
    return [ContainerPose(xyz=(x, y, float(table_z)), yaw=0.0) for x, y in places]


def wider_candidates(
    ctx,
    tmap,
    table_z,
    q,
    seeds=(),
    reach=None,
    cache=None,
    spot_z=None,
    passed=None,
    aside=None,
    held=None,
    skip=(),
    stop=None,
):
    """More spots, for where spot_candidates has none whose carry plans (owner 2026-11-19: "it
    should keep trying spots, assume tere will always be a spot for the lid") — the same checks of
    every one of them and the lid's way there too (_offers' `way`), the same (spot, tier, what it
    says), and in this order:

     - 'the whole table': every place on the table the cameras measured (_table_points), the hand
       straight down, then leaning SETDOWN_LEAN_DEG. spot_candidates' tiers each stop after a few
       tries (TIER_TRIES: the 'nearest' rings after 24, hardly past the second ring) and none looks
       further than 0.39-0.45 m from the box; this one goes through all of the table, the nearest
       where the hand holds the lid first: the reach of every place that passes is asked, or, where
       those are more than WHOLE_TABLE_TRIES, of that many of them evenly from the nearest to the
       farthest;
     - then every tier once more, and the whole table, with the hand leaning SETDOWN_LEAN_MAX_DEG:
       only where the lid still lands as a lean of SETDOWN_LEAN_DEG lands it (landing_tilt_deg; the
       canister is plan_carry's to keep it off: lid_low_over_box).

    `skip`: the spots offered before (_spot_key), `stop()`: _offers'. The rest as spot_candidates'."""
    passed = passed if passed is not None else SpotsPassed()
    spot_z = float(table_z if spot_z is None else spot_z)
    if stop is not None and stop():
        return  # (before the table's places are gone through: the clock first)
    from box_opening.runtime import approach

    said = "no spot of any tier has a carry that plans: anywhere on the table the cameras measured, nearest the hand"
    hand = approach._chain().fk([float(v) for v in q])[1]
    whole = [(WHOLE_TABLE, [(d, said) for d in _table_points(ctx, tmap, spot_z, near=hand[:2])])]
    passes = [(0.0, whole), (SETDOWN_LEAN_DEG, whole)]
    if SETDOWN_LEAN_MAX_DEG > SETDOWN_LEAN_DEG:
        passes.append((SETDOWN_LEAN_MAX_DEG, _spot_tiers(ctx, tmap, table_z, q, cache, spot_z) + whole))
    yield from _offers(
        ctx, tmap, q, seeds, reach, passed, passes, aside=aside, held=held, skip=skip, stop=stop, way=True
    )


def _spot_lean(d):
    """How far the hand leans setting the lid down at spot `d` (deg; 0: straight down)."""
    return round(float(getattr(d, "lean_deg", 0.0) or 0.0), 3)


def _spot_key(d):
    """A spot's identity for _Reach: where, to 0.1 mm, the hand's turn and its lean."""
    return (
        round(float(d.xyz[0]), 4),
        round(float(d.xyz[1]), 4),
        round(float(d.yaw) % (2 * math.pi), 3),
        _spot_lean(d),
    )


def _same_spot(a, b):
    """Whether spots `a` and `b` are the one spot for a carry planned to
    either: AHEAD_SAME_SPOT_M apart at most, the hand turned and leaning the same."""
    turn = abs(math.remainder(float(a.yaw) - float(b.yaw), 2 * math.pi))
    return (
        math.hypot(a.xyz[0] - b.xyz[0], a.xyz[1] - b.xyz[1]) <= AHEAD_SAME_SPOT_M
        and turn < 1e-3
        and _spot_lean(a) == _spot_lean(b)
    )


class _Reach:
    """drop_reachable kept by spot (_spot_key): (why, how long it took).
    `known`, the verdicts of another one from (nearly) the same joints —
    carry_ahead's, the arm within DRIFT_REPLAN_RAD of where it judged them
    — are taken as they are, their time saved (saved_s); every spot not
    among them is judged here (ik_s) and kept (`found`)."""

    def __init__(self, known=None, held=None, stands=None):
        self.known = dict(known or {})
        self.found = {}
        self.ik_s = 0.0
        self.saved_s = 0.0
        self.reused = 0
        self.held = held  # the joints the lid hangs level at, where not the ones it judges from (drop_reachable)
        self.stands = stands  # what else a way down is asked (drop_reachable's: LaterWay.descent)

    def __call__(self, ctx, drop, q, seeds=()):
        key = _spot_key(drop)
        if key in self.known:
            why, took = self.known[key]
            self.saved_s += took
            self.reused += 1
            return why
        if key in self.found:  # judged before from these joints (the same search, a later pass over the spot)
            return self.found[key][0]
        t = time.monotonic()
        if self.stands is None:
            why = drop_reachable(ctx, drop, q, seeds, held=self.held)
        else:
            why = drop_reachable(ctx, drop, q, seeds, held=self.held, stands=self.stands)
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

    @property
    def shut(self):
        """The knob read flush (press_demo.knob_call, below
        KNOB_SHUT_BELOW_MM): the only knob pressed again."""
        from box_opening.tasks.press_demo import knob_call

        return self.found and knob_call(self.up_mm) == "shut"


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
    from box_opening.tasks.press_demo import knob_call, knob_is_up

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
        # Not a stop (2026-11-12, flatbot, twice: the button popped and the finder, which looks for a flat disc at
        # the lid's height, lost it once it stood up, its depth scattered 59 mm; both runs ended here with the box
        # open). Where it was pressed is known, and the fingers tell a knob from air: the run goes on to the grip.
        return KnobLook(False, status=status)
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
            "have nothing to hold, so the lid is not taken off; the box is left open, and the next run "
            "would take it for shut and press it (its aim judges a box by its knob alone)" % (1000 * dz, best)
        )
    runner.note(
        "reach_knob",
        press=press,
        found=True,
        looks=looks,
        knob_up_mm=None if best is None else round(best, 1),
        popped=look.popped,
        call=knob_call(best),  # the band's: "shut" is pressed again, "band" gripped as "open", "unread" stops
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


def solve_grip_way(ctx, cfg, button, tilt, q_from, lid_z, shifted=False):
    """The way from joints `q_from` (the look pose) down to the knob of the
    `button` the press was aimed at, at `tilt`: the press's way in
    (reach_press.reach_from_here's move to the look pose and slide, then
    walk_or_turn) with the grip's stroke (grip_descent onto the lid at
    `lid_z`) where the press's was — every line one the runner flies
    (depth_press.line_refusal). Returns (reach_press.WayIn, None) or (None,
    why). A hand that turns down leaning comes down grip_shift_m further out
    from the arm than `button` (`shifted`: this call is that one)."""
    from box_opening.runtime import approach

    chain = approach._chain()
    # the way in stands over the lid the grip goes down to: where the arm's touch put the lid higher than
    # the cameras put the button (flatbot 2026-11-12 18:42: 44 mm, the wrist's depth reading far), the way
    # in planned for the cameras' height ended 22 mm over the grip's end, too low to come down from
    button = (float(button[0]), float(button[1]), max(float(button[2]), float(lid_z)))
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
    way = rp.WayIn(move, plan, down, None if down.walked else plan.slide_pts, grip)
    out = 0.0 if shifted else grip_shift_m(down.quat)
    d = math.hypot(button[0], button[1])
    if out > 0.0 and d > 1e-6:
        # a leaning hand grips with the knob under its pads' low half (grip_shift_m): the same way in to a
        # point that far further out, where it flies and turns down as far; else this one, on the knob's centre
        further = (button[0] * (1.0 + out / d), button[1] * (1.0 + out / d), button[2])
        again, _why = solve_grip_way(ctx, cfg, further, tilt, q_from, lid_z, shifted=True)
        if again is not None and abs(dp._lean_deg(again.down.quat) - dp._lean_deg(down.quat)) < 0.5:
            return again, None
    return way, None


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
    lift_m: float = None  # how far straight up the lift goes (lift_off_m's, or its shorter fallback's)


def lift_clear_targets(ctx, R, t, top_z=None):
    """Where the clear after the lift goes where reach_press.clear_targets'
    does not do (_clear_after_lift), in the order they are tried: as high as
    the carry's floor (Place.hover_for's: the held lid's underside
    CARRY_CLEAR_M over the box's top — the cameras', or the lid's top as the
    grip took it, `top_z` (_gripped_lid_top), where that stands higher),
    never lower than the hand, and back toward the arm along the box's
    bearing CLEAR_BACK_M, then
    REACH_PULLBACK_M, as clear_targets'. Theirs is HOME_CLEAR_M +
    CLEAR_EXTRA_M over the box's padded cuboid (worlds.planned_top_z) and
    what the hand's lean hangs the planner's gripper spheres lower: set for
    a planner world holding the padded box, where flatbot's is empty
    (world_empty.yaml). Over the pink canister this one stands 76 mm over
    the box's top. On the 0.86 / 0.91 m side counters of the chair study
    (study/fixes/highcounter, 11-06) the clear up to theirs met joint_6's
    limit in all 80 rows whose lift flew and no clear did (the 150 mm one
    entering seat/armrest_arm_1 in 28); and where one did fly it ended with
    joint_6 0.026-0.108 rad from its limit, inside the planner's margin in
    the 10 rows whose carry it refused there. Up to the carry's floor
    instead joint_6 keeps ~0.2 rad: the carry and home plan from it. The
    cameras' top alone put it too low where the lid stands higher than they
    say: in the study's rows whose press met it 45 mm over their top, the
    carry from a clear that stood at the lift's top took the lid's lowest
    point 0.5-5.0 mm over the canister's rim; over the lid's top as gripped,
    7.4 mm at least (today's clear's least)."""
    m = ctx.model
    top = float(ctx.cpose.xyz[2]) + float(m.dims[2])
    if top_z is not None:
        top = max(top, float(top_z))
    floor = tcp_z(top + float(m.lid_dims[2]) + core.CARRY_CLEAR_M)
    z = max(float(t[2]), floor)
    bx, by = float(ctx.cpose.xyz[0]), float(ctx.cpose.xyz[1])
    d = math.hypot(bx, by)
    return [[float(t[0]) - b * bx / d, float(t[1]) - b * by / d, z] for b in (rp.CLEAR_BACK_M, rp.REACH_PULLBACK_M)]


def _gripped_lid_top(chain, q_grip):
    """The lid's top as the grip took it, the knob held at joints `q_grip`:
    GRIP_BELOW_LID_M over the closed pads' reference (grip_descent's end,
    press_contact.pads_z) — the lid's height the grip went by (grip_lid_z:
    the arm's own touch where the press held, else the press's top)."""
    from box_opening.tasks.press_contact import pads_z

    return pads_z(chain, q_grip) + GRIP_BELOW_LID_M


def _least_room(chain, q):
    """The least room any joint of joints `q` has to its URDF limit (rad);
    inf for a chain that names none."""
    rooms = [
        min(float(v) - lo, hi - float(v))
        for v, (lo, hi) in zip(q, getattr(chain, "limits", ()) or (), strict=False)
        if lo is not None and hi is not None
    ]
    return min(rooms) if rooms else math.inf


def _first_clear(ctx, chain, q, quat, targets):
    """The clear from joints `q` to the first of `targets` whose straight
    line flies and leaves every joint room for the planner to start from
    (reach_press._near_a_limit), else the first that flies, as plan_way_out
    picks: (points, how far back, None) or (None, 0, why)."""
    _R, t = chain.fk(q)
    chosen, whys = None, []
    for target in targets:
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
    return None, 0.0, "; ".join(whys)


def _clear_after_lift(ctx, chain, q, quat, top_z=None):
    """The clear from the lifted lid at joints `q`: up and back toward the
    arm to reach_press.clear_targets' (_first_clear), as the way home's
    clear goes — unless none of their lines flies, or the one that does
    ends inside the planner's limit margin (PLANNER_LIMIT_MARGIN_RAD: the
    carry from there is refused START_IN_LIMIT_MARGIN, and no lid comes
    off), where the clear up to the carry's floor (lift_clear_targets, over
    the box's top or `top_z`, the lid's as gripped, the higher) is flown
    that leaves more room. Where the lid comes off today, as on every lower
    table recorded, nothing changes. Returns (points, how far back, None) or
    (None, 0, why)."""
    R, t = chain.fk(q)
    R = np.asarray(R, float)
    pts, back, why = _first_clear(ctx, chain, q, quat, rp.clear_targets(ctx, R, t))
    if pts is not None and _least_room(chain, pts[-1]) >= PLANNER_LIMIT_MARGIN_RAD:
        return pts, back, None
    low, low_back, low_why = _first_clear(ctx, chain, q, quat, lift_clear_targets(ctx, R, t, top_z))
    if low is not None and (pts is None or _least_room(chain, low[-1]) > _least_room(chain, pts[-1])):
        return low, low_back, None
    if pts is not None:
        return pts, back, None
    return None, 0.0, "no way up and back from the lifted lid flies (%s; up to the carry's floor: %s)" % (why, low_why)


def _lean_keeps_clear(model, lean_deg, clear_m):
    """Whether the lid, turned `lean_deg` about the knob with the hand, still
    ends the lift with its lowest edge over the rim: that edge hangs the
    lid's radius times the lean's sine lower, of the `clear_m` the lift goes
    past the cap."""
    lid_r = 0.5 * max(float(model.lid_dims[0]), float(model.lid_dims[1]))
    return lid_r * math.sin(math.radians(lean_deg)) < float(clear_m)


def _lift_line(chain, q_grip, t, up, quat, lean_deg):
    """The lift's joint waypoints from the knob held at `q_grip` (the tool
    at `t`) to `up` straight above it: the hand at `quat` all the way, or,
    `lean_deg` not 0, leaning that much further back toward the arm by the
    top (reach_press.reach_quat's axis; kinematics.walk, in the straight
    line's own steps). Returns (points, the hand's attitude at the top,
    None) or (None, None, why): the solver's, or the runner's gate's
    (dp.line_refusal)."""
    from box_opening.runtime import kinematics

    top = [float(t[0]), float(t[1]), float(t[2]) + up]
    d = math.hypot(float(t[0]), float(t[1]))
    if not lean_deg or d < 1e-6:
        end, (pts, why) = quat, chain.straight_line(q_grip, top, quat)
    else:
        axis = (float(t[1]) / d, -float(t[0]) / d, 0.0)
        end = list(rp.rotate_about_world_axis(quat, axis, math.radians(lean_deg)))
        pts, why = chain.walk(q_grip, top, end, step_m=kinematics.STEP_M, max_step_rad=kinematics.MAX_STEP_RAD)
    if pts is not None:
        why = dp.line_refusal(chain, pts)
    if pts is None or why is not None:
        return None, None, why
    return [[float(v) for v in p] for p in pts], [float(v) for v in end], None


def plan_lift(ctx, cfg, q_grip, quat):
    """The lid's way off the box from the knob held at joints `q_grip`, the
    hand at `quat` (LiftOff): straight up lift_off_m, then the clear
    (_clear_after_lift) — where either does not fly, the same from the
    shorter lift LIFT_OFF_MIN_CLEAR_M past the cap; and where neither does
    with the hand as it holds the knob, the same two with the hand leaning
    further back as it rises (LIFT_LEAN_DEG, the least that flies; the
    clear and all after it at that attitude, LiftOff.quat). The grip's
    lookahead, while the hand comes down to the knob. Returns (LiftOff,
    None) or (None, why): the full lift's, the hand as it holds the knob."""
    from box_opening.runtime import approach

    chain = approach._chain()
    world = _full_world(ctx)
    q_grip = [float(v) for v in q_grip]
    _R, t = chain.fk(q_grip)
    top_z = _gripped_lid_top(chain, q_grip)  # the clear's floor never under the lid's top as gripped
    first = None
    for lean in (0.0, *LIFT_LEAN_DEG):
        tried = None
        for clear_m in (LIFT_OFF_CLEAR_M, LIFT_OFF_MIN_CLEAR_M):
            up = lift_off_m(ctx.model, clear_m)
            if tried is not None and up >= tried - 1e-9:
                break  # no shorter than the one that did not fly
            if lean and not _lean_keeps_clear(ctx.model, lean, clear_m):
                continue  # the lid's low edge would end at the rim or under it
            tried = up
            lift, top_quat, why = _lift_line(chain, q_grip, t, up, quat, lean)
            if lift is None:
                first = first or (up, "the lift off the canister is not a straight line the arm can fly (%s)" % why)
                continue
            clear, back, why = _clear_after_lift(ctx, chain, lift[-1], top_quat, top_z)
            if clear is None:
                first = first or (up, why)
                continue
            if first is not None:
                log.info(
                    "[press_demo] %.0f mm up: %s — the lift %.0f mm%s instead"
                    % (
                        1000 * first[0],
                        first[1],
                        1000 * up,
                        ", the hand leaning %.0f deg further back by its top" % lean if lean else "",
                    )
                )
            return LiftOff(
                [
                    dp.compliant_line(ctx, "lid:lift", lift, top_quat, cfg.lift_speed, world),
                    dp.compliant_line(ctx, "lid:clear", clear, top_quat, rp.free_line_speed(ctx), world),
                ],
                back,
                q_grip=q_grip,
                q_top=lift[-1],
                quat=top_quat,
                lift_m=up,
            ), None
    return None, first[1]


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
    over_m: float = None  # how far past where the plan has the lid meet the table the touch ends: SETDOWN_PAST_M
    k_z: float = None  # the hand's spring straight down there, N/m
    contact: object = None  # the set-down watched as it flies (watch_setdown): where the lid met the table


# The set-down watched for where the lid meets the table (watch_setdown, a
# SetdownContact: tasks/press_contact's PressContact with the set-down's own
# rules) from where its slow stretch begins, SETDOWN_SLOW_M above where the
# plan puts that, on the execution's last stretch below there (PressContact
# final=True: never the lift off a box that stands lower). Replayed on the 16
# recorded set-downs of 10-29 and 10-30, a watch over the fast way down (from
# 0.07 m) met one 88 mm early, from 0.05 m none. Cut where it meets
# (SETDOWN_FROM_CONTACT: the reference held where it stands when the contact
# is seen), and the release opens there: the first touch is the lid let go —
# on its rim if it leans, to settle flat (in 10 of the 16 the hand went on
# down 8-20 mm after it, the lid settling or the knob slipping in the
# fingers, the spring pressing on).
SETDOWN_WATCH_M = SETDOWN_SLOW_M
SETDOWN_FROM_CONTACT = True
# A watch that failed before it had the sent trajectory's heights finds where the slow stretch
# begins by the joints the way down ends at (SetdownContact._descent_from): the sent point nearest
# them, no further off than this on any joint (the set-down's lines are sent 2 mm a point, a few
# milliradians: 2026-11-17 16:40's trajectory)
SLOW_Q_TOL_RAD = 0.02


class SetdownContact(PressContact):
    """The set-down's watch: a PressContact on where the lid meets the table, by the set-down's
    own rules (SETDOWN_SETTLE_S, SETDOWN_CREDIBLE_M, SETDOWN_FREE_REF_M and SETDOWN_HELD_M's notes).
    The fingers open only where credible() says the lid met the table:

     - no rise is judged until `settle_s` after it armed (where the slow stretch begins): the arm
       is still catching up with its reference, which reads as a rise (the three "contacts" of
       2026-11-13 and 11-17);
     - a rise that began no more than `credible_m` over where the plan has the lid meet the table
       (`top_z`) is the table: the reference is stopped where it stands (with the cut on) and the
       lid may be let go;
     - where a rise began is where the pads last came freely down (`_free`, _began): what holds
       them up from there on is one thing, whichever rule sees it and however far they have sunk
       into it by then. One that began higher is not the table (`early` keeps it): the watch
       settles again and the descent goes on, slowly;
     - the pads held up are held up by something, settle or not — stalled (they came down next to
       nothing while the reference came on: how "held") or the gap risen `held_m` since they last
       came freely down (how "risen"), met where they stand as it is seen: the table where that and
       where the rise began lie within `credible_m` of the plan; else something else, and the
       descent is stopped there, the lid not let go (verdict "held");
     - the touch ended with nothing met (verdict "never"), or the watch failed (verdict "failed":
       the descent is stopped — where it failed once it watches; where its slow stretch begins
       where it failed before that, the lift and the carry flying on, __call__): the lid is not
       let go.

    `slow_q`: the joints the slow stretch begins at (the way down's end), for a watch that failed
    before it had the trajectory's heights (_descent_from). read_rest(), once the execution is
    over, reads what the client kept of its feedback and no poll read."""

    def __init__(
        self,
        chain,
        client,
        expect_z,
        arm_z,
        end_z,
        cut=True,
        credible_m=None,
        settle_s=None,
        held_m=None,
        slow_q=None,
    ):
        self.credible_m = SETDOWN_CREDIBLE_M if credible_m is None else float(credible_m)
        self.settle_s = SETDOWN_SETTLE_S if settle_s is None else float(settle_s)
        self.held_m = SETDOWN_HELD_M if held_m is None else float(held_m)
        self.slow_q = None if slow_q is None else [float(v) for v in slow_q]
        super().__init__(chain, client, expect_z, arm_z, 0.0, end_z, cut=cut, final=True, abort_on_failure=True)

    def _reset(self):
        super()._reset()
        self.early = []  # rises that began above where the table can be: (when, the pads there, which rise, when seen)
        self._judge_from = None  # the rises are judged from here on
        self._free = None  # the least gap since the pads last came freely down (its sample): where a rise began
        self._n_free = 0  # kept samples in a row that say the pads come freely down ...
        self._n_stall = 0  # ... that they have stalled ...
        self._n_risen = 0  # ... that the gap stands held_m over where they last came freely down
        self.began = None  # (when, the pads there, the gap) where the rise last met began (_began)
        self._noted = None  # when the rise last noted in `early` began: one thing is noted once
        self.failed_when = None  # a watch that failed: "watching" its stretch | "before" it | "after" the flight
        self.fail_from = None  # ... the fraction it stops the descent from (None: nowhere, the flight flies on)
        self.read_after = 0  # the kept samples read once the execution was over (read_rest)
        self.seen_after = False  # ... and what it says of the lid was only seen in those

    def _back(self, z_ref, by):
        """The newest kept sample at which the reference's pads stood `by` or more over `z_ref`:
        what the pads have done while the reference came down that far. None: it has not, since
        the watch armed."""
        for s in reversed(self._hist):
            if s[2] - s[1] >= z_ref + by:
                return s
        return None

    def _judge(self, t, frac, z_ref, z_meas):
        gap = z_meas - z_ref
        self.last_gap = gap
        if self.contact is not None or self.verdict is not None:
            return
        sample = (t, gap, z_meas)
        if self._judge_from is None:
            self._judge_from = t + self.settle_s
        in_a_row = self._rule()[2]
        # where a rise began: the least gap since the pads last came freely down — they kept the
        # reference's pace over the last SETDOWN_FREE_REF_M of its descent, so what had risen
        # before that is over (the arm's catching up, something that gave way) and the count
        # starts from that stretch's least
        was = self._back(z_ref, SETDOWN_FREE_REF_M)
        self._n_free = self._n_free + 1 if was is not None and gap - was[1] <= SETDOWN_FREE_RISE_M else 0
        if self._free is None or gap < self._free[1]:
            self._free = sample
        elif self._n_free >= in_a_row:
            least = sample
            for s in reversed(self._hist):
                least = s if s[1] < least[1] else least
                if s is was:
                    break
            if least[1] > self._free[1]:
                self._free = least
        # held up, whatever the settle says, by the pads' own descent: stalled while the reference
        # came on (which the catching up never shows: the pads come down through it at half the
        # reference's pace and more), or the gap risen held_m since they last came freely down
        was = self._back(z_ref, SETDOWN_STALL_REF_M)
        self._n_stall = self._n_stall + 1 if was is not None and was[2] - z_meas < SETDOWN_STALL_PADS_M else 0
        self._n_risen = self._n_risen + 1 if gap - self._free[1] > self.held_m else 0
        if self._n_stall >= in_a_row or self._n_risen >= in_a_row:
            self._hist.append(sample)
            # met where the pads stand as it is seen (what holds them stands there, or they have sunk that
            # far into it); whether that is the table is asked of where the rise began (_met)
            self._met(t, frac, sample, "held" if self._n_stall >= in_a_row else "risen")
        elif t < self._judge_from:
            # catching up with its reference: kept for where a rise began, judged by nothing yet
            self._hist.append(sample)
            while self._hist and t - self._hist[0][0] > HISTORY_S:
                self._hist.popleft()
        else:
            super()._judge(t, frac, z_ref, z_meas)

    def _began(self):
        """Where the rise the pads stand in began: (when, the pads there, the gap) of the last kept
        sample, from where they last came freely down on (`_free`), whose gap lay within
        CONTACT_ONSET_M of the gap there — PressContact._met's onset, counted from there and not
        from the least of the rule that saw the rise (the fast one's is the least of its last 0.2
        s: of a rise that goes on, wherever the pads have sunk to by then)."""
        on = self._free
        for s in self._hist:
            if s[0] >= self._free[0] and s[1] - self._free[1] <= CONTACT_ONSET_M:
                on = s
        return (on[0], on[2], on[1])

    def _met(self, t, frac, least, how):
        super()._met(t, frac, least, how)
        band = self.top_z + self.credible_m
        self.began = self._began()
        t_c, z_c, _gap = self.contact
        if max(z_c, self.began[1]) <= band:
            # the table: the rise began inside the band (the pads had come freely down to there) and that
            # is where the rule that saw it has it
            if self.cut:
                self.f_stop = frac  # stopped where it stands now: it has gone on past the touch since it began
            return
        if how in ("held", "risen"):
            # something holds the lid up that was met above where the table can be — there still, or the
            # pads sunk into it since (inside the band, even): stopped here, not let go
            self.verdict, self.f_stop = "held", frac
            return
        # a rise that began above where the table can be is no table: noted where it began (once, however
        # often the rules see it go on, and not where the rule that saw it has it, which may lie inside the
        # band by then), settled again, and the descent goes on. It is what holds the pads up until they come
        # freely down again: held up by it — stalled on it, or the gap risen held_m from where it began — the
        # descent stops
        if self.began[1] > band:
            t_c, z_c = self.began[0], self.began[1]
        if self._noted != self.began[0]:
            self.early.append((t_c, z_c, how, t))
            self._noted = self.began[0]
        self.contact = self.detected_t = self.how = self.stop_z = self.f_stop = None
        self._resettle(t)
        self._judge_from = t + self.settle_s

    def _descent_from(self):
        """The fraction of the sent trajectory's time from which a watch that failed stops the
        descent: 0 where it failed watching (stopped where it failed); else where the stretch it
        watches begins — its own arm_frac where it has the trajectory's heights, and where building
        those is what failed, the last of the sent points nearest the joints the slow stretch
        begins at (`slow_q`, within SLOW_Q_TOL_RAD: no kinematics). None: not known, or the
        trajectory never comes down to that stretch — nothing is stopped."""
        if self.armed_t is not None:
            return 0.0
        if self.traj is not None:
            return self.arm_frac
        if self.slow_q is None:
            return None
        try:
            pts = list(self.client.last_execution["sent_traj"].points)
            t = np.array([p.time_from_start.sec + 1e-9 * p.time_from_start.nanosec for p in pts], float)
            off = np.array([list(p.positions) for p in pts], float) - np.asarray(self.slow_q, float)
            far = np.abs(np.remainder(off + math.pi, 2.0 * math.pi) - math.pi).max(axis=1)
            i = len(far) - 1 - int(np.argmin(far[::-1]))
            return float(t[i] / t[-1]) if far[i] <= SLOW_Q_TOL_RAD and t[-1] > 0.0 else None
        except Exception:  # noqa: BLE001 -- not known: nothing is stopped, and the lid is not let go
            return None

    def __call__(self, progress):
        """stop_when: True once the reference stands where the descent ends — where the lid met the
        table, or where something held it up. A watch that fails stops the descent, never the lift
        or the carry before it (it is the stop_when of the whole execution, polled from its first
        sample: a failure there used to cancel the flight where it was, the lid half lifted or in
        mid-carry, and the way back from there is no line the run checked): from where the stretch
        it watches begins (_descent_from), the reference not gone slowly down onto a table nothing
        watches for. Where that is not known the flight flies its planned end, as the set-down's
        watch did before 2026-11-17. Either way the lid is not let go (verdict "failed")."""
        if self.failed is None:
            try:
                self._feed_new()
            except Exception as e:  # noqa: BLE001 -- said; the descent stops, the lid is not let go
                self.failed = "%s: %s" % (type(e).__name__, e)
                self.verdict, self.f_stop = "failed", None
                self.failed_when = "watching" if self.armed_t is not None else "before"
                self.fail_from = self._descent_from()
                log.info(
                    "[press_demo] the set-down's watch failed (%s) — %s; the lid will not be let go"
                    % (
                        self.failed,
                        "the descent stops here"
                        if self.failed_when == "watching"
                        else "the descent will be stopped where its slow stretch begins"
                        if self.fail_from is not None
                        else "where its slow stretch begins is not known: the flight flies its planned end",
                    )
                )
        if self.failed is not None:
            if self.stopped_frac is None and self.fail_from is not None and progress >= self.fail_from:
                self.stopped_frac = float(progress)
                self.held_ref = None if self.traj is None else [float(v) for v in self.traj.at(progress)]
            return self.stopped_frac is not None
        if self.f_stop is None or progress < self.f_stop:
            return False
        self.stopped_frac = float(progress)
        self.held_ref = [float(v) for v in self.traj.at(progress)]
        return True

    def read_rest(self):
        """The execution over: what the client kept of its feedback and no poll read is read now —
        the last samples before its result came, or all of them where the watch was not polled
        while they came (client.execute polls a stop_when only once what the goal hosts has run:
        the set-down hosts nothing, choose_and_carry, but nothing here relies on that). Nothing is
        stopped by it any more; it says only whether the lid met the table. Until 2026-11-17's
        review the release asked credible() of a watch that had read none of them, and the run
        said "the set-down was not watched (no feedback of its last stretch)" of feedback that
        was kept. Returns how many it read."""
        before, had = self.samples, self.contact is not None or self.verdict is not None
        if self._sent is not None and getattr(self.client, "last_execution", None) is not self._sent:
            return 0  # another execution has gone out since the one it watched: nothing of that is left to read
        if self.failed is None:
            try:
                self._feed_new()
            except Exception as e:  # noqa: BLE001 -- said in its record; the lid is not let go
                self.failed = "%s: %s" % (type(e).__name__, e)
                self.verdict, self.f_stop, self.failed_when = "failed", None, "after"
        self.read_after = self.samples - before
        self.seen_after = not had and (self.contact is not None or self.verdict is not None)
        return self.read_after

    def credible(self):
        """Whether the lid met the table, at about the height the run has for it: the one case the
        fingers open in."""
        return self.contact is not None and self.verdict is None and self.failed is None

    def refusal(self):
        """Why the lid must not be let go here, once the set-down's execution is over: (verdict,
        the words), or None where it met the table (credible)."""
        above = ", ".join("%+.0f" % (1000 * (z - self.top_z)) for _t, z, _how, _seen in self.early)
        rises = "" if not self.early else " (rises %s mm over it were not taken for the table)" % above
        if self.verdict == "failed":
            if self.failed_when == "after":
                return "failed", (
                    "the set-down's watch failed (%s) reading what was kept of the flight once it was over: nothing "
                    "says the lid met the table" % self.failed
                )
            if self.stopped_frac is None:
                return "failed", (
                    "the set-down's watch failed (%s) before the descent, and where that begins was not known: the "
                    "touch flew to its planned end with nothing watching for the table" % self.failed
                )
            return "failed", "the set-down's watch failed (%s) and the descent was stopped %s" % (
                self.failed,
                "there" if self.failed_when == "watching" else "where its slow stretch begins",
            )
        if self.verdict == "held":
            stands = 1000 * (self.contact[1] - self.top_z)
            if stands > 1000 * self.credible_m or self.began is None:
                return "held", (
                    "something held the lid up %.0f mm over where the plan has it meet the table (more than %.0f): "
                    "that is not the table, and the descent was stopped there%s"
                    % (stands, 1000 * self.credible_m, rises)
                )
            # the pads have sunk into what they met to inside the band: it is still what they met above it
            return "held", (
                "something met %.0f mm over where the plan has the lid meet the table (more than %.0f) still held it "
                "up, %.0f mm over it: that is not the table, and the descent was stopped there%s"
                % (1000 * (self.began[1] - self.top_z), 1000 * self.credible_m, stands, rises)
            )
        if self.contact is None:
            if self.armed_t is None:
                return "never", (
                    "the set-down was not watched (no feedback of its last stretch): nothing says the lid met the "
                    "table"
                )
            return "never", (
                "the touch ended %.0f mm under where the plan has the lid meet the table and the arm felt no table%s"
                % (1000 * (self.top_z - self.end_z), rises)
            )
        return None

    def record(self):
        """PressContact.record's, and the set-down's own: whether the contact is one the fingers
        open on (`credible`), the band, the settle and the held-up rules it was judged by (`held_mm`
        the gap's rise; `stall_mm` the reference's descent and the most the pads came down in it;
        `free_mm` the reference's descent and the most the gap rose in it where the pads came
        freely down), the rises that began above the band and were not taken for the table
        (`early`: when after the goal went out, how far over the plan, which rise), where a rise
        that was met began (`began_above_mm`: where the pads last came freely down before it), and
        how many of its samples were read only once the execution was over
        (`read_after_flight`, read_rest: the last few, or all of them where no poll came while it
        flew — `seen_after_flight` where what it says of the lid was only seen in those, nothing
        cut)."""
        out = super().record()
        sent = None if self._sent is None else self._sent.get("sent_at")
        refused = self.refusal()
        if refused is not None:
            out["why"] = refused[1]
        out.update(
            credible=self.credible(),
            credible_above_mm=round(1000 * self.credible_m, 1),
            settle_s=self.settle_s,
            held_mm=round(1000 * self.held_m, 1),
            stall_mm=[round(1000 * SETDOWN_STALL_REF_M, 1), round(1000 * SETDOWN_STALL_PADS_M, 1)],
            free_mm=[round(1000 * SETDOWN_FREE_REF_M, 1), round(1000 * SETDOWN_FREE_RISE_M, 1)],
            early=[
                [None if sent is None else round(t_c - float(sent), 3), round(1000 * (z - self.top_z), 1), how]
                for t_c, z, how, _seen in self.early
            ],
            read_after_flight=self.read_after,
            seen_after_flight=self.seen_after,
        )
        if self.contact is not None and self.began is not None:
            out["began_above_mm"] = round(1000 * (self.began[1] - self.top_z), 1)
        return out


def watch_setdown(ctx, legs, carry):
    """The set-down of `carry` (plan_carry's) watched as it flies in the
    execution `legs` begins (the first motion leg's stop_when): a
    SetdownContact on where the lid meets the table — SETDOWN_WATCH_M above
    where the plan puts that on, the execution's last stretch below there —
    cut where it meets with SETDOWN_FROM_CONTACT. Kept on the carry
    (carry.contact): the release asks it whether the lid met the table
    (credible), and the reach_setdown row keeps its record. It is told the
    joints the slow stretch begins at (the way down's end): a watch that
    fails stops the descent from there, never the lift or the carry
    (SetdownContact.__call__). Returns it, or None: nothing to watch."""
    from box_opening.runtime import approach
    from box_opening.runtime.legs import Kind
    from box_opening.tasks.press_contact import pads_z

    lead = next((lg for lg in legs if lg.kind is Kind.MOTION), None)
    touch = carry.legs[-1] if carry.legs else None
    if lead is None or touch is None or touch.goal_joints is None:
        return None
    chain = approach._chain()
    end = pads_z(chain, touch.goal_joints)
    over = float(carry.over_m or 0.0)
    expect = end + over  # where the plan has the lid meet the table
    down = carry.legs[-2] if len(carry.legs) >= 2 else None  # 'place:down': it ends where the slow stretch begins
    carry.contact = SetdownContact(
        chain,
        ctx.client,
        expect,
        expect + SETDOWN_WATCH_M,
        end,
        cut=SETDOWN_FROM_CONTACT,
        slow_q=getattr(down, "goal_joints", None),
    )
    lead.stop_when = carry.contact
    return carry.contact


# The carry the planner refuses for where it starts or ends, inside its limit
# margin (PLANNER_LIMIT_MARGIN_RAD), flies as the arm's own walk instead
# (walk_carry): to the same hover at the same attitude, every step solved
# from the last, CARRY_WALK_STEP_M / CARRY_WALK_STEP_RAD apart, no joint
# jumping more than kinematics.MAX_STEP_RAD between two — held out of the
# chair and its user by the walk itself (runtime/body), in the runner's
# trajectory gate (depth_press.line_refusal), at the free-air lines' speed.
# The chair study's 0.86 / 0.91 m side counters (study/fixes/highcounter,
# 11-06): with the clear to the carry's floor and the leaning set-down, the
# planner refused the carry there in 58 of the 158 rows whose lid came up,
# the joint_6 0.026-0.085 rad from its limit; walked, the lid came off in
# all 158 (100 without), its way never nearer joint_6's limit than 0.021
# rad (the walk's own bound: kinematics.LIMIT_MARGIN_RAD). Like every
# plan-free line it is also held out of the cabinet below the counter's
# edges this run's maps showed (press_demo.keep_out_counter's boxes are in
# the walk's own body check); the study's numbers were measured without them.
CARRY_WALK_ON = ("START_IN_LIMIT_MARGIN", "GOAL_IN_LIMIT_MARGIN")
CARRY_WALK_STEP_M = 0.01
CARRY_WALK_STEP_RAD = math.radians(2.0)


def walk_carry(ctx, chain, q_from, hover, quat, world):
    """The carry from joints `q_from` to the tool at `hover`, `quat`, as the
    arm's own walk (CARRY_WALK_ON's), a 'lid:carry' compliant line at
    reach_press.free_line_speed: (Leg, None) or (None, why)."""
    from box_opening.runtime import kinematics as kin

    pts, why = chain.walk(
        q_from,
        hover,
        quat,
        step_m=CARRY_WALK_STEP_M,
        step_rad=CARRY_WALK_STEP_RAD,
        max_step_rad=kin.MAX_STEP_RAD,
    )
    if pts is None:
        return None, why
    pts = [[float(v) for v in p] for p in pts]
    why = dp.line_refusal(chain, pts)
    if why is not None:
        return None, why
    return dp.compliant_line(ctx, "lid:carry", pts, quat, rp.free_line_speed(ctx), world), None


def plan_carry(ctx, cfg, q_from, drop, held=None, over_box=False, way=None):
    """The carry from joints `q_from` (where the spot look stands) to the
    spot `drop` (planned in the planner's own world, worlds.PLANNER_WORLD:
    it is told nothing of the box or the table; where it refuses the start
    or the goal for its limit margin, the arm's own walk: walk_carry) —
    then the set-down's straight lines from its end (the way down to
    SETDOWN_SLOW_M over where the plan has the lid meet the table, the touch
    on to SETDOWN_PAST_M under it) and the retreat's back up. `held`: the
    joints the lid hangs level at, where those are not `q_from`
    (drop_reachable's). `over_box`, and always for a hand leaning past
    SETDOWN_LEAN_DEG: a carry that would bring the lid too low over the
    canister is refused (lid_low_over_box).

    `way` (a LaterWay: a spot of the wider search, and every spot after the
    first): the carry is the arm's own walk first — a straight line from
    hover to hover, at their height — and the planner's only where no walk
    flies; whichever it is, its own path is asked of the canister and of
    `way.path` (the carry's height all the way, and what the map shows
    standing under it: the planner knows no table, and a joint-space path
    between two hovers can sink a hand's width under them), and a leaning
    set-down's way down of `way.descent`. A planned carry refused for the
    canister with no `way` is flown as the walk where that clears it, before
    the spot is refused. Returns (Carry, None) or (None, why)."""
    from box_opening.runtime import approach
    from box_opening.runtime import kinematics as kin

    chain = approach._chain()
    world = _full_world(ctx)
    q_from = [float(v) for v in q_from]
    held = q_from if held is None else [float(v) for v in held]
    quat, contact, hover = setdown_poses(ctx, drop, held=held)
    asked = over_box or way is not None or _spot_lean(drop) > SETDOWN_LEAN_DEG

    def too_low(leg):
        """Why `leg`'s own path is no way for the lid here (the canister; `way`'s height and what stands), or None."""
        if not asked or leg.traj is None:
            return None
        pts = [list(p.positions) for p in leg.traj.points]
        why = lid_low_over_box(ctx, pts, held)
        return why if why is not None or way is None else way.path(pts)

    def walk():
        leg, why = walk_carry(ctx, chain, q_from, hover, quat, world)
        if leg is not None:
            why = too_low(leg)
        return (leg if why is None else None), why

    carry, walked, no_walk = None, False, None
    if way is not None:
        carry, no_walk = walk()
        walked = carry is not None
    if carry is None:
        try:
            carry, _st = rp._plan_motion(
                ctx, PlanState(joints=q_from, chain=0), "lid:carry", ("pose", hover, quat), world, TRANSIT_SPEED
            )
        except RuntimeError as e:
            if way is not None:
                return None, "the carry (walked: %s; planned: %s)" % (no_walk, rp._short(e))
            if not any(code in str(e) for code in CARRY_WALK_ON):
                return None, "the carry (%s)" % rp._short(e)
            carry, why = walk()
            if carry is None:
                return None, "the carry (%s; walked: %s)" % (rp._short(e), why)
            walked = True
            log.info("[press_demo] the planner refused the carry (%s): the arm's own walk instead" % rp._short(e))
    if carry.traj is not None and not walked:
        carry.traj = approach.on_branch_of(carry.traj, q_from)  # continues the clear, not a turn away
        # the runner's sanity gate, here where another way can still be taken: it gates every leg of an
        # execution before anything moves, so a carry that swings wide refused the close, the lift and the
        # set-down merged with it (flatbot 2026-11-12 18:42: arm_joint_2 0.444 rad against 0.444 allowed; the
        # run read the refusal as a grip that missed and never closed on a knob that was up). The arm's own
        # walk there instead, as for a start the planner refuses; a spot neither reaches is passed over
        bad = sanity_violations(carry.traj, SANITY_MARGIN_RAD)
        if bad:
            swung = "the planner's carry swings wide (%s)" % "; ".join(bad)
            if way is not None:
                return None, "the carry (walked: %s; planned: %s)" % (no_walk, swung)
            carry, why = walk()
            if carry is None:
                return None, "the carry (%s; walked: %s)" % (swung, why)
            walked = True
            log.info("[press_demo] %s: the arm's own walk instead" % swung)
    if carry.traj is not None and not walked:
        carry.goal_joints = [float(v) for v in carry.traj.points[-1].positions]
        # the planner's world may be empty (worlds.PLANNER_WORLD): the carry is held out of the
        # chair and its user as the plan-free lines are (runtime/body), and a spot whose carry
        # enters them is passed over for the next
        why = kin.body_refusal(chain, [list(p.positions) for p in carry.traj.points])
        if why is None:
            # ... and, where it is asked, off the canister, at its height and over nothing that stands
            # (from one spot to the next the way may cross the box and the table, and a hand leaning past
            # SETDOWN_LEAN_DEG hangs the lid's low edge lower than any carry flown)
            low = too_low(carry)
            if low is not None and way is None:
                # the walk before the spot is refused: on 2026-11-17 16:40's map the joint-space way to every
                # further spot brought the lid under the canister's top, and the walk to each cleared it
                carry, why = walk()
                if carry is None:
                    return None, "the carry (%s; walked: %s)" % (low, why)
                walked = True
                log.info(
                    "[press_demo] the planner's carry is no way for the lid (%s): the arm's own walk instead" % low
                )
            else:
                why = low
        if why is not None and not walked:
            if way is not None:
                return None, "the carry (walked: %s; planned: %s)" % (no_walk, why)
            return None, "the carry (%s)" % why
    if way is not None:
        log.info(
            "[press_demo] the lid's way to [%.2f, %.2f] is %s"
            % (
                drop.xyz[0],
                drop.xyz[1],
                "the arm's own walk, a straight line at the height it is carried at"
                if walked
                else "the planner's (no walk flies: %s): it keeps the carry's height, and nothing stands under it"
                % no_walk,
            )
        )
    carry.gains = dp.COMPLIANT_GAINS
    lines = []
    q = carry.goal_joints
    k_z = None
    z_at = {"place:down": contact[2] + SETDOWN_SLOW_M, "place:touch": contact[2] - SETDOWN_PAST_M}
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
    if way is not None:
        # a leaning hand's own body, down the lines it flies, and what stands on the table (LaterWay.descent)
        why = way.descent(drop, lines[0] + lines[1])
        if why is not None:
            return None, "the set-down (%s)" % why
    down_speed = max(float(cfg.setdown_speed), rp.free_line_speed(ctx))
    legs = [
        carry,
        dp.compliant_line(ctx, "place:down", lines[0], quat, down_speed, world),
        dp.compliant_line(ctx, "place:touch", lines[1], quat, PLACE_TOUCH_SPEED, world),
    ]
    return Carry(legs, lines[2], quat, drop, over_m=SETDOWN_PAST_M, k_z=k_z), None


def carry_refusal(carry):
    """Why `carry` (plan_carry's) would go into the body check's boxes as they stand now — the chair
    and its user, and the cabinet below every counter edge a map of this run showed
    (press_demo.keep_out_counter) — or None: its planned leg's goal, strictly, and its path, its
    set-down's lines and its retreat, each as plan_carry checked them when it was made. The carry
    carry_ahead planned while the lift flew was held out of the boxes in force then; the lid's own
    map, read after the lift, may show an edge they did not (choose_and_carry)."""
    from box_opening.runtime import approach
    from box_opening.runtime import kinematics as kin

    chain = approach._chain()
    for leg in carry.legs:
        traj = getattr(leg, "traj", None)
        if traj is None or not traj.points:
            continue
        pts = [[float(v) for v in p.positions] for p in traj.points]
        target = getattr(leg, "target", None)
        planned = target is not None and not getattr(leg, "plan_free", False)  # a plan-free line has no goal sent
        why = approach.goal_refusal(target, pts[0], pts[-1]) if planned else None
        if why is None:
            why = kin.body_refusal(chain, pts)
        if why is not None:
            return "%s (%s)" % (leg.name, why)
    why = kin.body_refusal(chain, carry.retreat) if carry.retreat else None
    return None if why is None else "place:retreat (%s)" % why


def plan_home_after(ctx, cfg, carry):
    """The retreat back up from the set-down and home planned from its
    end, with the lid lying at the spot (ctx.lid_at, for this plan only):
    the legs after the release, home under its own gains (MEDIUM), or
    (None, why). Planned with the carry while the hand comes down to the
    knob (CarrySearch.finish, the grip's lookahead) — never while the
    set-down flies: that execution hosts nothing (choose_and_carry)."""
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
    look pose where the way in did not start (from over the box, the clear
    first, where its free air flew in position and only the stroke did not
    start: rp.fly_way_in), the arm let go and held where
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
    # where the pads meet the box ends every press's stroke past it (review round 2: the second
    # too), and a contact well off the cameras' top refuses it (dp.watch_press)
    # (its free air in position control, rp.air_position: two executions, and `names` the last's)
    res, contact, names, over_box = rp.fly_way_in(ctx, runner, legs, way.stroke, top_z)
    last = res[-1]
    flown = all(r.ok for r in res)
    retimed = rp._note_flight(runner, "reach_way_in_flight", names, last) if last.flown_traj is not None else {}
    refused = dp.refuse_press(ctx, runner, contact, res)
    if refused is not None:
        # stopped where it was judged, the springs let go: back to the look pose, then home
        code = _way_back(
            ctx, cfg, runner, way, ctx.client.joints(), dp.DepthPress(free=True, why=refused, judged=False)
        )
        return (code if code is not None else _stop_home(ctx, cfg, runner, refused)), None
    if not flown and not _started(last) and over_box:
        # Its free air flew in position and arrived, and the stroke's own execution did not start
        # (rp.fly_way_in): the hand stands turned down at the stroke's start, 35-60 mm over the lid, not at
        # the look pose. It leaves as every other stop over the box does, up and back clear of it before
        # home is planned (rp._home_from_over_box), nothing pressed. An arm that does not get home says so
        # last: the STOP before it reads as a run a retry may follow
        why = "the press stroke did not start (%s) — nothing was pressed%s" % (
            last.detail or last.outcome,
            "" if press == 1 else " again",
        )
        # (at rest first: the clear is sent the instant the stroke was refused, and a driver that refused
        # the stroke for an arm still moving would refuse that too)
        runner.wait_still("the clear, after a stroke that did not start")
        if not rp._home_from_over_box(ctx, cfg, runner, dp.DepthPress(free=True, why=why, judged=False)):
            rp.let_go(ctx)
            log.info("[press_demo] STOP: the arm did not get home — it holds; home it with sheppy's home_arm")
        return 1, None
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
    up onto the stroke's line, under the exit's looser gate
    (kinematics.EXIT_MAX_STEP_RAD: its steps past MAX_STEP_RAD fly slow,
    depth_press.exit_line), then down it to where the pads stand
    GRIP_LID_ABOVE_HELD_M - GRIP_BELOW_LID_M over where the hold held them
    (grip_lid_z), each as joint points — or (None, None, why): the first
    refusal's. Where that rise or its way down does not fly, the same with
    the hand leaning LIFT_LEAN_DEG further back by the top (and back to the
    stroke's attitude on the way down), then GRIP_RISE_MIN_M each way. A
    leaning hand's way down ends grip_shift_m further out from the arm than
    the stroke's line; where none of those flies, on the line itself."""
    from box_opening.runtime import approach
    from box_opening.runtime.kinematics import EXIT_MAX_STEP_RAD, MAX_STEP_RAD, STEP_M

    chain = approach._chain()
    q_held = [float(v) for v in q_held]
    _R, t_held = chain.fk(q_held)
    _R, t_line = chain.fk([float(v) for v in way.stroke.q0])  # the stroke's line: the aimed knob
    quat = way.stroke.quat
    d = math.hypot(float(t_line[0]), float(t_line[1]))
    first = None
    # a leaning hand grips with the knob under its pads' low half: the way down ends that far out (grip_shift_m)
    shift = grip_shift_m(quat) if d > 1e-6 else 0.0
    for up, out in [(up, o) for o in dict.fromkeys((shift, 0.0)) for up in (GRIP_RISE_M, GRIP_RISE_MIN_M)]:
        top = [float(t_line[0]), float(t_line[1]), float(t_held[2]) + up]
        bottom = [
            top[0] + (out * float(t_line[0]) / d if out else 0.0),
            top[1] + (out * float(t_line[1]) / d if out else 0.0),
            float(t_held[2]) + GRIP_LID_ABOVE_HELD_M - GRIP_BELOW_LID_M,
        ]
        for lean in (0.0, *LIFT_LEAN_DEG) if d > 1e-6 else (0.0,):
            if not lean:
                rise, why = chain.straight_line(q_held, top, quat, max_step_rad=EXIT_MAX_STEP_RAD)
            else:  # the hand leaning further back toward the arm by the top, as the lift's (_lift_line)
                axis = (float(t_line[1]) / d, -float(t_line[0]) / d, 0.0)
                leaned = list(rp.rotate_about_world_axis(quat, axis, math.radians(lean)))
                rise, why = chain.walk(q_held, top, leaned, step_m=STEP_M, max_step_rad=EXIT_MAX_STEP_RAD)
            if rise is not None:
                why = dp.line_refusal(chain, rise)
            if rise is None or why is not None:
                first = first or "the rise off the button (%s)" % why
                continue
            if not lean:
                down, why = chain.straight_line(rise[-1], bottom, quat)
            else:  # back to the stroke's attitude on the way down
                down, why = chain.walk(rise[-1], bottom, quat, step_m=STEP_M, max_step_rad=MAX_STEP_RAD)
            if down is not None:
                why = dp.line_refusal(chain, down)
            if down is None or why is not None:
                first = first or "the way down to the knob (%s)" % why
                continue
            if first is not None:
                log.info(
                    "[press_demo] %s — the rise %.0f mm%s instead"
                    % (first, 1000 * up, ", the hand leaning %.0f deg further back by its top" % lean if lean else "")
                )
            return [[float(v) for v in p] for p in rise], [[float(v) for v in p] for p in down], None
    return None, None, first


def _grip_from_hold(ctx, cfg, runner, eyes, way, seen, room):
    """The grip straight from the first press's hold (GRIP_FROM_PRESS): the
    fingers opening to the knob's width as the hand rises off the button,
    once the pads are GRIP_OPEN_ABOVE_HELD_M over where the hold held them
    (else at the rise's top), the lift and the first spot the hand reaches planned while it does
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
    from box_opening.runtime import approach
    from box_opening.tasks.press_contact import PadsAbove, PressContact, pads_z

    chain = approach._chain()
    top_z, end_z = pads_z(chain, rise[-1]), pads_z(chain, down[-1])
    log.info(
        "[press_demo] GRIP STRAIGHT FROM THE PRESS: up %.0f mm, the fingers opening round the knob once the pads "
        "are %.1f mm over the hold (else at the top), then down to the knob — to where the open fingertips meet the "
        "lid, or the closed pads %.0f mm under the lid at %.3f (%.0f mm over where the press held them)"
        % (
            1000 * (top_z - pads_z(chain, seen["held_joints"])),
            1000 * GRIP_OPEN_ABOVE_HELD_M,
            1000 * GRIP_BELOW_LID_M,
            lid_z,
            1000 * GRIP_LID_ABOVE_HELD_M,
        )
    )
    _video(ctx, "GRIP straight from the press: up, the fingers open, down to the knob")

    opened = _gripper_leg(
        ctx,
        PlanState(joints=None, chain=0),
        "grip:open",
        grip_open_cmd(ctx.model),
        world,
        defer_join=True,
        send_with_previous_motion=True,
    )
    # sent under the rise once the pads have left the knob, as the arm measures them; never so, at its top
    off = PadsAbove(chain, ctx.client, held_z + GRIP_OPEN_ABOVE_HELD_M)
    opened.send_when = off
    # the rise ends at the hand's attitude at its top: the stroke's, or leaning further back (grip_from_hold_lines)
    quat_top = [float(v) for v in dp.mat_to_quat_xyzw(np.asarray(chain.fk(rise[-1])[0], float))]
    up = dp.exit_line(ctx, "grip:rise", rise, quat_top, rp.free_line_speed(ctx), world)
    lower = dp.compliant_line(ctx, "grip:down", down, quat, dp.TOUCH_SPEED, world)
    touch = None
    if GRIP_TOUCH:
        # stopped where the open fingertips meet the lid (or a knob), nothing pressed on down past it
        touch = PressContact(chain, ctx.client, end_z, top_z - GRIP_TOUCH_ARM_M, 0.0, end_z, cut=True)
        lower.stop_when = touch
    # the lookahead split over both (grip_ahead_rise, grip_ahead_down): the lift and the
    # first spot while the hand rises, the carry and home while it comes down
    res = runner.run(
        [up, opened],
        execute=True,
        lookahead=lambda _end: grip_ahead_rise(ctx, cfg, eyes, lower.goal_joints, quat, room),
    )
    first, rise_stood = runner.lookahead_result, runner.lookahead_stood_s
    runner.note(
        "grip_open",
        above_held_mm=round(1000 * GRIP_OPEN_ABOVE_HELD_M, 1),
        under_rise=off.at is not None,  # sent as the hand rose, the pads off the knob; else at the rise's top
        at_s=None if off.at is None else off.at[0],
        pads_z=None if off.at is None else off.at[1],
        highest_mm=None if off.highest is None else round(1000 * (off.highest - held_z), 1),
    )
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
    # the hand comes down only from where it cleared the knob, as the arm measured its pads at the top
    q_top = [float(v) for v in ctx.client.joints()]
    rose = max(pads_z(chain, q_top), off.highest if off.highest is not None else -math.inf) - held_z
    if rose < GRIP_RISE_LEAST_M:
        runner.note("reach_grip_refused", attempt=0, why="rose %.1f mm" % (1000 * rose), from_press=True)
        log.info(
            "[press_demo] the hand rose only %.0f mm off the hold, not the %.0f mm that clears a popped knob: not "
            "coming down on it — back to the look pose to look at the knob" % (1000 * rose, 1000 * GRIP_RISE_LEAST_M)
        )
        return _MISSED if _way_back(ctx, cfg, runner, way, q_top, dp.DepthPress(free=True)) is None else 1
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
    if touch is not None and touch.contact is not None:
        ahead = _grip_at_touch(ctx, runner, chain, touch, quat, held_z, world)

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


def _grip_at_touch(ctx, runner, chain, touch, quat, held_z, world):
    """The way down to the knob ended where the open fingertips met something (`touch`, the way down's
    PressContact): the hand back up touch_rise_m from where it stands, for the fingers to close round
    the knob's base and not into the lid — where that short line does not fly, they close where it
    stands, said. Returns None: the lift planned ahead started from the way down's planned end, lower
    than this, and is not flown; the close plans its own from where the hand is (_close_and_carry)."""
    from box_opening.tasks.press_contact import pads_z

    q = [float(v) for v in ctx.client.joints()]
    up = touch_rise_m(ctx.model)
    met = float(touch.contact[1])
    runner.note(
        "grip_touch",
        met_z=round(met, 4),
        over_held_mm=round(1000 * (met - held_z), 1),
        how=touch.how,
        pads_z=round(pads_z(chain, q), 4),
        rise_mm=round(1000 * up, 1),
    )
    log.info(
        "[press_demo] GRIP: the open fingertips met the lid with the pads at %.3f, %.0f mm over where the press held "
        "them — up %.1f mm for the pads to close round the knob's base" % (met, 1000 * (met - held_z), 1000 * up)
    )
    _R, t = chain.fk(q)
    pts, why = chain.straight_line(q, [float(t[0]), float(t[1]), float(t[2]) + up], quat)
    if pts is not None:
        why = dp.line_refusal(chain, pts)
    if pts is None or why is not None:
        log.info("[press_demo] GRIP: no way up off the touch (%s): the fingers close where the hand stands" % why)
        return None
    res = runner.run(
        [dp.compliant_line(ctx, "grip:settle", [[float(v) for v in p] for p in pts], quat, dp.TOUCH_SPEED, world)],
        execute=True,
    )
    if not all(r.ok for r in res):
        log.info(
            "[press_demo] GRIP: the way up off the touch did not fly (%s): the fingers close where the hand stands"
            % (res[-1].detail or res[-1].outcome)
        )
    return None


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
    press once more only where the knob reads flush (KnobLook.shut) and the
    lid did not rise (risen with its knob flush: knob_look's STOP), then the lid off
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
            if not knob.found:
                log.info(
                    "[press_demo] the knob is not judged after press %d (the wrist does not find the button): going "
                    "on to the grip where the press was aimed — the fingers close on a knob that is up, or on air"
                    % press
                )
                break
            if knob.popped:
                break
            if knob.up_mm is None:
                return end(
                    _stop_home(
                        ctx,
                        cfg,
                        runner,
                        "the wrist read no height for the knob after press %d — it is not pressed again, and the "
                        "lid is not taken off" % press,
                    )
                )
            if not knob.shut:
                # neither flush nor up: a popped knob reads here too, and pressing it latches the box again
                log.info(
                    "[press_demo] the knob is neither flush nor up (%.1f mm above the lid): not pressing it again, "
                    "which would shut an open box — taking the lid off as an open box's" % knob.up_mm
                )
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
    legs = rp.way_in_legs(ctx, cfg, way, air=False)  # one motion on the spring, however the press's way in flies
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
    if one:
        # the runner gates every leg before anything moves: a carry it would refuse is not flown merged with
        # the close and the lift (nothing would close), the lift goes alone and the carry is chosen after it
        gate = runner.refusal(placed.carry.legs)
        if gate is not None:
            log.info(
                "[press_demo] the carry planned ahead would be refused (%s: %s): the lift alone, then the spot "
                "look" % gate
            )
            one = False
    gate = runner.refusal(lift.legs)
    if gate is not None:
        return back_out("the lid has no way off the box (%s: %s) — not gripping it" % gate, opened=False)
    carried = None
    search = None
    ctx.lid_lifted = lift  # the lid's way back onto the box, for a STOP before the carry (put_lid_back)
    if one:
        carry, drop = placed.carry, placed.drop
        flying = [close, *lift.legs, *carry.legs]
        # where this set-down feels no table the lid goes on to another spot, chosen on the map this one was
        search = SpotSearch(
            ctx,
            cfg,
            eyes,
            runner,
            placed.tmap,
            placed.table_z,
            placed.spot_z,
            cache=None if placed.room is None else placed.room.cache,
            held=placed.q,
        )
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
            no_land=no_land_note(ctx, drop),
            table=table_note(ctx, placed.table_z, placed.spot_z, placed.passed),
            pooled=pooled_note(ctx, runner, placed.room, drop),  # the wrist's look-pose frame pooled in: shadow
            spot=1,  # the run's first spot for the lid (a set-down that feels no table goes on to a second)
        )
        _say_spot(ctx, eyes, (drop, placed.tier, placed.says), placed.refused, placed.tmap, placed.table_z)
        log.info(
            "[press_demo] CLOSE, then ONE MOTION: %s — %.0f mm straight up, %.0f mm back toward the arm, to "
            "[%.2f, %.2f] and down onto the table, %s"
            % (
                ", ".join(lg.name for lg in flying[1:]),
                1000 * _lift_m(ctx, lift),
                1000 * lift.back_m,
                drop.xyz[0],
                drop.xyz[1],
                _touch_says(carry, search.after_this(flight_s(runner, flying))),
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
        if placed is not None and placed.carry is None:
            log.info(
                "[press_demo] no spot whose carry and home both plan was found while the hand came down (%s): "
                "the lift alone, then the spot look"
                % ("; ".join(placed.refused) or (placed.passed.says() if placed.passed else "no spot the hand reaches"))
            )
        flying = [close, *lift.legs]
        log.info(
            "[press_demo] CLOSE, then ONE MOTION: %s — %.0f mm straight up, then %.0f mm back toward the arm"
            % (", ".join(lg.name for lg in lift.legs), 1000 * _lift_m(ctx, lift), 1000 * lift.back_m)
        )
        # while the lift and the clear fly, the carry to the scene camera's
        # best spot from where the clear ends (carry_ahead): flown after the
        # lift where the spot chosen there is the same
        _video(ctx, "CLOSE on the knob, then LIFT %.0f mm and CLEAR the box" % (1000 * _lift_m(ctx, lift)))
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
    return _release_and_home(ctx, cfg, runner, carry, drop, placed.tier, placed.home, None, search=search)


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
    passed: object = None  # the spots passed over and why (SpotsPassed)
    spot_z: float = None  # the table's height the spots were made at (setdown_table_z)


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


def carry_ahead(ctx, cfg, eyes, q_end, room=None, home=False, q_grip=None):
    """The lid's spot from the scene camera's map alone (spot_map, no wrist
    frame; `room`'s, the room on it judged while the press flew, where it
    is the box's still: room_for), its spot_candidates the hand reaches from
    `q_end`, where the clear ends, and the carry from there to the first
    that plans (plan_carry, at most CARRY_TRIES) — with `home`, and home
    after its set-down too (plan_home_after): a spot home does not plan
    from is passed over. The grip's lookahead (grip_ahead, with home): the
    close to the set-down then fly as one. Else the lift's: the spot is
    chosen after the lift, the wrist's frame in its map (choose_and_carry),
    and this carry flies only where that spot is its own. `q_grip`: the
    joints the knob is gripped at, where the lift is not on ctx yet (the
    grip's lookahead): the table the set-down goes by (setdown_table_z).
    Returns a CarryAhead. CarrySearch's two parts, one after the other."""
    return CarrySearch(ctx, cfg, eyes, q_end, room=room, home=home, q_grip=q_grip).start().finish()


class CarrySearch:
    """carry_ahead in two parts, each of which a motion can host (the grip
    straight from the press: grip_ahead_rise, grip_ahead_down), both on the
    main thread — the planner is called from nowhere else. start(): the
    room (room_for), the map and the first spot the hand reaches, no plan;
    finish(): the carry (and home after it) planned to that spot, the next
    ones tried where it is refused, and the CarryAhead — what carry_ahead
    returns, its spent_s both parts' time."""

    def __init__(self, ctx, cfg, eyes, q_end, room=None, home=False, q_grip=None):
        self.ctx, self.cfg, self.eyes, self.room, self.home = ctx, cfg, eyes, room, home
        self.q_grip = None if q_grip is None else [float(v) for v in q_grip]
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
        # the counter's edges this map shows, kept out of the carry and the set-down planned from it
        pd.keep_out_counter(ctx, tmap, table_z, "the lid's spot map", key="spot")
        rest = list(pd.rest_joints(self.cfg))
        self._reach = _Reach()
        spot_z, moved = setdown_table_z(ctx, table_z, scene_fix, self.q_grip)
        _say_setdown_table(spot_z, moved, table_z)
        self.ahead = CarryAhead(
            self.q,
            tmap=tmap,
            table_z=table_z,
            room_s=None if judged is None else judged.spent_s,
            room_wait_s=room_wait_s,
            room=judged,
            passed=SpotsPassed(),
            spot_z=spot_z,
        )
        self._spots = spot_candidates(
            ctx,
            tmap,
            table_z,
            self.q,
            seeds=(look_joints(rest), rest),
            reach=self._reach,
            cache=None if judged is None else judged.cache,
            spot_z=spot_z,
            passed=self.ahead.passed,
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
    (lid_spots: ~1.3 s of the 1.7-2.0 s the scene
    camera's spot took on 10-29's runs), on a thread of its own while the
    press flies (~8.5 s). Only the ranking needs the way the hand faces
    where the clear ends, known once the grip is planned: grip_ahead ranks
    from the cache. Returns the Room, its thread started."""
    _node, watcher, scene_fix, scene_q = eyes
    model, cpose, worlds = ctx.model, ctx.cpose, ctx.worlds
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
                lid_spots(model, cpose, worlds, room.tmap, table_z, cache=room.cache)
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
    return lift, None, carry_ahead(ctx, cfg, eyes, lift.legs[-1].goal_joints, room=room, home=True, q_grip=q_grip)


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
    return (
        lift,
        None,
        CarrySearch(ctx, cfg, eyes, lift.legs[-1].goal_joints, room=room, home=True, q_grip=q_grip).start(),
    )


def grip_ahead_down(first):
    """grip_ahead's second part, while the hand comes down to the knob: the
    carry to the spot grip_ahead_rise found and home after its set-down
    (CarrySearch.finish) — `first` its result. Returns grip_ahead's:
    (LiftOff, None, CarryAhead) or (None, why, None)."""
    lift, why, search = first
    if lift is None or search is None:
        return lift, why, None
    return lift, None, search.finish()


def _ahead_note(ahead, off, flew, chosen, reach, stale=None):
    """lid_spot's 'ahead': whether the carry carry_ahead planned flew, why
    not, how far the arm stood from where it was planned from (`off`, rad)
    and the time it saved at rest — its plan where it flew, and the reach
    verdicts taken from it (`reach`, _Reach); `stale`, why the lid's own map
    refused it (carry_refusal). None without one."""
    if ahead is None:
        return None
    why = None
    if not flew:
        if stale is not None:
            why = "the lid's own map keeps it out: %s" % stale
        elif ahead.carry is None:
            why = "none planned (%s)" % ("; ".join(ahead.refused) or "no spot the hand reaches")
        elif off > DRIFT_REPLAN_RAD:
            why = "the arm stands %.3f rad from where it was planned from (gate %.2f)" % (off, DRIFT_REPLAN_RAD)
        elif chosen is None:
            why = "no carry plans here"
        else:
            d = chosen[0]
            why = "the spot chosen with the lid lifted is %.0f mm from it%s%s" % (
                1000 * math.hypot(d.xyz[0] - ahead.drop.xyz[0], d.xyz[1] - ahead.drop.xyz[1]),
                ""
                if abs(math.remainder(float(d.yaw) - float(ahead.drop.yaw), 2 * math.pi)) < 1e-3
                else ", the hand turned otherwise",
                "" if _spot_lean(d) == _spot_lean(ahead.drop) else ", the hand leaning otherwise",
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


def _last_go(runner):
    """When this run last had the engine's go (time.time(), the stamp its rows carry), or None: the
    newest 'line_up' row of stage 'go' in its own record, which press_demo.line_up_watch writes as it
    hears one — before the run's first move (--line-up) and after each line-up in the middle of it
    (--guide). None too where the record cannot be read: the clock then counts from the run's
    start, the earlier."""
    path = getattr(runner, "_log_path", None)
    if path is None:
        return None
    last = None
    try:
        with open(path) as f:
            for line in f:
                if '"line_up"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("leg") == "line_up" and row.get("stage") == "go" and row.get("t") is not None:
                    last = float(row["t"])
    except (OSError, TypeError, ValueError):
        return last
    return last


class RunClock:
    """How long this run still has before the engine stops it (SPOT_S's note): left_s().

    The engine hands the run its budget in RUN_BUDGET_ENV as it starts it (node.budget_env), JSON:
    {"s": the seconds it has, "from": the time.monotonic() they count from (the engine and the run
    are processes on one host, and that clock is the host's: no step of the wall clock moves it),
    "go": whether they start again, whole, at each go the engine gives the run (a goal sent by
    hand: the time a person takes to move a chair is none of the run's)}. The go's time is the
    run's own record's (_last_go: the engine's go comes down the run's stdin to press_demo's ear,
    and the row the run writes as it hears it is the only stamp of it this side has; its wall-clock
    stamp is put on the engine's clock by what the two differ by now, and never dated later than
    now or earlier than the budget's own start).

    Three cases, and they are not alike:
      nothing handed (a run from the command line): no engine times it, left_s() is None, and
        SETDOWN_SPOTS alone bounds its spots;
      a budget handed: left_s() seconds, negative past it;
      something handed that does not read as one (`unreadable` says what): an engine IS timing
        this run and the run does not know by how much — left_s() is 0.0, so no further spot is
        ever started on it (the first review of 2026-11-19: it was taken for "nothing handed", and
        an unreadable budget would have let a run try every spot with the engine's Ctrl+C due).
    `runner`: the run's, for its record; `env`, `now`, `wall`: the tests'."""

    def __init__(self, runner=None, env=None, now=time.monotonic, wall=time.time):
        self.runner, self._now, self._wall = runner, now, wall
        self.s = self.since = None
        self.from_go = False
        self.unreadable = None
        self._go = (None, None)  # (the record's size when it was last read, the go found in it)
        raw = (os.environ if env is None else env).get(RUN_BUDGET_ENV)
        if raw is None:
            return
        try:
            told = json.loads(raw)
            if not isinstance(told, dict):
                raise ValueError("not an object")
            s, since, go = told["s"], told["from"], told.get("go", False)
            for v in (s, since):
                if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
                    raise ValueError("%r is not a number of seconds" % (v,))
            if not isinstance(go, bool):
                raise ValueError("go is %r, not true or false" % (go,))
            self.s, self.since, self.from_go = float(s), float(since), go
        except (KeyError, TypeError, ValueError) as e:
            self.unreadable = "%s=%r: %s" % (RUN_BUDGET_ENV, raw, e if not isinstance(e, KeyError) else "no %s" % e)
            log.info(
                "[press_demo] the run's time as the engine handed it cannot be read (%s): the engine times this "
                "run and the run does not know how long it has — a lid that feels no table at its spot goes back "
                "on the box, no further spot is started" % self.unreadable
            )

    def _go_since(self):
        """The engine's clock at this run's last go, or None (none heard, or no record to read)."""
        path = getattr(self.runner, "_log_path", None)
        try:
            size = None if path is None else os.path.getsize(path)
        except OSError:
            size = None
        if size is None:
            return None
        if size != self._go[0]:  # rows were written since: read again (every reach check asks the clock)
            self._go = (size, _last_go(self.runner))
        go = self._go[1]
        if go is None:
            return None
        now = float(self._now())
        return min(max(go - (float(self._wall()) - now), self.since), now)

    def left_s(self):
        """Seconds until the engine's stop, negative past it; None: nothing times this run; 0.0
        where what times it cannot be read."""
        if self.unreadable is not None:
            return 0.0
        if self.s is None:
            return None
        since = self.since
        if self.from_go:
            go = self._go_since()
            if go is not None:
                since = go
        return since + self.s - float(self._now())


def flight_s(runner, legs):
    """How long `legs` take to fly as the runner will re-time them (its preview's own sum: each
    merged group's profile), or None where that cannot be told (a group with no trajectory yet)."""
    from dataclasses import replace

    from box_opening.runtime import runner as rn
    from box_opening.runtime.legs import Kind

    try:
        total = 0.0
        for group in merge_groups(legs):
            if group[0].kind is not Kind.MOTION:
                continue
            if any(g.traj is None for g in group):
                return None
            _traj, info = rn.retime_group(
                [g.traj for g in group],
                [g.speed for g in group],
                rn.JOINT_VMAX,
                replace(runner.retime_params(group[0]), time_scale=runner.time_scale),
                folds=[g.fold for g in group],
                step_caps=rn._step_caps(group),
                **runner._near_user_kw(),
            )
            total += float(info["duration_s"])
        return total
    except Exception:  # noqa: BLE001 -- an estimate the clock is asked with: where it cannot be made, SPOT_S stands in
        return None


# the hand's farthest corner from its tool axis as the gate has it (clear_table.HAND_PARTS: the wrist
# camera's, 99 mm): the footprint a hand leaves round where it came down
HAND_REACH_M = max(
    math.hypot(max(-xa, xb), max(-ya, yb)) for _n, (xa, xb), (ya, yb), _b, _need in clear_table.HAND_PARTS
)


@dataclass
class SpotLeft:
    """A spot the lid was taken to and left again with the lid still in the fingers: which of the
    run's spots it was (`n`, 1 the first), where, its tier, and what the set-down's watch said
    (`verdict`: "never" | "held"; `why`, its words). `hand_xy`: where the hand's tool stood over the
    table there (a leaning hand stands arm-side of the lid's middle). `above_m`: for a "held", how
    far over where the plan had the table the lid was held up."""

    n: int
    drop: object
    tier: str
    verdict: str
    why: str
    hand_xy: tuple = None
    above_m: float = None

    def says(self):
        return "spot %d at [%.2f, %.2f]: %s" % (self.n, self.drop.xyz[0], self.drop.xyz[1], _why_short(self.verdict))


def _why_short(verdict):
    """A set-down's verdict in a few words, for the line that says why a spot was left."""
    return {
        "never": "no table felt by the touch's end",
        "held": "something held the lid up above where the table can be",
    }.get(verdict, "no table felt")


@dataclass
class Picked:
    """What SpotSearch.pick found: the carry that planned (None: no spot) and its spot (`chosen`:
    the spot, its tier, what it says), the spots the planner refused, what was passed over, the
    reach verdicts (`reach`: the tiers', which take the lift's lookahead's; `reaches`: all), whether the
    carry planned while the lift flew is the one (`flew`; `stale`, why the lid's own map refused
    that one), the plans' time, which searches it took (`searched`), whether the wrist looked again
    for it (`looked`), and whether the run's clock ended the choice (`timed_out`: the time left no
    longer held a spot and the way back to the box after it)."""

    carry: object = None
    chosen: tuple = None
    refused: list = field(default_factory=list)
    passed: object = None
    reach: object = None
    reaches: list = field(default_factory=list)  # every _Reach it asked, `reach` (the tiers' first) among them
    flew: bool = False
    stale: str = None
    carry_s: float = 0.0
    searched: list = field(default_factory=list)
    looked: bool = False
    timed_out: bool = False

    def ik_s(self):
        return sum(r.ik_s for r in self.reaches)

    def checks(self):
        return sum(len(r.found) for r in self.reaches)


class SpotSearch:
    """Where the lid goes, chosen with the lid in the fingers and the arm standing: after the lift
    (choose_and_carry) and after every set-down that felt no table (another_spot). It keeps what
    one choice hands the next: the map and the wrist's frames in it, the table's heights, the spots
    the lid was taken to and left (`left`: each set aside for the run with the footprint that was
    left there, `aside`), the joints the lid hangs level at (`held`: where the first carry began, the
    hand as the grip had it; later carries begin with the hand as a set-down left it), where the
    wrist last looked from (`looked_from`) and the run's clock.

    pick() takes the first spot whose carry plans, looked for in this order:
      1. spot_candidates, as every run before 2026-11-19 chose: the same tiers, the same order, at
         most CARRY_TRIES carries refused;
      2. none there: wider_candidates — the whole measured table, then the hand leaning as far as
         the press's look — CARRY_TRIES more;
      3. still none, and the map in hand was not made with the wrist's frame from where the arm
         stands now (the scene camera's alone, made before the press; or a look from another stand):
         the wrist looks from here (look_again), its frame is pooled in, and 1 and 2 once more.
    Nothing: no spot is left, and the caller's last resort is the box (put_lid_back).

    Every carry but the first spot's own from the tiers (planned as it was before 2026-11-19) is
    asked what a LaterWay asks: the arm's own walk first, its path at carry height over the table
    and over what stands on it, a leaning set-down's hand out of what stands (plan_carry's `way`).

    The run's clock is asked all through the choice of a spot after the first — before it begins,
    of every spot looked at, before each reach check and before each plan — and through the wider
    search and before the second look of the first spot's too: the moment what is left no longer
    holds a spot and the way back to the box after it (out_of_time), choosing stops, with the
    box's own time still in hand (Picked.timed_out)."""

    def __init__(self, ctx, cfg, eyes, runner, tmap, table_z, spot_z, cache=None, held=None, looked_from=None):
        self.ctx, self.cfg, self.eyes = ctx, cfg, eyes
        self.tmap, self.table_z, self.spot_z = tmap, table_z, spot_z
        self.cache = {} if cache is None else cache
        self.held = None if held is None else [float(v) for v in held]
        self.looked_from = None if looked_from is None else [float(v) for v in looked_from]
        self.frames = []  # the wrist's frames of the looks so far, each with the joints it was taken at
        self.left = []  # SpotLeft, in order
        self.spots = 1  # the spots the lid has been taken to: the one in flight counts
        self.clock = RunClock(runner)

    def _hand_xy(self, d):
        """Where the hand's tool stands over the table setting the lid down at `d` (setdown_poses'
        contact; `d`'s own xy where that cannot be told: a leaning spot with no grip to go by)."""
        try:
            return tuple(float(v) for v in setdown_poses(self.ctx, d, held=self.held)[1][:2])
        except Exception:  # noqa: BLE001 -- the footprint's centre then; never what stops the search
            return (float(d.xyz[0]), float(d.xyz[1]))

    def aside(self, d):
        """Whether the spot `d` is set aside for this run.

        After a spot where NOTHING was felt ("never"): `d`'s lid would lie on some of the place the
        lid was held over there — their centres nearer than the lid is wide. No table was felt
        there; what the map says of that place is not gone by again.

        After a spot where something HELD the lid up, which the map did not show: no part of the
        next lid or hand over the footprint that was left there, the lid's and the hand's (a disc
        of HAND_REACH_M round where the tool stood: its farthest corner, whichever way it is
        turned next). The lid's width alone let the next spot lie rim to rim with whatever held
        it, 108 mm away for a 105 mm lid, the hand's parts coming down over it (the review of
        2026-11-19)."""
        m = self.ctx.model
        lid_r = 0.5 * max(float(m.lid_dims[0]), float(m.lid_dims[1]))
        at = (float(d.xyz[0]), float(d.xyz[1]))
        hand = None
        for s in self.left:
            was = (float(s.drop.xyz[0]), float(s.drop.xyz[1]))
            if math.dist(at, was) < 2.0 * lid_r:
                return True
            if s.verdict != "held":
                continue
            hand = self._hand_xy(d) if hand is None else hand
            was_hand = was if s.hand_xy is None else s.hand_xy
            if (
                math.dist(at, was_hand) < lid_r + HAND_REACH_M
                or math.dist(hand, was) < HAND_REACH_M + lid_r
                or math.dist(hand, was_hand) < 2.0 * HAND_REACH_M
            ):
                return True
        return False

    def leave(self, drop, tier, verdict, why, above_m=None):
        """The lid leaves `drop` still in the fingers: set aside (aside). Returns the SpotLeft."""
        self.left.append(
            SpotLeft(len(self.left) + 1, drop, tier, verdict, why, hand_xy=self._hand_xy(drop), above_m=above_m)
        )
        return self.left[-1]

    def same_height(self, was):
        """The earlier spot where something held the lid up at the height it was held up at `was`
        (within SAME_HELD_M), or None: two of those are the table itself, higher than the run has it."""
        if was.verdict != "held" or was.above_m is None:
            return None
        for s in self.left:
            if s is not was and s.verdict == "held" and s.above_m is not None:
                if abs(s.above_m - was.above_m) <= SAME_HELD_M:
                    return s
        return None

    def no_more(self, need_s, own=False):
        """Why no further spot is started, or None: SETDOWN_SPOTS are spent, the run's time cannot
        be read, or its clock has less than `need_s` and the way back to the box after it (BOX_S,
        END_S) left. `own`: `need_s` is a planned carry's own flight, not a spot's time on record."""
        if self.spots >= SETDOWN_SPOTS:
            return "%d spots is as many as a run tries (SETDOWN_SPOTS)" % SETDOWN_SPOTS
        if self.clock.unreadable is not None:
            return "the run's time as the engine handed it cannot be read (%s)" % self.clock.unreadable
        left = self.clock.left_s()
        if left is not None and left < need_s + BOX_S + END_S:
            return (
                "%.0f s of the run's time are left: another spot and, should it feel no table either, the way "
                "back to the box take %.0f (%.0f %sand %.0f, on record)"
                % (
                    max(left, 0.0),
                    need_s + BOX_S + END_S,
                    need_s,
                    "by this spot's own carry " if own else "",
                    BOX_S + END_S,
                )
            )
        return None

    def after_this(self, flight=None):
        """Why a set-down about to fly (`flight` s: its carry's own time, None: SPOT_S's) will be
        followed by the box and not by another spot should it feel no table, or None: it is the
        last of SETDOWN_SPOTS, or by its end the run's clock will not hold another (no_more, asked
        of the time there will be then). What the line before a set-down promises is this."""
        if self.spots >= SETDOWN_SPOTS:
            return "the last of the %d spots a run tries" % SETDOWN_SPOTS
        if self.clock.unreadable is not None:
            return "the run's time cannot be read"
        left = self.clock.left_s()
        if left is None:
            return None
        then = left - (SPOT_S - SPOT_UP_S if flight is None else float(flight))
        if then < CHOICE_S + SPOT_S + BOX_S + END_S:
            return "the run's time will hold no further spot: %.0f s are left now" % max(left, 0.0)
        return None

    def out_of_time(self):
        """Whether the run's clock leaves no time for a spot still being looked for (SPOT_S) and the
        way back to the box after it: the choice's stop. Asked of the first spot's search too — the
        tiers themselves are gone through whatever the clock says, as they always were, but the
        wider search and the wrist's second look are work a run did not do before 2026-11-19 (up
        to ~16 s of reach checks where the hand reaches nothing), and they must not take the time
        the lid's way back to the box needs. With 31 s left where it says so, the box's 23 are
        still in hand whatever one more reach check or plan takes."""
        left = self.clock.left_s()
        return left is not None and left < SPOT_S + BOX_S + END_S

    def way(self, held=None):
        """The LaterWay a carry of this search is asked (the lid as the hand at `held` holds it:
        this search's own where it has one): the table the set-down goes by, or the cameras' where
        that stands higher — the less room of the two."""
        heights = [float(z) for z in (self.table_z, self.spot_z) if z is not None]
        # (no height at all: no table was measured, there is no map, and the height is in the words alone)
        return LaterWay(self.ctx, self.tmap, max(heights, default=0.0), self.held if self.held is not None else held)

    def pick(self, runner, q, ahead=None, off=None):
        """The first spot whose carry plans from joints `q` (the class's note), as a Picked.
        `ahead`, `off`: the carry planned while the lift flew and how far the arm stands from where
        it was planned from (choose_and_carry's: its reach verdicts are taken within
        DRIFT_REPLAN_RAD, its carry flies where the spot chosen is its own and nothing was left)."""
        from box_opening.primitives.look import look_joints
        from box_opening.tasks import press_demo as pd

        ctx = self.ctx
        rest = list(pd.rest_joints(self.cfg))
        near = ahead is not None and off is not None and off <= DRIFT_REPLAN_RAD
        later = bool(self.left)
        held = self.held if later else None  # the first carry begins where the lid hangs level: as before
        out = Picked(passed=SpotsPassed())
        offered = set()  # every spot a carry was planned to here: refused, and not planned to twice
        way = None  # (made on the map in hand each time round: the wrist's look gives another)

        def stop():
            if self.out_of_time():
                out.timed_out = True
            return out.timed_out

        def first(cands, wider):
            refused = 0
            for drop, tier, says in cands:
                if refused >= CARRY_TRIES:
                    break
                if _spot_key(drop) in offered:
                    continue  # (the tiers gone through once more after the wrist's look: the planner's answer stands)
                if (later or wider) and stop():
                    return False  # a plan can take a second and more: the clock before each
                offered.add(_spot_key(drop))
                reuse = near and not wider and not later and ahead.carry is not None and out.stale is None
                if reuse and _same_spot(drop, ahead.drop):
                    # planned before the lid's own map: held out of what that map shows too, or planned again
                    out.stale = carry_refusal(ahead.carry)
                    if out.stale is None:
                        out.carry, out.chosen, out.flew = ahead.carry, (ahead.drop, tier, says), True
                        return True
                t = time.monotonic()
                if wider or later:
                    carry, why = plan_carry(ctx, self.cfg, q, drop, held=held, over_box=True, way=way)
                else:  # the first spot's own, from the tiers: planned as it was before 2026-11-19
                    carry, why = plan_carry(ctx, self.cfg, q, drop)
                out.carry_s += time.monotonic() - t
                if carry is not None:
                    out.carry, out.chosen = carry, (drop, tier, says)
                    return True
                refused += 1
                out.refused.append("[%.2f, %.2f]: %s" % (drop.xyz[0], drop.xyz[1], why))
            return False

        while True:
            if later and stop():
                return out
            way = self.way(q)
            # a later spot's reach is asked what its carry will be (a leaning hand out of what stands, on
            # this map); the first spot's tiers as they always were, its wider search with a reach of its
            # own that asks it
            in_tiers = _Reach(
                ahead.known if near and not out.reaches else None, held=held, stands=way.descent if later else None
            )
            in_wider = in_tiers if later else _Reach(held=held, stands=way.descent)
            out.reaches += [in_tiers] if later else [in_tiers, in_wider]
            out.reach = out.reach or in_tiers
            common = dict(
                seeds=(look_joints(rest), rest),
                cache=self.cache,
                spot_z=self.spot_z,
                passed=out.passed,
                aside=self.aside if later else None,
                held=held,
            )
            # (from a spot the lid was left at, the way to the next is asked of every spot: _offers' `way`;
            # and the clock of every spot looked at)
            tiers = spot_candidates(
                ctx, self.tmap, self.table_z, q, reach=in_tiers, way=later, stop=stop if later else None, **common
            )
            out.searched.append("the tiers")
            if first(tiers, False):
                return out
            if stop():  # before the wider search's own work, not after a second or three of it
                return out
            log.info(
                "[press_demo] no spot of the tiers has a carry that plans (%s): looking wider — the whole of the "
                "table the cameras measured, then the hand leaning as far as %.0f deg"
                % ("; ".join(out.refused[-CARRY_TRIES:]) or out.passed.says(), SETDOWN_LEAN_MAX_DEG)
            )
            out.searched.append("wider")
            wider = wider_candidates(ctx, self.tmap, self.table_z, q, reach=in_wider, skip=offered, stop=stop, **common)
            if first(wider, True):
                return out
            if out.timed_out or out.looked or not self.look_again(runner, q):
                return out
            out.looked = True

    def look_again(self, runner, q, why=None):
        """The wrist's frame from where the arm stands, at joints `q`, pooled into the map (with
        the frames of the looks before it), where the map in hand was not made with one from here:
        the scene camera's alone (made before the press, from 14 cm over the table: edge-on), or
        with a look from another stand. What the map says may be stale by now, and from above the
        wrist sees table the scene camera does not. `why`: what is said of it where it is not for
        want of a spot (another_spot's, after something the map did not show held the lid up).
        Once a stand; not where the run's clock leaves no time for a spot after it. Returns
        whether there is a new map to look for a spot on."""
        from box_opening.geometry import ang_diff
        from box_opening.tasks import press_demo as pd

        ctx = self.ctx
        if self.looked_from is not None:
            moved = max(abs(ang_diff(a, b)) for a, b in zip(q, self.looked_from, strict=False))
            if moved <= DRIFT_REPLAN_RAD:
                return False  # the map in hand has the wrist's frame from here in it
        if self.out_of_time():
            return False
        node, watcher, scene_fix, scene_q = self.eyes
        log.info(
            "[press_demo] %s: the wrist looks at the table from here, and the spot is looked for on a map with its "
            "frame in it" % (why or "no spot on the map in hand, which was made before the arm stood here")
        )
        _video(ctx, "%s: the wrist looks at the table from here" % (why or "NO SPOT on the map in hand"))
        wrist = spot_look(node, watcher, runner)
        at = [float(v) for v in ctx.client.joints()]  # where the arm stood still for the frame
        self.looked_from = at
        n = 0 if wrist is None else len(wrist)
        tmap = None
        if n:
            self.frames.append((wrist, at))
            tmap = spot_map(
                ctx,
                None,
                at,
                scene_fix,
                scene_q,
                self.table_z,
                getattr(getattr(watcher, "grab", None), "wrist_lens_in_hand_m", None),
                more=self.frames,
            )
        runner.note("lid_look_again", wrist_points=n, mapped=tmap is not None, looks=len(self.frames), why=why)
        if tmap is None:
            log.info("[press_demo] the wrist's look from here adds no map of the table (%d points)" % n)
            return False
        self.tmap, self.cache = tmap, {}
        # the counter's edges this map shows, kept out of the carries planned on it
        pd.keep_out_counter(ctx, tmap, self.table_z, "the lid's spot map, looked at again", runner, key="spot")
        return True


def choose_and_carry(ctx, cfg, runner, eyes, ahead=None):
    """The lid lifted and clear of the box, in the fingers: its one stop.
    The wrist's frame (spot_look) and the scene camera's table make the map
    (spot_map) the spot is chosen on; the first of spot_candidates the carry plans to
    (plan_carry, at most CARRY_TRIES) is where the lid goes, one line saying
    which tier and why; none: wider_candidates' (SpotSearch.pick). `ahead` (carry_ahead's, made while the lift flew):
    the arm within DRIFT_REPLAN_RAD of where it was made, its reach
    verdicts are taken (_Reach), and its carry flies where the spot chosen
    is its own (_same_spot) — no plan at rest; else the carry is planned
    here, as without one. Then the carry and the set-down, ONE execution
    that hosts no planning (its watch is polled from its first sample: _fly_carry); the
    knuckle read (the lid slipped?); the release and the retreat, home
    planned while that flies (_release_and_home); home. The
    lid_spot note says how long each part took here, and whether the carry
    planned ahead flew (_ahead_note). Returns the exit code: 0 with the lid
    set down and the arm home."""
    from box_opening.geometry import ang_diff
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
    # the counter's edges the lifted lid's map shows (the wrist's frame in it), kept out of the carry
    pd.keep_out_counter(ctx, tmap, table_z, "the lid's spot map, the lid lifted", runner, key="spot")
    t_map = time.monotonic()
    # the reach of a spot judged from where the clear was to end is the
    # reach from here where the arm stands as near as the runner flies a
    # planned leg from (its drift gate): ahead's verdicts are taken, and its
    # carry, planned from there, flies from here
    off = None if ahead is None else max(abs(ang_diff(a, b)) for a, b in zip(q, ahead.q, strict=False))
    spot_z, moved = setdown_table_z(ctx, table_z, scene_fix)
    _say_setdown_table(spot_z, moved, table_z)
    search = SpotSearch(ctx, cfg, eyes, runner, tmap, table_z, spot_z, held=q, looked_from=q)
    if wrist is not None and len(wrist):
        search.frames.append((wrist, q))
    got = search.pick(runner, q, ahead=ahead, off=off)
    t_end = time.monotonic()
    carry, refused, chosen, passed, reach = got.carry, got.refused, got.chosen, got.passed, got.reach
    note = _ahead_note(ahead, off, got.flew, chosen, reach, got.stale)
    runner.note(
        "lid_spot",
        xyz=None if chosen is None else [round(v, 4) for v in chosen[0].xyz],
        tier=None if chosen is None else chosen[1],
        wrist_points=0 if wrist is None else len(wrist),
        mapped=tmap is not None,
        refused="; ".join(refused) or None,
        look_s=round(t_look - t0, 3),
        map_s=round(t_map - t_look, 3),
        candidates_s=round(t_end - t_map - got.carry_s, 3),
        ik_s=round(got.ik_s(), 3),
        spot_checks=got.checks(),  # the spots whose reach was judged here (the rest: ahead's)
        carry_s=round(got.carry_s, 3),
        no_land=no_land_note(ctx, None if chosen is None else chosen[0]),
        table=table_note(ctx, table_z, spot_z, passed),
        ahead=note,
        pooled=pooled_note(ctx, runner, None if ahead is None else ahead.room, None if chosen is None else chosen[0]),
        spot=1,
        searched=got.searched,
    )
    if note is not None:
        log.info(
            "[press_demo] %s"
            % (
                "the carry planned while the lift flew goes (%.2f s not spent here)" % note["saved_s"]
                if got.flew
                else "the carry planned while the lift flew is not flown: %s" % note["why"]
            )
        )
    if carry is None:
        left_s = search.clock.left_s()
        return put_lid_back(
            ctx,
            cfg,
            runner,
            "no carry plans to a spot on the table the cameras measured (%s)%s"
            % (
                "; ".join(refused) or passed.says(),
                # (said, where the clock and not the table ended the looking: the tiers were gone through)
                ""
                if not got.timed_out
                else ", and no more of the table is looked at: the run's time holds no spot found later and the way "
                "back to the box after it (%.0f s are left, they take %.0f)"
                % (max(left_s or 0.0, 0.0), SPOT_S + BOX_S + END_S),
            ),
        )
    drop, tier, says = chosen
    _say_spot(ctx, eyes, chosen, refused, search.tmap, table_z)
    _video(
        ctx,
        "CARRY to [%.2f, %.2f] (%s) and SET DOWN, let go where it meets the table" % (drop.xyz[0], drop.xyz[1], tier),
    )
    log.info(
        "[press_demo] ONE MOTION: %s — to [%.2f, %.2f] and down onto the table, %s"
        % (
            ", ".join(lg.name for lg in carry.legs),
            drop.xyz[0],
            drop.xyz[1],
            _touch_says(carry, search.after_this(flight_s(runner, carry.legs))),
        )
    )
    return _fly_carry(ctx, cfg, runner, carry, drop, tier, search)


def _fly_carry(ctx, cfg, runner, carry, drop, tier, search):
    """The carry to `drop` and its set-down, watched (watch_setdown), as ONE execution from where
    the arm stands with the lid — after the lift (choose_and_carry) or over a spot it left
    (another_spot) — then what the watch felt decides (_release_and_home: the release and home, or
    on to another spot with `search`). Returns the exit code."""
    watch_setdown(ctx, carry.legs, carry)  # where the lid meets the table, watched
    # No lookahead on this execution, as on the press's own (reach_press.fly_way_in). The client runs what a
    # goal hosts once the goal is accepted and only then starts to poll its stop_when (client.execute): home
    # was planned here (plan_home_after), and for as long as that took the set-down's watch saw nothing. A
    # plan still running where the lid met the table left the reference going on to the touch's end, 8 mm
    # under the plan, the spring pressing 21-34 mm into a table 13-26 mm over it; one that outlasted the
    # flight left the watch never polled, the run said "not watched" of feedback that was kept, and a lid
    # that stood on the table went back on the box (review 2026-11-17; run 1's grip lookahead overran its
    # host by 3.6 s). Home is planned while the retreat flies instead, where nothing is watched
    # (_release_and_home)
    res = runner.run(carry.legs, execute=True)
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
    return _release_and_home(ctx, cfg, runner, carry, drop, tier, None, None, search=search)


def _say_spot(ctx, eyes, chosen, refused, tmap, table_z):
    """Where the lid goes (`chosen`: the spot, its tier, what it says):
    kept (ctx.lid_drop), said with the spots the planner refused, and drawn
    (perception/viz; the lid video's scene panel)."""
    from box_opening.tasks import press_demo as pd

    drop, tier, says = chosen
    ctx.lid_drop = drop
    log.info(
        "[press_demo] lid goes to [%.2f, %.2f, %.3f] — %.0f mm from the box (%.0f mm outside its no-land radius), %s%s"
        % (
            drop.xyz[0],
            drop.xyz[1],
            drop.xyz[2],
            1000 * math.hypot(drop.xyz[0] - ctx.cpose.xyz[0], drop.xyz[1] - ctx.cpose.xyz[1]),
            1000 * box_clearance(ctx, drop),
            says,
            "; the planner refused %s" % "; ".join(refused) if refused else "",
        )
    )
    pd.viz_hook("spot", tmap, drop, tier, says, table_z)  # the map and the spot, drawn off this thread (perception/viz)
    _video_spot(ctx, eyes, drop)


def _touch_says(carry, box=None):
    """Where `carry`'s set-down lets the lid go, and what follows where it does not. `box`: why a
    set-down that feels no table will be followed by the box and not by another spot
    (SpotSearch.after_this: the run's time, the last of its spots), None where another spot will
    be tried — said as it will be: the line promised "on to another spot" of set-downs whose clock
    then sent them to the box (the review of 2026-11-19)."""
    return (
        "slowly from %.0f mm over where the plan has the lid meet it to %.0f mm under that, let go only where the "
        "arm feels it meet the table (no more than %.0f mm over the plan); felt nowhere, %s"
        % (
            1000 * SETDOWN_SLOW_M,
            1000 * (carry.over_m or 0.0),
            1000 * SETDOWN_CREDIBLE_M,
            "up again with it and on to another spot" if box is None else "back onto the box (%s)" % box,
        )
    )


def put_lid_back(ctx, cfg, runner, why):
    """A STOP with the lid in the fingers and the arm standing — the run's last resort for it (no
    carry to any spot plans, the wider search and the wrist's second look included: SpotSearch.pick;
    the carry was refused before it went out; or a set-down felt no table and no other spot is
    tried: carry_back, another_spot): the lid goes
    back where it came from, the one
    place known to take it, rather than staying in the fingers over the table. Built from where
    the arm stands (so no drift gate can refuse it): a straight line back over the box to the top
    of the lift (the hand turning back to the grip's attitude on the way where the carry turned
    it: the arm's own walk), and straight down to where the knob was held (ctx.lid_lifted) in one
    execution; the fingers open there; the hand rises straight up off the knob as the lid came up;
    home.

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
        if pts is None and name == "lid:back-over" and hasattr(chain, "walk") and _turned_from(chain, q, quat):
            # from the set-down's own attitude (the hand turned to the spot, or leaning: carry_back), no line
            # holds the grip's: the arm's own walk, turning back as it goes
            from box_opening.runtime import kinematics as kin

            pts, walked = chain.walk(
                q, target, quat, step_m=CARRY_WALK_STEP_M, step_rad=CARRY_WALK_STEP_RAD, max_step_rad=kin.MAX_STEP_RAD
            )
            refusal = refusal if pts is None else None
            if pts is None:
                refusal = "%s; walked: %s" % (refusal, walked)
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
        [float(v) for v in ctx.client.joints()], [grip_xyz[0], grip_xyz[1], grip_xyz[2] + _lift_m(ctx, lifted)], quat
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


def _turned_from(chain, q, quat, tol_deg=1.0):
    """Whether the hand at joints `q` stands turned from the attitude `quat` (more than `tol_deg`)."""
    from box_opening.perception.d405 import quat_to_mat

    R = np.asarray(chain.fk([float(v) for v in q])[0], float)
    cos = 0.5 * (float(np.trace(R.T @ np.asarray(quat_to_mat(*quat), float))) - 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > tol_deg


def carry_back(ctx, cfg, runner, carry, why, last_resort=None):
    """The set-down ended, the arm did not feel the lid meet the table (`why`, the watch's words),
    and no other spot is tried (`last_resort` says why not: another_spot's; None, a caller with no
    search to go on with): the lid is NOT let go. Nothing says there is a table under it —
    2026-11-17 16:40 let it go 64 mm over where the plan had the table, past the table's edge, onto
    the floor — and the fingers opening there is the one thing that cannot be taken back. Of what
    the run can do with a lid in the fingers, the box is the one place known to take it: back up to
    the carry's height the way the retreat goes (from where the arm stands: retreat_from_live),
    the fingers shut, then back over the box and down onto it (put_lid_back), the fingers open
    there, and home. Going on down past the touch's end was not chosen: nothing is known of what
    is under the lid there, and the hand's own reach below the table is not planned for. Where a
    line of it does not fly, the arm holds the lid where it stands, said. Returns 1: the lid did
    not come off."""
    world = _full_world(ctx)
    log.info(
        "[press_demo] SET-DOWN: no table felt — %s; the lid is not let go: back up with it, and back onto the box%s"
        % (why, "" if last_resort is None else " (%s)" % last_resort)
    )
    _video(ctx, "NO TABLE FELT: the lid is not let go, back onto the box")
    code = _back_up(ctx, runner, carry, why, world)
    if code is not None:
        return code
    return put_lid_back(
        ctx,
        cfg,
        runner,
        "the set-down felt no table (%s)%s" % (why, "" if last_resort is None else ", and %s" % last_resort),
    )


def _back_up(ctx, runner, carry, why, world):
    """With the lid still in the fingers after a set-down that felt no table (`why`): back up to
    the carry's height over that spot, the way the retreat goes, solved from where the arm stands
    (retreat_from_live), the fingers shut — 'lid:back-up'. None there; else 1, the STOP said: no
    line up solves, or it did not fly, and the arm holds the lid where it stands."""
    up, off = retreat_from_live(ctx, carry, world)
    if up is None:
        log.info(
            "[press_demo] STOP: the set-down felt no table (%s), and no way back up solves from where the arm "
            "stands (%s) — the arm holds the lid" % (why, off)
        )
        return 1
    up.name, up.start_hold_s = "lid:back-up", 0.0  # the fingers stay shut: nothing to wait for
    res = runner.run([up], execute=True)
    if not all(r.ok for r in res):
        log.info(
            "[press_demo] STOP: the set-down felt no table (%s), and the way back up with the lid %s (%s) — the "
            "arm holds the lid"
            % (why, "stopped" if _started(res[-1]) else "did not start", res[-1].detail or res[-1].outcome)
        )
        if _started(res[-1]):
            rp.let_go(ctx)
        return 1
    return None


def another_spot(ctx, cfg, runner, search, carry, drop, tier, refused, above_m=None):
    """The set-down at `drop` ended and the arm felt no table there (`refused`: the watch's verdict
    and words; `above_m`: for a "held", how far over the plan's table the lid was held up), the lid
    still in the fingers. Owner 2026-11-19, asked whether such a set-down should put the lid back
    on the box or try another spot first: "it should keep trying spots, assume tere will always be a
    spot for the lid". So the lid is neither let go nor taken back:

     - the spot is left (SpotSearch.leave): it, and the footprint that was left there, are set
       aside for the run (the lid's; the lid's and the hand's where something held the lid up);
     - the arm rises with the lid to the carry's height over it (_back_up);
     - where something the map did not show held the lid up, the wrist looks at the table from
       there before anything is chosen (SpotSearch.look_again): the map was wrong of that place;
     - the next spot is chosen from there by the same rules in the same order (SpotSearch.pick:
       every spot still on table the cameras measured, outside the no-land radius, reached by the
       hand, its carry held out of the chair, its user and the open canister, at carry height
       over the table and what stands on it), and where the tiers have none left, wider;
     - the lid is carried there and set down the same watched way (_fly_carry), and what that
       watch feels decides again: spot after spot.

    The box stays the last resort (carry_back from the touch's end, put_lid_back from over the
    spot), and its STOP says which it was: SETDOWN_SPOTS are spent; the run's clock leaves no time
    for another spot and the way back to the box after it (asked before the spot is left, all
    through the choice and, of the carry's own flight, before it is sent: SpotSearch.no_more,
    out_of_time; SPOT_S's note); something held the lid up at the same height at two spots — the
    table itself, higher than the run has it: no further spot is pressed on (SAME_HELD_M); or no
    spot is left on the table the cameras measured. Returns the exit code."""
    verdict, why = refused
    was = search.leave(drop, tier, verdict, why, above_m=above_m)
    same = search.same_height(was)
    if same is not None:
        stop = (
            "something held the lid up at the same height at two spots, %+.0f mm from where the plan has the table "
            "at spot %d and %+.0f mm at spot %d: that is probably the table itself, standing higher than the run "
            "has it, and it is not pressed on at another spot"
            % (1000 * same.above_m, same.n, 1000 * was.above_m, was.n)
        )
    else:
        stop = search.no_more(CHOICE_S + SPOT_S)
    left_s = search.clock.left_s()
    runner.note(
        "lid_spot_left",
        spot=was.n,
        xyz=[round(float(v), 4) for v in drop.xyz],
        tier=tier,
        verdict=verdict,
        why=why,
        above_mm=None if was.above_m is None else round(1000 * was.above_m, 1),
        same_height_as=None if same is None else same.n,  # probably the table itself: held up there as here
        time_left_s=None if left_s is None else round(left_s, 1),
        next="the box" if stop is not None else "another spot",
        why_the_box=stop,
    )
    if stop is not None:
        return carry_back(ctx, cfg, runner, carry, why, last_resort="no other spot is tried: %s" % stop)
    log.info(
        "[press_demo] SET-DOWN: no table felt at spot %d, [%.2f, %.2f] — %s; the lid is not let go: up with it and "
        "on to another spot (this one, and the footprint that was left there, set aside for this run%s)"
        % (
            was.n,
            drop.xyz[0],
            drop.xyz[1],
            why,
            "" if left_s is None else "; %.0f s of the run's time left" % left_s,
        )
    )
    _video(ctx, "NO TABLE FELT at spot %d: the lid is not let go, on to another spot" % was.n)
    code = _back_up(ctx, runner, carry, why, _full_world(ctx))
    if code is not None:
        return code
    runner.wait_still("the next spot for the lid")
    q = [float(v) for v in ctx.client.joints()]
    t0 = time.monotonic()
    looked = False
    if verdict == "held":
        # something stood there that the map did not show: the wrist looks before the next spot is chosen
        looked = search.look_again(runner, q, why="something the map did not show held the lid up at spot %d" % was.n)
        q = [float(v) for v in ctx.client.joints()]
    got = search.pick(runner, q)
    chosen, tried = got.chosen, "; ".join(s.says() for s in search.left)
    flight = None if got.carry is None else flight_s(runner, got.carry.legs)
    # the time once more before the carry is sent, for the flight this carry is (SPOT_S's until it was planned)
    stop = (
        None
        if got.carry is None
        else search.no_more((SPOT_S - SPOT_UP_S if flight is None else flight) + SPOT_UP_S, own=flight is not None)
    )
    left_s = search.clock.left_s()
    runner.note(
        "lid_spot",
        xyz=None if chosen is None or stop is not None else [round(v, 4) for v in chosen[0].xyz],
        tier=None if chosen is None else chosen[1],
        mapped=search.tmap is not None,
        refused="; ".join(got.refused) or None,
        candidates_s=round(time.monotonic() - t0 - got.carry_s, 3),
        ik_s=round(got.ik_s(), 3),
        spot_checks=got.checks(),
        carry_s=round(got.carry_s, 3),
        flight_s=None if flight is None else round(flight, 2),
        time_left_s=None if left_s is None else round(left_s, 1),
        timed_out=got.timed_out,
        no_land=no_land_note(ctx, None if chosen is None else chosen[0]),
        table=table_note(ctx, search.table_z, search.spot_z, got.passed),
        spot=search.spots + 1,
        left=[
            {"spot": s.n, "xyz": [round(float(v), 4) for v in s.drop.xyz], "tier": s.tier, "verdict": s.verdict}
            for s in search.left
        ],
        searched=got.searched,
        looked_again=looked or got.looked,
    )
    if got.carry is None and got.timed_out:
        return put_lid_back(
            ctx,
            cfg,
            runner,
            "no table was felt under the lid (%s), and no other spot is tried: the run's time ran out while one "
            "was looked for (%.0f s are left; a spot and the way back to the box after it take %.0f)"
            % (tried, max(left_s or 0.0, 0.0), SPOT_S + BOX_S + END_S),
        )
    if got.carry is None:
        return put_lid_back(
            ctx,
            cfg,
            runner,
            "no table was felt under the lid (%s), and no other spot on the table the cameras measured has a "
            "carry that plans (%s)" % (tried, "; ".join(got.refused) or got.passed.says()),
        )
    if stop is not None:
        return put_lid_back(
            ctx, cfg, runner, "no table was felt under the lid (%s), and no other spot is tried: %s" % (tried, stop)
        )
    search.spots += 1
    nxt, tier2, says = chosen
    _say_spot(
        ctx,
        eyes=search.eyes,
        chosen=(nxt, tier2, "%s; spot %d of this run, after %s" % (says, search.spots, tried)),
        refused=got.refused,
        tmap=search.tmap,
        table_z=search.table_z,
    )
    _video(
        ctx,
        "SPOT %d: CARRY to [%.2f, %.2f] (%s) and SET DOWN, let go where it meets the table"
        % (search.spots, nxt.xyz[0], nxt.xyz[1], tier2),
    )
    log.info(
        "[press_demo] SPOT %d, ONE MOTION: %s — to [%.2f, %.2f] and down onto the table, %s"
        % (
            search.spots,
            ", ".join(lg.name for lg in got.carry.legs),
            nxt.xyz[0],
            nxt.xyz[1],
            _touch_says(got.carry, search.after_this(flight)),
        )
    )
    return _fly_carry(ctx, cfg, runner, got.carry, nxt, tier2, search)


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


def _release_and_home(ctx, cfg, runner, carry, drop, tier, home, why, search=None):
    """The set-down of `carry` at `drop` flown: the knuckle read (the lid
    slipped?), and what its watch felt (carry.contact; first it reads what
    the client kept of the flight's feedback and no poll read: read_rest).
    The fingers open only where it felt the lid meet the table, at about the
    height the run has for it (SetdownContact.credible): then the release,
    the retreat starting as the fingers open (defer_join: the pads leave the
    knob in the first millimetres, the retreat's first ones come slower than
    that); then `home` — the legs after the release planned ahead. None, or
    one the retreat from where the arm stands ends too far from (`why`):
    home is planned while the retreat flies, from where it ends (the
    runner's lookahead: nothing is watched on that execution), and where
    that fails, after it. Slow and under MEDIUM as every home flies
    (core.plan_home). No such contact, the lid still in the fingers: it is
    not let go. It goes on to another spot (another_spot, with `search`:
    the SpotSearch this spot came from, or one made of what chose it ahead)
    where the watch watched the touch and felt no table here, or something
    held the lid up; and back onto the box (carry_back) where there is no
    search to go on with, or the set-down could not be watched at all (no
    watch, none of its stretch fed, a watch that failed: that says nothing
    of this spot, and the next set-down would fly as blind, its touch to
    its planned end with nothing watching for the table). A lid that slipped
    out on the way (the knuckle outside the grip band): the fingers open,
    the arm goes home, and the run ends STOP — no "the lid met the table" is
    said of it, whatever the watch felt. Returns the exit code: 0 with the
    lid let go on the table and the arm home."""
    world = _full_world(ctx)
    ok_q, pos, _stalled = ctx.client.gripper_cmd(None)  # query only: still on the knob?
    kept = not ok_q or pos is None or in_band(pos, cfg.grip_band)
    watch = getattr(carry, "contact", None)
    read_rest = getattr(watch, "read_rest", None)
    if read_rest is not None:
        read_rest()  # the feedback no poll read: the watch says what ALL of the flight's kept samples say
    watched = None if watch is None else watch.record()
    met = watch is not None and bool(watch.credible())  # the arm felt the lid meet the table where one can be
    refused = ("never", "the set-down was not watched") if watch is None else watch.refusal()
    # the driver's estimate of the force on the tool as the lid stands on the table, before the
    # fingers open (the last 0.2 s; the lid's own weight is in it), and its peak over the carry
    ee = getattr(ctx.client, "ee_force", None)
    last = getattr(ctx.client, "last_execution", None) or {}
    driver_force = None
    if ee is not None:
        at = ee(span_s=0.2)
        over = ee(since=last.get("sent_at")) if last.get("sent_at") is not None else None
        if at is not None or over is not None:
            driver_force = {"at_release": at, "over_the_carry": over}
    spots = 1 if search is None else search.spots  # which of the run's spots this set-down was at
    runner.note(
        "reach_setdown",
        knuckle=None if pos is None else round(float(pos), 3),
        kept=kept,
        credible=met,  # the lid met the table: the one case the fingers open in
        contact=watched,
        driver_force=driver_force,
        spot=spots,
        xyz=[round(float(v), 4) for v in drop.xyz],
    )
    if not met and kept:
        felt = refused[1] if refused else "nothing says the lid met the table"
        if search is None:
            return carry_back(ctx, cfg, runner, carry, felt)
        blind = watch is None or refused is None or refused[0] == "failed" or not (watched or {}).get("watched", True)
        if blind:
            return carry_back(
                ctx,
                cfg,
                runner,
                carry,
                felt,
                last_resort="no other spot is tried: a set-down that could not be watched says nothing of this "
                "spot, and the next would fly as blind",
            )
        above = (watched or {}).get("above_top_mm") if refused[0] == "held" else None
        return another_spot(
            ctx,
            cfg,
            runner,
            search,
            carry,
            drop,
            tier,
            (refused[0], felt),
            above_m=None if above is None else 0.001 * float(above),
        )
    if met and kept:
        # only of a lid still in the fingers: the flight view and the node read this line as the lid on the table
        log.info(
            "[press_demo] SET-DOWN CONTACT: the lid met the table %+.0f mm from where the plan put it; %s%s"
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
                " (seen only in the feedback read once the flight was over: nothing was cut)"
                if watched.get("seen_after_flight")
                else "",
            )
        )
    if not kept:
        # the fingers have shut past the knob: the lid left them on the way, and whatever the watch felt at
        # the table was not the lid in the fingers meeting it. Nothing here says where the lid lies. Until
        # 2026-11-17's review a contact printed "SET-DOWN CONTACT: the lid met the table" first, and the
        # flight view said "lid: on the table" of a lid that had slipped out
        log.info(
            "[press_demo] the fingers read %.3f at the table, outside the grip band %s: the lid slipped on the way%s"
            % (
                pos,
                list(cfg.grip_band),
                " (the set-down felt a contact %+.0f mm from where the plan put the table: not the lid's, which "
                "had left the fingers)" % watched["above_top_mm"]
                if met
                else " (and the set-down felt no table: %s)" % (refused[1] if refused else "not watched"),
            )
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
    # no home in hand (none was planned ahead: the spot chosen after the lift, whose set-down hosts no planning;
    # or the one planned ahead starts too far from where this retreat ends): planned while the retreat flies,
    # from where it ends, as the way back to the look pose hosts its home (_press_and_back) — it stood at the
    # hover for the plan before
    host = None if home_legs else (lambda end: rp._home_legs(ctx, cfg, start=end))
    res = runner.run([release, *up], execute=True, lookahead=host)
    home_ahead = None if host is None else runner.lookahead_result
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
        home_legs, home_ahead = [], None  # planned from where the arm stands now
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
        # the home planned while the retreat flew, where the arm stands where it starts (reach_press
        # ._planned_ahead); else planned here, at rest
        ok = rp._go_home(ctx, cfg, runner, "home from the lid (%s)" % why if why else "", ahead=home_ahead)
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
    # only ever said of a lid the arm felt meet the table (met; a lid that slipped ended above)
    log.info(
        "[press_demo] DONE — box open, lid on the table at [%.2f, %.2f] (%s)%s, let go where the arm felt it meet "
        "the table, arm home"
        % (
            drop.xyz[0],
            drop.xyz[1],
            tier,
            "" if spots == 1 else ", spot %d of this run (no table was felt at the %d before it)" % (spots, spots - 1),
        )
    )
    return 0
