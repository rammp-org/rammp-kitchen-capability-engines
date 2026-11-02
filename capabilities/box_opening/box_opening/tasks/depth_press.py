"""The depth press (the reach-out's press, press_demo's run): once the hand
is over the box, how far down — and whether the button went — from the
wrist camera's own depth and the arm's own joint torques.

Getting over the box is the mission's own, calibrated way: the scene camera,
the wrist's aim through TF, then the reach-out's slide in
(tasks/reach_press). Flatbot 2026-10-22: that put the fingertip over the
button, but 6 in too high — the stroke went down to where TF put the lid
and met nothing. TF is good enough to get there; the height is the one
thing it got wrong. So from over the box:

  TURN    the hand straight down where it stands, turning about the
          fingertip so it stays over the button (the reach-out's tilted
          hand; one the slide in already turned down is not turned again).
          If no straight-down hand reaches there — up to 5 deg short, a
          compliant press's up to 15 (COMPLIANT_TURN_SHORT_DEG) — it
          stops. A compliant reach-out solves the turn and its stroke
          before it flies toward the box (reach_press.way_down, from
          turn_walk and compliant_stroke), and flies them in the same
          execution as its re-aim and its slide in (reach_press.way_in_legs).
  DEPTH   the wrist camera, looking down along the tool, measures how far
          below its lens the lid is: the button's own ring if it sees it,
          else where the caller aimed the button (the reach-out's tilted
          look), backed by the surface straight under the fingers' line.
          Its own depth in its own frame, no transform.
  TOUCH   straight down, at most that far plus travel_m — the fingertips are
          never behind the lens — stopped by the first contact the joint
          torques feel. Where it stopped says how far in front of the lens
          the fingertips are: printed, every run.
  PUSH    button_travel_m further: the torques feeling the button bottom
          out is the press. The whole travel without a stop pressed nothing.
  RISE    straight back up to where the touch started, a pull at the
          fingertip stopping it (the reach-out's unload and its checks); a
          compliant reach-out rises back up its own stroke, in one
          execution with its clear and home (rise_back_up,
          reach_press._way_home),
  LOOK    and the camera judges the knob and whether the box moved.

What the touch measured is the box from then on: ctx.cpose is put where
the fingertips met the button (TF's lid was 76 mm high on 20:04, so every
home was planned from "inside" a box that was not there). After a contact,
a wrist camera that no longer finds the box holds the arm unless the scene
camera finds it (after_scene_check), the reach-out's own AFTER rule.
"""

import dataclasses
import math
import time
from dataclasses import dataclass

import numpy as np

from box_opening import LOGGER as log
from box_opening.constants import (
    COMPLIANT_RUN_GAINS,
    CONTACT_SPEED,
    FREE_LINE_SPEED,
    IMPEDANCE_KQ,
    SANITY_MARGIN_RAD,
    TCP_OFFSET_M,
)
from box_opening.models.container import ContainerPose
from box_opening.primitives.core import CONTACT_SHIFT_PAD_M, _full_world, _push_guard_and_verify
from box_opening.runtime.guards import GuardSpec, sanity_violations
from box_opening.runtime.kinematics import mat_to_quat_xyzw
from box_opening.tasks import reach_press as rp

DOWN_CONTACT_TOOL = (0.0, 0.0, TCP_OFFSET_M)  # the closed pads straight down, in tool_frame (the straight-down press's)
TURN_STEP_M = 0.005  # the turn's waypoints: this far apart ...
TURN_STEP_RAD = math.radians(2.0)  # ... and this much turn
TURN_MAX_JUMP_RAD = 0.15  # a bigger joint step between waypoints is no path to fly
# How far short of straight down the turn may stop, tried in order: exactly
# down can sit on the wrist's singularity (flatbot, the box 0.60 m out: a
# 0.26 rad jump in the last step). The stroke is vertical whatever the hand's
# lean, and the lens stays parallel to the fingers.
TURN_SHORT_DEG = (0.0, 2.0, 5.0)
# ... and a COMPLIANT press's turn in place after the slide, smallest lean
# first. Flatbot 2026-10-27 16:23: the slid-in fingertip [0.7213, -0.297,
# 0.3725] stood 0.780 m out, where no hand within 8 deg of straight down is
# found (0 of 120 random IK seeds at 0/2/5/7/8 deg, 9-14 of them at 10-20),
# so the turn refused and nothing was pressed; 12 deg short it flies. That
# press never looks with the wrist, so nothing needs the lens straight down,
# and its stroke is vertical at any lean. The lean costs sideways push: on
# pads that do not slide, the STIFF spring pushes sideways 0.17/0.20/0.28/
# 0.32/0.37/0.42 N per newton down at 0/5/8/10/12/15 deg over 15:19's box,
# 0.28/0.40 at 12/15 over 16:23's (flatbot's URDF), mostly along the line
# to the arm. TURN_SHORT_DEG's come first: every box they reach keeps its
# lean. The guarded press (_press_down) keeps TURN_SHORT_DEG: its close-up
# look and DEPTH assume the lens points down, and 5 deg already costs
# ~14 mm (_off_in_world).
COMPLIANT_TURN_SHORT_DEG = (0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0)
# An in-place turn (turn_down_path) whose joint arc — the sum of its steps'
# lengths in joint space — is longer than this is refused, and the next lean
# tried. Flatbot's URDF, boxes 0.64-0.82 m out under a lid at 0.3125: every
# turn that flies at the 30 deg tilt they plan at runs 0.49-1.91 rad (16:23's
# 12 deg one 1.46, the 10 deg one at its scene button 1.77); past 2.0 only at
# 35-45 deg, most of those swinging joint_1 about 1 rad over the box (the
# walks in from 16:23's re-aimed look at 10-15 deg swing it 1.6-1.9 rad,
# 2.65-3.21 rad of arc). Under lids at 0.22-0.40, at 30 deg, two turns go
# past it, both 0.78 m out under 0.22: exactly down, 2.04-2.10 rad with
# joint_1 swinging ~1.05; they turn 2 deg short instead, 1.84-1.86 rad.
TURN_MAX_ARC_RAD = 2.0
# A compliant press leaning this far or more says so, and its run record
# (depth_compliant_press) keeps the lean
LEAN_SAID_DEG = 5.0
LOOK_S = 2.0  # still frames are gathered this long at each look
LOOK_SETTLE_S = 0.3  # ... after the arm stopped
LOOK_HITS = 2  # sightings that agree within LOOK_AGREE_M make a look
LOOK_AGREE_M = 0.003
# A look that knows where the button must be (look's `expect`: after a
# centring move, the hand moved level by what the camera asked; after the
# press, back where the last look stood) takes no circle farther than this
# from there, nor a lid this much nearer or farther. Flatbot 2026-10-23 17:30:
# the look after a 21 mm move took a circle on the cabinet 288 mm away and
# 99 mm deeper in the frames Hough missed the button in, two of them agreed,
# and the run stopped "the camera's axes in TF are wrong?" with the button
# right under the fingers. A box the press slid farther than this is lost to
# the wrist camera, and the scene camera is asked (after_scene_check).
LOOK_EXPECT_M = 0.03
LOOK_EXPECT_DEPTH_M = 0.03
# The caller's aim (press_down_from_here's `aimed`: the reach-out's tilted
# look) is where the first look expects the button too. Flatbot 2026-10-27
# 14:33: the fingers stood over the aimed button, the first close-up took a
# circle on the lid's edge 64 mm off it, and CENTRE flew the hand off the box.
# Turned down there, the lens 65 mm beside the fingers sees the button under
# them at the frame's top edge, its ring cut off (77 % of it at lid height,
# 80 % needed): where no look confirms it, the aim stands in, if the camera's
# own depth under the fingers' line is within LOOK_EXPECT_DEPTH_M of the
# aimed button and the button within LOOK_EXPECT_M of under the fingers.
LENS_AHEAD_MAX_M = 0.25  # the fingertips are at most this far in front of the lens
# CENTRE: from over the box the fingers are put over the button the wrist
# camera sees — TF's aim from the tilted look landed the press 45 mm off its
# centre, on the lid's edge (flatbot 2026-10-22, run 20:04). Straight down,
# only the lens's sideways offset from the fingers matters, not where it
# sits along the tool, which is what tilted views get most wrong.
CENTRE_TOL_M = 0.003  # the button this close in world xy to the fingertip's vertical: press
CENTRE_MAX_M = 0.10  # farther than this off is not a re-centre: stop
CENTRE_MAX_MOVES = 2
CENTRE_SPEED = FREE_LINE_SPEED  # the centring move: free air, the fingertips well above the lid
BOX_MOVED_M = 0.005  # the button this far from where the camera saw it before: the box moved
LID_MOVED_M = 0.008  # ... or the lid this much nearer or farther
KNOB_UP_MM = 6.0  # press_demo.KNOB_UP_MIN_MM
# The lens offset CENTRE moves by is TF's mount, and the look after the move
# uses the same offset, so a wrong mount "converges" anyway, its whole error
# off the button (20:04: TF's lens y 0.100 against a true ~0.065 — the press
# 35 mm off, on the lid's edge). CENTRE runs only when TF's lens agrees with
# the measured one in the camera config (wrist_lens_in_hand_m) ...
MOUNT_XY_TOL_M = 0.005
MOUNT_Z_TOL_M = 0.020
# ... and the touch must meet the lid where TF's lens says the fingertips
# are in front of it, or nothing is pushed.
AHEAD_TOL_M = 0.020

# The camera's own frame with z turned up, so find_button's heights are
# minus depths below the lens: a virtual camera at the origin looking
# straight down. Nothing in it is a mount or a transform.
_LENS_DOWN = np.diag([1.0, -1.0, -1.0])

# TOUCH: the stroke flies at press_speed to TOUCH_SLOW_M above where the
# camera's depth and TF's mount put the lid, then at TOUCH_SPEED onto it.
# Flatbot 2026-10-23 16:02: all 112 mm at 0.35 met the lid moving — the
# 3 Nm touch tripped at 12.4 Nm in one sample, the push began 9 Nm loaded,
# and the box slid 6 mm. The slow stretch is the mount's own allowance
# (AHEAD_TOL_M) and some depth noise.
TOUCH_SLOW_M = AHEAD_TOL_M + 0.015
TOUCH_SPEED = CONTACT_SPEED
# A box the press slid this little, still standing (its lid's depth within
# LID_MOVED_M), is gripped where the camera sees it now; farther is a STOP
SLIDE_OK_M = 0.015

# A COMPLIANT press (press_down_from_here's `compliant`; owner 2026-10-27:
# "with impedance mode, lets remove the safety check for now that stops the
# arm when there's a bad reading or it thinks it hit or is going to hit
# something"): one stroke down, in the driver's joint impedance, to where
# its springs push COMPLIANT_PRESS_N straight down through the button's
# bottom (its end bent past the bottom: BEND_MAX_M), held push_hold_s, and
# back up. No close-up look, no stop at contact, no check: the arm yields.
# 15:19's touch tripped at 3.4 Nm on joint 4, ~13.5 N at the pads, and the
# button went.
COMPLIANT_PRESS_N = 15.0
# The compliant stroke's approach and press fly at this fraction of
# cfg.press_speed and TOUCH_SPEED: 0.175 and 0.075 for the pink box (owner
# 2026-10-28: "make the arm press down half as slow as it is now", half the
# speed)
COMPLIANT_STROKE_SLOWDOWN = 0.5
# the compliant run's gains (constants.GAINS_RIGID): ~1130 N/m straight down at
# the pads on 15:19's button's bottom (STIFF's was ~450)
COMPLIANT_GAINS = COMPLIANT_RUN_GAINS
# The driver's joint stiffness under that spec, N·m/rad (constants
# IMPEDANCE_KQ, kinova-gen3-driver interface/gains.cpp): the robot's, not the
# scene's. Taken from the profile the press legs send, so the stiffness at the
# pads, the overshoot, the stroke's bend and press_n always model the spring
# that flies: COMPLIANT_KQ must stay that spec's. The bend's direction is
# Kq^-1 J_z^T (stroke_bend): a spring stiffer or softer by one factor on every
# joint still pushes straight down from the same end, only harder or softer;
# one whose ratio between joints 1-4 and 5-7 is not this one's pushes sideways
# again.
COMPLIANT_KQ = IMPEDANCE_KQ[COMPLIANT_GAINS]
# how far the stroke's end lies below the button's bottom, whatever the
# stiffness: never a graze, never a ram
COMPLIANT_OVERSHOOT_M = (0.015, 0.06)
# the springs held pushing at least this much down at the pads (compliant_hold):
# something pushed back
COMPLIANT_FELT_N = 5.0
# The stroke's end BENT (stroke_bend). The driver's spring is a joint spring,
# tau = g + Kq (q_ref - q) - D (qd - qd_ref), with no Cartesian term
# (kinova-gen3-driver joint_impedance_mode.cpp, whose header says the tip
# stiffness "is not diagonal in the task frame"): the pads' compliance
# J Kq^-1 J^T couples, and a reference driven straight down past the button
# pushes sideways too. Flatbot's URDF, STIFF, the stroke as it flew before
# the bend, the pads held where they met the button's bottom: 17.3 N down
# and 5.3 N sideways at 15:19's box turned exactly down (2026-10-27; free
# to slide, they slip 13 mm across its 44 mm button), 23.3 N and 12.2 N at
# 16:23's 12 deg lean (an 18 mm slip), while k(q0) x the offset said 15 N at
# both. So the press leg goes straight down only to the button's bottom, at
# joints q_b, then on in joint space to q_end = q_b - F Kq^-1 J_z(q_b)^T:
# held at the bottom the arm stands at q_b, and the springs push F straight
# down there, nothing sideways (offline, both boxes: 15.00 N, 0.00 N). The
# model is the URDF and the profile's Kq: no friction, no gravity-model
# error. A button bottoming 5 mm below or 5-10 mm above where top_z puts
# its bottom leaves the pads 0.7-3.3 N sideways bent, 4.3-14.2 N straight
# (both boxes). In free air the pads would end 11 mm (15:19) to 14 mm
# (16:23) aside and 28-32 mm below the bottom, the hand turned ~4 deg off
# the attitude it held down to there (the bend's waypoints are no IK
# solutions). Over flatbot's workspace (boxes 0.50-0.82 m out, lids 0.25-
# 0.38, 0-15 deg short) every one of the 223 strokes its URDF solves bends,
# 4-23 mm aside; a bend that would move them more than BEND_MAX_M sideways
# is near a degenerate compliance, and the stroke flies straight, as before,
# and says why.
BEND_MAX_M = 0.03
BEND_STEP_RAD = 0.005  # the bend's waypoints: no joint moves more than this between two
# a bend dropping the pads less than this has nothing to bend (a chain whose
# pads barely move with its joints, as the test fakes' that stand still)
BEND_MIN_DROP_M = 0.001
# A compliant reach-out's way back up from the hold (rise_back_up): the
# stroke's own samples reversed, joined at the first one whose pads stand
# this far above where the spring held them, the held arm's step to it no
# joint longer than RISE_FIRST_STEP_MAX_RAD and nowhere lower than the held
# pads (RISE_CHECK_POINTS along it). The spring holds the arm 0.040-0.052
# rad off the stroke's commanded end, mostly on joints 2 and 4 (flatbot's
# URDF, STIFF, the pads held 32-33 mm above the end in 16:23's and 15:19's
# geometry, sliding on the button or held where they met it): past the
# runner's DRIFT_REPLAN_RAD 0.04, so a way up from the commanded end is
# refused and the arm stays pressing on the button. And the stroke's sample
# nearest the held arm in joint space lies near its END: a way up joined
# there (the recoil's reverse_tail) drives the pads 23-30 mm lower first, a
# second press. Joined 16-17 mm up instead, the step runs 0.14-0.18 rad and
# the pads only rise. Bent (stroke_bend), the arm held at the button's bottom
# is the stroke's own sample there, 0.035-0.046 rad off the commanded end,
# and joined 16 mm up its step runs 0.05-0.06 rad; flown straight, or held
# anywhere but at the bottom, it is as above.
RISE_JOIN_ABOVE_M = 0.015
RISE_FIRST_STEP_MAX_RAD = 0.25
RISE_CHECK_POINTS = 10


@dataclass
class LensSighting:
    """The button in the wrist camera's own frame (the hand's axes)."""

    x: float  # the button's centre from the lens axis, along the hand's x (m)
    y: float  # ... and its y
    lid_depth: float  # the lid ring below the lens (m), along the axis
    button_depth: float  # the button top below the lens
    up_mm: float  # the button above its own lid (refuse_open_box), or nan
    raw: object  # the find_button sighting, for the saved frame


def lid_depth_guesses(depth, n=4):
    """The nearest few flat surfaces in the frame, nearest first: the lid is
    the highest thing there, and may fill only a corner of the frame (1%)."""
    d = np.asarray(depth, float)
    d = d[np.isfinite(d) & (d > 0.08) & (d < 1.5)]
    if d.size < 200:
        return []
    hist, edges = np.histogram(d, bins=np.arange(0.08, 1.505, 0.005))
    order = [i for i in np.argsort(-hist) if hist[i] >= 0.01 * d.size][: 3 * n]
    out = []
    for i in sorted(order, key=lambda i: edges[i]):
        c = float(np.median(d[(d >= edges[i]) & (d < edges[i + 1])]))
        if all(abs(c - o) > 0.01 for o in out):
            out.append(c)
    return out[:n]


def find_in_camera(color, depth, k, model, dist=None, expect=None):
    """The button in one frame from a camera looking straight down along the
    tool, in that camera's own frame: (LensSighting, None) or (None, why).
    find_button's straight-down aim, handed the lens itself as its world.
    `expect`: (x, y, lid depth) where the button must be, in the same frame —
    the lid looked for at that depth only, and no circle farther than
    LOOK_EXPECT_M from there or LOOK_EXPECT_DEPTH_M off its lid depth."""
    from box_opening.perception.button_aim import find_button

    if color is None or depth is None or k is None:
        return None, "no frame"
    whys = []
    guesses, prior, reach = lid_depth_guesses(depth), None, None
    if expect is not None:
        # find_button's world is the lens turned down (_LENS_DOWN): its y is the camera's -y
        guesses, prior, reach = [float(expect[2])], (float(expect[0]), -float(expect[1])), LOOK_EXPECT_M
    for d0 in guesses:
        s, why = find_button(
            color,
            depth,
            k,
            _LENS_DOWN,
            (0.0, 0.0, 0.0),
            -d0 - float(model.dims[2]),
            model,
            dist=dist,
            prior_xy=prior,
            prior_max_m=reach,
        )
        if s is None:
            whys.append("at %.0f mm: %s" % (1000 * d0, why))
            continue
        up = float(s.above_lid_m)
        top = -float(s.xyz[2])
        seen = LensSighting(
            x=float(s.xyz[0]),
            y=-float(s.xyz[1]),
            lid_depth=top + up if np.isfinite(up) else top,
            button_depth=top,
            up_mm=1000.0 * up,
            raw=s,
        )
        if expect is not None and abs(seen.lid_depth - float(expect[2])) > LOOK_EXPECT_DEPTH_M:
            whys.append(
                "at %.0f mm: its lid %.0f mm below the lens, not the %.0f it must be"
                % (1000 * d0, 1000 * seen.lid_depth, 1000 * float(expect[2]))
            )
            continue
        return seen, None
    return None, "; ".join(whys) if whys else "no flat surface in depth"


# lens_in_hand at rest: when TF has no pose yet at the newest frame's stamp,
# the pose this much earlier — never before the arm stopped — is the same one
STILL_POSE_BACK_S = 0.15


def lens_in_hand(ctx, grab, still_since_ns=None):
    """Where the wrist lens is from the tool frame, in the hand's own axes:
    (lens xyz (m), R_hand_cam — the camera's axes in the hand's), from the
    camera's pose at its newest frame (TF: the arm's model and the camera's
    mount) against the arm's own kinematics at the joints it holds now —
    read while the arm is still, so the two are one moment. (x, y) is how
    far the lens axis sits beside the fingers' line, z how far along the
    tool (negative: behind the tool frame). R_hand_cam is the identity on
    flatbot's OAK, 180 deg about the lens axis on the bench's D405. None
    without a camera pose.

    `still_since_ns`: the arm has not moved since then (look()'s frames
    count from it). TF can lag the newest frame's stamp, and the lookup at
    it fails — both of a look's two frames did at 16:51 on 2026-10-23, and
    the depth press stopped "no camera pose" right over the box — so the
    pose STILL_POSE_BACK_S before it, never before the arm stopped, stands
    in: the camera has not moved in between."""
    from box_opening.perception.depth_source import camera_pose_at
    from box_opening.runtime import approach

    cam = camera_pose_at(grab)
    if cam is None and still_since_ns is not None and getattr(grab, "color_stamp", None) is not None:
        from rclpy.time import Time

        back = max(int(still_since_ns), _stamp_ns(grab.color_stamp) - int(STILL_POSE_BACK_S * 1e9))
        cam = camera_pose_at(grab, stamp=Time(nanoseconds=back).to_msg())
    if cam is None:
        return None
    rot, trans = cam
    R, t = approach._chain().fk(ctx.client.joints())
    R = np.asarray(R, float)
    lens = [float(v) for v in R.T @ (np.asarray(trans, float) - np.asarray(t, float))]
    return lens, R.T @ np.asarray(rot, float)


def depth_under_fingers(depth, k, lens, R_hand_cam=None):
    """The depth of whatever is straight under the fingers' line, from a lens
    beside them looking parallel (m), or None. `lens`, `R_hand_cam`: the
    lens's offset from the fingers' line and its axes, in the hand
    (lens_in_hand). The line's point at depth Z is at (-x, -y, Z) from the
    lens in the hand's axes, R_hand_cam.T of that in the camera's; walked
    from far to near, the surface under the fingers is where the depth read
    at that point's pixel equals Z."""
    kk = np.asarray(k, float)
    d = np.asarray(depth, float)
    h, w = d.shape
    r = np.eye(3) if R_hand_cam is None else np.asarray(R_hand_cam, float)
    p = r.T @ np.array([-float(lens[0]), -float(lens[1]), 0.0])
    px, py = float(p[0]), float(p[1])
    prev = None
    for z_line in np.arange(1.2, 0.12, -0.002):  # far to near: the point leaves the centre for the frame's edge
        u = kk[0, 0] * px / z_line + kk[0, 2]
        v = kk[1, 1] * py / z_line + kk[1, 2]
        if not (2.0 <= u < w - 3 and 2.0 <= v < h - 3):
            break  # out of the frame, and it only goes farther out
        ui, vi = int(round(u)), int(round(v))
        win = d[vi - 2 : vi + 3, ui - 2 : ui + 3]
        win = win[np.isfinite(win) & (win > 0.08) & (win < 1.5)]
        if win.size < 5:
            continue
        # < 0 while the line's point is beyond the surface seen there, >= 0
        # once it is nearer than that surface
        gap = float(np.median(win)) - float(z_line)
        if prev is not None and prev[0] < 0.0 <= gap:
            g0, z0 = prev
            return z0 + (float(z_line) - z0) * (-g0) / (gap - g0)  # where the gap closes
        prev = (gap, float(z_line))
    return None


def _stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def look(node, watcher, model, tag, runner=None, ctx=None, settled=False, expect=None):
    """Still frames after the arm stopped, until LOOK_HITS agree within
    LOOK_AGREE_M. Returns (LensSighting or None, depth under the fingers or
    None, the lens in the hand (lens_in_hand's pair) or None, status) —
    the median of the agreeing sightings. Without `ctx` there is no offset,
    and no depth under the fingers. `settled`: an at-rest read (rp._settled,
    0.45 s still) came just before, so frames count from now, without
    LOOK_SETTLE_S more. `expect`: where the button must be (find_in_camera),
    or a function of the lens pair giving that — frames then count once the
    lens is placed."""
    from box_opening.tasks.press_demo import _spin_detect, save_aim_frame

    g = watcher.grab
    t_end = time.monotonic() + (0.0 if settled else LOOK_SETTLE_S)
    while time.monotonic() < t_end:
        _spin_detect(node)
    after = node.get_clock().now().nanoseconds
    seen, under, last, why, frames, lens = [], [], None, "no wrist frame", 0, None
    t_end = time.monotonic() + LOOK_S
    while time.monotonic() < t_end:
        _spin_detect(node)
        if g.color is None or g.depth is None or g.k is None or g.color_stamp is None:
            continue
        st = _stamp_ns(g.color_stamp)
        if st <= after or st == last:
            continue  # shot while the arm was still moving, or seen already
        last = st
        frames += 1
        if lens is None and ctx is not None:
            lens = lens_in_hand(ctx, g, still_since_ns=after)
        z = None if lens is None else depth_under_fingers(g.depth, g.k, lens[0], lens[1])
        if z is not None:
            under.append(z)
        want = expect
        if callable(expect):
            if lens is None:
                why = "no camera pose to place the aimed button by"
                continue
            want = expect(lens)
        s, why = find_in_camera(g.color, g.depth, g.k, model, dist=getattr(g, "dist", None), expect=want)
        if s is None:
            continue
        seen.append(s)
        near = [o for o in seen if math.hypot(o.x - s.x, o.y - s.y) < LOOK_AGREE_M]
        # with ctx, the sighting is no use without the lens to place it by:
        # keep looking until the camera's pose is had too
        if len(near) >= LOOK_HITS and (lens is not None or ctx is None):
            med = LensSighting(
                x=float(np.median([o.x for o in near])),
                y=float(np.median([o.y for o in near])),
                lid_depth=float(np.median([o.lid_depth for o in near])),
                button_depth=float(np.median([o.button_depth for o in near])),
                up_mm=float(np.nanmedian([o.up_mm for o in near]))
                if any(np.isfinite(o.up_mm) for o in near)
                else float("nan"),
                raw=s.raw,
            )
            save_aim_frame(g, s.raw, tag)
            z_under = float(np.median(under)) if under else None
            if runner is not None:
                runner.note(
                    tag,
                    lens_xy_mm=[round(1000 * med.x, 1), round(1000 * med.y, 1)],
                    lid_depth_mm=round(1000 * med.lid_depth, 1),
                    under_fingers_mm=None if z_under is None else round(1000 * z_under, 1),
                    lens_in_hand_mm=None if lens is None else [round(1000 * v, 1) for v in lens[0]],
                    up_mm=None if not np.isfinite(med.up_mm) else round(med.up_mm, 1),
                    hits=len(near),
                    frames=frames,
                )
            return med, z_under, lens, "%d/%d frames" % (len(near), frames)
    z_under = float(np.median(under)) if under else None
    if runner is not None:
        runner.note(
            tag,
            under_fingers_mm=None if z_under is None else round(1000 * z_under, 1),
            frames=frames,
            why=why,
            lens_in_hand_mm=None if lens is None else [round(1000 * v, 1) for v in lens[0]],
        )
    return None, z_under, lens, "%d frame(s), the button in none (last: %s)" % (frames, why)


def _in_hand(s, lens, R_hand_cam=None):
    """The button top from the tool frame, in the hand's axes (m): the camera
    sees it at (s.x, s.y, s.button_depth) in its own axes, from a lens at
    `lens` turned by `R_hand_cam` (lens_in_hand)."""
    r = np.eye(3) if R_hand_cam is None else np.asarray(R_hand_cam, float)
    b = r @ np.array([float(s.x), float(s.y), float(s.button_depth)])
    return np.array([float(lens[0]) + b[0], float(lens[1]) + b[1], (float(lens[2]) if len(lens) > 2 else 0.0) + b[2]])


def expect_after_move(s, lens, R_hand_cam, R, d_world):
    """Where the camera must see the button of sighting `s` once the hand has
    moved by `d_world` holding its attitude `R` (the hand's axes in the
    world): (x, y, lid depth) in the camera's own frame, for look(expect=)."""
    r = np.eye(3) if R_hand_cam is None else np.asarray(R_hand_cam, float)
    b = _in_hand(s, lens, r) - np.asarray(R, float).T @ np.asarray(d_world, float)
    c = r.T @ (b - np.asarray(lens, float)[:3])
    return float(c[0]), float(c[1]), float(c[2]) + float(s.lid_depth) - float(s.button_depth)


def where_seen(s):
    """look(expect=) for a look from where `s` was seen: (x, y, lid depth)."""
    return float(s.x), float(s.y), float(s.lid_depth)


def aim_seen(ctx, aimed, lens_pair, up_mm=None):
    """The button the caller aimed at `aimed` (its top, in the world) as the
    wrist camera must see it from where the hand stands now: a LensSighting
    in the camera's own frame, from the lens at `lens_pair` (lens_in_hand's),
    its lid `up_mm` (the aim's knob reading) below its top. raw is None: no
    frame here saw it."""
    from box_opening.runtime import approach

    lens, r_hc = lens_pair
    r = np.eye(3) if r_hc is None else np.asarray(r_hc, float)
    R, t = approach._chain().fk(ctx.client.joints())
    b = np.asarray(R, float).T @ (np.asarray(aimed, float) - np.asarray(t, float))  # from the tool frame, in the hand
    c = r.T @ (b - np.asarray(lens, float)[:3])
    up = float(up_mm) if up_mm is not None and np.isfinite(up_mm) else float("nan")
    return LensSighting(
        x=float(c[0]),
        y=float(c[1]),
        lid_depth=float(c[2]) + (up / 1000.0 if np.isfinite(up) else 0.0),
        button_depth=float(c[2]),
        up_mm=up,
        raw=None,
    )


def aim_disagrees(ctx, aim, lens_pair, z_under):
    """Why the aim (aim_seen) may not stand in for a look that did not
    confirm it, or None: the camera reads no surface under the fingers'
    line, or one farther than LOOK_EXPECT_DEPTH_M from the aimed button's
    depth, or the button is farther than LOOK_EXPECT_M from under the
    fingers."""
    if z_under is None:
        return "the camera reads no depth under the fingers to check the aim by"
    if abs(float(z_under) - aim.button_depth) > LOOK_EXPECT_DEPTH_M:
        return "the camera reads the surface under the fingers %.0f mm below the lens, the aimed button %.0f" % (
            1000 * float(z_under),
            1000 * aim.button_depth,
        )
    miss = math.hypot(*_off_in_world(ctx, aim, lens_pair[0], lens_pair[1]))
    if miss > LOOK_EXPECT_M:
        return "the aimed button is %.0f mm from under the fingers" % (1000 * miss)
    return None


def under_fingers(s, lens, R_hand_cam=None):
    """The button from the fingers' line, along the hand's x and y (m).
    (0, 0): the fingers' line runs through it."""
    b = _in_hand(s, lens, R_hand_cam)
    return float(b[0]), float(b[1])


def _off_in_world(ctx, s, lens, R_hand_cam):
    """The button top from the fingertip's vertical, in world xy (m): what
    the vertical stroke misses it by. Along the hand's axes it would be off
    by the depth below the fingertips times the lean when the turn stopped
    short (TURN_SHORT_DEG: 5 deg, 14 mm)."""
    from box_opening.runtime import approach

    R, t = approach._chain().fk(ctx.client.joints())
    R, t = np.asarray(R, float), np.asarray(t, float)
    b = t + R @ _in_hand(s, lens, R_hand_cam)
    tip = t + R @ np.asarray(DOWN_CONTACT_TOOL, float)
    return float(b[0] - tip[0]), float(b[1] - tip[1])


def _hand_line(ctx, dx, dy, dz, quat):
    """Joint waypoints moving the tool by (dx, dy) along the hand's own x and
    y and dz in world z, holding `quat`: ([], None) when that is no move."""
    from box_opening.runtime import approach

    R, t = approach._chain().fk(ctx.client.joints())
    d = np.asarray(R, float)[:, 0] * dx + np.asarray(R, float)[:, 1] * dy
    return rp._line(ctx, [float(t[0] + d[0]), float(t[1] + d[1]), float(t[2] + dz)], quat)


def down_quat(model, xy, short_deg=0.0):
    """The straight-down hand at `xy`, yawed as the reach-out's — leaning
    back toward the arm by `short_deg`."""
    return rp.reach_quat(model, xy, short_deg)


def turn_target(model, tip, short_deg):
    """The turn's end: (quat, tool_frame xyz) for the straight-down hand,
    `short_deg` short, with its pads (DOWN_CONTACT_TOOL) at `tip`."""
    quat = down_quat(model, (float(tip[0]), float(tip[1])), short_deg)
    return quat, np.asarray(tip, float) - rp._rot(quat) @ np.asarray(DOWN_CONTACT_TOOL)


@dataclass
class WayDown:
    """How the hand comes to point straight down over the button, solved
    before anything flies (reach_press.way_down)."""

    pts: list  # the joint waypoints that turn it; [] when the hand already points down
    quat: list  # its attitude at their end
    short_deg: float  # how far short of straight down (walked: TURN_SHORT_DEG; turned: COMPLIANT_TURN_SHORT_DEG)
    walked: bool  # True: one walk from the look pose (walk_in_down), else the turn after the slide
    # a turn lower than the slide's end (reach_press.TURN_LOW_ABOVE_M): the
    # slide carried on straight down to there, and the tool where it ends;
    # None where the hand turns at the slide's own end; `above_m` how far
    # above the button's top the fingertip stands there
    slide: list = None
    start_xyz: list = None
    above_m: float = None


@dataclass
class StrokePlan:
    """The compliant press's stroke from the hand turned down at `q0`
    (compliant_stroke): `legs` [(name, joint points, speed)], 'depth:approach'
    at press_speed to TOUCH_SLOW_M above the button's top — none when the
    hand already stands that low — then 'depth:press' at TOUCH_SPEED,
    straight down to the button's bottom and, `bent`, on in joint space to
    `q_end` (stroke_bend), where the springs push COMPLIANT_PRESS_N straight
    down at the pads held at the bottom; not bent, straight on down
    (`bend_why` says why). Either way the pads' reference ends at `end_z`,
    `over` below the button's bottom, and `k` (N/m) is the spring at the
    pads at the bottom."""

    legs: list
    k: float
    over: float
    end_z: float
    t0: object  # the tool at q0 (world)
    pad0: object  # ... and the pads (DOWN_CONTACT_TOOL)
    q0: list
    quat: list
    q_end: list = None  # the stroke's last joints: the springs' reference at the hold
    bent: bool = False
    bend_m: float = 0.0  # how far the bend moves the pads' reference sideways from the bottom (m)
    bend_why: str = None  # why the stroke is not bent


def joint_arc(pts):
    """How far a path of joint waypoints travels in joint space (rad): the
    sum of its steps' lengths."""
    p = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def line_refusal(chain, pts):
    """Why the runner refuses to fly the plan-free line through joint
    waypoints `pts`, in its own words, or None: Runner._refusal's
    trajectory sanity gate (guards.sanity_violations, SANITY_MARGIN_RAD) on
    the trajectory rp.line_leg flies them as. Runner.run gates each leg on
    its own trajectory before anything moves, merged into one execution or
    not, and a proof proves only what it flies: flatbot's URDF, r 0.70 m
    az +15 deg under a lid at 0.26, the walk in 5 deg short swings joint_1
    0.251 -> 0.687 -> 0.261, 0.437 rad against the gate's 0.361, and the
    way in proven with it was refused once the look and the aim had flown
    (merge review 2026-10-28: 3 of 119 walks in under the lid at 0.3125)."""
    from box_opening.runtime import approach

    if len(pts) < 2:
        return None  # no line: nothing to fly
    # the arm's chain names its joints as the driver does; the test fakes name none
    names = getattr(chain, "joint_names", None) or ["joint_%d" % (i + 1) for i in range(len(pts[0]))]
    bad = sanity_violations(approach.line_trajectory(names, pts), SANITY_MARGIN_RAD)
    return "trajectory sanity gate: " + "; ".join(bad) if bad else None


def turn_walk(chain, model, q, tip, shorts=TURN_SHORT_DEG, max_arc_rad=None):
    """The walk from joints `q` to the straight-down hand with its pads at
    `tip` (turn_target), each step solved from the last at TURN_STEP_M and
    TURN_STEP_RAD, no joint jumping more than TURN_MAX_JUMP_RAD between two
    — nor, with `max_arc_rad`, the whole walk longer than that in joint
    space (joint_arc) — and one the runner flies (line_refusal): `shorts`
    tried in order. Returns (points, quat, short, None) or (None, None,
    None, why)."""
    whys = []
    for short in shorts:
        quat, goal = turn_target(model, tip, short)
        pts, why = chain.walk(q, goal, quat, step_m=TURN_STEP_M, step_rad=TURN_STEP_RAD, max_step_rad=TURN_MAX_JUMP_RAD)
        arc = None if pts is None or max_arc_rad is None else joint_arc(pts)
        if arc is not None and arc > max_arc_rad:
            pts, why = None, "a turn of %.2f rad of joint arc" % arc
        if pts is not None:
            why = line_refusal(chain, pts)
            if why is None:
                return pts, quat, short, None
        whys.append("%.0f deg short: %s" % (short, why))
    return None, None, None, "; ".join(whys)


def walk_down_path(chain, model, q, start_xyz, quat):
    """walk_in_down's path from joints `q` (turn_walk's return): to the
    straight-down hand whose pads stand where the hand at `quat` puts its
    fingertip (rp.CONTACT_TOOL) at the slide's end `start_xyz`. Only
    TURN_SHORT_DEG, a compliant press's too: from 16:23's re-aimed look
    (2026-10-27) the walks 10-15 deg short "fly" only by swinging joint_1
    1.6-1.9 rad, the elbow travelling 247-312 mm, and the 10 deg one is
    near-singular — its worst step 0.138 rad of TURN_MAX_JUMP_RAD's 0.15.
    Such a box slides in and turns in place (COMPLIANT_TURN_SHORT_DEG)."""
    tip = np.asarray(start_xyz, float) + rp._rot(quat) @ np.asarray(rp.CONTACT_TOOL)
    return turn_walk(chain, model, q, tip, TURN_SHORT_DEG)


def turn_down_path(chain, model, q, contact_tool, shorts=TURN_SHORT_DEG):
    """turn_down's path from joints `q`, about the fingertip (`contact_tool`,
    in tool_frame) where fk puts it: ([], the hand's quat, 0.0, None) when
    it is within 1 deg of straight down already, else turn_walk's return,
    no turn longer than TURN_MAX_ARC_RAD of joint arc."""
    R, t = chain.fk(q)
    R = np.asarray(R, float)
    if math.degrees(math.acos(float(np.clip(-R[2, 2], -1.0, 1.0)))) < 1.0:
        return [], mat_to_quat_xyzw(R), 0.0, None  # straight down already
    tip = np.asarray(t, float) + R @ np.asarray(contact_tool, float)
    return turn_walk(chain, model, q, tip, shorts, max_arc_rad=TURN_MAX_ARC_RAD)


def turn_down(ctx, runner, contact_tool, ref, q=None, shorts=TURN_SHORT_DEG):
    """Turn the hand straight down where it stands (`q`, else the arm's
    joints), about its fingertip (`contact_tool`, in tool_frame), guarded:
    `shorts` (turn_down_path's) tried in order. Returns (quat, moved, None)
    — moved False when it already points down — or (None, False, why). A
    compliant turn (unguarded, faster: rp.free_guard, rp.free_line_speed)
    stopped part-way, over the button, lets go there (rp.let_go_part_way)."""
    from box_opening.runtime import approach

    q = ctx.client.joints() if q is None else q
    pts, quat, short, why = turn_down_path(approach._chain(), ctx.model, q, contact_tool, shorts)
    if pts is None:
        return None, False, "no straight-down hand reaches here (%s)" % why
    if not pts:
        return quat, False, None  # straight down already
    if short:
        log.info("[press_demo] TURN: %.0f deg short of straight down (no hand nearer straight down flies here)" % short)
    # in place, the fingertip the pivot: free air, nothing nearer the lid
    res = runner.run(
        [rp.line_leg(ctx, "depth:turn", pts, quat, rp.free_line_speed(ctx), rp.free_guard(ctx, ref), _full_world(ctx))],
        execute=True,
    )
    if any(not r.ok for r in res):
        rp.let_go_part_way(ctx, res)
        return None, False, "the turn met something (%s)" % res[-1].detail
    return quat, True, None


def walk_in_down(ctx, start_xyz, quat, q=None):
    """The reach-out's slide in and the TURN as one walk from where the arm
    stands (its look pose; `q`, else the arm's joints): to the straight-down
    hand whose pads stand where the hand at `quat` would put its fingertip
    (rp.CONTACT_TOOL) at the slide's end `start_xyz`, the tool in a straight
    line, the attitude turning on the way — TURN_SHORT_DEG tried in order,
    a compliant press's too (walk_down_path). Flatbot, run 20:04's button
    at 35 deg: 7.7 s at the slide's speed against 12.0 s for the slide and
    the turn, the fingertip within 0.6 mm of the slide's line and height.
    Returns (points, the down quat, None) or (None, None, why)."""
    from box_opening.runtime import approach

    q = ctx.client.joints() if q is None else q
    pts, down, short, why = walk_down_path(approach._chain(), ctx.model, q, start_xyz, quat)
    if pts is None:
        return None, None, why
    if short:
        log.info("[press_demo] TURN: %.0f deg short of straight down (no hand nearer straight down flies here)" % short)
    return pts, down, None


def touch_leg(ctx, cfg, pts, quat, reference, name="depth:touch", speed=None, arrive_ok=False):
    """The stroke down, guarded at contact_nm on the compensated torques
    against the free-air `reference`: every trip is the first contact.
    `arrive_ok`: the stroke's fast first stretch, which is meant to arrive
    (TOUCH_SLOW_M short of the lid) and touches only a lid higher than
    expected."""
    guard = GuardSpec(touch_nm=cfg.contact_nm, trip="touch", gravity=rp._wrist_gravity(), reference=reference)

    def verify(v):
        if v.outcome == "touch":
            return True, "contact%s" % ("" if v.torque_peak is None else " at %.1f Nm" % v.torque_peak)
        if v.outcome == "arrived":
            return (
                (True, "arrived above the lid")
                if arrive_ok
                else (False, "no contact all the way to the measured depth")
            )
        return False, "touch %s" % v.outcome

    return rp.line_leg(
        ctx, name, pts, quat, cfg.press_speed if speed is None else speed, guard, _full_world(ctx), verify
    )


def _rise(ctx, runner, quat, top_z, ref):
    """Straight back up to `top_z`: off the button UNLOAD_CHECK_M and checked
    at rest, then the rest of the way and checked again — judged for a pull
    at the fingertip against the free-air `ref` (the reach-out's unload).
    False: the arm holds, the reason printed."""
    from box_opening.runtime import approach

    _R, t = approach._chain().fk(ctx.client.joints())
    first = min(float(t[2]) + rp.UNLOAD_CHECK_M, float(top_z))
    # the rise only after the at-rest check 15 mm up found nothing on the hand
    for name, z, strict, speed in (
        ("depth:unload", first, False, CONTACT_SPEED),
        ("depth:rise", float(top_z), True, FREE_LINE_SPEED),
    ):
        _R, t = approach._chain().fk(ctx.client.joints())
        pts, why = rp._line(ctx, [float(t[0]), float(t[1]), z], quat)
        if pts is None:
            log.info("[press_demo] STOP: the way up is not a straight line the arm can fly (%s) — the arm holds" % why)
            return False
        if pts:
            res = runner.run(
                [
                    rp.line_leg(
                        ctx,
                        name,
                        pts,
                        quat,
                        speed,
                        rp.unload_guard(ctx.model, ref, rp.tip_load_at(pts[0])),
                        _full_world(ctx),
                    )
                ],
                execute=True,
            )
            if any(not r.ok for r in res):
                log.info("[press_demo] STOP: %s" % rp._unload_stopped(res))
                return False
        why = rp._still_free(
            ctx,
            runner,
            ref,
            "%.0f mm up" % (1000 * rp.UNLOAD_CHECK_M) if not strict else "where the touch began",
            strict,
        )
        if why is not None:
            log.info("[press_demo] STOP: %s The arm holds; check the box" % why)
            return False
    return True


def _touch_lid(ctx, cfg, runner, quat, top_z, over, depth, lens):
    """TOUCH: straight down from where the hand stands, fast to TOUCH_SLOW_M
    short of where the camera's `depth` and TF's `lens` put the lid, then
    slowly, until the torques feel it — checked against where these
    fingertips can meet it. Returns (the leg that met it, for the recoil; the
    pads' point by the arm's own kinematics; None), or (None, None, the
    DepthPress to stop with)."""
    from box_opening.models.container import from_container
    from box_opening.runtime import approach

    m = ctx.model
    reach = float(depth) + float(cfg.travel_m)
    # fast to TOUCH_SLOW_M short of where the depth and TF's mount put the lid
    fast_m = max(0.0, float(depth) - (float(DOWN_CONTACT_TOOL[2]) - float(lens[2])) - TOUCH_SLOW_M)
    touched = []
    if fast_m > 0.0:
        fast, why = _hand_line(ctx, 0.0, 0.0, -fast_m, quat)
        if not fast:
            return (
                None,
                None,
                DepthPress(
                    free=True, why="the stroke down is not a straight line the arm can fly (%s)" % (why or "no length")
                ),
            )
        log.info(
            "[press_demo] TOUCH: straight down, %.0f mm fast, then slowly, at most %.0f mm in all, until the torques "
            "feel it" % (1000 * fast_m, 1000 * reach)
        )
        touch = touch_leg(ctx, cfg, fast, quat, over, name="depth:approach", arrive_ok=True)
        touched = runner.run([touch], execute=True)
        if any(not r.ok for r in touched):
            return None, None, DepthPress(free=_rise(ctx, runner, quat, top_z, over), why=touched[-1].detail)
    else:
        log.info(
            "[press_demo] TOUCH: straight down, slowly, at most %.0f mm, until the torques feel it" % (1000 * reach)
        )
    if not touched or touched[-1].outcome != "touch":
        stroke, why = _hand_line(ctx, 0.0, 0.0, -(reach - fast_m), quat)
        if not stroke:
            return (
                None,
                None,
                DepthPress(
                    free=_rise(ctx, runner, quat, top_z, over),
                    why="the stroke down is not a straight line the arm can fly (%s)" % (why or "no length"),
                ),
            )
        touch = touch_leg(ctx, cfg, stroke, quat, over, speed=TOUCH_SPEED)  # the leg that meets the lid, for the recoil
        touched = runner.run([touch], execute=True)
    if touched[-1].outcome == "touch":
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
    if any(not r.ok for r in touched):
        return None, None, DepthPress(free=_rise(ctx, runner, quat, top_z, over), why=touched[-1].detail)
    R1, t1 = approach._chain().fk(ctx.client.joints())
    down = top_z - float(t1[2])
    ahead = float(depth) - down
    ahead_tf = float(DOWN_CONTACT_TOOL[2]) - float(lens[2])  # where TF's mount puts the pads before the lens
    pad = np.asarray(t1, float) + np.asarray(R1, float) @ np.asarray(DOWN_CONTACT_TOOL)
    world_dz = float(pad[2]) - float(from_container(ctx.cpose, m.button_offset)[2])
    runner.note(
        "depth_touch",
        down_mm=round(1000 * down, 1),
        lid_depth_mm=round(1000 * depth, 1),
        fingertips_ahead_of_lens_mm=round(1000 * ahead, 1),
        tf_says_mm=round(1000 * ahead_tf, 1),
        pad_xyz=[round(float(v), 4) for v in pad],
        world_dz_mm=round(1000 * world_dz, 1),
    )
    log.info(
        "[press_demo] TOUCHED after %.0f mm down: the fingertips are %.0f mm in front of the wrist lens (the camera's "
        "mount in TF says %.0f)" % (1000 * down, 1000 * ahead, 1000 * ahead_tf)
    )
    if ahead > LENS_AHEAD_MAX_M or ahead < -float(cfg.travel_m):
        return (
            None,
            None,
            DepthPress(
                free=_rise(ctx, runner, quat, top_z, over),
                contact=True,
                why="the contact %.0f mm down is not where this hand's fingertips can meet the " "lid" % (1000 * down),
            ),
        )
    if abs(ahead - ahead_tf) > AHEAD_TOL_M:  # the mount is wrong after all, or the touch met something else
        return (
            None,
            None,
            DepthPress(
                free=_rise(ctx, runner, quat, top_z, over),
                contact=True,
                why="the touch met something %.0f mm from where the camera saw the lid — not "
                "pushing" % (1000 * abs(ahead - ahead_tf)),
            ),
        )
    return touch, pad, None


# An OPEN box (the knob already up) is never pressed: pressing shuts it. The
# grip needs the lid's height to a few mm — the knob stands only 5-10 mm — and
# the wrist camera's is 7-9 mm off the arm's own through TF's mount (touches
# 2026-10-23: the fingertips 122.7 and 124.6 mm in front of the lens, TF says
# 116). So the closed pads feel the lid beside the knob, halfway across its
# ring round the button, along the hand's x: the fingers close along it, so
# the pads' narrow side lies across the ring. The closed pads' footprint, half
# of it along the hand's x and y (robotiq_description's fingertip collision
# mesh), and what it must keep clear of the knob and of the lid's edge.
PAD_HALF_M = (0.0065, 0.0105)
LID_TOUCH_CLEAR_M = 0.004


def lid_touch_offset_m(model):
    """How far beside the button's centre, along the hand's x, the closed pads
    feel an open box's lid: halfway across the ring between the button and
    the lid's nearest edge — or None when that ring cannot take the pads'
    footprint with LID_TOUCH_CLEAR_M to spare on both sides (Ankou: 37 mm,
    8.8 mm clear of the knob and 7.5 mm of the edge)."""
    r_in = 0.5 * float(model.button_diameter_m)
    r_out = 0.5 * min(float(model.dims[0]), float(model.dims[1])) - math.hypot(
        float(model.button_offset[0]), float(model.button_offset[1])
    )
    d = 0.5 * (r_in + r_out)
    if (
        d - PAD_HALF_M[0] - r_in < LID_TOUCH_CLEAR_M
        or r_out - math.hypot(d + PAD_HALF_M[0], PAD_HALF_M[1]) < LID_TOUCH_CLEAR_M
    ):
        return None
    return d


def _feel_the_lid(ctx, cfg, runner, quat, top_z, over, depth, lens, seen, up_mm):
    """An open box, its knob `up_mm` above the lid, the fingers over the
    button: the closed pads go lid_touch_offset_m along the hand's x (the
    other way when that line does not fly), feel the lid there as the press
    feels the button (_touch_lid), and come back up. The box is then where
    the pads put it — the button's centre at the lid's felt height — and
    the knob up. Nothing presses the knob: the guarded reach-out that
    calls this ends there, the lid on (the full run is compliant:
    tasks/reach_lid_off)."""
    from box_opening.runtime import approach

    m = ctx.model
    d = lid_touch_offset_m(m)
    if d is None:
        return DepthPress(
            free=True,
            why="the box is already OPEN (the knob %.0f mm above its lid), and its lid has no "
            "ring round the button wide enough for the pads to feel it on — not pressing "
            "the knob down, which would shut it" % up_mm,
        )
    log.info(
        "[press_demo] ALREADY OPEN: the knob stands %.0f mm above its lid — not pressing it, which would shut the "
        "box; the pads feel the lid %.0f mm beside it for the grip" % (up_mm, 1000 * d)
    )
    R, _t = approach._chain().fk(ctx.client.joints())
    whys = []
    for sign in (1.0, -1.0):
        pts, why = _hand_line(ctx, sign * d, 0.0, 0.0, quat)  # level, along the hand's x
        if pts:
            break
        whys.append(why or "no length")
    else:
        return DepthPress(
            free=True,
            why="the box is already OPEN, and neither move beside the knob is a straight line "
            "the arm can fly (%s)" % "; ".join(whys),
        )
    shift = np.asarray(R, float)[:, 0] * sign * d  # the move, in the world
    res = runner.run(
        [rp.line_leg(ctx, "depth:beside", pts, quat, CENTRE_SPEED, rp.drag_guard(over), _full_world(ctx))], execute=True
    )
    if any(not r.ok for r in res):
        return DepthPress(free=False, why="the move beside the knob met something — the arm holds")
    _touch, pad, stop = _touch_lid(ctx, cfg, runner, quat, top_z, over, depth, lens)
    if stop is not None:
        return stop
    button = (float(pad[0] - shift[0]), float(pad[1] - shift[1]), float(pad[2]))
    seen["touch_xyz"] = tuple(round(v, 5) for v in button)
    ctx.cpose = rp.cpose_for_button(list(button), ctx.cpose.yaw, m.button_offset)
    runner.note(
        "depth_lid_felt",
        knob_up_mm=round(float(up_mm), 1),
        beside_mm=round(1000 * sign * d, 1),
        lid_xyz=[round(float(v), 4) for v in pad],
    )
    log.info(
        "[press_demo] the lid felt %.0f mm beside the knob: the box is put there, the knob gripped from its height"
        % (1000 * d)
    )
    if not _rise(ctx, runner, quat, top_z, over):
        return DepthPress(free=False, contact=True, popped=True, why="the way up stopped")
    return DepthPress(free=True, contact=True, popped=True, knob_mm=float(up_mm))


def _pads(chain, q):
    """The pads (DOWN_CONTACT_TOOL) in the world with the arm at joints `q`."""
    R, t = chain.fk(list(q))
    return np.asarray(t, float) + np.asarray(R, float) @ np.asarray(DOWN_CONTACT_TOOL, float)


def fingertip_jacobian(chain, q):
    """The pads' translation Jacobian at joints `q`, 3 x len(q) (m/rad):
    central differences of the arm's own kinematics at DOWN_CONTACT_TOOL."""
    q = np.asarray(q, float)
    J = np.zeros((3, len(q)))
    for i in range(len(q)):
        dq = np.zeros(len(q))
        dq[i] = 1e-5
        J[:, i] = (_pads(chain, q + dq) - _pads(chain, q - dq)) / 2e-5
    return J


def fingertip_compliance(chain, q, kq=COMPLIANT_KQ):
    """The pads' compliance (m/N, 3 x 3, world axes) under joint stiffness
    `kq` with the arm at `q`: C = J Kq^-1 J^T, J fingertip_jacobian's. Not
    diagonal: a push straight down moves the pads sideways too."""
    J = fingertip_jacobian(chain, q)
    return J @ np.diag(1.0 / np.asarray(kq, float)[: J.shape[1]]) @ J.T


def fingertip_stiffness_z(chain, q, kq=COMPLIANT_KQ):
    """How stiff the pads are straight down (N/m) under joint stiffness `kq`
    with the arm at `q`: 1 / C_zz (fingertip_compliance)."""
    c = float(fingertip_compliance(chain, q, kq)[2, 2])
    return float("inf") if c <= 0.0 else 1.0 / c


def stroke_bend(chain, q_b, kq=COMPLIANT_KQ):
    """The compliant stroke's bend from joints `q_b`, the pads at the
    button's bottom: its end q_end = q_b - F Kq^-1 J_z(q_b)^T, J_z the
    vertical row of fingertip_jacobian at q_b and F COMPLIANT_PRESS_N —
    scaled, where the pads' reference would then drop less or more below
    the bottom than COMPLIANT_OVERSHOOT_M allows, to drop that bound. Under
    the driver's joint spring, linear in joint space, that end is the
    static balance itself: the pads held at the bottom, the arm stands at
    q_b and the springs push F straight down there, whatever the arm's
    compliance couples. Returns (q_end, how far it moves the pads'
    reference sideways (m), None), or (None, that, why) where the stroke
    flies straight instead: pads that do not move with the joints, or less
    than BEND_MIN_DROP_M, a bend past BEND_MAX_M, or an end outside a
    joint's limits."""
    q_b = np.asarray(q_b, float)
    J = fingertip_jacobian(chain, q_b)
    inv = 1.0 / np.asarray(kq, float)[: len(q_b)]
    if float(J[2] @ (inv * J[2])) <= 0.0:
        return None, 0.0, "the pads do not move with the joints here"
    pad_b = _pads(chain, q_b)

    def end(force):
        q = q_b - force * inv * J[2]
        return q, float(pad_b[2] - _pads(chain, q)[2])

    force = COMPLIANT_PRESS_N
    q_end, drop = end(force)
    if drop < BEND_MIN_DROP_M:
        return None, 0.0, "a bend would drop the pads only %.1f mm" % (1000 * drop)
    bound = min(max(drop, COMPLIANT_OVERSHOOT_M[0]), COMPLIANT_OVERSHOOT_M[1])
    if bound != drop:
        force *= bound / drop
        q_end, drop = end(force)
    aside = float(np.hypot(*(_pads(chain, q_end) - pad_b)[:2]))
    if aside > BEND_MAX_M:
        return (
            None,
            aside,
            "a bend would move the pads' reference %.0f mm sideways, past BEND_MAX_M's %.0f"
            % (1000 * aside, 1000 * BEND_MAX_M),
        )
    within = getattr(chain, "within_limits", None)  # the arm's own chain has limits; the test fakes have none
    if within is not None and not within(list(q_end)):
        return None, aside, "the bend's end is outside a joint's limits"
    return q_end, aside, None


def compliant_stroke(chain, cfg, q0, quat, top_z, kq=COMPLIANT_KQ):
    """press_compliant's stroke from the hand turned down at joints `q0`,
    holding `quat`, onto the button whose top is at `top_z`, as the arm's
    own kinematics solves it: straight down to TOUCH_SLOW_M above the top,
    then to the button's bottom (its last joints q_b), then the bend
    (stroke_bend) on to where the springs under `kq` push COMPLIANT_PRESS_N
    straight down at the pads held at the bottom, as joint waypoints no
    joint moving more than BEND_STEP_RAD between two, in the same leg. No
    bend (stroke_bend's refusals, or pads that already stand within 5 mm of
    the bottom or below it, with no stroke down to it to bend from):
    straight on down to as far below the bottom as the spring there
    (fingertip_stiffness_z at q_b) needs to push COMPLIANT_PRESS_N, within
    COMPLIANT_OVERSHOOT_M, refused where the pads already stand below that.
    Returns (StrokePlan, None) or (None, why)."""
    q0 = [float(v) for v in q0]
    R0, t0 = chain.fk(q0)
    R0, t0 = np.asarray(R0, float), np.asarray(t0, float)
    pad0 = t0 + R0 @ np.asarray(DOWN_CONTACT_TOOL, float)
    bottom = float(top_z) - float(cfg.button_travel_m)

    def down_to(q, z):
        # the straight line from q until the pads stand at z, the attitude held: [] when already this low
        _R, t = chain.fk(q)
        tool_z = float(t0[2]) + (z - float(pad0[2]))  # the attitude held: the tool drops as the pads do
        if float(t[2]) - tool_z < 0.005:
            return [], None  # already this low
        pts, why = chain.straight_line(q, [float(t0[0]), float(t0[1]), tool_z], quat)
        return pts, (None if pts is not None else "the press stroke is not a straight line the arm can fly (%s)" % why)

    approach, why = down_to(q0, float(top_z) + TOUCH_SLOW_M)
    if approach is None:
        return None, why
    q_a = approach[-1] if approach else q0
    press, why = down_to(q_a, bottom)
    if press is None:
        return None, why
    q_b = np.asarray(press[-1] if press else q_a, float)
    k = fingertip_stiffness_z(chain, q_b, kq)
    if press:
        q_end, bend_m, bend_why = stroke_bend(chain, q_b, kq)
    else:
        # The pads already stand within 5 mm of the bottom, or below it: q_b
        # is where the hand stands, not where the button's bottom holds the
        # pads, and a bend from there drives the reference that far deeper.
        # Straight, as before the bend, pads already below where the press
        # would end are refused.
        dz = _pads_z(chain, q_b) - bottom
        q_end, bend_m = None, 0.0
        bend_why = "the pads already stand %.0f mm %s the button's bottom, with no stroke down to it to bend from" % (
            1000 * abs(dz),
            "above" if dz >= 0 else "below",
        )
    if q_end is not None:
        n = max(2, int(math.ceil(float(np.abs(q_end - q_b).max()) / BEND_STEP_RAD)))
        press = list(press) + [list(q_b + (q_end - q_b) * (i / n)) for i in range(1, n + 1)]
        end_z = float(_pads(chain, q_end)[2])
        over = bottom - end_z
    else:
        over = min(max(COMPLIANT_PRESS_N / k, COMPLIANT_OVERSHOOT_M[0]), COMPLIANT_OVERSHOOT_M[1])
        end_z = bottom - over
        if pad0[2] <= end_z:
            return None, "the pads already stand %.0f mm below where the press would end" % (1000 * (end_z - pad0[2]))
        press, why = down_to(q_a, end_z)  # straight from where the approach ends, as before the bend
        if press is None:
            return None, why
        q_end, bend_m = np.asarray(press[-1] if press else q_a, float), 0.0
    legs = [
        (name, pts, speed)
        for name, pts, speed in (
            ("depth:approach", approach, COMPLIANT_STROKE_SLOWDOWN * cfg.press_speed),
            ("depth:press", press, COMPLIANT_STROKE_SLOWDOWN * TOUCH_SPEED),
        )
        if pts
    ]
    return StrokePlan(
        legs, k, over, end_z, t0, pad0, q0, list(quat), [float(v) for v in q_end], bend_why is None, bend_m, bend_why
    ), None


def compliant_line(ctx, name, pts, quat, speed, world):
    """A compliant press's plan-free line (rp.line_leg), unguarded, under
    COMPLIANT_GAINS named on the leg: the spring the press is modelled on
    (COMPLIANT_KQ), and the one gains spec legs.can_merge asks of legs that
    fly as one execution."""
    leg = rp.line_leg(ctx, name, pts, quat, speed, None, world)
    leg.gains = COMPLIANT_GAINS
    return leg


def _lean_deg(quat):
    """How far short of straight down the hand at `quat` presses (deg, to 0.1)."""
    return round(math.degrees(rp.hand_lean(rp._rot(quat))), 1)


def press_text(stroke, cfg):
    """What the console says of a compliant stroke (compliant_stroke's)
    before it flies: how far down, past the button's bottom, bent how far
    aside (or why not), with what spring, held how long — and the lean,
    from LEAN_SAID_DEG on."""
    lean = _lean_deg(stroke.quat)
    if stroke.bent:
        how = (
            "straight down %.0f mm to the button's bottom, its reference on %.0f mm below it and %.0f mm aside, "
            "where the springs push ~%.0f N straight down on the pads held there (%.0f N/m at the pads)"
            % (
                1000 * (stroke.pad0[2] - stroke.end_z - stroke.over),
                1000 * stroke.over,
                1000 * stroke.bend_m,
                stroke.k * stroke.over,
                stroke.k,
            )
        )
    else:
        how = (
            "straight down %.0f mm to %.0f mm below the button's bottom — the spring pushes ~%.0f N there "
            "(%.0f N/m at the pads)%s"
            % (
                1000 * (stroke.pad0[2] - stroke.end_z),
                1000 * stroke.over,
                stroke.k * stroke.over,
                stroke.k,
                "; not bent: %s" % stroke.bend_why if stroke.bend_why else "",
            )
        )
    return "PRESS (compliant)%s: %s; held %.1f s, then back up. Nothing stops it but the e-stop" % (
        ", the hand leaning %.0f deg back toward the arm" % lean if lean >= LEAN_SAID_DEG else "",
        how,
        cfg.push_hold_s,
    )


def compliant_hold(ctx, cfg, runner, stroke, top_z, seen):
    """The compliant press's hold at the end of its stroke (`stroke`,
    compliant_stroke's) on the button whose top is at `top_z`: held
    push_hold_s where the springs left the pads, pressing — the hold IS the
    reading. What the pads pushed with is the joint springs' pull from the
    held arm toward the stroke's end (stroke.q_end, under COMPLIANT_KQ) as
    the pads feel it: the force f, world axes, with J^T f = Kq (q_end -
    q_held), least squares, J fingertip_jacobian's at the held arm. Its
    downward part is the press, COMPLIANT_FELT_N or more a press, and the
    rest the sideways shove. seen["touch_xyz"] is where the pads were held,
    pressing, and seen["held_joints"] the arm held there, the way back up's
    start (rise_back_up). The run record (depth_compliant_press) and the
    console say it, the bend, and how far from the aim (the pads at q0)
    the pads were held. Returns the verdict: a DepthPress, free, judged
    False (a compliant press does not look)."""
    from box_opening.runtime import approach
    from box_opening.runtime.branches import on_branch

    chain = approach._chain()
    k, end_z = stroke.k, stroke.end_z
    q_end = stroke.q_end if stroke.q_end is not None else (stroke.legs[-1][1][-1] if stroke.legs else stroke.q0)
    q_end = np.asarray(q_end, float)
    held = ctx.client.settled_efforts(duration_s=0.15, settle_s=max(0.0, float(cfg.push_hold_s) - 0.15))
    qh = held[1] if held is not None else ctx.client.joints()
    seen["held_joints"] = [float(v) for v in qh]
    qh = np.asarray(on_branch(qh, q_end), float)  # the reading is wrapped into (-pi, pi], the stroke's joints are not
    pad = _pads(chain, qh)
    pull = np.asarray(COMPLIANT_KQ, float)[: len(qh)] * (q_end - qh)
    f = np.linalg.lstsq(fingertip_jacobian(chain, qh).T, pull, rcond=None)[0]
    force, aside = float(-f[2]) + 0.0, float(np.hypot(f[0], f[1]))  # + 0.0: a spring at rest says 0 N, not -0
    felt = force >= COMPLIANT_FELT_N
    if felt:
        seen["touch_xyz"] = tuple(round(float(v), 5) for v in pad)  # where the pads were held, pressing
    off = pad[:2] - np.asarray(stroke.pad0, float)[:2]  # the held pads from the aim
    facts = dict(
        top_z=round(float(top_z), 4),
        end_z=round(end_z, 4),
        held_pad_xyz=[round(float(v), 4) for v in pad],
        stiffness_n_per_m=round(k, 1),
        press_n=round(force, 1),
        sideways_n=round(aside, 1),
        felt=felt,
        bent=bool(stroke.bent),
        bend_mm=round(1000 * float(stroke.bend_m), 1),
        held_off_aim_mm=[round(1000 * float(v), 1) for v in off],
    )
    if stroke.bend_why:
        facts["bend_why"] = stroke.bend_why
    lean = _lean_deg(stroke.quat)
    if lean >= LEAN_SAID_DEG:
        facts["lean_deg"] = lean
    runner.note("depth_compliant_press", **facts)
    log.info(
        "[press_demo] %s, %.0f N sideways: the pads held %.0f mm above the stroke's end, %.0f mm %s the button's "
        "top, %.0f mm from the aim"
        % (
            "PRESSED with ~%.0f N" % force if felt else "NOTHING PRESSED (%.0f N)" % force,
            aside,
            1000 * (pad[2] - end_z),
            1000 * abs(float(top_z) - pad[2]),
            "below" if pad[2] < float(top_z) else "above",
            1000 * float(np.hypot(*off)),
        )
    )
    return DepthPress(free=True, contact=felt, felt=felt, judged=False, press_n=force)


def compliant_rise(ctx, runner, stroke, held):
    """From the hold back up to where the stroke began: 'depth:rise' from
    where the arm stands back up the stroke (rise_back_up), else a straight
    line to its start (stroke.t0), the lean held — reach_press._rise_legs,
    as the one-execution way home rises — at rp.free_line_speed
    (compliant_line, labelled with the full world). A bent stroke's end is
    no IK solution: in free air the hand stands there 3-6 deg off the
    attitude the stroke held, and a straight line's first 2 mm step takes
    that whole turn (flatbot's URDF: 0.11-0.13 rad at 15:19's and 16:23's
    boxes, refused past MAX_STEP_RAD 0.76 m out). Returns the hold's
    verdict `held` (compliant_hold's), free False with why when the way up
    does not fly — the arm holds, let go (rp.let_go): on the button, its
    springs would push on toward the stroke's end, and part-way up toward
    the rise's next point."""
    up, _on_stroke, why = rp._rise_legs(ctx, stroke, ctx.client.joints(), _full_world(ctx))
    if up is None:
        rp.let_go(ctx)
        return dataclasses.replace(
            held, free=False, why="the way up is not a straight line the arm can fly (%s) — the arm holds" % why
        )
    if up:
        res = runner.run(up, execute=True)
        if any(not r.ok for r in res):
            rp.let_go(ctx)
            return dataclasses.replace(held, free=False, why="the way up stopped (%s) — the arm holds" % res[-1].detail)
    return held


def _pads_z(chain, q):
    return float(_pads(chain, q)[2])


def rise_back_up(chain, stroke, q_held, join_above_m=RISE_JOIN_ABOVE_M):
    """The way back up from the hold to where the stroke began, as joint
    waypoints ending exactly on the stroke's first sample (stroke.q0), or
    None: from the held arm `q_held` to the first of the stroke's own
    samples (compliant_stroke's approach then press, reversed: from its
    end up) whose pads (DOWN_CONTACT_TOOL) stand `join_above_m` or more
    above the held pads, then up the stroke from there. None when no
    sample stands that high, or the step to it moves a joint more than
    RISE_FIRST_STEP_MAX_RAD, or it puts the pads lower than they were held
    at any of RISE_CHECK_POINTS along it. The held arm is written on the
    stroke's side of the wrap first (runtime/branches): its reading is
    wrapped into (-pi, pi], the stroke's samples are not."""
    from box_opening.runtime.branches import on_branch

    samples = []
    for _name, pts, _speed in stroke.legs:
        for q in pts:
            q = [float(v) for v in q]
            if samples and max(abs(a - b) for a, b in zip(q, samples[-1], strict=False)) < 1e-9:
                continue  # the press starts where the approach ends
            samples.append(q)
    if not samples:
        return None
    up = samples[::-1]
    qh = on_branch(q_held, up[0])
    floor = _pads_z(chain, qh)
    join = next((i for i, q in enumerate(up) if _pads_z(chain, q) >= floor + float(join_above_m)), None)
    if join is None:
        return None
    step = np.asarray(up[join], float) - np.asarray(qh, float)
    if float(np.abs(step).max()) > RISE_FIRST_STEP_MAX_RAD:
        return None
    for k in range(1, RISE_CHECK_POINTS + 1):
        if _pads_z(chain, list(np.asarray(qh, float) + step * (k / RISE_CHECK_POINTS))) < floor:
            return None
    return [list(qh)] + up[join:]


def press_compliant(ctx, cfg, runner, contact_tool, seen, turned=None, top_z=None):
    """The COMPLIANT press (COMPLIANT_PRESS_N): from where the hand stands
    over the button, turned down (`turned`, else TURN here), straight down
    at press_speed to TOUCH_SLOW_M above the button's top `top_z`, then at
    TOUCH_SPEED to its bottom and on to the stroke's end below it, bent so
    the springs push straight down there (stroke_bend) — one execution —
    held push_hold_s, and back up to where it began. What it pressed with
    is the joint springs' pull from where they held the arm toward the
    stroke's end, seen at the pads (compliant_hold). Its TURN may stop up
    to 15 deg short of straight down (COMPLIANT_TURN_SHORT_DEG): the pads'
    centre still over the button, the stroke still vertical to the bottom.
    Nothing it reads stops it. The stroke, the hold and the rise are
    compliant_stroke, compliant_hold and compliant_rise: a compliant
    reach-out flies the same stroke as the end of its one-execution way in
    (reach_press.way_in_legs) and calls this only where that does not
    fly. From the stroke's start until the rise ends a stop lets go
    (rp.let_go): the stroke stopped part-way, the way up refused or
    stopped (compliant_rise), anything raised — and before it, a turn
    stopped part-way (turn_down's). A turn or a stroke that did not start
    moved nothing, and nothing is let go: after the stroke's, the arm
    stands over the button in free air, free to go home."""
    from box_opening.runtime import approach
    from box_opening.tasks.press_demo import viz_hook

    if top_z is None:
        return DepthPress(free=True, why="a compliant press needs the button's height (the aim or the scene camera)")
    if turned is not None:
        quat = list(turned)
    else:
        quat, _moved, why = turn_down(ctx, runner, contact_tool, None, shorts=COMPLIANT_TURN_SHORT_DEG)
        if quat is None:
            return DepthPress(free=why.startswith("no straight-down"), why=why)
    stroke, why = compliant_stroke(approach._chain(), cfg, ctx.client.joints(), quat, top_z)
    if stroke is None:
        return DepthPress(free=True, why=why)
    world = _full_world(ctx)
    legs = [compliant_line(ctx, name, pts, quat, speed, world) for name, pts, speed in stroke.legs]
    log.info("[press_demo] %s" % press_text(stroke, cfg))
    viz_hook("press", stroke, top_z, cfg)  # the stroke, drawn off this thread (perception/viz)
    try:
        res = runner.run(legs, execute=True)
        if len(res) == 1 and not res[0].ok and (res[0].outcome == "refused" or not res[0].progress):
            # nothing moved: the runner's gates refused the stroke, or the
            # driver or the client's own gates its goal before it flew. A
            # stroke with no legs (the pads already within 5 mm of its end,
            # compliant_stroke) flies nothing and has no result: it goes on
            # to the hold and the rise
            return DepthPress(
                free=True,
                why="the press stroke did not start (%s) — nothing was pressed" % (res[0].detail or res[0].outcome),
            )
        # the stroke meets the box by design: from here every world (home's
        # too) pads it for a shove, as _touch_lid's contact does
        ctx.contact_pad = CONTACT_SHIFT_PAD_M
        if any(not r.ok for r in res):
            rp.let_go(ctx)  # stopped part-way down, or on the button
            return DepthPress(free=False, why="the press stroke stopped (%s) — the arm holds" % res[-1].detail)
        return compliant_rise(ctx, runner, stroke, compliant_hold(ctx, cfg, runner, stroke, top_z, seen))
    except BaseException:
        # a Ctrl+C in the hold's read lands with no goal in flight (abort.py
        # raises at once): the springs would push on into the box
        rp.let_go(ctx)
        raise


@dataclass
class DepthPress:
    """What the depth press came back with."""

    free: bool  # the arm ended in free air where the touch began: it may go home
    contact: bool = False  # the fingertips touched something
    felt: bool = False  # the torques felt the button bottom out
    popped: bool = False  # the camera sees the knob up after
    moved: bool = False  # the camera sees the box moved
    shift_m: float = 0.0  # how far the wrist camera saw the button move (when it saw it again)
    lid_dz_m: float = 0.0  # ... and the lid's depth change
    knob_mm: float = float("nan")  # the knob above the lid at the after-look
    why: str = None  # why it stopped short, when it did
    lost: bool = False  # after a contact the wrist camera did not find the box again
    touch_xyz: tuple = None  # where the pads met the button (the arm's own kinematics)
    ref: object = None  # the free-air (efforts, joints) reading the press was judged against
    seen_at: tuple = None  # where the after-look saw the button (where_seen), for a look again from there
    judged: bool = True  # a look judged the knob after (a compliant press does not look)
    press_n: float = float("nan")  # a compliant press: what its spring held on the button


def press_down_from_here(
    node,
    ctx,
    cfg,
    runner,
    watcher,
    contact_tool,
    here=None,
    turned=None,
    knob_up_mm=None,
    aimed=None,
    compliant=False,
    top_z=None,
):
    """TURN, DEPTH, TOUCH, PUSH, RISE, LOOK from where the hand stands over
    the button (`contact_tool`: the fingertip's point in tool_frame at the
    hand's current attitude). `here`: the caller's at-rest reading where the
    arm stands now, nothing flown since; `turned`: the hand already points
    down at this quat (the reach-out slid in turning), no TURN. An open box
    — the knob KNOB_UP_MM up by any look here or by `knob_up_mm`, the
    caller's aim — is never pressed: the pads feel its lid beside the knob
    instead (_feel_the_lid). `aimed`: the button's top in the world by the
    caller's aim — the first look takes no circle far from it, and stands
    in for a look that does not confirm the button (aim_disagrees).
    `compliant`: press_compliant instead, down to the button's top at
    `top_z`. Never goes home itself: DepthPress.free says whether the arm
    may."""
    seen = {}
    if compliant:
        result = press_compliant(ctx, cfg, runner, contact_tool, seen, turned, top_z)
        return dataclasses.replace(result, touch_xyz=seen.get("touch_xyz"))
    result = _press_down(node, ctx, cfg, runner, watcher, contact_tool, seen, here, turned, knob_up_mm, aimed)
    return dataclasses.replace(result, ref=seen.get("ref"), touch_xyz=seen.get("touch_xyz"))


def _press_down(
    node, ctx, cfg, runner, watcher, contact_tool, seen, here=None, turned=None, knob_up_mm=None, aimed=None
):
    from box_opening.models.container import from_container
    from box_opening.runtime import approach

    m = ctx.model
    if here is None:
        here = rp._settled(ctx, "over the box")
    if here is None:
        return DepthPress(free=True, why="no steady efforts over the box")
    if turned is not None:
        quat, moved = list(turned), False
    else:
        quat, moved, why = turn_down(ctx, runner, contact_tool, here)
        if quat is None:
            return DepthPress(free=why.startswith("no straight-down"), why=why)
    # a turn that did not move leaves the arm where `here` was read
    over = rp._settled(ctx, "over the button, pointing down") if moved else here
    if over is None:
        return DepthPress(free=True, why="no steady efforts pointing down")
    seen["ref"] = over
    _R, t0 = approach._chain().fk(ctx.client.joints())
    top_z = float(t0[2])

    first = None if aimed is None else (lambda pair: where_seen(aim_seen(ctx, aimed, pair, knob_up_mm)))
    s, z_under, lens_pair, status = look(node, watcher, m, "depth-look", runner, ctx, settled=True, expect=first)
    if lens_pair is None:
        return DepthPress(
            free=True,
            why="no camera pose for the wrist's frames: where the lens sits from the fingers "
            "is unknown (%s)" % status,
        )
    lens, r_hc = lens_pair
    measured = getattr(getattr(watcher, "grab", None), "wrist_lens_in_hand_m", None)
    if measured is None:
        return DepthPress(
            free=True,
            why="no measured wrist lens offset (wrist_lens_in_hand_m) in the camera config — "
            "not centring on TF's mount",
        )
    measured = [float(v) for v in measured]
    if (
        math.hypot(lens[0] - measured[0], lens[1] - measured[1]) > MOUNT_XY_TOL_M
        or abs(lens[2] - measured[2]) > MOUNT_Z_TOL_M
    ):
        runner.note(
            "depth_mount_refused",
            lens_in_hand_mm=[round(1000 * v, 1) for v in lens],
            measured_mm=[round(1000 * v, 1) for v in measured],
        )
        return DepthPress(
            free=True,
            why="the wrist camera's mount in TF puts the lens at (%s) mm from the tool frame, "
            "the measured wrist_lens_in_hand_m at (%s) mm — fix the mount before centring "
            "on it"
            % (", ".join("%+.0f" % (1000 * v) for v in lens), ", ".join("%+.0f" % (1000 * v) for v in measured)),
        )
    trusted = s is None and aimed is not None  # no look confirms the button: the aim stands in
    if s is None and aimed is None:
        return DepthPress(
            free=True,
            why="the wrist camera does not see the button from over the box (%s) — not " "pressing blind" % status,
        )
    if trusted:
        s = aim_seen(ctx, aimed, lens_pair, knob_up_mm)
        why = aim_disagrees(ctx, s, lens_pair, z_under)
        runner.note(
            "depth_trusted_aim",
            ok=why is None,
            why=why,
            lens_xy_mm=[round(1000 * s.x, 1), round(1000 * s.y, 1)],
            button_depth_mm=round(1000 * s.button_depth, 1),
            under_fingers_mm=None if z_under is None else round(1000 * z_under, 1),
        )
        if why is not None:
            return DepthPress(
                free=True,
                why="the wrist camera does not see the button from over the box (%s), and %s "
                "— not pressing on the aim" % (status, why),
            )
        log.info(
            "[press_demo] DEPTH: the wrist camera does not confirm the button from over the box (%s) — going by "
            "the aim: its button %.0f mm below the lens, the camera reads %.0f mm under the fingers"
            % (status, 1000 * s.button_depth, 1000 * z_under)
        )
    # how far the knob stands above its lid, by every look that read it
    ups = [float(v) for v in (knob_up_mm, s.up_mm) if v is not None and np.isfinite(v)]

    # CENTRE: the fingertip's vertical through the button the camera sees,
    # in world xy — the stroke is vertical whatever the hand's lean
    off = _off_in_world(ctx, s, lens, r_hc)
    moves = 0
    while math.hypot(*off) > CENTRE_TOL_M:
        miss = math.hypot(*off)
        if miss > CENTRE_MAX_M:
            return DepthPress(
                free=True,
                why="the button the wrist camera sees is %.0f mm from under the fingers — farther "
                "than a re-centre should be (check the wrist camera's mount)" % (1000 * miss),
            )
        if moves >= CENTRE_MAX_MOVES:
            return DepthPress(
                free=True,
                why="the button is still %.0f mm from under the fingers after %d moves" % (1000 * miss, moves),
            )
        log.info(
            "[press_demo] CENTRE: the button %.0f mm from under the fingers (world x %+.0f, y %+.0f mm)"
            " — moving over it" % (1000 * miss, 1000 * off[0], 1000 * off[1])
        )
        R_move, t = approach._chain().fk(ctx.client.joints())
        pts, why = rp._line(ctx, [float(t[0]) + off[0], float(t[1]) + off[1], float(t[2])], quat)  # level
        if pts is None:
            return DepthPress(
                free=True, why="the move over the button is not a straight line the arm can fly (%s)" % why
            )
        if pts:
            res = runner.run(
                [rp.line_leg(ctx, "depth:centre", pts, quat, CENTRE_SPEED, rp.drag_guard(over), _full_world(ctx))],
                execute=True,
            )
            if any(not r.ok for r in res):
                return DepthPress(free=False, why="the move over the button met something — the arm holds")
        moves += 1
        if trusted:
            s = aim_seen(ctx, aimed, (lens, r_hc), knob_up_mm)  # no look confirms it: the aim, from here
        else:
            expect = expect_after_move(s, lens, r_hc, R_move, (off[0], off[1], 0.0))
            s, z_under, lens2, status = look(node, watcher, m, "depth-look", runner, ctx, expect=expect)
            lens, r_hc = lens2 or (lens, r_hc)
            if s is None:
                return DepthPress(free=True, why="the wrist camera lost the button after moving over it (%s)" % status)
            if np.isfinite(s.up_mm):
                ups.append(float(s.up_mm))
        off = _off_in_world(ctx, s, lens, r_hc)
        if math.hypot(*off) > 0.5 * miss:
            return DepthPress(
                free=True,
                why="moving over the button did not bring it under the fingers (%.0f -> %.0f mm)"
                " — the camera's axes in TF are wrong?" % (1000 * miss, 1000 * math.hypot(*off)),
            )
    runner.note(
        "depth_centred",
        moves=moves,
        off_mm=[round(1000 * v, 1) for v in off],
        lens_in_hand_mm=[round(1000 * v, 1) for v in lens],
        by_aim=trusted,
    )
    depth = s.lid_depth
    if trusted:
        # no nearer than the camera's own depth under the fingers puts the
        # lid either: the fast stretch stops TOUCH_SLOW_M short of both
        depth = min(depth, float(z_under) + s.lid_depth - s.button_depth)
    log.info(
        "[press_demo] DEPTH: the button %.1f mm from under the fingers%s%s; the lid %.0f mm below the wrist lens "
        "(under the fingers' line: %s)"
        % (
            1000 * math.hypot(*off),
            " by the aim" if trusted else "",
            "" if not moves else " after %d move(s)" % moves,
            1000 * depth,
            "n/a" if z_under is None else "%.0f mm" % (1000 * z_under),
        )
    )

    up = max(ups) if ups else float("nan")
    if np.isfinite(up) and up >= KNOB_UP_MM:
        return _feel_the_lid(ctx, cfg, runner, quat, top_z, over, depth, lens, seen, up)
    touch, pad, stop = _touch_lid(ctx, cfg, runner, quat, top_z, over, depth, lens)
    if stop is not None:
        return stop
    world_dz = float(pad[2]) - float(from_container(ctx.cpose, m.button_offset)[2])
    # the box is where the fingertips met its button, flush before the push:
    # every world from here on (the home's too) holds it there, not where TF
    # put it (76 mm high on 20:04, so every home started "inside" it). The
    # table is the scene camera's own measurement and is left as it is.
    seen["touch_xyz"] = tuple(round(float(v), 5) for v in pad)
    ctx.cpose = rp.cpose_for_button([float(v) for v in pad], ctx.cpose.yaw, m.button_offset)
    log.info(
        "[press_demo] the box is put where the fingertips met it: %+.0f mm from where TF had its button"
        % (1000 * world_dz)
    )

    pts, why = _hand_line(ctx, 0.0, 0.0, -float(cfg.button_travel_m), quat)
    if not pts:
        return DepthPress(
            free=_rise(ctx, runner, quat, top_z, over),
            contact=True,
            why="the push is not a straight line the arm can fly (%s)" % (why or "no length"),
        )
    # the backstop judged against the free-air reading over the box, not the
    # push's first sample: that is already loaded by whatever the touch
    # overshot (run 20:04: 5.8 Nm), and the push then rode to 13.5 Nm off
    # free air against touch_nm 10
    _g, verify = _push_guard_and_verify(m, cfg)
    guard = GuardSpec(touch_nm=float(m.touch_nm), trip="press", gravity=rp._wrist_gravity(), reference=over)
    push = rp.line_leg(ctx, "press:push", pts, quat, cfg.press_speed, guard, _full_world(ctx), verify)
    push.hold_s = float(cfg.push_hold_s)
    push.recoil_via = touch.traj  # the recoil carries on back up the touch
    res = runner.run([push], execute=True)  # held, then the runner's recoil lets go
    pushed = res[0]  # the recoil, when there was one, comes after it
    felt = pushed.ok and pushed.outcome == "touch"
    if felt:
        log.info(
            "[press_demo] PRESSED: the torques felt the button bottom out%s"
            % ("" if pushed.torque_peak is None else " (%.1f Nm)" % pushed.torque_peak)
        )
    elif pushed.ok:
        log.info("[press_demo] NOT PRESSED by feel: the whole %.0f mm push and no stop" % (1000 * cfg.button_travel_m))
    else:
        log.info("[press_demo] the push failed (%s)" % pushed.detail)
    if not _rise(ctx, runner, quat, top_z, over):
        return DepthPress(free=False, contact=True, felt=felt, why="the way up stopped")

    # after _rise's read, back where the last look stood
    after, _z, _lens, status = look(node, watcher, m, "depth-after", runner, ctx, settled=True, expect=where_seen(s))
    popped = after is not None and np.isfinite(after.up_mm) and after.up_mm >= KNOB_UP_MM
    if after is None:
        # the reach-out's AFTER rule: a box no camera finds after a contact
        # holds the arm — the caller asks the scene camera (after_scene_check)
        log.info(
            "[press_demo] AFTER: the wrist camera does not find the button%s (%s)"
            % (" (nor did it before the press, which went by the aim)" if trusted else " again", status)
        )
        return DepthPress(
            free=False,
            contact=True,
            felt=felt,
            lost=True,
            why="the wrist camera does not find the button after the press — the knob is not judged",
        )
    shift = math.hypot(after.x - s.x, after.y - s.y)
    # from the aim, the shift holds the aim's own miss (14:33: ~10 mm), so
    # only a slide too far to grip by is a move
    moved = shift > (SLIDE_OK_M if trusted else BOX_MOVED_M) or abs(after.lid_depth - s.lid_depth) > LID_MOVED_M
    log.info(
        "[press_demo] AFTER: the knob %s; the button %.0f mm from where %s, the lid %+.0f mm"
        % (
            "UP %.0f mm — the box is open" % after.up_mm
            if popped
            else ("%.1f mm above the lid" % after.up_mm if np.isfinite(after.up_mm) else "not judged"),
            1000 * shift,
            "the aim put it" if trusted else "it was",
            1000 * (after.lid_depth - s.lid_depth),
        )
    )
    if moved:
        # home is planned around the box where the camera sees it now: the
        # sighting's shift, turned from the camera's axes into the hand's
        # (R_hand_cam: 180 deg on the bench's D405), then into the world
        R2, _t2 = approach._chain().fk(ctx.client.joints())
        dh = _in_hand(after, lens, r_hc) - _in_hand(s, lens, r_hc)
        dh[2] = 0.0  # the lid's depth change is not a shift
        d = np.asarray(R2, float) @ dh
        c = ctx.cpose
        ctx.cpose = ContainerPose(xyz=(float(c.xyz[0] + d[0]), float(c.xyz[1] + d[1]), float(c.xyz[2])), yaw=c.yaw)
    return DepthPress(
        free=True,
        contact=True,
        felt=felt,
        popped=popped,
        moved=moved,
        shift_m=float(shift),
        lid_dz_m=float(after.lid_depth - s.lid_depth),
        knob_mm=float(after.up_mm),
        seen_at=where_seen(after),
    )


def after_scene_check(ctx, runner, result, scene_fix, locator):
    """After a contact the wrist camera lost the box (DepthPress.lost): the
    scene camera looks again, as the reach-out's AFTER does. Seen, the arm
    may go home around it — shifted by what the scene camera saw it move,
    from where the arm measured it; unseen, the arm holds. Returns the
    DepthPress to finish with."""
    from box_opening.tasks.press_demo import SCENE_LOCATE_S

    if not result.lost:
        return result
    again = locator.locate(SCENE_LOCATE_S) if locator is not None and scene_fix is not None else None
    if again is None:
        runner.note("depth_after_scene", found=False)
        log.info(
            "[press_demo] STOP: neither camera finds the box after the press — it may have left the table. The arm "
            "holds over where it was; check the box, then home the arm with sheppy's home_arm"
        )
        return result
    d = np.asarray(again.top_xyz, float) - np.asarray(scene_fix.top_xyz, float)
    moved = math.hypot(d[0], d[1]) > rp.SCENE_MOVED_M or abs(d[2]) > rp.SCENE_MOVED_M
    runner.note(
        "depth_after_scene",
        found=True,
        scene_moved_mm=round(1000 * math.hypot(d[0], d[1]), 1),
        scene_dz_mm=round(1000 * d[2], 1),
    )
    log.info(
        "[press_demo] AFTER: the scene camera puts the box %.0f mm from where it was, %+.0f mm in height"
        % (1000 * math.hypot(d[0], d[1]), 1000 * d[2])
    )
    if moved:
        # the scene camera's own shift: its box can stand 8-10 cm from the
        # arm's, but where it saw the box go is its own two looks' difference
        c = ctx.cpose
        ctx.cpose = ContainerPose(xyz=(float(c.xyz[0] + d[0]), float(c.xyz[1] + d[1]), float(c.xyz[2])), yaw=c.yaw)
    return dataclasses.replace(result, free=True, moved=moved)


def home_why(result):
    """What the home after a depth press says first: why it stopped short,
    when it did (2026-10-23 17:30 went home "DEPTH PRESS done" and told why
    only after), and a moved box is a STOP before the arm goes anywhere."""
    if result.why is not None:
        return "STOP: %s" % result.why
    return "STOP: the press MOVED the box — check it is still standing on the table" if result.moved else ""


def exit_code(result):
    """The process exit code for a DepthPress, its verdict printed: 0 the
    knob up, 1 anything else."""
    if result.why is not None:
        log.info("[press_demo] STOP: %s" % result.why)
        return 1
    if result.moved:
        log.info("[press_demo] STOP: the press MOVED the box — check it is still standing on the table")
        return 1
    if not result.judged:
        # a compliant press: no look after it, the press itself is the verdict
        if result.felt:
            log.info(
                "[press_demo] PRESSED: ~%.0f N held on the button (a compliant run does not look at the knob)"
                % result.press_n
            )
            return 0
        log.info("[press_demo] STOP: the stroke pressed nothing (%.0f N at its end)" % result.press_n)
        return 1
    if result.popped:
        log.info("[press_demo] the box is open")
        return 0
    log.info("[press_demo] STOP: %s, and the knob is not up" % ("pressed by feel" if result.felt else "not pressed"))
    return 1


def finish(ctx, result, homed):
    """The process exit code after a depth press and the home that followed
    it (`homed`: the arm got there). An arm left away from home is never a
    success. What the press established goes into the mission's record."""
    frames = getattr(ctx, "mission_frames", None)
    if frames is not None:
        facts = dict(
            popped=None if not result.contact or result.lost or not result.judged else bool(result.popped),
            touch_xyz=None if result.touch_xyz is None else list(result.touch_xyz),
        )
        if result.felt:
            facts["stage"] = "pressed"
        frames.note(**facts)
    code = exit_code(result)
    if not homed:
        if result.free:
            log.info("[press_demo] STOP: the arm did not get home — it holds; home it with sheppy's home_arm")
        code = max(code, 1)
    return code
