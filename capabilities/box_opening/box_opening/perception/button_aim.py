"""Aiming the press from staging: the button's circle, nothing else.

At staging the wrist camera hangs 12-14 cm above the lid and the button is
the largest circle in the frame. The plateau detector (depth_source) gated
that circle behind the whole box top — height band, footprint against the
model's dims, solid fill, no image border — and at 25 cm every one of those
is a way to refuse a box the camera is looking straight at (2026-09-17: a
4.1-inch OXO refused 103/104 frames as "footprint 0.11x0.10 vs 0.07x0.07"
with the button dead centre in the image). The scene camera has already
said which thing is the box; from here the only question is where its
button is, and a circle of the button's diameter, at lid height, sitting
on a flat lid, answers it without knowing the box's footprint at all.

Owner's design (2026-09-17): scene OWL finds the box, the arm flies over
it, then OpenCV's circle alone aims the press. The OWL-gated wrist scan
that was the fallback when the scene camera found nothing is gone: no
scene fix stops the run (owner, 2026-10-22).

The reach-out press (flatbot: the box 0.80-0.89 m out, where the hand
cannot point straight down) sees the button from a tilted pose, 29-39 deg
off the lid's normal. There the button is an ellipse, the disc is not flat
in camera depth and an image annulus is not a lid annulus, so a camera
axis more than OBLIQUE_MIN_DEG off straight down takes find_button_oblique:
the frame is rectified onto a horizontal plane at the lid's prior height,
where the button is a circle again, and every gate is judged in base z.
The lid's height is measured in the frame, not trusted — the scene camera
has put a lid top 21 mm below table_z + dims.z.
"""

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from box_opening.perception.depth_source import (
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    TOP_RESIDUAL_MAX_M,
)

MIN_RANGE_M = 0.08  # closer than this the button fills the frame; no aim
RADIUS_WINDOW = (0.7, 1.4)  # of the model diameter's pixel radius, as button_circle_refine
DISC_STD_MAX_M = 0.006  # the button top is flat
ANNULUS = (1.15, 1.5)  # of r: the ring around the button is LID, not table
ANNULUS_TOL_M = 0.012
ANNULUS_MIN_FRAC = 0.8
AGREE_M = 0.004  # hits that agree within this (planar) count together
MIN_HITS = 3
# A tilted view's aim may fix on two hits that agree (ButtonAimer's min_hits,
# tasks/reach_press), found among this many newest (agreeing_fix): its hits
# come rarer than straight down's — three agreeing took 14, 19 and 25 frames
# on flatbot 2026-10-27 — and at 15:19 the first hit did not agree, so the
# last-three rule waited for three new ones.
AIM_WINDOW = 4
KNOB_MAX_UP_M = 0.025  # a popped knob's top is at most this far above the lid (met at ~10 and at 17.7 mm)
# The press point is the CENTRE of the seam circle, and nothing overrides
# it (owner: "make sure the gripper goes to the center of the circle
# segmented out by openCV"). For four days a "dimple" detector did: the
# lowest patch of depth inside the disc, taken for the place a finger
# presses, after a +9.6 mm touch on 2026-09-17 was read as "pads on a ring
# around a dimple". That surface was the button itself, already popped
# (this knob stands ~10 mm proud when up: the aim frame of 2026-09-21 13:35
# has the whole disc +10 mm above its own lid). The patch was stereo noise
# at the sensor's 1 mm steps: it fired on one real aim in four, moved that
# press 13 mm off the centre, and that press missed.

# The oblique path (find_button_oblique).
OBLIQUE_MIN_DEG = 10.0  # optical axis this far off straight down: rectify first
OBLIQUE_MAX_INCIDENCE_DEG = 50.0  # view ray vs the lid's normal at the button, beyond which no aim
# The same angle at the prior (or the optical axis' hit) is only a sanity
# bound: a prior 10-12 cm outward reads 10-14 deg more than the button does.
OBLIQUE_PRIOR_MAX_DEG = 65.0
OBLIQUE_BUTTON_PX = 24.0  # the button's radius in the rectified image (fixes the scale)
OBLIQUE_HALF_M = 0.16  # rectified half-extent: a 12 cm prior error plus the annulus
OBLIQUE_RADIUS_WINDOW = (0.85, 1.2)  # of OBLIQUE_BUTTON_PX; a lid off its prior scales the circle
OBLIQUE_MAX_CIRCLES = 8
LID_PRIOR_TOL_M = 0.035  # the measured lid within this of its prior height
PRIOR_MAX_M = 0.15  # a circle farther than this from prior_xy is not the button
# ... and from straight down, where a circle's place is read straight off the
# frame: farther than this from where the box is, it is not this box's
# button. Flatbot 2026-10-23 15:09: a 47 mm circle on a cabinet's side, 28 cm
# from the box, aimed the press and the centring flew 31 cm to it. The
# allowance: the wrist mount's TF, 3.5 cm off sideways until
# rammp-deployments#128, and the scene camera's own few centimetres.
DOWN_PRIOR_MAX_M = 0.07
# of the disc's rectified cells with valid depth. Seen at a slant the glossy
# button holds depth in patches, the lid ring around it more (flatbot
# 2026-10-27 14:04: the button's circle found at its size and place in 31
# frames, refused in all for depth on 9-24 % of the disc while the ring held
# 39 % at one height, and the disc's own patches lay flat on it). Every other
# gate still has to pass: the disc flat, the ring one lid at the lid's
# height, the rays meeting it, the size and the place.
DISC_MIN_FILL = 0.15
# A tilted view's aim reads the depth of this many still frames at once
# (scene.median_depth, holes ignored): the patches move frame to frame
FUSE_FRAMES = 5
ANNULUS_MIN_FILL = 0.3  # of the annulus' cells; a popped knob hides its far side
KNOB_NOISE_M = 0.0025  # disc and ring medians closer than this are one height
KNOB_SPLIT_M = 0.006  # disc this far above its lid: two circles the centroid tells apart
# a first frame is still when the camera's pose this long before its stamp
# is its pose at it: over two of the wrist camera's frame intervals (the
# OAK's ~75-90 ms, run logs 2026-10-22), so a stricter test than two
# consecutive frames — and the reach-out aims after a 0.46 s at-rest read
FIRST_LOOKBACK_S = 0.2
OBLIQUE_WINDOW = 3.0  # of the circle's rectified radius: the cells a candidate's gates read
# The lid's prior height is judged against the table THIS frame sees. Since
# its repair the wrist reads every height ~20 mm low (flatbot 2026-10-27: its
# own table 18.1-21.6 mm below the scene camera's in the four reach-out
# aims, its lid rings 22-32 mm below the scene's lid), and the live aims
# cleared LID_PRIOR_TOL_M by only 2.7-13.0 mm: a slightly worse day refuses
# every frame and presses nothing. A common offset moves the wrist's table
# and lid alike, so the two prior-height gates move by it (and past
# TABLE_RECTIFY_M the rectifying plane: the circles at their own scale). The
# press point's xy and the heights reported stay as measured:
# the 14:33 close-up found the button 4 mm from the measured aim, and 18 mm
# from that aim re-slid to the scene camera's height.
TABLE_OFFSET_MAX_M = 0.030  # a wrist farther off than this is recalibrated at the source, not aimed through
# ... and only while tan(incidence at the prior) x the offset stays within
# this: how far apart the press xy slid by the measured heights and by the
# offset ones can be. Past ~20 mm which of them is right decides whether
# the press lands on the 22 mm-radius button.
TABLE_OFFSET_XY_MAX_M = 0.020
TABLE_SUBSAMPLE = 3  # the rectified grid's every third cell each way: under 1 ms
TABLE_MIN_CELLS = 500  # of those, below half the box's height (the live frames held 6.5-7k)
TABLE_BIN_M = 0.002  # the table: the mode of their heights in bins this wide,
TABLE_BAND_M = 0.004  # then the median of the heights within this of its centre
TABLE_AGREE_M = 0.004  # two of a fix's accepted offsets this close raise its z (ButtonAimer.fix)
# The two prior-height gates, though, judge the lid against the table THIS
# frame sees however far that stands from the one given (owner 2026-10-30:
# the scene camera gives the arm a general location, the wrist finds the
# lid): 2026-10-30 10:09 and 10:10 the wrist's table stood 43-55 mm below
# the scene camera's, past both bounds above, its lid with it, and every
# frame was refused. Only a "table" farther off than this is no table.
TABLE_GATE_MAX_M = 0.150
# ... and past this the frame is rectified again on the plane a box's height
# above that table: on the plane at the scene camera's lid the circles are
# scaled by its error. 2026-10-30 10:40 and 10:58 the wrist's table stood
# 57-59 mm below the scene camera's: the 44 mm knob read ~17% small, near
# the radius window's floor, and both regrips' re-aims took the lid's 60 mm
# ring (59.5, 61.2 mm) for it, 22-26 mm off
TABLE_RECTIFY_M = 0.010


@dataclass(frozen=True)
class ButtonSighting:
    xyz: tuple  # the PRESS POINT, base frame: the circle's centre xy, the disc's median z
    uv: tuple  # the seam circle's pixel centre — the press point's pixel
    r_px: float
    # top-down: the camera z-depth at the centre pixel; oblique: |P - C|,
    # the press point's distance from the camera along its ray
    range_m: float
    diameter_m: float
    # how far the disc stands above the lid ring around it, in THIS frame
    # (nan without depth on the ring): ~0 closed, ~+10 mm with the knob up.
    # A difference inside one frame — the mount and the table cancel.
    above_lid_m: float = float("nan")
    lid_z_m: float = float("nan")  # the lid ring's measured base z (oblique path)
    incidence_deg: float = 0.0  # view ray vs the lid's +z normal at the press point
    oblique: bool = False  # found by find_button_oblique
    disc_fill: float = float("nan")  # the share of the disc with depth (oblique path)
    # the table this frame sees less the table the aim was given (m, oblique
    # path, _table_offset): accepted — the prior-height gates moved by it —
    # else nan; and raw, as read, accepted or not. xyz and lid_z_m do not
    # include it.
    table_offset_m: float = float("nan")
    table_offset_raw_m: float = float("nan")


def centre_depth(depth, u, v, r):
    """The range at the circle's centre (m), or nan: the 7x7 window there,
    or — that window empty — the disc's (0.8 r) median, which the gates hold
    flat anyway. The flat pink disc seen straight down reads patchily, and
    its centre window was empty in a whole aim's 38 frames (flatbot
    2026-10-23 16:34) while the disc around it held depth."""
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    win = depth[max(0, vi - 3) : vi + 4, max(0, ui - 3) : ui + 4]
    valid = win[(win > DEPTH_MIN_M) & (win < DEPTH_MAX_M) & np.isfinite(win)]
    if valid.size >= 5:
        return float(np.median(valid))
    v0, u0, d2 = _window_d2(h, w, u, v, r)
    disc = depth[v0 : v0 + d2.shape[0], u0 : u0 + d2.shape[1]][d2 < (0.8 * r) ** 2]
    disc = disc[(disc > DEPTH_MIN_M) & (disc < DEPTH_MAX_M) & np.isfinite(disc)]
    return float(np.median(disc)) if disc.size >= 20 else float("nan")


def circle_gates(depth, k, rot_cam, trans_cam, u, v, r, lid_z):
    """Why this Hough circle is not the button, or None (it may be).

    depth is metres at colour pixels. lid_z: the expected lid top height
    in base. The centre must sit at lid height; the disc must be flat; the
    annulus around it must ALSO be at lid height — a rounded lid corner
    passes the first two and fails the third (the real frame of 2026-09-17
    had four of them at Hough radii 25-34 px).

    "Lid height" is the lid this frame measures — the annulus around the
    circle — when that stands within LID_PRIOR_TOL_M of `lid_z`, else
    `lid_z` itself. `lid_z` is the scene camera's table plus the box's
    height, and that table swung 0.180-0.200 m over one afternoon with the
    box standing still (flatbot 2026-10-23): judged against it alone, the
    real button read "12 mm from lid height" and was refused."""
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if ui - 3 < 0 or vi - 3 < 0 or ui + 4 > w or vi + 4 > h:
        return "at the image border"
    zc = centre_depth(depth, u, v, r)
    if not np.isfinite(zc):
        return "no depth at the centre"
    kk = np.asarray(k, float)
    here = ring_height(depth, kk, rot_cam, trans_cam, u, v, r)
    if np.isfinite(here) and abs(here - lid_z) <= LID_PRIOR_TOL_M:
        lid_z = here  # the lid where this frame measures it
    pc = np.array([(u - kk[0, 2]) / kk[0, 0] * zc, (v - kk[1, 2]) / kk[1, 1] * zc, zc])
    p = np.asarray(rot_cam, float) @ pc + np.asarray(trans_cam, float)
    # at lid height — or ABOVE it by as much as a popped knob stands: an
    # open box's button is still its button, and the mission must see it to
    # refuse to press it shut (tasks.press_demo.refuse_open_box)
    if p[2] - lid_z < -TOP_RESIDUAL_MAX_M or p[2] - lid_z > KNOB_MAX_UP_M:
        return "centre %.0f mm from lid height" % (1000 * (p[2] - lid_z))
    v0, u0, d2 = _window_d2(h, w, u, v, ANNULUS[1] * r)
    win = depth[v0 : v0 + d2.shape[0], u0 : u0 + d2.shape[1]]
    disc = win[d2 < (0.8 * r) ** 2]
    disc = disc[(disc > DEPTH_MIN_M) & (disc < DEPTH_MAX_M) & np.isfinite(disc)]
    if disc.size < 20 or float(np.std(disc)) > DISC_STD_MAX_M:
        return "disc not flat (%d px, std %.1f mm)" % (disc.size, 1000 * float(np.std(disc)) if disc.size else 0.0)
    ring = (d2 > (ANNULUS[0] * r) ** 2) & (d2 < (ANNULUS[1] * r) ** 2)
    rd = win[ring]
    ok = (rd > DEPTH_MIN_M) & (rd < DEPTH_MAX_M) & np.isfinite(rd)
    if ok.sum() < 30:
        return "no depth around the circle"
    # heights of the ring's points, in base z (the camera looks down; each
    # pixel's own depth is lifted)
    rv, ru = np.nonzero(ring)
    ring_pts = _lift(depth, kk, rot_cam, trans_cam, ru[ok] + u0, rv[ok] + v0)
    frac = float(np.mean(np.abs(ring_pts[:, 2] - lid_z) < ANNULUS_TOL_M))
    if frac < ANNULUS_MIN_FRAC:
        return "ring around it is not lid (%.0f%% at lid height) — a corner?" % (100 * frac)
    return None


def _window_d2(h, w, u, v, r_max):
    """(v0, u0, d2): the squared pixel distance from (u, v) over the image's
    window that holds every pixel nearer than r_max — outside it no
    strict d2 < r_max ** 2 test can hold, so masks over it are the
    full-frame masks, in the same row-major order. A centre off the frame
    gets an empty window (this numpy's mgrid refuses a stop below its
    start)."""
    m = int(math.ceil(r_max)) + 1
    v0 = min(h, max(0, int(math.floor(v)) - m))
    u0 = min(w, max(0, int(math.floor(u)) - m))
    v1 = max(v0, min(h, int(math.ceil(v)) + m + 1))
    u1 = max(u0, min(w, int(math.ceil(u)) + m + 1))
    yy, xx = np.mgrid[v0:v1, u0:u1]
    return v0, u0, (xx - u) ** 2 + (yy - v) ** 2


def _lift(depth, k, rot_cam, trans_cam, us, vs):
    """Base-frame points for the pixels (us, vs) from their own depth."""
    kk = np.asarray(k, float)
    zc = depth[vs, us]
    pc = np.c_[(us - kk[0, 2]) / kk[0, 0] * zc, (vs - kk[1, 2]) / kk[1, 1] * zc, zc]
    return pc @ np.asarray(rot_cam, float).T + np.asarray(trans_cam, float)


def disc_height(depth, k, rot_cam, trans_cam, u, v, r):
    """The base-frame height of the button's top: the median over the disc
    inside the seam circle (u, v, r), or nan without depth there. One pixel
    at the centre is noisy by 2-3 mm at 25 cm; the disc is what the closed
    pads meet."""
    h, w = depth.shape
    v0, u0, d2 = _window_d2(h, w, u, v, r)
    win = depth[v0 : v0 + d2.shape[0], u0 : u0 + d2.shape[1]]
    disc = d2 < r**2
    disc &= (win > DEPTH_MIN_M) & (win < DEPTH_MAX_M) & np.isfinite(win)
    vs, us = np.nonzero(disc)
    if len(us) < 50:
        return float("nan")
    return float(np.median(_lift(depth, k, rot_cam, trans_cam, us + u0, vs + v0)[:, 2]))


def ring_height(depth, k, rot_cam, trans_cam, u, v, r):
    """The base-frame height of the LID around the button: the median over
    the annulus the gates judge (ANNULUS), or nan."""
    h, w = depth.shape
    v0, u0, d2 = _window_d2(h, w, u, v, ANNULUS[1] * r)
    win = depth[v0 : v0 + d2.shape[0], u0 : u0 + d2.shape[1]]
    ring = (d2 > (ANNULUS[0] * r) ** 2) & (d2 < (ANNULUS[1] * r) ** 2)
    ring &= (win > DEPTH_MIN_M) & (win < DEPTH_MAX_M) & np.isfinite(win)
    vs, us = np.nonzero(ring)
    if len(us) < 30:
        return float("nan")
    return float(np.median(_lift(depth, k, rot_cam, trans_cam, us + u0, vs + v0)[:, 2]))


def view_tilt_deg(rot_cam):
    """The angle between the camera's optical axis and straight down."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(np.asarray(rot_cam, float)[2, 2])))))


def find_button(
    color_bgr, depth, k, rot_cam, trans_cam, table_z, model, dist=None, lid_z=None, prior_xy=None, prior_max_m=None
):
    """The button in one still frame: (ButtonSighting, None) or (None, why).
    The sighting's xyz is the PRESS POINT: the seam circle's centre.

    `dist` (the colour stream's plumb_bob coefficients) corrects the
    centre for lens distortion before it becomes a ray — the stream is not
    rectified, and near the image edge a pinhole lift was 1-2 mm off
    (2026-09-23). The depth is still read at the raw pixel, where the
    aligned depth lives.

    `lid_z`: the lid top's expected base z (default table_z + dims.z) and
    `prior_xy`: where the button is expected in base xy. Only the oblique
    path uses lid_z: the top-down staging aim's lid is at table_z + dims.z
    whatever lid_z says. Both use prior_xy: from straight down a circle
    farther than DOWN_PRIOR_MAX_M (or `prior_max_m`, a caller that knows
    the place better) from it is not the button. A camera axis more than
    OBLIQUE_MIN_DEG off straight down goes to find_button_oblique with the
    same arguments."""
    if rot_cam is not None and view_tilt_deg(rot_cam) > OBLIQUE_MIN_DEG:
        return find_button_oblique(
            color_bgr, depth, k, rot_cam, trans_cam, table_z, model, dist=dist, lid_z=lid_z, prior_xy=prior_xy
        )
    import cv2

    if color_bgr is None or depth is None or k is None:
        return None, "no frame"
    kk = np.asarray(k, float)
    lid_z = float(table_z) + float(model.dims[2])
    rng = float(trans_cam[2]) - lid_z  # the camera looks straight down at staging
    if rng < MIN_RANGE_M:
        return None, "camera %.0f mm above the lid — too close to aim" % (1000 * rng)
    r_px = kk[0, 0] * (float(model.button_diameter_m) / 2.0) / rng
    r_lo, r_hi = int(RADIUS_WINDOW[0] * r_px), int(math.ceil(RADIUS_WINDOW[1] * r_px))
    if r_lo < 6:
        return None, "button would be %.0f px — too far to aim" % r_px
    gray = cv2.medianBlur(cv2.cvtColor(np.ascontiguousarray(color_bgr), cv2.COLOR_BGR2GRAY), 3)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=2 * r_px,
        param1=90,
        param2=14,
        minRadius=r_lo,
        maxRadius=r_hi,
    )
    if circles is None:
        return None, "no circle of %.0f-%.0f px radius in the frame" % (r_lo, r_hi)
    whys = []
    prior_max = DOWN_PRIOR_MAX_M if prior_max_m is None else float(prior_max_m)
    for u, v, r in circles[0][:8]:  # Hough's own order: strongest first
        why = circle_gates(depth, kk, rot_cam, trans_cam, float(u), float(v), float(r), lid_z)
        if why is not None:
            whys.append(why)
            continue
        zc = centre_depth(depth, float(u), float(v), float(r))
        # the centre's xy from its own (undistorted) pixel at its own range;
        # the height is the disc's median — the honest "top" for the
        # calibration line
        uu, vv = float(u), float(v)
        if dist is not None and np.any(np.asarray(dist, float)):
            uu, vv = cv2.undistortPoints(np.array([[[uu, vv]]], np.float64), kk, np.asarray(dist, float), P=kk)[0, 0]
        pc = np.array([(uu - kk[0, 2]) / kk[0, 0] * zc, (vv - kk[1, 2]) / kk[1, 1] * zc, zc])
        p = np.asarray(rot_cam, float) @ pc + np.asarray(trans_cam, float)
        if prior_xy is not None:
            miss = math.hypot(p[0] - float(prior_xy[0]), p[1] - float(prior_xy[1]))
            if miss > prior_max:
                whys.append("circle %.0f mm from where the box is (max %.0f)" % (1000 * miss, 1000 * prior_max))
                continue
        z_disc = disc_height(depth, kk, rot_cam, trans_cam, float(u), float(v), float(r))
        z_top = z_disc if np.isfinite(z_disc) else float(p[2])
        z_ring = ring_height(depth, kk, rot_cam, trans_cam, float(u), float(v), float(r))
        return ButtonSighting(
            xyz=(float(p[0]), float(p[1]), z_top),
            uv=(float(u), float(v)),
            r_px=float(r),
            range_m=zc,
            diameter_m=2.0 * float(r) * zc / kk[0, 0],
            above_lid_m=float(z_disc - z_ring),
        ), None
    return None, "%d circle(s), none the button: %s" % (len(circles[0]), "; ".join(whys[:3]))


def _project(pts, k, rot_cam, trans_cam, dist, shape):
    """Raw pixels (N, 2) of base-frame points (N, 3); nan behind the camera
    or, by the pinhole, beyond a quarter frame outside the image (there a
    distortion polynomial can fold a point back into the frame)."""
    pc = (np.atleast_2d(np.asarray(pts, float)) - np.asarray(trans_cam, float)) @ np.asarray(rot_cam, float)
    return _project_cam(pc, k, dist, shape)


def _project_cam(pc, k, dist, shape):
    """_project for points (N, 3) already in the camera frame."""
    import cv2

    kk = np.asarray(k, float)
    out = np.full((len(pc), 2), np.nan, pc.dtype)
    z = np.where(pc[:, 2] > 1e-6, pc[:, 2], np.nan)
    u = kk[0, 0] * pc[:, 0] / z + kk[0, 2]
    v = kk[1, 1] * pc[:, 1] / z + kk[1, 2]
    h, w = shape
    with np.errstate(invalid="ignore"):
        near = (u > -0.25 * w) & (u < 1.25 * w) & (v > -0.25 * h) & (v < 1.25 * h)
    if dist is None:
        out[near] = np.c_[u[near], v[near]]
    elif near.any():
        uv, _ = cv2.projectPoints(pc[near].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), kk, dist)
        out[near] = uv.reshape(-1, 2)
    return out


def _pixel_rays(us, vs, k, dist):
    """Normalised camera coordinates (x/z, y/z) of raw pixels."""
    import cv2

    kk = np.asarray(k, float)
    if dist is None:
        return (us - kk[0, 2]) / kk[0, 0], (vs - kk[1, 2]) / kk[1, 1]
    nrm = cv2.undistortPoints(np.c_[us, vs].astype(np.float64).reshape(-1, 1, 2), kk, dist).reshape(-1, 2)
    return nrm[:, 0], nrm[:, 1]


def _incidence_deg(p, cam):
    """The angle between the ray p -> camera and the lid's +z normal."""
    ray = np.asarray(cam, float) - np.asarray(p, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, ray[2] / float(np.linalg.norm(ray))))))


def _oblique_circle(lifted, cu, cv, r, s, c0, g, h0, cam, rb, h_gate=None):
    """((P, h_disc, h_ring, f), None) for the rectified circle (cu, cv, r),
    or (None, why). lifted: (n, n, 3) base points of the cells' depth; f
    scales the rectified circle to metric at the height it lies at. Only the
    cells within OBLIQUE_WINDOW * r of the circle are read: every sample a
    gate counts lies within 2.5 rb of the centre and a knob's height of it,
    and such a sample's ray crosses h0 no farther out than that. h_gate:
    where this frame should see the lid (h0 moved by the table offset,
    default h0); the two prior-height gates alone read it."""
    h_gate = h0 if h_gate is None else float(h_gate)
    half = int(math.ceil(OBLIQUE_WINDOW * r))
    i0, j0 = max(0, int(cv) - half), max(0, int(cu) - half)
    win = lifted[i0 : int(cv) + half + 1, j0 : int(cu) + half + 1]
    yy = np.arange(i0, i0 + win.shape[0], dtype=np.float32)[:, None]  # the window's cells
    xx = np.arange(j0, j0 + win.shape[1], dtype=np.float32)[None, :]
    zz = win[..., 2]
    fin = np.isfinite(zz)
    cell = (xx - cu) ** 2 + (yy - cv) ** 2 < (0.8 * r) ** 2
    zd = zz[cell & fin]
    if zd.size < max(20, DISC_MIN_FILL * cell.sum()):
        return None, "too little depth on the disc (%d of %d cells)" % (zd.size, cell.sum())
    h_circle = float(np.median(zd))
    if not -LID_PRIOR_TOL_M - TOP_RESIDUAL_MAX_M <= h_circle - h_gate <= LID_PRIOR_TOL_M + KNOB_MAX_UP_M:
        return None, "disc %.0f mm from the lid's prior height" % (1000 * (h_circle - h_gate))
    p_rect = np.array([g[0] + (cu - c0) / s, g[1] + (cv - c0) / s, h0])

    def centre(h):
        # the circle on h0 is a circle at h seen through C: slide it along
        # the rays (exact for a circle parallel to the plane)
        return cam + (p_rect - cam) * (cam[2] - h) / (cam[2] - h0)

    def levels(p):
        # disc and lid ring by each sample's OWN base xy: metric, whatever the view
        q = np.where(fin, (win[..., 0] - p[0]) ** 2 + (win[..., 1] - p[1]) ** 2, np.inf)
        return q, zz[q < (0.8 * rb) ** 2], zz[(q > (ANNULUS[0] * rb) ** 2) & (q < (ANNULUS[1] * rb) ** 2)]

    # Which height the circle lies at: the disc's ("disc"), the lid ring's
    # ("ring"), or halfway between ("mid").
    at = "disc"
    p = centre(h_circle)
    q, _, zr = levels(p)
    if zr.size >= 30:
        h_lid = float(np.median(zr))
        if h_circle - h_lid < -KNOB_NOISE_M:
            # Recessed: the circle is the hole's rim, on the lid, and through
            # it shows the floor's far part (the near rim hides the rest).
            # Slid to the floor's height the centre went tan(incidence) x
            # the depth away from the camera: 8 mm at -10 mm and 39 deg.
            at = "ring"
        elif h_circle - h_lid > KNOB_SPLIT_M:
            # A proud knob shows two circles: its top's edge, at the disc's
            # height, and its base on the lid, at the ring's — and they meet
            # the rays tan(incidence) * height apart. Its top is wholly in
            # view: the centroid of the points nearer its height than the
            # lid's says which.
            with np.errstate(invalid="ignore"):
                top = (q < (2.5 * rb) ** 2) & (np.abs(zz - h_circle) < np.abs(zz - h_lid))
            if top.sum() >= 20:
                cxy = win[top][:, :2].mean(0)
                at = "ring" if np.hypot(*(centre(h_lid)[:2] - cxy)) < np.hypot(*(p[:2] - cxy)) else "disc"
        elif h_circle - h_lid > KNOB_NOISE_M:
            # A knob a few mm proud: too low for its top's points to stand
            # out of the depth noise, so which circle Hough took is unknown.
            # Halfway between them is off by at most half their split
            # (tan(incidence) x height / 2: 2.4 mm at 6 mm and 39 deg).
            at = "mid"
        h_circle = {"disc": h_circle, "ring": h_lid, "mid": 0.5 * (h_circle + h_lid)}[at]
    for _ in range(2):  # the heights around the moved centre, and the centre again
        p = centre(h_circle)
        _, zd, zr = levels(p)
        if zd.size < 20 or zr.size < 30:
            break
        h_d, h_r = float(np.median(zd)), float(np.median(zr))
        h_circle = {"disc": h_d, "ring": h_r, "mid": 0.5 * (h_d + h_r)}[at]
    p = centre(h_circle)
    _, zd, zr = levels(p)
    h_disc = float(np.median(zd)) if zd.size else h_circle
    h_ring = float(np.median(zr)) if zr.size else h_circle

    def cells(area, h):  # the rectified cells a metric area at height h covers
        return area * (s * (cam[2] - h0) / (cam[2] - h)) ** 2

    expect = cells(math.pi * (0.8 * rb) ** 2, h_disc)
    if zd.size < max(20, DISC_MIN_FILL * expect):
        return None, "too little depth on the disc (%d of ~%.0f cells)" % (zd.size, expect)
    if float(np.std(zd)) > DISC_STD_MAX_M:
        return None, "disc not flat in base z (std %.1f mm)" % (1000 * float(np.std(zd)))
    expect = cells(math.pi * (ANNULUS[1] ** 2 - ANNULUS[0] ** 2) * rb**2, h_ring)
    if zr.size < max(30, ANNULUS_MIN_FILL * expect):
        return None, "too little depth around the circle (%d of ~%.0f cells)" % (zr.size, expect)
    frac = float(np.mean(np.abs(zr - h_ring) < ANNULUS_TOL_M))
    if frac < ANNULUS_MIN_FRAC:
        return None, "ring around it is not one lid (%.0f%% at its height) — a corner?" % (100 * frac)
    # The same ring by where each cell's RAY crosses its plane. Past the
    # lid's far rim the box hides the table from this view, so no sample's
    # own xy lands there and the ring above reads all lid for a circle
    # straddling that rim; its rays went on past the lid, to land lower. A
    # ray met higher up is not counted: a popped knob hides the far side.
    fr = (cam[2] - h_ring) / (cam[2] - h0)
    rx = cam[0] + (g[0] + (xx - c0) / s - cam[0]) * fr - p[0]
    ry = cam[1] + (g[1] + (yy - c0) / s - cam[1]) * fr - p[1]
    rho2 = rx**2 + ry**2
    zb = zz[fin & (rho2 > (ANNULUS[0] * rb) ** 2) & (rho2 < (ANNULUS[1] * rb) ** 2)]
    on_lid = int(np.sum(np.abs(zb - h_ring) < ANNULUS_TOL_M))
    past = int(np.sum(zb <= h_ring - ANNULUS_TOL_M))
    if on_lid + past < 30:
        return None, "no lid seen around the circle (%d cells)" % (on_lid + past)
    if on_lid < ANNULUS_MIN_FRAC * (on_lid + past):
        return None, "ring around it is not one lid (%.0f%% of its rays meet it) — a corner?" % (
            100.0 * on_lid / (on_lid + past)
        )
    if abs(h_ring - h_gate) > LID_PRIOR_TOL_M:
        return None, "lid ring %.0f mm from its prior height" % (1000 * (h_ring - h_gate))
    up = h_disc - h_ring
    if up < -TOP_RESIDUAL_MAX_M or up > KNOB_MAX_UP_M:
        return None, "disc %.0f mm from the lid around it" % (1000 * up)
    fill = zd.size / max(cells(math.pi * (0.8 * rb) ** 2, h_disc), 1.0)
    return (np.r_[p[:2], h_disc], h_disc, h_ring, (cam[2] - h_circle) / (cam[2] - h0), fill), None


def _table_offset(lifted, h0, table_z, inc_deg):
    """(accepted, raw): how far the table this frame sees stands from
    `table_z`, the one the aim was given (m). The table is the mode, in
    TABLE_BIN_M bins, of the lifted heights below half the box's height
    (every TABLE_SUBSAMPLE-th cell each way), then the median of those
    within TABLE_BAND_M of its centre. Both nan with fewer than
    TABLE_MIN_CELLS there; accepted is nan unless the offset is within
    TABLE_OFFSET_MAX_M and tan(inc_deg) x the offset (the incidence at the
    prior) within TABLE_OFFSET_XY_MAX_M."""
    nan = float("nan")
    if table_z is None or not np.isfinite(table_z) or not float(h0) > float(table_z):
        return nan, nan
    z = lifted[::TABLE_SUBSAMPLE, ::TABLE_SUBSAMPLE, 2]
    z = z[np.isfinite(z)]
    z = z[z < float(h0) - 0.5 * (float(h0) - float(table_z))]
    if z.size < TABLE_MIN_CELLS:
        return nan, nan
    lo = float(z.min())
    mode = lo + (int(np.argmax(np.bincount(((z - lo) / TABLE_BIN_M).astype(int)))) + 0.5) * TABLE_BIN_M
    raw = float(np.median(z[np.abs(z - mode) <= TABLE_BAND_M])) - float(table_z)
    if abs(raw) > TABLE_OFFSET_MAX_M or math.tan(math.radians(inc_deg)) * abs(raw) > TABLE_OFFSET_XY_MAX_M:
        return nan, raw
    return raw, raw


def _table_words(off, off_raw):
    """What a refusal says of the table its frame saw (_table_offset): the
    lid judged against it, or not (TABLE_GATE_MAX_M); the fix's z raised by
    it (taken out), or not."""
    if not np.isfinite(off_raw):
        return ""
    if abs(off_raw) > TABLE_GATE_MAX_M:
        return (
            "; the table in view %+.0f mm from the one given, too far to be the table: the lid judged against "
            "the one given" % (1000 * off_raw)
        )
    return "; the table in view %+.0f mm from the one given, the lid judged against it, %s" % (
        1000 * off_raw,
        "taken out" if np.isfinite(off) else "not taken out",
    )


def _prior_miss_m(cu, cv, s, c0, g, h0, cam, prior_xy, h_gate=None):
    """How near the rectified circle (cu, cv) can come to prior_xy at any
    height the gates allow it: the circle on h0 slid along the rays over
    that band of heights, around h_gate as theirs is (default h0), is a
    segment in xy."""
    h_gate = h0 if h_gate is None else float(h_gate)
    p_rect = np.array([g[0] + (cu - c0) / s, g[1] + (cv - c0) / s, h0])
    ends = [
        (cam + (p_rect - cam) * (cam[2] - h) / (cam[2] - h0))[:2]
        for h in (h_gate + LID_PRIOR_TOL_M + KNOB_MAX_UP_M, h_gate - LID_PRIOR_TOL_M - TOP_RESIDUAL_MAX_M)
    ]
    seg = ends[1] - ends[0]
    w = np.asarray(prior_xy, float) - ends[0]
    t = min(1.0, max(0.0, float(w @ seg) / max(float(seg @ seg), 1e-12)))
    return float(np.hypot(*(w - t * seg)))


def _lid_plane(prior_xy, h0, kk, rot, cam, dd, hw):
    """Where the rectified grid is centred on the plane z = h0: prior_xy
    there, else where the optical axis meets it. (g, None), or (None, why)."""
    hh, ww = hw
    if prior_xy is not None:
        g = np.array([float(prior_xy[0]), float(prior_xy[1]), h0])
        uv = _project(g, kk, rot, cam, dd, (hh, ww))[0]
        if not (np.all(np.isfinite(uv)) and 0 <= uv[0] <= ww - 1 and 0 <= uv[1] <= hh - 1):
            return None, "the prior button (%.3f, %.3f) projects outside the image" % (g[0], g[1])
        return g, None
    axis = rot[:, 2]
    if axis[2] > -1e-3:
        return None, "the optical axis does not meet the lid plane"
    return cam + axis * (h0 - cam[2]) / axis[2], None


def _rectified(color_bgr, depth, kk, rot, cam, dd, g, rb):
    """The frame resampled onto the horizontal plane through `g` on a metric
    grid centred there (OBLIQUE_BUTTON_PX for the button's radius `rb`),
    each cell's depth lifted along its own raw pixel's ray: ((s, c0, gray,
    lifted), None) — px per metre, the grid's centre cell, the rectified
    grey image, the cells' base points — or (None, why)."""
    import cv2

    hh, ww = depth.shape[:2]
    s = OBLIQUE_BUTTON_PX / rb  # rectified px per metre
    n = 2 * int(round(OBLIQUE_HALF_M * s)) + 1
    c0 = (n - 1) / 2.0
    ax = ((np.arange(n) - c0) / s).astype(np.float32)
    # the cells in the camera frame, in float32 and by rows and columns
    # (columns along base x, rows along base y): R^T (g - C) + x R^T ex + y R^T ey
    rf = rot.astype(np.float32)
    pc = (rot.T @ (g - cam)).astype(np.float32) + ax[None, :, None] * rf[0] + ax[:, None, None] * rf[1]
    uv = _project_cam(pc.reshape(-1, 3), kk, dd, (hh, ww))
    with np.errstate(invalid="ignore"):
        valid = (uv[:, 0] >= 0) & (uv[:, 0] <= ww - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= hh - 1)
    if valid.sum() < 100:
        return None, "the lid plane around the button is not in the frame"
    uvf = np.where(valid[:, None], uv, np.float32(-1.0)).astype(np.float32)
    mx = uvf[:, 0].reshape(n, n)
    my = uvf[:, 1].reshape(n, n)
    rect = cv2.remap(np.ascontiguousarray(color_bgr), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    vgrid = valid.reshape(n, n)
    gray[~vgrid] = np.uint8(np.median(gray[vgrid]))  # no edge where the frame ends
    gray = cv2.medianBlur(gray, 3)
    # depth by nearest pixel, lifted along that pixel's own ray
    us = np.rint(uvf[valid, 0]).astype(int)
    vs = np.rint(uvf[valid, 1]).astype(int)
    d = np.asarray(depth)[vs, us].astype(float)
    ok = np.isfinite(d) & (d > DEPTH_MIN_M) & (d < DEPTH_MAX_M)
    if ok.sum() < 100:
        return None, "no depth on the lid plane around the button"
    xn, yn = _pixel_rays(us[ok], vs[ok], kk, dd)
    lifted = np.full((n * n, 3), np.nan, np.float32)
    lifted[np.flatnonzero(valid)[ok]] = np.c_[xn * d[ok], yn * d[ok], d[ok]] @ rot.T + cam
    lifted = lifted.reshape(n, n, 3)
    return (s, c0, gray, lifted), None


def find_button_oblique(color_bgr, depth, k, rot_cam, trans_cam, table_z, model, dist=None, lid_z=None, prior_xy=None):
    """The button seen from a tilted pose: (ButtonSighting, None) or (None, why).

    The frame is resampled onto the horizontal plane z = h0 (lid_z, default
    table_z + dims.z) on a metric grid centred on prior_xy (else where the
    optical axis meets the plane). There a horizontal circle is a circle.
    Each cell's depth is lifted along its own raw pixel's ray, so every gate
    reads base z and the lid ring's height is measured, not assumed: it
    must lie within LID_PRIOR_TOL_M of h0 moved by the table offset — a
    box's height above the table this frame reads (_table_offset,
    TABLE_GATE_MAX_M). The circle found on h0 is then moved along the
    camera's rays to the disc's measured height h:
    P = C + (P' - C) (Cz - h) / (Cz - h0)."""
    import cv2

    if color_bgr is None or depth is None or k is None:
        return None, "no frame"
    kk = np.asarray(k, float)
    rot = np.asarray(rot_cam, float)
    cam = np.asarray(trans_cam, float)
    dd = np.asarray(dist, float).ravel() if dist is not None and np.any(np.asarray(dist, float)) else None
    h0 = float(table_z) + float(model.dims[2]) if lid_z is None else float(lid_z)
    rb = float(model.button_diameter_m) / 2.0
    hh, ww = depth.shape[:2]
    if cam[2] - h0 < MIN_RANGE_M:  # the range is at least this
        return None, "camera %.0f mm above the lid — too close to aim" % (1000 * (cam[2] - h0))
    g, why = _lid_plane(prior_xy, h0, kk, rot, cam, dd, (hh, ww))
    if g is None:
        return None, why
    inc = _incidence_deg(g, cam)  # at the prior, not the button: a sanity bound only
    if inc > OBLIQUE_PRIOR_MAX_DEG:
        return None, "the lid seen %.0f deg off its normal — too oblique to aim" % inc
    got, why = _rectified(color_bgr, depth, kk, rot, cam, dd, g, rb)
    if got is None:
        return None, why
    s, c0, gray, lifted = got
    # the lid where this frame should see it: a box's height above the table
    # it sees — h0 moved by that table's offset from table_z, as read
    # (TABLE_GATE_MAX_M); the fix's z is raised only by an offset taken out
    # (TABLE_OFFSET_*, ButtonAimer.fix). Past TABLE_RECTIFY_M the frame is
    # rectified again on that plane, so the circles are measured at the
    # scale they lie at (TABLE_RECTIFY_M's 10-30 regrips)
    table_off, table_raw = _table_offset(lifted, h0, table_z, inc)
    seen = bool(np.isfinite(table_raw) and abs(table_raw) <= TABLE_GATE_MAX_M)
    h_gate = h0 + (table_raw if seen else 0.0)
    if seen and abs(table_raw) > TABLE_RECTIFY_M:
        g1, _why = _lid_plane(prior_xy, h_gate, kk, rot, cam, dd, (hh, ww))
        got1, _why = (None, None) if g1 is None else _rectified(color_bgr, depth, kk, rot, cam, dd, g1, rb)
        if got1 is not None:
            h0, g = h_gate, g1
            s, c0, gray, lifted = got1

    r0 = OBLIQUE_BUTTON_PX
    r_lo, r_hi = int(OBLIQUE_RADIUS_WINDOW[0] * r0), int(math.ceil(OBLIQUE_RADIUS_WINDOW[1] * r0))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=2 * r0,
        param1=90,
        param2=14,
        minRadius=r_lo,
        maxRadius=r_hi,
    )
    if circles is None:
        return None, "no circle of %d-%d px radius on the rectified lid plane" % (r_lo, r_hi)
    cands = [(float(u), float(v), float(r)) for u, v, r in circles[0]]  # strongest first
    if prior_xy is not None:
        # nearest the prior first (the scene camera chose that box); the grid
        # is centred on the prior, so a cell's offset from c0 is its offset
        # from it ON h0. The sort is stable: equally near, strongest first.
        # A lid above h0 meets the rays nearer the camera than h0 does, so
        # the cut is where the circle can lie at any allowed height; the
        # metric check after the correction is the one that holds.
        off = [math.hypot(u - c0, v - c0) / s for u, v, _ in cands]
        near = sorted(
            (
                i
                for i in range(len(cands))
                if _prior_miss_m(*cands[i][:2], s, c0, g, h0, cam, prior_xy, h_gate) <= PRIOR_MAX_M
            ),
            key=lambda i: off[i],
        )
        if not near:
            return None, "%d circle(s), all over %.0f mm from the prior" % (len(cands), 1000 * PRIOR_MAX_M)
        cands = [cands[i] for i in near]
    cands = cands[:OBLIQUE_MAX_CIRCLES]
    whys = []
    for cu, cv, r in cands:
        hit, why = _oblique_circle(lifted, cu, cv, r, s, c0, g, h0, cam, rb, h_gate)
        if hit is None:
            whys.append(why)
            continue
        p, h_disc, h_ring, f, fill = hit
        if prior_xy is not None and math.hypot(p[0] - prior_xy[0], p[1] - prior_xy[1]) > PRIOR_MAX_M:
            whys.append("%.0f mm from the prior" % (1000 * math.hypot(p[0] - prior_xy[0], p[1] - prior_xy[1])))
            continue
        inc = _incidence_deg(p, cam)
        if inc > OBLIQUE_MAX_INCIDENCE_DEG:
            whys.append("seen %.0f deg off its normal — too oblique" % inc)
            continue
        rng = float(np.linalg.norm(p - cam))
        diameter = 2.0 * r / s * f
        uv_p = _project(p, kk, rot, cam, dd, (hh, ww))[0]
        return ButtonSighting(
            xyz=(float(p[0]), float(p[1]), h_disc),
            uv=(float(uv_p[0]), float(uv_p[1])),
            r_px=float(kk[0, 0] * diameter / 2.0 / rng),
            range_m=rng,
            diameter_m=float(diameter),
            above_lid_m=h_disc - h_ring,
            lid_z_m=h_ring,
            incidence_deg=inc,
            oblique=True,
            disc_fill=min(1.0, fill),
            table_offset_m=table_off,
            table_offset_raw_m=table_raw,
        ), None
    return None, "%d circle(s) on the rectified lid plane, none the button: %s%s" % (
        len(cands),
        "; ".join(whys[:3]),
        _table_words(table_off, table_raw),
    )


def agreeing_median(points, tol=AGREE_M, n=MIN_HITS):
    """The median of the last n points when they agree pairwise within tol
    (planar), else None."""
    if len(points) < n:
        return None
    last = np.asarray(points[-n:], float)
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(last[i, :2] - last[j, :2])) > tol:
                return None
    return tuple(float(v) for v in np.median(last, axis=0))


def agreeing_fix(points, above, n=MIN_HITS, window=AIM_WINDOW, tol=AGREE_M):
    """(xyz, indices): a fix and the hits it is made from, or None.

    n=3 (MIN_HITS): the last three agree pairwise within tol (planar), the
    fix their median — agreeing_median, the rule every straight-down aim
    keeps (any n but 2 is the last n). n=2: of the last `window` hits, the
    newest two that agree within tol and on the knob — their `above` (each
    hit's above_lid_m) within KNOB_NOISE_M, or one of the two without one —
    the fix their mean; else the newest three of them that agree pairwise,
    their median. The knob decides whether the box is open
    (press_demo.refuse_open_box at 6 mm; shut boxes read +0.4 to +3.7 mm on
    2026-10-27): two hits say it only when they say it alike, and one
    outlying reading cannot move the median of three."""
    n = int(n)
    if n != 2:
        xyz = agreeing_median(points, tol=tol, n=n)
        return None if xyz is None else (xyz, list(range(len(points) - n, len(points))))
    up = [float("nan")] * len(points) if above is None else [float(v) for v in above]
    last = range(max(0, len(points) - int(window)), len(points))

    def near(i, j):
        return math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) <= tol

    def same_knob(i, j):  # a hit without a reading disagrees with none
        return not (math.isfinite(up[i]) and math.isfinite(up[j])) or abs(up[i] - up[j]) <= KNOB_NOISE_M

    def newest_first(k):  # by the newest hit in them, then the next newest
        return sorted(itertools.combinations(last, k), key=lambda c: c[::-1], reverse=True)

    for i, j in newest_first(2):
        if near(i, j) and same_knob(i, j):
            return tuple(float(v) for v in np.mean(np.asarray([points[i], points[j]], float), axis=0)), [i, j]
    for c in newest_first(3):
        if all(near(i, j) for i, j in itertools.combinations(c, 2)):
            return tuple(float(v) for v in np.median(np.asarray([points[i] for i in c], float), axis=0)), list(c)
    return None


# lid_in_view: the lid is the surface standing the container's height above
# the table the frame sees most of, judged against THAT table so the wrist
# mount's height in TF (5 cm off on flatbot, 2026-10-23) cannot move it
LID_VIEW_BAND_M = 0.035  # the patch this close to table + dims.z
LID_VIEW_MIN_FRAC = 0.15  # of the lid's area at its depth: less is not a lid to stage over
LID_VIEW_MAX_FRAC = 2.5  # ... and more is some other raised surface


def lid_in_view(depth, k, rot, trans, model, stride=2):
    """Where the container's lid is in a wrist frame, by depth alone: the
    largest patch standing model.dims[2] above the table this frame sees
    most of, its centroid in the planning frame. For bringing the arm over
    a lid the aim could not find the button on: a lid the frame cuts off is
    placed by the part in view, biased toward the frame's middle — the aim
    from over it then sees it whole. Returns ((x, y, z), info) — info:
    frac (of the lid's area), cut (it reaches the depth image's edge),
    table_z — or (None, why)."""
    import cv2

    d = np.asarray(depth, float)[::stride, ::stride]
    h, w = d.shape
    vv, uu = np.mgrid[0:h, 0:w]
    ok = np.isfinite(d) & (d > 0.05)
    if ok.sum() < 200:
        return None, "too little depth in the wrist frame"
    K = np.asarray(k, float)
    x = (uu * stride - K[0, 2]) / K[0, 0] * d
    y = (vv * stride - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1) @ np.asarray(rot, float).T + np.asarray(trans, float)
    z = P[..., 2]
    zs = z[ok]
    hist, edges = np.histogram(zs, bins=np.arange(zs.min(), zs.max() + 0.01, 0.005))
    table_z = float(0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]))
    lid_z = table_z + float(model.dims[2])
    mask = (ok & (np.abs(z - lid_z) <= LID_VIEW_BAND_M)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None, "nothing stands the container's height above the table in view (the table at %.3f)" % table_z
    comp = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    dl = float(np.median(d[comp]))
    area = float(model.lid_dims[0]) * float(model.lid_dims[1]) * (math.pi / 4.0 if model.round else 1.0)
    frac = float(comp.sum()) / (area * (K[0, 0] / dl) * (K[1, 1] / dl) / stride**2)
    if frac < LID_VIEW_MIN_FRAC:
        return None, "only %.0f%% of a lid at lid height in view" % (100 * frac)
    if frac > LID_VIEW_MAX_FRAC:
        return None, "the surface at lid height is %.1fx the lid — not this container" % frac
    edge = ~cv2.erode(ok.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edge |= np.pad(np.zeros((h - 2, w - 2), bool), 1, constant_values=True)
    cut = bool((comp & cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).any())
    xyz = P[comp].mean(axis=0)
    return (float(xyz[0]), float(xyz[1]), float(np.median(z[comp]))), {"frac": frac, "cut": cut, "table_z": table_z}


class StillDepth:
    """The depth of the last FUSE_FRAMES still frames taken from one camera
    pose, fused per pixel (scene.median_depth: holes ignored). A glossy lid
    seen at a slant holds depth in patches that move from frame to frame
    (flatbot 2026-10-27 14:04: 9 % and 24 % of the button's disc in two
    frames of one look); together the frames cover more of it than one."""

    def __init__(self, n=FUSE_FRAMES):
        self.n = int(n)
        self.cam = None  # the pose the frames held were taken from
        self.depths = []

    def add(self, cam, depth):
        """The fused depth, this still frame's with the ones before it — only
        while the camera stands where the first of them was taken."""
        from box_opening.perception.depth_source import camera_is_still
        from box_opening.perception.scene import median_depth

        if self.cam is None or not camera_is_still(self.cam, cam):
            self.cam, self.depths = cam, []
        self.depths = (self.depths + [np.asarray(depth, np.float32)])[-self.n :]
        return self.depths[0] if len(self.depths) == 1 else median_depth(self.depths)

    def reset(self):
        self.cam, self.depths = None, []


class ButtonAimer:
    """Frames from a D405Grabber, still ones only, until enough hits agree
    (agreeing_fix). A tilted view's aim takes `min_hits` (2: two agreeing
    among the last `window` — the reach-out's look); a straight-down one,
    where most frames hit, at least MIN_HITS in a row whatever min_hits
    says. lid_z and prior_xy go to find_button (see there). A tilted view's
    frame is judged on the depth of the still frames before it too
    (StillDepth), and its fix's z raised by the wrist's table offset (fix)."""

    def __init__(self, grab, model, table_z, lid_z=None, prior_xy=None, min_hits=MIN_HITS, window=AIM_WINDOW):
        self.grab = grab
        self.model = model
        self.table_z = float(table_z)
        self.lid_z = None if lid_z is None else float(lid_z)
        self.prior_xy = None if prior_xy is None else (float(prior_xy[0]), float(prior_xy[1]))
        self.min_hits = int(min_hits)
        self.window = int(window)
        self.t0 = time.monotonic()
        self.last_why = None
        self.frames = 0
        self.hits = 0
        self.points = []
        self.above_lid = []
        self.table_offsets = []  # each hit's (accepted, raw) table offset, nan where none (ButtonSighting)
        self.hit_s = []  # each hit's time since the aimer was built (s)
        self.oblique = []  # each hit: found by find_button_oblique
        self._last_stamp = None
        self._last_depth_stamp = None  # the depth that last voted (depth_paired)
        self._last_cam = None
        self.last_frame = None  # (the snapshot it judged last, its camera pose or None): perception/viz draws it
        self.last_sighting = None
        self.pair_dt_ms = []  # each hit's depth-minus-colour stamp (ms), when known
        self.still_depth = StillDepth()

    def tick(self):
        """One frame if there is a new one: True when a fix is ready."""
        from box_opening.perception.d405 import frame_snapshot
        from box_opening.perception.depth_source import (
            camera_is_still,
            camera_pose_at,
            depth_key,
            depth_paired,
            pair_dt_ms,
        )

        g = self.grab
        # one read of the frame: the dedup key, the pose's stamp and the
        # pixels judged all belong to it even if a newer one lands meanwhile
        f = frame_snapshot(g)
        if f.color is None or f.depth is None or f.k is None or f.color_stamp is None:
            self.last_why = "no wrist frame yet"
            return False
        stamp = (f.color_stamp.sec, f.color_stamp.nanosec)
        if stamp == self._last_stamp:
            return False
        paired, why = depth_paired(f, self._last_depth_stamp)
        if not paired:
            # not consumed: its depth may not have landed yet, and the next
            # tick tries this colour frame again; a stalled depth never aims
            self.last_why = why
            return False
        self._last_stamp = stamp
        self._last_depth_stamp = depth_key(f)
        self.frames += 1
        cam = camera_pose_at(g, stamp=f.color_stamp)
        self.last_frame = (f, cam)  # references: nothing is copied
        if cam is None:
            self.last_why = "no camera pose for the frame"
            return False
        prev = self._last_cam
        if prev is None:
            # the first frame has no frame before it: the camera's pose
            # FIRST_LOOKBACK_S before its stamp stands in for one (TF keeps
            # the history) — the same still test, over at least one frame
            # interval, so a parked arm's first frame votes
            off = float(getattr(g, "stamp_offset_s", 0.0) or 0.0)
            prev = camera_pose_at(g, offset_s=off - FIRST_LOOKBACK_S, stamp=f.color_stamp)
        still = camera_is_still(prev, cam)
        self._last_cam = cam
        if not still:
            self.still_depth.reset()
            self.last_why = "camera moving"
            return False
        rot, trans = cam
        depth = self.still_depth.add(cam, f.depth) if view_tilt_deg(rot) > OBLIQUE_MIN_DEG else f.depth
        s, why = find_button(
            f.color,
            depth,
            f.k,
            rot,
            trans,
            self.table_z,
            self.model,
            dist=f.dist,
            lid_z=self.lid_z,
            prior_xy=self.prior_xy,
        )
        if s is None:
            self.last_why = why
            return False
        self.hits += 1
        self.last_sighting = s
        self.points.append(s.xyz)
        self.above_lid.append(s.above_lid_m)
        self.table_offsets.append((s.table_offset_m, s.table_offset_raw_m))
        self.hit_s.append(time.monotonic() - self.t0)
        self.oblique.append(bool(getattr(s, "oblique", False)))  # a sighting without the flag: straight down
        dt = pair_dt_ms(f)
        if dt is not None:
            self.pair_dt_ms.append(dt)
        return self.fix() is not None

    def _need(self):
        """How many agreeing hits make the fix (agreeing_fix's n): min_hits
        while the newest hit is a tilted view's, else at least MIN_HITS."""
        if self.oblique and self.oblique[-1]:
            return self.min_hits
        return max(self.min_hits, MIN_HITS)

    def fix_hits(self):
        """The hits the fix is made from (indices into points, above_lid,
        table_offsets and hit_s), or None without a fix."""
        got = agreeing_fix(self.points, self.above_lid, self._need(), self.window)
        return None if got is None else got[1]

    def _judged(self):
        """The fix's own hits — a pair need not be the last two — or,
        without a fix, the last _need() of them."""
        idx = self.fix_hits()
        return idx if idx is not None else list(range(max(0, len(self.points) - self._need()), len(self.points)))

    def button_above_lid_m(self):
        """How far the button stands above its own lid, over the hits the
        fix was made from (median), or None when none of them could say."""
        vals = [self.above_lid[i] for i in self._judged() if np.isfinite(self.above_lid[i])]
        return float(np.median(vals)) if vals else None

    def table_offset_m(self, raw=False):
        """How far the wrist reads its own table from the one it was given
        (m), over the hits the fix was made from: the median of their
        accepted offsets when at least two agree within TABLE_AGREE_M (the
        most that do), else nan — a hit whose offset was not accepted counts
        for nothing. raw: the median of their raw offsets, accepted or not,
        for the aim's note: drift shows there before it fails a run."""
        pairs = [self.table_offsets[i] for i in self._judged()]
        if raw:
            vals = [v for _a, v in pairs if np.isfinite(v)]
            return float(np.median(vals)) if vals else float("nan")
        vals = sorted(a for a, _v in pairs if np.isfinite(a))
        best = []
        for i, lo in enumerate(vals):
            near = [v for v in vals[i:] if v - lo <= TABLE_AGREE_M]
            if len(near) > len(best):
                best = near
        return float(np.median(best)) if len(best) >= 2 else float("nan")

    def fix(self, measured=False):
        """The press point when enough hits agree (agreeing_fix: the median
        of the last MIN_HITS; a tilted view's, the mean of its newest
        agreeing pair or the median of three of its last `window`), else
        None. Its xy is as measured; its z is RAISED by the wrist's
        table offset where that is known and below the given table
        (table_offset_m, over the same hits): the wrist that reads its table
        20 mm low reads the lid 20 mm low too. It is never lowered: the
        compliant press goes by the higher of this and the scene camera's
        lid (reach_press), and a lowered aim would no longer bound the press
        from below should the scene camera's table and lid ever read low.
        measured: the z as the hits measured it — what the wrist's own later
        readings are judged against (reach_press._aim)."""
        got = agreeing_fix(self.points, self.above_lid, self._need(), self.window)
        if got is None or measured:
            return None if got is None else got[0]
        xyz = got[0]
        off = self.table_offset_m()
        if np.isfinite(off) and off < 0.0:
            return (xyz[0], xyz[1], xyz[2] - off)
        return xyz

    def status(self):
        idx = self.fix_hits()
        return "%d of %d hits agree in %d frames%s" % (
            0 if idx is None else len(idx),
            self.hits,
            self.frames,
            "" if self.last_why is None else " (last reject: %s)" % self.last_why,
        )
