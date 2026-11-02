"""Finding the box from the scene camera, before the arm moves.

The fixed scene camera sees the whole workspace from a metre away. That is
the wrong camera for AIMING a press — a lid is 30 pixels wide from there —
and exactly the right one for the question the mission used to answer by
flying the wrist camera around: where, roughly, is the box? A coarse pose
from here sends the arm straight to staging above the box; the wrist camera
then takes its precise fix from 12 cm, where it is good.

Semantics first, geometry second. The scene is not the same every time and
carries other containers, plates and cables, so a box-sized plateau alone
is not enough: a detector says WHICH thing is the box (a pixel box), and
the depth points inside that box that sit at lid height above the table
give its position. The detector is sheppy's always-on scene segmenter
(perception/segmenter_source: RF-DETR at 5 Hz, its holder box turned into
the band where the lid stands) where the container config names its
classes, with the OWL node's scene instance as the fallback; a fix
commits when two sightings agree (commit_fix). No smoothness or footprint
gates — the scene depth at a metre is noisier than those gates were tuned
for, and the wrist fix that follows carries the precision.

A camera level with the lid never sees a top face (flatbot's, on a table
that high — tables change, so which view holds is not assumed): when the
top-face lift finds nothing, the nearest object in the pixel box is checked
against the table around it and placed half its width behind the face the
camera sees (box_from_side_points). With no survey, the table the box was
found on is measured for the planner too (table_extent).
"""

import time
from dataclasses import dataclass, replace

import numpy as np

from box_opening import LOGGER as log
from box_opening.perception.clear_table import MAP_R_M
from box_opening.perception.d405 import stable_fix
from box_opening.perception.depth_source import container_pose_from_top
from box_opening.perception.planes import fit_plane

LID_BAND_M = 0.035  # the top must sit this close to table + dims.z
TOP_SLAB_M = 0.015  # the lid top: points within this of the highest surface
TOP_PCTL = 97  # "highest" as a percentile, so flying pixels do not define it
MIN_LID_POINTS = 40
YAW_MIN_POINTS = 150  # fewer than this and the min-area rect's yaw is noise
# the top slab must be a FACE, not a rim: a taller container's near face
# crosses the lid band as a thin strip, and a strip must not read as a box
MIN_FOOTPRINT_FRAC = 0.4  # of the model's shorter side, as a 10-90 % spread
# ... and not much LARGER than the model either: a black 19 cm tray next to
# the 10 cm OXO passed every other gate and the OWL's best box flipped
# between the two (2026-09-17). The longer spread must fit the longer side.
MAX_FOOTPRINT_FRAC = 1.4


@dataclass(frozen=True)
class SceneFix:
    pose: object  # ContainerPose (bottom-centre origin, yaw)
    top_xyz: tuple  # the lid top's centre in base_link
    n_points: int
    score: float  # the detector score behind it (the newest sighting's)
    table_z_scene: float  # the table height the scene depth itself shows, or nan
    table_extent: object = None  # (x0, x1, y0, y1) where it shows that table, when no survey says
    partial: str = ""  # how the frame cut the box when it was finished from the part in view, else ""
    # the scene's points around the box (float32, base frame): what stands on
    # the table there, for where the lid may go (perception/clear_table)
    table_points: object = None
    source: str = "owl"  # the newest sighting's detector: "segmenter" | "owl"
    n_sightings: int = 1  # the agreeing sightings behind it (commit_fix)
    sources: tuple = ()  # ... and each one's detector, oldest first


def robust_yaw(xy):
    """The square's orientation, mod 90 deg, as the angle whose axis-aligned
    3rd-97th percentile box is tightest. A min-area rectangle is set by the
    outermost points, so one stray sets it; a 3 % trim is not. The trim
    must stay small, though: a square's projection barely changes width
    between 10 % and 90 %, so a 10-90 box hardly knows the angle at all."""
    xy = np.asarray(xy, float)

    def best_of(th):
        # every angle's rotation in one stacked matmul and its trimmed box
        # in one percentile call (a Python loop of these was 16 ms)
        c, s = np.cos(-th), np.sin(-th)
        rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        lo, hi = np.percentile(xy @ rot.transpose(0, 2, 1), [3, 97], axis=1)
        return th[int(np.argmin(np.prod(hi - lo, axis=-1)))]

    best = best_of(np.radians(np.arange(0.0, 90.0, 1.0)))
    best = best_of(best + np.radians(np.arange(-1.0, 1.0, 0.1)))
    return float(best % (np.pi / 2))


def box_from_scene_points(pts_base, uv, bbox, table_z, model):
    """Coarse container pose from scene depth points inside a pixel box.

    pts_base: (N, 3) depth points in base_link; uv: their colour pixels
    (N, 2); bbox: (x0, y0, x1, y1) in colour pixels. Returns
    (ContainerPose, top_xyz, n_points) or (None, why, 0)."""
    x0, y0, x1, y1 = bbox
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    if not inside.any():
        return None, "no depth inside the box", 0
    lid_z = float(table_z) + float(model.dims[2])
    z = pts_base[:, 2]
    band = inside & (np.abs(z - lid_z) < LID_BAND_M)
    if band.sum() < MIN_LID_POINTS:
        return (
            None,
            "only %d depth points at lid height inside the box (need %d)" % (int(band.sum()), MIN_LID_POINTS),
            int(band.sum()),
        )
    # The TOP face is the highest surface in the pixel box. From the scene
    # camera's oblique view the box's near face is in the box too, and its
    # upper part sits inside the lid band: taking every band point dragged
    # the centre 5 cm toward the camera (live, 2026-09-16). Keep only the
    # slab just under the highest points.
    top_z = float(np.percentile(z[band], TOP_PCTL))
    lid = band & (z > top_z - TOP_SLAB_M)
    n = int(lid.sum())
    if n < MIN_LID_POINTS:
        return None, "only %d depth points on the top face (need %d)" % (n, MIN_LID_POINTS), n
    top = pts_base[lid]
    centre = np.median(top, axis=0)
    # Flying pixels at lid height land inside the box too, and one stray a
    # hand-width away is enough to stretch a min-area rectangle across the
    # gate. So the footprint is measured as the 10th-90th percentile spread
    # along the slab's principal axes, which a stray cannot move, and the
    # rectangle that gives the yaw is fitted to the core between the 5th
    # and 95th percentiles only.
    xy = top[:, :2] - centre[:2]
    _w, vecs = np.linalg.eigh(np.cov(xy.T))
    proj = xy @ vecs  # columns: minor, major axis
    lo10, hi90 = np.percentile(proj, [10, 90], axis=0)
    spread = hi90 - lo10
    need = MIN_FOOTPRINT_FRAC * min(float(model.dims[0]), float(model.dims[1]))
    if spread.min() < need:
        return (
            None,
            (
                "the top slab is a strip %.0fx%.0f mm across, not a face (need %.0f mm) — "
                "something taller crossing lid height?" % (1000 * spread[0], 1000 * spread[1], 1000 * need)
            ),
            n,
        )
    too_big = MAX_FOOTPRINT_FRAC * max(float(model.dims[0]), float(model.dims[1]))
    if spread.max() > too_big:
        return (
            None,
            (
                "the top is %.0fx%.0f mm, the container %.0fx%.0f — another container"
                % (1000 * spread[0], 1000 * spread[1], 1000 * model.dims[0], 1000 * model.dims[1])
            ),
            n,
        )
    yaw = robust_yaw(xy) if n >= YAW_MIN_POINTS else 0.0
    pose = container_pose_from_top((centre[0], centre[1], centre[2]), yaw, model, table_z)
    return pose, (float(centre[0]), float(centre[1]), float(centre[2])), n


# A camera level with the lid sees the box's SIDE, never its top face (the
# flatbot scene OAK on a table this high; a lower table shows it the top).
SIDE_DEPTH_SLACK_M = 0.03  # the nearest object: this far past its own width behind its front
ABOVE_TABLE_M = 0.015  # an object point sits at least this far above the table
TABLE_RING_M = (0.08, 0.30)  # the table is read this far (horizontally) from the box's centre
TABLE_SLAB_M = 0.015  # ... from the points within this of the ring's commonest height
MIN_TABLE_POINTS = 300
TABLE_EXTENT_R_M = 0.6  # the table's extent is read this far around the box


def table_extent(pts_base, table_z, centre_xy, radius=TABLE_EXTENT_R_M):
    """(x0, x1, y0, y1): where the scene depth shows the table — the points
    within TABLE_SLAB_M of its height, within `radius` of the box — as their
    2nd-98th percentile box, so stray points do not set it. None when too
    few points say. Only what the camera sees: the table behind the box, or
    out of view, is not in it."""
    d = np.hypot(pts_base[:, 0] - centre_xy[0], pts_base[:, 1] - centre_xy[1])
    on = pts_base[(d < radius) & (np.abs(pts_base[:, 2] - table_z) < TABLE_SLAB_M)]
    if len(on) < MIN_TABLE_POINTS:
        return None
    lo, hi = np.percentile(on[:, :2], 2, axis=0), np.percentile(on[:, :2], 98, axis=0)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def seen_table_extent(pts_base, table_z, centre_xy):
    """(extent, assumed): table_extent, or, when too few points say, the
    ring square table_around measured the height from (assumed True), so a
    measured height never reaches the planner as no table at all."""
    ext = table_extent(pts_base, table_z, centre_xy)
    if ext is not None:
        return ext, False
    r = TABLE_RING_M[1]
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    return (cx - r, cx + r, cy - r, cy + r), True


def table_around(pts_base, centre_xy, below_z, ring=TABLE_RING_M):
    """The table's height under `centre_xy`: the commonest height (1 cm
    bins) in a ring around it below `below_z`, refined by a plane through
    the points near that height and read at the centre. A camera level with
    the table also sees under it (flatbot 2026-10-22: streaks down to the
    floor through the gap below its front edge); those spread over every
    height while the table piles into one. nan when the ring holds too
    little."""
    d = np.hypot(pts_base[:, 0] - centre_xy[0], pts_base[:, 1] - centre_xy[1])
    ringed = pts_base[(d > ring[0]) & (d < ring[1]) & (pts_base[:, 2] < below_z)]
    if len(ringed) < MIN_TABLE_POINTS:
        return float("nan")
    hist, edges = np.histogram(ringed[:, 2], bins=np.arange(ringed[:, 2].min(), below_z + 0.01, 0.01))
    k = int(np.argmax(hist))
    slab = ringed[np.abs(ringed[:, 2] - 0.5 * (edges[k] + edges[k + 1])) < TABLE_SLAB_M]
    if len(slab) < MIN_TABLE_POINTS:
        return float("nan")
    fit = fit_plane(slab, tol_m=0.008, iters=150, max_points=3000)
    if fit is None or abs(fit[0][2]) < 0.9:
        return float(np.median(slab[:, 2]))
    n, c, _frac = fit
    return float(c[2] - (n[0] * (centre_xy[0] - c[0]) + n[1] * (centre_xy[1] - c[1])) / n[2])


def box_from_side_points(
    pts_base, uv, bbox, table_z, model, cam_xyz, partial=False, anchor_rng=None, sides=(), side_is_lid=True
):
    """Coarse container pose from a camera level with the lid. It sees the
    box's side, never its top face, and the wall behind a round lid falls
    inside the pixel box too (flatbot 2026-10-22: 22 of 252 points, at lid
    height, stretched the top-face lift's "top" to 621 mm). So: the nearest
    object in the box, its top checked against the table around it
    (`table_z` when surveyed), its centre half its width behind the face the
    camera sees. Returns (ContainerPose, top_xyz, n_points, table_z) or
    (None, why, n_points, table_z or nan).

    `partial`: the frame cuts the box off at its side (a round container
    only, scene_fix_from_boxes): what is in view is not centred on it and
    is narrower than it, so the centre is the lid's circle through the arc
    of its side in view (lid_circle) instead, and no width is asked;
    `sides`, the planes of the box's sides the frame does not cut
    (uncut_sides), which that circle must stay inside, and `side_is_lid`,
    whether they are the lid's own (lid_circle). A cut box the lid stands
    over (PARTIAL_OVER_M, inside the box's pixel columns) holds only the
    holder under it, and is refused; so is a circle that something taller
    than the lid stands inside (PARTIAL_OVER_PAD_M): its arc took in that
    thing's side.

    `anchor_rng`: `bbox` is the band above a holder box whose front stands
    this far from the camera (holder_front_rng). A lid standing on that
    holder is within a lid's reach of it; when the nearest thing in the band
    stands further behind, the band looks past the holder at whatever stands
    behind it, and is refused (detbench: the plain band lifted 320 of 624
    phantom bowls, this 35). Only ever a refusal: an accepted lift is the
    plain one to the millimetre — the slab cut at the holder's front instead
    moved good fixes 29-52 mm, where something nearer than the canister
    lies inside its holder box (flatbot 2026-10-22 14:43)."""
    x0, y0, x1, y1 = bbox
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    if inside.sum() < MIN_LID_POINTS:
        return (
            None,
            "only %d depth points inside the box (need %d)" % (int(inside.sum()), MIN_LID_POINTS),
            int(inside.sum()),
            float("nan"),
        )
    cam = np.asarray(cam_xyz, float)
    q = pts_base[inside]
    rng = np.linalg.norm(q - cam, axis=1)
    half = 0.5 * min(float(model.dims[0]), float(model.dims[1]))
    p5 = np.percentile(rng, 5)
    if anchor_rng is not None and not partial:
        gap = float(p5) - float(anchor_rng)
        if gap > 2 * half + SIDE_DEPTH_SLACK_M:
            return (
                None,
                (
                    "the nearest thing above the holder stands %.0f mm behind the holder's front: the band "
                    "looks past it, no lid stands on it" % (1000 * gap)
                ),
                int(inside.sum()),
                float("nan"),
            )
    near = q[rng <= p5 + 2 * half + SIDE_DEPTH_SLACK_M]
    upper = near[near[:, 2] >= np.median(near[:, 2])]
    along = upper[:, :2].mean(axis=0) - cam[:2]  # the view across the box, horizontally
    along /= np.linalg.norm(along)
    across = np.array([-along[1], along[0]])

    def centre_of(p):
        rel = p[:, :2] - cam[:2]
        return cam[:2] + (np.percentile(rel @ along, 5) + half) * along + np.median(rel @ across) * across

    if table_z is None:
        table_z = table_around(pts_base, centre_of(upper), float(np.percentile(near[:, 2], TOP_PCTL)) - 0.03)
        if np.isnan(table_z):
            return None, "no table around it", len(near), table_z
    obj = near[near[:, 2] > table_z + ABOVE_TABLE_M]
    n = len(obj)
    if n < MIN_LID_POINTS:
        return None, "only %d depth points above the table (need %d)" % (n, MIN_LID_POINTS), n, table_z
    top_z = float(np.percentile(obj[:, 2], TOP_PCTL))
    lid_z = float(table_z) + float(model.dims[2])
    if abs(top_z - lid_z) > LID_BAND_M:
        return (
            None,
            "its top is %+.0f mm from lid height (the table at %.3f + %.0f mm)"
            % (1000 * (top_z - lid_z), table_z, 1000 * model.dims[2]),
            n,
            table_z,
        )
    if partial:
        # the lid's points from one lid diameter around what the box shows, at its height
        d = float(model.lid_dims[0])
        seed = near[:, :2].mean(axis=0)
        reach = np.linalg.norm(pts_base - cam, axis=1) <= np.percentile(rng, 5) + 2 * half + SIDE_DEPTH_SLACK_M
        around = reach & (np.hypot(pts_base[:, 0] - seed[0], pts_base[:, 1] - seed[1]) <= d + PARTIAL_REACH_M)
        above = (pts_base[:, 2] > top_z + PARTIAL_OVER_M[0]) & (pts_base[:, 2] <= top_z + PARTIAL_OVER_M[1])
        cols = (uv[:, 0] >= x0) & (uv[:, 0] <= x1)  # the lid stands on its holder, inside the box's columns
        over = int((around & cols & above).sum())
        if over >= MIN_LID_POINTS:
            return (
                None,
                (
                    "%d depth points stand %.0f-%.0f mm over its top around it: the box holds what is under the lid, "
                    "not the lid" % (over, 1000 * PARTIAL_OVER_M[0], 1000 * PARTIAL_OVER_M[1])
                ),
                n,
                table_z,
            )
        band = pts_base[
            around & (pts_base[:, 2] > top_z - float(model.lid_dims[2])) & (pts_base[:, 2] <= top_z + 0.005)
        ]
        centre, why = lid_circle(band, model, cam, sides=sides, side_is_lid=side_is_lid)
        if centre is None:
            return None, why, n, table_z
        under = reach & (
            np.hypot(pts_base[:, 0] - centre[0], pts_base[:, 1] - centre[1]) <= 0.5 * d + PARTIAL_OVER_PAD_M
        )
        stand = int((under & above).sum())
        if stand >= MIN_LID_POINTS:
            return (
                None,
                (
                    "%d depth points stand %.0f-%.0f mm over its top where the lid's circle is: the arc took in "
                    "something taller than the lid" % (stand, 1000 * PARTIAL_OVER_M[0], 1000 * PARTIAL_OVER_M[1])
                ),
                n,
                table_z,
            )
        pose = container_pose_from_top((centre[0], centre[1], top_z), 0.0, model, table_z)  # round: yaw unused
        return pose, (centre[0], centre[1], top_z), n, float(table_z)
    lo, hi = np.percentile((obj[:, :2] - cam[:2]) @ across, [10, 90])
    width = float(hi - lo)
    need = MIN_FOOTPRINT_FRAC * min(float(model.dims[0]), float(model.dims[1]))
    if width < need:
        return None, "it is %.0f mm across, not a container (need %.0f mm)" % (1000 * width, 1000 * need), n, table_z
    too_big = MAX_FOOTPRINT_FRAC * max(float(model.dims[0]), float(model.dims[1]))
    if width > too_big:
        return (
            None,
            "it is %.0f mm across, the container %.0f — another container"
            % (1000 * width, 1000 * max(model.dims[0], model.dims[1])),
            n,
            table_z,
        )
    cx, cy = centre_of(obj)
    yaw = robust_yaw(obj[:, :2] - (cx, cy)) if n >= YAW_MIN_POINTS else 0.0
    pose = container_pose_from_top((cx, cy, top_z), yaw, model, table_z)
    return pose, (float(cx), float(cy), top_z), n, float(table_z)


def holder_front_rng(pts_base, uv, holder, cam_xyz):
    """How far the front of a holder box stands from the camera: the 5th
    percentile of the ranges of the depth points inside its pixel box, or
    None when fewer than MIN_LID_POINTS are."""
    x0, y0, x1, y1 = (float(v) for v in holder[:4])
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    if inside.sum() < MIN_LID_POINTS:
        return None
    return float(np.percentile(np.linalg.norm(pts_base[inside] - np.asarray(cam_xyz, float), axis=1), 5))


@dataclass(frozen=True)
class SceneBoxFix:
    pose: object  # ContainerPose, or None when no box was this container
    top: tuple  # the lid top's centre (base) — None without a pose
    n: int  # lid points behind it
    box: object  # the candidate box that won ([x0, y0, x1, y1]), or None
    tried: tuple  # why each refused candidate was refused, in order
    table_z: float = float("nan")  # the table the winner was measured against
    lift: str = ""  # "top" or "side": which view of it won
    clipped: tuple = ()  # (score, box, edges) of each candidate refused at the frame's edge
    partial: tuple = ()  # the edges the frame cut the winner at, when it was finished from part of it
    # with rivals=True: the tops of the OTHER candidates that pass on their
    # own more than RIVAL_M from the winner's — two places, not one box
    rivals: tuple = ()
    tried_boxes: tuple = ()  # the box (the candidate's own object) each of `tried` refused: what perception/viz reads


# An OWL box this close to the frame's edge has its container running off
# the frame: the camera sees a part of it, and a lift of a part puts the box
# off by up to half of it (flatbot 2026-10-23: the canister cut by the scene
# image's edge was fixed 7 cm short of where the wrist camera then found it,
# and the staging aimed past its button)
FRAME_EDGE_PX = 3


# ... unless the container is round and the frame cuts it at its side: then
# the arc of the lid's side band in view is a circle of the lid's known
# diameter, fitted with that radius (lid_circle). Flatbot 2026-10-23: the
# canister half off the frame put 147 deg of its lid in view and the fit
# put it 3 cm from where the wrist camera found it, the old lift 7 cm.
PARTIAL_INLIER_M = 0.006  # a lid point this close to the circle is on it (the side OAK's noise at 0.3 m)
PARTIAL_MIN_ARC_DEG = 60.0  # the points on it spread over at least this much of it
# A tight OWL box on a cut lid holds a short arc, and a circle of known
# radius slides along a short arc: live 2026-10-23 the same canister fixed
# at [0.497, -0.285] in most frames and [0.445, -0.235] in the rest, where
# it stood. So the lid's points are taken from one lid diameter around what
# the box shows (PARTIAL_REACH_M), the circle is found by RANSAC, and a
# rival circle this far from the best that holds nearly as many points is
# no fix at all.
PARTIAL_REACH_M = 0.02  # beyond one lid diameter from the box's own near points
PARTIAL_RIVAL_M = 0.035
PARTIAL_RIVAL_FRAC = 0.85
PARTIAL_RANSAC_ITERS = 400
# Not every two places is a slide. Half out of the frame, a lid can show its
# depth only as a strip along its own silhouette ray (the rest of its side is
# past the frame's edge, or in the strip beside it where the OAK gives no
# depth), and the circle tangent to that strip from its other side holds the
# same points on its hidden back: the lid's mirror across its silhouette,
# ~100 mm away. On detbench's P4 frames (flatbot 2026-10-23, the canister half
# off the frame; found 2026-10-27) the mirror holds 62-99 % as many of the
# band's points as the lid, and more than the lid in most calls at 17:27,
# where only the rival rule stood between it and a fix 103 mm off. The box's
# side the frame does not cut is the lid's silhouette, or the holder's around
# it: the lid stands inside it, the mirror past it. So a circle that runs past
# that side by more than this is not the box's lid. Over the 2,496 cut-box
# calls on detbench's P2 and P4 frames (6 box sources, 4 pixel parities, 4 box
# draws), the circles near the lid that could win (0.85 of the best's points)
# run up to 22.7 mm past a lid-tight box (the depth smears past the lid's edge
# and pulls them; in every call one runs 9.8 mm past or less) and stay 4.5+ mm
# inside a holder's; the mirror runs 30.8+ mm past a holder's, 73+ mm past a
# lid's.
PARTIAL_BOX_SLACK_M = 0.020
# The bound is not only a refusal: where the circle it takes out was the rival
# of the one it keeps, a "two places" becomes a fix. That is sound only on a
# lid's worth of arc. With the lid all but out of view, or its sliver in the
# strip where the OAK gives no depth, the band is the holder's brim tip or the
# smear behind it, and every circle through it is a guess: the one the box's
# side keeps is no likelier the lid than the one it takes out. Detbench
# 2026-10-23 13:32 (the canister ~90 % off the frame, a frame the acceptance
# skips as difficult): the kept circle held 45 points or fewer and stood 51 mm
# past the lid's own silhouette, and the stream committed it in 300 of 300 runs
# where c693865 refused it as two places. So a circle the bound settles two
# places for must hold this many points. Over detbench's P2-P4 frames (768
# gate calls: 4 parities, 4 box draws, segmenter and OWL) the circles it
# settled, all on P4, held 127-167 and were fixed 7-21 mm from the reference;
# the wrong ones it settled on re-cut frames (the lid cropped 75-80 % out of
# view with the no-depth strip, or the frame's edge moved 80 px in) held 45-84
# (found 2026-10-28, review of b2fb875).
PARTIAL_SETTLE_MIN_POINTS = 100
# A cut box that holds only the holder under the lid (a "blue bowl" box as it
# is, an OWL "a pink lid" that boxed the brim) lifts its top at the brim,
# 13-34 mm under the lid's and inside LID_BAND_M, and the lid's circle fitted
# to the brim put the box 43-47 mm off on detbench P4 once the mirror was no
# second place (the rival rule had refused them by chance). The lid then
# stands over the band: MIN_LID_POINTS this far over the lifted top, inside
# the band's own reach and disc and the box's own pixel columns, refuse the
# box. Over the 429 cut boxes of every detbench source on P2 and P4 that reach
# the circle: a lid's band has at most 12 points there (the two frames whose
# button stood up included), a brim 18+ mm under the lid 81-300 (the 2
# shallower brims that pass fix 16 and 20 mm off). The lid stands ON the
# holder, inside the box's columns; counted over the whole disc, a taller
# thing beside the box refused a good cut box too (a bottle 80 mm taller 18 cm
# beside the half-off canister at 17:27; at 14:40 a lid-wide one 27 cm off,
# where c693865 fixed it 11.6 mm from the reference). Something taller
# standing right behind the box, inside its columns, still refuses it.
PARTIAL_OVER_M = (0.020, 0.050)
# ... and nothing stands that far over the top where the fitted lid is (within
# its radius and this): the lid is the container's top. When the band took in
# the side of something taller beside or behind the canister, the circle slid
# toward it, and that thing stands inside the circle. With the count above
# kept to the box's columns, review 1's taller cylinders 1-5 cm beside or
# behind the cut canister (detbench P2/P3/P4, ray-cast into the depth) pulled
# 17 cut fixes 31-58 mm off that b2fb875 had refused (14 by its whole-disc
# count); this refuses 14 of them, and 17 fixes within 30 mm. Over all 1,104
# of those gate calls: offs 48 -> 33 (b2fb875 35, c693865 398), ok 722 -> 710
# (b2fb875 513); over review 0's 3,782 with a second object: offs 146 -> 143
# (b2fb875 152), ok 3017 -> 3004 (b2fb875 2431). Nothing changes on detbench's
# own frames.
PARTIAL_OVER_PAD_M = 0.010
# The segmenter candidates (segmenter_source.SceneCand.kind) whose box is its
# holder's: a cut holder grown over its lid band, and the band above an uncut
# one. Their sides stand outside the lid's silhouette; every other box (a lid
# RF-DETR boxed on its holder, a container boxed as it is, an OWL lid box) has
# the lid's own (lid_circle's side_is_lid).
HOLDER_SIDE_KINDS = ("grown", "band")


def pixel_plane(axis, at, pts_base, uv, cam_xyz, R_base_color):
    """The plane through the camera of the colour pixels whose coordinate
    `axis` (0: u, 1: v) is `at`, as its base-frame normal n: n . (p - cam) > 0
    for a point p that projects beyond it (a larger u or v). The image's scale
    and centre along that axis are read off the cloud itself — its pixels `uv`
    are its projection into the colour image, u = f x / z + c in the colour
    optical frame — so no intrinsics are passed in. None when the cloud holds
    too little."""
    R = np.asarray(R_base_color, float)
    step = max(1, len(pts_base) // 2000)
    pc = (np.asarray(pts_base[::step], float) - np.asarray(cam_xyz, float)) @ R  # the colour optical frame
    px = np.asarray(uv[::step], float)[:, axis]
    front = pc[:, 2] > 1e-6
    if front.sum() < MIN_LID_POINTS:
        return None
    f, c = np.polyfit(pc[front, axis] / pc[front, 2], px[front], 1)
    m = np.zeros(3)
    m[axis], m[2] = f, c - float(at)
    return R @ m


# the side of a pixel box across from each frame edge: (that side, its axis, its coordinate in
# the box, +1 when past it is +u/+v)
_ACROSS = {
    "left": ("right", 0, 2, 1.0),
    "right": ("left", 0, 0, -1.0),
    "top": ("bottom", 1, 3, 1.0),
    "bottom": ("top", 1, 1, -1.0),
}


def uncut_sides(box, edges, pts_base, uv, cam_xyz, R_base_color):
    """The sides of pixel box [x0, y0, x1, y1] across from the frame `edges`
    that cut it, as planes through the camera (pixel_plane): each a base-frame
    normal n with n . (p - cam) > 0 for a point past the box. A box the frame
    cuts at both its sides along an axis has none on it."""
    sides = []
    for e in edges:
        other, axis, k, sign = _ACROSS[e]
        if other in edges:
            continue
        n = pixel_plane(axis, float(box[k]), pts_base, uv, cam_xyz, R_base_color)
        if n is not None:
            sides.append(sign * n)
    return tuple(sides)


def past_sides(centres, sides, cam_xyz, z, r):
    """How far (m, horizontally, at height `z`) each circle of radius `r`
    centred at `centres` (M x 2, base frame) runs past the furthest of the
    planes `sides` (uncut_sides); negative when it stays inside them all."""
    cam = np.asarray(cam_xyz, float)
    c = np.atleast_2d(np.asarray(centres, float))
    out = np.full(len(c), -np.inf)
    for n in sides:
        n = np.asarray(n, float)
        h = max(float(np.hypot(n[0], n[1])), 1e-9)
        out = np.maximum(out, ((c - cam[:2]) @ n[:2] + n[2] * (z - cam[2])) / h + r)
    return out


def lid_circle(band, model, cam_xyz, sides=(), side_is_lid=True):
    """The centre (x, y) of a round lid from the points of its side band a
    level camera sees (`band`, base frame): the circle of the lid's radius
    through the most of them (RANSAC over point pairs, the centre behind the
    pair as the camera sees it), refined on those. Returns ((cx, cy), None)
    or (None, why).

    `sides`: the planes of its box's sides the frame does not cut
    (uncut_sides). A circle that runs past them by more than
    PARTIAL_BOX_SLACK_M is no candidate, before the best and its rival are
    picked. That is not only a refusal: where the circle it takes out was
    the rival of the best one inside (the lid's mirror, detbench P4), two
    places become one fix, and then only on PARTIAL_SETTLE_MIN_POINTS.
    `side_is_lid`: those sides are the lid's own silhouette (the box is the
    lid, or the container itself), not a holder's around it. Refined past a
    lid's side, the circle was pulled there by the smear past the silhouette,
    and the RANSAC circle it was refined from, inside the bound, is the fix;
    refined past a holder's side, it is refused."""
    r = 0.5 * float(model.lid_dims[0])
    if len(band) < MIN_LID_POINTS:
        return None, "only %d points on the lid's side in view (need %d)" % (len(band), MIN_LID_POINTS)
    P = np.asarray(band, float)[:, :2]
    cam = np.asarray(cam_xyz, float)[:2]
    rng = np.random.default_rng(0)  # the same frame, the same fix
    cands = []
    for _ in range(PARTIAL_RANSAC_ITERS):
        i, j = rng.choice(len(P), 2, replace=False)
        m, chord = 0.5 * (P[i] + P[j]), P[j] - P[i]
        dist = float(np.linalg.norm(chord))
        if dist < 0.01 or dist >= 2 * r:
            continue
        normal = np.array([-chord[1], chord[0]]) / dist
        h = float(np.sqrt(r * r - 0.25 * dist * dist))
        for c in (m + h * normal, m - h * normal):
            if np.dot(c - m, m - cam) > 0.0:  # behind the pair, from the camera
                on = np.abs(np.linalg.norm(P - c, axis=1) - r) < PARTIAL_INLIER_M
                cands.append((int(on.sum()), c))
    if not cands:
        return None, "no %.0f mm circle through the lid's side in view" % (2000 * r)
    beyond = []  # the circles the bound took out
    if len(sides):
        z = float(np.median(np.asarray(band, float)[:, 2]))
        past = past_sides([cc for _k, cc in cands], sides, cam_xyz, z, r)
        beyond = [kc for kc, p in zip(cands, past, strict=False) if p > PARTIAL_BOX_SLACK_M]
        cands = [kc for kc, p in zip(cands, past, strict=False) if p <= PARTIAL_BOX_SLACK_M]
        if not cands:
            return None, "every %.0f mm circle through the lid's side in view runs %.0f+ mm past its box" % (
                2000 * r,
                1000 * PARTIAL_BOX_SLACK_M,
            )
    cands.sort(key=lambda kc: -kc[0])
    best_n, c = cands[0][0], cands[0][1].copy()
    rival = next((k for k, cc in cands if np.linalg.norm(cc - c) > PARTIAL_RIVAL_M), 0)
    if rival >= PARTIAL_RIVAL_FRAC * best_n:
        return None, "the lid's arc in view fits two places %.0f+ mm apart" % (1000 * PARTIAL_RIVAL_M)
    if best_n < PARTIAL_SETTLE_MIN_POINTS and any(
        k >= PARTIAL_RIVAL_FRAC * best_n and np.linalg.norm(cc - c) > PARTIAL_RIVAL_M for k, cc in beyond
    ):
        return None, (
            "the lid's arc in view fits two places %.0f+ mm apart, one past its box: the %d points of the "
            "other are too few to settle it (need %d)" % (1000 * PARTIAL_RIVAL_M, best_n, PARTIAL_SETTLE_MIN_POINTS)
        )
    picked = c.copy()
    on = np.abs(np.linalg.norm(P - c, axis=1) - r) < PARTIAL_INLIER_M
    for _it in range(30):
        dv = P[on] - c
        dist = np.maximum(np.linalg.norm(dv, axis=1), 1e-9)
        step = np.linalg.lstsq(-dv / dist[:, None], -(dist - r), rcond=None)[0]
        c = c + step
        if np.linalg.norm(step) < 1e-7:
            break
    if len(sides):
        # where the depth runs to the frame's edge (a right-edge cut: the OAK's
        # no-depth strip is at its left), the smear past the lid's silhouette
        # pulls the refined circle 20-26 mm past a lid box (detbench P2 frames
        # re-cut through the lid, as it is and mirrored: 39 of 528 calls); the
        # circle it was refined from fixes 35 of them 20-29 mm from the
        # reference, where c693865 put 14 further than 30 mm. A holder's side
        # stands 18-33 mm outside the lid's circle (P4): past it, the circle is
        # not the lid.
        out = float(past_sides(c, sides, cam_xyz, z, r)[0])
        if out > PARTIAL_BOX_SLACK_M:
            if not side_is_lid:
                return None, "the lid's circle in view, refined, runs %.1f mm past its box" % (1000 * out)
            c = picked
    on = np.abs(np.linalg.norm(P - c, axis=1) - r) < PARTIAL_INLIER_M
    if on.sum() < MIN_LID_POINTS:
        return None, "only %d of the lid's side points lie on a %.0f mm circle" % (int(on.sum()), 2000 * r)
    ang = np.sort(np.arctan2(P[on, 1] - c[1], P[on, 0] - c[0]))
    arc = 360.0 - float(np.degrees(np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max()))
    if arc < PARTIAL_MIN_ARC_DEG:
        return None, "only %.0f deg of the lid's circle in view (need %.0f)" % (arc, PARTIAL_MIN_ARC_DEG)
    return (float(c[0]), float(c[1])), None


def frame_edges(box, image_wh, edge_px=FRAME_EDGE_PX):
    """The edges ("left", "top", "right", "bottom") of a w x h image that
    the pixel box [x0, y0, x1, y1] reaches."""
    w, h = image_wh
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    hits = (
        ("left", x0 <= edge_px),
        ("top", y0 <= edge_px),
        ("right", x1 >= w - 1 - edge_px),
        ("bottom", y1 >= h - 1 - edge_px),
    )
    return tuple(e for e, hit in hits if hit)


def side_cut(edges, R_base_color):
    """True when every edge in `edges` cuts the view across, not up or down:
    its image direction is level in the base frame (a camera hung level, as
    flatbot's scene OAK is, cuts a box at its left or right side)."""
    R = np.asarray(R_base_color, float)
    col = {"left": 0, "right": 0, "top": 1, "bottom": 1}
    return all(abs(R[2, col[e]]) < 0.5 for e in edges)


def off_frame_ways(edges, R_base_color):
    """The base-frame axis ("+y", ...) each image edge in `edges` lies
    toward: the image's own left and right mean nothing on a camera hung
    upside down."""
    R = np.asarray(R_base_color, float)
    out = {"right": R[:, 0], "left": -R[:, 0], "bottom": R[:, 1], "top": -R[:, 1]}
    ways = []
    for e in edges:
        d = out[e]
        k = int(np.argmax(np.abs(d)))
        ways.append(("+" if d[k] > 0 else "-") + "xyz"[k])
    return ways


def off_frame_why(score, edges, R_base_color, frame="arm_mount", src="OWL"):
    """What to tell the operator about a box the detector `src` found
    running off the frame at `edges`: which way that is in `frame`
    (off_frame_ways), and the way back."""
    ways = off_frame_ways(edges, R_base_color)
    back = ", ".join(("-" if w[0] == "+" else "+") + w[1] for w in ways)
    return (
        "the %s's box (%.2f) runs off the scene camera's frame toward %s %s: it sees part of the "
        "container, and a part is not where the box is — move the box toward %s until all of it is "
        "in the camera's view" % (src, score, frame, " and ".join(ways), back)
    )


# The top face is seen only from above it. A camera that stands less than
# this slope above the lid (rise over horizontal distance to the box, ~6
# deg) sees the lid's side band edge-on, and that band passes the top-face
# lift's footprint gates: flatbot 2026-10-22, camera z 0.400, lid top 0.397 —
# the top lift won or lost on a few pixels of the OWL box and moved the fix
# 13 mm and the lid 7.5 mm between runs of the same placement.
TOP_VIEW_MIN_SLOPE = 0.1


def sees_top_face(pts_base, uv, bbox, lid_z, cam_xyz):
    """Does a camera at `cam_xyz` stand high enough over a lid at `lid_z`
    to see its top face, for the depth points inside `bbox`?"""
    x0, y0, x1, y1 = bbox
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    if not inside.any():
        return True  # the lift refuses it with its own reason
    cam = np.asarray(cam_xyz, float)
    d = float(np.median(np.hypot(pts_base[inside, 0] - cam[0], pts_base[inside, 1] - cam[1])))
    return cam[2] - float(lid_z) >= TOP_VIEW_MIN_SLOPE * d


# Two candidates of one message that both pass this far apart (xy) are two
# places, not one box: the message is ambiguous, and the OWL settles it
# (SceneLocator.locate). A band above some other bowl lifts whatever stands
# behind it; 33 of the 35 detbench phantoms the gate still passed are caught
# this way when the canister is in view too, with no false ambiguity in 248
# multi-candidate samples.
RIVAL_M = 0.035


def box_key(box):
    """A pixel box as a tuple of floats: candidates are compared by value,
    whichever sequence type carried them."""
    return tuple(float(v) for v in box[:4])


def scene_fix_from_boxes(
    boxes,
    pts_base,
    uv,
    table_z,
    model,
    cam_xyz=None,
    side_table_z=None,
    image_wh=None,
    cam_R=None,
    src="OWL",
    rivals=False,
):
    """A detector's candidates (score, [x0, y0, x1, y1][, holder]) best
    first: every candidate lifted to the lid slab (box_from_scene_points)
    when a table height is known (`table_z`) and the camera stands above the
    lid (sees_top_face), then — with `cam_xyz`, the camera's position —
    every candidate from its side (box_from_side_points), against
    `side_table_z` when surveyed or the table around it. Which view works
    depends on how high the table stands against the camera, so neither is
    assumed. Two passes, not both lifts per candidate: the side lift never
    measures how far a container runs along the view, so a tray the top-face
    lift just refused would pass it ahead of the real box. The first that is
    this container wins. What the live locate() does with every message, and
    what an offline replay of recorded frames computes the same way. With
    `image_wh` (the colour image's w, h), a candidate reaching the frame's
    edge is refused before either lift (FRAME_EDGE_PX) — unless the model
    is round and the frame cuts it at its side (`cam_R`, the camera's axes
    in the base frame, says which image edges are), when the side lift
    finishes it from the part in view (partial), inside the box's sides the
    frame does not cut (uncut_sides).

    A candidate's third field, when it has one, is the holder box under a
    lid band (segmenter_source.SceneCand): an uncut one arms the side
    lift's looks-past refusal (holder_front_rng). A SceneCand of a kind in
    HOLDER_SIDE_KINDS has its holder's sides, not its lid's (lid_circle's
    `side_is_lid`); an OWL box is the lid's. `src` names the detector
    in every reason. With `rivals`, every OTHER candidate is gated on its
    own too, and the tops that pass more than RIVAL_M from the winner's are
    the fix's `rivals`."""
    # (why, the candidate's own box) for each refusal, in order: one list, so
    # SceneBoxFix's tried and tried_boxes never come apart (_tried)
    refused = []
    clipped = []
    # (score, box, the side edges a round box is finished across or (), the holder that anchors it or None,
    # whether the box's sides are its lid's)
    cands = []
    for c in boxes:
        score, box = c[0], c[1]
        holder = c[2] if len(c) > 2 else None
        side_is_lid = getattr(c, "kind", None) not in HOLDER_SIDE_KINDS
        edges = frame_edges(box, image_wh) if image_wh is not None else ()
        if not edges:
            cands.append((score, box, (), holder, side_is_lid))
        elif model.round and cam_R is not None and side_cut(edges, cam_R):
            cands.append((score, box, edges, None, side_is_lid))
        else:
            clipped.append((score, box, edges))
            refused.append(("it runs off the frame at its %s edge (%s %.2f)" % (" and ".join(edges), src, score), box))
    got = None
    if table_z is not None:
        lid_z = float(table_z) + float(model.dims[2])
        for score, box, edges, _holder, _side_is_lid in cands:
            if edges:
                continue  # only the side lift finishes a box from part of it
            if cam_xyz is not None and not sees_top_face(pts_base, uv, box, lid_z, cam_xyz):
                continue
            pose, top, n = box_from_scene_points(pts_base, uv, box, table_z, model)
            if pose is not None:
                got = SceneBoxFix(
                    pose, top, n, box, table_z=float(table_z), lift="top", clipped=tuple(clipped), **_tried(refused)
                )
                break
            refused.append(("%s (%s %.2f)" % (top, src, score), box))
    if got is None and cam_xyz is not None:
        for score, box, edges, holder, side_is_lid in cands:
            anchor = None if holder is None else holder_front_rng(pts_base, uv, holder, cam_xyz)
            sides = uncut_sides(box, edges, pts_base, uv, cam_xyz, cam_R) if edges else ()
            pose, top, n, tz = box_from_side_points(
                pts_base,
                uv,
                box,
                side_table_z,
                model,
                cam_xyz,
                partial=bool(edges),
                anchor_rng=anchor,
                sides=sides,
                side_is_lid=side_is_lid,
            )
            if pose is not None:
                got = SceneBoxFix(
                    pose, top, n, box, table_z=tz, lift="side", clipped=tuple(clipped), partial=edges, **_tried(refused)
                )
                break
            if edges:
                clipped.append((score, box, edges))
                why = "from the side, cut off at the frame's %s edge, %s (%s %.2f)" % (
                    " and ".join(edges),
                    top,
                    src,
                    score,
                )
                refused.append((why, box))
            else:
                refused.append(("from the side, %s (%s %.2f)" % (top, src, score), box))
    if got is None:
        return SceneBoxFix(None, None, 0, None, clipped=tuple(clipped), **_tried(refused))
    if rivals and len(boxes) > 1:
        won = box_key(got.box)
        others = []
        for c in boxes:
            if box_key(c[1]) == won:
                continue
            alone = scene_fix_from_boxes(
                [c],
                pts_base,
                uv,
                table_z,
                model,
                cam_xyz=cam_xyz,
                side_table_z=side_table_z,
                image_wh=image_wh,
                cam_R=cam_R,
                src=src,
            )
            if (
                alone.pose is not None
                and float(np.hypot(alone.top[0] - got.top[0], alone.top[1] - got.top[1])) > RIVAL_M
            ):
                others.append(tuple(float(v) for v in alone.top))
        if others:
            got = replace(got, rivals=tuple(others))
    return got


def _tried(refused):
    """SceneBoxFix's tried and tried_boxes from the gate's (why, box) pairs."""
    return {"tried": tuple(why for why, _box in refused), "tried_boxes": tuple(box for _why, box in refused)}


def table_from_scene_points(pts_base, uv, image_h, row_frac=0.6):
    """The table height the scene depth shows, from a plane fit over the
    lower part of the (upright) view; nan when there is no clean plane.
    Logged beside the surveyed height so a wrong survey, or a knocked
    camera, shows up in every run."""
    low = uv[:, 1] > row_frac * image_h
    if low.sum() < 500:
        return float("nan")
    # a few thousand points are plenty for a table plane, and this number
    # is only PRINTED: over the full lower image (~150k points) the RANSAC
    # cost 1.6-2.3 s of a 3.9 s locate (profiled live 2026-09-17)
    fit = fit_plane(pts_base[low], tol_m=0.008, iters=150, max_points=3000)
    if fit is None or fit[2] < 0.3:
        return float("nan")
    n, c, _frac = fit
    if abs(n[2]) < 0.9:
        return float("nan")  # the dominant plane down there is not horizontal
    return float(c[2])


# |image-down . base +z| below this: the camera looks too steeply down for
# "which way up" to mean anything
UPRIGHT_COS = 0.5


def image_upside_down(T_base_color):
    """True when the colour image's down axis points up in the base frame,
    False when it points down, None when the camera looks too steeply down
    to tell."""
    down_z = float(np.asarray(T_base_color)[2, 1])
    if abs(down_z) < UPRIGHT_COS:
        return None
    return down_z > 0


# --- the vote: two sightings make a fix -------------------------------------
# One frame never moves the arm (d405.stable_fix): a scene fix commits when
# the two newest sightings of one locate() agree within VOTE_TOL_M. With the
# segmenter at 5 Hz that costs one frame, 0.2 s.
VOTE_N, VOTE_TOL_M = 2, 0.020
# The scene OWL is enabled when the segmenter has made no fix this long after
# the streams were ready — and at once when the segmenter is down or a message
# is ambiguous. The OWL needs ~1.3 s an inference and two boxes to commit.
OWL_AFTER_S = 1.0
SEG_DOWN = ("unavailable", "absent", "silent", "wrong_camera", "wrong_size", "stale")  # SegmenterRung.state
SRC_LABEL = {"segmenter": "segmenter", "owl": "OWL"}  # how each source is named in the gate's reasons


@dataclass(frozen=True)
class Sighting:
    """One message's fix, as the vote counts it."""

    source: str  # "segmenter" | "owl"
    score: float  # the detector's score of the candidate that passed
    top: tuple  # the lid top's centre (base)
    pose: object  # ContainerPose
    n: int  # lid points behind it
    partial: tuple  # the edges the frame cut it at, when it was finished from part of it
    table_z: float  # the table it was measured against
    lift: str  # "top" | "side"
    box: tuple  # the box the gate lifted
    det_box: tuple  # the detector's own box (the holder under a band)


def commit_fix(sightings, segmenter_saw, n=VOTE_N, tol=VOTE_TOL_M, owl_settles=False, segmenter_places=()):
    """(top, None) when the newest `n` sightings agree pairwise within `tol`
    — their median (d405.stable_fix) — else (None, why), why one of "too
    few", "disagree", "cut OWL box unconfirmed" and "two places unsettled".

    `segmenter_saw`: the segmenter is live and has seen the container in
    this look (a sighting, or a message that passed in two places). The
    OWL's box of a cut lid (a partial lift) then counts only with the
    segmenter beside it: a segmenter sighting in the pair, or one of
    `segmenter_places` within `tol` of each of the pair's sightings, as a
    third sighting would be. The segmenter can confirm it, so it must.
    Down, or live but blind (no food container, or every band refused), it
    confirms nothing, and the OWL's cut boxes commit on their own. Before
    the cut lid's arc was bounded by its box's uncut side, the OWL's circle
    at detbench 163450 slid 46 mm in 39 % of its commits, so it waited on
    any live segmenter; a blind one then left all 15 cut frames unfixed.
    Bounded, those commits land 12-14 mm off and all 15 commit
    (detbench/spec blind_stream).

    `segmenter_places`: the tops of the places this look's segmenter
    messages passed in two (SceneBoxFix.top and .rivals). Such a message is
    no sighting, so with only a sighting to confirm it, a look whose every
    frame passed two places — the canister half off the frame, beside a
    second bowl whose band passes too — held the OWL's cut pair until it
    timed out: NO BOX, nothing moved (the 2026-10-28 merge review of #43 +
    #57). On detbench's 15 cut frames every segmenter fix of the canister
    lies within 17.8 mm of each sighting of every OWL cut pair that agrees.

    `owl_settles`: a segmenter message of this look passed in two places
    (SceneBoxFix.rivals). Two later segmenter sightings that agree on one of
    them do not say which is the container — a frame where the canister's
    band failed and a phantom bowl's passed, twice, would agree on the
    phantom — so the pair must hold an OWL sighting: the OWL boxes the lid
    itself."""
    if len(sightings) < n:
        return None, "too few"
    top = stable_fix([(s.top, i) for i, s in enumerate(sightings)], tol, n)
    if top is None:
        return None, "disagree"
    pair = sightings[-n:]
    beside = any(s.source == "segmenter" for s in pair) or any(
        all(np.linalg.norm(np.subtract(p, s.top)) <= tol for s in pair) for p in segmenter_places
    )
    if segmenter_saw and any(s.partial and s.source == "owl" for s in pair) and not beside:
        return None, "cut OWL box unconfirmed"
    if owl_settles and not any(s.source == "owl" for s in pair):
        return None, "two places unsettled"
    return top, None


def _newer(stamp, since, last):
    """Is a segmenter frame stamped `stamp` (ns) newer than `since` and than
    `last`, the frame gated before it (a SegFrame or None)?"""
    if stamp is None:
        return True  # unstamped: only "not the frame gated last" decides
    if since is not None and stamp <= since:
        return False
    return last is None or last.stamp_ns is None or stamp > last.stamp_ns


def _top_xy(top):
    return "[%.3f, %.3f]" % (float(top[0]), float(top[1]))


class SceneLocator:
    """The ROS side: the scene grabber, the scene segmenter and OWL rungs
    and the calibration, answering `locate()` with a SceneFix or None."""

    def __init__(
        self, node, cfg, model, table_z, T_base_link, spins=True, tag_px=None, scene_cfg=None, base_frame=None
    ):
        from box_opening.perception.owl_source import OwlRung
        from box_opening.perception.scene import SceneGrabber
        from box_opening.perception.segmenter_source import SegmenterRung

        self.node = node
        # spins=False: the node is spun elsewhere (the mission's perception
        # thread); locate() then only waits, it must not spin it too
        self.spins = bool(spins)
        self.cfg = cfg
        self.model = model
        self.table_z = float(table_z) if table_z is not None else None
        # None: the robot's TF carries the camera's mount (a world with
        # external_static_tf), read as base_frame <- the camera's link
        self.T_base_link = None if T_base_link is None else np.asarray(T_base_link, dtype=float)
        self.base_frame = base_frame
        self.grab = SceneGrabber(node, keep=3, need_depth=True, scene_cfg=scene_cfg)
        self.owl = OwlRung(node, cfg, camera="scene")
        # sheppy's scene segmenter, for a container whose config names the
        # classes it is boxed as (scene_detect.segmenter_classes); None: the
        # scene OWL alone finds it, as before
        classes = tuple(getattr(cfg, "segmenter_classes", ()) or ())
        self.seg = SegmenterRung(node, classes) if classes else None
        self.seg_box = getattr(cfg, "segmenter_box", "holder")
        self.last_why = None
        self.last_timing = {}
        # source -> the raw boxes of the last message locate() gated (the detection set keeps them)
        self.last_boxes = {}
        self.last_judged = {}  # source -> (its candidates, SceneBoxFix, colour frame, T_base_color): what viz draws
        self.last_seg_state = None  # the segmenter's state when locate() or prime() last looked
        self._clock, self._sleep = time.monotonic, time.sleep  # tests drive them
        # where the calibration saw the cabinet's tag (scene_calib.scene_tag_px):
        # checked once per locate, before any fix is trusted
        self.tag_px = tag_px
        self.moved_note = None  # "tag not in view" and the like, for the log
        if self.seg is not None and not self.seg.available:
            log.info(
                "[scene] the scene segmenter cannot be read (%s): the scene OWL alone finds the box" % self.seg.why
            )

    def base_to_link(self):
        """T(base <- the camera's link): the calibration's, or TF's."""
        if self.T_base_link is not None:
            return self.T_base_link
        return self.grab.lookup(self.base_frame, self.grab.link_frame)

    def base_to_color(self):
        T_bl, T_lc = self.base_to_link(), self.grab.link_to_color()
        return None if T_bl is None or T_lc is None else T_bl @ T_lc

    def ready(self):
        return not self.grab.missing() and self.grab.depth_to_color() is not None and self.base_to_color() is not None

    def _unplaced_why(self):
        if self.grab.depth_to_color() is None:
            return "no TF between the scene camera's depth and colour frames"
        if self.T_base_link is None:
            return "no TF from %s to the scene camera (%s) — is sheppy's static_tf node up?" % (
                self.base_frame,
                self.grab.link_frame,
            )
        return "no TF from the scene camera's link to its colour frame"

    def _seg_args(self, now):
        """(now, frame_id, (w, h), stamp ns) of the grabber's own newest colour
        frame: what a segmenter message must match (SegmenterRung.state)."""
        from box_opening.perception.segmenter_source import stamp_ns

        g = self.grab
        color = g.color
        wh = None if color is None else (int(color.shape[1]), int(color.shape[0]))
        return now, getattr(g, "color_frame_id", None), wh, stamp_ns(getattr(g, "color_stamp", None))

    def prime(self):
        """Ahead of a locate() (at construction; before the look after a
        press): the scene OWL enabled now where the segmenter cannot answer —
        none for this container, or down — so the OWL's first box (~0.7 s,
        inferred on the enable edge) comes while the caller does something
        else. A live segmenter, or one still in its first second, leaves the
        OWL off and off the GPU: locate() enables it itself when it must."""
        if self.seg is None or not self.seg.available:
            self.owl.enable()
            return
        self.last_seg_state = self.seg.state(*self._seg_args(self._clock()))
        if self.last_seg_state in SEG_DOWN:
            self.owl.enable()

    def locate(self, timeout_s):
        """The box, from the scene frames that arrive from now on. Each new
        segmenter message's boxes (the band above the holder) go through the
        gate (scene_fix_from_boxes) as they come; the scene OWL's boxes too
        once the segmenter is down, a message is ambiguous, or OWL_AFTER_S
        has passed with no fix. A fix commits when the two newest sightings
        agree (commit_fix). Sightings belong to this call only: the looks
        after a press never mix with those before it. None, with
        `last_why`, when nothing commits in `timeout_s`."""
        from box_opening.perception.owl_source import bboxes_in_msg
        from box_opening.perception.segmenter_source import segmenter_candidates, stamp_ns

        clock = self._clock
        g = self.grab
        seg = self.seg
        t0 = clock()
        self.last_timing = {}  # where a locate's seconds went (the SCENE line prints it)
        self.last_boxes = {}
        self.last_judged = {}
        # only frames newer than the newest one held now: those kept from
        # before the call (before a press, say) must not vote
        since = stamp_ns(getattr(g, "color_stamp", None))
        t_ready = None
        owl_on = False

        def owl_now():
            nonlocal owl_on
            if not owl_on:
                self.owl.enable()
                owl_on = True
                self.last_timing["owl_s"] = 0.0 if t_ready is None else round(clock() - t_ready, 2)

        try:
            # no segmenter for this container (or none this install can read),
            # or prime() found it down: the OWL from the start (an enable
            # repeated refreshes the node's window)
            if seg is None or not seg.available or getattr(self.owl, "enabled", False):
                owl_now()
            heard_owl = False  # any message at all from the scene OWL
            last_owl = last_seg = None  # the messages gated last, per source
            n_seg = n_seg_boxes = 0  # segmenter frames gated in this call, and the boxes of its classes in them
            tried = {"segmenter": [], "owl": []}  # why each refused candidate was refused, per source
            ambiguous = {"segmenter": [], "owl": []}  # (top, rivals) of each message that passed in two places
            # a segmenter message passed in two places: from then on in this
            # look, only a pair with an OWL sighting in it commits (commit_fix)
            owl_settles = False
            sightings = []
            unconfirmed = False  # an OWL pair on a cut lid was held back: nothing the segmenter saw stood beside it
            off_frame = None  # (score, edges, R_base_color, src) of the best candidate cut by the frame's edge
            state = seg_args = m_seg = None
            found = None
            if self.spins:
                import rclpy
            while clock() - t0 < timeout_s:
                if self.spins:
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                else:
                    self._sleep(0.02)
                if not self.ready():
                    continue
                if t_ready is None:
                    t_ready = clock()
                    self.last_timing["streams_s"] = round(t_ready - t0, 2)
                    # the detectors need upright frames (upside down, RF-DETR
                    # found the box in 7 of 37 detbench frames, the OWL in 15),
                    # and nothing here turns them: a camera hung upside down
                    # rotates them on the device
                    if image_upside_down(self.base_to_color()):
                        self.last_why = (
                            "TF has the scene camera's frames upside down — its detectors need them upright: "
                            "rotate them on the device (the OAK driver's color/depth i_rotation 180, "
                            "rammp-deployments #133) with the upright mount in TF"
                        )
                        return None
                    # is the calibration's tag where it was? Said, never
                    # refused: it rides the cabinet door (scene_calib
                    # TAG_MOVED_PX); a re-aimed camera is caught by what its
                    # fixes measure instead (press_demo.scene_trust)
                    from box_opening.perception.scene_calib import scene_camera_moved

                    _moved, why = scene_camera_moved(g.color, self.tag_px)
                    self.moved_note = why
                    if since is None:  # no frame at the call: the first one counts
                        first = stamp_ns(getattr(g, "color_stamp", None))
                        since = None if first is None else first - 1
                now = clock()
                if seg is not None:
                    m_seg = seg.latest  # one read: the perception thread replaces it whole
                    seg_args = self._seg_args(now)
                    state = seg.state(*seg_args, frame=m_seg)
                    self.last_seg_state = state
                if not owl_on and (seg is None or state in SEG_DOWN or now - t_ready >= OWL_AFTER_S):
                    owl_now()
                msgs = []
                if state == "live" and m_seg is not last_seg and _newer(m_seg.stamp_ns, since, last_seg):
                    last_seg = m_seg
                    n_seg += 1
                    self.last_timing.setdefault("seg_s", round(now - t_ready, 2))
                    self.last_boxes["segmenter"] = [[c, float(s)] + [float(v) for v in b] for c, s, b in m_seg.objects]
                    n_seg_boxes += len(m_seg.objects)
                    cands = segmenter_candidates(m_seg.objects, seg.classes, seg_args[2], box_kind=self.seg_box)
                    if cands:
                        msgs.append(("segmenter", cands))
                if owl_on and self.owl.latest is not None:
                    heard_owl = True
                    # the newest BOX, not the newest message: a heartbeat that
                    # lands milliseconds after it must not hide it (2026-09-23)
                    m = self.owl.fresh_box()
                    if m is not None and m is not last_owl:
                        last_owl = m
                        boxes = bboxes_in_msg(m)
                        self.last_timing.setdefault("bbox_s", round(now - t_ready, 2))
                        self.last_boxes["owl"] = [[float(b[4])] + [float(v) for v in b[:4]] for b in boxes]
                        if boxes:
                            msgs.append(("owl", [(b[4], b[:4]) for b in boxes]))
                if not msgs:
                    continue
                # every candidate of every new message, best first, against the
                # container's geometry, on one cloud: the first whose top is at
                # lid height AND the container's size passes
                t_cloud = clock()
                cloud = g.cloud_in_color_frame(depth=g.depth_median(), stride=2)
                if cloud is None:
                    self.last_why = "no scene depth cloud"
                    return None
                pc, uv = cloud
                T_base_color = self.base_to_color()
                if T_base_color is None:
                    continue
                pts_base = pc @ T_base_color[:3, :3].T + T_base_color[:3, 3]
                h, w = g.color.shape[:2]
                # the top-face lift needs a table height up front: surveyed,
                # else read off the image rows (no clean plane there is no
                # refusal: the side lift reads the table around each box)
                top_table_z = self.table_z
                if top_table_z is None:
                    measured = table_from_scene_points(pts_base, uv, h)
                    top_table_z = None if np.isnan(measured) else measured
                for source, cands in msgs:
                    label = SRC_LABEL[source]
                    got = scene_fix_from_boxes(
                        cands,
                        pts_base,
                        uv,
                        top_table_z,
                        self.model,
                        cam_xyz=T_base_color[:3, 3],
                        side_table_z=self.table_z,
                        image_wh=(w, h),
                        cam_R=T_base_color[:3, :3],
                        src=label,
                        rivals=True,
                    )
                    tried[source].extend(got.tried)
                    self.last_judged[source] = (cands, got, g.color, T_base_color)  # what perception/viz draws
                    for score, _box, edges in got.clipped:
                        if off_frame is None or score > off_frame[0]:
                            off_frame = (score, edges, T_base_color[:3, :3], label)
                    if got.pose is None:
                        continue
                    if got.rivals:
                        # two places pass as this container: no sighting, and
                        # the OWL, which boxes the lid itself, settles which
                        ambiguous[source].append((got.top, got.rivals))
                        owl_settles = owl_settles or source == "segmenter"
                        owl_now()
                        continue
                    key = box_key(got.box)
                    c = next(c for c in cands if box_key(c[1]) == key)
                    det = getattr(c, "det_box", None)
                    sightings.append(
                        Sighting(
                            source,
                            float(c[0]),
                            tuple(float(v) for v in got.top),
                            got.pose,
                            int(got.n),
                            tuple(got.partial),
                            float(got.table_z),
                            got.lift,
                            key,
                            key if det is None else box_key(det),
                        )
                    )
                    # a live segmenter confirms a cut OWL box only once it has seen the container in this look:
                    # a sighting beside it, or a place one of its messages passed in two (commit_fix)
                    seg_saw = state == "live" and (
                        bool(ambiguous["segmenter"]) or any(s.source == "segmenter" for s in sightings)
                    )
                    places = [p for at, riv in ambiguous["segmenter"] for p in (at, *riv)]
                    top, why = commit_fix(sightings, seg_saw, owl_settles=owl_settles, segmenter_places=places)
                    unconfirmed = unconfirmed or why == "cut OWL box unconfirmed"
                    if top is not None:
                        found = (top, sightings[-VOTE_N:], pts_base, uv, got, T_base_color)
                        t_fix = clock()
                        self.last_timing["fix_s"] = round(t_fix - t_ready, 2)
                        self.last_timing["lift_s"] = round(t_fix - t_cloud, 2)
                        break
                if found is not None:
                    break
            if found is None:
                n_owl = sum(s.source == "owl" for s in sightings)
                if g.missing():
                    why = "scene camera streams missing: %s" % ", ".join(g.missing())
                elif t_ready is None:
                    why = self._unplaced_why()
                else:
                    # three different failures wear the same "no box" face: the
                    # camera is down, a detector is silent (not started, still
                    # loading, or getting no frames — a launch-started OWL
                    # instance was seen like that, 2026-09-16), or it is alive
                    # and honestly sees no box
                    owl_part = _owl_why(
                        timeout_s, owl_on or seg is None, heard_owl, tried["owl"], ambiguous["owl"], n_owl, unconfirmed
                    )
                    if seg is None:
                        why = owl_part
                    else:
                        if state in SEG_DOWN:
                            seg_part = seg.describe(state, *seg_args[1:], frame=m_seg)
                        elif n_seg == 0:
                            seg_part = "no frame newer than the look in %.1f s" % timeout_s
                        elif n_seg_boxes == 0:
                            seg_part = "saw no %s in %d frames" % (" or ".join(seg.classes), n_seg)
                        elif ambiguous["segmenter"]:
                            top, riv = ambiguous["segmenter"][-1]
                            seg_part = (
                                "two places pass as this container, %s and %s, and the OWL did not settle which"
                                % (_top_xy(top), _top_xy(riv[0]))
                            )
                        elif unconfirmed:
                            seg_part = "the OWL's box of the cut lid is not confirmed by the segmenter's"
                        elif len(sightings) == n_owl:
                            # the same box is refused for the same reason 5 times a second: the last 3 reasons
                            # that differ
                            last = list(dict.fromkeys(reversed(tried["segmenter"])))[:3][::-1]
                            seg_part = "none of its %d boxes is this container: %s" % (
                                len(tried["segmenter"]),
                                "; ".join(last),
                            )
                        elif len(sightings) - n_owl == 1:
                            seg_part = "one of its boxes passed in %d frames, and a fix needs two that agree" % n_seg
                        else:
                            seg_part = "its fixes did not agree within %.0f mm" % (1000 * VOTE_TOL_M)
                        why = "scene segmenter: %s; scene OWL: %s" % (seg_part, owl_part)
                    if off_frame is not None:
                        score, edges, R, label = off_frame
                        why = "%s; %s" % (
                            off_frame_why(score, edges, R, frame=self.base_frame or "base_link", src=label),
                            why,
                        )
                self.last_why = why
                return None
            top, pair, pts_base, uv, got, T_base_color = found
            top = tuple(float(v) for v in top)
            newest = pair[-1]
            pose = container_pose_from_top(top, newest.pose.yaw, self.model, newest.table_z)
            table_seen = got.table_z if got.lift == "side" else table_from_scene_points(pts_base, uv, g.color.shape[0])
            # no survey: the table the box was found on is the one the
            # planner must know about (WorldStore.table_extent)
            extent = None
            if self.table_z is None:
                extent, assumed = seen_table_extent(pts_base, got.table_z, top[:2])
                if assumed:
                    log.info(
                        "[scene] too few table points to measure its extent; planning the %.2f m ring "
                        "square around the box at the measured height" % (2 * TABLE_RING_M[1])
                    )
            partial = ""
            if got.partial:
                partial = "cut off by the frame toward %s %s" % (
                    self.base_frame or "base_link",
                    " and ".join(off_frame_ways(got.partial, T_base_color[:3, :3])),
                )
            near = np.hypot(pts_base[:, 0] - top[0], pts_base[:, 1] - top[1]) < MAP_R_M
            return SceneFix(
                pose=pose,
                top_xyz=top,
                n_points=newest.n,
                score=newest.score,
                table_z_scene=table_seen,
                table_extent=extent,
                partial=partial,
                table_points=pts_base[near].astype(np.float32),
                source=newest.source,
                n_sightings=len(pair),
                sources=tuple(s.source for s in pair),
            )
        finally:
            self.owl.disable()


def _owl_why(timeout_s, asked, heard, tried, ambiguous, n_sightings, unconfirmed):
    """Why the scene OWL gave no fix: today's three texts, and the vote's."""
    if not asked:
        return "not asked: the segmenter was live"
    if not heard:
        return (
            "the scene OWL is SILENT (no heartbeat in %.1f s) — not started, still "
            "loading, or not receiving frames: check the launch terminal" % timeout_s
        )
    if unconfirmed and n_sightings >= VOTE_N:
        return "its boxes of the cut lid wait on the segmenter's"
    if n_sightings == 1:
        return "one of its boxes passed in %.1f s, and a fix needs two that agree" % timeout_s
    if n_sightings > 1:
        return "its fixes did not agree within %.0f mm" % (1000 * VOTE_TOL_M)
    if ambiguous:
        top, riv = ambiguous[-1]
        return "two places pass as this container, %s and %s" % (_top_xy(top), _top_xy(riv[0]))
    if not tried:
        return "the scene OWL saw no box in %.1f s" % timeout_s
    return "none of the OWL's %d candidate(s) is this container: %s" % (len(tried), "; ".join(tried[-3:]))
