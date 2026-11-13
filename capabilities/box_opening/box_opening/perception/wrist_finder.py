"""Finding the box from the wrist camera, where the scene camera cannot see it.

Parked side-on to a kitchen cabinet (the seated user's chair beside a counter
it cannot go under) the scene camera does not see the box: the chair study's
problem 3 (2026-11-06). The wrist camera does, from a survey pose the arm takes
for it. This module turns one or more of its RGB-D frames into the fix the
rest of the run reads from the scene camera's (scene_source.SceneFix): where
the lid's top is, the container's pose on the counter, the counter's height
and extent, and the points the lid's spot is chosen on (clear_table.table_map).

Geometry first, the button second. A survey frame shows a counter and what
stands on it — the box, cups, a tray, the wall — and no detector is needed
(boxes from /perception/object_detections for the wrist camera, when the
perception stack publishes them, only break a tie). In each frame:

1. the counter: a LEVEL plane through the depth (planes.fit_plane under
   clear_table.TABLE_LEVEL_NZ, as the lid spot's map fits it) that spreads
   wider than a container — a lid top seen close is no counter;
2. what stands on it at lid height: the points in the lid band above that
   plane, as connected patches on a CELL_M grid of the plane (metres, not
   pixels: a near lid and a far one read the same);
3. the container by its geometry: a solid top (a cup's rim is a ring) a lid
   across (the model's lid_dims), at the model's height, nothing standing
   above it (a bottle's shoulder crosses lid height too), the counter seen
   around it;
4. the button where the view allows: the aim's own circle detector
   (button_aim.find_button, rectified when the view is oblique) run on the
   patch — a circle there confirms it; a place no frame that could check it
   shows a button on (a closed tin, a jar) is no container.

The frames' patches are then pooled by place. Two places that pass are two
candidates, and the finder refuses rather than guess which one is the box;
so it does with a lid the view cuts (look again with it whole), and with a
top no frame's view could check the button on: geometry alone does not tell
the box from a closed tin of its size (side-cabinet renders without the box,
a 105 mm tin alone: a fix at it in 48 of 96 frames, every one a view the
button could not be checked from), so it is a place to look at again from
nearer above, not a fix — as the staging aim commits only on the button's
circle (detect.require_button_circle).
The holder (its ~152 mm flange 45 mm under the lid) is not asked for: from a
survey pose the canister hides its far side, and the lid's solid top and its
button already tell the box from cups, trays and tins. Nothing about the
counter or what stands on it is configured: the model's geometry
(config/containers) is the only prior. The fix's confidence: 1.0 with the
button in two frames or more, 0.9 in one, 0.2 less when every view cut the
lid; with the circle detector off (check_buttons False, geometry alone) 0.7
from two frames or more, else 0.6.

Its heights are the wrist camera's own. The lid's height above the counter
it sees is right (92-103 mm on the real looks), but both stand 6-10 mm low
in base z straight down and 20-60 mm low tilted (the wrist mount's error
since its repair; the scene camera's table corrected it until now): the
fix's table_z_scene, top_xyz z and pose z are no table for the planner
(press_demo.adopt_scene_table) and no lid height to fly fast down to — the
touch measures the lid.
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np

from box_opening.perception.clear_table import MAP_R_M, STANDS_M, TABLE_LEVEL_NZ, WRIST_NEAREST_M
from box_opening.perception.planes import fit_plane

# --- the counter -------------------------------------------------------------
MAX_RANGE_M = 1.0  # depth farther than this is too coarse for a 15 mm slab (the OAK's ~1 % of range)
PLANE_TOL_M = 0.008  # a counter point lies this close to its plane (clear_table._heights' fit)
PLANE_ITERS = 300
PLANE_MAX_POINTS = 6000
MAX_PLANES = 3  # level planes tried per frame, largest first: a close lid top can out-count the counter
MIN_PLANE_POINTS = 1500
# a counter spreads at least this far both ways (its inliers' 5-95 % extent along their axes):
# a lid top is a level plane too (its extent 66-87 mm on the real straight-down looks), and so
# is the holder's flange (67-129 mm across its ring); the counter around the box read 186-218
# mm from 0.33 m, the closest the wrist looks down at it
COUNTER_MIN_M = 0.15
# the counter must show around the box (scene_source.TABLE_RING_M's ring): what stands on it
# stands on THIS plane, not on another seen past it
RING_M = (0.08, 0.30)
MIN_RING_POINTS = 150

# --- what stands on it at lid height ------------------------------------------
CELL_M = 0.005  # the plane's grid: a 105 mm lid is 21 cells across
# The lid band, about the model's dims.z (121 mm). The scene camera has measured the canister's
# top 87.9, 98.6 and 129.2 mm above its table (presssim defaults' lid_top_above_table), and the
# wrist reads it 89-106 mm above the counter it sees itself in the real captures that show it
# (its table and lid err alike, 20-60 mm low in base z since its repair: the height is read
# above its own counter, never against a table given to it). Room above for a popped knob.
LID_BELOW_M = 0.045
LID_ABOVE_M = 0.025
TOP_SLAB_M = 0.015  # the lid's top: the densest 15 mm of the patch's heights (scene_source.TOP_SLAB_M)
KNOB_UP_M = 0.025  # a popped knob stands at most this far above its lid (button_aim.KNOB_MAX_UP_M)
MIN_SLAB_CELLS = 40  # 1000 mm2: a tenth of a 105 mm lid
# the slab's outline (cv2.fitEllipse, so neither the knob's hole nor the denser near half moves
# it): its axes against the lid's sides
AXES_FRAC = (0.70, 1.25)
# the slab cells over the outline's area: a lid is solid (0.77-1.09 on the real looks that found
# it, a cut one lowest), an open cup's rim a ring (0.35-0.54 on the side-cabinet renders, noise
# thickening it): 0.55 let a noisy rim pass beside the box 7 times in 60 (no wrong fix: two places)
MIN_FILL = 0.65
CORE_FRAC = 0.30  # of the lid's diameter: the core around the centre ...
CORE_MIN_TOP = 0.6  # ... where what shows at all must mostly be the top, not lower (inside a cup)
ABOVE_M = 0.030  # points this far above the top, over it, are something taller than the box
MAX_ABOVE_FRAC = 0.10  # ... over this share of the slab's points
EDGE_PX = 3  # a top this close to the view's edge is cut by it (scene_source.FRAME_EDGE_PX)
DEPTH_BAND_FRAC = 0.02  # a column (row) with depth in fewer of its pixels is outside the view

# --- the button ----------------------------------------------------------------
BUTTON_NEAR_M = 0.030  # the button circle sits this close to the lid's centre (a 22 mm-radius knob)
CHECK_MAX_INCIDENCE_DEG = 50.0  # the aim's own bound (button_aim.OBLIQUE_MAX_INCIDENCE_DEG)
CHECK_MIN_RANGE_M = 0.12
CHECK_MAX_RANGE_M = 0.70  # past this the knob is under 25 px across, where the aim's gates were never tuned
CHECK_MARGIN_PX = 30  # the lid's centre this far inside the image, or the view does not allow it

# --- pooling ---------------------------------------------------------------------
# two passing patches closer than half a lid are one place (two lids cannot stand that close)
SAME_PLACE_FRAC = 0.5


@dataclass(frozen=True)
class WristFrame:
    """One wrist RGB-D frame, as the captures save it (color, depth, k, rot_cam, trans_cam):
    `depth` in metres aligned to the colour image, the colour camera's optical frame at
    (rot_cam, trans_cam) in the planning frame (arm_mount). `boxes`: the wrist camera's
    detector boxes [(score, [x0, y0, x1, y1]), ...] when the perception stack publishes them,
    else None."""

    color: object
    depth: object
    k: object
    rot_cam: object
    trans_cam: object
    dist: object = None
    boxes: object = None
    tag: str = ""


def frame_from_capture(path, fused=True, boxes=None):
    """A saved capture's frame_000.npz as a WristFrame: its fused depth when it kept one."""
    z = np.load(str(path), allow_pickle=False)
    depth = z["depth_fused"] if fused and "depth_fused" in z.files else z["depth"]
    return WristFrame(
        color=z["color"],
        depth=np.asarray(depth, np.float32),
        k=np.asarray(z["k"], float),
        rot_cam=np.asarray(z["rot_cam"], float),
        trans_cam=np.asarray(z["trans_cam"], float),
        dist=np.asarray(z["dist"], float) if "dist" in z.files else None,
        boxes=boxes,
        tag=str(path),
    )


@dataclass(frozen=True)
class Patch:
    """One thing standing at lid height in one frame, and how the container's gates read it."""

    frame: int
    xy: tuple  # the top's outline centre (base)
    top_z: float  # the top's height (base z: the slab's median)
    table_z: float  # the counter under it (its plane, read at xy)
    height_m: float  # top_z above that counter
    axes_m: tuple  # the outline's (minor, major) axes
    fill: float
    core_top: float  # nan: nothing showed in the core
    above_frac: float
    n_slab: int
    cut: bool  # the frame's edge cuts it
    why: str  # "" when it passes as the container
    button: str = "unchecked"  # "seen" | "none" (the view showed it, no circle) | "unchecked"
    button_xyz: tuple = None
    button_why: str = ""
    boxed: object = None  # True/False: inside a detector box of its frame; None: no boxes
    yaw: float = 0.0  # a square lid's (scene_source.fitted_yaw); 0 for a round one
    rim: bool = False  # placed by the colour image's rim, its depth top too long (by_its_rim)


@dataclass(frozen=True)
class Place:
    """The passing patches of every frame that stand at one place."""

    xy: tuple
    top_z: float
    table_z: float
    patches: tuple
    button: str  # "seen" in any frame | "none": checked, never seen | "unchecked"
    button_xyz: tuple = None


@dataclass(frozen=True)
class WristFind:
    """What the wrist frames say: `fix` the scene_source.SceneFix the run reads (source
    "wrist"), or None with `why`. `confidence` in [0, 1]. `clouds`: [(points, table_z)] per
    frame that showed a counter, around the box — clear_table.table_map's input for the lid's
    spot; the fix's table_points are the same points, each frame's moved onto one counter
    height (table_z_scene), since each pose's z errs its own way. Its z is the wrist's own
    (the module's note): not the planner's table. `places`: every place that passed;
    `patches`: every patch judged, passing or not."""

    fix: object
    why: str
    confidence: float = 0.0
    button_xyz: tuple = None
    clouds: tuple = ()
    places: tuple = ()
    patches: tuple = ()
    frames: int = 0
    counters: int = 0  # frames that showed a counter
    timing_s: dict = field(default_factory=dict)


def lift(frame, max_range=MAX_RANGE_M):
    """(points (N, 3) in the planning frame, pixels (N, 2) as (u, v), at the edge of the view
    (N,) bool) of a frame's depth, without the hand (WRIST_NEAREST_M: the fingertips are 123
    mm out) or what is too far. The view's edge is the image's, moved in past the bands of
    columns and rows where the depth shows nothing (DEPTH_BAND_FRAC): the OAK's stereo leaves
    one down the colour image's left side, and a lid half in it is cut there. A hole inside a
    surface (a glossy knob's, a shadow beside the lid) is no edge. A frame without a depth
    image lifts to no points."""
    if frame.depth is None or np.ndim(frame.depth) != 2 or np.size(frame.depth) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), int), np.zeros(0, bool)
    d = np.asarray(frame.depth, np.float32)
    ok = np.isfinite(d) & (d > WRIST_NEAREST_M) & (d < max_range)
    v, u = np.nonzero(ok)
    z = d[v, u].astype(float)
    kk = np.asarray(frame.k, float)
    x = (u - kk[0, 2]) / kk[0, 0] * z
    y = (v - kk[1, 2]) / kk[1, 1] * z
    pts = np.stack([x, y, z], -1) @ np.asarray(frame.rot_cam, float).T + np.asarray(frame.trans_cam, float)
    edge = np.zeros(len(u), bool)
    for axis, px in ((0, u), (1, v)):
        shown = np.flatnonzero(ok.mean(axis=axis) > DEPTH_BAND_FRAC)
        if len(shown):
            edge |= (px - shown[0] <= EDGE_PX) | (shown[-1] - px <= EDGE_PX)
    return pts, np.stack([u, v], -1), edge


def _plane_z(plane, xy):
    """The plane's z at xy (planes are (normal up, centroid, ...))."""
    n, c = plane[0], plane[1]
    return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / n[2])


def level_planes(pts, n_max=MAX_PLANES):
    """The level planes through `pts` that could be a counter, largest first: [(normal up,
    centroid, inlier mask)], each fitted on the points no earlier plane holds, kept when it
    holds MIN_PLANE_POINTS and spreads COUNTER_MIN_M both ways."""
    out = []
    free = np.ones(len(pts), bool)
    rng = np.random.default_rng(0)
    for _ in range(n_max):
        if free.sum() < MIN_PLANE_POINTS:
            break
        fit = fit_plane(
            pts[free], tol_m=PLANE_TOL_M, iters=PLANE_ITERS, max_points=PLANE_MAX_POINTS, rng=rng, min_nz=TABLE_LEVEL_NZ
        )
        if fit is None:
            break
        n, c, _frac = fit
        n = n if n[2] > 0 else -n
        on = free & (np.abs((pts - c) @ n) < PLANE_TOL_M)
        if on.sum() < MIN_PLANE_POINTS:
            break
        free &= ~on
        if n[2] < TABLE_LEVEL_NZ:  # its inliers' own fit leans (clear_table._heights asks the same)
            continue
        xy = pts[on, :2] - c[:2]
        _w, vecs = np.linalg.eigh(np.cov(xy.T))
        lo, hi = np.percentile(xy @ vecs, [5, 95], axis=0)
        if float(np.min(hi - lo)) >= COUNTER_MIN_M:
            out.append((n, c, on))
    return out


def _cells(xy, x0, y0):
    return np.floor((xy[:, 0] - x0) / CELL_M).astype(int), np.floor((xy[:, 1] - y0) / CELL_M).astype(int)


def _grid(xy, pad):
    """`xy` on a CELL_M occupancy grid, `pad` cells around it: (mask uint8, x0, y0, ix, iy)."""
    x0, y0 = xy.min(axis=0) - pad * CELL_M
    ix, iy = _cells(xy, x0, y0)
    mask = np.zeros((int(iy.max()) + 1 + pad, int(ix.max()) + 1 + pad), np.uint8)
    mask[iy, ix] = 1
    return mask, x0, y0, ix, iy


def _densest_slab(h):
    """The bottom of the TOP_SLAB_M window of heights that holds the most of `h`: the lid's top
    (the largest level surface of the patch: a popped knob's top is a sixth of the lid's)."""
    hs = np.sort(h)
    n = np.searchsorted(hs, hs + TOP_SLAB_M, side="right") - np.arange(len(hs))
    return float(hs[int(np.argmax(n))])


def lid_band(model):
    """(lo, hi): the heights above the counter a lid top may stand at."""
    return float(model.dims[2]) - LID_BELOW_M, float(model.dims[2]) + LID_ABOVE_M


def _judge(model, pts, edge, h, sel, plane, i_frame):
    """The container's gates on one patch (`sel`: its points among `pts`, `h` their heights
    above `plane`): a Patch, its `why` empty when it passes."""
    import cv2

    d_lid = float(max(model.lid_dims[0], model.lid_dims[1]))
    d_min = float(min(model.lid_dims[0], model.lid_dims[1]))
    level = _densest_slab(h[sel])
    slab = sel & (h >= level) & (h <= level + TOP_SLAB_M)
    top = sel & (h >= level) & (h <= level + TOP_SLAB_M + KNOB_UP_M)  # the lid, with its knob if it is up
    n_slab = int(slab.sum())
    height = float(np.median(h[slab]))
    cut = bool(edge[top].any())
    mask, x0, y0, _ix, _iy = _grid(pts[top, :2], 4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_cells = int(mask.sum())
    got = {"frame": i_frame, "height_m": height, "n_slab": n_slab, "cut": cut}

    def patch(why, xy, **kw):
        tz = _plane_z(plane, xy)
        return Patch(xy=(float(xy[0]), float(xy[1])), top_z=tz + height, table_z=tz, why=why, **got, **kw)

    nan2 = (float("nan"), float("nan"))
    if n_cells < MIN_SLAB_CELLS:
        xy = np.median(pts[top, :2], axis=0)
        return patch("a top of %.0f cm2, under a tenth of the lid's" % (1e4 * n_cells * CELL_M**2), xy, **_nan(nan2))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)
    if len(outline) < 5:
        return patch("no outline to its top", np.median(pts[top, :2], axis=0), **_nan(nan2))
    (cu, cv_), (a1, a2), _ang = cv2.fitEllipse(outline)
    xy = (x0 + (cu + 0.5) * CELL_M, y0 + (cv_ + 0.5) * CELL_M)
    axes = (min(a1, a2) * CELL_M, max(a1, a2) * CELL_M)
    fill = n_cells / max(math.pi / 4.0 * a1 * a2, 1.0)
    # the core: what shows there at all must be the top (a cup shows its inside, lower)
    d_c = np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1])
    core = d_c < CORE_FRAC * d_lid
    n_core_top = int((core & top).sum())
    n_core_low = int((core & (h > STANDS_M) & (h < level)).sum())
    core_top = n_core_top / float(n_core_top + n_core_low) if n_core_top + n_core_low >= 10 else float("nan")
    above = int(((d_c < 0.5 * axes[1] + 2 * CELL_M) & (h > level + TOP_SLAB_M + ABOVE_M)).sum())
    kw = {"axes_m": axes, "fill": float(fill), "core_top": core_top, "above_frac": above / float(max(n_slab, 1))}
    if not model.round:
        from box_opening.perception.scene_source import fitted_yaw

        kw["yaw"] = fitted_yaw(pts[slab, :2] - np.asarray(xy), n_slab, model)
    size = "%.0fx%.0f mm" % (1e3 * axes[0], 1e3 * axes[1])
    if axes[1] > AXES_FRAC[1] * d_lid:
        return patch("its top %s, wider than the lid's %.0f" % (size, 1e3 * d_lid), xy, **kw)
    if fill < MIN_FILL:
        return patch("its top a ring, %.0f%% filled — an open cup's rim?" % (100 * fill), xy, **kw)
    if np.isfinite(core_top) and core_top < CORE_MIN_TOP:
        return patch("its middle %.0f%% lower than its top — open, not a lid" % (100 * (1 - core_top)), xy, **kw)
    if kw["above_frac"] > MAX_ABOVE_FRAC:
        return patch(
            "%d points stand over %.0f mm above its top: taller than the container" % (above, 1e3 * ABOVE_M), xy, **kw
        )
    if not counter_around(pts, xy, plane):
        return patch("no counter shows around it", xy, **kw)
    if axes[0] < AXES_FRAC[0] * d_min:
        # a top this narrow at the edge of the view may be the lid, cut: a candidate to look at again
        why = "cut off by the view" if cut else "narrower than the lid's %.0f" % (1e3 * d_min)
        return patch("its top %s, %s" % (size, why), xy, **kw)
    return patch("", xy, **kw)


def _nan(nan2):
    return {"axes_m": nan2, "fill": float("nan"), "core_top": float("nan"), "above_frac": float("nan")}


def counter_around(pts, xy, plane):
    """Does the counter `plane` show in the ring around xy (RING_M)?"""
    on = plane[2]
    d = np.hypot(pts[on, 0] - xy[0], pts[on, 1] - xy[1])
    return int(((d > RING_M[0]) & (d < RING_M[1])).sum()) >= MIN_RING_POINTS


def patches_on(model, pts, edge, plane, i_frame):
    """Every patch standing in the lid band on `plane` (connected on the CELL_M grid), judged."""
    import cv2

    n, c, _on = plane
    h = (pts - c) @ n
    lo, hi = lid_band(model)
    band = (h >= lo) & (h <= hi)
    if band.sum() < MIN_SLAB_CELLS:
        return []
    occ, _x0, _y0, ix, iy = _grid(pts[band, :2], 2)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    idx = np.flatnonzero(band)
    point_lab = lab[iy, ix]
    out = []
    for k in range(1, n_lab):
        sel = np.zeros(len(pts), bool)
        sel[idx[point_lab == k]] = True
        if stats[k, cv2.CC_STAT_AREA] >= MIN_SLAB_CELLS and sel.sum() >= MIN_SLAB_CELLS:
            out.append(_judge(model, pts, edge, h, sel, plane, i_frame))
    return out


def _incidence_deg(p, cam):
    v = np.asarray(cam, float) - np.asarray(p, float)
    return float(np.degrees(np.arccos(np.clip(v[2] / max(np.linalg.norm(v), 1e-9), -1.0, 1.0))))


def check_button(frame, model, patch):
    """The aim's own circle detector on a passing patch: (state, button xyz or None, why).
    state "seen": a button circle within BUTTON_NEAR_M of its centre; "none": the view
    allowed the check and held none; "unchecked": the view does not allow it."""
    from box_opening.perception import button_aim as ba

    cam = np.asarray(frame.trans_cam, float)
    p = np.array([patch.xy[0], patch.xy[1], patch.top_z])
    rng = float(np.linalg.norm(p - cam))
    inc = _incidence_deg(p, cam)
    rot = np.asarray(frame.rot_cam, float)
    pc = rot.T @ (p - cam)
    kk = np.asarray(frame.k, float)
    h, w = np.asarray(frame.depth).shape[:2]
    if frame.color is None:
        return "unchecked", None, "no colour image"
    if pc[2] <= 0:
        return "unchecked", None, "behind the camera"
    u, v = kk[0, 0] * pc[0] / pc[2] + kk[0, 2], kk[1, 1] * pc[1] / pc[2] + kk[1, 2]
    if not (CHECK_MARGIN_PX <= u <= w - 1 - CHECK_MARGIN_PX and CHECK_MARGIN_PX <= v <= h - 1 - CHECK_MARGIN_PX):
        return "unchecked", None, "its centre at the frame's edge"
    if inc > CHECK_MAX_INCIDENCE_DEG:
        return "unchecked", None, "its top seen %.0f deg off straight down" % inc
    if not (CHECK_MIN_RANGE_M <= rng <= CHECK_MAX_RANGE_M):
        return "unchecked", None, "%.0f mm away" % (1e3 * rng)
    # the top-down path puts the lid at table_z + dims.z: give it the measured top that way
    oblique = ba.view_tilt_deg(rot) > ba.OBLIQUE_MIN_DEG
    table_z = patch.table_z if oblique else patch.top_z - float(model.dims[2])
    s, why = ba.find_button(
        frame.color,
        np.asarray(frame.depth, np.float32),
        kk,
        rot,
        cam,
        table_z,
        model,
        dist=frame.dist,
        lid_z=patch.top_z,
        prior_xy=patch.xy,
        prior_max_m=BUTTON_NEAR_M,
    )
    if s is None:
        return "none", None, why
    miss = math.hypot(s.xyz[0] - patch.xy[0], s.xyz[1] - patch.xy[1])
    if miss > BUTTON_NEAR_M:
        return "none", None, "the nearest circle %.0f mm from the top's centre" % (1e3 * miss)
    return "seen", tuple(float(v) for v in s.xyz), ""


# A lid on a plain table reads WIDER than it is in the wrist's depth: the stereo has nothing to
# match on the table beside it and carries the lid's depth out over it (flatbot 2026-11-10 17:45,
# a white folding table: the 105 mm lid a solid 116 x 184 mm top, 92 % filled, stretched away
# from the lens; with the depth's steps left in, 190 x 225). The colour image has its rim where
# it is. A top refused only for being too LONG — its short side still a lid's — is looked at
# again in colour, laid flat on the plane at its own height (RIM_PX_M a pixel): one circle of
# the lid's radius there, inside the top the depth showed, is the lid, and its centre the place.
RIM_PX_M = 0.002
RIM_HALF_M = 0.20  # the square laid flat reaches this far from the depth top's centre
RIM_R_FRAC = (0.85, 1.15)  # the circle's radius, of the lid's
RIM_SHORT_FRAC = 2.0  # the depth top no more than this many lids across its short side: a smeared lid, not a tray
RIM_HOUGH = (120.0, 22.0)  # cv2.HoughCircles param1 (the edge), param2 (the votes)
# Only where the view looks down on the lid. Seen from far off to its side (the wrist turned where
# the arm rests, 60-65 deg off straight down, 0.6 m away) the colour laid flat stretches 2 mm of
# the lid's height into 4 mm along the view, and the circle found there stood 36-54 mm short of
# the button the look pose then saw (flatbot 2026-11-11 19:13 and 19:29) — where the depth top's
# own centre, long as it read (72 x 158 mm), stood 9 and 6 mm from it: from the side the stereo
# carries the lid's depth out both ways along the view, and its middle stays the lid's. So past
# RIM_MAX_INCIDENCE_DEG a top too long by no more than SIDE_LONG_FRAC lids, its short side a
# lid's or less, is the lid where its centre is.
RIM_MAX_INCIDENCE_DEG = 50.0
SIDE_LONG_FRAC = 1.6


def rim_circle(frame, plane, patch, model):
    """The lid's rim in `frame`'s colour image, for the too-long depth top `patch` on `plane`:
    the (x, y) of the one circle of the lid's radius on the plane at the top's height whose
    centre lies inside the top, or None (no colour image, no such circle)."""
    import cv2

    if frame.color is None:
        return None
    d_lid = float(max(model.lid_dims[0], model.lid_dims[1]))
    n = int(round(2 * RIM_HALF_M / RIM_PX_M))
    g = (np.arange(n) + 0.5) * RIM_PX_M - RIM_HALF_M
    X, Y = np.meshgrid(patch.xy[0] + g, patch.xy[1] + g)
    nrm, c, _on = plane
    # the plane lifted to the top's height, z at each (x, y)
    Z = c[2] - (nrm[0] * (X - c[0]) + nrm[1] * (Y - c[1])) / nrm[2] + patch.height_m / abs(nrm[2])
    P = np.stack([X, Y, Z], -1) - np.asarray(frame.trans_cam, float)
    pc = P @ np.asarray(frame.rot_cam, float)
    kk = np.asarray(frame.k, float)
    zc = np.where(pc[..., 2] > 1e-3, pc[..., 2], np.nan)
    u = (kk[0, 0] * pc[..., 0] / zc + kk[0, 2]).astype(np.float32)
    v = (kk[1, 1] * pc[..., 1] / zc + kk[1, 2]).astype(np.float32)
    flat = cv2.remap(np.asarray(frame.color), np.nan_to_num(u, nan=-1.0), np.nan_to_num(v, nan=-1.0), cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY) if flat.ndim == 3 else flat, (0, 0), 1.5)
    r_px = 0.5 * d_lid / RIM_PX_M
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=0.5 * r_px,
        param1=RIM_HOUGH[0],
        param2=RIM_HOUGH[1],
        minRadius=int(RIM_R_FRAC[0] * r_px),
        maxRadius=int(math.ceil(RIM_R_FRAC[1] * r_px)),
    )
    if circles is None:
        return None
    inside = []
    for cu, cv_, _r in circles[0]:
        xy = (patch.xy[0] + (cu + 0.5) * RIM_PX_M - RIM_HALF_M, patch.xy[1] + (cv_ + 0.5) * RIM_PX_M - RIM_HALF_M)
        if math.hypot(xy[0] - patch.xy[0], xy[1] - patch.xy[1]) <= 0.5 * patch.axes_m[1]:
            inside.append(xy)
    if not inside:
        return None
    # HoughCircles gives the strongest first; a second inside the top, more than a lid's radius off, is two places
    if any(math.hypot(q[0] - inside[0][0], q[1] - inside[0][1]) > 0.5 * d_lid for q in inside[1:]):
        return None
    return float(inside[0][0]), float(inside[0][1])


def by_its_rim(frame, plane, patch, model):
    """`patch` as the colour image's rim has it (rim_circle), where the depth refused it only as
    too long: the same top, placed at the circle's centre and passing; else `patch` as it was."""
    d_lid = float(max(model.lid_dims[0], model.lid_dims[1]))
    if "wider than the lid's" not in patch.why or not np.isfinite(patch.axes_m[0]):
        return patch
    if patch.axes_m[0] > RIM_SHORT_FRAC * d_lid or patch.cut:
        return patch
    p = np.array([patch.xy[0], patch.xy[1], patch.top_z])
    if _incidence_deg(p, frame.trans_cam) > RIM_MAX_INCIDENCE_DEG:
        side = patch.axes_m[1] <= SIDE_LONG_FRAC * d_lid and patch.axes_m[0] <= AXES_FRAC[1] * d_lid
        return replace(patch, why="") if side else patch
    xy = rim_circle(frame, plane, patch, model)
    if xy is None:
        return patch
    tz = _plane_z(plane, xy)
    return replace(patch, xy=xy, table_z=tz, top_z=tz + patch.height_m, why="", rim=True)


def _boxed(frame, patch):
    """Is the patch's centre inside one of its frame's detector boxes? None without boxes."""
    if frame.boxes is None:
        return None
    cam = np.asarray(frame.trans_cam, float)
    pc = np.asarray(frame.rot_cam, float).T @ (np.array([patch.xy[0], patch.xy[1], patch.top_z]) - cam)
    if pc[2] <= 0:
        return False
    kk = np.asarray(frame.k, float)
    u, v = kk[0, 0] * pc[0] / pc[2] + kk[0, 2], kk[1, 1] * pc[1] / pc[2] + kk[1, 2]
    return any(b[1][0] <= u <= b[1][2] and b[1][1] <= v <= b[1][3] for b in frame.boxes)


def frame_patches(frame, model, i_frame=0):
    """(the patches judged in one frame, its counter plane or None, its points). Every
    counter-sized level plane is tried, largest first; the frame's counter is the first a
    patch passes on, else the largest."""
    pts, _uv, edge = lift(frame)
    if len(pts) < MIN_PLANE_POINTS:
        return [], None, pts
    planes = level_planes(pts)
    out, counter = [], None
    for plane in planes:
        got = [by_its_rim(frame, plane, q, model) for q in patches_on(model, pts, edge, plane, i_frame)]
        if counter is None and any(not p.why for p in got):
            counter = plane
        out.extend(got)
    return out, (counter if counter is not None else (planes[0] if planes else None)), pts


def _pool(passing, d_lid):
    """The passing patches grouped by place (greedy, strongest first): [[patch, ...], ...]."""
    groups = []
    for p in sorted(passing, key=lambda p: -p.n_slab):
        for g in groups:
            if math.hypot(p.xy[0] - g[0].xy[0], p.xy[1] - g[0].xy[1]) < SAME_PLACE_FRAC * d_lid:
                g.append(p)
                break
        else:
            groups.append([p])
    return groups


def _place(group):
    whole = [p for p in group if not p.cut] or group
    seen = [p.button_xyz for p in group if p.button == "seen"]
    state = "seen" if seen else ("none" if any(p.button == "none" for p in group) else "unchecked")
    return Place(
        xy=tuple(float(v) for v in np.median([p.xy for p in whole], axis=0)),
        top_z=float(np.median([p.top_z for p in whole])),
        table_z=float(np.median([p.table_z for p in whole])),
        patches=tuple(group),
        button=state,
        button_xyz=tuple(float(v) for v in np.median(seen, axis=0)) if seen else None,
    )


def _xy(p):
    return "[%.3f, %.3f]" % (p.xy[0], p.xy[1])


# Two places pass and another camera's detector has the container at one of them (`seen_at`): that one is
# taken where it stands within this many lid widths of the detector's place and every other place at
# least a lid's width further from it. Flatbot 2026-11-12 18:41, a table with tape rolls, a bowl and a
# bottle on it: the wrist's geometry passed 2-8 lid-sized tops in each of its 4 views ("refusing to
# guess", four times), the box among them at [0.755, -0.219], 33 mm from where the scene camera's
# segmenter had it and the next place 0.37 m from it; the run then went by the scene camera's own
# place for it, 87 mm high, and no reach planned. The same afternoon the two cameras put one box 99 mm
# apart (18:42).
NEAR_LIDS = 1.5


def _nearest_alone(live, seen_at, d_lid):
    """The one of the places `live` another detector's (x, y) `seen_at` names (NEAR_LIDS), or None."""
    by = sorted(live, key=lambda q: math.hypot(q.xy[0] - seen_at[0], q.xy[1] - seen_at[1]))
    d = [math.hypot(q.xy[0] - seen_at[0], q.xy[1] - seen_at[1]) for q in by[:2]]
    if d[0] <= NEAR_LIDS * d_lid and (len(d) < 2 or d[1] >= d[0] + d_lid):
        return by[0]
    return None


def find_container(frames, model, check_buttons=True, reachable=None, seen_at=None):
    """The container in wrist frames (WristFrame, one or more), as a WristFind: its `fix` a
    scene_source.SceneFix with source "wrist" — or None, and `why`, when no counter shows,
    nothing on it is the container, the one lid-sized top shows no button where the view
    shows its top, no view could check its button (look again nearer above it), or two
    places pass (it refuses to guess). `check_buttons` False: geometry alone (no circle
    detector), a top no button was asked of a fix at 0.6-0.7. `reachable` (a lid top's (x, y, z) -> bool; None: every
    place counts): a place it calls out of reach is no candidate beside one it does not, and where
    none is within reach there is no fix (it says so). `seen_at` ((x, y) where another camera's detector has
    the container, or None): of several places that pass, the one it names alone (_nearest_alone) is
    taken, placed by the wrist's own points."""
    import time

    from box_opening.models.container import ContainerPose
    from box_opening.perception.scene_source import SceneFix, table_extent

    t0 = time.monotonic()
    frames = list(frames)
    d_lid = float(max(model.lid_dims[0], model.lid_dims[1]))
    patches, counters, n_depth = [], {}, 0
    for i, f in enumerate(frames):
        got, counter, pts = frame_patches(f, model, i)
        n_depth += len(pts) >= MIN_PLANE_POINTS
        if counter is not None:
            counters[i] = (counter, pts)
        for p in got:
            if not p.why:
                if check_buttons:
                    state, xyz, why = check_button(f, model, p)
                    p = replace(p, button=state, button_xyz=xyz, button_why=why)
                p = replace(p, boxed=_boxed(f, p))
            patches.append(p)
    timing = {"judge_s": round(time.monotonic() - t0, 3)}
    passing = [p for p in patches if not p.why]
    places = tuple(_place(g) for g in _pool(passing, d_lid))
    live = [q for q in places if q.button != "none"]

    def refuse(why):
        return WristFind(
            None, "wrist camera: %s" % why, 0.0, None, (), places, tuple(patches), len(frames), len(counters), timing
        )

    if not n_depth:
        return refuse("too little depth in %d frame%s" % _n(len(frames)))
    if not counters:
        return refuse("no counter in %d frame%s (no level plane wider than a container)" % _n(len(frames)))
    if not passing:
        cut = [p for p in patches if p.why.endswith("cut off by the view")]
        if cut:
            return refuse(
                "the container may stand at %s, cut off by the view (%s): look again with it whole"
                % (_xy(cut[0]), cut[0].why)
            )
        whys = list(dict.fromkeys(p.why for p in patches))
        return refuse(
            "nothing on the counter is the container (%d thing%s at lid height%s)"
            % (*_n(len(patches)), (": " + "; ".join(whys[:3])) if whys else "")
        )
    if not live:
        q = places[0]
        why = next((p.button_why for p in q.patches if p.button == "none"), "")
        return refuse("a lid-sized top at %s, but no button circle where the view shows it (%s)" % (_xy(q), why))
    if reachable is not None and live:
        # What the arm cannot reach is not a container this run could open, so it is no rival of one it can
        # (2026-11-12, a wooden table with a bowl and two tape rolls 1.0-1.2 m out behind the box at 0.81 m:
        # "two places pass", twice, and nothing opened). `reachable`: the run's own word on a lid top's place
        # (reach_press.out_of_reach), the arm's kinematic bound and no number of the scene's.
        near = [q for q in live if reachable((q.xy[0], q.xy[1], q.top_z))]
        if not near:
            return refuse(
                "the lid-sized top%s at %s stand%s out of the arm's reach"
                % ("s" if len(live) > 1 else "", " and ".join(_xy(q) for q in live[:3]), "" if len(live) > 1 else "s")
            )
        live = near
    if len(live) > 1:
        boxed = [q for q in live if any(p.boxed for p in q.patches)]
        named = _nearest_alone(live, seen_at, d_lid) if seen_at is not None and len(boxed) != 1 else None
        if len(boxed) != 1 and named is None:
            return refuse(
                "two places pass as this container, %s and %s (button %s, %s): refusing to guess"
                % (_xy(live[0]), _xy(live[1]), live[0].button, live[1].button)
            )
        # the wrist camera's detector boxed one of them alone, or another camera's names one alone:
        # semantics break the tie
        live = boxed if len(boxed) == 1 else [named]
    q = live[0]
    if check_buttons and q.button != "seen":
        # geometry alone: a closed tin of the lid's size passes every gate the button does not
        why = next((p.button_why for p in q.patches if p.button_why), "")
        return refuse(
            "a lid-sized top at %s, but no view could check its button (%s): look again nearer above it, "
            "within %.0f deg of straight down and %.2f m" % (_xy(q), why, CHECK_MAX_INCIDENCE_DEG, CHECK_MAX_RANGE_M)
        )
    frames_seen = sorted({p.frame for p in q.patches})
    if q.button == "seen":
        # frames, not patches: one frame judges one lid on every counter-sized plane it holds
        conf = 1.0 if len({p.frame for p in q.patches if p.button == "seen"}) >= 2 else 0.9
    else:
        conf = 0.7 if len(frames_seen) >= 2 else 0.6
    if all(p.cut for p in q.patches):
        conf -= 0.2
    # the button circle is where the press goes; the lid's outline centre where it shows none
    xy = q.button_xyz[:2] if q.button_xyz is not None else q.xy
    top = (float(xy[0]), float(xy[1]), q.top_z)
    # the counter's points around the box, per frame (each pose errs its own way: table_map);
    # pooled into the fix's one cloud, each frame's are set at their height above its own
    # counter on the place's (q.table_z): one plane, as room_map's single fit reads them —
    # two poses' counters 20-60 mm apart in z were two tables there, one "standing" on the other
    clouds, levelled = [], []
    for i in frames_seen:
        counter, pts = counters.get(i, (None, None))
        if counter is not None:
            near = np.hypot(pts[:, 0] - top[0], pts[:, 1] - top[1]) < MAP_R_M
            cloud = pts[near].astype(np.float32)
            clouds.append((cloud, _plane_z(counter, top[:2])))
            moved = cloud.copy()
            moved[:, 2] = q.table_z + (pts[near] - counter[1]) @ counter[0]
            levelled.append(moved)
    points = np.concatenate(levelled) if levelled else None
    extent = None if points is None else table_extent(points.astype(float), q.table_z, top[:2])
    best = max(q.patches, key=lambda p: p.n_slab)
    fix = SceneFix(
        pose=ContainerPose(xyz=(top[0], top[1], q.table_z), yaw=float(best.yaw)),
        top_xyz=top,
        n_points=int(best.n_slab),
        score=float(conf),
        table_z_scene=float(q.table_z),
        table_extent=extent,
        partial="cut off by the wrist camera's view" if all(p.cut for p in q.patches) else "",
        table_points=points,
        source="wrist",
        n_sightings=len(frames_seen),
        sources=("wrist",) * len(frames_seen),
    )
    timing["find_s"] = round(time.monotonic() - t0, 3)
    why = "the container at [%.3f, %.3f], its top %.0f mm above the counter, in %d of %d frame%s; button %s" % (
        top[0],
        top[1],
        1e3 * (q.top_z - q.table_z),
        len(frames_seen),
        *_n(len(frames)),
        q.button,
    )
    return WristFind(
        fix, why, conf, q.button_xyz, tuple(clouds), places, tuple(patches), len(frames), len(counters), timing
    )


def _n(k):
    return k, "" if k == 1 else "s"
