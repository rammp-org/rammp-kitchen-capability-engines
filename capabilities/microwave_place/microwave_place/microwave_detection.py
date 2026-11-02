"""Where the container goes in the open microwave, from one scene-camera frame and its cavity mask.

Entry points:
    estimate_interior(depth_m, K, world_T_cam, objects_3d, cavity_mask) -> dict, or None when not detected
    fit_interior(depth_m, K, world_T_cam, cavity_mask) -> dict; raises NotDetected with the reason
        one frame; pure numpy/opencv
    locate(engine, ...) -> dict
        the live scene camera inside a running action: SAM3's cavity mask on each frame, `frames`
        detected frames that agree (of up to `max_frames`), their median, and the frame nearest it;
        raises NotLocated otherwise
    SceneCamera(engine, ...).grab(prompt, deadline), scene_frame(camera, prompt, timeout_s)
        one live frame: SAM3's mask for a prompt, the depth frame of its stamp, the camera pose at it
    cavity_mask_from_detect(response) -> HxW bool or None
        the cavity mask from a rammp_perception_interfaces/srv/Detect response

This is for one microwave, whose dimensions are known (the constants below). It is measured from the
frame face around its opening: the flat face, in one vertical plane, where the closed door's seal
rests. The only other assumption is that the oven stands upright.

Oven frame (horizontal axes in the gravity-aligned world xy plane):
    n : horizontal, into the oven (= insertion axis)
    w : z x n, horizontal, the oven's LEFT as seen from the front
    A : coordinate along w,  S : coordinate along n,  Z : world z

Steps
  1. The opening's edges in the image. The cavity mask (SAM3 "inside of microwave") is everything
     seen through the opening, so its outline is the opening's inner edge, except where it runs
     along the image border or along something in front of the opening (the open door). A straight
     line is fitted to each side of the outline.
  2. The frame plane: a vertical plane fitted to the depth points in a thin ring just outside the
     mask whose surfaces face the camera and stand vertical. Its normal is the insertion axis, its
     offset the opening's in/out position. It needs enough inliers, a low rms, and inliers on at
     least two of the four strips around the opening.
  3. Each side of the outline cast onto that plane: a vertical side gives A of the opening's left or
     right edge, a horizontal side Z of its top or bottom edge. A side is used only when it is
     straight in the image, clear of the image border, not in front of something nearer than the
     frame (an occluder), and vertical or horizontal on the plane. At least one vertical and one
     horizontal edge are needed; where both edges of a pair are seen, their gap must be the
     opening's size.
  4. The opening rectangle from those edges and its known size. It must contain the whole cast
     mask, and what is seen through it must lie within the oven's depth behind the plane.
  5. The point: centred sideways, half-way from the frame plane to the back wall, on the floor
     (the turntable glass).
Any requirement that fails: not detected. Nothing is guessed.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as _Rot

# ---------------------------------------------------------------------------------------------
# This microwave: Comfee retro CMO-MMP02AA (0.9 cu ft). Metres, tape-measured by hand (2026-11-02).
# ---------------------------------------------------------------------------------------------
OPENING = (0.3048, 0.1905)  # opening width, height in the frame plane [12 x 7.5 in]
RECESS = 0.0286  # control panel face -> frame plane around the opening
DEPTH = 0.3048  # frame plane -> back wall [12 in]; the side walls are flush with the opening's edges
FLOOR = 0.0159  # floor (turntable glass top) above the opening's bottom edge [5/8 in]
HEIGHT = 0.1746  # floor -> ceiling (with FLOOR: the ceiling is flush with the opening's top)

# 1. the outline's sides
SIDE_MIN_PX = 20  # a side of the outline shorter than this is not measured
SIDE_RMS_PX = 1.5  # ... nor one that is not straight to this (rms px off its line; no lens distortion)
BORDER_PX = (
    12  # a side running this close to the image border is the border, not an edge (SAM3 stops a few px short of it)
)
OCCLUDER_BAND_PX = (2, 8)  # just outside a side: what is seen there ...
OCCLUDER_FRONT = 0.015  # ... is an occluder where it is this far in front of the frame plane ...
OCCLUDER_SHARE = 0.3  # ... in this share of the band's depth points (the frame face itself is on the plane)
# 2. the frame plane
RING = (3, 0.04)  # ring outside the mask: from this many px (the frame's stereo depth spreads ~1 cm into
# the opening; the ring needs only the frame itself) to this far (m) out
VERTICAL_NZ = 0.25  # a ring point's normal within ~15 deg of horizontal (a vertical surface) ...
FACING = 0.2  # ... and facing the camera (|cos| to its line of sight at least this)
NORMAL_AGREE_DEG = 20.0  # an inlier's normal within this of the plane's
PLANE_TOL = 0.006  # a ring point this close to the plane is an inlier
PLANE_MIN = 80  # inliers needed ...
PLANE_RMS = 0.004  # ... with at most this rms ...
STRIP_MIN = 20  # ... and at least this many on each of at least two strips
CORNER = 0.1  # the strips are beside the middle of each side: this share of its length off each end is left out
# 3-4. edges and the rectangle
SIDE_UPRIGHT_DEG = 30.0  # an outline side within this of world-up in the image is vertical, within this of
# square to it horizontal; in between it is neither and is not used
RECT_RMS_PX = 2.0  # the upright rectangle must fit the sides it uses to this (px rms)
RING_MIN = 50  # with at least this many depth points on the frame strips ...
RING_TOL = 0.015  # ... their median must be within this of the final (depth-refined) plane ...
RING_TOL_COARSE = 0.03  # ... and within this of the plane fitted to the outline alone (the mask's edge can sit a
# little outside the opening, which places the known-size rectangle a little too near)
RING_SAMPLE = 300  # frame depth points used in the refinement ...
RING_WEIGHT = 200.0  # ... each weighted so 1 cm off the plane counts as 2 px off an edge
SAME_POSE_M, SAME_POSE_DEG = 0.02, 3.0  # passing poses further apart than this: two answers, not detected
UPRIGHT_DEG = 5.0  # a cast side within this of vertical / horizontal on the plane
EDGE_TOL = 0.015  # the gap between opposite edges vs OPENING; also how far the cast mask may stick out of
# the rectangle. A mask edge is good to ~2 px (~3 mm at 0.7 m with fx 284) and the plane's offset to ~5 mm,
# which seen at 60 deg off-axis moves a cast edge ~9 mm sideways: 1.5 cm covers both, and stays well under
# the 2.9 cm control-panel step that a wrong edge or plane would show.
BEHIND = (0.01, DEPTH + 0.03)  # what is seen through the opening lies this far behind the plane ...
BEHIND_EDGE = 0.015  # m: what is seen through the opening is taken this far in from the mask's edge
BEHIND_SHARE = 0.9  # ... in this share of the mask's depth points (eroded by RING[0]) ...
BEHIND_MIN = 50  # ... of which at least this many are needed

_SIDES = ("left", "right", "top", "bottom")


class NotDetected(Exception):
    """This frame does not show the interior; the message says which requirement failed."""


def _backproject(depth, K):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth.astype(np.float32)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)


def _normals(Pw, valid, k=3, max_jump=0.02):
    """Organized-cloud normals by central differences over k px.
    Returns unit normals (sign arbitrary), ok mask."""
    n = np.zeros_like(Pw)
    ok = np.zeros(valid.shape, bool)
    dx = Pw[k:-k, 2 * k :] - Pw[k:-k, : -2 * k]
    dy = Pw[2 * k :, k:-k] - Pw[: -2 * k, k:-k]
    cr = np.cross(dx, dy)
    nn = np.linalg.norm(cr, axis=-1, keepdims=True)
    v = (
        valid[k:-k, 2 * k :]
        & valid[k:-k, : -2 * k]
        & valid[2 * k :, k:-k]
        & valid[: -2 * k, k:-k]
        & valid[k:-k, k:-k]
        & (np.linalg.norm(dx, axis=-1) < max_jump * 2 * k)
        & (np.linalg.norm(dy, axis=-1) < max_jump * 2 * k)
        & (nn[..., 0] > 1e-12)
    )
    n[k:-k, k:-k] = cr / np.maximum(nn, 1e-12)
    ok[k:-k, k:-k] = v
    return n, ok


# ---------------------------------------------------------------------------------------------
# 1. The outline's sides
# ---------------------------------------------------------------------------------------------


def _clean(mask):
    """The mask's largest connected region with its holes filled, and that region's outline (N,2 px)."""
    m = np.asarray(mask, bool).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None, None
    c = max(cs, key=cv2.contourArea)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c], -1, 1, -1)
    return filled > 0, c[:, 0, :].astype(float)


def _sides(contour, shape):
    """The outline's straight sides: dict(p0, p1 (px endpoints on the fitted line), run (its outline
    px), rms, why) with why None for a side that may be an edge, else the reason it is not."""
    H, W = shape
    perim = len(contour)
    poly = cv2.approxPolyDP(contour.astype(np.int32)[:, None, :], max(3.0, 0.02 * perim), True)[:, 0, :]
    key = {(int(x), int(y)): i for i, (x, y) in enumerate(contour.astype(int))}
    idx = sorted(key[(int(x), int(y))] for x, y in poly)
    out = []
    for i0, i1 in zip(idx, idx[1:] + [idx[0] + perim], strict=True):
        run = contour[np.arange(i0, i1 + 1) % perim]
        if len(run) < SIDE_MIN_PX:
            continue
        trim = max(2, len(run) // 8)  # the corners round off
        core = run[trim:-trim]
        mid = core.mean(0)
        d = np.linalg.svd(core - mid)[2][0]
        rms = float(np.sqrt(np.mean(((core - mid) @ np.array([-d[1], d[0]])) ** 2)))
        t = (run[[0, -1]] - mid) @ d
        why = None
        if rms > SIDE_RMS_PX:
            why = f"not straight ({rms:.1f} px rms)"
        on_border = (
            (core[:, 0] <= BORDER_PX)
            | (core[:, 1] <= BORDER_PX)
            | (core[:, 0] >= W - 1 - BORDER_PX)
            | (core[:, 1] >= H - 1 - BORDER_PX)
        )
        if on_border.mean() > 0.3:
            why = "image border"
        out.append(dict(p0=mid + t[0] * d, p1=mid + t[1] * d, run=run, rms=rms, why=why))
    return out


def _seg_dist(px, a, b, beside=None):
    """Distance (px) of points px (N,2) to the segment a-b; with beside=f, inf for points not beside its
    middle (whose foot on its line falls within f of its length from either end)."""
    ab = b - a
    t = ((px - a) @ ab) / max(float(ab @ ab), 1e-9)
    d = np.linalg.norm(px - (a + np.clip(t, 0.0, 1.0)[:, None] * ab), axis=1)
    return d if beside is None else np.where((t < beside) | (t > 1 - beside), np.inf, d)


# ---------------------------------------------------------------------------------------------
# 2. The frame plane
# ---------------------------------------------------------------------------------------------


def _second_strip(inl, strip):
    """Inliers on the second-best strip: the frame face is the plane around the opening on two strips at
    least, where a plane beside only one side of it (the open door's edge, the control panel) is not."""
    per = np.sort(np.bincount(strip[inl]))
    return int(per[-2]) if len(per) > 1 else 0


def _vertical_plane(P, N, strip, seed=0, iters=400):
    """A vertical plane (a line in the world xy plane) through ring points P (M,3) with normals N, each
    on the strip beside outline side `strip` (M,): by RANSAC over the planes with inliers on two strips,
    then least squares on the inliers. The plane with the most inliers on its second-best strip wins.
    (unit horizontal normal (3,), offset, inlier mask) or None."""
    if len(P) < PLANE_MIN:
        return None
    xy = P[:, :2]
    nxy = N[:, :2] / np.maximum(np.linalg.norm(N[:, :2], axis=1, keepdims=True), 1e-9)
    cos_agree = np.cos(np.radians(NORMAL_AGREE_DEG))

    def inliers(nrm, off):
        return (np.abs(xy @ nrm - off) < PLANE_TOL) & (np.abs(nxy @ nrm) > cos_agree)

    rng = np.random.default_rng(seed)
    best, best_score = None, (STRIP_MIN - 1, 0)
    for _ in range(iters):
        i, j = rng.choice(len(xy), 2, replace=False)
        d = xy[j] - xy[i]
        if np.linalg.norm(d) < 0.02:
            continue
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        inl = inliers(nrm, nrm @ xy[i])
        score = (_second_strip(inl, strip), int(inl.sum()))
        if score > best_score:
            best, best_score = inl, score
    if best is None:
        return None
    inl = best
    for _ in range(3):
        mid = xy[inl].mean(0)
        d = np.linalg.svd(xy[inl] - mid)[2][0]
        nrm = np.array([-d[1], d[0]])
        off = float(nrm @ mid)
        inl = inliers(nrm, off)
        if _second_strip(inl, strip) < STRIP_MIN:
            return None
    return np.array([nrm[0], nrm[1], 0.0]), off, inl


# ---------------------------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------------------------


def _upright_rect(yaw, cx, cy, cz):
    """The opening in world for a pose: (corners {bl, br, tl, tr}, insertion axis n, the oven's left w);
    (cx, cy, cz) is the middle of its bottom edge, n points into the oven."""
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    w = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    b = np.array([cx, cy, cz])
    up = np.array([0.0, 0.0, OPENING[1]])
    half = w * OPENING[0] / 2
    return dict(bl=b + half, br=b - half, tl=b + half + up, tr=b - half + up), n, w


def fit_interior(depth_m, K, world_T_cam, cavity_mask, door_mask=None):
    """depth_m: (H,W) metres aligned to the color image, 0 = invalid. K: 3x3. world_T_cam: 4x4
    camera-optical -> world (z up). cavity_mask: HxW bool, SAM3 "inside of microwave" on the same
    image. door_mask: optional HxW bool, the open door (a side along it is not an edge).

    The opening is a rectangle of known size (OPENING) standing upright in the frame plane. Its four
    unknowns (where its bottom edge's middle is, and its turn about vertical) are fitted to the straight
    sides of the cavity mask's outline in the image, by perspective: one vertical and one horizontal side
    are enough. Depth only checks the result: the frame strips must lie on the fitted plane, and what is
    seen through the opening must lie inside the oven's depth behind it (this also tells which way the
    oven faces). Two different poses that both pass mean "not detected".

    Returns, in world: point (where the container goes), insertion_axis, lateral_axis_left,
    opening_corners (bottom-right, bottom-left, top-left, top-right in the frame plane, as seen from
    the front), floor_z, pins, fit (rms px, ring/behind checks), edges (each side of the outline: side,
    px, world segment, used or why not). Raises NotDetected with the reason."""
    if cavity_mask is None:
        raise NotDetected("no cavity mask")
    K = np.asarray(K, float).reshape(3, 3)
    T = np.asarray(world_T_cam, float)
    R, cam = T[:3, :3], T[:3, 3]
    cTw = np.linalg.inv(T)
    depth = np.asarray(depth_m, np.float32)
    H, W = depth.shape
    mask, contour = _clean(cavity_mask)
    if mask is None or mask.sum() < 400:
        raise NotDetected(f"cavity mask too small ({0 if mask is None else int(mask.sum())} px)")
    door = np.zeros((H, W), bool) if door_mask is None else np.asarray(door_mask, bool)
    door_grown = cv2.dilate(door.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 3.0)
    Pw = _backproject(np.where(valid, depth, 0), K) @ R.T.astype(np.float32) + cam.astype(np.float32)
    Nw, nok = _normals(Pw, valid)

    def proj(X):
        X = np.atleast_2d(np.asarray(X, float))
        pc = X @ cTw[:3, :3].T + cTw[:3, 3]
        uv = pc @ K.T
        return uv[:, :2] / np.maximum(uv[:, 2:], 1e-6)

    # 1. the outline's straight sides, each vertical or horizontal in the world (from where world-up
    #    points in the image at the side's middle)
    sides = _sides(contour, (H, W))
    out_px = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    bv, bu = np.nonzero((out_px >= OCCLUDER_BAND_PX[0]) & (out_px <= OCCLUDER_BAND_PX[1]))  # just outside the mask
    # what is seen through the opening: the mask without the band where the frame's stereo depth spreads into
    # it (~1 cm at the frame's distance)
    d_in = float(np.median(depth[mask & valid])) if (mask & valid).any() else 1.0
    in_px = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    inner = in_px > max(RING[0], BEHIND_EDGE * K[0, 0] / d_in)
    seen = Pw[inner & valid].astype(float)
    if len(seen) < BEHIND_MIN:
        raise NotDetected(f"{len(seen)} depth points seen through the opening (need {BEHIND_MIN})")
    edges = []
    for s in sides:
        e = dict(px=[s["p0"].tolist(), s["p1"].tolist()], side=None, why=s["why"], world=None, s=s)
        edges.append(e)
        if e["why"]:
            continue
        mid = 0.5 * (s["p0"] + s["p1"])
        ray = R @ (np.linalg.inv(K) @ np.r_[mid, 1.0])
        at = cam + ray * float(np.median(seen @ (R @ np.array([0, 0, 1.0])) - cam @ (R @ np.array([0, 0, 1.0]))))
        upd = proj(at + [0, 0, 0.05])[0] - proj(at)[0]
        upd /= np.linalg.norm(upd)
        dd = (s["p1"] - s["p0"]) / np.linalg.norm(s["p1"] - s["p0"])
        cos_up = abs(float(dd @ upd))
        e["up"], e["mid"] = upd, mid
        if cos_up > np.cos(np.radians(SIDE_UPRIGHT_DEG)):
            e["kind"] = "vertical"
        elif cos_up < np.sin(np.radians(SIDE_UPRIGHT_DEG)):
            e["kind"] = "horizontal"
        else:
            e["why"] = f"neither vertical nor horizontal in the image ({np.degrees(np.arccos(cos_up)):.0f} deg from up)"
            continue
        mine = _seg_dist(np.c_[bu, bv].astype(float), s["p0"], s["p1"]) <= OCCLUDER_BAND_PX[1] + 1
        if mine.any() and door_grown[bv[mine], bu[mine]].mean() > 0.2:
            e["why"] = "along the door"
    usable = [e for e in edges if not e["why"]]
    vert = sorted((e for e in usable if e["kind"] == "vertical"), key=lambda e: e["s"]["p0"][0] + e["s"]["p1"][0])
    horz = sorted((e for e in usable if e["kind"] == "horizontal"), key=lambda e: e["s"]["p0"][1] + e["s"]["p1"][1])
    if not vert or not horz or len(vert) > 2 or len(horz) > 2:
        why = ", ".join(f"{e.get('kind', '?')}: {e['why']}" for e in edges if e["why"]) or "-"
        raise NotDetected(
            f"need 1-2 vertical and 1-2 horizontal sides; have {len(vert)} and {len(horz)} (not used: {why})"
        )

    # 2. fit the upright rectangle to those sides: every labelling the image allows, from several turns
    # Facing the oven, its left is on the image left. A lone side is named by where the cavity is: below a
    # top edge, above a bottom edge, right of a left edge, left of a right edge.
    ys, xs = np.nonzero(mask)
    cen = np.array([xs.mean(), ys.mean()])

    def lone(e, vertical):
        if vertical:
            right = np.array([-e["up"][1], e["up"][0]])  # in the image, square to world-up
            right = right if right[0] >= 0 else -right
            return ("left",) if (cen - e["mid"]) @ right > 0 else ("right",)
        return ("top",) if (cen - e["mid"]) @ e["up"] < 0 else ("bottom",)

    def labellings():
        vl = [("left", "right")] if len(vert) == 2 else [lone(vert[0], True)]
        hl = [("top", "bottom")] if len(horz) == 2 else [lone(horz[0], False)]
        for a in vl:
            for b in hl:
                yield list(zip(vert, a, strict=True)) + list(zip(horz, b, strict=True))

    KEYS = dict(left=("bl", "tl"), right=("br", "tr"), top=("tl", "tr"), bottom=("bl", "br"))

    def resid(p, lab):
        C = _upright_rect(*p)[0]
        P = {k: proj(v)[0] for k, v in C.items()}
        out = []
        for e, side in lab:
            run = e["s"]["run"]
            core = run[len(run) // 8 : len(run) - len(run) // 8 : 2]
            a, b = P[KEYS[side][0]], P[KEYS[side][1]]
            nrm = np.array([-(b - a)[1], (b - a)[0]])
            out.append((core - a) @ (nrm / max(np.linalg.norm(nrm), 1e-9)))
        return np.concatenate(out)

    centre = np.median(seen, axis=0)
    toward = centre[:2] - cam[:2]
    sol = []
    for lab in labellings():
        for yaw0 in np.radians(np.arange(0.0, 360.0, 20.0)):
            n0 = np.array([np.cos(yaw0), np.sin(yaw0)])
            if n0 @ toward <= 0:  # the camera must be in front of the opening
                continue
            p0 = np.r_[yaw0, centre[:2] - 0.12 * n0, centre[2] - OPENING[1] / 2]
            r = least_squares(resid, p0, args=(lab,), loss="soft_l1", f_scale=2.0, max_nfev=200)
            rms = float(np.sqrt(np.mean(r.fun**2)))
            if rms <= RECT_RMS_PX:
                sol.append((rms, r.x, lab))

    # 3. the depth checks, on every pose that fits the outline
    def check(p, ring_tol=RING_TOL_COARSE):
        C, n, w = _upright_rect(*p)
        b = np.asarray(p[1:])
        if n @ (b - cam) <= 0:
            return "the camera is behind the opening"
        behind = (seen - b) @ n
        share = float(np.mean((behind > BEHIND[0]) & (behind < BEHIND[1])))
        if share < BEHIND_SHARE:
            return f"only {share * 100:.0f}% of what is seen through it lies inside the oven's depth behind it"
        ring = valid & nok & (out_px >= RING[0]) & (out_px <= RING[0] + 15)
        Pr = Pw[ring].astype(float)
        a = (Pr - b) @ w
        z = Pr[:, 2] - b[2]
        on = (np.abs(a) < OPENING[0] / 2 + 0.04) & (z > -0.04) & (z < OPENING[1] + 0.04)
        if on.sum() >= RING_MIN:
            off = float(np.median((Pr[on] - b) @ n))
            if abs(off) > ring_tol:
                return f"the frame's depth is {off * 100:+.1f} cm off the fitted plane"
        return None

    passing = []
    for rms, p, lab in sorted(sol, key=lambda x: x[0]):
        why = check(p)
        if why is None:
            passing.append((rms, p, lab))
    if not passing:
        reasons = (
            sorted({check(p) for _, p, _ in sol} - {None})
            if sol
            else [f"no pose fits the sides within {RECT_RMS_PX} px"]
        )
        raise NotDetected("; ".join(reasons[:3]))
    rms, p, lab = passing[0]
    for rms2, p2, _ in passing[1:]:
        if np.linalg.norm(p2[1:] - p[1:]) > SAME_POSE_M or abs(
            (p2[0] - p[0] + np.pi) % (2 * np.pi) - np.pi
        ) > np.radians(SAME_POSE_DEG):
            raise NotDetected(
                f"two different poses fit ({rms:.1f} and {rms2:.1f} px rms, "
                f"{np.linalg.norm(p2[1:] - p[1:]) * 100:.0f} cm apart)"
            )

    # 4. refine with the frame's depth: the outline's edges fix the turn and the opening's shape; the frame
    #    strips' depth points (on the plane the fit found) fix its distance, which the edges alone pin only
    #    loosely when the view is steep
    C, n, w = _upright_rect(*p)
    b = np.asarray(p[1:])
    ring = valid & nok & (out_px >= RING[0]) & (out_px <= RING[0] + 15)
    Pr = Pw[ring].astype(float)
    a_r, z_r = (Pr - b) @ w, Pr[:, 2] - b[2]
    on = (np.abs(a_r) < OPENING[0] / 2 + 0.04) & (z_r > -0.04) & (z_r < OPENING[1] + 0.04)
    on &= np.abs((Pr - b) @ n) < RING_TOL_COARSE  # the frame points the outline-only fit is near to
    depth_used = int(on.sum())
    if depth_used >= RING_MIN:
        Pf = Pr[on][np.random.default_rng(0).permutation(depth_used)[:RING_SAMPLE]]

        def resid2(q):
            n_, b_ = _upright_rect(*q)[1], np.asarray(q[1:])
            return np.r_[resid(q, lab), RING_WEIGHT * ((Pf - b_) @ n_)]

        q = least_squares(resid2, p, loss="soft_l1", f_scale=2.0, max_nfev=200).x
        rms_q = float(np.sqrt(np.mean(resid(q, lab) ** 2)))
        if rms_q > RECT_RMS_PX:
            raise NotDetected(f"the outline and the frame's depth disagree (edges {rms_q:.1f} px rms after both)")
        why = check(q, RING_TOL)
        if why is not None:
            raise NotDetected(f"after adding the frame's depth: {why}")
        p, rms = q, rms_q
        C, n, w = _upright_rect(*p)
        b = np.asarray(p[1:])

    # 5. the point
    for e, side in lab:
        e["side"] = side
        e["world"] = [C[KEYS[side][0]].tolist(), C[KEYS[side][1]].tolist()]
    used = [side for _, side in lab]
    pins = dict(
        in_out_and_turn=[
            f"opening outline ({', '.join(used)} sides), {OPENING[0] * 100:.2f} x {OPENING[1] * 100:.2f} cm"
        ],
        sideways=[f"{k} edge" for k in ("left", "right") if k in used],
        height=[f"{k} edge" for k in ("bottom", "top") if k in used],
    )
    return dict(
        point=(b + n * DEPTH / 2 + np.array([0.0, 0.0, FLOOR])).tolist(),
        insertion_axis=n.tolist(),
        lateral_axis_left=w.tolist(),
        opening_corners=[C[k].tolist() for k in ("br", "bl", "tl", "tr")],
        floor_z=float(b[2] + FLOOR),
        pins=pins,
        fit=dict(rms_px=rms, poses_fitting=len(sol), poses_passing=len(passing), frame_depth_points=depth_used),
        edges=[{k: v for k, v in e.items() if k not in ("s", "kind", "up", "mid")} for e in edges],
    )


def estimate_interior(depth_m, K, world_T_cam, objects_3d=None, cavity_mask=None, door_mask=None):
    """fit_interior, or None when not detected (also when there is no cavity mask). objects_3d is not
    used (kept for the call signature)."""
    try:
        return fit_interior(depth_m, K, world_T_cam, cavity_mask, door_mask)
    except NotDetected:
        return None


# ---------------------------------------------------------------------------------------------
# Live: SAM3's cavity mask on the scene camera, several detected frames that agree (inside a running action)
# ---------------------------------------------------------------------------------------------

DEPTH_SCALE = {"16UC1": 0.001, "mono16": 0.001, "32FC1": 1.0}  # depth encoding -> metres per unit
SAM3_SERVICE = "/perception/sam3/detect"  # rammp_perception_interfaces/srv/Detect
SAM3_PROMPT = "inside of microwave"
SAM3_IMAGE_TOPIC = "/scene_camera/color/image_raw"
SAM3_THRESHOLD = 0.5  # SAM3 keeps objects scoring at least this (the cavity scored 0.82-0.95 in the recordings)
SAM3_WAIT_S = 2.0  # the SAM3 service must be up within this
PAIR_MAX_S = 0.01  # the depth frame paired with SAM3's color frame: stamps this close (the camera stamps both alike)
DEPTH_KEEP = 60  # depth frames kept for pairing (~2 s at 30 Hz; a SAM3 call takes a few hundred ms)


class NotLocated(Exception):
    """No agreed interior; the message says what was seen."""


def _stamp(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def cavity_mask_from_detect(response):
    """HxW bool: the highest-scoring object's mask in a rammp_perception_interfaces/srv/Detect response
    (one prompt, e.g. "inside of microwave"); None when SAM3 failed or found nothing. Raises ValueError
    when objects come without a usable label image."""
    if not response.success or not response.detections.objects:
        return None
    img = response.detections.label_image
    if img.encoding not in ("mono16", "16UC1") or not img.height:
        raise ValueError(f"SAM3 label image: encoding {img.encoding!r}, {img.width}x{img.height} (expected mono16)")
    labels = np.frombuffer(bytes(img.data), np.dtype(">u2" if img.is_bigendian else "<u2"))
    labels = labels.reshape(img.height, img.step // 2)[:, : img.width]
    top = max(response.detections.objects, key=lambda o: o.score)
    return labels == top.mask_id


class SceneCamera:
    """The live scene camera inside a running action of `engine`: SAM3 (sam3_service) masks on its color
    frames, each paired with the depth frame of the same stamp and with the camera pose in `frame` from TF
    at that stamp. topics: (depth, camera_info, ...). Its subscriptions and client end with the action.
    Raises NotLocated when the SAM3 service is not up."""

    def __init__(
        self,
        engine,
        frame,
        topics,
        sam3_service=SAM3_SERVICE,
        sam3_threshold=SAM3_THRESHOLD,
        sam3_image_topic=SAM3_IMAGE_TOPIC,
    ):
        import threading
        from collections import deque

        from cv_bridge import CvBridge
        from rammp_perception_interfaces.srv import Detect
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.frame, self._Detect = frame, Detect
        self._service, self._threshold, self._image_topic = sam3_service, sam3_threshold, sam3_image_topic
        self._depths, self._latest, self._lock = deque(maxlen=DEPTH_KEEP), {}, threading.Lock()
        self._bridge = CvBridge()

        def keep_depth(m):
            with self._lock:
                self._depths.append(m)

        self._tf = engine.create_action_tf_buffer()
        engine.create_action_subscription(Image, topics[0], keep_depth, qos_profile_sensor_data)
        engine.create_action_subscription(
            CameraInfo, topics[1], lambda m: self._latest.update(info=m), qos_profile_sensor_data
        )
        # The engine has no public action-scoped service client; its action scope makes one (as for the
        # subscriptions above) and destroys it safely when the action ends.
        self._sam3 = engine._action_scope().create_client(Detect, sam3_service)
        if not self._sam3.wait_for_service(timeout_sec=SAM3_WAIT_S):
            raise NotLocated(f"SAM3 service {sam3_service} not available within {SAM3_WAIT_S:.0f} s")

    def grab(self, prompt, deadline):
        """One SAM3 call for `prompt` on the next color frame. (shot, None) with shot = dict(depth_m (HxW m),
        K (3x3), world_T_cam (4x4, optical -> `frame`), mask (HxW bool, SAM3's best object), stamp); (None, why)
        when SAM3 found nothing or there is no depth frame, camera_info or TF at its frame; (None, None) when
        time.monotonic() passed `deadline` first. Raises NotLocated on a depth encoding or image size it
        cannot use."""
        import time

        from rclpy.duration import Duration
        from rclpy.time import Time

        future = self._sam3.call_async(
            self._Detect.Request(prompts=[prompt], image_topic=self._image_topic, score_threshold=self._threshold)
        )
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            self._sam3.remove_pending_request(future)
            return None, None
        response = future.result()
        mask = cavity_mask_from_detect(response)
        if mask is None:
            return None, f"SAM3 found no {prompt!r}" + (f" ({response.message})" if response.message else "")
        stamp = _stamp(response.detections)
        with self._lock:
            depth = min(self._depths, key=lambda m: abs(_stamp(m) - stamp), default=None)
        info = self._latest.get("info")
        if depth is None or info is None or abs(_stamp(depth) - stamp) > PAIR_MAX_S:
            return None, "no depth frame or camera_info at SAM3's frame"
        if depth.encoding not in DEPTH_SCALE:
            raise NotLocated(f"scene depth encoding {depth.encoding!r} (expected one of {sorted(DEPTH_SCALE)})")
        try:
            t = self._tf.lookup_transform(
                self.frame, depth.header.frame_id, Time.from_msg(depth.header.stamp), timeout=Duration(seconds=0.5)
            ).transform
        except Exception as e:  # noqa: BLE001 -- tf2's lookup errors: this frame is not used
            return None, f"no TF {self.frame} <- {depth.header.frame_id} at the frame's stamp ({type(e).__name__})"
        T = np.eye(4)
        T[:3, :3] = _Rot.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
        T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        depth_m = self._bridge.imgmsg_to_cv2(depth).astype(np.float32) * DEPTH_SCALE[depth.encoding]
        if depth_m.shape != mask.shape:
            raise NotLocated(f"SAM3's mask is {mask.shape[::-1]} px, the depth image {depth_m.shape[::-1]}")
        K = np.array(info.k, float).reshape(3, 3)
        return dict(depth_m=depth_m, K=K, world_T_cam=T, mask=mask, stamp=_stamp(depth)), None


def locate(
    engine,
    frame,
    topics,
    frames,
    timeout_s,
    agree_m,
    agree_deg,
    log,
    sam3_service=SAM3_SERVICE,
    sam3_prompt=SAM3_PROMPT,
    sam3_threshold=SAM3_THRESHOLD,
    sam3_image_topic=SAM3_IMAGE_TOPIC,
    max_frames=None,
    camera=None,
):
    """The interior in `frame` (gravity-aligned, z up: arm_mount) from the scene camera, inside a
    running action of `engine`. topics: (depth, camera_info, object_detections_3d); the 3D detections
    are not used. Each round asks SAM3 (sam3_service) for sam3_prompt on the next color frame, pairs
    its mask with the depth frame of the same stamp, and fits that with the camera pose from TF at the
    stamp; frames that are not detected are skipped.

    max_frames None: once `frames` are detected, every point must be within agree_m of their median and
    every insertion axis within agree_deg of the median heading. max_frames > frames: up to max_frames
    are detected (until timeout_s), and at least `frames` of them must lie within agree_m and agree_deg of
    the median of all detected; only those are used. camera: a SceneCamera to use (None: one is made).

    Returns {point, insertion_axis, floor_z (medians of the frames used), frames (used), detected,
    spread_m, spread_deg, fit (the used frame's fit_interior result nearest the median point), scene (that
    frame: depth_m, K, world_T_cam, mask, stamp)}; raises NotLocated."""
    import time

    if camera is None:
        camera = SceneCamera(engine, frame, topics, sam3_service, sam3_threshold, sam3_image_topic)
    want = frames if max_frames is None else max(int(frames), int(max_frames))
    found, misses, asked = [], [], 0
    deadline = time.monotonic() + timeout_s
    while len(found) < want and time.monotonic() < deadline:
        shot, why = camera.grab(sam3_prompt, deadline)
        if shot is None and why is None:
            break
        asked += 1
        if shot is None:
            misses.append(why)
            continue
        try:
            found.append((fit_interior(shot["depth_m"], shot["K"], shot["world_T_cam"], shot["mask"]), shot))
        except NotDetected as e:
            misses.append(str(e))
            log(f"frame not detected: {e}")
    if len(found) < frames:
        why = "; ".join(dict.fromkeys(misses[-3:])) or "-"
        raise NotLocated(
            f"the interior was detected in {len(found)} of {asked} SAM3 frames in {timeout_s:.0f} s (need {frames}); "
            f"last reasons: {why}"
        )

    def medians(items):
        pts = np.array([r["point"] for r, _ in items])
        yaws = np.array([np.arctan2(r["insertion_axis"][1], r["insertion_axis"][0]) for r, _ in items])
        yaw = yaws[0] + float(np.median((yaws - yaws[0] + np.pi) % (2 * np.pi) - np.pi))
        off_m = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
        off_deg = np.degrees(np.abs((yaws - yaw + np.pi) % (2 * np.pi) - np.pi))
        return np.median(pts, axis=0), yaw, off_m, off_deg

    detected = len(found)
    mid, yaw, off_m, off_deg = medians(found)
    if max_frames is not None:
        agree = (off_m <= agree_m) & (off_deg <= agree_deg)
        log(
            f"located in {len(found)} frames, cm from their median: {np.round(off_m * 100, 1).tolist()}, "
            f"deg: {np.round(off_deg, 1).tolist()}"
        )
        if agree.sum() < frames:
            raise NotLocated(
                f"only {int(agree.sum())} of {len(found)} detected frames lie within {agree_m * 100:.1f} cm and "
                f"{agree_deg:.1f} deg of their median (need {frames}): cm off {np.round(off_m * 100, 1).tolist()}, "
                f"deg off {np.round(off_deg, 1).tolist()}"
            )
        found = [f for f, ok in zip(found, agree, strict=True) if ok]
        mid, yaw, off_m, off_deg = medians(found)
    spread_m, spread_deg = float(np.max(off_m)), float(np.max(off_deg))
    if max_frames is None and (spread_m > agree_m or spread_deg > agree_deg):
        raise NotLocated(
            f"{frames} detected frames disagree: points within {spread_m * 100:.1f} cm of their median (max "
            f"{agree_m * 100:.1f}), insertion axes within {spread_deg:.1f} deg (max {agree_deg:.1f})"
        )
    fit, shot = found[int(np.argmin(off_m))]
    return dict(
        point=mid.tolist(),
        insertion_axis=[float(np.cos(yaw)), float(np.sin(yaw)), 0.0],
        floor_z=float(np.median([r["floor_z"] for r, _ in found])),
        frames=len(found),
        detected=detected,
        spread_m=spread_m,
        spread_deg=spread_deg,
        fit=fit,
        scene=shot,
    )


def scene_frame(camera, prompt, timeout_s):
    """One scene-camera frame with SAM3's mask for `prompt` (e.g. "microwave"), its depth and the camera
    pose at its stamp (SceneCamera.grab's shot), within timeout_s; raises NotLocated with the reasons."""
    import time

    deadline, misses = time.monotonic() + timeout_s, []
    while time.monotonic() < deadline:
        shot, why = camera.grab(prompt, deadline)
        if shot is not None:
            return shot
        if why is None:
            break
        misses.append(why)
    raise NotLocated(
        f"no scene frame with SAM3's {prompt!r} in {timeout_s:.0f} s; last reasons: "
        + ("; ".join(dict.fromkeys(misses[-3:])) or "-")
    )
