"""The fridge door's direction (its outward normal, horizontal, in the arm's frame) measured from the handle, so the
grasp does not have to take the door as square to the arm.

measure_door_normal (the method): the printed handle's own model (RAMMP.Fridge.Handle5.STL, the file that was printed,
next to this module) registered to the scene camera's points on SAM 3's 'handle' mask. The handle's mounting plate sits
flat on the door, so the model's door normal (its -z) is the door's.
1. All of the mask's points with depth, into the arm's frame.
2. The model, sampled on its surface, starts with its bar's front at the mask's nearest layer (per row, within
   FRONT_LAYER_M of the row's nearest point), at each yaw of ICP_YAW_STARTS_DEG.
3. 4-DoF point-to-plane ICP (x, y, z and yaw about the vertical: the door is vertical): each data point matched to the
   nearest model point on a face turned toward the camera, Huber-weighted, those farther than ICP_MATCH_M left out of
   the step. The start with the lowest final cost wins.
4. The look fails with the reason (DoorNormalError), never a default normal, on: too few points, too few of them on the
   model (FIT_MIN_INLIER_FRACTION), too large a residual (FIT_MAX_RMS_M), the model moving off the handle, or a normal
   more than MAX_OFF_CAMERA_DEG from facing the camera.
11-06 scene frames (one wheelchair position, 4 runs x 8 looks, nothing moving): yaw +5.8 to +9.5 deg, SD 0.9 deg,
consecutive looks within 0.7 deg typically (2.9 at most), run means +7.1 to +7.7 deg; 2.8-3.1 mm RMS, 56-66% of the
mask's points on the model. The fitted height is loose (+-1 cm: the handle is nearly symmetric top to bottom).

measure_bar_front_normal (kept, not the default): a vertical plane (RANSAC + SVD) on the bar's front surface alone.
The front is only ~12 px (~3 cm) wide at 0.7-0.9 m and the stereo depth is smoothed across it: on the same frames SD
2.9 deg per look and ~9 deg below the model fit.

combine_looks: the mean of several looks' normals; it refuses with too few looks or too wide a spread.

Yaw convention (yaw_deg): atan2(n_y, -n_x) of the outward normal n. 0 means the door is square to the arm (n = -x).
Positive means n turns toward the arm's left (+y): the door's right-hand end (toward -y) is nearer the arm.

numpy and scipy (cKDTree) only: no ROS and no open3d, so it runs offline on saved frames. make_detector gives the
fridge's HandleDetector with its door_normal_from_handle hook using measure_door_normal (it imports the ROS side only
when called). The STL has to be installed next to this module (setup.py's package_data).
"""

import os
from dataclasses import dataclass, field

import numpy as np

DEPTH_RANGE_M = (0.05, 2.0)  # as door_common.perception.DEPTH_RANGE_M
ROWS_TRIM = 0.15  # the mask's rows used: all but this fraction at its top and bottom (the bar's legs, their bends)
FRONT_LAYER_M = 0.006  # the bar's front: per mask row, the points within this (camera depth) of the row's nearest
# (11-06: 6, 10 and 20 mm gave the same mean within 0.3 deg; 6 mm the smallest SD)
EDGE_TRIM_PX = 0  # front pixels also cut from each row's left and right end. 11-06 (32 looks of one view): 0 px mean
# -1.3 SD 2.9 deg; 1 px -0.6 / 3.1; 2 px +0.2 / 3.7 -- the bar is only ~12 px wide, so cutting leaves too few points
RANSAC_ITERS = 400
RANSAC_INLIER_M = 0.003  # a front point within this of the plane is an inlier (the depth's noise is ~1.5 mm RMS here)
RANSAC_MIN_PAIR_M = 0.01  # RANSAC's two sample points at least this far apart (horizontally)
MIN_POINTS = 150  # fewer bar-front points (or inliers): the look fails (11-06: ~600-700)
MIN_INLIER_FRACTION = 0.5
MIN_SPAN_M = 0.015  # the inliers' extent across the bar (11-06: ~2.5-3 cm)
MAX_RMS_M = 0.003  # the inliers' RMS distance from the plane (11-06: ~1.5 mm)
MAX_OFF_CAMERA_DEG = 60.0  # the normal at most this far (horizontally) from pointing back at the camera
COMBINE_MIN_LOOKS = 3
COMBINE_MAX_SPREAD_DEG = 6.0  # max - min yaw over the looks combined (11-06, model fit: 1.7-3.2 deg over 8 looks)
SEED = 0  # RANSAC's random seed: a frame gives the same answer every time


class DoorNormalError(RuntimeError):
    """The door's normal could not be measured from this look. The message says why."""


@dataclass
class DoorNormal:
    """One look's door normal, in the arm's frame."""

    normal: np.ndarray  # (3,) unit, horizontal, out of the door toward the camera
    yaw_deg: float  # yaw_deg(normal)
    centre: np.ndarray  # (3,) the inliers' mean: a point on the fitted plane
    inliers: int  # bar-front points on the plane
    points: int  # bar-front points the plane was fit to
    rms_m: float  # inliers' RMS distance from the plane
    span_m: float  # inliers' extent across the bar, along the door (5th to 95th pct)
    height_m: float  # inliers' vertical extent (5th to 95th pct)
    debug: dict = field(default_factory=dict, repr=False)  # arm-frame points: front_in, front_out, rest (of the mask)


def yaw_deg(n):
    """atan2(n_y, -n_x): 0 with the outward normal along the arm's -x (square to the arm), + toward the arm's left."""
    return float(np.degrees(np.arctan2(n[1], -n[0])))


def normal_from_yaw(yaw):
    """The horizontal outward normal with yaw `yaw` (deg), as yaw_deg reads it."""
    a = np.radians(yaw)
    return np.array([-np.cos(a), np.sin(a), 0.0])


def _points(depth_mm, K, base_T_cam, mask):
    """The arm-frame points of `mask`'s pixels with valid depth, and their (v, u)."""
    vs, us = np.nonzero(mask)
    d = np.asarray(depth_mm, float)[vs, us] / 1000.0
    ok = np.isfinite(d) & (d >= DEPTH_RANGE_M[0]) & (d <= DEPTH_RANGE_M[1])
    vs, us, d = vs[ok], us[ok], d[ok]
    cam = np.stack([(us - K[0, 2]) / K[0, 0] * d, (vs - K[1, 2]) / K[1, 1] * d, d], axis=1)
    T = np.asarray(base_T_cam, float)
    return cam @ T[:3, :3].T + T[:3, 3], vs, us


def _tls(xy):
    """Total least squares line through 2D points: (unit normal, centre, RMS distance)."""
    c = xy.mean(axis=0)
    _, s, vt = np.linalg.svd(xy - c, full_matrices=False)
    return vt[1], c, float(s[1] / np.sqrt(len(xy)))


def fit_vertical_plane(P, toward, inlier_m=RANSAC_INLIER_M, iters=RANSAC_ITERS, seed=SEED):
    """A vertical plane through arm-frame points `P` (N x 3): RANSAC on their horizontal (x, y), then an SVD on the
    inliers. Returns (unit horizontal normal pointing along `toward`, centre (3,), inlier bool mask, RMS m)."""
    xy = P[:, :2]
    if len(xy) < 2:
        raise DoorNormalError(f"{len(xy)} points to fit a plane to")
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        i, j = rng.integers(len(xy), size=2)
        dv = xy[j] - xy[i]
        length = float(np.hypot(*dv))
        if length < RANSAC_MIN_PAIR_M:
            continue
        n = np.array([-dv[1], dv[0]]) / length
        inl = np.abs((xy - xy[i]) @ n) < inlier_m
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None or best_n < 2:
        raise DoorNormalError(f"RANSAC found no plane among {len(xy)} points")
    for _ in range(2):  # refine: SVD on the inliers, then the inliers of that
        n, c, _ = _tls(xy[best])
        best = np.abs((xy - c) @ n) < inlier_m
        if best.sum() < 2:
            raise DoorNormalError("the plane's refit kept no inliers")
    n, c, rms = _tls(xy[best])
    n3 = np.array([n[0], n[1], 0.0])
    if n3 @ np.asarray(toward, float) < 0:
        n3 = -n3
    return n3, np.array([c[0], c[1], float(P[best, 2].mean())]), best, rms


def measure_bar_front_normal(depth_mm, K, base_T_cam, handle_mask, edge_trim_px=None):
    """The door's outward normal from one look at the handle's bar front (module docstring): depth_mm (H x W, mm,
    aligned to the colour image the mask is on), K 3x3, base_T_cam 4x4 (arm frame <- camera), handle_mask (H x W bool,
    SAM 3's handle). edge_trim_px: None = EDGE_TRIM_PX. Returns DoorNormal; raises DoorNormalError saying why."""
    trim = EDGE_TRIM_PX if edge_trim_px is None else int(edge_trim_px)
    mask = np.asarray(handle_mask, bool)
    if base_T_cam is None:
        raise DoorNormalError("no arm <- camera transform")
    if not mask.any():
        raise DoorNormalError("empty handle mask")
    T = np.asarray(base_T_cam, float)
    vs, _ = np.nonzero(mask)
    v0, v1 = int(vs.min()), int(vs.max())
    cut = int(ROWS_TRIM * (v1 - v0))
    rows = np.zeros_like(mask)
    rows[v0 + cut : v1 - cut + 1] = True
    P, pv, pu = _points(depth_mm, K, T, mask & rows)
    if len(P) < MIN_POINTS:
        raise DoorNormalError(f"{len(P)} handle points with depth (needs {MIN_POINTS})")
    d = np.asarray(depth_mm, float)[pv, pu] / 1000.0
    nearest = np.full(int(pv.max()) + 1, np.inf)
    np.minimum.at(nearest, pv, d)
    front = d <= nearest[pv] + FRONT_LAYER_M
    if trim > 0:
        lo = np.full(int(pv.max()) + 1, np.iinfo(np.int64).max)
        hi = np.full(int(pv.max()) + 1, -1)
        np.minimum.at(lo, pv[front], pu[front])
        np.maximum.at(hi, pv[front], pu[front])
        front &= (pu >= lo[pv] + trim) & (pu <= hi[pv] - trim)
    F = P[front]
    if len(F) < MIN_POINTS:
        raise DoorNormalError(f"{len(F)} points on the bar's front (needs {MIN_POINTS})")
    cam = T[:3, 3]
    toward = cam - F.mean(axis=0)
    toward[2] = 0.0
    n, centre, inl, rms = fit_vertical_plane(F, toward)
    along = F[inl] @ np.cross([0.0, 0.0, 1.0], n)
    span = float(np.percentile(along, 95) - np.percentile(along, 5))
    height = float(np.percentile(F[inl, 2], 95) - np.percentile(F[inl, 2], 5))
    to_cam = cam - centre
    to_cam[2] = 0.0
    off = float(np.degrees(np.arccos(np.clip(n @ to_cam / np.linalg.norm(to_cam), -1.0, 1.0))))
    res = DoorNormal(
        normal=n,
        yaw_deg=yaw_deg(n),
        centre=centre,
        inliers=int(inl.sum()),
        points=len(F),
        rms_m=rms,
        span_m=span,
        height_m=height,
        debug={"front_in": F[inl], "front_out": F[~inl], "rest": P[~front]},
    )
    problems = []
    if res.inliers < MIN_POINTS:
        problems.append(f"{res.inliers} bar-front inliers (needs {MIN_POINTS})")
    if res.inliers < MIN_INLIER_FRACTION * res.points:
        problems.append(f"only {res.inliers} of {res.points} bar-front points on the plane")
    if span < MIN_SPAN_M:
        problems.append(f"the bar's front spans {span * 100:.1f} cm across (needs {MIN_SPAN_M * 100:.1f})")
    if rms > MAX_RMS_M:
        problems.append(f"the bar's front is {rms * 1000:.1f} mm RMS off its plane (max {MAX_RMS_M * 1000:.0f})")
    if off > MAX_OFF_CAMERA_DEG:
        problems.append(f"the normal faces {off:.0f} deg away from the camera (max {MAX_OFF_CAMERA_DEG:.0f})")
    if problems:
        raise DoorNormalError(
            f"door normal from the bar's front (yaw {res.yaw_deg:+.1f} deg): " + "; ".join(problems)
        )
    return res


# --- registering the printed handle's model (the default method) ---------------------------------------------------
HANDLE_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAMMP.Fridge.Handle5.STL")
# the STL (the file that was printed) is in mm: x along the bar (247.65 long, vertical on the door), y across it (63.5),
# z out of the door: the mounting plate's back (z 80, flat on the door) to the bar's front (z 0.9-1). Its door normal
# (out of the door) is the model's -z
MODEL_SAMPLES = 6000  # points sampled on the model's surface, by area
ICP_ITERS = 40
ICP_MATCH_M = 0.02  # data points farther than this from the model (after the start) are left out of a step
ICP_HUBER_M = 0.003  # Huber's knee on the point-to-plane residual
ICP_YAW_STARTS_DEG = (-20.0, 0.0, 20.0)  # starts tried (the best final cost wins)
FIT_INLIER_M = 0.005  # a mask point within this of the fitted model counts as on it
FIT_MIN_POINTS = 300  # fewer mask points with depth: the look fails
FIT_MIN_INLIER_FRACTION = 0.6  # of the mask's points on the model's footprint, within FIT_INLIER_M of the model
# (11-06 still scene frames: 77-91%)
FIT_REF_RANGE_M = 0.75  # FIT_INLIER_M and FIT_MAX_RMS_M hold at this camera range; farther, they grow with the
# stereo depth noise, ~ range^2 (11-06: the handle at 0.93 m fit only 44-49% within 5 mm, 0.74 m fit fine)
FIT_MAX_RMS_M = 0.0035  # RMS point-to-model distance of the inliers (11-06: 2.7-3.1 mm)
FIT_MIN_FOOTPRINT_FRACTION = 0.5  # of the mask's points with depth, on the model's footprint (11-06 still: 72-75%)
FOOTPRINT_MARGIN_M = 0.005  # the footprint: the model's outline (x, y in its frame) grown by this ...
FOOTPRINT_FRONT_M = 0.01  # ... from this in front of the bar to ...
FOOTPRINT_BEHIND_M = 0.01  # ... this behind the plate's back (depth noise; the door beside the plate is outside x, y)
FOOTPRINT_FROM_ITER = 10  # a cold fit uses the footprint from this iteration on (the pose is rough before) ...
FOOTPRINT_FIT = False  # ... when this is set; else the footprint only judges the fit (FIT_MIN_FOOTPRINT_FRACTION).
# Off: on 11-06 fitting to the footprint alone made the still looks noisier (SD 1.7 vs 0.9 deg) and did not fix the
# oblique views of the rosbag
WARM_ITERS = 8  # a warm start (prev): at most this many iterations
ICP_MAX_POINTS = 1200  # ICP's steps use at most this many of the mask's points (a fixed random subset): speed
FIT_MAX_STEP_M = 0.10  # the fit may move the model at most this far from where it started


@dataclass
class HandleFit:
    """One look's handle model registration, in the arm's frame."""

    normal: np.ndarray  # (3,) the door's outward normal: the model's -z, horizontal
    yaw_deg: float  # yaw_deg(normal)
    position: np.ndarray  # (3,) the model's bar-front centre (the middle of the bar's front face)
    R: np.ndarray  # 3x3 model -> arm rotation
    t: np.ndarray  # (3,) model (m) -> arm translation: arm = R @ model_m + t
    rms_m: float  # inliers' RMS distance to the model
    inlier_fraction: float  # of the mask points with depth, within FIT_INLIER_M of the model
    points: int  # mask points with depth
    footprint_fraction: float  # of those, on the model's footprint (the rest: door / side seen past the handle)
    debug: dict = field(default_factory=dict, repr=False)  # arm-frame: data_in, data_out, model (visible samples)


_MODEL = None


def load_handle_model(path=HANDLE_STL, n=MODEL_SAMPLES, seed=SEED):
    """(points (n x 3, m), normals (n x 3), bar-front centre (3,), centre (3,), bounds lo (3,), hi (3,)) of the
    binary STL at `path`, sampled on its surface."""
    global _MODEL
    if _MODEL is not None and _MODEL[0] == (path, n):
        return _MODEL[1]
    with open(path, "rb") as f:
        raw = f.read()
    count = int(np.frombuffer(raw[80:84], "<u4")[0])
    if len(raw) != 84 + 50 * count:
        raise DoorNormalError(f"{path}: not a binary STL ({len(raw)} bytes for {count} triangles)")
    tri = np.frombuffer(raw[84:], np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"].astype(float)
    tri /= 1000.0  # mm -> m
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    keep = area > 0
    tri, cross, area = tri[keep], cross[keep], area[keep]
    normals = cross / (2 * area[:, None])
    rng = np.random.default_rng(seed)
    k = rng.choice(len(tri), n, p=area / area.sum())
    a, b = rng.random(n), rng.random(n)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    pts = tri[k, 0] + a[:, None] * (tri[k, 1] - tri[k, 0]) + b[:, None] * (tri[k, 2] - tri[k, 0])
    lo, hi = tri.reshape(-1, 3).min(axis=0), tri.reshape(-1, 3).max(axis=0)
    centre = (lo + hi) / 2
    front = np.array([centre[0], centre[1], lo[2]])  # the bar's front face, at the middle of the bar
    model = (pts, normals[k], front, centre, lo, hi)
    _MODEL = ((path, n), model)
    return model


def _model_rotation(yaw):
    """Model -> arm rotation for door yaw `yaw` (deg): model x up, model -z along the door's outward normal."""
    n = normal_from_yaw(yaw)
    up = np.array([0.0, 0.0, 1.0])
    zc = -n
    yc = np.cross(zc, up)
    return np.stack([up, yc, zc], axis=1)


def _bar_front_point(P, d, pv):
    """Rough handle point to start from: the median of the mask's nearest layer (per row, within FRONT_LAYER_M)."""
    nearest = np.full(int(pv.max()) + 1, np.inf)
    np.minimum.at(nearest, pv, d)
    return np.median(P[d <= nearest[pv] + FRONT_LAYER_M], axis=0)


def _footprint(P, R, t, model):
    """Which of P lie on the handle's footprint in the model's frame: within FOOTPRINT_MARGIN_M of its outline (x, y),
    from FOOTPRINT_FRONT_M in front of the bar to FOOTPRINT_BEHIND_M behind the plate's back. The door or the fridge's
    side seen past the handle's outline is left out."""
    M = (P - t) @ R
    lo, hi = model[4], model[5]
    m = FOOTPRINT_MARGIN_M
    return (
        (M[:, 0] > lo[0] - m) & (M[:, 0] < hi[0] + m) & (M[:, 1] > lo[1] - m) & (M[:, 1] < hi[1] + m)
        & (M[:, 2] > lo[2] - FOOTPRINT_FRONT_M) & (M[:, 2] < hi[2] + FOOTPRINT_BEHIND_M)
    )


def _icp(P, cam, R, t, model, iters, footprint_from):
    """4-DoF (x, y, z, yaw) point-to-plane ICP of the model onto data P from the pose (R, t) (arm = R @ model + t).
    From iteration `footprint_from` on, only data on the model's footprint (_footprint) is used. Returns
    (R, t, cost, footprint mask)."""
    from scipy.spatial import cKDTree

    pts, nrm = model[0], model[1]
    full = P
    if len(P) > ICP_MAX_POINTS:  # speed: the steps use a fixed subset; the cost and the checks use every point
        P = P[np.random.default_rng(SEED).choice(len(P), ICP_MAX_POINTS, replace=False)]
    on = np.ones(len(P), bool)
    for i in range(iters):
        if FOOTPRINT_FIT and i >= footprint_from:
            on = _footprint(P, R, t, model)
            if on.sum() < 10:
                return R, t, np.inf, on
        X = pts @ R.T + t
        Nw = nrm @ R.T
        vis = np.einsum("ij,ij->i", Nw, cam - X) > 0  # faces turned toward the camera
        X, Nw = X[vis], Nw[vis]
        dist, idx = cKDTree(X).query(P[on])
        use = dist < ICP_MATCH_M
        if use.sum() < 10:
            return R, t, np.inf, on
        q, m, p = X[idx[use]], Nw[idx[use]], P[on][use]
        r = np.einsum("ij,ij->i", m, p - q)
        c = np.array([*(R @ model[3] + t)[:2], 0.0])  # rotate about the vertical through the model's centre
        lever = np.cross([0.0, 0.0, 1.0], q - c)
        J = np.column_stack([np.einsum("ij,ij->i", m, lever), m])  # d(model point)/d(yaw rad, tx, ty, tz) . normal
        w = np.where(np.abs(r) < ICP_HUBER_M, 1.0, ICP_HUBER_M / np.maximum(np.abs(r), 1e-12))
        A = J.T @ (J * w[:, None]) + np.eye(4) * 1e-9
        delta = np.linalg.solve(A, J.T @ (w * r))
        # delta[0] turns the model counter-clockwise (from above) about the vertical through c; yaw_deg counts the
        # other way (+ turns the outward normal, ~-x, toward +y)
        Rz = _model_rotation(-float(np.degrees(delta[0]))) @ _model_rotation(0.0).T
        t = Rz @ (t - c) + c + delta[1:]
        R = Rz @ R
        settled = abs(delta[0]) < 1e-4 and np.linalg.norm(delta[1:]) < 1e-4
        if settled and (i >= footprint_from or not FOOTPRINT_FIT):
            break
    P = full
    on = _footprint(P, R, t, model)
    X = pts @ R.T + t
    dist = cKDTree(X).query(P[on])[0] if on.any() else np.array([np.inf])
    cost = float(np.sum(np.minimum(dist, ICP_MATCH_M) ** 2)) + ICP_MATCH_M**2 * float((~on).sum())
    return R, t, cost, on


def measure_door_normal(depth_mm, K, base_T_cam, handle_mask, stl_path=HANDLE_STL, prev=None):
    """The door's outward normal from one look: the printed handle's model (stl_path) registered to the SAM 3 mask's
    points with depth, 4-DoF (position and yaw about the vertical: the door is vertical), point-to-plane ICP with a
    Huber loss, matched only to model faces turned toward the camera, and -- once roughly in place -- only to the mask's
    points on the model's footprint (the door or the fridge's side seen past the handle left out). Cold: from yaws
    ICP_YAW_STARTS_DEG with the bar's front at the mask's nearest layer, the best fit wins. prev (a HandleFit, e.g. the
    last frame's): a warm start from its pose, WARM_ITERS at most. Returns HandleFit; raises DoorNormalError (why) on a
    poor fit, never a default normal."""
    mask = np.asarray(handle_mask, bool)
    if base_T_cam is None:
        raise DoorNormalError("no arm <- camera transform")
    if not mask.any():
        raise DoorNormalError("empty handle mask")
    T = np.asarray(base_T_cam, float)
    P, pv, pu = _points(depth_mm, K, T, mask)
    if len(P) < FIT_MIN_POINTS:
        raise DoorNormalError(f"{len(P)} handle points with depth (needs {FIT_MIN_POINTS})")
    d = np.asarray(depth_mm, float)[pv, pu] / 1000.0
    start = _bar_front_point(P, d, pv)
    cam = T[:3, 3]
    model = load_handle_model(stl_path)
    if prev is not None:
        # the warm start, and one fresh start (yaw 0 at the mask's front): the yaw can have two minima ~10 deg apart
        # (11-06, handle at 0.55 m), and a warm start alone stays in whichever it was in
        R0 = _model_rotation(0.0)
        fits = [
            _icp(P, cam, prev.R, prev.t, model, WARM_ITERS, 0),
            _icp(P, cam, R0, start - R0 @ model[2], model, WARM_ITERS + 4, 0),
        ]
        R, t, _, on = min(fits, key=lambda f: f[2])
    else:
        fits = []
        for y0 in ICP_YAW_STARTS_DEG:
            R0 = _model_rotation(y0)
            fits.append(_icp(P, cam, R0, start - R0 @ model[2], model, ICP_ITERS, FOOTPRINT_FROM_ITER))
        R, t, _, on = min(fits, key=lambda f: f[2])
    from scipy.spatial import cKDTree

    X = model[0] @ R.T + t
    vis = np.einsum("ij,ij->i", model[1] @ R.T, cam - X) > 0
    F = P[on]
    dist = cKDTree(X).query(F)[0] if len(F) else np.zeros(0)
    n = -R[:, 2]
    position = R @ model[2] + t
    noise = max(1.0, (float(np.linalg.norm(position - cam)) / FIT_REF_RANGE_M) ** 2)  # depth noise ~ range^2
    inl = dist < FIT_INLIER_M * noise
    res = HandleFit(
        normal=n,
        yaw_deg=yaw_deg(n),
        position=position,
        R=R,
        t=t,
        rms_m=float(np.sqrt(np.mean(dist[inl] ** 2))) if inl.any() else float("inf"),
        inlier_fraction=float(inl.mean()) if len(inl) else 0.0,
        points=len(P),
        footprint_fraction=float(on.mean()),
        debug={"data_in": F[inl], "data_out": np.vstack([F[~inl], P[~on]]), "model": X[vis], "start": start},
    )
    problems = []
    if on.sum() < FIT_MIN_POINTS:
        problems.append(f"{int(on.sum())} mask points on the model's footprint (needs {FIT_MIN_POINTS})")
    if res.footprint_fraction < FIT_MIN_FOOTPRINT_FRACTION:
        problems.append(
            f"only {res.footprint_fraction * 100:.0f}% of the mask's points on the model's footprint "
            f"(needs {FIT_MIN_FOOTPRINT_FRACTION * 100:.0f}%)"
        )
    if res.inlier_fraction < FIT_MIN_INLIER_FRACTION:
        problems.append(
            f"only {res.inlier_fraction * 100:.0f}% of the footprint's points within {FIT_INLIER_M * noise * 1000:.1f} mm of "
            f"the model (needs {FIT_MIN_INLIER_FRACTION * 100:.0f}%)"
        )
    if res.rms_m > FIT_MAX_RMS_M * noise:
        problems.append(f"{res.rms_m * 1000:.1f} mm RMS off the model (max {FIT_MAX_RMS_M * noise * 1000:.1f})")
    if np.linalg.norm(position - start) > FIT_MAX_STEP_M:
        problems.append(f"the model moved {np.linalg.norm(position - start) * 100:.0f} cm from the handle's front")
    to_cam = cam - position
    to_cam[2] = 0.0
    off = float(np.degrees(np.arccos(np.clip(n @ to_cam / np.linalg.norm(to_cam), -1.0, 1.0))))
    if off > MAX_OFF_CAMERA_DEG:
        problems.append(f"the normal faces {off:.0f} deg away from the camera (max {MAX_OFF_CAMERA_DEG:.0f})")
    if problems:
        raise DoorNormalError(f"handle model fit (yaw {res.yaw_deg:+.1f} deg): " + "; ".join(problems))
    return res


@dataclass
class CombinedNormal:
    normal: np.ndarray
    yaw_deg: float
    spread_deg: float  # max - min yaw of the looks
    sd_deg: float  # SD of the looks' yaws
    looks: int


def combine_looks(normals, min_looks=COMBINE_MIN_LOOKS, max_spread_deg=COMBINE_MAX_SPREAD_DEG):
    """The mean of several looks' outward normals (DoorNormal or (3,) arrays). Raises DoorNormalError with fewer than
    `min_looks` or a yaw spread (max - min) over `max_spread_deg`."""
    ns = [np.asarray(getattr(x, "normal", x), float) for x in normals]
    if len(ns) < min_looks:
        raise DoorNormalError(f"{len(ns)} look(s) with a door normal (needs {min_looks})")
    yaws = np.array([yaw_deg(n) for n in ns])
    spread = float(yaws.max() - yaws.min())
    if spread > max_spread_deg:
        raise DoorNormalError(
            f"the looks' door yaws spread {spread:.1f} deg (max {max_spread_deg:g}): "
            + ", ".join(f"{y:+.1f}" for y in yaws)
        )
    m = np.mean([n / np.linalg.norm(n) for n in ns], axis=0)
    m[2] = 0.0
    m /= np.linalg.norm(m)
    return CombinedNormal(m, yaw_deg(m), spread, float(yaws.std(ddof=1)) if len(yaws) > 1 else 0.0, len(ns))


def make_detector(**kwargs):
    """The fridge's HandleDetector(door_normal_from_handle=True, **kwargs), its _normal_from_handle replaced by
    measure_door_normal (the handle model's fit) on the picked mask. A look that can't
    be measured is a miss (last_miss says why), never a default normal. The detection's 'normal' is this look's;
    combine several looks with combine_looks before relying on it."""
    from door_common.common import say

    from . import perception

    class ModelNormalHandleDetector(perception.HandleDetector):
        last_door_normal = None

        def _normal_from_handle(self, mask, depth, K, base_T_cam):
            try:
                res = measure_door_normal(depth, K, base_T_cam, mask)
            except DoorNormalError as e:
                self.last_miss = str(e)
                self.last_door_normal = None
                return None
            self.last_door_normal = res
            say(
                f"  door normal from the handle's model: yaw {res.yaw_deg:+.1f} deg (0 = square to the arm), "
                f"{res.inlier_fraction * 100:.0f}% of {res.points} pts on it, {res.rms_m * 1000:.1f} mm RMS, bar front "
                f"at {np.round(res.position, 3)}"
            )
            return res.normal.copy()

    return ModelNormalHandleDetector(door_normal_from_handle=True, **kwargs)
