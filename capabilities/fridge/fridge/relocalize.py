"""Where the fridge is now: re-detected before the close, after the wheelchair has been moved by hand since the open.

The open's door file is true in the arm's base frame as the open ended. The arm rides on the wheelchair, so moving the
chair moves the whole fridge in that frame: along x and y, and turned about vertical. When an executed open ends, the
wrist camera's depth is kept as the reference (`capture_reference`): the fridge's points (the cabinet and the open door,
cropped by the door file's model, and by SAM 3's mask of RELOC_SAM3_PROMPTS when it answers), with the door file's hinge
it was taken with. Before the close, a new frame is aligned to it (`relocalize`): the rigid move (dx, dy, yaw) that
carries the reference's fridge onto the points seen now. It needs no guess: a grid of moves is scored against the new
points, the best few are refined by point-to-plane ICP, and the result is gated on how well it fits. The door file is
then carried by that move (`moved_door`) in memory only: the file and its reference stay a pair, so every close
re-detects against the same reference.

11-13 close-start frames (fridge_data/runs/*_close/*_close_start.npz), offline, depth only (no SAM 3 mask): a ~35 cm
manual move back measured 33.4 cm back, 3.6 cm sideways, 1.6 deg turned. Four views with the chair still, from
different arm poses: within 1 cm along x and 2 deg, but 1-7 cm sideways -- the inside of the cabinet constrains
sideways least. The one pair it got wrong (the reference taken after the move, seeing mostly the door) fit 21% of its
points within 2 cm, against 41-80% when right: RELOC_MIN_INLIER_FRAC.
"""

import json
import time

import numpy as np
from door_common import common as dc
from door_common.common import Refused, say
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from . import constants as F
from .motion import opened
from .world import FridgeModel

REF_FILE_NAME = "door_ref.npz"  # next to the door file
# the door file's positions and directions (arm base frame) that a move of the fridge carries; the rest are sizes,
# heights and angles of the door itself
POINT_KEYS = (
    "hinge",
    "hinge_pivot",
    "hinge_estimate",
    "closed_grasp_pos",
    "closed_handle",
    "closed_hold_pos",
    "front_hold",
    "door_mid",
    "detected_edge",
)
VECTOR_KEYS = ("closed_normal",)


def ref_file():
    """The reference frame's file, next to the door file (door_common sets where that is)."""
    return dc.DOOR_FILE.parent / REF_FILE_NAME


# --- geometry: pure functions (offline-testable on saved frames) ---------------------------------------------------


def voxel(points, size):
    """One point per `size` m cube."""
    if not len(points):
        return points
    _, keep = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(keep)]


def frame_points(frame, ee_pos, mask=None):
    """The frame's depth as points in the arm's base frame (N, 3), RELOC_VOXEL_M apart: within RELOC_DEPTH_M of the
    camera, inside `mask` (image-sized bool) if given, and not within RELOC_HAND_CLEAR_M of the tool point `ee_pos`
    (the hand and what it holds)."""
    z = np.nan_to_num(np.asarray(frame["depth_mm"], float), nan=0.0) / 1000.0
    ok = (z > F.RELOC_DEPTH_M[0]) & (z < F.RELOC_DEPTH_M[1])
    if mask is not None:
        ok &= np.asarray(mask, bool)
    v, u = np.nonzero(ok)
    zz = z[v, u]
    K = np.asarray(frame["K"], float)
    cam = np.c_[(u - K[0, 2]) * zz / K[0, 0], (v - K[1, 2]) * zz / K[1, 1], zz]
    T = np.asarray(frame["base_T_cam"], float)
    p = cam @ T[:3, :3].T + T[:3, 3]
    p = p[np.linalg.norm(p - np.asarray(ee_pos[:3], float), axis=1) > F.RELOC_HAND_CLEAR_M]
    return voxel(p, F.RELOC_VOXEL_M)


def _box_gap(points, box):
    """How far each point is outside FridgeModel box `box` (0 inside)."""
    loc = R.from_euler("z", box["rpy_deg"][2], degrees=True).inv().apply(points - np.asarray(box["position"], float))
    return np.linalg.norm(np.maximum(np.abs(loc) - np.asarray(box["dims"], float) / 2, 0.0), axis=-1)


def fridge_boxes(door):
    """The fridge as the door file models it: the cabinet, the open door and its handle."""
    deg = float(door["door_open_deg"])
    model = FridgeModel(opened(door, deg))
    return model.body() + model.slab(deg) + model.handle(deg)


def crop(points, boxes, margin):
    """The points within `margin` of any of `boxes`."""
    if not len(points):
        return points
    return points[np.min([_box_gap(points, b) for b in boxes], axis=0) < margin]


def _rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _apply(points, x):
    return points @ _rz(x[2]).T + np.r_[x[0], x[1], 0.0]


def _normals(points, k=12):
    """Each point's surface normal, from its k nearest neighbours."""
    _, nn = cKDTree(points).query(points, k)
    q = points[nn] - points[nn].mean(axis=1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", q, q))[1][:, :, 0]


def icp(src, dst, x0, iters=40, trim=0.7):
    """The move x = (dx, dy, yaw: about the base's z) carrying `src` onto `dst`, point-to-plane from `x0`, the worst
    1 - `trim` of the pairs (and any over 8 cm apart) left out each step. Returns (x, median gap m, fraction of
    `src` within RELOC_INLIER_M of `dst`)."""
    nd, tree, x = _normals(dst), cKDTree(dst), np.array(x0, float)
    for _ in range(iters):
        q = _apply(src, x)
        d, j = tree.query(q)
        keep = (d < np.quantile(d, trim)) & (d < 0.08)
        if keep.sum() < 10:
            break
        n, qq = nd[j[keep]], q[keep]
        r = ((qq - dst[j[keep]]) * n).sum(axis=1)
        rel = qq - np.r_[x[0], x[1], 0.0]
        J = np.c_[n[:, 0], n[:, 1], n[:, 1] * rel[:, 0] - n[:, 0] * rel[:, 1]]
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        x += step
        if np.abs(step).max() < 1e-5:
            break
    d, _ = tree.query(_apply(src, x))
    return x, float(np.median(d)), float((d < F.RELOC_INLIER_M).mean())


def align(ref, cur):
    """The fridge's move from the reference points `ref` to the current ones `cur`, without a guess: every move of a
    grid (RELOC_SEARCH_M, RELOC_SEARCH_YAW_DEG) scored on coarse points, the best RELOC_STARTS distinct ones refined
    by icp on 2 cm points, the best of those on the full points. Returns (x, median gap m, inlier fraction)."""
    rc, cc = voxel(ref, 0.04), voxel(cur, 0.03)
    tree = cKDTree(cc)
    (sx, sy), sy_deg = F.RELOC_SEARCH_M, F.RELOC_SEARCH_YAW_DEG
    scored = []
    for yaw in np.radians(np.arange(-sy_deg, sy_deg + 1e-9, 3.0)):
        turned = rc @ _rz(yaw).T
        for dx in np.arange(-sx, sx + 1e-9, 0.05):
            for dy in np.arange(-sy, sy + 1e-9, 0.05):
                d, _ = tree.query(turned + np.r_[dx, dy, 0.0], distance_upper_bound=0.04)
                scored.append((float(np.isfinite(d).mean()), dx, dy, yaw))
    scored.sort(key=lambda s: -s[0])
    starts = []
    for _, dx, dy, yaw in scored:
        if all(np.hypot(dx - a, dy - b) > 0.08 or abs(yaw - c) > np.radians(5) for a, b, c in starts):
            starts.append((dx, dy, yaw))
        if len(starts) == F.RELOC_STARTS:
            break
    r2, c2 = voxel(ref, 0.02), voxel(cur, 0.02)
    best = max((icp(r2, c2, s, iters=25) for s in starts), key=lambda fit: fit[2])
    return icp(ref, cur, best[0])


def moved_door(door, x):
    """`door` (a door dict) carried by the fridge's move x = (dx, dy, yaw) in the arm's base frame."""
    rot = _rz(x[2])
    out = dict(door)
    for k in POINT_KEYS:
        if out.get(k) is not None:
            p = np.asarray(out[k], float)
            q = rot[:2, :2] @ p[:2] + np.asarray(x[:2], float)
            out[k] = [float(q[0]), float(q[1]), *map(float, p[2:])]
    for k in VECTOR_KEYS:
        if out.get(k) is not None:
            out[k] = [float(c) for c in rot @ np.asarray(out[k], float)]
    return out


def trusted(med, frac, n_ref, n_cur):
    """Why the fit is not trusted, or ''."""
    if min(n_ref, n_cur) < F.RELOC_MIN_POINTS:
        return f"too few points on the fridge ({n_ref} in the reference, {n_cur} now; {F.RELOC_MIN_POINTS} needed)"
    if frac < F.RELOC_MIN_INLIER_FRAC:
        return (
            f"only {frac * 100:.0f}% of the fridge's points line up within {F.RELOC_INLIER_M * 100:.0f} cm "
            f"({F.RELOC_MIN_INLIER_FRAC * 100:.0f}% needed)"
        )
    if med > F.RELOC_MAX_MEDIAN_M:
        return f"its points are {med * 1000:.0f} mm apart (median; at most {F.RELOC_MAX_MEDIAN_M * 1000:.0f})"
    return ""


# --- on the robot -------------------------------------------------------------------------------------------------


def _grab():
    """A wrist frame and SAM 3's mask of RELOC_SAM3_PROMPTS on it (the union of its instances; None when SAM 3 isn't
    there or finds none -- the depth alone is used then), and how the mask came out, in words."""
    cam = dc.camera()
    saved = cam.sam3_prompts
    cam.sam3_prompts = list(F.RELOC_SAM3_PROMPTS)  # this one question; the handle's prompts back after it
    try:
        instances, frame = cam.sam3_detect()
    finally:
        cam.sam3_prompts = saved
    if instances and frame is not None and frame.get("base_T_cam") is not None:
        mask = np.zeros(instances[0][1].shape, bool)
        for _, m, _ in instances:
            mask |= m
        return frame, mask, f"SAM 3 '{' / '.join(F.RELOC_SAM3_PROMPTS)}': {len(instances)} instance(s)"
    return dc.grab_frames(1)[0], None, "depth only (no SAM 3 mask)"


def _points(ai, margin, boxes):
    """The fridge's points in a new wrist frame: inside SAM 3's mask when it gives enough of them, else depth only;
    within `margin` of `boxes`. Returns (points, how)."""
    frame, mask, how = _grab()
    ee = ai.get_state()["ee_pos"]
    pts = crop(frame_points(frame, ee, mask), boxes, margin) if mask is not None else None
    if pts is None or len(pts) < F.RELOC_MIN_POINTS:
        if pts is not None:
            how = f"{how}, only {len(pts)} points in it: depth only"
        pts = crop(frame_points(frame, ee), boxes, margin)
    return pts, how


def capture_reference(ai):
    """Keep the wrist camera's view of the fridge as the reference the close re-detects it against, with the door file
    as it is now (call when an executed open ends: the door file is then true). Never raises: without a reference the
    close plans against the door file as before."""
    try:
        door = load_door_or_none()
        if door is None:
            return
        pts, how = _points(ai, F.RELOC_REF_MARGIN_M, fridge_boxes(door))
        ref_file().unlink(missing_ok=True)
        if len(pts) < F.RELOC_MIN_POINTS:
            say(f"fridge reference NOT kept: {len(pts)} points on the fridge ({how}); the close won't re-detect it")
            return
        np.savez(
            ref_file(),
            points=pts.astype(np.float32),
            hinge=np.asarray(door["hinge"], float),
            door_open_deg=float(door["door_open_deg"]),
            taken=time.time(),
        )
        say(f"fridge reference kept for the close's re-detection: {len(pts)} points ({how})")
    except Exception as e:  # noqa: BLE001 -- the open has finished either way
        say(f"fridge reference NOT kept ({e!r}); the close won't re-detect the fridge")


def load_door_or_none():
    """The door file, or None without one."""
    return json.loads(dc.DOOR_FILE.read_text()) if dc.DOOR_FILE.exists() else None


def relocalize(ai, door):
    """`door` carried to where the fridge is now (the wheelchair moved since the open), and the move found (dx, dy,
    yaw). Without a reference for this door file it says so and returns (door, None): the close then plans against
    the door file as before. Raises Refused when the fit is not trusted (trusted)."""
    path = ref_file()
    if not path.exists():
        say("  fridge NOT re-detected: no reference from the open (an open from before this, or a dry run)")
        return door, None
    ref = np.load(path)
    if float(np.linalg.norm(np.asarray(ref["hinge"], float)[:2] - np.asarray(door["hinge"], float)[:2])) > 1e-3:
        say("  fridge NOT re-detected: the reference is from another door file (an open since it, or a dry run's)")
        return door, None
    t0 = time.monotonic()
    ref_pts = np.asarray(ref["points"], float)
    # where the fridge may be now: within the search's reach of where it was
    cur, how = _points(ai, max(F.RELOC_SEARCH_M) + F.RELOC_REF_MARGIN_M, fridge_boxes(door))
    if min(len(ref_pts), len(cur)) < F.RELOC_MIN_POINTS:
        raise Refused(
            f"re-detecting the fridge: {trusted(0.0, 1.0, len(ref_pts), len(cur))} -- is the wrist camera facing it? "
            "(relocalize false plans against the door file as before)"
        )
    x, med, frac = align(ref_pts, cur)
    say(
        f"  fridge re-detected: moved {x[0] * 100:+.1f} cm along x, {x[1] * 100:+.1f} cm along y, turned "
        f"{np.degrees(x[2]):+.1f} deg since the open; {frac * 100:.0f}% of its points within "
        f"{F.RELOC_INLIER_M * 100:.0f} cm, median {med * 1000:.0f} mm ({how}; {len(ref_pts)} points then, {len(cur)} "
        f"now; {time.monotonic() - t0:.1f} s)"
    )
    why = trusted(med, frac, len(ref_pts), len(cur))
    if why:
        raise Refused(
            f"re-detecting the fridge: {why} -- not trusted. Is the wrist camera facing the fridge? "
            "(relocalize false plans against the door file as before)"
        )
    return moved_door(door, x), x
