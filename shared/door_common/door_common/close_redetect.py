"""Before the close: find the open door again, from a fresh look, and move the door model to where it is.

The close plans against the door model the open saved (door file: hinge, closed face, door height, sizes). The
appliance or the robot may have moved since, and the push-open's 90 deg is taken, not measured. So the close looks
again, as the open does: the scene camera (RF-DETR) says where the microwave is, the wrist camera at the view pose
measures it. From the wrist camera's points (the arm's own links taken out, `drop_near`) it finds:

* the appliance's FRONT FACE's depth -- the frontmost flat surface parallel to the saved closed door, i.e. the control
  panel: 11-16 it sat within 2 mm of the closed door's plane, and read the same with the door open or closed, while the
  frame round the opening is 2 cm further back. Its DIRECTION is the saved one: the wrist detection measured it on the
  whole closed door; the panel left visible with the door open (16 cm wide) gave it 5 deg apart in two looks;
* the OPEN DOOR -- seen nearly edge-on from the view pose (11-16: a narrow strip from the hinge toward the camera), so
  as a LINE from above: the x, y of the points in front of the face, a long straight run turned well away from the
  face and reaching back to it.

The door's line gives the opening angle; where it meets the face, the hinge. The model is shifted (not turned) to put
its hinge there and its face at the measured depth: its sizes, height and the handle's place on the door stay the
open's.
Several looks must agree (`combine`); nothing is compared against an
earlier look.

Pure numpy (its own RANSAC for the door's line): the tests feed it points sampled off planes; tests/arm also runs
it on the 11-16 frames.
"""

import numpy as np

CROP_PAST_DOOR_M = 0.25  # points within the door's length + this of the saved hinge (horizontally) ...
CROP_Z_LOW_M = 0.03  # ... from this far above the saved door's bottom (no table) ...
CROP_Z_HIGH_M = 0.05  # ... to this far above its top: from the view pose, looking down at a door nearly edge-on, its
# TOP EDGE is what the wrist camera sees of it (11-16: z 0.43-0.48, the saved top 0.488)
EDGE_ON_M = 0.03  # the camera this close to the door's plane sees it edge-on: which face is unknowable, so its middle
# (at most half the door's thickness off, not all of it)
TOP_EDGE_M = 0.04  # door points whose median z is within this of the saved top are its top edge: the door's middle
ARM_CLEAR_M = 0.09  # points this close to the arm's links (drop_near) are the arm, not the appliance
RANSAC_ITERS = 400
PLANE_TOL_M = 0.012  # a point this close to the door's line (from above) is on it
FACE_NEAR_SAVED_M = 0.15  # the face's depth looked for within this of the saved face's ...
FACE_BIN_M = 0.005  # ... in a histogram of depths along the saved normal, this fine ...
FACE_SHEET_M = 0.005  # ... a flat surface: a peak with at least FACE_MIN_POINTS within this of its depth; the
# frontmost peak is the face. Narrow: 11-16 the panel and the frame behind the open door were peaks 2 cm apart (a
# 1 cm window merged them and took the frame, putting the hinge 2.3 cm too deep)
FACE_MIN_POINTS = 300
FACE_MIN_SHARE = 0.35  # ... with at least this share of the strongest such surface's points (11-16 at 94 deg: edge
# pixels between the door and the panel made a sparse "surface" 2.5 cm in front, 191 points against the panel's 1049)
FACE_MIN_WIDTH_M = 0.08  # ... spanning at least this along the face (5th-95th pct): the panel is ~16 cm wide; the open
# door near its hinge, which 11-16 at 94 deg piled up at one depth too, only its thickness (~4 cm)
DOOR_MIN_DEG = 50.0  # the door's line at least this far from the face's
DOOR_IN_FRONT_M = 0.03  # its points this far in front of the face, at least ...
DOOR_REACH_FACE_M = 0.06  # ... some within this of it: the door reaches back to the face at the hinge ...
DOOR_MIN_SPAN_M = 0.15  # ... and running at least this far out
DOOR_MIN_POINTS = 150
MAX_LINES = 4  # lines taken off the points in front of the face while looking for the door
OPEN_DEG_RANGE = (30.0, 130.0)  # a measured opening outside this: refuse
# looks (combine): each one measured on its own; they must agree this well, and the median is used
AGREE_HINGE_M = 0.015
AGREE_DEG = 3.0
AGREE_YAW_DEG = 2.0


def _rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _signed_deg(a, b):
    """The angle (deg) from horizontal vector a to b, counter-clockwise seen from above."""
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def _horizontal(v):
    v = np.array([v[0], v[1], 0.0], float)
    return v / np.linalg.norm(v)


def drop_near(pts, centres, radius=ARM_CLEAR_M):
    """`pts` without those within `radius` of any of `centres` (M x 3: points along the arm's links, the gripper's)."""
    pts = np.asarray(pts, float)
    centres = np.asarray(centres, float).reshape(-1, 3)
    if len(centres) == 0 or len(pts) == 0:
        return pts
    keep = np.ones(len(pts), bool)
    for c in centres:  # one centre at a time: no N x M matrix
        keep &= np.sum((pts - c) ** 2, axis=1) > radius**2
    return pts[keep]


def _line_fits(xy, rng, iters=RANSAC_ITERS, tol=PLANE_TOL_M):
    """Lines (from above) taken off the 2D points `xy` biggest first, up to MAX_LINES: [(unit normal, c: n.p = c,
    inlier mask over `xy`)], each refitted to its inliers."""
    found, rest_idx = [], np.arange(len(xy))
    for _ in range(MAX_LINES):
        rest = xy[rest_idx]
        if len(rest) < DOOR_MIN_POINTS:
            break
        best, best_n = None, -1
        for _ in range(iters):
            a, b = rest[rng.choice(len(rest), 2, replace=False)]
            d = b - a
            norm = np.linalg.norm(d)
            if norm < 1e-6:
                continue
            n = np.array([-d[1], d[0]]) / norm
            inl = np.abs(rest @ n - n @ a) <= tol
            k = int(inl.sum())
            if k > best_n:
                best, best_n = inl, k
        if best is None or best_n < DOOR_MIN_POINTS:
            break
        q = rest[best]
        centre = q.mean(axis=0)
        n = np.linalg.svd(q - centre, full_matrices=False)[2][-1]
        inl = np.abs(rest @ n - n @ centre) <= tol
        mask = np.zeros(len(xy), bool)
        mask[rest_idx[inl]] = True
        found.append((n, float(n @ centre), mask))
        rest_idx = rest_idx[~inl]
    return found


def _saved(door, dims):
    n0 = _horizontal(door["closed_normal"])
    hinge0 = np.asarray(door["hinge"], float)
    g0 = np.asarray(door["closed_grasp_pos"], float)
    face0 = g0 - np.r_[n0[:2] * dims[0], 0.0]
    length = float(np.linalg.norm((g0 - hinge0)[:2])) + dims[2]
    return n0, hinge0, g0, face0, length


def measure(door, pts, cam_pos, dims, seed=0):
    """One look: where the face and the open door are in `pts` (N x 3, the arm's frame, the arm already taken out),
    seen from `cam_pos`, against the door file `door` (dict) and `dims` = common.door_dims(door). Returns dict(yaw (deg,
    the face turned from the saved one), deg (the door's opening), shift (x, y m, after the turn), the hinge it puts
    the model's at, and what was seen); raises ValueError (why) when there is no face, no open door, or an
    implausible opening."""
    rng = np.random.default_rng(seed)
    sign = float(door.get("open_sign", 1.0))
    n0, hinge0, _, face0, length = _saved(door, dims)
    door_t = dims[1]
    z_lo, z_hi = door["door_z"] if door.get("door_z") else (hinge0[2] - 0.15, hinge0[2] + 0.15)

    pts = np.asarray(pts, float)
    near = np.linalg.norm(pts[:, :2] - hinge0[:2], axis=1) <= length + CROP_PAST_DOOR_M
    inside = (pts[:, 2] >= z_lo + CROP_Z_LOW_M) & (pts[:, 2] <= z_hi + CROP_Z_HIGH_M)
    pts = pts[near & inside]
    if len(pts) < FACE_MIN_POINTS:
        raise ValueError(f"only {len(pts)} points near the saved door -- is the microwave in the camera's view?")

    # the front face: the frontmost flat surface along the saved normal (the panel), near the saved face's depth
    d = pts @ n0
    c_saved = float(n0 @ face0)
    edges = np.arange(c_saved - FACE_NEAR_SAVED_M, c_saved + FACE_NEAR_SAVED_M + FACE_BIN_M, FACE_BIN_M)
    counts = np.histogram(d, edges)[0]
    k = max(1, int(round(FACE_SHEET_M / FACE_BIN_M)))
    sheet = np.convolve(counts, np.ones(2 * k + 1, int), mode="same")  # points within ~FACE_SHEET_M of each bin
    t0 = np.array([-n0[1], n0[0], 0.0])  # along the face

    def wide(i):
        centre = (edges[i] + edges[i + 1]) / 2
        q = pts[np.abs(d - centre) <= FACE_SHEET_M] @ t0
        return len(q) > 0 and float(np.percentile(q, 95) - np.percentile(q, 5)) >= FACE_MIN_WIDTH_M

    peaks = [
        i
        for i in range(len(sheet))
        if sheet[i] >= FACE_MIN_POINTS
        and (i == 0 or sheet[i] >= sheet[i - 1])
        and (i == len(sheet) - 1 or sheet[i] > sheet[i + 1])
        and wide(i)
    ]
    if not peaks:
        raise ValueError(
            f"no front face (a flat surface facing the arm, {FACE_MIN_WIDTH_M * 100:.0f} cm wide or more, within "
            f"{FACE_NEAR_SAVED_M * 100:.0f} cm of the saved face's depth) -- is the microwave where the open left it?"
        )
    strongest = max(sheet[i] for i in peaks)
    peaks = [i for i in peaks if sheet[i] >= FACE_MIN_SHARE * strongest]
    peak = peaks[-1]  # the frontmost: depth along n0 grows toward the arm
    centre = (edges[peak] + edges[peak + 1]) / 2
    q_f = pts[np.abs(d - centre) <= FACE_SHEET_M]
    n_f, c_f = n0, float(np.median(q_f @ n0))

    # the open door, from above: a long straight run in front of the face, turned away from it, reaching back to it
    ahead = pts[pts @ n_f - c_f >= DOOR_IN_FRONT_M]
    cos_door = np.cos(np.radians(DOOR_MIN_DEG))
    why_not = []
    door_line = None
    for n, c, mask in _line_fits(ahead[:, :2], rng):
        q = ahead[mask]
        n3 = np.r_[n, 0.0]
        along = np.array([n[1], -n[0], 0.0])
        out = q @ n_f - c_f
        span = float(np.percentile(q @ along, 95) - np.percentile(q @ along, 5))
        if abs(n3 @ n_f) > cos_door:
            why_not.append(f"{len(q)} pts {np.degrees(np.arccos(min(abs(n3 @ n_f), 1))):.0f} deg off the face's line")
        elif float(np.min(out)) > DOOR_REACH_FACE_M:
            why_not.append(f"{len(q)} pts not reaching the face (nearest {np.min(out) * 100:.0f} cm)")
        elif span < DOOR_MIN_SPAN_M:
            why_not.append(f"{len(q)} pts only {span * 100:.0f} cm long")
        else:
            door_line = (n3, c, q)
            break
    if door_line is None:
        raise ValueError(
            f"no open door in front of the face ({len(ahead)} points there"
            + (f"; lines seen: {'; '.join(why_not)}" if why_not else "")
            + ") -- is the door open?"
        )
    n_s, _, q_s = door_line
    deg_file = float(door.get("door_open_deg", 90.0))
    outer_guess = np.r_[_rot(sign * deg_file) @ n0[:2], 0.0]  # the outer face's normal, opened as the file says
    if n_s @ outer_guess < 0:
        n_s = -n_s
    c_s = float(np.median(q_s @ n_s))
    if float(np.median(q_s[:, 2])) >= z_hi - TOP_EDGE_M:
        seen = "top edge"  # across the door's thickness: its middle
        c_outer = c_s + door_t / 2
    elif abs(float(n_s @ cam_pos) - c_s) < EDGE_ON_M:
        seen = "edge-on"
        c_outer = c_s + door_t / 2
    else:
        seen = "outer" if float(n_s @ cam_pos) - c_s >= 0 else "inner"
        c_outer = c_s if seen == "outer" else c_s + door_t  # the door's body is on the outer face's -n side

    # the opening from the door's line; the hinge where the door's outer face meets the front face (the door's outer
    # corner on the hinge side: what the open saves as `hinge`, the detected door edge)
    deg = sign * _signed_deg(n_f[:2], n_s[:2])
    if not OPEN_DEG_RANGE[0] <= deg <= OPEN_DEG_RANGE[1]:
        raise ValueError(f"the door is open {deg:.0f} deg (outside {OPEN_DEG_RANGE}) -- not the door?")
    hinge = np.linalg.solve(np.array([n_f[:2], n_s[:2]]), np.array([c_f, c_outer]))
    return {
        "yaw": 0.0,
        "deg": float(deg),
        "shift": [float(v) for v in hinge - hinge0[:2]],
        "hinge": [float(v) for v in hinge],
        "face_depth": float(c_f),
        "face_points": int(len(q_f)),
        "door_points": int(len(q_s)),
        "door_face_seen": seen,
    }


def move_door(door, m):
    """The door file `door` (dict) moved as measurement `m` (measure's, or combine's median) says: shifted m["shift"]
    (its hinge onto the measured one), door_open_deg m["deg"], and its face at the measured depth (face_behind_grasp:
    the model's face is the grasp less it, so the close's door model has its face where the camera saw it, whatever
    the default -- 11-16 the default put it 2.9 cm behind the door's measured face). Heights unchanged."""
    moved = dict(door)
    dxy = np.asarray(m["shift"], float)
    for k in ("hinge", "closed_grasp_pos", "closed_handle", "door_mid"):
        if door.get(k) is not None:
            p = np.asarray(door[k], float)
            moved[k] = [float(p[0] + dxy[0]), float(p[1] + dxy[1]), *[float(v) for v in door[k][2:]]]
    n0 = _horizontal(door["closed_normal"])
    moved["face_behind_grasp"] = round(
        float(n0[:2] @ np.asarray(moved["closed_grasp_pos"][:2])) - float(m["face_depth"]), 4
    )
    moved["door_open_deg"] = round(float(m["deg"]), 1)
    return moved


def combine(looks):
    """Several looks' measurements -> their median, once they agree (hinge within AGREE_HINGE_M of the median's, the
    opening within AGREE_DEG, the yaw within AGREE_YAW_DEG); raises ValueError (how they disagree) otherwise."""
    if not looks:
        raise ValueError("no look found the door")
    hinge = np.median([m["hinge"] for m in looks], axis=0)
    deg = float(np.median([m["deg"] for m in looks]))
    yaw = float(np.median([m["yaw"] for m in looks]))
    off = [
        (float(np.linalg.norm(np.asarray(m["hinge"]) - hinge)), abs(m["deg"] - deg), abs(m["yaw"] - yaw)) for m in looks
    ]
    worst = tuple(max(o[i] for o in off) for i in range(3))
    if worst[0] > AGREE_HINGE_M or worst[1] > AGREE_DEG or worst[2] > AGREE_YAW_DEG:
        raise ValueError(
            f"the {len(looks)} looks disagree: hinge up to {worst[0] * 100:.1f} cm apart "
            f"(max {AGREE_HINGE_M * 100:.1f}), "
            f"opening {worst[1]:.1f} deg (max {AGREE_DEG:g}), face {worst[2]:.1f} deg (max {AGREE_YAW_DEG:g})"
        )
    shift = np.median([m["shift"] for m in looks], axis=0)
    return {
        "face_depth": float(np.median([m["face_depth"] for m in looks])),
        "yaw": yaw,
        "deg": deg,
        "shift": [float(v) for v in shift],
        "hinge": [float(v) for v in hinge],
        "looks": len(looks),
        "spread": {"hinge_m": round(worst[0], 4), "deg": round(worst[1], 1), "yaw_deg": round(worst[2], 1)},
        "door_face_seen": looks[0]["door_face_seen"],
    }


def frame_points(frame, stride=2, box2d=None):
    """A camera frame (WristCamera.grab's dict) as points in the arm's frame (N x 3), every `stride`-th pixel, inside
    `box2d` ([x0, y0, x1, y1] pixels) if given; None without its arm base <- camera transform."""
    T = frame.get("base_T_cam")
    if T is None:
        return None
    depth = np.asarray(frame["depth_mm"], float) / 1000.0
    K = np.asarray(frame["K"], float)
    vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    d = depth[vs, us]
    ok = np.isfinite(d) & (d > 0.05) & (d < 2.5)
    if box2d is not None:
        x0, y0, x1, y1 = box2d
        ok &= (us >= x0) & (us <= x1) & (vs >= y0) & (vs <= y1)
    d, us, vs = d[ok], us[ok], vs[ok]
    cam = np.stack([d * (us - K[0, 2]) / K[0, 0], d * (vs - K[1, 2]) / K[1, 1], d], axis=1)
    T = np.asarray(T, float)
    return cam @ T[:3, :3].T + T[:3, 3]
