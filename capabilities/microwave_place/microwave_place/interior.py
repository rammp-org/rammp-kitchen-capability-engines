"""Model-based estimate of an open microwave's interior box from one RGB-D frame.

Pure numpy/scipy (no ROS, no cv2). Entry point:
    estimate_interior(color_bgr, depth_m, K, world_T_cam, objects_3d) -> dict

Model: the oven is a box standing on a level table; its front is vertical and faces the camera; the
cavity is a box (same yaw as the oven) behind a rectangular opening. With the door open, the frame
around the opening sits a few cm behind the outer front (control panel) - the door thickness - so
the front is modelled as two parallel planes.

Oven frame used internally (horizontal axes lie in the gravity-aligned world xy plane):
    n_in : horizontal unit normal of the front pointing INTO the oven (= insertion axis)
    w    : z x n_in, horizontal, points to the oven's LEFT as seen from the front (image left)
    A    : coordinate along w,  S : coordinate along n_in,  Z : world z,  dS = S - S(opening plane)

Pipeline
  1. ROI + weak priors from the "microwave" detection (3D box if it has a pose, else 2D box +
     camera bearing).
  2. Yaw (mod 90 deg): global Manhattan vote of all vertical-surface normals on the oven, then
     circular-mean refine.
  3. For each of the (usually two) camera-facing box faces: outer front plane = first strong mode
     of S over front-facing points; opening plane = deepest strong mode <= 7 cm behind it (door
     recess); table = dominant level surface just below/in front; opening = max-sum rectangle in
     (A, Z) of "hole" vs "frame" pixels, where each pixel's camera ray is intersected with the
     opening plane. The face with the best opening is the front.
  4. Interior surfaces measured where visible: side wall (normal ~ w), back wall (normal ~ n_in),
     floor and ceiling (level points); otherwise the opening edge (+ a view-geometry correction)
     or the prior is used. Every quantity is reported under 'measured' or 'assumed'.
"""

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

# ---------------- priors / thresholds (metres) ----------------
PRIOR = dict(
    interior_w=(0.24, 0.40, 0.30),  # (min, max, nominal)
    interior_d=(0.22, 0.36, 0.29),  # opening plane -> back wall
    interior_h=(0.14, 0.26, 0.19),
    floor_above_table=(0.02, 0.09, 0.055),
    outer_h=(0.20, 0.36, 0.27),
)
HOLE_T = 0.03  # a point this far behind the opening plane is "seen through the opening"
FRAME_T = 0.015  # a point this close to either front plane is front material
# the frame around the opening may sit up to this far behind the outer front (door thickness)
RECESS_MAX = 0.07
CELL = 0.005  # grid cell for the opening search


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


def _pick_microwave(objects_3d):
    """Among 'microwave' detections prefer one with a 3D pose whose extents look like a whole oven
    (the open door is often detected as a second, smaller 'microwave')."""
    mws = [o for o in objects_3d if str(o.get("cls", "")).lower() == "microwave"]
    if not mws:
        return None

    def score(o):
        if not o.get("has_pose"):
            return -1.0 + 0.1 * float(o.get("score", 0))
        s = np.sort(np.asarray(o["size"], float))[::-1]
        return float(o.get("score", 0)) - 2.0 * np.abs(s - np.array([0.46, 0.31, 0.27])).sum()

    return max(mws, key=score)


def _max_sum_rect(G):
    """Maximum-sum axis-aligned sub-rectangle of G (rows x cols).
    Returns (value, r0, r1, c0, c1), inclusive."""
    R, C = G.shape
    cs = np.vstack([np.zeros((1, C)), np.cumsum(G, 0)])
    best = (-np.inf, 0, 0, 0, 0)
    idx = np.arange(C + 1)[None, :]
    for r0 in range(R):
        rows = cs[r0 + 1 :] - cs[r0]  # column sums for rows r0..r0+k
        pre = np.hstack([np.zeros((rows.shape[0], 1)), np.cumsum(rows, 1)])
        runmin = np.minimum.accumulate(pre, axis=1)
        argmin = np.maximum.accumulate(np.where(pre <= runmin, idx, 0), axis=1)
        gain = pre[:, 1:] - runmin[:, :-1]  # best sum of a run ending at column j
        k, j = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[k, j] > best[0]:
            best = (float(gain[k, j]), r0, r0 + k, int(argmin[k, j]), int(j))
    return best


def _manhattan_yaw(nxy, yaw0, prior_weight=0.15, iters=4):
    """Yaw (rad, defined mod 90 deg) of the dominant rectangular structure from horizontal normals
    (N,2). Global vote over the 90 deg period (+-4 deg window, weak preference for yaw0), then
    circular-mean refinement.
    Returns (yaw, n_used, residual_std_deg) or None."""
    th = np.arctan2(nxy[:, 1], nxy[:, 0])
    if th.size < 50:
        return None
    grid = np.radians(np.arange(-45.0, 45.0, 1.0))
    votes = np.zeros(len(grid))
    for s in range(0, th.size, 20000):
        d = np.angle(np.exp(4j * (th[None, s : s + 20000] - grid[:, None]))) / 4.0
        votes += (np.abs(d) < np.radians(4.0)).sum(1)
    votes *= 1.0 + prior_weight * np.cos(4 * (grid - yaw0))
    yaw = grid[int(np.argmax(votes))]
    n_used, spread = 0, np.nan
    for _ in range(iters):
        r = np.angle(np.exp(4j * (th - yaw))) / 4.0
        m = np.abs(r) < np.radians(8.0)
        if m.sum() < 50:
            return None
        yaw = yaw + np.angle(np.mean(np.exp(4j * r[m]))) / 4.0
        n_used, spread = int(m.sum()), float(np.degrees(np.std(r[m])))
    return yaw, n_used, spread


def _mode_median(x, bin_w, pick="max", min_frac=0.3, halfwin=None):
    """Histogram mode of x; pick='max' (largest bin) or 'first' (lowest-valued bin with >= min_frac
    of max, climbed to its local peak). Returns the median of samples within +-halfwin of that bin
    centre."""
    if x.size == 0:
        return None
    lo, hi = float(np.min(x)), float(np.max(x))
    edges = np.arange(lo, hi + bin_w, bin_w)
    if len(edges) < 2:
        return float(np.median(x))
    h, e = np.histogram(x, bins=edges)
    if pick == "first":
        k = int(np.flatnonzero(h >= min_frac * h.max())[0])
        while k + 1 < len(h) and h[k + 1] > h[k]:
            k += 1
    else:
        k = int(np.argmax(h))
    c = 0.5 * (e[k] + e[k + 1])
    hw = halfwin if halfwin is not None else bin_w * 1.5
    return float(np.median(x[np.abs(x - c) < hw]))


def estimate_interior(color_bgr, depth_m, K, world_T_cam, objects_3d, debug=False):
    """color_bgr (H,W,3) is only used for its shape (geometry-only method).
    depth_m: (H,W) metres aligned to color, 0 = invalid. K: 3x3.
    world_T_cam: 4x4 camera-optical -> world (z up).
    objects_3d: list of dicts (cls, score, box [x1,y1,x2,y2], has_pose, pos, quat xyzw, size;
    pose in camera optical).
    Returns a dict; on failure 'error' is a string and the geometric keys are absent."""
    out = dict(error=None, measured={}, assumed={}, notes=[])
    K = np.asarray(K, float).reshape(3, 3)
    T = np.asarray(world_T_cam, float)
    depth = np.asarray(depth_m, np.float32)
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 3.0)
    Pc = _backproject(np.where(valid, depth, 0), K)
    Pw = Pc @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
    cam = T[:3, 3]
    Nw, nok = _normals(Pw, valid)
    nz = np.abs(Nw[..., 2])
    level = nok & (nz > 0.93)  # surface within ~21 deg of horizontal
    vert = nok & (nz < 0.30)  # surface within ~17 deg of vertical
    Z = Pw[..., 2]
    out["camera"] = dict(
        position_world=cam.tolist(),
        pitch_down_deg=float(np.degrees(np.arcsin(np.clip(-T[2, 2], -1, 1)))),
    )

    # ---------- 1. ROI + priors from the microwave detection ----------
    mw = _pick_microwave(objects_3d)
    if mw is None:
        out["error"] = "no microwave detection"
        return out
    x1, y1, x2, y2 = (float(t) for t in mw["box"])
    bw, bh = x2 - x1, y2 - y1
    box2d = np.zeros((H, W), bool)
    box2d[
        int(max(0, y1 - 0.03 * bh)) : int(min(H, y2 + 0.03 * bh)),
        int(max(0, x1 - 0.03 * bw)) : int(min(W, x2 + 0.03 * bw)),
    ] = True
    roi = np.zeros((H, W), bool)
    roi[
        int(max(0, y1 - 0.1 * bh)) : int(min(H, y2 + 0.35 * bh)),
        int(max(0, x1 - 0.1 * bw)) : int(min(W, x2 + 0.1 * bw)),
    ] = True
    if mw.get("has_pose"):
        Rw = T[:3, :3] @ _Rot.from_quat(mw["quat"]).as_matrix()
        cw = T[:3, :3] @ np.asarray(mw["pos"], float) + T[:3, 3]
        size = np.asarray(mw["size"], float)
        bear = cw[:2] - cam[:2]
        bear /= np.linalg.norm(bear)
        hor = [k for k in range(3) if abs(Rw[2, k]) < 0.5]
        if len(hor) == 2 and min(size[hor]) < 0.85 * max(size[hor]):
            kd = hor[int(np.argmin(size[hor]))]  # depth axis = shorter horizontal extent
        else:
            kd = max(
                hor, key=lambda k: abs(Rw[:2, k] @ bear) / max(np.linalg.norm(Rw[:2, k]), 1e-9)
            )
        n_prior = Rw[:2, kd] / np.linalg.norm(Rw[:2, kd])
        n_prior = n_prior * np.sign(n_prior @ bear)
        half_xy = 0.5 * max(size[k] for k in hor) + 0.06
    else:
        P = Pw[box2d & valid]
        if len(P) < 100:
            out["error"] = "no depth in microwave 2D box"
            return out
        cw = np.median(P, 0)
        n_prior = cw[:2] - cam[:2]
        n_prior /= np.linalg.norm(n_prior)
        half_xy = 0.35
        out["notes"].append("microwave detection has no 3D pose; yaw prior = camera bearing")
    bear_ang = float(np.arctan2(cw[1] - cam[1], cw[0] - cam[0]))
    near_det = (
        (np.abs(Pw[..., 0] - cw[0]) < half_xy)
        & (np.abs(Pw[..., 1] - cw[1]) < half_xy)
        & (np.abs(Z - cw[2]) < 0.3)
    )
    yaw0 = float(np.arctan2(n_prior[1], n_prior[0]))
    out["detection"] = dict(
        box=[x1, y1, x2, y2],
        score=float(mw.get("score", 0)),
        center_world=cw.tolist(),
        yaw_deg=float(np.degrees(yaw0)),
    )

    # ---------- 2. Yaw (mod 90) ----------
    cand = box2d & vert & near_det
    nxy = Nw[cand][:, :2]
    nxy = nxy / np.maximum(np.linalg.norm(nxy, axis=1, keepdims=True), 1e-9)
    my = _manhattan_yaw(nxy, yaw0)
    if my is None:
        out["error"] = "could not estimate oven yaw from vertical surfaces"
        return out
    yaw_m, n_yaw, yaw_spread = my

    # ---------- 3. Front face + opening, for each camera-facing face ----------
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rcam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    rw = rcam @ T[:3, :3].T  # pixel rays in world

    def face(yaw):
        f = dict(yaw=yaw)
        n3 = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        w3 = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # w3 = z x n3
        A = Pw[..., 0] * w3[0] + Pw[..., 1] * w3[1]
        S = Pw[..., 0] * n3[0] + Pw[..., 1] * n3[1]
        nS = np.abs(Nw[..., 0] * n3[0] + Nw[..., 1] * n3[1])
        nA = np.abs(Nw[..., 0] * w3[0] + Nw[..., 1] * w3[1])
        a_cam, s_cam = float(cam[:2] @ w3[:2]), float(cam[:2] @ n3[:2])
        frontish = box2d & near_det & vert & (nS > 0.9)
        if frontish.sum() < 200:
            f["error"] = "too few front-facing points"
            return f
        s_outer = _mode_median(S[frontish], 0.005, pick="first", min_frac=0.35, halfwin=0.008)
        rel = S[frontish] - s_outer
        h, e = np.histogram(rel, bins=np.arange(-0.0125, RECESS_MAX + 0.0125, 0.005))
        hs = np.convolve(h, [1, 1, 1], "same")
        peaks = [
            k
            for k in range(len(hs))
            if hs[k] >= 0.15 * hs.max()
            and hs[k] >= hs[max(k - 1, 0)]
            and hs[k] >= hs[min(k + 1, len(hs) - 1)]
        ]
        recess = 0.0
        if peaks:
            c = 0.5 * (e[max(peaks)] + e[max(peaks) + 1])
            if c > 0.012:
                recess = float(np.median(rel[np.abs(rel - c) < 0.008]))
        s_front = s_outer + recess
        if s_front - s_cam < 0.15:
            f["error"] = "front face implausibly close to the camera"
            return f
        dS = S - s_front
        dS_outer = S - s_outer
        front_mask = frontish & ((np.abs(dS) < 0.01) | (np.abs(dS_outer) < 0.01))
        if front_mask.sum() < 100:
            f["error"] = "too few front-plane points"
            return f
        fA_lo, fA_hi = np.percentile(A[front_mask], [1, 99])
        fZ_lo, fZ_hi = np.percentile(Z[front_mask], [1, 99.5])
        # table: dominant level surface just below / in front of / beside the oven
        near = (
            roi
            & level
            & (dS > -0.35)
            & (dS < 0.45)
            & (A > fA_lo - 0.25)
            & (A < fA_hi + 0.25)
            & (Z < fZ_lo + 0.05)
        )
        table_z = (
            _mode_median(Z[near], 0.005, pick="max", halfwin=0.01) if near.sum() > 200 else None
        )
        top_z = float(fZ_hi)
        # opening: intersect every pixel ray with the opening plane
        denom = rw[..., 0] * n3[0] + rw[..., 1] * n3[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            tpl = (s_front - s_cam) / denom
        Xp = cam + tpl[..., None] * rw
        Ap = Xp[..., 0] * w3[0] + Xp[..., 1] * w3[1]
        Zp = Xp[..., 2]
        z_floor_guess = table_z if table_z is not None else float(fZ_lo)
        a_lo, a_hi = fA_lo - 0.03, fA_hi + 0.03
        z_lo, z_hi = z_floor_guess, top_z + 0.01
        inwin = (tpl > 0) & (Ap > a_lo) & (Ap < a_hi) & (Zp > z_lo) & (Zp < z_hi)
        is_frame = inwin & valid & ((np.abs(dS) < FRAME_T) | (np.abs(dS_outer) < FRAME_T))
        is_hole = inwin & valid & (dS > HOLE_T)
        nAc = max(1, int(np.ceil((a_hi - a_lo) / CELL)))
        nZc = max(1, int(np.ceil((z_hi - z_lo) / CELL)))

        def grid(mask):
            g = np.zeros((nZc, nAc))
            ia = ((Ap[mask] - a_lo) / CELL).astype(int).clip(0, nAc - 1)
            iz = ((z_hi - Zp[mask]) / CELL).astype(int).clip(0, nZc - 1)  # row 0 = top
            np.add.at(g, (iz, ia), 1)
            return g

        gh, gf = grid(is_hole), grid(is_frame)
        tot = gh + gf
        score = np.where(tot > 0, (gh - 2.0 * gf) / np.maximum(tot, 1), 0.0)
        val, r0, r1, c0, c1 = _max_sum_rect(score)
        op = dict(
            top=z_hi - r0 * CELL,
            bot=z_hi - (r1 + 1) * CELL,
            left=a_lo + (c1 + 1) * CELL,
            right=a_lo + c0 * CELL,
        )
        f.update(
            n3=n3,
            w3=w3,
            A=A,
            S=S,
            nS=nS,
            nA=nA,
            a_cam=a_cam,
            s_cam=s_cam,
            s_outer=s_outer,
            recess=recess,
            s_front=s_front,
            dS=dS,
            front_mask=front_mask,
            fA=(float(fA_lo), float(fA_hi)),
            table_z=table_z,
            fZ_lo=float(fZ_lo),
            top_z=top_z,
            Ap=Ap,
            Zp=Zp,
            inwin=inwin,
            score=score,
            op=op,
            op_value=val,
        )
        if (op["left"] - op["right"]) < 0.15 or (op["top"] - op["bot"]) < 0.08:
            f["error"] = (
                "opening not found ({:.2f} x {:.2f} m) - door closed or oven not facing "
                "camera?".format(op["left"] - op["right"], op["top"] - op["bot"])
            )
        return f

    faces = [yaw_m + k * np.pi / 2 for k in range(4)]
    faces = [
        y for y in faces if np.cos(y - bear_ang) > 0.2
    ]  # insertion axis points away from the camera
    faces.sort(key=lambda y: -np.cos(y - yaw0))
    hyps = [face(y) for y in faces]
    good = [f for f in hyps if not f.get("error")]
    out["measured"]["face_hypotheses"] = [
        dict(
            yaw_deg=float(np.degrees(f["yaw"])),
            error=f.get("error"),
            opening_score=float(f.get("op_value", 0.0)),
        )
        for f in hyps
    ]
    if not good:
        out["error"] = hyps[0]["error"] if hyps else "no camera-facing face"
        return out
    F = max(good, key=lambda f: f["op_value"])
    yaw, n3, w3, A, dS, nS, nA = (F[k] for k in ("yaw", "n3", "w3", "A", "dS", "nS", "nA"))
    a_cam, s_cam, s_front, op = F["a_cam"], F["s_cam"], F["s_front"], F["op"]
    op_top, op_bot, op_left, op_right = op["top"], op["bot"], op["left"], op["right"]
    top_z = F["top_z"]
    out["measured"]["yaw_deg"] = float(np.degrees(yaw))
    out["measured"]["yaw_support"] = dict(
        n_normals=n_yaw,
        residual_std_deg=yaw_spread,
        diff_vs_detection_deg=float(np.degrees(np.angle(np.exp(1j * (yaw - yaw0))))),
    )
    out["measured"]["front_face"] = dict(
        n_points=int(F["front_mask"].sum()),
        outer_dist_from_camera=float(F["s_outer"] - s_cam),
        A_range=list(F["fA"]),
        frame_recess_behind_outer=F["recess"],
    )
    if F["table_z"] is not None:
        table_z = float(F["table_z"])
        out["measured"]["table_z"] = table_z
    else:
        table_z = F["fZ_lo"]
        out["assumed"]["table_z"] = "lowest front-face point (no level table points found)"
    out["measured"]["outer_top_z"] = top_z
    if not (PRIOR["outer_h"][0] <= top_z - table_z <= PRIOR["outer_h"][1]):
        out["notes"].append("outer height %.3f outside prior" % (top_z - table_z))
    out["measured"]["opening"] = dict(
        A_right=float(op_right),
        A_left=float(op_left),
        z_bottom=float(op_bot),
        z_top=float(op_top),
        width=float(op_left - op_right),
        height=float(op_top - op_bot),
    )
    Ap, Zp, inwin = F["Ap"], F["Zp"], F["inwin"]
    through = inwin & (Ap > op_right) & (Ap < op_left) & (Zp > op_bot) & (Zp < op_top)
    inside = through & valid & (dS > HOLE_T)
    D = s_front - s_cam  # camera distance in front of the opening plane

    def edge_shift(dist_cam_to_edge):
        # with the adjoining wall visible, the hole boundary appears shifted towards the opening
        # centre
        return abs(dist_cam_to_edge) * HOLE_T / (D + HOLE_T)

    # ---------- 4a. Back wall ----------
    zm, zh = 0.5 * (op_top + op_bot), op_top - op_bot
    bk = inside & vert & (nS > 0.85) & (dS > 0.12) & (np.abs(Z - zm) < 0.4 * zh)
    lo, hi, nom = PRIOR["interior_d"]
    depth_in = None
    if bk.sum() > 60:
        depth_in = float(np.median(dS[bk]))
        out["measured"]["back_wall"] = dict(
            n=int(bk.sum()),
            depth=depth_in,
            A_range=[float(np.percentile(A[bk], 5)), float(np.percentile(A[bk], 95))],
        )
    if depth_in is None or not (lo <= depth_in <= hi):
        out["assumed"]["interior_depth"] = "prior {:.2f} m (back wall {})".format(
            nom,
            "not seen" if depth_in is None else "measured %.3f, rejected" % depth_in,
        )
        depth_in = nom

    # ---------- 4b. Side walls ----------
    side = (
        inside
        & vert
        & (nA > 0.85)
        & (dS > 0.04)
        & (dS < depth_in - 0.04)
        & (Z > op_bot + 0.02)
        & (Z < op_top - 0.02)
    )
    am = 0.5 * (op_left + op_right)
    walls = {}
    for name, sgn, edge in (("left", +1, op_left), ("right", -1, op_right)):
        visible = (
            sgn * (edge - a_cam) > 0.02
        )  # camera on the other side of the opening -> wall visible
        m = side & (sgn * (A - am) > 0)
        from_open = edge + sgn * (edge_shift(edge - a_cam) if visible else 0.0)
        if visible and m.sum() > 80:
            meas = float(np.median(A[m]))
            walls[name] = meas
            out["measured"]["%s_wall" % name] = dict(
                A=meas, n=int(m.sum()), opening_edge_estimate=float(from_open)
            )
            if abs(meas - from_open) > 0.03:
                out["notes"].append(
                    f"{name} wall measured {meas:.3f} vs opening-edge estimate {from_open:.3f}"
                )
        else:
            walls[name] = float(from_open)
            out["assumed"]["%s_wall" % name] = (
                "= opening edge (wall hidden from camera)"
                if not visible
                else "= opening edge + view correction (too few wall points)"
            )
    # a wall is never well inside the opening, nor far outside it
    wall_left = float(np.clip(walls["left"], op_left - 0.01, op_left + 0.06))
    wall_right = float(np.clip(walls["right"], op_right - 0.06, op_right + 0.01))

    # ---------- 4c. Floor ----------
    fl = (
        inside
        & level
        & (Z > table_z + 0.01)
        & (Z < op_bot + 0.03)
        & (A > wall_right + 0.015)
        & (A < wall_left - 0.015)
        & (dS < depth_in - 0.02)
    )
    floor_from_open = op_bot - (edge_shift(cam[2] - op_bot) if cam[2] > op_bot else 0.0)
    if fl.sum() > 40:
        floor_z = float(np.median(Z[fl]))
        out["measured"]["floor"] = dict(
            z=floor_z,
            n=int(fl.sum()),
            z_p10=float(np.percentile(Z[fl], 10)),
            z_p90=float(np.percentile(Z[fl], 90)),
            dS_range=[float(np.percentile(dS[fl], 5)), float(np.percentile(dS[fl], 95))],
            opening_edge_estimate=float(floor_from_open),
        )
    else:
        floor_z = float(floor_from_open)
        out["assumed"]["floor"] = "= opening bottom edge (too few level points inside)"
    lo, hi, nom = PRIOR["floor_above_table"]
    if not (lo <= floor_z - table_z <= hi):
        out["notes"].append(
            f"floor {floor_z - table_z:.3f} m above table is outside prior "
            f"[{lo:.2f}, {hi:.2f}]; clamped"
        )
        floor_z = float(np.clip(floor_z, table_z + lo, table_z + hi))

    # ---------- 4d. Ceiling ----------
    ce = (
        inside
        & level
        & (Z > op_top - 0.04)
        & (Z < top_z)
        & (A > wall_right + 0.015)
        & (A < wall_left - 0.015)
    )
    ceil_from_open = op_top + (edge_shift(op_top - cam[2]) if cam[2] < op_top else 0.0)
    if ce.sum() > 40:
        ceil_z = float(np.median(Z[ce]))
        out["measured"]["ceiling"] = dict(
            z=ceil_z, n=int(ce.sum()), opening_edge_estimate=float(ceil_from_open)
        )
    else:
        ceil_z = float(ceil_from_open)
        out["assumed"]["ceiling"] = "= opening top edge"

    # ---------- 5. Assemble ----------
    width = wall_left - wall_right
    a_c = 0.5 * (wall_left + wall_right)
    s0, s1 = s_front, s_front + depth_in
    s_c = 0.5 * (s0 + s1)

    def to_world(a, s, z):
        return np.array([a * w3[0] + s * n3[0], a * w3[1] + s * n3[1], z])

    bottom_middle = to_world(a_c, s_c, floor_z)
    corners = [
        to_world(a, s, z)
        for z in (floor_z, ceil_z)
        for (a, s) in ((wall_right, s0), (wall_left, s0), (wall_left, s1), (wall_right, s1))
    ]
    out.update(
        bottom_middle=bottom_middle.tolist(),
        insertion_axis=n3.tolist(),
        lateral_axis_left=w3.tolist(),
        floor_z=float(floor_z),
        table_z=float(table_z),
        ceiling_z=float(ceil_z),
        front_plane=dict(
            normal_in=n3.tolist(), offset=float(s_front), outer_offset=float(F["s_outer"])
        ),
        opening_center=to_world(
            0.5 * (op_left + op_right), s_front, 0.5 * (op_top + op_bot)
        ).tolist(),
        opening_corners=[
            to_world(a, s_front, z).tolist()
            for (a, z) in (
                (op_right, op_bot),
                (op_left, op_bot),
                (op_left, op_top),
                (op_right, op_top),
            )
        ],
        interior_box=dict(
            center=to_world(a_c, s_c, 0.5 * (floor_z + ceil_z)).tolist(),
            yaw=float(yaw),
            size=[
                float(depth_in),
                float(width),
                float(ceil_z - floor_z),
            ],  # [along insertion, lateral, vertical]
            size_axes=["insertion_axis", "lateral_axis_left", "world_z"],
            corners=[c.tolist() for c in corners],
        ),  # 0-3 floor (front-right, front-left, back-left, back-right), 4-7 ceiling
        floor_rect=[corners[k].tolist() for k in range(4)],
    )
    for key, val in (("interior_w", width), ("interior_h", ceil_z - floor_z)):
        lo, hi, _ = PRIOR[key]
        if not (lo <= val <= hi):
            out["notes"].append(f"{key} {val:.3f} outside prior [{lo:.2f}, {hi:.2f}]")
    if debug:
        out["_debug"] = dict(
            score=F["score"],
            inside=inside,
            fl=fl,
            side=side,
            back=bk,
            ceil=ce,
            front_mask=F["front_mask"],
            yaw_cand=cand,
            Pw=Pw,
            A=A,
            dS=dS,
        )
    return out
