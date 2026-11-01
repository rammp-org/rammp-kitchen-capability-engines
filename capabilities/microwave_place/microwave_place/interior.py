"""Where the container goes in the open microwave, from one scene-camera depth frame.

Pure numpy/scipy (no ROS). Entry point:
    estimate_interior(depth_m, K, world_T_cam, objects_3d) -> dict, or None when not detected

This is for one microwave, whose interior dimensions are known (below). Its pose is measured in the
point cloud by fitting those dimensions to the inside surfaces the camera sees; nothing else about
the oven is assumed. A frame that does not show enough of the inside to fit is not detected.

Oven frame used internally (horizontal axes lie in the gravity-aligned world xy plane):
    n_in : horizontal, into the oven (= insertion axis)
    w    : z x n_in, horizontal, the oven's LEFT as seen from the front
    A    : coordinate along w,  S : coordinate along n_in,  Z : world z

Steps
  1. Where to look: the microwave's 3D box (object_lifter, /perception/object_detections_3d). The
     box is used for nothing else.
  2. The oven's yaw (mod 90 deg) from the directions of its vertical surfaces.
  3. For each of the four ways the oven can face, fit the known interior to the point cloud:
     - in/out: the back wall, the frame around the opening and the outer front, DEPTH and RECESS
       apart; pinned by the back wall, or by the frame and the outer front together;
     - sideways: the left wall;
     - height: the floor and the ceiling, HEIGHT apart;
     - and the camera must see those surfaces through the fitted opening.
     Exactly one way must fit; otherwise the frame is not detected.
  4. The point: the opening's centre sideways, half-way from the opening to the back wall, on the
     floor.

The scene camera sits to the right of the opening, so the right wall is never in view; it is not
used.
"""

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

# This microwave: medians over 124 frames of it at rest (take 1, 2026-10-30).
DEPTH = 0.296  # frame around the opening -> back wall
HEIGHT = 0.185  # floor (turntable top) -> ceiling
OPENING = (0.265, 0.185)  # width, height
LEFT_WALL = 0.036  # left wall, outside the opening's left edge
RECESS = 0.032  # frame around the opening, behind the outer front (the door's thickness)
FLOOR_ABOVE_OPENING = 0.003  # floor, above the opening's bottom edge

TOL = 0.01  # a point this close to a fitted surface supports it
MIN_POINTS = dict(back=60, front=100, left=80, level=40)  # support needed per surface


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
    """Among 'microwave' detections with a 3D pose, the one whose extents look like a whole oven
    (the open door is often detected as a second, smaller 'microwave'); None if there is none."""
    mws = [
        o for o in objects_3d if str(o.get("cls", "")).lower() == "microwave" and o.get("has_pose")
    ]
    if not mws:
        return None

    def score(o):
        s = np.sort(np.asarray(o["size"], float))[::-1]
        return float(o.get("score", 0)) - 2.0 * np.abs(s - np.array([0.46, 0.31, 0.27])).sum()

    return max(mws, key=score)


def _manhattan_yaw(nxy, iters=4):
    """Yaw (rad, defined mod 90 deg) of the dominant rectangular structure from horizontal normals
    (N,2): a vote over the 90 deg period (+-4 deg window), then circular-mean refinement. None if
    fewer than 50 normals support it."""
    th = np.arctan2(nxy[:, 1], nxy[:, 0])
    if th.size < 50:
        return None
    grid = np.radians(np.arange(-45.0, 45.0, 1.0))
    votes = np.zeros(len(grid))
    for s in range(0, th.size, 20000):
        d = np.angle(np.exp(4j * (th[None, s : s + 20000] - grid[:, None]))) / 4.0
        votes += (np.abs(d) < np.radians(4.0)).sum(1)
    yaw = grid[int(np.argmax(votes))]
    for _ in range(iters):
        r = np.angle(np.exp(4j * (th - yaw))) / 4.0
        m = np.abs(r) < np.radians(8.0)
        if m.sum() < 50:
            return None
        yaw = yaw + np.angle(np.mean(np.exp(4j * r[m]))) / 4.0
    return float(yaw)


def _mode(x, bin_w=0.005):
    """Median of the samples within TOL of the densest bin of x; None if x is empty."""
    if x.size == 0:
        return None
    h, e = np.histogram(x, bins=np.arange(x.min(), x.max() + 2 * bin_w, bin_w))
    k = int(np.argmax(np.convolve(h, [1, 1, 1], "same")))
    c = 0.5 * (e[k] + e[k + 1])
    return float(np.median(x[np.abs(x - c) < TOL]))


def _fit(P, N, vert, yaw, cam):
    """The interior with its insertion axis at this yaw, fitted to the points; None if the inside
    surfaces it needs are not there."""
    n3 = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    w3 = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # w3 = z x n3
    A, S, Z = P @ w3, P @ n3, P[:, 2]
    nA, nS, nZ = N @ w3, N @ n3, N[:, 2]
    facing_out = vert & (nS < -0.85)  # back wall, frame, outer front
    facing_right = vert & (nA < -0.85)  # left wall
    up, down = nZ > 0.93, nZ < -0.93  # floor, ceiling

    # in/out: the back wall, the frame and the outer front vote for the frame's position
    s_out = S[facing_out]
    s_front = _mode(np.concatenate([s_out - DEPTH, s_out, s_out + RECESS]))
    if s_front is None:
        return None
    in_S = (S > s_front + 0.03) & (S < s_front + DEPTH - 0.03)

    # sideways: the left wall
    a_mid = _mode(A[facing_right & in_S] - OPENING[0] / 2 - LEFT_WALL)
    if a_mid is None:
        return None
    a_left = a_mid + OPENING[0] / 2 + LEFT_WALL
    in_A = (A > a_mid - OPENING[0] / 2 + 0.015) & (A < a_left - 0.015)

    # height: the floor and the ceiling vote for the floor's height
    z_floor = _mode(np.concatenate([Z[up & in_S & in_A], Z[down & in_S & in_A] - HEIGHT]))
    if z_floor is None:
        return None
    in_Z = (Z > z_floor + 0.02) & (Z < z_floor + HEIGHT - 0.02)

    z_bot = z_floor - FLOOR_ABOVE_OPENING
    back = facing_out & in_A & in_Z & (np.abs(S - s_front - DEPTH) < TOL)
    left = facing_right & in_S & in_Z & (np.abs(A - a_left) < TOL)
    level = (
        in_S
        & in_A
        & ((up & (np.abs(Z - z_floor) < TOL)) | (down & (np.abs(Z - z_floor - HEIGHT) < TOL)))
    )
    around = (
        facing_out
        & (np.abs(A - a_mid) < OPENING[0] / 2 + 0.12)
        & (Z > z_bot - 0.08)
        & (Z < z_bot + OPENING[1] + 0.08)
    )
    frame = around & (np.abs(S - s_front) < TOL)
    outer = around & (np.abs(S - s_front + RECESS) < TOL)
    # in/out is pinned by the back wall, or by the frame and the outer front together (one front
    # plane alone could be either)
    pinned = (
        back.sum() >= MIN_POINTS["back"] or min(frame.sum(), outer.sum()) >= MIN_POINTS["front"]
    )
    if not pinned or left.sum() < MIN_POINTS["left"] or level.sum() < MIN_POINTS["level"]:
        return None

    # the camera sees the inside through the opening: each line of sight crosses the frame plane
    # inside the opening
    inner = back | left | level
    ca, cs, cz = cam @ w3, cam @ n3, cam[2]
    t = (s_front - cs) / (S[inner] - cs)
    a_hit, z_hit = ca + t * (A[inner] - ca), cz + t * (Z[inner] - cz)
    through = (
        (t > 0)
        & (t < 1)
        & (np.abs(a_hit - a_mid) < OPENING[0] / 2 + 0.02)
        & (z_hit > z_bot - 0.02)
        & (z_hit < z_bot + OPENING[1] + 0.02)
    )
    if through.mean() < 0.9:
        return None

    def to_world(a, s, z):
        return np.array([a * w3[0] + s * n3[0], a * w3[1] + s * n3[1], z])

    return dict(
        point=to_world(a_mid, s_front + 0.5 * DEPTH, z_floor).tolist(),
        insertion_axis=n3.tolist(),
        lateral_axis_left=w3.tolist(),
        # in the frame plane, seen from the front: bottom-right, bottom-left, top-left, top-right
        opening_corners=[
            to_world(a_mid + da * OPENING[0] / 2, s_front, z_bot + dz * OPENING[1]).tolist()
            for da, dz in ((-1, 0), (1, 0), (1, 1), (-1, 1))
        ],
        floor_z=z_floor,
    )


def estimate_interior(depth_m, K, world_T_cam, objects_3d):
    """depth_m: (H,W) metres aligned to the color image, 0 = invalid. K: 3x3.
    world_T_cam: 4x4 camera-optical -> world (z up).
    objects_3d: list of dicts (cls, score, box [x1,y1,x2,y2], has_pose, pos, quat xyzw, size;
    pose in camera optical), as in /perception/object_detections_3d.
    Returns, in world: point (where the container goes), insertion_axis, lateral_axis_left,
    opening_corners and floor_z; or None when not detected."""
    K = np.asarray(K, float).reshape(3, 3)
    T = np.asarray(world_T_cam, float)
    depth = np.asarray(depth_m, np.float32)
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 3.0)
    Pc = _backproject(np.where(valid, depth, 0), K)
    Pw = Pc @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
    cam = T[:3, 3]
    Nw, nok = _normals(Pw, valid)

    # ---------- 1. Where to look: the microwave's 3D box ----------
    mw = _pick_microwave(objects_3d)
    if mw is None:
        return None
    Rw = T[:3, :3] @ _Rot.from_quat(mw["quat"]).as_matrix()
    hor = [k for k in range(3) if abs(Rw[2, k]) < 0.5]
    if len(hor) != 2:  # the box is not upright
        return None
    cw = T[:3, :3] @ np.asarray(mw["pos"], float) + T[:3, 3]
    half_xy = 0.5 * max(mw["size"][k] for k in hor) + 0.06
    near_box = (
        (np.abs(Pw[..., 0] - cw[0]) < half_xy)
        & (np.abs(Pw[..., 1] - cw[1]) < half_xy)
        & (np.abs(Pw[..., 2] - cw[2]) < 0.3)
    )
    x1, y1, x2, y2 = (float(t) for t in mw["box"])
    bw, bh = x2 - x1, y2 - y1
    box2d = np.zeros((H, W), bool)
    box2d[
        int(max(0, y1 - 0.03 * bh)) : int(min(H, y2 + 0.03 * bh)),
        int(max(0, x1 - 0.03 * bw)) : int(min(W, x2 + 0.03 * bw)),
    ] = True
    sel = box2d & near_box & nok
    P = Pw[sel].astype(float)
    N = Nw[sel].astype(float)
    N *= np.sign(((cam - P) * N).sum(1, keepdims=True))  # the side the camera sees
    vert = np.abs(N[:, 2]) < 0.30  # surface within ~17 deg of vertical

    # ---------- 2. Yaw from the point cloud ----------
    nxy = N[vert][:, :2]
    yaw = _manhattan_yaw(nxy / np.maximum(np.linalg.norm(nxy, axis=1, keepdims=True), 1e-9))
    if yaw is None:
        return None

    # ---------- 3-4. The interior, for each way the oven can face ----------
    fits = [
        f for f in (_fit(P, N, vert, yaw + k * np.pi / 2, cam) for k in range(4)) if f is not None
    ]
    if len(fits) != 1:  # none fits, or more than one does: cannot tell
        return None
    return fits[0]
