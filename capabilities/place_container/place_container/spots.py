"""Where on the table the container can go: the spots the UI offers for place_here's tap-to-place. Pure (numpy only,
no ROS), so it is tested on its own and on recorded frames.

From one scene-camera frame (depth aligned to colour, its K, arm_mount <- camera optical frame):

1. Every depth pixel -> a point in arm_mount.
2. The table: the scene segmenter's `table` mask(s) when the frame has any, else every point in the workspace (a
   height band and the reach annulus, widened). A RANSAC plane, near horizontal, refined by least squares.
   A table that looks tilted (more than `max_tilt_deg`) is refused: a level table seen tilted means the scene
   camera's mount (arm_mount -> scene_camera tf) is off, and every spot would be off with it.
3. The table's top, seen from above: a `cell` grid of the plane's inliers, small holes (depth gaps) filled.
4. Obstacles: points `obstacle_h` above the plane over the table, and the masks of every other detected object;
   points in `exclude` (the hand and the container it holds) are left out.
5. A grid every `step` m, anchored at arm_mount's origin, so a spot keeps its id from frame to frame. A spot is kept
   if it is in `reach` (distance from the arm's base axis), the table is under its whole footprint plus
   `edge_margin`, and nothing is within its footprint plus `clearance`.
6. Each spot projected into the colour image: its grid cell's bounding box, for the UI's `objects`.

`local_table_z` gives the table's height at the chosen spot: a plane fitted to the inliers around it.
"""

import numpy as np

TABLE_CLASS = "table"  # the scene segmenter's class (rammp-perception docs/interface.md)

DEFAULTS = {
    "step": 0.09,  # m between spots
    "footprint_r": 0.087,  # m, the container with its lip (plan.BOWL_RADIUS + plan.LIP_WIDTH)
    "edge_margin": 0.03,  # m of table beyond the footprint
    "clearance": 0.03,  # m between the footprint and anything on the table
    "reach": (0.70, 1.10),  # m from the arm's base axis (the node passes its `reach` parameter)
    "table_z_range": (-0.10, 0.50),  # m, arm_mount: where a table may be (the plane fit without a table mask)
    "max_tilt_deg": 3.0,  # a table tilted more than this to the camera: its mount is off, refused
    "search_tilt_deg": 20.0,  # planes up to this tilt are found (and then refused with the reason)
    "inlier_tol": 0.012,  # m from the plane = the table
    "obstacle_h": (0.015, 0.40),  # m above the plane = something on the table
    "obstacle_min_points": 3,  # depth points in one cell before it counts as an obstacle (flying pixels)
    "cell": 0.02,  # m, the top-view grid
    "close_cells": 2,  # holes in the seen table up to ~2x this many cells are filled (depth gaps at grazing view)
    "support_frac": 1.0,  # share of the footprint + margin that must be seen as table (after the filling)
    "max_spots": 60,
    "min_table_points": 500,
    "exclude": (),  # [(x, y, r)]: obstacles within r of (x, y) are ignored (the hand holding the container)
}


class Scene:
    """One scene-camera frame. depth_m: HxW metres, 0 = none, aligned to the colour image. K: 3x3. T_base_cam: 4x4,
    arm_mount <- the colour optical frame. labels: HxW mask ids (the segmenter's label_image) or None; objects:
    [(perception_class, mask_id)] of that frame."""

    def __init__(self, depth_m, K, T_base_cam, labels=None, objects=(), stamp=0.0, source=""):
        self.depth_m = np.asarray(depth_m, dtype=np.float32)
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        self.T = np.asarray(T_base_cam, dtype=float)
        self.labels = None if labels is None else np.asarray(labels)
        self.objects = list(objects)
        self.stamp, self.source = stamp, source

    @property
    def size(self):
        return self.depth_m.shape[1], self.depth_m.shape[0]  # (width, height)


class Search:
    """find_spots' result: the spots ([{id, x, y, z}]), the plane (a, b, c: z = a x + b y + c), the table's inlier
    points (for local_table_z), where the table came from ("segmenter" / "depth plane"), and `problem` (why there are
    no spots, for the user) or ""."""

    def __init__(self, spots=(), plane=None, inliers=None, source="", problem="", tilt_deg=None):
        self.spots, self.plane, self.inliers = list(spots), plane, inliers
        self.source, self.problem, self.tilt_deg = source, problem, tilt_deg

    def by_id(self):
        return {s["id"]: s for s in self.spots}


# -- geometry ----------------------------------------------------------------------------------------------------


def cloud(depth_m, K, T, min_m=0.15, max_m=3.0):
    """(points (N, 3) in the base frame, flat pixel indices (N,)) of every pixel with depth in [min_m, max_m]."""
    depth = np.asarray(depth_m, dtype=np.float32)
    h, w = depth.shape
    idx = np.flatnonzero((depth > min_m) & (depth < max_m))
    v, u = np.divmod(idx, w)
    z = depth.ravel()[idx].astype(float)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1) @ T[:3, :3].T + T[:3, 3]
    return pts, idx


def project(points, K, T):
    """Pixels (N, 2) and camera-frame depth (N,) of base-frame points."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    Ti = np.linalg.inv(T)
    pc = pts @ Ti[:3, :3].T + Ti[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]], axis=1)
    return uv, z


def plane_z(plane, x, y):
    a, b, c = plane
    return a * np.asarray(x) + b * np.asarray(y) + c


def tilt_deg(plane):
    return float(np.degrees(np.arctan(np.hypot(plane[0], plane[1]))))


def fit_plane(points, max_tilt_deg=20.0, tol=0.012, iters=300, sample=6000, seed=0):
    """The plane with the most points within `tol`, among those tilted at most `max_tilt_deg`: (a, b, c) with
    z = a x + b y + c, least-squares refined on its inliers. None with fewer than 3 points or no such plane."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    rng = np.random.default_rng(seed)
    sub = pts if len(pts) <= sample else pts[rng.choice(len(pts), sample, replace=False)]
    cos_max = np.cos(np.radians(max_tilt_deg))
    best, best_n = None, 0
    for _ in range(iters):
        p = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if abs(n[2]) < cos_max:
            continue
        count = int(np.count_nonzero(np.abs((sub - p[0]) @ n) < tol))
        if count > best_n:
            best, best_n = (n, p[0]), count
    if best is None:
        return None
    n, p0 = best
    inl = np.abs((pts - p0) @ n) < tol
    plane = _lstsq_plane(pts[inl])
    # one more pass on the refined plane's own inliers
    inl = np.abs(pts[:, 2] - plane_z(plane, pts[:, 0], pts[:, 1])) < tol
    return _lstsq_plane(pts[inl]) if np.count_nonzero(inl) >= 3 else plane


def _lstsq_plane(pts):
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    return tuple(float(v) for v in coef)


def local_table_z(inliers, x, y, radius=0.10, min_points=50, max_std=0.01):
    """The table's height (m, arm_mount) at (x, y): a plane fitted to the table points within `radius`. Raises
    ValueError (why) with too few points there or a rough fit."""
    pts = np.asarray(inliers, dtype=float)
    near = pts[np.hypot(pts[:, 0] - x, pts[:, 1] - y) < radius] if len(pts) else pts
    if len(near) < min_points:
        raise ValueError(f"only {len(near)} table points within {radius * 100:.0f} cm of the spot (need {min_points})")
    plane = _lstsq_plane(near)
    std = float(np.std(near[:, 2] - plane_z(plane, near[:, 0], near[:, 1])))
    if std > max_std:
        raise ValueError(f"the table around the spot is not flat to the camera ({std * 100:.1f} cm rough)")
    return float(plane_z(plane, x, y))


def _window_sum(grid, kernel):
    """For each cell, the sum of `grid` over `kernel` (odd-sized, centred) around it; outside the grid counts 0."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(grid.astype(float), ((ph, ph), (pw, pw)))
    out = np.zeros(grid.shape, dtype=float)
    for di, dj in zip(*np.nonzero(kernel), strict=True):
        out += padded[di : di + grid.shape[0], dj : dj + grid.shape[1]]
    return out


def _close(mask, cells=1):
    """Morphological closing with a (2 cells + 1) square: fills holes up to about 2 x `cells` wide (depth dropouts,
    the gaps between far depth rows) without growing the outer edge."""
    k = np.ones((2 * cells + 1, 2 * cells + 1), dtype=bool)
    dilated = _window_sum(mask, k) > 0
    return _window_sum(~dilated, k) == 0


def _disk(radius_cells):
    r = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx**2 + yy**2) <= radius_cells**2


class _Grid:
    """A top-view occupancy grid over [lo, hi] (m) at `cell`."""

    def __init__(self, lo, hi, cell):
        self.lo, self.cell = np.asarray(lo, dtype=float), cell
        self.shape = tuple(int(n) for n in np.ceil((np.asarray(hi) - self.lo) / cell).astype(int) + 1)

    def index(self, xy):
        ij = np.floor((np.asarray(xy) - self.lo) / self.cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] < self.shape[1])
        return ij, ok

    def counts(self, xy):
        grid = np.zeros(self.shape, dtype=np.int32)
        ij, ok = self.index(xy)
        np.add.at(grid, (ij[ok, 0], ij[ok, 1]), 1)
        return grid


# -- the search -------------------------------------------------------------------------------------------------


def find_spots(scene, **overrides):
    """The free spots on the table in `scene` (a Scene), as a Search. Never raises on what it sees: a Search with
    `problem` says why there are none."""
    cfg = {**DEFAULTS, **overrides}
    pts, idx = cloud(scene.depth_m, scene.K, scene.T)
    if len(pts) == 0:
        return Search(problem="the scene camera's depth image is empty")
    lo_r, hi_r = cfg["reach"]

    # 2. the table's points: the segmenter's table mask, else the workspace
    table_ids, other_ids = [], []
    if scene.labels is not None and scene.labels.shape == scene.depth_m.shape:
        for cls, mask_id in scene.objects:
            if mask_id:
                (table_ids if cls == TABLE_CLASS else other_ids).append(int(mask_id))
    label_of = scene.labels.ravel()[idx] if table_ids or other_ids else None
    source, table_pts = "", None
    if table_ids:
        cand = pts[np.isin(label_of, table_ids)]
        if len(cand) >= cfg["min_table_points"]:
            source, table_pts = "segmenter", cand
    if table_pts is None:
        r = np.hypot(pts[:, 0], pts[:, 1])
        zlo, zhi = cfg["table_z_range"]
        work = (pts[:, 2] > zlo) & (pts[:, 2] < zhi) & (r > lo_r - 0.3) & (r < hi_r + 0.3)
        source, table_pts = "depth plane", pts[work]
        if len(table_pts) < cfg["min_table_points"]:
            return Search(source=source, problem="the scene camera sees no table in reach")
    plane = fit_plane(table_pts, cfg["search_tilt_deg"], cfg["inlier_tol"])
    if plane is None:
        return Search(source=source, problem="no flat table surface in the scene camera's depth")
    height = table_pts[:, 2] - plane_z(plane, table_pts[:, 0], table_pts[:, 1])
    inliers = table_pts[np.abs(height) < cfg["inlier_tol"]]
    if len(inliers) < cfg["min_table_points"]:
        return Search(plane=plane, source=source, problem="too little of the table is seen")
    tilt = tilt_deg(plane)
    if tilt > cfg["max_tilt_deg"]:
        return Search(
            plane=plane,
            inliers=inliers,
            source=source,
            tilt_deg=tilt,
            problem=(
                f"the table looks {tilt:.0f} degrees tilted to the scene camera, so the camera's calibration "
                "(its mount on the chair) is off and a spot from it would be too"
            ),
        )

    # 3. the table seen from above
    reach_pad = cfg["footprint_r"] + cfg["edge_margin"] + cfg["clearance"]
    lo = inliers[:, :2].min(axis=0) - reach_pad
    hi = inliers[:, :2].max(axis=0) + reach_pad
    grid = _Grid(lo, hi, cfg["cell"])
    support = _close(grid.counts(inliers[:, :2]) > 0, int(cfg["close_cells"]))

    # 4. obstacles: above the plane over the table, and every other detected object's mask
    above = pts[:, 2] - plane_z(plane, pts[:, 0], pts[:, 1])
    hlo, hhi = cfg["obstacle_h"]
    obst = pts[(above > hlo) & (above < hhi)]
    if other_ids:
        obst = np.vstack([obst, pts[np.isin(label_of, other_ids) & (above > -cfg["inlier_tol"])]])
    for x, y, r in cfg["exclude"]:
        obst = obst[np.hypot(obst[:, 0] - x, obst[:, 1] - y) > r]
    blocked = grid.counts(obst[:, :2]) >= cfg["obstacle_min_points"]

    # 5. the grid of candidate spots and its checks
    disk_edge = _disk((cfg["footprint_r"] + cfg["edge_margin"]) / cfg["cell"])
    seen = _window_sum(support, disk_edge) / disk_edge.sum()
    disk_clear = _disk((cfg["footprint_r"] + cfg["clearance"]) / cfg["cell"])
    near_obst = _window_sum(blocked, disk_clear) > 0
    step = cfg["step"]
    i0, j0 = np.ceil(lo / step).astype(int)
    i1, j1 = np.floor(hi / step).astype(int)
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    xy = np.stack([ii.ravel() * step, jj.ravel() * step], axis=1)
    keys = np.stack([ii.ravel(), jj.ravel()], axis=1)
    r = np.hypot(xy[:, 0], xy[:, 1])
    cells, inside = grid.index(xy)  # the cell the spot's centre falls in
    ok = inside & (r >= lo_r) & (r <= hi_r)
    ok[inside] &= seen[cells[inside, 0], cells[inside, 1]] >= cfg["support_frac"] - 1e-9
    ok[inside] &= ~near_obst[cells[inside, 0], cells[inside, 1]]
    counts = dict(
        grid=int(len(xy)),
        in_reach=int(np.count_nonzero(inside & (r >= lo_r) & (r <= hi_r))),
        free=int(np.count_nonzero(ok)),
    )
    spots = []
    for (x, y), (i, j) in zip(xy[ok], keys[ok], strict=True):
        z = float(plane_z(plane, x, y))
        spots.append(dict(id=f"spot_{i}_{j}".replace("-", "m"), x=round(float(x), 4), y=round(float(y), 4), z=z))
    if len(spots) > cfg["max_spots"]:  # the ones nearest the middle of the reach
        mid = 0.5 * (lo_r + hi_r)
        spots = sorted(spots, key=lambda s: abs(np.hypot(s["x"], s["y"]) - mid))[: cfg["max_spots"]]
    result = Search(spots, plane, inliers, source, tilt_deg=tilt)
    result.counts = counts
    if not spots:
        if counts["in_reach"] == 0:
            result.problem = f"none of the table the camera sees is within the arm's reach ({lo_r:.2f}-{hi_r:.2f} m)"
        else:
            result.problem = "no free spot big enough for the container within the arm's reach"
    return result


def ui_objects(spots, scene, step=DEFAULTS["step"], shrink=0.7):
    """The spots as the UI's `objects` (docs/UI_JSON_SCHEMA.md): each its grid cell (shrunk by `shrink`) projected
    into the scene camera's colour image, as a box. Spots outside the image or behind the camera are left out."""
    width, height = scene.size
    half = 0.5 * step * shrink
    out = []
    for s in spots:
        corners = [[s["x"] + dx, s["y"] + dy, s["z"]] for dx in (-half, half) for dy in (-half, half)]
        corners = np.array(corners + [[s["x"], s["y"], s["z"]]])
        uv, z = project(corners, scene.K, scene.T)
        if np.any(z <= 0.05):
            continue
        cu, cv = uv[-1]
        if not (0 <= cu < width and 0 <= cv < height):
            continue
        box = [
            float(np.clip(uv[:4, 0].min(), 0, width - 1)),
            float(np.clip(uv[:4, 1].min(), 0, height - 1)),
            float(np.clip(uv[:4, 0].max(), 0, width - 1)),
            float(np.clip(uv[:4, 1].max(), 0, height - 1)),
        ]
        out.append(
            {
                "instance_id": s["id"],
                "perception_class": "spot",
                "box": [round(v, 1) for v in box],
                "centroid": [round(float(cu), 1), round(float(cv), 1)],
                "source": "scene",
            }
        )
    return out


# -- a made-up table, for mock runs and the tests -----------------------------------------------------------------


def look_at(eye, target):
    """arm_mount <- an optical frame (z forward, x right, y down) at `eye` looking at `target`."""
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


MOCK_TABLE = dict(z=0.20, x=(0.45, 1.40), y=(-0.90, 0.60))
# boxes on the table: (class, (x0, y0), (x1, y1), height); a cup and a microwave
MOCK_BOXES = (("drink cup", (0.90, 0.00), (0.98, 0.08), 0.12), ("microwave", (1.05, -0.45), (1.40, 0.05), 0.30))
MOCK_CAMERA = dict(eye=(0.30, -0.60, 0.55), target=(0.95, -0.05, 0.20))


def synthetic_scene(table=MOCK_TABLE, boxes=MOCK_BOXES, camera=MOCK_CAMERA, size=(640, 400), f=284.0, labels=True):
    """A Scene of a level table with boxes on it, rendered by ray casting (floor at z=-0.5 elsewhere), with a label
    image (table = mask 1, boxes 2, 3, ...)."""
    width, height = size
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    T = look_at(np.asarray(camera["eye"], dtype=float), camera["target"])
    v, u = np.mgrid[0:height, 0:width]
    rays = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f, np.ones_like(u, dtype=float)], axis=-1).reshape(-1, 3)
    dirs = rays @ T[:3, :3].T  # base frame, per unit of camera z
    eye = T[:3, 3]
    depth = np.full(len(rays), np.inf)
    label = np.zeros(len(rays), dtype=np.uint16)

    def hit_plane(z0, xr, yr, mask_id):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z0 - eye[2]) / dirs[:, 2]
        p = eye + t[:, None] * dirs
        hit = (t > 0) & (p[:, 0] >= xr[0]) & (p[:, 0] <= xr[1]) & (p[:, 1] >= yr[0]) & (p[:, 1] <= yr[1])
        closer = hit & (t < depth)
        depth[closer], label[closer] = t[closer], mask_id

    hit_plane(-0.5, (-10, 10), (-10, 10), 0)
    hit_plane(table["z"], table["x"], table["y"], 1)
    for k, (_, (x0, y0), (x1, y1), h) in enumerate(boxes):
        lo = np.array([x0, y0, table["z"]])
        hi = np.array([x1, y1, table["z"] + h])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - eye) / dirs, (hi - eye) / dirs
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < depth)
        depth[hit], label[hit] = tmin[hit], k + 2
    depth[~np.isfinite(depth)] = 0.0
    objects = [(TABLE_CLASS, 1)] + [(cls, k + 2) for k, (cls, *_) in enumerate(boxes)]
    return Scene(
        depth.reshape(height, width),
        K,
        T,
        labels=label.reshape(height, width) if labels else None,
        objects=objects if labels else (),
        source="mock",
    )
