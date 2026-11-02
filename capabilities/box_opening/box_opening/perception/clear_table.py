"""Where on the table there is room for the lid: a top-down map of what the
scene camera sees standing around the box.

The lid used to go to a spot derived from the box alone — beside it, inside
the table the camera saw — and nothing asked what else stood there. Flatbot
2026-10-23 17:57 set it down on the front edge of a microwave 14 cm in front
of the box: the wrist camera, which rides 65 mm out along the hand's y
(world +x at the press attitude), came down on the microwave and broke. The
scene camera had the microwave in the middle of its frame.

From 14 cm above the table the scene camera sees the FRONT of whatever
stands on it well (the microwave: 34 cells, 26 points each, 83 % of them
standing) and the table itself poorly — at that grazing angle its points
are sparse and speckled. So the map answers what stands where and how tall,
not whether the table is flat. Heights are measured from the table plane
those same points show (fit_plane), not from the base frame's z: through
TF's mount the flatbot scene camera's table leans 6 deg, its far side 3-4
cm "below" the table.

What the camera could not see (behind the microwave, past its frame) is not
known to be clear, so nothing comes down there; past a table's edge it sees
the floor, so no lid goes down there.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from box_opening.geometry import ang_diff, yaw_about_world_z
from box_opening.perception.d405 import quat_to_mat
from box_opening.perception.planes import fit_plane
from box_opening.worlds import SEEN_PREFIX

CELL_M = 0.02
MAP_R_M = 0.6  # the map holds what the scene camera sees this far around the box
MAP_PAD_M = 0.25  # ... on a grid this much wider, unseen: past its edge nothing is known
PLANE_BAND_M = 0.08  # the table plane is fitted to points this close to the table's height
STANDS_M = 0.015  # a point this far above the table plane stands on it
OFF_M = 0.03  # ... and this far below it is past the table's edge
# a cell holds something only when enough of its points stand: a few
# flying pixels in a cell of table are speckle, not a thing
MIN_STANDING = 3
STANDING_FRAC = 0.25
# ... and a thing covers more than a cell or two: speckle comes in ones and
# twos (17:57: two "13 cm" cells of flying pixels off the microwave's dark
# door, 15 cm in front of it), the microwave covered 34
MIN_THING_CELLS = 3
MIN_SEEN_POINTS = 2

# What comes down with the lid, in the HAND frame (x the fingers' closing
# axis, y toward the wrist camera): each part a rectangle, the lowest it
# reaches above the fingertips, and how far it must miss anything (the
# scene camera's TF errs 1-2 cm away from the box it is corrected at, the
# map's cells are 2 cm). The lid is the model's disc below them, and needs
# the least: at the set-down's speed a lid brushing what stands beside it
# is harmless. The wrist camera needs the most — it is what broke. It is the
# OAK-D Pro (97 x 30 mm) TF mounts at +65 mm along hand y, its lens 116-123
# mm above the pads (TF said 116, the touch measured 123, 2026-10-23), its
# body a little lower toward the hand.
LID_CLEAR_M = 0.015
HAND_PARTS = (
    ("the fingers", (-0.055, 0.055), (-0.035, 0.035), 0.0, 0.02),  # the 2F-85 shut on the knob, and its body
    ("the wrist camera", (-0.05, 0.05), (0.0, 0.085), 0.10, 0.04),
)
UP_SLACK_M = 0.02  # a part clears a thing only by this much more than their heights say
ROOM_STEP_M = 0.01  # rooms this close are the same room ...
# ... and past this much, a spot the hand turns less to, nearer the box, is
# as good. A spot's room is what is left once every part has its own margin
# (LID_CLEAR_M, HAND_PARTS), and those margins already hold what the map
# gets wrong: the wrist camera's 40 mm is the scene camera's 2 cm of TF
# error and a 2 cm cell. One cell (CELL_M) more than they ask is enough;
# more guards again against what they already guard against. It was 0.05:
# 2026-10-29 12:02 took a spot with 46 mm to spare that the hand turned 91
# deg to, and joint_7 turned 151 deg, 6.0 s of the 7.8 s carry. Rooms are
# counted to the nearest ROOM_STEP_M, so 15 mm counts as enough.
ENOUGH_M = 0.02
SPOT_STEP_M = 0.02
SPOT_REACH_M = 0.20  # spots are sought at most this much farther from the box than the nearest allowed
SPOT_APART_M = 0.04  # the spots offered are at least this far apart
MAX_SPOTS = 8
# the hand may come down turned about its own axis, the lid with it (a
# round lid does not care; a square one lands square): joint_7 is
# continuous, and turned a quarter the wrist camera faces along the table's
# open band instead of into what stands across it (17:57: 31 mm of room
# turned, 20 mm not)
TURNS = (0.0, math.pi / 2, -math.pi / 2, math.pi)
# the hand's rotations this close are the same rotation, and the spot
# nearer the box is as good: the planner's choice of arm turns joint_7 more
# than that past the hand's own rotation (12:02: 151 deg for 91)
ROTATION_STEP_RAD = math.radians(10.0)


@dataclass(frozen=True)
class TableMap:
    """What stands on the table, on a CELL_M grid in the base frame."""

    x0: float  # the grid's corner
    y0: float
    cell: float
    tall: np.ndarray  # (nx, ny): the height above the table of what stands in a cell; nan where nothing does
    seen: np.ndarray  # (nx, ny) bool: the camera saw into the cell
    off: np.ndarray  # (nx, ny) bool: what it saw there is below the table — past its edge
    box_xy: tuple  # where the box stands

    def centres(self):
        nx, ny = self.tall.shape
        xs = self.x0 + (np.arange(nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys, indexing="ij")

    def standing(self):
        """(n, 3): every cell something stands in, as x, y and its height."""
        X, Y = self.centres()
        on = np.isfinite(self.tall)
        return np.stack([X[on], Y[on], self.tall[on]], -1)


def _neighbours(mask):
    """How many of each cell's 8 neighbours are set."""
    p = np.pad(mask.astype(np.int16), 1)
    n = np.zeros(mask.shape, np.int16)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a or b:
                n += p[1 + a : p.shape[0] - 1 + a, 1 + b : p.shape[1] - 1 + b]
    return n


def _small_things(mask, min_cells):
    """The 8-connected groups of `mask` smaller than min_cells, as a mask."""
    out = np.zeros_like(mask)
    todo = mask.copy()
    for i, j in zip(*np.nonzero(mask), strict=False):
        if not todo[i, j]:
            continue
        group, stack = [], [(i, j)]
        todo[i, j] = False
        while stack:
            a, b = stack.pop()
            group.append((a, b))
            for u in (a - 1, a, a + 1):
                for v in (b - 1, b, b + 1):
                    if 0 <= u < mask.shape[0] and 0 <= v < mask.shape[1] and todo[u, v]:
                        todo[u, v] = False
                        stack.append((u, v))
        if len(group) < min_cells:
            out[tuple(np.array(group).T)] = True
    return out


def _heights(pts, table_z, box_xy, radius):
    """(points within `radius` of the box, their heights above the table
    plane they show), or None when they show none."""
    pts = np.asarray(pts, float)
    pts = pts[np.hypot(pts[:, 0] - box_xy[0], pts[:, 1] - box_xy[1]) < radius]
    band = pts[np.abs(pts[:, 2] - float(table_z)) < PLANE_BAND_M]
    fit = fit_plane(band, tol_m=0.008, iters=300, max_points=5000) if len(band) >= 3 else None
    if fit is None or abs(fit[0][2]) < 0.9:
        return None
    normal, centroid, _frac = fit
    normal = normal if normal[2] > 0 else -normal
    return pts, (pts - centroid) @ normal


def table_map(clouds, box_xy, radius=MAP_R_M, cell=CELL_M):
    """The TableMap of `clouds` — [(points in the base frame, the table
    height they were measured on), ...]: the scene camera's, and the wrist
    camera's when it looked down at staging — within `radius` of the box,
    or None when none shows a table plane. Each cloud's heights are its
    own table plane's (each camera's mount errs its own way); their points
    pool, so a cell the wrist saw bare from above outvotes the speckle the
    scene camera saw there edge-on."""
    cx, cy = float(box_xy[0]), float(box_xy[1])
    x0, y0 = cx - radius - MAP_PAD_M, cy - radius - MAP_PAD_M
    n = int(math.ceil(2 * (radius + MAP_PAD_M) / cell))
    count, n_up, n_off = (np.zeros(n * n, np.int64) for _ in range(3))
    top = np.full(n * n, -np.inf)
    shown = False
    for pts, table_z in clouds:
        got = None if pts is None else _heights(pts, table_z, (cx, cy), radius)
        if got is None:
            continue
        shown = True
        pts, h = got
        ix = np.floor((pts[:, 0] - x0) / cell).astype(int)
        iy = np.floor((pts[:, 1] - y0) / cell).astype(int)
        ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        flat, h = ix[ok] * n + iy[ok], h[ok]
        up = h > STANDS_M
        count += np.bincount(flat, minlength=n * n)
        n_up += np.bincount(flat[up], minlength=n * n)
        n_off += np.bincount(flat[h < -OFF_M], minlength=n * n)
        np.maximum.at(top, flat[up], h[up])
    if not shown:
        return None
    count, n_up, n_off, top = (a.reshape(n, n) for a in (count, n_up, n_off, top))
    stands = (n_up >= MIN_STANDING) & (n_up >= STANDING_FRAC * count)
    stands &= ~_small_things(stands, MIN_THING_CELLS)
    seen = count >= MIN_SEEN_POINTS
    seen |= _neighbours(seen) >= 5  # a hole in what it saw, not a place it could not see
    off = ~stands & (2 * n_off > count) & (count >= MIN_SEEN_POINTS)
    off &= ~_small_things(off, MIN_THING_CELLS)  # an edge runs along the table, it is not a speck
    return TableMap(
        x0=x0, y0=y0, cell=cell, tall=np.where(stands, top, np.nan), seen=seen | stands, off=off, box_xy=(cx, cy)
    )


def _hand_axes(quat):
    """The hand's x and y axes in the base frame's xy, at a top-down attitude."""
    r = quat_to_mat(*quat)
    return r[:2, 0], r[:2, 1]


def hand_yaw(quat):
    """Which way the hand at `quat` faces about the vertical: the heading
    of its x axis (the fingers' closing axis) in the base frame, rad.
    That axis stays level when the reach-out leans the hand back
    (reach_press.reach_quat leans it about that axis)."""
    x, _y = _hand_axes(quat)
    return math.atan2(float(x[1]), float(x[0]))


def _to_square(d, half):
    """Distance from a point to a cell, from the point's offset to its centre."""
    return np.maximum(np.abs(d) - half, 0.0)


def clearance(tmap, xy, quat, lid_r, tips_h, near_m=0.25):
    """(metres, what): the room the hand coming straight down to `xy` has
    past what each part needs (LID_CLEAR_M, HAND_PARTS) from everything the
    map knows stands there or could not see, and the part with the least —
    the lid's disc (`lid_r`, its underside on the table) or one of
    HAND_PARTS, the fingertips `tips_h` above the table. Negative when a
    part comes nearer anything than it may."""
    sx, sy = float(xy[0]), float(xy[1])
    c = tmap.cell
    nx, ny = tmap.tall.shape
    i0, i1 = max(0, int((sx - near_m - tmap.x0) / c)), min(nx, int((sx + near_m - tmap.x0) / c) + 1)
    j0, j1 = max(0, int((sy - near_m - tmap.y0) / c)), min(ny, int((sy + near_m - tmap.y0) / c) + 1)
    dx, dy = np.meshgrid(
        tmap.x0 + (np.arange(i0, i1) + 0.5) * c - sx, tmap.y0 + (np.arange(j0, j1) + 0.5) * c - sy, indexing="ij"
    )
    win = (slice(i0, i1), slice(j0, j1))
    tall, seen, off = np.nan_to_num(tmap.tall[win], nan=-1.0), tmap.seen[win], tmap.off[win]
    half = 0.5 * c
    best, what = near_m, None
    lid = (tall > 0.0) | ~seen | off
    if lid.any():
        d = float(np.min(np.hypot(_to_square(dx, half), _to_square(dy, half))[lid])) - float(lid_r) - LID_CLEAR_M
        if d < best:
            best, what = d, "the lid"
    ax, ay = _hand_axes(quat)
    hx, hy = dx * ax[0] + dy * ax[1], dx * ay[0] + dy * ay[1]
    for name, (xa, xb), (ya, yb), bottom, need in HAND_PARTS:
        hits = (tall > float(tips_h) + bottom - UP_SLACK_M) | ~seen
        if not hits.any():
            continue
        gx = np.maximum(np.maximum(xa - hx, hx - xb) - half, 0.0)
        gy = np.maximum(np.maximum(ya - hy, hy - yb) - half, 0.0)
        d = float(np.min(np.hypot(gx, gy)[hits])) - need
        if d < best:
            best, what = d, name
    return best, what


def carry_clearance(tmap, start_xy, end_xy, lid_r, tips_h, lid_h):
    """(metres, what, where): the room, as clearance() says it, the hand
    carrying the lid from above the box to above the spot has — the
    fingertips `tips_h` up, the lid hanging `lid_h` below them — from what
    stands taller than it flies. The hand may turn on the way, so each part
    is the disc it sweeps turning. Only what stands is in its way up there:
    an unseen cell is the far side of something seen."""
    st = tmap.standing()
    a, b = np.asarray(start_xy, float)[:2], np.asarray(end_xy, float)[:2]
    worst = (float("inf"), None, None)
    if not len(st):
        return worst
    half = 0.5 * tmap.cell
    parts = [("the lid", float(lid_r) + LID_CLEAR_M, float(tips_h) - float(lid_h))]
    for name, (xa, xb), (ya, yb), bottom, need in HAND_PARTS:
        parts.append((name, math.hypot(max(-xa, xb), max(-ya, yb)) + need, float(tips_h) + bottom))
    steps = max(1, int(math.ceil(np.linalg.norm(b - a) / tmap.cell)))
    for t in np.linspace(0.0, 1.0, steps + 1):
        p = a + t * (b - a)
        d = np.hypot(_to_square(st[:, 0] - p[0], half), _to_square(st[:, 1] - p[1], half))
        for name, r, bottom in parts:
            hits = st[:, 2] > bottom - UP_SLACK_M
            if hits.any():
                c = float(np.min(d[hits])) - r
                if c < worst[0]:
                    worst = (c, name, (float(p[0]), float(p[1])))
    return worst


@dataclass(frozen=True)
class Spot:
    xy: tuple
    turn: float  # the hand's turn about its axis from the press attitude, rad
    room_m: float  # the room the set-down and the carry have past what each part needs
    what: str  # the part that comes nearest
    rotation: float  # how far the hand turns about the vertical from where it holds the lid to here, rad (-pi, pi]


def _rank(s):
    """A clear Spot's rank, the best least: its room (ROOM_STEP_M apart,
    no more than ENOUGH_M counted), then how far the hand turns to it
    (ROTATION_STEP_RAD apart)."""
    return -round(min(s.room_m, ENOUGH_M) / ROOM_STEP_M), round(abs(s.rotation) / ROTATION_STEP_RAD)


def clear_spots(
    tmap,
    box_xy,
    zone,
    min_from_box,
    quat_at,
    lid_r,
    tips_h,
    carry_tips_h,
    lid_h,
    step=SPOT_STEP_M,
    judged_out=None,
    hand_at=None,
    cache=None,
):
    """(spots, why): every spot in `zone` (x0, x1, y0, y1) at least
    `min_from_box` from the box, and at most SPOT_REACH_M more, where the
    hand comes down, at one of TURNS,
    and carries there with room for every part (clearance) — the roomiest
    first, then (past ENOUGH_M) the one the hand turns least to and nearest
    the box — at most MAX_SPOTS, SPOT_APART_M apart. `quat_at(xy)` is the
    press attitude there. `hand_at` is which way the hand holding the lid
    faces (hand_yaw), and each Spot's `rotation` how far it turns from
    there to come down at the spot; None: the press attitude at the box,
    so the turn alone. Each spot keeps the turn that ranks best there.
    `why` says, when there is none, what was in the way.

    The spots are judged `step` apart, and again half a step around each
    that came within a step of clear: a spot's room changes no faster than
    it moves, and the one clear spot beside the 17:57 box fell between the
    coarse grid's points. `judged_out`, a list, receives every Spot judged
    (tasks/lid_room_view draws them). `cache`, a dict, keeps each spot's
    room at each turn and on the carry (clearance, carry_clearance): the
    same map judged again — the hand facing another way (`hand_at`) — takes
    them from it, only the ranking left to do (reach_lid_off.room_ahead)."""
    bx, by = float(box_xy[0]), float(box_xy[1])
    x0, x1, y0, y1 = zone
    held = hand_yaw(quat_at((bx, by))) if hand_at is None else float(hand_at)
    judged = {}

    def judge(sx, sy):
        key = (round(sx, 4), round(sy, 4))
        if key in judged:
            return judged[key]
        if (
            not (x0 - 1e-9 <= sx <= x1 + 1e-9 and y0 - 1e-9 <= sy <= y1 + 1e-9)
            or not min_from_box <= math.hypot(sx - bx, sy - by) <= min_from_box + SPOT_REACH_M
        ):
            judged[key] = None
            return None
        q0 = quat_at((sx, sy))
        turned = []
        for t in TURNS:
            q = yaw_about_world_z(q0, t)
            if cache is not None and (key, t) in cache:
                room, what = cache[key, t]
            else:
                room, what = clearance(tmap, (sx, sy), q, lid_r, tips_h)
                if cache is not None:
                    cache[key, t] = room, what
            turned.append(
                Spot(
                    xy=(float(sx), float(sy)),
                    turn=t,
                    room_m=room,
                    what=what or "nothing",
                    rotation=ang_diff(hand_yaw(q), held),
                )
            )
        clear = [s for s in turned if s.room_m >= 0.0]
        if not clear:
            judged[key] = max(turned, key=lambda s: s.room_m)
            return judged[key]
        if cache is not None and (key, "carry") in cache:
            carry, c_what = cache[key, "carry"]
        else:
            carry, c_what, _at = carry_clearance(tmap, (bx, by), (sx, sy), lid_r, carry_tips_h, lid_h)
            if cache is not None:
                cache[key, "carry"] = carry, c_what
        clear = [replace(s, room_m=carry, what="%s on the carry" % c_what) if carry < s.room_m else s for s in clear]
        judged[key] = min(clear, key=lambda s: (*_rank(s), -s.room_m))
        return judged[key]

    coarse = [judge(sx, sy) for sx in np.arange(x0, x1 + 1e-9, step) for sy in np.arange(y0, y1 + 1e-9, step)]
    half = 0.5 * step
    for s in [s for s in coarse if s is not None and s.room_m >= -step]:
        for ox in (-half, 0.0, half):
            for oy in (-half, 0.0, half):
                judge(s.xy[0] + ox, s.xy[1] + oy)
    every = [s for s in judged.values() if s is not None]
    if judged_out is not None:
        judged_out.extend(every)
    found = [s for s in every if s.room_m >= 0.0]
    nearest = max(every, key=lambda s: s.room_m, default=None)
    found.sort(key=lambda s: (*_rank(s), math.hypot(s.xy[0] - bx, s.xy[1] - by)))
    picked = []
    for s in found:
        if all(math.hypot(s.xy[0] - p.xy[0], s.xy[1] - p.xy[1]) >= SPOT_APART_M for p in picked):
            picked.append(s)
        if len(picked) == MAX_SPOTS:
            break
    why = None
    if not picked:
        why = (
            "nowhere in x %.2f to %.2f, y %.2f to %.2f has room for the lid and the hand among what the cameras "
            "see" % (x0, x1, y0, y1)
        )
        if nearest is not None:
            why += " (the roomiest, [%.2f, %.2f]: %s %.0f mm short of the room it needs)" % (
                nearest.xy[0],
                nearest.xy[1],
                nearest.what,
                -1000 * nearest.room_m,
            )
    return picked, why


WRIST_NEAREST_M = 0.15  # nearer the wrist lens than this is the hand itself: the fingertips are 123 mm out


def depth_points(depth, k, rot_cam, trans_cam, stride=2):
    """A depth frame (metres, aligned to `k`) as points in the base frame,
    the camera at (rot_cam, trans_cam) there — the wrist's frame at staging,
    looking straight down at the table around the box (its lid ~22 cm off),
    without the fingers it can see in front of it."""
    d = np.asarray(depth, np.float32)[::stride, ::stride]
    v, u = np.nonzero(np.isfinite(d) & (d > WRIST_NEAREST_M) & (d < 2.0))
    z = d[v, u].astype(float)
    kk = np.asarray(k, float)
    x = (u * stride - kk[0, 2]) / kk[0, 0] * z
    y = (v * stride - kk[1, 2]) / kk[1, 1] * z
    return np.stack([x, y, z], -1) @ np.asarray(rot_cam, float).T + np.asarray(trans_cam, float)


# What stands on the table, for the PLANNER. The spot search above checks
# the hand at the set-down and the carry's straight line; the planner plans
# every motion of the whole arm, and knew nothing the cameras saw (17:57:
# the microwave was in no world it was given). Here the map's standing
# squares become cuboids for its collision world (worlds.WorldStore.seen):
# each square's column from the table up to the tallest point standing in
# it. Nothing is surveyed or named: whatever the cameras see standing this
# run, and only that, is in it.
# rammp-curobo holds 60 cuboids a world (collision_cache_obb); the floor
# keep-out, the table, the box and the placed lid take 4 of them
PLANNER_BOXES = 40
BOX_CLEAR_M = 0.03  # what stands this close to the box's cuboid is the box: its holder, its edges' flying pixels
ARM_CLEAR_M = 0.10  # ... and this close to the arm's centre line when the world is made, the arm (or touching it)
KEEP_CLEAR_M = 0.10  # no merged cuboid covers free table this close to a lid spot: HAND_PARTS reach 0.099 m out
MERGE_GAP = 10  # squares: only cuboids this close are merged while any such pair may be


def _grow(mask, i, j, order):
    """The rectangle of `mask` squares grown from (i, j), a side at a time
    in `order` (0: +y, 1: -y, 2: +x, 3: -x), as (i0, i1, j0, j1)."""
    nx, ny = mask.shape
    i0, i1, j0, j1 = i, i, j, j
    grew = True
    while grew:
        grew = False
        for side in order:
            if side == 0 and j1 + 1 < ny and mask[i0 : i1 + 1, j1 + 1].all():
                j1, grew = j1 + 1, True
            elif side == 1 and j0 > 0 and mask[i0 : i1 + 1, j0 - 1].all():
                j0, grew = j0 - 1, True
            elif side == 2 and i1 + 1 < nx and mask[i1 + 1, j0 : j1 + 1].all():
                i1, grew = i1 + 1, True
            elif side == 3 and i0 > 0 and mask[i0 - 1, j0 : j1 + 1].all():
                i0, grew = i0 - 1, True
    return i0, i1, j0, j1


def _cover(mask):
    """Rectangles of `mask` squares that cover all of them: each time, of
    the rectangles grown from every square (along y first, and along x
    first), the one covering the most squares not covered yet."""
    cands = np.array(
        sorted(
            {
                _grow(mask, i, j, o)
                for i, j in zip(*np.nonzero(mask), strict=False)
                for o in ((0, 1, 2, 3), (2, 3, 0, 1))
            }
        ),
        int,
    ).reshape(-1, 4)
    a, b, c, d = cands.T
    todo = mask.copy()
    out = []
    while todo.any():
        P = np.pad(todo.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        gain = P[b + 1, d + 1] - P[a, d + 1] - P[b + 1, c] + P[a, c]
        k = int(np.argmax(gain))
        todo[a[k] : b[k] + 1, c[k] : d[k] + 1] = False
        out.append(tuple(int(v) for v in cands[k]))
    return out


def _merge(rects, blocked, forbid, budget):
    """Merge rectangles [i0, i1, j0, j1, height] pairwise until at most
    `budget` remain: each time the pair whose enclosing rectangle, at the
    taller one's height, blocks the least volume not blocked yet. No merge
    may raise `blocked` (the height blocked per square, 0 where free) where
    `forbid` is set. Returns the rectangles (over budget when nothing may
    merge). A pair's cost changes only where a merge blocked more, so only
    the pairs over it are judged again."""
    live = {k: list(r) for k, r in enumerate(rects)}
    nxt = len(live)

    def judge(rp, rq, gap):
        if max(max(rp[0], rq[0]) - min(rp[1], rq[1]), max(rp[2], rq[2]) - min(rp[3], rq[3])) > gap:
            return None
        a, b, c, d = min(rp[0], rq[0]), max(rp[1], rq[1]), min(rp[2], rq[2]), max(rp[3], rq[3])
        h = max(rp[4], rq[4])
        up = h - blocked[a : b + 1, c : d + 1]
        if (forbid[a : b + 1, c : d + 1] & (up > 1e-9)).any():
            return None
        return float(np.clip(up, 0.0, None).sum()), [a, b, c, d, h]

    for gap in (MERGE_GAP, max(blocked.shape)):
        keys = sorted(live)
        pairs = {}
        for x, p in enumerate(keys):
            for q in keys[x + 1 :]:
                got = judge(live[p], live[q], gap)
                if got is not None:
                    pairs[(p, q)] = got
        while len(live) > budget and pairs:
            (p, q), (_cost, r) = min(pairs.items(), key=lambda kv: (kv[1][0], kv[0]))
            del live[p], live[q]
            a, b, c, d, h = r
            blocked[a : b + 1, c : d + 1] = np.maximum(blocked[a : b + 1, c : d + 1], h)
            pairs = {k: v for k, v in pairs.items() if p not in k and q not in k}
            for k, (_c, u) in list(pairs.items()):  # their volume changed only where this one blocks
                if u[0] <= b and a <= u[1] and u[2] <= d and c <= u[3]:
                    got = judge(live[k[0]], live[k[1]], gap)
                    if got is None:
                        del pairs[k]
                    else:
                        pairs[k] = got
            for s_ in live:
                got = judge(live[s_], r, gap)
                if got is not None:
                    pairs[(s_, nxt)] = got
            live[nxt] = r
            nxt += 1
        if len(live) <= budget:
            break
    return [live[k] for k in sorted(live)]


def obstacle_boxes(tmap, table_z, box_xy, box_half, arm_points=None, keep_clear=(), budget=PLANNER_BOXES):
    """(cuboids, note): what `tmap` knows stands on the table, as at most
    `budget` cuboid dicts (name, position, dims) for the planner's world,
    standing on the table at `table_z` (heights are the map's, above the
    plane the points showed), and what was made of them.

    Left out: what stands within BOX_CLEAR_M of the box's cuboid (centred
    on `box_xy`, `box_half` its half-width: the world's container cuboid is
    the box), and what reaches within ARM_CLEAR_M of `arm_points` (the
    arm's centre line, kinematics.ArmChain.points: the arm itself, which
    would put every start in collision). The squares are covered by
    rectangles, merged where that blocks the least table not standing
    (conservative: a merged cuboid is the taller one's height), never over
    the box, the arm, or free table within KEEP_CLEAR_M of a `keep_clear`
    (x, y): the lid spots. When even then there are too many, the smallest
    are left out, and the note says so."""
    c = tmap.cell
    X, Y = tmap.centres()
    tz = float(table_z)
    stands = np.isfinite(tmap.tall)
    H = np.where(stands, tmap.tall, 0.0)
    near_box = (np.abs(X - float(box_xy[0])) - 0.5 * c < float(box_half) + BOX_CLEAR_M) & (
        np.abs(Y - float(box_xy[1])) - 0.5 * c < float(box_half) + BOX_CLEAR_M
    )
    forbid = near_box.copy()
    under_arm = np.zeros_like(stands)
    if arm_points is not None and len(arm_points) > 1:
        pts = np.asarray(arm_points, float)
        for a, b in zip(pts[:-1], pts[1:], strict=False):
            for t in np.linspace(0.0, 1.0, max(2, int(np.ceil(np.linalg.norm(b - a) / c)) + 1)):
                s = a + t * (b - a)
                near = np.hypot(_to_square(X - s[0], 0.5 * c), _to_square(Y - s[1], 0.5 * c)) <= ARM_CLEAR_M
                forbid |= near
                under_arm |= near & stands & (H >= s[2] - tz - ARM_CLEAR_M)
    for x, y in keep_clear:
        forbid |= ~stands & (np.hypot(_to_square(X - x, 0.5 * c), _to_square(Y - y, 0.5 * c)) <= KEEP_CLEAR_M)
    mask = stands & ~near_box & ~under_arm
    mask &= ~_small_things(mask, MIN_THING_CELLS)  # what is left of the box's or the arm's surroundings is not a thing
    rects = [
        [a, b, cc, d, float(H[a : b + 1, cc : d + 1][mask[a : b + 1, cc : d + 1]].max())]
        for a, b, cc, d in _cover(mask)
    ]
    blocked = np.where(mask, H, 0.0)
    rects = _merge(rects, blocked, forbid, budget)
    left_out = 0
    if len(rects) > budget:
        rects.sort(key=lambda r: -(r[1] - r[0] + 1) * (r[3] - r[2] + 1) * r[4])
        left_out, rects = len(rects) - budget, rects[:budget]
    boxes = []
    for k, (a, b, cc, d, h) in enumerate(sorted(rects)):
        boxes.append(
            {
                "name": "%s%02d" % (SEEN_PREFIX, k),
                "position": [
                    float(tmap.x0 + 0.5 * (a + b + 1) * c),
                    float(tmap.y0 + 0.5 * (cc + d + 1) * c),
                    tz + 0.5 * h,
                ],
                "dims": [float((b - a + 1) * c), float((d - cc + 1) * c), float(h)],
            }
        )
    covered = np.zeros_like(stands)
    for a, b, cc, d, _h in rects:
        covered[a : b + 1, cc : d + 1] = True
    note = dict(
        boxes=len(boxes),
        squares=int(mask.sum()),
        free_blocked=int((covered & ~mask).sum()),
        under_arm=int(under_arm.sum()),
        left_out=left_out,
        tallest_m=round(max((r[4] for r in rects), default=0.0), 3),
    )
    return boxes, note
