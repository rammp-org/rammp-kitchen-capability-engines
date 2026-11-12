"""Close the fridge: push the open door most of the way shut with the hand, which may hold a container -- no grasp.

The motion -- holding the container, every leg holds the hand's tilt (HOLD_LEVEL: it never tips it; turning about
vertical is free); with an empty hand the route may tilt it (cuRobo found no level route to most push starts, 11-10 sim):
1. To a push start (a free leg, cuRobo's route): behind the door -- round the hinge past the open door (where the door
   file says it is: the push-open leaves it at PUSH_OPEN_TO_DEG), clear of it there by CLOSE_START_GAP_M -- at a
   distance from the hinge out near the free edge (CLOSE_RADII_FROM_EDGE_M, furthest first: leverage), at a height
   within the door, the hand as it is now or turned about vertical. cuRobo's world has the door, its handle, the
   cabinet and the chair; the container -- a sphere carried with the hand (CONTAINER_M, CONTAINER_CENTRE_TOOL_M), which
   cuRobo doesn't see -- is checked along its route here. The fridge is never shifted to stand in for it. Only if no
   route plans is it given the old one: back in front of the door's free edge, then sideways.
2. The push: ONE leg (local IK, a pose every CLOSE_POINT_DEG) round the hinge at that distance toward shut, until the
   hand's real box (HAND_BOX_TOOL_M) or the container -- whichever touches first, at any angle, the handle included --
   would have the door at `close_to_deg`: it swings shut by itself from there. Any part may push: the hand's box or
   the container. Checked against the cabinet only: it pushes the door. Only this is scripted: it is contact, which
   cuRobo can't plan.
3. Back straight out up to `retreat_dist` (the door, now further shut, in the world).
One trajectory, stopping at the end of the push. The door is planned where the open saved it, shifted by the
wheelchair's move. The push starts are tried furthest from the hinge first, then by height (nearest the hand), then by
hand turn; at most CLOSE_TRIES of those whose geometry works are planned by cuRobo per wheelchair move.

Before that, the wheelchair is moved straight back or forward by the smallest of CLOSE_CHAIR_MOVES_M whose plan passes
(or by the goal's chair_back); the arm, held, must clear the door all through the move (the route to the push start is
planned from where it is against the door at CHAIR_MOVE_STEPS points along the move). A plan saves its move in the
door file, and an execute goal with chair_moved uses it; with the goal's `drive_base` the base drives it instead and
the close is planned again against the door shifted by the move the base really made.
"""

import collections
import itertools

import numpy as np
from door_common import common as dc
from door_common import open_door as shared
from door_common.common import Refused, save_door_geometry, say
from scipy.spatial.transform import Rotation as R

from . import constants as F
from . import planning as pl
from .motion import load_door, logged_run, opened, shifted_by
from .world import FridgeModel

# The goal's settings, and their defaults: F.CLOSE_SETTINGS holds what a close of this fridge doesn't vary.
GOAL_SETTINGS = {
    "door_deg": None,  # how far the door is open now (deg); None: the door file's
    "retreat_dist": F.CLOSE_SETTINGS["retreat_dist"],
    "push_z": None,  # the push's height (m); None: searched within the door
    "push_radius": None,  # the push's distance from the hinge (m); None: searched (CLOSE_RADII_FROM_EDGE_M)
    "chair_back": None,  # how far the chair is moved back (m, - forward); 0: not moved; None: the plan picks it
    "close_to_deg": F.CLOSE_TO_DEG,  # swing the door shut until it is open this far (deg)
    "chair_moved": False,  # the chair has been moved as the plan asked: required to execute with a move
}
TRIED = []  # the last close's push starts planned by cuRobo: [(move, radius, z, yaw, via, outcome)] -- for the logs
GEOMETRY_CACHE = {}  # CloseGeometry.push's results this close (run clears it): the same for every wheelchair move
RETREAT_DOOR_DEG = 20.0  # the back-out plans with the door this much further shut than where the push left it
SWEEP_STEP_DEG = 1.0  # the push's contact search: round the hinge this much at a time, then bisected


def holding(settings):
    """True when the close carries the container: run sets it from the gripper at the close's start (closed past
    CLOSE_HOLDING_GRIPPER)."""
    return bool(settings.get("container"))


def gripper_holds(st):
    """True when the arm state `st`'s gripper is closed: the hand holds the container."""
    return float(st["gripper_pos"]) > F.CLOSE_HOLDING_GRIPPER


def _box_gap(points, box):
    """Smallest signed distance (m) from `points` to obstacle dict `box` (yaw-only), negative inside."""
    c = np.asarray(box["position"], float)
    rot = R.from_euler("z", box["rpy_deg"][2], degrees=True)
    local = rot.inv().apply(np.asarray(points, float) - c)
    q = np.abs(local) - np.asarray(box["dims"], float) / 2
    return float(np.min(np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)))


def _box_gaps(points, box):
    """_box_gap for many point sets at once: `points` (N, M, 3) -> (N,), each set's smallest signed distance to `box`
    (yaw only). One numpy pass: the close's geometry checks thousands of hand poses (11-12: 357k calls one at a time
    were 55 of a close's 87 s)."""
    a = np.radians(box["rpy_deg"][2])
    c, s = np.cos(a), np.sin(a)
    d = np.asarray(points, float) - np.asarray(box["position"], float)
    local = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], axis=-1)
    q = np.abs(local) - np.asarray(box["dims"], float) / 2
    gap = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)
    return gap.min(axis=-1)



def _box_corners_at(box, z):
    """The 4 corners of obstacle dict `box` (yaw only) in plan, at height `z`."""
    c = np.asarray(box["position"], float)
    d = np.asarray(box["dims"], float) / 2
    rot = R.from_euler("z", box["rpy_deg"][2], degrees=True)
    return [np.r_[(c + rot.apply([x, y, 0.0]))[:2], z] for x in (-d[0], d[0]) for y in (-d[1], d[1])]


def _hand_grid(box, n=4):
    """Points on the surface of box ((x0, x1), (y0, y1), (z0, z1)) (corners included), in its own frame."""
    axes = [np.linspace(lo, hi, n) for lo, hi in box]
    pts = []
    for k in range(3):  # each pair of opposite faces: a grid on it
        for side in box[k]:
            g = np.meshgrid(*[axes[i] if i != k else np.array([side]) for i in range(3)], indexing="ij")
            pts.append(np.stack([x.ravel() for x in g], axis=1))
    return np.vstack(pts)


_HAND_GRID = {}


def _hand_points(pos, quat, n=4):
    """Points on the surface of the hand's box (HAND_BOX_TOOL_M; corners included), in arm_mount."""
    key = (n, tuple(map(tuple, F.HAND_BOX_TOOL_M)))
    if key not in _HAND_GRID:
        _HAND_GRID[key] = _hand_grid(F.HAND_BOX_TOOL_M, n)
    return np.asarray(pos, float) + R.from_quat(quat).apply(_HAND_GRID[key])


def _sphere_dirs(n=40):
    """`n` directions spread over the unit sphere (Fibonacci)."""
    k = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * k / n), np.pi * (1 + 5**0.5) * k
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


_DIRS = _sphere_dirs()


def container_centre(pos, quat):
    """The held container's centre (arm_mount) with the tool at (`pos`, `quat`)."""
    return np.asarray(pos, float) + R.from_quat(quat).apply([0.0, 0.0, F.CONTAINER_CENTRE_TOOL_M])


def body_points(pos, quat, container):
    """Points on what may touch the door: the hand's real box, and with `container` the held container (a sphere)."""
    pts = _hand_points(pos, quat)
    if container:
        pts = np.vstack([pts, container_centre(pos, quat) + _DIRS * (F.CONTAINER_M / 2)])
    return pts


def hand_gap(model, deg, pos, quat):
    """How far the hand's real box (HAND_BOX_TOOL_M) is from the door slab open `deg` (m, negative: into it)."""
    return min(_box_gap(_hand_points(pos, quat), b) for b in model.slab(deg))


class CloseGeometry:
    """Where the close's hand goes, for the door `door` (as planned: shifted) open `deg0`, the hand turned `quat`, the
    push at height `z`; `container`: the hand holds the container (a sphere carried with it)."""

    def __init__(self, door, deg0, quat, z, settings, stop_deg, container=False):
        self.model = FridgeModel(door)
        self.df = df = self.model.df
        self.deg0, self.quat, self.z, self.stop_deg = deg0, np.asarray(quat, float), float(z), float(stop_deg)
        self.s = settings
        self.container = container
        self.fwd = np.r_[-df.n, 0.0]  # into the fridge, the closed door's normal
        self.side = np.r_[-df.e, 0.0]  # along the closed door, toward the hinge
        self.h = np.r_[df.hinge[:2], self.z]
        self.door_box = self.model.slab(deg0) + self.model.handle(deg0)  # the open door -- cuRobo's route avoids it too
        self.end_door = self.model.slab(self.stop_deg) + self.model.handle(self.stop_deg)  # where the push leaves it

    def b_of(self, p):
        return float((np.asarray(p) - self.h) @ self.fwd)

    def radii(self):
        """The push's distances from the hinge to try (the tool point's), furthest first."""
        if self.s.get("push_radius") is not None:
            return [float(self.s["push_radius"])]
        return [self.df.length - d for d in sorted(F.CLOSE_RADII_FROM_EDGE_M) if self.df.length - d > 0.1]

    def at(self, radius, deg):
        """The tool point `radius` from the hinge where the door's mid-plane would be open `deg`."""
        return self.df.pos(radius, 0.0, deg, self.z)

    def body(self, pos):
        return body_points(pos, self.quat, self.container)

    def _body_offsets(self):
        """body()'s points relative to the tool point: the hand's orientation is fixed for this geometry, so every pose
        of it is these offsets moved (computed once)."""
        if getattr(self, "_off", None) is None:
            self._off = body_points(np.zeros(3), self.quat, self.container)
        return self._off

    def clear_of_door(self, pos):
        """How far (m) the hand and the container at tool position `pos` are from the open door and the cabinet
        (negative: into them)."""
        return float(self.clear_of_door_many(np.asarray(pos, float)[None, :])[0])

    def clear_of_door_many(self, positions):
        """clear_of_door for tool positions (N, 3) at once -> (N,)."""
        pts = np.asarray(positions, float)[:, None, :] + self._body_offsets()[None, :, :]
        return np.min([_box_gaps(pts, b) for b in self.door_box + self.model.body()], axis=0)

    def at_many(self, radius, degs):
        """at() for many door angles (deg) -> (N, 3): the tool point `radius` from the hinge on the door's mid-plane."""
        df = self.df
        a = df.sign * np.radians(np.asarray(degs, float))
        w = df.e * radius + df.n * df.mid
        xy = df.hinge[:2] + np.stack([np.cos(a) * w[0] - np.sin(a) * w[1], np.sin(a) * w[0] + np.cos(a) * w[1]], axis=1)
        return np.column_stack([xy, np.full(len(a), self.z)])

    def push_start(self, radius):
        """The door angle (deg) round the hinge where the push starts at `radius`: the first past the open door, every
        CLOSE_START_STEP_DEG up to CLOSE_START_MAX_DEG, whose hand (and container) clears the door and the cabinet by
        CLOSE_START_GAP_M; None if none does. Every angle checked in one pass."""
        n = int(np.floor((F.CLOSE_START_MAX_DEG + 1e-9 - self.deg0) / F.CLOSE_START_STEP_DEG))
        if n < 1:
            return None
        degs = self.deg0 + F.CLOSE_START_STEP_DEG * np.arange(1, n + 1)
        ok = np.nonzero(self.clear_of_door_many(self.at_many(radius, degs)) >= F.CLOSE_START_GAP_M)[0]
        return float(degs[ok[0]]) if len(ok) else None

    def sweep(self, radius, deg0):
        """The push's points [(deg turned round the hinge, pos)] from the start (`radius`, `deg0`) toward shut until the
        hand or the container touches the door at stop_deg (its handle included); None if it never would. The push
        spirals in from `radius` to CLOSE_PUSH_END_RADIUS_M (if further out) as it turns: out at the edge while the
        door is wide open (leverage), nearer the hinge -- and the arm -- by the end (a whole ~100 deg at the edge ran the
        arm into its joint limits, 11-10 sim); anywhere on the door keeps it pushed."""
        p0 = self.at(radius, deg0)
        v0 = p0 - self.h
        r0 = float(np.linalg.norm(v0[:2]))
        r1 = min(r0, F.CLOSE_PUSH_END_RADIUS_M)
        span = max(deg0 - self.stop_deg, 1.0)
        off = self._body_offsets()

        def at_many(ds):  # the start turned d deg round the hinge toward shut, drawn in (the hand's orientation kept)
            ds = np.asarray(ds, float)
            a = self.df.sign * np.radians(-ds)
            xy = np.stack([np.cos(a) * v0[0] - np.sin(a) * v0[1], np.sin(a) * v0[0] + np.cos(a) * v0[1]], axis=1)
            xy *= ((r0 + (r1 - r0) * np.minimum(ds / span, 1.0)) / r0)[:, None]
            return self.h + np.column_stack([xy, np.full(len(ds), v0[2])])

        def gaps(ds):  # the hand (and container) to the door at stop_deg, for each d
            pts = at_many(ds)[:, None, :] + off[None, :, :]
            return np.min([_box_gaps(pts, b) for b in self.end_door], axis=0)

        step, limit = SWEEP_STEP_DEG, deg0 - self.stop_deg + 30.0
        if gaps([0.0])[0] <= 0.0:
            return None
        # every whole step round, at once: the first that touches (as stepping one at a time, giving up past `limit`)
        k_max = int(np.floor(limit / step)) + 1
        hit = np.nonzero(gaps(step * np.arange(1, k_max + 1)) <= 0.0)[0]
        if not len(hit):
            return None
        d = step * float(hit[0])  # the last step still clear of the door
        lo, hi = d, d + step  # land on the touch
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if gaps([mid])[0] > 0.0 else (lo, mid)
        d_end = hi
        n = max(1, int(np.ceil(d_end / F.CLOSE_POINT_DEG)))
        if not self.meets_door_everywhere(p0):
            return None
        ds = d_end * np.arange(1, n + 1) / n
        return list(zip(ds.tolist(), at_many(ds), strict=True))

    def meets_door_everywhere(self, p0):
        """True when the push from tool position `p0` meets the door. The door is a slab from the hinge out to its free
        edge (length); the push turns round the hinge at a radius that only shrinks (sweep) and at a height within the
        door. So it meets the door iff, at its start -- its furthest out -- the part of the hand (and container) nearest
        the hinge is within the door's length."""
        radial = (p0 - self.h)[:2] / np.linalg.norm((p0 - self.h)[:2])
        inner = float(np.min((self.body(p0) - self.h)[:, :2] @ radial))  # the body's nearest reach toward the hinge
        return inner <= self.df.length

    def push(self, radius):
        """(start deg, sweep) for a push at `radius` (push_start, sweep), or (None, why). Computed once per close and
        relative to the hinge (GEOMETRY_CACHE): a wheelchair move shifts the door and the hand's start together."""
        key = (round(radius, 4), round(self.z - float(self.model.door["door_z"][0]), 4), tuple(np.round(self.quat, 6)), self.container,
               round(self.deg0, 3), round(self.stop_deg, 3), self.df.sign,
               tuple(np.round(self.df.n, 6)))
        hit = GEOMETRY_CACHE.get(key)
        if hit is None:
            deg0 = self.push_start(radius)
            if deg0 is None:
                hit = (None, f"at {radius * 100:.0f} cm from the hinge no push start clears the door (open "
                             f"{self.deg0:.0f} deg) by {F.CLOSE_START_GAP_M * 100:.0f} cm")
            else:
                sweep = self.sweep(radius, deg0)
                hit = ((deg0, [(d, p - self.h) for d, p in sweep]) if sweep is not None
                       else (None, f"at {radius * 100:.0f} cm from the hinge the push would miss the door"))
            GEOMETRY_CACHE[key] = hit
        if hit[0] is None:
            return hit
        return hit[0], [(d, self.h + p) for d, p in hit[1]]

    def goals(self, p_now, route_world, radius, via, q_reach=None):
        """The close's goals from the hand at `p_now` for a push at `radius`: to the push start (with `via`: in front
        of the free edge first, then sideways), the push. Returns (goals, info) or (None, why)."""
        deg0, sweep = self.push(radius)
        if deg0 is None:
            return None, sweep
        start = self.at(radius, deg0)
        end = sweep[-1][1]
        q = self.quat
        lvl = pl.HOLD_LEVEL if self.container else pl.HOLD_NONE  # level only for the container: it stays upright
        goals = []
        if via:
            b_edge = min(self.b_of(c) for box in self.door_box for c in _box_corners_at(box, self.z))
            b_safe = min(self.b_of(p_now), b_edge - F.HAND_BOX_TOOL_M[2][1] - 0.06)
            a_now = float((np.asarray(p_now) - self.h) @ self.side)
            a0 = float((start - self.h) @ self.side)
            goals += [
                pl.free(self.h + self.side * a_now + self.fwd * b_safe, q, "back in front of the free edge",
                        route_world, blend=0.05, hold=lvl, world_label="the open door"),
                pl.free(self.h + self.side * a0 + self.fwd * b_safe, q, "sideways past the hinge", route_world,
                        blend=0.05, hold=lvl, world_label="the open door"),
            ]
        if q_reach is not None:  # the arm configuration the search found clear there: cuRobo plans to those joints
            goals.append(pl.to_joints(q_reach, start, q, "to the push start", route_world, blend=0.02,
                                      world_label="the open door"))
        else:
            goals.append(pl.free(start, q, "to the push start", route_world, blend=0.02, hold=lvl,
                                 world_label="the open door"))
        # the push: ONE leg (no stop) through its points round the hinge, the hand as it is
        goals.append(pl.through([(p, q) for _, p in sweep], f"push {sweep[-1][0]:.0f} deg round the hinge",
                                self.model.cabinet(), hold=lvl, world_label="the cabinet (pushing)"))
        return goals, dict(radius=radius, start_deg=deg0, d_end=sweep[-1][0], end=end, start=start)

    def retreat_goals(self, end):
        """Back straight out from the push's end, the first of CLOSE_RETREAT_M (up to retreat_dist) that plans."""
        world = self.model.full(max(self.stop_deg - RETREAT_DOOR_DEG, 0.0))
        out = []
        for dist in F.CLOSE_RETREAT_M:
            if dist <= self.s["retreat_dist"] + 1e-9:
                out.append(pl.line(end - self.fwd * dist, self.quat, f"back out {dist * 100:.0f} cm", world,
                                   contact=False, hold=pl.HOLD_LEVEL if self.container else pl.HOLD_NONE,
                                   world_label="the door, further shut"))
        return out


def _turn(v, deg, df):
    """Vector `v` (3D, from the hinge) turned `deg` about vertical in the door's opening sense (negative: toward
    shut)."""
    xy = df._rot(deg) @ np.asarray(v, float)[:2]
    return np.r_[xy, v[2]]


def search_starts(planner, geos, heights, turns, st, fridge_boxes, p_now, route_world):
    """The push starts to hand to cuRobo, in order: every height (geos' keys), distance from the hinge (furthest first)
    and hand orientation (`turns`) whose start clears the open door (CloseGeometry.push), within the
    mount's reach and the arm's stretch, that the arm reaches (its own IK, all of a distance's starts and seeds solved
    together) with the whole arm -- its links as capsules -- clear of the door and the cabinet (`fridge_boxes`).
    At most CLOSE_TRIES_PER_RADIUS per distance. Returns ([(radius, z, yaw, geo, joints)], why none).

    Lazy: a distance's starts are worked out in order, SEARCH_CHUNK at a time (geometry, then the IK of the chunk), and
    the distance stops at its CLOSE_TRIES_PER_RADIUS-th clear one -- the same starts, in the same order, as working out
    every one first (ik_batch solves each row on its own), without the geometry and IK of the rest (11-12: computing all
    of them took ~80 of a close's 87 s)."""
    reach = shared.MAX_REACH
    fk = planner.fk
    seeds = np.array([np.asarray(st["position"], float)] + _ik_seeds(fk))
    k = len(seeds)  # every start from every seed, solved together
    radii = next(iter(geos.values())).radii()
    out, counts = [], collections.Counter()
    for radius in radii:
        planner.check_budget()  # the search runs between cuRobo's plans: it counts too

        def starts(radius=radius):  # (z, yaw, geo, start) in the order tried, at most CLOSE_SEARCH_MAX_IK
            n = 0
            for z in heights:
                for yaw, pitch in turns:
                    geo = geos[(z, yaw, pitch)]
                    goals, info = geo.goals(p_now, route_world, radius, via=False)
                    if goals is None:
                        counts["no start clears the door"] += 1
                        continue
                    start = info["start"]
                    if reach is not None and float(np.linalg.norm(start)) > reach:
                        counts["beyond the arm's reach"] += 1
                        continue
                    if not _within_stretch(fk, start, geo.quat):
                        counts["beyond the arm's stretch"] += 1
                        continue
                    yield z, yaw, geo, start
                    n += 1
                    if n >= F.CLOSE_SEARCH_MAX_IK:
                        return

        gen, n_here = starts(), 0
        while n_here < F.CLOSE_TRIES_PER_RADIUS:
            chunk = list(itertools.islice(gen, SEARCH_CHUNK))
            if not chunk:
                break
            pos = np.repeat(np.array([h[3] for h in chunk]), k, axis=0)
            quat = np.repeat(np.array([h[2].quat for h in chunk]), k, axis=0)
            q, ok = fk.ik_batch(pos, quat, np.tile(seeds, (len(chunk), 1)))
            for i, (z, yaw, geo, _start) in enumerate(chunk):
                sols = [q[i * k + j] for j in range(k) if ok[i * k + j]]
                clear = next((s_ for s_ in sols if arm_clearance(fk, s_, fridge_boxes) >= F.CLOSE_ARM_CLEAR_M), None)
                if clear is None:
                    counts["no arm configuration clear of the door" if sols else "no arm configuration"] += 1
                    continue
                out.append((radius, z, yaw, geo, clear))
                n_here += 1
                if n_here >= F.CLOSE_TRIES_PER_RADIUS:
                    break
    why = "no push start: " + ", ".join(f"{n} {k_}" for k_, n in counts.most_common()) if not out else ""
    return out, why


SEARCH_CHUNK = 16  # search_starts: push starts worked out (geometry, then IK) this many at a time


_STRETCH = {}
WRIST_BEHIND_TOOL_M = 0.12  # the flange (the wrist's end) this far behind the tool point along its z (planner's note)


def _within_stretch(fk, pos, quat):
    """False when the wrist for a tool at (`pos`, `quat`) is further from the shoulder than the arm ever puts it (the
    most of 3000 random configurations, from its own model): no IK can reach it."""
    if fk not in _STRETCH:
        rng = np.random.default_rng(1)
        lo = np.where(fk.limited, fk.lower, -np.pi)
        hi = np.where(fk.limited, fk.upper, np.pi)
        qs = lo + (hi - lo) * rng.random((3000, len(lo)))
        pos_, quat_ = fk.poses(qs)
        wrist = pos_ - R.from_quat(quat_).apply([0.0, 0.0, WRIST_BEHIND_TOOL_M])
        shoulder = fk.joint_points(np.zeros(len(lo)))[1]
        _STRETCH[fk] = (shoulder, float(np.max(np.linalg.norm(wrist - shoulder, axis=1))) + 0.01)
    shoulder, stretch = _STRETCH[fk]
    wrist = np.asarray(pos, float) - R.from_quat(quat).apply([0.0, 0.0, WRIST_BEHIND_TOOL_M])
    return float(np.linalg.norm(wrist - shoulder)) <= stretch


def arm_clearance(fk, q, boxes):
    """How far (m) the arm at joints `q` -- each link a capsule CLOSE_ARM_RADIUS_M thick along the chain from the
    shoulder to the tool -- is from `boxes` (negative: into one)."""
    pts = fk.joint_points(q)[1:]  # from the shoulder: the base doesn't move
    seg = []
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.03)) + 1)
        seg.append(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None])
    samples = np.vstack(seg)
    return min(_box_gap(samples, b) for b in boxes) - F.CLOSE_ARM_RADIUS_M


def hand_turned(quat, yaw, pitch):
    """The hand at `quat` turned `yaw` deg about vertical, after tipping its approach (tool z) `pitch` deg down."""
    rot = R.from_quat(quat)
    if pitch:
        z = rot.apply([0.0, 0.0, 1.0])
        axis = np.cross(z, [0.0, 0.0, -1.0])  # tipping z toward straight down
        if np.linalg.norm(axis) > 1e-6:
            rot = R.from_rotvec(axis / np.linalg.norm(axis) * np.radians(pitch)) * rot
    return (R.from_euler("z", yaw, degrees=True) * rot).as_quat()


_SEEDS = {}


def _ik_seeds(fk):
    """CLOSE_IK_SEEDS fixed configurations spread over the arm's joint range: where its own IK starts from when the close
    checks a push start is reachable at all."""
    if fk not in _SEEDS:
        rng = np.random.default_rng(0)
        lo = np.where(fk.limited, fk.lower, -np.pi)
        hi = np.where(fk.limited, fk.upper, np.pi)
        _SEEDS[fk] = [lo + (hi - lo) * rng.random(len(lo)) for _ in range(F.CLOSE_IK_SEEDS)]
    return _SEEDS[fk]


def container_route_gap(fk, motion, boxes, upto):
    """How close (m) the held container comes to `boxes` over the first `upto` legs of `motion` (cuRobo's route: it
    doesn't see the container)."""
    gap = np.inf
    for leg in motion.legs[:upto]:
        if not len(leg.q):
            continue
        pos, quats = fk.poses(np.asarray(leg.q)[:: max(1, len(leg.q) // 40)])
        for p, q in zip(pos, quats, strict=True):
            c = container_centre(p, q)
            gap = min(gap, min(_box_gap(c[None, :], b) for b in boxes) - F.CONTAINER_M / 2)
    return gap


def max_tilt_deg(fk, qs, quat):
    """Largest tilt (deg) of the tool over joints `qs` away from orientation `quat`: roll and pitch only -- turning
    about vertical is free (as HOLD_LEVEL)."""
    if not len(qs):
        return 0.0
    _, quats = fk.poses(qs)
    up0 = R.from_quat(quat).inv().apply([0.0, 0.0, 1.0])
    ups = R.from_quat(quats).inv().apply([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(ups @ up0, -1.0, 1.0))).max())


def plan_close(planner, door, st, settings, shift, close_to_deg, move=0.0, chair_dir=F.CHAIR_BACK_DIR, quiet=False):
    """The close from the arm state `st` against `door` shifted by `shift` (the wheelchair's move: `move` m along
    `chair_dir`, as planned): the push starts whose geometry works (distance from the hinge furthest first, then height,
    then hand turn), at most CLOSE_TRIES of them planned by cuRobo; the first whose whole motion plans. Returns (Motion,
    info) or (None, why)."""
    deg0 = float(settings["door_deg"] if settings.get("door_deg") is not None else door["door_open_deg"])
    planned = shifted_by(opened(door, deg0), shift)
    model = FridgeModel(planned)
    container = holding(settings)
    if move:  # the arm, held, against the door all along the move
        from .push_open import swept_world

        # the door where it stands now is left out: the hand is already there (the move-0 plan judged that)
        route_world = swept_world(opened(door, deg0), move, chair_dir, skip_start=True)
    else:
        route_world = model.full(deg0)
    fridge_boxes = model.full(deg0)
    lo, hi = planned["door_z"][0] + F.CLOSE_Z_MARGIN_M, planned["door_z"][1] - F.CLOSE_Z_MARGIN_M
    p_now, quat = np.asarray(st["ee_pos"][:3], float), np.asarray(st["ee_pos"][3:7], float)
    if settings.get("push_z") is not None:
        if not lo <= settings["push_z"] <= hi:
            raise Refused(f"push_z {settings['push_z']:.3f} is not within the door ({lo:.3f}-{hi:.3f}).")
        heights = [float(settings["push_z"])]
    else:  # the whole door, nearest the hand first
        heights = [float(z) for z in sorted(np.arange(lo, hi + 1e-9, F.CLOSE_Z_STEP_M), key=lambda z: abs(z - p_now[2]))]
    geos = {}
    turns = [(yaw, pitch) for pitch in ((0,) if container else F.CLOSE_PITCHES_DEG) for yaw in F.CLOSE_YAWS_DEG]
    for z in heights:
        for yaw, pitch in turns:
            geos[(z, yaw, pitch)] = CloseGeometry(planned, deg0, hand_turned(quat, yaw, pitch), z, settings,
                                                  close_to_deg, container)
            geos[(z, yaw, pitch)].pitch = pitch
    cands, why = search_starts(planner, geos, heights, turns, st, fridge_boxes, p_now, route_world)
    if not cands:
        return None, why
    for radius, z, yaw, geo, q_reach in cands[: F.CLOSE_TRIES]:
        for via in (False, True):
            goals, info = geo.goals(p_now, route_world, radius, via, q_reach=None if container else q_reach)
            # from where the arm is: a start inside an obstacle raises StartNotClear (run then tries the other moves if
            # only the fridge's boxes block it) -- but not with the door swept along a wheelchair move in the world,
            # where it rules out that move only
            motion, why = planner.motion(goals, st["position"], start_pose=(p_now, quat), from_arm=not move)
            TRIED.append((move, radius, z, yaw, via, "route + push planned" if motion is not None else str(why)[:200]))
            if motion is None:
                if "START_STATE" in str(why) and str(why).startswith(f"{goals[0].label}:"):
                    # the arm's own start is blocked in this move's world (the door swept along it): every start, hand
                    # turn and route starts from there and plans in that same world -- this move is out
                    return None, f"with this move, the arm's start pose is blocked: {why}"
                why = f"{radius * 100:.0f} cm from the hinge, z {z:.3f}, hand turned {yaw:+.0f} deg: {why}"
                if pl.goal_unreachable(why):  # a pose of it can't be reached at all: the route round won't help
                    break
                continue
            if container:  # cuRobo doesn't see the container: its route there, checked here
                gap = container_route_gap(planner.fk, motion, fridge_boxes, len(goals) - 1)
                if gap < F.CONTAINER_CLEARANCE_M:
                    why = (f"{radius * 100:.0f} cm from the hinge, z {z:.3f}, hand turned {yaw:+.0f} deg: the container "
                           f"would pass {gap * 100:.1f} cm from the door or cabinet on the way to the push")
                    TRIED.append((move, radius, z, yaw, via, why))
                    continue
            back = None
            for g in geo.retreat_goals(info["end"]):
                back, _ = planner.motion([g], motion.q_end, start_pose=(info["end"], geo.quat))
                if back is not None:
                    break
            if back is None:
                why = f"{radius * 100:.0f} cm from the hinge, z {z:.3f}, hand turned {yaw:+.0f} deg: no back-out plans"
                TRIED.append((move, radius, z, yaw, via, why))
                break
            motion = pl.Motion(motion.goals + back.goals, motion.legs + back.legs, motion.q0)
            info.update(z=z, via=via, deg0=deg0, model=model, yaw=yaw, pitch=getattr(geo, "pitch", 0))
            if not quiet:
                say(
                    f"  push {radius * 100:.0f} cm from the hinge at z {z:.3f}, starting at {info['start_deg']:.0f} deg "
                    f"round it (the door open {deg0:.0f}), the hand turned {yaw:+.0f} deg, tipped "
                    f"{getattr(geo, 'pitch', 0):.0f} deg down; {info['d_end']:.0f} deg round to the door at "
                    f"{close_to_deg:.0f} deg"
                    + (" (via in front of the free edge)" if via else "")
                    + (", holding the container" if container else "")
                )
            return motion, info
    return None, why


LAST_MOVE_WHY = []  # the last close: [(wheelchair move, why it didn't plan)] for each move tried that didn't
LAST_INFO = None  # the last close plan's push (radius, height, hand turn, start and push angles): for the logs
LAST_CHAIR_BACK = None  # the move the last close planned with (m, + back): the node says it where the goal was sent


def how_moved(move):
    if not move:
        return "none"
    return f"~{abs(move) * 100:.0f} cm {'back' if move > 0 else 'forward'}"


def move_chair(chair_back):
    """What the person does to the wheelchair (`chair_back` m: + back, - forward), as the close's and open's messages
    say it."""
    if chair_back < 0:
        return f"Move the WHEELCHAIR straight forward ~{-chair_back * 100:.0f} cm (toward the fridge, not turning it)"
    return f"Move the WHEELCHAIR straight back ~{chair_back * 100:.0f} cm (away from the fridge, without turning it)"


def run(
    ai,
    planner,
    execute,
    chair_back=None,
    close_to_deg=F.CLOSE_TO_DEG,
    chair_moved=False,
    chair_back_dir=F.CHAIR_BACK_DIR,
    speed=1.0,
    log_args=None,
    drive_chair=None,
    **settings,
):
    """The close as the node runs it, logged to RUNS_DIR/<time>_close. `planner`: this action's planning.Planner.
    `settings`: GOAL_SETTINGS' (door_deg, push_z, ...). `chair_back` None: the smallest move that plans, saved in the
    door file; an execute goal with chair_moved uses the saved one. `drive_chair` (the goal's drive_base): an execute
    goal drives the chair itself instead of needing `chair_moved`. Returns True when it finished (or the dry run
    passed); raises Refused."""
    global LAST_CHAIR_BACK, LAST_INFO
    LAST_CHAIR_BACK, LAST_INFO = None, None
    LAST_MOVE_WHY.clear()
    TRIED.clear()
    GEOMETRY_CACHE.clear()
    settings = {**{k: v for k, v in GOAL_SETTINGS.items() if k not in ("chair_back", "close_to_deg", "chair_moved")},
                **{k: v for k, v in settings.items() if v is not None or k in ("door_deg", "push_z")}}

    def task():
        global LAST_CHAIR_BACK, LAST_INFO
        nonlocal chair_back
        door = load_door()
        if "open_sign" not in door or door.get("door_open_deg") is None:
            raise Refused(f"{dc.DOOR_FILE} has no opened door: the swing writes it.")
        st = ai.get_state()
        settings["container"] = gripper_holds(st)  # the gripper closed: it holds the container, carried with the hand
        say(
            f"fridge close: door from {dc.DOOR_FILE}, open ~{door['door_open_deg']:.0f} deg; pushed to "
            f"~{close_to_deg:.0f} deg; "
            + (f"holding a {F.CONTAINER_M * 100:.1f} cm container" if holding(settings) else "no container")
            + f"; gripper {float(st['gripper_pos']):.3f} (left as is)"
        )
        found = None
        if chair_back is None and execute and chair_moved:  # the move the plan asked for was made
            if door.get("close_chair_back") is None:
                raise Refused("chair_moved, but no close has planned a wheelchair move: send a dry run first.")
            chair_back = float(door["close_chair_back"])
            say(f"the wheelchair was moved {how_moved(chair_back)}, as the last close plan asked")
        moves = F.CLOSE_CHAIR_MOVES_M if chair_back is None else (float(chair_back),)
        # with the chair already moved (chair_moved), the door is where the move put it: no move left to make
        moved_already = execute and chair_moved
        why = ""
        planner.search()
        for move in moves:
            shift = np.asarray(chair_back_dir, float) * move
            try:
                motion, info = plan_close(planner, door, st, settings, shift, close_to_deg,
                                          move=0.0 if moved_already else move, chair_dir=chair_back_dir,
                                          quiet=chair_back is None)
            except pl.StartNotClear as e:
                # the hand starts inside the door (or its handle) as planned with this move: the door moves with the
                # wheelchair, so another move may clear it -- only the chair's own layers rule
                # every move out. With no boxes named (the planner's explanation can fail), the other moves are tried:
                # a start that is really blocked fails each of them after one plan (plan_close)
                if e.blocking() and not e.only_fridge():
                    raise
                motion, info = None, str(e)
            if motion is not None:
                found = (motion, info)
                chair_back = move
                break
            why = info
            LAST_MOVE_WHY.append((move, str(why)))
            if move == 0:
                say(f"  without a wheelchair move: {why}")
        if found is None:
            if len(moves) > 1:
                lo, hi = min(moves), max(moves)
                raise Refused(f"no wheelchair move from {-lo * 100:.0f} cm forward to {hi * 100:.0f} cm back lets the "
                              f"close plan (last: {why}) -- refusing.")
            raise Refused(f"the close doesn't plan with the wheelchair moved {how_moved(chair_back)}: {why}")
        if len(moves) > 1:
            say(f"wheelchair move for the close: {how_moved(chair_back)} (the smallest that plans)")
            dc.save_door_geometry(close_chair_back=chair_back)
        LAST_CHAIR_BACK = chair_back
        motion, info = found
        LAST_INFO = {k: info.get(k) for k in ("radius", "z", "yaw", "pitch", "start_deg", "d_end")}
        say(f"  planning: {planner.summary()}")
        if not execute:
            say("(plan only: planned as if the wheelchair has already moved"
                + (f"; with execute true and drive_base the base drives it {how_moved(chair_back)} first)"
                   if drive_chair is not None and chair_back else ")"))
            return True
        driven = False
        if chair_back and not chair_moved:
            if drive_chair is None:
                raise Refused(
                    f"{move_chair(chair_back)}, then send this goal again with chair_moved true. If it is still back "
                    "from the last close, leave it."
                )
            say(f"\n>>> drive_base: driving the wheelchair {how_moved(chair_back)} for the close")
            moved = drive_chair(chair_back)  # within the base_tolerance of chair_back, or it raises
            driven = True
            shift = np.asarray(chair_back_dir, float) * moved
            say(f"the close planned again against the door shifted by the {moved * 100:.1f} cm the base moved")
            planner.search()  # a new search: the drive doesn't count against it
            motion, info = plan_close(planner, door, ai.get_state(), settings, shift, close_to_deg)
            if motion is None:
                raise Refused(f"after the wheelchair move the close no longer plans ({info}) -- the arm holds.")
        planner.join(motion, info["model"].cabinet())
        if holding(settings):  # the container stays upright: never tilted
            tilt = max_tilt_deg(planner.fk, motion.joined.q, st["ee_pos"][3:7])
            say(f"hand tilt over the whole close: at most {tilt:.1f} deg (limit {F.CLOSE_MAX_TILT_DEG:.0f})")
            if tilt > F.CLOSE_MAX_TILT_DEG:
                raise Refused("the hand would tilt -- refusing, arm untouched.")
        ok = planner.run(motion, "close")
        if ok:
            save_door_geometry(door_open_deg=float(close_to_deg))
            say(
                f"door left at ~{close_to_deg:.0f} deg to swing shut. "
                + ("The base stays where it drove: drive" if driven else "Move")
                + " the wheelchair forward again before the next open (it detects the door afresh)."
            )
        return ok

    ok = logged_run("close", ai, log_args or {}, speed, task)
    say(f"\nCLOSE {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
    return ok

