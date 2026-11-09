"""Close the fridge: push the open door most of the way shut with the hand, which may hold a container -- no grasp.

The motion, all planned by cuRobo (planning.py), every leg holding the hand's tilt (HOLD_LEVEL: it never tips what it
holds; turning about vertical is free on the way):
1. To the swing start (a free leg; the open door, its handle and the cabinet in the world): `side_past_hinge` sideways
   past the hinge (further, until the hand clears the open door by CLOSE_START_GAP_M) and `push_radius` from it, at the
   swing height, the hand as it is now or turned about vertical (CLOSE_YAWS_DEG: never tilted). cuRobo finds the way there round the open door; only if it finds none is it
   given the old route: back in front of the door's free edge, then sideways past the hinge.
2. The swing: ONE leg (local IK, a pose every CLOSE_POINT_DEG) round the hinge toward shut, the radius spiralling in to
   `push_radius_end`, until the hand (its real size, HAND_BOX_TOOL_M) meets the door at `close_to_deg`: the door
   swings shut by itself from there. It is checked against the cabinet only: it pushes the door.
3. Back straight out up to `retreat_dist` (the door, now further shut, in the world).
One trajectory, stopping at the end of the swing. The door is planned where the open saved it, shifted by the
wheelchair's move and by `hold_offset_x` (the container: the hand stays back by its size, so it -- not the hand --
reaches the door; it is not in the collision model).

Before that, the wheelchair is moved straight back or forward by the smallest of CLOSE_CHAIR_MOVES_M whose plan passes
(or by the goal's chair_back); the arm, held, must clear the door all through the move (the swing start's route is
planned from where it is against the door at CHAIR_MOVE_STEPS points along the move). A plan saves its move in the
door file, and an execute goal with chair_moved uses it; with the goal's `drive_base` the base drives it instead and
the close is planned again against the door shifted by the move the base really made. Swing heights are tried within
the door, nearest the hand first (CLOSE_Z_TRIES), unless the goal gives push_z.
"""

import numpy as np
from door_common import common as dc
from door_common.common import Refused, save_door_geometry, say
from scipy.spatial.transform import Rotation as R

from . import constants as F
from . import planning as pl
from .motion import load_door, logged_run, opened, shifted_by
from .world import FridgeModel

# The goal's settings, and their defaults: F.CLOSE_SETTINGS holds what a close of this fridge doesn't vary.
GOAL_SETTINGS = {
    "door_deg": None,  # how far the door is open now (deg); None: the door file's
    **{k: F.CLOSE_SETTINGS[k] for k in ("side_past_hinge", "retreat_dist")},
    "push_z": None,  # the swing's height (m); None: searched within the door
    **{k: F.CLOSE_SETTINGS[k] for k in ("push_radius", "push_radius_end", "hold_offset_x")},
    "chair_back": None,  # how far the chair is moved back (m, - forward); 0: not moved; None: the plan picks it
    "close_to_deg": F.CLOSE_TO_DEG,  # swing the door shut until it is open this far (deg)
    "chair_moved": False,  # the chair has been moved as the plan asked: required to execute with a move
}
RETREAT_DOOR_DEG = 20.0  # the back-out plans with the door this much further shut than where the swing left it


def _box_gap(points, box):
    """Smallest signed distance (m) from `points` to obstacle dict `box` (yaw-only), negative inside."""
    c = np.asarray(box["position"], float)
    rot = R.from_euler("z", box["rpy_deg"][2], degrees=True)
    local = rot.inv().apply(np.asarray(points, float) - c)
    q = np.abs(local) - np.asarray(box["dims"], float) / 2
    return float(np.min(np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)))


def hand_gap(model, deg, pos, quat):
    """How far the hand's real box (HAND_BOX_TOOL_M) is from the door slab open `deg` (m, negative: into it)."""
    # a grid on the box's faces, corners included: the slab is thin, its corners alone could straddle it
    pts = _hand_points(pos, quat)
    return min(_box_gap(pts, b) for b in model.slab(deg))


def _hand_points(pos, quat, n=4):
    """Points on the surface of the hand's box (corners included), in arm_mount."""
    box = F.HAND_BOX_TOOL_M
    axes = [np.linspace(lo, hi, n) for lo, hi in box]
    pts = []
    for k in range(3):  # each pair of opposite faces: a grid on it
        for side in box[k]:
            g = np.meshgrid(*[axes[i] if i != k else np.array([side]) for i in range(3)], indexing="ij")
            pts.append(np.stack([x.ravel() for x in g], axis=1))
    return np.asarray(pos, float) + R.from_quat(quat).apply(np.vstack(pts))


class CloseGeometry:
    """Where the close's hand goes, for the door `door` (as planned: shifted) open `deg0`, the hand turned `quat`, the
    swing at height `z`."""

    def __init__(self, door, deg0, quat, z, settings, stop_deg):
        self.model = FridgeModel(door)
        self.df = df = self.model.df
        self.deg0, self.quat, self.z, self.stop_deg = deg0, np.asarray(quat, float), float(z), float(stop_deg)
        self.s = settings
        self.fwd = np.r_[-df.n, 0.0]  # into the fridge, the closed door's normal
        self.side = np.r_[-df.e, 0.0]  # along the closed door, toward the hinge
        self.h = np.r_[df.hinge[:2], self.z]

    def pos(self, a, b):
        return self.h + self.side * a + self.fwd * b

    def a_of(self, p):
        return float((np.asarray(p) - self.h) @ self.side)

    def b_of(self, p):
        return float((np.asarray(p) - self.h) @ self.fwd)

    def swing_start(self):
        """(a, b) of the swing start: side_past_hinge past the hinge, or further until the hand clears the open door by
        CLOSE_START_GAP_M; None if none within 25 cm."""
        r = self.s["push_radius"]
        a = self.s["side_past_hinge"]
        while a < min(r, 0.25):
            b = -np.sqrt(r**2 - a**2)
            if hand_gap(self.model, self.deg0, self.pos(a, b), self.quat) >= F.CLOSE_START_GAP_M:
                return a, b
            a += 0.01
        return None

    def sweep(self, a0, b0):
        """The swing's points [(d deg round the hinge, pos)] from the start (a0, b0) to where the hand meets the door
        at stop_deg; None if it never does within 60 deg past the door."""
        r0 = self.pos(a0, b0) - self.h
        span = max(np.degrees(np.arccos(np.clip(r0[:2] @ self.df.e / np.linalg.norm(r0[:2]), -1, 1))) - 15.0, 10.0)
        ratio = self.s["push_radius_end"] / self.s["push_radius"]

        def at(d):
            shrink = 1.0 - (1.0 - ratio) * min(d / span, 1.0)
            return self.h + np.r_[self.df._rot(-d) @ r0[:2] * shrink, 0.0]

        def gap(d):
            return hand_gap(self.model, self.stop_deg, at(d), self.quat)

        d, step = 0.0, 1.0
        while gap(d + step) > 0.0:
            d += step
            if d > self.deg0 + 60:
                return None
        lo, hi = d, d + step  # land on the touch
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if gap(mid) > 0.0 else (lo, mid)
        d_end = hi
        n = max(1, int(np.ceil(d_end / F.CLOSE_POINT_DEG)))
        return [(d_end * k / n, at(d_end * k / n)) for k in range(1, n + 1)]

    def goals(self, p_now, route_world, via):
        """The close's goals from the hand at `p_now`: to the swing start (with `via`: in front of the free edge first,
        then sideways), the swing, the back-out. Returns (goals, info) or (None, why)."""
        start = self.swing_start()
        if start is None:
            return None, "no swing start within 25 cm past the hinge clears the open door"
        a0, b0 = start
        sweep = self.sweep(a0, b0)
        if sweep is None:
            return None, "the swing never reaches the door in the model"
        end = sweep[-1][1]
        u_end = float((end[:2] - self.df.hinge[:2]) @ self.df.e)
        if u_end + F.CLOSE_HAND_HALF_M > self.df.u_handle - F.CLOSE_HANDLE_MARGIN_M:
            return None, "the hand would end on the handle -- lower push_radius"
        q, lvl = self.quat, pl.HOLD_LEVEL
        goals = []
        if via:
            free_edge = self.model.df.pos(self.df.length, 0.0, self.deg0, self.z)
            b_safe = min(self.b_of(p_now), self.b_of(free_edge) - F.HAND_BOX_TOOL_M[2][1] - 0.06)
            goals += [
                pl.free(self.pos(self.a_of(p_now), b_safe), q, "back in front of the free edge", route_world,
                        blend=0.05, hold=lvl, world_label="the open door"),
                pl.free(self.pos(a0, b_safe), q, "sideways past the hinge", route_world, blend=0.05, hold=lvl,
                        world_label="the open door"),
            ]
        goals.append(pl.free(self.pos(a0, b0), q, "to the swing start", route_world, blend=0.02, hold=lvl,
                             world_label="the open door"))
        # the swing: ONE leg (no stop) through its points round the hinge, the hand as it is
        goals.append(pl.through([(p, q) for _, p in sweep], f"swing {sweep[-1][0]:.0f} deg round the hinge",
                                self.model.cabinet(), hold=lvl, world_label="the cabinet (pushing)"))
        return goals, dict(a0=a0, b0=b0, d_end=sweep[-1][0], u_end=u_end, end=end)

    def retreat_goals(self, end):
        """Back straight out from the swing's end, the first of CLOSE_RETREAT_M (up to retreat_dist) that plans."""
        world = self.model.full(max(self.stop_deg - RETREAT_DOOR_DEG, 0.0))
        out = []
        for dist in F.CLOSE_RETREAT_M:
            if dist <= self.s["retreat_dist"] + 1e-9:
                out.append(pl.line(end - self.fwd * dist, self.quat, f"back out {dist * 100:.0f} cm", world,
                                   contact=False, hold=pl.HOLD_LEVEL, world_label="the door, further shut"))
        return out


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
    `chair_dir`, as planned) and by hold_offset_x: the first swing height whose whole motion plans. Returns (Motion,
    info) or (None, why)."""
    deg0 = float(settings["door_deg"] if settings.get("door_deg") is not None else door["door_open_deg"])
    base = opened(door, deg0)
    hold_dxy = (float(settings["hold_offset_x"]), 0.0)
    planned = shifted_by(shifted_by(base, shift), hold_dxy)
    model = FridgeModel(planned)
    if move:  # the arm, held, against the door all along the move
        from .push_open import swept_world

        # the door where it stands now is left out: the hand is already there (the move-0 plan judged that)
        route_world = swept_world(shifted_by(base, hold_dxy), move, chair_dir, skip_start=True)
    else:
        route_world = model.full(deg0)
    lo, hi = planned["door_z"][0] + F.CLOSE_Z_MARGIN_M, planned["door_z"][1] - F.CLOSE_Z_MARGIN_M
    p_now, quat = np.asarray(st["ee_pos"][:3], float), np.asarray(st["ee_pos"][3:7], float)
    if settings.get("push_z") is not None:
        if not lo <= settings["push_z"] <= hi:
            raise Refused(f"push_z {settings['push_z']:.3f} is not within the door ({lo:.3f}-{hi:.3f}).")
        heights = [float(settings["push_z"])]
    else:
        heights = sorted(np.arange(lo, hi + 1e-9, F.CLOSE_Z_STEP_M), key=lambda z: abs(z - p_now[2]))
        heights = [float(z) for z in heights[: F.CLOSE_Z_TRIES]]
    why = "no height tried"
    for z, yaw in [(z, yaw) for z in heights for yaw in F.CLOSE_YAWS_DEG]:
        # the hand turned `yaw` about vertical (its tilt kept: whatever it holds stays upright)
        q_yaw = (R.from_euler("z", yaw, degrees=True) * R.from_quat(quat)).as_quat()
        geo = CloseGeometry(planned, deg0, q_yaw, z, settings, close_to_deg)
        for via in (False, True):
            goals, info = geo.goals(p_now, route_world, via)
            if goals is None:
                why = f"z {z:.3f}, hand turned {yaw:+.0f} deg: {info}"
                break
            # from where the arm is: a start inside an obstacle raises StartNotClear (run then tries the other moves if
            # only the fridge's boxes block it) -- but not with the door swept along a wheelchair move in the world,
            # where it rules out that move only
            motion, why = planner.motion(goals, st["position"], start_pose=(p_now, quat), from_arm=not move)
            if motion is None:
                if "START_STATE" in str(why) and str(why).startswith(f"{goals[0].label}:"):
                    # the arm's own start is blocked in this move's world (the door swept along it): every height,
                    # hand turn and route starts from there and plans in that same world -- this move is out
                    return None, f"with this move, the arm's start pose is blocked: {why}"
                why = f"z {z:.3f}, hand turned {yaw:+.0f} deg: {why}"
                if pl.goal_unreachable(why):  # a pose of it can't be reached at all: the route round won't help
                    break
                continue
            back = None
            for g in geo.retreat_goals(info["end"]):
                back, _ = planner.motion([g], motion.q_end, start_pose=(info["end"], q_yaw))
                if back is not None:
                    break
            if back is None:
                why = f"z {z:.3f}, hand turned {yaw:+.0f} deg: no back-out from the swing's end plans"
                break
            motion = pl.Motion(motion.goals + back.goals, motion.legs + back.legs, motion.q0)
            info.update(z=z, via=via, deg0=deg0, model=model, yaw=yaw)
            if not quiet:
                say(
                    f"  z {z:.3f}: swing start {info['a0'] * 100:+.0f} cm past the hinge, {info['b0'] * 100:+.0f} cm "
                    f"out, the hand turned {yaw:+.0f} deg; {info['d_end']:.0f} deg round it to the door at "
                    f"{close_to_deg:.0f} deg"
                    + (" (via in front of the free edge)" if via else "")
                )
            return motion, info
    return None, why


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
    global LAST_CHAIR_BACK
    LAST_CHAIR_BACK = None
    settings = {**{k: v for k, v in GOAL_SETTINGS.items() if k not in ("chair_back", "close_to_deg", "chair_moved")},
                **{k: v for k, v in settings.items() if v is not None or k in ("door_deg", "push_z")}}

    def task():
        global LAST_CHAIR_BACK
        nonlocal chair_back
        door = load_door()
        if "open_sign" not in door or door.get("door_open_deg") is None:
            raise Refused(f"{dc.DOOR_FILE} has no opened door: the swing writes it.")
        st = ai.get_state()
        say(
            f"fridge close: door from {dc.DOOR_FILE}, open ~{door['door_open_deg']:.0f} deg; swung to "
            f"~{close_to_deg:.0f} deg; holding a {F.CONTAINER_M * 100:.1f} cm container (hand kept "
            f"{-settings['hold_offset_x'] * 100:.1f} cm back, incl. {F.CONTAINER_CLEARANCE_M * 100:.1f} cm margin); "
            f"gripper {float(st['gripper_pos']):.3f} (left as is)"
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

