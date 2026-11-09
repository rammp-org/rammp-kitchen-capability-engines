"""The push-open, the open's default: after the PUSH_OPEN_PULL_DEG pull, stopped there, and the release (the gripper
opened with the hand still), off the handle, round the door's free edge, and the door pushed on to PUSH_OPEN_TO_DEG with the
side of the hand -- one trajectory under joint impedance, the wheelchair moved first if the push needs it.

The motion (planning.py), planned with the open before anything moves and again from where the release leaves the arm:
1. Straight back off the handle BACK_OFF_M (local IK; the door with a window round the hand in cuRobo's check).
2. Round the open door to the push start (cuRobo; door, handle, cabinet in its world): in behind the door,
   PUSH_OPEN_GAP_M off its inner face, the hand turned PARALLEL to the door (fingers along it toward the hinge, rolled
   so its flat side faces it) and PUSH_OPEN_DZ_M off the handle's height. cuRobo finds the way round the free edge;
   only if it finds none is it given one point, PUSH_OPEN_EDGE_CLEAR_M past the edge.
3. ONE leg (local IK, checked against the cabinet: it pushes the door): onto the inner face until the side of the
   hand is PUSH_OPEN_PRELOAD_M past touching it (the real hand's size, HAND_BOX_TOOL_M), round with the door to
   PUSH_OPEN_TO_DEG, the hand held in the door's frame, then PUSH_OPEN_AWAY_M straight off it.
   Then, in the same leg (no stop, the hand turning the short way), to in front of the middle of the fridge, facing
   into it, upright (finish_pose), the hand's real box PUSH_OPEN_FINISH_CLEAR_M clear of the open door and its handle
   all the way. The hand placements are tried for that first (plan_finished_push); if none plans, the push-open ends
   at the step-off.
A few hand placements are tried in order (PUSH_OPEN_AT_M x PUSH_OPEN_APPROACH_DEG x tool y up/down); the first whose
whole motion plans.

If none plans with the wheelchair where it is, the smallest move of CHAIR_MOVES_M after which one does (+ back,
- forward): the back-off runs alone, the hand waits while the chair moves (the push start's route is planned
from there against the door at CHAIR_MOVE_STEPS points along the move), and the rest is planned again against the door
where it really is. Without drive_base the goal ends there, asking for the move (NEXT_STEP); an open goal with
chair_moved then pushes (push_after_chair_move).
"""

import numpy as np
from door_common.common import DoorFrame, Refused, save_door_geometry, say
from scipy.spatial.transform import Rotation as R

from . import constants as F
from . import planning as pl
from .close_door import _box_gap, _hand_points, move_chair
from .motion import load_door, opened, shifted
from .world import FridgeModel, hand_reach

NEXT_STEP = ""  # what the person does next, said where the goal was sent (the node reads it): the wheelchair move


def hand_quat(df, deg, approach_deg, y_up):
    """The hand behind the door open `deg`: approach level, `approach_deg` off "along the door toward the hinge"
    toward "into the cabinet"; tool y up/down (`y_up` +-1); then rolled PUSH_OPEN_ROLL_DEG about the approach."""
    e2, n2 = df._rot(deg) @ df.e, df._rot(deg) @ df.n
    a = np.radians(approach_deg)
    z = np.r_[-(np.cos(a) * e2 + np.sin(a) * n2), 0.0]
    y = np.array([0.0, 0.0, y_up])
    return (
        R.from_matrix(np.column_stack([np.cross(y, z), y, z])) * R.from_euler("z", F.PUSH_OPEN_ROLL_DEG, degrees=True)
    ).as_quat()


def facing_fridge(df, y_up):
    """Approach along the closed door's inward normal (into the fridge), tool y up (+1, upright -- as the grasp) or
    down (-1)."""
    fwd = -np.r_[df.n, 0.0]
    y = np.array([0.0, 0.0, y_up])
    return R.from_matrix(np.column_stack([np.cross(y, fwd), y, fwd])).as_quat()


def finish_pose(door, out=None):
    """Where the push-open ends: in front of the middle of the fridge's opening (PUSH_OPEN_FINISH_U of the door's width
    from the hinge, `out` -- default the first of PUSH_OPEN_FINISH_OUT_M -- out from the closed door's face, the door's
    mid-height), facing into the fridge, upright. (pos, quat)."""
    out = F.PUSH_OPEN_FINISH_OUT_M[0] if out is None else out
    df = DoorFrame(door, float(door.get("open_sign", 1.0)))
    lo, hi = (float(v) for v in door["door_z"])
    pos = df.pos(df.length * F.PUSH_OPEN_FINISH_U, df.t / 2 + out, 0.0, (lo + hi) / 2)
    return pos, facing_fridge(df, 1.0)


def back_off_goal(model, deg, pose):
    """Straight back off the handle from the tool pose `pose` (pos, quat) holding it, the door open `deg`."""
    pos, quat = np.asarray(pose[0], float), np.asarray(pose[1], float)
    approach = R.from_quat(quat).as_matrix()[:, 2]
    world = model.holding((deg,), model.df.u_handle, float(model.door.get("handle_mid_z", model.door["closed_handle"][2])),
                          F.HAND_WINDOW_U_M, F.HAND_WINDOW_Z_M)
    return pl.line(
        pos - approach * F.BACK_OFF_M, quat, f"back {F.BACK_OFF_M * 100:.0f} cm off the handle", world,
        contact=False, blend=0.03, world_label="the door, hand window",
    )


def placements():
    """(u along the door from the hinge, fingers off along-the-door, tool y up/down) in the order tried."""
    return [(u, a, y) for u in F.PUSH_OPEN_AT_M for a in F.PUSH_OPEN_APPROACH_DEG for y in (1.0, -1.0)]


def push_goals(model, deg0, z_hold, placement, route_world, via_from=None, finish=None):
    """The goals from the hand off the handle to the end of the push, for one hand placement: (optionally) a point past
    the free edge, the push start, onto the inner face, round with the door. `route_world`: what the route to the push
    start avoids. `via_from`: the back-off's tool position -- add the point past the free edge (its height and
    distance off the door). `finish` (pos, quat): after the step-off, on to there in the same leg."""
    u_push, approach_deg, y_up = placement
    df = model.df
    z_in = z_hold + F.PUSH_OPEN_DZ_M
    q_in = hand_quat(df, deg0, approach_deg, y_up)
    # the hand's side toward the inner face: how far the side of the hand reaches past the tool point toward the door
    side = hand_reach(q_in, np.r_[df._rot(deg0) @ df.n, 0.0], F.HAND_BOX_TOOL_M)
    v_push = -df.t / 2 - side + F.PUSH_OPEN_PRELOAD_M
    v_in = -(df.t / 2 + F.PUSH_OPEN_GAP_M)
    goals = []
    if via_from is not None:
        d = df._rot(deg0).T @ (np.asarray(via_from[:2]) - df.hinge[:2])
        v_out = max(float(d @ df.n) - df.mid, F.AROUND_OUT_M)
        goals.append(pl.free(df.pos(df.length + F.PUSH_OPEN_EDGE_CLEAR_M, v_out, deg0, float(via_from[2])), q_in,
                             "past the free edge", route_world, blend=0.06, world_label="the open door"))
    goals.append(pl.free(df.pos(u_push, v_in, deg0, z_in), q_in, "in behind the door", route_world, blend=0.02,
                         world_label="the open door"))
    # the push: ONE leg onto the inner face and round with the door (the hand held in the door's frame, a pose every
    # PUSH_OPEN_POINT_DEG) to exactly PUSH_OPEN_TO_DEG, where it comes to REST, still against the door: the door isn't
    # held, so a hand still moving at 90 would let it coast on past. Its last PUSH_EASE_DEG at most PUSH_END_JOINT_SPEED
    # whatever the goal's speed (push_caps), so the door is as slow at the end however fast the push ran
    n = max(1, int(np.ceil((F.PUSH_OPEN_TO_DEG - deg0) / F.PUSH_OPEN_POINT_DEG)))
    degs = [deg0] + [deg0 + (F.PUSH_OPEN_TO_DEG - deg0) * k / n for k in range(1, n + 1)]
    poses = [(df.pos(u_push, v_push, d, z_in), q_in if k == 0 else hand_quat(df, d, approach_deg, y_up))
             for k, d in enumerate(degs)]
    worlds = {}
    for d in degs:
        if d not in worlds:  # each step against the door where it is then: its front panel, edges, inner part all but
            # round the hand
            worlds[d] = (f"the door at {d:.0f} deg (its front and edges; the hand's window left out)",
                         model.pushed(d, u_push, z_in, *F.PUSH_DOOR_WINDOW_M, F.PUSH_DOOR_PANEL_M))
    goals.append(pl.through(poses, f"onto the inner face, push to {F.PUSH_OPEN_TO_DEG} deg and stop there",
                            model.cabinet(), caps=push_caps(degs), world_label="the cabinet (pushing)",
                            segment_worlds=[worlds[d] for d in degs]))
    # then, from rest, straight off the inner face -- the close starts from there, the door left at 90 stays -- and on
    # to in front of the fridge (`finish`), one more leg
    deg = float(F.PUSH_OPEN_TO_DEG)
    off = [(df.pos(u_push, v_push - F.PUSH_OPEN_AWAY_M, deg, z_in), hand_quat(df, deg, approach_deg, y_up))]
    label = f"{F.PUSH_OPEN_AWAY_M * 100:.0f} cm off the door"
    if finish is not None:
        off.append((np.asarray(finish[0], float), np.asarray(finish[1], float)))
        label += ", then in front of the fridge"
    goals.append(pl.through(off, label, model.cabinet(), world_label="the cabinet (off the door)",
                            segment_worlds=[worlds[deg]] * len(off)))
    return goals


def push_caps(degs):
    """The push leg's speed caps (fractions of planning.LOCAL_JOINT_SPEED) at the door angles `degs`: 1 until the last
    PUSH_EASE_DEG, then at most PUSH_END_JOINT_SPEED rad/s at the speed the run's contact legs get (planning.Planner.join
    scales a local leg by rammp_arm's joint and contact speeds, which the goal's `speed` has scaled already: this is
    planned inside the run). The leg ends at rest either way; this makes how slowly it gets there the same at any
    speed."""
    import rammp_arm

    scale = rammp_arm.JOINT_SPEED * rammp_arm.contact_speed_scale() / pl.LOCAL_JOINT_SPEED
    end = min(1.0, F.PUSH_END_JOINT_SPEED / max(1e-6, pl.LOCAL_JOINT_SPEED * scale))
    return [end if d >= F.PUSH_OPEN_TO_DEG - F.PUSH_EASE_DEG - 1e-6 else 1.0 for d in degs]


def finish_gap(model, step_off, finish):
    """How close (m) the hand's real box (HAND_BOX_TOOL_M) comes to the door open PUSH_OPEN_TO_DEG or its handle on the
    way from the step-off to `finish` (both (pos, quat)), the hand turning evenly as the leg does."""
    goal = pl.Goal(np.asarray(finish[0], float), np.asarray(finish[1], float), "", [], path="line")
    poses, _ = pl.path_poses(goal, np.asarray(step_off[0], float), np.asarray(step_off[1], float))
    deg = float(F.PUSH_OPEN_TO_DEG)
    boxes = model.slab(deg) + model.handle(deg)
    return min(min(_box_gap(_hand_points(p, q), b) for b in boxes) for p, q in poses)


def plan_finished_push(planner, door, q_off, off_pose, route_world, model=None, from_arm=False):
    """plan_push, ending in front of the fridge (finish_pose, the first of PUSH_OPEN_FINISH_OUT_M out from it that any
    hand placement plans); else the push ending at its step-off. Returns (Motion, placement, how the push ends in
    words) or (None, why, "")."""
    no_finish = "no stand-off tried"
    for out in F.PUSH_OPEN_FINISH_OUT_M:
        fin = finish_pose(door, out)
        push, placed = plan_push(planner, door, q_off, off_pose, route_world, model, from_arm, finish=fin)
        if push is not None:
            return push, placed, f"then {out * 100:.0f} cm in front of the fridge, facing it"
        no_finish = placed
    push, placed = plan_push(planner, door, q_off, off_pose, route_world, model, from_arm)
    if push is None:
        return None, placed, ""
    return push, placed, f"ends {F.PUSH_OPEN_AWAY_M * 100:.0f} cm off the door (not on to in front of it: {no_finish})"


def plan_push(planner, door, q_off, off_pose, route_world, model=None, from_arm=False, finish=None):
    """The push from the hand off the handle (joints `q_off`, tool `off_pose`), the door as `door`: the first hand
    placement whose whole motion plans, cuRobo routing straight to the push start first, then via a point past the
    free edge. `finish` (pos, quat): the push's leg goes on to there (push_goals), the hand's real box at least
    PUSH_OPEN_FINISH_CLEAR_M off the door all the way. Returns (Motion, placement) or (None, why)."""
    model = model or FridgeModel(door)
    deg0 = float(door["door_open_deg"])
    why = "no placement tried"
    for placement in placements():
        for via in (None, off_pose[0]):
            goals = push_goals(model, deg0, float(off_pose[0][2]), placement, route_world, via_from=via, finish=finish)
            if finish is not None:
                gap = finish_gap(model, goals[-1].through[-2], finish)
                if gap < F.PUSH_OPEN_FINISH_CLEAR_M:
                    why = f"on to in front of the fridge, the hand would come {gap * 100:.1f} cm from the door"
                    break
            motion, why = planner.motion(goals, q_off, start_pose=off_pose, from_arm=from_arm)
            if motion is not None:
                return motion, (*placement, via is not None)
            if "START_STATE" in why:  # the hand's start is blocked in this world: no placement changes that
                return None, why
            if pl.goal_unreachable(why):  # a pose of it can't be reached at all: a via point won't help
                break
    return None, why


def swept_world(door, move, chair_dir, skip_start=False):
    """The open door and its handle at CHAIR_MOVE_STEPS points along a wheelchair move of `move` (the arm, held, must
    clear them all) and the cabinet where the move ends. `skip_start`: leave out the door where it stands now (the
    points then run from one step along the move to its end): the arm is already there, maybe within the planner's
    padding of it, and every plan would refuse its start for a door the move takes away."""
    out = []
    steps = np.linspace(0.0, move, F.CHAIR_MOVE_STEPS + int(skip_start))[int(skip_start):]
    for k, s in enumerate(steps):
        m = FridgeModel(shifted(door, float(s), chair_dir), prefix=f"m{k}_")
        deg = float(door["door_open_deg"])
        out += m.slab(deg) + m.handle(deg)
    return out + FridgeModel(shifted(door, move, chair_dir)).cabinet()


def _how(move):
    return f"~{abs(move) * 100:.0f} cm {'back' if move > 0 else 'forward'}"


def _after(move):
    """How the plan goes on after the back-off, in words."""
    return "no wheelchair move needed" if not move else f"after the wheelchair moves {_how(move)}"


class PushOpen:
    """What follows the pull: plans the back-off, wheelchair move and push from the hand on the handle, the gripper
    open (`plan`), and runs them (`execute`), re-planned from where the pull really ended."""

    def __init__(self, chair_dir=F.CHAIR_BACK_DIR):
        self.chair_dir = chair_dir
        self.chair_move = None
        self.motion = None  # back-off + push (no move), or the back-off alone (a move)
        self.push = None  # with a move: the push after it, as planned
        self.placement = None
        self.finish = ""  # how the push ends (plan_finished_push), in words

    def plan(self, planner, door, q_hold, hold_pose, from_arm=False):
        """From the hand on the handle (joints `q_hold`, tool `hold_pose`) with the door as `door` (open as far as the
        pull leaves it). `from_arm`: that is where the arm is now. Returns (ok, message)."""
        model = FridgeModel(door)
        deg0 = float(door["door_open_deg"])
        back = back_off_goal(model, deg0, hold_pose)
        off, why = planner.motion([back], q_hold, start_pose=hold_pose, from_arm=from_arm)
        if off is None:
            return False, f"push-open: {why}"
        off_pose = (back.pos, back.quat)
        for move in F.CHAIR_MOVES_M:
            if move == 0:
                moved, route = door, model.full(deg0)
            else:
                moved, route = shifted(door, move, self.chair_dir), swept_world(door, move, self.chair_dir)
            push, placed, finish = plan_finished_push(planner, moved, off.q_end, off_pose, route, FridgeModel(moved))
            if push is None:
                why = placed
                continue
            self.chair_move, self.placement = move, placed
            self.finish = finish
            if move == 0:
                self.motion, self.push = pl.Motion(off.goals + push.goals, off.legs + push.legs, off.q0), None
            else:
                self.motion, self.push = off, push
            return True, f"push-open OK, {_after(move)} ({self.describe()}); the push {self.finish}"
        moves = ", ".join(f"{m * 100:+.0f}" for m in F.CHAIR_MOVES_M)
        return False, f"push-open: no hand placement behind the door plans (wheelchair moves {moves} cm; last: {why})"

    def describe(self):
        u_push, approach_deg, y_up, via = self.placement
        return (
            f"hand {u_push * 100:.0f} cm from the hinge, fingers {approach_deg:+.0f} deg off along-the-door, tool y "
            f"{'up' if y_up > 0 else 'down'}{', via a point past the free edge' if via else ''}"
        )

    def dry_run(self, drive_chair):
        """The dry run's NEXT_STEP: what an executed open would do after the pull."""
        global NEXT_STEP
        move = self.chair_move or 0.0
        if not move:
            NEXT_STEP = "Plan OK: after the pull it lets go, backs off the handle and pushes the door open (no wheelchair move)."
        elif drive_chair is not None:
            NEXT_STEP = (
                f"Plan OK: after the pull it lets go, backs off the handle, DRIVES THE BASE {_how(move)} "
                "(drive_base) and pushes the door open."
            )
        else:
            NEXT_STEP = (
                f"Plan OK: after the pull it lets go, backs off the handle and stops, asking you to "
                f"{move_chair(move)[0].lower() + move_chair(move)[1:]}; then send the open with execute true and "
                "chair_moved true to push the door open."
            )
        say(f"  {NEXT_STEP}")

    def execute(self, ai, planner, drive_chair=None):
        """After the swing (stopped at its angle) and the release (the gripper opened with the hand still): planned
        again from where the arm is (else the plan checked before the swing, if the arm is where it starts), then off the
        bar, round the door and the push as ONE motion under joint impedance. With a wheelchair move: the back-off,
        then the move (driven, or asked for)."""
        global NEXT_STEP
        say("\n===== off the handle, round the door, push further open =====")
        door = load_door()
        deg0 = float(door["door_open_deg"])
        if deg0 > F.PUSH_OPEN_TO_DEG - F.PUSH_OPEN_MIN_PUSH_DEG + 2:  # a pull may overshoot a little
            raise Refused(f"the door is already ~{deg0:.0f} deg open (push to {F.PUSH_OPEN_TO_DEG}).")
        say(f"  {after_release(self, planner, ai, door)}")
        planner.join(self.motion, FridgeModel(door).cabinet())
        if not self.chair_move:
            ok = planner.run(self.motion, "off the handle + round the door + push-open", impedance=AFTER_RELEASE)
            say(f"PUSH-OPEN {'DONE' if ok else 'STOPPED SHORT'}")
            if ok:
                save_door_geometry(door_open_deg=float(F.PUSH_OPEN_TO_DEG))
            return ok
        if not planner.run(self.motion, "off the handle", impedance=AFTER_RELEASE):
            return False
        move = self.chair_move
        if drive_chair is not None:
            # the hand stays where the back-off left it, clear of the door all through the move (planned so); the
            # push is then planned against the door shifted by the move the base really made
            say(f"\n>>> drive_base: driving the wheelchair {_how(move)} for the push")
            moved = drive_chair(move)  # within the base_tolerance of `move`, or it raises
            save_door_geometry(push_after_chair_move=float(moved))
            return push_after_chair_move(ai, planner, True, self.chair_dir)
        save_door_geometry(push_after_chair_move=float(move))
        NEXT_STEP = (
            f"Pull done and the hand is off the handle. {move_chair(move)}, then send the open with execute true and "
            "chair_moved true to push the door open."
        )
        say(f"\n>>> {NEXT_STEP}")
        return True


AFTER_RELEASE = {**F.PULL_GAINS, "path_tol": F.AFTER_RELEASE_PATH_TOL_RAD}  # everything after the release: compliant


def _release(ai):
    """Open the gripper, slowly, the hand still (the swing has stopped at its angle). Refused if the gripper's own
    reported position (its joint state, not a camera) still says closed after the command: backing off then would drag
    the door."""
    ai.gripper(0.0, speed=F.GRIP_OPEN_SPEED)
    if float(ai.get_state()["gripper_pos"]) > 0.2:
        raise Refused("the gripper did not open -- the hand is still on the handle.")


def after_release(after, planner, ai, door):
    """What follows the release (`after`: PushOpen, ReleaseAround, ...; execute_open let go), planned again from where
    the arm is now; if that fails, the plan `after` checked before the swing, when the arm is where it starts. Returns
    what it did, in words; raises Refused (the gripper open, the arm holding)."""
    st = ai.get_state()
    planner.search()  # a new search: the swing's time doesn't count against it
    checked = dict(after.__dict__)
    ok, msg = after.plan(planner, door, st["position"], (st["ee_pos"][:3], st["ee_pos"][3:7]), from_arm=True)
    if ok:
        return f"from where the release ended: {msg}"
    first = checked.get("motion")
    if first is not None:
        off = float(np.degrees(np.max(np.abs(pl._short_delta(first.q0, st["position"])))))
        if off <= pl.START_TOL_DEG:
            after.__dict__.update(checked)
            return f"the plan checked before the swing (planned again from here: {msg})"
    raise Refused(f"{msg} -- the hand is off the handle (gripper open), the arm holds where it is")


def push_after_chair_move(ai, planner, execute, chair_dir):
    """The push-open's last part, after the open asked for the wheelchair move and it was made: the push from where the
    hand is (off the handle), against the door file's door shifted by that move; the door file then holds the door
    where it now is. A dry run plans it only."""
    door = load_door()
    move = door.get("push_after_chair_move")
    if not move:
        raise Refused("no push waiting for a wheelchair move: chair_moved is for after an open asked for one.")
    say(f"\n===== push further open, the wheelchair moved {_how(move)} =====")
    st = ai.get_state()
    if float(st["gripper_pos"]) > 0.2:
        raise Refused("gripper is closed -- the push starts with the hand off the handle, as the open left it.")
    moved = shifted(door, move, chair_dir)
    model = FridgeModel(moved)
    deg0 = float(moved["door_open_deg"])
    pose = (np.asarray(st["ee_pos"][:3]), np.asarray(st["ee_pos"][3:7]))
    planner.search()  # a new search: the base's drive (or the goal before) doesn't count against it
    motion, placed, finish = plan_finished_push(planner, moved, st["position"], pose, model.full(deg0), model,
                                                from_arm=True)
    if motion is None:
        raise Refused(
            f"no hand placement behind the door plans from here, with the door {_how(move)} from where it was "
            f"recorded ({placed}) -- check the wheelchair moved as asked."
        )
    say(f"  push planned: {len(motion.goals)} legs, hand {placed[0] * 100:.0f} cm from the hinge; it {finish}")
    if not execute:
        say("  (dry run: the push plans from here; nothing moves)")
        return True
    planner.join(motion, model.cabinet())
    ok = planner.run(motion, "push-open", impedance=AFTER_RELEASE)
    say(f"PUSH-OPEN {'DONE' if ok else 'STOPPED SHORT'}")
    if ok:  # the door where it now is, for the close; no push waiting
        save_door_geometry(
            **{
                k: moved[k]
                for k in ("hinge", "hinge_pivot", "closed_grasp_pos", "closed_handle", "door_mid")
                if moved.get(k) is not None
            },
            door_open_deg=float(F.PUSH_OPEN_TO_DEG),
            push_after_chair_move=None,
        )
    return ok


def planned_door_after(door, push, after):
    """The door file a dry run's save_door writes: the door open PUSH_OPEN_TO_DEG (`push`), shifted by the wheelchair
    move the push-open planned (`after`)."""
    out = opened(door, F.PUSH_OPEN_TO_DEG) if push else dict(door)
    if push and after is not None and after.chair_move:
        out = shifted(out, after.chair_move, after.chair_dir)
    return out

