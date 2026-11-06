"""The push-open, the open's default: after a PUSH_OPEN_PULL_DEG pull, let go, back off the handle, have the wheelchair
moved if the push needs it, and push the door on.

1. After the pull (still holding the handle): let go and back straight off the handle BACK_OFF_M (release_and_back_off).
2. The smallest wheelchair move of CHAIR_MOVES_M (+ back, - forward; 0 first) after which the push plans from there,
   the hand clear of the door all through the move. 0: straight on to 3. Else the goal ends there, asking for the move
   (NEXT_STEP); an open goal with chair_moved then does 3 against the door shifted by it (push_after_chair_move). An
   action can't prompt, as for the close.
3. The push: along the open door to past its free edge, round it -- turning the hand PARALLEL to the door (fingers along
   it toward the hinge, rolled vertical) and dropping a little -- in behind the door, then the side of the hand onto the
   inner face and the push, held in the door's frame, round to PUSH_OPEN_TO_DEG. A few hand placements are tried in
   order (PUSH_OPEN_AT_M x PUSH_OPEN_APPROACH_DEG x tool y up/down); the first whose whole route passes every gate.

Only the push may touch the door model; it is checked against the cabinet. All of it is planned with the grasp, before
any motion; again from where the pull really ended, before letting go; and the push once more from where the hand
really is. It runs at the contact speed.
"""

import json

import numpy as np
import rammp_arm
from capability_engine import sim
from door_common import common as dc
from door_common import demo_replay
from door_common import open_door as shared
from door_common.common import (
    DoorFrame,
    Refused,
    clearance,
    drop,
    make_sim,
    plan_straight_line,
    quiet,
    save_door_geometry,
    say,
    solve_ik,
)
from scipy.spatial.transform import Rotation as R

from . import constants as F
from .close_door import move_chair
from .motion import AfterSwing, door_after, send_legs, trajectory_keep_out

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


def _touch_v(scene, rb, df, deg, u, z, quat, v_far, q_seed):
    """Door-frame v (off the mid-plane) where the hand at (u, z, quat) just touches the door model at `deg`, found by
    bisection from `v_far` (clear) toward the inner face; None if no IK there."""
    bodies = df.bodies(scene, rb, deg)
    try:

        def clear(v):
            q, err = solve_ik(scene, rb, df.pos(u, v, deg, z), quat, q_seed)
            for _ in range(3):
                if err <= sim.MAX_IK_ERR:
                    break
                q, err = solve_ik(scene, rb, df.pos(u, v, deg, z), quat, q)
            return None if err > sim.MAX_IK_ERR else clearance(rb, q, bodies)[0]

        c = clear(v_far)
        if c is None or c <= 0.0:
            return None
        lo, hi = v_far, -df.t / 2  # lo: clear, hi: the inner face itself
        for _ in range(12):
            mid = (lo + hi) / 2
            cm = clear(mid)
            if cm is None:
                return None
            lo, hi = (mid, hi) if cm > 0.0 else (lo, mid)
        return hi
    finally:
        drop(rb, bodies)


def push_open_points(scene, rb, door, st, u_push, approach_deg, y_up, off_handle=True):
    """Keypoints [(pos, quat, label)] from the hand (`st`: on the handle at the end of the pull, or with `off_handle`
    False already backed off it) to the end of the push for one hand placement, and the index of the first push keypoint
    (legs from there are checked against the cabinet only: they touch the door on purpose). (None, None) if the hand
    can't be placed against the inner face."""
    df = DoorFrame(door, door["open_sign"])
    deg0 = float(door["door_open_deg"])
    ee = np.asarray(st["ee_pos"], float)
    z, q_hold = float(ee[2]), ee[3:7]
    d = df._rot(deg0).T @ (ee[:2] - df.hinge[:2])  # the hand in the open door's frame (u along, v off it)
    u_hold, v_out = float(d @ df.e), float(d @ df.n) - df.mid + (F.PUSH_OPEN_OFF_HANDLE_M if off_handle else 0.0)
    u_edge = df.length + F.PUSH_OPEN_EDGE_CLEAR_M
    v_in, z_in = -(df.t / 2 + F.PUSH_OPEN_GAP_M), z + F.PUSH_OPEN_DZ_M
    q_in = hand_quat(df, deg0, approach_deg, y_up)
    v_touch = _touch_v(scene, rb, df, deg0, u_push, z_in, q_in, v_in, np.asarray(st["position"], float))
    if v_touch is None:
        return None, None
    v_push = v_touch + F.PUSH_OPEN_PRELOAD_M  # past touching, into the inner face
    pts = [(ee[:3], q_hold, "end of the pull, holding" if off_handle else "off the handle")]
    if off_handle:
        pts.append((df.pos(u_hold, v_out, deg0, z), q_hold, "off the handle"))
    pts += [
        (df.pos(u_edge, v_out, deg0, z), q_hold, "past the free edge"),
        (df.pos(u_edge, v_in, deg0, z_in), q_in, "round the edge, turned"),
        (df.pos(u_push, v_in, deg0, z_in), q_in, "in behind the door"),
    ]
    first_push = len(pts)
    pts.append((df.pos(u_push, v_push, deg0, z_in), q_in, "onto the inner face"))
    n = max(1, int(np.ceil((F.PUSH_OPEN_TO_DEG - deg0) / F.PUSH_OPEN_ARC_STEP_DEG)))
    for k in range(1, n + 1):
        deg = deg0 + (F.PUSH_OPEN_TO_DEG - deg0) * k / n
        quat = hand_quat(df, deg, approach_deg, y_up)
        pts.append((df.pos(u_push, v_push, deg, z_in), quat, f"push to {deg:.0f} deg"))
    return pts, first_push


def plan_push_open(scene, rb, door, st, off_handle=True):
    """Try the hand placements in order; the first whose whole route passes every gate (demo_replay.plan, from `st`; on
    the handle, or `off_handle` False backed off it). Returns (keypoints, joint legs, placement) or (None, None,
    None)."""
    tried = 0
    for u_push in F.PUSH_OPEN_AT_M:
        for approach_deg in F.PUSH_OPEN_APPROACH_DEG:
            for y_up in (1.0, -1.0):
                tried += 1
                with quiet():
                    pts, first_push = push_open_points(scene, rb, door, st, u_push, approach_deg, y_up, off_handle)
                    legs = (
                        None
                        if pts is None
                        else demo_replay.plan(
                            scene, rb, door, st, [(pp, qq) for pp, qq, _ in pts[1:]], first_push, lift_off=off_handle
                        )
                    )
                if legs is not None and trajectory_keep_out(
                    st["position"], [(pp, qq) for pp, qq, _ in pts], legs, rammp_arm.contact_speed_scale()
                ):
                    legs = None  # the arm would refuse to send it
                if legs is not None:
                    return pts, legs, (u_push, approach_deg, y_up, tried)
    return None, None, None


def shifted(door, chair_move, chair_dir):
    """A copy of `door` as it stands in the arm's frame once the wheelchair has moved `chair_move` (m, + back): its
    recorded positions moved by `chair_dir` x that (the node's chair_back_dir)."""
    dxy = np.asarray(chair_dir, float) * chair_move
    out = dict(door)
    for k in ("hinge", "hinge_pivot", "closed_grasp_pos", "closed_handle", "door_mid"):
        if out.get(k) is not None:
            out[k] = [out[k][0] + dxy[0], out[k][1] + dxy[1], *out[k][2:]]
    return out


def back_off(scene, rb, ee, q):
    """The release's straight back-off off the handle from tool pose `ee` / joints `q`, planned as release_and_back_off
    plans it: the arm state where it ends ({position, ee_pos}), or None if it fails a gate."""
    ee = np.asarray(ee, float)
    approach = R.from_quat(ee[3:7]).as_matrix()[:, 2]
    with quiet():
        back = plan_straight_line(scene, rb, ee[:3], ee[3:7], -approach, F.BACK_OFF_M, q, "back-off")
    if back is None:
        return None
    return {"position": list(back[-1]), "ee_pos": list(np.r_[ee[:3] - approach * F.BACK_OFF_M, ee[3:7]])}


def _clear_through_move(scene, rb, door, q, chair_move, chair_dir):
    """The arm at joints `q` (it moves with the wheelchair) stays CHAIR_MOVE_MIN_CLEAR_M off the door model all through
    the wheelchair's move."""
    for s in np.linspace(0.0, chair_move, F.CHAIR_MOVE_CHECK_STEPS):
        d = shifted(door, float(s), chair_dir)
        bodies = DoorFrame(d, d["open_sign"]).bodies(scene, rb, float(d["door_open_deg"]))
        try:
            if clearance(rb, np.asarray(q, float), bodies)[0] < F.CHAIR_MOVE_MIN_CLEAR_M:
                return False
        finally:
            drop(rb, bodies)
    return True


def plan_after_chair_move(scene, rb, door, st_off, chair_dir, moves=None):
    """The first wheelchair move of `moves` (default CHAIR_MOVES_M: smallest first) after which the push plans from the
    hand off the handle (`st_off`), the hand clear of the door all through the move. Returns (move, keypoints, legs,
    placement) or None."""
    for move in F.CHAIR_MOVES_M if moves is None else moves:
        if shared.CHECKPOINT is not None:
            shared.CHECKPOINT()
        if move and not _clear_through_move(scene, rb, door, st_off["position"], move, chair_dir):
            continue
        pts, legs, placed = plan_push_open(scene, rb, shifted(door, move, chair_dir), st_off, off_handle=False)
        if pts is not None:
            return move, pts, legs, placed
    return None


def _after(move):
    """How the plan goes on after the back-off, in words."""
    return "no wheelchair move needed" if not move else f"after the wheelchair moves {_how(move)}"


def _how(move):
    return f"~{abs(move) * 100:.0f} cm {'back' if move > 0 else 'forward'}"


class PushOpen(AfterSwing):
    """Plans the release, back-off, wheelchair move and push with the swing (`chair_move`: the move the chosen plan
    needs); `push_open` runs them, re-planned from where the swing ends."""

    def __init__(self, chair_dir=F.CHAIR_BACK_DIR):
        self.chair_dir, self.chair_move = chair_dir, None

    def check(self, scene, rb, geom, cand, explain):
        off = back_off(scene, rb, cand["wps"][-1], cand["arc"][-1])
        if off is None:
            return False, f"push-open: the {F.BACK_OFF_M * 100:.0f} cm back-off off the handle fails a gate"
        found = plan_after_chair_move(scene, rb, door_after(geom, cand), off, self.chair_dir)
        if found is None:
            moves = ", ".join(f"{m * 100:+.0f}" for m in F.CHAIR_MOVES_M)
            return (
                False,
                f"push-open: no hand placement behind the door passes every gate (wheelchair moves {moves} cm)",
            )
        self.chair_move = found[0]
        return True, f"push-open OK, {_after(found[0])}"


def _push(ai, pts, legs, placed, deg0):
    """Send the planned push (the hand already off the handle); the door file then has the door at the push's end."""
    u_push, approach_deg, y_up, tried = placed
    say(
        f"door ~{deg0:.0f} deg; placement {tried}: hand {u_push * 100:.0f} cm from the hinge, fingers "
        f"{approach_deg:+.0f} deg off along-the-door, tool y {'up' if y_up > 0 else 'down'}; push to "
        f"{F.PUSH_OPEN_TO_DEG} deg"
    )
    for pp, _, label in pts:
        say(f"  {label[:30]:30s} -> {np.round(pp, 3)}")
    ok, _ = send_legs(ai, [(pp, qq) for pp, qq, _ in pts], legs, speed_scale=rammp_arm.contact_speed_scale())
    say(f"PUSH-OPEN {'DONE' if ok else 'STOPPED SHORT'}")
    if ok:
        save_door_geometry(door_open_deg=float(F.PUSH_OPEN_TO_DEG))  # the close's default
    return ok


def push_open(ai, execute, after, drive_chair=None):
    """After the pull: let go, back off the handle, and push the door on -- or, if the push needs the wheelchair moved
    (`after`, the PushOpen that planned it), stop there and ask for the move (NEXT_STEP); with `drive_chair` (the
    goal's drive_base: base_drive.move_chair) drive the base that far instead and push from there. Planned
    and sim-gated from where the arm really is before letting go (a failed gate: nothing moves, the hand stays on the
    handle)."""
    global NEXT_STEP
    say("\n===== let go, back off the handle, push further open =====")
    if not execute:
        move = after.chair_move or 0.0
        if not move:
            NEXT_STEP = (
                "Plan OK: after the pull it lets go, backs off the handle and pushes the door open (no wheelchair "
                "move)."
            )
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
        return True
    door = json.loads(dc.DOOR_FILE.read_text())
    st = ai.get_state()
    if float(st["gripper_pos"]) < 0.2:
        raise Refused("gripper is open -- the push-open starts from the end of the pull, still holding the handle.")
    deg0 = float(door["door_open_deg"])
    if deg0 > F.PUSH_OPEN_TO_DEG - F.PUSH_OPEN_MIN_PUSH_DEG + 2:  # the door as reached: a pull may overshoot a little
        raise Refused(f"the door is already ~{deg0:.0f} deg open (push to {F.PUSH_OPEN_TO_DEG}).")
    scene, rb = make_sim()
    off = back_off(scene, rb, st["ee_pos"], st["position"])
    found = None if off is None else plan_after_chair_move(scene, rb, door, off, after.chair_dir)
    if found is None:
        raise Refused(
            "from where the pull ended, no back-off + wheelchair move + push passes every gate -- NOT releasing, the "
            "hand stays on the handle. A release_only goal lets go and backs off."
        )
    say(f"push-open from here: {_after(found[0])} (chosen again once the hand is off the handle)")
    if not dc.release_and_back_off(ai, F.BACK_OFF_M, True, park=False):
        return False
    # the move asked for is planned from where the hand really is now: the arm doesn't move again until the push, so
    # the push after the move plans from exactly here (from the predicted back-off, a plan near a gate can flip)
    found = plan_after_chair_move(scene, rb, door, ai.get_state(), after.chair_dir)
    if found is None:
        raise Refused(
            f"from where the hand ended up off the handle, no wheelchair move lets the push plan -- the door is "
            f"~{deg0:.0f} deg open, the hand off it."
        )
    move = found[0]
    if move and drive_chair is not None:
        # the hand stays where the back-off left it, clear of the door all through the move (plan_after_chair_move
        # checked it); the push is then planned against the door shifted by the move the base really made
        say(f"\n>>> drive_base: driving the wheelchair {_how(move)} for the push")
        moved = drive_chair(move)  # within the base_tolerance of `move`, or it raises
        # only now: a drive stopped part way leaves no push waiting, so a chair_moved goal can't push a door that
        # isn't where it would think
        save_door_geometry(push_after_chair_move=float(moved))
        return push_after_chair_move(ai, True, after.chair_dir)
    if move:
        save_door_geometry(push_after_chair_move=float(move))
        NEXT_STEP = (
            f"Pull done and the hand is off the handle. {move_chair(move)}, then send the open with execute true and "
            "chair_moved true to push the door open."
        )
        say(f"\n>>> {NEXT_STEP}")
        return True
    return _push(ai, *found[1:], deg0)


def push_after_chair_move(ai, execute, chair_dir):
    """The push-open's last part, after the open asked for the wheelchair move and it was made: the push from where the
    hand is (off the handle), against the door file's door shifted by that move; the door file then holds the door
    where it now is. A dry run plans it only."""
    if not dc.DOOR_FILE.exists():
        raise Refused(f"need {dc.DOOR_FILE}: the open writes it.")
    door = json.loads(dc.DOOR_FILE.read_text())
    move = door.get("push_after_chair_move")
    if not move:
        raise Refused("no push waiting for a wheelchair move: chair_moved is for after an open asked for one.")
    say(f"\n===== push further open, the wheelchair moved {_how(move)} =====")
    st = ai.get_state()
    if float(st["gripper_pos"]) > 0.2:
        raise Refused("gripper is closed -- the push starts with the hand off the handle, as the open left it.")
    scene, rb = make_sim()
    moved = shifted(door, move, chair_dir)
    found = plan_after_chair_move(scene, rb, moved, st, chair_dir, moves=(0.0,))
    if found is None:
        raise Refused(
            f"no hand placement behind the door passes every gate from here, with the door {_how(move)} from where it "
            "was recorded -- check the wheelchair moved as asked."
        )
    if not execute:
        say("  (dry run: the push plans from here; nothing moves)")
        return True
    ok = _push(ai, *found[1:], float(door["door_open_deg"]))
    if ok:  # the door where it now is, for the close; no push waiting
        save_door_geometry(
            **{
                k: moved[k]
                for k in ("hinge", "hinge_pivot", "closed_grasp_pos", "closed_handle", "door_mid")
                if moved.get(k) is not None
            },
            push_after_chair_move=None,
        )
    return ok
