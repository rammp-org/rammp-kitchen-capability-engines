"""The push-open option: after a PUSH_OPEN_PULL_DEG pull (still holding the handle), let go and push the door on.

The route is computed for today's door, as one smooth sim-gated trajectory: straight out off the handle, along the open
door to past its free edge, round it -- turning the hand PARALLEL to the door (fingers along it toward the hinge, rolled
vertical) and dropping a little -- in behind the door, then the side of the hand onto the inner face and the push, held
in the door's frame, round to PUSH_OPEN_TO_DEG. A few hand placements are tried in order (PUSH_OPEN_AT_M x
PUSH_OPEN_APPROACH_DEG x tool y up/down); the first whose whole route passes every gate is used.

Only the push may touch the door model; it is checked against the cabinet. Planned with the grasp, from where the pull
would end; again from the real grasp; and once more from where the pull really ended, before letting go. It runs at the
contact speed.
"""

import json
import time

import numpy as np
import rammp_arm
from capability_engine import sim
from door_common import common as dc
from door_common import demo_replay
from door_common.common import DoorFrame, Refused, clearance, drop, make_sim, quiet, save_door_geometry, say, solve_ik
from scipy.spatial.transform import Rotation as R

from . import constants as F
from .motion import AfterSwing, door_after, send_legs


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


def push_open_points(scene, rb, door, st, u_push, approach_deg, y_up):
    """Keypoints [(pos, quat, label)] from the hand on the handle (`st`, the end of the pull) to the end of the push for
    one hand placement, and the index of the first push keypoint (legs from there are checked against the cabinet only:
    they touch the door on purpose). (None, None) if the hand can't be placed against the inner face."""
    df = DoorFrame(door, door["open_sign"])
    deg0 = float(door["door_open_deg"])
    ee = np.asarray(st["ee_pos"], float)
    z, q_hold = float(ee[2]), ee[3:7]
    d = df._rot(deg0).T @ (ee[:2] - df.hinge[:2])  # the hand in the open door's frame (u along, v off it)
    u_hold, v_out = float(d @ df.e), float(d @ df.n) - df.mid + F.PUSH_OPEN_OFF_HANDLE_M
    u_edge = df.length + F.PUSH_OPEN_EDGE_CLEAR_M
    v_in, z_in = -(df.t / 2 + F.PUSH_OPEN_GAP_M), z + F.PUSH_OPEN_DZ_M
    q_in = hand_quat(df, deg0, approach_deg, y_up)
    v_touch = _touch_v(scene, rb, df, deg0, u_push, z_in, q_in, v_in, np.asarray(st["position"], float))
    if v_touch is None:
        return None, None
    v_push = v_touch + F.PUSH_OPEN_PRELOAD_M  # past touching, into the inner face
    pts = [
        (ee[:3], q_hold, "end of the pull, holding"),
        (df.pos(u_hold, v_out, deg0, z), q_hold, "off the handle"),
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


def plan_push_open(scene, rb, door, st):
    """Try the hand placements in order; the first whose whole route passes every gate (demo_replay.plan, from `st`).
    Returns (keypoints, joint legs, placement) or (None, None, None)."""
    tried = 0
    for u_push in F.PUSH_OPEN_AT_M:
        for approach_deg in F.PUSH_OPEN_APPROACH_DEG:
            for y_up in (1.0, -1.0):
                tried += 1
                with quiet():
                    pts, first_push = push_open_points(scene, rb, door, st, u_push, approach_deg, y_up)
                    legs = (
                        None
                        if pts is None
                        else demo_replay.plan(
                            scene, rb, door, st, [(pp, qq) for pp, qq, _ in pts[1:]], first_push, lift_off=True
                        )
                    )
                if legs is not None:
                    return pts, legs, (u_push, approach_deg, y_up, tried)
    return None, None, None


class PushOpen(AfterSwing):
    """Plans the push-open with the swing; `push_open` runs it, re-planned from where the swing ends."""

    def check(self, scene, rb, geom, cand, explain):
        st = {"position": list(cand["arc"][-1]), "ee_pos": list(cand["wps"][-1])}
        ok = plan_push_open(scene, rb, door_after(geom, cand), st)[0] is not None
        return ok, ("push-open OK" if ok else "push-open: no hand placement behind the door passes every gate")


def push_open(ai, execute):
    """After the pull: let go and push the door further open. Planned and sim-gated from where the arm really is, before
    anything moves (a failed gate: nothing moves, the hand stays on the handle)."""
    say("\n===== release + push further open =====")
    if not execute:
        say("  (planned above from the planned swing end -- see the chosen tilt's line; re-planned from the real one)")
        return True
    door = json.loads(dc.DOOR_FILE.read_text())
    st = ai.get_state()
    if float(st["gripper_pos"]) < 0.2:
        raise Refused("gripper is open -- the push-open starts from the end of the pull, still holding the handle.")
    deg0 = float(door["door_open_deg"])
    if deg0 >= F.PUSH_OPEN_TO_DEG - 5:
        raise Refused(f"the door is already ~{deg0:.0f} deg open (push to {F.PUSH_OPEN_TO_DEG}).")
    scene, rb = make_sim()
    pts, legs, placed = plan_push_open(scene, rb, door, st)
    if pts is None:
        n = len(F.PUSH_OPEN_AT_M) * len(F.PUSH_OPEN_APPROACH_DEG) * 2
        raise Refused(f"no hand placement of {n} gives a push-open route that passes every gate.")
    u_push, approach_deg, y_up, tried = placed
    say(
        f"door ~{deg0:.0f} deg; placement {tried}: hand {u_push * 100:.0f} cm from the hinge, fingers "
        f"{approach_deg:+.0f} deg off along-the-door, tool y {'up' if y_up > 0 else 'down'}; push to "
        f"{F.PUSH_OPEN_TO_DEG} deg"
    )
    for pp, _, label in pts:
        say(f"  {label[:30]:30s} -> {np.round(pp, 3)}")
    say("opening the gripper ...")
    ai.gripper(0.0)
    time.sleep(0.5)
    ok, _ = send_legs(ai, [(pp, qq) for pp, qq, _ in pts], legs, speed_scale=rammp_arm.contact_speed_scale())
    say(f"PUSH-OPEN {'DONE' if ok else 'STOPPED SHORT'}")
    if ok:
        save_door_geometry(door_open_deg=float(F.PUSH_OPEN_TO_DEG))  # the close's default
    return ok
