"""PART 2's geometry (11-06): straight back along the measured door normal, the hand turning TURN_DEG about
the vertical through the tool point with its fingertips away from the door (the way the bar turns as the door opens)."""
import numpy as np
from scipy.spatial.transform import Rotation as R
import rammp_arm
from capability_engine import sim

TURN_DEG = 10.0


def goal(ee, out, pull_m, turn_deg=TURN_DEG):
    """The pull's end pose (pos, quat xyzw, turn sign) from tool pose ee [x y z qx qy qz qw], `out` the horizontal
    outward door normal."""
    ee = np.asarray(ee, float)
    a = R.from_quat(ee[3:7]).as_matrix()[:, 2]
    sgn = float(np.sign(np.cross(a, out)[2]))  # +: turning about +z swings the approach toward the outward normal
    if sgn == 0:
        raise ValueError("the hand's approach is parallel to the door normal -- no turn direction")
    return ee[:3] + out * pull_m, (R.from_euler("z", sgn * turn_deg, degrees=True) * R.from_quat(ee[3:7])).as_quat(), sgn


def knots(sim_pair, ee, q, out, pull_m, turn_deg=TURN_DEG):
    """(knots, cart_lengths, message) for the pull from tool pose ee / joints q; knots None when it doesn't solve."""
    gp, gq, _ = goal(ee, out, pull_m, turn_deg)
    msgs = []
    k, lengths = rammp_arm.cartesian_knots(sim_pair, np.asarray(ee, float), np.asarray(q, float), [(gp, gq)],
                                           log=msgs.append)
    return k, lengths, (msgs[-1].strip() if msgs else "")


def arc_poses(ee, out, toward_hinge, out_m, side_m, step_m=0.02):
    """The pull along the door's swing (11-06): from tool pose ee [x y z qx qy qz qw], the circle that leaves
    straight out along `out` (the measured outward normal) and passes through ee + out*out_m + toward_hinge*side_m; the hand
    turns with it about the vertical, as the door (and its bar) does. Returns (poses [(pos, quat)], turn deg, radius m)."""
    ee = np.asarray(ee, float)
    if side_m <= 0 or out_m <= 0:
        raise ValueError("arc pull: out_m and side_m must be > 0")
    rad = (side_m ** 2 + out_m ** 2) / (2 * side_m)  # centre on the line ee + toward_hinge * rad
    c = ee[:3] + toward_hinge * rad
    r0 = ee[:3] - c
    total = float(np.arctan2(out_m, rad - side_m))
    sgn = float(np.sign(np.cross(r0, out)[2]))
    n = max(2, int(np.ceil(rad * total / step_m)))
    q0 = R.from_quat(ee[3:7])
    poses = []
    for i in range(1, n + 1):
        rot = R.from_euler("z", sgn * total * i / n)
        poses.append((c + rot.apply(r0), (rot * q0).as_quat()))
    return poses, float(np.degrees(total)), rad


def swing_poses(ee, hinge, out, hinge_sign, out_m, spacing_m=0.02):
    """The pull as the fridge's door swing (Cailyn's door_common.common.door_arc_waypoints, as open_door._swing_waypoints
    calls it): about the measured hinge pin, the hand turning with the door, from tool pose ee [x y z qx qy qz qw], swung
    until the tool is out_m out along the outward normal `out`. direction = -hinge_sign (open_door.SWING_DIRECTION = -1
    for this fridge, hinged on the handle's left). Returns (poses [(pos, quat)], swing deg, radius m)."""
    from door_common.common import door_arc_waypoints
    ee = np.asarray(ee, float); hinge = np.asarray(hinge, float)
    direction = -int(hinge_sign)
    r0 = ee[:2] - hinge[:2]
    rad = float(np.linalg.norm(r0))
    deg = None
    for d in np.arange(0.5, 150.0, 0.25):
        a = np.radians(direction * d)
        p = hinge[:2] + np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ r0
        if float(np.dot(p - ee[:2], out[:2])) >= out_m:
            deg = float(d)
            break
    if deg is None:
        raise ValueError(f"the door's swing (radius {rad:.3f} m) never takes the tool {out_m:.2f} m out")
    wps = door_arc_waypoints(sim.Pose(tuple(ee[:3]), tuple(ee[3:7])), tuple(hinge), rad * np.radians(deg), spacing_m,
                             direction=direction, rotate_orientation=True)
    return [(np.asarray(w.position, float), np.asarray(w.orientation, float)) for w in wps], deg, rad


def swing_by_deg(ee, hinge, hinge_sign, deg, spacing_m=0.02):
    """open_door._swing_waypoints with the side grasp's numbers: Cailyn's door_arc_waypoints about the hinge pin, `deg`
    of swing (arc length = radius x angle), direction -hinge_sign (SWING_DIRECTION -1 for this left-hinged fridge), the
    hand turning with the door. Returns (poses [(pos, quat)], radius m)."""
    from door_common.common import door_arc_waypoints
    ee = np.asarray(ee, float); hinge = np.asarray(hinge, float)
    rad = float(np.linalg.norm(ee[:2] - hinge[:2]))
    wps = door_arc_waypoints(sim.Pose(tuple(ee[:3]), tuple(ee[3:7])), tuple(hinge), rad * np.radians(deg), spacing_m,
                             direction=-int(hinge_sign), rotate_orientation=True)
    return [(np.asarray(w.position, float), np.asarray(w.orientation, float)) for w in wps], rad


# --- the swing's joints with the arm's spare joint (11-06): plain step-by-step IK (rammp_arm.cartesian_knots) follows
# whatever elbow the grasp has, and turning the hand with the door then folds J6 into its 115 deg guard (~51 deg of swing
# at best from 23 grasp configurations). Resolved-rate IK instead: each waypoint solved exactly (position + orientation),
# the null space of the hand's task used to keep J2/J4/J6 away from their guards.
NS_GUARDS_DEG = np.array([np.inf, 125.0, np.inf, 144.0, np.inf, 115.0, np.inf])  # J2 (limit 128.9), J4, J6 (sim guards)
NS_WEIGHTS = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 4.0, 0.0])  # J6 the one that runs out
NS_ACTIVE_DEG = np.array([np.inf, 110.0, np.inf, 120.0, np.inf, 60.0, np.inf])  # the push starts past these
NS_STEP_DEG = 3.0  # the null-space relief per waypoint, at most (the elbow drifts, never jumps)
NS_ITERS = 60
NS_TOL_M, NS_TOL_RAD = 2e-4, 2e-3
NS_DAMP = 0.02
NS_MAX_STEP_DEG = 8.0  # largest joint change between consecutive waypoints (2 cm apart)


def _movable(rb):
    import pybullet as pb
    c = rb.physics_client_id
    return [j for j in range(pb.getNumJoints(rb.robot_id, physicsClientId=c))
            if pb.getJointInfo(rb.robot_id, j, physicsClientId=c)[2] != pb.JOINT_FIXED]


def _jacobian(scene, rb, q, movable):
    """6x7 geometric Jacobian of the tool point in the ARM's base frame (the mount is tilted: rotate pybullet's world J)."""
    import pybullet as pb
    c = rb.physics_client_id
    sim.clearance(rb, q, [])  # sets the arm, gripper open
    pos = [pb.getJointState(rb.robot_id, j, physicsClientId=c)[0] for j in movable]
    jl, ja = pb.calculateJacobian(rb.robot_id, rb.end_effector_id, [0.0, 0.0, 0.0], pos, [0.0] * len(pos),
                                  [0.0] * len(pos), physicsClientId=c)
    cols = [movable.index(j) for j in sim.ARM]
    rb_inv = R.from_quat(scene.robot_base_pose.orientation).inv().as_matrix()
    return np.vstack([rb_inv @ np.asarray(jl)[:, cols], rb_inv @ np.asarray(ja)[:, cols]])


def swing_knots_ns(scene, rb, q_start, poses):
    """Joint knots through poses [(pos, quat) in the arm's base frame] from q_start, the spare joint keeping J2/J4/J6 off
    their guards. Returns (knots, worst {J6, J4} deg, message); knots None with the reason when a pose can't be held."""
    movable = _movable(rb)
    q = np.asarray(q_start, float).copy()
    out, worst6, worst4 = [], 0.0, 0.0
    for k, (pt, qt) in enumerate(poses):
        q_prev = q.copy()
        rt = R.from_quat(qt)
        # one bounded null-space step (the hand doesn't move), only for joints past their active threshold ...
        act = np.radians(NS_ACTIVE_DEG)
        over = np.where(np.isfinite(act), np.sign(q) * np.maximum(np.abs(q) - np.where(np.isfinite(act), act, 0.0), 0.0), 0.0)
        if np.any(over != 0):
            J = _jacobian(scene, rb, q, movable)
            Jp = J.T @ np.linalg.inv(J @ J.T + NS_DAMP ** 2 * np.eye(6))
            ns = (np.eye(7) - Jp @ J) @ (-NS_WEIGHTS * over)
            m = float(np.max(np.abs(ns)))
            if m > 1e-9:
                q = q + ns * min(1.0, np.radians(NS_STEP_DEG) / m)
        # ... then the hand onto this waypoint (damped least squares: the smallest joint change)
        for _ in range(NS_ITERS):
            pos, quat = sim.tool_pose(scene, rb, q)
            e = np.r_[np.asarray(pt) - pos, (rt * R.from_quat(quat).inv()).as_rotvec()]
            J = _jacobian(scene, rb, q, movable)
            Jp = J.T @ np.linalg.inv(J @ J.T + NS_DAMP ** 2 * np.eye(6))
            q = q + np.clip(Jp @ e, -0.1, 0.1)
            if np.linalg.norm(e[:3]) < NS_TOL_M and np.linalg.norm(e[3:]) < NS_TOL_RAD:
                break
        pos, quat = sim.tool_pose(scene, rb, q)
        err_p = float(np.linalg.norm(np.asarray(pt) - pos))
        err_r = float(np.degrees((rt * R.from_quat(quat).inv()).magnitude()))
        jump = float(np.degrees(np.max(np.abs(q - q_prev))))
        j4, j6 = float(np.degrees(abs(q[3]))), float(np.degrees(abs(q[5])))
        worst4, worst6 = max(worst4, j4), max(worst6, j6)
        why = (f"pose {k + 1}/{len(poses)}: " + ("IK error %.1f mm / %.1f deg" % (err_p * 1000, err_r) if err_p > 1e-3 or err_r > 0.5
               else "a joint jumps %.1f deg" % jump if jump > NS_MAX_STEP_DEG
               else "J6 %.1f deg (guard 115)" % j6 if j6 >= NS_GUARDS_DEG[5]
               else "J4 %.1f deg (guard 144)" % j4 if j4 >= NS_GUARDS_DEG[3] else ""))
        if not why.endswith(": "):
            return None, {"J6": worst6, "J4": worst4}, why
        out.append(q.copy())
    return out, {"J6": worst6, "J4": worst4}, ""


def swing_hand_turn(ee, hinge, hinge_sign, deg, hand_deg, spacing_m=0.02):
    """Cailyn's arc for the hand's PATH (door_arc_waypoints about the hinge pin, `deg` of door swing), the hand turning
    only `hand_deg` over it (evenly) instead of the door's full `deg`: under impedance the bar may turn a few degrees in
    the open gripper (11-06), and the wrist (J6) keeps room. Returns (poses [(pos, quat)], radius m)."""
    from door_common.common import door_arc_waypoints
    ee = np.asarray(ee, float); hinge = np.asarray(hinge, float)
    rad = float(np.linalg.norm(ee[:2] - hinge[:2]))
    direction = -int(hinge_sign)
    wps = door_arc_waypoints(sim.Pose(tuple(ee[:3]), tuple(ee[3:7])), tuple(hinge), rad * np.radians(deg), spacing_m,
                             direction=direction, rotate_orientation=False)
    q0, n = R.from_quat(ee[3:7]), len(wps)
    return [(np.asarray(w.position, float), (R.from_euler("z", direction * np.radians(hand_deg) * (i + 1) / n) * q0).as_quat())
            for i, w in enumerate(wps)], rad


def self_clearance(rb, q, max_m=0.10):
    """The smallest distance between the arm's own links at joints q (gripper open) -- every pair at least two apart
    along the chain (base, the 7 arm links), and the hand's links (after the last arm joint) against the arm links up to
    two before it. (distance m, 'link vs link'); 9.0 when nothing is within max_m. 11-06: a route round the door's free
    edge folded the arm into itself (the door model clearance never sees that)."""
    import pybullet as pb
    c, rid = rb.physics_client_id, rb.robot_id
    sim.clearance(rb, q, [])  # sets the arm, gripper open
    arm = list(sim.ARM)
    chain = [-1] + arm
    hand = list(range(arm[-1] + 1, pb.getNumJoints(rid, physicsClientId=c)))
    name = lambda i: "base" if i < 0 else pb.getJointInfo(rid, i, physicsClientId=c)[12].decode()  # noqa: E731
    pairs = [(a, b) for i, a in enumerate(chain) for b in chain[i + 2:]] + [(a, h) for a in chain[:-2] for h in hand]
    best = (9.0, "")
    for a, b in pairs:
        for pt in pb.getClosestPoints(rid, rid, max_m, linkIndexA=a, linkIndexB=b, physicsClientId=c):
            if pt[8] < best[0]:
                best = (float(pt[8]), f"{name(a)} vs {name(b)}")
    return best
