"""Grasping the fridge handle from the side: line up beside the handle on its hinge side, then go straight in along the
door, the fingers closing across the bar from its front and back.

The grasp's depth is the front grasp's: the finger pads close on the bar where the front grasp closes them
(BAR_FROM_DETECTED_M in front of the detected handle point, the point the camera gives being behind the bar's real front;
the front grasp's GRIP_EXT was found by trial). Seen from above, the hand is the front grasp's turned 90 degrees about
the vertical: the approach (tool z) runs along the door face toward the free edge and the fingers close along the door
normal, one finger between the bar and the door.

Planned in the node's sim with NO gates (11-06, for testing): IK for the waypoint and every 2 cm of the straight-in;
the numbers a gate would judge are reported only.

plan_side_curve (11-06): the same grasp reached by ONE smooth route from the start pose instead of turn waypoint ->
forward -> corner -> in. The fingertips follow a curve that ends tangent to a straight slide along the door at the
grasp's depth; the hand turns onto the curve's tangent on the way, so near the end the fingertips lead and the hand
trails (a car pulling into a parking space), and nothing of the gripper gets deeper than it is in the slide.
"""

import numpy as np
import pybullet as pb
import rammp_arm
from door_common import common as dc
from door_common import open_door as shared
from door_common.common import Refused, drop, say
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

from capability_engine import sim

from . import constants as F

SIDE_STANDOFF_M = 0.12  # waypoint 1: this far along the door (hinge side) from where the pads centre on the bar
# (8, then 10 cm until 11-06 evening, then 12 cm on the arm)
SIDE_REACH_IN_M = 0.17  # waypoint 1 -> waypoint 3: straight in along the door this far (11-06: 16, then 17 cm, i.e. 5 cm
# past where the pads centre on the bar)
SIDE_CORNER_RADIUS_M = 0.03  # the route rounds the corner at waypoint 1 (forward -> in along the door) with this radius,
# so the hand doesn't turn 90 deg at speed (11-06: one smooth route, no stops)
SIDE_TURN_BACK_M = 0.15  # the turn waypoint: waypoint 1 pulled this far straight back from the door (toward the arm), the
# hand already turned sideways -- cuRobo turns the hand there, clear of the fridge (11-06: the fingertip hit the fridge
# when cuRobo went straight to waypoint 1), then the hand goes straight forward to waypoint 1
SIDE_DEEPER_M = 0.055  # 11-06 on the arm: at the front grasp's depth the fingers met the bar's front surface. Tried 4 and
# 3 cm; with the scene camera's handle point, 5 cm (the pads 0.6 cm past the detected point, toward the door)
# the pads' centre on the bar, in front of the detected handle point along the door normal: where the front grasp
# closes them at no tilt (tool point GRIP_EXT - GRASP_X_ADJUST in front of the handle, pads GRASP_PADS_PAST_TOOL past it)
SIDE_INSERT_DEG = 70.0  # 11-06: the hand goes in at this angle from the door normal -- 90 slides along
# the door; 70 comes in 20 deg off the door's surface (from in front of it), the hand turned with its travel, so the
# fingers move along their own axis and every point before the grasp is shallower than the grasp
BAR_FROM_DETECTED_M = F.GRIP_EXT - F.GRASP_X_ADJUST - shared.GRASP_PADS_PAST_TOOL
# the bar's back edge to the door's face, measured 11-06 (TESTING.md): what the finger between them has to fit in
BAR_BACK_TO_FACE_M = 2.5 * 0.0254
BAR_DEPTH_M = 0.5 * 0.0254  # the bar, front to back (~0.5 in, 11-06)
FINGER_OPEN_HALF_M = 0.0425  # the 2F-85 open: each finger's inner face this far from the closing axis


def side_frame(n, hinge_sign=1):
    """(into the door, along the door toward the hinge side), both horizontal unit vectors, for outward normal `n`.
    `hinge_sign` +1: the hinge on the handle's left as seen from the arm (door square to the arm, n = -x: into = +x,
    toward the hinge = +y, the arm's left); -1: hinged on the right (toward the hinge = -y)."""
    if hinge_sign not in (1, -1):
        raise ValueError(f"hinge_sign must be +1 (hinge on the handle's left) or -1 (right), not {hinge_sign!r}")
    out = np.array([n[0], n[1], 0.0])
    out /= np.linalg.norm(out)
    return -out, hinge_sign * np.cross(out, [0.0, 0.0, 1.0])


def hinge_sign_of(h, n, hinge):
    """+1 when the hinge point `hinge` is on the handle `h`'s left (side_frame's default), -1 on its right."""
    _, left = side_frame(n)
    return 1 if float(np.dot(np.asarray(hinge, float) - np.asarray(h, float), left)) >= 0 else -1


def front_hand(orient, n):
    """The level front grasp's hand (open_door.grasp_pose at no tilt, without its reach and height gates, which are the
    front grasp's): the detector's handle orientation flipped to approach, yawed to face the door square."""
    quat = R.from_quat(orient) * R.from_euler("y", np.pi)
    z0, into = quat.as_matrix()[:, 2], -np.asarray(n, float)
    yaw = float(np.arctan2(z0[0] * into[1] - z0[1] * into[0], z0[0] * into[0] + z0[1] * into[1]))
    return R.from_euler("z", yaw) * quat


def hinge_from_handle(h, n, measured, hinge_sign=1):
    """The hinge pin as the fridge measures it from the handle (hinge_measured with `measured`, constants.MEASURED's
    names): handle_centre_to_face_m + pin_behind_face_m into the door, hinge_along_door_m toward the hinge side
    (`hinge_sign`: side_frame's)."""
    into, hinge_side = side_frame(n, hinge_sign)
    return (np.asarray(h, float) + into * (measured["handle_centre_to_face_m"] + measured["pin_behind_face_m"])
            + hinge_side * measured["hinge_along_door_m"])


def side_grasp_poses(h, orient, n, hinge_sign=1):
    """The side grasp for detected handle `h` (handle orientation `orient`, outward door normal `n`, hinge on the
    handle's left for `hinge_sign` +1, right for -1): (bar point where the pads close, waypoint 3 (the end of the
    reach-in), waypoint 1, hand quaternion xyzw, the turn waypoint). Mirrored for -1: the same offsets in the door's
    frame (0.6 cm into the door, 0.8 cm past the bar toward the free edge, the handle's height)."""
    into, hinge_side = side_frame(n, hinge_sign)
    # the front grasp's hand turned 90 deg about the vertical (away from the hinge): its approach (into the door)
    # becomes "along the door, away from the hinge side"; its finger axis (along the door) becomes the door normal
    quat = (R.from_euler("z", -90 * hinge_sign, degrees=True) * front_hand(orient, n)).as_quat()
    approach = R.from_quat(quat).as_matrix()[:, 2]
    if np.dot(approach, -hinge_side) < 0.999:
        raise Refused(f"side grasp: the hand's approach {np.round(approach, 3)} isn't along the door -- refusing")
    tilt = np.radians(90.0 - SIDE_INSERT_DEG)  # the insert direction: along the door (away from the hinge), turned into it
    want = -hinge_side * np.cos(tilt) + into * np.sin(tilt)
    turn_z = float(np.arctan2(np.cross(approach, want)[2], np.dot(approach, want)))
    quat = (R.from_euler("z", turn_z) * R.from_quat(quat)).as_quat()
    approach = R.from_quat(quat).as_matrix()[:, 2]
    if np.dot(approach, want) < 0.999:
        raise Refused(f"side grasp: the hand's approach {np.round(approach, 3)} isn't the insert direction {np.round(want, 3)}")
    bar = np.asarray(h, float) - into * (BAR_FROM_DETECTED_M - SIDE_DEEPER_M)
    centred = bar - approach * shared.GRASP_PADS_PAST_TOOL  # the tool point with the pads centred on the bar
    waypoint = centred - approach * SIDE_STANDOFF_M
    grasp = waypoint + approach * SIDE_REACH_IN_M
    turn = waypoint - into * SIDE_TURN_BACK_M
    return bar, grasp, waypoint, quat, turn


def door_for(h, orient, n, hinge):
    """The closed door dict the planners check against, built as the open builds it from this detection."""
    into, _ = side_frame(n)
    g = np.asarray(h, float) - into * (F.GRIP_EXT - F.GRASP_X_ADJUST)  # the level front grasp's tool point
    if shared.DOOR_Z_FROM_HANDLE is None:
        raise Refused("side grasp: no DOOR_Z_FROM_HANDLE -- configure the fridge (open_door.configure) first")
    return {
        "door_z": [float(v) for v in float(h[2]) + np.asarray(shared.DOOR_Z_FROM_HANDLE, float)],  # as open_task
        "closed_normal": list(n),
        "closed_grasp_pos": list(g),
        "closed_handle": list(h),
        "hinge": list(hinge),
        "hinge_pivot": list(hinge),
        "door_open_deg": 0.0,
        "open_sign": -1.0,
        **shared.door_model(n, g, h),
    }


def plan_side_grasp(scene, rb, h, orient, n, hinge, q0):
    """The side grasp's joints from `q0`, NO gates (11-06, for testing): IK for the waypoint from each of
    open_door._config_seeds, the configuration with the least reconfiguration, then IK every STEP_M along the straight-in.
    Every number that a gate would have judged (IK error, joint jumps, the door model's clearance) is reported, not acted
    on. Returns a dict (poses, the waypoint joints and the straight-in leg, every configuration found)."""
    bar, grasp, waypoint, quat, turn = side_grasp_poses(h, orient, n)
    approach = R.from_quat(quat).as_matrix()[:, 2]
    into, _ = side_frame(n)
    door = door_for(h, orient, n, hinge)

    def line(q, start, direction, dist):
        """IK every STEP_M along a straight line, from joints `q`: (joints per step, per-step numbers)."""
        out, info = [], []
        k = max(1, int(np.ceil(dist / dc.STEP_M)))
        for i in range(1, k + 1):
            nq, err = sim.solve_ik(scene, rb, start + direction * dist * i / k, quat, q)
            nq = np.asarray(nq, float)
            info.append({"ik_err_m": float(err), "jump_deg": shared._joint_dist_deg(q, nq)})
            out.append(nq)
            q = nq
        return out, info

    configs = []
    for seed in shared._config_seeds(q0):
        q, err = sim.solve_ik(scene, rb, turn, quat, seed)
        q = np.asarray(q, float)
        if all(shared._joint_dist_deg(c["q"], q) > shared.SAME_CONFIG_DEG for c in configs):
            configs.append({"q": q, "ik_err_m": float(err), "reconfig_deg": shared._joint_dist_deg(q0, q),
                            "near_limit": shared._near_limit(rb, q, shared.CUROBO_LIMIT_MARGIN_DEG)})
    # converged, then inside every joint's limits (11-06: a turn pose with J6 3.5 deg past its limit), then the least
    # reconfiguration -- a choice among configurations, not a gate
    best = min(configs, key=lambda c: (c["ik_err_m"] > sim.MAX_IK_ERR, bool(c["near_limit"]), c["reconfig_deg"]))
    fwd, fwd_info = line(best["q"], turn, into, SIDE_TURN_BACK_M)
    leg, steps = line(fwd[-1], waypoint, approach, SIDE_REACH_IN_M)
    # the smooth route: forward to SIDE_CORNER_RADIUS_M short of waypoint 1, a quarter circle round the corner, then in
    r = SIDE_CORNER_RADIUS_M
    a_pts = [turn + into * (SIDE_TURN_BACK_M - r) * i / max(1, int(np.ceil((SIDE_TURN_BACK_M - r) / dc.STEP_M)))
             for i in range(1, max(1, int(np.ceil((SIDE_TURN_BACK_M - r) / dc.STEP_M))) + 1)]
    centre = waypoint - into * r + approach * r
    arc = [centre - approach * r * np.cos(th) + into * r * np.sin(th) for th in np.linspace(0, np.pi / 2, 7)[1:]]
    n_in = max(1, int(np.ceil((SIDE_REACH_IN_M - r) / dc.STEP_M)))
    in_pts = [waypoint + approach * (r + (SIDE_REACH_IN_M - r) * i / n_in) for i in range(1, n_in + 1)]
    route_pts, route, q = [*a_pts, *arc, *in_pts], [], best["q"]
    for pt in route_pts:
        nq, err = sim.solve_ik(scene, rb, pt, quat, q)
        q = np.asarray(nq, float)
        route.append(q)
    bodies = dc.DoorFrame(door, door["open_sign"]).bodies(scene, rb, 0.0)
    try:
        cl = min((dc.clearance(rb, x, bodies) for x in [best["q"], *fwd, *leg]), key=lambda t: t[0])
    finally:
        drop(rb, bodies)
    say(f"side grasp (no checks): turn {np.round(turn, 3)} -> {SIDE_TURN_BACK_M * 100:.0f} cm forward -> waypoint 1 "
        f"{np.round(waypoint, 3)} -> {SIDE_REACH_IN_M * 100:.0f} cm in -> waypoint 3 {np.round(grasp, 3)}; turn IK err "
        f"{best['ik_err_m'] * 100:.2f} cm, forward worst IK err {max(st['ik_err_m'] for st in fwd_info) * 100:.2f} cm, "
        f"{best['reconfig_deg']:.0f} deg reconfiguration; straight-in worst IK err "
        f"{max(st['ik_err_m'] for st in steps) * 100:.2f} cm, worst step {max(st['jump_deg'] for st in steps):.1f} deg; "
        f"{cl[0] * 100:.1f} cm from the door model ({cl[1]})")
    return {
        "handle": [float(v) for v in h],
        "normal": [float(v) for v in n],
        "hinge": [float(v) for v in hinge],
        "bar": [float(v) for v in bar],
        "grasp": [float(v) for v in grasp],
        "waypoint": [float(v) for v in waypoint],
        "turn": [float(v) for v in turn],
        "turn_back_m": SIDE_TURN_BACK_M,
        "quat": [float(v) for v in quat],
        "approach": [float(v) for v in approach],
        "standoff_m": SIDE_STANDOFF_M,
        "reach_in_m": SIDE_REACH_IN_M,
        "deeper_m": SIDE_DEEPER_M,
        "door": door,
        "q0": [float(v) for v in q0],
        "chosen": {"q_turn": [float(v) for v in best["q"]], "fwd": [[float(v) for v in x] for x in fwd],
                   "q_wp": [float(v) for v in fwd[-1]], "leg": [[float(v) for v in x] for x in leg], "fwd_steps": fwd_info,
                   "route": [[float(v) for v in x] for x in route], "route_pts": [[float(v) for v in x] for x in route_pts],
                   "reconfig_deg": best["reconfig_deg"], "ik_err_m": best["ik_err_m"], "steps": steps,
                   "min_clear_m": float(cl[0]), "min_clear_link": str(cl[1]), "why": ""},
        "tried": [{"q_wp": [float(v) for v in c["q"]], "reconfig_deg": c["reconfig_deg"], "ik_err_m": c["ik_err_m"], "why": ""}
                  for c in configs],
    }


# --- the curved approach (plan_side_curve) ---
CURVE_LEAD_M = dc.FINGERTIP_PAST_TOOL  # the curve is drawn for the fingertips, this far past the tool point along the
# approach: once the hand is on the curve's tangent, they lead and the rest of the hand trails them
CURVE_TIPS_STRAIGHT_M = 0.08  # the route ends straight along the door at the grasp's depth (the approach direction):
# the fingertips run straight this far before they reach the bar's centre line, so the back finger enters the gap
# behind the bar already square to the door and at its depth (the slide before was 17 cm for the tool point = 11 cm
# for the fingertips; this is 15 cm)
CURVE_STEP_M = 0.01  # route knots: the tool point and the fingertips move at most this far ...
CURVE_STEP_DEG = 2.0  # ... and the hand turns at most this much from one knot to the next
CURVE_IK_TOL_M = 0.001  # every knot's IK within this (position) ...
CURVE_IK_TOL_DEG = 0.5  # ... and this (orientation) of its pose, else that shape is not used
CURVE_KNOT_DEG = 2.0  # knots are added until no joint moves more than this from one to the next (_track) ...
CURVE_MAX_JUMP_DEG = 3.0  # ... a step that stays above this is a jump (an IK branch switch): that shape is not used
CURVE_TIMING = "joint"  # how the route is timed (route_samples): "mixed" = move_joints with route_lengths as
# cart_lengths, "joint" = in joint radians at JOINT_SPEED (run_side.py's move_joints(knots))
CURVE_CLEAR_ENOUGH_M = 0.03  # door clearance on the curve past this earns nothing more (_score)
CURVE_LIMIT_MARGIN_DEG = 10.0  # ... and every joint at least this far inside its limit
# The curve's shape (side_curve_poses): mid_depth_m: the fingertips' depth half way along the curve, from the slide's
# line (negative = toward the arm); start_slope: the curve's first direction, depth change per metre along the door
# (0 = start along the door, negative = also back from it); turn_end: how far along the curve (0..1) the hand has
# finished turning onto the curve's tangent; rise_end: how far along it the hand is at the grasp's height.
# CURVE_SHAPE was picked by plan_curve.py's search on the 11-06 detection (the most door clearance on the curve, then
# the shortest time); plan_side_curve tries CURVE_SHAPES and keeps the best that passes.
CURVE_SHAPE = {"mid_depth_m": -0.015, "start_slope": -0.6, "turn_end": 0.65, "rise_end": 0.8}
# tried in this order until one passes with CURVE_GOOD_CLEAR_M on the curve: the later ones first take the fingertips
# further back from the door, for a door turned toward the arm on the hinge side (at home the fingertips are only
# ~2 cm short of a square door's slide line; a door turned 15 deg puts them ~5 cm past it)
CURVE_SHAPES = (CURVE_SHAPE,
                {**CURVE_SHAPE, "mid_depth_m": -0.03, "start_slope": -1.2},
                {**CURVE_SHAPE, "mid_depth_m": -0.045, "start_slope": -2.0, "turn_end": 0.75},
                {**CURVE_SHAPE, "mid_depth_m": -0.06, "start_slope": -3.0, "turn_end": 0.8},
                {**CURVE_SHAPE, "mid_depth_m": -0.03, "start_slope": 0.0})
CURVE_GOOD_CLEAR_M = 0.015
# the bar for the sim's checks (the door model has no handle): BAR_DEPTH_M deep, this wide along the door (not
# measured; 1 in, generous), handle_length_m tall, its back BAR_BACK_TO_FACE_M in front of the door's face; and its two
# mounts, from the bar back to the face, this tall, at its ends
BAR_WIDTH_M = 0.0254
BAR_MOUNT_H_M = 0.02


def _smoothstep(x):
    """0 -> 1 with zero first and second derivatives at both ends (quintic), clamped outside [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


def _quintic(d0, slope0, mid, at=0.5):
    """Coefficients c0..c5 of d(t) on [0, 1] with d(0) = d0, d'(0) = slope0, d(at) = mid, d(1) = d'(1) = d''(1) = 0."""
    k = np.arange(6)
    rows = [k == 0, np.where(k == 1, 1.0, 0.0), at**k, np.ones(6), k.astype(float), (k * (k - 1)).astype(float)]
    return np.linalg.solve(np.array(rows, float), np.array([d0, slope0, mid, 0.0, 0.0, 0.0]))


def _yaw(a, b):
    """Signed angle about the vertical from horizontal direction `a` to `b`."""
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


def side_curve_poses(tool0, quat0, n, grasp, quat_end, bar, shape):
    """The curved route's tool poses from tool pose (`tool0`, `quat0`) to the side grasp (`grasp`, `quat_end`; pads
    on `bar`), door normal `n`: a list of (position, quaternion xyzw, segment), every CURVE_STEP_M / CURVE_STEP_DEG.

    In the door's frame (depth along `into` from the slide's line, `w` along the door toward the hinge) the fingertips
    (CURVE_LEAD_M past the tool point) go from where they start to the slide's start on a curve whose depth is a
    quintic in the progress t (0..1) with w falling linearly: depth `shape["mid_depth_m"]` at t = 0.5, starting off in
    direction `start_slope`, ending on the slide's line with zero slope and zero curvature (so the route's turn rate
    is zero where the slide begins). The approach's angle off `into` blends (quintic, done at t = turn_end) from the
    start's to the curve's tangent, the rest of the start's tilt goes with it; the height reaches the grasp's at t =
    rise_end. Segments: "turn" (the hand still turning onto the tangent), "pull-in" (on the tangent), "slide" (the
    straight along the door, at the grasp's depth and in its orientation). Raises Refused for a shape that would
    take the fingertips deeper than the slide."""
    into, hinge_side = side_frame(n)
    up = np.array([0.0, 0.0, 1.0])
    tool0, grasp, bar = (np.asarray(v, float) for v in (tool0, grasp, bar))
    r0, r_end = R.from_quat(quat0), R.from_quat(quat_end)
    a0 = r0.as_matrix()[:, 2]
    lead0 = tool0 + CURVE_LEAD_M * a0
    d0, w0 = float(np.dot(lead0 - grasp, into)), float(np.dot(lead0 - grasp, hinge_side))
    w_slide = _slide_from_grasp(grasp, bar, n)
    w_lead = w_slide - CURVE_LEAD_M  # the fingertips where the slide starts
    span = w0 - w_lead
    if span < 0.05:
        raise Refused(f"side curve: the start's fingertips are {w0 * 100:.1f} cm along the door from the grasp, not "
                      f"past the slide's start ({w_lead * 100:.1f} cm) by 5 cm -- no room for a curve")
    c = _quintic(d0, shape["start_slope"] * span, shape["mid_depth_m"])
    t = np.linspace(0.0, 1.0, 1601)
    depth = np.polyval(c[::-1], t)
    slope = np.polyval((c[1:] * np.arange(1, 6))[::-1], t)  # d depth / dt
    # a door turned against the arm can put the start's fingertips past the slide's line (deeper): then the curve must
    # first take them back behind it, and never deeper than they start; once behind it, they stay behind it
    behind = np.flatnonzero(depth <= 0.0)
    if (behind.size == 0 or depth[behind[0]:].max() > 1e-4 or depth.max() > max(d0, 0.0) + 1e-4
            or (d0 > 0.0 and slope[0] >= 0.0)):
        worst = depth[behind[0]:].max() if behind.size else depth.max()
        raise Refused(f"side curve {shape}: the fingertips would go {max(worst, depth.max() - max(d0, 0.0)) * 100:.1f}"
                      f" cm past the slide's line (they start {d0 * 100:+.1f} cm from it)")
    # the approach's angle off `into`, toward the free edge (= -hinge_side): the start's, then the curve's tangent
    a0h = a0 - up * np.dot(a0, up)
    g0 = float(np.arctan2(np.dot(a0h, -hinge_side), np.dot(a0h, into)))
    g_tan = np.arctan2(span, slope)  # the tangent (slope along `into`, span along -hinge_side per unit t)
    blend = _smoothstep(t / shape["turn_end"])
    gam = g0 + (g_tan - g0) * blend
    dirs = np.outer(np.cos(gam), into) - np.outer(np.sin(gam), hinge_side)
    yaw = np.unwrap(np.arctan2(a0h[0] * dirs[:, 1] - a0h[1] * dirs[:, 0], a0h[0] * dirs[:, 0] + a0h[1] * dirs[:, 1]))
    yaw_end = _yaw(a0h, -hinge_side)
    rest = (r0.inv() * R.from_rotvec(-yaw_end * up) * r_end).as_rotvec()  # the start's tilt etc.: the end's turned back
    rots = R.from_rotvec(yaw[:, None] * up) * r0 * R.from_rotvec(blend[:, None] * rest)
    z = lead0[2] + (grasp[2] - lead0[2]) * _smoothstep(t / shape["rise_end"])
    lead = grasp + np.outer(depth, into) + np.outer(w0 - t * span, hinge_side)
    lead[:, 2] = z
    tools = lead - CURVE_LEAD_M * rots.as_matrix()[:, :, 2]
    # knots where the tool point's, the fingertips' or the hand's turn's running total (each in units of its step)
    # passes the next whole number; the dense samples are ~0.2 mm apart
    quats = rots.as_quat()
    turn = np.r_[0.0, np.cumsum(2 * np.arccos(np.clip(np.abs(np.sum(quats[1:] * quats[:-1], axis=1)), 0, 1)))]
    prog = np.maximum.reduce([
        np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(tools, axis=0), axis=1))] / CURVE_STEP_M,
        np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(lead, axis=0), axis=1))] / CURVE_STEP_M,
        np.degrees(turn) / CURVE_STEP_DEG])
    idx = np.r_[np.flatnonzero(np.diff(np.floor(prog)) > 0) + 1, len(t) - 1]
    idx = np.unique(idx[idx > 0])
    out = [(tools[i], quats[i], "turn" if blend[i] < 1.0 else "pull-in") for i in idx]
    k = int(np.ceil(w_slide / CURVE_STEP_M))
    for i in range(1, k + 1):
        out.append((grasp + hinge_side * w_slide * (1 - i / k), np.asarray(quat_end, float), "slide"))
    return out


def _slide_from_grasp(grasp, bar, n, hinge_sign=1, back=None):
    """How far back along the slide (`back`, from the grasp; default along the door toward the hinge) the straight
    slide starts (tool point): the fingertips run straight CURVE_TIPS_STRAIGHT_M before they reach the bar's centre."""
    if back is None:
        _, back = side_frame(n, hinge_sign)
    return float(np.dot(np.asarray(bar, float) - grasp, back)) + CURVE_TIPS_STRAIGHT_M + dc.FINGERTIP_PAST_TOOL


def slide_back(quat):
    """Back along the hand's insert (minus its approach axis), horizontal unit: the slide runs from grasp + back * w."""
    b = -R.from_quat(quat).as_matrix()[:, 2]
    b[2] = 0.0
    return b / np.linalg.norm(b)


def _unwrap(q_start, knots):
    """Knots made continuous from `q_start`: each continuous joint steps the short way from the last (as
    rammp_arm.joint_trajectory sends them)."""
    out, prev = [], np.asarray(q_start, float)
    for k in knots:
        d = np.asarray(k, float) - prev
        d[rammp_arm.CONTINUOUS] = (d[rammp_arm.CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
        prev = prev + d
        out.append(prev.copy())
    return out


def route_samples(q0, knots, cart_lengths=None, speed_scale=1.0):
    """(t, q, qd) of the route from `q0` through joint `knots` as move_joints times it (rammp_arm.joint_trajectory:
    one minimum-jerk time law over the whole path, in joint radians at JOINT_SPEED, or with `cart_lengths` in metres
    at CART_SPEED), sampled with a C1 (PCHIP) curve through the knots instead of straight joint segments -- what
    run_side.py's _smooth_min_jerk_samples sends."""
    path = np.array([np.asarray(q0, float), *_unwrap(q0, knots)])
    seg = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg if cart_lengths is None else cart_lengths)])
    speed = (rammp_arm.JOINT_SPEED if cart_lengths is None else rammp_arm.CART_SPEED) * speed_scale
    duration = max(rammp_arm.MIN_DURATION_S, 1.875 * s[-1] / speed,
                   1.875 * float(np.sum(seg)) / (rammp_arm.JOINT_SPEED * speed_scale))
    keep = np.r_[True, np.diff(s) > 1e-9]
    f = PchipInterpolator(s[keep], path[keep], axis=0)
    tt = np.linspace(0.0, duration, max(2, int(np.ceil(duration / rammp_arm.DT)) + 1))
    u = tt / duration
    q = f(s[-1] * (10 * u**3 - 15 * u**4 + 6 * u**5))
    qd = np.gradient(q, tt, axis=0)
    qd[0] = qd[-1] = 0.0
    return tt, q, qd


def speed_dip(v):
    """How far (fraction of its peak) speed profile `v` dips below rising-then-falling: 0 for a single smooth bell,
    0.5 when it halves somewhere before picking up again -- the stop-and-go a route's timing gives."""
    v = np.asarray(v, float)
    k = int(np.argmax(v))
    env = np.r_[np.maximum.accumulate(v[: k + 1]), np.maximum.accumulate(v[k:][::-1])[::-1][1:]]
    return float(np.max(env - v) / max(v[k], 1e-9))


def route_motion(scene, rb, t, q):
    """The tool's (positions, linear speed m/s, turn rate rad/s) along samples `q` at times `t`."""
    poses = [sim.tool_pose(scene, rb, x) for x in q]
    pos = np.array([p_[0] for p_ in poses])
    rot = R.from_quat([p_[1] for p_ in poses])
    v = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t)]
    w = np.r_[0.0, [(rot[i].inv() * rot[i + 1]).magnitude() for i in range(len(rot) - 1)] / np.diff(t)]
    return pos, v, w


def add_bar_model(scene, rb, door, h, n):
    """The handle bar and its two mounts as collision boxes (the door model has none), placed from the door model's
    face: [(id, name)], remove with `drop`."""
    c = rb.physics_client_id
    into, hinge_side = side_frame(n)
    out = -into
    face = np.asarray(door["closed_grasp_pos"], float) + into * dc.door_dims(door)[0]
    h = np.asarray(h, float)
    back = h + into * (float(np.dot(face - h, into)) - BAR_BACK_TO_FACE_M)  # the bar's back, at the handle point
    base = np.asarray(scene.robot_base_pose.position, float)
    yaw = float(np.arctan2(out[1], out[0]))
    half_len = F.MEASURED["handle_length_m"] / 2

    def box(centre, half):
        cs = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(half), physicsClientId=c)
        return pb.createMultiBody(0, cs, basePosition=(base + centre).tolist(),
                                  baseOrientation=R.from_euler("z", yaw).as_quat().tolist(), physicsClientId=c)

    bodies = [(box(back - into * BAR_DEPTH_M / 2, [BAR_DEPTH_M / 2, BAR_WIDTH_M / 2, half_len]), "handle bar")]
    for sgn in (-1.0, 1.0):
        mc = back + into * BAR_BACK_TO_FACE_M / 2 + np.array([0, 0, sgn * (half_len + BAR_MOUNT_H_M / 2)])
        bodies.append((box(mc, [BAR_BACK_TO_FACE_M / 2, BAR_WIDTH_M / 2, BAR_MOUNT_H_M / 2]), "handle mount"))
    return bodies


def link_clearances(rb, q, bodies):
    """{link name: (distance, body name)}: each robot link's distance to the nearest of `bodies` at joints `q`, the
    gripper open (as sim.clearance)."""
    sim.clearance(rb, q, [])  # sets the arm and opens the fingers
    c, out = rb.physics_client_id, {}
    for bid, name in bodies:
        for pt in pb.getClosestPoints(rb.robot_id, bid, sim.CLEARANCE_QUERY_M, physicsClientId=c):
            ln = pb.getJointInfo(rb.robot_id, pt[3], physicsClientId=c)[12].decode() if pt[3] >= 0 else "base"
            if ln not in out or pt[8] < out[ln][0]:
                out[ln] = (float(pt[8]), name)
    return out


def _back_finger_links(scene, rb, q, into):
    """The gripper links of the finger nearer the door at joints `q` (the one that goes between the bar and the door):
    every robotiq link named with that side."""
    c = rb.physics_client_id
    sim.tool_pose(scene, rb, q)
    names = [pb.getJointInfo(rb.robot_id, j, physicsClientId=c)[12].decode() for j in range(pb.getNumJoints(rb.robot_id, physicsClientId=c))]
    depth = {side: float(np.dot(pb.getLinkState(rb.robot_id, names.index(f"arm_robotiq_85_{side}_finger_tip_link"),
                                                physicsClientId=c)[4], into)) for side in ("left", "right")}
    side = max(depth, key=depth.get)
    return {nm for nm in names if nm.startswith("arm_robotiq_85_") and f"_{side}_" in nm}


def _limit_margin_deg(rb, qs):
    """(smallest distance in deg of any limited joint to its limit over joints `qs`, which joint)."""
    best = (np.inf, "")
    for j, jj in enumerate(sim.ARM):
        info = pb.getJointInfo(rb.robot_id, jj, physicsClientId=rb.physics_client_id)
        lo, hi = info[8], info[9]
        if lo < hi:
            m = float(np.degrees(min(np.min(np.asarray(qs)[:, j] - lo), np.min(hi - np.asarray(qs)[:, j]))))
            if m < best[0]:
                best = (m, f"J{j + 1}")
    return best


def _ik(scene, rb, pos, quat, q):
    """sim.solve_ik seeded at `q`, re-solved up to 4 times toward CURVE_IK_TOL_M / 5; the joints continuous from q."""
    nq, err = sim.solve_ik(scene, rb, pos, quat, q)
    for _ in range(4):
        if err <= CURVE_IK_TOL_M * 0.2:
            break
        nq, err = sim.solve_ik(scene, rb, pos, quat, nq)
    return _unwrap(q, [nq])[0], float(err)


def _track(scene, rb, poses, q0):
    """IK every pose of `poses` seeded from the last knot (from `q0`). Where a joint would move more than
    CURVE_KNOT_DEG from one knot to the next, the step is halved (position interpolated, orientation slerped -- the
    knots are CURVE_STEP_M apart on a smooth curve, so the halves are on it to well under 1 mm), down to 1/32 of it: a
    step still too big there is a jump (the IK switching branches), reported as such. Returns (the poses used, their
    knots made continuous from q0, per-knot position error m, orientation error deg, largest joint step deg)."""
    out_p, knots, perr, aerr, jump = [], [], [], [], []
    q, prev = np.asarray(q0, float), None

    def go(pa, pb_, depth):
        nonlocal q
        nq, err = _ik(scene, rb, pb_[0], pb_[1], q)
        step = float(np.degrees(np.max(np.abs(nq - q))))
        if step > CURVE_KNOT_DEG and pa is not None and depth < 5:
            mid = ((np.asarray(pa[0]) + pb_[0]) / 2,
                   (R.from_quat(pa[1]) * R.from_rotvec((R.from_quat(pa[1]).inv() * R.from_quat(pb_[1])).as_rotvec()
                                                       / 2)).as_quat(), pb_[2])
            go(pa, mid, depth + 1)
            go(mid, pb_, depth + 1)
            return
        fk_q = sim.tool_pose(scene, rb, nq)[1]
        out_p.append(pb_)
        knots.append(nq)
        perr.append(err)
        aerr.append(float(np.degrees((R.from_quat(fk_q).inv() * R.from_quat(pb_[1])).magnitude())))
        jump.append(step)
        q = nq

    tool0, quat0 = sim.tool_pose(scene, rb, q)
    prev = (tool0, quat0, "start")
    for pose in poses:
        pose = (np.asarray(pose[0], float), np.asarray(pose[1], float), pose[2])
        go(prev, pose, 0)
        prev = pose
    return out_p, knots, perr, aerr, jump


def route_lengths(q0, knots, pts, quats, tool0, quat0):
    """move_joints' cart_lengths for the route that time it by whichever limit binds on each segment: the segment's
    largest joint move at JOINT_SPEED, its tool travel at CART_SPEED, or its turn (ROT_EQUIV_M per rad) at
    CART_SPEED, all as metres at CART_SPEED. With these the minimum-jerk peak keeps every joint under JOINT_SPEED
    and the tool under CART_SPEED everywhere along the route, not only on average."""
    path = np.array([np.asarray(q0, float), *_unwrap(q0, knots)])
    dq = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    p = np.array([np.asarray(tool0, float), *np.asarray(pts, float)])
    r = R.from_quat([np.asarray(quat0, float), *np.asarray(quats, float)])
    dx = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dth = np.array([(r[i].inv() * r[i + 1]).magnitude() for i in range(len(r) - 1)])
    v = rammp_arm.CART_SPEED
    return [float(max(1e-5, a * v / rammp_arm.JOINT_SPEED, b, c * rammp_arm.ROT_EQUIV_M)) for a, b, c in zip(dq, dx, dth)]


def _route_report(scene, rb, q0, poses, knots, bodies, back, full=False):
    """Per-knot clearances (all links but the back finger, and the back finger) and per-segment minima; and the route
    timed as CURVE_TIMING says (route_samples): duration, peak joint speeds, the tool's peak speed and its speed_dip.
    `full`: also the other timing and the clearance of every 2nd sample (slower; for reports)."""
    rows = []
    for (pos, quat, seg), q in zip(poses, knots):
        cl = link_clearances(rb, q, bodies)
        other = min(((d, f"{ln} vs {b}") for ln, (d, b) in cl.items() if ln not in back), default=(9.0, ""))
        bk = min(((d, f"{ln} vs {b}") for ln, (d, b) in cl.items() if ln in back), default=(9.0, ""))
        rows.append({"seg": seg, "clear": other, "back": bk})
    segs = {}
    for r in rows:
        s = segs.setdefault(r["seg"], {"clear": (9.0, ""), "back": (9.0, ""), "knots": 0})
        s["knots"] += 1
        s["clear"], s["back"] = min(s["clear"], r["clear"]), min(s["back"], r["back"])
    tool0, quat0 = sim.tool_pose(scene, rb, np.asarray(q0, float))
    lengths = route_lengths(q0, knots, [p_[0] for p_ in poses], [p_[1] for p_ in poses], tool0, quat0)
    timing = {}
    for name, cl_ in (("joint", None), ("mixed", lengths)):
        if name != CURVE_TIMING and not full:
            continue
        t, q, qd = route_samples(q0, knots, cl_)
        _, v, w = route_motion(scene, rb, t[::2], q[::2])
        timing[name] = {"duration_s": float(t[-1]), "peak_joint_speed": float(np.max(np.abs(qd))),
                        "peak_joint_speed_each": [float(x) for x in np.max(np.abs(qd), axis=0)],
                        "peak_tool_speed": float(v.max()), "peak_turn_rate": float(w.max()),
                        "tool_speed_dip": speed_dip(v), "combined_speed_dip": speed_dip(v + w * rammp_arm.ROT_EQUIV_M)}
        if name == CURVE_TIMING and full:
            timing["sample_clear"] = min((min(((d, f"{ln} vs {b}") for ln, (d, b) in link_clearances(
                rb, x, bodies).items() if ln not in back), default=(9.0, "")) for x in q[::2]))
    timing.setdefault("sample_clear", (float("nan"), "not checked (full=False)"))
    timing["lengths"] = lengths
    return rows, segs, timing


def _score(r):
    """A route's merit for plan_side_curve's choice: clearance on the curve (cm, no credit past CURVE_CLEAR_ENOUGH_M),
    less 4 points per unit of speed_dip (stop-and-go: 0.25 dip costs as much as a cm of clearance), less 1 per
    second."""
    tm = r["timing"][CURVE_TIMING]
    return (min(r["curve_clear"][0], CURVE_CLEAR_ENOUGH_M) * 100 - 4 * tm["combined_speed_dip"] - tm["duration_s"])


def plan_side_curve(scene, rb, h, orient, n, hinge, q0, shapes=None, warm=None, full=False):
    """The side grasp reached by ONE smooth route from joints `q0` (side_curve_poses), NO gates on the arm side (as
    plan_side_grasp): for each of `shapes` (default CURVE_SHAPES) the route is IK'd knot by knot from q0; a shape is
    dropped when a knot misses its pose (CURVE_IK_TOL_*), a joint jumps (CURVE_MAX_JUMP_DEG) or comes within
    CURVE_LIMIT_MARGIN_DEG of its limit -- planning choices, not checks on the arm. Of the rest: the most clearance to
    the door, its body and the bar on the curve (every link; the back finger, which must get to the door in the slide,
    only as far as it is never deeper than in the slide), then the shortest time. Returns plan_side_grasp's dict with
    chosen["route"] (the joint knots after q0), "route_pts", "route_quats", "route_seg" and the numbers; raises Refused
    when no shape works. The default shapes are tried best first and the first that passes is returned (~0.5 s).
    `warm`: an earlier plan (this function's dict) for a nearby detection: its route's knots seed the reach check of
    the slide's ends instead of the 10 configuration seeds. `full`: also time the route the other way and check the
    clearance of every trajectory sample (slower, for reports)."""
    bar, grasp, _waypoint, quat, _turn = side_grasp_poses(h, orient, n)
    into, hinge_side = side_frame(n)
    approach = R.from_quat(quat).as_matrix()[:, 2]
    door = door_for(h, orient, n, hinge)
    tool0, quat0 = sim.tool_pose(scene, rb, np.asarray(q0, float))
    bodies = dc.DoorFrame(door, door["open_sign"]).bodies(scene, rb, 0.0) + add_bar_model(scene, rb, door, h, n)
    # the slide's two ends must be in reach in some configuration, whatever curve leads there (a door turned with its
    # hinge side away from the arm puts the slide's start beyond the arm's reach: refuse that up front, saying so)
    w_slide = _slide_from_grasp(grasp, bar, n)
    for name, pt in (("start", grasp + hinge_side * w_slide), ("end", grasp)):
        seeds = shared._config_seeds(q0)
        if warm is not None:  # the earlier plan's knot nearest this point, else the configuration seeds
            wp_ = np.asarray(warm["chosen"]["route_pts"], float)
            seeds = [np.asarray(warm["chosen"]["route"][int(np.argmin(np.linalg.norm(wp_ - pt, axis=1)))], float)]
        sols = [_ik(scene, rb, pt, quat, seed) for seed in seeds]
        q_best, err = min(sols, key=lambda s: s[1])
        if err > CURVE_IK_TOL_M and warm is not None:
            sols = [_ik(scene, rb, pt, quat, seed) for seed in shared._config_seeds(q0)]
            q_best, err = min(sols, key=lambda s: s[1])
        if err > CURVE_IK_TOL_M:
            drop(rb, bodies)
            raise Refused(f"side curve: the slide's {name} {np.round(pt, 3)} (the door's normal {np.round(n, 3)}) is "
                          f"out of the arm's reach: the closest IK is {err * 100:.1f} cm off with J4 at "
                          f"{np.degrees(q_best[3]):.0f} deg (straight = 0)")
    results, why = [], []
    try:
        for shape in shapes or CURVE_SHAPES:
            try:
                poses = side_curve_poses(tool0, quat0, n, grasp, quat, bar, shape)
            except Refused as e:
                why.append(str(e))
                continue
            poses, knots, perr, aerr, jump = _track(scene, rb, poses, q0)
            margin, mj = _limit_margin_deg(rb, knots)
            bad = []
            if max(perr) > CURVE_IK_TOL_M or max(aerr) > CURVE_IK_TOL_DEG:
                i = int(np.argmax(perr))
                bad.append(f"IK off its pose by {max(perr) * 100:.2f} cm / {max(aerr):.2f} deg (knot {i})")
            if max(jump) > CURVE_MAX_JUMP_DEG:
                bad.append(f"a joint jumps {max(jump):.1f} deg between knots (knot {int(np.argmax(jump))})")
            if margin < CURVE_LIMIT_MARGIN_DEG:
                bad.append(f"{mj} {margin:.1f} deg from its limit")
            if bad:
                why.append(f"side curve {shape}: " + "; ".join(bad))
                continue
            back = _back_finger_links(scene, rb, knots[-1], into)
            rows, segs, timing = _route_report(scene, rb, q0, poses, knots, bodies, back, full)
            curve = [r for r in rows if r["seg"] != "slide"]
            slide_back = segs["slide"]["back"][0]
            deeper = min(r["back"][0] for r in curve) - slide_back  # >= 0: the back finger never deeper than in the slide
            results.append({"shape": dict(shape), "poses": poses, "knots": knots, "perr": perr, "aerr": aerr,
                            "jump": jump, "margin": (margin, mj), "rows": rows, "segs": segs, "timing": timing,
                            "back": sorted(back), "curve_clear": min(r["clear"] for r in curve), "back_deeper_m": deeper})
            if deeper > -0.002 and results[-1]["curve_clear"][0] >= CURVE_GOOD_CLEAR_M and shapes is None:
                break  # good enough: the default list is in order of preference
    finally:
        drop(rb, bodies)
    ok = [r for r in results if r["back_deeper_m"] > -0.002]
    if not ok:
        raise Refused("side curve: no shape works -- " + " | ".join(
            why + [f"{r['shape']}: the back finger {-r['back_deeper_m'] * 100:.1f} cm deeper than in the slide"
                   for r in results]))
    # the most clearance on the curve (no credit past 4 cm), then the shortest time
    best = max(ok, key=_score)
    sg = best["segs"]
    say(f"side curve (no checks): {len(best['knots'])} knots from the start to {np.round(grasp, 3)}, shape "
        f"{best['shape']}; {best['timing'][CURVE_TIMING]['duration_s']:.2f} s at full speed; clearance on the curve "
        f"{best['curve_clear'][0] * 100:.1f} cm ({best['curve_clear'][1]}), in the slide "
        f"{sg['slide']['clear'][0] * 100:.1f} cm ({sg['slide']['clear'][1]}); back finger in the slide "
        f"{sg['slide']['back'][0] * 100:.1f} cm ({sg['slide']['back'][1]}, the model's); worst IK "
        f"{max(best['perr']) * 1000:.1f} mm / {max(best['aerr']):.2f} deg, worst joint step {max(best['jump']):.1f} deg, "
        f"{best['margin'][1]} {best['margin'][0]:.0f} deg inside its limit")
    fl = lambda v: [float(x) for x in v]  # noqa: E731
    return {
        "handle": fl(h), "normal": fl(n), "hinge": fl(hinge), "bar": fl(bar), "grasp": fl(grasp),
        "waypoint": fl(grasp + hinge_side * w_slide),  # where the straight slide starts
        "quat": fl(quat), "approach": fl(approach), "deeper_m": SIDE_DEEPER_M, "door": door, "q0": fl(q0),
        "tool0": fl(tool0), "quat0": fl(quat0), "slide_m": w_slide, "tips_straight_m": CURVE_TIPS_STRAIGHT_M,
        "chosen": {
            "route": [fl(q) for q in best["knots"]],
            "route_pts": [fl(p[0]) for p in best["poses"]],
            "route_quats": [fl(p[1]) for p in best["poses"]],
            "route_seg": [p[2] for p in best["poses"]],
            "shape": best["shape"], "ik_err_m": best["perr"], "ik_err_deg": best["aerr"], "jump_deg": best["jump"],
            "limit_margin_deg": best["margin"][0], "limit_margin_joint": best["margin"][1],
            "clear": [[float(r["clear"][0]), r["clear"][1]] for r in best["rows"]],
            "back_clear": [[float(r["back"][0]), r["back"][1]] for r in best["rows"]],
            "back_links": best["back"],
            "segments": {k: {"knots": v["knots"], "clear_m": float(v["clear"][0]), "clear_link": v["clear"][1],
                             "back_m": float(v["back"][0]), "back_link": v["back"][1]} for k, v in sg.items()},
            "timing_used": CURVE_TIMING,
            "cart_lengths": best["timing"]["lengths"] if CURVE_TIMING == "mixed" else None,
            "timing": {**{k: v for k, v in best["timing"].items() if k not in ("sample_clear", "lengths")},
                       "sample_clear_m": float(best["timing"]["sample_clear"][0]),
                       "sample_clear_link": best["timing"]["sample_clear"][1]},
            "min_clear_m": float(best["curve_clear"][0]), "min_clear_link": best["curve_clear"][1],
        },
        "tried": [{"shape": r["shape"], "curve_clear_m": float(r["curve_clear"][0]), "curve_clear_link": r["curve_clear"][1],
                   "back_deeper_m": float(r["back_deeper_m"]), "score": float(_score(r)), "knots": len(r["knots"]),
                   "max_jump_deg": max(r["jump"]), "limit_margin_deg": r["margin"][0],
                   "limit_margin_joint": r["margin"][1], "sample_clear_m": float(r["timing"]["sample_clear"][0]),
                   "timing": {k: r["timing"][k] for k in ("joint", "mixed") if k in r["timing"]}} for r in results],
        "refused": why,
    }


# --- the general route (plan_side_general): cuRobo through free space to a searched slide configuration, then the
# straight slide, joined into one route ---
GEN_LIMIT_MARGIN_DEG = 6.0  # the whole slide this far inside every joint limit (cuRobo refuses goals within 5.7)
GEN_J1_OFFSETS_DEG = (0.0, -30.0, 30.0)  # IK seeds for the slide: the current joints with J1 and J3 offset ...
GEN_J3_OFFSETS_DEG = (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, -120.0, 120.0, 180.0)  # ... each wrist flipped too
GEN_SAME_CONFIG_DEG = 10.0
GEN_MAX_CONFIGS_AFTER = 12  # ... and this many when what follows the grasp (`after`) must also work from it
GEN_MAX_CONFIGS = 4  # slide_configs stops after this many  # slide configurations closer than this (every joint) are one
GEN_SLIDE_STEP_M = 0.01  # slide knots
GEN_KNOT_MIN_DEG = 0.5  # cuRobo's trajectory, thinned to knots at least this far apart (largest joint)
GEN_FACE_DEEPER_M = 0.02  # cuRobo's door face this much deeper than the model's (the model's back finger overlaps the
# model's door by ~1.3 cm in the slide; the real open gripper fits between bar and door)
GEN_GOAL_OUT_M = 0.04  # cuRobo's goal: the slide's start this far out of the door (its collision spheres call any
# pose at the grasp's depth a collision with the door: 2 cm out still did, 3 cm planned on 11-06)
GEN_MERGE_M = 0.08  # from cuRobo's goal the hand merges into the slide's line over this much travel along the door
GEN_MERGE_MIN_M = 0.02  # the merge never shorter than this
GEN_SUBDIVIDE = 3  # _ik_line halves a piece up to this many times where the joints move a lot
GEN_TIPS_MIN_M = 0.02  # the shortest slide: the fingertips this much straight travel before the bar (clears the bar's
# side as the hand comes in from out of the door)
GEN_REACH_STEP_M = 0.01  # slide-length search step
GEN_REACH_SEEDS = 20  # IK seeds (nearest the start) for measuring the slide line's reach
# (depth eased in, zero slope at both ends; the hand already in the grasp's orientation), then slides
GEN_PROTECT_M = 0.11  # fast_timing's protect_tail: the route's knots within this of the end are not smoothed
GEN_JUNCTION_TOL_M = 0.005  # the timed route within this of the slide's line over the last GEN_JUNCTION_LAST_M
GEN_JUNCTION_LAST_M = 0.10
WORLDS_DIR = "/home/snorlax/.ros/rammp_box_opening/worlds"  # mounted at this path in the planner container


class CuroboClient:
    """Planning-only calls to the cuRobo planner container (/rammp_curobo/set_world, /rammp_curobo/plan_to_joints):
    nothing here moves the arm. `node` must be spun by an executor in another thread."""

    def __init__(self, node, timeout_s=10.0):
        from rammp_curobo_interfaces.action import PlanToJoints
        from rammp_curobo_interfaces.srv import SetWorld
        from rclpy.action import ActionClient

        self._pt, self._sw, self.timeout_s, self.world = PlanToJoints, SetWorld, timeout_s, None
        self._plan = ActionClient(node, PlanToJoints, "/rammp_curobo/plan_to_joints")
        self._world = node.create_client(SetWorld, "/rammp_curobo/set_world")

    def _wait(self, fut, what):
        import time

        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > self.timeout_s:
                raise Refused(f"cuRobo: no answer to {what} in {self.timeout_s:.0f} s")
            time.sleep(0.002)
        return fut.result()

    def set_world(self, path):
        """Load world YAML `path` (absolute) into cuRobo -- replaces whatever world it had. Skipped if already loaded."""
        if path == self.world:
            return
        if not self._world.wait_for_service(timeout_sec=self.timeout_s):
            raise Refused("cuRobo: no /rammp_curobo/set_world service")
        res = self._wait(self._world.call_async(self._sw.Request(world=path)), "set_world")
        if not res.success:
            raise Refused(f"cuRobo refused the world {path}: {res.message}")
        self.world = path

    def plan(self, start, goal):
        """cuRobo's joint trajectory from `start` to `goal` (rad, arm_joint_1..7): (positions (N, 7), times, planning
        time s, goal mismatch rad). Raises Refused when it fails."""
        if not self._plan.wait_for_server(timeout_sec=self.timeout_s):
            raise Refused("cuRobo: no /rammp_curobo/plan_to_joints action server")
        gh = self._wait(self._plan.send_goal_async(self._pt.Goal(target_joints=[float(v) for v in goal],
                                                                 start_joints=[float(v) for v in start])), "the goal")
        if not gh.accepted:
            raise Refused("cuRobo rejected the plan_to_joints goal")
        res = self._wait(gh.get_result_async(), "plan_to_joints").result
        if not res.success:
            raise Refused(f"cuRobo: no plan ({res.message})")
        tr = res.trajectory
        order = [list(tr.joint_names).index(f"arm_joint_{i}") for i in range(1, 8)]
        q = np.array([[p.positions[k] for k in order] for p in tr.points], float)
        t = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in tr.points])
        return q, t, float(res.planning_time), float(res.goal_mismatch_rad)


def side_world(door, h, n, hinge_cut=None):
    """cuRobo's world for the side grasp, from the door dict (door_for) of detection `h`, `n`: the door slab (its face
    GEN_FACE_DEEPER_M deeper than the model's), the fridge body behind it, the handle bar and its two mounts. Writes
    it to WORLDS_DIR (named by its content) and returns (path, obstacles). No chair, person or floor. `hinge_cut`:
    the door and body end this far along the door toward the hinge from the handle (start_hinge_cut) instead of at the
    hinge."""
    import hashlib
    import os

    into, hinge_side = side_frame(n, hinge_sign_of(door["closed_handle"], n, door["hinge"]))
    up = np.array([0.0, 0.0, 1.0])
    face_off, door_t, past_handle, body_depth = dc.door_dims(door)
    g0, hinge, h = (np.asarray(door[k], float) for k in ("closed_grasp_pos", "hinge", "closed_handle"))
    face = g0 + into * face_off  # on the model's face, at the handle
    lo, hi = -past_handle, float(np.dot(hinge - face, hinge_side))
    if hinge_cut is not None:
        hi = min(hi, float(hinge_cut))
    z_lo, z_hi = door["door_z"][0] - dc.DOOR_Z_MARGIN, door["door_z"][1] + dc.DOOR_Z_MARGIN
    yaw = float(np.degrees(np.arctan2(into[1], into[0])))
    mid_u, mid_z = (lo + hi) / 2, (z_lo + z_hi) / 2

    def box(name, centre, dims):
        return {"name": name, "position": [round(float(v), 4) for v in centre], "rpy_deg": [0.0, 0.0, round(yaw, 3)],
                "dims": [round(float(v), 4) for v in dims]}

    def at(depth, u, z):
        p = face + into * depth + hinge_side * u
        p[2] = z
        return p

    face_d = GEN_FACE_DEEPER_M
    bar_back = float(np.dot(face - h, into)) - BAR_BACK_TO_FACE_M  # the bar's back, from the handle point
    half_len = F.MEASURED["handle_length_m"] / 2
    obs = [box("fridge_door", at(face_d + door_t / 2, mid_u, mid_z), [door_t, hi - lo, z_hi - z_lo]),
           box("fridge_body", at(face_d + door_t + body_depth / 2, mid_u, mid_z), [body_depth, hi - lo, z_hi - z_lo]),
           box("handle_bar", at(-BAR_BACK_TO_FACE_M - BAR_DEPTH_M / 2, 0.0, h[2]), [BAR_DEPTH_M, BAR_WIDTH_M, 2 * half_len])]
    gap = BAR_BACK_TO_FACE_M + face_d
    for sgn, nm in ((1, "handle_mount_top"), (-1, "handle_mount_bottom")):
        obs.append(box(nm, at(-BAR_BACK_TO_FACE_M + gap / 2, 0.0, h[2] + sgn * (half_len + BAR_MOUNT_H_M / 2)),
                       [gap, BAR_WIDTH_M, BAR_MOUNT_H_M]))
    del bar_back, up
    lines = ["# fridge side grasp (side_grasp.side_world): door, body, handle bar from the scene-camera detection",
             "base_frame: arm_mount", "obstacles:"]
    for o in obs:
        lines += [f"- name: {o['name']}", f"  position: {o['position']}", f"  rpy_deg: {o['rpy_deg']}",
                  f"  dims: {o['dims']}"]
    lines += ["objects: []", "targets: []", ""]
    text = "\n".join(lines)
    path = os.path.join(WORLDS_DIR, f"fridge_side-{hashlib.sha1(text.encode()).hexdigest()[:10]}.yaml")
    if not os.path.exists(path):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    return path, obs


GEN_START_CUT_MARGIN_M = 0.05  # start_hinge_cut: this much short of where the arm is (cuRobo's spheres are fat)


def start_hinge_cut(scene, rb, q, door, n):
    """Where along the door (from the handle toward the hinge) the arm at joints `q` starts, counting only its parts
    within 5 cm of the door's face plane or past it, less GEN_START_CUT_MARGIN_M: the arm is really there, so the
    fridge (door + body, as modelled from the detection) cannot be. None when no part is that close to the face."""
    into, hinge_side = side_frame(n, hinge_sign_of(door["closed_handle"], n, door["hinge"]))
    face = np.asarray(door["closed_grasp_pos"], float) + into * dc.door_dims(door)[0]
    sim.clearance(rb, q, [])  # sets the arm, gripper open
    c, us = rb.physics_client_id, []
    for j in range(-1, pb.getNumJoints(rb.robot_id, physicsClientId=c)):
        lo, hi = pb.getAABB(rb.robot_id, j, physicsClientId=c)
        for corner in np.array(np.meshgrid(*zip(lo, hi))).T.reshape(-1, 3):
            if np.dot(corner - face, into) > -0.05:
                us.append(float(np.dot(corner - face, hinge_side)))
    return (min(us) - GEN_START_CUT_MARGIN_M) if us else None


GEN_RETREATS_M = (0.03, 0.05, 0.08)  # _retreat: how far straight out of the door the start may back off


def _retreat(scene, rb, q_start, into, curobo, goal):
    """For a start cuRobo calls a collision with the (deepened, inflated) door: back the tool straight out of the
    door (along -into, orientation kept) by each of GEN_RETREATS_M in turn, IK every GEN_SLIDE_STEP_M, and plan with
    cuRobo from there. Returns (retreat knots, cuRobo's (q, t, planning s, mismatch), distance) or None."""
    p0, q0r = sim.tool_pose(scene, rb, np.asarray(q_start, float))
    for d in GEN_RETREATS_M:
        k = int(np.ceil(d / GEN_SLIDE_STEP_M))
        knots, q, ok = [], np.asarray(q_start, float), True
        for i in range(1, k + 1):
            nq, err = _ik(scene, rb, p0 - into * d * i / k, q0r, q)
            if err > CURVE_IK_TOL_M or np.degrees(np.max(np.abs(nq - q))) > CURVE_MAX_JUMP_DEG:
                ok = False
                break
            knots.append(nq)
            q = nq
        if not ok:
            continue
        try:
            return knots, curobo.plan(knots[-1], goal), d
        except Refused:
            continue
    return None


def _gen_seeds(q):
    """IK seeds around joints `q`: J1 (GEN_J1_OFFSETS_DEG) x J3 (GEN_J3_OFFSETS_DEG) offsets, each also with the wrist
    flipped (J5 + 180, -J6, J7 + 180: the same hand on the other wrist branch)."""
    out = []
    for flip in (False, True):
        for d1 in GEN_J1_OFFSETS_DEG:
            for d3 in GEN_J3_OFFSETS_DEG:
                s = np.array(q, float)
                s[0] += np.radians(d1)
                s[2] += np.radians(d3)
                if flip:
                    s[4], s[5], s[6] = s[4] + np.pi, -s[5], s[6] + np.pi
                out.append(s)
    return out


def _ik_line(scene, rb, p0, p1, quat, q0, depth=0):
    """Joints along the straight piece p0 -> p1 (hand orientation `quat`) from joints q0 at p0, ending at p1: the
    piece's end IK'd from q0; where a joint moves more than CURVE_MAX_JUMP_DEG, the piece is halved and each half done
    the same way (up to GEN_SUBDIVIDE times), so a fast-turning stretch near the edge of reach gets more knots instead
    of being refused. None if the IK misses, or a joint still jumps at the finest split (an IK branch switch)."""
    q1, err = _ik(scene, rb, p1, quat, q0)
    if err > CURVE_IK_TOL_M:
        return None
    if np.degrees(np.max(np.abs(q1 - q0))) <= CURVE_MAX_JUMP_DEG:
        return [q1]
    if depth >= GEN_SUBDIVIDE:
        return None
    mid = (np.asarray(p0, float) + np.asarray(p1, float)) / 2
    a = _ik_line(scene, rb, p0, mid, quat, q0, depth + 1)
    if a is None:
        return None
    b = _ik_line(scene, rb, mid, p1, quat, a[-1], depth + 1)
    return None if b is None else a + b


def slide_configs(scene, rb, slide, quat, q_start, max_configs=None):
    """Every distinct arm configuration that does the whole straight slide (`slide`: tool points, start first) in hand
    orientation `quat`: IK'd from each _gen_seeds(q_start) at the slide's start, then knot by knot along it. Kept
    when every knot is within CURVE_IK_TOL_M, no joint moves more than CURVE_MAX_JUMP_DEG between knots, and every
    joint stays GEN_LIMIT_MARGIN_DEG inside its limits. Sorted: least reconfiguration from q_start (largest joint
    move), then the most limit margin. Returns (configs, how many seeds were tried)."""
    # nearest seeds first (smallest offsets, unflipped), and stop after GEN_MAX_CONFIGS: the ranking prefers the least
    # reconfiguration, which the near seeds find
    seeds = _gen_seeds(q_start)
    seeds.sort(key=lambda s_: float(np.max(np.abs(sim.short_delta(q_start, s_)))))
    found = []
    for seed in seeds:
        if len(found) >= (max_configs or GEN_MAX_CONFIGS):
            break
        q, err = _ik(scene, rb, slide[0], quat, seed)
        if err > CURVE_IK_TOL_M or any(np.degrees(np.max(np.abs(sim.short_delta(c["knots"][0], q)))) < GEN_SAME_CONFIG_DEG
                                       for c in found):
            continue
        knots, ok, prev_pt = [q], True, slide[0]
        for pt in slide[1:]:
            seg = _ik_line(scene, rb, prev_pt, pt, quat, knots[-1])  # subdivides where the joints move a lot
            if seg is None:
                ok = False
                break
            knots.extend(seg)
            prev_pt = pt
        if not ok:
            continue
        margin, mj = _limit_margin_deg(rb, knots)
        if margin < GEN_LIMIT_MARGIN_DEG:
            continue
        # the configuration's joints near q_start's (J1/J3/J5/J7 spin freely: the short way round)
        q_near = np.asarray(q_start, float) + sim.short_delta(q_start, knots[0])
        knots = _unwrap(q_near, knots[1:])
        knots = [q_near, *knots]
        found.append({"knots": knots, "margin": (margin, mj),
                      "reconfig_deg": float(np.degrees(np.max(np.abs(knots[0] - np.asarray(q_start, float)))))})
    found.sort(key=lambda c: (round(c["reconfig_deg"], 0), -c["margin"][0]))
    return found, len(seeds)


def plan_side_general(scene, rb, h, orient, n, hinge, q_start, curobo, retime=None, tool_caps=True, after=None, prefer=None):
    """The side grasp from ANY start joints `q_start`: the straight slide along the door into the confirmed end pose
    (side_grasp_poses; CURVE_TIPS_STRAIGHT_M of fingertip travel before the bar), in the first configuration of
    slide_configs that cuRobo (`curobo`: a CuroboClient; the door, its body and the handle loaded as its world --
    replacing the planner's world) can reach from q_start; cuRobo's trajectory and the slide's knots joined into ONE
    route. `retime(path, dt, tool_fn=...) -> (t, q, qd)` (fast_timing.samples) times it for the checks: the timed
    route within GEN_JUNCTION_TOL_M of the slide's line over its last GEN_JUNCTION_LAST_M (where the back finger
    enters the gap behind the bar). Returns a dict (plan_side_curve's keys where they apply: chosen["route"] is the
    knots after q_start, for move_joints). Raises Refused, with every configuration's reason, when nothing works."""
    import time

    t_all = time.time()
    sgn = hinge_sign_of(h, n, hinge)  # which side of the handle the hinge is on: the slide comes in from there
    bar, grasp, _wp, quat, _turn = side_grasp_poses(h, orient, n, sgn)
    into, hinge_side = side_frame(n, sgn)
    back = slide_back(quat)  # the slide: grasp + back * w (SIDE_INSERT_DEG off the door normal)
    door = door_for(h, orient, n, hinge)
    # The slides length adapts to the arms reach (11-06: a door turned 27 deg, free edge nearer the arm, left only ~10 cm
    # of the slide line reachable in the grasp orientation, and a fixed 15 cm slide + 8 cm merge was refused although a
    # shorter one works): the longest slide <= the full one (CURVE_TIPS_STRAIGHT_M of fingertip travel before the bar)
    # whose whole merge + slide has an arm configuration, down to GEN_TIPS_MIN_M of fingertip travel
    w_full = _slide_from_grasp(grasp, bar, n, sgn, back=back)
    w_min = w_full - CURVE_TIPS_STRAIGHT_M + GEN_TIPS_MIN_M
    near = sorted(_gen_seeds(q_start), key=lambda s_: float(np.max(np.abs(sim.short_delta(q_start, s_)))))[:GEN_REACH_SEEDS]
    reach, d = 0.0, 0.0
    while d <= w_full + GEN_MERGE_M + 1e-9:  # how far along the slide line from the grasp the hand can be
        if not any(_ik(scene, rb, grasp + back * d, quat, s_)[1] <= CURVE_IK_TOL_M for s_ in near):
            break
        reach, d = d, d + GEN_REACH_STEP_M
    configs, n_seeds, tried = [], 0, []
    w_slide = min(w_full, max(reach, w_min))  # the reach estimate is a quick one: always try down to the shortest
    while w_slide >= w_min - 1e-9:
        merge_len = float(np.clip(reach - w_slide, GEN_MERGE_MIN_M, GEN_MERGE_M))  # the merge: whatever reach is left
        k = int(np.ceil(w_slide / GEN_SLIDE_STEP_M))
        slide = [grasp + back * w_slide * (1 - i / k) for i in range(k + 1)]
        # knots by the merge's real path (along the door AND out of it), not its along-door length alone: a short merge
        # still moves GEN_GOAL_OUT_M out of the door (11-06: 2 knots for a 1 cm merge = 8-9 deg joint jumps)
        km = max(4, int(np.ceil(np.hypot(merge_len, GEN_GOAL_OUT_M) / (GEN_SLIDE_STEP_M / 2))))
        merge = [grasp + back * (w_slide + merge_len * (1 - i / km)) - into * GEN_GOAL_OUT_M * (1 - _smoothstep(i / km))
                 for i in range(km)]
        configs, n_seeds = slide_configs(scene, rb, merge + slide, quat, q_start,
                                         max_configs=GEN_MAX_CONFIGS_AFTER if after is not None else None)
        tried.append(f"{w_slide * 100:.0f}+{merge_len * 100:.0f}")
        if configs:
            break
        w_slide -= GEN_REACH_STEP_M
    t_cfg = time.time() - t_all
    if not configs:
        raise Refused(f"side general: no arm configuration does a slide + merge to {np.round(grasp, 3)} (normal "
                      f"{np.round(n, 3)}): the slide line is reachable {reach * 100:.0f} cm from the grasp in the grasp "
                      f"orientation; tried slide+merge (cm) {', '.join(tried) or 'none'} (shortest allowed "
                      f"{w_min * 100:.0f} cm: {GEN_TIPS_MIN_M * 100:.0f} cm of fingertip travel before the bar), "
                      f"{n_seeds} IK seeds, {GEN_LIMIT_MARGIN_DEG:g} deg inside every joint limit")
    say(f"side general: slide {w_slide * 100:.0f} cm (full {w_full * 100:.0f}; the line reachable {reach * 100:.0f} cm from "
        f"the grasp), merge {merge_len * 100:.0f} cm")
    world, obstacles = side_world(door, h, n)
    curobo.set_world(world)
    why, chosen, cut, retreat = [], None, None, None
    why_after = []  # configurations what follows the grasp (`after`) doesn't work from, checked lazily below
    if prefer is not None:  # e.g. the least joint reversal on the way in first (11-06: a reversal = a visible stop)
        configs = sorted(configs, key=lambda c: (round(float(prefer(q_start, c)), 1), c.get("reconfig_deg", 0.0)))
        say("side general: configurations by preference: " + ", ".join(f"{prefer(q_start, c):.0f}" for c in configs))
    t_cu = time.time()
    u_needed = float(np.dot(merge[0] - np.asarray(h, float), hinge_side)) + 0.05  # the merge's start, + 5 cm
    i = 0
    while i < len(configs):
        c = configs[i]
        if after is not None and not c.get("_after_ok"):  # what follows the grasp (e.g. the door swing) must work from
            r = after(np.asarray(c["knots"][-1], float))  # this configuration's grasp joints: checked only when it is next
            if r is not None:
                why_after.append(f"config {np.round(np.degrees(c['knots'][-1]), 0).tolist()}: {r}")
                i += 1
                continue
            c["_after_ok"] = True
        try:
            cq, ct, cu_s, mism = curobo.plan(q_start, c["knots"][0])
        except Refused as e:
            if "START_STATE_WORLD_COLLISION" in str(e) and cut is None:
                # the arm is where it is: the modelled fridge can't reach past it toward the hinge. End the door
                # and body short of the arm (start_hinge_cut) and try again -- once, and only if that leaves the
                # door in front of the whole merge + slide
                cut = start_hinge_cut(scene, rb, q_start, door, n)
                if cut is not None and cut >= u_needed:
                    say(f"side general: the start collides with the modelled fridge in cuRobo; the door/body end "
                        f"{cut * 100:.0f} cm from the handle toward the hinge (where the arm starts) instead of at "
                        "the hinge")
                    world, obstacles = side_world(door, h, n, hinge_cut=cut)
                    curobo.set_world(world)
                    continue
                cut = False  # the arm starts in front of the door: back straight out of it first (below)
            if "START_STATE_WORLD_COLLISION" in str(e) and retreat is None:
                retreat = _retreat(scene, rb, q_start, into, curobo, c["knots"][0])
                if retreat is not None:
                    cq, ct, cu_s, mism = retreat[1]
                    say(f"side general: the start is within cuRobo's margin of the modelled door: backing "
                        f"{retreat[2] * 100:.0f} cm straight out of it first, in the same route")
                    chosen = (c, cq, ct, cu_s, mism)
                    break
                raise Refused(f"side general: cuRobo says the START joints collide with this detection's fridge, "
                              f"and backing straight out up to {GEN_RETREATS_M[-1] * 100:.0f} cm doesn't clear it: {e}")
            why.append(f"config {np.round(np.degrees(c['knots'][0]), 0).tolist()} ({c['reconfig_deg']:.0f} deg "
                       f"reconfiguration): {e}")
            if "START_STATE" in str(e):  # the start itself: the same for every configuration
                raise Refused(f"side general: cuRobo refuses the START joints with this detection's fridge: {e}")
            i += 1
            continue
        chosen = (c, cq, ct, cu_s, mism)
        break
    t_cu = time.time() - t_cu
    if chosen is None:
        raise Refused(f"side general: no slide configuration both works for what follows the grasp and has a cuRobo "
                      f"path ({len(configs)} tried) -- " + " | ".join(why_after + why))
    if why_after:
        say(f"side general: {len(why_after)} grasp configuration(s) skipped: what follows the grasp doesn't work from them")
    c, cq, ct, cu_s, mism = chosen
    # one route: the retreat's knots (if any), cuRobo's trajectory thinned to knots, then the merge's and the
    # slide's knots (cuRobo ends on the merge's start)
    pre = retreat[0] if retreat is not None else []
    cq = _unwrap(q_start, [*pre, *cq[1:]])
    knots, last = [], np.asarray(q_start, float)
    for i, q in enumerate(cq):
        if i == len(cq) - 1 or np.degrees(np.max(np.abs(q - last))) >= GEN_KNOT_MIN_DEG:
            knots.append(q)
            last = q
    n_free = len(knots)
    slide_knots = _unwrap(knots[-1] if knots else q_start, c["knots"])
    route = knots + slide_knots[1:]
    seg = ["free"] * n_free + ["merge"] * (km - 1) + ["slide"] * (len(slide_knots) - km)
    out = {
        "handle": [float(v) for v in h], "normal": [float(v) for v in n], "hinge": [float(v) for v in hinge],
        "bar": [float(v) for v in bar], "grasp": [float(v) for v in grasp], "quat": [float(v) for v in quat],
        "waypoint": [float(v) for v in slide[0]], "curobo_goal": [float(v) for v in merge[0]], "slide_m": w_slide, "tips_straight_m": CURVE_TIPS_STRAIGHT_M,
        "q0": [float(v) for v in q_start], "door": door, "world": world, "world_obstacles": obstacles, "hinge_sign": sgn,
        "world_hinge_cut_m": cut,
        "chosen": {"route": [[float(v) for v in q] for q in route], "route_seg": seg, "n_free": n_free,
                   "slide_config": [float(v) for v in c["knots"][0]],
                   "retreat_m": retreat[2] if retreat is not None else 0.0, "reconfig_deg": c["reconfig_deg"],
                   "limit_margin_deg": c["margin"][0], "limit_margin_joint": c["margin"][1],
                   "curobo_points": int(len(cq) + 1), "curobo_planning_s": cu_s, "curobo_mismatch_rad": mism,
                   "curobo_duration_s": float(ct[-1]) if len(ct) else 0.0, "configs": len(configs),
                   "configs_refused_by_curobo": why},
        "times": {"configs_s": t_cfg, "curobo_s": t_cu},
    }
    if retime is not None:
        tool_fn = (lambda q: sim.tool_pose(scene, rb, q)[0]) if tool_caps else None
        # the timing's shape smoothing leaves the last GEN_PROTECT_M of the route exact (the slide into the handle),
        # and the speed only falls over the merge + slide (+3 knots): one slow-down into the handle
        d_end = [float(np.linalg.norm(sim.tool_pose(scene, rb, q)[0] - grasp)) for q in route]
        targs = {"protect_tail": int(sum(d <= GEN_PROTECT_M for d in d_end)),
                 "monotone_tail": int(sum(x != "free" for x in seg) + 3)}
        out["chosen"]["timing_args"] = targs
        t, qs, qd = retime(np.array([np.asarray(q_start, float), *route]), rammp_arm.DT, tool_fn=tool_fn, **targs)
        out["_timed"] = (t, qs, qd)  # numpy, for checks in this process (not JSON)
        pos = np.array([sim.tool_pose(scene, rb, q)[0] for q in qs])
        w = (pos - grasp) @ back
        dev = np.linalg.norm((pos - grasp) - np.outer(w, back), axis=1)  # off the slide's line
        # the route's final stretch along the slide: from the last time it was more than the slide's length out
        start = int(np.flatnonzero(w > w_slide + 0.005)[-1]) + 1 if np.any(w > w_slide + 0.005) else 0
        last = slice(start, None)
        near = [i for i in range(start, len(w)) if w[i] <= GEN_JUNCTION_LAST_M]
        worst_near = float(dev[near].max()) if near else float("nan")
        out["chosen"]["timing"] = {"duration_s": float(t[-1]), "peak_joint_speed": float(np.max(np.abs(qd))),
                                   "peak_tool_speed": float(np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)
                                                                   / np.diff(t))),
                                   "slide_dev_max_m": float(dev[last].max()), "last_10cm_dev_max_m": worst_near}
        if not near or worst_near > GEN_JUNCTION_TOL_M:
            raise Refused(f"side general: the timed route strays {worst_near * 1000:.1f} mm from the slide's line in "
                          f"its last {GEN_JUNCTION_LAST_M * 100:.0f} cm (allowed {GEN_JUNCTION_TOL_M * 1000:.0f} mm)")
    # the free part against the full door model + bar (cuRobo saw a deeper face, maybe a door cut short at the start):
    # nothing may go further into the model than the start already is (5 mm slack)
    door_b = dc.DoorFrame(door, door["open_sign"]).bodies(scene, rb, 0.0)
    try:
        start_cl = float(dc.clearance(rb, np.asarray(q_start, float), door_b)[0])
    finally:
        drop(rb, door_b)
    cl = free_clearance(scene, rb, out)
    out["chosen"]["clearance"] = {k: [float(v[0]), v[1]] for k, v in cl.items()}
    out["chosen"]["start_clear_m"] = start_cl
    worst = min(cl.get("free", (9.0, "")), cl.get("free back finger", (9.0, "")))
    if worst[0] < min(start_cl, 0.0) - 0.005:
        raise Refused(f"side general: cuRobo's path goes {-worst[0] * 100:.1f} cm into the modelled door ({worst[1]}; "
                      f"the start is {start_cl * 100:+.1f} cm from it)")
    out["times"]["total_s"] = time.time() - t_all
    return out


def free_clearance(scene, rb, plan):
    """Sim check of a plan_side_general plan: the smallest distance of any link to the door model + bar (add_bar_model)
    over its cuRobo knots and over its slide knots, the back finger apart: {segment: (m, link)}."""
    door, n = plan["door"], np.asarray(plan["normal"], float)
    bodies = dc.DoorFrame(door, door["open_sign"]).bodies(scene, rb, 0.0) + add_bar_model(scene, rb, door, plan["handle"], n)
    into, _ = side_frame(n)
    try:
        route = [np.asarray(q, float) for q in plan["chosen"]["route"]]
        back = _back_finger_links(scene, rb, route[-1], into)
        res = {}
        for q, s in zip(route, plan["chosen"]["route_seg"]):
            cl = link_clearances(rb, q, bodies)
            for part, pick in (("", lambda ln: ln not in back), (" back finger", lambda ln: ln in back)):
                m = min(((d, f"{ln} vs {b}") for ln, (d, b) in cl.items() if pick(ln)), default=(9.0, ""))
                key = s + part
                res[key] = min(res.get(key, (9.0, "")), m)
        return res
    finally:
        drop(rb, bodies)
