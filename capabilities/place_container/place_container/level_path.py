"""Level paths for the place: the container carried from wherever the arm holds it to the spot and down,
keeping the tilt it has at the start, in the start's arm posture, the elbow kept clear of the chair and its user.

The start is the reference: however the start holds the container is "level", and the container's up
direction at the start (in the tool frame) must stay pointing up at every sample (tilt_from). The hand
only turns about vertical.

Each path is a straight tool line (positions linear, orientation slerped) cut into knots. The arm has
7 joints for a 6-number tool pose, so at every knot there is a one-parameter family of arm shapes with
the same tool pose (the elbow swinging about the shoulder-wrist line). A beam search over that family
(the null space of the tool Jacobian, NULL_STEPS each way, re-solved exactly by IK) picks, knot after
knot, the shapes furthest from the chair and user boxes (Chair, the boxes cuRobo plans around, padded
as it pads them) with the smallest joint steps, continued from the start's own joints: the start's
posture is kept, never flipped. The chosen knots are timed as the driver will run them
(rammp_arm.joint_trajectory, minimum jerk through the knots, timed in metres of tool travel) and EVERY
timed sample is checked (check_samples): container tilt, clearance from the boxes, the container's
height over the table, joint limits, joint jumps. What runs is exactly what was checked.
"""

from pathlib import Path

import numpy as np
import pybullet as p
import rammp_arm
import yaml
from capability_engine import sim
from scipy.spatial.transform import Rotation, Slerp

from . import plan
from .plan import Refused, wrap, wrap_deg

CHAIR_DIR = Path(__file__).parent / "chair"  # the robot's planner layers, copied (chair/SOURCE)
CHAIR_LAYERS = ("platform", "seat", "person")  # as the planner composes them (its floor is not a chair part)
CLEAR_MIN_M = 0.015  # the model's arm (meshes) stays this far outside the padded boxes: cuRobo's arm is
# padded spheres, fatter (door_common, 11-10: sim-clear line-ups were 1-9 mm into cuRobo's person/lap)
CLEAR_WANT_M = 0.06  # more clearance than this earns the search nothing
CARRY_KNOT_M = 0.02  # tool travel between the carry's knots
LINE_KNOT_M = 0.01  # ... and the line down's (densified to plan.PATH_STEP for the lowering)
NULL_STEPS = np.radians([-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0])  # elbow swings tried per knot (the carry)
LINE_NULL_STEPS = np.radians([-10.0, -4.0, 0.0, 4.0, 10.0])  # ... on the line down (it only continues)
BEAM = 3  # arm-shape paths kept between knots
MAX_KNOT_STEP_DEG = 12.0  # no joint turns more than this between two knots
MAX_TILT_DEG = 2.0  # the container's tilt from how the start holds it, at every sample
MIN_BOTTOM_ABOVE_TABLE = 0.10  # m: the container's bottom over the table on the carry
CLEARANCE_WEIGHT = 200.0  # search cost per m of clearance short of CLEAR_WANT_M (vs 1 per rad^2 of joint step)


class Chair:
    """The chair and user boxes in the model's PyBullet world, padded as cuRobo pads them
    (padding_assumed_m per side); clearance() of the arm's links (not its base and shoulder, which never
    move and were carved around cuRobo's spheres of them, not these meshes)."""

    def __init__(self, model):
        self.model = model
        self.client = model.rb.physics_client_id
        base = sim.make_sim()[0].robot_base_pose
        self.boxes = []  # (body, name)
        for layer in CHAIR_LAYERS:
            data = yaml.safe_load((CHAIR_DIR / f"layer_mebot_{layer}.yaml").read_text())
            pad = float(data["meta"]["padding_assumed_m"])
            for o in data["obstacles"]:
                half = [float(d) / 2 + pad for d in o["dims"]]
                at = plan_pose_in_world(base, o["position"])
                cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
                body = p.createMultiBody(0, cs, basePosition=at[0], baseOrientation=at[1], physicsClientId=self.client)
                self.boxes.append((body, f"{layer}/{o['name']}"))
        if not self.boxes:
            raise Refused(f"no chair boxes in {CHAIR_DIR}: without them nothing is checked against the chair")
        self.fixed = {-1, *range(sim.ARM[0] + 1)}  # the root, the links before joint 1, and the shoulder
        # the held container: a cylinder (lip outer radius, BOWL_DEPTH tall) placed by container_pose()
        cs = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=plan.BOWL_RADIUS + plan.LIP_WIDTH,
            height=plan.BOWL_DEPTH,
            physicsClientId=self.client,
        )
        self.container = p.createMultiBody(0, cs, basePosition=[0, 0, -10], physicsClientId=self.client)
        self.up_tool = None  # set_container(): the container's up in the tool frame (None: no container checked)
        self.base = base

    def set_container(self, up_tool):
        """Check the held container too, its axis `up_tool` (the tool frame), as the start holds it."""
        self.up_tool = None if up_tool is None else np.asarray(up_tool, dtype=float)

    def clearance(self, q):
        """(m, part, box): the smallest distance of the arm's links -- and of the held container, once
        set_container() -- to a padded box at joints q (negative: inside)."""
        rb = self.model.rb
        for j, jj in enumerate(sim.ARM):
            p.resetJointState(rb.robot_id, jj, float(q[j]), physicsClientId=self.client)
        best = (np.inf, "", "")
        for body, name in self.boxes:
            for pt in p.getClosestPoints(rb.robot_id, body, CLEAR_WANT_M + 0.05, physicsClientId=self.client):
                if pt[3] not in self.fixed and pt[8] < best[0]:
                    link = p.getJointInfo(rb.robot_id, pt[3], physicsClientId=self.client)[12].decode()
                    best = (float(pt[8]), link, name)
        if self.up_tool is not None:
            centre, axis = container_pose(self.model, q, self.up_tool)
            quat = z_to(axis)
            at = plan_pose_in_world(self.base, centre)[0]
            world_q = p.multiplyTransforms([0, 0, 0], self.base.orientation, [0, 0, 0], list(quat))[1]
            p.resetBasePositionAndOrientation(self.container, at, world_q, physicsClientId=self.client)
            for body, name in self.boxes:
                for pt in p.getClosestPoints(self.container, body, CLEAR_WANT_M + 0.05, physicsClientId=self.client):
                    if pt[8] < best[0]:
                        best = (float(pt[8]), "the container", name)
        return best


def plan_pose_in_world(base, position):
    """A box centre given in arm_mount, as (position, quaternion) in the PyBullet world (the robot's base pose)."""
    pos, quat = p.multiplyTransforms(base.position, base.orientation, list(map(float, position)), [0, 0, 0, 1])
    return list(pos), list(quat)


def container_pose(model, q, up_tool):
    """(centre, axis) of the held container at joints q, in arm_mount: the fingertips pinch the lip, the
    centre BOWL_RADIUS on along the hand's horizontal heading, half BOWL_DEPTH down its axis (up_tool)."""
    T = model.fk(q, "tip")
    axis = T[:3, :3] @ up_tool
    h = T[:3, 2] - (T[:3, 2] @ axis) * axis
    h = h / max(np.linalg.norm(h), 1e-6)
    return T[:3, 3] + plan.BOWL_RADIUS * h - 0.5 * plan.BOWL_DEPTH * axis, axis


def z_to(axis):
    """The quaternion (xyzw) turning +z onto unit `axis` (the shortest turn)."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c = np.cross([0.0, 0.0, 1.0], axis)
    s = float(np.linalg.norm(c))
    if s < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0]) if axis[2] > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    return Rotation.from_rotvec(c / s * np.arctan2(s, axis[2])).as_quat()


def bottom_z(model, q):
    """The container's bottom height at joints q: the fingertips (the lip) less BOWL_DEPTH."""
    return float(model.fk(q, "tip")[2, 3]) - plan.BOWL_DEPTH


def tip_drop(model, q):
    """How far the fingertips (the lip) are below the tool frame at joints q (the hand's pitch)."""
    return float(model.fk(q)[2, 3] - model.fk(q, "tip")[2, 3])


def up_in_tool(R0):
    """The world's up at the start, in the tool frame: the container's up direction, as the start holds it."""
    return R0.T @ np.array([0.0, 0.0, 1.0])


def tilt_from(R, up_tool):
    """How far (deg) the container's up direction (`up_tool`, in the tool frame) is from vertical at tool rotation R."""
    return float(np.degrees(np.arccos(np.clip((R @ up_tool)[2], -1.0, 1.0))))


def turned_to(R0, heading):
    """R0 turned about vertical so the tool's horizontal heading points along `heading` (the tilt kept)."""
    h0 = R0[:, 2].copy()
    h0[2] = 0.0
    if np.linalg.norm(h0) < 0.3:
        raise Refused("the hand points too steeply up or down to tell which way it faces")
    h0 /= np.linalg.norm(h0)
    yaw = np.arctan2(h0[0] * heading[1] - h0[1] * heading[0], h0 @ heading)
    return (Rotation.from_rotvec([0.0, 0.0, yaw]) * Rotation.from_matrix(R0)).as_matrix()


def null_direction(model, q, eps=1e-4):
    """A unit joint direction that leaves the tool pose unchanged (the elbow's swing), at joints q."""
    T0 = model.fk(q)
    J = np.zeros((6, 7))
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        T = model.fk(q + dq)
        J[:3, i] = (T[:3, 3] - T0[:3, 3]) / eps
        J[3:, i] = Rotation.from_matrix(T[:3, :3] @ T0[:3, :3].T).as_rotvec() / eps
    _, _, vt = np.linalg.svd(J)
    n = vt[-1]
    return n / np.linalg.norm(n)


def knot_poses(p_from, R_from, p_to, R_to, step):
    """Tool poses along the straight line p_from -> p_to, orientation slerped, about `step` apart (not the start)."""
    n = max(1, int(np.ceil(np.linalg.norm(np.asarray(p_to) - p_from) / step)))
    sl = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R_from, R_to])))
    return [(p_from + s * (np.asarray(p_to) - p_from), sl([s])[0].as_quat()) for s in np.linspace(0, 1, n + 1)[1:]]


def search_line(model, chair, q_from, p_to, R_to, step, label, null_steps=NULL_STEPS):
    """Joint knots along the straight tool line from q_from's pose to (p_to, R_to): at each knot, the arm
    shapes reached by swinging the elbow (NULL_STEPS along null_direction) from each kept path's last
    knot, re-solved exactly by IK; the BEAM paths with the least cost (joint steps + clearance short of
    CLEAR_WANT_M) kept. Shapes inside CLEAR_MIN_M or more than MAX_KNOT_STEP_DEG from the knot before are
    dropped. [q, ...] (not q_from); Refused when no shape continues the line."""
    T = model.fk(q_from)
    beam = [(0.0, [np.asarray(q_from, dtype=float)], np.inf)]  # (cost, knots, worst clearance)
    poses = knot_poses(T[:3, 3], T[:3, :3], p_to, R_to, step)
    for k, (pos, quat) in enumerate(poses):
        nxt = []
        for cost, path, worst in beam:
            q_prev = path[-1]
            n = null_direction(model, q_prev)
            seen = []
            for a in null_steps:
                q = model.ik(pos, quat, q_prev + a * n, restarts=0)
                if q is None:
                    continue
                (q,) = rammp_arm._unwrap_from(q_prev, [q])
                if np.max(np.abs(np.degrees(q - q_prev))) > MAX_KNOT_STEP_DEG:
                    continue
                if any(np.max(np.abs(q - s)) < np.radians(0.5) for s in seen):
                    continue
                seen.append(q)
                c, _, _ = chair.clearance(q)
                if c < CLEAR_MIN_M:
                    continue
                step_cost = float(np.sum((q - q_prev) ** 2))
                clear_cost = CLEARANCE_WEIGHT * max(0.0, CLEAR_WANT_M - c) * step
                nxt.append((cost + step_cost + clear_cost, path + [q], min(worst, c)))
        if not nxt:
            c, link, box = chair.clearance(beam[0][1][-1])
            raise Refused(
                f"{label}: no arm shape continues the line at {k}/{len(poses)} (tool at {np.round(pos, 3).tolist()}): "
                f"every elbow position is within {CLEAR_MIN_M * 100:.1f} cm of the chair and user boxes or needs a "
                f"joint jump (last kept: {link} {c * 100:.1f} cm from {box}) -- nothing moved"
            )
        nxt.sort(key=lambda e: e[0])
        beam = nxt[:BEAM]
    return beam[0][1][1:]


def timed(q_from, knots, model, speed_scale):
    """(t, q, qd) as the driver will run the knots from q_from: minimum jerk, timed in metres of tool travel."""
    path = [np.asarray(q_from, dtype=float), *rammp_arm._unwrap_from(q_from, knots)]
    tools = [model.fk(q)[:3, 3] for q in path]
    lengths = []
    for k in range(len(path) - 1):
        a, b, qa, qb = tools[k], tools[k + 1], path[k], path[k + 1]
        # a segment's time: its tool travel at CART_SPEED, but no less than its biggest joint step at
        # JOINT_SPEED (the elbow's swing is in the knots, not in the tool's travel)
        joint = float(np.max(np.abs(qb - qa))) * rammp_arm.CART_SPEED / rammp_arm.JOINT_SPEED
        lengths.append(max(float(np.linalg.norm(b - a)), joint, 1e-4))
    samples = rammp_arm.joint_trajectory(path[0], path[1:], lengths, speed_scale)
    return samples, lengths


def check_samples(
    model, chair, samples, up_tool, table_z, label, speed_scale, min_bottom=MIN_BOTTOM_ABOVE_TABLE, vertical=False
):
    """Every timed sample (t, q, qd): the container within MAX_TILT_DEG of how the start holds it, the arm
    and the container CLEAR_MIN_M outside the padded boxes, the container's bottom `min_bottom` over the
    table (None: not checked), joint limits, no joint faster than JOINT_SPEED x speed_scale (+25%), no
    jump between samples, and (vertical) the tool on the vertical line through the first sample's (3 mm).
    Refused naming the first failure; else a summary dict."""
    _, q_samples, qd_samples = samples
    worst_tilt, worst_clear, low = 0.0, (np.inf, "", ""), np.inf
    vmax = 1.25 * rammp_arm.JOINT_SPEED * speed_scale
    x0 = model.fk(q_samples[0])[:2, 3]
    for i, q in enumerate(q_samples):
        if i == 0:
            continue  # the start itself is taken as it is: only where the path goes is checked
        T = model.fk(q)
        where = f"{label} at {i / max(len(q_samples) - 1, 1) * 100:.0f}% (tool {np.round(T[:3, 3], 3).tolist()})"
        a = tilt_from(T[:3, :3], up_tool)
        worst_tilt = max(worst_tilt, a)
        if a > MAX_TILT_DEG:
            raise Refused(f"{where}: the container would tilt {a:.1f} deg (max {MAX_TILT_DEG:g}) -- nothing moved")
        v = float(np.max(np.abs(qd_samples[i])))
        if v > vmax:
            raise Refused(
                f"{where}: a joint at {np.degrees(v):.0f} deg/s (max {np.degrees(vmax):.0f}) -- nothing moved"
            )
        if vertical and np.linalg.norm(T[:2, 3] - x0) > 0.003:
            raise Refused(f"{where}: leaves the vertical line by {np.linalg.norm(T[:2, 3] - x0) * 100:.1f} cm")
        c = chair.clearance(q)
        worst_clear = min(worst_clear, c)
        if c[0] < CLEAR_MIN_M:
            raise Refused(
                f"{where}: {c[1]} {c[0] * 100:.1f} cm from {c[2]} (min {CLEAR_MIN_M * 100:.1f} cm outside the padded "
                "chair and user boxes) -- nothing moved"
            )
        bottom = bottom_z(model, q) - table_z
        low = min(low, bottom)
        if min_bottom is not None and bottom < min_bottom - 1e-3:
            raise Refused(
                f"{where}: the container's bottom {bottom * 100:.1f} cm over the table (min {min_bottom * 100:.0f})"
            )
        bad = model.limit_violations(q)
        if bad:
            raise Refused(f"{where}: past a joint limit: {', '.join(bad)} -- nothing moved")
        if i and np.max(np.abs(np.degrees(np.asarray(q) - q_samples[i - 1]))) > 5.0:
            raise Refused(f"{where}: a joint jumps between two samples -- nothing moved")
    return dict(
        tilt_deg=worst_tilt, clear_m=worst_clear[0], clear_link=worst_clear[1], clear_box=worst_clear[2], bottom_m=low
    )


def carry(model, chair, q_from, spot, table_z, heading, step=CARRY_KNOT_M):
    """The carry: from q_from's tool pose straight to above the spot (container centre over it, the hand
    turned about vertical to `heading`), at the LOWEST height that plans, from the container's bottom
    MIN_BOTTOM_ABOVE_TABLE over the table up to the start's height (the line down after it is then the
    shortest). (tool height over the table, knots); Refused when no height plans."""
    T0 = model.fk(q_from)
    R_to = turned_to(T0[:3, :3], heading)
    tip = R_to @ np.array([0.0, 0.0, model.flange_to_tip - rammp_arm.TOOL_Z])  # tool frame -> fingertips
    centre = tip + plan.BOWL_RADIUS * np.asarray(heading)  # tool frame -> container centre (at the lip)
    low = MIN_BOTTOM_ABOVE_TABLE + plan.BOWL_DEPTH - tip[2]  # tool height over the table: bottom 10 cm up
    high = max(low, T0[2, 3] - table_z)
    why = []
    for h in sorted({round(h, 3) for h in np.linspace(low, high, 4)}):
        p_to = np.array([spot[0] - centre[0], spot[1] - centre[1], table_z + h])
        try:
            return h, search_line(model, chair, q_from, p_to, R_to, step, f"carry at {h * 100:.0f} cm")
        except Refused as e:
            why.append(str(e))
    raise Refused("no carry height plans: " + " | ".join(why))


def descent_line(model, chair, q_top, floor_z):
    """The straight line down from q_top's tool pose to floor_z (orientation kept), searched like the carry
    (LINE_KNOT_M knots, elbow kept clear), then densified to plan.PATH_STEP by joint interpolation for the
    lowering: (tool z values, joints) as plan.plan_descent returns them; the first entry is q_top."""
    T = model.fk(q_top)
    knots = search_line(
        model, chair, q_top, [T[0, 3], T[1, 3], floor_z], T[:3, :3], LINE_KNOT_M, "line down", LINE_NULL_STEPS
    )
    path = [np.asarray(q_top, dtype=float), *knots]
    qs = [path[0]]
    for a, b in zip(path[:-1], path[1:], strict=True):
        dz = abs(model.fk(a)[2, 3] - model.fk(b)[2, 3])
        n = max(1, int(round(dz / plan.PATH_STEP)))
        qs += [a + (b - a) * s for s in np.linspace(0, 1, n + 1)[1:]]
    qs = np.array(qs)
    zs = np.array([model.fk(q)[2, 3] for q in qs])
    for q in qs:  # joint interpolation between 1 cm knots: still the straight line
        Tq = model.fk(q)
        if np.linalg.norm(Tq[:2, 3] - T[:2, 3]) > 0.003:
            raise Refused(f"line down: leaves the vertical by {np.linalg.norm(Tq[:2, 3] - T[:2, 3]) * 100:.1f} cm")
    return zs, qs


def wrap_continuous(q):
    """Continuous joints into (-pi, pi], as the driver reports them."""
    q = np.asarray(q, dtype=float).copy()
    q[rammp_arm.CONTINUOUS] = wrap(q[rammp_arm.CONTINUOUS])
    return q


__all__ = [
    "Chair",
    "carry",
    "check_samples",
    "descent_line",
    "search_line",
    "tilt_from",
    "timed",
    "turned_to",
    "up_in_tool",
    "wrap_continuous",
    "wrap_deg",
]
