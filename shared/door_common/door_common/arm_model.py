"""The arm as the door code checks it, in numpy: no PyBullet.

The model is the driver's own URDF (/arm/robot_description, or a file offline), read by kinematics.Chain from
arm_mount (the planning frame, rammp_arm.FRAME) to arm_tool_frame (cuRobo's ee_link, self.arm's tool frame) -- the
frames every door pose is in. It gives what the door code asked of the PyBullet copy before:

* `pose(q)`: forward kinematics, the tool's (position, quaternion xyzw).
* `ik(pos, quat, seed)`: local IK from a nearby configuration (damped least squares on the exact Jacobian, the hard
  joint limits held), on the seed's branch: (joints, error in m). The error counts orientation too
  (ROT_ERR_EQUIV_M per radian), so a solution that reaches the point with the hand turned wrong fails the IK gates.
* `clearance(q, bodies)`: the smallest distance from the arm to box obstacles (`Box`, yaw only: the door slab, the
  appliance's body, the handle), as the fridge's close checks it: each link from J1 to J7 a capsule ARM_RADIUS_M
  thick, the 2F-85 as its measured box (HAND_BOX_TOOL_M, the fridge's), a held container as a sphere. Coarser than
  PyBullet's meshes (the capsules are fatter than the links), so a margin tuned on the meshes is met with room to
  spare or refused -- never passed closer than it was.

The chair and the user are not here: cuRobo checks every motion against them (its world's chair layers), the moves
run exactly as planned here through a one-leg blend (common.curobo_validates), as the fridge's are.
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
from capability_engine import sim
from scipy.spatial.transform import Rotation as R

from .kinematics import JOINTS, ROOT, Chain

# --- IK ---
IK_ITERS = 200  # iterations per solve (PyBullet's was 200 too)
IK_TOL_M, IK_TOL_RAD = 1e-4, 1e-3  # converged
IK_DAMP = 0.02  # damped least squares
IK_MAX_STEP_RAD = 0.1  # per iteration, per joint
ROT_ERR_EQUIV_M = 0.1  # ik's error: position error + this x orientation error (rad) -- 1 deg off counts 1.7 mm

# --- the arm's shape, for clearance ---
ARM_RADIUS_M = 0.06  # each link J1 -> J7 a capsule this thick (the fridge's CLOSE_ARM_RADIUS_M): the Gen3's
# actuators are ~9-10 cm across, so the capsule is the link plus 1-1.5 cm
ARM_SAMPLE_M = 0.01  # a capsule's axis sampled this often
# the real 2F-85 (open) in the tool frame, from its collision meshes in the driver's URDF (the fridge's, measured
# 11-06): fingertips +-8.0 cm across (tool x), the body +-3.8 cm thick (tool y; the bracelet +-4.2 cm), the bracelet's
# back 21 cm behind the tool frame. Open or closed alike: the open hand is the wider one
HAND_BOX_TOOL_M = ((-0.080, 0.080), (-0.042, 0.042), (-0.21, 0.045))
HAND_SAMPLE_M = 0.01  # the hand box's surface sampled this often (a gap is exact to within half of it)


class Box(NamedTuple):
    """An obstacle box in the arm's frame: centre (xyz), half extents, yaw about z (rad)."""

    centre: tuple
    half: tuple
    yaw: float


class Sphere(NamedTuple):
    """Something held in the gripper, as a sphere: its radius and its centre in the tool frame."""

    radius: float
    centre_tool: tuple


def box_gaps(points, box):
    """Signed distance (m, negative inside) from each of `points` (..., 3) to `box`."""
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    d = np.asarray(points, float) - np.asarray(box.centre, float)
    local = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], axis=-1)
    q = np.abs(local) - np.asarray(box.half, float)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def _hand_grid(box, step):
    """Points on the surface of box ((x0, x1), (y0, y1), (z0, z1)), corners included, at most `step` apart, in its own
    frame."""
    axes = [np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / step)) + 1)) for lo, hi in box]
    pts = []
    for k in range(3):
        for side in box[k]:
            g = np.meshgrid(*[axes[i] if i != k else np.array([side]) for i in range(3)], indexing="ij")
            pts.append(np.stack([x.ravel() for x in g], axis=1))
    return np.vstack(pts)


def _overlap(c1, r1, h1, box):
    """How deep (m) the box at centre `c1`, rotation matrix `r1` (columns: its axes), half extents `h1` overlaps `box`:
    the least overlap over the 15 separating axes (> 0: they intersect; <= 0: apart)."""
    a2 = R.from_euler("z", box.yaw).as_matrix()
    h2 = np.asarray(box.half, float)
    d = np.asarray(box.centre, float) - np.asarray(c1, float)
    axes = [r1[:, i] for i in range(3)] + [a2[:, i] for i in range(3)]
    axes += [np.cross(r1[:, i], a2[:, j]) for i in range(3) for j in range(3)]
    best = np.inf
    for ax in axes:
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        ax = ax / n
        reach = np.sum(h1 * np.abs(r1.T @ ax)) + np.sum(h2 * np.abs(a2.T @ ax))
        best = min(best, reach - abs(float(d @ ax)))
        if best <= 0:
            return best
    return best


def _rodrigues(axis, angle):
    """The rotation matrix turning `angle` (rad) about unit `axis`."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rotvec(m):
    """The rotation vector of rotation matrix `m` (the log map); scipy's near a half turn, where it is ill-posed."""
    c = (np.trace(m) - 1.0) / 2.0
    if c < -0.99:
        return R.from_matrix(m).as_rotvec()
    angle = np.arccos(min(1.0, c))
    w = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / 2.0
    return w if angle < 1e-9 else w * (angle / np.sin(angle))


class ArmModel:
    """The arm's kinematics and shape from its URDF (see the module's docstring)."""

    def __init__(self, urdf_xml, **chain_kw):
        self.chain = Chain(urdf_xml, **chain_kw)  # chain_kw: root / tip / joints, for a model named otherwise
        self._fingers = self._finger_chains(urdf_xml, chain_kw.get("root", ROOT), chain_kw.get("joints", JOINTS))
        self.lo = np.where(self.chain.limited, self.chain.lower, -np.inf)
        self.hi = np.where(self.chain.limited, self.chain.upper, np.inf)
        # the chain as numpy steps for the IK's one-pass FK + Jacobian (no scipy in the loop)
        self._steps = [(T[:3, :3].copy(), T[:3, 3].copy(), axis, k) for T, axis, k in self.chain.steps]
        self._hand = _hand_grid(HAND_BOX_TOOL_M, HAND_SAMPLE_M)
        self._hand_c = np.array([(lo + hi) / 2 for lo, hi in HAND_BOX_TOOL_M])
        self._hand_h = np.array([(hi - lo) / 2 for lo, hi in HAND_BOX_TOOL_M])

    @staticmethod
    def _finger_chains(urdf_xml, root, joints):
        """A chain to each link named *finger* (the 2F-85's), over the arm's joints and the gripper's (held open, at
        0). None when the model has none."""
        robot = ET.fromstring(urdf_xml)
        names = [lk.get("name") for lk in robot.findall("link") if "finger" in lk.get("name", "")]
        extra = [
            j.get("name")
            for j in robot.findall("joint")
            if j.get("type") in ("revolute", "continuous") and j.get("name") not in joints
        ]
        chains = []
        for name in names:
            try:
                chains.append(Chain(urdf_xml, root=root, tip=name, joints=list(joints) + extra))
            except ValueError:  # not below the root
                continue
        return (chains, len(extra)) if chains else None

    def finger_points(self, q):
        """The finger links' origins (the gripper open) at arm joints `q`, in arm_mount: (N, 3). Without finger links in
        the model, the hand box's fingertip corners (further out: stricter)."""
        if self._fingers is None:
            pos, quat = self.pose(q)
            (x0, x1), (y0, y1), (_, z1) = HAND_BOX_TOOL_M
            tips = np.array([[x, y, z1] for x in (x0, x1) for y in (y0, y1)])
            return pos + R.from_quat(quat).apply(tips)
        chains, n_extra = self._fingers
        q_full = np.r_[np.asarray(q, float), np.zeros(n_extra)]
        return np.array([c.pose(q_full)[0] for c in chains])

    def _fk_jac(self, q):
        """The tool's position and rotation matrix at joints `q`, and its 6x7 Jacobian (linear, then angular, in
        arm_mount), in one pass down the chain -- kinematics.Chain.pose and .jacobian together, the same numbers."""
        rot, pos = np.eye(3), np.zeros(3)
        axes, points, cols = [], [], []
        for r0, t0, axis, k in self._steps:
            pos = pos + rot @ t0
            rot = rot @ r0
            if axis is not None:
                axes.append(rot @ axis)
                points.append(pos)
                cols.append(k)
                rot = rot @ _rodrigues(axis, q[k])
        z, r = np.array(axes), pos - np.array(points)  # each joint's axis, and the lever arm to the tool
        J = np.zeros((6, len(q)))
        np.add.at(
            J[:3].T,
            cols,
            np.c_[
                z[:, 1] * r[:, 2] - z[:, 2] * r[:, 1],
                z[:, 2] * r[:, 0] - z[:, 0] * r[:, 2],
                z[:, 0] * r[:, 1] - z[:, 1] * r[:, 0],
            ],
        )
        np.add.at(J[3:].T, cols, z)
        return pos, rot, J

    def pose(self, q):
        """(position, quaternion xyzw) of the tool at joints `q`, in arm_mount."""
        pos, rot, _ = self._fk_jac(np.asarray(q, float))
        return pos, R.from_matrix(rot).as_quat()

    def ik(self, pos, quat, seed, wrap=True):
        """Joints putting the tool at (`pos`, `quat`), continued from `seed` (its branch kept), the hard joint limits
        held: (joints, error in m -- position plus ROT_ERR_EQUIV_M per radian of orientation). `wrap`: the free joints
        wrapped to [-180, 180) as commanded (sim.wrap_joints); False keeps them continuous from the seed."""
        q = np.clip(np.asarray(seed, float).copy(), self.lo, self.hi)
        target, pos = R.from_quat(quat).as_matrix(), np.asarray(pos, float)
        lam = IK_DAMP**2 * np.eye(6)
        for _ in range(IK_ITERS + 1):  # the last pass only measures the last step's result
            p, rot, J = self._fk_jac(q)
            e = np.r_[pos - p, _rotvec(target @ rot.T)]
            if np.linalg.norm(e[:3]) < IK_TOL_M and np.linalg.norm(e[3:]) < IK_TOL_RAD:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + lam, e)
            q = np.clip(q + np.clip(dq, -IK_MAX_STEP_RAD, IK_MAX_STEP_RAD), self.lo, self.hi)
        err = float(np.linalg.norm(e[:3]) + ROT_ERR_EQUIV_M * np.linalg.norm(e[3:]))
        return (sim.wrap_joints(q) if wrap else q), err

    def joint_points(self, q):
        """Each joint's origin (J1..J7) and the tool's, in arm_mount: (8, 3)."""
        return self.chain.joint_points(q)

    def lowest(self, q):
        """(z, name) of the lowest joint origin past J1 or the tool at joints `q`."""
        pts = self.joint_points(q)[1:]
        names = [f"arm J{k}" for k in range(2, len(pts) + 1)] + ["tool"]
        k = int(np.argmin(pts[:, 2]))
        return float(pts[k, 2]), names[k]

    def _shape(self, q, held=None):
        """The arm at `q` as point sets, each with the radius it is grown by and its name: the capsules' axes, the
        hand box's surface (with the box's centre and rotation, for the overlap test), the held sphere's centre."""
        pts = self.joint_points(q)
        out = []
        for k in range(len(pts) - 2):  # J1 -> J2 ... J6 -> J7; the hand box covers J7 -> the tool
            a, b = pts[k], pts[k + 1]
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / ARM_SAMPLE_M)) + 1)
            out.append((a + (b - a) * np.linspace(0.0, 1.0, n)[:, None], ARM_RADIUS_M, f"arm J{k + 1}-J{k + 2}", None))
        pos, quat = self.pose(q)
        rot = R.from_quat(quat)
        out.append((pos + rot.apply(self._hand), 0.0, "gripper", (pos + rot.apply(self._hand_c), rot.as_matrix())))
        if held is not None:
            out.append(((pos + rot.apply(held.centre_tool))[None, :], float(held.radius), "container", None))
        return out

    def clearance(self, q, bodies, closed=False, held=None):
        """(smallest distance in m, "<part> vs <body>") from the arm at joints `q` to `bodies` [(Box, name)] --
        negative: into one; (9.0, "") with none. `closed` is accepted for the old signature: the hand is its open box
        either way (the wider). `held`: a Sphere carried in the gripper, checked too."""
        best = (9.0, "")
        if not bodies:
            return best
        for pts, radius, part, solid in self._shape(q, held):
            for box, name in bodies:
                gap = float(box_gaps(pts, box).min()) - radius
                if solid is not None:  # the hand: an obstacle inside it touches none of its surface points
                    depth = _overlap(solid[0], solid[1], self._hand_h, box)
                    if depth > 0:
                        gap = min(gap, -depth)
                if gap < best[0]:
                    best = (gap, f"{part} vs {name}")
        return best


# --- the process's one model (the driver's URDF: set_model_provider; offline, a file) ---
_PROVIDER: Callable[[], str] | None = None
_MODEL: ArmModel | None = None


class ModelError(Exception):
    pass


def set_model_provider(fn):
    """Where the URDF comes from: fn() -> its text (self.arm.robot_description). None: none. Rebuilds on next use."""
    global _PROVIDER, _MODEL
    _PROVIDER, _MODEL = fn, None


def load(urdf_path):
    """The model from a URDF file (offline: the bench, tests), replacing any provider's."""
    global _PROVIDER, _MODEL
    text = Path(urdf_path).read_text()
    _PROVIDER, _MODEL = (lambda: text), None
    return model()


def model() -> ArmModel:
    """The process's one ArmModel, built on first use from the provider."""
    global _MODEL
    if _MODEL is None:
        if _PROVIDER is None:
            raise ModelError("no arm model: set_model_provider(self.arm.robot_description) or load(<urdf>) first")
        text = _PROVIDER()
        _MODEL = ArmModel(text)
        # cuRobo's joint-limit band (sim.limit_message / limit_room) on this model's limits, as sim.make_sim sets them
        sim.ARM_LIMITS[:] = sim.arm_limits(text)
    return _MODEL


def has_model() -> bool:
    return _MODEL is not None or _PROVIDER is not None
