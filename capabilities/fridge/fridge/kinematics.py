"""The arm's forward kinematics, from its own model (/arm/robot_description), in numpy.

cuRobo plans every motion; this only checks what it planned: that a leg on or near the door (straight in to the
handle, the swing, the pushes) keeps the tool on the line or arc it is meant to follow, and where a planned
configuration puts the tool. The chain is read from the URDF the driver publishes, from `arm_mount` (the planning
frame) to `arm_tool_frame` (cuRobo's ee_link, self.arm's tool frame).
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

JOINTS = [f"arm_joint_{i}" for i in range(1, 8)]
ROOT, TIP = "arm_mount", "arm_tool_frame"

# local IK (Chain.ik): from a nearby configuration, on its branch -- the joint goals of the legs on or along the door
IK_ITERS = 60
IK_TOL_M, IK_TOL_RAD = 2e-4, 2e-3
IK_DAMP = 0.02  # damped least squares
IK_MAX_STEP_RAD = 0.1  # per iteration, per joint
CONTINUOUS_RAD = 6.28  # cuRobo's range for a continuous joint
LIMIT_MARGIN_RAD = 0.1  # cuRobo's joint_limit_margin_rad: a goal this close to a limit it refuses
# the arm's spare joint: each solve also turns the arm in the hand's null space (the hand doesn't move) to keep the
# limited joints (J2, J4, J6) toward the middle of their range, J6 the one that runs out first (the side grasp, 11-06:
# turning the hand with the door folds J6 into its limit)
NS_WEIGHTS = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 4.0, 0.0])
NS_ACTIVE_RAD = np.radians([180.0, 110.0, 180.0, 120.0, 180.0, 60.0, 180.0])  # ... only past these (|q|): J2, J4, J6
NS_STEP_RAD = np.radians(3.0)  # at most this much null-space turn per solve: the elbow drifts, never jumps


def _floats(el, attr, default):
    return np.array([float(v) for v in (default if el is None else el.get(attr, default)).split()])


class Chain:
    """FK of the URDF chain `root` -> `tip` over the joints `joints` (rad, in that order)."""

    def __init__(self, urdf_xml, root=ROOT, tip=TIP, joints=JOINTS):
        robot = ET.fromstring(urdf_xml)
        by_child = {j.find("child").get("link"): j for j in robot.findall("joint")}
        chain, link = [], tip
        while link != root:
            joint = by_child.get(link)
            if joint is None:
                raise ValueError(f"robot model: no joint chain from {root} to {tip} (stuck at {link})")
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.steps = []  # (fixed 4x4 origin, joint axis or None, index into `joints`)
        for joint in reversed(chain):
            origin = joint.find("origin")
            T = np.eye(4)
            T[:3, :3] = R.from_euler("xyz", _floats(origin, "rpy", "0 0 0")).as_matrix()
            T[:3, 3] = _floats(origin, "xyz", "0 0 0")
            kind = joint.get("type")
            if kind in ("revolute", "continuous"):
                axis = _floats(joint.find("axis"), "xyz", "1 0 0")
                self.steps.append((T, axis / np.linalg.norm(axis), joints.index(joint.get("name"))))
            elif kind == "fixed":
                self.steps.append((T, None, None))
            else:
                raise ValueError(f"robot model: joint {joint.get('name')} is {kind}, not revolute/fixed")
        # the joints' limits (rad); a continuous joint has +-2 pi, as cuRobo makes it
        self.lower, self.upper = np.full(len(joints), -CONTINUOUS_RAD), np.full(len(joints), CONTINUOUS_RAD)
        self.limited = np.zeros(len(joints), bool)
        for joint in robot.findall("joint"):
            if joint.get("name") in joints and joint.get("type") == "revolute":
                lim = joint.find("limit")
                k = joints.index(joint.get("name"))
                self.lower[k], self.upper[k] = float(lim.get("lower")), float(lim.get("upper"))
                self.limited[k] = True

    def poses(self, qs):
        """Tool poses for joint configurations `qs` ((N, 7) or (7,)): (positions (N, 3), quaternions xyzw (N, 4))."""
        qs = np.atleast_2d(np.asarray(qs, float))
        T = np.broadcast_to(np.eye(4), (len(qs), 4, 4)).copy()
        for origin, axis, k in self.steps:
            T = T @ origin
            if axis is not None:
                rot = np.broadcast_to(np.eye(4), (len(qs), 4, 4)).copy()
                rot[:, :3, :3] = R.from_rotvec(axis[None, :] * qs[:, k : k + 1]).as_matrix()
                T = T @ rot
        return T[:, :3, 3], R.from_matrix(T[:, :3, :3]).as_quat()

    def joint_points(self, q):
        """Positions (arm_mount) of each joint's origin along the chain, then the tool's, at joints `q`: the arm as a
        polyline (the close checks it, as capsules, against where the door may be)."""
        q = np.asarray(q, float)
        T, pts = np.eye(4), []
        for origin, axis, k in self.steps:
            T = T @ origin
            if axis is not None:
                pts.append(T[:3, 3].copy())
                rot = np.eye(4)
                rot[:3, :3] = R.from_rotvec(axis * q[k]).as_matrix()
                T = T @ rot
        pts.append(T[:3, 3].copy())
        return np.array(pts)

    def pose(self, q):
        """(position, quaternion xyzw) of the tool at joints `q`."""
        pos, quat = self.poses(q)
        return pos[0], quat[0]

    def jacobian(self, q):
        """6x7 Jacobian of the tool at joints `q` (rows: linear velocity, then angular velocity in the root frame,
        the rotation-vector rate the IK's error uses), exact from one forward pass: a revolute joint with world
        axis z through point p moves the tool at z x (tip - p) and turns it at z."""
        q = np.asarray(q, float)
        T = np.eye(4)
        axes, points, cols = [], [], []
        for origin, axis, k in self.steps:
            T = T @ origin
            if axis is not None:
                axes.append(T[:3, :3] @ axis)
                points.append(T[:3, 3].copy())
                cols.append(k)
                rot = np.eye(4)
                rot[:3, :3] = R.from_rotvec(axis * q[k]).as_matrix()
                T = T @ rot
        J = np.zeros((6, len(q)))
        for z, p, k in zip(axes, points, cols, strict=True):
            J[:3, k] += np.cross(z, T[:3, 3] - p)
            J[3:, k] += z
        return J

    def fk_jac_batch(self, qs):
        """For joint configurations `qs` (N, 7): tool positions (N, 3), rotation matrices (N, 3, 3) and Jacobians
        (N, 6, 7) as jacobian(), all at once."""
        qs = np.atleast_2d(np.asarray(qs, float))
        n = len(qs)
        T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        axes, points, cols = [], [], []
        for origin, axis, k in self.steps:
            T = T @ origin
            if axis is not None:
                axes.append(T[:, :3, :3] @ axis)
                points.append(T[:, :3, 3].copy())
                cols.append(k)
                rot = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
                rot[:, :3, :3] = R.from_rotvec(axis[None, :] * qs[:, k : k + 1]).as_matrix()
                T = T @ rot
        tip = T[:, :3, 3]
        J = np.zeros((n, 6, qs.shape[1]))
        for z, p, k in zip(axes, points, cols, strict=True):
            J[:, :3, k] += np.cross(z, tip - p)
            J[:, 3:, k] += z
        return tip, T[:, :3, :3], J

    def ik_batch(self, pos, quat, q_seed, iters=30, tol_m=0.003, tol_rad=np.radians(2.0)):
        """ik() for many targets at once: tool positions `pos` (N, 3), quaternions `quat` (N, 4), seeds `q_seed`
        (N, 7) -- damped least squares, all N together (no null-space step). Returns (q (N, 7), ok (N,)): ok where it
        converged and is not within LIMIT_MARGIN_RAD of a limit."""
        q = np.array(q_seed, float, copy=True)
        target = R.from_quat(quat)
        pos = np.asarray(pos, float)
        done = np.zeros(len(q), bool)
        lam = IK_DAMP**2 * np.eye(6)
        for _ in range(iters + 1):
            tip, rot, J = self.fk_jac_batch(q)
            e = np.concatenate([pos - tip, (target * R.from_matrix(rot).inv()).as_rotvec()], axis=1)
            done = (np.linalg.norm(e[:, :3], axis=1) < tol_m) & (np.linalg.norm(e[:, 3:], axis=1) < tol_rad)
            if done.all():
                break
            step = np.einsum("nji,nj->ni", J, np.linalg.solve(J @ J.transpose(0, 2, 1) + lam, e[..., None])[..., 0])
            q = np.where(done[:, None], q, q + np.clip(step, -IK_MAX_STEP_RAD, IK_MAX_STEP_RAD))
        near = ((q < self.lower + LIMIT_MARGIN_RAD) | (q > self.upper - LIMIT_MARGIN_RAD)).any(axis=1)
        return q, done & ~near

    def jacobian_fd(self, q, eps=1e-6):
        """jacobian by finite differences (eight forward passes): the check of the exact one."""
        q = np.asarray(q, float)
        qs = np.vstack([q] + [q + eps * np.eye(len(q))[k] for k in range(len(q))])
        pos, quat = self.poses(qs)
        r0 = R.from_quat(quat[0])
        J = np.zeros((6, len(q)))
        J[:3] = ((pos[1:] - pos[0]) / eps).T
        J[3:] = ((R.from_quat(quat[1:]) * r0.inv()).as_rotvec() / eps).T
        return J

    def ik(self, pos, quat, q_seed, iters=IK_ITERS, tol_m=IK_TOL_M, tol_rad=IK_TOL_RAD):
        """Joints putting the tool at (`pos`, `quat`), continued from `q_seed` (a nearby configuration, whose branch
        it keeps), the spare joint used to keep J2/J4/J6 toward the middle of their range. Returns (q, why): q None
        with why when it doesn't converge or ends within LIMIT_MARGIN_RAD of a limit."""
        q = np.asarray(q_seed, float).copy()
        target = R.from_quat(quat)
        # one bounded null-space step (the hand doesn't move) pulling back the limited joints already past
        # NS_ACTIVE_RAD -- the side grasp's resolved-rate IK (11-06); nothing when none is
        over = np.sign(q) * np.maximum(np.abs(q) - NS_ACTIVE_RAD, 0.0) * NS_WEIGHTS
        if np.any(over != 0):
            J = self.jacobian(q)
            Jp = J.T @ np.linalg.inv(J @ J.T + IK_DAMP**2 * np.eye(6))
            ns = (np.eye(len(q)) - Jp @ J) @ -over
            m = float(np.max(np.abs(ns)))
            if m > 1e-12:
                q = q + ns * min(1.0, NS_STEP_RAD / m)
        for it in range(iters + 1):  # the last pass only checks the last step's result
            p, qt = self.pose(q)
            e = np.r_[np.asarray(pos, float) - p, (target * R.from_quat(qt).inv()).as_rotvec()]
            if np.linalg.norm(e[:3]) < tol_m and np.linalg.norm(e[3:]) < tol_rad:
                break
            if it == iters:
                return None, f"local IK did not converge ({np.linalg.norm(e[:3]) * 1000:.1f} mm, {np.degrees(np.linalg.norm(e[3:])):.1f} deg off)"
            J = self.jacobian(q)
            Jp = J.T @ np.linalg.inv(J @ J.T + IK_DAMP**2 * np.eye(6))
            q = q + np.clip(Jp @ e, -IK_MAX_STEP_RAD, IK_MAX_STEP_RAD)
        near = (q < self.lower + LIMIT_MARGIN_RAD) | (q > self.upper - LIMIT_MARGIN_RAD)
        if near.any():
            k = int(np.argmax(near))
            return None, f"J{k + 1} at {np.degrees(q[k]):.0f} deg, within {np.degrees(LIMIT_MARGIN_RAD):.0f} deg of its limit"
        return q, ""
