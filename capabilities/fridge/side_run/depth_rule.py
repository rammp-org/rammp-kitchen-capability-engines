"""The depth rule (11-06): along the whole route no part of the arm (any link, gripper open) is deeper into
the door than the arm's deepest point at the grasp itself. Measured exactly in the sim: a large box whose face is a
plane square to the door, and pybullet's closest distance from every robot link to it."""
import numpy as np
import pybullet as p
from capability_engine import sim

DEPTH_TOL_M = 0.003  # numerical slack: the slide keeps the hand's depth to ~0.2 mm
_D0 = 0.30  # the box's face this far deeper than the reference point (nothing of the arm gets there)
_slabs = {}


def _slab(rb, into, ref):
    c = rb.physics_client_id
    key = (c, tuple(np.round(into, 6)), tuple(np.round(ref, 5)))
    if key not in _slabs:
        for k in [k for k in _slabs if k[0] == c]:
            p.removeBody(_slabs.pop(k), physicsClientId=c)
        half = [1.0, 3.0, 3.0]
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c)
        yaw = float(np.arctan2(into[1], into[0]))
        centre = np.asarray(ref, float) + np.asarray(into, float) * (_D0 + half[0])
        _slabs[key] = p.createMultiBody(0, shape, -1, list(centre), p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=c)
    return _slabs[key]


def deepest(rb, q, into, ref):
    """(depth, link): how far the arm's deepest point at joints q is past `ref` along the horizontal `into`."""
    into = np.asarray(into, float).copy(); into[2] = 0.0; into /= np.linalg.norm(into)
    sid = _slab(rb, into, ref)
    sim.clearance(rb, q, [])  # sets the arm, gripper open
    c = rb.physics_client_id
    best, link = 9.0, ""
    for pt in p.getClosestPoints(rb.robot_id, sid, 2.0, physicsClientId=c):
        if pt[8] < best:
            best = pt[8]
            link = p.getJointInfo(rb.robot_id, pt[3], physicsClientId=c)[12].decode() if pt[3] >= 0 else "base"
    return _D0 - best, link


def check(rb, qs, into, ref, q_grasp):
    """Worst violation of the depth rule over joint samples qs: (excess_m, sample index, link, limit_m). Excess > 0
    means some part of the arm is that much deeper than the deepest point at the grasp (q_grasp)."""
    limit, _ = deepest(rb, q_grasp, into, ref)
    worst = (-9.0, -1, "", limit)
    for i, q in enumerate(qs):
        d, ln = deepest(rb, q, into, ref)
        if d - limit > worst[0]:
            worst = (d - limit, i, ln, limit)
    return worst
