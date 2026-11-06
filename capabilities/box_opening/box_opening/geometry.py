"""Pure geometry helpers — duplicated from rammp_curobo.geometry so the
host-side package (not available outside the planner container) is not
required to run the mission or the launch file."""

import math


def ang_diff(a, b):
    """Signed angular difference a − b, wrapped to (−π, π]."""
    d = (float(a) - float(b)) % (2.0 * math.pi)
    return d - 2.0 * math.pi if d > math.pi else d


def euler_deg_to_quat_xyzw(rpy_deg):
    """ZYX Euler angles [roll, pitch, yaw] in degrees → quaternion [x, y, z, w]."""
    r, p, y = (math.radians(v) for v in rpy_deg)
    cr, cp, cy = math.cos(r / 2), math.cos(p / 2), math.cos(y / 2)
    sr, sp, sy = math.sin(r / 2), math.sin(p / 2), math.sin(y / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def rotate_about_world_axis(quat_xyzw, axis, angle):
    """q(axis, angle) in the world frame, times quat_xyzw (xyzw; `axis` a
    unit vector, `angle` in radians)."""
    s, aw = math.sin(angle / 2), math.cos(angle / 2)
    ax, ay, az = (s * float(a) for a in axis)
    bx, by, bz, bw = quat_xyzw
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def yaw_about_world_z(quat_xyzw, yaw):
    """Pre-multiply quat_xyzw by a pure yaw rotation about the world Z axis.

    Equivalent to q_yaw(world) * quat_xyzw, where yaw is in radians."""
    qx, qy, qz, qw = quat_xyzw
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    return [
        cy * qx - sy * qy,
        cy * qy + sy * qx,
        cy * qz + sy * qw,
        cy * qw - sy * qz,
    ]
