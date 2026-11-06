from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ------------------------------------------------------------------ rotations


def axis_vector(spec: str) -> np.ndarray:
    """'+z' -> [0, 0, 1], '-x' -> [-1, 0, 0]. A bare 'z' means '+z'."""
    s = str(spec).strip().lower()
    if len(s) == 1:
        s = "+" + s
    if len(s) != 2 or s[0] not in "+-" or s[1] not in _AXIS_INDEX:
        raise ValueError(f"axis must look like '+z' or '-x', got {spec!r}")
    v = np.zeros(3)
    v[_AXIS_INDEX[s[1]]] = 1.0 if s[0] == "+" else -1.0
    return v


def quat_to_matrix(q_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w) with w >= 0 (Shepperd)."""
    m = np.asarray(r, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        ]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    qa = np.array(q)
    qa /= np.linalg.norm(qa)
    return -qa if qa[3] < 0.0 else qa


def axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues: rotation by `angle` (rad, right-handed) about unit `axis`."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def rotvec_from_matrix(r: np.ndarray) -> np.ndarray:
    """Log map: rotation matrix -> rotation vector (axis * angle, angle in [0, pi])."""
    q = matrix_to_quat(r)  # w >= 0, so the angle is the short way round
    v = q[:3]
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3)
    return v / s * (2.0 * math.atan2(s, q[3]))


def facing_rotation(
    optical_axis_ee: Sequence[float],
    ref_axis_ee: Sequence[float],
    ref_axis_fridge: Sequence[float],
    roll_rad: float,
) -> np.ndarray:
    """R_FE: end-effector orientation in the fridge frame that "faces the fridge".

    Convention: fridge +z points INTO the back wall (away from the arm), so the EE
    optical axis maps to fridge +z. With roll = 0, `ref_axis_ee` maps to
    `ref_axis_fridge`; roll then rotates the EE about its own optical axis
    (right-handed about that axis). Both reference axes must be perpendicular to
    their optical/z axes.
    """
    a = np.asarray(optical_axis_ee, dtype=float)
    b = np.asarray(ref_axis_ee, dtype=float)
    c = np.asarray(ref_axis_fridge, dtype=float)
    a, b, c = (v / np.linalg.norm(v) for v in (a, b, c))
    d = np.array([0.0, 0.0, 1.0])  # look along fridge +z, into the wall
    if abs(float(a @ b)) > 1e-6:
        raise ValueError(
            "the EE roll reference axis must be perpendicular to the optical axis"
        )
    if abs(float(c @ d)) > 1e-6:
        raise ValueError(
            "the fridge roll reference axis must be the fridge x or y axis"
        )
    m_e = np.column_stack([a, b, np.cross(a, b)])  # EE basis, in EE coordinates
    m_f = np.column_stack([d, c, np.cross(d, c)])  # where those axes go, in F
    return (m_f @ m_e.T) @ axis_angle_matrix(a, roll_rad)


def upright_reference(up_F: Sequence[float], min_in_plane: float) -> np.ndarray:
    """World "up" projected onto the fridge x/y plane (the wall plane), unit length.

    Used as the fridge-side roll reference, so the camera comes out upright however
    the RANSAC node happens to orient x/y in the wall. `min_in_plane` is the
    smallest allowed length of the projection before normalising, i.e. the cosine
    of the largest allowed tilt of the wall from vertical.
    """
    u = np.array(up_F, dtype=float)
    u = u / np.linalg.norm(u)
    u[2] = 0.0  # drop the component along fridge z (the wall normal)
    n = float(np.linalg.norm(u))
    if n < min_in_plane:
        tilt = math.degrees(math.acos(min(n, 1.0)))
        raise ValueError(f"the wall is {tilt:.0f} deg from vertical; 'up' is not in it")
    return u / n


def view_rotation(up_ref_P: Sequence[float]) -> np.ndarray:
    """R_PV: the view a camera has when it looks along plane +z with `up_ref_P` up.

    V uses the optical-frame convention, x right, y down, z forward (into the
    plane), so it is right-handed. `up_ref_P` must be a unit vector in the plane,
    as returned by upright_reference().
    """
    z = np.array([0.0, 0.0, 1.0])
    u = np.asarray(up_ref_P, dtype=float)
    return np.column_stack([np.cross(z, u), -u, z])


def gravity_aligned_rotation(z_axis: Sequence[float], up: Sequence[float]) -> np.ndarray:
    """Rotation whose columns are the plane axes of the old servo node's frame.

      z = `z_axis` (the plane normal, INTO the plane)
      x = up x z, horizontal along the face -> u = LEFT as seen from the robot
      y = z x x, along the face              -> v = UP the face

    Built from gravity only, so it is the same whatever in-plane x/y the frame
    that supplied `z_axis` (RANSAC) has.
    """
    z = np.asarray(z_axis, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=float), z)
    n = float(np.linalg.norm(x))
    if n < 1e-3:
        raise ValueError("plane is horizontal; no 'up' direction on its face")
    x = x / n
    return np.column_stack([x, np.cross(z, x), z])


# ------------------------------------------------------------------ transforms


def make_transform(r: np.ndarray, p: Sequence[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = np.asarray(p, dtype=float)
    return t


def transform_from_pq(p: Sequence[float], q_xyzw: Sequence[float]) -> np.ndarray:
    return make_transform(quat_to_matrix(q_xyzw), p)


def invert_transform(t: np.ndarray) -> np.ndarray:
    r_t = t[:3, :3].T
    return make_transform(r_t, -r_t @ t[:3, 3])


# ------------------------------------------------------------------ shaping


def clamp_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > limit and n > 0.0:
        return v * (limit / n)
    return v


def radial_deadzone(x: float, y: float, deadzone: float) -> Tuple[float, float]:
    """Circular deadzone on a 2D stick, rescaled so the output still reaches 1."""
    r = math.hypot(x, y)
    if r <= deadzone or r == 0.0:
        return 0.0, 0.0
    scale = (min(r, 1.0) - deadzone) / (1.0 - deadzone) / r
    return x * scale, y * scale


def scalar_deadzone(v: float, deadzone: float) -> float:
    a = abs(v)
    if a <= deadzone:
        return 0.0
    return math.copysign((min(a, 1.0) - deadzone) / (1.0 - deadzone), v)


def trigger_to_unit(raw: float, released: float, pressed: float) -> float:
    """Map a trigger axis to 0 (released) .. 1 (fully pressed)."""
    t = (raw - released) / (pressed - released)
    return min(max(t, 0.0), 1.0)


def ramp_vector(
    current: np.ndarray,
    target: np.ndarray,
    max_accel: float,
    max_decel: float,
    dt: float,
) -> np.ndarray:
    """Move `current` toward `target` with a bounded rate of change (vector norm).

    Slowing down (|target| < |current|) uses `max_decel`, so stopping can be made
    quicker than starting.
    """
    slowing = np.linalg.norm(target) < np.linalg.norm(current)
    limit = (max_decel if slowing else max_accel) * dt
    return current + clamp_norm(target - current, limit)


def limit_to_workspace(
    p: np.ndarray,
    v: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    slowdown_time_s: float,
) -> Tuple[np.ndarray, List[bool]]:
    """Per-axis soft walls on a box [lo, hi].

    Velocity toward a wall is capped at distance / slowdown_time_s, so the point
    slows as it approaches and never crosses. Outside the box, only motion back
    toward it is allowed (the limiter never pushes; it only withholds).
    Returns the limited velocity and which axes were limited.
    """
    out = np.array(v, dtype=float)
    limited = [False, False, False]
    for i in range(3):
        cap_hi = max((hi[i] - p[i]) / slowdown_time_s, 0.0)
        cap_lo = min((lo[i] - p[i]) / slowdown_time_s, 0.0)
        clipped = min(max(out[i], cap_lo), cap_hi)
        if clipped != out[i]:
            limited[i] = True
            out[i] = clipped
    return out, limited
