from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

Vec = np.ndarray

ACCEPTED = "accepted"
DEADBAND = "deadband"
TILT = "tilt"
SHIFT = "shift"
STEP = "step"


def _unit(v: Sequence[float]) -> Vec:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("cannot normalize a zero-length vector")
    return v / n


def _angle(a: Vec, b: Vec) -> float:
    """Angle between two unit vectors, rad. atan2 form: exact near 0."""
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))


def _clip_norm(v: Vec, max_norm: float) -> Vec:
    n = float(np.linalg.norm(v))
    return v * (max_norm / n) if n > max_norm else v


def detector_rotation(normal: Sequence[float], up: Sequence[float]) -> Vec:
    """The plane_detection node's frame axes for a wall with this normal.

    Mirrors plane_rotation() in plane_detection/src/plane_frame.cpp (gravity
    branch): z = normal, y = world DOWN projected into the plane, x = y cross z.
    Teleop's _R_FE is defined against these axes, so the followed orientation must
    be rebuilt with them (not with gravity_aligned_rotation(), whose x/y are the
    detector's rotated 180 deg about z) or the camera roll flips.
    """
    z = _unit(normal)
    down = -_unit(up)
    y = down - float(down @ z) * z
    if float(np.linalg.norm(y)) < 1e-3:
        raise ValueError("plane is horizontal; no 'down' direction on its face")
    y = _unit(y)
    x = _unit(np.cross(y, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


@dataclass(frozen=True)
class FollowParams:
    w0: float = 2.0  # rad/s, filter natural frequency (settles in ~2.5 s)
    max_rate: float = math.radians(4.0)  # rad/s, cap on the normal's swing
    max_shift_rate: float = 0.02  # m/s, cap on the wall offset's change
    deadband_deg: float = 1.0
    deadband_m: float = 0.005
    max_tilt_deg: float = 15.0  # from the homing normal
    max_shift_m: float = 0.05  # from the homing wall offset
    max_step_deg: float = 5.0  # from the follower's current normal
    max_step_m: float = 0.03  # from the follower's current offset

    def problems(self) -> List[str]:
        out = []
        for name in ("w0", "max_rate", "max_shift_rate", "max_tilt_deg",
                     "max_shift_m", "max_step_deg", "max_step_m"):
            if not getattr(self, name) > 0.0:
                out.append(f"{name} must be > 0, got {getattr(self, name)}")
        if self.deadband_deg < 0.0 or self.deadband_m < 0.0:
            out.append("deadband_deg and deadband_m must be >= 0")
        if self.max_tilt_deg >= 60.0:
            out.append("max_tilt_deg must be < 60 (the wall must stay a wall)")
        if self.deadband_deg >= self.max_step_deg:
            out.append("deadband_deg must be < max_step_deg")
        if self.deadband_m >= self.max_step_m:
            out.append("deadband_m must be < max_step_m")
        return out


class PlaneFollower:
    def __init__(
        self,
        normal: Sequence[float],
        d: float,
        anchor: Sequence[float],
        params: FollowParams,
    ) -> None:
        bad = params.problems()
        if bad:
            raise ValueError("; ".join(bad))
        self.p = params
        self._c = np.asarray(anchor, dtype=np.float64).copy()
        self._n0 = _unit(normal)
        self._e0 = float(d) - float(self._n0 @ self._c)
        self._n = self._n0.copy()
        self._e = self._e0
        self._n_dot = np.zeros(3)
        self._e_dot = 0.0
        self._n_ref = self._n0.copy()
        self._e_ref = self._e0

    # ------------------------------------------------------------ output

    @property
    def normal(self) -> Vec:
        return self._n.copy()

    @property
    def d(self) -> float:
        return float(self._n @ self._c) + self._e

    @property
    def origin(self) -> Vec:
        """The anchor projected onto the current plane."""
        return self._c + self._e * self._n

    @property
    def angular_velocity(self) -> Vec:
        """Rate of turn of the normal, rad/s (for diagnostics)."""
        return np.cross(self._n, self._n_dot)

    @property
    def tilt_deg(self) -> float:
        """Current normal's angle from the homing normal."""
        return math.degrees(_angle(self._n, self._n0))

    @property
    def shift_m(self) -> float:
        """Current wall offset at the anchor relative to homing, + = further in."""
        return self._e - self._e0

    @property
    def ref_tilt_deg(self) -> float:
        return math.degrees(_angle(self._n_ref, self._n0))

    @property
    def ref_shift_m(self) -> float:
        return self._e_ref - self._e0

    # ------------------------------------------------------------ input

    def offer(self, normal: Sequence[float], d: float) -> str:
        """Gate one plane estimate; adopt it as the reference if it passes.

        Returns ACCEPTED, DEADBAND (fine, but too close to matter), or the gate
        that refused it: TILT, SHIFT (vs homing) or STEP (vs the current plane).
        """
        n = _unit(normal)
        d = float(d)
        if float(n @ self._n0) < 0.0:  # same plane, other sign convention
            n, d = -n, -d
        e = d - float(n @ self._c)
        if math.degrees(_angle(n, self._n0)) > self.p.max_tilt_deg:
            return TILT
        if abs(e - self._e0) > self.p.max_shift_m:
            return SHIFT
        if (
            math.degrees(_angle(n, self._n)) > self.p.max_step_deg
            or abs(e - self._e) > self.p.max_step_m
        ):
            return STEP
        if (
            math.degrees(_angle(n, self._n_ref)) <= self.p.deadband_deg
            and abs(e - self._e_ref) <= self.p.deadband_m
        ):
            return DEADBAND
        self._n_ref, self._e_ref = n, e
        return ACCEPTED

    def brake(self) -> None:
        """Come to rest smoothly, with no reversal.

        For a critically damped filter moving at v, a reference v / w0 ahead of
        the current state is approached monotonically: the motion decays to a stop
        instead of overshooting the current point and coming back.
        """
        self._n_ref = _unit(self._n + self._n_dot / self.p.w0)
        self._e_ref = self._e + self._e_dot / self.p.w0

    def step(self, dt: float) -> None:
        """Advance the filter by dt seconds toward the reference."""
        if not dt > 0.0:
            return
        w0 = self.p.w0
        substeps = max(1, int(math.ceil(dt * w0 / 0.05)))
        h = dt / substeps
        for _ in range(substeps):
            # normal, on the unit sphere: the error is the tangent vector toward
            # the reference whose length is the angle between them
            t = self._n_ref - float(self._n_ref @ self._n) * self._n
            tn = float(np.linalg.norm(t))
            err = t * (_angle(self._n, self._n_ref) / tn) if tn > 1e-12 else np.zeros(3)
            acc = w0 * w0 * err - 2.0 * w0 * self._n_dot
            self._n_dot = _clip_norm(self._n_dot + acc * h, self.p.max_rate)
            self._n = _unit(self._n + self._n_dot * h)
            self._n_dot = self._n_dot - float(self._n_dot @ self._n) * self._n
            # offset at the anchor, scalar
            acc_e = w0 * w0 * (self._e_ref - self._e) - 2.0 * w0 * self._e_dot
            lim = self.p.max_shift_rate
            self._e_dot = min(max(self._e_dot + acc_e * h, -lim), lim)
            self._e += self._e_dot * h
