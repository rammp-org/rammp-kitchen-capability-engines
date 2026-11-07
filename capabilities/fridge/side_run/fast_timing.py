"""Smooth, velocity-limited timing for a joint route (this process only; the shared rammp_arm is unchanged).

rammp_arm times a route with ONE minimum-jerk law over the cumulative largest-joint move, capped by JOINT_SPEED on every
joint, through straight joint segments between knots. Here:
1. The route is resampled evenly along its joint-space length and SMOOTHED (a Gaussian along the path), so a kink where two
   planned pieces meet (11-06: cuRobo's 0.5 deg knots -> the merge's 2-3 deg knots) becomes a gentle bend. The last
   `protect_tail` knots (the straight slide into the handle) and both ends are kept exactly.
2. Speed along the path is limited by every joint's velocity (a fraction of the driver's max_ref_speed), by the
   acceleration that a BEND needs (|q''(s)| v^2 <= a_max), by the path acceleration (forward/backward pass), and,
   with a tool_fn, by the tool's speed (TOOL_MAX, TOOL_NEAR_END near the end) -- the limits themselves smoothed so they
   don't step.
Usage: import fast_timing; fast_timing.install(rammp_arm, tool_fn=..., protect_tail=...) before move_joints."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

DRIVER_MAX = np.array([1.40, 1.40, 1.40, 1.40, 1.22, 1.22, 1.22])  # the driver's max_ref_speed (rad/s, J1..J7)
SPEED_FRACTION = 0.8
ACCEL = np.array([1.5] * 7)  # rad/s^2 per joint: path speed-up/slow-down AND the acceleration a bend may take
N = 1500  # samples along the path
SMOOTH_RAD = 0.12  # Gaussian sigma along the path (rad of joint-space length) for smoothing the route's shape
LIMIT_SMOOTH = 0.03  # ... and for smoothing the speed limits (so a limit never steps)
TOOL_MAX = 0.25  # m/s, the tool point anywhere on the route (with a tool_fn)
TOOL_NEAR_END = 0.12  # m/s within NEAR_END_M of the route's end: going into the handle, the back finger ~1 cm off the door
NEAR_END_M = 0.10


def _smooth_path(path, protect_tail):
    """The route resampled at N points evenly along its joint-space length, Gaussian-smoothed except near both ends and
    over the last `protect_tail` knots, which are kept exactly. Returns (s, q)."""
    path = np.asarray(path, float)
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]
    path = path[keep]
    sk = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    L = float(sk[-1])
    s = np.linspace(0.0, L, N)
    q = PchipInterpolator(sk, path, axis=0)(s)
    ds = s[1] - s[0]
    sig = max(1.0, SMOOTH_RAD / ds)
    qs = gaussian_filter1d(q, sig, axis=0, mode="nearest")
    # weight 0 at the start, over the protected tail and at the end; 1 elsewhere, ramped over ~3 sigma
    s_tail = sk[max(0, len(sk) - 1 - protect_tail)] if protect_tail else L
    ramp = 3 * sig * ds
    w = np.clip(s / ramp, 0, 1) * np.clip((s_tail - s) / ramp, 0, 1) * np.clip((L - s) / ramp, 0, 1)
    q = qs * w[:, None] + q * (1 - w[:, None])
    s2 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]  # re-measure along the smoothed shape
    return s2, q


def samples(path, dt=0.02, speed_fraction=SPEED_FRACTION, tool_fn=None, protect_tail=0, monotone_tail=0, knot_tool_cap=None):
    """(t, q, qd) every `dt` along the smoothed route through joint `path` (knots). `monotone_tail`: over the route's
    last knots (e.g. the merge + slide into the handle) the speed only decreases -- one slow-down, no slow-fast-slow."""
    pa = np.asarray(path, float)
    sk = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pa, axis=0), axis=1))]
    mono_frac = sk[max(0, len(sk) - 1 - monotone_tail)] / max(sk[-1], 1e-9) if monotone_tail else 1.0
    s, q = _smooth_path(path, protect_tail)
    L = float(s[-1])
    if L < 1e-6:
        return None
    f = PchipInterpolator(s, q, axis=0)
    s = np.linspace(0.0, L, N)
    q = f(s)
    ds = s[1] - s[0]
    q1 = np.gradient(q, ds, axis=0)  # dq/ds
    q2 = np.gradient(q1, ds, axis=0)  # d2q/ds2: the bend
    vmax = np.min((DRIVER_MAX * speed_fraction) / (np.abs(q1) + 1e-9), axis=1)
    vmax = np.minimum(vmax, np.min(np.sqrt(ACCEL / (np.abs(q2) + 1e-9)), axis=1))  # |q''| v^2 <= a
    if tool_fn is not None:
        tool = np.array([tool_fn(qi) for qi in q])
        dtool = np.linalg.norm(np.gradient(tool, ds, axis=0), axis=1) + 1e-9
        d_end = np.linalg.norm(tool - tool[-1], axis=1)
        frac = np.clip((d_end - NEAR_END_M) / 0.03 + 0.5, 0, 1)  # blend the two caps over ~3 cm
        vmax = np.minimum(vmax, (TOOL_NEAR_END + (TOOL_MAX - TOOL_NEAR_END) * frac) / dtool)
        if knot_tool_cap is not None:  # per-knot tool speed caps (m/s): e.g. a stretch that must take a given time
            caps = np.asarray(knot_tool_cap, float)
            ok = np.r_[True, np.diff(sk) > 1e-12]
            cap_s = np.interp(s, sk[ok] * (L / max(sk[-1], 1e-12)), caps[ok])
            vmax = np.minimum(vmax, cap_s / dtool)
    w = max(3, int(LIMIT_SMOOTH / ds))
    vmax = gaussian_filter1d(minimum_filter1d(vmax, size=2 * w + 1, mode="nearest"), w / 2, mode="nearest")
    amax = np.min(ACCEL / (np.abs(q1) + 1e-9), axis=1)
    v = vmax.copy()
    v[0] = v[-1] = 0.0
    for i in range(1, N):  # forward: accelerate
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * amax[i - 1] * ds))
    for i in range(N - 2, -1, -1):  # backward: decelerate into the end
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * amax[i + 1] * ds))
    if monotone_tail:  # from there on only slow down (a running minimum), then re-check the slow-down before it
        i0 = int(mono_frac * (N - 1))
        v[i0:] = np.minimum.accumulate(v[i0:])
        for i in range(N - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * amax[i + 1] * ds))
    vm = np.maximum((v[1:] + v[:-1]) / 2, 1e-6)
    t_s = np.r_[0.0, np.cumsum(ds / vm)]
    T = float(t_s[-1])
    t = np.arange(0.0, T + dt / 2, dt)
    if t[-1] < T:
        t = np.r_[t, T]
    # s(t), then a light smoothing of the speed in time (rounds the corners of the accel profile: less jerk)
    s_t = np.interp(t, t_s, s)
    sd = np.gradient(s_t, t)
    sd = gaussian_filter1d(sd, 2.0, mode="constant", cval=0.0)  # zero speed beyond both ends: no jolt at the stop
    sd[0] = sd[-1] = 0.0
    s_t = np.r_[0.0, np.cumsum((sd[1:] + sd[:-1]) / 2 * np.diff(t))]
    s_t *= L / s_t[-1]  # end exactly at the end
    q = f(s_t)
    qd = np.gradient(q, t, axis=0)
    qd[0] = qd[-1] = 0.0
    return t, q, qd


def install(rammp_arm, speed_fraction=SPEED_FRACTION, tool_fn=None, protect_tail=0, monotone_tail=0, knot_tool_cap=None):
    """Make rammp_arm.joint_trajectory use this timing in this process (move_joints calls it). `tool_fn(q)` -> tool xyz
    turns on the tool-speed caps; `protect_tail`: the route's last knots kept exactly (no smoothing), e.g. the slide."""
    def joint_trajectory(q0, knots, cart_lengths=None, speed_scale=1.0):
        path = [np.asarray(q0, float)] + rammp_arm._unwrap_from(q0, knots)
        return samples(path, rammp_arm.DT, speed_fraction * min(1.0, speed_scale), tool_fn, protect_tail, monotone_tail,
                       knot_tool_cap)
    rammp_arm.joint_trajectory = joint_trajectory
