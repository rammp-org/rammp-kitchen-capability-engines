"""One speed profile for every unguarded motion — the human-motion pass.

What read as "robotic" (measured 2026-09-03): every leg started and ended
at rest (the planner pins terminal velocity), so a run made ~8 dead stops
where a person makes 3; and every leg was a symmetric bell uniformly slowed
to 0.75 — sluggish through the middle, still abrupt at the end.

This module re-times a joint path. POSITIONS ARE NEVER TOUCHED: the path
the planner validated is the path flown (the contract warp.py established
for guarded descents); only the timestamps change, and the velocities and
accelerations annotating them are recomputed from those timestamps.

The profile, along the path's joint-space arc length:
  - ease out of rest at `accel`, cruise at `cruise` x the per-joint velocity
    cap, ease into rest at `decel` — decel < accel, so the arrival is the
    long half (the precision signature of human reaching);
  - corners (direction changes along the path, including the junction
    between two chained legs) are taken at the speed their turning
    acceleration allows — a shallow corner flows, a sharp one slows, a
    reversal (> reversal_deg) is a genuine stop — and a chained junction,
    where one is set, at no more than junction_cruise, whatever its legs'
    own fractions (a compliant run's, runtime/runner.py
    Runner.retime_params);
  - per-joint velocity never exceeds vmax_margin x the joint limit, so the
    gates runtime/driver.py applies before every goal (URDF velocity
    limits, per-interval continuity at 3 x vmax x dt) hold by
    construction — and refuse the goal if not;
  - nor, where one is set (qd_ceiling), an ABSOLUTE per-joint ceiling the
    cruise fraction does not scale: what a compliant arm's wrist follows
    (runtime/runner.py Runner.retime_params).

Guarded strokes are NOT re-timed here: their fast-then-slow warp and the
guard's rebaseline/arm fractions are tuned to contact and stay in warp.py.
"""

import math
from dataclasses import dataclass

import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening.runtime.stamps import secs, set_stamp


@dataclass(frozen=True)
class RetimeParams:
    accel: float = 6.0  # rad/s^2 along the path: easing OUT of rest / a corner
    decel: float = 3.0  # rad/s^2 easing INTO rest / a corner (longer: precision)
    corner_accel: float = 4.0  # turning acceleration allowed through a corner
    reversal_deg: float = 150.0  # a direction change beyond this is a stop
    corner_window: float = 0.03  # rad of path on each side over which a turn is measured
    vmax_margin: float = 0.9  # per-joint velocity cap, fraction of the limit
    v_floor: float = 0.02  # rad/s: never divide by a smaller average speed
    smooth_pts: int = 4  # the profile's tops are rounded over this many samples
    first_dt: float = 0.02  # point 0 sits one step in the future (planner convention)
    # around a chained junction the planner's samples crowd together (each
    # leg decelerated to rest there); the spline executes the kink's turn
    # inside the two adjacent intervals, so those must be real chords —
    # samples within junction_window of the kink are thinned to at least
    # junction_chord apart (dropped, never moved; the tail of a planned leg
    # is straight to well under a milliradian over such a chord)
    junction_chord: float = 0.015
    junction_window: float = 0.06
    # the robot's own acceleration limit: bounds the LOCAL turn at a kink
    # (two chained legs meet inside one sample interval)
    amax: float = 25.0
    # the JTC's quintic concentrates a turn: its peak acceleration runs
    # ~3x the mean over the interval and lands mostly in the shorter of the
    # two adjacent intervals, so the local bound budgets amax/5
    # (measured on the spline model: 39 rad/s^2 at a 60 deg junction
    # without it, review 2026-09-03)
    spline_peak_factor: float = 5.0
    # samples closer than this along the path are merged into a neighbour:
    # cuRobo's float32 output carries one-ulp near-duplicates (2e-7 rad)
    # that would otherwise become nanosecond intervals on the wire
    min_ds: float = 2e-4
    # uniform time dilation applied AFTER profiling (1.0 = as profiled): the
    # operator's whole-run slow mode (--speed-scale), not a per-leg speed.
    # Positions are untouched; the intervals stretch by 1/time_scale, so the
    # velocities annotating them scale by time_scale and the accelerations
    # by time_scale**2 (driver.dilate's rule). The driver's quintic fits
    # through all three: full-speed velocities on stretched intervals made
    # its reference wobble and run backwards, for 20-255 ms of a run at 0.5
    # and over a third of it at 0.3 and below (review 2026-10-27)
    time_scale: float = 1.0
    # an ABSOLUTE per-joint speed ceiling, rad/s (None: none), no margin
    # applied: it bounds what a leg's cruise fraction of vmax leaves and is
    # never scaled by it, so a slow leg it does not bind keeps its speed.
    # Written as a smaller vmax instead (0.5, which the 0.9 margin makes
    # 0.45 at full speed) it is 0.3 x 0.9 x 0.5 = 0.135 rad/s on a 0.3 line
    # (16:23's way in 7.25 -> 12.4 s, offline 2026-10-28). A compliant run
    # sets it (constants.COMPLIANT_QD_CEILING): in the driver's joint
    # impedance the wrist follows no faster than ~0.54 rad/s
    qd_ceiling: tuple = None
    # a chained junction's cruise fraction ceiling (None: none): neither its
    # sample nor the one on either side of it — the ends of the two
    # intervals the driver's quintic turns the junction in — cruises faster,
    # whatever the legs' speeds. Without it a junction takes the lower of its
    # two legs' fractions, so lines that cruise faster meet faster too, up to
    # what the corner bounds allow, and those budget a kink over both
    # adjacent intervals while the quintic turns it in the shorter one. A
    # compliant run sets it (constants.COMPLIANT_JUNCTION_SPEED): its
    # free-air lines cruise faster and still meet as they did at
    # FREE_LINE_SPEED
    junction_cruise: float = None


def _dedup(positions, min_ds):
    """Drop samples closer than min_ds (joint-space) to the last kept one:
    the planner's terminal hold, float32 one-ulp near-duplicates, and the
    shared junction point when legs are chained. The final sample is
    always kept (the goal); a sample is only ever DROPPED, never moved."""
    n = len(positions)
    keep = [0]
    for k in range(1, n):
        if np.linalg.norm(positions[k] - positions[keep[-1]]) >= min_ds:
            keep.append(k)
    if keep[-1] != n - 1:
        if len(keep) > 1 and np.linalg.norm(positions[n - 1] - positions[keep[-1]]) < min_ds:
            keep[-1] = n - 1  # replace the near-duplicate with the true goal
        else:
            keep.append(n - 1)
    return np.asarray(keep)


def _thin_junctions(q, seg, chord, window):
    """Keep samples near each chained junction at least `chord` apart (within
    `window` of path on both sides). A planner crowds its samples where a
    leg decelerates to rest; the JTC's quintic turns the kink inside the
    two intervals adjacent to it, and 1e-3 rad intervals made that a
    230-300 rad/s^2 kick (review 2026-09-03). Segment ends are never
    dropped."""
    n = len(q)
    if n < 3:
        return np.arange(n)
    ds = np.linalg.norm(np.diff(q, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(ds)])
    keep = np.ones(n, dtype=bool)
    for k in range(1, n):
        if seg[k] == seg[k - 1]:
            continue
        j = k - 1  # the kink: last sample of the earlier segment
        last = j
        m = j - 1
        while m > 0 and seg[m] == seg[j] and cum[j] - cum[m] <= window:
            if cum[last] - cum[m] < chord:
                keep[m] = False
            else:
                last = m
            m -= 1
        last = j
        m = k
        while m < n - 1 and seg[m] == seg[k] and cum[m] - cum[j] <= window:
            if cum[m] - cum[last] < chord:
                keep[m] = False
            else:
                last = m
            m += 1
    return np.flatnonzero(keep)


def concat_paths(trajs, min_ds=2e-4, junction_chord=0.0, junction_window=0.0):
    """Chained trajectories -> (positions (N,dof), segment index per sample).
    Junction duplicates are dropped; the shared sample belongs to the
    EARLIER segment (its speed is the min of both, see cruise_per_sample);
    the samples around each junction are thinned to real chords."""
    parts, seg = [], []
    for i, tr in enumerate(trajs):
        q = np.asarray([[float(v) for v in pt.positions] for pt in tr.points])
        parts.append(q)
        seg.extend([i] * len(q))
    q = np.vstack(parts)
    seg = np.asarray(seg)
    keep = _dedup(q, min_ds)
    q, seg = q[keep], seg[keep]
    if junction_chord > 0 and len(trajs) > 1:
        keep = _thin_junctions(q, seg, junction_chord, junction_window)
        q, seg = q[keep], seg[keep]
    return q, seg


def velocity_caps(positions, vmax, margin):
    """Per-sample path-speed cap so that every joint stays under margin x
    its limit: on interval i, v <= vmax_j * ds_i / |dq_ij| for all j."""
    dq = np.diff(positions, axis=0)
    ds = np.linalg.norm(dq, axis=1)
    vmax = np.asarray(vmax, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_joint = np.where(np.abs(dq) > 0, vmax[None, :] * ds[:, None] / np.abs(dq), np.inf)
    cap_int = margin * per_joint.min(axis=1)
    cap = np.empty(len(positions))
    cap[0], cap[-1] = cap_int[0], cap_int[-1]
    cap[1:-1] = np.minimum(cap_int[:-1], cap_int[1:])
    return cap, ds


def joint_caps(positions, vmax, params):
    """velocity_caps at params.vmax_margin, lowered where params.qd_ceiling
    binds (rad/s, no margin): the per-sample path-speed bound under which
    every interval keeps every joint below both. Returns (cap, ds)."""
    cap, ds = velocity_caps(positions, vmax, params.vmax_margin)
    if params.qd_ceiling is not None:
        cap = np.minimum(cap, velocity_caps(positions, params.qd_ceiling, 1.0)[0])
    return cap, ds


def corner_caps(positions, ds, params):
    """Per-sample speed cap from the direction change at that sample, and
    which samples are genuine stops (reversal apexes).

    Two bounds, the lower wins:
      - the turn measured over a fixed PATH window on each side (a
        planner's samples crowd together near a leg's end, so adjacent-
        sample directions there are noise) against the COMFORT turning
        acceleration corner_accel over the window's length;
      - the turn between the two ADJACENT intervals against the robot's
        own amax over their length — at a chained junction the kink is
        executed inside one sample interval, and a window-sized cap there
        commanded ~300 rad/s^2 (review 2026-09-03).
    A reversal (> reversal_deg) stops — only at its apex (the local
    maximum of the turn), never across its whole neighbourhood.
    Returns (cap, stop_mask)."""
    n = len(positions)
    cap = np.full(n, np.inf)
    turn = np.zeros(n)
    cum = np.concatenate([[0.0], np.cumsum(ds)])
    w = params.corner_window
    rev = math.radians(params.reversal_deg)

    def angle(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))

    def bound(theta, L, a):
        s_ = math.sin(theta / 2.0)
        return np.inf if s_ < 1e-6 else math.sqrt(a * L / (2.0 * s_))

    for k in range(1, n - 1):
        i = int(np.searchsorted(cum, cum[k] - w, side="right")) - 1
        i = max(0, min(i, k - 1))
        j = int(np.searchsorted(cum, cum[k] + w, side="left"))
        j = min(n - 1, max(j, k + 1))
        theta_w = angle(positions[k] - positions[i], positions[j] - positions[k])
        theta_l = angle(positions[k] - positions[k - 1], positions[k + 1] - positions[k])
        turn[k] = theta_l
        cap[k] = min(
            bound(theta_w, cum[j] - cum[i], params.corner_accel),
            bound(theta_l, ds[k - 1] + ds[k], params.amax / params.spline_peak_factor),
        )
    stop = np.zeros(n, dtype=bool)
    k = 1
    while k < n - 1:
        if turn[k] >= rev:
            j = k
            while j + 1 < n - 1 and turn[j + 1] >= rev:
                j += 1
            apex = k + int(np.argmax(turn[k : j + 1]))
            stop[apex] = True
            cap[apex] = 0.0
            k = j + 1
        else:
            k += 1
    return cap, stop


def cruise_per_sample(seg, speeds, junction_cruise=None):
    """Segment speed (the leg's cruise fraction) per sample; the sample a
    junction shares takes the lower of the two. Under `junction_cruise`
    (RetimeParams.junction_cruise) neither that sample nor the one on
    either side of it cruises faster than it."""
    speeds = np.asarray(speeds, dtype=float)
    cr = speeds[seg].astype(float)
    for k in range(1, len(seg)):
        if seg[k] != seg[k - 1]:
            cr[k - 1] = min(cr[k - 1], cr[k])
            if junction_cruise is not None:
                lo, hi = max(0, k - 2), min(len(seg), k + 1)
                cr[lo:hi] = np.minimum(cr[lo:hi], junction_cruise)
    return cr


def speed_profile(positions, cruise, vmax, params, v_start=0.0, v_end=0.0):
    """Path speed at every sample (rad/s of joint-space arc length)."""
    cap, ds = velocity_caps(positions, vmax, params.vmax_margin)
    ccap, _stop = corner_caps(positions, ds, params)
    target = np.minimum(np.minimum(np.asarray(cruise) * cap, ccap), cap)
    # the ceiling after the cruise fraction, never scaled by it; from here
    # on `cap` is the bound both set (the rounding's fill, the ends)
    cap, _ = joint_caps(positions, vmax, params)
    target = np.minimum(target, cap)
    n = len(positions)
    if params.smooth_pts > 0 and n > 2 * params.smooth_pts + 2:
        # round the tops of the TARGET (its lower envelope with a moving
        # average only ever lowers it); the bounded passes below then
        # guarantee the accel/decel envelope on the rounded target
        w = 2 * params.smooth_pts + 1
        finite = np.where(np.isfinite(target), target, cap)
        avg = np.convolve(finite, np.ones(w) / w, mode="same")
        inner = slice(params.smooth_pts, n - params.smooth_pts)
        target[inner] = np.minimum(target[inner], avg[inner])
    v = np.array(target, dtype=float)
    v[0] = min(v_start, cap[0])
    v[-1] = min(v_end, cap[-1])
    for k in range(1, n):  # ease out: bounded acceleration along the path
        v[k] = min(v[k], math.sqrt(v[k - 1] ** 2 + 2.0 * params.accel * ds[k - 1]))
    for k in range(n - 2, -1, -1):  # ease in: bounded (gentler) deceleration
        v[k] = min(v[k], math.sqrt(v[k + 1] ** 2 + 2.0 * params.decel * ds[k]))
    return v, ds


def _time_scale(params):
    """The slow-mode factor with the one clamp both of its uses share: the
    intervals stretch by 1/ts and the wire velocities shrink by ts, and a
    clamp on only one of them would put the two out of step."""
    return max(float(params.time_scale), 1e-3)


def times_from_profile(v, ds, params, cap=None):
    dt = ds / np.maximum(0.5 * (v[:-1] + v[1:]), params.v_floor)
    # an interval between two rest samples (a 2-point path, the two sides
    # of a reversal apex) is a triangle profile, not a crawl at v_floor —
    # bounded by the interval's per-joint cap (a trapezoid when it binds)
    both_rest = (v[:-1] < 1e-9) & (v[1:] < 1e-9)
    if both_rest.any():
        a, d = params.accel, params.decel
        vpk = np.sqrt(2.0 * ds[both_rest] * a * d / (a + d))
        if cap is not None:
            vcap = np.minimum(cap[:-1], cap[1:])[both_rest]
            over = vpk > vcap
            vpk = np.minimum(vpk, vcap)
            ramp_len = vpk**2 / (2.0 * a) + vpk**2 / (2.0 * d)
            cruise = np.where(over, (ds[both_rest] - ramp_len) / np.maximum(vpk, 1e-9), 0.0)
            dt[both_rest] = vpk / a + vpk / d + np.maximum(cruise, 0.0)
        else:
            dt[both_rest] = vpk / a + vpk / d
    dt = dt / _time_scale(params)
    t = np.empty(len(v))
    t[0] = params.first_dt
    t[1:] = params.first_dt + np.cumsum(dt)
    return t


def to_message(joint_names, positions, t, v=None):
    """Positions as given; velocities from the profile along the local
    tangent (never the central difference of a micro-interval);
    accelerations by central differences of those; rest at both ends."""
    n = len(positions)
    vel = np.zeros_like(positions)
    if n > 2:
        chord = positions[2:] - positions[:-2]
        if v is None:
            vel[1:-1] = chord / (t[2:] - t[:-2])[:, None]
        else:
            norm = np.maximum(np.linalg.norm(chord, axis=1), 1e-12)
            vel[1:-1] = chord / norm[:, None] * np.asarray(v)[1:-1, None]
    acc = np.zeros_like(positions)
    if n > 2:
        acc[1:-1] = (vel[2:] - vel[:-2]) / (t[2:] - t[:-2])[:, None]
    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    for k in range(n):
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in positions[k]]
        p.velocities = [float(x) for x in vel[k]]
        p.accelerations = [float(x) for x in acc[k]]
        set_stamp(p.time_from_start, float(t[k]))
        msg.points.append(p)
    return msg


def _segment_start_fracs(seg, t, n_legs):
    """Time fraction at which each input leg's samples begin.

    A leg whose samples all merged away (a zero-length move) inherits the
    previous leg's start, so the list always has one entry per input and
    never points past the end."""
    total = float(t[-1])
    out = []
    for i in range(n_legs):
        if i == 0:
            out.append(0.0)  # the motion starts when the goal does
            continue
        k = next((j for j in range(len(seg)) if seg[j] == i), None)
        if k is None:
            out.append(out[-1])
        else:
            out.append(float(t[k]) / total if total > 0 else 0.0)
    return out


def retime_group(trajs, speeds, vmax, params=RetimeParams()):  # noqa: B008 - frozen dataclass
    """Chained unguarded legs -> ONE re-timed JointTrajectory + a report.

    speeds: each leg's cruise fraction (its Leg.speed). Returns
    (trajectory, info) with info = {duration_s, junction_speeds (rad/s at
    each chained junction), stops (junctions taken at rest), n_points}.
    """
    positions, seg = concat_paths(trajs, params.min_ds, params.junction_chord, params.junction_window)
    if len(positions) < 2:
        return trajs[0], {"duration_s": 0.0, "junction_speeds": [], "stops": 0, "n_points": len(positions)}
    cruise = cruise_per_sample(seg, speeds, params.junction_cruise)
    v, ds = speed_profile(positions, cruise, vmax, params)
    cap, _ = joint_caps(positions, vmax, params)
    t = times_from_profile(v, ds, params, cap)
    junctions = [k for k in range(1, len(seg)) if seg[k] != seg[k - 1]]
    jspeeds = [float(v[k - 1]) for k in junctions]
    info = {
        "duration_s": float(t[-1]),
        "junction_speeds": jspeeds,
        "stops": int(sum(1 for s in jspeeds if s < params.v_floor * 2)),
        "n_points": int(len(positions)),
        # where each input leg begins as a TIME fraction of the result. A
        # guarded descent merged behind its approach arms its torque guard
        # here, so it watches its own regime and not the approach's
        # dynamics (runner._run_motion).
        "seg_start_fracs": _segment_start_fracs(seg, t, len(trajs)),
    }
    # v is the FULL-speed profile and t is stretched by 1/time_scale, so
    # the wire velocities shrink by the same factor; to_message
    # differentiates them over t, so the accelerations shrink by its
    # square. Only this copy is scaled: the report stays on the profile,
    # and slow mode counts the same junctions as stops as full speed does.
    return to_message(trajs[0].joint_names, positions, t, v * _time_scale(params)), info


def positions_to_traj(joint_names, positions):
    """A bare position path as a JointTrajectory (times are placeholders —
    retime_group reads positions only)."""
    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    for k, q in enumerate(positions):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.velocities = [0.0] * len(q)
        pt.accelerations = [0.0] * len(q)
        set_stamp(pt.time_from_start, (k + 1) * 0.02)
        msg.points.append(pt)
    return msg


def reverse_tail(traj, progress, live, arc_rad, min_ds=2e-4):
    """The executed tail of a descent, reversed, starting at `live`.

    A guard trip leaves the arm pressed against what it touched. The way
    out that needs no planner and no new collision check is the path it
    just flew, reversed — it was validated for this world and it leads
    directly away from the contact.

    The walk back starts at the sample NEAREST the live stop, never at
    the deepest planned one: the arm stopped short of that, and stepping
    "back" to it would drive INTO the contact (the 2026-09-02 re-press).
    `live` is then the first point, which is also what the executor's
    start gate wants. Returns None when there is nothing to reverse.
    """
    pts = list(traj.points)
    if len(pts) < 2:
        return None
    total = secs(pts[-1].time_from_start)
    cut = total * max(0.0, min(1.0, float(progress)))
    done = [p for p in pts if secs(p.time_from_start) <= cut]
    # cancel latency carries the arm past the last fully elapsed point
    if len(done) < len(pts):
        done.append(pts[len(done)])
    if len(done) < 2:
        return None
    from box_opening.runtime.branches import on_branch

    q = np.asarray([[float(v) for v in p.positions] for p in done])
    live = np.asarray(on_branch(live, q[-1]), dtype=float)  # (forward_tail)
    i = int(np.argmin(np.linalg.norm(q - live[None, :], axis=1)))
    out, arc = [q[i]], 0.0
    for k in range(i, 0, -1):
        arc += float(np.linalg.norm(q[k] - q[k - 1]))
        out.append(q[k - 1])
        if arc >= arc_rad:
            break
    if len(out) < 2:
        return None
    out = np.asarray(out)
    if np.linalg.norm(out[0] - live) > min_ds:
        out = np.vstack([live, out])  # the stop itself is the first point
    else:
        out[0] = live
    return out


def forward_tail(traj, live, arc_rad, min_ds=2e-4):
    """The UNEXECUTED continuation of a stopped descent: from the sample
    nearest `live` onward, until arc_rad of joint path is covered,
    starting at `live`.

    A guard stopped the arm partway down a planned, collision-checked
    stroke; the rest of that stroke is a validated path straight on
    through the contact. Cutting the push from it needs no planner call
    while the fingers sit on the button. Returns None when less than half
    the asked arc remains (the caller then plans the push instead).
    """
    from box_opening.runtime.branches import on_branch

    pts = list(traj.points)
    if len(pts) < 2:
        return None
    q = np.asarray([[float(v) for v in p.positions] for p in pts])
    # the reading is wrapped into (-pi, pi], the path is on the planner's
    # branch: a continuous joint past +/-pi would read 2*pi from its own path
    live = np.asarray(on_branch(live, q[0]), dtype=float)
    i = int(np.argmin(np.linalg.norm(q - live[None, :], axis=1)))
    out, arc = [q[i]], 0.0
    for k in range(i, len(q) - 1):
        arc += float(np.linalg.norm(q[k + 1] - q[k]))
        out.append(q[k + 1])
        if arc >= arc_rad:
            break
    if arc < 0.5 * arc_rad or len(out) < 2:
        return None
    out = np.asarray(out)
    if np.linalg.norm(out[0] - live) > min_ds:
        out = np.vstack([live, out])
    else:
        out[0] = live
    return out


def check_like_executor(msg, vmax, continuity_slack=3.0):
    """The gates the old planner-side executor applied: velocity limits,
    monotonic time, per-interval continuity. The kinova-gen3-ros2 driver
    applies none of them, so runtime/driver.py runs these in front of every
    goal. Returns a list of problems (empty = go)."""
    pos = np.asarray([[float(x) for x in p.positions] for p in msg.points])
    vel = np.asarray([[float(x) for x in p.velocities] for p in msg.points])
    t = np.asarray([secs(p.time_from_start) for p in msg.points])
    vmax = np.asarray(vmax, dtype=float)
    problems = []
    if (np.abs(vel).max(axis=0) > vmax * 1.01).any():
        problems.append("velocity exceeds limit")
    dts = np.diff(t, prepend=0.0)
    if (dts <= 0).any():
        problems.append("non-monotonic time_from_start")
    else:
        steps = np.abs(np.diff(pos, axis=0))
        if (steps > continuity_slack * vmax[None, :] * dts[1:, None]).any():
            problems.append("discontinuity")
    return problems
