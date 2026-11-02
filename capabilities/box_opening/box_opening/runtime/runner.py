"""The Runner: gates, merging, retries, run log (spec §6).

Dry-run is the default; execute=True alone arms motion — the human on the
physical e-stop is the gate, never a typed word (owner, 2026-09-16).
First unexpected outcome stops the task with the arm holding; the runner
never auto-continues past a cancel.
"""

import json
import math
import re
import time
from collections import namedtuple
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from box_opening import LOGGER as log
from box_opening.constants import (
    COMPLIANT_ACCEL,
    COMPLIANT_DECEL,
    COMPLIANT_JUNCTION_SPEED,
    COMPLIANT_QD_CEILING,
    CONTACT_SPEED,
    DRIFT_REPLAN_RAD,
    JOINT_VMAX,
    RECOIL_ARC_RAD,
    RECOIL_SPEED,
    SANITY_MARGIN_RAD,
    START_GATE_RAD,
    state_dir,
)
from box_opening.geometry import ang_diff
from box_opening.runtime.guards import (
    ARM_AFTER_CAP,
    GROUP_SETTLE_S,
    WARP_SETTLE_FRAC,
    CompensatedGuard,
    TorqueGuard,
    sanity_violations,
)
from box_opening.runtime.legs import (
    Kind,
    Leg,
    VerifyCtx,
    merge_groups,
)
from box_opening.runtime.planner import plan_record
from box_opening.runtime.retime import (
    RetimeParams,
    positions_to_traj,
    retime_group,
    reverse_tail,
)
from box_opening.runtime.stamps import secs
from box_opening.runtime.warp import warp_trajectory


@dataclass
class LegResult:
    leg_name: str
    outcome: str  # arrived | touch | failed | refused | skipped
    ok: bool
    detail: str = ""
    torque_peak: float = None
    progress: float = None
    t_wall: float = None
    # where the FINGERTIPS were when a guard tripped (TF, base frame): the
    # arm measuring the surface it touched, independent of the camera and
    # of every model constant
    contact_xyz: list = None
    # how the execution flew: "impedance" or "position", and the driver's
    # ImpedanceGains profile it was sent with (None in position)
    mode: str = None
    gains: int = None
    # ... and a custom spec's zeta (constants.CustomGains), which its name
    # does not say (RIGID flew at 0.7 from 14:21 to 16:23 2026-10-28): None
    # for a driver profile (the driver's own zeta) and in position
    gains_zeta: float = None
    # an arrival's error: the live joints the moment the driver called it
    # arrived against where the trajectory flown ends — the worst joint
    # (rad, wrap-aware), and the tool frame's offset (mm) and turn (deg) by
    # the arm's own kinematics. A goal ends when its time does, so in
    # impedance it is what the spring has not yet caught up. Measured, never
    # judged: 2026-10-27 16:23 flew at MEDIUM and TF put the tool 32 mm and
    # 5.3 deg off its commanded look pose, where that day's three position
    # runs (14:33, 15:19, 15:34) agree with one another to 0.1 mm
    end_err_rad: float = None
    end_err_mm: float = None
    end_err_deg: float = None
    # ... and each joint's (mrad), signed by its travel into the end: + past
    # it, - short of it (_signed_end_error), and the worst joint's name. The
    # 15:45 bag (2026-10-28): the look arrived joint_2 +45 mrad PAST, the
    # home joint_6 -253 SHORT, which end_err_rad reads alike
    end_err_mrad: list = None
    end_err_joint: str = None
    # the execution's side file, from the run record's folder: the trajectory
    # sent, point by point, and the driver's feedback of what flew
    # (Runner._keep_flight)
    flight: str = None
    # a gripper leg's knuckle when it finished (rad, 0.0 open .. 0.8 closed)
    gripper_pos: float = None
    # what tripped the guard (guards.TorqueGuard.trip_report): the joint, its
    # deviation from the baseline in force, and the run-up — so a false trip
    # can be read off the log instead of re-flown
    trip: dict = field(default=None, compare=False, repr=False)
    lookahead: object = field(default=None, compare=False, repr=False)
    # the trajectory this execution actually sent (a merged group's one
    # profile, a re-timed line): what `progress` is a fraction of, so the
    # recoil reverses the path flown, in its own time base
    flown_traj: object = field(default=None, compare=False, repr=False)


def _hold(seconds):
    """Hold still (the driver holds its last position). Its own function so
    a test can see the hold without waiting it out."""
    time.sleep(float(seconds))


def _arrival_error(live, traj):
    """(worst joint rad, tool mm, tool deg): the arm at `live` against the
    end of `traj`, the trajectory flown (LegResult.end_err_*). Joints
    wrap-aware — HOME's joint_3 sits 4.6 mrad from +-pi, either side of it
    the same arm — and the tool frame by the arm's own kinematics, which a
    chain that is not ready leaves None. A measurement only: nothing here
    may cost the run."""
    try:
        end = [float(v) for v in traj.points[-1].positions]
        live = [float(v) for v in live]
        rad = max(abs(ang_diff(a, b)) for a, b in zip(live, end, strict=False))
    except Exception:  # noqa: BLE001 - nothing to measure against
        return None, None, None
    try:
        from box_opening.runtime import approach
        from box_opening.runtime.kinematics import rotvec

        chain = approach._chain()
        R1, t1 = chain.fk(end)
        R2, t2 = chain.fk(live)
        mm = 1000.0 * float(np.linalg.norm(np.asarray(t1, float) - np.asarray(t2, float)))
        deg = math.degrees(float(np.linalg.norm(rotvec(np.asarray(R1, float) @ np.asarray(R2, float).T))))
    except Exception:  # noqa: BLE001 - no chain (no URDF for its root, ...): no tool error
        mm = deg = None
    return tuple(None if v is None or not math.isfinite(v) else float(v) for v in (rad, mm, deg))


# -- the flight log: how each execution arrived, and how the arm settled -------
# "Arrived" is the driver's trajectory clock running out: it reads no
# goal_tolerance (kinova-gen3-driver issue #17), so in the joint impedance
# the arm is still moving then. The 15:45 bag (2026-10-28, 49 Hz): the look
# arrived with joint_2 45-47 mrad PAST its end, the tool 14 mm off, and
# crept back for 0.55 s; the home arrived with joint_6 0.25 rad SHORT, the
# tool 86-88 mm off, and came within 2 mm at +1.15 s, 1 mm at +1.70 s. The
# record kept one unsigned worst joint at the arrival, which reads both
# alike, and nothing after it. Now each arrival's settle is measured over
# SETTLE_WINDOW_S after it, per joint and at the tool, from the
# /joint_states the client keeps (client.joint_history), and noted when the
# next goal goes out or the run ends (Runner.flush): never waited for.
SETTLE_WINDOW_S = 1.5
SETTLE_WITHIN_MM = (5.0, 2.0, 1.0)
# a joint's travel into the end of what flew is read where it last stood
# this far from the end: a joint that finished moving early holds there to
# float noise (cuRobo's float32 samples, an ulp 2.4e-7 rad), whose steps
# have no sign
TRAVEL_FROM_RAD = 1e-3

# an arrival whose settle is still to be noted (Runner._keep_flight): its
# execution's number and legs, when it arrived (time.monotonic(),
# time.time()), the reading end_err_* measured, the end of what flew and
# each joint's travel into it (_travel_signs)
_Settling = namedtuple("_Settling", "seq legs arrived live end signs")


def _wrapped(d):
    """ang_diff's wrap, (-pi, pi], over an array."""
    d = np.mod(np.asarray(d, float), 2.0 * math.pi)
    return np.where(d > math.pi, d - 2.0 * math.pi, d)


def _travel_signs(traj):
    """+1 or -1 for each joint: the way it travels into the end of `traj`,
    from the last sample TRAVEL_FROM_RAD or more off the end — its last
    step that is not float noise. +1 for a joint that never stands that
    far off: it does not move, and its error reads along its own axis."""
    pts = np.asarray([p.positions for p in traj.points], float)
    to_end = pts[-1] - pts
    signs = np.ones(pts.shape[1])
    for j in range(pts.shape[1]):
        far = np.flatnonzero(np.abs(to_end[:, j]) >= TRAVEL_FROM_RAD)
        if len(far):
            signs[j] = 1.0 if to_end[far[-1], j] > 0.0 else -1.0
    return signs


def _signed_end_error(live, traj):
    """(each joint's error in mrad, the worst joint's name): the arm at
    `live` against the end of `traj`, the trajectory flown, wrap-aware and
    signed by each joint's travel into the end (_travel_signs) — + past it,
    - short of it (LegResult.end_err_mrad). A measurement only: (None,
    None) when there is nothing to measure against."""
    try:
        err = _travel_signs(traj) * _wrapped(np.subtract(np.asarray(live, float), traj.points[-1].positions))
        worst = int(np.argmax(np.abs(err)))
        names = list(traj.joint_names)
    except Exception:  # noqa: BLE001 - nothing to measure against
        return None, None
    if not np.isfinite(err).all():
        return None, None
    name = names[worst] if len(names) == len(err) else "joint_%d" % (worst + 1)
    return [round(1000.0 * float(v), 1) for v in err], name


def _within(ts, mm, thr):
    """When the tool, `mm` from its end at the times `ts`, came within `thr`
    for good: where the line between its last reading outside and the next
    meets `thr`. 0.0 when it never was outside, None when the last reading
    still is."""
    if mm[-1] > thr:
        return None
    out = np.flatnonzero(mm > thr)
    if not len(out):
        return 0.0
    k = int(out[-1])
    return round(float(ts[k] + (ts[k + 1] - ts[k]) * (mm[k] - thr) / (mm[k] - mm[k + 1])), 3)


def settle_report(rows, end, signs, window_s, chain=None):
    """What the arm did over an arrival's settle window, `window_s` long,
    from `rows` — (s after the arrival, joints, joint velocities or None),
    in time order, the arrival's own reading first at 0 — against `end`,
    the joints the trajectory flown ends at. The settle note's fields:

      samples       the readings after the arrival's own; last_s, when the
                    last of them was measured (s after the arrival, 0.0
                    with none): what "for good" holds to; gap_ms, the
                    longest stretch of the window without one (its
                    resolution: the aim spins the node once a frame)
      peak_qd       each joint's fastest |qd| after the arrival (rad/s)
      past_mrad     each joint's furthest past the end along its travel
                    (`signs`, _travel_signs): + it went past by so much, -
                    it stayed short, so much at its closest
      max_mm        the tool's furthest from where `end` puts it; end_mm,
                    where the last reading puts it (what is left: a sag)
      within_Nmm_s  when the tool came within N mm (SETTLE_WITHIN_MM) for
                    good; None where the window's last reading is outside

    With no reading after the arrival's own (samples 0: nothing spun the
    node, as after a run's last motion) end_mm and within_Nmm_s are None:
    nothing says where the tool went, nor that it stayed. The tool by
    `chain` (ArmChain.tool_positions, else fk() row by row); without one
    its fields are None."""
    ts = np.asarray([r[0] for r in rows], float)
    qs = np.asarray([r[1] for r in rows], float)
    qds = [r[2] for r in rows[1:] if r[2] is not None]
    past = (np.asarray(signs, float) * _wrapped(qs - np.asarray(end, float))).max(axis=0)
    seen = len(rows) > 1
    out = {
        "samples": len(rows) - 1,
        "last_s": round(float(ts[-1]), 3),
        "gap_ms": round(1000.0 * float(np.diff(np.append(ts, max(float(window_s), ts[-1]))).max()), 1),
        "peak_qd": [round(float(v), 3) for v in np.abs(np.asarray(qds, float)).max(axis=0)] if qds else None,
        "past_mrad": [round(1000.0 * float(v), 1) for v in past],
    }
    mm = None
    if chain is not None:
        try:
            at = np.vstack([np.asarray(end, float), qs])
            many = getattr(chain, "tool_positions", None)
            p = np.asarray(many(at) if many is not None else [chain.fk(q)[1] for q in at], float)
            mm = 1000.0 * np.linalg.norm(p[1:] - p[0], axis=1)
        except Exception:  # noqa: BLE001 - no tool: the joints still say
            mm = None
    out["max_mm"] = None if mm is None else round(float(mm.max()), 1)
    out["end_mm"] = None if mm is None or not seen else round(float(mm[-1]), 1)
    for thr in SETTLE_WITHIN_MM:
        out["within_%gmm_s" % thr] = None if mm is None or not seen else _within(ts, mm, thr)
    return out


def _rounded(rows, places):
    """`rows`, lists of floats, rounded to `places` and as lists: in one
    pass where they are all as long (the side file is written between two
    motions)."""
    if rows and len({len(r) for r in rows}) == 1:
        return np.round(np.asarray(rows, float), places).tolist()
    return [[round(float(v), places) for v in r] for r in rows]


def _flight_doc(seq, names, outcome, flew, info, traj, arrived, every):
    """An execution's side file (Runner._keep_flight): the trajectory sent,
    point by point — t, q, qd, qdd, as the driver samples it — and every
    `every`-th of the driver's feedback (client.FEEDBACK_EVERY): s after
    the goal went out, the driver's fraction of the trajectory's time, and
    the joints it measured."""
    sent = info.get("sent_traj") or traj
    pts = list(sent.points)
    fb = list(info.get("feedback_q") or ())
    t0 = float(info["sent_at"])
    return {
        "seq": seq,
        "legs": list(names),
        "outcome": outcome,
        "mode": flew.get("mode"),
        "gains": flew.get("gains"),
        "gains_zeta": flew.get("gains_zeta"),
        "joint_names": list(sent.joint_names),
        "sent_t": round(arrived[1] - (arrived[0] - t0), 3),  # time.time() when the goal went out
        "ended_s": round(arrived[0] - t0, 4),  # when the client returned its result, s after that
        "trajectory": {
            "t": [round(secs(p.time_from_start), 4) for p in pts],
            "q": _rounded([list(p.positions) for p in pts], 6),
            "qd": _rounded([list(p.velocities) for p in pts], 5),
            "qdd": _rounded([list(p.accelerations) for p in pts], 4),
        },
        "feedback": {
            "every": every,
            "t": [round(float(s[0]) - t0, 4) for s in fb],
            "fraction": [round(float(s[1]), 5) for s in fb],
            "q": _rounded([list(s[2]) for s in fb], 6),
        },
    }


def _path_len(traj):
    """Joint-space arc length of a trajectory."""
    pts = list(traj.points)
    return sum(
        math.sqrt(sum((x - y) ** 2 for x, y in zip(a.positions, b.positions, strict=False)))
        for a, b in zip(pts, pts[1:], strict=False)
    )


def _rescale_contact_expectation(group, lead, flown):
    """Re-time a guarded tail's expected-contact fraction onto the merged
    path it is actually flying.

    The stroke's `contact_path_frac` says where along ITS OWN path contact
    is expected; the execution's progress counts the approach in front of
    it too. Without this the verify compares two different time bases and
    an honest contact reads as an early strike. The leg's own fraction is
    restored afterwards: a replan re-times it against its own fresh
    trajectory (_restore_execution_profile) before this runs again."""
    retime, frac = lead.retime, lead.contact_path_frac
    if retime is None or frac is None:
        return
    lengths = [_path_len(g.traj) for g in group]
    total = sum(lengths)
    if total <= 0:
        return
    before = sum(lengths[: group.index(lead)])
    lead.contact_path_frac = (before + float(frac) * lengths[group.index(lead)]) / total
    try:
        retime(flown)
    finally:
        lead.contact_path_frac = frac


def _restore_execution_profile(leg):
    """A replan swaps in a fresh planner-native trajectory; anything the
    original had baked in must be re-applied or honestly downgraded.

    Warped legs carry speed=1.0 as a do-not-dilate sentinel — executing
    an UNWARPED replacement at that sentinel is a full-speed descent
    into contact (review 2026-09-02). Re-warp the fresh trajectory; when
    it cannot be warped, run the WHOLE leg at the slow contact speed —
    never faster than intended. A leg with a retime hook (the press
    expects contact at a time fraction of the trajectory it will
    actually fly) gets it called on the final trajectory."""
    w = leg.warp
    if w is not None:
        fast, slow_speed, slow_frac = w[:3]
        # a 4th element: seconds into the slow zone the guard takes its
        # baseline AND arms (the staging press's free-air descent set it,
        # until 2026-10-28: a slow zone that starts from rest, where the
        # warp's own rule would baseline still)
        settle_s = w[3] if len(w) > 3 else None
        warped, arm_frac = warp_trajectory(leg.traj, slow_frac, fast, slow_speed)
        if arm_frac is None:
            leg.speed = float(slow_speed)
            leg.guard = replace(leg.guard, rebaseline_after=None)
        elif settle_s is not None:
            leg.traj = warped
            total = secs(warped.points[-1].time_from_start)
            armed_at = min(arm_frac + max(WARP_SETTLE_FRAC, settle_s / max(total, 1e-6)), ARM_AFTER_CAP)
            leg.guard = replace(leg.guard, rebaseline_after=armed_at, arm_after=armed_at)
        else:
            leg.traj = warped
            # arm_after rides with the rebaseline: the fresh trajectory has
            # its own time base, and a guard armed before its NEW slow zone
            # judges slow-zone efforts against the fast baseline — the lid-
            # released-mid-air trip (review 2026-09-02)
            arm_after = leg.guard.arm_after
            if arm_after is not None:
                arm_after = min(max(arm_after, arm_frac + 0.05), 0.95)  # settle, capped
            leg.guard = replace(leg.guard, rebaseline_after=arm_frac, arm_after=arm_after)
    if leg.retime is not None:
        leg.retime(leg.traj)


def _free_air_line(leg):
    """A plan-free line flown on its own, slower than full speed, whose guard
    (if any) judges an OBSTRUCTION against a free-air rest reference and is
    armed from the first sample: nothing about it — no arming fraction, no
    rebaseline, no contact-time expectation, no recoil — depends on the
    trajectory's time base, so it may fly a re-timed profile instead of a
    dilated one."""
    g = leg.guard
    return (
        leg.plan_free
        and leg.warp is None
        and leg.guard_from is None
        and leg.retime is None
        and leg.contact_path_frac is None
        and 0.0 < float(leg.speed) < 1.0
        and (
            g is None
            or (
                g.trip == "obstruction"
                and g.reference is not None
                and g.gravity is not None
                and g.arm_after is None
                and g.rebaseline_after is None
            )
        )
    )


def _path_speed_peak(traj):
    pts = traj.points
    best = 0.0
    for a, b in zip(pts, pts[1:], strict=False):
        dt = secs(b.time_from_start) - secs(a.time_from_start)
        if dt > 0.0:
            ds = math.sqrt(sum((x - y) ** 2 for x, y in zip(a.positions, b.positions, strict=False)))
            best = max(best, ds / dt)
    return best


def line_profile(leg, params, time_scale=1.0):
    """The free-air line `leg` as a re-timed profile, or None to fly it as
    before (dilated by its speed).

    Dilating a full-speed profile by 1/speed scales its accelerations by
    speed**2: at CONTACT_SPEED 0.15 the line eases in and out at 0.135 and
    0.068 rad/s^2 and a 15 cm line is all ramp (4.55 s). Here the SAME
    positions are re-timed with the accelerations scaled by speed (not
    speed**2) and the cruise lowered until no sample's path speed exceeds
    the dilated profile's peak: never faster than it flies today, only less
    time spent getting up to speed and back down. Under a speed ceiling (a
    compliant run's, Runner.retime_params) the profile flies even where it
    takes the longer: the dilated line eases at 6 x speed**2 and knows no
    ceiling — at 0.55 a 0.6 rad joint_6 line would run that joint at 0.61
    rad/s, and above 0.5 its ease is harsher than the compliant one."""
    from box_opening.runtime.driver import dilate

    s = float(leg.speed)
    ref = dilate(leg.traj, s * float(time_scale))
    v_max = _path_speed_peak(ref)
    p = replace(params, accel=params.accel * s, decel=params.decel * s, time_scale=float(time_scale))
    cruise = s
    for _ in range(6):
        traj, _info = retime_group([leg.traj], [cruise], JOINT_VMAX, p)
        peak = _path_speed_peak(traj)
        if peak <= v_max * (1.0 + 1e-6):
            if p.qd_ceiling is not None:
                return traj
            return traj if secs(traj.points[-1].time_from_start) < secs(ref.points[-1].time_from_start) else None
        cruise *= 0.98 * v_max / peak
    return None


class Runner:
    def __init__(self, client, log_dir=None, margin_rad=SANITY_MARGIN_RAD):
        self.client = client
        self.margin_rad = margin_rad
        self._log_dir = Path(log_dir if log_dir is not None else state_dir() / "runs")
        self._log_path = None
        # an overlapped gripper command that outlives a run(): sent at fix
        # commit or on arrival at the hop, joined lazily before anything
        # that needs the fingers settled (audit 2026-09-02)
        self._pending = None  # (leg, handle, t0)
        self._sent_early = set()  # ids of gripper legs dispatched under a motion
        # the human-motion profile for unguarded groups (retime.py); the
        # mission may replace it with the container's motion: block. Every
        # re-timing reads it through retime_params (a compliant run's own)
        self.retime = RetimeParams()
        self.last_retime = None  # report of the most recent re-timed group
        # whole-run slow mode (--speed-scale): dilates every motion, guarded
        # strokes included — a first attempt at a new placement runs at a
        # fraction of speed without touching any per-leg tuning
        self.time_scale = 1.0
        # the flight log: the executions sent so far, which number their side
        # files, and the last arrival, its settle noted when the next goal
        # goes out (_keep_flight, _note_settle)
        self._flights = 0
        self._settling = None

    def retime_params(self):
        """What this runner re-times a motion with: every group it profiles
        (a guarded tail rides its approach's), a free-air line's profile and
        the recoil, in flight and in the preview alike, so what the preview
        shows is what flies; slow mode is the caller's (time_scale). A
        compliant client's (client.compliant) eases out of rest at
        COMPLIANT_ACCEL and into it at COMPLIANT_DECEL, no interval runs a
        joint past COMPLIANT_QD_CEILING, and no junction cruises faster than
        COMPLIANT_JUNCTION_SPEED: what the arm follows in the driver's joint
        impedance (constants). The flag is read at every call, never kept:
        press_demo builds the runner before run_reach_press sets it.
        Positions are never touched. A stroke
        flown on its own under a contact guard keeps its dilated or warped
        timing, tuned to the contact (no compliant leg carries a guard). A
        guarded run, which flies in position, keeps self.retime as it is,
        as every other position run does."""
        if not getattr(self.client, "compliant", False):
            return self.retime
        return replace(
            self.retime,
            accel=COMPLIANT_ACCEL,
            decel=COMPLIANT_DECEL,
            qd_ceiling=COMPLIANT_QD_CEILING,
            junction_cruise=COMPLIANT_JUNCTION_SPEED,
        )

    # -- preview -----------------------------------------------------------
    def preview(self, legs):
        rows = [
            "%-18s %-7s %9s %-18s %5s %8s %7s %7s"
            % (
                "leg",
                "kind",
                "speed",
                "world",
                "chain",
                "time_s",
                "plan_s",
                "solve_s",
            )
        ]
        plan_total = solve_total = 0.0
        # the time an UNGUARDED leg flies is its re-timed profile's, not
        # the planner's duration over its speed: preview what will run
        flown = {}
        for group in merge_groups(legs):
            if group[0].kind is Kind.MOTION and group[0].guard is None and all(g.traj is not None for g in group):
                try:
                    _, info = retime_group(
                        [g.traj for g in group],
                        [g.speed for g in group],
                        JOINT_VMAX,
                        replace(self.retime_params(), time_scale=self.time_scale),
                    )
                    per = info["duration_s"] / len(group)
                    for g in group:
                        flown[id(g)] = per if len(group) > 1 else info["duration_s"]
                except Exception:
                    pass
        for leg in legs:
            flies = "-"
            line = (
                line_profile(leg, self.retime_params(), self.time_scale)
                if leg.kind is Kind.MOTION and leg.traj is not None and _free_air_line(leg)
                else None
            )
            if id(leg) in flown:
                flies = "%.2f" % flown[id(leg)]
            elif line is not None:
                flies = "%.2f" % secs(line.points[-1].time_from_start)
            elif leg.kind is Kind.MOTION and leg.traj is not None:
                flies = "%.2f" % (secs(leg.traj.points[-1].time_from_start) / (leg.speed * self.time_scale))
            speed_txt = "%5.2f" % leg.speed
            if leg.warp:
                # the profile is baked into the timing; showing 1.00 would
                # read as "transit speed" when it is fast-then-contact
                speed_txt = "%.2f>%.2f" % (leg.warp[0], leg.warp[1])
            plan_total += leg.plan_s or 0.0
            solve_total += leg.plan_server_s or 0.0
            rows.append(
                "%-18s %-7s %9s %-18s %5d %8s %7s %7s"
                % (
                    leg.name,
                    leg.kind.value,
                    speed_txt,
                    leg.world,
                    leg.chain,
                    flies,
                    "-" if leg.plan_s is None else "%.2f" % leg.plan_s,
                    "-" if leg.plan_server_s is None else "%.2f" % leg.plan_server_s,
                )
            )
        if plan_total > 0.0:
            # round trip vs. what the planner says it spent solving: the
            # difference is transport/queueing, and it is the half we have
            # not yet accounted for (see Leg.plan_s)
            rows.append(
                "%-18s %-7s %9s %-18s %5s %8s %7.2f %7.2f"
                % ("(planning total)", "", "", "", "", "", plan_total, solve_total)
            )
        return "\n".join(rows)

    # -- gates (checked BEFORE anything executes) ---------------------------
    def _refusal(self, leg):
        if not self.client.motion_enabled():
            # the driver executes whatever it is sent — it has no dry-run
            # parameter the way the old planner-side executor did — so the
            # client's latch, set only by --execute, is the software gate
            return "motion disabled: the client was not armed (--execute)"
        if leg.kind is Kind.MOTION:
            # Transit-speed legs require the fullest world KNOWN: "full"
            # (bench + container), or "bench" in the pre-detection epoch —
            # a container cannot be modeled before one is seen (the look,
            # the sweeps, a no-box home). Slow (contact-speed) unguarded legs are
            # the one exception: a post-contact retreat must plan against
            # the interaction world its descent used — in the full world
            # its start would read as inside the container.
            if (
                leg.guard is None
                and leg.speed > CONTACT_SPEED
                and not leg.world.startswith(("full", "bench"))
                # retreat and lift both ascend OUT of the corridor just
                # descended, planned collision-free in that same
                # interaction world — faster-than-contact is fine for them
                # (owner: fast up, 2026-08-26; lift added 2026-08-28).
                # Retreat already runs at TRANSIT_SPEED here, so a 0.35
                # lift is the milder of the two.
                and not (
                    # a recoil reverses the corridor just descended — the
                    # same case as retreat/lift, by construction
                    leg.name.startswith(("retreat", "lift", "recoil")) and leg.world.startswith("interaction")
                )
            ):
                return (
                    "transit-speed MOTION leg planned against %r — full or "
                    "bench world required (spec §6)" % leg.world
                )
            if leg.traj is not None:
                bad = sanity_violations(leg.traj, self.margin_rad)
                if bad:
                    return "trajectory sanity gate: " + "; ".join(bad)
            if leg.guard is not None and not self.client.efforts_present():
                return "no effort fields in /joint_states — guarded legs " "refuse to run (spec §6)"
        return None

    # -- run ----------------------------------------------------------------
    def run(self, legs, execute, lookahead=None):
        """Execute legs in merge groups.

        `lookahead(end_joints)` — builds the NEXT phase's legs while this
        run's last unguarded motion flies (planned from that motion's
        predicted end joints). Its return value lands in
        self.lookahead_result; a failure leaves None and the caller builds
        after the run as before. Never hosted on a guarded stroke."""
        self.lookahead_result = None
        log.info(self.preview(legs))
        if not execute:
            log.info("dry-run complete — nothing moved (add --execute)")
            return [LegResult(leg.name, "skipped", True) for leg in legs]
        # --execute alone arms it, for every CLI (owner 2026-09-16: "remove
        # the need to type yes"): the human on the physical e-stop is the
        # gate, Ctrl+C stops everything. This line is printed once per run
        # attempt; the abort drill counts it.
        log.info("EXECUTING (human on the physical e-stop; Ctrl+C stops everything)")

        for leg in legs:
            why = self._refusal(leg)
            if why:
                log.info("REFUSED: %s — %s" % (leg.name, why))
                res = LegResult(leg.name, "refused", False, why)
                self._log(res, leg)
                return [res]

        results = []
        after_touch = False  # the previous motion stopped ON something
        next_chain = max((leg.chain for leg in legs), default=0) + 1
        groups = merge_groups(legs)
        host = None  # the last unguarded MOTION group hosts the lookahead
        if lookahead is not None:
            for g in reversed(groups):
                if g[0].kind is Kind.MOTION and all(x.guard is None for x in g):
                    host = g
                    break
        for gi, group in enumerate(groups):
            lead = group[0]
            # Worlds are pushed at PLAN time (core._plan_motion) and by the
            # replan path per leg; execution itself never consults the
            # collision world, so nothing is pushed here.

            # A deferred gripper command must be settled before anything
            # that depends on the fingers having arrived: another gripper
            # leg, or any guarded motion (the press descends with them
            # closed). Everything else — a plain transit, a replan — is
            # exactly what we want it to overlap with.
            if self._pending is not None and (lead.kind is Kind.GRIPPER or any(g.guard is not None for g in group)):
                res = self._join_pending()
                results.append(res)
                if not res.ok:
                    log.info("STOP: leg %s -> %s (%s) — arm holds" % (res.leg_name, res.outcome, res.detail))
                    return results

            if lead.kind is Kind.GRIPPER:
                if id(lead) in self._sent_early:
                    continue  # dispatched under the previous motion; pending
                if lead.send_when_still:
                    self.wait_still(lead.name)  # what the fingers close on stops moving first
                    if lead.when_still is not None:
                        try:
                            lead.when_still()
                        except Exception as e:  # what rides the wait is its caller's, never the leg's
                            log.info(
                                "[runner] %s: what was to run once the arm stood still failed (%s)" % (lead.name, e)
                            )
                if lead.defer_join and lead.verify is None:
                    handle = self.client.gripper_send(lead.gripper_cmd)
                    if handle is not None:
                        self._pending = (lead, handle, time.monotonic())
                        continue
                    # send failed — fall through to the blocking path so
                    # the failure is reported the same way as ever
                res = self._run_gripper(lead)
            else:
                # Free-air drift replans on MEASURED drift only (0.04 rad;
                # owner: fewer pauses, 2026-08-26). But after a TOUCH the
                # predicted start is wrong BY DESIGN — the guard stopped
                # the arm early — and a sub-gate joint delta is still a
                # multi-mm Cartesian shove INTO the thing just touched:
                # the pre-planned retreat re-pressed the button at full
                # speed, guardless, and stalled the arm (field
                # 2026-09-02). Post-touch, replan from live always; the
                # arm stop-and-holds at contact for the ~0.6 s it costs.
                if (after_touch or self._drifted(group)) and any(g.plan_free for g in group):
                    # a plan-free line (the arm's own straight line, built
                    # from live joints) re-planned by the planner would be a
                    # bowed path under a straight line's guard: fail it, the
                    # caller rebuilds it from where the arm is
                    res = LegResult(
                        lead.name,
                        "refused",
                        False,
                        "a plan-free line no longer starts where the arm is — rebuild it from live joints",
                    )
                    self._log(res, lead)
                    results.append(res)
                    log.info("STOP: leg %s -> refused (%s) — arm holds" % (res.leg_name, res.detail))
                    return results
                if after_touch or self._drifted(group):
                    # a later leg's plan failing from the fresh chain is
                    # usually an unlucky cuRobo family draw (the
                    # 2026-09-01 INVALID_START home was one) — fresh
                    # draws are cheap, so try thrice before giving up;
                    # deterministic refusals just fail three times
                    for _attempt in range(3):
                        regroup, next_chain = self._replan_group(group, next_chain)
                        if regroup is not None:
                            break
                        log.info("[runner] replan of %s failed (attempt %d/3)" % (group[0].name, _attempt + 1))
                    group = regroup
                    if group is None:
                        res = LegResult(lead.name, "failed", False, "re-plan from live state failed")
                        self._log(res, lead)
                        results.append(res)
                        return results
                    lead = group[0]
                # A group was merged on the trajectories its legs had THEN. A
                # lazy group had none, and a replan (above, or under the
                # recoil) plans each leg from the predicted end of the one
                # before it, which the client hands the planner on HOME's
                # branch: a whole turn from that end when it is written the
                # other way round (legs.MERGE_GAP_RAD). Replayed offline
                # 2026-10-28, a lazy retreat and home flew as one goal that
                # turned joint_3 9.23 rad. One goal flies only legs that
                # meet; the rest fly next, as goals of their own
                parts = merge_groups(group)
                if len(parts) > 1:
                    for before, after in zip(parts, parts[1:], strict=False):
                        a, b = before[-1], after[0]
                        gap = (
                            np.subtract(b.traj.points[0].positions, a.traj.points[-1].positions)
                            if a.traj is not None and b.traj is not None
                            else [float("nan")]
                        )
                        log.info(
                            "[runner] %s no longer starts where %s ends (%.3f rad apart as written) — it flies "
                            "as a goal of its own" % (b.name, a.name, float(np.max(np.abs(gap))))
                        )
                    groups[gi + 1 : gi + 1] = parts[1:]
                    if host is group:
                        host = parts[-1]
                    group, lead = parts[0], parts[0][0]
                # a RELEASE overlaps the replan above, never the motion:
                # the arm must not move away from what it dropped before
                # the fingers have settled
                if self._pending is not None and self._pending[0].join_before_motion:
                    res = self._join_pending()
                    results.append(res)
                    if not res.ok:
                        log.info("STOP: leg %s -> %s (%s) — arm holds" % (res.leg_name, res.outcome, res.detail))
                        return results
                after_touch = False
                hook = None
                if group is host:
                    end = list(group[-1].goal_joints)

                    def hook(_end=end):
                        return lookahead(_end)

                # a deferred gripper leg flagged to ride THIS motion is sent
                # the moment the goal is accepted (the fingers move while
                # the arm moves); the usual join still gates what follows
                early = None
                if gi + 1 < len(groups):
                    cand = groups[gi + 1][0]
                    if (
                        cand.kind is Kind.GRIPPER
                        and cand.defer_join
                        and cand.verify is None
                        and cand.send_with_previous_motion
                        and not cand.send_when_still
                        and self._pending is None
                        and all(g.guard is None for g in group)
                    ):
                        early = cand
                if early is not None:
                    inner = hook

                    def hook(_leg=early, _inner=inner):
                        handle = self.client.gripper_send(_leg.gripper_cmd)
                        if handle is not None:
                            self._pending = (_leg, handle, time.monotonic())
                            self._sent_early.add(id(_leg))
                        return _inner() if _inner is not None else None

                res = self._run_motion(group, while_running=hook)
                if group is host and res.lookahead is not None:
                    self.lookahead_result = res.lookahead
                after_touch = res.outcome == "touch"
            for g in group:
                self._log(res, g)
            results.append(res)
            if not res.ok:
                log.info("STOP: leg %s -> %s (%s) — arm holds" % (res.leg_name, res.outcome, res.detail))
                return results
            # the press semantics belong to the group's GUARDED member: a
            # press merged as the tail of its approach still holds and
            # recoils (group[0] is then the unguarded approach)
            owner = next((g for g in group if g.guard is not None), lead)
            if (
                owner.kind is Kind.MOTION
                and owner.guard is not None
                and owner.guard.trip == "press"
                and res.outcome in ("touch", "arrived")
            ):
                # a good press leaves the arm on the button — whether the
                # push met a stop or ran its full bound — while the next leg
                # is planned: recoil along the stroke first. A FAILED leg is
                # left holding where it struck — the operator needs to see
                # that.
                if owner.hold_s > 0.0:
                    _hold(owner.hold_s)  # held on the button: the driver holds position, still pressing
                nxt = next((g for g in groups[gi + 1 :] if g[0].kind is Kind.MOTION), None)
                handled, moved, next_chain = self._reflex_recoil(owner, res, nxt, next_chain, results)
                if moved:
                    # the recoil moved the arm: a next group its own hook did
                    # not plan from the recoil's end was planned from where
                    # the press stopped — sent as-is, the driver's first
                    # waypoint would put the arm back down on the button,
                    # unguarded. Replan it from live.
                    after_touch = not handled
        # a pending gripper command deliberately OUTLIVES the run: the next
        # phase's planning is what it overlaps with (finish() collects it)
        return results

    def start_gripper(self, name, cmd, execute, world="full"):
        """Dispatch a gripper command NOW, outside any leg sequence, to
        overlap the planning that follows (the press close goes out the
        moment the fix commits, audit 2026-09-02). Same gates as a leg;
        joined lazily like any deferred command. Returns False if refused."""
        leg = Leg(
            name=name,
            kind=Kind.GRIPPER,
            traj=None,
            speed=0.0,
            guard=None,
            world=world,
            chain=0,
            target=None,
            goal_joints=None,
            gripper_cmd=cmd,
            defer_join=True,
        )
        if not execute:
            log.info("[runner] %s: dry-run, not sent" % name)
            return True
        why = self._refusal(leg)
        if why:
            log.info("REFUSED: %s — %s" % (name, why))
            return False
        if self._pending is not None:
            res = self._join_pending()
            if not res.ok:
                return False
        handle = self.client.gripper_send(cmd)
        if handle is None:
            res = self._run_gripper(leg)  # send failed: blocking path, honest
            self._log(res, leg)
            return res.ok
        self._pending = (leg, handle, time.monotonic())
        return True

    def join_pending(self):
        """Collect a pending gripper command now, mid-run (None when nothing
        is pending): where what comes next needs the fingers where it sent
        them and run() would not join it first — it joins before a gripper
        leg, a guarded motion or a join_before_motion leg's motion only, and
        a compliant way in is none of them."""
        if self._pending is None:
            return None
        return self._join_pending()

    def finish(self):
        """Collect any pending gripper command (mission end / exit)."""
        return self.join_pending()

    def wait_still(self, before):
        """Wait, the arm at rest, until it stands still (client.wait_still),
        before `before`, which wants it so: a camera read at rest — the aim,
        the spot look — or a gripper leg sent once the arm is still
        (Leg.send_when_still, the grip's close). Noted 'still': how long it
        took (settle_s) and whether it timed out, the run carrying on all
        the same. Never moves or stops anything: a wait between motions,
        never during one. A client that keeps no stream to wait on waits
        for nothing. Returns (settle_s, timed_out), or None."""
        wait = getattr(self.client, "wait_still", None)
        if wait is None:
            return None
        settle_s, timed_out = wait()
        self.note("still", before=before, settle_s=round(float(settle_s), 3), timed_out=bool(timed_out))
        return settle_s, timed_out

    def _join_pending(self):
        leg, handle, t0 = self._pending
        self._pending = None
        res = self._join_gripper(leg, handle, t0)
        self._log(res, leg)
        return res

    # -- helpers -------------------------------------------------------------
    def _drifted(self, group):
        if any(g.traj is None for g in group):
            return True  # lazy: planned here, from live, for the first time
        start = group[0].traj.points[0].positions
        live = self.client.joints()
        return max(abs(ang_diff(a, b)) for a, b in zip(live, start, strict=False)) > DRIFT_REPLAN_RAD

    def _reflex_recoil(self, leg, res, next_group, next_chain, results):
        """Back off a press contact along the path just flown, at once.

        The guard stops the arm ON the button and the considered retreat
        then takes 0.5-1.3 s to plan; a person recoils in about a tenth of
        a second. Reversing the descent's own executed tail needs no
        planner and no new collision check — that path was flown
        milliseconds ago, in this world, and it leads directly away from
        what was touched. The considered leg is planned WHILE the recoil
        flies, from the recoil's own end, so the arm is off the button
        before the planner is even asked.

        A push is short (button_travel_m) and starts where the touch
        stopped: reversing it alone ends on the touch's contact pose. The
        leg's `recoil_via` (the touch's trajectory) carries the recoil on
        back up the touch stroke until RECOIL_ARC_RAD is covered.

        Returns (next_group was planned from the recoil's end, the arm
        moved, next_chain).
        """
        flown = res.flown_traj if res.flown_traj is not None else leg.traj
        if flown is None or res.progress is None:
            return False, False, next_chain
        path = reverse_tail(flown, res.progress, self.client.joints(), RECOIL_ARC_RAD)
        if path is None:
            return False, False, next_chain
        path = self._recoil_on_up_the_touch(leg, path)
        traj, _info = retime_group(
            [positions_to_traj(flown.joint_names, path)],
            [RECOIL_SPEED],
            JOINT_VMAX,
            replace(self.retime_params(), time_scale=self.time_scale),
        )
        reflex = Leg(
            name="recoil",
            kind=Kind.MOTION,
            traj=traj,
            speed=1.0,  # the profile is baked in
            guard=None,
            world=leg.world,
            world_path=leg.world_path,
            chain=next_chain,
            target=None,
            goal_joints=[float(v) for v in path[-1]],
        )
        why = self._refusal(reflex)
        if why:
            log.info("REFUSED recoil — %s" % why)
            return False, False, next_chain
        hook = None
        if next_group is not None:

            def hook(_end=list(reflex.goal_joints)):  # noqa: B006 - binds the goal now
                return self._replan_group(next_group, next_chain + 1, start=_end)

        self._note_settle()  # the last arrival's, its window cut by this goal
        t0 = time.monotonic()
        try:
            outcome, info = self.client.execute(traj, 1.0, guard=None, while_running=hook)
        except KeyboardInterrupt:
            self._keep_aborted([reflex.name], traj, t0)
            raise
        arrived = (time.monotonic(), time.time())
        ok = outcome == "arrived"
        flew = self._how_it_flew(outcome, info, traj)
        rres = LegResult(
            reflex.name,
            outcome,
            ok,
            info.get("message", ""),
            progress=info.get("progress"),
            t_wall=arrived[0] - t0,
            flight=self._keep_flight([reflex.name], outcome, flew, info, traj, arrived),
            **flew,
        )
        planned = False
        if ok and next_group is not None:
            regroup, _chain = info.get("while_running") or (None, None)
            planned = regroup is not None
        self._log(rres, reflex)
        if ok:
            results.append(rres)
            log.info(
                "  recoil — off the contact in %.2f s%s"
                % (
                    rres.t_wall or 0.0,
                    " (next leg planned during it)" if planned else "",
                )
            )
        else:
            # not fatal: the arm is somewhere along a path it just flew; the
            # next leg replans from live (run() forces it: the arm moved)
            log.info("  recoil did not complete (%s) — the next leg replans from live" % (rres.detail or outcome))
        return (ok and planned), True, (next_chain + 2 if planned else next_chain + 1)

    @staticmethod
    def _recoil_on_up_the_touch(leg, path):
        """`path` (the press's reversed tail) carried on back along the
        touch stroke it continued (leg.recoil_via) until RECOIL_ARC_RAD of
        joint arc is covered. Only when the touch's path passes within
        START_GATE_RAD of where the reversed press ends, on the same
        branch: anything else is not the stroke this press continued."""
        via = leg.recoil_via
        if via is None:
            return path
        path = np.asarray(path, float)
        arc = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if arc >= RECOIL_ARC_RAD - 1e-3:
            return path
        ext = reverse_tail(via, 1.0, path[-1], RECOIL_ARC_RAD - arc)
        if ext is None or len(ext) < 2 or float(np.linalg.norm(ext[0] - path[-1])) > 1e-6:
            return path
        if float(np.linalg.norm(ext[1] - ext[0])) > START_GATE_RAD:
            return path
        if len(ext) > 2 and np.linalg.norm(ext[2] - ext[0]) < np.linalg.norm(ext[2] - ext[1]):
            # the touch's sample nearest the press's start lies just PAST it
            # (further down the stroke): skip it, never a step back down
            ext = np.vstack([ext[:1], ext[2:]])
        # reverse_tail stops at the first SAMPLE past its arc: end exactly
        # on the budget, however far apart the touch's samples are
        left, out = RECOIL_ARC_RAD - arc, [ext[0]]
        for a, b in zip(ext, ext[1:], strict=False):
            d = float(np.linalg.norm(b - a))
            if d >= left:
                if left > 1e-9:
                    out.append(a + (b - a) * (left / d))
                break
            out.append(b)
            left -= d
        return np.vstack([path, np.asarray(out)[1:]]) if len(out) > 1 else path

    def _replan_group(self, group, next_chain, start=None):
        """Re-plan each leg of the group from live state (or from `start`:
        a recoil plans the next leg from its own predicted end while it is
        still flying), same targets."""
        live = self.client.joints() if start is None else list(start)
        if any(leg.plan_free for leg in group):
            # the arm's own straight line, guarded for that line: the
            # planner's path is not it (the recoil's lookahead lands here too)
            log.info(
                "REFUSED replan of %s — a plan-free line is rebuilt from live joints, never re-planned" % group[0].name
            )
            return None, next_chain
        for leg in group:
            # Push THIS leg's world before re-planning it. Without this the
            # replan used whatever world happened to be loaded — for the
            # merged [retreat, home] group that meant re-planning `home` at
            # transit speed against an interaction world, whose obstacles
            # are deliberately capped (review 2026-08-28).
            ok, msg = self.client.set_world(leg.world_path or leg.world)
            if not ok:
                log.info("REFUSED replan of %s — set_world failed: %s" % (leg.name, msg))
                return None, next_chain
            kind, *rest = leg.target
            n_refused = len(getattr(self.client, "refusals", ()))
            t_plan = time.monotonic()
            if kind == "pose":
                off = float(rest[2]) if len(rest) > 2 else 0.0
                plan = self.client.plan_to_pose(rest[0], rest[1], live, approach_offset_m=off)
            else:
                plan = self.client.plan_to_joints(rest[0], live)
            plan_s = time.monotonic() - t_plan
            # the run record keeps what was planned at flight: the plan this
            # leg now flies, or the one that failed, in a note of its own —
            # the failure's row shows the leg's old plan, or none (lazy)
            record = plan_record(plan, leg.target, plan_s, getattr(self.client, "refusals", ())[n_refused:])
            if plan is None or not plan.success:
                self.note("replan_failed", name=leg.name, plan=record)
                return None, next_chain
            # replanned trajectories pass the SAME sanity gate as pre-built
            # ones — a wandering replan executed ungated defeats spec §6
            bad = sanity_violations(plan.trajectory, self.margin_rad)
            if bad:
                log.info("REFUSED replan of %s — %s" % (leg.name, "; ".join(bad)))
                return None, next_chain
            leg.traj = plan.trajectory
            leg.plan = dict(record, replanned=True)
            leg.chain = next_chain
            leg.goal_joints = list(plan.trajectory.points[-1].positions)
            # indices into the OLD trajectory mean nothing in this one
            leg.waypoint = getattr(plan, "waypoint", None)
            leg.guard_from = None if leg.waypoint is None else int(leg.waypoint[0])
            _restore_execution_profile(leg)
            # the pre-execution gates ran against the ORIGINAL trajectory;
            # a replan produces a new one and must clear them again
            why = self._refusal(leg)
            if why:
                log.info("REFUSED replan of %s — %s" % (leg.name, why))
                return None, next_chain
            live = leg.goal_joints
        return group, next_chain + 1

    def _settle_frac(self, flown_s):
        """The fraction of an execution lasting `flown_s` (as flown, slow
        mode included) that a guard waits after a junction or a launch
        before it takes its baseline and arms.

        GROUP_SETTLE_S is wall time at FULL operator speed. --speed-scale
        dilates every transient the settle exists to wait out, so it
        dilates the settle too: the guard then arms at the same point
        ALONG THE PATH however slowly the run is watched. It did not, and
        slow mode armed the chained press 87 mm above the button instead of
        the 70 mm that four full-speed runs had validated — in the corner's
        wake, where it tripped on nothing at its first armed instant (bench
        2026-09-21; the 09-15 false trip was at that same height)."""
        settle_s = GROUP_SETTLE_S / max(float(self.time_scale), 1e-3)
        return max(WARP_SETTLE_FRAC, settle_s / max(float(flown_s), 1e-6))

    def _run_motion(self, group, while_running=None):
        # The member that OWNS this execution's contact semantics. Today
        # can_merge forbids guarded legs in a group, so this is group[0];
        # the lookup exists so that relaxing can_merge cannot silently run
        # a guarded stroke with the lead's guard (None), the lead's speed
        # and the lead's verify — which would be an unguarded press with no
        # error and no log line (review 2026-08-28).
        guarded = [g for g in group if g.guard is not None]
        if len(guarded) > 1:
            raise RuntimeError(
                "merged group has %d guarded legs (%s) — one execution can "
                "carry at most one contact guard" % (len(guarded), ", ".join(g.name for g in guarded))
            )
        if guarded and guarded[0] is not group[-1]:
            raise RuntimeError(
                "guarded leg %r is not last in its merged group — a trip "
                "must not strand queued motion behind it" % guarded[0].name
            )
        lead = guarded[0] if guarded else group[0]
        guard_spec = lead.guard
        if lead.guard is None or len(group) > 1:
            # ONE profile over the whole group — ease out, cruise at each
            # leg's speed fraction, flow through the junctions, long ease in
            # — baked into the timestamps, flown at the 1.0 sentinel.
            # A guarded TAIL rides the same profile: the approach and the
            # descent it leads into are one motion with no stop between
            # them, and the descent keeps its own (slow) cruise fraction.
            params = replace(self.retime_params(), time_scale=self.time_scale)
            traj, self.last_retime = retime_group([g.traj for g in group], [g.speed for g in group], JOINT_VMAX, params)
            exec_speed = 1.0
            if guard_spec is not None:
                # Arm where the guarded leg BEGINS, plus a settle: before
                # that the arm is still flying its approach, and the corner
                # into the descent leaves a torque offset that a contact
                # threshold cannot tell from a touch. The baseline is taken
                # at that SAME point rather than at the junction, so the
                # offset is absorbed into it instead of being measured
                # against a stale one (field 2026-09-15: a 3 Nm threshold
                # tripped at exactly the first armed instant, 87 mm above
                # the button, on nothing).
                at = self.last_retime["seg_start_fracs"][group.index(lead)]
                settle = self._settle_frac(float(self.last_retime.get("duration_s") or 0.0))
                armed_at = min(max(guard_spec.arm_after or 0.0, at + settle), ARM_AFTER_CAP)
                guard_spec = replace(guard_spec, rebaseline_after=armed_at, arm_after=armed_at)
                _rescale_contact_expectation(group, lead, traj)
        else:
            traj = group[0].traj
            exec_speed = lead.speed * self.time_scale
            flown = line_profile(lead, self.retime_params(), self.time_scale) if _free_air_line(lead) else None
            if flown is not None:
                traj, exec_speed = flown, 1.0
            elif guard_spec is not None and lead.warp is None and lead.guard_from is not None:
                # A descent flown on its own gets the SAME rule as a merged
                # tail above: baseline and arming where its straight line
                # begins, plus a settle. Its guard used to arm at a fixed
                # fraction of the leg (arm_after), which covered the free-
                # air approach only while the line below it crawled; once
                # the line flew a real profile the guard was live while the
                # arm was still braking into the waypoint, and 3.08 Nm of
                # that read as a touch three inches above the box (bench
                # 2026-09-21). A warped leg keeps the warp's own slow-zone
                # rebaseline, which is measured on the path it flies.
                total = secs(traj.points[-1].time_from_start)
                if total > 0.0:
                    at = secs(traj.points[lead.guard_from].time_from_start) / total
                    settle = self._settle_frac(total / exec_speed)
                    armed_at = min(max(guard_spec.arm_after or 0.0, at + settle), ARM_AFTER_CAP)
                    guard_spec = replace(guard_spec, rebaseline_after=armed_at, arm_after=armed_at)

        def make_guard():
            if not guard_spec:
                return None
            if guard_spec.gravity is not None:
                return CompensatedGuard(
                    guard_spec.touch_nm,
                    guard_spec.gravity,
                    reference=guard_spec.reference,
                    hang=guard_spec.hang,
                    rebaseline_after=guard_spec.rebaseline_after,
                    arm_after=guard_spec.arm_after,
                )
            return TorqueGuard(
                guard_spec.touch_nm,
                rebaseline_after=guard_spec.rebaseline_after,
                arm_after=guard_spec.arm_after,
            )

        guard = make_guard()
        # a compliant run's impedance tuning: the group's one gains value
        # (legs.can_merge merges only legs under the same one), sent when it
        # names a profile; with None the client flies its run default
        gains = group[0].gains
        self._note_settle()  # the last arrival's, its window cut by this goal
        t0 = time.monotonic()
        try:
            outcome, info = self.client.execute(
                traj,
                exec_speed,
                guard=guard,
                while_running=while_running,
                stop_when=lead.stop_when,
                **({} if gains is None else {"gains": gains}),
            )
        except KeyboardInterrupt:
            self._keep_aborted([g.name for g in group], traj, t0)
            raise
        arrived = (time.monotonic(), time.time())
        flew = self._how_it_flew(outcome, info, traj)
        if guard is not None:
            info.setdefault("torque_peak", guard.peak)
        if info.get("while_running_error"):
            log.info("  lookahead plan failed (%s) — planning after the leg" % info["while_running_error"])
        contact = self.client.contact_xyz() if outcome == "touch" else None
        ok = self._leg_ok(lead, outcome)
        detail = info.get("message", "")
        # a stop the leg ASKED for (stop_when) is not an outcome for its
        # contact verify to judge: a reach cut at the junction because the
        # box was not yet confirmed must read as "stopped", never as a
        # failed touch (field harness 2026-09-16)
        if lead.verify is not None and not (outcome == "stopped" and lead.stop_when is not None):
            g_ok, g_pos, _ = self.client.gripper_cmd(None)  # query only
            ok, detail = lead.verify(
                VerifyCtx(
                    outcome=outcome,
                    gripper_pos=g_pos if g_ok else None,
                    progress=info.get("progress"),
                    torque_peak=info.get("torque_peak"),
                    contact_xyz=contact,
                )
            )
        t_wall = time.monotonic() - t0
        return LegResult(
            group[-1].name if len(group) > 1 else lead.name,
            outcome,
            ok,
            detail,
            torque_peak=info.get("torque_peak"),
            progress=info.get("progress"),
            t_wall=t_wall,
            contact_xyz=contact,
            trip=guard.trip_report() if guard is not None else None,
            lookahead=info.get("while_running"),
            flown_traj=traj,
            flight=self._keep_flight([g.name for g in group], outcome, flew, info, traj, arrived),
            **flew,
        )

    def _how_it_flew(self, outcome, info, traj):
        """LegResult's mode, gains and end_err_* for one execution of `traj`,
        a motion's or the recoil's. On an arrival: where the arm is the
        moment the driver called it arrived, against the end of what flew —
        the cached sample, no wait and no settle, since a wait would hide
        what the spring had not caught up; each joint's error signed by its
        travel into the end (_signed_end_error). What it did after is the
        settle note's (_note_settle). Logged, never a gate (a compliant run
        stops for nothing it reads)."""
        live = self.client.joints() if outcome == "arrived" else None
        rad, mm, deg = _arrival_error(live, traj) if live is not None else (None, None, None)
        mrad, worst = _signed_end_error(live, traj) if live is not None else (None, None)
        gains = info.get("gains")
        return {
            "mode": "impedance" if getattr(self.client, "compliant", False) else "position",
            # a profile's number, or a custom spec's name (constants.GAINS_RIGID)
            # and its zeta, which one name has flown more than one of
            "gains": getattr(gains, "name", gains),
            "gains_zeta": getattr(gains, "zeta", None),
            "end_err_rad": rad,
            "end_err_mm": mm,
            "end_err_deg": deg,
            "end_err_mrad": mrad,
            "end_err_joint": worst,
        }

    def _keep_flight(self, names, outcome, flew, info, traj, arrived):
        """After an execution of the legs `names` (`traj`, what flew, as
        _how_it_flew's `flew`) the client returned at `arrived`
        (time.monotonic(), time.time()) — a Ctrl+C's, raised by then
        (_keep_aborted): its number, its side file next to
        the run record — <record>.flights/<number>-<first leg>.json, what
        went to the driver point by point and its feedback of what flew
        (_flight_doc) — and, an arrival, its settle, noted when the next goal
        goes out (_note_settle). Written now, the motion over: never while it
        flies. Returns the side file's path from the record's folder
        (LegResult.flight); None where nothing went to the driver (the
        client's own gates refused it: no info["sent_at"]) or nothing could
        be written. A measurement only: nothing here may cost the run."""
        if "sent_at" not in info:
            return None
        self._flights += 1
        seq = self._flights
        if outcome == "arrived":
            try:
                self._settling = _Settling(
                    seq,
                    list(names),
                    arrived,
                    [float(v) for v in self.client.joints()],
                    [float(v) for v in traj.points[-1].positions],
                    _travel_signs(traj),
                )
            except Exception:  # noqa: BLE001 - no settle to note
                self._settling = None
        try:
            path = self._record().with_suffix(".flights") / ("%02d-%s.json" % (seq, re.sub(r"[^\w.-]+", "_", names[0])))
            path.parent.mkdir(exist_ok=True)
            doc = _flight_doc(
                seq, names, outcome, flew, info, traj, arrived, getattr(self.client, "FEEDBACK_EVERY", None)
            )
            path.write_text(json.dumps(doc))
            return str(path.relative_to(self._log_dir))
        except Exception as e:  # noqa: BLE001 - a side file must never cost the run
            log.info("[runner] the flight of %s is not kept (%s)" % (names[0], e))
            return None

    def _keep_aborted(self, names, traj, t0):
        """A Ctrl+C ended the execution of the legs `names` (`traj`) begun at
        `t0`: the client raises once the cancel and the release are out (its
        abort path, its backstop) and hands nothing back, so its side file is
        kept from the call's own info (client.last_execution), its outcome
        "aborted" — the motion that was stopped is the one most worth
        reading. Nothing of it went out (no "sent_at" since `t0`): nothing
        kept. Its legs get no row, as before, and no settle: nothing
        arrived. Never in the unwinding's way: whatever goes wrong is
        printed, and the KeyboardInterrupt goes on."""
        try:
            info = getattr(self.client, "last_execution", None)
            if not isinstance(info, dict) or float(info.get("sent_at", t0 - 1.0)) < t0:
                return
            arrived = (time.monotonic(), time.time())
            self._keep_flight(names, "aborted", self._how_it_flew("aborted", info, traj), info, traj, arrived)
        except Exception as e:  # noqa: BLE001 - a measurement only: the Ctrl+C goes on
            log.info("[runner] the flight of %s is not kept (%s)" % (names[0], e))

    def _note_settle(self, ending=False):
        """The last arrival's settle (_keep_flight kept it), noted now: this
        runner's next execution is about to go out, or the run is over
        (`ending`, flush). Its window runs from the arrival to
        SETTLE_WINDOW_S after it, the first goal sent to the driver since
        (client.goals_sent: the hold that primes the impedance, a release)
        or now, whichever comes first. `cut` names what ended it: "next
        goal", such a goal; "next execution", this runner's next, about to
        go out (whether the client's gates then send it or not); "run end",
        the run's end — and at it any window nothing was heard through to
        its end (nothing spins the node after the run's last motion); None,
        the whole window. In it the /joint_states the client heard
        (joint_history), each placed by its stamp: the arm's settle per
        joint and at the tool (settle_report), against the end of what
        flew. Nothing waits for it and nothing gates on it; a client that
        keeps no history notes none."""
        pend, self._settling = self._settling, None
        history = getattr(self.client, "joint_history", None)
        if pend is None or history is None:
            return
        try:
            t0, w0 = pend.arrived
            later = [float(t) for t in getattr(self.client, "goals_sent", list)() if float(t) > t0]
            until = min([t0 + SETTLE_WINDOW_S, time.monotonic()] + later)
            # a stamp is the driver's clock, time.time()'s: on this one by
            # the arrival's two readings; an unstamped message by when heard
            off = t0 - w0
            heard = sorted(
                (((s.stamp + off if s.stamp else s.t) - t0, s) for s in history(since=t0)), key=lambda r: r[0]
            )
            rows = [(0.0, pend.live, None)] + [(dt, s.q, s.qd) for dt, s in heard if 0.0 < dt <= until - t0]
            if later and until == min(later):
                cut = "next goal"
            elif ending and (until < t0 + SETTLE_WINDOW_S or not any(dt >= until - t0 for dt, _s in heard)):
                cut = "run end"
            elif until < t0 + SETTLE_WINDOW_S:
                cut = "next execution"
            else:
                cut = None
            try:
                from box_opening.runtime import approach

                chain = approach._chain()
            except Exception:  # noqa: BLE001 - no chain: the joints alone
                chain = None
            fields = settle_report(rows, pend.end, pend.signs, until - t0, chain)
            self.note(
                "settle",
                seq=pend.seq,
                legs=pend.legs,
                arrived_t=round(w0, 3),
                window_s=round(until - t0, 3),
                cut=cut,
                **fields,
            )
        except Exception as e:  # noqa: BLE001 - a measurement only: nothing here may cost the run
            log.info("[runner] the settle after %s is not measured (%s)" % (pend.legs[0], e))

    def flush(self):
        """The run is over: the last arrival's settle noted now, its window
        cut where the run ends. Nothing spins the node after the last motion,
        so what the client heard by then is all there is (press_demo's
        exit)."""
        self._note_settle(ending=True)

    @staticmethod
    def _leg_ok(leg, outcome):
        if outcome == "stopped":
            # only a SEARCH may end early and count as done: it exists to find
            # something, not to reach its goal
            return leg.stop_when is not None
        if leg.guard is None:
            return outcome == "arrived"
        if leg.guard.trip == "setdown":
            return outcome == "touch"
        if leg.guard.trip == "obstruction":
            return outcome == "arrived"  # a trip means we struck the lid/rim
        return outcome == "touch"  # touch / press: the leg's verify refines it

    def _join_gripper(self, leg, handle, t0):
        """Collect an overlapped gripper command. Identical verdicts to
        the blocking path — only the waiting moved."""
        ok, pos, stalled = self.client.gripper_join(handle)
        outcome = "arrived" if ok else "failed"
        detail = "gripper at %.3f%s (overlapped)" % (
            pos,
            " (stalled)" if stalled else "",
        )
        if ok and leg.verify is not None:
            ok, detail = leg.verify(VerifyCtx(outcome=outcome, gripper_pos=pos))
        return LegResult(leg.name, outcome, ok, detail, t_wall=time.monotonic() - t0, gripper_pos=float(pos))

    def _run_gripper(self, leg):
        t0 = time.monotonic()
        ok, pos, stalled = self.client.gripper_cmd(leg.gripper_cmd)
        outcome = "arrived" if ok else "failed"
        detail = "gripper at %.3f%s" % (pos, " (stalled)" if stalled else "")
        if ok and leg.verify is not None:
            ok, detail = leg.verify(VerifyCtx(outcome=outcome, gripper_pos=pos))
        return LegResult(leg.name, outcome, ok, detail, t_wall=time.monotonic() - t0, gripper_pos=float(pos))

    def _record(self):
        """This run's jsonl, named when its first row or side file is written."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        if self._log_path is None:
            self._log_path = self._log_dir / time.strftime("run-%Y%m%d-%H%M%S.jsonl")
        return self._log_path

    def _append(self, row):
        """One row onto this run's jsonl (created on the first row)."""
        with open(self._record(), "a") as f:
            f.write(json.dumps(row) + "\n")

    def note(self, kind, **fields):
        """A non-leg row in the run log (the detected fix, the touch's
        verdict): a missed press is diagnosable from disk, not from a
        terminal paste (bench 2026-09-03)."""
        self._append({"t": time.time(), "leg": kind, "kind": "note", **fields})

    def _log(self, res, leg):
        row = {
            "t": time.time(),
            "leg": leg.name,
            "kind": leg.kind.value,
            "world": leg.world,
            "speed": leg.speed,
            # the trajectories are not log material (and asdict deep-copies them)
            **asdict(replace(res, lookahead=None, flown_traj=None)),
        }
        row.pop("leg_name", None)
        row.pop("lookahead", None)
        row.pop("flown_traj", None)
        if getattr(leg, "plan", None) is not None:
            row["plan"] = leg.plan  # what the planner said of this leg's plan, and how its end met the target
        self._append(row)
