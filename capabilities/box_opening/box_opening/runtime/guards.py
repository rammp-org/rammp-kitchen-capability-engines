"""Contact guard, trajectory sanity gate, and guarded-descent bookkeeping.

TorqueGuard is the palm-demo pattern with the spec §6 hardening: the
baseline is anchored at the first execution feedback with
progress > 0 (never at goal-accept), and guarded runs REFUSE to start
without effort fields (enforced by the Runner, which owns the streams).
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

from box_opening.geometry import ang_diff
from box_opening.runtime.stamps import secs

RECENT_SAMPLES = 60  # effort readings kept for a trip's report (~1.5 s of a live stream)
# The pull at the fingertip (CompensatedGuard `hang`) is judged on the mean of
# this many distinct readings (~0.1 s of /joint_states at ~46 Hz), never on
# one: 0.1 s into a depth press's unload, 23 ms after the recoil stopped,
# joint_4 rang 6.6-8.7 Nm reading to reading, -3.9 to +3.3 N of "pull" about
# a mean under 0.5, and the one high reading stopped the arm over the pressed
# box with nothing on the hand (field 2026-10-23 17:18; the 16:02 run rang as
# hard and passed by luck). A box coming along pulls steadily: the mean finds
# it two or three readings later, about a millimetre of the unload.
HANG_SAMPLES = 5


class TorqueGuard:
    def __init__(self, touch_nm, rebaseline_after=None, arm_after=None):
        self.touch_nm = float(touch_nm)
        self.armed = False
        self._baseline = None
        self.peak = 0.0
        # Progress fraction before which deviations are OBSERVED but never
        # trip: a set-down's contact physically cannot happen in the first
        # half of a stroke that ends 5 mm below the surface, yet the fast
        # warp segment's motion dynamics tripped a 4.0 Nm threshold 64 ms
        # in and the lid was released 110 mm up (field 2026-09-02).
        self.arm_after = None if arm_after is None else float(arm_after)
        self._progress = 0.0
        # Time fraction at which a warped descent changes speed. The guard
        # stays ARMED throughout — coverage is not reduced — but its
        # reference is re-taken once the arm is in the slow regime, so the
        # dynamic-torque shift from decelerating is not mistaken for
        # contact, and the touch is judged against a same-regime baseline.
        self.rebaseline_after = None if rebaseline_after is None else float(rebaseline_after)
        self._rebaselined = False
        # the run-up to a trip, for the log (trip_report): a false trip
        # used to leave one number behind, and that one misleading
        self._recent = deque(maxlen=RECENT_SAMPLES)
        self._baseline_at = None  # (monotonic s, progress) the baseline in force was taken
        self._trip = None
        self._recent_key = None  # the reading last kept in _recent

    def on_progress(self, progress):
        self._progress = float(progress)
        if progress > 0.0:
            self.armed = True
        if self.rebaseline_after is not None and not self._rebaselined and progress >= self.rebaseline_after:
            self._rebaselined = True
            self._baseline = None  # re-captured on the next efforts reading

    def on_efforts(self, wrist_efforts):
        if not self.armed or wrist_efforts is None:
            return False
        now = time.monotonic()
        eff = [float(v) for v in wrist_efforts]
        self._remember(now, eff, eff)
        if self._baseline is None:
            self._baseline = eff
            self._baseline_at = (now, self._progress)
            # the peak is the deviation from the baseline IN FORCE: what was
            # seen against an earlier one (the free-air dynamics before a
            # merged tail's junction, a warp's fast zone) is not what a trip
            # is judged on — "torque_peak 12.9" on a 3 Nm false trip was
            # that (bench 2026-09-21)
            self.peak = 0.0
            return False
        devs = [abs(a - b) for a, b in zip(eff, self._baseline, strict=False)]
        dev = max(devs)
        self.peak = max(self.peak, dev)
        if self.arm_after is not None and self._progress < self.arm_after:
            return False
        if dev > self.touch_nm:
            self._trip = (now, devs.index(dev), dev, eff)
            return True
        return False

    def _remember(self, now, eff, key):
        # the client polls once per callback it spins (~1 kHz of execution
        # feedback) and /joint_states arrives at ~46 Hz: the same message is
        # handed in 10-20 times running, and kept each time it filled the
        # run-up with copies of four to seven readings (field 2026-10-22)
        if key != self._recent_key:
            self._recent_key = key
            self._recent.append((now, self._progress, eff))

    def trip_report(self):
        """What tripped the guard, for the run log — or None. `joint` indexes
        the efforts the guard is fed (the client's wrist joints); `recent`
        is the run-up, newest last: [s before the trip, progress, efforts...]."""
        if self._trip is None:
            return None
        t, joint, dev, eff = self._trip
        return {
            "joint": joint,
            "dev_nm": round(dev, 3),
            "baseline": [round(v, 3) for v in self._baseline],
            "efforts": [round(v, 3) for v in eff],
            "baseline_age_s": round(t - self._baseline_at[0], 3),
            "baseline_at_progress": round(self._baseline_at[1], 4),
            "progress": round(self._progress, 4),
            "recent": [[round(ts - t, 3), round(p, 4)] + [round(v, 3) for v in e] for ts, p, e in self._recent],
        }


class CompensatedGuard(TorqueGuard):
    """A TorqueGuard that judges only what gravity does NOT explain.

    Fed the live joints with every effort reading (client.execute), it takes
    out the change the arm's own weight makes between the baseline's joints
    and these (kinematics.GravityModel). A plan-free reach line 15 cm long
    at 0.8-0.9 m changes joint_4's gravity torque by 1.1-1.9 Nm — the size
    of a canister hanging off a finger — so a raw guard either trips on the
    reach itself or misses the canister (review 2026-10-22).

    `reference`: (efforts, joints) read at rest in free air, used as the
    baseline instead of the first moving sample: a leg that STARTS on
    something (the lift off the button) is judged against the arm with
    nothing on it, not against the load it is leaving.

    `hang`: (k, newtons), k the wrist joints' holding torque per newton
    pulling straight down at the fingertip (GravityModel.tip_load). The
    residual's projection on k reads as that pull, and past `newtons` over
    HANG_SAMPLES readings the guard trips. A surface pushes UP and reads
    negative: leaving the button never trips it, a box coming along with the
    finger does."""

    wants_joints = True

    def __init__(self, touch_nm, gravity, reference=None, hang=None, rebaseline_after=None, arm_after=None):
        super().__init__(
            touch_nm, rebaseline_after=None if reference is not None else rebaseline_after, arm_after=arm_after
        )
        self._gravity = gravity
        self._g0 = None
        self._k = None if hang is None else [float(v) for v in hang[0]]
        self._kk = None if hang is None else sum(v * v for v in self._k)
        self.hang_n = None if hang is None else float(hang[1])
        self.down_n = None  # the pull at the fingertip (N, + down): the mean of the latest HANG_SAMPLES readings
        self.peak_down_n = None
        self._pulls = deque(maxlen=HANG_SAMPLES)
        self._why = None
        self._g_key = self._g_last = None  # the joints gravity was last evaluated at, and its value
        if reference is not None:
            eff, q = reference
            self._baseline = [float(v) for v in eff]
            self._g0 = [float(v) for v in gravity(q)]
            self._baseline_at = (time.monotonic(), 0.0)

    def on_efforts(self, wrist_efforts, joints=None):
        if joints is None:
            # no joints, nothing to take gravity out with: judged raw, the
            # guard trips sooner, never later
            return super().on_efforts(wrist_efforts)
        if not self.armed or wrist_efforts is None:
            return False
        now = time.monotonic()
        eff = [float(v) for v in wrist_efforts]
        key = tuple(float(v) for v in joints)
        if key != self._g_key:
            # gravity is a function of the joints alone: one model
            # evaluation (0.7 ms on the Orin) per /joint_states message,
            # not one per feedback callback the client spun
            self._g_key, self._g_last = key, [float(v) for v in self._gravity(joints)]
        g = self._g_last
        fresh = (tuple(eff), key) != self._recent_key  # a new /joint_states message, not a repeat
        self._remember(now, eff, (tuple(eff), key))
        if self._baseline is None or self._g0 is None:
            self._baseline, self._g0 = eff, g
            self._baseline_at = (now, self._progress)
            self.peak = 0.0
            return False
        r = [(a - b) - (gq - g0) for a, b, gq, g0 in zip(eff, self._baseline, g, self._g0, strict=False)]
        devs = [abs(v) for v in r]
        dev = max(devs)
        self.peak = max(self.peak, dev)
        if self._k is not None and self._kk > 0.0:
            if fresh or not self._pulls:
                self._pulls.append(sum(k * v for k, v in zip(self._k, r, strict=False)) / self._kk)
            self.down_n = sum(self._pulls) / len(self._pulls)
            self.peak_down_n = self.down_n if self.peak_down_n is None else max(self.peak_down_n, self.down_n)
        if self.arm_after is not None and self._progress < self.arm_after:
            return False
        if dev > self.touch_nm:
            self._trip, self._why = (now, devs.index(dev), dev, eff), "torque"
            return True
        if self.down_n is not None and len(self._pulls) == HANG_SAMPLES and self.down_n > self.hang_n:
            self._trip, self._why = (now, devs.index(dev), dev, eff), "hang"
            return True
        return False

    def trip_report(self):
        rep = super().trip_report()
        if rep is not None:
            rep["compensated"] = True
            rep["why"] = self._why
            rep["down_n"] = None if self.down_n is None else round(self.down_n, 2)
        return rep


# Time fraction a speed change needs to settle after a guard's rebaseline
# before the guard may judge efforts against the new baseline. Tuned for a
# WARP, whose scale ramps gently over RAMP_POINTS samples.
WARP_SETTLE_FRAC = 0.05
# How long the arm needs after a merge JUNCTION — a corner where one leg's
# cruise becomes the next one's — before a contact threshold means anything
# there. A TIME, not a fraction: what the arm needs after a speed and
# direction change does not scale with how far the trajectory goes, and a
# fraction of a short one is a shorter wait for the same event.
#
# Field 2026-09-15: the first chained approach + press tripped a 3 Nm touch
# threshold at exactly its first armed instant, 87 mm above the button, with
# nothing there. The re-timed profile was already at the descent cruise by
# the junction; what was stale was the BASELINE, taken at the junction and
# compared 0.14 s later. The Runner now takes the baseline AT the arming
# point, so a steady offset left by the corner is absorbed rather than
# measured, and waits this long after the junction to do it.
#
# Wall time AT FULL OPERATOR SPEED: --speed-scale dilates it with the motion
# (runner._settle_frac), so slow mode arms the guard at the same point along
# the path as the run it is a slow view of.
GROUP_SETTLE_S = 0.25
# A guard is never armed later than this fraction of its stroke: a
# degenerate split must still leave it able to trip at the very end.
ARM_AFTER_CAP = 0.95


@dataclass(frozen=True)
class GuardSpec:
    touch_nm: float
    # what a trip MEANS for the leg (runner._leg_ok, and the recoil):
    #   "touch"       the stroke went looking for a surface — a trip found it
    #   "press"       the bounded push — a trip (its stop) or arriving, both good
    #   "setdown"     the trip IS the success
    #   "obstruction" nothing should be met — a trip is a strike
    trip: str
    # time fraction at which a warped descent enters its slow zone;
    # the Runner hands it to TorqueGuard so the baseline is re-taken
    # in the regime the touch actually happens in
    rebaseline_after: float = None
    # progress fraction before which the guard observes but cannot trip
    # (set-down: contact is only possible at the stroke's very end)
    arm_after: float = None
    # The arm's own weight, judged on the live joints: a callable joints ->
    # the wrist joints' holding torque (kinematics.GravityModel.wrist_holding).
    # Set, the Runner builds a CompensatedGuard with the two below.
    gravity: object = field(default=None, compare=False)
    # (efforts, joints) at rest in free air: the baseline, in place of the
    # first moving sample
    reference: tuple = field(default=None, compare=False)
    # (k, newtons): trip on this much pull straight down at the fingertip
    hang: tuple = field(default=None, compare=False)


def sanity_violations(traj, margin_rad):
    """Per-joint excursion beyond |start->end| + margin: planner wandered.

    Wrap-aware: reported positions wrap to (-pi, pi], so the series is
    unwrapped by accumulating ang_diff deltas before measuring excursion
    (joint_3 sits AT +pi at home — spec §3)."""
    out = []
    for j, name in enumerate(traj.joint_names):
        pos = [p.positions[j] for p in traj.points]
        unwrapped = [pos[0]]
        for prev, cur in zip(pos, pos[1:], strict=False):
            unwrapped.append(unwrapped[-1] + ang_diff(cur, prev))
        allowed = abs(unwrapped[-1] - unwrapped[0]) + margin_rad
        excursion = max(unwrapped) - min(unwrapped)
        if excursion > allowed:
            out.append("%s excursion %.3f rad > |Δ| + margin %.3f" % (name, excursion, allowed))
    return out


def in_band(pos, band):
    lo, hi = band
    return lo <= float(pos) <= hi


def time_fraction_at_path_fraction(traj, path_frac):
    """Time fraction at which `traj` has covered `path_frac` of its own
    joint-space path length.

    Execution feedback reports progress as elapsed/duration — a TIME
    fraction (the driver's fraction_complete, trajectory_executor.cpp).
    Callers that know
    where along the PATH contact is expected must convert, because the two
    only coincide for a constant-speed profile and cuRobo's is not one.
    Trajectory points are uniformly spaced in time, so a point's time
    fraction is just its index over the count.
    """
    pts = list(traj.points)
    if len(pts) < 2:
        return float(path_frac)
    cum, total = [0.0], 0.0
    for a, b in zip(pts, pts[1:], strict=False):
        total += math.sqrt(sum((x - y) ** 2 for x, y in zip(a.positions, b.positions, strict=False)))
        cum.append(total)
    if total <= 0.0:
        return float(path_frac)

    duration = secs(pts[-1].time_from_start)
    if duration <= 0.0:
        return float(path_frac)
    target = float(path_frac) * total
    for i, c in enumerate(cum):
        if c >= target:
            return secs(pts[i].time_from_start) / duration
    return 1.0
