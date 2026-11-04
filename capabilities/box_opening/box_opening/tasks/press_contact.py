"""Contact in the compliant press: where the pads met what they press, as
the arm measures it, and the stroke ended that far past it.

The compliant press (depth_press.compliant_stroke) drives one stroke down,
in the driver's joint impedance, to a fixed depth below where the CAMERAS
put the button: its end lies the button's travel and the spring's
overshoot below the top they measured, and nothing read on the way stops
it (owner 2026-10-27). The force then follows how wrong that height is.
Flown 10-29 and 10-30, 25 presses held 9.6-89.5 N against the 15 N the
spring was sized for (the depth_compliant_press rows); 10-30 13:28's pads
met the box 32 mm above where the scene camera put its top, and pressed on
42 mm past that.

The arm says where the pads met the box. The driver flies the reference
the trajectory sets; the pads follow it under the joint springs, a few
millimetres behind it in free air, and stop following where something
pushes back: the pads' height measured (the driver's feedback, its joints
through the arm's kinematics) and the reference's (the trajectory sent, at
the driver's fraction of its time) part, the gap growing as fast as the
reference goes on. Replayed on the 25 recorded presses' flight files: in
free air, once the slow leg's start has settled, the gap wanders 0.1-1.3
mm (sd); where something pushes back it rises 10-25 mm/s; the speed change
where the slow leg begins dips it 3-5 mm and it recovers at 5-6 mm/s.
CONTACT_RISE_M over the least gap of the last CONTACT_WINDOW_S, on
CONTACT_SAMPLES kept samples in a row, met every one of the 25 and none of
the slow leg's starts; where it met is the least gap's sample.

The stroke then ends where its reference stands as far below the contact
as it was planned to go below the top the cameras measured — the button's
travel and the spring's overshoot (PressContact.past_m) — and never deeper
than planned: only a stroke whose end lies deeper than that is cut, the
reference held where it is cut (client.execute's stop_when: the goal
cancelled, the driver holding its last reference). Where the cameras put
the top right, contact comes near it and the planned end stands: the
10-30 morning's six presses fly as flown. Where the box stood higher than
they said, the press is cut: replayed, 10-30 13:28's 45 N would have been
~23 N, 13:35's 89.5 N ~47 N. A press with no contact flies as planned
(compliant_hold says what it felt, as before). Watching only, PressContact
says where it met and what it would have cut, and cuts nothing
(PRESS_FROM_CONTACT False, or cut=False).

Review round 2 (KCE #23): the press watches its whole descent, from where
its stroke begins to come down (watch_press's arm_z), not only its slow leg:
something standing higher than that is met too. And a contact well away from
where the cameras put the button refuses the press (`verdict`): met more
than CONTACT_EARLY_M above their top, something else is in the way ("early");
nothing met down to CONTACT_LATE_M below it, the box is not where it was seen
("late"); the stroke ended with nothing met ("never"). The stroke is stopped
where it is judged, nothing pressed past it, and the run goes back up and
home. A watcher that fails stops the press there too (abort_on_failure),
said; the set-down's watch (reach_lid_off.watch_setdown) flies its planned
end instead, said.
"""

from collections import deque

import numpy as np

from box_opening import LOGGER as log

# where a press met the box (PressContact): the gap between the pads' height
# measured and the reference's rose more than CONTACT_RISE_M above its least
# in the last CONTACT_WINDOW_S, on CONTACT_SAMPLES kept samples in a row
# (client.FEEDBACK_EVERY's, ~90 Hz) — 15 mm/s and faster: something hard, or
# the box met while the reference still slowed; or, once the slow leg's start
# has settled (CONTACT_SETTLE_S), more than CONTACT_SLOW_RISE_M above its least
# since — a yield the pads press on down into, the gap growing slower
CONTACT_WINDOW_S = 0.2
CONTACT_RISE_M = 0.003
CONTACT_SAMPLES = 3
CONTACT_SETTLE_S = 0.6
CONTACT_SLOW_RISE_M = 0.005
# ... and it met the box at the last sample before that rise whose gap lay
# within CONTACT_ONSET_M of the least: where the rise began (the least itself
# may be a noise dip a few samples before it)
CONTACT_ONSET_M = 0.001
HISTORY_S = 3.0  # how far back the onset is looked for
# The stroke ends as far past the contact as it was planned past the top
# (PressContact.past_m), never deeper than planned. False: watched only, nothing cut
PRESS_FROM_CONTACT = True
# a cut shallower than this leaves the planned end: not worth a cancel
CUT_MIN_M = 0.001
# The refusal band around the top the cameras measured (PressContact's early_m, late_m; review
# round 2). Replayed through the whole-descent watch, 29 of the 30 presses flown 2026-10-29 to
# 11-03 met the box between CONTACT_SEEN_MM[0] and [1] mm of that top (+ above); the 30th, 10-30
# 13:35 (89.5 N held: the wrist camera put that box ~50 mm low), met it 50 mm above, which the
# watch armed at TOUCH_SLOW_M never saw. Early lies above the highest of the 29 and below where
# the descent begins (~56 mm over the top: nothing higher is watched), late well below the lowest
CONTACT_SEEN_MM = (-13.3, 37.5)
CONTACT_EARLY_M = 0.045
CONTACT_LATE_M = 0.025
# nothing met is judged by the pads as measured, not the reference (which runs on past a contact
# while its rise is counted): late once they stand this much below the limit, room for a contact
# just above it that they press a little into before it is seen
CONTACT_LAG_M = 0.003


def pads_z(chain, q):
    """The pads' height (depth_press.DOWN_CONTACT_TOOL) with the arm at `q`."""
    from box_opening.tasks.depth_press import DOWN_CONTACT_TOOL

    R, t = chain.fk([float(v) for v in q])
    return float(np.asarray(t, float)[2] + (np.asarray(R, float) @ np.asarray(DOWN_CONTACT_TOOL, float))[2])


class _Traj:
    """A sent trajectory's joints at a fraction of its time, and the pads'
    height at each of its points (one FK each, once)."""

    def __init__(self, chain, traj):
        pts = list(traj.points)
        self.t = np.array([p.time_from_start.sec + 1e-9 * p.time_from_start.nanosec for p in pts], float)
        self.q = np.array([list(p.positions) for p in pts], float)
        self.z = np.array([pads_z(chain, q) for q in self.q], float)
        self.T = float(self.t[-1]) if len(self.t) else 0.0

    def z_at(self, frac):
        """The reference's pads height at `frac`, from its points' (no FK)."""
        return float(np.interp(frac * self.T, self.t, self.z))

    def at(self, frac):
        """The reference's joints at `frac` of the trajectory's time."""
        t = float(np.clip(frac * self.T, self.t[0], self.t[-1]))
        i = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2))
        a = (t - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-9)
        return self.q[i] + a * (self.q[i + 1] - self.q[i])

    def frac_reaching(self, z, frac0):
        """The first fraction from `frac0` on at which the reference's pads
        stand at `z` or lower (linear between points), or None: never."""
        t0 = frac0 * self.T
        for i in range(len(self.t) - 1):
            if self.t[i + 1] < t0:
                continue
            za, zb = self.z[i], self.z[i + 1]
            if zb > z:
                continue
            a = 0.0 if za <= z else (za - z) / max(za - zb, 1e-12)
            t = max(t0, self.t[i] + a * (self.t[i + 1] - self.t[i]))
            return t / self.T if self.T > 0 else 1.0
        return None

    def frac_final_below(self, z):
        """The fraction from which the reference's pads stand at `z` or
        lower to the trajectory's end (linear between points): 0.0 where
        they never stand above it, None where they end above it."""
        if not len(self.z) or self.z[-1] > z:
            return None
        above = np.nonzero(self.z > z)[0]
        if not len(above):
            return 0.0
        i = int(above[-1])  # the last point above z; the next one is not
        a = (self.z[i] - z) / max(self.z[i] - self.z[i + 1], 1e-12)
        t = self.t[i] + a * (self.t[i + 1] - self.t[i])
        return t / self.T if self.T > 0 else 1.0


class PressContact:
    """A compliant press watched as it flies: client.execute's stop_when.

    Built for a stroke (depth_press.compliant_stroke's) planned onto a
    button whose top the cameras put at `top_z`: it watches from where the
    stroke's slow leg begins (`arm_z`, TOUCH_SLOW_M above the top) on, and
    ends the stroke `past_m` below where the pads met the box (the stroke's
    own depth below the top), where that is above the stroke's end
    (`end_z`) — with `cut`; else it only watches. Fed what the client keeps
    of the driver's feedback (client.last_execution's feedback_q: when, the
    fraction, the joints) and the trajectory it sent (sent_traj), each kept
    sample once (feed). held_ref is the reference the driver holds where it
    cut: what the hold's spring pulls toward (depth_press.compliant_hold).
    record() is what the run record keeps of it. With `final` it watches
    only the execution's last stretch below `arm_z` (_Traj.frac_final_below):
    the set-down's, which flies in one execution from a lift off a box that
    may stand lower (reach_lid_off.watch_setdown)."""

    def __init__(
        self,
        chain,
        client,
        top_z,
        arm_z,
        past_m,
        end_z,
        cut=True,
        final=False,
        slow_z=None,
        early_m=None,
        late_m=None,
        abort_on_failure=False,
    ):
        self.chain, self.client = chain, client
        self.top_z, self.arm_z = float(top_z), float(arm_z)
        # the slow rise is judged once the reference has stood below slow_z CONTACT_SETTLE_S (the
        # slow leg's start: its speed change dips the gap); the fast one from arming on
        self.slow_z = self.arm_z if slow_z is None else float(slow_z)
        self.past_m, self.end_z = float(past_m), float(end_z)
        self.cut = bool(cut) and PRESS_FROM_CONTACT
        self.final = bool(final)
        # the refusal band (None: none): met more than early_m above top_z, or nothing met down
        # to late_m below it, stops the stroke where it is judged (verdict)
        self.early_m = None if early_m is None else float(early_m)
        self.late_m = None if late_m is None else float(late_m)
        self.abort_on_failure = bool(abort_on_failure)
        self.traj = None
        self._sent = None  # the info whose trajectory self.traj is
        self._reset()

    def _reset(self):
        """Nothing seen yet: a new execution starts its watch afresh."""
        self._last_t = None  # the newest kept sample fed
        self.armed_t = None
        self.arm_frac = 0.0  # where watching may begin (final: the last stretch below arm_z; None never)
        self._win = deque()  # (t, gap, pads z measured) since arming, the last CONTACT_WINDOW_S
        self._hist = deque()  # ... the last HISTORY_S of them: where a rise began
        self._cand = None  # (the least gap, its sample) a rise is counted from
        self._n_above = 0
        self._slow_least = None  # the least gap since the slow leg's start settled
        self._n_slow = 0
        self.how = None  # "fast" | "slow": which rise met it
        self.contact = None  # (t, the pads' height there, the gap there): where the pads met the box
        self.detected_t = None
        self.stop_z = None  # the reference's pads where the stroke ends past the contact
        self.f_stop = None
        self.stopped_frac = None
        self.held_ref = None
        self.samples = 0
        self.last_gap = None
        self.slow_t = None  # when the reference first stood below slow_z
        self.late_at = None  # (t, the pads measured) where nothing met was judged late
        self.verdict = None  # "early" | "late" | "failed": the press refused, stopped where it was judged
        self.failed = None  # what the watcher raised, where it failed

    # -- the samples ------------------------------------------------------------------
    def feed(self, t, frac, q):
        """One kept sample: when it was heard (s), the driver's fraction of
        the trajectory's time, the joints it measured."""
        if self.traj is None:
            return
        self.samples += 1
        if self.armed_t is None and (self.arm_frac is None or frac < self.arm_frac):
            return  # before the stretch it watches (final): nothing to judge
        if self.armed_t is None and self.traj.z_at(frac) > self.arm_z + 0.005:
            return  # well above the slow leg: nothing to judge, no kinematics spent
        z_ref = pads_z(self.chain, self.traj.at(frac))
        if self.armed_t is None:
            if z_ref > self.arm_z:
                return
            self.armed_t = t
        if self.slow_t is None and z_ref <= self.slow_z:
            self.slow_t = t
        z_meas = pads_z(self.chain, q)
        gap = z_meas - z_ref
        self.last_gap = gap
        if self.contact is not None or self.verdict is not None:
            return
        sample = (t, gap, z_meas)
        win, hist = self._win, self._hist
        win.append(sample)
        hist.append(sample)
        while win and t - win[0][0] > CONTACT_WINDOW_S:
            win.popleft()
        while hist and t - hist[0][0] > HISTORY_S:
            hist.popleft()
        # a fast rise: over the least of the last CONTACT_WINDOW_S
        if self._cand is not None and gap - self._cand[1] > CONTACT_RISE_M:
            self._n_above += 1
        else:
            least = min(win, key=lambda s: s[1])
            self._cand, self._n_above = (least, 1) if gap - least[1] > CONTACT_RISE_M else (None, 0)
        # a slow one: over the least since the slow leg's start settled
        if self.slow_t is not None and t - self.slow_t >= CONTACT_SETTLE_S:
            if self._slow_least is None or gap < self._slow_least[1]:
                self._slow_least, self._n_slow = sample, 0
            elif gap - self._slow_least[1] > CONTACT_SLOW_RISE_M:
                self._n_slow += 1
            else:
                self._n_slow = 0
        if self._n_above >= CONTACT_SAMPLES:
            self._met(t, frac, self._cand, "fast")
        elif self._n_slow >= CONTACT_SAMPLES:
            self._met(t, frac, self._slow_least, "slow")
        elif self.late_m is not None and z_meas < self.top_z - self.late_m - CONTACT_LAG_M:
            # the pads this far below the top the cameras measured, nothing met: the box is not
            # where it was seen. Stopped here (the next poll), nothing pressed past it
            self.verdict, self.f_stop, self.late_at = "late", frac, (t, z_meas)

    def _met(self, t, frac, least, how):
        """Contact, seen at `t` (`frac` of the trajectory) as a rise over the
        gap `least` (a sample): it began at the last sample from that one on
        whose gap lay within CONTACT_ONSET_M of it."""
        onset = least
        for s in self._hist:
            if s[0] >= least[0] and s[1] - least[1] <= CONTACT_ONSET_M:
                onset = s
        t_c, g_c, z_c = onset
        self.contact = (t_c, z_c, g_c)
        self.detected_t, self.how = t, how
        if self.early_m is not None and z_c > self.top_z + self.early_m:
            self.verdict, self.f_stop = "early", frac  # something else is in the way: stopped now
            return
        if self.late_m is not None and z_c < self.top_z - self.late_m:
            self.verdict, self.f_stop = "late", frac  # the box is not where it was seen: stopped now
            return
        stop_z = z_c - self.past_m
        if stop_z > self.end_z + CUT_MIN_M:
            self.stop_z = stop_z
            if self.cut:
                self.f_stop = self.traj.frac_reaching(stop_z, frac)

    def _feed_new(self):
        info = getattr(self.client, "last_execution", None)
        if not isinstance(info, dict) or info.get("sent_traj") is None:
            return
        if info is not self._sent:  # a new execution: the trajectory it sent, watched afresh
            self._reset()
            self._sent, self.traj = info, _Traj(self.chain, info["sent_traj"])
            if self.final:
                self.arm_frac = self.traj.frac_final_below(self.arm_z)
        kept = info.get("feedback_q")
        if not kept or (self._last_t is not None and kept[-1][0] <= self._last_t):
            return  # polled at every spin (~1 kHz); a sample is kept every 10th feedback
        new = []
        for s in reversed(kept):  # only the newest: the deque holds up to a minute of them
            if self._last_t is not None and s[0] <= self._last_t:
                break
            new.append(s)
        for s in reversed(new):
            self._last_t = s[0]
            self.feed(float(s[0]), float(s[1]), s[2])

    def __call__(self, progress):
        """stop_when: True once the reference stands where the stroke ends
        past the contact — the goal is cancelled and the driver holds it."""
        if self.failed is not None:
            return self.abort_on_failure
        try:
            self._feed_new()
        except Exception as e:  # noqa: BLE001 -- said; the press stops, the set-down flies as planned
            self.failed = "%s: %s" % (type(e).__name__, e)
            log.info(
                "[press_demo] the contact watch failed (%s) — %s"
                % (self.failed, "the press stops here" if self.abort_on_failure else "it flies its planned end")
            )
            if not self.abort_on_failure:
                self.f_stop = None
                return False
            self.verdict = "failed"
            self.stopped_frac = float(progress)
            self.held_ref = None if self.traj is None else [float(v) for v in self.traj.at(progress)]
            return True
        if self.f_stop is None or progress < self.f_stop:
            return False
        self.stopped_frac = float(progress)
        self.held_ref = [float(v) for v in self.traj.at(progress)]
        return True

    def refusal(self):
        """Why the press must not go on from here, once its execution is over: (verdict, the
        STOP's words), or None. "never": watched to the stroke's end, nothing met, where a band
        is set (a stroke that ends above the late limit)."""
        above = None if self.contact is None else 1000 * (self.contact[1] - self.top_z)
        if self.verdict == "early":
            return self.verdict, (
                "the press met something %.0f mm above the button's top as the cameras saw it (more than %.0f): "
                "something else is in the way — the press was stopped there; nothing was pressed"
                % (above, 1000 * self.early_m)
            )
        if self.verdict == "late":
            how_low = -above if above is not None else 1000 * (self.top_z - self.late_at[1])
            return self.verdict, (
                "the pads met nothing down to %.0f mm below the button's top as the cameras saw it (more than %.0f): "
                "the box is not where it was seen — the press was stopped there" % (how_low, 1000 * self.late_m)
            )
        if self.verdict == "failed":
            return self.verdict, "the press's contact watch failed (%s) — the press was stopped there" % self.failed
        if self.contact is None and self.armed_t is not None and self.late_m is not None:
            return "never", ("the press stroke ended with the pads meeting nothing: the box is not where it was seen")
        return None

    # -- the record --------------------------------------------------------------------
    def record(self):
        """The run record's 'contact': where the pads met the box (the
        pads' height, how far above the top the cameras measured, when after
        the goal went out), the stroke's end past it and whether it was cut
        there (`cut_mm` how much short of its planned end), or why not."""
        sent = None if self._sent is None else self._sent.get("sent_at")
        refused = self.refusal()
        out = {
            "watched": self.armed_t is not None,
            "cut_on": self.cut,
            "samples": self.samples,
            "past_mm": round(1000 * self.past_m, 1),
            "armed_above_top_mm": round(1000 * (self.arm_z - self.top_z), 1),
            "band_mm": None
            if self.early_m is None and self.late_m is None
            else [
                None if self.late_m is None else -round(1000 * self.late_m),
                None if self.early_m is None else round(1000 * self.early_m),
            ],
            "verdict": None if refused is None else refused[0],
        }
        if self.failed is not None:
            out["failed"] = self.failed
        if self.contact is None:
            out["met"] = False
            return out
        t_c, z_c, _g = self.contact
        out.update(
            met=True,
            how=self.how,  # a fast rise or a slow one
            z=round(z_c, 4),
            above_top_mm=round(1000 * (z_c - self.top_z), 1),
            at_s=None if sent is None else round(t_c - float(sent), 3),
            detected_after_ms=round(1000 * (self.detected_t - t_c)),
            # where the reference would end past it, and how far short of the planned end
            end_z=None if self.stop_z is None else round(self.stop_z, 4),
            cut_mm=None if self.stop_z is None else round(1000 * (self.stop_z - self.end_z), 1),
            cut=self.stopped_frac is not None,
        )
        if refused is not None:
            out["why"] = refused[1]
        elif self.stop_z is None:
            out["why"] = "the planned end is no deeper than that past the contact"
        elif not self.cut:
            out["why"] = "watched only"
        elif self.stopped_frac is None:
            out["why"] = "the stroke ended before its reference got there"
        return out


class PadsAbove:
    """A gripper leg's gate (Leg.send_when): True once the pads, as the arm
    measures them, stand at `z` or higher — the newest sample the client
    keeps of the driver's feedback (client.last_execution's feedback_q), one
    FK for each new one (~90 Hz; polled at every spin). The grip straight
    from the press opens the fingers so: off the knob, not on it
    (reach_lid_off._grip_from_hold). `at` is where it said so: (when, after
    the goal went out, s; the pads' height)."""

    def __init__(self, chain, client, z):
        self.chain, self.client, self.z = chain, client, float(z)
        self._sent = None
        self._last_t = None
        self.at = None
        self.highest = None  # the highest the pads stood while it watched

    def __call__(self, progress):
        info = getattr(self.client, "last_execution", None)
        if not isinstance(info, dict) or info.get("sent_traj") is None:
            return False
        if info is not self._sent:  # a new execution: watched afresh
            self._sent, self._last_t, self.at, self.highest = info, None, None, None
        kept = info.get("feedback_q")
        if not kept or (self._last_t is not None and kept[-1][0] <= self._last_t):
            return False
        t, _frac, q = kept[-1][0], kept[-1][1], kept[-1][2]
        self._last_t = t
        z = pads_z(self.chain, q)
        self.highest = z if self.highest is None else max(self.highest, z)
        if z < self.z:
            return False
        sent = info.get("sent_at")
        self.at = (None if sent is None else round(float(t) - float(sent), 3), round(z, 4))
        return True
