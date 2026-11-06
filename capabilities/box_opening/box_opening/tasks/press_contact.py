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

The press study (2026-11-05): beside the real watch, a shadow judges the same
samples by a candidate fast rule (SHADOW_*, watch_beside), cutting and
refusing nothing; the record says what it made of the press. And the record
keeps the longest gap between the kept samples watched (feedback_gap_max_ms).
What the watch decides follows from those samples alone — each judged once,
in order, at the time it was heard — not from when it is polled; but a gap in
them delays it, and one of 120 ms at the onset of the box met 50 mm high
(10-30 13:35) turned its refusal into a press 1.57 s later (replayed). The
samples are kept by the feedback callback on the thread that spins the
client's node, the main one: a watch on a thread of its own would see the same
gaps. On the 33 strokes flown none was longer than 21 ms. A change of the
driver's global speed scale mid-stroke changes the arm's lag behind the
reference, and a speed-up reads as a rise: the watch reads the scale the driver
applies from how fast the fraction it reports goes (SCALE_*), and a change of
it re-settles the watch, both rules started again.
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
# The fast rule a shadow watch judges beside the real one on every press (PressContact.watch_beside,
# depth_press.watch_press): 4 mm over the least of the last 0.4 s, on 1 kept sample (both rules'
# count, as CONTACT_SAMPLES is), the slow rule's settle and rise as they are. Replayed on the 33
# strokes flown 10-29 to 11-04 (press study I-S2, its C3) it met all 32 by the fast rule, 254 / 320
# / 412 ms after the onset (median / p90 / max) against today's 287 / 674 / 1355, none falsely, and
# still refused the box met 50 mm above the cameras' top. But it was chosen on those same strokes,
# and the largest free-air rise over 0.4 s before an onset was 3.4 mm. It cuts and refuses nothing:
# its verdict is in the press's record, for the presses that fly to score
SHADOW_RISE_M = 0.004
SHADOW_WINDOW_S = 0.4
SHADOW_SAMPLES = 1
# The driver's global speed scale changed mid-stroke (press study E-S10): the reference speeds up or
# slows down, the arm's lag behind it changes as at the slow leg's start, and a speed-up reads as a
# rise (0.5 -> 1.0: 8 % of the changes injected false contacts, 0.25 -> 1.0 62 %). The client sees
# the scale the driver applies, slewed: the driver's trajectory clock runs at it
# (trajectory_executor.cpp tick: traj_t += dt x applied, fraction = traj_t / duration), so the
# fraction it reports advances at scale / T, and T x dfrac / dt over the kept samples' heard times
# is the applied scale (PressContact._scale). Over SCALE_SPAN_S it read 0.94-1.07 through the 198
# impedance flights recorded 10-29 to 11-04 (all at 1.0; 0.96-1.06 on their last 60 mm), no change
# of SCALE_CHANGE among them. A change of more than that from the scale the watch settled at
# re-settles it (_resettle): both rules start again, as at the slow leg's start. A gap in the kept
# samples (the node not spun) bunches the heard times after it: one longer than SCALE_GAP_S, the
# scale is read afresh from SCALE_GAP_S after it (a shorter one puts it at most 15 % off; a 120 ms
# one read across would put it 60 % off)
SCALE_SPAN_S = 0.2
SCALE_CHANGE = 0.2
SCALE_GAP_S = 0.03


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
    may stand lower (reach_lid_off.watch_setdown). `rise_m`, `window_s` and
    `rise_samples` set its fast rule (None: CONTACT_RISE_M, CONTACT_WINDOW_S,
    CONTACT_SAMPLES, read as it judges). A change of the driver's applied
    speed scale while it watches re-settles it (_scale, SCALE_*)."""

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
        rise_m=None,
        window_s=None,
        rise_samples=None,
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
        self.rise_m = None if rise_m is None else float(rise_m)
        self.window_s = None if window_s is None else float(window_s)
        self.rise_samples = None if rise_samples is None else int(rise_samples)
        self.shadow = None  # a watch judging beside this one, on its samples (watch_beside)
        self.traj = None
        self._sent = None  # the info whose trajectory self.traj is
        self._reset()

    def _reset(self):
        """Nothing seen yet: a new execution starts its watch afresh."""
        self._last_t = None  # the newest kept sample fed
        self._prev_t = None  # ... the one fed before it (feed)
        self.gap_max_s = None  # the longest between two kept samples, from where it arms on
        self._scale_q = deque()  # (t, frac) of the armed samples the applied speed scale is read over
        self._scale_from = None  # ... read again from this heard time on, after a gap (SCALE_GAP_S)
        self.scale_ref = None  # the applied speed scale the watch settled at (_scale)
        self.scale_seen = None  # (least, most) of the applied speed scale read
        self.resettles = []  # (t, the scale from, to): where a change of it re-settled the watch
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
        prev, self._prev_t = self._prev_t, t
        if self.armed_t is None and (self.arm_frac is None or frac < self.arm_frac):
            return  # before the stretch it watches (final): nothing to judge
        if self.armed_t is None and self.traj.z_at(frac) > self.arm_z + 0.005:
            return  # well above the slow leg: nothing to judge, no kinematics spent
        z_ref = pads_z(self.chain, self.traj.at(frac))
        if self.armed_t is None and z_ref > self.arm_z:
            return
        if prev is not None:
            # how long since the kept sample before, from where it arms on: the longest is in the
            # record (feedback_gap_max_ms). A gap there delays what it decides, and one at the
            # onset of a box met high may miss its fast rise (press study N12)
            self.gap_max_s = max(self.gap_max_s or 0.0, t - prev)
        self._arm(t, z_ref)
        resettled = self._scale(t, frac, prev)
        z_meas = pads_z(self.chain, q)
        self._judge(t, frac, z_ref, z_meas)
        if self.shadow is not None and self.shadow.failed is None:
            try:  # the same sample judged by the shadow: no kinematics of its own
                self.shadow._arm(t, z_ref)
                if resettled:  # the same change of scale re-settles it too
                    self.shadow._resettle(t)
                self.shadow._judge(t, frac, z_ref, z_meas)
            except Exception as e:  # noqa: BLE001 -- the shadow's own failure: in its record, nothing more
                self.shadow.failed = "%s: %s" % (type(e).__name__, e)

    def _arm(self, t, z_ref):
        """A sample with the reference's pads at `z_ref`, at or below where it arms: watched from
        here on, the slow rise once the reference has stood below slow_z."""
        if self.armed_t is None:
            self.armed_t = t
        if self.slow_t is None and z_ref <= self.slow_z:
            self.slow_t = t

    def _scale(self, t, frac, prev):
        """The driver's applied speed scale read at an armed sample heard at `t` (`prev` the kept
        sample before it): the trajectory's time x how far its fraction went over the last
        SCALE_SPAN_S of them, by their heard times (SCALE_*). A reading more than SCALE_CHANGE off
        the scale the watch settled at re-settles it (_resettle). Returns whether it did."""
        if prev is not None and t - prev > SCALE_GAP_S:
            self._scale_q.clear()  # heard bunched after it: no measure of the driver's clock
            self._scale_from = t + SCALE_GAP_S
        if not 0.0 < frac < 1.0 or self.traj.T <= 0.0 or (self._scale_from is not None and t < self._scale_from):
            return False  # not flying yet, or done: its clock stands
        win = self._scale_q
        win.append((t, frac))
        while len(win) > 1 and t - win[1][0] >= SCALE_SPAN_S:
            win.popleft()
        t0, f0 = win[0]
        if t - t0 < SCALE_SPAN_S:
            return False
        s = self.traj.T * (frac - f0) / (t - t0)
        lo, hi = self.scale_seen or (s, s)
        self.scale_seen = (min(lo, s), max(hi, s))
        if self.scale_ref is None:
            self.scale_ref = s
            return False
        if abs(s / max(self.scale_ref, 1e-3) - 1.0) <= SCALE_CHANGE:
            return False
        self.resettles.append((t, self.scale_ref, s))
        self.scale_ref = s
        self._resettle(t)
        return True

    def _resettle(self, t):
        """The applied speed scale changed at `t` (_scale): the arm's lag behind the reference
        changes with its speed, so both rules start again, as at the slow leg's start — the fast
        rise over the least from here on, the slow one once it has settled again."""
        self._win.clear()
        self._cand, self._n_above = None, 0
        self._slow_least, self._n_slow = None, 0
        if self.slow_t is not None:
            self.slow_t = t

    def _rule(self):
        """The fast rule it judges by: (rise, window, samples in a row — both rules')."""
        return (
            CONTACT_RISE_M if self.rise_m is None else self.rise_m,
            CONTACT_WINDOW_S if self.window_s is None else self.window_s,
            CONTACT_SAMPLES if self.rise_samples is None else self.rise_samples,
        )

    def _judge(self, t, frac, z_ref, z_meas):
        """The rules on one armed sample: the pads measured at `z_meas`, the reference's at
        `z_ref`."""
        gap = z_meas - z_ref
        self.last_gap = gap
        if self.contact is not None or self.verdict is not None:
            return
        rise, window, in_a_row = self._rule()
        sample = (t, gap, z_meas)
        win, hist = self._win, self._hist
        win.append(sample)
        hist.append(sample)
        while win and t - win[0][0] > window:
            win.popleft()
        while hist and t - hist[0][0] > HISTORY_S:
            hist.popleft()
        # a fast rise: over the least of the last CONTACT_WINDOW_S
        if self._cand is not None and gap - self._cand[1] > rise:
            self._n_above += 1
        else:
            least = min(win, key=lambda s: s[1])
            self._cand, self._n_above = (least, 1) if gap - least[1] > rise else (None, 0)
        # a slow one: over the least since the slow leg's start settled
        if self.slow_t is not None and t - self.slow_t >= CONTACT_SETTLE_S:
            if self._slow_least is None or gap < self._slow_least[1]:
                self._slow_least, self._n_slow = sample, 0
            elif gap - self._slow_least[1] > CONTACT_SLOW_RISE_M:
                self._n_slow += 1
            else:
                self._n_slow = 0
        if self._n_above >= in_a_row:
            self._met(t, frac, self._cand, "fast")
        elif self._n_slow >= in_a_row:
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
            if self.shadow is not None:  # the shadow afresh too, on the same trajectory
                self.shadow._reset()
                self.shadow._sent, self.shadow.traj, self.shadow.arm_frac = info, self.traj, self.arm_frac
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

    def watch_beside(self, rise_m, window_s, rise_samples):
        """A shadow watch beside this one (self.shadow): the same press judged by the fast rule
        `rise_m` over the least of the last `window_s`, on `rise_samples` kept samples in a row
        (both rules' count, as CONTACT_SAMPLES is), the slow rule's settle and rise as they are.
        It judges each sample this one does, as this one judges it (no kinematics of its own),
        and cuts and refuses nothing: never a stop_when, it stops nothing, and one that fails
        only says so. Its verdict is in this one's record ('shadow'). Returns it."""
        self.shadow = PressContact(
            self.chain,
            self.client,
            self.top_z,
            self.arm_z,
            self.past_m,
            self.end_z,
            cut=False,
            final=self.final,
            slow_z=self.slow_z,
            early_m=self.early_m,
            late_m=self.late_m,
            rise_m=rise_m,
            window_s=window_s,
            rise_samples=rise_samples,
        )
        return self.shadow

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
        there (`cut_mm` how much short of its planned end), or why not; the
        longest between two kept samples it watched (`feedback_gap_max_ms`),
        the driver's applied speed scale read (`speed_scale`, least and
        most) and where a change of it re-settled the watch (`resettles`),
        and what a shadow beside it made of the press (`shadow`)."""
        sent = None if self._sent is None else self._sent.get("sent_at")
        refused = self.refusal()
        out = {
            "watched": self.armed_t is not None,
            "cut_on": self.cut,
            "samples": self.samples,
            "feedback_gap_max_ms": None if self.gap_max_s is None else round(1000 * self.gap_max_s, 1),
            # the driver's applied speed scale as the fraction's rate read it, and where a change
            # of it re-settled the watch (s after the goal went out, the scale from, to)
            "speed_scale": None if self.scale_seen is None else [round(v, 3) for v in self.scale_seen],
            "resettles": [
                [None if sent is None else round(t - float(sent), 3), round(a, 3), round(b, 3)]
                for t, a, b in self.resettles
            ],
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
        if self.shadow is not None:
            out["shadow"] = self._shadow_record()
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

    def _shadow_record(self):
        """The record's 'shadow': its rule; whether it met the box, how, where (the pads' height,
        how far above the cameras' top) and how long after this watch's detection (ms, - before
        it); its own verdict (early | late: it would have refused the press there); and
        `stopped_first` where this watch stopped the stroke before it met anything."""
        s = self.shadow
        rise, window, in_a_row = s._rule()
        out = {
            "rule": {"rise_mm": round(1000 * rise, 1), "window_s": round(window, 3), "samples": in_a_row},
            "met": s.contact is not None,
            "verdict": s.verdict,
        }
        if s.failed is not None:
            out["failed"] = s.failed
        if s.contact is None:
            out["stopped_first"] = self.stopped_frac is not None
            return out
        z_c = s.contact[1]
        out.update(
            how=s.how,
            z=round(z_c, 4),
            above_top_mm=round(1000 * (z_c - self.top_z), 1),
            after_real_ms=None if self.detected_t is None else round(1000 * (s.detected_t - self.detected_t)),
        )
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
