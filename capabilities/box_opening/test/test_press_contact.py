"""Contact in the compliant press (tasks/press_contact): where the pads met
the box, from the gap between their height measured and the reference's,
and the stroke ended as far past it as it was planned past the cameras'
top — never deeper than planned."""

from types import SimpleNamespace

import numpy as np
import pytest

from box_opening.tasks import depth_press as dp
from box_opening.tasks import press_contact as pc

PAD = dp.DOWN_CONTACT_TOOL[2]  # the pads below the tool frame (TCP_OFFSET_M)
TOP = 0.30  # where the cameras put the button's top (pads height)
END = TOP - 0.012 - 0.020  # the stroke's planned end: its travel and a 20 mm overshoot below that
HZ = 90.0  # the client's kept samples


class _Chain:
    """Joint 1 is the tool frame's height; nothing else moves."""

    def fk(self, q):
        return np.eye(3), np.array([0.0, 0.0, float(q[0])])


def _stamp(t):
    return SimpleNamespace(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def _traj(z_pads, dt):
    """A trajectory through the pads heights `z_pads`, `dt` apart."""
    return SimpleNamespace(
        points=[
            SimpleNamespace(positions=[z - PAD] + [0.0] * 6, time_from_start=_stamp(0.02 + i * dt))
            for i, z in enumerate(z_pads)
        ]
    )


def _ref(z0=0.40, z1=END, speed=0.025):
    """The reference's pads straight down from z0 to z1 at `speed` (m/s)."""
    n = int(round((z0 - z1) / speed / 0.05))
    return np.linspace(z0, z1, n + 1), 0.05


def _fly(det, client, z_pads, dt, measured, cut_seen=None):
    """The client flying `z_pads`: every kept sample, the measured pads at
    measured(t, z_ref); stop_when polled after each. Returns (the fraction
    it stopped at or None, the samples)."""
    traj = _traj(z_pads, dt)
    T = 0.02 + (len(z_pads) - 1) * dt
    info = {"sent_at": 100.0, "sent_traj": traj, "feedback_q": []}
    client.last_execution = info
    t, frac = 100.0, 0.0
    while frac < 1.0:
        frac = min(1.0, (t - 100.0) / T)
        z_ref = float(np.interp(frac * T, [0.02 + i * dt for i in range(len(z_pads))], z_pads))
        z_m = measured(t - 100.0, z_ref)
        info["feedback_q"].append((t, frac, [z_m - PAD] + [0.0] * 6, None))
        if det(frac):
            return frac, info
        t += 1.0 / HZ
    return None, info


def _watch(cut=True):
    client = SimpleNamespace(last_execution=None)
    det = pc.PressContact(_Chain(), client, TOP, TOP + dp.TOUCH_SLOW_M, TOP - END, END, cut=cut)
    return det, client


def _free(noise=0.0003, lag=-0.002, seed=0):
    rng = np.random.default_rng(seed)
    return lambda t, z: z + lag + float(rng.normal(0.0, noise))


def test_a_hard_stop_is_met_where_the_pads_stopped_and_the_stroke_ends_as_far_past_it_as_planned():
    """The box 10 mm higher than the cameras put it: the pads stop at 0.31.
    The stroke was planned 32 mm past the top; it ends 32 mm past where
    they stopped — 10 mm short of its planned end — and the driver holds
    the reference there."""
    det, client = _watch()
    z, dt = _ref()
    free = _free()
    stopped, _info = _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), 0.31))
    assert stopped is not None
    t_c, z_c, _g = det.contact
    assert z_c == pytest.approx(0.31, abs=0.0015)
    assert det.stop_z == pytest.approx(z_c - (TOP - END))
    assert pc.pads_z(_Chain(), det.held_ref) == pytest.approx(det.stop_z, abs=0.0005)  # the reference held there
    rec = det.record()
    assert rec["met"] and rec["cut"] and rec["cut_on"] and rec["above_top_mm"] == pytest.approx(10.0, abs=1.5)
    assert rec["cut_mm"] == pytest.approx(10.0, abs=1.5) and rec["past_mm"] == pytest.approx(32.0)
    assert 0 < rec["detected_after_ms"] <= 1000 * pc.CONTACT_WINDOW_S + 1000 * pc.CONTACT_SAMPLES / HZ


def test_the_speed_change_where_the_slow_leg_begins_is_no_contact():
    """The reference slows where the slow leg begins; the measured pads dip
    4 mm below it and come back at 5 mm/s (the recorded presses: 3-5 mm,
    5-6 mm/s). No contact, and the stroke flies to its end."""
    det, client = _watch()
    fast = np.linspace(0.40, TOP + dp.TOUCH_SLOW_M, 9)  # 0.05 s apart: ~70 mm/s
    slow = np.linspace(TOP + dp.TOUCH_SLOW_M, END, 54)[1:]  # ~25 mm/s
    z = np.concatenate([fast, slow])
    t_arm = 0.02 + 8 * 0.05

    def measured(t, zr):
        dip = 0.0
        if t > t_arm:
            s = t - t_arm
            dip = -0.004 * min(1.0, s / 0.1) if s < 0.1 else -0.004 + 0.005 * min(s - 0.1, 0.8)
        return zr - 0.002 + dip

    stopped, _info = _fly(det, client, z, 0.05, measured)
    assert stopped is None and det.contact is None
    rec = det.record()
    assert (rec["watched"], rec["cut_on"], rec["past_mm"], rec["met"], rec["verdict"]) == (
        True,
        True,
        32.0,
        False,
        None,
    )


def _yield(onset):
    """The pads press on down into something that yields from `onset` on, at
    half the reference's speed (10-29 14:31, 14:45; 10-30 13:35:48): the gap
    grows 12.5 mm/s, slower than a fast rise."""

    def measured(t, zr):
        return zr - 0.002 if zr - 0.002 > onset else onset - 0.5 * (onset - (zr - 0.002))

    return measured


def test_a_slow_yield_is_met_where_it_begins():
    """Begun once the slow leg's start has settled: the slow rise meets it
    where the pads began to lag, and the stroke ends as far past that as
    planned."""
    det, client = _watch()
    z, dt = _ref()
    onset = TOP + 0.015
    stopped, _info = _fly(det, client, z, dt, _yield(onset))
    assert stopped is not None and det.how == "slow"
    assert det.contact[1] == pytest.approx(onset, abs=0.002)
    assert det.record()["cut_mm"] == pytest.approx(15.0, abs=2.0)


def test_a_slow_yield_begun_as_the_slow_leg_starts_is_met_late_and_cuts_less():
    """Begun inside CONTACT_SETTLE_S, where the slow leg's start still dips
    the gap: no slow rise is judged there, so it is met once that has
    settled, lower than it began — the stroke cut less, never deeper than
    planned. (The recorded presses that met the box this early rose fast:
    10-30 13:28, 21 mm/s.)"""
    det, client = _watch()
    z, dt = _ref()
    onset = TOP + 0.032
    stopped, _info = _fly(det, client, z, dt, _yield(onset))
    assert stopped is not None and det.how == "slow"
    assert onset - 0.012 < det.contact[1] < onset
    assert 0.0 < det.record()["cut_mm"] < 32.0


def test_a_contact_below_the_cameras_top_leaves_the_stroke_as_planned():
    """The pads stop 10 mm below the top: as far past that as planned is
    deeper than the stroke goes. Never deeper than planned: nothing cut."""
    det, client = _watch()
    z, dt = _ref()
    free = _free()
    stopped, _info = _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), TOP - 0.010))
    assert stopped is None and det.contact is not None and det.stop_z is None
    rec = det.record()
    assert rec["met"] and not rec["cut"] and rec["why"] == "the planned end is no deeper than that past the contact"


@pytest.mark.parametrize("off", ["cut", "flag"])
def test_watched_only_it_says_where_it_met_and_cuts_nothing(monkeypatch, off):
    """cut=False, or PRESS_FROM_CONTACT off: the contact is met and recorded
    with where the stroke would have ended, nothing cut."""
    if off == "flag":
        monkeypatch.setattr(pc, "PRESS_FROM_CONTACT", False)
    det, client = _watch(cut=off != "cut")
    z, dt = _ref()
    free = _free()
    stopped, _info = _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), 0.31))
    assert stopped is None and det.held_ref is None
    rec = det.record()
    assert rec["met"] and not rec["cut"] and not rec["cut_on"] and rec["why"] == "watched only"
    assert rec["cut_mm"] == pytest.approx(10.0, abs=1.5)  # what it would have cut


def test_nothing_is_judged_before_the_slow_leg():
    """Above where the slow leg begins the reference moves fast and the gap
    with it: a jump there is no contact."""
    det, client = _watch()
    z, dt = _ref()

    def measured(t, zr):
        return zr + (0.02 if zr > TOP + dp.TOUCH_SLOW_M + 0.01 else -0.002)

    stopped, _info = _fly(det, client, z, dt, measured)
    assert stopped is None and det.contact is None and det.armed_t is not None


def test_a_reference_already_past_where_the_stroke_should_end_stops_at_once():
    """Contact seen late (the samples read in a batch after the lookahead):
    the reference is already past where the stroke should end — it stops at
    the next poll, not never."""
    det, client = _watch()
    z, dt = _ref()
    T = 0.02 + (len(z) - 1) * dt
    traj = _traj(z, dt)
    info = {"sent_at": 0.0, "sent_traj": traj, "feedback_q": []}
    client.last_execution = info
    free = _free()
    ts = np.arange(0.0, T, 1.0 / HZ)
    for t in ts:
        frac = t / T
        z_ref = float(np.interp(t, [0.02 + i * dt for i in range(len(z))], z))
        info["feedback_q"].append((t, frac, [max(free(t, z_ref), 0.31) - PAD] + [0.0] * 6, None))
    assert det(0.99)  # every sample at once: met, and the reference long past its stop
    assert det.stopped_frac == 0.99


def test_a_new_execution_is_watched_afresh():
    det, client = _watch()
    z, dt = _ref()
    free = _free()
    assert _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), 0.31))[0] is not None
    stopped, _info = _fly(det, client, z, dt, _free(seed=1))  # the next execution: free air all the way
    assert stopped is None and det.contact is None and det.stopped_frac is None


def test_a_watcher_that_fails_stops_nothing():
    class Broken(_Chain):
        def fk(self, q):
            raise ValueError("no chain")

    client = SimpleNamespace(last_execution={"sent_at": 0.0, "sent_traj": _traj([0.4, 0.3], 0.05), "feedback_q": []})
    det = pc.PressContact(Broken(), client, TOP, TOP + 0.035, 0.032, END)
    assert det(0.5) is False and det(0.6) is False  # the set-down's: it flies its planned end
    assert det.failed == "ValueError: no chain" and det.record()["failed"] == det.failed  # said, not swallowed
    assert pc.PressContact(_Chain(), SimpleNamespace(), TOP, TOP + 0.035, 0.032, END)(0.5) is False  # no execution


def _set_down(final):
    """reach_lid_off.watch_setdown's watch on one execution: the lift off a
    box standing lower than where the watch begins (the pads held back 6 mm
    by the lid's latch, then springing after the reference as it comes
    free), up and over, and down to the touch's end 10 mm over where the
    plan has the lid meet the table — which it meets 40 mm higher."""
    table = 0.20  # where the plan has the lid meet the table (pads)
    arm, end, met = table + 0.08, table + 0.010, table + 0.040
    client = SimpleNamespace(last_execution=None)
    det = pc.PressContact(_Chain(), client, table, arm, 0.0, end, cut=True, final=final)
    z = np.concatenate([np.linspace(0.25, 0.40, 31), np.linspace(0.40, end, 80)[1:]])

    def measured(t, zr):
        if t < 0.3:
            return zr - 0.006  # the latch holds the lid, and the pads with it
        if t < 1.6:
            return zr - 0.001  # free: the pads a millimetre behind the rising reference
        return max(zr + 0.002, met)  # down: a little above it, until the table holds the lid

    stopped, _info = _fly(det, client, z, 0.05, measured)
    return det, stopped, met, end


def test_a_final_watch_judges_only_the_last_stretch_below_where_it_arms_and_cuts_where_the_lid_meets():
    """final=True: the lid springing free in the lift below where the watch
    begins is not judged; the last stretch down is, and it is cut where the
    lid meets the table (past 0), above the touch's end — the lid let go
    there."""
    det, stopped, met, end = _set_down(final=True)
    assert stopped is not None and det.how == "fast"
    assert det.contact[1] == pytest.approx(met, abs=0.002)
    assert det.stop_z == pytest.approx(det.contact[1]) and det.stop_z > end
    assert pc.pads_z(_Chain(), det.held_ref) < met  # the detection's lag under where it met ...
    assert pc.pads_z(_Chain(), det.held_ref) > met - 0.015  # ... no more
    rec = det.record()
    assert rec["cut"] and rec["above_top_mm"] == pytest.approx(40.0, abs=2.0) and rec["past_mm"] == 0.0


def test_a_watch_from_the_first_time_under_where_it_arms_would_take_the_lift_for_the_table():
    """Without final, the same execution is watched from its start, below
    where the watch begins: the pads springing after the reference as the
    lid comes free off the box rise like a contact, and the set-down is cut
    there — the lid let go over the box."""
    det, stopped, met, _end = _set_down(final=False)
    assert stopped is not None and det.contact[1] < 0.30  # met in the lift, not at the table
    assert abs(det.contact[1] - met) > 0.02


def test_a_final_watch_of_a_trajectory_that_ends_above_where_it_arms_watches_nothing():
    client = SimpleNamespace(last_execution=None)
    det = pc.PressContact(_Chain(), client, TOP, TOP + 0.08, 0.0, TOP + 0.09, cut=True, final=True)
    stopped, _info = _fly(det, client, np.linspace(0.30, 0.45, 20), 0.05, _free())
    assert stopped is None and det.armed_t is None and det.arm_frac is None


# -- review round 2: the whole descent watched, and the refusal band -------------------------------


def _descent(end=END, start=0.40):
    """depth_press.watch_press's watch on a stroke from `start` down: armed DESCENT_ARM_M under
    it (its last stretch below that), the slow rise from the slow leg's start, the band on."""
    client = SimpleNamespace(last_execution=None)
    det = pc.PressContact(
        _Chain(),
        client,
        TOP,
        start - dp.DESCENT_ARM_M,
        TOP - end,
        end,
        cut=True,
        final=True,
        slow_z=TOP + dp.TOUCH_SLOW_M,
        early_m=pc.CONTACT_EARLY_M,
        late_m=pc.CONTACT_LATE_M,
        abort_on_failure=True,
    )
    return det, client


def test_a_contact_above_the_slow_leg_is_met_as_the_pads_meet_it():
    """Review round 2: armed only TOUCH_SLOW_M above the top, the watch did not see a box met
    higher until the reference got there, pushing on into it (10-30 13:35: met 50 mm above
    the top, seen at 37 mm). Watching the whole descent, it is met as the pads meet it."""
    hit = TOP + 0.040  # above the slow leg (35 mm), inside the band (45 mm)
    z, dt = _ref()
    free = _free()
    old, oc = _watch()
    _fly(old, oc, z, dt, lambda t, zr: max(free(t, zr), hit))
    new, nc = _descent()
    stopped, _info = _fly(new, nc, z, dt, lambda t, zr: max(free(t, zr), hit))
    assert stopped is not None and new.verdict is None and new.refusal() is None
    assert new.contact[1] == pytest.approx(hit, abs=0.0015)
    assert new.detected_t - new.contact[0] <= pc.CONTACT_WINDOW_S + pc.CONTACT_SAMPLES / HZ
    assert old.detected_t - new.detected_t > 0.15  # the old watch saw it once its reference got low enough
    assert new.stop_z == pytest.approx(new.contact[1] - (TOP - END))  # pressed its own depth past it, no more


def test_a_contact_well_above_the_cameras_top_refuses_the_press_where_it_is_met():
    """Something else in the way: met more than CONTACT_EARLY_M above the top. The stroke is
    stopped there, nothing pressed past it, and the run is told why."""
    hit = TOP + pc.CONTACT_EARLY_M + 0.010
    det, client = _descent()
    z, dt = _ref()
    free = _free()
    stopped, _info = _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), hit))
    assert stopped is not None and det.verdict == "early" and det.stop_z is None
    assert pc.pads_z(_Chain(), det.held_ref) > hit - 0.012  # held a few mm into it, not the stroke's depth
    verdict, why = det.refusal()
    assert verdict == "early" and "55 mm above" in why and "something else is in the way" in why
    assert det.record()["verdict"] == "early" and det.record()["band_mm"] == [-25, 45]


def test_nothing_met_well_below_the_top_refuses_the_press_there():
    """The box is not where it was seen (here, 40 mm lower): the pads pass CONTACT_LATE_M below
    the top meeting nothing. Stopped there: nothing pressed past it."""
    det, client = _descent()
    z, dt = _ref()
    free = _free()
    stopped, _info = _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), TOP - 0.040))
    assert stopped is not None and det.verdict == "late" and det.contact is None
    assert det.late_at[1] == pytest.approx(TOP - pc.CONTACT_LATE_M - pc.CONTACT_LAG_M, abs=0.001)
    verdict, why = det.refusal()
    assert verdict == "late" and "the box is not where it was seen" in why


def test_a_contact_met_late_within_the_band_is_pressed():
    """Met 15 mm below the top (the recorded presses: as low as 13 mm): no refusal; as far past
    it as planned is deeper than the stroke goes, so it flies as planned."""
    det, client = _descent()
    z, dt = _ref()
    free = _free()
    stopped, _info = _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), TOP - 0.015))
    assert stopped is None and det.contact is not None and det.verdict is None and det.refusal() is None


def test_a_contact_met_below_the_late_limit_refuses_the_press():
    """Met, but lower than CONTACT_LATE_M below the top: the box is not where it was seen."""
    end = TOP - 0.060
    det, client = _descent(end=end)
    z, dt = _ref(z1=end)
    free = _free()
    stopped, _info = _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), TOP - pc.CONTACT_LATE_M - 0.002))
    assert stopped is not None and det.verdict == "late"


def test_a_stroke_that_ends_with_nothing_met_is_refused():
    """A stroke whose end stands above the late limit, nothing met all the way: refused once it
    is over ("never")."""
    end = TOP - 0.020
    det, client = _descent(end=end)
    z, dt = _ref(z1=end)
    stopped, _info = _fly(det, client, z, dt, _free())
    assert stopped is None and det.verdict is None and det.contact is None
    verdict, why = det.refusal()
    assert verdict == "never" and "meeting nothing" in why


def test_the_slow_legs_speed_change_is_no_contact_watched_from_the_descents_start():
    """Watched from the descent's start, the slow rise is still judged only once the slow leg's
    start has settled (slow_z): the gap's dip there and its recovery are no contact. Free air
    down past the late limit: refused late, never met."""
    det, client = _descent()
    fast = np.linspace(0.40, TOP + dp.TOUCH_SLOW_M, 9)
    slow = np.linspace(TOP + dp.TOUCH_SLOW_M, END, 54)[1:]
    z = np.concatenate([fast, slow])
    t_arm = 0.02 + 8 * 0.05

    def measured(t, zr):
        dip = 0.0
        if t > t_arm:
            s_ = t - t_arm
            dip = -0.004 * min(1.0, s_ / 0.1) if s_ < 0.1 else -0.004 + 0.005 * min(s_ - 0.1, 0.8)
        return zr - 0.002 + dip

    _fly(det, client, z, 0.05, measured)
    assert det.contact is None and det.armed_t is not None and det.slow_t > det.armed_t
    assert det.verdict == "late"


def test_a_watch_that_fails_stops_the_press_there():
    """Review round 2: a watcher that raised was swallowed and the full stroke flew. The press's
    watch (abort_on_failure) says it and stops the stroke at once; the run is told why."""

    class Broken(_Chain):
        def fk(self, q):
            raise ValueError("no chain")

    client = SimpleNamespace(last_execution={"sent_at": 0.0, "sent_traj": _traj([0.4, 0.3], 0.05), "feedback_q": []})
    det = pc.PressContact(Broken(), client, TOP, TOP + 0.035, 0.032, END, abort_on_failure=True)
    assert det(0.4) is True and det(0.5) is True and det.stopped_frac == 0.4
    verdict, why = det.refusal()
    assert verdict == "failed" and "ValueError: no chain" in why
