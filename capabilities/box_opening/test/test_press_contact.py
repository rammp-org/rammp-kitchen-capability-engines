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


def _fly(det, client, z_pads, dt, measured, cut_seen=None, lost=None, clock=None, heard=None):
    """The client flying `z_pads`: every kept sample, the measured pads at
    measured(t, z_ref); stop_when polled after each. lost(t): a sample the
    client did not keep (a gap in the feedback). clock(t): the driver's
    applied speed scale, its trajectory clock running at it (None: 1.0).
    heard(t): when the client heard the sample sent at t (None: then).
    Returns (the fraction it stopped at or None, the samples)."""
    traj = _traj(z_pads, dt)
    T = 0.02 + (len(z_pads) - 1) * dt
    info = {"sent_at": 100.0, "sent_traj": traj, "feedback_q": []}
    client.last_execution = info
    t, frac, traj_t = 100.0, 0.0, 0.0
    while frac < 1.0:
        frac = min(1.0, (t - 100.0) / T) if clock is None else min(1.0, traj_t / T)
        z_ref = float(np.interp(frac * T, [0.02 + i * dt for i in range(len(z_pads))], z_pads))
        z_m = measured(t - 100.0, z_ref)
        if clock is not None:
            traj_t += clock(t - 100.0) / HZ
        if lost is not None and lost(t - 100.0) and frac < 1.0:
            t += 1.0 / HZ
            continue
        at = t if heard is None else 100.0 + heard(t - 100.0)
        info["feedback_q"].append((at, frac, [z_m - PAD] + [0.0] * 6, None))
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


# -- the press study (2026-11-05): a shadow beside the watch, and the feedback's gaps --------------


def _shadowed(**kw):
    """_descent's watch with the candidate fast rule judging beside it (watch_press's shadow)."""
    det, client = _descent(**kw)
    det.watch_beside(pc.SHADOW_RISE_M, pc.SHADOW_WINDOW_S, pc.SHADOW_SAMPLES)
    return det, client


def _without(rec):
    return {k: v for k, v in rec.items() if k != "shadow"}


# what the watch decides, one of each: (the measured pads, given free air), and its name
_STREAMS = [
    lambda free: lambda t, zr: max(free(t, zr), TOP + 0.010),  # met 10 mm high: cut past it
    lambda free: lambda t, zr: max(free(t, zr), TOP + pc.CONTACT_EARLY_M + 0.010),  # refused early
    lambda free: lambda t, zr: max(free(t, zr), TOP - 0.040),  # refused late
    lambda free: _yield(TOP + 0.015),  # a slow yield
    lambda free: free,  # free air to the end
]
_STREAM_IDS = ["cut", "early", "late", "yield", "free"]


@pytest.mark.parametrize("measured", _STREAMS, ids=_STREAM_IDS)
def test_a_shadow_beside_the_watch_changes_nothing_it_decides(measured):
    """The real watch with a shadow beside it stops the stroke where it did without one, its
    record the same but for the shadow's."""
    alone, ac = _descent()
    beside, bc = _shadowed()
    z, dt = _ref()
    a = _fly(alone, ac, z, dt, measured(_free()))[0]
    b = _fly(beside, bc, z, dt, measured(_free()))[0]
    assert a == b and alone.held_ref == beside.held_ref and alone.refusal() == beside.refusal()
    assert _without(beside.record()) == alone.record() and "shadow" not in alone.record()
    assert not beside.shadow.cut and beside.shadow.stopped_frac is None  # it cuts nothing, asked to stop nothing


def test_the_shadow_meets_a_slow_yield_by_its_fast_rule_before_the_watch_does():
    """A yield the pads press on into at half the reference's speed (12.5 mm/s): the real fast
    rule needs 15 mm/s and the slow rule meets it once the slow leg's start has settled; the
    candidate's 4 mm over 0.4 s meets it by the fast rule, earlier, near where it began."""
    det, client = _shadowed()
    z, dt = _ref()
    onset = TOP + 0.015
    stopped, _info = _fly(det, client, z, dt, _yield(onset))
    assert stopped is not None and det.how == "slow"
    shadow = det.record()["shadow"]
    assert shadow["rule"] == {"rise_mm": 4.0, "window_s": 0.4, "samples": 1}
    assert shadow["met"] and shadow["how"] == "fast" and shadow["verdict"] is None
    assert shadow["after_real_ms"] < 0  # before the real watch's detection
    assert shadow["z"] == pytest.approx(onset, abs=0.002) and shadow["above_top_mm"] == pytest.approx(15.0, abs=2.0)


def test_the_shadow_says_it_would_have_refused_a_press_the_watch_let_go_on():
    """A box met 55 mm above the cameras' top that yields at 12.5 mm/s (10-30 13:35's, the press
    study's +50 mm box): the real fast rule misses it, and the slow rule meets it once the slow
    leg's start has settled, 20 mm lower, inside the band — pressed and cut past there. The
    shadow meets it where it began, and says it would have refused the press; it stops nothing."""
    det, client = _shadowed()
    z, dt = _ref()
    stopped, _info = _fly(det, client, z, dt, _yield(TOP + 0.055))
    rec = det.record()
    assert stopped is not None and det.how == "slow" and det.verdict is None and rec["cut"]
    assert rec["above_top_mm"] < 1000 * pc.CONTACT_EARLY_M
    shadow = rec["shadow"]
    assert shadow["met"] and shadow["how"] == "fast" and shadow["verdict"] == "early"
    assert shadow["above_top_mm"] == pytest.approx(55.0, abs=2.0) and shadow["after_real_ms"] < -1000


def test_a_stroke_the_watch_stopped_before_the_shadow_met_anything_says_so():
    """A shadow that has met nothing where the real watch stops the stroke saw no more of it."""
    det, client = _descent()
    det.watch_beside(0.050, 0.4, 10_000)  # a rise no press makes, for longer than any stroke
    z, dt = _ref()
    free = _free()
    stopped, _info = _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), TOP + 0.010))
    shadow = det.record()["shadow"]
    assert stopped is not None and not shadow["met"] and shadow["stopped_first"] is True
    det2, client2 = _descent(end=TOP - 0.020)
    det2.watch_beside(0.050, 0.4, 10_000)
    _fly(det2, client2, *_ref(z1=TOP - 0.020), _free())  # free air to the stroke's end: refused "never"
    assert det2.record()["shadow"] == {
        "rule": {"rise_mm": 50.0, "window_s": 0.4, "samples": 10_000},
        "met": False,
        "verdict": None,
        "stopped_first": False,
    }


def test_a_shadow_that_fails_leaves_the_watch_as_it_is():
    """The shadow raising: said in its record; the real watch decides as it would alone."""
    alone, ac = _descent()
    det, client = _shadowed()

    def broken(*_a):
        raise ValueError("shadow")

    det.shadow._judge = broken
    z, dt = _ref()
    free_a, free_b = _free(), _free()
    a = _fly(alone, ac, z, dt, lambda t, zr: max(free_a(t, zr), TOP + 0.010))[0]
    b = _fly(det, client, z, dt, lambda t, zr: max(free_b(t, zr), TOP + 0.010))[0]
    assert a == b is not None and det.failed is None and _without(det.record()) == alone.record()
    assert det.record()["shadow"]["failed"] == "ValueError: shadow" and not det.record()["shadow"]["met"]


def test_the_shadow_is_watched_afresh_with_each_execution():
    det, client = _shadowed()
    z, dt = _ref()
    free = _free()
    _fly(det, client, z, dt, lambda t, zr: max(free(t, zr), TOP + 0.010))
    assert det.shadow.contact is not None
    _fly(det, client, z, dt, _free(seed=1))  # the next execution: free air down to late
    assert det.shadow.contact is None and det.shadow.traj is det.traj and det.shadow._sent is client.last_execution


def test_the_record_keeps_the_longest_gap_in_the_feedback_watched():
    """A 120 ms stretch the client kept no feedback of, on the watched descent: in the record.
    A longer one before the watch arms is no part of it."""
    det, client = _watch()
    z, dt = _ref()
    t_armed = 0.02 + (0.40 - (TOP + dp.TOUCH_SLOW_M)) / 0.025  # where the reference reaches arm_z

    def lost(t):
        return 0.5 <= t < 0.7 or t_armed + 0.2 <= t < t_armed + 0.32

    stopped, _info = _fly(det, client, z, dt, _free(), lost=lost)
    assert stopped is None and det.contact is None
    assert 120.0 < det.record()["feedback_gap_max_ms"] < 120.0 + 2000.0 / HZ  # the samples either side of it
    quiet, qc = _watch()
    _fly(quiet, qc, z, dt, _free())
    assert quiet.record()["feedback_gap_max_ms"] == pytest.approx(1000.0 / HZ, abs=0.2)  # every sample kept
    unarmed, uc = _watch()
    assert unarmed.record()["feedback_gap_max_ms"] is None  # nothing watched


# -- the driver's global speed scale changed mid-stroke (press study E-S10) ---------------------


def _lagging(k=0.4, tau=0.1, noise=0.0003, seed=0):
    """Free air under the joint springs: the pads lag the reference by `k` (s) x its speed (the
    slow leg's start dips the gap up to 0.41 mm per mm/s, press study E-S10), settling with time
    constant `tau`: a change of its speed moves the gap as a contact's rise would."""
    rng = np.random.default_rng(seed)
    last = {}

    def measured(t, zr):
        if last:
            dt = t - last["t"]
            v = (last["z"] - zr) / dt if dt > 0 else 0.0
            last["lag"] = k * v if last["lag"] is None else last["lag"] + (k * v - last["lag"]) * min(1.0, dt / tau)
        else:
            last["lag"] = None
        last.update(t=t, z=zr)
        return zr + (last["lag"] or 0.0) + float(rng.normal(0.0, noise))

    return measured


def _rescaled(t_c, s0, s1, slew=2.0):
    """The driver's applied speed scale: `s0`, from `t_c` on slewed to `s1` at `slew` a second
    (the supervisor's 2.0)."""

    def clock(t):
        if t < t_c:
            return s0
        return min(s1, s0 + slew * (t - t_c)) if s1 > s0 else max(s1, s0 - slew * (t - t_c))

    return clock


@pytest.mark.parametrize("s0, s1", [(0.25, 1.0), (0.5, 1.0), (1.0, 0.5)])
def test_a_change_of_the_drivers_speed_scale_mid_stroke_re_settles_the_watch(s0, s1, monkeypatch):
    """The driver's global speed scale changed while the pads come down through free air, 25 mm
    above the box: the reference's speed changes and the pads' lag behind it with it. Raised,
    the gap rises as on a contact, and a watch that never re-settles meets the box in free air.
    The watch reads the applied scale from how fast the driver's fraction goes, and re-settles
    where it changes: it meets the box where the pads do, as does the shadow beside it."""
    box = TOP - 0.005
    t_c = (0.40 - (TOP + 0.025)) / (0.025 * s0)  # the reference 25 mm above the top

    def fly():
        det, client = _shadowed(start=0.37)
        lag = _lagging()
        _fly(det, client, *_ref(), lambda t, zr: max(lag(t, zr), box), clock=_rescaled(t_c, s0, s1))
        return det.record()

    rec = fly()
    assert rec["met"] and rec["above_top_mm"] == pytest.approx(-5.0, abs=1.5)
    assert rec["how"] == ("fast" if s1 > s0 else "slow")  # met at 12.5 mm/s: slower than a fast rise
    assert rec["speed_scale"] == pytest.approx([min(s0, s1), max(s0, s1)], abs=0.01)
    assert rec["resettles"] and all(t_c < at < t_c + 0.4 + abs(s1 - s0) / 2.0 for at, _a, _b in rec["resettles"])
    assert abs(s1 / rec["resettles"][-1][2] - 1.0) <= pc.SCALE_CHANGE  # settled at the new scale
    shadow = rec["shadow"]
    assert shadow["met"] and shadow["above_top_mm"] == pytest.approx(-5.0, abs=1.5) and shadow["verdict"] is None
    monkeypatch.setattr(pc, "SCALE_CHANGE", float("inf"))  # never re-settled: the watch before E-S10
    blind = fly()
    assert blind["met"] and blind["resettles"] == []
    if s1 > s0:  # sped up: met in free air, 20 mm and more above the box
        assert blind["above_top_mm"] > 15.0
    else:  # slowed: the lag shrinks, the gap falls, nothing read as a rise
        assert blind["above_top_mm"] == pytest.approx(-5.0, abs=1.5)


@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("measured", _STREAMS, ids=_STREAM_IDS)
def test_a_steady_speed_scale_re_settles_nothing(measured, scale, monkeypatch):
    """At a steady scale (1.0, or 0.5 all the way) the watch reads it and decides as one that
    never re-settles: the same stop, the same record."""

    def fly():
        det, client = _shadowed()
        clock = None if scale == 1.0 else (lambda t: scale)
        stopped = _fly(det, client, *_ref(), measured(_free()), clock=clock)[0]
        return stopped, det.held_ref, det.record()

    watched = fly()
    assert watched[2]["resettles"] == [] and watched[2]["speed_scale"] == pytest.approx([scale, scale], abs=1e-6)
    monkeypatch.setattr(pc, "SCALE_CHANGE", float("inf"))
    assert fly() == watched


def test_a_gap_in_the_feedback_is_no_change_of_speed_scale(monkeypatch):
    """The node not spun for 120 ms on the watched descent (press study N12): the samples sent
    meanwhile are heard bunched at its end, as though the driver's clock had stood and then
    run. The scale is read afresh after such a gap: nothing re-settles."""
    t_s = 0.02 + (0.40 - (0.37 - dp.DESCENT_ARM_M)) / 0.025 + 0.5  # 0.5 s after it arms
    t_e = t_s + 0.12

    def heard(t):
        return t_e - 1e-5 * HZ * (t_e - t) if t_s <= t < t_e else t  # in the order sent, at its end

    def fly():
        det, client = _shadowed(start=0.37)
        _fly(det, client, *_ref(), _free(), heard=heard)
        return det.record()

    rec = fly()
    assert rec["feedback_gap_max_ms"] > 110.0 and rec["resettles"] == []
    assert rec["speed_scale"] == pytest.approx([1.0, 1.0], abs=0.15)
    monkeypatch.setattr(pc, "SCALE_GAP_S", float("inf"))  # read across the gap: it would
    assert fly()["resettles"]
