"""A look flies gently (constants.LOOK_*; core.as_look): slow, easing softly out of rest and into it, in
the driver's joint impedance under MEDIUM. Owner 2026-11-16, of the wrist look: "we need to make the arm
look more docile here because it looks snappy and scary"."""

import numpy as np
import pytest
from conftest import Q0, FakeClient
from conftest import leg as make_leg
from conftest import runner as make_runner

from box_opening.constants import (
    COMPLIANT_ACCEL,
    COMPLIANT_DECEL,
    COMPLIANT_QD_CEILING,
    GAINS_MEDIUM,
    HOME_SPEED,
    JOINT_VMAX,
    LOOK_ACCEL,
    LOOK_DECEL,
    LOOK_GAINS,
    LOOK_SPEED,
    TRANSIT_SPEED,
)
from box_opening.primitives.core import as_look
from box_opening.runtime.retime import positions_to_traj, retime_group

NAMES = ["joint_%d" % i for i in range(1, 8)]


def _wrist_turn(rad=0.6, n=40):
    """The wrist look's kind of leg: joints 5-7 turn `rad` where the arm rests, the arm above them still."""
    end = list(Q0)
    for j in (4, 5, 6):
        end[j] += rad
    pts = [list(np.asarray(Q0) + (np.asarray(end) - np.asarray(Q0)) * k / n) for k in range(n + 1)]
    return positions_to_traj(NAMES, pts)


def _flown(traj, speed, params):
    """(duration s, peak joint speed rad/s, seconds from rest to half that peak) of `traj` re-timed."""
    out, info = retime_group([traj], [speed], JOINT_VMAX, params)
    t = np.array([p.time_from_start.sec + p.time_from_start.nanosec / 1e9 for p in out.points])
    q = np.array([p.positions for p in out.points])
    v = np.abs(np.diff(q, axis=0) / np.diff(t)[:, None]).max(axis=1)
    half = t[1:][v >= 0.5 * v.max()][0] - t[0]
    return info["duration_s"], float(v.max()), float(half)


def test_a_look_is_slower_than_a_home_soft_and_eases_its_own_way():
    look = as_look(make_leg("survey:angle-1", start=Q0, end=Q0, speed=TRANSIT_SPEED))
    assert (look.speed, look.compliant, look.gains, look.ease) == (
        LOOK_SPEED,
        True,
        GAINS_MEDIUM,
        (LOOK_ACCEL, LOOK_DECEL),
    )
    assert LOOK_GAINS == GAINS_MEDIUM and LOOK_SPEED < HOME_SPEED < TRANSIT_SPEED
    assert LOOK_ACCEL < COMPLIANT_ACCEL / 3 and LOOK_DECEL < COMPLIANT_DECEL / 2


def test_the_runner_retimes_a_look_with_its_own_ease_whatever_the_runs_mode(tmp_path):
    for compliant in (True, False):
        client = FakeClient()
        client.compliant = compliant
        runner = make_runner(client, tmp_path)
        plain = runner.retime_params(make_leg("reach:look", start=Q0, end=Q0))
        look = runner.retime_params(as_look(make_leg("survey:angle-1", start=Q0, end=Q0)))
        assert (look.accel, look.decel) == (LOOK_ACCEL, LOOK_DECEL) and look.qd_ceiling == COMPLIANT_QD_CEILING
        assert (plain.accel, plain.decel) != (LOOK_ACCEL, LOOK_DECEL)
        assert runner.retime_params().accel == plain.accel  # no leg: the runner's own, as before


def test_a_wrist_turn_flown_as_a_look_is_slow_and_starts_softly(tmp_path):
    """The same 0.6 rad turn of the wrist as the run flew it until 2026-11-16 (a transit under the compliant
    profile) and as a look: about twice as long, no joint over a third of a radian a second, and several
    times as long getting up to speed."""
    client = FakeClient()
    client.compliant = True
    runner = make_runner(client, tmp_path)
    traj = _wrist_turn()
    was_s, was_peak, was_half = _flown(traj, TRANSIT_SPEED, runner.retime_params(make_leg("x", start=Q0, end=Q0)))
    look = as_look(make_leg("survey:angle-1", start=Q0, end=Q0))
    now_s, now_peak, now_half = _flown(traj, look.speed, runner.retime_params(look))
    assert was_peak == pytest.approx(0.45, abs=0.03)  # the wrist's ceiling in the joint impedance
    assert now_peak <= LOOK_SPEED * 0.9 * 1.222 + 0.01 < 0.30
    assert 1.6 * was_s <= now_s <= 3.2  # a view costs a second or two more, and no more
    # out of rest: the speed it gains each second, to half its peak (rad/s^2, the fastest joint)
    was_onset, now_onset = 0.5 * was_peak / was_half, 0.5 * now_peak / now_half
    assert now_onset <= 0.35 * was_onset


def test_the_wrist_looks_views_fly_as_looks(ctx, tmp_path, monkeypatch):
    """survey.fly_view: every view the wrist looks from, the front's and the side survey's alike."""
    from types import SimpleNamespace

    from box_opening.tasks import survey as sv

    client = ctx.client
    runner = make_runner(client, tmp_path)
    flown = []
    monkeypatch.setattr(runner, "run", lambda legs, execute=False: flown.extend(legs) or [SimpleNamespace(ok=True)])
    band = SimpleNamespace(path_refusal=lambda traj: None)
    view = SimpleNamespace(name="survey:angle-1", joints=list(Q0))
    assert sv.fly_view(ctx, runner, view, band, True) == ("flown", None)
    (leg,) = flown
    assert (leg.speed, leg.compliant, leg.gains, leg.ease) == (LOOK_SPEED, True, LOOK_GAINS, (LOOK_ACCEL, LOOK_DECEL))
