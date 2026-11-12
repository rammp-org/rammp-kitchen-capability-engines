import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening.runtime.guards import (
    TorqueGuard,
    in_band,
    sanity_violations,
)


def _traj(rows, dt=0.5):
    t = JointTrajectory()
    t.joint_names = ["joint_%d" % i for i in range(1, len(rows[0]) + 1)]
    for i, row in enumerate(rows):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in row]
        p.velocities = [0.0] * len(row)
        p.accelerations = [0.0] * len(row)
        p.time_from_start.sec = int(i * dt)
        p.time_from_start.nanosec = int((i * dt % 1) * 1e9)
        t.points.append(p)
    return t


def test_guard_baseline_anchored_at_progress():
    g = TorqueGuard(touch_nm=3.0)
    assert g.on_efforts([9.0, 9.0, 9.0, 9.0]) is False  # not armed: ignored
    g.on_progress(0.0)
    assert g.armed is False  # progress 0 != started
    g.on_progress(0.01)
    assert g.on_efforts([1.0, 1.0, 1.0, 1.0]) is False  # first sample = baseline
    assert g.on_efforts([2.0, 1.0, 1.0, 1.0]) is False  # dev 1.0 < 3.0
    assert g.on_efforts([1.0, 5.0, 1.0, 1.0]) is True  # dev 4.0 > 3.0
    assert g.peak == pytest.approx(4.0)


def test_guard_ignores_none_efforts():
    g = TorqueGuard(touch_nm=3.0)
    g.on_progress(0.5)
    assert g.on_efforts(None) is False


def test_sanity_gate_flags_wandering_joint():
    # joint_1 wanders 1.0 rad out and back on a 0.1 rad net move
    bad = _traj([[0.0, 0.0], [1.0, 0.05], [0.1, 0.1]])
    good = _traj([[0.0, 0.0], [0.05, 0.05], [0.1, 0.1]])
    assert sanity_violations(good, margin_rad=0.35) == []
    v = sanity_violations(bad, margin_rad=0.35)
    assert len(v) == 1 and "joint_1" in v[0]


def test_sanity_gate_wrap_aware():
    # joint crossing the pi boundary: 3.10 -> -3.10 is a 0.08 rad move
    t = _traj([[3.10], [3.14], [-3.10]])
    assert sanity_violations(t, margin_rad=0.35) == []


def test_in_band():
    assert in_band(0.6, (0.55, 0.75))
    assert not in_band(0.8, (0.55, 0.75))  # closed on air


def test_time_fraction_conversion_tracks_the_path_not_the_clock():
    """progress is elapsed/duration; contact is expected at a DISTANCE.

    A profile that covers most of its path early must report a LOWER time
    fraction for the same path fraction than a constant-speed one would."""
    from box_opening.runtime.guards import time_fraction_at_path_fraction

    # front-loaded: 90% of the path covered in the first half of the points
    fast_first = _traj([[0.0], [0.45], [0.9], [0.95], [1.0]], dt=1.0)
    frac = time_fraction_at_path_fraction(fast_first, 0.9)
    assert frac < 0.9, "front-loaded motion reaches 90%% of path early in time"

    # uniform motion: time fraction and path fraction agree
    uniform = _traj([[0.0], [0.25], [0.5], [0.75], [1.0]], dt=1.0)
    assert time_fraction_at_path_fraction(uniform, 0.5) == pytest.approx(0.6, abs=0.21)
    # degenerate inputs fall back to the requested fraction
    assert time_fraction_at_path_fraction(_traj([[0.0]], dt=1.0), 0.42) == pytest.approx(0.42)


def test_a_trip_says_what_tripped_it():
    """Bench 2026-09-21 13:10: a false trip 87 mm above the button, and the
    log's only number — "torque_peak 12.9" against a 3 Nm threshold — was
    the deviation from the GROUP-START baseline, seen long before the guard
    re-took its baseline and armed. Nothing said which joint, how far from
    the baseline in force, or how soon after it was taken. A trip now
    carries that, and the peak is measured from the baseline in force."""
    g = TorqueGuard(3.0, rebaseline_after=0.5, arm_after=0.5)
    g.on_progress(0.1)
    assert not g.on_efforts([10.0, 1.0, 1.0, 1.0])
    assert not g.on_efforts([22.0, 1.0, 1.0, 1.0])  # 12 Nm of free-air dynamics, unarmed
    assert g.peak == pytest.approx(12.0)
    g.on_progress(0.5)  # the baseline is re-taken here, and the guard arms
    assert not g.on_efforts([20.0, 1.0, 1.0, 1.0])
    assert g.peak == 0.0  # ... so the peak starts again with it
    assert not g.on_efforts([20.5, 1.0, 2.0, 1.0])
    assert g.trip_report() is None
    g.on_progress(0.52)
    assert g.on_efforts([20.4, 1.0, 4.6, 1.0])
    assert g.peak == pytest.approx(3.6)
    rep = g.trip_report()
    assert rep["joint"] == 2 and rep["dev_nm"] == pytest.approx(3.6)
    assert rep["baseline"] == [20.0, 1.0, 1.0, 1.0] and rep["efforts"] == [20.4, 1.0, 4.6, 1.0]
    assert rep["baseline_at_progress"] == pytest.approx(0.5) and rep["progress"] == pytest.approx(0.52)
    assert rep["baseline_age_s"] >= 0.0
    # the run-up to the trip: (seconds before the trip, progress, efforts), newest last
    assert rep["recent"][-1][1:] == [0.52, 20.4, 1.0, 4.6, 1.0]
    assert rep["recent"][-1][0] == pytest.approx(0.0, abs=1e-3)
    assert len(rep["recent"]) == 5
