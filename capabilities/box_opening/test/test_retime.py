"""The human-motion re-timer: positions untouched, limits by construction,
flow through shallow corners, stop at reversals, long ease-in."""

import math

import numpy as np
import pytest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from box_opening.constants import JOINT_VMAX
from box_opening.runtime.retime import (
    RetimeParams,
    check_like_executor,
    retime_group,
    speed_profile,
    velocity_caps,
)


def _traj(points, dt=0.02):
    msg = JointTrajectory()
    msg.joint_names = ["joint_%d" % i for i in range(1, 8)]
    for k, q in enumerate(points):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in q]
        p.velocities = [0.0] * 7
        p.accelerations = [0.0] * 7
        t = (k + 1) * dt
        p.time_from_start.sec = int(t)
        p.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
        msg.points.append(p)
    return msg


def _line(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _arrays(msg):
    q = np.asarray([p.positions for p in msg.points])
    v = np.asarray([p.velocities for p in msg.points])
    t = np.asarray([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
    return q, v, t


def test_positions_untouched_and_gates_hold():
    src = _traj(_line([0] * 7, [0.5, -0.3, 0.2, 0.0, 0.4, -0.2, 0.1], 60))
    out, info = retime_group([src], [0.75], JOINT_VMAX)
    q, v, t = _arrays(out)
    assert np.allclose(q, np.asarray([p.positions for p in src.points]))
    assert check_like_executor(out, JOINT_VMAX) == []
    assert np.all(np.diff(t) > 0)
    assert np.allclose(v[0], 0) and np.allclose(v[-1], 0)  # rest at both ends
    assert np.abs(v).max(axis=0).max() <= 0.9 * max(JOINT_VMAX) + 1e-9
    assert info["stops"] == 0 and info["junction_speeds"] == []


def test_arrival_is_the_long_half():
    """Asymmetric ease: decel < accel, so the slow-down into rest takes
    longer than the ease out of it — the precision signature."""
    src = _traj(_line([0] * 7, [0.8, 0.6, 0.0, 0.0, 0.5, 0.0, 0.0], 80))
    out, _ = retime_group([src], [1.0], JOINT_VMAX)
    q, v, t = _arrays(out)
    ds = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = ds / np.diff(t)
    k = int(s.argmax())
    assert t[-1] - t[k] > 1.3 * t[k]  # tail longer than head


def test_shallow_corner_flows_and_reversal_stops():
    a = _line([0] * 7, [0.4, 0, 0, 0, 0, 0, 0], 30)
    b = _line([0.4, 0, 0, 0, 0, 0, 0], [0.4, 0.4, 0, 0, 0, 0, 0], 30)  # 90 deg corner
    out, info = retime_group([_traj(a), _traj(b)], [0.75, 0.75], JOINT_VMAX)
    assert 50 <= info["n_points"] <= 59  # junction de-duplicated, neighbours thinned
    assert info["stops"] == 0
    assert 0.15 < info["junction_speeds"][0] < 1.0  # slowed, not stopped
    assert check_like_executor(out, JOINT_VMAX) == []
    back = _line([0.4, 0, 0, 0, 0, 0, 0], [0.0] * 7, 30)  # full reversal
    out2, info2 = retime_group([_traj(a), _traj(back)], [0.75, 0.75], JOINT_VMAX)
    assert info2["stops"] == 1 and info2["junction_speeds"][0] == 0.0
    assert check_like_executor(out2, JOINT_VMAX) == []


def test_each_leg_keeps_its_own_cruise_fraction():
    a = _line([0] * 7, [0.5, 0, 0, 0, 0, 0, 0], 40)
    b = _line([0.5, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0], 40)  # straight on
    out, _ = retime_group([_traj(a), _traj(b)], [0.75, 0.35], JOINT_VMAX)
    q, v, t = _arrays(out)
    s = np.linalg.norm(np.diff(q, axis=0), axis=1) / np.diff(t)
    fast = s[10:30].max()
    slow = s[50:70].max()
    assert slow < 0.6 * fast  # the carry-class leg cruises slower


def test_per_joint_cap_binds_on_the_busiest_joint():
    q = _line([0] * 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 50)  # only joint_7 moves
    cap, _ = velocity_caps(np.asarray(q), JOINT_VMAX, 0.9)
    assert np.allclose(cap[1:-1], 0.9 * JOINT_VMAX[6])


def test_duplicates_and_short_paths_do_not_break():
    q = [[0.1] * 7] * 5 + _line([0.1] * 7, [0.3] * 7, 3) + [[0.3] * 7] * 4
    out, info = retime_group([_traj(q)], [0.75], JOINT_VMAX)
    assert info["n_points"] == 3  # the holds at both ends collapsed
    assert check_like_executor(out, JOINT_VMAX) == []
    single = _traj([[0.2] * 7])
    same, info1 = retime_group([single], [0.75], JOINT_VMAX)
    assert same is single and info1["n_points"] == 1


def test_speed_profile_respects_caps_everywhere():
    q = np.asarray(_line([0] * 7, [0.9, 0.9, 0, 0, 0, 0, 0], 70))
    v, _ = speed_profile(q, np.full(len(q), 1.0), JOINT_VMAX, RetimeParams())
    cap, _ = velocity_caps(q, JOINT_VMAX, 0.9)
    assert np.all(v <= cap + 1e-9) and v[0] == 0.0 and v[-1] == 0.0


def test_time_scale_is_a_uniform_dilation_after_profiling():
    src = _traj(_line([0] * 7, [0.6, 0.2, 0, 0, 0, 0, 0], 50))
    fast, i1 = retime_group([src], [0.75], JOINT_VMAX, RetimeParams())
    slow, i2 = retime_group([src], [0.75], JOINT_VMAX, RetimeParams(time_scale=0.25))
    assert i2["duration_s"] == pytest.approx(4 * (i1["duration_s"] - 0.02) + 0.02, rel=1e-6)
    assert check_like_executor(slow, JOINT_VMAX) == []


def _corner_legs():
    """Two chained legs meeting at a 60 deg corner: a merged group, what
    runner._run_motion re-times."""
    p1 = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p2 = [0.4 + 0.4 * math.cos(math.radians(60)), 0.4 * math.sin(math.radians(60)), 0, 0, 0, 0, 0]
    return [_traj(_line([0] * 7, p1, 30)), _traj(_line(p1, p2, 30))]


def _runs_backwards(msg, nsub=48):
    """Samples at which the JTC's quintic moves AGAINST its own interval's
    chord: the reference stepping back along the path."""
    from jtc_model import quintic_coeffs

    q, v, t = _arrays(msg)
    acc = np.asarray([p.accelerations for p in msg.points])
    back = 0
    for k in range(len(q) - 1):
        h = t[k + 1] - t[k]
        c = quintic_coeffs(q[k], v[k], acc[k], q[k + 1], v[k + 1], acc[k + 1], h)
        s = np.linspace(0.0, h, nsub)[:, None]
        qd = c[1] + 2 * c[2] * s + 3 * c[3] * s**2 + 4 * c[4] * s**3 + 5 * c[5] * s**4
        chord = q[k + 1] - q[k]
        back += int((qd @ chord < -1e-9 * np.linalg.norm(chord)).sum())
    return back


@pytest.mark.parametrize("ts", [0.5, 0.25, 0.2])
def test_slow_mode_scales_the_velocities_and_accelerations_too(ts):
    """--speed-scale stretches the intervals by 1/ts; the velocities
    annotating them must shrink by ts and the accelerations by ts**2
    (driver.dilate's rule). The driver's quintic fits through all three:
    handed full-speed velocities, 2x / 4x / 5x the speed flown at 0.5 /
    0.25 / 0.2, its reference wobbled through every interval and ran
    backwards, and check_like_executor passed it (review 2026-10-27)."""
    from jtc_model import peak_accel

    legs = _corner_legs()
    fast, i1 = retime_group(legs, [0.75, 0.75], JOINT_VMAX, RetimeParams())
    slow, i2 = retime_group(legs, [0.75, 0.75], JOINT_VMAX, RetimeParams(time_scale=ts))
    q, v, t = _arrays(slow)
    # the annotated speed is the speed the samples are flown at
    flown = np.linalg.norm(q[2:] - q[:-2], axis=1) / (t[2:] - t[:-2])
    moving = flown > 1e-3
    ratio = np.linalg.norm(v[1:-1], axis=1)[moving] / flown[moving]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.02)
    # an exact dilation of the full-speed message (first_dt cancels)
    _q1, v1, _t1 = _arrays(fast)
    a1 = np.asarray([p.accelerations for p in fast.points])
    a = np.asarray([p.accelerations for p in slow.points])
    assert np.allclose(v, v1 * ts, rtol=1e-9, atol=1e-12)
    assert np.allclose(a, a1 * ts**2, rtol=1e-9, atol=1e-12)
    # so the spline is the full-speed one, slowed: no sharper kick, no step back
    assert peak_accel(slow)[0] <= peak_accel(fast)[0] * ts**2 * (1 + 1e-6)
    assert _runs_backwards(slow) == 0
    # the report stays on the profile: slow mode miscounts no junction as a stop
    assert i2["junction_speeds"] == i1["junction_speeds"]
    assert i2["stops"] == i1["stops"] == 0
    assert check_like_executor(slow, JOINT_VMAX) == []


def test_full_speed_message_is_the_profile_bit_for_bit():
    """At the default 1.0, which every sheppy task flies, the scaling
    changes nothing: the message is bit for bit the profile's own speeds
    on its own times, what it was before the velocities followed the
    scale."""
    from box_opening.runtime import retime as R

    legs, p = _corner_legs(), RetimeParams()
    out, _ = retime_group(legs, [0.75, 0.75], JOINT_VMAX, RetimeParams(time_scale=1.0))
    q, seg = R.concat_paths(legs, p.min_ds, p.junction_chord, p.junction_window)
    v, ds = speed_profile(q, R.cruise_per_sample(seg, [0.75, 0.75]), JOINT_VMAX, p)
    cap, _ = velocity_caps(q, JOINT_VMAX, p.vmax_margin)
    ref = R.to_message(legs[0].joint_names, q, R.times_from_profile(v, ds, p, cap), v)
    assert len(out.points) == len(ref.points)
    for got, want in zip(_arrays(out), _arrays(ref), strict=False):
        assert np.array_equal(got, want)
    acc = [list(pt.accelerations) for pt in out.points]
    assert np.array_equal(acc, [list(pt.accelerations) for pt in ref.points])


def test_two_point_path_is_a_triangle_not_a_crawl():
    """An interval between two rest samples is timed accelerate-then-
    decelerate, not at the 0.02 rad/s floor (13 s for 0.26 rad)."""
    src = _traj([[0.0] * 7, [0.1] * 7])
    out, info = retime_group([src], [0.75], JOINT_VMAX)
    assert info["duration_s"] < 1.0
    assert check_like_executor(out, JOINT_VMAX) == []


def test_reversal_stops_only_at_its_apex():
    a = _line([0] * 7, [0.4, 0, 0, 0, 0, 0, 0], 30)
    back = _line([0.4, 0, 0, 0, 0, 0, 0], [0.0] * 7, 30)
    out, info = retime_group([_traj(a), _traj(back)], [0.75, 0.75], JOINT_VMAX)
    q, v, t = _arrays(out)
    s = np.linalg.norm(np.diff(q, axis=0), axis=1) / np.diff(t)
    assert info["stops"] == 1
    assert info["duration_s"] < 3.0  # no multi-second crawl around the apex
    assert (s < 0.05).sum() <= 2  # only the intervals touching the apex are slow


def test_float32_near_duplicates_are_merged():
    """cuRobo emits float32 positions: one-ulp near-duplicates (2e-7 rad)
    must not become nanosecond intervals and 1e5 rad/s^2 on the wire."""
    q = _line([0] * 7, [0.5, 0, 0, 0, 0, 0, 0], 40)
    q.insert(20, q[20] + np.array([0, 0, 0, 0, 0, 2.4e-7, 0]))
    out, info = retime_group([_traj(q)], [0.75], JOINT_VMAX)
    assert info["n_points"] == 40
    _, _, t = _arrays(out)
    assert np.diff(t).min() > 1e-4
    acc = np.asarray([p.accelerations for p in out.points])
    assert np.abs(acc).max() < 30.0


def test_kink_between_chained_legs_is_bounded_by_the_robot_accel():
    """Two legs meeting at a kink inside one sample interval: the cap must
    come from the LOCAL geometry and amax, not from a wide window."""
    from box_opening.runtime.retime import corner_caps

    # dense samples near the junction, as a planner's deceleration tail has
    a = _line([0] * 7, [0.3, 0, 0, 0, 0, 0, 0], 20) + _line([0.3, 0, 0, 0, 0, 0, 0], [0.302, 0, 0, 0, 0, 0, 0], 6)[1:]
    b = (
        _line([0.302, 0, 0, 0, 0, 0, 0], [0.302, 0.002, 0, 0, 0, 0, 0], 6)[1:]
        + _line([0.302, 0.002, 0, 0, 0, 0, 0], [0.302, 0.3, 0, 0, 0, 0, 0], 20)[1:]
    )
    q = np.asarray(a + b)
    from box_opening.runtime.retime import RetimeParams

    p = RetimeParams()
    ds = np.linalg.norm(np.diff(q, axis=0), axis=1)
    cap, stop = corner_caps(q, ds, p)
    k = 24  # the kink sample
    L_local = ds[k - 1] + ds[k]
    assert cap[k] <= math.sqrt(p.amax * L_local / (2 * math.sin(math.pi / 4))) + 1e-9
    assert not stop.any()


def test_long_rest_to_rest_interval_respects_the_joint_cap():
    """A 2-point path longer than the triangle can cover under the cap
    becomes a trapezoid at the cap, never a peak above 0.9 x the limit."""
    src = _traj([[0.0] * 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # 1 rad on joint_7
    out, info = retime_group([src], [0.75], JOINT_VMAX)
    _, v, t = _arrays(out)
    # velocities on the wire are zero at both samples; check the implied
    # average speed against the cap instead
    assert 1.0 / (t[-1] - t[0]) < 0.9 * JOINT_VMAX[6]
    assert check_like_executor(out, JOINT_VMAX) == []
    # the crawl is gone but the cap held: duration bounded below by ds/cap
    assert info["duration_s"] > 1.0 / (0.9 * JOINT_VMAX[6])


def test_executed_acceleration_at_a_crowded_junction_stays_under_amax():
    """The real defect: two legs meeting at a kink with the planner's
    crowded end samples. The JTC's quintic must not be asked for more
    than the robot's 25 rad/s^2 anywhere — measured on the spline, not
    on the cap formula (review 2026-09-03)."""
    from jtc_model import peak_accel

    def crowded_tail(a, b, n_far=20, n_near=12, tail=0.02):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        far = _line(a, b - d * tail, n_far)
        near = _line(b - d * tail, b, n_near)[1:]  # 12 samples over the last 2 cm
        return far + near

    def crowded_head(a, b, n_near=12, n_far=20, head=0.02):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        near = _line(a, a + d * head, n_near)
        far = _line(a + d * head, b, n_far)[1:]
        return near + far

    P0 = [0.0] * 7
    P1 = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    P2 = [0.4 + 0.4 * math.cos(math.radians(60)), 0.4 * math.sin(math.radians(60)), 0, 0, 0, 0, 0]
    legs = [_traj(crowded_tail(P0, P1)), _traj(crowded_head(P1, P2))]
    out, info = retime_group(legs, [0.75, 0.75], JOINT_VMAX)
    assert check_like_executor(out, JOINT_VMAX) == []
    assert info["junction_speeds"][0] > 0.15  # it flows
    peak, _ = peak_accel(out)
    print("junction %.2f rad/s, executed peak %.1f rad/s^2" % (info["junction_speeds"][0], peak))
    assert peak <= 25.0, peak


def test_reverse_tail_starts_at_the_live_stop_and_never_goes_deeper():
    """The walk back begins at the sample NEAREST the stop, not at the
    deepest planned one — stepping to that would drive into the contact."""
    from box_opening.runtime.retime import reverse_tail

    down = _line([0] * 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], 50)
    src = _traj(down)
    live = np.asarray(down[30])  # the guard stopped here, short of the plan
    path = reverse_tail(src, progress=0.62, live=live, arc_rad=0.09)
    assert np.allclose(path[0], live)  # the executor's start gate
    depth = [float(q[6]) for q in path]
    assert all(b <= a + 1e-9 for a, b in zip(depth, depth[1:], strict=False))  # only back out
    assert depth[0] - depth[-1] >= 0.09 - 1e-6  # the arc was covered
    # a trip at the very start has nothing to reverse
    assert reverse_tail(src, progress=0.0, live=np.asarray(down[0]), arc_rad=0.09) is None


def test_reverse_tail_covers_the_cancel_overshoot():
    """Cancel latency carries the arm past the last fully elapsed point;
    the retrace must include it."""
    from box_opening.runtime.retime import reverse_tail

    down = _line([0] * 7, [0.4, 0, 0, 0, 0, 0, 0], 20)
    path = reverse_tail(_traj(down), progress=0.5, live=np.asarray(down[10]), arc_rad=0.05)
    assert path is not None and len(path) >= 3


def test_forward_tail_continues_the_stopped_stroke_from_live():
    """The push is cut from the touch's own unexecuted continuation: it
    starts at the live stop and runs down the validated path for the
    asked arc; too little remaining path -> None (the caller plans)."""
    from box_opening.runtime.retime import forward_tail

    down = _line([0] * 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], 50)
    live = np.asarray(down[30])  # the guard stopped at sample 30
    path = forward_tail(_traj(down), live, arc_rad=0.05)
    assert np.allclose(path[0], live)
    depth = [float(q[6]) for q in path]
    assert all(b >= a - 1e-9 for a, b in zip(depth, depth[1:], strict=False))  # only onward
    assert depth[-1] - depth[0] >= 0.05 - 1e-6
    # stopped 1 mm from the end: nothing left to continue along
    assert forward_tail(_traj(down), np.asarray(down[-2]), arc_rad=0.05) is None


def test_the_group_reports_where_each_leg_begins_in_time():
    """A guarded descent merged behind its approach needs to know WHEN the
    descent starts, so its torque guard can arm there instead of watching
    the approach's own dynamics (runner._run_motion)."""
    a = _line([0] * 7, [0.5, 0, 0, 0, 0, 0, 0], 40)
    b = _line([0.5, 0, 0, 0, 0, 0, 0], [0.6, 0, 0, 0, 0, 0, 0], 40)  # short, slow
    out, info = retime_group([_traj(a), _traj(b)], [0.75, 0.15], JOINT_VMAX)
    fracs = info["seg_start_fracs"]
    assert len(fracs) == 2
    assert fracs[0] == 0.0
    assert 0.0 < fracs[1] < 1.0
    # and it is a real TIME fraction of the result
    _q, _v, t = _arrays(out)
    at = float(t[-1]) * fracs[1]
    reached = float(np.interp(at, t, np.linalg.norm(_q - _q[0], axis=1)))
    assert abs(reached - 0.5) < 0.02  # the junction is where leg b starts


# -- the absolute ceiling (RetimeParams.qd_ceiling): what a compliant arm's wrist follows --

WRIST_CEILING = (math.inf,) * 4 + (0.45,) * 3  # constants.COMPLIANT_QD_CEILING
# stamps are whole nanoseconds (stamps.set_stamp): an interval at the ceiling reads
# a few parts in 1e8 over it


def _joint_speeds(msg):
    """Each interval's per-joint speed, as the driver flies the samples."""
    q, _v, t = _arrays(msg)
    return np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None]


def test_the_ceiling_is_absolute_a_slow_line_it_does_not_bind_flies_as_before():
    """The ceiling caps what a leg's cruise fraction leaves and no fraction
    scales it. A 0.3 walk like 16:23's (joint_1 -1.35, joint_3 +1.20,
    joint_6 -0.63 rad) runs its wrist well under 0.45 rad/s, so it flies
    exactly as without the ceiling: the shoulder is not slowed. The same
    0.45 written as a smaller vmax is 0.3 x 0.9 x 0.5 on it, and slows the
    whole line (16:23's way in 7.25 -> 12.4 s, offline 2026-10-28)."""
    walk = _traj(_line([0] * 7, [-1.35, 0.37, 1.2, 0.83, 0.1, -0.63, 0.2], 80))
    free, _ = retime_group([walk], [0.3], JOINT_VMAX, RetimeParams())
    capped, info = retime_group([walk], [0.3], JOINT_VMAX, RetimeParams(qd_ceiling=WRIST_CEILING))
    assert _joint_speeds(capped)[:, 4:].max() <= 0.45 * (1 + 1e-6)
    assert np.abs(_arrays(capped)[1][:, 4:]).max() <= 0.45 * (1 + 1e-6)  # the velocities on the wire too
    for got, want in zip(_arrays(capped), _arrays(free), strict=False):
        assert np.array_equal(got, want)
    _, as_vmax = retime_group([walk], [0.3], JOINT_VMAX[:4] + [0.5] * 3, RetimeParams())
    assert as_vmax["duration_s"] > 1.2 * info["duration_s"]


def test_a_full_speed_leg_holds_its_wrist_to_the_ceiling_and_a_shoulder_move_flies_as_before():
    """At 1.0 the re-timer let joint_6 run 0.9 x 1.222 = 1.10 rad/s; in the
    driver's joint impedance it made 0.54 (15:45 2026-10-28, its command at
    the 9 N·m clamp). Under the ceiling a full-speed leg holds the wrist to
    0.45 and slows its whole path with it, positions untouched; a move of
    joints 1-4 alone, which the ceiling does not name, flies as before."""
    look = _traj(_line([0] * 7, [0.1, 1.2, 0.5, 0.7, 0.3, 1.0, 0.1], 60))
    free, _ = retime_group([look], [1.0], JOINT_VMAX, RetimeParams())
    capped, info = retime_group([look], [1.0], JOINT_VMAX, RetimeParams(qd_ceiling=WRIST_CEILING))
    assert _joint_speeds(free)[:, 5].max() > 0.8  # what it asked without
    assert _joint_speeds(capped)[:, 4:].max() <= 0.45 * (1 + 1e-6)
    assert np.abs(_arrays(capped)[1][:, 4:]).max() <= 0.45 * (1 + 1e-6)
    assert np.array_equal(_arrays(capped)[0], _arrays(free)[0])  # positions never touched
    assert info["duration_s"] > 1.0 / 0.45  # 1 rad of joint_6 under 0.45 rad/s
    assert check_like_executor(capped, JOINT_VMAX) == []
    arm = _traj(_line([0] * 7, [0.3, 1.2, 0.5, 0.7, 0.0, 0.0, 0.0], 60))
    a_free, _ = retime_group([arm], [1.0], JOINT_VMAX, RetimeParams())
    a_capped, _ = retime_group([arm], [1.0], JOINT_VMAX, RetimeParams(qd_ceiling=WRIST_CEILING))
    for got, want in zip(_arrays(a_capped), _arrays(a_free), strict=False):
        assert np.array_equal(got, want)


def test_a_rest_to_rest_interval_averages_under_the_ceiling_too():
    """An interval between two rest samples is timed apart from the profile
    (times_from_profile's triangle, a trapezoid where the cap binds): its
    cap is the ceiling's too, so 1 rad of joint_6 there averages under
    0.45 rad/s, not under the 1.10 the joint limit alone allows. Averages:
    the driver's quintic between two rest samples peaks at 15/8 of that
    (0.77 rad/s here; 1.58 under the joint cap alone, as before this). The
    mission's lines and turns are sampled every 2-5 mm and the planner's
    plans densely, and there the quintic follows each interval's speed
    within a few per cent (1 % on a straight line)."""
    src = _traj([[0.0] * 7, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    out, info = retime_group([src], [1.0], JOINT_VMAX, RetimeParams(qd_ceiling=WRIST_CEILING))
    _, _, t = _arrays(out)
    assert 1.0 / (t[-1] - t[0]) < 0.45
    assert info["duration_s"] > 1.0 / 0.45
    assert check_like_executor(out, JOINT_VMAX) == []


# -- the junction ceiling (RetimeParams.junction_cruise): a compliant run's faster lines meet as at 0.3 --


def _near_reversal(turn_deg):
    """16:23's rise into its walk back in miniature: a stroke sampled every
    0.023 rad, then a walk every 0.15 rad, turning `turn_deg` in joint space
    where they meet (joints 1, 4 and 6 reversing at 132 deg)."""
    d1 = np.array([1.0, 0, 0, 0.6, 0, 0.5, 0])
    d1 /= np.linalg.norm(d1)
    e = np.array([0, 0.8, 0.3, 0, 0, 0, 0.0])
    e -= e.dot(d1) * d1
    e /= np.linalg.norm(e)
    th = math.radians(turn_deg)
    d2 = math.cos(th) * d1 + math.sin(th) * e
    p0 = np.zeros(7)
    p1 = p0 + d1 * 0.023 * 11
    p2 = p1 + d2 * 0.15 * 5
    return [_traj(_line(p0, p1, 12)), _traj(_line(p1, p2, 6))]


def test_the_junction_ceiling_holds_the_junction_and_the_sample_either_side_and_no_other():
    """Without one, the sample a junction shares cruises at the lower of its
    two legs' fractions and every other at its own leg's. Under
    junction_cruise neither that sample nor the one on either side of it —
    the ends of the two intervals the driver's quintic turns the junction
    in — cruises faster, whichever leg each is of; one slower keeps its
    own, and every other sample its leg's. Nothing sets one but a compliant
    run (Runner.retime_params)."""
    from box_opening.runtime.retime import cruise_per_sample

    seg = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    speeds = [0.55, 0.55, 0.175]  # a walk, a turn, the stroke's approach
    assert list(cruise_per_sample(seg, speeds)) == [0.55] * 7 + [0.175] * 4
    held = cruise_per_sample(seg, speeds, 0.3)
    assert list(held) == [0.55, 0.55, 0.3, 0.3, 0.3, 0.55, 0.3, 0.175, 0.175, 0.175, 0.175]
    assert RetimeParams().junction_cruise is None


@pytest.mark.parametrize("turn_deg", [132.0, 60.0, 31.5])
def test_lines_that_cruise_faster_under_the_junction_ceiling_meet_as_the_slower_lines_did(turn_deg):
    """Offline 2026-10-28, flatbot's URDF: at COMPLIANT_FREE_LINE_SPEED a
    junction of two free-air lines took their 0.55 and flowed through up to
    1.83x as fast as at FREE_LINE_SPEED, and the driver's quintic, which
    turns it inside the shorter adjacent interval, was asked up to 3.4x the
    acceleration there (16:23's rise into its walk back 47 -> 59 rad/s^2).
    Under a compliant run's COMPLIANT_JUNCTION_SPEED the junction meets at
    the speed the slower lines gave it and the quintic is asked what it was,
    while the lines between cruise faster: shorter, the same positions, no
    stop gained or lost."""
    from dataclasses import replace

    from jtc_model import peak_accel

    from box_opening.constants import (
        COMPLIANT_ACCEL,
        COMPLIANT_DECEL,
        COMPLIANT_FREE_LINE_SPEED,
        COMPLIANT_JUNCTION_SPEED,
        FREE_LINE_SPEED,
    )

    legs = _near_reversal(turn_deg)
    p = RetimeParams(accel=COMPLIANT_ACCEL, decel=COMPLIANT_DECEL, qd_ceiling=WRIST_CEILING)
    slow, was = retime_group(legs, [FREE_LINE_SPEED] * 2, JOINT_VMAX, p)
    loose, free = retime_group(legs, [COMPLIANT_FREE_LINE_SPEED] * 2, JOINT_VMAX, p)
    held, info = retime_group(
        legs, [COMPLIANT_FREE_LINE_SPEED] * 2, JOINT_VMAX, replace(p, junction_cruise=COMPLIANT_JUNCTION_SPEED)
    )
    assert free["junction_speeds"][0] > 1.3 * was["junction_speeds"][0]  # what it did without
    assert peak_accel(loose)[0] > 1.5 * peak_accel(slow)[0]
    assert info["junction_speeds"] == pytest.approx(was["junction_speeds"])
    assert peak_accel(held)[0] <= 1.02 * peak_accel(slow)[0]
    assert info["duration_s"] < 0.85 * was["duration_s"] and info["stops"] == was["stops"] == 0
    assert np.array_equal(_arrays(held)[0], _arrays(loose)[0])  # positions never touched
    assert check_like_executor(held, JOINT_VMAX) == []
