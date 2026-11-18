"""Offline checks for the non-contact stop-at-joins hardware test path."""

from types import SimpleNamespace

import numpy as np
import pytest

from microwave_place import place_motion, smoothness_test


class Arm:
    def __init__(self, fail=0):
        self.calls = []
        self.fail = fail

    def plan_to_joints(self, goal, start):
        self.calls.append((np.asarray(goal), np.asarray(start)))
        if len(self.calls) == self.fail:
            return SimpleNamespace(success=False, message="blocked", q=np.zeros((0, 7)))
        middle = np.asarray(start) + 0.5 * place_motion.joint_delta(start, goal)
        return SimpleNamespace(success=True, message="", q=np.asarray([start, middle, goal]))


def goals():
    return [[5, 0, 0, 0, 0, 0, 0], [5, 5, 0, 0, 0, 0, 0], [5, 5, 5, 0, 0, 0, 0]]


def test_plans_three_sequential_curobo_legs():
    arm = Arm()
    legs, speed = smoothness_test.plan_test(arm, np.zeros(7), goals(), 0.1)

    assert len(legs) == 3
    assert speed == 0.1
    assert arm.calls[1][1] == pytest.approx(arm.calls[0][0])
    assert arm.calls[2][1] == pytest.approx(arm.calls[1][0])


def test_retimming_stops_at_exactly_both_joins():
    legs, speed = smoothness_test.plan_test(Arm(), np.zeros(7), goals(), 0.1)
    trajectory = smoothness_test.test_trajectory(np.zeros(7), legs, speed)

    assert len(trajectory["stop_at"]) == 2
    for join, boundary in zip(trajectory["join_times"], (legs[0][-1], legs[1][-1]), strict=True):
        index = int(np.argmin(np.abs(trajectory["times"] - join)))
        assert trajectory["positions"][index] == pytest.approx(boundary)
        assert trajectory["velocities"][index] == pytest.approx(np.zeros(7), abs=1e-12)


@pytest.mark.parametrize(
    ("waypoints", "speed", "message"),
    [
        (goals()[:2], 0.1, "exactly three"),
        ([[21, 0, 0, 0, 0, 0, 0], *goals()[1:]], 0.1, "maximum 20 deg"),
        (goals(), 0.21, "speed_scale"),
    ],
)
def test_rejects_unbounded_test_inputs(waypoints, speed, message):
    with pytest.raises(place_motion.Refused, match=message):
        smoothness_test.plan_test(Arm(), np.zeros(7), waypoints, speed)


def test_refuses_any_curobo_leg_failure():
    with pytest.raises(place_motion.Refused, match="leg 2: cuRobo refused: blocked"):
        smoothness_test.plan_test(Arm(fail=2), np.zeros(7), goals(), 0.1)


def test_csv_records_command_measurement_velocity_error_and_faults(tmp_path):
    legs, speed = smoothness_test.plan_test(Arm(), np.zeros(7), goals(), 0.1)
    trajectory = smoothness_test.test_trajectory(np.zeros(7), legs, speed)
    samples = [
        (float(t), np.asarray(q) + 0.001, np.asarray(qd))
        for t, q, qd in zip(trajectory["times"], trajectory["positions"], trajectory["velocities"], strict=True)
    ]
    path = tmp_path / "evidence.csv"

    evidence = smoothness_test.write_csv(path, samples, ["controller: fault (faults=1)"], trajectory)

    assert evidence["samples"] == len(samples)
    assert evidence["max_tracking_error_deg"] == pytest.approx(np.degrees(0.001))
    assert len(evidence["measured_join_velocity_deg_s"]) == 2
    text = path.read_text()
    assert "command_velocity_rad_s" in text
    assert "measured_velocity_rad_s" in text
    assert "tracking_error_rad" in text
    assert "controller: fault" in text
