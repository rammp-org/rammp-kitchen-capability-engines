"""MIB differential drive behavior shared by the ROS fake and mobile fridge replay."""

import math

import pytest
from base_sim import DifferentialDrive


def test_low_profile_ignores_lateral_and_ramps_after_latency():
    drive = DifferentialDrive(traction=(1, 1), encoder_scale=(1, 1))
    drive.command(0.0, 2.0, 0.0)
    assert drive.odometry(0.10)[0] == 0.0
    drive.set_mode(0.10, True)
    drive.command(0.10, 2.0, 0.0, 4.0)
    assert drive.odometry(0.13)[3] == 0.0
    sample = drive.odometry(0.24)
    assert sample[3] == pytest.approx(0.05)
    assert sample[1] == sample[4] == sample[5] == 0.0
    for t in (0.25, 0.40, 0.55, 0.70, 0.85):
        drive.command(t, 2.0, 0.0, 4.0)
    assert drive.odometry(1.0)[3] == pytest.approx(0.35)
    assert len(drive.twists) == 7
    assert len(drive.samples) == 4


def test_watchdog_brakes_and_mode_exit_discards_queued_command():
    drive = DifferentialDrive(latency_s=0.04, traction=(1, 1), encoder_scale=(1, 1))
    drive.set_mode(0.0, True)
    drive.command(0.0, 0.10, 0.0)
    assert drive.odometry(0.32)[3] == pytest.approx(0.10)
    assert drive.odometry(0.44)[3] == pytest.approx(0.05)
    assert drive.odometry(0.60)[3] == 0.0
    before = drive.truth
    drive.command(0.61, 0.10, 0.0)
    drive.set_mode(0.62, False)
    drive.set_mode(0.63, True)
    drive.advance(1.0)
    assert drive.truth == before


def test_in_place_yaw_rotates_the_odom_datum_about_the_axle():
    drive = DifferentialDrive(
        axle_pose=(1, 2, 0),
        odom_to_axle=(0.5, 0),
        latency_s=0.0,
        traction=(1, 1),
        encoder_scale=(1, 1),
    )
    drive.set_mode(0.0, True)
    for t in range(10):
        drive.command(t / 10, 0.0, 0.20)
    x, y, yaw, vx, vy, wz = drive.odometry(1.0)
    assert drive.truth[:2] == (1, 2)
    assert yaw > 0
    assert (x, y) == pytest.approx((1 - 0.5 * math.cos(yaw), 2 - 0.5 * math.sin(yaw)))
    assert (vx, vy, wz) == pytest.approx((0, -0.10, 0.20))


def test_encoder_odom_and_ground_truth_can_diverge():
    drive = DifferentialDrive(traction=(0.85, 0.95), encoder_scale=(1.0, 1.0))
    drive.set_mode(0.0, True)
    for t in range(10):
        drive.command(t / 10, 0.15, 0.0)
    x, y, yaw, *_ = drive.odometry(1.0)
    assert y == yaw == 0.0
    assert x > drive.truth[0]
    assert drive.truth[1] > 0.0
