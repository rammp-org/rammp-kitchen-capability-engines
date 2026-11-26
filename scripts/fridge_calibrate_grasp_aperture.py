#!/usr/bin/env python3
"""Record a settled, operator-labelled gripper close without commanding motion."""

import argparse
import csv
import math
import statistics
import time
from pathlib import Path


def capture(node, label, path, seconds):
    import rclpy
    from rammp_arm_interfaces.msg import GripperState
    from rclpy.qos import qos_profile_sensor_data

    rows = {}

    def sample(msg):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if (
            stamp != (0, 0)
            and msg.present
            and math.isfinite(msg.position)
            and 0 <= msg.position <= 1
        ):
            rows[stamp] = (label, *stamp, msg.position, 85 * (1 - msg.position), msg.current)

    sub = node.create_subscription(GripperState, "/gripper_state", sample, qos_profile_sensor_data)
    try:
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            rclpy.spin_once(node, timeout_sec=min(remaining, 0.1))
    finally:
        node.destroy_subscription(sub)
    if len(rows) < 3:
        raise ValueError("Need at least three distinct valid gripper samples")
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("label", "sec", "nanosec", "closure", "gap_mm", "motor_current"))
        writer.writerows(rows.values())
    gaps = [row[4] for row in rows.values()]
    low, median, high = min(gaps), statistics.median(gaps), max(gaps)
    print(f"{label}: n={len(gaps)} gap min/median/max {low:.2f}/{median:.2f}/{high:.2f} mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=("empty", "bar"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=2)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")

    import rclpy

    rclpy.init()
    node = rclpy.create_node("fridge_grasp_aperture")
    try:
        capture(node, args.label, args.output, args.seconds)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
