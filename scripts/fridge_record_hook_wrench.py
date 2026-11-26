#!/usr/bin/env python3
"""Record hook telemetry or compare production hinge-load scores from labelled pulls."""

import argparse
import csv
import math
import re
import time
from pathlib import Path

JOINTS = tuple(f"arm_joint_{index}" for index in range(1, 8))
SCORE = re.compile(r"hook late pull: (\S+) N m resisting hinge moment from (\d+) samples")


def record(node, path, seconds=None):
    import rclpy
    from rammp_arm_interfaces.msg import EeState
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    counts = {"joint_residual": 0, "flange_wrench": 0}
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "source",
                "sec",
                "nanosec",
                "frame",
                *[f"j{i}_nm" for i in range(1, 8)],
                "px_m",
                "py_m",
                "pz_m",
                "fx_n",
                "fy_n",
                "fz_n",
                "tx_nm",
                "ty_nm",
                "tz_nm",
            )
        )

        def write(source, msg, values):
            stamp = msg.header.stamp
            writer.writerow((source, stamp.sec, stamp.nanosec, msg.header.frame_id, *values))
            counts[source] += 1

        def residual(msg):
            if len(msg.name) != len(msg.effort):
                return
            effort = dict(zip(msg.name, msg.effort, strict=False))
            if all(joint in effort and math.isfinite(effort[joint]) for joint in JOINTS):
                write("joint_residual", msg, [effort[joint] for joint in JOINTS] + [""] * 9)

        def wrench(msg):
            vectors = (msg.pose.position, msg.wrench.force, msg.wrench.torque)
            values = [getattr(vector, axis) for vector in vectors for axis in ("x", "y", "z")]
            if msg.header.frame_id == "arm_mount" and all(map(math.isfinite, values)):
                write("flange_wrench", msg, [""] * 7 + values)

        subs = [
            node.create_subscription(
                JointState, "/contact_monitor/residual", residual, qos_profile_sensor_data
            ),
            node.create_subscription(EeState, "/ee_state", wrench, qos_profile_sensor_data),
        ]
        print("Recording. Ctrl-C after the supervised pulls.", flush=True)
        try:
            if seconds is None:
                rclpy.spin(node)
            else:
                deadline = time.monotonic() + seconds
                while (remaining := deadline - time.monotonic()) > 0:
                    rclpy.spin_once(node, timeout_sec=min(remaining, 0.1))
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            for sub in subs:
                node.destroy_subscription(sub)
    print(counts)
    if not all(counts.values()):
        raise ValueError(
            "Missing joint residual or flange wrench samples. Check driver diagnostics and DDS."
        )


def scores(paths):
    values = []
    for path in paths:
        matches = SCORE.findall(path.read_text())
        if len(matches) != 1:
            raise ValueError(f"{path}: expected one production hook score")
        value, count = matches[0]
        if int(count) < 3 or not math.isfinite(float(value)):
            raise ValueError(
                f"{path}: need at least three finite samples in the production angle window"
            )
        values.append(float(value))
    return values


def propose(retained, unloaded):
    high_unloaded = max(0.0, max(scores(unloaded)))
    low_retained = min(scores(retained))
    # Console scores are rounded to 0.001 N m. Require a gap wider than rounding uncertainty.
    if low_retained - high_unloaded <= 0.002:
        raise ValueError("No threshold separates the labelled trials with room for log rounding")
    return high_unloaded, low_retained, (high_unloaded + low_retained) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recorder = commands.add_parser("record")
    recorder.add_argument("output", type=Path)
    recorder.add_argument("--seconds", type=float)
    summary = commands.add_parser("propose")
    summary.add_argument("--retained", type=Path, nargs="+", required=True)
    summary.add_argument("--unloaded", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    try:
        if args.command == "propose":
            unloaded, retained, threshold = propose(args.retained, args.unloaded)
            print(f"Max unloaded {unloaded:.3f}, min retained {retained:.3f} N m")
            print(f"Candidate HOOK_MIN_HINGE_LOAD_NM = {threshold:.4f}")
            return

        if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
            parser.error("--seconds must be finite and positive")

        import rclpy

        rclpy.init()
        node = rclpy.create_node("fridge_hook_wrench")
        try:
            record(node, args.output, args.seconds)
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
