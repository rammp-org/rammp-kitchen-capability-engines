"""Exercise the read-only recorders against the fake arm's live ROS topics."""

import csv
import runpy
from pathlib import Path

import pytest
import rclpy
from sensor_msgs.msg import JointState

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
GRASP = runpy.run_path(str(SCRIPTS / "fridge_calibrate_grasp_aperture.py"))
HOOK = runpy.run_path(str(SCRIPTS / "fridge_record_hook_wrench.py"))


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_grasp_and_hook_recorders(arm_env, tmp_path):
    _, _, fake = arm_env
    collector = rclpy.create_node("test_calibration_collector")
    # FakeDriver already supplies gripper state and flange wrench. Supply the optional
    # contact observer topic beside them, with reverse joint order as a regression case.
    pub = fake.create_publisher(JointState, "/contact_monitor/residual", 10)

    def residual():
        msg = JointState(
            name=list(reversed(HOOK["JOINTS"])), effort=[float(i) for i in range(7, 0, -1)]
        )
        msg.header.stamp = fake.get_clock().now().to_msg()
        pub.publish(msg)

    timer = fake.create_timer(0.02, residual)
    try:
        for label, closure in (("empty", 1.0), ("bar", 0.8)):
            fake.set_gripper(closure)
            path = tmp_path / f"{label}.csv"
            GRASP["capture"](collector, label, path, 1.0)
            assert all(
                float(row["gap_mm"]) == pytest.approx(85 * (1 - closure)) for row in rows(path)
            )

        path = tmp_path / "hook.csv"
        HOOK["record"](collector, path, 1.0)
        records = rows(path)
        assert any(
            row["source"] == "flange_wrench" and row["frame"] == "arm_mount" for row in records
        )
        residuals = [row for row in records if row["source"] == "joint_residual"]
        assert residuals
        assert [float(residuals[0][f"j{i}_nm"]) for i in range(1, 8)] == list(range(1, 8))
    finally:
        fake.destroy_timer(timer)
        collector.destroy_node()


def test_propose_uses_hinge_moment_and_requires_separation(tmp_path):
    retained, unloaded = tmp_path / "retained.log", tmp_path / "unloaded.log"
    retained.write_text("  hook late pull: 0.120 N m resisting hinge moment from 8 samples\n")
    unloaded.write_text("  hook late pull: 0.020 N m resisting hinge moment from 4 samples\n")
    assert HOOK["propose"]([retained], [unloaded]) == pytest.approx((0.020, 0.120, 0.070))
    unloaded.write_text("  hook late pull: 0.119 N m resisting hinge moment from 4 samples\n")
    with pytest.raises(ValueError, match="No threshold separates"):
        HOOK["propose"]([retained], [unloaded])
    unloaded.write_text("  hook late pull: 0.000 N m resisting hinge moment from 1 samples\n")
    with pytest.raises(ValueError, match="three finite samples"):
        HOOK["propose"]([retained], [unloaded])
