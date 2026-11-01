"""Stage 3, planned moves (cuRobo in the driver): go_to_ee_pose and its holds, speed and
control modes, and go_to_ee_poses / go_to_joint_configs, several targets in one motion.

The multi-target tests skip when the driver does not serve /go_to_ee_poses (a driver image
older than kinova-gen3-ros2#65)."""

import time

import numpy as np
import pytest
import rammp_arm
from conftest import angle_deg, joint_err
from scipy.spatial.transform import Rotation as R

from capability_engine import ArmError
from capability_engine.arm import HOLD_FIXED, HOLD_LEVEL, IMPEDANCE, JointTarget, PoseTarget

pytestmark = pytest.mark.motion

END_M = rammp_arm.GO_TO_TRACK_M
END_DEG = 2.0
HOLD_DEG = 3.0  # how far a held orientation (or tilt) may wander along the path
BLEND_MARGIN_M = 0.01  # a blended target is passed within its radius plus this
STOPPED_M_S = 0.01  # tool speed at an unblended target
OFFSET = np.array([0.0, 0.05, 0.05])
FAR = [2.0, 0.0, 0.5]
SPIN_DEG = 15.0  # HOLD_LEVEL: a turn about arm_mount z keeps the tilt
BLEND_M = 0.03
SWING = np.array([0.1, 0, 0, 0, 0, 0, 0])  # J1: the tool swings ~6 cm


def needs(cell, action):
    if not cell.observer.serves(action):
        pytest.skip(f"the driver does not serve {action} (driver image older than #65)")


def tilt_deg(quat):
    """The angle between the tool's z and arm_mount's z."""
    z = R.from_quat(quat).apply([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))


def ends_at(cell, pos, quat):
    cell.observer.wait_still()
    p, q = cell.observer.tool()
    assert np.linalg.norm(p - pos) < END_M, p - pos
    assert angle_deg(q, quat) < END_DEG


def test_go_to_ee_pose(cell, home):
    p0, q0 = cell.observer.tool()
    feedback = []
    ok = cell.act(lambda arm, ctx: arm.go_to_ee_pose(p0 + OFFSET, q0, feedback=feedback.append))
    assert ok
    ends_at(cell, p0 + OFFSET, q0)
    phases = {f.phase for f in feedback}
    assert "executing" in phases, phases


def test_go_to_ee_pose_hold_fixed(cell, home):
    p0, q0 = cell.observer.tool()
    t0 = time.monotonic()
    target = p0 + OFFSET * [1, -1, 1]
    assert cell.act(lambda arm, ctx: arm.go_to_ee_pose(target, q0, orientation_hold=HOLD_FIXED))
    ends_at(cell, target, q0)
    _, _, quat = cell.observer.tool_since(t0)
    assert max(angle_deg(q, q0) for q in quat) < HOLD_DEG


def test_go_to_ee_pose_hold_level(cell, home):
    p0, q0 = cell.observer.tool()
    q_target = (R.from_euler("z", SPIN_DEG, degrees=True) * R.from_quat(q0)).as_quat()
    t0 = time.monotonic()
    ok = cell.act(lambda arm, ctx: arm.go_to_ee_pose(p0, q_target, orientation_hold=HOLD_LEVEL))
    assert ok
    ends_at(cell, p0, q_target)
    _, _, quat = cell.observer.tool_since(t0)
    tilts = [tilt_deg(q) for q in quat]
    assert max(tilts) - min(tilts) < HOLD_DEG


def test_an_unreachable_pose_returns_false_without_moving(cell, home):
    t0 = time.monotonic()
    ok, last_error = cell.act(
        lambda arm, ctx: (arm.go_to_ee_pose(FAR, [0, 0, 0, 1]), arm.last_error)
    )
    assert not ok and last_error
    assert cell.observer.motion_span(t0) is None


def test_go_to_speed_scale(cell, home):
    p0, q0 = cell.observer.tool()
    durations = {}
    for scale, target in ((0.3, p0 + OFFSET), (0.6, p0)):
        t0 = time.monotonic()
        assert cell.act(lambda arm, ctx, s=scale, p=target: arm.go_to_ee_pose(p, q0, speed_scale=s))
        cell.observer.wait_still()
        first, last = cell.observer.motion_span(t0)
        durations[scale] = last - first
    assert durations[0.6] < 0.75 * durations[0.3], durations


def test_go_to_in_impedance(cell, home):
    p0, q0 = cell.observer.tool()
    for gains in ("medium", None):  # a profile, then the session's default
        assert cell.act(
            lambda arm, ctx, g=gains: arm.go_to_ee_pose(
                p0 + OFFSET, q0, control_mode=IMPEDANCE, gains=g
            )
        ), gains
        ends_at(cell, p0 + OFFSET, q0)
        assert cell.act(lambda arm, ctx: arm.go_to_joint_config(home))


def test_go_to_ee_poses_blends(cell, home, record_property):
    """Through two corners: with radius 0 the tool stops on each; with a radius it passes
    within it (the planner may still stop: the README says so). Both end on the last target."""
    needs(cell, "/go_to_ee_poses")
    p0, q0 = cell.observer.tool()
    corners = [p0 + [0.0, 0.05, 0.0], p0 + [0.0, 0.05, 0.05]]
    durations = {}
    for blend in (0.0, BLEND_M):
        assert cell.act(lambda arm, ctx: arm.go_to_joint_config(home))
        cell.observer.wait_still()
        targets = [PoseTarget(c, q0, blend_radius=blend) for c in corners]
        targets.append(PoseTarget(p0, q0))
        t0 = time.monotonic()
        assert cell.act(lambda arm, ctx, ts=targets: arm.go_to_ee_poses(ts))
        ends_at(cell, p0, q0)
        t, pos, _ = cell.observer.tool_since(t0)
        ts, speed = cell.observer.tool_speed(t0)
        for c in corners:
            d = np.linalg.norm(pos - c, axis=1)
            assert d.min() < blend + BLEND_MARGIN_M, f"passed {d.min():.3f} m from a corner"
            if blend == 0.0:  # radius 0: it stops on the corner
                near = np.abs(ts - t[int(d.argmin())]) < 0.3
                assert speed[near].min() < STOPPED_M_S, speed[near].min()
        first, last = cell.observer.motion_span(t0)
        durations[blend] = last - first
    record_property("durations_s", durations)  # blended should be shorter; not promised


def test_go_to_joint_configs(cell, home):
    needs(cell, "/go_to_joint_configs")
    targets = [JointTarget(home + SWING, blend_radius=BLEND_M), JointTarget(home - SWING)]
    assert cell.act(lambda arm, ctx: arm.go_to_joint_configs(targets))
    cell.observer.wait_still()
    err = np.degrees(joint_err(cell.observer.q(), home - SWING))
    assert err.max() < rammp_arm.GO_TO_TRACK_DEG, err


def test_go_to_ee_poses_refuses_a_zero_quaternion(cell, home):
    t0 = time.monotonic()
    p0, _ = cell.observer.tool()
    with pytest.raises(ArmError):
        cell.act(lambda arm, ctx: arm.go_to_ee_poses([PoseTarget(p0, (0, 0, 0, 0))]))
    assert cell.observer.motion_span(t0) is None
