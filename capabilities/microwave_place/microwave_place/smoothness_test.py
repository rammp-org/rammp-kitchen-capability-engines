"""Non-contact, attended test of the approach trajectory's two stop-at-joins transitions."""

import csv
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rammp_arm

from . import place_motion as pm

WAYPOINTS = 3
MAX_LEG_DEG = 20.0
MAX_SPEED_SCALE = 0.20
START_TOL_DEG = 0.3
END_TOL_DEG = 1.0


def plan_test(arm, q0, waypoints_deg, speed_scale):
    """cuRobo-plan three short joint legs in the current collision world; nothing moves."""
    goals = np.radians(np.asarray(waypoints_deg, dtype=float))
    if goals.shape != (WAYPOINTS, len(rammp_arm.JOINTS)) or not np.isfinite(goals).all():
        raise pm.Refused("smoothness test needs exactly three finite J1-J7 waypoint arrays in degrees")
    speed_scale = float(speed_scale)
    if not np.isfinite(speed_scale) or not 0.01 <= speed_scale <= MAX_SPEED_SCALE:
        raise pm.Refused(f"smoothness test speed_scale must be in [0.01, {MAX_SPEED_SCALE:.2f}]")
    legs, start = [], np.asarray(q0, dtype=float)
    for index, goal in enumerate(goals, 1):
        direct = float(np.max(np.abs(pm.wrap_deg(goal - start))))
        if direct > MAX_LEG_DEG:
            raise pm.Refused(
                f"smoothness test leg {index} changes a joint {direct:.1f} deg " f"(maximum {MAX_LEG_DEG:.0f} deg)"
            )
        plan = arm.plan_to_joints(goal, start)
        if not plan.success:
            raise pm.Refused(f"smoothness test leg {index}: cuRobo refused: {plan.message}")
        q = np.asarray(plan.q, dtype=float)
        if q.shape[0] < 2 or q.shape[1:] != (len(rammp_arm.JOINTS),):
            raise pm.Refused(f"smoothness test leg {index}: cuRobo returned no executable trajectory")
        start_error = float(np.max(np.abs(pm.wrap_deg(q[0] - start))))
        end_error = float(np.max(np.abs(pm.wrap_deg(q[-1] - goal))))
        if start_error > START_TOL_DEG or end_error > END_TOL_DEG:
            raise pm.Refused(
                f"smoothness test leg {index}: cuRobo endpoint mismatch " f"{start_error:.2f}/{end_error:.2f} deg"
            )
        travel = pm.joint_travel_deg(q)
        if float(np.max(travel)) > 2 * MAX_LEG_DEG:
            raise pm.Refused(
                f"smoothness test leg {index}: cuRobo detour has joint travel " f"{np.round(travel, 1).tolist()} deg"
            )
        legs.append(q)
        start = q[-1]
    return legs, speed_scale


def test_trajectory(q0, legs, speed_scale):
    """Retiming used by move_joints, with stops at exactly the two leg boundaries."""
    first, second, third = legs
    knots = pm.approach_knots(first, second, third)
    stops = pm.approach_stop_indices(first, second, third)
    samples = rammp_arm.joint_trajectory(q0, knots, speed_scale=speed_scale, stop_at=stops)
    if samples is None or len(stops) != 2:
        raise pm.Refused("smoothness test did not produce two interior segment joins")
    times, positions, velocities = samples
    join_times = []
    boundaries = (first[-1], second[-1])
    for boundary in boundaries:
        index = int(np.argmin(np.linalg.norm(positions - boundary, axis=1)))
        if np.max(np.abs(pm.wrap_deg(positions[index] - boundary))) > 1e-6:
            raise pm.Refused("smoothness test retiming did not preserve a segment boundary")
        if np.max(np.abs(velocities[index])) > 1e-9:
            raise pm.Refused("smoothness test retiming did not stop at a segment boundary")
        join_times.append(float(times[index]))
    return {
        "knots": knots,
        "stop_at": stops,
        "times": times,
        "positions": positions,
        "velocities": velocities,
        "join_times": join_times,
        "duration_s": float(times[-1]),
    }


def save_plan(state_dir, q0, waypoints_deg, legs, trajectory, world_message):
    directory = Path(state_dir) / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = directory / f"smoothness_{stamp}.json"
    record = {
        "kind": "non_contact_stop_at_joins",
        "start_q": np.asarray(q0).tolist(),
        "waypoints_deg": np.asarray(waypoints_deg).tolist(),
        "legs": [leg.tolist() for leg in legs],
        "stop_at": list(trajectory["stop_at"]),
        "join_times_s": trajectory["join_times"],
        "duration_s": trajectory["duration_s"],
        "world": world_message,
    }
    path.write_text(json.dumps(record, indent=2))
    return path


def _faults(message):
    faults = []
    for status in message.status:
        level = status.level[0] if isinstance(status.level, bytes) else int(status.level)
        values = {value.key: value.value for value in status.values}
        count = values.get("faults", "0").strip()
        if level >= 2 or count not in ("", "0", "0.0"):
            faults.append(f"{status.name}: {status.message} (faults={count})")
    return faults


class Recorder:
    """Action-scoped joint/diagnostic recorder; callbacks never command the arm."""

    def __init__(self, node):
        from diagnostic_msgs.msg import DiagnosticArray
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        self._lock = threading.Lock()
        self._started = None
        self.samples = []
        self.faults = []
        node.create_action_subscription(JointState, "/joint_states", self._joint, qos_profile_sensor_data)
        node.create_action_subscription(DiagnosticArray, "/diagnostics", self._diagnostic, qos_profile_sensor_data)

    def start(self):
        self._started = time.monotonic()

    def _joint(self, message):
        if self._started is None or not set(rammp_arm.JOINTS) <= set(message.name):
            return
        index = [list(message.name).index(name) for name in rammp_arm.JOINTS]
        if len(message.position) != len(message.name) or len(message.velocity) != len(message.name):
            return
        sample = (
            time.monotonic() - self._started,
            np.asarray([message.position[i] for i in index], dtype=float),
            np.asarray([message.velocity[i] for i in index], dtype=float),
        )
        with self._lock:
            self.samples.append(sample)

    def _diagnostic(self, message):
        found = _faults(message)
        if found:
            with self._lock:
                self.faults.extend(item for item in found if item not in self.faults)

    def monitor(self):
        with self._lock:
            return bool(self.faults)

    def finish(self, path, trajectory):
        with self._lock:
            samples, faults = list(self.samples), list(self.faults)
        return write_csv(path, samples, faults, trajectory)


def write_csv(path, samples, faults, trajectory):
    """Write commanded/measured states and return join velocity/tracking evidence."""
    if not samples:
        raise pm.Refused("smoothness test recorded no joint states")
    command_t = np.asarray(trajectory["times"], dtype=float)
    command_q = np.asarray(trajectory["positions"], dtype=float)
    command_qd = np.asarray(trajectory["velocities"], dtype=float)
    measured_t = np.asarray([sample[0] for sample in samples], dtype=float)
    measured_q = np.asarray([sample[1] for sample in samples], dtype=float)
    measured_qd = np.asarray([sample[2] for sample in samples], dtype=float)
    command_move = np.flatnonzero(np.linalg.norm(command_qd, axis=1) > 1e-3)
    measured_move = np.flatnonzero(np.linalg.norm(measured_qd, axis=1) > 1e-3)
    alignment = 0.0
    if len(command_move) and len(measured_move):
        alignment = measured_t[measured_move[0]] - command_t[command_move[0]]
    relative = np.clip(measured_t - alignment, command_t[0], command_t[-1])
    expected_q = np.column_stack([np.interp(relative, command_t, command_q[:, j]) for j in range(7)])
    expected_qd = np.column_stack([np.interp(relative, command_t, command_qd[:, j]) for j in range(7)])
    error = expected_q - measured_q
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "elapsed_s",
                "joint",
                "command_position_rad",
                "measured_position_rad",
                "command_velocity_rad_s",
                "measured_velocity_rad_s",
                "tracking_error_rad",
                "controller_faults",
            ]
        )
        fault_text = " | ".join(faults)
        for i, elapsed in enumerate(measured_t):
            for joint, name in enumerate(rammp_arm.JOINTS):
                writer.writerow(
                    [
                        f"{elapsed:.6f}",
                        name,
                        f"{expected_q[i, joint]:.9f}",
                        f"{measured_q[i, joint]:.9f}",
                        f"{expected_qd[i, joint]:.9f}",
                        f"{measured_qd[i, joint]:.9f}",
                        f"{error[i, joint]:.9f}",
                        fault_text,
                    ]
                )
    join_velocity = []
    for join in trajectory["join_times"]:
        index = int(np.argmin(np.abs(relative - join)))
        join_velocity.append(np.degrees(measured_qd[index]).tolist())
    return {
        "samples": len(samples),
        "alignment_s": alignment,
        "max_tracking_error_deg": float(np.max(np.abs(np.degrees(error)))),
        "measured_join_velocity_deg_s": join_velocity,
        "controller_faults": faults,
    }
