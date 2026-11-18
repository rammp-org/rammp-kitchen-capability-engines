"""Read-only commissioning checks and active-URDF collision inventory for microwave placement."""

import json
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import place_motion as pm

PASS = "PASS"
FAIL = "FAIL"
CAMERA_MAX_AGE_S = 1.0
CAMERA_WAIT_S = 5.0


def check(call):
    """Run one independent preflight check and preserve its actionable failure."""
    try:
        detail = call()
        return {"status": PASS, "detail": detail}
    except Exception as error:  # noqa: BLE001 -- every check must report instead of hiding later checks
        if type(error).__name__ == "Cancelled":
            raise
        return {"status": FAIL, "detail": str(error), "error_type": type(error).__name__}


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def urdf_collision_inventory(urdf_text):
    """JSON-safe list of collision shapes declared on each active-URDF link."""
    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError as error:
        raise ValueError(f"active robot_description is invalid XML: {error}") from error
    inventory = []
    for link in (element for element in root.iter() if _tag(element) == "link"):
        collisions = []
        for collision in (element for element in link if _tag(element) == "collision"):
            geometry = next((element for element in collision if _tag(element) == "geometry"), None)
            shape = next(iter(geometry), None) if geometry is not None else None
            if shape is None:
                continue
            origin = next((element for element in collision if _tag(element) == "origin"), None)
            record = {
                "type": _tag(shape),
                "origin_xyz": (origin.get("xyz", "0 0 0").split() if origin is not None else ["0", "0", "0"]),
                "origin_rpy": (origin.get("rpy", "0 0 0").split() if origin is not None else ["0", "0", "0"]),
            }
            record.update(shape.attrib)
            collisions.append(record)
        if collisions:
            inventory.append({"link": link.get("name", ""), "collision_shapes": collisions})
    return inventory


def collision_configuration_report(path, urdf_text):
    """Audit a geometry YAML against its schema and the active URDF without creating collision bodies."""
    data = pm.read_collision_geometry(path)
    inventory = urdf_collision_inventory(urdf_text)
    available = {item["link"] for item in inventory}
    links = data.get("robot_collision_links", {})
    configured = sorted(
        name
        for names in (links.values() if isinstance(links, dict) else [])
        if isinstance(names, list)
        for name in names
        if isinstance(name, str)
    )
    missing_links = sorted(set(configured) - available)
    validation_error = ""
    try:
        pm.validate_collision_geometry(data)
    except pm.InsertionCollisionModelUnavailable as error:
        validation_error = str(error)
    return {
        "path": str(path),
        "complete": not validation_error and not missing_links,
        "validation_error": validation_error,
        "missing_measurements": (
            list(data.get("missing_measurements", [])) if isinstance(data.get("missing_measurements", []), list) else []
        ),
        "configured_robot_links": configured,
        "configured_links_missing_from_active_urdf": missing_links,
        "active_urdf_collision_inventory": inventory,
        "verified_measurements": data.get("verified_measurements", {}),
    }


def diagnostic_faults(diagnostics):
    """Normalize converted ArmInterface diagnostics into concise fault strings."""
    faults = []
    for status in diagnostics.status:
        values = (
            status.values if isinstance(status.values, dict) else {value.key: value.value for value in status.values}
        )
        count = str(values.get("faults", "0")).strip()
        try:
            nonzero = bool(count) and float(count) != 0.0
        except ValueError:
            nonzero = True
        if int(status.level) >= 2 or nonzero:
            faults.append(f"{status.name}: {status.message} (faults={count or 'unknown'})")
    return faults


def save_report(state_dir, checks):
    directory = Path(state_dir) / "preflight"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = directory / f"preflight_{stamp}.json"
    path.write_text(
        json.dumps({"ready": all(item["status"] == PASS for item in checks.values()), "checks": checks}, indent=2)
    )
    return path


class SceneProbe:
    """Action-scoped color/depth/info and timestamped-TF freshness probe; never requests perception."""

    def __init__(self, node, frame, color_topic, depth_topic, info_topic):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self._node, self._frame = node, frame
        self._lock, self._latest = threading.Lock(), {}
        self._tf = node.create_action_tf_buffer()

        def keep(name):
            def callback(message):
                with self._lock:
                    self._latest[name] = (message, time.monotonic())

            return callback

        node.create_action_subscription(Image, color_topic, keep("color"), qos_profile_sensor_data)
        node.create_action_subscription(Image, depth_topic, keep("depth"), qos_profile_sensor_data)
        node.create_action_subscription(CameraInfo, info_topic, keep("info"), qos_profile_sensor_data)

    @staticmethod
    def _stamp_ns(message):
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def wait(self, checkpoint, timeout_s=CAMERA_WAIT_S, max_age_s=CAMERA_MAX_AGE_S):
        from rclpy.duration import Duration
        from rclpy.time import Time

        deadline = time.monotonic() + timeout_s
        latest = {}
        while time.monotonic() < deadline:
            checkpoint()
            with self._lock:
                latest = dict(self._latest)
            if set(latest) == {"color", "depth", "info"}:
                break
            time.sleep(0.02)
        missing = sorted({"color", "depth", "info"} - set(latest))
        if missing:
            raise RuntimeError(f"no fresh scene-camera {', '.join(missing)} within {timeout_s:g} s")
        now = time.monotonic()
        ages = {name: now - received for name, (_, received) in latest.items()}
        stale = {name: age for name, age in ages.items() if age > max_age_s}
        if stale:
            raise RuntimeError(f"stale scene-camera streams: {stale} (limit {max_age_s:g} s)")
        depth = latest["depth"][0]
        if not depth.header.frame_id:
            raise RuntimeError("scene depth frame_id is empty")
        try:
            self._tf.lookup_transform(
                self._frame,
                depth.header.frame_id,
                Time.from_msg(depth.header.stamp),
                timeout=Duration(seconds=0.5),
            )
        except Exception as error:  # noqa: BLE001 -- tf2 exposes several lookup exception classes
            raise RuntimeError(
                f"no TF {self._frame} <- {depth.header.frame_id} at the latest depth stamp ({type(error).__name__})"
            ) from error
        color, info = latest["color"][0], latest["info"][0]
        stamps = {"color": self._stamp_ns(color), "depth": self._stamp_ns(depth), "info": self._stamp_ns(info)}
        if stamps["color"] <= 0 or stamps["depth"] <= 0:
            raise RuntimeError(f"scene color/depth timestamps must be valid ROS times: {stamps}")
        return {
            "receipt_age_s": {name: round(age, 3) for name, age in ages.items()},
            "stamp_ns": stamps,
            "color_depth_skew_ms": round(abs(stamps["color"] - stamps["depth"]) / 1e6, 3),
            "tf": f"{self._frame} <- {depth.header.frame_id}",
            "depth_encoding": depth.encoding,
            "size": [int(depth.width), int(depth.height)],
        }


def state_detail(joints, flange, gripper):
    return {
        "joint_position_deg": np.degrees(joints.position).round(3).tolist(),
        "joint_velocity_deg_s": np.degrees(joints.velocity).round(3).tolist(),
        "joint_age_s": round(float(joints.age_s), 3),
        "flange_frame": flange.frame_id,
        "flange_position_m": np.asarray(flange.position).round(4).tolist(),
        "flange_age_s": round(float(flange.age_s), 3),
        "gripper_position": round(float(gripper.position), 4),
        "gripper_effort": round(float(gripper.effort), 4),
        "gripper_current": round(float(gripper.current), 4),
        "gripper_present": bool(gripper.present),
        "gripper_age_s": round(float(gripper.age_s), 3),
    }
