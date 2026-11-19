# ruff: noqa: E501, T201  -- PoC debugging tool, not capability code
"""cuRobo's collision model as Foxglove markers: the planner's own world (its planner.yaml, layers and padding, built
the way the planner node builds it) and the arm's collision spheres at the live /joint_states, both in the planner's
base frame (arm_mount). A sphere that reaches into a box, and that box, are red: the same test cuRobo's start-state
check makes (sphere radius > its centre's signed distance to the padded box). Read-only: it subscribes
/arm/robot_description and /joint_states and publishes /curobo_viz/markers, nothing else.

Runs in the rammp-curobo image (run_collision_viz.sh). usage: collision_viz.py <planner.yaml> [hz]"""

import sys
import tempfile
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rammp_curobo import CuRoboPlanner
from rammp_curobo.config import load_planner_config
from rammp_curobo.contract import check_robot_dir
from rammp_curobo.urdf_model import write_urdf
from rammp_curobo.world import _signed_distances
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

LAYER_COLOR = {"seat": (0.4, 0.5, 0.9), "person": (1.0, 0.6, 0.2), "platform": (0.6, 0.6, 0.6)}
HIT = (1.0, 0.0, 0.0)


def rgba(c, a):
    return ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=float(a))


class Viz(Node):
    def __init__(self, cfg_path, hz):
        super().__init__("curobo_collision_viz")
        self.cfg_path, self.q, self.planner = cfg_path, None, None
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(String, "/arm/robot_description", self._model, qos)
        self.create_subscription(
            JointState, "/joint_states", self._joints, qos_profile_sensor_data
        )  # best effort, as published
        self.pub = self.create_publisher(MarkerArray, "/curobo_viz/markers", 1)
        self.create_timer(1.0 / hz, self._tick)
        self.last_report = ""

    def _model(self, msg):
        if self.planner is not None:
            return
        self.get_logger().info("arm model received; building the planner's world (no warm-up) ...")
        cfg, cfg_dir = load_planner_config(self.cfg_path)
        cfg["planner"]["warmup"] = (
            False  # only its kinematics and world are used: nothing is planned
        )
        self._dir = tempfile.TemporaryDirectory(prefix="curobo_viz_")
        xml = check_robot_dir(
            cfg, cfg_dir, msg.data
        )  # the model as the planner node prepares it (and its checks)
        self.planner = CuRoboPlanner(
            cfg, config_dir=cfg_dir, urdf_path=write_urdf(xml, self._dir.name)
        )
        kc = self.planner._motion_gen.kinematics.kinematics_config
        idx_to_name = {int(v): k for k, v in kc.link_name_to_idx_map.items()}
        self.sphere_link = [
            idx_to_name.get(int(i), "?") for i in kc.link_sphere_idx_map.cpu().numpy()
        ]
        self.frame = self.planner._cfg.get("base_frame") or "arm_mount"
        self.get_logger().info(
            f"world: {self.planner.world_summary()}; {len(self.sphere_link)} spheres; frame {self.frame}"
        )

    def _joints(self, msg):
        if self.planner is None:
            return
        pos = dict(zip(msg.name, msg.position, strict=False))
        if all(n in pos for n in self.planner.joint_names):
            self.q = [pos[n] for n in self.planner.joint_names]

    def _tick(self):
        if self.planner is None or self.q is None:
            return
        spheres = self.planner._spheres(self.q)
        live = spheres[:, 3] > 0
        boxes = self.planner._world_cuboids
        keys = list(boxes)
        dist = (
            _signed_distances(spheres[:, :3], boxes) if keys else np.zeros((len(spheres), 0))
        )  # (N spheres, M boxes)
        depth = spheres[:, 3:4] - dist
        depth[~live] = -np.inf
        hit_sphere = (depth > 0).any(axis=1)
        hit_box = (depth > 0).any(axis=0)
        stamp = self.get_clock().now().to_msg()
        out = MarkerArray()
        out.markers.append(Marker(action=Marker.DELETEALL))
        for m, key in enumerate(keys):
            b = boxes[key]
            x, y, z, qw, qx, qy, qz = (float(v) for v in b["pose"])
            mk = Marker(type=Marker.CUBE, action=Marker.ADD, ns="world", id=m)
            mk.header.frame_id, mk.header.stamp = self.frame, stamp
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = x, y, z
            (
                mk.pose.orientation.w,
                mk.pose.orientation.x,
                mk.pose.orientation.y,
                mk.pose.orientation.z,
            ) = qw, qx, qy, qz
            mk.scale.x, mk.scale.y, mk.scale.z = (float(d) for d in b["dims"])
            layer = self.planner._world_sources[key]
            mk.color = (
                rgba(HIT, 0.55)
                if hit_box[m]
                else rgba(LAYER_COLOR.get(layer, (0.3, 0.8, 0.3)), 0.2)
            )
            out.markers.append(mk)
            if hit_box[m]:
                t = Marker(type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, ns="hit_labels", id=m)
                t.header.frame_id, t.header.stamp = self.frame, stamp
                t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, z
                t.scale.z, t.color = 0.03, rgba((1, 1, 1), 1)
                t.text = f"{key} {1000 * depth[:, m].max():.0f} mm"
                out.markers.append(t)
        for i in np.flatnonzero(live):
            mk = Marker(
                type=Marker.SPHERE,
                action=Marker.ADD,
                ns="spheres_hit" if hit_sphere[i] else "spheres",
                id=int(i),
            )
            mk.header.frame_id, mk.header.stamp = self.frame, stamp
            mk.pose.position = Point(
                x=float(spheres[i, 0]), y=float(spheres[i, 1]), z=float(spheres[i, 2])
            )
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 2 * float(spheres[i, 3])
            mk.color = rgba(HIT, 0.9) if hit_sphere[i] else rgba((0.2, 0.9, 0.3), 0.35)
            out.markers.append(mk)
        self.pub.publish(out)
        hits = sorted(
            (
                (float(depth[i, m]), self.sphere_link[i], i, keys[m])
                for i in np.flatnonzero(hit_sphere)
                for m in np.flatnonzero(depth[i] > 0)
            ),
            reverse=True,
        )
        report = (
            "; ".join(f"{link} sphere {i} {1000 * d:.0f} mm into {key}" for d, link, i, key in hits)
            or "no sphere in any box"
        )
        if report != self.last_report:
            self.get_logger().info(
                f"joints {np.round(np.degrees(self.q), 1).tolist()} deg: {report}"
            )
            self.last_report = report


def main():
    rclpy.init()
    node = Viz(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0)
    t0 = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            if node.planner is None and time.time() - t0 > 30:
                node.get_logger().warn(
                    "no model on /arm/robot_description yet (arm_state_publisher up?)"
                )
                t0 = time.time()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
