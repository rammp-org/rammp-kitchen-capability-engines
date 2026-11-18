#!/usr/bin/env python3
"""Live collision view for Foxglove: what cuRobo and the sim check the arm against, coloured by contact.

Publishes a visualization_msgs/MarkerArray on /collision_viz (add it to a 3D panel; fixed frame arm_mount):

- arm_spheres   cuRobo's collision spheres of the arm (robot.yaml, + collision_sphere_buffer), locked to their link
                frames, so they move with TF. Green clear, yellow within --near-m of a box, red inside one.
- curobo_world  the planner's world boxes as cuRobo checks them: planner.yaml's layers (platform, seat, person) and
                its boot world's (the floor), each padded world_padding per side. Grey.
- door_world    the door model while the close has it in cuRobo's world (door_common's door_world_*.yaml, there
                only while it is loaded), padded as the planner pads it. Orange.
- sim_door      the sim's door model from the door file (door slab at --door-deg, the microwave body), unpadded: what
                the close's own checks use (>= 3 cm). Blue. Needs door_common importable; skipped otherwise.
- status        text over the arm: the closest sphere and box, and by how much (negative = inside).

The files are re-read every second, the distances every frame (--rate Hz). Run on the host, ROS sourced:

    python3 shared/door_common/door_common/collision_viz.py                 # sheppy's config, the default door file
    python3 shared/door_common/door_common/collision_viz.py --door-deg 0    # the sim door shown shut

Nothing here talks to the arm or the planner: it reads their config files and /tf only.
"""

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

FRAME = "arm_mount"  # cuRobo's world frame, and the door file's
ROBOT_CONFIG = os.environ.get("SHEPPY_ROBOT_CONFIG", "~/sheppy-local/december_2026/config/robots/current")
PLANNER_ROOT = "/config/robot"  # how the planner's container sees ROBOT_CONFIG (paths inside a world file)

GREEN, YELLOW, RED = (0.1, 0.8, 0.2, 0.35), (1.0, 0.85, 0.0, 0.7), (1.0, 0.1, 0.1, 0.9)
COLOURS = {
    "curobo_world": (0.6, 0.6, 0.6, 0.25),
    "door_world": (1.0, 0.5, 0.0, 0.45),
    "sim_door": (0.2, 0.4, 1.0, 0.35),
}


def quat_rpy_deg(rpy_deg):
    """cuRobo's rpy (deg) -> quaternion (x, y, z, w), as rammp_curobo.geometry.euler_deg_to_quat_xyzw."""
    r, p, y = (math.radians(float(a)) for a in rpy_deg)
    cr, sr, cp, sp, cy, sy = (
        math.cos(r / 2),
        math.sin(r / 2),
        math.cos(p / 2),
        math.sin(p / 2),
        math.cos(y / 2),
        math.sin(y / 2),
    )
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rot(q):
    """Rotation matrix of quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class Box:
    def __init__(self, ns, name, position, dims, quat):
        self.ns, self.name = ns, name
        self.position, self.dims, self.quat = (
            np.asarray(position, float),
            np.asarray(dims, float),
            tuple(quat),
        )
        self.r = rot(self.quat)

    def distance(self, centre):
        """Signed distance from point `centre` to the box surface (negative inside)."""
        p = self.r.T @ (np.asarray(centre) - self.position)
        q = np.abs(p) - self.dims / 2
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(q.max()), 0.0))


def host_path(path, relative_to):
    """A path from a planner file, on this host: ones under the planner's /config/robot mapped to ROBOT_CONFIG."""
    path = str(path)
    if path.startswith(PLANNER_ROOT + "/"):
        return Path(ROBOT_CONFIG).expanduser() / path[len(PLANNER_ROOT) + 1 :]
    return Path(path) if os.path.isabs(path) else Path(relative_to) / path


def layer_boxes(path, ns, pad):
    data = yaml.safe_load(Path(path).read_text())
    name = data.get("layer", Path(path).stem)
    return [
        Box(
            ns,
            f"{name}/{o['name']}",
            o["position"],
            [d + 2 * pad for d in o["dims"]],
            quat_rpy_deg(o.get("rpy_deg", [0, 0, 0])),
        )
        for o in data["obstacles"]
    ]


def world_boxes(path, ns, pad, skip_layers=()):
    """A world file's obstacles and its layers' boxes, padded `pad` per side."""
    data = yaml.safe_load(Path(path).read_text())
    boxes = [
        Box(
            ns,
            o["name"],
            o["position"],
            [d + 2 * pad for d in o["dims"]],
            quat_rpy_deg(o.get("rpy_deg", [0, 0, 0])),
        )
        for o in data.get("obstacles") or []
    ]
    for layer in data.get("layers") or []:
        lp = host_path(layer, Path(path).parent)
        if lp.resolve() not in skip_layers:
            boxes += layer_boxes(lp, "curobo_world", pad)  # the floor: the planner's own, grey
    return boxes


class CollisionViz(Node):
    def __init__(self, args):
        super().__init__("collision_viz")
        self.args = args
        self.cfg_dir = Path(args.config).expanduser()
        self.pub = self.create_publisher(MarkerArray, "/collision_viz", 1)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.spheres, self.boxes, self.loaded_at, self.notes = {}, [], 0.0, []
        self.create_timer(1.0 / args.rate, self.tick)

    def load(self):
        """Re-read the planner config, the door world (if one is loaded) and the door file."""
        notes = []
        planner = yaml.safe_load((self.cfg_dir / "planner.yaml").read_text())
        pad = float(planner.get("world_padding", 0.02))
        robot = yaml.safe_load((self.cfg_dir / planner.get("robot", "robot.yaml")).read_text())["robot_cfg"][
            "kinematics"
        ]
        buf = float(robot.get("collision_sphere_buffer", 0.0))
        self.spheres = {
            link: [
                (np.asarray(s["center"], float), float(s["radius"]) + buf) for s in spheres if float(s["radius"]) > 0
            ]
            for link, spheres in robot["collision_spheres"].items()
        }
        config_layers = [host_path(lp, self.cfg_dir) for lp in planner.get("layers") or []]
        boxes = [b for lp in config_layers for b in layer_boxes(lp, "curobo_world", pad)]
        door_worlds = sorted(glob.glob(str(self.cfg_dir / "door_world" / "door_world_*.yaml")), key=os.path.getmtime)
        if door_worlds:  # cuRobo has the door (the close deletes the file once its boot world is back)
            boxes += world_boxes(door_worlds[-1], "door_world", pad, {p.resolve() for p in config_layers})
            notes.append(f"door in cuRobo's world ({Path(door_worlds[-1]).name})")
        else:
            boxes += world_boxes(self.cfg_dir / planner.get("world", "world.yaml"), "curobo_world", pad)
        boxes += self.sim_door(notes)
        self.boxes, self.notes = boxes, notes

    def sim_door(self, notes):
        door_file = Path(self.args.door_file)
        if not door_file.exists():
            notes.append(f"no door file {door_file}")
            return []
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a file: door_common beside it
            from door_common import common as dc
        except Exception as e:  # noqa: BLE001 -- any import failure: just no sim door
            notes.append(f"sim door skipped (door_common: {e.__class__.__name__})")
            return []
        door = yaml.safe_load(door_file.read_text())  # JSON is YAML
        df = dc.DoorFrame(door, door.get("open_sign", 1.0))
        boxes, _ = dc.door_model_boxes(door, df.bodies_point(self.args.door_deg))
        return [
            Box(
                "sim_door",
                name,
                centre,
                2 * np.asarray(half),
                (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
            )
            for name, centre, half, yaw in boxes
        ]

    def tick(self):
        now = time.monotonic()
        if now - self.loaded_at > 1.0:
            try:
                self.load()
            except Exception as e:  # noqa: BLE001 -- keep the last view; say why on screen
                self.notes = [f"config not read: {e}"]
            self.loaded_at = now
        stamp = self.get_clock().now().to_msg()
        out = MarkerArray()
        out.markers.append(Marker(action=Marker.DELETEALL))
        for i, b in enumerate(self.boxes):
            m = Marker(ns=b.ns, id=i, type=Marker.CUBE, action=Marker.ADD, frame_locked=True)
            m.header.frame_id, m.header.stamp = FRAME, stamp
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, b.position)
            (
                m.pose.orientation.x,
                m.pose.orientation.y,
                m.pose.orientation.z,
                m.pose.orientation.w,
            ) = map(float, b.quat)
            m.scale.x, m.scale.y, m.scale.z = map(float, b.dims)
            m.color = ColorRGBA(**dict(zip("rgba", COLOURS[b.ns], strict=True)))
            out.markers.append(m)
        worst, k = None, 0
        for link, spheres in self.spheres.items():
            try:
                t = self.tf.lookup_transform(FRAME, link, Time()).transform
            except Exception:  # noqa: BLE001 -- no TF for this link yet: draw it, uncoloured
                t = None
            if t is not None:
                r = rot((t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w))
                o = np.array([t.translation.x, t.translation.y, t.translation.z])
            for j, (c, radius) in enumerate(spheres):
                colour = GREEN
                if t is not None and self.boxes:
                    w = r @ c + o
                    d, box = min(((b.distance(w) - radius, b) for b in self.boxes), key=lambda x: x[0])
                    colour = RED if d < 0 else YELLOW if d < self.args.near_m else GREEN
                    if worst is None or d < worst[0]:
                        worst = (d, f"{link}[{j}]", box)
                m = Marker(ns="arm_spheres", id=k, type=Marker.SPHERE, action=Marker.ADD, frame_locked=True)
                m.header.frame_id, m.header.stamp = link, stamp
                m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, c)
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 2 * radius
                m.color = ColorRGBA(**dict(zip("rgba", colour, strict=True)))
                out.markers.append(m)
                k += 1
        text = Marker(ns="status", id=0, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, frame_locked=True)
        text.header.frame_id, text.header.stamp = FRAME, stamp
        text.pose.position.z, text.pose.orientation.w, text.scale.z = 0.9, 1.0, 0.04
        lines = list(self.notes)
        if worst is None:
            lines.insert(0, "no TF for the arm's links (is /tf up?)")
        else:
            d, sphere, box = worst
            state = "INSIDE" if d < 0 else "near" if d < self.args.near_m else "clear"
            lines.insert(0, f"{state}: {sphere} {d * 100:+.1f} cm from {box.ns} {box.name}")
        text.text = "\n".join(lines)
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        out.markers.append(text)
        self.pub.publish(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=f"{ROBOT_CONFIG}/curobo", help="the planner's config dir on this host")
    ap.add_argument("--door-file", default="/tmp/door_common/door.json", help="the door the open saved")
    ap.add_argument("--door-deg", type=float, default=90.0, help="sim door opening to show (deg)")
    ap.add_argument("--near-m", type=float, default=0.03, help="yellow within this of a box (m)")
    ap.add_argument("--rate", type=float, default=10.0, help="Hz")
    args = ap.parse_args()
    rclpy.init()
    node = CollisionViz(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
