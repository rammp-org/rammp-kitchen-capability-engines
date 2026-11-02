"""A video of a dry run's planned open, in the sim: the arm (the driver's model), the door swinging with the hand and
the microwave's body, from the side and from above.

    python3 -m microwave.render_plan /microwave_data/runs/<time>_open        # -> <run>/plan.mp4

A dry run saves <run>/plan.json (open_door._save_plan). Run this where the arm driver publishes /arm/robot_description
(the microwave container on the robot). cuRobo's two moves -- the line-up and the push-open's up-and-back -- are
cuRobo's own plans when the dry run could ask the planner for them (plan_to_*; nothing moved); otherwise a straight
joint-space blend from their start to their checked goal, labelled so.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import rclpy
from capability_engine import sim
from door_common.common import DoorFrame, door_dims, say
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String

CONTINUOUS = (0, 2, 4, 6)  # J1, J3, J5, J7 turn without limit: blend them the short way
STEP_DEG = 2.0  # at most this much joint motion per frame
HOLD_FRAMES = 12  # frames for a gripper open / close, and at the end
WIDTH, HEIGHT = 640, 480
DOOR_RGBA, BODY_RGBA = (0.25, 0.5, 0.85, 0.85), (0.8, 0.8, 0.8, 1.0)


def _urdf(timeout_s=10.0):
    """The driver's URDF, from /arm/robot_description (latched)."""
    rclpy.init()
    node = rclpy.create_node("microwave_render_plan")
    got = {}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, "/arm/robot_description", lambda m: got.setdefault("urdf", m.data), qos)
    t0 = time.time()
    while "urdf" not in got and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if "urdf" not in got:
        raise SystemExit(f"no /arm/robot_description in {timeout_s:g} s -- is the arm driver up?")
    return got["urdf"]


def _blend(q0, q1):
    """Joint configurations from q0 to q1, the continuous joints the short way, at most STEP_DEG apart."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = q1 - q0
    d[list(CONTINUOUS)] = (d[list(CONTINUOUS)] + np.pi) % (2 * np.pi) - np.pi
    n = max(1, int(np.ceil(np.degrees(np.max(np.abs(d))) / STEP_DEG)))
    return [q0 + d * k / n for k in range(1, n + 1)]


def _through(q, path):
    """From q along the joint configurations `path`, densified to STEP_DEG."""
    out = []
    for nq in path:
        out += _blend(q, nq)
        q = out[-1]
    return out


class Scene3D:
    """The arm, the door slab and the body, as visual-only shapes in the sim's client."""

    def __init__(self, door):
        self.scene, self.rb = sim.make_sim()
        self.c = self.rb.physics_client_id
        self.base = np.asarray(self.scene.robot_base_pose.position)
        self.sign = float(door.get("open_sign", 1.0))
        self.df = DoorFrame(door, self.sign)
        face_off, self.t, past_handle, self.body_depth = door_dims(door)
        z_lo, z_hi = door["door_z"]
        self.zc, self.zh = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2
        self.door_id = self._box([self.t / 2, self.df.length / 2, self.zh], DOOR_RGBA)
        body = self._box([self.body_depth / 2, self.df.length / 2, self.zh], BODY_RGBA)
        self._place(body, self.df.length / 2, -(self.t / 2 + self.body_depth / 2), 0.0)
        self.set_door(0.0)

    def _box(self, half, rgba):
        vs = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=self.c)
        return p.createMultiBody(0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs, physicsClientId=self.c)

    def _place(self, body, u, v, deg):
        """A box centred at door-frame (u, v) at opening `deg`, its x along the door's normal."""
        c = self.df.pos(u, v, deg, self.zc)
        a, b = self.df.pos(0.0, 0.0, deg, 0.0), self.df.pos(1.0, 0.0, deg, 0.0)
        yaw = np.arctan2(b[1] - a[1], b[0] - a[0]) - np.pi / 2  # box y along the door (hinge -> free edge)
        p.resetBasePositionAndOrientation(
            body, (self.base + c).tolist(), R.from_euler("z", yaw).as_quat().tolist(), physicsClientId=self.c
        )

    def set_door(self, deg):
        self._place(self.door_id, self.df.length / 2, 0.0, deg)

    def set_arm(self, q, closed):
        for j, jj in enumerate(sim.ARM):
            p.resetJointState(self.rb.robot_id, jj, float(q[j]), physicsClientId=self.c)
        for jj in sim.FINGER_JOINTS:
            a = sim.FINGER_CLOSE_SIGN[jj] * sim.FINGER_CLOSED_RAD if closed else sim.FINGERTIP_OPEN_RAD
            p.resetJointState(self.rb.robot_id, jj, a, physicsClientId=self.c)

    def tool(self, q):
        return sim.tool_pose(self.scene, self.rb, q)[0]

    def door_deg_at(self, tool_xyz):
        """The door's opening with the hand on its handle at `tool_xyz`."""
        return max(0.0, self.df.angle_of(tool_xyz))

    def frame(self, label):
        target = self.base + np.r_[self.df.hinge[:2] * 0.6, self.zc]
        views = []
        for yaw, pitch, dist in ((-140.0, -25.0, 1.5), (-90.0, -89.0, 1.6)):  # from behind-right of the arm, above
            view = p.computeViewMatrixFromYawPitchRoll(target.tolist(), dist, yaw, pitch, 0.0, 2)
            proj = p.computeProjectionMatrixFOV(55.0, WIDTH / HEIGHT, 0.05, 5.0)
            _, _, rgba, _, _ = p.getCameraImage(
                WIDTH, HEIGHT, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=self.c
            )
            views.append(cv2.cvtColor(np.reshape(np.asarray(rgba, np.uint8), (HEIGHT, WIDTH, 4)), cv2.COLOR_RGBA2BGR))
        img = np.hstack(views)
        cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        return img


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("run_dir", type=Path, help="a dry run's folder, with plan.json")
    a.add_argument("--out", type=Path, help="the video (default <run_dir>/plan.mp4)")
    a.add_argument("--fps", type=int, default=30)
    args = a.parse_args()
    plan = json.loads((args.run_dir / "plan.json").read_text())
    if plan["door"] is None:
        raise SystemExit("plan.json has no door (no door height was detected) -- nothing to draw it from")
    urdf = _urdf()
    sim.set_model_provider(lambda: urdf)
    w = Scene3D(plan["door"])

    # (joints, gripper closed, door angle or None = follow the hand, label) per frame
    frames = []
    q = plan["start"]
    lineup = plan.get("line_up_curobo")
    for label, path in (
        (
            "line-up (cuRobo's plan)" if lineup else "line-up (cuRobo: goal checked, path blended)",
            lineup or [plan["line_up"]],
        ),
        ("straight in", plan["straight_in"]),
    ):
        for nq in _through(q, path):
            frames.append((nq, False, 0.0, label))
        q = frames[-1][0]
    frames += [(q, True, 0.0, "close the gripper")] * HOLD_FRAMES
    for nq in _through(q, plan["swing"]):
        frames.append((nq, True, None, f"swing {plan['swing_deg']} deg (tilt {plan['tilt_deg']} deg)"))
    q = frames[-1][0]
    swing_end_deg = w.door_deg_at(w.tool(q))
    po = plan["push_open"]
    if po is not None:
        frames += [(q, False, swing_end_deg, "let go")] * HOLD_FRAMES
        up = po.get("up_back_curobo")
        label = "up and back (cuRobo's plan)" if up else "up and back (cuRobo: goal checked, path blended)"
        for nq in _through(q, up or [po["up_back"]]):
            frames.append((nq, False, swing_end_deg, label))
        q = frames[-1][0]
        for leg in po["legs"]:
            pushing = leg["label"] == "push open"
            start_xy = w.tool(q)[:2] - w.df.hinge[:2] if pushing else None
            for nq in _through(q, leg["joints"]):
                deg = swing_end_deg
                if pushing:  # the door turns with the point the hand pushes
                    now = w.tool(nq)[:2] - w.df.hinge[:2]
                    turn = np.degrees(np.arctan2(start_xy[0] * now[1] - start_xy[1] * now[0], start_xy @ now))
                    deg = swing_end_deg + turn * w.sign
                frames.append((nq, False, deg, f"push-open: {leg['label']}"))
            q = frames[-1][0]
    frames += [frames[-1]] * HOLD_FRAMES

    out = args.out or args.run_dir / "plan.mp4"
    video = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * WIDTH, HEIGHT))
    for k, (nq, closed, deg, label) in enumerate(frames):
        w.set_arm(nq, closed)
        w.set_door(w.door_deg_at(w.tool(nq)) if deg is None else deg)
        w.set_arm(nq, closed)  # tool() moved the joints; put them back with the fingers
        video.write(w.frame(label))
        if k % 50 == 0:
            say(f"  frame {k}/{len(frames)}")
    video.release()
    say(f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
