"""The microwave_place capability: put the container the gripper holds into the open microwave.

Actions:
  locate  the open microwave's interior from the scene camera (interior.locate): the spot on its
          floor, the insertion axis and the floor height, in arm_mount. No motion.
  place   locate, capture the scene camera's depth as obstacles, then plan the whole place, check
          every configuration of every leg against the oven model and the obstacles (place.py:
          adjust, lift, transit in front of the oven, straight in, lower in impedance until the floor
          stops it, open, straight back out) and run it. The arm is left in impedance, held where the
          back-out ended.

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets `execute`.
Otherwise every goal is a dry run that locates, plans and checks, and returns the plan. Someone stands
at the e-stop whenever allow_execute is on.

Every `place`, dry run or not, publishes what it planned on /microwave_place/plan
(visualization_msgs/MarkerArray, frame arm_mount, latched): the oven model, the obstacle voxels, the
spot, the tool's path on each leg with its tightest clearance, and the container at pre-insert, inserted
and resting. A leg that failed its check is red, with its closest pair drawn. Look at it in Foxglove
before running with `execute`.

    ros2 launch microwave_place microwave_place.launch.py use_orchestrator:=false allow_execute:=true
    make action CAP=microwave_place ACTION=locate
    make action CAP=microwave_place ACTION=place PARAMS='{"execute": false}'
"""

import time
from pathlib import Path

import numpy as np
import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run
from capability_engine.arm import HOLD_LEVEL
from geometry_msgs.msg import Point
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from . import interior, place

PLAN_TOPIC = "/microwave_place/plan"
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
DEPTH_SCALE = {"16UC1": 0.001, "mono16": 0.001, "32FC1": 1.0}  # depth encoding -> metres per unit
DEPTH_RANGE = (0.15, 3.0)  # m, depth outside this is not a point
STILL_DEG = 0.5  # the arm may move at most this much (any joint) while the scene is captured
LEG_COLORS = {  # rgb per leg on the marker topic
    "adjust": (0.6, 0.6, 0.6),
    "lift": (1.0, 0.85, 0.1),
    "transit": (0.2, 0.5, 1.0),
    "insert": (0.1, 0.8, 0.3),
    "lower": (0.9, 0.2, 0.9),
    "back out": (0.1, 0.9, 0.9),
}

# ROS parameter -> default.
PARAMS = {
    "allow_execute": False,
    # flange face to the closed fingertips (m): sets where the container centre lands
    "flange_to_tip": place.FLANGE_TO_TIP,
    # the hand points this many deg down while placing, so the container touches first
    "pitch_deg": place.PLACE_PITCH_DEG,
    # contact: vz (m/s, + up) back up to this after moving down
    "vz_stop": -0.01,
    # the hold after contact, this far (m) below the contact point, while the gripper opens
    "hold_press": 0.003,
    # held in impedance at the start of the lowering this long (s); must end still
    "settle_s": 3.0,
    # contact more than this (m) above where the bottom meets the floor = not the floor
    "early_stop": 0.03,
    # joint impedance for the settle, lowering, hold and back-out (one set: the stiffness never changes
    # mid-place). Default: the driver's documented starting point (rammp_arm.IMPEDANCE)
    "impedance_kq": list(rammp_arm.IMPEDANCE["kq"]),
    "impedance_zeta": rammp_arm.IMPEDANCE["zeta"],
    "impedance_torque_limit": list(rammp_arm.IMPEDANCE["torque_limit"]),
    # the hand's speed (m/s) where the floor should be: what the contact test was tuned at
    "lower_speed": 0.02,
    # speed_scale of the carrying moves (go_to_ee_pose)
    "go_to_speed_scale": 0.4,
    # speed_scale of the squaring turn (move_joints)
    "adjust_speed_scale": 1.0,
    # the back-out's peak speed (m/s)
    "retreat_speed": 0.06,
    # the squaring turn may correct at most this much (deg)
    "max_adjust_deg": place.MAX_ADJUST_DEG,
    # the arm's tool frame vs the model's (m): further apart = wrong model or tool configuration
    "tool_frame_tol": 0.015,
    # where the descent recordings are kept
    "state_dir": "/config/microwave_place",
    # the scene camera and the 3D detections
    "depth_topic": "/scene_camera/depth/image_rect_raw",
    "camera_info_topic": "/scene_camera/color/camera_info",
    "detections_topic": "/perception/object_detections_3d",
    # locate: this many detected frames, within this long (s), that agree to agree_m / agree_deg
    "frames": 5,
    "timeout_s": 10.0,
    "agree_m": 0.01,
    "agree_deg": 2.0,
    # the container's bottom this far (m) above the oven floor on the way in
    "insert_clearance": 0.03,
    # in front of the oven, the container's far edge this far (m) from its outer front
    "standoff": 0.05,
    # the straight line in, peak speed (m/s)
    "insert_speed": 0.03,
    # clearances (m) checked at every configuration of every leg, in the sim: every arm/gripper link from
    # the oven model and the scene voxels ...
    "clear_link": 0.03,
    # ... the held container from them ...
    "clear_container": 0.02,
    # ... and, on the legs into, in and out of the oven, links and container from the oven model
    "clear_inside": 0.015,
    # checked configurations are at most this far apart (m, any link frame or container rim point)
    "check_step_m": 0.005,
    # obstacles from the scene camera: this many new depth frames (taken after locate, arm still) ...
    "obstacle_frames": 3,
    # ... their points as cubes this size (m) ...
    "voxel_m": 0.03,
    # ... within this distance (m) of arm_mount's origin (the arm's reach with the container) ...
    "region_m": 1.4,
    # ... except points this close (m) to the oven model (its own surfaces, modelled already) ...
    "surface_band": 0.03,
    # ... or to the arm and the held container where they were (they are the robot itself)
    "self_band": 0.04,
}
PLACE = {
    "type": "object",
    "properties": {
        "execute": {"type": "boolean", "default": False},
        # an empty-gripper test run: don't refuse an open gripper
        "allow_empty_gripper": {"type": "boolean", "default": False},
    },
}


class Capability(CapabilityEngine):
    display_name = "Place container in microwave"
    perception_classes = ["microwave"]
    required_resources = ["arm", "gripper"]
    actions = [
        Action("place", "Put the held container into the open microwave", params_schema=PLACE),
        Action("locate", "Find the spot inside the open microwave", params_schema={"type": "object"}),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("microwave_place", **node_kwargs)
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        self.cfg["gains"] = {
            "kq": [float(v) for v in self.cfg["impedance_kq"]],
            "zeta": float(self.cfg["impedance_zeta"]),
            "torque_limit": [float(v) for v in self.cfg["impedance_torque_limit"]],
        }
        # for the life of the node and latched, so the last plan stays visible after its action ends
        self._plan_pub = self.create_publisher(MarkerArray, PLAN_TOPIC, LATCHED)
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def execute(self, action_id, params, ctx):
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing located, planned or moved")
            return {"executed": False, "mock": True}
        log = self.get_logger().info
        try:
            if action_id == "locate":
                return self._locate(ctx, log)
            return {**self._place(params, execute, ctx, log), "note": note}
        except (place.Refused, interior.NotLocated) as e:
            raise EngineError("refused", str(e), recoverable=True) from e

    def _locate(self, ctx, log):
        ctx.progress("locate", 0.0)
        cfg = self.cfg
        target = interior.locate(
            self,
            rammp_arm.FRAME,
            (cfg["depth_topic"], cfg["camera_info_topic"], cfg["detections_topic"]),
            cfg["frames"],
            cfg["timeout_s"],
            cfg["agree_m"],
            cfg["agree_deg"],
            log,
        )
        log(
            f"microwave in {rammp_arm.FRAME}: spot {np.round(target['point'], 3).tolist()}, insertion axis "
            f"{np.round(target['insertion_axis'], 3).tolist()}, floor z {target['floor_z']:.3f} "
            f"({target['frames']} frames within {target['spread_m'] * 100:.1f} cm, {target['spread_deg']:.1f} deg)"
        )
        return {**target, "frame": rammp_arm.FRAME}

    def _place(self, params, execute, ctx, log):
        model = place.Model(self.cfg["flange_to_tip"], self.cfg["pitch_deg"])
        self._check_tool_frame(model, log)
        gripper = float(self.arm.gripper_state().position)
        if gripper < 0.05 and not params.get("allow_empty_gripper", False):
            raise place.Refused("the gripper is fully open -- it isn't holding the container")
        target = self._locate(ctx, log)
        ctx.progress("obstacles", 0.0)
        cloud, q_seen = self._scene_points(ctx, log)
        oven = place.Oven(target)
        obs = place.Obstacles(model, oven, place.camera_on_left(model.fk(q_seen)[:3, :3]))
        view = place.View(oven)
        try:
            n = obs.add_scene(cloud, q_seen, self.cfg)
            view.voxels, view.voxel_m = obs.voxels, obs.voxel_m
            log(
                f"obstacles: the oven model (5 boxes) + {n['voxels']} voxels of {self.cfg['voxel_m'] * 100:.0f} cm "
                f"({n['points']} scene points within {self.cfg['region_m']:.1f} m, {n['off_oven']} of them off the "
                f"oven model; {n['on_robot']} voxels on the arm or the container left out)"
            )
            ctx.progress("plan", 0.0)
            plan = place.plan_microwave(model, obs, self.arm.raw_joints(), self.cfg, self._curobo, view, log=log)
            steps = self._describe(plan, view, log)
            if not execute:
                log(f"DRY RUN -- all checks passed, nothing commanded; the plan is on {PLAN_TOPIC}")
                return {"executed": False, "target": target, **steps}

            why = self.arm.not_ready()
            if why is not None:
                raise place.Refused(why)
            rec = place.Recorder()
            try:
                place.run(
                    self.arm,
                    model,
                    obs,
                    plan,
                    self.cfg,
                    ctx,
                    log,
                    self._joints,
                    rec,
                    self._curobo,
                    view,
                    lambda: self._show(view),
                )
            finally:
                path = rec.write(Path(self.cfg["state_dir"]) / "descents")
                if path is not None:
                    log(f"recorded {len(rec.rows)} /ee_state messages -> {path}")
            return {
                "executed": True,
                "target": target,
                **self._describe(plan, view, log),
                "recording": None if path is None else str(path),
            }
        finally:
            self._show(view)  # also what failed, in red
            obs.remove()

    # -- helpers -------------------------------------------------------------------------------

    def _joints(self):
        return np.asarray(self.arm.get_state()["position"], dtype=float)

    def _curobo(self, pos, quat, start):
        """cuRobo's plan (nothing moves) for the tool to (pos, quat) from joints `start`, the tilt kept."""
        return self.arm.plan_to_pose(pos, quat, start, hold=HOLD_LEVEL)

    def _check_tool_frame(self, model, log):
        """The arm's frames against the model's. /ee_state (self.arm.ee_state()) is the FLANGE,
        arm_end_effector_link; get_state()['ee_pos'] is that + TOOL_Z along its z, the TOOL frame
        (arm_tool_frame) that go_to_ee_pose and cuRobo target and model.fk() gives. Both are compared:
        reading one as the other is 12 cm off and refused."""
        q = self._joints()
        ee = self.arm.ee_state()
        flange = model.fk(q, "flange")
        tool_reported = np.asarray(self.arm.get_state()["ee_pos"][:3], dtype=float)
        tool = model.fk(q)[:3, 3]
        off = float(np.linalg.norm(ee.position - flange[:3, 3]))
        tool_off = float(np.linalg.norm(tool_reported - tool))
        turn = float(
            np.degrees((Rotation.from_quat(ee.orientation) * Rotation.from_matrix(flange[:3, :3]).inv()).magnitude())
        )
        log(
            f"flange: /ee_state {np.round(ee.position, 3).tolist()} ({ee.frame_id}), model "
            f"{np.round(flange[:3, 3], 3).tolist()}: {off * 100:.1f} cm, {turn:.1f} deg apart; tool frame: arm "
            f"{np.round(tool_reported, 3).tolist()}, model {np.round(tool, 3).tolist()}: {tool_off * 100:.1f} cm apart"
        )
        if ee.frame_id and ee.frame_id != rammp_arm.FRAME:
            raise place.Refused(f"/ee_state is in {ee.frame_id!r}, not {rammp_arm.FRAME!r}")
        if max(off, tool_off) > self.cfg["tool_frame_tol"] or turn > place.TOOL_FRAME_TOL_DEG:
            raise place.Refused(
                "the arm's flange / tool frame doesn't match the model's -- check the tool configuration"
            )

    def _scene_points(self, ctx, log):
        """The scene camera's points in arm_mount from `obstacle_frames` new depth frames (the camera's pose
        from TF at each frame's stamp), and the arm's joints while they were taken. Refused if the frames
        don't come or the arm moved meanwhile: there is no obstacle check without them."""
        from cv_bridge import CvBridge
        from rclpy.duration import Duration
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo, Image

        cfg, latest = self.cfg, {}
        tf = self.create_action_tf_buffer()
        self.create_action_subscription(
            Image, cfg["depth_topic"], lambda m: latest.update(depth=m), qos_profile_sensor_data
        )
        self.create_action_subscription(
            CameraInfo, cfg["camera_info_topic"], lambda m: latest.update(info=m), qos_profile_sensor_data
        )
        q0 = self.arm.raw_joints()
        bridge, clouds, last = CvBridge(), [], None
        deadline = time.monotonic() + cfg["timeout_s"]
        while len(clouds) < cfg["obstacle_frames"] and time.monotonic() < deadline:
            ctx.checkpoint()
            depth, info = latest.get("depth"), latest.get("info")
            stamp = None if depth is None else (depth.header.stamp.sec, depth.header.stamp.nanosec)
            if depth is None or info is None or stamp == last:
                time.sleep(0.02)
                continue
            last = stamp
            if depth.encoding not in DEPTH_SCALE:
                raise place.Refused(f"scene depth encoding {depth.encoding!r} (expected one of {sorted(DEPTH_SCALE)})")
            try:
                t = tf.lookup_transform(
                    rammp_arm.FRAME,
                    depth.header.frame_id,
                    Time.from_msg(depth.header.stamp),
                    timeout=Duration(seconds=0.5),
                ).transform
            except Exception as e:  # noqa: BLE001 -- tf2's lookup errors: this frame is not used
                log(f"no TF {rammp_arm.FRAME} <- {depth.header.frame_id} at the frame's stamp ({type(e).__name__})")
                continue
            R = Rotation.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
            d = bridge.imgmsg_to_cv2(depth).astype(np.float32) * DEPTH_SCALE[depth.encoding]
            K = np.array(info.k, dtype=float).reshape(3, 3)
            v, u = np.nonzero(np.isfinite(d) & (d > DEPTH_RANGE[0]) & (d < DEPTH_RANGE[1]))
            z = d[v, u].astype(float)
            cam = np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
            clouds.append(cam @ R.T + [t.translation.x, t.translation.y, t.translation.z])
        if len(clouds) < cfg["obstacle_frames"]:
            raise place.Refused(
                f"obstacles: {len(clouds)} scene depth frames with TF in {cfg['timeout_s']:.0f} s "
                f"(need {cfg['obstacle_frames']}) -- not planning without them"
            )
        q1 = self.arm.raw_joints()
        moved = float(np.max(np.abs(place.wrap_deg(q1 - q0))))
        if moved > STILL_DEG:
            raise place.Refused(f"obstacles: the arm moved {moved:.1f} deg while the scene was captured")
        return np.vstack(clouds), q1

    def _describe(self, plan, view, log):
        """The plan's numbers for the log and the result: the height it reaches in at, and each leg's
        tightest clearance."""
        info, h = plan["info"], plan["info"]["height"]
        floor = info["floor_z"]
        log(
            f"reach in with the tool at z={info['z_in']:.3f} ({(info['z_in'] - floor) * 100:.1f} cm above the glass): "
            f"allowed {(h['z_min'] - floor) * 100:.1f}..{(h['z_max'] - floor) * 100:.1f} cm, the {h['top']} closest to "
            f"the ceiling; {info['reach_in'] * 100:.0f} cm straight in along {np.round(info['axis'], 3).tolist()}; "
            f"adjust turns the hand {info['adjust_deg']:.1f} deg"
        )
        clearance = []
        for leg, r in view.legs.items():
            c = dict(leg=leg, ok=r["ok"], samples=r["samples"], pair=r["pair"])
            c["min_m"] = None if r["min_m"] is None else round(r["min_m"], 4)
            c["need_m"] = None if r["need_m"] is None else round(r["need_m"], 4)
            clearance.append(c)
            log(f"{leg:9s} {r['samples']:5d} configurations checked, {_clearance_text(r)}")
        return {
            "plan": dict(
                pre=np.round(info["pre"], 3).tolist(),
                end=np.round(info["end"], 3).tolist(),
                z_in=round(info["z_in"], 3),
                height=dict(z_min=round(h["z_min"], 3), z_max=round(h["z_max"], 3), top=h["top"]),
                reach_in=round(info["reach_in"], 3),
                adjust_deg=round(info["adjust_deg"], 1),
            ),
            "clearance": clearance,
            "markers": PLAN_TOPIC,
        }

    def _show(self, view):
        """Publish `view` on PLAN_TOPIC (latched)."""
        self._plan_pub.publish(_markers(view, self.get_clock().now().to_msg()))


def _clearance_text(r):
    if r["min_m"] is None:
        text = f"nothing within {place.QUERY_M * 100:.0f} cm"
    else:
        text = f"tightest {r['min_m'] * 100:.1f} cm ({r['pair']}, needs {r['need_m'] * 100:.1f})"
    return text if r["ok"] else f"FAILED: {r['problem']}"


def _color(rgb, a=1.0):
    return ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(a))


def _point(p):
    return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))


def _markers(view, stamp):
    """The MarkerArray of a place.View, in arm_mount: namespaces oven, voxels, target, path, clearance,
    closest, container."""
    out = [Marker(action=Marker.DELETEALL)]  # what an earlier place showed
    out[0].header.frame_id, out[0].header.stamp = rammp_arm.FRAME, stamp

    def marker(ns, kind, color, scale, centre=(0.0, 0.0, 0.0), R=None):
        m = Marker(ns=ns, id=len(out), type=kind, action=Marker.ADD, color=color)
        m.header.frame_id, m.header.stamp = rammp_arm.FRAME, stamp
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
        m.pose.position = _point(centre)
        x, y, z, w = Rotation.from_matrix(np.eye(3) if R is None else R).as_quat()
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = x, y, z, w
        out.append(m)
        return m

    for name, centre, R, half in view.oven.world_boxes():
        rgb = (0.55, 0.45, 0.35) if name == place.FLOOR_BOX else (0.7, 0.7, 0.75)
        marker("oven", Marker.CUBE, _color(rgb, 0.35), 2 * half, centre, R)
    if len(view.voxels):
        m = marker("voxels", Marker.CUBE_LIST, _color((1.0, 0.5, 0.0), 0.6), [view.voxel_m] * 3)
        m.points = [_point(p) for p in view.voxels]
    marker("target", Marker.SPHERE, _color((0.0, 1.0, 0.0)), [0.02] * 3, view.oven.spot)
    m = marker("target", Marker.ARROW, _color((0.0, 1.0, 0.0)), [0.008, 0.016, 0.02])
    m.points = [_point(view.oven.spot - 0.15 * view.oven.n), _point(view.oven.spot)]
    for leg, r in view.legs.items():
        rgb = LEG_COLORS.get(leg.rstrip("'"), (1.0, 1.0, 1.0)) if r["ok"] else (1.0, 0.0, 0.0)
        if len(r["tool"]) >= 2:
            m = marker("path", Marker.LINE_STRIP, _color(rgb), [0.004, 0.0, 0.0])
            m.points = [_point(p) for p in r["tool"]]
        m = marker("clearance", Marker.TEXT_VIEW_FACING, _color(rgb), [0.0, 0.0, 0.02], r["tool"][-1] + [0, 0, 0.03])
        m.text = f"{leg}: {_clearance_text(r)}"
        if r["between"] is not None:
            m = marker("closest", Marker.LINE_LIST, _color((1.0, 0.0, 0.0) if not r["ok"] else rgb), [0.003, 0, 0])
            m.points = [_point(r["between"][0]), _point(r["between"][1])]
    for label, (centre, R) in view.containers.items():
        d = 2 * place.LIP_RADIUS
        marker("container", Marker.CYLINDER, _color((0.4, 0.7, 1.0), 0.5), [d, d, place.BOWL_DEPTH], centre, R)
        m = marker("container", Marker.TEXT_VIEW_FACING, _color((0.4, 0.7, 1.0)), [0, 0, 0.015], centre + [0, 0, 0.07])
        m.text = label
    return MarkerArray(markers=out)


def main():
    run(Capability)


if __name__ == "__main__":
    main()
