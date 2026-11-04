"""The microwave_place capability: put the held container into the open microwave, and take it out again.

  microwave_detection.py   where the microwave is, from the scene camera (SAM3 masks + depth)
  place_motion.py          planning, checking and running the arm (Walter Hsieh's place_container motion,
                           with the transit, the straight lines and the release the microwave needs)
  rim_detection.py         take_out: where the container's rim is, from the wrist camera (grab_container's SAM3
                           rim detection)
  take_out.py              take_out: planning and running the arm
  microwave_place_node.py  this action server: parameters, the actions, the plan and state files

Everything is in arm_mount (x forward, y left, z up).

Action:
  place    the tupperware, gripped by its side (after grab_container), into the open microwave:
             1. locate (microwave_detection.locate): the opening's known size fitted to SAM3's cavity mask,
                frame after frame until `frames` consecutive detected frames agree (agree_m / agree_deg);
             2. obstacle (place_motion.microwave_obstacles): the microwave as one box, from the depth of SAM3's
                "microwave" mask; only the first motion (side shift, transit to in front of the door) avoids it,
                nothing after it is collision-checked;
             3. plan (place_motion.search_place): for each of side_shifts_cm (the hand moved level to its
                left first) and each of heading_offsets_deg: the side shift, the target and pre-insert pose
                reachable, the straight line in, cuRobo's level transit to the pre-insert pose (refused if it
                swings any joint more than TRANSIT_TRAVEL_MAX_DEG), then on the arm model: the side shift and
                the transit clear_min from the microwave box, the tilt within tilt_max_deg everywhere, the line
                in ending on the target, the line down, the release tilt and the back-out planned (none of
                them collision-checked). The first that passes wins. A dry run stops here and returns the plan
                (also written to state_dir/plans);
             4. with execute (place_motion.run_place): the side shift, cuRobo's transit and the line in as ONE
                motion (insert_speed peak), the impedance lowering until the glass stops it, the gripper opened
                to RELEASE_OPEN in place (still in impedance), the hand pitched RELEASE_TILT_DEG nose-down
                (impedance) so the container slides off, a stop, then position control and the level line back
                out, the hand kept tilted, to above the pre-insert pose.

  take_out the container out of the open microwave, from the take-out start pose (the wrist camera looking at
             it, the gripper open; take_out.py has the steps): the rim's nearest point from the wrist camera
             (rim_detection), the hand turned 90 deg and reached straight in along its heading to `grasp_depth` /
             `grasp_up` from that point (position control, the last `impedance_in` in joint impedance), the gripper
             closed in position control, a `take_lift` raise and the level line back out. Planned on the arm model
             first; a dry run stops there and returns the plan (also written to state_dir/plans).

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets `execute`.
Otherwise every goal is a dry run that locates, plans and checks, and returns the plan. Someone stands at
the e-stop whenever allow_execute is on. Only the first motion (side shift, transit) is checked against the
microwave; from the line in on nothing is collision-checked. Any check that fails refuses with the reason.

    ros2 launch microwave_place microwave_place.launch.py use_orchestrator:=false allow_execute:=true
    make action CAP=microwave_place ACTION=place PARAMS='{"execute": false}'
"""

import json
import time
from pathlib import Path

import numpy as np
import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run

from . import microwave_detection, place_motion, rim_detection, take_out

# ROS parameter -> default.
PARAMS = {
    "allow_execute": False,
    # flange face to the closed fingertips (m)
    "flange_to_tip": place_motion.FLANGE_TO_TIP,
    # contact: vz (m/s, + up) back up to this after moving down; 0.0 came 4.5 s late on the flatbot
    "vz_stop": -0.01,
    # the hold after contact, this far (m) below the contact point, while the gripper opens
    "hold_press": 0.003,
    # held in impedance at the start of the lowering this long (s); must end still
    "settle_s": 0.2,
    # contact more than this (m) above where the box's bottom meets the glass = not the glass
    "early_stop": 0.03,
    # joint impedance for the settle, lowering, hold, gripper opening and release tilt (one set: the stiffness never
    # changes mid-place). Default: the driver's documented starting point (rammp_arm.IMPEDANCE)
    "impedance_kq": list(rammp_arm.IMPEDANCE["kq"]),
    "impedance_zeta": rammp_arm.IMPEDANCE["zeta"],
    "impedance_torque_limit": list(rammp_arm.IMPEDANCE["torque_limit"]),
    # the hand's speed where the glass should be (m/s): what the contact test was tuned at
    "lower_speed": 0.12,
    # the plans (plans/, last_plan.json) and the descent recordings (descents/)
    "state_dir": "/config/microwave_place",
    # locate: the scene camera's depth and intrinsics (the 3D detections are not used)
    "depth_topic": "/scene_camera/depth/image_rect_raw",
    "camera_info_topic": "/scene_camera/color/camera_info",
    "detections_topic": "/perception/object_detections_3d",
    # detect continuously until `frames` consecutive detected frames lie within agree_m / agree_deg of their
    # median (at most timeout_s)
    "frames": 3,
    "timeout_s": 20.0,
    "agree_m": 0.01,
    "agree_deg": 2.0,
    # SAM3's prompt for the whole oven (its outside, for the obstacles)
    "microwave_prompt": "microwave",
    # the tool this far (m) over the turntable glass at the end of the line in (the box's bottom
    # place_motion.BOX_ABOVE lower), then lowered from there
    "above": 0.055,
    # the container's spot this far (m) back from the turntable's middle toward the opening, along the oven's axis
    "closer": 0.02,
    # the hand's heading off the oven's axis (deg, + = left), tried in this order; the first that passes wins
    "heading_offsets_deg": [0.0],
    # before the transit, the hand moved this far (cm) level to its left (- = right), tried in this order
    # (each with every heading); the first that passes wins
    "side_shifts_cm": [3.0, -3.0, 6.0, -6.0, 9.0, -9.0],
    # the side shift and the transit: every arm and gripper link at least this far (m) from the microwave box
    "clear_min": 0.01,
    # the hand's tilt vs the start, anywhere on the transit and the lines (deg)
    "tilt_max_deg": 2.0,
    # cuRobo's world files go here; the planner's container must see this directory at the SAME path
    "worlds_dir": "/home/snorlax/.ros/rammp_box_opening/worlds",
    # the hand's peak speed (m/s; position control, min-jerk: about half of it on average) on the approach
    # (side shift + transit + line in, one motion) and the back-out
    "insert_speed": 0.16,
    # take_out: the wrist camera (colour, aligned depth, camera_info); the rim is found on it (rim_detection)
    "wrist_topics": [
        "/wrist_camera/color/image_raw",
        "/wrist_camera/depth/image_rect_raw",
        "/wrist_camera/color/camera_info",
    ],
    # SAM3's prompts: the whole container with its sleeve, then its rim on the same frame
    "container_prompt": "food container together with its blue sleeve",
    "rim_prompt": "wide rim of blue sleeve",
    "sam3_threshold": 0.5,
    # the rim: this many consecutive frames whose container and rim masks overlap (IoU) at least rim_mask_iou
    "rim_frames": 3,
    "rim_mask_iou": 0.85,
    "rim_timeout_s": 60.0,
    # the grasp: the tool frame this far (m) past the rim point along the hand's heading (0: on it)
    "grasp_depth": -0.05,
    # ... and this far (m) above it: the tool's height for the pinch over the detected rim point (2026-11-03: the hand
    # placed by hand at the right height, tool z 0.332, over the rim the wrist camera found at z 0.302; +0.5 cm
    # after the first runs)
    "grasp_up": 0.035,
    # the pre-grasp pose this far (m) back from the grasp along the hand's level heading: the line in's length
    "reach_in": 0.25,
    # the gripper's opening for the reach in (0 = fully open .. 1 closed)
    "grip_open": 0.0,
    # the last this far (m) of the line in, in joint impedance (a short stop to switch); position control before it, and
    # from the gripper's close on
    "impedance_in": 0.10,
    # take_out's joint impedance: the driver's cap (kq <= 400); the place's (impedance_*, kq 80/30) let the arm sag
    # under its own weight (2026-11-03). The torque limits are the place's (impedance_torque_limit)
    "take_impedance_kq": [400.0, 400.0, 400.0, 400.0, 200.0, 200.0, 200.0],
    "take_impedance_zeta": 0.7,
    # the hand's peak speed (m/s) on that impedance part of the line in: half the approach's insert_speed
    "impedance_speed": 0.08,
    # after closing: straight up this far (m), then the place's level retract
    "take_lift": 0.03,
}
EXECUTE = {"type": "object", "properties": {"execute": {"type": "boolean", "default": False}}}
PLACE = EXECUTE


def _json_default(o):
    if isinstance(o, np.ndarray | np.generic):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


class Capability(CapabilityEngine):
    display_name = "Place in microwave"
    perception_classes = []
    required_resources = ["arm", "gripper"]
    actions = [
        Action("place", "Place the held container in the open microwave", params_schema=PLACE),
        Action("take_out", "Take the container out of the open microwave", params_schema=EXECUTE),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("microwave_place", **node_kwargs)
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        self.cfg["gains"] = {
            "kq": [float(v) for v in self.cfg["impedance_kq"]],
            "zeta": float(self.cfg["impedance_zeta"]),
            "torque_limit": [float(v) for v in self.cfg["impedance_torque_limit"]],
        }
        self.cfg["take_gains"] = {
            "kq": [float(v) for v in self.cfg["take_impedance_kq"]],
            "zeta": float(self.cfg["take_impedance_zeta"]),
            "torque_limit": [float(v) for v in self.cfg["impedance_torque_limit"]],
        }
        self.state_dir = Path(self.cfg["state_dir"])
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def execute(self, action_id, params, ctx):
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing planned or moved")
            return {"executed": False, "mock": True}
        log = self.get_logger().info
        try:
            model = place_motion.Model(self.cfg["flange_to_tip"])
            run_action = self._take_out if action_id == "take_out" else self._place
            return {**run_action(model, execute, ctx, log), "note": note}
        except (place_motion.Refused, microwave_detection.NotLocated, rim_detection.NotDetected) as e:
            raise EngineError("refused", str(e), recoverable=True) from e

    def _place(self, model, execute, ctx, log):
        c = self.cfg
        q0 = np.asarray(self.arm.raw_joints(), dtype=float)
        oven, loc, boxes, world, world_info, stamp = self._oven_world(ctx, log)
        checker = place_motion.Checker(model, boxes)
        try:
            # 4-5. the first heading that passes every stage
            ctx.progress("plan", 0.0)
            cand, plan, res, attempts = place_motion.search_place(self.arm, model, checker, q0, oven, c, log)
            summary, record = place_motion.summarize(
                model, q0, loc, oven, world, world_info, boxes, cand, plan, res, attempts, c
            )
            path = self._save_plan(record, stamp)
            summary["plan_file"] = str(path)
            log(
                f"plan: side shift {cand['shift_cm']:+.0f} cm, heading {cand['offset_deg']:+.0f} deg, clearance "
                + ", ".join(f"{k} {v[0]:.1f} cm" for k, v in summary["clearance_cm"].items())
                + f", tilt {summary['max_tilt_deg']:.1f} deg, approach (shift + transit + line in "
                f"{cand['insert_m'] * 100:.0f} cm) in {summary['duration_s']['approach']:.1f} s -> {path}"
            )
            if not execute:
                log("DRY RUN -- all checks passed, nothing commanded")
                return {"executed": False, "plan": summary}
            # 6. execute
            result = place_motion.run_place(
                self.arm,
                model,
                q0,
                oven,
                cand,
                plan,
                c,
                ctx,
                log,
                self.state_dir / "descents",
                res,
            )
            return {**result, "plan": summary}
        finally:
            checker.close()

    def _take_out(self, model, execute, ctx, log):
        """take_out (take_out.py): the rim from the wrist camera, then turn + reach in, close, raise, retract."""
        c = self.cfg
        q0 = np.asarray(self.arm.raw_joints(), dtype=float)
        ctx.progress("detect", 0.0)
        wrist = rim_detection.WristCamera(self, rammp_arm.FRAME, list(c["wrist_topics"]))
        rim = rim_detection.detect_rim(
            wrist,
            c["container_prompt"],
            c["rim_prompt"],
            c["sam3_threshold"],
            int(c["rim_frames"]),
            c["rim_mask_iou"],
            c["rim_timeout_s"],
            log,
            ctx.checkpoint,
        )
        cand = take_out.grasp_pose(model, q0, rim["point"], c["reach_in"], c["grasp_depth"], c["grasp_up"])
        ctx.progress("plan", 0.0)
        legs = take_out.plan_take_out(model, q0, cand, c)
        T_pre = model.fk(legs["turn"][-1])
        summary = dict(
            rim_point=np.round(rim["point"], 3).tolist(),
            rim_depths_m=rim["info"]["depths_m"],
            rim_frame_points=rim["info"]["frame_points"],
            rim_frames_spread_cm=round(rim["info"]["frames_spread_m"] * 100, 1),
            rim_mask_iou_min=round(rim["iou_min"], 3),
            start_tool=np.round(model.fk(q0)[:3, 3], 3).tolist(),
            pre_grasp=np.round(cand["pre"], 3).tolist(),
            grasp=np.round(cand["target"], 3).tolist(),
            heading=np.round(cand["heading"], 3).tolist(),
            fingers_line_from_up_deg=round(float(np.degrees(np.arccos(np.clip(T_pre[2, 0], -1.0, 1.0)))), 1),
            hand_pitch_deg=round(float(np.degrees(np.arcsin(-T_pre[2, 2]))), 1),
            joint_change_deg=dict(
                turn=np.round(np.degrees(place_motion.joint_delta(legs["turn"][0], legs["turn"][-1])), 1).tolist(),
                line_in=np.round(np.degrees(place_motion.joint_delta(legs["line"][0], legs["line"][-1])), 1).tolist(),
            ),
            end_off_cm=round(legs["end"][0] * 100, 1),
            lift_cm=round(c["take_lift"] * 100, 1),
            out_end=np.round(model.fk(legs["out"][-1])[:3, 3], 3).tolist(),
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._save(f"plans/take_out_{stamp}.json", dict(summary=summary, q0=q0, **legs))
        log(f"take-out plan: {json.dumps(summary)}")
        if not execute:
            log("DRY RUN -- planned, nothing commanded")
            return {"executed": False, "plan": summary}
        return {**take_out.run_take_out(self.arm, model, q0, cand, legs, c, ctx, log), "plan": summary}

    def _oven_world(self, ctx, log):
        """locate the oven (scene camera), its one obstacle box, cuRobo's world set: (oven, loc, boxes, world file,
        world info, stamp)."""
        c = self.cfg

        # 2. locate
        ctx.progress("locate", 0.0)
        topics = (c["depth_topic"], c["camera_info_topic"], c["detections_topic"])
        camera = microwave_detection.SceneCamera(self, rammp_arm.FRAME, topics)
        loc = microwave_detection.locate(
            self,
            rammp_arm.FRAME,
            topics,
            int(c["frames"]),
            c["timeout_s"],
            c["agree_m"],
            c["agree_deg"],
            log,
            camera=camera,
            extra_prompt=c["microwave_prompt"],
        )
        oven = place_motion.Oven(loc["fit"])
        log(
            f"microwave: point {np.round(oven.point, 3).tolist()}, axis {np.round(oven.n[:2], 3).tolist()}, glass z "
            f"{oven.floor_z:.3f} (the frame nearest the median of {loc['frames']} of {loc['detected']}, "
            f"within {loc['spread_m'] * 100:.1f} cm, {loc['spread_deg']:.1f} deg)"
        )

        # 3. obstacles
        ctx.progress("obstacles", 0.0)
        shot = loc["body"]  # the locate's frame of SAM3's "microwave" (the oven's body)
        boxes, curobo_boxes, world_info = place_motion.microwave_obstacles(
            oven, shot["depth_m"], shot["K"], shot["world_T_cam"], shot["mask"]
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        world = self._write_world(curobo_boxes, stamp)
        ok, msg = self.arm.set_world(str(world))
        log(
            f"world: the microwave as one box from {world_info['points']} depth points, in oven coordinates (cm) "
            f"along {world_info['along_cm']}, across {world_info['across_cm']}, up {world_info['up_cm']} -> {world}"
        )
        if not ok:
            raise place_motion.Refused(f"cuRobo's set_world({world}) failed: {msg}")

        return oven, loc, boxes, world, world_info, stamp

    # -- state files -----------------------------------------------------------------------------

    def _write_world(self, boxes, stamp):
        """cuRobo's world file in worlds_dir (the planner reads it at this same absolute path)."""
        directory = Path(self.cfg["worlds_dir"])
        if not directory.is_absolute():
            raise place_motion.Refused(
                f"worlds_dir {directory} is not an absolute path: the planner reads the file by its path"
            )
        path = directory / f"microwave_place_{stamp}.yaml"
        try:
            place_motion.write_world(path, boxes, rammp_arm.FRAME)
        except OSError as e:
            raise place_motion.Refused(
                f"can't write cuRobo's world {path} ({e}) -- set worlds_dir, mounted in both containers"
            ) from e
        return path

    def _save_plan(self, record, stamp):
        path = self.state_dir / "plans" / f"plan_{stamp}.json"
        self._save(f"plans/plan_{stamp}.json", record)
        self._save("last_plan.json", record)
        return path

    def _save(self, name, record):
        try:
            path = self.state_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1, default=_json_default))
        except OSError as e:
            raise place_motion.Refused(f"can't save {name} in {self.state_dir} ({e}) -- set state_dir") from e


def main():
    run(Capability)


if __name__ == "__main__":
    main()
