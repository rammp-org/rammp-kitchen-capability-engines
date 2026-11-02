"""The microwave_place capability: put the held container into the open microwave.

  microwave_detection.py   where the microwave is, from the scene camera (SAM3 masks + depth)
  place_motion.py          planning, checking and running the arm (Walter Hsieh's place_container motion,
                           with the transit, the straight lines and the release the microwave needs)
  microwave_place_node.py  this action server: parameters, the actions, the plan and state files

Everything is in arm_mount (x forward, y left, z up).

Actions:
  place    the tupperware, gripped by its side (after grab_container), into the open microwave:
             1. checks: the arm may be commanded, the gripper is closed, the tool frame matches the model;
             2. locate (microwave_detection.locate): SAM3's cavity mask and the opening's known size; up to
                `max_frames` frames, `frames` of them within agree_m / agree_deg of their median;
             3. obstacles (place_motion.microwave_obstacles): the oven as a shell of boxes around its cavity
                (its outside from SAM3's "microwave" depth) and what stands in front of it (the open door) as
                depth columns; cuRobo's copy has the shell's inner faces pulled back PAD_COMP (cuRobo pads
                2 cm, the gaps inside are 1-2 cm);
             4. heading (place_motion.search_heading): each of heading_offsets_deg in turn (the hand turned
                that far off the oven's axis); the first that passes every stage wins: the target and the
                pre-insert pose reachable, the straight line in clear, cuRobo's level transit to the
                pre-insert pose (refused if it swings any joint more than TRANSIT_TRAVEL_MAX_DEG), then
                the check of everything on the arm model: the transit and the line in clear_min from every
                true box, the tilt within tilt_max_deg, the line in ending on the target, the line down,
                the release lift and the back-out with the fingers opened to RELEASE_OPEN.
                A dry run stops here and returns the plan (also written to state_dir/plans);
             5. with execute (place_motion.run_place): the transit as checked (transit_speed x cuRobo's
                timing), the line in again from where the arm is, checked again (insert_speed), the
                impedance lowering until the glass stops it, position control, the release lift, the
                gripper opened to RELEASE_OPEN, and the line in's joints reversed to the pre-insert pose.
  retreat  the release lift and back-out alone, after a place stopped with the gripper open: retraces the
           last executed line in, the fingers checked as they are.
  reset    back to the start position (where the last place started, or `save`d): straight up, then a joint
           move; checked against the last plan's obstacles. Gripper closed after.

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets `execute`.
Otherwise every goal is a dry run that locates, plans and checks, and returns the plan. Someone stands at
the e-stop whenever allow_execute is on. Any check that fails refuses with the reason; nothing unchecked is
sent.

    ros2 launch microwave_place microwave_place.launch.py use_orchestrator:=false allow_execute:=true
    make action CAP=microwave_place ACTION=place PARAMS='{"execute": false}'
"""

import json
import time
from pathlib import Path

import numpy as np
import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run

from . import microwave_detection, place_motion

# ROS parameter -> default.
PARAMS = {
    "allow_execute": False,
    # flange face to the closed fingertips (m): sets where the container centre is (model.reach)
    "flange_to_tip": place_motion.FLANGE_TO_TIP,
    # the hold the model checks a carried container against (reset's tilt check)
    "pitch_deg": place_motion.PLACE_PITCH_DEG,
    # contact: vz (m/s, + up) back up to this after moving down; 0.0 came 4.5 s late on the flatbot
    "vz_stop": -0.01,
    # the hold after contact, this far (m) below the contact point, while the gripper opens
    "hold_press": 0.003,
    # held in impedance at the start of the lowering this long (s); must end still
    "settle_s": 3.0,
    # contact more than this (m) above where the box's bottom meets the glass = not the glass
    "early_stop": 0.03,
    # joint impedance for the settle, lowering, hold, back-out and reset (one set: the stiffness never
    # changes mid-place). Default: the driver's documented starting point (rammp_arm.IMPEDANCE)
    "impedance_kq": list(rammp_arm.IMPEDANCE["kq"]),
    "impedance_zeta": rammp_arm.IMPEDANCE["zeta"],
    "impedance_torque_limit": list(rammp_arm.IMPEDANCE["torque_limit"]),
    # the hand's speed where the glass should be (m/s): what the contact test was tuned at
    "lower_speed": 0.02,
    # speed_scale of reset's joint moves (move_joints)
    "reset_speed_scale": 1.0,
    # the arm's tool frame vs the model's (m): further apart = wrong model or tool configuration
    "tool_frame_tol": 0.015,
    # the plans (plans/, last_plan.json), the start position and the descent recordings (descents/)
    "state_dir": "/config/microwave_place",
    # locate: the scene camera's depth and intrinsics (the 3D detections are not used)
    "depth_topic": "/scene_camera/depth/image_rect_raw",
    "camera_info_topic": "/scene_camera/color/camera_info",
    "detections_topic": "/perception/object_detections_3d",
    # up to max_frames detected frames within timeout_s; at least `frames` of them within agree_m / agree_deg
    # of their median
    "frames": 3,
    "max_frames": 5,
    "timeout_s": 20.0,
    "agree_m": 0.01,
    "agree_deg": 2.0,
    # SAM3's prompt for the whole oven (its outside, for the obstacles)
    "microwave_prompt": "microwave",
    # the tool this far (m) over the turntable glass at the end of the line in (the box's bottom
    # place_motion.BOX_ABOVE lower), then lowered from there
    "above": 0.08,
    # the hand's heading off the oven's axis (deg, + = left), tried in this order; the first that passes wins
    "heading_offsets_deg": [0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0, 25.0, -25.0, 30.0, -30.0]
    + [35.0, -35.0, 40.0, -40.0],
    # every arm and gripper link at least this far (m) from every true obstacle box
    "clear_min": 0.01,
    # the hand's tilt vs the start, anywhere on the transit and the lines (deg)
    "tilt_max_deg": 2.0,
    # also keep the held box clear_min from the obstacles
    "check_container": False,
    # cuRobo's world files go here; the planner's container must see this directory at the SAME path
    "worlds_dir": "/home/snorlax/.ros/rammp_box_opening/worlds",
    # the transit at this fraction of cuRobo's timing
    "transit_speed": 1.0,
    # the hand's speed (m/s; position control) on the line in, the release lift and the back-out
    "insert_speed": 0.16,
}
EXECUTE = {"type": "object", "properties": {"execute": {"type": "boolean", "default": False}}}
PLACE = {
    "type": "object",
    "properties": {
        **EXECUTE["properties"],
        # an empty-gripper test run: don't refuse an open gripper
        "allow_empty_gripper": {"type": "boolean", "default": False},
    },
}
RESET = {
    "type": "object",
    "properties": {
        **EXECUTE["properties"],
        "save": {"type": "boolean", "default": False},
        # nothing in the (closed) gripper: skip the container-tilt check on the way back
        "empty": {"type": "boolean", "default": False},
    },
}


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
        Action("retreat", "Back the hand straight out of the oven", params_schema=EXECUTE),
        Action("reset", "Back to the start position", params_schema=RESET),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("microwave_place", **node_kwargs)
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        self.cfg["gains"] = {
            "kq": [float(v) for v in self.cfg["impedance_kq"]],
            "zeta": float(self.cfg["impedance_zeta"]),
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
            model = place_motion.Model(self.cfg["flange_to_tip"], self.cfg["pitch_deg"])
            if action_id == "place":
                return {**self._place(model, params, execute, ctx, log), "note": note}
            if action_id == "retreat":
                insert = self._load(
                    "last_insert.json", "no line in saved: the back-out retraces the last executed line in"
                )
                return {**place_motion.retreat(self.arm, model, insert, self.cfg, execute, ctx, log), "note": note}
            return {**self._reset(model, params, execute, log), "note": note}
        except (place_motion.Refused, microwave_detection.NotLocated) as e:
            raise EngineError("refused", str(e), recoverable=True) from e

    def _place(self, model, params, execute, ctx, log):
        c = self.cfg
        # 1. checks
        place_motion.ready(self.arm)
        q0 = np.asarray(self.arm.raw_joints(), dtype=float)
        place_motion.check_tool_frame(self.arm, model, q0, c["tool_frame_tol"], log)
        gripper = float(self.arm.gripper_state().position)
        if gripper < 0.05 and not params.get("allow_empty_gripper", False):
            raise place_motion.Refused("the gripper is fully open -- it isn't holding the container")

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
            max_frames=int(c["max_frames"]),
            camera=camera,
        )
        oven = place_motion.Oven(loc["fit"])
        log(
            f"microwave: point {np.round(oven.point, 3).tolist()}, axis {np.round(oven.n[:2], 3).tolist()}, glass z "
            f"{oven.floor_z:.3f} (the frame nearest the median of {loc['frames']} of {loc['detected']}, "
            f"within {loc['spread_m'] * 100:.1f} cm, {loc['spread_deg']:.1f} deg)"
        )

        # 3. obstacles
        ctx.progress("obstacles", 0.0)
        shot = microwave_detection.scene_frame(camera, c["microwave_prompt"], c["timeout_s"])
        boxes, curobo_boxes, world_info = place_motion.microwave_obstacles(
            model, q0, oven, shot["depth_m"], shot["K"], shot["world_T_cam"], shot["mask"]
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        world = self._write_world(curobo_boxes, stamp)
        ok, msg = self.arm.set_world(str(world))
        log(
            f"world: {len(boxes)} boxes ({world_info['columns']} depth columns, "
            f"{world_info['cell'] * 100:.0f} cm) -> {world}"
        )
        if not ok:
            raise place_motion.Refused(f"cuRobo's set_world({world}) failed: {msg}")

        checker = place_motion.Checker(model, boxes)
        try:
            # 4-5. the first heading that passes every stage
            ctx.progress("plan", 0.0)
            cand, plan, res, attempts = place_motion.search_heading(self.arm, model, checker, q0, oven, c, log)
            summary, record = place_motion.summarize(
                model, q0, loc, oven, world, world_info, boxes, cand, plan, res, attempts, c
            )
            path = self._save_plan(record, stamp)
            summary["plan_file"] = str(path)
            log(
                f"plan: heading {cand['offset_deg']:+.0f} deg, clearance "
                + ", ".join(f"{k} {v[0]:.1f} cm" for k, v in summary["clearance_cm"].items())
                + f", tilt {summary['max_tilt_deg']:.1f} deg, transit {summary['duration_s']['transit']:.0f} s, "
                f"line in {cand['insert_m'] * 100:.0f} cm in {summary['duration_s']['insert']:.0f} s -> {path}"
            )
            if not execute:
                log("DRY RUN -- all checks passed, nothing commanded")
                return {"executed": False, "plan": summary}
            # 6. execute
            result = place_motion.run_place(
                self.arm,
                model,
                checker,
                boxes,
                q0,
                oven,
                cand,
                plan,
                c,
                ctx,
                log,
                self._save,
                self.state_dir / "descents",
            )
            return {**result, "plan": summary}
        finally:
            checker.close()

    def _reset(self, model, params, execute, log):
        if params.get("save", False):
            q = np.asarray(self.arm.raw_joints(), dtype=float)
            self._save("start.json", dict(q=[float(v) for v in q], time=time.time()))
            T = model.fk(q)
            log(f"saved the start position: hand at x={T[0, 3]:.3f} y={T[1, 3]:.3f} z={T[2, 3]:.3f}")
            return {"saved": True}
        start = self._load("start.json", "no start position saved: run reset with save, or a place first")
        plan = self._load("last_plan.json", "no place plan saved: reset is checked against the last plan's obstacles")
        # Nothing in the hand (the gripper is open, or the goal says so): the way back may turn the hand.
        is_open = float(self.arm.gripper_state().position) <= place_motion.OPEN_STALLED
        empty = is_open or bool(params.get("empty", False))
        q_now = np.asarray(self.arm.raw_joints(), dtype=float)
        moves = place_motion.reset_moves(model, q_now, np.asarray(start["q"], dtype=float), empty=empty)
        place_motion.check_against(
            model,
            plan["boxes"],
            [("reset", [q_now] + [q for _, q in moves], place_motion.TRANSIT_STEP_DEG)],
            self.cfg["clear_min"],
            closed=not is_open,
        )
        steps = [dict(name=n, tool=np.round(model.fk(q)[:3, 3], 3).tolist()) for n, q in moves]
        if not execute:
            return {"executed": False, "plan": steps}
        place_motion.ready(self.arm)
        place_motion.joint_moves(self.arm, model, moves, self.cfg, log)
        self.arm.gripper(1.0)  # ready for the next run
        log("Back at the start position.")
        return {"executed": True, "plan": steps}

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

    def _load(self, name, missing):
        path = self.state_dir / name
        if not path.exists():
            raise place_motion.Refused(missing)
        return json.loads(path.read_text())


def main():
    run(Capability)


if __name__ == "__main__":
    main()
