"""The place_container capability: put the held food container down at a spot on the table.

Ported from RAMMP-table (place_bowl.py + kinova_arm.py), where it placed the container on a marked
sticky note on the flatbot (2026-10-26). Everything is in arm_mount (x forward, y left, z true up).

Actions:
  place_here  the UI's tap on a `table`: tap-to-place. The scene camera's view of the table gets a grid of
           free spots (spots.py: the segmenter's table mask or a plane fit to depth, the reach, the table's
           edge, what is on it); the user taps one and presses Place here (or Use marked spot), and it is
           placed for real there and ends back at the start (capability_ui.RunUI: Stop on every panel).
           Every action shows its phases and any error on the UI panel and is recorded (a rosbag + run.json,
           bag.py).
  mark     the closed fingertips touching the target -> saved as the spot (its z = the table). No motion.
  place    the container, gripped by its lip, to the spot ({x, y, table_z}, or the marked one):
             1. adjust: square the hand in place to the nearest hold (level, pointing `pitch_deg` down);
             2. lift to SAFE_HEIGHT above the table, if lower; 3. transit to above the spot; 3b. the
             approach straight down to `lower_from` above where the container meets the table
             -- 1-3 are ONE go_to_joint_configs motion through our planned joints (cuRobo, around the
             chair's collision volumes, blended at the waypoints), stopping above the spot; 3b runs
             right after on move_joints in position mode, through the planned straight line's joints
             (cuRobo planned each 4 cm of it on its own and swung the hand out and back on every one,
             plan.split_carry); `carry_targets` pose: go_to_ee_poses for 1-3b; `blend` false: separate
             moves as before;
             4. the rest of the straight line down in the driver's IMPEDANCE mode
             until contact, then hold `hold_press` below it (lower.py); 5. open, drop `release_drop`;
             6. back out: straight back, or a short way back and then a turn of the hand that swings
             the fingertips off the lip (plan.plan_swing).
           The arm is left in impedance, held where the retreat ended.
  retreat  only step 6 (after a place stopped with the gripper open).
  reset    back to the start position (where the last place started, or `save`d): straight up,
           then to the start's joints, one cuRobo motion (go_to_joint_configs, around the chair's
           collision volumes); gripper closed.

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets `execute`.
Otherwise every goal is a dry run that plans and checks and returns the plan. Someone stands at the
e-stop whenever allow_execute is on.

    ros2 launch place_container place_container.launch.py use_orchestrator:=false allow_execute:=true
    make action CAP=place_container ACTION=mark
    make action CAP=place_container ACTION=place PARAMS='{"execute": false}'
"""

import contextlib
import json
import time
from pathlib import Path

import numpy as np
import rammp_arm
from capability_engine import Action, ArmError, Cancelled, CapabilityEngine, EngineError, run
from capability_engine.arm import HOLD_LEVEL, HOLD_NONE, SUCCESSFUL, JointTarget, PoseTarget
from capability_ui import RunUI, cancel_own_run
from rammp_arm import _unwrap_from
from scipy.spatial.transform import Rotation

from . import level_path, lower, plan, spots, ui
from .scene import SceneCamera

# The arm moved 8 in (0.2032 m) forward on the flatbot's chair (rammp-deployments #156, merged
# 2026-11-07 17:56 UTC). Everything here is in arm_mount, which moved with the arm: the arm's own
# geometry (flange_to_tip, pitch, reach as kinematics) still holds, but a spot or start position saved
# before then was saved relative to a chair and table that are now 0.2032 m nearer in x.
ARM_MOVED = 1794074180.0  # 2026-11-07T17:56:20Z
# deg: after the carry, the planned line down is used if the arm is within this of its first joints
# (lower() starts the line from the measured joints, so this much is a smooth start, not a jump)
REPLAN_DEG = 1.0

# ROS parameter -> default: the flatbot's values (2026-10-26, unless noted).
PARAMS = {
    "allow_execute": False,
    # flange face to the closed fingertips (m): sets where the container centre lands
    "flange_to_tip": plan.FLANGE_TO_TIP,
    # the hand points this many deg down while placing, so the container touches first
    "pitch_deg": plan.PLACE_PITCH_DEG,
    # contact: vz (m/s, + up) back up to this after moving down; 0.0 came 4.5 s late on the flatbot
    "vz_stop": -0.01,
    # the hold after contact, this far (m) below the contact point, while the gripper opens
    "hold_press": 0.003,
    # after the gripper opens, the hand goes this far (m) straight down before the retreat, which then
    # doesn't lift: the open fingers leave the lip downward (lifting first dragged the container,
    # 2026-11-10). 0.0 = off (the retreat lifts by hold_press as before)
    "release_drop": 0.01,
    # held in impedance at the start this long (s) before going down; must end still. 0 = no pause:
    # the arm goes compliant as the descent starts (contact_ignore_m covers its sag)
    "settle_s": 0.0,  # flatbot 2026-11-10: contact_ignore_m covers the sag; 3.0 added ~4 s
    # contact is ignored until the descent's command has gone this far down (m): the sag when the arm
    # goes compliant must not read as moving down and being stopped
    "contact_ignore_m": 0.08,  # 0.03 armed mid catch-up: a false contact (flatbot 2026-11-10)
    # contact more than this (m) above where the bottom meets the table = not the table
    "early_stop": 0.03,
    # joint impedance for the settle, lowering, hold, retreat and reset, one set so the stiffness never
    # changes mid-place (move_joints takes custom gains, not the driver's named profiles). Default: the
    # driver's documented starting point (rammp_arm.IMPEDANCE), NOT the "medium" profile RAMMP-table
    # used -- compare the settle's sag in the recording (medium: 1.0 cm on the flatbot, 2026-10-26)
    "impedance_kq": list(rammp_arm.IMPEDANCE["kq"]),
    "impedance_zeta": rammp_arm.IMPEDANCE["zeta"],
    "impedance_torque_limit": list(rammp_arm.IMPEDANCE["torque_limit"]),
    # the slow impedance lowering starts this far (m) above where the container's bottom meets the table;
    # the carrying motion takes the hand straight down to there first (position mode, fast). 0.0 = the
    # whole way down from the carrying height slow, as before (~14 s at the flatbot's spot, 2026-11-10)
    "lower_from": 0.06,
    # speeds. The lowering stays at the 2 cm/s its contact test was tuned at; the rest is 2x RAMMP-table's
    # (2026-10-29): the hand's speed where the table should be (m/s)
    "lower_speed": 0.02,
    # speed_scale of the carrying moves (go_to_ee_pose; was 0.2)
    "go_to_speed_scale": 0.6,  # flatbot 2026-11-10 (was 0.4)
    # speed_scale of the squaring turn with blend false (move_joints; was 0.5)
    "adjust_speed_scale": 1.0,
    # adjust -> lift -> transit as ONE go_to_ee_poses motion (cuRobo, blended at the waypoints, all at
    # go_to_speed_scale); false = separate moves as on 2026-10-29 (adjust on move_joints)
    "blend": True,
    # with blend: "pose" = the squaring turn is the motion's first target (unheld: it changes the tilt);
    # "joints" = a go_to_joint_config to our checked joints first (if cuRobo swings the hand for the
    # pose, as it did 2026-10-08), then lift -> transit blended
    "adjust_mode": "pose",
    # with blend: "level" = level_path.py: however the start holds the container is the reference; the
    # carry goes straight to above the spot (the lowest height that plans, the container's bottom >= 10 cm
    # over the table), the hand only turning about vertical, the elbow kept clear of the chair and user by
    # the arm's spare joint, every timed sample checked (tilt, clearance, height, limits) and exactly that
    # trajectory run; then straight down (flatbot 2026-11-12: a joint-space carry from a home-based start
    # flipped the arm and tipped the container 70-90 deg); "joints" = through OUR planned joints
    # (go_to_joint_configs, blended, cuRobo's path unchecked); "pose" = hand poses, cuRobo picks the joints
    # -- on 2026-11-10 it chose another posture at the spot (J1 +101, J3 -147 deg from the start)
    "carry_targets": "level",
    # the retreat's peak speed (m/s; was 0.03)
    "retreat_speed": 0.10,  # flatbot 2026-11-10 (was 0.06)
    # speed_scale of reset's joint moves with reset_mode legacy (move_joints; was ~0.5)
    "reset_speed_scale": 1.0,
    # reset: "curobo" = one go_to_joint_configs motion (up, then the start), planned around the chair's
    # collision volumes, at go_to_speed_scale; "legacy" = the impedance move_joints moves (no chair check)
    "reset_mode": "curobo",
    # blend_radius (m) where a blended motion passes a waypoint near the table or the container (the
    # table is not in cuRobo's world), and up at the carrying height
    "blend_radius_near": 0.03,
    "blend_radius_far": 0.05,
    # before lowering, ask cuRobo whether the ends of the moves it doesn't plan (the line down, the
    # retreat) are clear of the chair's collision volumes
    "curobo_check": True,
    # spot distance from the arm base axis (m) where the whole place plans with the full descent
    # (flatbot, table z 0.198: test_place_bowl.py --sweep, 2026-10-29). Arm kinematics only: the sweep
    # had no chair collision volumes, so re-measure it with the planner's. Outside it the plan decides
    "reach": [0.70, 1.10],
    # where the line down can't reach the table in the arm's posture, switch to another posture that can
    # (go_to_joint_config: a bigger move, cuRobo in joint space, no level hold); off = refuse instead
    "allow_reconfigure": False,
    # the squaring turn may correct at most this much (deg)
    "max_adjust_deg": plan.MAX_ADJUST_DEG,
    # a marked spot older than this (min) is refused: the chair may have moved
    "max_spot_age_min": 30.0,
    # the arm's tool frame vs the model's (m): further apart = wrong model or tool configuration
    "tool_frame_tol": 0.015,
    # where the spot, the start position and the descent recordings are kept
    "state_dir": "/config/place_container",
    # a spot or start position saved before this (unix s) is refused: the arm has moved on the chair
    # since (default: ARM_MOVED)
    "state_valid_after": ARM_MOVED,
    # a failure is shown on the user's UI panel with an OK button; the action waits this long (s) for
    # the OK before it ends with the error (the panel closes when the action ends). 0 = don't wait
    "ui_error_wait_s": 60.0,
    # after letting go: "blended" = drop, back out (and swing) and go back to the start as ONE
    # go_to_joint_configs motion in impedance (cuRobo, blended), so place ends at the start; "legacy"
    # = the separate drop / back_out moves of 2026-11-10, ending by the container (then reset)
    "return_mode": "legacy",
    # speed_scale of that motion's legs by the container (the drop and the retreat); the way up and
    # over to the start goes at go_to_speed_scale
    "return_speed_scale": 0.3,
    # a rosbag + run.json for every action run, in <state_dir>/runs/ (bag.py); false = none
    "record_bag": True,
    # arm state / scene camera frames per second in the bag
    "bag_state_hz": 25.0,
    "bag_frame_hz": 1.0,
    # the scene camera's colour, camera_info and depth in the bag (context; this capability uses no camera)
    "bag_camera": True,
    "bag_camera_topics": [
        "/scene_camera/color/image_raw",
        "/scene_camera/color/camera_info",
        "/scene_camera/depth/image_rect_raw",
    ],
    # mock only: place fails at this phase (carry, lower, open, return), to try the UI's error panel
    # from the UI (which sends no params); "" = none. A goal's own mock_fail wins
    "mock_fail": "",
    # place_here's tap-to-place (spots.py): the scene camera's depth (aligned to colour), its camera_info, and
    # the scene segmenter's detections (the table's mask; "" = the plane fit to depth only)
    "scene_depth_topic": "/scene_camera/depth/image_rect_raw",
    "scene_info_topic": "/scene_camera/color/camera_info",
    "detections_topic": "/perception/object_detections",
    # wait this long (s) for the scene camera's depth, camera_info and tf
    "scene_wait_s": 3.0,
    # the offered spots: this far apart (m); the table must reach this far past the container's footprint (m),
    # and nothing on it may be closer to the footprint than this (m)
    "spot_step": 0.09,
    "spot_edge_margin": 0.03,
    "spot_clearance": 0.03,
    # the table's height at the chosen spot (a plane fitted to the depth around it) is raised by this (m): the
    # descent stops on contact, so a table found too high only makes it stop later; too low would be pushed
    # into faster (the approach ends lower_from above it)
    "table_z_bias": 0.01,
    # a table that looks more tilted than this (deg) to the scene camera: its mount is off, no spots
    "max_table_tilt_deg": 3.0,
    # with a marked spot: the camera's table height there more than this (m) from the arm's mark = no spots
    "table_check_tol": 0.03,
    # where (m, arm_mount z) a table may be, for the plane fit without the segmenter's table mask
    "table_z_range": [-0.10, 0.50],
}
EXECUTE = {"type": "object", "properties": {"execute": {"type": "boolean", "default": False}}}
PLACE = {
    "type": "object",
    "properties": {
        **EXECUTE["properties"],
        # the container centre (m, arm_mount); all three, or none for the marked spot
        "x": {"type": "number"},
        "y": {"type": "number"},
        "table_z": {"type": "number"},
        # an empty-gripper test run: don't refuse an open gripper
        "allow_empty_gripper": {"type": "boolean", "default": False},
        # mock only: fail at this phase (carry, lower, open, return) to see the error on the UI
        "mock_fail": {"type": "string"},
    },
}
RESET = {
    "type": "object",
    "properties": {
        **EXECUTE["properties"],
        "save": {"type": "boolean", "default": False},
        # nothing in the (closed) gripper: skip the container-tilt check on the way back
        "empty": {"type": "boolean", "default": False},
        # empty hand only: allow a joint to turn more than 120 deg on the way back (one big cuRobo joint
        # move, around the chair's volumes) -- to recover from a different arm posture
        "big_turn": {"type": "boolean", "default": False},
    },
}


def _panel_open(ctx):
    """Whether the run's UI panel goal is still open (or opening): the engine's own run record (PR #42).
    True when it can't tell, so nothing is re-sent needlessly. From microwave (feature/14)."""
    run = getattr(ctx, "_run", None)
    if run is None or not hasattr(run, "ui_goal_handle"):
        return True
    return bool(run.ui_goal_handle is not None or getattr(run, "ui_opening", False))


class _PanelContext:
    """The action's ctx, with every progress() also put on the user's UI panel in words
    (ui.progress_panel). Everything else is ctx's own. `panel` False (standalone: no orchestrator,
    so no session token) skips the panel altogether instead of a logged no-op on every call."""

    def __init__(self, ctx, action_id, panel=True):
        self._ctx, self._action_id, self._panel = ctx, action_id, panel
        self.bag = None  # the run's RunBag while it records: each phase goes in it too
        self.ui_run = None  # place_here: its RunUI, which shows each phase with Stop
        self.phase = None  # the last phase reported
        self.camera = None  # place_here: the scene camera's subscriptions, made once per action

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def update_user_panel(self, **kwargs):
        return self._ctx.update_user_panel(**kwargs) if self._panel else False

    def progress(self, phase, pct, message="", pausable=True):
        self._ctx.progress(phase, pct, message, pausable)
        self.phase = phase
        if self.bag is not None:
            self.bag.event("phase", dict(phase=phase, pct=pct, message=message))
        if self.ui_run is not None:
            text, bar = ui.PHASES.get(phase, (message or phase, None))
            self.ui_run.moving(text, bar, phase)
        else:
            self.update_user_panel(panel_json=ui.progress_panel(self._action_id, phase, message))


class Capability(CapabilityEngine):
    display_name = "Place container"
    # The UI offers this engine's actions when the user taps a table (perception's class name)
    perception_classes = ["table"]
    required_resources = ["arm", "gripper"]
    actions = [
        # The UI's tap on a table: pick a spot on the scene camera's view, then the place for real, back to the start
        Action("place_here", ui.PLACE_HERE),
        Action("place", "Place the container on the table", params_schema=PLACE),
        Action("mark", "Save the fingertip position as the spot", params_schema={"type": "object"}),
        Action("retreat", "Back the hand straight out", params_schema=EXECUTE),
        Action("reset", "Back to the start position", params_schema=RESET),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("place_container", **node_kwargs)
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
        """Runs the action with the user's UI panel showing each phase; a failure is shown there too
        (with an OK button, waited for up to ui_error_wait_s) before the action ends with it.
        Standalone (no orchestrator, no session token) every panel call is a logged no-op.
        place_here is the UI's tap-to-place (_place_here)."""
        if action_id == "place_here":
            return self._place_here(params, ctx)
        ctx = _PanelContext(ctx, action_id, panel=self.orchestrated)
        first = {"place": "plan", "reset": "reset", "retreat": "retreat"}
        first = first.get(action_id, action_id)
        while True:
            ctx.update_user_panel(panel_json=ui.progress_panel(action_id, first))
            try:
                with self._recording(action_id, params) as bag:
                    ctx.bag = bag
                    result = self._act(action_id, params, ctx)
                    if bag is not None:
                        bag.result = result  # into run.json when the bag closes
                break
            except Cancelled:
                raise
            except EngineError as e:
                self._show_error(ctx, action_id, str(e))
                raise
            except Exception as e:  # a bug: still say so on the UI, then let the engine report it
                self._show_error(ctx, action_id, f"Unexpected error ({type(e).__name__}): {e}")
                raise
        if bag is not None:
            self.get_logger().info(f"run recorded: {bag.run_dir} (bag + run.json)")
            result = {**result, "bag_path": str(bag.path), "recording_errors": list(bag.errors)}
        ctx.update_user_panel(panel_json=ui.done_panel(action_id, "Done"))
        return result

    # -- place_here: the UI's tap-to-place ---------------------------------------------------------

    def _place_here(self, params, ctx):
        """The spot picker (_pick_spot), then the place there for real under a progress panel with Stop, then
        a Done notice. A refusal before anything moved goes back to the picker with the reason; a failure after
        the arm moved ("STOPPED ...") is shown and ends the action; Stop shows what was left where. Goal params
        x, y, table_z (all three) skip the picker (driving it by CLI)."""
        pctx = _PanelContext(ctx, "place_here", panel=self.orchestrated)
        run_ui = RunUI(pctx, ui.title("place_here"), cancel=lambda: cancel_own_run(self), logger=self.get_logger())
        pctx.ui_run = run_ui
        if not self.mock and not self.cfg["allow_execute"]:
            run_ui.refuse(ui.REFUSED_OFF)
            raise EngineError("refused", ui.REFUSED_OFF, recoverable=True)
        given = {k: params[k] for k in ("x", "y", "table_z") if k in params}
        note = ""
        while True:
            choice = given or self._pick_spot(pctx, run_ui, note)
            try:
                result = run_ui.motion(
                    ui.MOVING,
                    lambda choice=choice: self._place_here_run(pctx, choice),
                    stopped_note=lambda: ui.stopped_note(pctx.phase),
                )
                break
            except Cancelled:
                raise
            except EngineError as e:
                if run_ui.user_stopped:  # motion() has shown "Stopped"
                    raise
                if e.recoverable and "STOPPED" not in str(e) and not given and run_ui.connected:
                    # refused before anything moved: back to the picker, with the reason
                    self.get_logger().warning(f"place_here: {e} -- back to the spot picker")
                    note = f"Couldn't place it there: {e}. "
                    continue
                run_ui.error(f"Couldn't place the container: {e}")
                raise
            except Exception as e:  # a bug: still say so on the UI, then let the engine report it
                if not run_ui.user_stopped:
                    run_ui.error(f"Unexpected error ({type(e).__name__}): {e}")
                raise
        run_ui.done(ui.PLACED, mock=self.mock)
        return result

    def _place_here_run(self, pctx, choice):
        """The place itself (recorded): at `choice` (x, y, table_z), or {} for the marked spot."""
        with self._recording("place_here", choice) as bag:
            pctx.bag = bag
            result = self._act("place_here", dict(choice), pctx)
            if bag is not None:
                bag.result = result
        result = {**result, "spot": dict(choice) or "marked"}
        if bag is not None:
            self.get_logger().info(f"run recorded: {bag.run_dir} (bag + run.json)")
            result = {**result, "bag_path": str(bag.path), "recording_errors": list(bag.errors)}
        return result

    def _pick_spot(self, pctx, run_ui, note=""):
        """Ask the user where to put the container: the free spots on the scene camera's view of the table as
        tappable boxes, Place here (after a tap), Use marked spot (if a valid one is saved). Returns the place's
        params: {x, y, table_z} of the tapped spot, or {} for the marked spot. Without a UI (standalone): the
        marked spot (mock: the first free spot), else refused."""
        marked = self._marked_spot()
        look = True
        while True:
            if look:
                search, scene = self._find_spots(pctx, marked)
                objects = [] if search.problem else spots.ui_objects(search.spots, scene, self.cfg["spot_step"])
                shown = {o["instance_id"] for o in objects}
                by_id = {k: s for k, s in search.by_id().items() if k in shown}
                look = False
            tapped = {"id": None}

            def on_tap(instance_id, by_id=by_id, tapped=tapped):
                if instance_id in by_id:
                    tapped["id"] = instance_id
                    pctx.update_user_panel(selected_instance_id=instance_id)  # the highlight follows the tap

            if objects:
                buttons, message = [ui.PLACE], ui.pick_message(len(objects), note)
            else:
                buttons, message = [ui.LOOK], ui.no_spot_message(search.problem, marked is not None, note)
            buttons += [ui.MARKED] if marked is not None else []
            answer = run_ui.ask(message, buttons, objects=objects, on_tap=on_tap)
            note = ""
            if answer is None:
                if run_ui.connected:
                    raise EngineError("no_answer", "no spot was chosen on the UI in time", recoverable=True)
                if marked is not None:
                    return {}
                if self.mock and by_id:
                    s = next(iter(by_id.values()))
                    return dict(x=s["x"], y=s["y"], table_z=s["z"])
                raise EngineError(
                    "refused",
                    "place_here without the UI needs x, y and table_z, or a marked spot (run mark)",
                    recoverable=True,
                )
            if answer == ui.LOOK["button_id"]:
                look = True
            elif answer == ui.MARKED["button_id"]:
                return {}
            elif tapped["id"] is None:
                note = ui.TAP_FIRST
            else:
                spot = by_id[tapped["id"]]
                table_z, why = self._table_z(search, spot)
                if table_z is not None:
                    self.get_logger().info(
                        f"place_here: spot {spot['id']} x={spot['x']:.3f} y={spot['y']:.3f}, table z "
                        f"{table_z:.3f} (incl. +{self.cfg['table_z_bias']:.3f} bias), from the {search.source}"
                    )
                    return dict(x=spot["x"], y=spot["y"], table_z=table_z)
                note = f"Can't measure the table there ({why}). Tap another spot. "

    def _find_spots(self, pctx, marked):
        """(spots.Search, spots.Scene) of the scene camera's newest frame (mock: a made-up table). A Search
        with `problem` when there is nothing to offer: the camera isn't ready, no table, the camera's view of
        the table disagrees with the arm (tilted, or not at the marked height)."""
        if self.mock:
            scene = spots.synthetic_scene()
        else:
            if pctx.camera is None:
                pctx.camera = SceneCamera(
                    self,
                    rammp_arm.FRAME,
                    self.cfg["scene_depth_topic"],
                    self.cfg["scene_info_topic"],
                    self.cfg["detections_topic"],
                )
            try:
                scene = pctx.camera.grab(float(self.cfg["scene_wait_s"]), pctx.checkpoint)
            except RuntimeError as e:
                return spots.Search(problem=f"the scene camera isn't ready ({e})"), None
        search = spots.find_spots(
            scene,
            step=float(self.cfg["spot_step"]),
            edge_margin=float(self.cfg["spot_edge_margin"]),
            clearance=float(self.cfg["spot_clearance"]),
            footprint_r=plan.BOWL_RADIUS + plan.LIP_WIDTH,
            reach=tuple(float(v) for v in self.cfg["reach"]),
            table_z_range=tuple(float(v) for v in self.cfg["table_z_range"]),
            max_tilt_deg=float(self.cfg["max_table_tilt_deg"]),
            exclude=self._hand_exclusion(),
        )
        if not search.problem and marked is not None:
            # the camera against the arm: the table's height where the fingertips marked it
            off = float(spots.plane_z(search.plane, marked["x"], marked["y"])) - float(marked["table_z"])
            if abs(off) > float(self.cfg["table_check_tol"]):
                search.spots = []
                search.problem = (
                    f"the scene camera sees the table {abs(off) * 100:.0f} cm {'higher' if off > 0 else 'lower'} "
                    f"than the arm measured it at the marked spot, so the camera's calibration is off"
                )
        self.get_logger().info(
            f"place_here: {len(search.spots)} free spots from the {search.source or 'scene camera'} "
            f"{getattr(search, 'counts', '')}, table tilt "
            + ("?" if search.tilt_deg is None else f"{search.tilt_deg:.1f} deg")
            + (f"; {search.problem}" if search.problem else "")
        )
        return search, scene

    def _hand_exclusion(self):
        """[(x, y, r)]: the hand and the container it holds, which the scene camera may see over the table, are
        not obstacles. None in mock or without the arm's state."""
        if self.mock:
            return ()
        try:
            x, y = (float(v) for v in self.arm.get_state()["ee_pos"][:2])
        except Exception as e:  # no state: nothing left out (more conservative)
            self.get_logger().warning(f"place_here: no hand position to leave out of the obstacles ({e})")
            return ()
        return ((x, y, plan.BOWL_RADIUS + plan.LIP_WIDTH + 0.06),)

    def _table_z(self, search, spot):
        """(table_z for the place at `spot`, "") or (None, why): a plane fitted to the table's depth around the
        spot, raised by table_z_bias (erring high is safe: the descent stops on contact)."""
        try:
            z = spots.local_table_z(search.inliers, spot["x"], spot["y"])
        except ValueError as e:
            return None, str(e)
        if abs(z - spot["z"]) > 0.02:
            return None, f"the table there is {abs(z - spot['z']) * 100:.0f} cm off the table's plane"
        return z + float(self.cfg["table_z_bias"]), ""

    def _marked_spot(self):
        """The saved spot.json if it can be used (saved after state_valid_after, younger than
        max_spot_age_min), else None. Never raises: it only decides whether Use marked spot is offered."""
        try:
            spot = self._load("spot.json", "no marked spot")
        except (plan.Refused, OSError, ValueError):
            return None
        if (time.time() - float(spot.get("time", 0.0))) / 60 > float(self.cfg["max_spot_age_min"]):
            return None
        return spot

    def _recording(self, action_id, params):
        """The run's RunBag (a context manager), or a no-op one with record_bag off. A recording
        that can't start refuses the action before anything moves."""
        if not self.cfg["record_bag"]:
            return contextlib.nullcontext(None)
        from .bag import RecordingError, RunBag  # lazy: the planning tests need no rosbag2

        camera = tuple(self.cfg["bag_camera_topics"]) if self.cfg["bag_camera"] else ()
        try:
            return RunBag(
                self,
                self.state_dir,
                action_id,
                params,
                {k: v for k, v in self.cfg.items() if k != "gains"},
                camera,
                frame_hz=float(self.cfg["bag_frame_hz"]),
                state_hz=float(self.cfg["bag_state_hz"]),
            )
        except RecordingError as e:
            why = f"{e}. Next: check the disk and state_dir, or run with record_bag:=false"
            raise EngineError("recording", why, recoverable=True) from e

    def _show_error(self, ctx, action_id, message):
        """The failure on the user's panel with an OK button, waited for (see _wait_for_button) so it
        is read before the action ends and the panel closes."""
        self._wait_for_button(ctx, ui.error_panel(action_id, message), ui.OK)

    def _wait_for_button(self, ctx, panel, button_id):
        """`panel` on the UI until its `button_id` is tapped (True), or ui_error_wait_s passes or the
        action is cancelled (False). Pushed again only once the UI's goal for it has ended (the bridge
        dropped or aborted it): re-sending an open panel every second made it flash (microwave, 11-12).
        False at once if the panel can't be shown (standalone, no bridge)."""
        if not ctx.update_user_panel(panel_json=panel):
            return False
        deadline = time.monotonic() + float(self.cfg["ui_error_wait_s"])
        refresh = time.monotonic() + 1.0
        while time.monotonic() < deadline and not ctx.cancelled():
            if ui.is_button(ctx.poll_ui_event(), button_id):
                return True
            if time.monotonic() >= refresh:
                if not _panel_open(ctx):
                    ctx.update_user_panel(panel_json=panel)
                refresh = time.monotonic() + 1.0
            time.sleep(0.1)
        return False

    def _mock(self, action_id, params, ctx):
        """No hardware: place walks through its phases on the panel (ui.MOCK_PHASE_S each), so the
        UI can be tried in a sim; `mock_fail` (a phase; the goal's, else the node parameter) ends it
        there with a refusal instead."""
        fail = params.get("mock_fail") or self.cfg["mock_fail"]
        for phase in ui.MOCK_PHASES if action_id in ("place", "place_here") else ():
            ctx.checkpoint()
            ctx.progress(phase, 0.0)
            if phase == fail:
                raise EngineError(
                    "refused",
                    f"mock: a failure at '{phase}', as asked (mock_fail). Next: run again without mock_fail",
                    recoverable=True,
                )
            time.sleep(ui.MOCK_PHASE_S)
        return {"executed": False, "mock": True}

    def _act(self, action_id, params, ctx):
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            return self._mock(action_id, params, ctx)
        log = self.get_logger().info
        try:
            model = plan.Model(self.cfg["flange_to_tip"], self.cfg["pitch_deg"])
            if action_id == "mark":
                return self._mark(model, log)
            if action_id == "place_here":
                # The UI's tap: for real, at the chosen spot (x, y, table_z; none = the marked spot), back to the start
                if not self.cfg["allow_execute"]:
                    raise plan.Refused(
                        "placing is switched off on this robot (allow_execute is false). Next: start "
                        "place_container with allow_execute:=true, with someone at the e-stop"
                    )
                result = self._place(model, params, True, ctx, log)
                if self.cfg["return_mode"] != "blended":  # legacy: the separate reset brings it back
                    ctx.progress("reset", 0.0)
                    self._reset(model, {"execute": True, "empty": True}, True, log)
                return result
            if action_id == "place":
                return {**self._place(model, params, execute, ctx, log), "note": note}
            if action_id == "retreat":
                return {**self._retreat(model, execute, log), "note": note}
            return {**self._reset(model, params, execute, log), "note": note}
        except plan.Refused as e:
            raise EngineError("refused", str(e), recoverable=True) from e
        except ArmError as e:  # the driver or its state, outside this capability (Cancelled is not one)
            # (the engine logs every EngineError's message as an error: /rosout, and a run's bag)
            why = plan.explain_failure(f"{type(e).__name__}: {e}", [action_id])
            raise EngineError(type(e).__name__, why, recoverable=True) from e

    def availability(self, instance_id, world_state):
        """The UI is offered only place_here (one tap, no params). place / mark / retreat / reset stay
        callable directly: from the UI, with no params, they would only dry-run (microwave's UI actions
        do the same, feature/14-microwave-close)."""
        return [a for a in super().availability(instance_id, world_state) if a.action_id == "place_here"]

    # -- actions -------------------------------------------------------------------------------

    def _mark(self, model, log):
        tip = model.fk(self._joints(), "tip")[:3, 3]
        spot = dict(x=float(tip[0]), y=float(tip[1]), table_z=float(tip[2]), time=time.time(), source="mark")
        self._save("spot.json", spot)
        log(
            f"fingertip at x={tip[0]:.3f} y={tip[1]:.3f} z={tip[2]:.3f} m, "
            f"{np.hypot(*tip[:2]):.2f} m from the base axis"
        )
        return {"spot": [spot["x"], spot["y"]], "table_z": spot["table_z"]}

    def _place(self, model, params, execute, ctx, log):
        spot, table_z = self._spot(params, log)
        q_now = self._joints()
        self._check_tool_frame(model, q_now, log)
        gripper = float(self.arm.gripper_state().position)
        if gripper < 0.05 and not params.get("allow_empty_gripper", False):
            raise plan.Refused(
                f"the gripper is at {gripper:.2f} (0 = open; under 0.05 holds nothing): "
                "it isn't holding the container. "
                "Close it on the lip, or set allow_empty_gripper for an empty test run"
            )
        plan.check_lowering(self.cfg["lower_speed"], self.cfg["vz_stop"], rammp_arm.CART_SPEED)
        r = float(np.hypot(*spot))
        lo, hi = self.cfg["reach"]
        if not lo <= r <= hi:
            log(
                f"NOTE: the spot is {r:.2f} m from the arm's base axis, outside the reach param {lo:.2f}-{hi:.2f} m "
                "(measured without the chair volumes); the plan decides"
            )
        if self.cfg["blend"] and self.cfg["carry_targets"] == "level":
            return self._place_level(model, spot, table_z, q_now, execute, ctx, log)
        moves, descent, info = plan.plan_place(
            model,
            q_now,
            spot,
            table_z,
            max_adjust=self.cfg["max_adjust_deg"],
            allow_reconfigure=self.cfg["allow_reconfigure"],
            reach=(lo, hi),
            lower_from=self.cfg["lower_from"],
            log=log,
        )
        if info["reconfigure"]:
            log(
                "NOTE: the transit changes the arm's posture (cuRobo, joint space): a bigger move, the container's "
                "tilt is checked only along a straight joint path -- watch it"
            )
        steps = self._describe(model, moves, descent, table_z, log)
        log(
            f"adjust turns the hand {info['adjust_deg']:.1f} deg. cuRobo avoids the chair's collision volumes; the "
            "table and what is on it are NOT in its world -- keep the way to the spot clear"
        )
        # With joint targets the approach (3b) runs on move_joints after the carrying motion, not
        # through cuRobo: it must be the straight line it was planned as.
        carry, approach = plan.split_carry(moves)
        on_line = bool(approach) and self.cfg["blend"] and self.cfg["carry_targets"] == "joints"
        if on_line:
            problems = plan.check_approach(model, carry[-1][1], approach)
            if problems:
                raise plan.Refused("the approach down to the slow line is not straight: " + "; ".join(problems))
        # Before anything moves (no pause at the top): the ends of the approach, the line down and the
        # retreat against the chair's volumes, on the planned line (the one from where the arm ends is
        # the same line down, a few mm apart at most).
        self._check_contact_zone(model, descent, table_z, log, q_approach=carry[-1][1] if on_line else None)
        if not execute:
            if self.cfg["blend"] and self.cfg["curobo_check"]:
                if self.cfg["carry_targets"] == "joints":
                    self._check_carry_joints(carry if on_line else moves, log)
                else:
                    self._check_carry(info["waypoints"], log)
            if on_line:
                log(f"approach: {self._approach_text(model, carry[-1][1], approach)}")
            log("DRY RUN -- all checks passed, nothing commanded")
            return {"executed": False, "plan": steps}

        self._ready()
        self._save("start.json", dict(q=[float(v) for v in q_now], time=time.time()))
        if self.cfg["blend"] and self.cfg["carry_targets"] == "joints":
            self._carry_joints(model, carry if on_line else moves, table_z, ctx, log)
            if on_line:
                self._approach(model, carry[-1][1], approach, log)
        elif self.cfg["blend"]:
            self._carry(info["waypoints"], table_z, ctx, log)
        else:
            self._carry_separately(model, moves, ctx, log)
        return self._lower_and_let_go(model, spot, table_z, q_now, descent, info, ctx, log)

    def _lower_and_let_go(self, model, spot, table_z, q_now, descent, info, ctx, log, up_tool=None):
        """After the carrying motion (and the approach): the container's level checked above the spot,
        the line down (again from where the arm IS when it ended elsewhere), then lower.py's impedance
        lowering, letting go and backing out (or the blended return)."""
        q_top = self._joints()
        if up_tool is not None:  # the level place: off how the START held it
            tilt = level_path.tilt_from(model.fk(q_top)[:3, :3], up_tool)
            if tilt > level_path.MAX_TILT_DEG:
                raise plan.Refused(
                    f"STOPPED above the spot: the container is {tilt:.1f} deg from how the start held it "
                    f"(max {level_path.MAX_TILT_DEG:g}) -- nothing let go"
                )
        elif model.tilt(q_top) > 1.0:
            raise plan.Refused(
                f"STOPPED above the spot: the container is {model.tilt(q_top):.1f} deg off level after the carrying "
                "motion (max 1.0); "
                "nothing let go. Next: reset, then run again with adjust_mode:=joints or blend:=false"
            )
        # The arm ended on the planned joints (a joint-target carry does): the planned line down starts
        # here, use it -- planning it again (~60 IK solves) held the arm still above the spot. Otherwise
        # (pose targets: cuRobo chose the joints) the line again, from where the arm IS, or a line planned
        # from other joints would start with a jump between the two postures.
        off_deg = float(np.max(np.abs(plan.wrap_deg(q_top - np.asarray(descent[1][0])))))
        if off_deg > REPLAN_DEG and up_tool is not None:  # the level line was searched clear: not replanned blind
            raise plan.Refused(
                f"STOPPED above the spot: the arm ended {off_deg:.1f} deg from the planned line's start -- "
                "nothing let go"
            )
        if off_deg > REPLAN_DEG:
            log(f"the arm ended {off_deg:.1f} deg from the planned joints: the line down planned again from there")
            descent = plan.plan_descent(model, q_top, info["floor_z"], log=log)
        if not plan.reaches(descent, table_z):
            raise plan.Refused(
                f"STOPPED above the spot: {plan.reach_refusal(descent, table_z, spot)} in the posture cuRobo "
                "chose -- nothing let go"
            )

        rec, cfg = lower.Recorder(), {**self.cfg, "table_z": table_z, "bowl_depth": plan.BOWL_DEPTH}
        try:
            ctx.progress("lower", 0.0)
            hold_q = lower.lower(self.arm, model, descent, cfg, rec, log)
            ctx.progress("open", 0.0)
            lower.let_go(self.arm, rec, log)
            drop = self.cfg["release_drop"]
            if self.cfg["return_mode"] == "blended":
                # the drop, the retreat (and swing) and the way back to the start: ONE blended motion
                ctx.progress("return", 0.0)
                moves = plan.return_moves(model, hold_q, q_now, drop=drop, log=log)
                lower.go_back(self.arm, model, moves, self.cfg, log)
                self.arm.gripper(1.0)  # ready for the next run, as reset leaves it
                where = "Back at the start position (position control)."
            else:
                time.sleep(0.5)
                hold_q = lower.sink(self.arm, model, hold_q, drop, self.cfg["gains"], rec, log)
                ctx.progress("retreat", 0.0)
                # with the drop, the retreat stays down (no lift while the fingers may still be by the lip)
                release = 0.0 if drop > 0 else self.cfg["hold_press"]
                lower.back_out(self.arm, model, hold_q, cfg, rec, log, release=release)
                where = "The arm is still in impedance mode, by the container."
        finally:
            path = rec.write(self.state_dir / "descents")
            if path is not None:
                log(f"recorded {len(rec.rows)} /ee_state messages -> {path} (plot: plot_descent.py)")
        log(f"Done: container placed, hand clear. {where}")
        return {"executed": True, "recording": str(path)}

    def _retreat(self, model, execute, log):
        gripper = float(self.arm.gripper_state().position)
        if gripper > lower.OPEN_STALLED:
            raise plan.Refused(
                f"the gripper is at {gripper:.2f}, still closed past {lower.OPEN_STALLED} -- "
                "backing out would drag what "
                "it holds. Open it, then retreat"
            )
        q = self._joints()
        line = plan.plan_retreat(model, q, log=log)
        end = model.fk(line[-1])[:3, 3]
        if not execute:
            return {"executed": False, "to": np.round(end, 3).tolist()}
        self._ready()
        lower.back_out(self.arm, model, q, self.cfg, lower.Recorder(), log)
        return {"executed": True, "to": np.round(end, 3).tolist()}

    def _reset(self, model, params, execute, log):
        if params.get("save", False):
            q = self._joints()
            self._save("start.json", dict(q=[float(v) for v in q], time=time.time()))
            T = model.fk(q)
            log(f"saved the start position: hand at x={T[0, 3]:.3f} y={T[1, 3]:.3f} z={T[2, 3]:.3f}")
            return {"saved": True}
        start = self._load("start.json", "no start position saved: run reset with save, or a place first")
        # Nothing in the hand (the gripper is open, or the goal says so): the way back may turn the hand.
        empty = float(self.arm.gripper_state().position) <= lower.OPEN_STALLED or bool(params.get("empty", False))
        big_turn = bool(params.get("big_turn", False))
        if big_turn and not empty:
            raise plan.Refused("big_turn is only for an empty hand: open the gripper or set empty, then reset")
        moves = plan.reset_moves(
            model, self._joints(), np.asarray(start["q"], dtype=float), empty=empty, big_turn=big_turn
        )
        steps = [dict(name=n, tool=np.round(model.fk(q)[:3, 3], 3).tolist()) for n, q in moves]
        if not execute:
            return {"executed": False, "plan": steps}
        self._ready()
        if self.cfg["reset_mode"] == "legacy":
            lower.joint_moves(self.arm, model, moves, self.cfg, log)
        else:
            # After a place the arm holds in impedance a few cm off its setpoint: leave it on the
            # current joints first. Joint targets on the measured joints' branch (J1/J3/J5/J7 spin
            # freely: a wrapped target could send one the long way round).
            lower.exit_impedance(self.arm, log)
            qs = _unwrap_from(self.arm.raw_joints(), [q for _, q in moves])
            radii = plan.reset_radii(model, self._joints(), moves, self.cfg["blend_radius_near"])
            targets = [
                JointTarget(q, blend_radius=r, speed_scale=self.cfg["go_to_speed_scale"])
                for q, r in zip(qs, radii, strict=True)
            ]
            names = " -> ".join(n for n, _ in moves)
            log(f"-> reset: {names} (go_to_joint_configs, speed x{self.cfg['go_to_speed_scale']}, blend {radii})")
            if not self.arm.go_to_joint_configs(targets, "reset"):
                legs = [n for n, _ in moves]
                where = {n: model.fk(q)[:3, 3] for n, q in moves}
                raise plan.Refused("STOPPED in reset: " + plan.explain_failure(self.arm.last_error, legs, where))
        self.arm.gripper(1.0)  # ready for the next run, up here clear of the container
        log("Back at the start position.")
        return {"executed": True, "plan": steps}

    # -- helpers -------------------------------------------------------------------------------

    def _joints(self):
        return np.asarray(self.arm.get_state()["position"], dtype=float)

    def _ready(self):
        why = self.arm.not_ready()
        if why is not None:
            raise plan.Refused(
                f"the arm can't be commanded now: {why}. Next: clear that (e-stop, control), then run again"
            )

    def _carry(self, waypoints, table_z, ctx, log):
        """adjust -> lift -> transit as ONE go_to_ee_poses motion: cuRobo plans every leg around the
        chair's collision volumes and blends them, so the arm doesn't stop at the waypoints."""
        speed = self.cfg["go_to_speed_scale"]
        poses = [w for w in waypoints if not w.get("joints")]
        if self.cfg["adjust_mode"] == "joints":
            # the squaring turn to our own checked joints first (a joint-space plan: no swing), then
            # the rest from there
            w = poses.pop(0)
            ctx.progress("adjust", 0.0)
            (q,) = _unwrap_from(self.arm.raw_joints(), [w["q"]])
            log(f"-> adjust (go_to_joint_config, speed x{speed})")
            if not self.arm.go_to_joint_config(q, "adjust", speed_scale=speed):
                raise plan.Refused(
                    "STOPPED: " + plan.explain_failure(self.arm.last_error, ["adjust"], {"adjust": w["pos"]})
                )
        if poses:
            radii = plan.blend_radii(poses, table_z, self.cfg["blend_radius_near"], self.cfg["blend_radius_far"])
            targets = [
                PoseTarget(
                    w["pos"],
                    w["quat"],
                    hold=HOLD_LEVEL if w["hold"] == "level" else HOLD_NONE,
                    blend_radius=r,
                    speed_scale=speed,
                )
                for w, r in zip(poses, radii, strict=True)
            ]
            names = [w["name"] for w in poses]
            label = " -> ".join(names)
            ctx.progress(names[0] if len(names) == 1 else "carry", 0.0)
            log(f"-> {label} (go_to_ee_poses, one motion, speed x{speed}, blend {radii} m)")
            if not self.arm.go_to_ee_poses(targets, label):
                where = {w["name"]: w["pos"] for w in poses}
                raise plan.Refused("STOPPED: " + plan.explain_failure(self.arm.last_error, names, where))
        for w in waypoints:
            if w.get("joints"):  # transit-reconfigure: to these exact joints, the posture that reaches
                ctx.progress(w["name"], 0.0)
                (q,) = _unwrap_from(self.arm.raw_joints(), [w["q"]])
                log(f"-> {w['name']} (go_to_joint_config, speed x{speed})")
                if not self.arm.go_to_joint_config(q, w["name"], speed_scale=speed):
                    raise plan.Refused(
                        "STOPPED: " + plan.explain_failure(self.arm.last_error, [w["name"]], {w["name"]: w["pos"]})
                    )

    def _carry_joints(self, model, moves, table_z, ctx, log):
        """adjust -> lift -> transit as ONE go_to_joint_configs motion through plan_place's joints
        (every leg checked level and above the table, on the start's posture): cuRobo plans around the
        chair's volumes and blends, but can't pick another posture."""
        speed = self.cfg["go_to_speed_scale"]
        names = [n for n, _ in moves]
        qs = _unwrap_from(self.arm.raw_joints(), [q for _, q in moves])
        where = {n: model.fk(q)[:3, 3] for n, q in moves}
        radii = plan.blend_radii(
            [dict(pos=where[n]) for n in names], table_z, self.cfg["blend_radius_near"], self.cfg["blend_radius_far"]
        )
        targets = [JointTarget(q, blend_radius=r, speed_scale=speed) for q, r in zip(qs, radii, strict=True)]
        label = " -> ".join(names)
        ctx.progress("carry", 0.0)
        log(f"-> {label} (go_to_joint_configs, one motion, speed x{speed}, blend {radii} m)")
        if not self.arm.go_to_joint_configs(targets, label):
            raise plan.Refused("STOPPED: " + plan.explain_failure(self.arm.last_error, names, where))

    def _approach_text(self, model, q_top, approach):
        lengths = plan.approach_lengths(model, q_top, approach)
        z_top, z_end = model.fk(q_top)[2, 3], model.fk(approach[-1][1])[2, 3]
        return (
            f"straight down {sum(lengths) * 100:.0f} cm (z {z_top:.3f} -> {z_end:.3f}) through {len(approach)} "
            f"joint waypoint(s), move_joints in position mode, speed x{self.cfg['go_to_speed_scale']}"
        )

    def _approach(self, model, q_top, approach, log):
        """3b: from above the spot straight down to where the slow line starts, on move_joints through
        the planned line's joints (check_approach checked it straight and level), position mode, timed
        in metres of tool travel. Not a cuRobo leg: cuRobo doesn't keep a path straight (split_carry);
        its ends were checked against the chair's volumes (contact_zone_checks)."""
        lengths = plan.approach_lengths(model, q_top, approach)
        log(f"-> approach: {self._approach_text(model, q_top, approach)}")
        ok = self.arm.move_joints(
            [q for _, q in approach], cart_lengths=lengths, speed_scale=self.cfg["go_to_speed_scale"]
        )
        if not ok:
            where = {"approach": model.fk(approach[-1][1])[:3, 3]}
            raise plan.Refused(
                "STOPPED above the spot, nothing let go: "
                + plan.explain_failure(self.arm.last_error, ["approach"], where)
            )

    def _chair(self, model):
        """The chair and user boxes in the model's sim (level_path.Chair), made once per sim."""
        key = model.rb.physics_client_id
        if getattr(self, "_chair_cache", None) is None or self._chair_cache[0] != key:
            self._chair_cache = (key, level_path.Chair(model))
        return self._chair_cache[1]

    def _place_level(self, model, spot, table_z, q_now, execute, ctx, log):
        """carry_targets level (level_path.py): the start, however it holds the container, is the reference.
        The carry goes straight to above the spot (the lowest height whose path plans, the container's bottom
        >= 10 cm over the table), the hand only turning about vertical, the elbow kept clear of the chair and
        its user; then straight down, fast to lower_from and slow in impedance until it feels the table.
        Every timed sample of the carry, the approach and the line down is checked before anything moves,
        and exactly those trajectories run."""
        chair = self._chair(model)
        q0 = np.asarray(self.arm.raw_joints(), dtype=float)  # as the driver reports them: the plan starts here
        T0 = model.fk(q0)
        up = level_path.up_in_tool(T0[:3, :3])
        chair.set_container(up)  # the held container is checked against the chair and user too
        heading = plan.approach_direction(spot)
        speed = self.cfg["go_to_speed_scale"]
        t_plan = time.monotonic()
        height, knots = level_path.carry(model, chair, q0, spot, table_z, heading)
        (t, q, qd), _ = level_path.timed(q0, knots, model, speed)
        s_carry = level_path.check_samples(model, chair, (t, q, qd), up, table_z, "carry", speed)
        # The lip is at the fingertips, below the tool frame by the hand's pitch: the tool-frame heights the
        # line down and the lowering use (table_z + BOWL_DEPTH where the bottom meets the table) move up by it
        table_tool = table_z + level_path.tip_drop(model, knots[-1])
        floor_z = table_tool + max(plan.BOWL_DEPTH - plan.PRESS_BELOW, plan.MIN_TOOL_HEIGHT)
        line = level_path.descent_line(model, chair, knots[-1], floor_z)
        if not plan.reaches(line, table_tool):
            raise plan.Refused(plan.reach_refusal(line, table_tool, spot) + " -- nothing moved")
        approach, descent = plan.split_approach(line, table_tool, self.cfg["lower_from"])
        if approach:  # through every point of the searched line, not only split_approach's 4 cm waypoints
            k = len(line[0]) - len(descent[0])
            approach = [("approach'", qq) for qq in line[1][1:k]] + [("approach", line[1][k])]
            appr = level_path.timed(knots[-1], [qq for _, qq in approach], model, speed)[0]
            s_appr = level_path.check_samples(
                model,
                chair,
                appr,
                up,
                table_z,
                "approach",
                speed,
                min_bottom=None,
                vertical=True,
            )
        slow = (None, descent[1], np.zeros_like(descent[1]))  # the lowering is slow (lower_speed): no speed check
        s_line = level_path.check_samples(
            model, chair, slow, up, table_z, "line down", speed, min_bottom=None, vertical=True
        )
        self._check_contact_zone(model, descent, table_tool, log, q_approach=knots[-1] if approach else None)
        top = model.fk(knots[-1])[:3, 3]
        log(
            f"carry: straight to above the spot at {height * 100:.0f} cm (tool {np.round(top, 3).tolist()}), "
            f"{len(knots)} knots, {t[-1]:.1f} s at speed x{speed}: container within {s_carry['tilt_deg']:.1f} deg of "
            f"how the start holds it, arm >= {s_carry['clear_m'] * 100:.1f} cm from the padded chair/user boxes "
            f"(closest: {s_carry['clear_link']} to {s_carry['clear_box']}), container bottom >= "
            f"{s_carry['bottom_m'] * 100:.0f} cm over the table"
        )
        log(
            "carry joints (deg): "
            + " ".join(f"J{j + 1} {np.degrees(q[:, j]).min():.0f}..{np.degrees(q[:, j]).max():.0f}" for j in range(7))
        )
        if approach:
            log(
                f"approach: straight down {(line[0][0] - descent[0][0]) * 100:.0f} cm in {appr[0][-1]:.1f} s "
                f"(position control, every 2 mm point of the line); tilt {s_appr['tilt_deg']:.1f} deg, "
                f">= {s_appr['clear_m'] * 100:.1f} cm from the boxes"
            )
        zs = descent[0]
        log(
            f"lower: from z={zs[0]:.3f} down to at most {zs[-1]:.3f} ({(zs[0] - zs[-1]) * 100:.0f} cm) in impedance "
            f"until it feels the table; tilt {s_line['tilt_deg']:.1f} deg, >= {s_line['clear_m'] * 100:.1f} cm from "
            f"the boxes (closest: {s_line['clear_link']} to {s_line['clear_box']}). "
            f"Planned in {time.monotonic() - t_plan:.0f} s. "
            "The table and what is on it are NOT modelled: keep the way to the spot clear"
        )
        steps = [
            dict(name="carry", tool=np.round(top, 3).tolist(), above_table=round(float(top[2] - table_z), 3)),
            dict(name="lower", from_z=round(float(zs[0]), 3), to_z=round(float(zs[-1]), 3)),
        ]
        if not execute:
            log("DRY RUN -- all checks passed, nothing commanded")
            return {"executed": False, "plan": steps}

        self._ready()
        off = float(np.degrees(np.max(np.abs(plan.wrap(np.asarray(self.arm.raw_joints()) - q0)))))
        if off > 0.5:
            raise plan.Refused(
                f"the arm moved {off:.1f} deg while the place was planned -- nothing moved; run it again"
            )
        self._save("start.json", dict(q=[float(v) for v in q_now], time=time.time()))
        ctx.progress("carry", 0.0)
        log(f"-> carry: the checked trajectory as planned (execute_joint_trajectory, {t[-1]:.1f} s)")
        res = self.arm.execute_joint_trajectory(
            rammp_arm.JOINTS, list(zip(t, q, qd, strict=True)), path_tolerance=rammp_arm.PATH_TOL_RAD
        )
        if res.error_code != SUCCESSFUL:
            raise plan.Refused(
                "STOPPED in the carry, nothing let go: "
                + plan.explain_failure(f"carry ended {res.error_code} {res.error_string}", ["carry"], {"carry": top})
            )
        if approach:  # exactly the checked samples, from where the carry ended
            off = float(np.degrees(np.max(np.abs(plan.wrap(np.asarray(self.arm.raw_joints()) - appr[1][0])))))
            if off > 1.0:
                raise plan.Refused(f"STOPPED above the spot: the arm is {off:.1f} deg from the approach's start")
            log(f"-> approach: the checked straight line down ({appr[0][-1]:.1f} s)")
            res = self.arm.execute_joint_trajectory(
                rammp_arm.JOINTS, list(zip(*appr, strict=True)), path_tolerance=rammp_arm.PATH_TOL_RAD
            )
            if res.error_code != SUCCESSFUL:
                raise plan.Refused(
                    "STOPPED in the approach, nothing let go: "
                    + plan.explain_failure(f"approach ended {res.error_code} {res.error_string}", ["approach"])
                )
        info = {"floor_z": floor_z}
        return self._lower_and_let_go(model, spot, table_tool, q_now, descent, info, ctx, log, up_tool=up)

    def _check_carry_joints(self, moves, log):
        """Dry run: cuRobo plans each joint leg of the carrying motion (nothing moves), from the
        measured joints and then each planned one."""
        names = [n for n, _ in moves]
        start = self.arm.raw_joints()
        for k, q in enumerate(_unwrap_from(start, [q for _, q in moves])):
            try:
                res = self.arm.plan_to_joints(q, start)
            except ArmError as e:
                raise plan.Refused(
                    f"cuRobo check of {names[k]}: " + plan.explain_failure(f"{type(e).__name__}: {e}", [names[k]])
                ) from e
            if not res.success:
                why = plan.explain_failure(f"leg {k + 1}/{len(names)}: {res.message}", names)
                raise plan.Refused(f"DRY RUN, cuRobo check: {why}")
            log(f"cuRobo check: {names[k]} plans in joint space ({res.planning_time:.2f} s)")
            start = q

    def _carry_separately(self, model, moves, ctx, log):
        """blend false: each move on its own, as on 2026-10-29 (adjust on move_joints, the rest
        go_to_ee_pose level)."""
        for name, q in moves:
            ctx.progress(name, 0.0)
            if name.startswith("adjust"):
                # a small turn in place: our checked joint move (cuRobo swung the hand sideways and back
                # for it, 2026-10-08)
                ok = self.arm.move_joints([q], speed_scale=self.cfg["adjust_speed_scale"])
            elif name == "transit-reconfigure":
                # to these exact joints: the posture whose line down reaches the table
                log(f"-> {name} (go_to_joint_config, speed x{self.cfg['go_to_speed_scale']})")
                (q,) = _unwrap_from(self.arm.raw_joints(), [q])
                ok = self.arm.go_to_joint_config(q, name, speed_scale=self.cfg["go_to_speed_scale"])
            else:
                T = model.fk(q)
                log(f"-> {name} (go_to_ee_pose, level, speed x{self.cfg['go_to_speed_scale']})")
                ok = self.arm.go_to_ee_pose(
                    T[:3, 3],
                    Rotation.from_matrix(T[:3, :3]).as_quat(),
                    name,
                    speed_scale=self.cfg["go_to_speed_scale"],
                    orientation_hold=HOLD_LEVEL,
                )
            if not ok:
                where = {name: model.fk(q)[:3, 3]}
                raise plan.Refused("STOPPED: " + plan.explain_failure(self.arm.last_error, [name], where))

    def _check_carry(self, waypoints, log):
        """Dry run: cuRobo plans each leg of the carrying motion on its own (nothing moves), from our
        IK joints at the leg before, so a target in the chair's volumes or out of reach is refused
        before anything moves. The real motion plans them again from where the arm is."""
        names = [w["name"] for w in waypoints]
        where = {w["name"]: w["pos"] for w in waypoints}
        starts = _unwrap_from(self.arm.raw_joints(), [w["q"] for w in waypoints])
        start = self.arm.raw_joints()
        for k, w in enumerate(waypoints):
            joints = w.get("joints") or (k == 0 and self.cfg["adjust_mode"] == "joints")
            try:
                if joints:
                    res = self.arm.plan_to_joints(starts[k], start)
                else:
                    hold = HOLD_LEVEL if w["hold"] == "level" else HOLD_NONE
                    res = self.arm.plan_to_pose(w["pos"], w["quat"], start, hold)
            except ArmError as e:
                raise plan.Refused(
                    f"cuRobo check of {w['name']}: " + plan.explain_failure(f"{type(e).__name__}: {e}", [w["name"]])
                ) from e
            if not res.success:
                why = plan.explain_failure(f"leg {k + 1}/{len(names)}: {res.message}", names, where)
                raise plan.Refused(f"DRY RUN, cuRobo check: {why}")
            log(f"cuRobo check: {w['name']} plans ({res.planning_time:.2f} s)")
            start = starts[k]

    def _check_contact_zone(self, model, descent, table_z, log, q_approach=None):
        """cuRobo plans (nothing moves) to the ends of the approach (from `q_approach`, if it runs on
        move_joints), the line down and the retreat, which run on move_joints and so are never checked
        against the chair's collision volumes otherwise."""
        if not self.cfg["curobo_check"]:
            log("curobo_check is off: the approach, the line down and the retreat are NOT checked against the chair")
            return
        for label, q_from, q_to in plan.contact_zone_checks(model, descent, table_z, q_approach):
            try:
                res = self.arm.plan_to_joints(q_to, q_from)
            except ArmError as e:
                raise plan.Refused(
                    f"cuRobo check of {label}: " + plan.explain_failure(f"{type(e).__name__}: {e}", [label])
                ) from e
            if not res.success and "VALIDATION_FAILED" in str(res.message):
                # cuRobo validates its trajectory only once it has accepted both ends as clear (a colliding
                # end fails earlier, INVALID_START/GOAL_STATE_WORLD_COLLISION), and the move runs on
                # move_joints, not this trajectory: a rejected trajectory (flatbot 2026-11-12, the retreat:
                # J2 acceleration 25.3-25.5 > 25.0, every time) says nothing about the chair
                log(
                    f"cuRobo check: the end of {label} is clear of the chair's collision volumes (cuRobo's own "
                    f"trajectory for it failed its validation, not used: {res.message})"
                )
                continue
            if not res.success:
                where = {label: model.fk(q_to)[:3, 3]}
                raise plan.Refused(
                    f"cuRobo check of {label} (nothing moved): " + plan.explain_failure(res.message, [label], where)
                )
            log(f"cuRobo check: the end of {label} is clear of the chair's collision volumes")

    def _check_tool_frame(self, model, q, log):
        reported = np.asarray(self.arm.get_state()["ee_pos"][:3], dtype=float)
        modelled = model.fk(q)[:3, 3]
        off = float(np.linalg.norm(reported - modelled))
        log(
            f"tool frame: arm says {np.round(reported, 3).tolist()}, "
            f"model {np.round(modelled, 3).tolist()} ({off * 100:.1f} cm apart)"
        )
        if off > self.cfg["tool_frame_tol"]:
            raise plan.Refused(
                f"the arm reports its tool frame at {np.round(reported, 3).tolist()} m, the model at "
                f"{np.round(modelled, 3).tolist()}: {off * 100:.1f} cm apart (tool_frame_tol "
                f"{self.cfg['tool_frame_tol'] * 100:.1f} cm). Next: check the driver's tool configuration (Robotiq "
                "2F-85, arm.tool_z 0.12 m) and that the driver's URDF is flatbot's"
            )

    def _spot(self, params, log):
        given = [k for k in ("x", "y", "table_z") if k in params]
        if given and len(given) < 3:
            raise plan.Refused("give all of x, y and table_z, or none (the marked spot)")
        if given:
            return np.array([params["x"], params["y"]], dtype=float), float(params["table_z"])
        spot = self._load(
            "spot.json", "no marked spot: run mark with the fingertips on the target, or give x, y, table_z"
        )
        age = (time.time() - spot["time"]) / 60
        log(f"marked spot ({age:.0f} min old): x={spot['x']:.3f} y={spot['y']:.3f}, table z {spot['table_z']:.3f}")
        if age > self.cfg["max_spot_age_min"]:
            raise plan.Refused(
                f"the marked spot is {age:.0f} min old -- the chair or table may have moved; mark it again"
            )
        return np.array([spot["x"], spot["y"]], dtype=float), float(spot["table_z"])

    def _describe(self, model, moves, descent, table_z, log):
        steps = []
        for name, q in moves:
            T = model.fk(q)
            steps.append(dict(name=name, tool=np.round(T[:3, 3], 3).tolist(), above_table=round(T[2, 3] - table_z, 3)))
            log(
                f"{name:10s} tool {np.round(T[:3, 3], 3).tolist()}  {(T[2, 3] - table_z) * 100:5.1f} cm above the table"
            )
        zs = descent[0]
        log(
            f"lower      one move (~{self.cfg['lower_speed'] * 100:.0f} cm/s at the table) from z={zs[0]:.3f} "
            f"down to at most {zs[-1]:.3f} "
            f"({(zs[0] - zs[-1]) * 100:.0f} cm), stopping when it feels the table"
        )
        steps.append(dict(name="lower", from_z=round(float(zs[0]), 3), to_z=round(float(zs[-1]), 3)))
        return steps

    def _save(self, name, record):
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            (self.state_dir / name).write_text(json.dumps(record, indent=2))
        except OSError as e:
            raise plan.Refused(f"can't save {name} in {self.state_dir} ({e}) -- set state_dir") from e

    def _load(self, name, missing):
        path = self.state_dir / name
        if not path.exists():
            raise plan.Refused(missing)
        record = json.loads(path.read_text())
        saved = float(record.get("time", 0.0))
        if saved < float(self.cfg["state_valid_after"]):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(saved))
            raise plan.Refused(
                f"{path} was saved {when}, before the arm moved on the chair "
                f"(state_valid_after={self.cfg['state_valid_after']:.0f}) -- "
                + (
                    "run mark again"
                    if name == "spot.json"
                    else "put the arm at a start position and run reset with save"
                )
            )
        return record


def main():
    run(Capability)


if __name__ == "__main__":
    main()
