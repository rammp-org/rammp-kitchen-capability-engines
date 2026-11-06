"""The fridge_peering_mode capability: look inside an open fridge with the wrist camera.

Action `peer` (streaming.PeerSession.run): go to the fridge the scene camera sees (its 2D box and
depth), wait for the plane detector's fridge TF, plan to a standoff in front of the back wall with
go_to_ee_pose, then (unless `teleop` is false) stream
ee_twist as a closed-loop servo: the joystick slides the camera along the wall, which the servo
follows as the detector re-estimates it. Teleop runs until `max_teleop_s` (success) or a cancel.
The joystick only moves the camera: no buttons are read.

No fridge TF to home on (and `search_when_no_fridge`): the same servo runs in a search mode, the
stick moving the camera a little (orientation held) so the detector gets another view. Once a
fridge TF is measured with the arm still (the stick let go), it homes by itself.

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets `execute`.
Otherwise the goal is a dry run: the standoff is planned (nothing moves) and teleop publishes what
it would stream on ~/commanded_twist. There is no deadman and this node never publishes /estop:
someone stands at the e-stop whenever allow_execute is on.

    ros2 launch fridge_peering_mode fridge_peering_mode.launch.py use_orchestrator:=false \
        allow_execute:=true
    make action CAP=fridge_peering_mode ACTION=peer PARAMS='{"execute": false}'
    ros2 service call /fridge_peering_mode/control rammp_orchestrator_interfaces/srv/Control \
        '{command: 2}'  # cancel: ends teleop
"""

from capability_engine import (
    Action,
    ArmError,
    CapabilityEngine,
    ControlTaken,
    EngineError,
    EStopped,
    StreamClosed,
    run,
)
from capability_engine.driver_state import Refused

from . import geometry as geo
from .streaming import PeerSession, facing, follow_params

ENGINE_ID = "fridge_peering_mode"  # = the package, its image and /<id>/... (make action CAP=)
FOLLOW_MODES = ("off", "idle", "continuous")
SCENE_APPROACHES = ("robot_forward", "camera_ray", "from_arm")

# ROS parameter -> default. Frames, topics and what is measured on the fridge are per robot (the
# fragment's params:). /ee_state's frame and the goals' frame are arm.frame, not one of these.
PARAMS = {
    "allow_execute": False,
    # teleop streams ee_twist at rate_hz; the driver closes the session after stream_timeout_s with
    # no setpoint (> 1 / rate_hz)
    "rate_hz": 50.0,
    "stream_timeout_s": 0.2,
    # the plane detector publishes base_frame -> fridge_frame, +z INTO the back wall. A fridge frame
    # whose +z points toward the arm is refused (verify_fridge_z), never silently flipped
    "base_frame": "arm_mount",
    "fridge_frame": "fridge_frame",
    "verify_fridge_z": True,
    # /ee_state reports driver_ee_frame (the flange, the driver's --ee-frame); ee_link is what is
    # put at the standoff, limited and oriented, rigid to it. arm_tool_frame is 0.120 m past the
    # flange
    "driver_ee_frame": "arm_end_effector_link",
    "ee_link": "arm_tool_frame",
    "static_tf_timeout_s": 5.0,
    # the camera: optical_axis of ee_link looks along fridge +z; with roll_deg 0
    # roll_reference_ee_axis lines up with roll_reference_fridge_axis (+-x or +-y); roll_deg turns
    # it about the optical axis
    "optical_axis": "+z",
    "roll_reference_ee_axis": "+x",
    "roll_reference_fridge_axis": "+x",
    "roll_deg": 180.0,
    # world up (up_axis of up_frame): the stick's u = left / v = up, and the fridge axes rebuilt
    # while following. Must match the plane detector's fixed_frame +z (a mismatch is warned about at
    # homing)
    "up_frame": "arm_mount",
    "up_axis": "+z",
    # the standoff: ee_link standoff_distance back from the wall (m), offset along fridge x / y (m).
    # Homing waits up to fridge_tf_wait_s for a fridge TF measured follow_settle_s after the arm
    # last moved. CHECK: the detector drops depth closer than its min_range (0.20 m) to the CAMERA;
    # closer than that, following holds its plane
    "fridge_tf_wait_s": 5.0,
    "standoff_distance": 0.40,
    "offset_x": 0.0,
    "offset_y": 0.0,
    # no usable fridge TF within fridge_tf_wait_s: with teleop, search -- the stick moves ee_link
    # within +-search_range_m (m) of where it was, its orientation held (u = left, v = up, standoff
    # = forward / back, level), so the detector gets another view. It homes by itself once a
    # fridge TF is measured with the arm still. false: fail with no_fridge (as teleop: false
    # always does)
    "search_when_no_fridge": True,
    "search_range_m": 0.15,
    # scene homing, first: the scene camera's scene_class box (scene_detections_topic) and depth
    # (scene_depth_topic, ALIGNED TO COLOUR) place a point at scene_box_point (fractions across /
    # down the box: 0.75 = 1/4 from its right border), its depth the scene_depth_percentile of the
    # fridge's mask pixels within scene_window_frac x the box around it (scene_depth_range_m, at
    # least scene_min_samples), the median of scene_frames detections within scene_wait_s.
    # ee_link goes scene_standoff_m in front of it, the camera looking along scene_approach (level):
    # "robot_forward" (scene_forward_axis of base_frame), "camera_ray" (from the scene camera
    # through the point) or "from_arm" (from the arm base through it). Skipped, with a warning, when
    # it finds nothing or cannot plan
    "scene_homing": True,
    "scene_detections_topic": "/perception/object_detections",
    "scene_depth_topic": "/scene_camera/depth/image_rect_raw",
    "scene_camera_info_topic": "/scene_camera/color/camera_info",
    "scene_class": "fridge",
    "scene_min_score": 0.5,
    "scene_box_point": [0.75, 0.5],
    "scene_window_frac": 0.1,
    "scene_depth_percentile": 50.0,
    "scene_depth_range_m": [0.3, 3.0],
    "scene_min_samples": 30,
    "scene_frames": 5,
    "scene_wait_s": 3.0,
    "scene_standoff_m": 0.40,
    "scene_approach": "robot_forward",
    "scene_forward_axis": "+x",
    # a go_to whose start is inside cuRobo's joint-limit margin (START_IN_LIMIT_MARGIN) is
    # retried once after turning only those joints unwind_extra_rad (rad) past the margin, by a
    # slow direct joint trajectory (not collision-checked; refused if it needs > unwind_max_rad)
    "unwind_limits": True,
    "unwind_extra_rad": 0.05,
    "unwind_max_rad": 0.25,
    # the axes ee_twist setpoints are in (the driver: arm_mount)
    "twist_frame": "arm_mount",
    # the servo: full stick slides the target at max_speed on the wall and standoff_speed toward it
    # (m/s); v = v_ff + kp_lin * error, w = w_ff + kp_ang * error (1/s), capped at max_lin (m/s) and
    # max_ang (rad/s)
    "max_speed": 0.08,
    "standoff_speed": 0.05,
    "kp_lin": 2.0,
    "kp_ang": 2.0,
    "max_lin": 0.12,
    "max_ang": 0.4,
    # the target may lead the arm by max_lead_m; past lead_slowdown_m it slows, smoothly, to 0 there
    "max_lead_m": 0.05,
    "lead_slowdown_m": 0.02,
    # the target's limits (m, for the ee_link origin): u (+ = left) and v (+ = up) from the homing
    # plane origin, and the distance from the wall, which must contain standoff_distance
    "u_limits": [-0.35, 0.35],
    "v_limits": [-0.30, 0.30],
    "standoff_limits": [0.08, 0.45],
    # following the detector's re-estimates of the wall: "off" (the plane homing used), "idle" (only
    # with the stick at neutral, estimates ignored until follow_settle_s after it moved) or
    # "continuous" (pair with the detector's max_joint_velocity motion gate). The fridge does not
    # move, so the plane is converged on, never tracked: gated against homing (follow_max_tilt_deg,
    # follow_max_shift_m) and the followed plane (follow_max_step_*), deadbanded, then a critically
    # damped filter (follow_w0, rad/s) capped at follow_max_rate_deg_s and follow_max_shift_rate
    # (m/s)
    "follow_mode": "continuous",
    "follow_settle_s": 0.5,
    "follow_stale_s": 2.0,
    "follow_w0": 2.0,
    "follow_max_rate_deg_s": 4.0,
    "follow_max_shift_rate": 0.02,
    "follow_deadband_deg": 1.0,
    "follow_deadband_m": 0.005,
    "follow_max_tilt_deg": 15.0,
    "follow_max_shift_m": 0.05,
    "follow_max_step_deg": 5.0,
    "follow_max_step_m": 0.03,
    # the joystick (an Xbox-style pad on joy_node, xpad layout: check every index with ros2 topic
    # echo /joy). Axes: u + = left, v + = up, standoff + = closer (-1 = off); deadzone, then sign,
    # then expo. No buttons are read
    "joy_topic": "/joy",
    "axis_u": 0,
    "axis_v": 1,
    "axis_standoff": -1,
    "axis_signs": [1.0, 1.0, 1.0],
    "deadzone": 0.1,
    "expo": 2.0,
    # older than this is not used (s): /joy holds the target (> 1 / joy_node's autorepeat_rate), the
    # fridge TF is not followed (<= 0 disables; a static TF is exempt), /ee_state sends a zero twist
    "joy_timeout_s": 0.25,
    "fridge_tf_timeout_s": 1.0,
    "ee_state_timeout_s": 0.2,
}


def _problems(cfg) -> list[str]:
    """Everything wrong with the parameters; the node refuses to start on any."""
    out = []
    for name in ("rate_hz", "joy_timeout_s", "ee_state_timeout_s", "fridge_tf_wait_s",
                 "static_tf_timeout_s", "max_lin", "max_ang", "max_lead_m", "expo",
                 "follow_stale_s", "standoff_distance", "search_range_m", "scene_wait_s",
                 "scene_standoff_m", "scene_window_frac", "unwind_max_rad"):
        if not cfg[name] > 0.0:
            out.append(f"{name} must be > 0, got {cfg[name]}")
    for name in ("max_speed", "standoff_speed", "kp_lin", "kp_ang", "follow_settle_s",
                 "unwind_extra_rad"):
        if cfg[name] < 0.0:
            out.append(f"{name} must be >= 0, got {cfg[name]}")
    if cfg["rate_hz"] > 0.0 and not cfg["stream_timeout_s"] > 1.0 / cfg["rate_hz"]:
        out.append("stream_timeout_s must exceed 1 / rate_hz, or the driver closes the session")
    if not 0.0 <= cfg["deadzone"] < 1.0:
        out.append("deadzone must be in [0, 1)")
    if not 0.0 <= cfg["lead_slowdown_m"] < cfg["max_lead_m"]:
        out.append("need 0 <= lead_slowdown_m < max_lead_m")
    if len(cfg["axis_signs"]) != 3:
        out.append("axis_signs must be [u, v, standoff]")
    for name in ("u_limits", "v_limits", "standoff_limits"):
        lim = cfg[name]
        if not (len(lim) == 2 and lim[0] < lim[1]):
            out.append(f"{name} must be [min, max]")
    lo, hi = (list(cfg["standoff_limits"]) + [0.0, 0.0])[:2]
    if not lo <= cfg["standoff_distance"] <= hi:
        out.append(
            f"standoff_distance {cfg['standoff_distance']} lies outside standoff_limits "
            f"[{lo}, {hi}]: teleop would start outside its own limits"
        )
    if cfg["scene_approach"] not in SCENE_APPROACHES:
        out.append(
            f"scene_approach must be one of {SCENE_APPROACHES}, got {cfg['scene_approach']!r}"
        )
    bp = list(cfg["scene_box_point"])
    if not (len(bp) == 2 and all(0.0 <= v <= 1.0 for v in bp)):
        out.append("scene_box_point must be [across, down], each in [0, 1]")
    zr = list(cfg["scene_depth_range_m"])
    if not (len(zr) == 2 and 0.0 < zr[0] < zr[1]):
        out.append("scene_depth_range_m must be [min, max], 0 < min < max")
    if not 0.0 <= cfg["scene_depth_percentile"] <= 100.0:
        out.append("scene_depth_percentile must be in [0, 100]")
    if not 0.0 <= cfg["scene_min_score"] <= 1.0:
        out.append("scene_min_score must be in [0, 1]")
    for name in ("scene_frames", "scene_min_samples"):
        if cfg[name] < 1:
            out.append(f"{name} must be >= 1, got {cfg[name]}")
    if cfg["follow_mode"] not in FOLLOW_MODES:
        out.append(f"follow_mode must be one of {FOLLOW_MODES}, got {cfg['follow_mode']!r}")
    out += [f"follow_*: {p}" for p in follow_params(cfg).problems()]
    try:
        facing(cfg)
        geo.axis_vector(cfg["up_axis"])
        geo.axis_vector(cfg["scene_forward_axis"])
    except ValueError as exc:
        out.append(str(exc))
    return out


# EStopped is a Refused: it comes first.
_ARM_ERROR_CODES = (
    (EStopped, "estopped"),
    (ControlTaken, "control_taken"),
    (StreamClosed, "stream_closed"),
    (Refused, "refused"),
)


def _error_code(exc: ArmError) -> str:
    return next((code for cls, code in _ARM_ERROR_CODES if isinstance(exc, cls)), "arm_error")


class Capability(CapabilityEngine):
    display_name = "Look inside the fridge"
    perception_classes = ["fridge"]
    required_resources = ["arm"]  # the gripper is never commanded
    actions = [
        Action(
            "peer",
            "Look inside the fridge",
            params_schema={
                "type": "object",
                "properties": {
                    "execute": {"type": "boolean", "default": False},
                    # scales arm.go_to_speed_scale for the move to the standoff
                    "speed": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                        "default": 1.0,
                    },
                    # false: stop at the standoff (and fail with no_fridge instead of searching)
                    "teleop": {"type": "boolean", "default": True},
                    # end teleop (successfully) after this long in total; null: only on a cancel
                    "max_teleop_s": {
                        "type": ["number", "null"],
                        "exclusiveMinimum": 0,
                        "default": None,
                    },
                },
            },
        )
    ]

    def __init__(self):
        super().__init__(ENGINE_ID)
        self.cfg = {
            name: self.declare_parameter(name, default).value for name, default in PARAMS.items()
        }
        bad = _problems(self.cfg)
        if bad:
            raise ValueError("fridge_peering_mode parameters: " + "; ".join(bad))
        moves = self.cfg["allow_execute"]
        self.get_logger().info(
            f"allow_execute={moves}: "
            + ("the arm WILL move on execute goals" if moves else "every goal is a dry run")
        )

    def execute(self, action_id, params, ctx):
        speed = float(params.get("speed", 1.0))
        if not 0.0 < speed <= 1.0:
            raise EngineError("bad_params", "'speed' must be in (0, 1]", recoverable=True)
        max_teleop_s = params.get("max_teleop_s")
        if max_teleop_s is not None and not float(max_teleop_s) > 0.0:
            raise EngineError("bad_params", "'max_teleop_s' must be > 0 or null", recoverable=True)
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = ""
        if params.get("execute") and not execute:
            note = "allow_execute is false: ran as a DRY RUN"
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing detected or moved")
            return {"executed": False, "mock": True}

        ctx.checkpoint()
        ctx.progress(action_id, 0.0, note or ("executing" if execute else "dry run"))
        # Its joystick subscription, tf buffer and debug publishers are this action's.
        session = PeerSession(self, ctx, self.cfg, execute=execute, speed=speed)
        try:
            out = session.run(teleop=bool(params.get("teleop", True)), max_teleop_s=max_teleop_s)
        except ArmError as e:  # Cancelled is not one: it propagates, as the engine expects
            raise EngineError(
                _error_code(e), f"{e} -- the arm holds where it is", recoverable=True
            ) from e
        return {"executed": execute, "note": note, **out}


def main():
    run(Capability)


if __name__ == "__main__":
    main()