# Fridge Peering Mode — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @SwapnilPande
**Tier:** experimental

## What it does

Looks inside an open fridge with the wrist camera and takes food out of it. `peer` homes the wrist
camera in front of the fridge's back wall (from the scene camera, then the plane detector's
`fridge_frame`) and streams `ee_twist` so a 2D joystick slides the camera over the wall. While the
arm is still, SAM3 finds food containers in the wrist camera and the UI shows them as tappable
boxes; a tap ends teleop, picks that container (microwave_place's take-out: approach, straight line
in, close on the brim, lift, withdraw) and backs the arm out of the fridge holding it. `pick` does
the same pick from wherever the arm is. Without it, the user cannot look into or take food from
the fridge.

The arm moves only with the `allow_execute` parameter AND the goal's `execute`; otherwise every
goal is a dry run that plans and checks but commands nothing. There is no deadman: someone stands
at the e-stop whenever `allow_execute` is on.

## Publishes

| Topic                                     | Type                         | Rate                | Meaning                                                                                                    |
| ----------------------------------------- | ---------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------- |
| `/fridge_peering_mode/status`             | `std_msgs/String`            | 1 Hz                | Health topic: JSON `state`, `action`, `instance_id`, `mode`, `registered`, `mock`                          |
| `/fridge_peering_mode/sam3_overlay`       | `sensor_msgs/Image`          | ~1 Hz, latched      | SAM3 debug: the wrist frame with the food boxes and their instance ids (and the pick's container mask)     |
| `/fridge_peering_mode/in_view`            | `std_msgs/String`            | 1 Hz                | During `peer` and `pick`: JSON of the tracked food and drink now (`instance_id`, class, score, box, 3D point in the arm frame) |
| `/fridge_peering_mode/ui_panel`           | `std_msgs/String`            | on change, latched  | During `peer` only: JSON `{ui_session, panel_json, objects}`, what the UI is shown (for testing without it) |
| `/fridge_peering_mode/goal_pose`          | `geometry_msgs/PoseStamped`  | per homing, latched | During `peer` only: the homing goal (tool frame)                                                           |
| `/fridge_peering_mode/commanded_twist`    | `geometry_msgs/TwistStamped` | `rate_hz`           | During `peer` only: the twist streamed (or, in a dry run, that would be)                                   |
| `/fridge_peering_mode/servo_target`       | `geometry_msgs/PoseStamped`  | `rate_hz`           | During `peer` only: the servo's target for `ee_link`                                                       |
| `/fridge_peering_mode/followed_plane`     | `geometry_msgs/PoseStamped`  | `rate_hz`           | During `peer` only: the followed wall plane                                                                |

The UI gets the tappable objects and the wrist colour stream through `ctx.update_user_panel()` and
`ctx.publish_overlay_image()` (the UI bridge), only with an orchestrator session.

## Subscribes

| Topic                                                  | Type                                       | Required        | Meaning                                                                                                   |
| ------------------------------------------------------ | ------------------------------------------ | --------------- | --------------------------------------------------------------------------------------------------------- |
| `/tf`, `/tf_static`                                    | `tf2_msgs/TFMessage`                       | yes             | Arm frames and the plane detector's `fridge_frame`                                                        |
| `/joy`                                                 | `sensor_msgs/Joy`                          | for teleop      | The joystick (`joy_topic`)                                                                                |
| `/perception/object_detections`                        | `rammp_perception_interfaces/ObjectDetections` | for scene homing | The scene segmenter's fridge                                                                           |
| `/scene_camera/depth/image_rect_raw`, `.../camera_info` | `sensor_msgs/Image`, `CameraInfo`         | for scene homing | Scene depth (aligned to colour) and intrinsics                                                           |
| `/wrist_camera/color/image_raw`, `/wrist_camera/depth/image_rect_raw`, `/wrist_camera/color/camera_info` | `sensor_msgs/Image`, `CameraInfo` | for picking | `wrist_topics`                                         |
| `/fridge_peering_mode/simulate_tap`                    | `std_msgs/String`                          | no              | During `peer` only: a tap without the UI: the bridge's `{"event_type": "object_tapped", "instance_id": ...}` JSON, a bare instance id, or `""` for the best container |

SAM3: while `peer` or `pick` runs it holds a tracking session (`/perception/sam3/start_tracking`,
renewed with `renew_tracking`, ended with `stop_tracking`; kept across `peer`'s handoff to `pick`)
for `track_prompts` on the wrist colour topic, and reads `/perception/sam3/object_detections`. The
SAM3 node runs one session for the whole robot: another client's `start_tracking` replaces it (this
module restarts it within ~5 s). The grasp mask itself is a one-shot `/perception/sam3/detect` at
full resolution, locked onto the tracked object. Without tracking it falls back to one-shot
detections; without SAM3 at all, picking is off for that action (warned) and teleop still runs.

## Orchestrator interface

### UI

A `fridge` tap offers one button (`availability()` returns only it; the orchestrator sends no
params). It runs `peer` with fixed params under `capability_ui.RunUI` (`shared/capability_ui`),
from `fridge_peering_mode/ui_flow.py`, its wording in `ui_text.py`:

| Action id        | Button               | Offered on a tap of | Runs                                             |
| ---------------- | -------------------- | ------------------- | ------------------------------------------------ |
| `look_in_fridge` | "Look Inside Fridge" | `fridge`            | `peer` `{"execute": true, "max_teleop_s": 90.0}` |

`peer` and `pick` stay callable directly (dry runs by default, the CLI) but are not offered to the
UI: from it they would only dry-run. A `pick` goal sent while `look_in_fridge` runs is refused
(busy): the handoff is only for a directly sent `peer`.

- `allow_execute` false (not mock): a "not started" notice with Done, then `execution_disabled`
  (recoverable). Nothing moves.
- While it runs: one panel with **Stop**, from the press to the end, through scene homing, homing,
  search or teleop, and a tap's pick. Its message follows the steps ("Moving the wrist camera in
  front of the fridge's back wall.", "Move the joystick to look around. Hold still to find food,
  then tap a container to pick it up.", "Taking the container: the arm is moving.", ...). The food
  boxes (tap-to-pick) are on the same panel, over the wrist camera's overlay, which is shown with
  its first frame (`ctx.set_overlay_visible(True)`; the orchestrator hides it when the run ends).
- Stop cancels the engine's own run, as the orchestrator's cancel does: in teleop the stream's next
  send raises `Cancelled` and closing the stream holds the arm; in a planned move `self.arm` stops
  it. A "Stopped" notice then says what the hand holds (from the run's last progress phase, e.g.
  "The hand is holding the container, part way out of the fridge.").
- Teleop ends by itself after 90 s (`ui_flow.LOOK_S`), with a Done notice that says how to keep
  looking. The orchestrator cancels a run that sends no progress for 120 s, and teleop sends none
  after it starts; a progress also goes out before each final notice for the same reason.
- A failure (`no_tf`, `no_fridge`, `refused`, `pick_failed`, `estopped`, `control_taken`,
  `stream_closed`, ...) or an exception: a "failed" notice with the reason in plain words and what
  the hand holds, Done only. A tap's pick refused before anything moved still goes back to teleop.
- Success: Done ("Holding the container outside the fridge." after a pick). Mock: "Mock: ... No arm
  motion."
- Every final notice waits at most 60 s. An orchestrator cancel shows nothing more. Without a UI
  session nothing waits; FoodTargets then mirrors its panel on `~/ui_panel` and takes
  `~/simulate_tap`, as before.

The bridge's events have one reader while the run moves, RunUI's Stop watcher; the object taps it
reads are handed to FoodTargets, so a tap never races Stop. `ui_bridge_node_name` (default
`ui_bridge_node`) selects `/ui_bridge_node/request_user_input` and its `/update` service. JSON
shapes: [UI_JSON_SCHEMA.md](../../../docs/UI_JSON_SCHEMA.md).

### Services and actions

Served under `/fridge_peering_mode/`, types from `rammp_orchestrator_interfaces`:

| Name                                  | Type                   | Meaning                                       |
| ------------------------------------- | ---------------------- | --------------------------------------------- |
| `/fridge_peering_mode/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/fridge_peering_mode/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/fridge_peering_mode/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/fridge_peering_mode/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

| Action | Params | Result payload |
| ------ | ------ | -------------- |
| `peer` | `execute` (bool, `false`), `speed` ((0, 1], `1.0`), `teleop` (bool, `true`), `max_teleop_s` (number or null), `pick` (bool, `true`), `label` (string, `""`: `pick_prompt`; what the overlay boxes are of) | `{executed, note, ended, scene_homed, homes, searches, teleop_s}`; `ended`: `max_teleop_s`, `homed`, `scene_homed`, `picked` (then `pick`: the pick's result, below) or `handed_off` (a `pick` goal ended it) |
| `pick` | `execute` (bool, `false`), `speed` ((0, 1], `1.0`), `label` (string: a tracked id like `food_2` from `~/in_view`; or a class / prompt that must match exactly one object in view; `""`: the only food or drink in view) | `{executed, note, plan}` (dry run) or `{executed, gripper, left_m, tool, plan, ...}` |

`label` names one object, never a choice among several. With SAM3 tracking (`track_prompts`), a
tracked id (`food_2`: SAM3 numbers one counter across all its prompts, so `food_1`, `drink_2`,
`food_3`) is exactly that object, and a tracked class (`food`) or none must match exactly one in
view; otherwise the pick is refused with the ids in view. Any other label is a one-shot SAM3 prompt
that must also match exactly one. The grasp is the same whatever the label: a pinch on a lid's brim.

A `pick` goal sent while `peer` teleops is not rejected as busy: its acceptance waits (up to 15
s) while `peer` leaves teleop, the arm holding, and succeeds with `ended: "handed_off"`; then the
pick runs from there, SAM3's tracking session (and so its ids) carried over. A `peer` not
teleoperating (homing, waiting for the fridge TF) is cancelled instead; a `peer` busy with a tap's
pick refuses the goal.

Error codes: `bad_params`, `no_tf`, `no_fridge`, `refused` (homing), `pick_refused` (`pick`:
nothing moved), `pick_failed` (the pick moved and stopped; the arm holds), and the arm's
`estopped`, `control_taken`, `stream_closed`, `arm_error`. A `peer` pick refused before anything
moved goes back to teleop instead of failing.

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until the orchestrator
is up. Needs the `arm` and the `gripper`.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A deployment sets one
per robot with an `overrides:` entry in its profile; sheppy passes them as a params file.

| Name               | Default | Meaning                                                   |
| ------------------ | ------- | --------------------------------------------------------- |
| `use_orchestrator` | `true`  | `false` runs standalone: no registration, drive it by CLI |
| `mock`             | `false` | `true` skips hardware (the mock alternative sets it)      |

| Name | Default | Meaning |
| --- | --- | --- |
| `ui_bridge_node_name` | `"ui_bridge_node"` | The UI bridge node the `look_in_fridge` panels and the wrist overlay go to |
| `allow_execute` | `false` | `false`: every goal is a dry run (plans, streams nothing, picks nothing). `true` only attended, someone on the e-stop |
| `rate_hz` | `50.0` | Teleop's ee_twist setpoint rate (Hz) |
| `stream_timeout_s` | `0.2` | The driver closes the stream after this long with no setpoint (s, > 1 / rate_hz) |
| `base_frame` | `"arm_mount"` | Parent frame of the plane detector's fridge TF |
| `fridge_frame` | `"fridge_frame"` | The plane detector's fridge frame, +z INTO the back wall |
| `verify_fridge_z` | `true` | Refuse a fridge frame whose +z points toward the arm (never flipped) |
| `driver_ee_frame` | `"arm_end_effector_link"` | The frame /ee_state reports (the flange) |
| `ee_link` | `"arm_tool_frame"` | The frame placed at the standoff, limited and oriented (rigid to the flange) |
| `static_tf_timeout_s` | `5.0` | Wait this long for driver_ee_frame -> ee_link (s) |
| `optical_axis` | `"+z"` | ee_link's axis that looks along fridge +z |
| `roll_reference_ee_axis` | `"+x"` | With roll_deg 0 this ee_link axis lines up with roll_reference_fridge_axis |
| `roll_reference_fridge_axis` | `"+x"` | See roll_reference_ee_axis |
| `roll_deg` | `180.0` | The camera's roll about the optical axis (deg) |
| `up_frame` | `"arm_mount"` | Frame of world up |
| `up_axis` | `"+z"` | World up in up_frame; must match the plane detector's fixed_frame +z |
| `fridge_tf_wait_s` | `5.0` | Homing waits this long for a fridge TF measured with the arm still (s) |
| `standoff_distance` | `0.4` | ee_link this far in front of the back wall at homing (m) |
| `offset_x` | `0.0` | Homing offset along fridge x (m) |
| `offset_y` | `0.0` | Homing offset along fridge y (m) |
| `scene_homing` | `true` | First move in front of the fridge the scene camera sees (skipped, warned, if it can't) |
| `scene_detections_topic` | `"/perception/object_detections"` | Scene segmenter detections |
| `scene_depth_topic` | `"/scene_camera/depth/image_rect_raw"` | Scene depth, aligned to colour |
| `scene_camera_info_topic` | `"/scene_camera/color/camera_info"` | Scene camera intrinsics |
| `scene_class` | `"fridge"` | The class to home on |
| `scene_min_score` | `0.5` | Detections below this score are ignored |
| `scene_box_point` | `[0.75, 0.5]` | The point homed on, as fractions [across, down] of the box |
| `scene_window_frac` | `0.1` | Depth window around it, as a fraction of the box |
| `scene_depth_percentile` | `50.0` | Percentile of the window's fridge-mask depths used |
| `scene_depth_range_m` | `[0.3, 3.0]` | Depths outside [min, max] are ignored (m) |
| `scene_min_samples` | `30` | At least this many valid depth pixels |
| `scene_frames` | `5` | Median of this many detections |
| `scene_wait_s` | `3.0` | Wait at most this long for them (s) |
| `scene_standoff_m` | `0.4` | ee_link this far in front of the scene point (m) |
| `scene_approach` | `"robot_forward"` | `robot_forward`, `camera_ray` or `from_arm`: the direction the camera looks |
| `scene_forward_axis` | `"+x"` | base_frame's forward axis, for robot_forward |
| `unwind_limits` | `true` | Retry a go_to refused for starting inside cuRobo's joint-limit margin after turning those joints out |
| `unwind_extra_rad` | `0.05` | Turn them this far past the margin (rad) |
| `unwind_max_rad` | `0.25` | Refuse an unwind larger than this (rad) |
| `twist_frame` | `"arm_mount"` | The axes of ee_twist setpoints |
| `max_speed` | `0.08` | Target speed on the wall at full stick (m/s) |
| `standoff_speed` | `0.05` | Target speed toward / away from the wall at full stick (m/s) |
| `kp_lin` | `2.0` | Servo position gain (1/s) |
| `kp_ang` | `2.0` | Servo orientation gain (1/s) |
| `max_lin` | `0.12` | Cap on the commanded linear speed (m/s) |
| `max_ang` | `0.4` | Cap on the commanded angular speed (rad/s) |
| `max_lead_m` | `0.05` | The target leads the arm by at most this (m) |
| `lead_slowdown_m` | `0.02` | Past this lead the target slows, to 0 at max_lead_m (m) |
| `u_limits` | `[-0.35, 0.35]` | Target limits left (+) / right of the homing origin (m) |
| `v_limits` | `[-0.3, 0.3]` | Target limits up (+) / down of the homing origin (m) |
| `standoff_limits` | `[0.08, 0.45]` | Target distance from the wall (m); contains standoff_distance |
| `follow_mode` | `"continuous"` | `off`, `idle` (stick at neutral only) or `continuous`: following the detector's re-estimates of the wall |
| `follow_settle_s` | `0.5` | An estimate (or detection) counts only this long after the arm last moved (s); also gates tap-to-pick's detections |
| `follow_stale_s` | `2.0` | Warn when no new estimate for this long (s) |
| `follow_w0` | `2.0` | Following filter's natural frequency (rad/s) |
| `follow_max_rate_deg_s` | `4.0` | Cap on the followed normal's turn rate (deg/s) |
| `follow_max_shift_rate` | `0.02` | Cap on the followed wall offset's rate (m/s) |
| `follow_deadband_deg` | `1.0` | Ignore estimates within this angle of the followed plane (deg) |
| `follow_deadband_m` | `0.005` | ... and within this offset (m) |
| `follow_max_tilt_deg` | `15.0` | Reject estimates tilted more than this from homing (deg) |
| `follow_max_shift_m` | `0.05` | Reject estimates shifted more than this from homing (m) |
| `follow_max_step_deg` | `5.0` | Reject a jump of more than this from the followed plane (deg) |
| `follow_max_step_m` | `0.03` | Reject a jump of more than this from the followed plane (m) |
| `joy_topic` | `"/joy"` | The joystick |
| `axis_u` | `0` | Axis index: left (+) / right |
| `axis_v` | `1` | Axis index: up (+) / down |
| `axis_standoff` | `-1` | Axis index: closer (+) / farther; -1 = off |
| `axis_signs` | `[1.0, 1.0, 1.0]` | Signs of [u, v, standoff] |
| `deadzone` | `0.1` | Stick deadzone (0..1) |
| `expo` | `2.0` | Stick expo (1 = linear) |
| `joy_timeout_s` | `0.25` | /joy older than this holds the target (s) |
| `fridge_tf_timeout_s` | `1.0` | Fridge TF older than this is not used (s; <= 0 disables; static TF exempt) |
| `ee_state_timeout_s` | `0.2` | /ee_state older than this sends a zero twist (s) |
| `search_when_no_fridge` | `true` | No fridge TF: search teleop (orientation held) until one appears; `false`: fail with no_fridge |
| `search_range_m` | `0.15` | Search box half-size around where it started (m) |
| `pick_enabled` | `true` | Tap-to-pick during `peer` (the goal's `pick` too). The `pick` action ignores it |
| `wrist_topics` | `["/wrist_camera/color/image_raw", "/wrist_camera/depth/image_rect_raw", "/wrist_camera/color/camera_info"]` | Wrist [colour, depth aligned to colour, camera_info] |
| `track_enabled` | `true` | SAM3 tracks food and drink on the wrist camera through `peer` and `pick` (one session, kept across the handoff); `false`: one-shot detections only |
| `track_prompts` | `["food", "drink"]` | What it tracks; their ids (`food_3`) are what a `pick`'s `label` names |
| `track_redetect_s` | `2.0` | Re-detect this often: new items get ids, tracked ones keep theirs (s, > 0) |
| `track_max_objects` | `10` | At most this many tracked |
| `track_lease_s` | `10.0` | The session's lease, renewed every 2 s (used only when the SAM3 node has `lease_enabled`) |
| `wrist_buffer_frames` | `180` | Wrist colour and depth frames buffered (each); SAM3's frame must still be in it when its answer comes (6 s at 30 Hz) |
| `pick_prompt` | `"food container"` | SAM3 prompt (and the UI label) for food containers |
| `pick_sam3_threshold` | `0.5` | SAM3 score threshold |
| `pick_detect_period_s` | `1.0` | Detections during teleop at most this often, only while the arm is still (s) |
| `ui_overlay_hz` | `10.0` | Rate of the wrist colour stream to the UI overlay (Hz; 0 = not published) |
| `pick_frames` | `3` | The pick's detection: this many consecutive agreeing masks |
| `pick_mask_iou` | `0.85` | ... agreeing at this mask IoU |
| `pick_timeout_s` | `20.0` | ... within this long (s) |
| `pick_select_iou` | `0.3` | The tapped container: the mask overlapping the tapped one by at least this |
| `pick_point` | `"top"` | `top` (top edge of the near face) or `nearest` (the microwave's nearest depths) |
| `pick_face_band_m` | `0.02` | `top`: the near face is this deep (m) |
| `pick_top_percentile` | `95.0` | `top`: the face's height percentile taken as its top edge |
| `pick_grasp_depth` | `0.0` | Tool frame this far past the point along the heading (m) |
| `pick_grasp_up` | `0.035` | Tool frame this far above the point (m). CHECK on the fridge |
| `pick_reach_in` | `0.3` | Pre-grasp this far back from the grasp: the straight line in (m) |
| `pick_grip_open` | `0.0` | Gripper opening for the reach in (0 open .. 1 closed) |
| `pick_world` | `"/config/robot/curobo/world.yaml"` | cuRobo world for the approach, resolved in the planner's container; must confirm flatbot's platform, seat, person and floor layers. `""`: whatever is loaded |
| `pick_speed_scale` | `0.3` | speed_scale of the approach (x the goal's `speed`) |
| `pick_approach_corridor_m` | `0.1` | Refuse an approach taking the tool farther than this off the straight line to pre-grasp (m) |
| `pick_line_speed` | `0.16` | Peak tool speed of the line in, the withdrawal and the leave (m/s) |
| `pick_impedance_in_m` | `0.1` | The last this much of the line in is in joint impedance (m) |
| `pick_impedance_speed` | `0.08` | Peak tool speed of that part (m/s) |
| `pick_settle_s` | `0.2` | Impedance hold before it (s) |
| `pick_impedance_kq` | `[400.0, 400.0, 400.0, 400.0, 200.0, 200.0, 200.0]` | Joint impedance stiffness, J1..J7 |
| `pick_impedance_zeta` | `0.7` | Joint impedance damping ratio |
| `pick_impedance_torque_limit` | `[39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0]` | Joint impedance torque limits, J1..J7 (Nm) |
| `pick_lift_m` | `0.03` | Straight lift after closing (m) |
| `pick_leave_back_m` | `0.2` | After the withdrawal, straight and level this much farther out of the fridge (m; 0 = none) |
| `pick_leave_to_joints` | `false` | Then cuRobo to pick_leave_joints (the container's tilt is not held) |
| `pick_leave_joints` | `[0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]` | Where to, J1..J7 (rad) |

`self.arm` reads these `arm.*` parameters once at startup and refuses changes afterwards. They are
set per robot like any other; `arm.mock_urdf` names the URDF `make_sim()` loads in mock runs (the
real driver supplies its own). `arm.sender_id` is the one left out of the real fragment: its
default is the engine id.

| Name | Default | Meaning |
| ---- | ------- | ------- |
| `arm.tool_z`                   | `0.12`                                             | Tool frame offset along the flange z (m)                   |
| `arm.dt`                       | `0.02`                                             | Trajectory sample spacing (s)                              |
| `arm.joint_speed`              | `0.85`                                             | Peak joint speed of joint moves (rad/s)                    |
| `arm.cart_speed`               | `0.16`                                             | Peak tool speed of Cartesian paths (m/s)                   |
| `arm.contact_joint_speed`      | `0.5`                                              | Peak joint speed of contact moves (rad/s)                  |
| `arm.contact_cart_speed`       | `0.09`                                             | Peak tool speed of contact moves (m/s)                     |
| `arm.min_duration_s`           | `1.0`                                              | Shortest trajectory duration (s)                           |
| `arm.cart_step_m`              | `0.02`                                             | Longest Cartesian step before IK (m)                       |
| `arm.cart_step_deg`            | `5.0`                                              | Largest Cartesian turn per step before IK (deg)            |
| `arm.rot_equiv_m`              | `0.15`                                             | Travel a 1 rad turn counts as when timing a path (m)       |
| `arm.max_knot_jump_deg`        | `20.0`                                             | Largest joint change between Cartesian knots (deg)         |
| `arm.path_tol_rad`             | `0.15`                                             | Per-joint tracking tolerance of trajectories (rad)         |
| `arm.gripper_force`            | `0.5`                                              | Default gripper force (0..1)                               |
| `arm.gripper_speed`            | `0.5`                                              | Default gripper speed (0..1)                               |
| `arm.frame`                    | `"arm_mount"`                                      | Frame of go_to_ee_pose goals and /ee_state                 |
| `arm.sender_id`                | the engine id (RammpArm standalone: `"microwave"`) | sender_id on goals (the engine id under the engine)        |
| `arm.go_to_speed_scale`        | `0.3`                                              | speed_scale of go_to goals (0.01..1)                       |
| `arm.go_to_timeout_s`          | `120.0`                                            | go_to planning plus execution limit (s)                    |
| `arm.go_to_track_m`            | `0.01`                                             | go_to_ee_pose: largest tool error at the end (m)           |
| `arm.go_to_track_deg`          | `2.0`                                              | go_to_joint_config: largest end joint error (deg)          |
| `arm.max_ik_err`               | `0.02`                                             | IK position error above which a pose is unreachable (m)    |
| `arm.max_step_jump_deg`        | `20.0`                                             | Largest joint change of a planned step (deg)               |
| `arm.j6_guard_deg`             | `115.0`                                            | Largest absolute J6 the sim accepts (deg)                  |
| `arm.j4_guard_deg`             | `144.0`                                            | Largest absolute J4 the sim accepts (deg)                  |
| `arm.max_continuous_delta_deg` | `170.0`                                            | Largest short-way step of a free joint (deg)               |
| `arm.keep_out_err`             | `9.0`                                              | IK error reported for a keep-out solution (m)              |
| `arm.ik_max_iterations`        | `200`                                              | PyBullet IK iterations per solve                           |
| `arm.clearance_query_m`        | `0.5`                                              | clearance() ignores bodies farther than this (m)           |
| `arm.fingertip_open_rad`       | `0.0`                                              | Sim finger joint angle with the gripper open (rad)         |
| `arm.finger_closed_rad`        | `0.7`                                              | Sim finger joint angle with the gripper closed (rad)       |
| `arm.state_wait_s`             | `10.0`                                             | First arm call: wait for the driver's state (s)            |
| `arm.state_stale_s`            | `0.5`                                              | Motion state older than this is an error (s)               |
| `arm.diagnostics_stale_s`      | `3.0`                                              | /diagnostics older than this is an error (s)               |
| `arm.goal_accept_s`            | `5.0`                                              | Wait for the driver to accept or reject a goal (s)         |
| `arm.result_margin_s`          | `30.0`                                             | A trajectory's result may come this long after its end (s) |
| `arm.poll_s`                   | `0.01`                                             | Poll period of a running motion (s)                        |
| `arm.cancel_settle_s`          | `0.4`                                              | Wait for a cancelled goal to settle (s)                    |
| `arm.halt_status_wait_s`       | `0.2`                                              | After a HALTED goal, wait for an e-stop status (s)         |
| `arm.planner_wait_s`           | `5.0`                                              | Wait for the cuRobo planner's action server (s)            |
| `arm.go_to_settle_s`           | `0.3`                                              | After a go_to goal, let the state catch up (s)             |
| `arm.gripper_period_s`         | `0.02`                                             | gripper() setpoint period (s)                              |
| `arm.gripper_moved`            | `0.002`                                            | gripper(): position change that counts as moving (0..1)    |
| `arm.gripper_judge_after_s`    | `0.5`                                              | gripper(): judge the fingers still only after (s)          |
| `arm.gripper_still_s`          | `0.3`                                              | gripper(): still this long means settled (s)               |
| `arm.plan_timeout_s`           | `30.0`                                             | Wait for a cuRobo plan (s)                                 |
| `arm.service_wait_s`           | `5.0`                                              | Wait for a service server to come up (s)                   |
| `arm.service_call_s`           | `10.0`                                             | Wait for a service's answer (s)                            |
| `arm.stream_timeout_s`         | `0.1`                                              | stream(): default setpoint timeout of a session (s)        |
| `arm.publisher_match_s`        | `1.0`                                              | Wait for a new publisher to reach the driver (s)           |
| `arm.close_stream_wait_s`      | `1.0`                                              | Wait for /close_stream's answer (s)                        |
| `arm.mock_joints`              | `[0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]`        | Mock mode: the joints it starts at (rad, J1..J7)           |
| `arm.mock_urdf`                | `""`                                               | Mock mode: the URDF file make_sim() loads (empty: none)    |

## Hardware

None of its own: the arm, gripper and cameras are the driver's and perception's.

## Alternatives

| id                         | What it is                                                                          |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `fridge_peering_mode_real` | The module proper                                                                   |
| `fridge_peering_mode_mock` | Same interface, no hardware: every action returns `{"executed": false, "mock": true}` |
