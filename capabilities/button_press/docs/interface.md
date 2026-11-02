# Button Press Capability — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @EvuhLi
**Tier:** experimental

## What it does

Presses one button on the microwave's control panel with the Kinova arm. Its one action,
`press_button`, finds the microwave (scene camera first, then the wrist camera's dome detector),
stages in front of the panel with a cuRobo move, re-detects the button from about 25 cm, plans and
gates every straight leg in a kinematic sim, then goes to a pre-press spot, presses by touch (the
driver's `ee_twist` stream, stopped by the wrist torques), and returns to where the arm started.
It is activated when the user selects a `microwave`. Without it, no kitchen task can start the
microwave.

**The arm moves only with `allow_execute: true`.** Otherwise every action is a dry run: detect,
plan and gate, nothing commanded, and the result says `dry_run: true`.

A check that refuses a leg before it is sent backs the arm off to its last safe pose (the start,
or the pre-press spot). A motion the driver fails (an e-stop, control lost, a tracking abort) or a
cancel leaves the arm holding where it stopped, and the action ends there. The approach, the press
and the return are not pausable: a cancel during them can leave the fingertip at the panel.

## Publishes

| Topic                       | Type                        | Rate            | Meaning                                                                                                   |
| --------------------------- | --------------------------- | --------------- | --------------------------------------------------------------------------------------------------------- |
| `/button_press/status`      | `std_msgs/String`           | 1 Hz            | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |
| `/button_press/target_pose` | `geometry_msgs/PoseStamped` | once per action | The planned pre-press tool pose, for viewing                                                              |

Through `self.arm` it uses the arm driver's state topics, `/execute_joint_trajectory`,
`/go_to_ee_pose`, the `ee_twist` stream and `/setpoint/gripper`, only while an action runs.

## Subscribes

Only while an action runs; the scene camera and the detections only during the first look.

| Topic                                                                                                                 | Type                                               | Required | Meaning                                             |
| --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | -------- | --------------------------------------------------- |
| `/tf`, `/tf_static`                                                                                                   | `tf2_msgs/TFMessage`                               | yes      | Arm and camera mounts into `arm_mount`              |
| `/wrist_camera/color/{image_raw,camera_info}`                                                                         | `Image`, `CameraInfo`                              | yes      | Dome detection                                      |
| `/wrist_camera/<depth_topic>`                                                                                         | `sensor_msgs/Image`                                | yes      | Depth **aligned to colour**                         |
| `/scene_camera/color/{image_raw,camera_info}`, `/scene_camera/<scene_depth_topic>`, `/scene_camera/depth/camera_info` | `Image`, `CameraInfo`                              | no       | First look; without it the wrist camera looks first |
| `/perception/object_detections`                                                                                       | `rammp_perception_interfaces/msg/ObjectDetections` | no       | The scene segmenter's microwave narrows the search  |

## Orchestrator interface

Served under `/button_press/`, types from `rammp_orchestrator_interfaces`:

| Name                           | Type                   | Meaning                                       |
| ------------------------------ | ---------------------- | --------------------------------------------- |
| `/button_press/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/button_press/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/button_press/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/button_press/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

| Action         | Params                                                                                                                                                                              | Result payload                                            |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `press_button` | `target` (`start_30s` default, `timer_clock`, `stop_eco`, `power_level`, `wgt_time_defrost`), `presses` (0–10, default 1; 0 = go to the spot and back), `no_return` (default false) | `{"dry_run", "pressed", "returned", "moved", "log_file"}` |

A failed action's `error_code` is `<step>_failed`, the step being `preflight`, `detect`, `stage`,
`redetect`, `plan`, `approach`, `press` or `return`; the node's log says why ("HOLDING HERE": the
arm holds where it stopped). After a failure in `approach`, `press` or `return` the fingertip may
be at the panel, so those are never `recoverable`: a retry could press the button again, and a
cuRobo move from there could hit the microwave (it is not in cuRobo's world). The same goes for a
cancel during those steps. Move the arm clear by hand before the next action.

```bash
make action CAP=button_press ACTION=press_button PARAMS='{"target": "start_30s"}'
```

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment (flatbot's values). A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy passes them as a
params file.

| Name                | Default (fragment)     | Meaning                                                              |
| ------------------- | ---------------------- | -------------------------------------------------------------------- |
| `use_orchestrator`  | `true`                 | `false` runs standalone: no registration, drive it by CLI            |
| `mock`              | `false`                | `true` skips hardware (the mock alternative sets it)                 |
| `allow_execute`     | `false`                | `true` lets the action move the arm. Attended runs only              |
| `mount_rpy_deg`     | `[0.0, 30.0, 0.0]`     | The arm base's orientation in `arm_mount` (the robot's `mount.yaml`) |
| `camera_frame`      | `""`                   | Wrist camera optical frame; `""` = the colour image's own `frame_id` |
| `camera_ns`         | `/wrist_camera`        | Wrist camera namespace                                               |
| `depth_topic`       | `depth/image_rect_raw` | Wrist depth under `camera_ns`, aligned to colour                     |
| `far`               | `scene`                | First look: `scene` (scene camera, falls back to wrist) or `wrist`   |
| `scene_ns`          | `/scene_camera`        | Scene camera namespace                                               |
| `scene_depth_topic` | `depth/image_rect_raw` | Scene depth under `scene_ns`                                         |
| `detections`        | `true`                 | Narrow the scene search to the segmenter's microwave box             |
| `log_dir`           | `""`                   | A JSONL run log per action in this directory; `""` = none            |

Per-rig constants that are not parameters yet: `PREPRESS_EE_OFFSET_M` / `PREPRESS_EE_QUAT_PANEL`
in `press_button.py`, and the scene finder's `FACE_CORNER_TO_START_MM` (refine with
`tools/scene_offset_from_logs.py` from run logs).

`self.arm` reads these `arm.*` parameters once at startup and refuses changes
afterwards. They are set per robot like any other. The fragment sets
`arm.go_to_track_m` to 0.05: the long cuRobo moves only need to end near their goal.

| Name                           | Default                                            | Meaning                                                    |
| ------------------------------ | -------------------------------------------------- | ---------------------------------------------------------- |
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
| `arm.go_to_track_m`            | `0.05`                                             | go_to_ee_pose: largest tool error at the end (m)           |
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

No devices of its own: it reaches the arm, the gripper and both cameras through their drivers'
topics. It needs the arm with a closed 2F-85 gripper (preflight closes an open one on an execute
run), a wrist RGB-D camera with depth aligned to colour (from ~22 cm), and optionally a fixed scene
camera seeing the microwave's front.

**Per robot:** the camera mounts in tf (rammp-deployments `transforms/`), `mount_rpy_deg`, and the
pre-press offset measured on that robot against those mounts.

## Alternatives

| id                  | What it is                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------ |
| `button_press_real` | The module proper                                                                          |
| `button_press_mock` | Same interface, no hardware — every action succeeds at once as a dry run, nothing is moved |
