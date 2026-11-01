# Fridge Capability — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @dhyi-svg
**Tier:** experimental

## What it does

Opens and closes the fridge door with the Kinova arm. On an `open` goal it finds the handle from
the wrist camera (door-plane fit over the whole image → the vertical bar sticking out of it),
grasps the handle squared to the door, swings the door open by exactly the asked angle, lets go,
backs off and goes back round the door's free edge to its inner side, facing the fridge, always
ending upright. A `close` goal pushes the door to about 45° with the side of the hand and it
swings shut by itself. Every leg is planned and sim-gated before anything moves; a failed check
stops the run with the arm holding where it is. Without it, no kitchen task can reach anything
in the fridge.

**The arm moves only if the node runs with `allow_execute: true` and the goal sets
`execute`.** Otherwise every goal is a dry run: detect, plan and gate, nothing commanded.

cuRobo does the free-space move to the line-up point in front of the handle. Everything on or
near the door (straight in, the swing, the route round the door, the push-open, the close) stays
on joint paths planned and checked here, against a box model of the door and cabinet, and those
checked joints are what is sent.

The open saves the door's geometry (`<data_dir>/door.json`) for the close; every run is logged
to `<data_dir>/runs/<time>_open/` (events, every trajectory sent, arm state at 25 Hz, camera
frames) and every detection frame to `<data_dir>/detections/`.

## Publishes

| Topic            | Type              | Rate | Meaning                                                                                                   |
| ---------------- | ----------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| `/fridge/status` | `std_msgs/String` | 1 Hz | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |

## Subscribes

| Topic                                            | Type                     | Required | Meaning                                     |
| ------------------------------------------------ | ------------------------ | -------- | ------------------------------------------- |
| `/wrist_camera/color/image_raw`                  | `sensor_msgs/Image`      | yes      | Handle detection (`color_topic`)            |
| `/wrist_camera/color/camera_info`                | `sensor_msgs/CameraInfo` | yes      | Intrinsics (`camera_info_topic`)            |
| `/wrist_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image`      | yes      | Depth **aligned to colour** (`depth_topic`) |
| `/tf`, `/tf_static`                              | `tf2_msgs/TFMessage`     | yes      | Camera pose in `arm.frame` (hand-eye calib) |

The camera subscriptions exist only while an action runs; the engine destroys them when it ends.
`self.arm` subscribes to the arm driver's state topics the same way. The SAM 3 client
(`handle_source: sam3`) is made once and lives as long as the node.

## Orchestrator interface

Served under `/fridge/`, types from `rammp_orchestrator_interfaces`:

| Name                     | Type                   | Meaning                                       |
| ------------------------ | ---------------------- | --------------------------------------------- |
| `/fridge/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/fridge/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/fridge/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/fridge/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

| Action  | Params                                                                                                                                                                                                                                                                                                               | Result payload                                     |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `open`  | `execute` (bool, default false), `speed` ((0, 1\], default 1.0: scales every arm motion), `swing_deg` (30–110; default 90, or 50 with the push-open: exactly this swing or it refuses), `push_open` (bool, default false), `release_only` (bool, default false: let go and back off the handle, after a stopped run) | `{"executed": bool, "note": str, "runs_dir": str}` |
| `close` | `execute`, `speed` as above, plus the settings below                                                                                                                                                                                                                                                                 | `{"executed": bool, "note": str, "runs_dir": str}` |

`close` needs the door file the open saved. The gripper is never touched (it may hold a container)
and the hand never tilts: every planned configuration keeps its roll and pitch within 2° of where it
starts, or the close refuses before moving.

| Setting                           | Default         | Meaning                                                                                                     |
| --------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------- |
| `chair_back`                      | `0.30`          | How far the wheelchair is moved back first (m); the plan shifts the recorded door by it. `0`: not moved     |
| `chair_moved`                     | `false`         | The wheelchair has been moved back. Required to execute with `chair_back` > 0; a dry run plans as if it had |
| `close_to_deg`                    | `45.0`          | Swing the door shut until it is open this far (deg); it swings shut by itself from there                    |
| `door_deg`                        | door file       | How far the door is open now (deg). Give it if the door was moved by hand                                   |
| `push_radius` / `push_radius_end` | `0.30` / `0.28` | The hand's distance from the hinge during the swing, start / end (m)                                        |
| `push_z`                          | searched        | Height of the swing (m); refused outside the door. Default: the first height within it that passes          |
| `hold_offset_x`                   | `-0.1651`       | Plan with the door shifted in x (m): the 6.5 in container. `0` if not holding it                            |
| `side_past_hinge`                 | `0.08`          | Swing start, sideways past the hinge (m)                                                                    |
| `retreat_dist`                    | `0.20`          | Back straight out up to this far after the swing (m)                                                        |

Failures: `refused` (a check failed; the message says which and the arm holds), `stopped`
(a motion stopped short), `bad_params`. On an executed success it marks the object `opened`
(true after `open`, false after `close`). Nothing measures whether the door shut: watch it.

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until
the orchestrator is up.

## Before a run

- **Start pose.** Nothing moves the camera to the fridge. The arm starts pointed at the closed door,
  roughly square-on, 50–60 cm from it, with the handle in view and the door filling most of the
  image.
- **Placement.** The open prints where the wheelchair should go and refuses with a suggestion when
  it does not fit. At the placement that passed (handle x 0.50–0.56, y −0.45 to −0.39), a 90° door's
  free edge is ~10 cm in front of the arm's base, so the hand passes right beside the base (~0.67 m
  up) and sweeps across in front of it. The sim has no wheelchair or person in it: keep that space
  clear.
- **Measure** a new fridge: the `*_m` parameters below. A wrong hinge drags the door or the fridge
  during the swing.
- **Watch where the fingers close:** `GRIP_EXT` is the microwave's and nothing checks the grasp on
  this handle. If the fingers hit the door or close short of the bar, change it (bigger = shorter).

How to run it on the arm: [TESTING.md](TESTING.md).

## Known limits

- **Not on flatbot yet.** The keep-out is a stand-in for an obstacle zone: it is checked on every
  plan and in `move_joints`, but on `go_to_*` only at the goal, and cuRobo's path is not previewed.
  flatbot's cuRobo world is empty on rammp-deployments dev (the chair world is on the unmerged
  `feature/flatbot-curobo-chair`). No fridge motion on flatbot until that is resolved.
- **The 30° mount is sim-checked only.** Set `mount_tilt_deg: 30`. The planning runs in `arm_mount`,
  which is level on every robot; the parameter sets the reach limits and placement advice. All
  hardware runs were on a level arm. On the 30° model the 90° open and the close plan with the
  fridge ~14–22 cm further from the arm than on the level one, and the push-open refuses there.
  The wheelchair keep-out's size was measured on the level rig.
- **The wheelchair is moved by a person.** An action can't prompt, so `chair_moved` stands in for
  the old script's prompt. A `MOVE_BASE` intent (`ctx.request_intent`) could replace it once an
  engine handles one.
- **`handle_source: sam3` is unmeasured here.** Its grasp offsets were never measured on this
  fridge: re-measure before using it. Depth stays the default, though the OAK may struggle on glossy
  handles.
- **The push-open does not keep the door open.** Its one hardware run (2026-10-24) finished its
  trajectory and the door swung shut.
- **The close has hit the handle.** On 2026-10-24 the hand shut the door by hitting the handle, not
  the door face, and carried on with the planned swing after the door had shut.
- **The close has hit the wheelchair.** Before the keep-out, a close ran the arm into the back of
  the chair. It has not been run on the arm since.
- **The container is not in the collision model.** The plan only keeps the hand back by its size.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file.

| Name                      | Default                                          | Meaning                                                                                  |
| ------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `use_orchestrator`        | `true`                                           | `false` runs standalone: no registration, drive it by CLI                                |
| `mock`                    | `false`                                          | `true` skips hardware (the mock alternative sets it)                                     |
| `allow_execute`           | `false`                                          | `true` lets execute goals move the arm. Attended runs only                               |
| `data_dir`                | `/fridge_data`                                   | Door file, grasp record, run logs, detection frames (a host volume)                      |
| `color_topic`             | `/wrist_camera/color/image_raw`                  | Wrist colour image                                                                       |
| `camera_info_topic`       | `/wrist_camera/color/camera_info`                | Wrist camera intrinsics                                                                  |
| `depth_topic`             | `/wrist_camera/aligned_depth_to_color/image_raw` | Wrist depth, aligned to colour                                                           |
| `mount_tilt_deg`          | `0`                                              | The arm's forward lean in its mount: `0` (dojo) or `30` (flatbot, gen_1_5)               |
| `camera_frame`            | `""`                                             | Camera optical frame; `""` = the colour image's frame_id                                 |
| `handle_source`           | `depth`                                          | `depth` (bar sticking out of the door) or `sam3` (SAM 3's handle mask)                   |
| `sam3_service`            | `/perception/sam3/detect`                        | SAM 3's Detect service (`handle_source: sam3`)                                           |
| `sam3_prompts`            | `[handle]`                                       | SAM 3 text prompts for the handle; the best-scoring instance is used                     |
| `sam3_threshold`          | `0.3`                                            | SAM 3 score threshold                                                                    |
| `handle_centre_to_face_m` | `0.053`                                          | Middle of the handle bar to the door's front surface, perpendicular to it (m)            |
| `pin_behind_face_m`       | `0.04675`                                        | Door's front surface to the hinge pin's axis, perpendicular to it (m)                    |
| `hinge_along_door_m`      | `0.3683`                                         | Handle bar centre line to the hinge pin's axis, along the door at handle height (m)      |
| `door_width_m`            | `0.4572`                                         | Hinge-side edge to free edge, measured at the door's top (m)                             |
| `door_depth_m`            | `0.10668`                                        | Front face to the innermost point of its inside, bins and shelves included (m)           |
| `door_above_handle_m`     | `0.3175`                                         | Middle of the handle bar up to the door's top (m)                                        |
| `door_below_handle_m`     | `0.4572`                                         | Middle of the handle bar down to the door's bottom (m)                                   |
| `body_depth_m`            | `0.5`                                            | Cabinet behind the door (m)                                                              |
| `chair_keep_out_behind_m` | `0.381`                                          | No part of the arm may be further than this behind the arm base while on its right (m)   |
| `chair_back_dir`          | `[1.0, 0.0]`                                     | Which way moving the chair back moves the fridge in the arm's frame (x, y): forward = +x |

Per-rig constants that are not parameters yet, in `fridge/constants.py`: the grasp corrections
(`GRIP_EXT`, `GRASP_X_ADJUST`, `VERTICAL_CORR`), the handle gates and the routes round the door. `GRASP_PITCHES_DEG` is `door_common.open_door`'s.

`self.arm` reads these `arm.*` parameters once at startup and refuses changes
afterwards. They are set per robot like any other; `arm.mock_urdf` names the
URDF `make_sim()` loads in mock runs (the real driver supplies its own).
`arm.sender_id` is the one left out of the real fragment: its default is the
engine id, so each capability sends its goals under its own name.

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

No devices of its own: it reaches the arm, the gripper and the wrist camera through their
drivers' topics. It needs the arm with a 2F-85 gripper, a wrist RGB-D camera with depth aligned
to colour, and a calibrated camera tf (hand-eye) into `arm.frame`. It needs no GPU.

## Alternatives

| id            | What it is                                                                         |
| ------------- | ---------------------------------------------------------------------------------- |
| `fridge_real` | The module proper                                                                  |
| `fridge_mock` | Same interface, no hardware — for developing against this module without owning it |
