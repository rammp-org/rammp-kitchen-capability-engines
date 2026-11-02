# Microwave Place — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @seanliyucheng
**Tier:** experimental

## What it does

Puts the container the gripper is already holding (after `grab_container`) into the open microwave. It
finds the microwave's opening with the scene camera (SAM3 masks and depth), models the oven and the open
door as obstacle boxes, picks a hand heading the arm can reach, plans cuRobo's level transit to in front
of the opening and a straight line in, checks all of it on the arm model, lowers the container onto the
turntable glass in impedance until the glass stops it, lifts slightly, opens the gripper part way and
backs out along the line it came in on. Without it the microwave demo has no way to put the food in.

## Publishes

| Topic                     | Type              | Rate | Meaning                                                                                                   |
| ------------------------- | ----------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| `/microwave_place/status` | `std_msgs/String` | 1 Hz | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |

## Subscribes

| Topic                                | Type                                           | Required | Meaning                                          |
| ------------------------------------ | ---------------------------------------------- | -------- | ------------------------------------------------ |
| `/tf`, `/tf_static`                  | `tf2_msgs/TFMessage`                           | yes      | The scene camera's pose in `arm_mount`           |
| `/scene_camera/depth/image_rect_raw` | `sensor_msgs/Image`                            | yes      | Depth aligned to color, paired with SAM3's frame |
| `/scene_camera/color/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Intrinsics                                       |
| `/perception/object_detections_3d`   | `rammp_perception_interfaces/ObjectDetections` | no       | Declared for the locate topics; not used         |

It also calls `/perception/sam3/detect` (`rammp_perception_interfaces/srv/Detect`, prompts "inside of
microwave" and "microwave" on `/scene_camera/color/image_raw`), and through `self.arm` the arm driver and
the cuRobo planner (`plan_to_pose` with `HOLD_LEVEL`, `set_world` with a world file in `worlds_dir`).

## Orchestrator interface

Served under `/microwave_place/`, types from `rammp_orchestrator_interfaces`:

| Name                              | Type                   | Meaning                                       |
| --------------------------------- | ---------------------- | --------------------------------------------- |
| `/microwave_place/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/microwave_place/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/microwave_place/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/microwave_place/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

| Action    | Params                                                           | Result payload                                                                  |
| --------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `place`   | `execute` (default false), `allow_empty_gripper` (default false) | `{"executed": bool, "plan": {...}, "note": str}`; with execute also `recording` |
| `retreat` | `execute` (default false)                                        | `{"executed": bool, "lift_cm", "clearance_cm", "to", "note"}`                   |
| `reset`   | `execute` (default false), `save`, `empty`                       | `{"executed": bool, "plan": [steps]}` or `{"saved": true}`                      |

The arm moves only when the node runs with `allow_execute:=true` and the goal sets `execute`; otherwise
every goal is a dry run that plans, checks and returns the plan. A refusal fails the goal with error code
`refused` (recoverable) and the reason.

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until the orchestrator is up.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment.

| Name                                                       | Default                                       | Meaning                                                                   |
| ---------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------- |
| `use_orchestrator`                                         | `true`                                        | `false` runs standalone: no registration, drive it by CLI                 |
| `mock`                                                     | `false`                                       | `true` skips hardware (the mock alternative sets it)                      |
| `allow_execute`                                            | `false`                                       | `true` lets `execute` goals move the arm (attended runs only)             |
| `state_dir`                                                | `/config/microwave_place`                     | Plans, the start position, the descent recordings                         |
| `worlds_dir`                                               | `/home/snorlax/.ros/rammp_box_opening/worlds` | cuRobo's world files; the planner's container sees it at the same path    |
| `frames`, `max_frames`                                     | `3`, `5`                                      | Locate: `frames` of up to `max_frames` detected frames must agree         |
| `agree_m`, `agree_deg`                                     | `0.01`, `2.0`                                 | ... within this of their median                                           |
| `timeout_s`                                                | `20.0`                                        | Locate (and the "microwave" frame): give up after this (s)                |
| `microwave_prompt`                                         | `"microwave"`                                 | SAM3's prompt for the oven's outside                                      |
| `above`                                                    | `0.08`                                        | The tool over the glass at the end of the line in (m)                     |
| `heading_offsets_deg`                                      | `[0, 5, -5, … 40, -40]`                       | The hand's headings off the oven's axis, tried in order                   |
| `clear_min`                                                | `0.01`                                        | Every arm and gripper link from every obstacle box (m)                    |
| `tilt_max_deg`                                             | `2.0`                                         | The hand's tilt vs the start                                              |
| `check_container`                                          | `false`                                       | Also keep the held box `clear_min` from the obstacles                     |
| `transit_speed`                                            | `1.0`                                         | The transit at this fraction of cuRobo's timing                           |
| `insert_speed`                                             | `0.16`                                        | The line in, the release lift and the back-out (m/s)                      |
| `lower_speed`                                              | `0.02`                                        | The lowering's speed where the glass should be (m/s; contact tuned at it) |
| `vz_stop`, `hold_press`, `settle_s`, `early_stop`          | `-0.01`, `0.003`, `3.0`, `0.03`               | The lowering's contact test and hold                                      |
| `impedance_kq`, `impedance_zeta`, `impedance_torque_limit` | the driver's `rammp_arm.IMPEDANCE`            | Joint impedance for the lowering and reset                                |
| `reset_speed_scale`                                        | `1.0`                                         | speed_scale of reset's joint moves                                        |
| `flange_to_tip`, `pitch_deg`                               | `0.17955`, `5.0`                              | The hold model                                                            |
| `tool_frame_tol`                                           | `0.015`                                       | The arm's tool frame vs the model's (m)                                   |
| `depth_topic`, `camera_info_topic`, `detections_topic`     | the scene camera's                            | Locate's topics                                                           |

`self.arm` reads the `arm.*` parameters once at startup; they are listed with their defaults in the real
fragment and described in the example capability's interface page.

## Hardware

The Kinova Gen3 arm with the Robotiq 2F-85 gripper and the scene camera, through the arm driver, the
cuRobo planner and rammp-perception's SAM3 node. No devices of its own.

## Alternatives

| id                     | What it is                                                                         |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `microwave_place_real` | The module proper                                                                  |
| `microwave_place_mock` | Same interface, no hardware — for developing against this module without owning it |
