# Microwave Place — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @seanliyucheng
**Tier:** experimental

## What it does

Puts the container the gripper is already holding (after `grab_container`) into the open microwave. It
finds the microwave's opening with the scene camera (the known-size opening fitted to SAM3's cavity mask,
checked with depth), models the microwave's body as one obstacle box, tries side shifts and hand headings
until one the arm can reach passes, plans a side shift and cuRobo's level transit to in front of the
opening (both checked against the box on the arm model) and a straight line in, and runs the three as one
position-control motion. It lowers the container onto the turntable glass in impedance until the glass
stops it, opens the gripper part way in place, pitches the hand 10 deg nose-down so the container slides
off the fingers, and backs out level with the hand tilted. Nothing from the line in on is
collision-checked. Without it the microwave demo has no way to put the food in.

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

| Name                                                       | Default                                       | Meaning                                                                                                                             |
| ---------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `allow_execute`                                            | `false`                                       | `true` lets `execute` goals move the arm (attended runs only)                                                                       |
| `state_dir`                                                | `/config/microwave_place`                     | Plans and the descent recordings                                                                                                    |
| `worlds_dir`                                               | `/home/snorlax/.ros/rammp_box_opening/worlds` | cuRobo's world files; the planner's container sees it at the same path                                                              |
| `frames`                                                   | `3`                                           | Locate: this many consecutive detected frames must agree                                                                            |
| `agree_m`, `agree_deg`                                     | `0.01`, `2.0`                                 | ... within this of their median                                                                                                     |
| `timeout_s`                                                | `20.0`                                        | Locate (with the "microwave" frame): give up after this (s)                                                                         |
| `microwave_prompt`                                         | `"microwave"`                                 | SAM3's prompt for the oven's outside (the obstacle)                                                                                 |
| `above`                                                    | `0.055`                                       | The tool over the glass at the end of the line in (m)                                                                               |
| `closer`                                                   | `0.02`                                        | The spot this far back from the turntable's middle toward the opening, along the oven's axis (m); the pre-insert pose moves with it |
| `side_shifts_cm`                                           | `[3, -3, 6, -6, 9, -9]`                       | The hand moved this far level to its left (- = right) before the transit, tried in order                                            |
| `heading_offsets_deg`                                      | `[0.0]`                                       | The hand's headings off the oven's axis (+ = left), tried in order with each side shift                                             |
| `clear_min`                                                | `0.01`                                        | The side shift and the transit: every arm and gripper link from the microwave box (m)                                               |
| `tilt_max_deg`                                             | `2.0`                                         | The hand's tilt vs the start, on the transit and the lines                                                                          |
| `insert_speed`                                             | `0.16`                                        | Peak speed of the approach (side shift + transit + line in, one motion) and the back-out (m/s)                                      |
| `lower_speed`                                              | `0.12`                                        | The lowering's speed where the glass should be (m/s; contact tuned at it)                                                           |
| `vz_stop`, `hold_press`, `settle_s`, `early_stop`          | `-0.01`, `0.003`, `0.2`, `0.03`               | The lowering's contact test and hold                                                                                                |
| `impedance_kq`, `impedance_zeta`, `impedance_torque_limit` | the driver's `rammp_arm.IMPEDANCE`            | Joint impedance for the lowering, the gripper opening and the release tilt                                                          |
| `flange_to_tip`                                            | `0.17955`                                     | Flange face to the closed fingertips (m)                                                                                            |
| `depth_topic`, `camera_info_topic`, `detections_topic`     | the scene camera's                            | Locate's topics (the 3D detections are not used)                                                                                    |

The capability engine itself also reads `use_orchestrator` (default `true`; `false` runs standalone: no
registration, drive it by CLI) and `mock` (default `false`; `true` skips hardware, the mock alternative
sets it).

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
