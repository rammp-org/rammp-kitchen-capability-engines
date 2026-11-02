# Box Opening Capability — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @ChrissCox
**Tier:** experimental

## What it does

Opens the food container: presses the button on its lid so the lid comes
loose, grips the lid by that button, lifts it off, sets it down on the table
beside the container, and goes home. `open`:

1. **Locate:** the scene camera finds the container before the arm moves (the
   OWL detector, or the scene segmenter's `food container` mask where it has
   one), with its lid's height from the scene camera's depth. Nothing about the
   table or the container's place is configured: both come from the cameras on
   every run.
1. **Look:** the arm moves to a pose looking down at the lid at 45 degrees, and
   the wrist camera aims at the button.
1. **Press:** the fingers, closed, press the button in joint impedance and feel
   where the lid is.
1. **Grip:** straight from the press, the fingers rise, open, come down around
   the button and close on it. A grip that misses goes back to the look pose and
   tries again from there, a centimetre lower.
1. **Carry:** the lid is lifted, carried and set down beside the container in
   one motion, at a spot the scene camera sees is clear.
1. **Release and home:** the fingers open as the arm rises off the lid, and the
   arm goes home.

`press` is the first three steps and home: the button pops, the lid stays on.
`home` moves the arm home from wherever it holds, stepping a joint off its
limit first where the planner cannot start from it. `check` reports whether the
robot is ready for a run (driver, planner, TF, gripper, both cameras and the
OWL detectors) and moves nothing.

Each action runs box opening's command line for it (`press_demo`, `home_arm`,
`preflight`) as a child process, the way it ran on flatbot until now. Its
console lines are this node's log lines, and its run log's rows are the
action's progress. Planning and motion go straight to the arm driver
(`/execute_joint_trajectory`, `/setpoint/gripper`) and the cuRobo planner
(`/rammp_curobo/plan_to_pose`, `plan_to_joints`), not yet through `self.arm`.

The arm moves only when the node runs with `allow_execute: true` and the goal
sets `execute`. Otherwise the run is a dry run: it finds the container, plans
and checks every motion, and sends none.

## Publishes

| Topic                 | Type              | Rate | Meaning                                                                                                   |
| --------------------- | ----------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| `/box_opening/status` | `std_msgs/String` | 1 Hz | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |

Inside the container, the run and the two OWL detectors also talk over
`/box_opening/owl_*` (the detectors' boxes, and when to look), and with
`viz: true` the run draws what it sees on `/box_opening/viz/*` for Foxglove.

## Subscribes

While an action runs; the OWL detectors read the cameras from the node's start.

| Topic                                | Type                                           | Required | Meaning                                                    |
| ------------------------------------ | ---------------------------------------------- | -------- | ---------------------------------------------------------- |
| `/joint_states`                      | `sensor_msgs/JointState`                       | yes      | The arm's joints, efforts and gripper knuckle              |
| `/gripper_state`                     | `rammp_arm_interfaces/GripperState`            | `check`  | The gripper is present                                     |
| `/arm/robot_description`             | `std_msgs/String`                              | yes      | The arm's URDF, rooted at `arm_mount` (the run's FK chain) |
| `/wrist_camera/color/image_raw`      | `sensor_msgs/Image`                            | yes      | The aim at the button, the grip's close-up                 |
| `/wrist_camera/color/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Its intrinsics                                             |
| `/wrist_camera/depth/image_rect_raw` | `sensor_msgs/Image`                            | yes      | Depth aligned to colour                                    |
| `/scene_camera/color/image_raw`      | `sensor_msgs/Image`                            | yes      | Finding the container, the lid's spot                      |
| `/scene_camera/color/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Its intrinsics                                             |
| `/scene_camera/depth/image_rect_raw` | `sensor_msgs/Image`                            | yes      | Scene depth aligned to colour                              |
| `/scene_camera/depth/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Its intrinsics                                             |
| `/perception/object_detections`      | `rammp_perception_interfaces/ObjectDetections` | no       | The scene segmenter's masks, used where they name the box  |
| `/tf`, `/tf_static`                  | `tf2_msgs/TFMessage`                           | yes      | The finger tips and both cameras in `arm_mount`            |

## Orchestrator interface

Served under `/box_opening/`, types from `rammp_orchestrator_interfaces`:

| Name                          | Type                   | Meaning                                       |
| ----------------------------- | ---------------------- | --------------------------------------------- |
| `/box_opening/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/box_opening/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/box_opening/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/box_opening/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

It is activated when the user selects a `food container`. `open` and `press`
are offered while the container's lid is on; `home` and `check` always.

| Action  | Params                                                                   | Result payload                                                                                     |
| ------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `open`  | `execute` (bool, default `false`)                                        | `{"executed", "lid_off", "lid_xyz", "presses", "grips", "duration_s", "video", "run_log", "note"}` |
| `press` | `execute` (bool, default `false`)                                        | the same; `lid_off` is always false                                                                |
| `home`  | `execute` (bool, default `false`), `lift_first_m` (0 to 0.15, default 0) | the same, without the lid                                                                          |
| `check` | none                                                                     | the same, and `checks`: each PASS, WARN and FAIL line                                              |

`lid_xyz` is where the lid was set down, in `arm_mount` (m). `video` is the MP4
of both cameras from the grip to the set-down, where one was recorded.
`lift_first_m` lifts the tool straight up that far before planning home, for a
start the planner refuses (fingers at the table). When the lid comes off, the
container's object state gets `lid: "off"` and `lid_xyz`.

Progress phases, in order: `locate` (5%), `look` (10%), `press` (35%), `grip`
(50%), `carry` (85%), `release` (90%), `home` (100%). A phase is reported as
its motion ends. A run is one motion from the press to the set-down, so it
cannot pause once it has started (every phase reports `pausable: false`); a
cancel stops it at once. The cancel is the run's own Ctrl+C: the motion in
flight is cancelled, and the action ends once the driver has confirmed it.

A run that stops fails the action with one of these codes:

| Code        | Recoverable | Meaning                                                                |
| ----------- | ----------- | ---------------------------------------------------------------------- |
| `not_ready` | yes         | Refused before anything moved: a driver, planner, camera or TF missing |
| `not_found` | yes         | The scene camera found no container                                    |
| `not_home`  | yes         | The arm could not get to its start pose                                |
| `box_moved` | yes         | The press moved the container                                          |
| `arm_holds` | no          | A motion stopped part of the way: the arm holds there; send `home`     |
| `stopped`   | no          | The run was ended from outside                                         |
| `failed`    | yes         | Anything else; the message is the run's last STOP line                 |

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until
the orchestrator is up.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file.

| Name               | Default                | Meaning                                                                      |
| ------------------ | ---------------------- | ---------------------------------------------------------------------------- |
| `use_orchestrator` | `true`                 | `false` runs standalone: no registration, drive it by CLI                    |
| `mock`             | `false`                | `true` skips hardware (the mock alternative sets it)                         |
| `allow_execute`    | `false`                | `false`: every goal is a dry run, whatever it asks                           |
| `container`        | `"ankou_pink.yaml"`    | The container's config, in the image's `config/containers`, or a path        |
| `world`            | `"world_flatbot.yaml"` | The robot's planning frame, cameras and fixed body, in the image's `config`  |
| `state_dir`        | `""`                   | Where runs keep their logs, captures and videos; `""` = `~/.ros/box_opening` |
| `owl`              | `true`                 | Run the wrist's and the scene camera's OWL detectors in this node's process  |
| `video`            | `true`                 | Record both cameras from the grip to the set-down                            |
| `viz`              | `false`                | Draw what the perception sees and decides for Foxglove                       |
| `speed_scale`      | `1.0`                  | Every motion slowed to this fraction of its speed (0 \< x \<= 1)             |

The engine base declares the `arm.*` parameters too; box opening does not use
`self.arm` yet, so they change nothing here.

## Hardware

The Kinova Gen3 and its Robotiq 2F-85 through the arm driver, the wrist and the
scene OAK-D cameras through their own nodes, and the GPU for OWLv2 (`gpus: all`
in the real fragment). OWLv2's weights come from the host's Hugging Face cache,
mounted read-only at `/models/hf`.

## Alternatives

| id                 | What it is                                                                            |
| ------------------ | ------------------------------------------------------------------------------------- |
| `box_opening_real` | The capability proper                                                                 |
| `box_opening_mock` | Same interface, no hardware: each action reports the real run's phases, nothing moves |
