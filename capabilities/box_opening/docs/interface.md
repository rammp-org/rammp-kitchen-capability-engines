# Box Opening Capability — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @ChrissCox
**Tier:** experimental

## What it does

Opens the food container: presses the button on its lid so the lid comes
loose, grips the lid by that button, lifts it off, sets it down on the table
beside the container, and goes home. It has one action, `open_box`:

1. **Ready:** the run checks the robot is ready (driver, planner, TF, gripper,
   both cameras) and refuses before anything moves if not; an arm away from its
   rest pose is brought there first.
1. **Locate:** the scene camera finds the container before the arm moves: the
   scene segmenter's box of the selected container (its `instance_id` on
   `/perception/object_detections`, and a lid box stacked on it), with its lid's
   height from the scene camera's depth. Nothing about the table or the
   container's place is configured: both come from the cameras on every run.
   A selected box that stands on another container's box (RF-DETR sometimes
   boxes the lid as a second container, with its own id) is resolved to the
   holder under it, the selected box its lid. The id finds it at this first
   look. From then on it is also followed by
   where it was boxed: a track perception drops while the arm hides the box,
   and finds again under a new id, is still the selected container. Every
   world-state write names the id the goal came with.
1. **Look:** the arm moves to a pose looking down at the lid at 45 degrees, and
   the wrist camera aims at the button.
1. **Press:** the fingers, closed, press the button in joint impedance. The arm
   feels where the pads meet the lid, watching its whole way down, and the
   stroke ends the button's travel past that. A press that meets something
   more than 45 mm above the button's top as the cameras saw it, or nothing
   down to 25 mm below it, is refused: stopped there, back up and home
   (`contact_early`, `contact_late`).
1. **Grip:** straight from the press, the fingers rise, open, come down around
   the button and close on it. A grip that misses goes back to the look pose and
   tries again from there, a centimetre lower.
1. **Carry:** the lid is lifted, carried and set down beside the container in
   one motion, at a spot the scene camera sees is clear. Never inside the
   no-land radius: the lid's rim at least 50 mm off the container's (0.155 m
   centre to centre for the pink canister), round both where the press held
   the button and where the cameras put the container.
1. **Release and home:** the fingers open as the arm rises off the lid, and the
   arm goes home.

If the run has to stop with the lid in the fingers and the arm standing (no
carry to any spot plans, or the run's own checks refuse the carry before it
starts), it puts the lid back on the box first: back over the box and straight
down to where the knob was gripped, the fingers open, the hand rises off the
knob, and the arm goes home. The action still fails (`failed`, recoverable).
After a cancel, an e-stop or a motion the driver stopped part way, nothing more
moves: the arm holds where it stopped (`arm_holds`, or the cancel). After the
e-stop (the driver's `HALTED`) or lost control (`NOT_AUTHORIZED`), the run ends
at once and sends nothing more, no let-go, no home and no gripper command
(`arm_halted`).

The action runs box opening's command line (`press_demo`) as a child process,
the way it ran on flatbot until now. Its
console lines are this node's log lines, and its run log's rows are the
action's progress. Planning and motion go straight to the arm driver
(`/execute_joint_trajectory`, `/setpoint/gripper`) and the cuRobo planner
(`/rammp_curobo/plan_to_pose`, `plan_to_joints`), not yet through `self.arm`.

The arm moves only on a node with `allow_execute: true`, set for an attended
run with someone at the e-stop. With it `false` (the default), `open_box` is
reported unavailable (with why), and a goal that comes anyway fails with
`not_allowed` before anything runs. There is no dry run. The mock moves nothing
and is always offered.

## Publishes

| Topic                 | Type                                   | Rate     | Meaning                                                                                                   |
| --------------------- | -------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| `/box_opening/status` | `std_msgs/String`                      | 1 Hz     | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |
| `/setpoint/gripper`   | `rammp_arm_interfaces/GripperSetpoint` | per grip | The run's gripper commands, to the arm driver                                                             |

With `viz: true` the run also draws what it sees on `/box_opening/viz/*`
(`scene/annotated/compressed`, `wrist/annotated/compressed`, `markers`,
`status`) for Foxglove.

## Subscribes

While an action runs (its run subscribes; the node itself reads nothing).

| Topic                                | Type                                           | Required | Meaning                                                                                                                                                                   |
| ------------------------------------ | ---------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/joint_states`                      | `sensor_msgs/JointState`                       | yes      | The arm's joints, efforts and gripper knuckle                                                                                                                             |
| `/arm/robot_description`             | `std_msgs/String`                              | yes      | The arm's URDF, rooted at `arm_mount` (the run's FK chain)                                                                                                                |
| `/wrist_camera/color/image_raw`      | `sensor_msgs/Image`                            | yes      | The aim at the button, the grip's close-up                                                                                                                                |
| `/wrist_camera/color/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Its intrinsics                                                                                                                                                            |
| `/wrist_camera/depth/image_rect_raw` | `sensor_msgs/Image`                            | yes      | Depth aligned to colour                                                                                                                                                   |
| `/scene_camera/color/image_raw`      | `sensor_msgs/Image`                            | yes      | Finding the container, the lid's spot                                                                                                                                     |
| `/scene_camera/color/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Its intrinsics                                                                                                                                                            |
| `/scene_camera/depth/image_rect_raw` | `sensor_msgs/Image`                            | yes      | Scene depth aligned to colour                                                                                                                                             |
| `/scene_camera/depth/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Its intrinsics                                                                                                                                                            |
| `/perception/object_detections`      | `rammp_perception_interfaces/ObjectDetections` | yes      | The scene segmenter's boxes: the selected container's                                                                                                                     |
| `/tf`, `/tf_static`                  | `tf2_msgs/TFMessage`                           | yes      | The finger tips and both cameras in `arm_mount`                                                                                                                           |
| `/diagnostics`                       | `diagnostic_msgs/DiagnosticArray`              | no       | The arm driver's: its link to the arm and its e-stop. A run does not start while the link is not running, and a goal refused while it is down ends the run (`arm_halted`) |
| `/ee_state`                          | `rammp_arm_interfaces/EeState`                 | no       | The driver's estimate of the force on the tool: logged at the press's hold and at the set-down (`driver_force`)                                                           |
| `/contact_monitor/residual`          | `sensor_msgs/JointState`                       | no       | The driver's contact monitor's residual (with its `contact_debug_topic` on): its peaks logged per motion (`driver_contact`)                                               |

Where the arm driver's contact monitor enforces (its `contact_detection`), each
motion sets its thresholds by `/set_contact_thresholds`: the driver's `DEFAULT`
in free air (the look, the way home), `LOOSE` where the run meets anything on
purpose, and `DEFAULT` again as the run ends. Its contact stop (`CONTACT_STOP`)
ends the run as the e-stop does (`arm_halted`). In `observe` the run sets
nothing and logs what the monitor saw of each motion.

## Orchestrator interface

Served under `/box_opening/`, types from `rammp_orchestrator_interfaces`:

| Name                          | Type                   | Meaning                                       |
| ----------------------------- | ---------------------- | --------------------------------------------- |
| `/box_opening/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/box_opening/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/box_opening/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/box_opening/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

It is activated when the user selects a `food container`. `open_box` is offered
where `allow_execute` is on, until the world state holds the container `opened`,
and only while the world state's gripper holds nothing (`gripper.holding`, which
a run stopped with the lid in the fingers leaves set).

| Action     | Params | Result payload                                                                                                                      |
| ---------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `open_box` | none   | `{"executed", "lid_off", "lid_xyz", "presses", "grips", "duration_s", "video", "run_log"}`, and `world_state` when the lid came off |

`lid_xyz` is where the lid was set down, in `arm_mount` (m). `video` is the MP4
of both cameras from the grip to the set-down, where one was recorded. As the
lid is let go on the table, the container's object record gets `opened: true`.
It is written then, not at the run's end, so a run that stops or is cancelled on
its way home still leaves the container open. `world_state` says `opened`, or
`not updated` where the orchestrator refused the write (also logged as an
error). `open_box` takes no params: any fails with `bad_params`
before anything runs.

While the lid is in the fingers the world state's gripper says so: `holding`
is `<instance_id>:lid` from the grip that held it to its release (or its
put-back), when it is `null` again. A cancel or a stop in between leaves it
said; it is written as the run goes, since the engine takes no world-state
write once an action is cancelled.

Progress `pct` is the fraction of the run done, from 0 to 1. The phases, in
order: `locate` (0.05), `look` (0.1), `press` (0.35), `grip` (0.5), `carry`
(0.85), `release` (0.9), `home` (1.0). A phase is reported as
its motion ends, its message what the run is doing then, in a person's words
(`pressing the button`, `setting the lid down`; `: stopped` where that motion
stopped). A run is one motion from the press to the set-down, so it
cannot pause once it has started (every phase reports `pausable: false`); a
cancel stops it at once. The cancel is the run's own Ctrl+C: the motion in
flight is cancelled, and the action ends once the driver has confirmed it, with
no video encoded and no frames written on the way out.

A run that stops fails the action with one of these codes. One that is not
recoverable also sets the world state's `system_status.arm` to
`{"state": "error", "error_code", "message"}`, which the orchestrator honours:
no object or action flow until the arm's domain is normal again. With the lid
still in the fingers, no code is recoverable.

| Code            | Recoverable | Meaning                                                                                                                                                                                                                                 |
| --------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bad_params`    | no          | The goal had params; `open_box` takes none. Nothing ran                                                                                                                                                                                 |
| `not_allowed`   | no          | `allow_execute` is off on this node. Nothing ran                                                                                                                                                                                        |
| `timeout`       | yes         | The run was still going after 90 s and was stopped its own way (one Ctrl+C); its stop's worst (a second Ctrl+C, then a kill) still ends before the orchestrator's fixed 120 s cancel; not recoverable with the lid still in the fingers |
| `not_ready`     | yes         | Refused before anything moved: a driver, planner, camera or TF missing                                                                                                                                                                  |
| `not_found`     | yes         | The scene camera found no container                                                                                                                                                                                                     |
| `not_home`      | yes         | The arm could not get to its start pose                                                                                                                                                                                                 |
| `contact_early` | yes         | The press met something more than 45 mm above the button's top as the cameras saw it: stopped there, nothing pressed, the arm went home                                                                                                 |
| `contact_late`  | yes         | The press met nothing down to 25 mm below that top, or at all: the box is not where it was seen; the arm went home                                                                                                                      |
| `watch_failed`  | no          | The press's contact watch failed: the press stopped there                                                                                                                                                                               |
| `box_moved`     | yes         | The press moved the container                                                                                                                                                                                                           |
| `not_popped`    | yes         | Pressed, but the knob did not come up (or the stroke pressed nothing); the arm went home                                                                                                                                                |
| `no_button`     | yes         | The look found no button; nothing was pressed                                                                                                                                                                                           |
| `no_press`      | yes         | No press plans to the button; nothing was pressed or moved                                                                                                                                                                              |
| `lid_put_back`  | yes         | No carry for the lid: it is back on the box and the arm home                                                                                                                                                                            |
| `lid_dropped`   | no          | The arm got home, but the lid slipped out of the fingers on the way                                                                                                                                                                     |
| `arm_holds`     | no          | A motion stopped part of the way: the arm holds there until cleared                                                                                                                                                                     |
| `arm_halted`    | no          | The driver halted a motion (the e-stop) or took control away: the run sent nothing after it                                                                                                                                             |
| `stopped`       | no          | The run was ended from outside                                                                                                                                                                                                          |
| `failed`        | no          | Anything else, a traceback included; the message is the run's last STOP line (or line). Nothing says where the arm is                                                                                                                   |

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until
the orchestrator is up.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file.

| Name               | Default                | Meaning                                                                                                                   |
| ------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `use_orchestrator` | `true`                 | `false` runs standalone: no registration, drive it by CLI                                                                 |
| `mock`             | `false`                | `true` skips hardware (the mock alternative sets it)                                                                      |
| `allow_execute`    | `false`                | `true` only for attended runs; `false`: `open_box` unavailable, nothing runs                                              |
| `container`        | `"ankou_pink.yaml"`    | The container's config, in the image's `config/containers`, or a path                                                     |
| `world`            | `"world_flatbot.yaml"` | The robot's planning frame, cameras and fixed body, in the image's `config`                                               |
| `state_dir`        | `""`                   | Where runs keep their logs, captures and videos; `""` = `~/.ros/box_opening`                                              |
| `video`            | `false`                | Record both cameras from the grip to the set-down (the newest 20 kept); on for attended data collection                   |
| `viz`              | `false`                | Draw what the perception sees and decides for Foxglove                                                                    |
| `mission_frames`   | `false`                | Debug: each executed run keeps its camera frames (~10 MB, the newest 20), and its run record's flight files and room maps |
| `speed_scale`      | `1.0`                  | Every motion slowed to this fraction of its speed (0 \< x \<= 1)                                                          |

Every run keeps its run log, `<state_dir>/runs/run-<stamp>.jsonl` (about 20 KB).
The folder keeps the newest 100, each with its side files.

The engine base declares the `arm.*` parameters too; box opening does not use
`self.arm` yet, so they change nothing here.

## Hardware

The Kinova Gen3 and its Robotiq 2F-85 through the arm driver, and the wrist and
the scene OAK-D cameras through their own nodes. No GPU: box opening runs no
model; the container's box comes from sheppy's scene segmenter (RF-DETR), which
the deployment must run.

## Alternatives

| id                 | What it is                                                                                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `box_opening_real` | The capability proper                                                                                                                                                                 |
| `box_opening_mock` | Same interface, no hardware: a successful real run's phases, result (`mock: true`, a stand-in `lid_xyz`) and world-state writes (`holding`, `opened`); it cannot pause; nothing moves |
