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
1. **Survey** (only with `wrist_survey` on, and only without a selection): where
   the scene camera finds no container, as with the chair parked side-on to a
   counter it cannot go under (the counter then stands beside the arm, out of
   the scene camera's view), the wrist camera looks for it from one or two poses
   over the arm's side. They are chosen on each run from the arm's kinematics
   and the chair model alone, for any counter height of 0.20-0.45 m in
   `arm_mount`: clear of the chair and its user, high over where any such counter
   could stand, each planned by the planner as the look is, its way there held
   the same way: no part of the arm goes into where such a counter could stand,
   or deeper than the rest pose already has it there, nor nearer the user than
   the planner keeps it. Where a lid-sized top
   is seen but its button cannot be checked from there, the camera looks once
   more nearer above it. The container found this way goes on to the look and
   the press as the scene camera's would, and the counter the wrist saw is the
   table the lid's spot is chosen on. Nothing found, or two places that could be
   it, is `not_found`, nothing pressed, the arm home.
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

### Watching a run fly

The node writes a flight view of each run to its stdout: one short line a step as
it starts and as it ends (`ok`, `FAIL`, or `??` where the run could not tell and
went on), a `note` for what worked but looked off, a `next` for what follows a
failure, and a result that says where the box, the lid and the arm were left.
Everything else the node logs goes to stderr, so:

```bash
docker logs -f sheppy-box_opening 2>/dev/null   # the flight view alone
docker logs -f sheppy-box_opening               # with every full line under its [INFO] prefix
```

```text
BOX OPENING   20:53:57
   find      ...   looking for the box
 1 FIND      ok    box 0.66 m out (wrist camera, 1 view)        18.0 s
   look      ...   moving to look at the button
 2 LOOK      ok    button found, 43 mm across, 0.68 m out        3.0 s
   press     ...   pressing the button
 3 PRESS     ok    21 N                                          4.0 s
             note  lid touched 41 mm higher than the camera read it
   grip      ...   rising 45 mm, opening the fingers, coming down round the knob
 4 GRIP      ok    knob held                                     8.0 s
 5 LIFT      ok
 6 CARRY     ok    lid to 20 cm from the box
 7 SET DOWN  ok    let go where the lid met the table
   home      ...   letting go and going home
 8 HOME      ok                                                  5.0 s
RESULT  DONE — box open, lid on the table, arm home           45 s
```

It is read off the run's own console lines (`box_opening/flight_view.py`) and
changes nothing the run does.

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
| `/wrist_camera/color/image_raw`      | `sensor_msgs/Image`                            | yes      | The aim at the button, the grip's close-up; with `wrist_survey`, the survey for a container the scene camera cannot see                                                   |
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

It is activated when the user selects a `food container` (the scene
segmenter's class), or a `food_container`: the same class without its space,
for an object list typed by hand (the orchestrator's
`interactive_perception_node` reads `add <instance_id> <perception_class>` split
on spaces). `open_box` is offered
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
order: `locate` (0.03 for each wrist survey pose, its message
`looking for the container with the wrist camera`; 0.05 once the container is
found), `look` (0.1), `press` (0.35), `grip` (0.5), `carry`
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

| Code            | Recoverable | Meaning                                                                                                                                                                                                                                                                                                             |
| --------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bad_params`    | no          | The goal had params; `open_box` takes none. Nothing ran                                                                                                                                                                                                                                                             |
| `not_allowed`   | no          | `allow_execute` is off on this node. Nothing ran                                                                                                                                                                                                                                                                    |
| `timeout`       | yes         | The run was still going after 90 s, box opening's own budget, and was stopped its own way (one Ctrl+C); its stop's worst (a second Ctrl+C, then a kill) ends within 120 s. The orchestrator's own limit (`execute_timeout_s`) is configurable and off by default. Not recoverable with the lid still in the fingers |
| `not_ready`     | yes         | Refused before anything moved: a driver, planner, camera or TF missing                                                                                                                                                                                                                                              |
| `not_found`     | yes         | The scene camera found no container (with `wrist_survey`, nor did the wrist camera's survey: nothing, or two places that could be it); nothing was pressed                                                                                                                                                          |
| `not_home`      | yes         | The arm could not get to its start pose                                                                                                                                                                                                                                                                             |
| `contact_early` | yes         | The press met something more than 45 mm above the button's top as the cameras saw it: stopped there, nothing pressed, the arm went home                                                                                                                                                                             |
| `contact_late`  | yes         | The press met nothing down to 25 mm below that top, or at all: the box is not where it was seen; the arm went home                                                                                                                                                                                                  |
| `watch_failed`  | no          | The press's contact watch failed: the press stopped there                                                                                                                                                                                                                                                           |
| `box_moved`     | yes         | The press moved the container                                                                                                                                                                                                                                                                                       |
| `not_popped`    | yes         | Pressed, but the knob did not come up (or the stroke pressed nothing); the arm went home                                                                                                                                                                                                                            |
| `no_button`     | yes         | The look found no button; nothing was pressed                                                                                                                                                                                                                                                                       |
| `no_press`      | yes         | No press plans to the button; nothing was pressed or moved                                                                                                                                                                                                                                                          |
| `lid_put_back`  | yes         | No carry for the lid: it is back on the box and the arm home                                                                                                                                                                                                                                                        |
| `lid_dropped`   | no          | The arm got home, but the lid slipped out of the fingers on the way                                                                                                                                                                                                                                                 |
| `arm_holds`     | no          | A motion stopped part of the way: the arm holds there until cleared                                                                                                                                                                                                                                                 |
| `arm_halted`    | no          | The driver halted a motion (the e-stop) or took control away: the run sent nothing after it                                                                                                                                                                                                                         |
| `stopped`       | no          | The run was ended from outside                                                                                                                                                                                                                                                                                      |
| `failed`        | no          | Anything else, a traceback included; the message is the run's last STOP line (or line). Nothing says where the arm is                                                                                                                                                                                               |

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until
the orchestrator is up.

## The UI

Started by the orchestrator with the UI (the token it pushes for the execution
in hand), `open_box` shows its own panels through the base class's
`ctx.update_user_panel()`, using `capability_ui.RunUI`. The design is Chris's
(`demo/ui`): one panel for as long as the run goes, its one button the
orchestrator's cancel.

While the run goes:

```json
{
  "title": "Opening the container",
  "message": "Pressing the button",
  "progress": {"phase": "press", "pct": 35.0, "pausable": false},
  "buttons": [{"button_id": "stop", "label": "Stop"}]
}
```

`message` is the progress message of the phase the run is in, `pct` the same
fraction on the UI bar's scale, 0 to 100 (`docs/UI_JSON_SCHEMA.md`; the
progress the orchestrator gets stays 0 to 1). A panel is sent as the phase
changes, never re-sent unchanged; one the bridge dropped is re-sent at most
once a second. The one button is **Stop**, and it is the orchestrator's cancel:
the base class's own `CANCEL`, so the run gets its Ctrl+C (the motion in flight
is cancelled first and the arm holds where it is), the action ends cancelled,
and nothing is written to the world state after it. There is no Pause: a run
cannot pause. Once the run is being stopped (Stop, the orchestrator's cancel,
or the 90 s budget) the panel says so (`Stopping: the arm stops where it is.`)
and has no button.

Once the run has ended, one notice with **Done**:

| How it ended                  | Title                            | Message                                                                                                                    |
| ----------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| The lid off                   | `Opening the container`          | `Lid off: it is on the table beside the container, and the arm is home.` (the mock: `Mock: ... No arm motion.`)            |
| Stop                          | `Opening the container: stopped` | `Stopped. The arm is holding still where it was.`, and `The lid is (may be) still in the fingers.` where a grip had closed |
| An error code (table above)   | `Opening the container: failed`  | The code in a person's words (`node.ERROR_TEXT`), and the lid's note as above                                              |
| `bad_params`, `not_allowed`   | `... : not started`              | What was refused; nothing moved                                                                                            |
| The orchestrator's own cancel | none                             | The orchestrator has the UI back                                                                                           |

The notice is shown after the run has ended and its result is decided, and
waits for Done at most 60 s (`capability_ui`'s `NOTICE_TIMEOUT_S`): not
tapping it never changes the result, and it is not part of the run's 90 s
budget (`RUN_BUDGET_S`, timed on the run alone). Standalone (no token) nothing
is sent, and a run is as it was without the UI.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file.

| Name                 | Default                | Meaning                                                                                                                                          |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `use_orchestrator`   | `true`                 | `false` runs standalone: no registration, drive it by CLI                                                                                        |
| `mock`               | `false`                | `true` skips hardware (the mock alternative sets it)                                                                                             |
| `allow_execute`      | `false`                | `true` only for attended runs; `false`: `open_box` unavailable, nothing runs                                                                     |
| `container`          | `"ankou_pink.yaml"`    | The container's config, in the image's `config/containers`, or a path                                                                            |
| `world`              | `"world_flatbot.yaml"` | The robot's planning frame, cameras and fixed body, in the image's `config`                                                                      |
| `state_dir`          | `""`                   | Where runs keep their logs, captures and videos; `""` = `~/.ros/box_opening`                                                                     |
| `video`              | `false`                | Record both cameras from the grip to the set-down (the newest 20 kept); on for attended data collection                                          |
| `viz`                | `false`                | Draw what the perception sees and decides for Foxglove                                                                                           |
| `mission_frames`     | `false`                | Debug: each executed run keeps its camera frames (~10 MB, the newest 20), and its run record's flight files and room maps                        |
| `speed_scale`        | `1.0`                  | Every motion slowed to this fraction of its speed (0 \< x \<= 1)                                                                                 |
| `body_layers`        | see below              | The planner config whose chair and user the arm is held out of; `""` turns the check off (a bench with no chair)                                 |
| `planner_worlds_as`  | `""`                   | The folder the planner's container mounts for box opening's worlds, as the planner names it; `""` = no world is sent (see "The planner's world") |
| `planner_worlds_dir` | `""`                   | The same folder as this node writes it; `""` = `planner_worlds` under `state_dir`                                                                |
| `wrist_survey`       | `false`                | Where the scene camera finds no container, look for it with the wrist camera (see below); needs the body model on                                |
| `wrist_first`        | `false`                | Facing a table: the wrist camera finds the container in front of the arm, the wrist turned where the arm rests (see below)                       |

### The planner's world

Box opening plans in the world the planner's deployment starts it with and sends
it none, with one exception: while a way HOME is planned (the first home, a run's
own homes, `home_arm`'s), the planner is handed the table the run's cameras
measured, and nothing else. A planner that knows no table plans HOME through one;
judging its plans afterwards refuses what it could have planned around.

It is off unless the deployment shares a folder with the planner, the way
rammp-feeding's does: `planner_worlds_as` names that folder as the planner's
container mounts it (read-only), and the node writes into `planner_worlds_dir`
(by default `planner_worlds` under `state_dir`). The file is the planner's own
start-up scene (read from `body_layers`' planner config) with one block more, the
measured table: the extent the scene camera saw, its top 20 mm under the measured
top so that a hand resting near the table is still a start the planner accepts.
It is sent by `set_world` before a home is planned and the planner's own world is
put back by `set_world` as soon as the plans are made (and by the first plan of
the next run, should a run die in between). Every planned home is still judged
against the measured top afterwards. With no table measured, a planner whose own
world is empty, a goal that would itself stand in the block, or a planner that
does not take the file, the planner keeps its own world and nothing else changes.

`body_layers` defaults to `/config/robot/curobo/planner.yaml`, the planner's own
config: the fragment mounts the robot's `config/robots/current` at `/config/robot`
read-only, as the planner's node does. Its layers (the chair, the seat, the person
in it) and its arm spheres, with the hand's own in `config/robot_body.yaml`, hold
every plan-free line and every planner goal out of the chair and the user, and the
same file slows the arm near the user. Unreadable, a run stops before it moves; on,
a run starts only on a planner started with that same config and `world ""`.

`wrist_survey` is off by default: a run whose scene camera finds no container
stops with `not_found` before anything moves. On, such a run surveys with the
wrist camera instead (the Survey step above), adding about 20-40 s: choosing
the poses (about 10 s), up to two survey poses and two looks again, each a
planned move, a still frame and the finder (`tasks/survey.py`,
`perception/wrist_finder.py`). It runs only for a
run without a selection (see the gap below), only with the body model on, and
the survey's frames are kept as captures (`captures/wrist-survey-*`). The
heights the wrist reads are its own (its mount's error, more the more its view
tilts): the press goes by the higher of the survey's top and the aim's, and
the press's contact band refuses the rest (`contact_early`).

`wrist_first` is off by default. On (a table faced from the front, where the
scene camera sees no lid whole where the arm presses), a run without a selection
looks for the container with the wrist camera before anything else: the three
wrist joints turn where the arm rests, the arm above them standing still, so the
camera sees the table in front. The views are chosen on each run from the arm's
kinematics, the chair model and the camera's own view, for any table height of
0.15-0.35 m in `arm_mount`: up to four, the one that shows the most of where the
arm presses first, and before them one aimed where the scene camera has the
container, if it has it at all (a fix placed roughly from a box its frame cut
included). Each view is a planned move of the wrist (about a second), a still
frame and the finder; the run goes on at the first that shows the lid whole. A
lid a view cuts is looked at again with the wrist turned to it (at most twice a
run) and is the container only where that look shows it whole. What the wrist
finds is taken over the scene camera's; where it finds none the scene camera's
fix is used, and with neither the run stops with `not_found`, saying what each
camera said. It adds about 1.5 s where the first view shows the container and
up to about 10 s where none does; its frames are kept as captures
(`captures/wrist-survey-*`). Only with the body model on, never with a selection.

### The survey and the orchestrator's selection: an interface gap

The orchestrator selects a container from the scene camera's detections
(`/perception/object_detections`, by `instance_id`), and passes that selection
to `open_box`. A container the scene camera cannot see, the case the survey is
for, has no detection and no `instance_id`, so the orchestrator cannot select
it, and `open_box` is never offered for it. And for a selected container the
scene camera does not find at the run, the wrist cannot tell which container
it is: the survey would find whatever stands on the side counter. So a run with
a selection does not survey; it stops with `not_found` as before, saying so.
Today the survey serves runs without a selection (`use_orchestrator: false`,
the CLI action).

Proposal, for the orchestrator and perception owners (nothing here changes
them): either (a) the survey's container is written into the world state as a
detection of its own after a survey (an object with an `instance_id`, its
pose in `arm_mount`, `source: wrist`), which the user can then select, the next
`open_box` passing that id; or (b) `open_box` is offered with no selected
instance where the deployment surveys (`wrist_survey` on), for "the container
beside the chair", the engine writing the found container's record as it
writes `opened` today. (a) keeps the selection the user's; (b) needs no new
perception topic.

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
