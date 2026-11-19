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
   it, is a container no camera sees: the chair is guided (see "A container no
   camera sees"); in a run by hand it is `not_found`, nothing pressed, the arm
   home.
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
   one motion, at a spot the cameras see is clear. Never inside the
   no-land radius: the lid's rim at least 50 mm off the container's (0.155 m
   centre to centre for the pink canister), round both where the press held
   the button and where the cameras put the container. And only on table the
   cameras measured this run: the whole lid, its rim 15 mm more, on cells the
   run's map saw at the table's height (not unseen, not past an edge, not on
   what stands there), the hand's parts over such cells too, and inside the
   table's extent where the fix measured one, the lid's rim 30 mm from its
   sides. That holds for every spot, the fallbacks included (the most room
   seen where nothing reads clear, beside the box, the nearest table point).
   A spot is looked for round the container as far out from the arm as the
   container stands, no longer only inside the bench's reach band (x
   0.28-0.65 m). With no map of the table (no camera's points show one) there
   is no spot.
1. **Set down:** fast to 80 mm over where the run has the lid meet the table,
   then slowly, to 8 mm under it at most. That height is the cameras' table
   moved by what the fingers found: the table is taken as far under the lid's
   top as the cameras saw it, the top where the fingers hold the knob (on
   2026-11-13 and 11-17 the wrist camera read the lid's top, and so the table,
   46-55 mm low). The arm feels for the table on the slow stretch, from 0.6 s
   after it begins (the arm is still catching up with its reference until then,
   which reads as a contact): the descent stops where the lid meets the table.
   A rise met more than 35 mm over where the table should be is not the table,
   and the descent goes on, the arm settling again for 0.6 s; a rise that goes
   on for 15 mm of the descent is the lid held up, and stops it. Held up no
   more than those 35 mm over where the table should be, it is the table (met
   while the arm settled); held up higher, it is not, and the lid is not let
   go. Nothing is planned while the set-down flies, so every sample of it is
   read as it comes; what was not read by its end is read then. With the spot
   chosen after the lift, the way home is planned while the arm rises off the
   lid.
1. **Release and home:** the fingers open, as the arm rises off the lid, only
   where the set-down felt the lid meet the table at about that height; then
   the arm goes home. The lid is never let go anywhere else.

If the run has to stop with the lid in the fingers and the arm standing (no
spot on the table the cameras measured that the hand reaches and a carry plans
to, the run's own checks refuse the carry before it starts, or the set-down
ended without the arm feeling the lid meet the table: nothing met down to its
end, something holding the lid up well above the table, or its watch failed,
which stops the descent where it failed, or where the slow stretch begins if it
failed before that, and never the lift or the carry),
it puts the lid back on the box first: back up to the carry's height where it
had come down, back over the box and straight down to where the knob was
gripped, the fingers open, the hand rises off the knob, and the arm goes home.
The action still fails (`lid_put_back`, recoverable), and its message says
which of these it was. Where a line of that way back does not fly, the arm
holds the lid where it stands (`arm_holds`).
After a cancel, an e-stop or a motion the driver stopped part way, nothing more
moves: the arm holds where it stopped (`arm_holds`, or the cancel). After the
e-stop (the driver's `HALTED`) or lost control (`NOT_AUTHORIZED`), the run ends
at once and sends nothing more, no let-go, no home and no gripper command
(`arm_halted`).

A container the arm cannot press from where the chair stands does not end the
run (owner 2026-11-17, of a run that did: "it should NOT just fail EVER"). Where
the run finds, once it has found the container, that no press plans from there,
or its wrist sees a lid-sized top that stands out of the arm's reach, the arm
goes to rest with only its wrist turned to look, and the user is told which way
and how far to move the chair until the container is where the arm opens it.
Then they are asked, and at their yes the run finds the container again and
goes on (see "Lining up in the middle of a run"). On the UI's panel where the
goal has one; on the node's console, answered `y` or `n`, where it has none.
Nor does a container no camera sees at all (owner 2026-11-19): the user is told
that the arm cannot see it and to move the chair in front of it, and the same
guidance takes over the moment a camera has it (see "A container no camera
sees").

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
It says `lid on the table` only of a set-down the run felt meet the table; a
set-down that felt none reads `7 SET DOWN  FAIL  the arm felt no table under the lid; the lid was not let go`, then `8 PUT BACK  ok`, and the result's `lid: on the box`.
A lid that slipped out of the fingers on the way is never `on the table`
either, whatever the set-down felt: `5 CARRY  FAIL  the lid slipped out of the fingers on the way`,
and the result's `lid: slipped out of the fingers on the way (where it lies is not known)`.
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
changes nothing the run does. A press that does not plan says why, each tilt's
own reason (`no press plans from here (0.79 m out): the slide in swings the arm wide (30, 35 deg); the press stroke would pass too close to the user (40, 45 deg)`);
only a container whose every look stands past the arm's reach is said to be out
of it. A line-up is a step of its own, `LINE UP`, with a `say` line each time
what the user is told changes and an `ask` line for the question (see "With no
UI").

## Publishes

| Topic                 | Type                                   | Rate     | Meaning                                                                                                   |
| --------------------- | -------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| `/box_opening/status` | `std_msgs/String`                      | 1 Hz     | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |
| `/setpoint/gripper`   | `rammp_arm_interfaces/GripperSetpoint` | per grip | The run's gripper commands, to the arm driver                                                             |

With `viz: true` the run also draws what it sees on `/box_opening/viz/*`
(`scene/annotated/compressed`, `wrist/annotated/compressed`, `markers`,
`status`) for Foxglove.

## Subscribes

While an action runs (its run subscribes; the node itself reads only
`/box_opening/answer`).

| Topic                                | Type                                           | Required | Meaning                                                                                                                                                                                                            |
| ------------------------------------ | ---------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/joint_states`                      | `sensor_msgs/JointState`                       | yes      | The arm's joints, efforts and gripper knuckle                                                                                                                                                                      |
| `/arm/robot_description`             | `std_msgs/String`                              | yes      | The arm's URDF, rooted at `arm_mount` (the run's FK chain)                                                                                                                                                         |
| `/wrist_camera/color/image_raw`      | `sensor_msgs/Image`                            | yes      | The aim at the button, the grip's close-up; with `wrist_survey`, the survey for a container the scene camera cannot see                                                                                            |
| `/wrist_camera/color/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Its intrinsics                                                                                                                                                                                                     |
| `/wrist_camera/depth/image_rect_raw` | `sensor_msgs/Image`                            | yes      | Depth aligned to colour                                                                                                                                                                                            |
| `/scene_camera/color/image_raw`      | `sensor_msgs/Image`                            | yes      | Finding the container, the lid's spot                                                                                                                                                                              |
| `/scene_camera/color/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Its intrinsics                                                                                                                                                                                                     |
| `/scene_camera/depth/image_rect_raw` | `sensor_msgs/Image`                            | yes      | Scene depth aligned to colour                                                                                                                                                                                      |
| `/scene_camera/depth/camera_info`    | `sensor_msgs/CameraInfo`                       | yes      | Its intrinsics                                                                                                                                                                                                     |
| `/perception/object_detections`      | `rammp_perception_interfaces/ObjectDetections` | yes      | The scene segmenter's boxes: the selected container's                                                                                                                                                              |
| `/tf`, `/tf_static`                  | `tf2_msgs/TFMessage`                           | yes      | The finger tips and both cameras in `arm_mount`                                                                                                                                                                    |
| `/diagnostics`                       | `diagnostic_msgs/DiagnosticArray`              | no       | The arm driver's: its link to the arm and its e-stop. A run does not start while the link is not running, and a goal refused while it is down ends the run (`arm_halted`)                                          |
| `/ee_state`                          | `rammp_arm_interfaces/EeState`                 | no       | The driver's estimate of the force on the tool: logged at the press's hold and at the set-down (`driver_force`)                                                                                                    |
| `/contact_monitor/residual`          | `sensor_msgs/JointState`                       | no       | The driver's contact monitor's residual (with its `contact_debug_topic` on): its peaks logged per motion (`driver_contact`)                                                                                        |
| `/box_opening/answer`                | `std_msgs/String`                              | no       | The person's `y` or `n`, for a goal with no UI to press on: `y` answers a line-up's question, `n` cancels the action at any moment (see "With no UI"). The node's own subscription, held only while an action runs |

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

The orchestrator's token for one execution comes in the `execute_action` goal
itself, its `token` field (the base class keeps it for the action's length; it
is what the UI bridge takes the run's panel by, see "The UI"). Until the engine
update of 2026-11-16 it was pushed through `/box_opening/acquire_control`, which
the base class no longer serves: an orchestrator of that older protocol and
this engine do not reach each other, and the image builds
`rammp_orchestrator_interfaces` at the base class's pin for the goal's field.

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

`lid_xyz` is where the lid was set down, in `arm_mount` (m), its height the
table's as the set-down went by it (the cameras' moved by the fingers' hold of
the lid's top). `video` is the MP4
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
order: `locate` (0 with `looking for the container` once the run is up, and
with `going to the start pose` where the arm does not rest at its start pose
and the run's first motion takes it there; 0.03 for each wrist survey pose, its
message `looking for the container with the wrist camera`; 0.05 once the
container is found), `look` (0.1), `press` (0.35), `grip` (0.5), `carry`
(0.85), `release` (0.9), `home` (1.0). A phase is reported as
its motion ends, its message what the run is doing then, in a person's words
(`pressing the button`, `setting the lid down`; `: stopped` where that motion
stopped). A run that finds no press from where the chair stands stays in
`locate` at the fraction it had: `no press plans from here: wait, the arm comes to rest first`
(`out of reach from here: wait, ...` where its wrist sees the container out of
the arm's reach, `no container in view: wait, ...` where no camera sees one),
`going to rest` where it homed for it, then the line-up's own progress (phase
`line_up`, at 0), and from the user's yes `looking for the container again`.
A run is one motion from the press to the set-down, so it
cannot pause once it has started (every phase reports `pausable: false`); a
cancel stops it at once. The cancel is the run's own Ctrl+C: the motion in
flight is cancelled, and the action ends once the driver has confirmed it, with
no video encoded and no frames written on the way out.

A run that stops fails the action with one of these codes. One that is not
recoverable also sets the world state's `system_status.arm` to
`{"state": "error", "error_code", "message"}`, which the orchestrator honours:
no object or action flow until the arm's domain is normal again. With the lid
still in the fingers, no code is recoverable.

| Code                  | Recoverable | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bad_params`          | no          | The goal had params; `open_box` takes none. Nothing ran                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `not_allowed`         | no          | `allow_execute` is off on this node. Nothing ran                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `bad_config`          | no          | A setting has a value it cannot have (`air_control` neither `impedance` nor `position`). Nothing ran                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `line_up_unavailable` | no          | What watches the container or judges its place could not start, or ended while lining up (the message carries what it said). The run was never told to go; a wrist that looked is back at rest                                                                                                                                                                                                                                                                                                                                                                                               |
| `not_lined_up`        | yes         | The container was not where the arm opens it, or was never seen by a camera, in the time a line-up has (see "How long a line-up may take"), the UI went, or a line-up in the middle of a run had no time left under the orchestrator's 120 s. Nothing was pressed; a wrist that looked is back at rest. The message says which it was (`no camera saw the container within ...` where none did), and carries the last instruction the user was given and, in the middle of a run, why the chair was guided; the next `open_box` then lines up first, unless no camera had seen the container |
| `timeout`             | yes         | The run was still going after 90 s and was stopped its own way (one Ctrl+C); its stop's worst (a second Ctrl+C, then a kill) still ends before the orchestrator's fixed 120 s cancel; not recoverable with the lid still in the fingers                                                                                                                                                                                                                                                                                                                                                      |
| `not_ready`           | yes         | Refused before anything moved: a driver, planner, camera or TF missing                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `not_found`           | yes         | No camera could look for the container at all (the scene camera's streams or its TF missing, the scene segmenter down, silent or showing another camera, and no wrist frame judged): a fault no moving of the chair cures; nothing was pressed, the arm home. A container the cameras looked for and do not see is not this: the chair is guided (see "A container no camera sees"). Only a run by hand (`press_demo` without `--guide`) ends `not_found` for it, as it always has                                                                                                           |
| `not_home`            | yes         | The arm could not get to its start pose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `contact_early`       | yes         | The press met something more than 45 mm above the button's top as the cameras saw it: stopped there, nothing pressed, the arm went home                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `contact_late`        | yes         | The press met nothing down to 25 mm below that top, or at all: the box is not where it was seen; the arm went home                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `watch_failed`        | no          | The press's contact watch failed: the press stopped there                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `box_moved`           | yes         | The press moved the container                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `not_popped`          | yes         | Pressed, but the knob did not come up (or the stroke pressed nothing); the arm went home                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `no_button`           | yes         | The look found no button; nothing was pressed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `no_press`            | yes         | A stop before the press that names no other code (the container's config has no reach-out press, the fingers did not shut, the knob's height unread): nothing was pressed. A container no press plans to from where the chair stands is not this: the chair is guided (see "Lining up in the middle of a run")                                                                                                                                                                                                                                                                               |
| `lid_put_back`        | yes         | No spot for the lid on the table the cameras measured (or no carry to one), or a set-down that did not feel the table: the lid was not let go there, it is back on the box and the arm home                                                                                                                                                                                                                                                                                                                                                                                                  |
| `lid_dropped`         | no          | The arm got home, but the lid slipped out of the fingers on the way                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `arm_holds`           | no          | A motion stopped part of the way: the arm holds there until cleared                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `arm_halted`          | no          | The driver halted a motion (the e-stop) or took control away: the run sent nothing after it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `stopped`             | no          | The run was ended from outside                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `failed`              | no          | Anything else, a traceback included; the message is the run's last STOP line (or line). Nothing says where the arm is                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until
the orchestrator is up.

### The UI

Each goal carries the orchestrator's token for that execution (its `token`
field), which the orchestrator also gives the UI bridge and the arm driver. The
base class keeps the token for the action's length and drops it as the action
ends; a goal sent by hand carries none. The run's own
goals to the arm driver carry no token (`sender_id` alone): box opening runs
with the driver's `arbitration_mode: disabled`; under `enforced` the driver
refuses each of them (`NOT_AUTHORIZED`, which the run's STOP line names). The
base class also holds a publisher on the bridge's `/engine_overlay/image_raw`;
box opening sends nothing on it.

While `open_box` runs with that token in hand, the run shows its own panel on
the UI (the base class's `ctx.update_user_panel()`, to the bridge's
`request_user_input`):

```json
{
  "title": "Opening the container",
  "message": "Pressing the button",
  "progress": {"phase": "press", "pct": 35.0, "message": "Pressing the button", "pausable": false},
  "buttons": [{"button_id": "cancel", "label": "Cancel"}]
}
```

`message` is the progress message of the phase the run is in, `pct` the same
fraction on the UI bar's scale, 0 to 100 (`docs/UI_JSON_SCHEMA.md`; the
progress the orchestrator gets stays 0 to 1). The one button is **Cancel**, and
it is the orchestrator's cancel: the base class's own `CANCEL`, so the run gets
its Ctrl+C (the motion in flight is cancelled first and the arm holds where it
is), the action ends cancelled, and nothing is written to the world state after
it. There is no Pause: a run cannot pause. Once the run is being stopped (Cancel,
the orchestrator's cancel, or the 90 s budget) the panel says so
(`Cancelled: the arm is stopping where it is`) and has no button left.

The run's own panel asks the user nothing and the run never waits on it (a
line-up's panel does: below). It is sent as
the phase changes, and again every second so a panel the bridge took down comes
back with its Cancel; the base class closes it as the action ends, with the
token. The node acquires no `ui_control` of its own. Standalone (no token)
nothing is sent. How a run ended is not shown on a panel of its own: the
orchestrator's panel returns as the action ends, and the result is in the
action's result and the node's log.

#### Lining up first

With `line_up: true`, `open_box` first shows the same panel with an instruction
in place of the run's phase, in a loop of the base class's example's shape
(`ctx.update_user_panel`, `ctx.poll_ui_event`): where the container stands
against the arm's reach, as what the user does about it.

**The arm never flies on from a line-up without the user's yes.** With the UI
that is **Open** on the panel. A goal that has no UI to ask on (it came without
the orchestrator's token, as one sent by hand does, or the UI bridge is not
there) lines up the same way on the node's console and is answered `y` or `n`
on `/box_opening/answer` (see "With no UI"); until 2026-11-17 such a goal was
refused (`line_up_unavailable`). `line_up: true` means "line up before the
first move"; `false`, "guide only when needed" (see "Lining up in the middle of
a run"). Every goal's settings (`line_up`, whether it has the UI, whether it
came with the orchestrator's token (`orchestrated`), `line_up_far_m`,
`air_control`, `wrist_first`, `wrist_alone`) are in the node's log as it
starts: a `ros2 param set` lives only in the running node, and a restart puts
the deployment's values back.

| Panel message                                                     | Buttons      | When                                                                                                                                                                                            |
| ----------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Looking for the container`                                       | Cancel       | Nothing has looked yet: the wrist is turning to its view, the scene camera's first look has not ended                                                                                           |
| `The arm cannot see the container. Move the chair in front of it` | Cancel       | What watches has looked and has no container, or one that was seen has not been for 1.5 s. No direction, no distance: none is known                                                             |
| `The arm sees two possible containers. Move in front of only one` | Cancel       | What watches sees more than one thing that could be the container and cannot tell which is meant (stands 3 s before the line above replaces it, and the other way)                              |
| `Move closer, about 15 cm`                                        | Cancel       | Outside the boundary: closer, back, left or right, the larger miss                                                                                                                              |
| `Hold still, checking`                                            | Cancel       | Inside, not yet still for 1 s, or not yet judged                                                                                                                                                |
| `Ready. Open the container?`                                      | Open, Cancel | Inside, still, and the run plans for it there                                                                                                                                                   |
| `Press open again when you are in place`                          | none         | The line-up's time passed (35 s from the goal's start; with no UI, `Out of time. Start open_box again when you are in place` after 10 minutes): shown 3 s, then the action fails `not_lined_up` |
| `Container not seen in time. Press open again in front of it`     | none         | ... and no camera saw the container in all that time (with no UI, `Out of time: the container was never seen. Start open_box again in front of it`): the same end                               |
| `The arm is not at its rest pose: home it, then press open again` | none         | The arm stands off rest (more than a wrist a line-up left turned): shown 3 s, then the action fails `not_home`                                                                                  |
| `The wrist is turning back to rest`                               | Cancel       | The line-up is over and the wrist that looked turns back                                                                                                                                        |
| `Ending`                                                          | Cancel       | The line-up is over and no wrist turned: the run is ending                                                                                                                                      |

`progress.phase` is `looking` (the first three messages), `move`, `hold`,
`ready`, `go`, `timed_out` or `unable`,
and `pct` how near the container is to the boundary (100 inside). No message
is longer than 63 characters, the panel's longest before 2026-11-19. The
orchestrator's progress reports phase `line_up` at 0 with the same message (in
the middle of a run too, between the run's own phases); an instruction whose
distance alone changed, and `Hold still, checking` taking an instruction's
place or giving way to one, are reported no oftener than once a second (the
panel itself is sent as it always was). A container on the boundary's edge
reads inside it on one wrist frame and a centimetre or two outside on the
next: the instruction (`Move closer a little`) then stands, on the panel too,
and `Hold still, checking` takes its place only once the container has read
inside for half a second without a break. The question is not held back by
that: it is asked the moment the container is inside, still and planned for.
The run starts only on **Open**, and only where it is pressed on a `Ready`
panel whose container was seen within the last second; then the panel is the
run's. `Ready` is kept through half a second of frames that cannot be judged or
say no, so one bad frame does not take the button away. **Cancel** is the
orchestrator's cancel, as on the run's panel, and is read from the first panel
to the run's last. With `wrist_first` off nothing moves while lining up; with
it on the fingers close and the wrist turns to look as soon as the line-up
starts, and turns back if it ends without **Open** (see below). No world state
is written while lining up, but for an arm left holding (`arm_holds`).

The 35 s are what the orchestrator's 120 s leave: the run's 90 s budget less
a long run after its go (55 s: about 37 s clean by the two runs of 2026-11-12
and the pacing since, more where a press or a grip is made twice), so a run
started on the last second still ends its own way and is not stopped part way.
A run's own budget is shortened by what its line-up took. After the go of a
line-up whose wrist held the container, the run's first scene look is 2 s, not
its usual 15: on both runs of 2026-11-12 that look found nothing in front of
the arm, and the arm would stand through it with the wrist turned.
`not_lined_up` is recoverable, and its message carries the watch's last word
on why no container was seen, where that is why. Where no camera saw the
container at any time of the line-up the message says that, not that the
container was not where the arm opens it:
`no camera saw the container within 35 s: nothing was pressed (move the chair so that the container is in front of the arm, then press open again)`.

The boundary is not typed in. It is a grid of places where the run's own proof
plans a press (`reach_zone`, filled by `reach_probe` from the robot's URDF and
chair model, for the lid height the cameras read) and is kept under
`state_dir`'s `reach_zone`. `line_up_far_m` bounds how far out it is trusted:
a container further out is told to come closer at once, with no grid needed,
and no proof is spent past it. A lid height read a little either side of a
grid's 2 cm bucket keeps the grid it has.
A proof takes half a second to two (about one on average), and the grid is
filled from the container outward: on a height never met before, a container in place is judged after
some 4 s (flatbot's computer, 2026-11-16), and the whole grid, some 500 proofs,
takes 8 minutes across the line-ups that follow; each line-up keeps what it
asked as it ends. `reach_probe --urdf <the arm's URDF> --lid-z <height>` fills
one ahead of time: run in the image the node runs in and without `--table-z`
(a grid's key follows the code, and a fill on a table is another grid).
`reach_probe` runs at low priority (nice 10) on one thread; if it ends or
cannot start, the line-up ends at once (`line_up_unavailable`, with what it
said), never a whole line-up of `Hold still, checking`.

What follows the container while the panel is up depends on `wrist_first`:

- **`wrist_first` on (the wrist looks):** the run's own command line is started
  at once with `--line-up`, so that one process alone ever commands the arm. If
  the arm stands at rest, or at rest but for a wrist an earlier line-up left
  turned, it closes the fingers, as every run does at its start, and turns the
  wrist to the front view that shows the most of where the arm presses: the
  three wrist joints, once, flown as a look (slow and eased). From there it judges every wrist frame as it comes, the user
  driving the chair the while. A lid-sized top beyond the arm's reach is no
  rival of one within it (the run's own rule); where none is within reach yet
  the nearest alone is followed roughly, enough to say which way to move. With
  the orchestrator's selection a find is said only where the run would take it
  for the selected container (the run's own tests; `wrist_alone` for a list
  typed by hand). While the wrist sees the container the scene camera's
  sightings are left out. On **Open** the node tells it to go and the same
  process runs on from that view. The four joints above the wrist, which
  rest within a few hundredths of a radian of the rest pose, are brought onto
  it in the same turn (a centimetre or two at the hand), as in the run's own
  wrist look. An arm that stands anywhere else is not moved by a line-up, and
  no waiting brings its **Ready**: the panel says to home it and the action
  ends at once, `not_home`, recoverable.
- **`wrist_first` off:** `box_watch` follows the container by the scene camera
  alone and holds no client for the arm or the gripper; nothing moves before
  **Open**, and the run starts after it.

Either way `reach_probe --serve` judges where the container stands (the grid,
and the run's proof for the box itself once it is inside), and can move nothing.

How a line-up that turned the wrist ends without **Open**: out of time, the run
is told so, turns the wrist back to rest as gently and ends, nothing pressed
(its exit 5); the panel says the wrist is turning back and a cancel stops that
turn too. **Cancel** (the panel's or the orchestrator's) is the run's Ctrl+C as
every cancel is: a turn in flight stops and the arm holds where it is, the
wrist where it looked. The next line-up looks on from there; a run with
`line_up` off homes first as always. A wrist turn of the line-up's own that
stopped part way, out or back, is the arm holding (the run's exit 6:
`arm_holds`, not recoverable, the arm's error in the world state), as is a run
that had to be stopped because it did not end within 25 s of being told.

**What it cannot see it never asks of.** Without the wrist the scene camera
alone follows the container, and on flatbot that camera does not see a lid whole
in front of the arm (the reach map of 2026-11-11): the panel says
`The arm cannot see the container. Move the chair in front of it`, though the
container may stand in front of the arm already, and the action ends
`not_lined_up`, with the watch's reason in the message. A container a camera's
frame cuts is placed roughly,
enough to say which way to move and never to ask on. With `line_up` off,
`open_box` runs as it always has. The mock lines up on a box that comes into
reach over six `mock_phase_s`, for rehearsing the UI. A line-up whose watch
cannot start, or ends of itself, fails (`line_up_unavailable`, or what the run
said: `not_ready` and the like): no run goes on without its **Open**.

Not yet run on the robot. The wrist's watch was replayed (2026-11-16) on the 20
wrist frames kept from the runs of 2026-11-11 to 11-13, each judged alone as
the watch judges it: as first written it found the container in 4 (another
lid-sized top in view, 1.08 m out on both runs that opened the box, and it
refused to guess); with the reach rule above, in 14, at the run's own place
(within 3 mm) on 9 of the 12 the run found and at no other place, 0.2 s a
frame. Whether it holds the container while the chair moves is unmeasured: the
frames are stills. A lid the one view cuts inside the boundary is said roughly
and never asked on, so the panel stays on `Hold still, checking` there.

#### Lining up in the middle of a run

A run that finds no press from where the chair stands is not ended. Run 2 of
2026-11-17 was (`FAILED at step 2 (PRESS) ... arm: where the look left it`),
with `line_up` off as the owner flies. Now, with `line_up` on or off:

- The run (the same process: it is started with `--guide`) says so with each
  tilt's reason (`NO WAY IN: ...`, no `STOP`), and brings the arm to rest if its
  look left it anywhere else: after the aimed button's press did not plan, it
  goes home from the look pose first, as it always has; from a side survey's
  pose it homes. A home that fails ends the run, the arm holding
  (`arm_holds`): the chair is never told to move with the arm standing out.
- Until the arm is at rest and the run has asked to be guided, nothing says
  the chair is to move: the panel and the progress say
  `No press plans from here: wait, the arm comes to rest first` (then
  `Going to rest` as that home ends), the console
  `next  the run does not end here: wait while the arm comes to rest`. The
  first word about moving is the line-up's own (the progress's
  `No press plans from here: ... Move the chair as told`, then the
  instructions).
- It forgets what its cameras measured from where the chair stood (the table's
  height and extent, the counter's edges it was held out of, the container's
  pose): after the move they are measured again, as at a run's first look.
- It asks the engine to guide and watches as a line-up before a run does: the
  wrist turned to look (with `wrist_first`; the wrist the run's own look turned
  looks on from where it stands), the same panel, the same boundary. The place
  where the run found no press is not inside for this line-up, whatever the
  boundary's own proof says of it: the run has the planner and the table's
  edges, the proof has neither, and the user is shown the way to a place that
  works, not asked on the place just refused. Every such place of the goal is
  held against each of its line-ups, and against the line-up carried to the
  next `open_box` (see "How long a line-up may take"), by where it is on the
  table whatever height the watch reads the lid at: the 5 cm cell of the
  boundary that holds it does not work, so the container must stand 3 cm or
  more outside that cell to be asked on. (A watch that reads the unmoved
  container further than that from where the run had it, as the scene camera
  can against the wrist's, is not caught by this.)
- On the user's yes the run is told to go, finds the container again from where
  the wrist stands and carries on; its own panel has the screen again, saying
  `Looking for the container again` until the run's next step, and so does the
  progress: nothing tells the user to move once they have said yes. If no
  press plans there either it asks again, and goes on only on another yes.
- Without the yes it ends as a line-up before a run does: **Cancel** (or `n`)
  is the run's Ctrl+C, read at every moment; out of time the run turns its wrist
  back to rest and the action fails `not_lined_up`, recoverable, its message
  carrying the last instruction (`last told: Move closer, about 15 cm`) and why
  no press had planned. If what judges the boundary (`reach_probe`) ends, the
  run is told to stop the same way (`line_up_unavailable`); if the run itself
  ends, the action fails with what it said.

What is guided: `no reach-out press plans for the scene camera's box` (before
the arm moves toward the container), `no press reaches the aimed button` (after
the look and the aim), and a lid-sized top the wrist's look sees out of the
arm's reach while the scene camera has no container (until now `not_found`).
With `wrist_survey` on (and no selection) the survey over the arm's side keeps
its turn before that last one: the container may stand beside the chair, and a
lid-sized top far off in front is then not it. Where the survey finds the
container the run opens it and nobody is guided; where it finds none either,
the chair is guided to the top in front (the arm brought to rest from the
survey's pose first).
A container no camera sees at all is guided too, since 2026-11-19 (below).
Not guided: the guarded press's own stops (`press_demo --guarded`, never the
engine's), and a run by hand (`press_demo` without `--guide`), which stops
there as it always has. Nothing more is tried to get in from where the chair
stands (a way round where the planner's path swings wide): the planning is as
it was.

#### A container no camera sees

The owner was asked (2026-11-19): "A box no camera sees at all still ends as
'not found'. Should that become guidance too ('move so the container is in
front of the arm')?" and answered yes. Until then such a run said `NO BOX`,
went home and failed `not_found`. Now, in every run the engine starts
(`--guide`), each find that ends with no container goes into the same guidance
as a container out of reach: the first look, the look after a line-up's yes,
and the look after a guidance's yes where the run's own look does not find what
the watch had held. That covers every reason the cameras give for having
looked and found none: nothing lid-sized in view, a lid a view cuts, two places
that could be the container, and a selected container that is in none of the
scene camera's frames or that no wrist find can be tied to.

- The run says `NOT SEEN: no camera sees the container (...)` with what each
  camera said (no `NO BOX`, no `STOP`), notes it in its record (the
  `reach_refused` row: `guided: true`, `what: "unseen"`, the look's seconds and
  the segmenter's verdict, then a `guide` row), brings the arm to rest if a
  side survey left it anywhere else, and asks the engine to guide with no place
  (`box: null`): none was measured, and nothing is held against the line-up.
- The arm rests with only its wrist turned. Where the wrist looks: where it
  looks for every line-up, the front view that shows the most of where the arm
  itself presses, solved on each run from the arm's kinematics, the chair model
  and the camera's own view. That is where the container has to be brought, and
  it stays true as the chair is driven. It is not turned toward the table as
  the cameras measured it before the move: where that stood in the arm's frame
  is true of nowhere once the chair has moved, which is why the run forgets it.
- The user is told at once, with **Cancel** alone:
  `The arm cannot see the container. Move the chair in front of it` (the run
  has looked, so not `Looking for the container` while the wrist turns). Where
  the cameras see more than one thing that could be the container and cannot
  tell which is meant: `The arm sees two possible containers. Move in front of only one`
  (a lid-sized top out of the arm's reach is no rival of one within it, so the
  chair's place can settle it). The user is never told to move the container or
  anything on the table: the scene is not changed, only where the robot stands.
- The moment a camera sees the container (the wrist's frames, or the scene
  camera's sightings while the wrist has none) the ordinary guidance takes
  over: which way and how far, `Hold still, checking`, `Ready. Open the container?`.
  Only what is known is said: a sighting the boundary cannot judge yet says
  `Hold still, checking` (the boundary is worked out from the first sighting
  on: `reach_probe` needs the lid's height to know which grid to use, so on a
  lid height it has not met before the first direction comes some seconds after
  the first sighting), and one with no known way in says
  `The arm cannot open it from here`. Lost again for 1.5 s, the user is told
  again that the arm cannot see it.
- Everything else is the guidance's own: the same time (see "How long a
  line-up may take"), **Cancel** or `n` at any moment ending with the run's
  Ctrl+C, nothing moving toward the container without a yes, and at the yes a
  fresh find. A user who never brings the container into view gets the end of
  one who never brings it into reach: `not_lined_up`, recoverable, the wrist
  back at rest, with the words for what was missing (the panel's
  `Container not seen in time. Press open again in front of it`, the error's
  `no camera saw the container within ...`). After that end the next `open_box`
  does not line up first: a line-up that never saw the container has nothing to
  go on from.

Under the orchestrator's 120 s this guidance is seldom given in the goal that
found no container: that find has spent the scene camera's whole first look
(15 s) and the wrist's views by then, which leaves under the 8 s a guidance
needs of the 35. The action then fails at once, `not_lined_up`, recoverable
(`no camera sees the container, and this run had no time left to guide the chair ...`),
and the next `open_box` lines up first with its whole 35 s, which is where the
user is told. A goal sent by hand is guided in the goal itself, for up to 10
minutes.

What still ends `not_found`: cameras that could not look at all. A guided run
is guided for an unseen container only where a camera did look: the scene
segmenter gave the look at least one of the scene camera's frames to judge, or
the wrist's finder judged a frame of its own. Where neither did (streams or TF
missing, the segmenter down or silent, no wrist frame), no moving of the chair
brings a container into view, and the run says `NO BOX` with what was missing
and ends as it always has. Once a line-up is under way the same distinction is
not made: a watch whose camera stops giving frames reads on the panel as a
container that is not seen, and its own reason is in the error's message.

Not flown, and never run against the robot's cameras: the run's side is tested
with its cameras and arm stood in for, the node's with the run stood in for by
a script that says what `press_demo` says.

#### With no UI

A goal with no UI is said and asked on the node's console, through the flight
view: one line each time what it says changes (a distance that only shrinks,
or `Hold still, checking` and an instruction by turns, no oftener than once a
second), and the question as `y / n`. Each is on the console once: the node's
own copy of a line-up's line (`line-up: Move closer, about 15 cm`) is kept in
the goal's file of full lines (`runs/console-<stamp>.log`), not on the console
beside the flight view's:

```text
 1 FIND      ok    box 0.79 m out (wrist camera, 1 view)                 8.0 s
 2 PRESS     FAIL  no press plans from here (0.79 m out): the slide in swings the arm wide (30, 35 deg); the press stroke would pass too close to the user (40, 45 deg)
             next  the run does not end here: wait while the arm comes to rest
   line up   ...   the arm rests and the wrist looks: move the chair as told
             say   Move closer, about 15 cm
             say   Hold still, checking
             ask   Ready. Open the container?   y / n
             how   answer in open_box_terminal, or publish y or n on /box_opening/answer (std_msgs/String)
 3 LINE UP   ok    yes                                                  23.0 s
   find      ...   looking for the box again
```

With the UI the same lines are on the console too, the question as
`Ready. Open the container?   (Open on the UI)`.

The node has no terminal of its own (it runs in a container under sheppy, and a
goal with no UI is sent by a one-shot `ros2 action send_goal`), so the answer
comes as a message: `/box_opening/answer`, `std_msgs/String`, `y` (or `yes`) or
`n` (or `no`), any case. The node hears it only while an action runs, and an
answer said before the action began is not kept.

- `y` is **Open**'s press: taken only while the question stands, on a container
  seen within the last second. One that comes at any other time is not a yes;
  the console says `that y was not to a question: answer once it asks`.
- `n` is **Cancel**, at any moment of the action, the run's flight included: the
  run's Ctrl+C, as every cancel is.
- With the UI live the answers are read as well as the panel's buttons.

`open_box_terminal` is the terminal for it, in the node's image:

```bash
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
  <the image the node runs> ros2 run box_opening open_box_terminal
```

It sends the `open_box` goal as a goal sent by hand is sent (no token), shows
what the node says as it goes (the action's progress: the run's steps, why no
press plans where none does, a line-up's instructions and its question), sends
the `y` or `n` typed and Enter (a `y` only while the question stands), and ends
with the action's result: `DONE`, `CANCELLED`, or `FAILED` with the error code
and whether it can be started again. Ctrl+C cancels the goal (the action's own
cancel) and then waits for that result as for any other, up to 45 s, the
node's own stop at its worst; a second Ctrl+C leaves without it. A Ctrl+C
before the goal is sent just leaves. From any other terminal
`ros2 topic pub --once /box_opening/answer std_msgs/msg/String "{data: y}"`
says the same. The mock lines up with no UI the same way, for rehearsing it.

Not yet run on the robot, nor against a node that moves an arm: the node's side
is tested with the run stood in for by a script that asks to be guided as
`press_demo` does, the run's side with its cameras and arm stood in for, and
the terminal whole against the mock node over DDS inside the test container
(`test_open_box_terminal_e2e.py`: a yes, an `n`, a Ctrl+C, each to the node's
result).

#### How long a line-up may take

| The goal                                                           | A line-up before the run       | A line-up in the middle of it                                                                                                                                                                           | The run's own budget                              |
| ------------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| with the orchestrator's token (from the UI, or the UI bridge down) | ends 35 s after the goal began | ends 35 s after the goal began: what the run's look and proof left of them (run 2 of 2026-11-17 said "no press plans" 16 s in: about 19 s). With under 8 s left it is not given: `not_lined_up` at once | 90 s from the goal's start, the line-up inside it |
| no token (sent by hand, `open_box_terminal`)                       | 10 minutes                     | 10 minutes, each time                                                                                                                                                                                   | 90 s from the last yes                            |

Under the orchestrator a line-up is given only where a run of 55 s still fits
after it inside the 90; the orchestrator cancels at 120 s whatever the engine
does. The 55 s are an estimate, not a measurement of a guided run (run 1 of
2026-11-17 took 33.5 s and the runs of 2026-11-13 about 40 s from the first row
to home): a guided run that takes longer still meets the run's own stop at
90 s, part way through a motion, as a run after a line-up before it can.
After an `open_box` that ended `not_lined_up` from a line-up in the middle of
its run, the next one within 5 minutes lines up before its first move whatever
`line_up` says, so that "press open again" goes on from the line-up with its
whole 35 s and not from another look and proof. Once: a line-up that then never
sees the container does not ask for another.

A goal sent by hand has no orchestrator to cancel it, and the owner's word for
it is "don't end, just tell you how far to move until it's close enough then
asking if ready y/n". Each of its line-ups waits 10 minutes for the person
(`CONSOLE_LINE_UP_S` in `node.py`), the arm at rest with its wrist turned the
while, and then ends as a timed one does (the wrist back to rest,
`not_lined_up`, recoverable). The time a person takes is no part of such a
run's budget: its 90 s start again at each yes.

A `CANCEL` on `/box_opening/control` that comes with no action running is
taken for the action about to start (the user's cancel overtook the goal): a
goal that starts within 3 s of it ends cancelled before anything runs, once.
With the token pushed ahead of the goal (the protocol before 2026-11-16) it is
kept for that token's execution instead.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file.

| Name                  | Default                | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_orchestrator`    | `true`                 | `false` runs standalone: no registration, drive it by CLI                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `mock`                | `false`                | `true` skips hardware (the mock alternative sets it)                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `ui_bridge_node_name` | `"ui_bridge_node"`     | The UI bridge's node, whose `request_user_input` the base class's panel calls go to (see "The UI")                                                                                                                                                                                                                                                                                                                                                                                                               |
| `mock_phase_s`        | `0.0`                  | The mock only: how long each of its phases lasts (s); `0` and its run is over at once. A second or two rehearses the UI on it with no arm (the panel, its bar, Cancel)                                                                                                                                                                                                                                                                                                                                           |
| `line_up`             | `false`                | Line up before a run's first move: the user is told which way to move, asked, and the run goes on only on their yes (**Open** on the UI; `y` on `/box_opening/answer` for a goal with no UI: see "Lining up first"). Off (the default), `open_box` starts its run at once and lines up only where it finds no press from where the chair stands ("Lining up in the middle of a run"). With `wrist_first` the wrist turns to look while the user lines up; without it only the scene camera follows the container |
| `line_up_far_m`       | `0.0`                  | How far from the arm's base (m) the line-up trusts the boundary: a container further out is told to come closer. `0`: as far as the run's proof plans a press (on flatbot about 0.90 m; flown at 0.63 and 0.68 m)                                                                                                                                                                                                                                                                                                |
| `air_control`         | `"impedance"`          | How the arm flies the free air before the press: `impedance`, every leg in the driver's joint impedance, one motion to under the button (as flown 2026-11-12); `position`, the look and the way in's free-air lines in position control and slower, about a second's stand while the springs come in, then the press on them as ever (see "The free air before the press"). `position` is not flown                                                                                                              |
| `allow_execute`       | `false`                | `true` only for attended runs; `false`: `open_box` unavailable, nothing runs                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `container`           | `"ankou_pink.yaml"`    | The container's config, in the image's `config/containers`, or a path                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `world`               | `"world_flatbot.yaml"` | The robot's planning frame, cameras and fixed body, in the image's `config`                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `state_dir`           | `""`                   | Where runs keep their logs, captures and videos; `""` = `~/.ros/box_opening`                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `video`               | `false`                | Record both cameras from the grip to the set-down (the newest 20 kept); on for attended data collection                                                                                                                                                                                                                                                                                                                                                                                                          |
| `viz`                 | `false`                | Draw what the perception sees and decides for Foxglove                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `mission_frames`      | `false`                | Debug: each executed run keeps its camera frames (~10 MB, the newest 20), and its run record's flight files and room maps                                                                                                                                                                                                                                                                                                                                                                                        |
| `speed_scale`         | `1.0`                  | Every motion slowed to this fraction of its speed (0 \< x \<= 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `body_layers`         | see below              | The planner config whose chair and user the arm is held out of; `""` turns the check off (a bench with no chair)                                                                                                                                                                                                                                                                                                                                                                                                 |
| `planner_worlds_as`   | `""`                   | The folder the planner's container mounts for box opening's worlds, as the planner names it; `""` = no world is sent (see "The planner's world")                                                                                                                                                                                                                                                                                                                                                                 |
| `planner_worlds_dir`  | `""`                   | The same folder as this node writes it; `""` = `planner_worlds` under `state_dir`                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `wrist_survey`        | `false`                | Where the scene camera finds no container, look for it with the wrist camera (see below); needs the body model on                                                                                                                                                                                                                                                                                                                                                                                                |
| `wrist_first`         | `false`                | Facing a table: the wrist camera finds the container in front of the arm, the wrist turned where the arm rests (see below)                                                                                                                                                                                                                                                                                                                                                                                       |
| `wrist_alone`         | `false`                | With `wrist_first`: a selection the scene camera's detections do not hold (a list typed by hand) is looked for by the wrist alone, taken only where it sees one container (see below)                                                                                                                                                                                                                                                                                                                            |
| `full_console`        | `false`                | A run's full lines on the node's console too, between the flight view's. Off: they are kept in `runs/console-<stamp>.log` of the state folder (the result's last line names it), and the console is the flight view                                                                                                                                                                                                                                                                                              |

### The free air before the press

A run flies every leg in the driver's joint impedance (the press needs its
spring), and in that mode the arm follows loosely: on the two runs that opened
the box on 2026-11-12 the look and the way in flew 7-10 mm behind their
reference (20-28 mm at most), hung 4-8 mm under their path and ended the look
8-10 mm off it. With `air_control: position` the free air before the press
flies in position control instead:

- the look (`reach:look`) at half each joint's speed limit, and the way in's
  free-air lines (`reach:re-aim`, `reach:slide-in`, `depth:turn`) at 0.3 of it,
  both easing out of rest at 2 rad/s² and into it at 1.5;
- the arm then stands at the stroke's start while the driver is put in its
  joint impedance there and its spring fades in (0.8 s, run-log row
  `air_switch`);
- the stroke (`depth:approach`, `depth:press`), the hold, the grip and
  everything after fly as they did. The grip's own way down to the knob stays
  one motion in the impedance.

What it costs: the way in is two motions with a stop between, not one, and a
run is about 2.5 s longer (re-timed on those two runs' paths: the look 1.4 to
1.6-1.9 s, the free air 1.8-2.1 to 2.6-3.2 s, and the stand). And in position
the arm does not give way to what it meets in the air as the spring does; these
legs carry no contact guard (the guard a position run's lines carry needs an
at-rest reading before each and 0.15 of the speed limit: not a reuse), and they
end 35-60 mm over where the cameras put the lid. A driver that will not enter
its impedance at the stroke's start (asked twice) gets no stroke: nothing is
pressed, and the arm leaves up and back clear of the box before home, as it
does where the stroke's own goal does not start. The default, `impedance`, is
the run as flown; any other value is refused before anything runs
(`bad_config`). `position` has not been flown.

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
has found none (the chair is then guided, see "A container no camera sees"; a
run by hand stops with `not_found`). On, such a run surveys with the
wrist camera first (the Survey step above), adding about 20-40 s: choosing
the poses (about 10 s), up to two survey poses and two looks again, each a
planned move, a still frame and the finder (`tasks/survey.py`,
`perception/wrist_finder.py`). It runs only for a
run without a selection (see the gap below), only with the body model on, and
the survey's frames are kept as captures (`captures/wrist-survey-*`). The
heights the wrist reads are its own (its mount's error, more the more its view
tilts): the press goes by the higher of the survey's top and the aim's, and
the press's contact band refuses the rest (`contact_early`).

`wrist_first` is off by default. On (a table faced from the front, where the
scene camera sees no lid whole where the arm presses), a run looks for the
container with the wrist camera before anything else: the three
wrist joints turn where the arm rests, the arm above them standing still, so the
camera sees the table in front. The views are chosen on each run from the arm's
kinematics, the chair model and the camera's own view, for any table height of
0.15-0.35 m in `arm_mount`: up to four, the one that shows the most of where the
arm presses first, and before them one aimed where the scene camera has the
container, if it has it at all (a fix placed roughly from a box its frame cut
included). Each view is a planned move of the wrist, a still frame and the
finder; the run goes on at the first that shows the lid whole. The move is flown
as a look (`LOOK_SPEED`, `LOOK_ACCEL`, `LOOK_DECEL` in `constants.py`): a quarter of each joint's speed limit, easing out of rest at
0.8 rad/s² and into it at 0.6, under the run's own gains (slow, not loose), so the hand turns slowly
where the user sits (a 0.6 rad turn of the three wrist joints takes 2.9 s where
it took 1.7 s as a transit; not yet flown). A
lid a view cuts is looked at again with the wrist turned to it (at most twice a
run) and is the container only where that look shows it whole. What the wrist
finds is taken over the scene camera's; where it finds none the scene camera's
fix is used, and with neither the run has found no container and says what each
camera said (the chair is guided, see "A container no camera sees"; a run by
hand stops with `not_found`). It adds about 3 s where the first view shows the container and
up to about 15 s where none does (estimates from the look's re-timing); its frames are kept as captures
(`captures/wrist-survey-*`). Only with the body model on.

With the orchestrator's selection the wrist looks too, and what it finds is the
selected container only where it is tied to the selection. The selection is one
of the scene camera's detections, so the tie is the scene camera's own view of
that instance: the wrist's lid must stand no more than 75 mm to either side of
the scene camera's line of sight to it, and, where the scene camera placed the
selected container (a fix: whole, cut by its frame, or placed roughly), within
0.20 m of that place along the line; with no fix, between the lines of sight to
the two sides of the instance's own box in the scene image (a lid's width more
past a side the frame cuts it at). A camera has a thing's direction right and
its range wrong (the two cameras' fixes of one box differed 80 mm along that
line and 8 mm across it), and two containers cannot stand nearer than 152 mm
centre to centre, so a lid within 75 mm of the line is the selected one and not
a neighbour beside it. A second lid-sized top in the same direction, a find that
is not tied, or no find: the run goes on with the scene camera's own fix of the
selected container where that is a careful one, and has found no container
where it is only a rough one or there is none, saying what each camera said.
Where the selected container was in none of the scene camera's frames the wrist
is not turned: nothing would tie its find to the selection. Either way the
chair is then guided (see "A container no camera sees"; a run by hand stops
with `not_found`): the guidance's watch says a wrist find only where the run
would take it for the selected container, so the user is guided until the scene
camera has the selected container. The instance the
world state is written for is the selected one in every case.

`wrist_alone` (off by default) is for that last case: a selection that is no
detection of the scene camera's at all, as when the UI's objects come from a
list typed by hand (the orchestrator's `interactive_perception_node`) and the
selected id is in none of the segmenter's frames. With it on, the wrist looks
from its own views, tied to nothing, and what it finds is taken only where it is
the one container its frames hold; a second lid-sized top anywhere in them (a
place that passes, or a top a view cut), or two places the finder cannot choose
between, and the run has found no container, saying so (the chair is guided,
the user told `The arm sees two possible containers. Move in front of only one`;
a run by hand stops with `not_found`). The look stops at the
first view that shows a lid whole, so "one" is one in the views looked from by
then, not on the whole table. With it on the arm opens whichever single
container stands in front of it, under the selected id: nothing checks that it
is the one the user meant. A selection the scene camera does hold is tied as
above, with or without it.

### The side survey and the orchestrator's selection: an interface gap

(This is `wrist_survey`, the poses over the arm's side. `wrist_first`, above,
does run with a selection.)

The orchestrator selects a container from the scene camera's detections
(`/perception/object_detections`, by `instance_id`), and passes that selection
to `open_box`. A container the scene camera cannot see, the case the survey is
for, has no detection and no `instance_id`, so the orchestrator cannot select
it, and `open_box` is never offered for it. And for a selected container the
scene camera does not find at the run, the wrist cannot tell which container
it is: the survey would find whatever stands on the side counter. So a run with
a selection does not survey; it has found no container, saying so (and the
chair is guided for it as for any container no camera sees).
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
