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
order: `locate` (0 with `looking for the container` once the run is up, and
with `going to the start pose` where the arm does not rest at its start pose
and the run's first motion takes it there; 0.03 for each wrist survey pose, its
message `looking for the container with the wrist camera`; 0.05 once the
container is found), `look` (0.1), `press` (0.35), `grip` (0.5), `carry`
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

| Code                  | Recoverable | Meaning                                                                                                                                                                                                                                                                  |
| --------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bad_params`          | no          | The goal had params; `open_box` takes none. Nothing ran                                                                                                                                                                                                                  |
| `not_allowed`         | no          | `allow_execute` is off on this node. Nothing ran                                                                                                                                                                                                                         |
| `bad_config`          | no          | A setting has a value it cannot have (`air_control` neither `impedance` nor `position`). Nothing ran                                                                                                                                                                     |
| `line_up_unavailable` | yes or no   | With `line_up` on: the goal has no UI to ask on (recoverable: set `line_up` false, or send it from the UI), or what watches the container or judges its place could not start or ended (not recoverable; the message carries what it said). The run was never told to go |
| `not_lined_up`        | yes         | With `line_up` on: the container was not where the arm opens it within 40 s, or the UI went. Nothing was pressed; a wrist that looked is back at rest                                                                                                                    |
| `timeout`             | yes         | The run was still going after 90 s and was stopped its own way (one Ctrl+C); its stop's worst (a second Ctrl+C, then a kill) still ends before the orchestrator's fixed 120 s cancel; not recoverable with the lid still in the fingers                                  |
| `not_ready`           | yes         | Refused before anything moved: a driver, planner, camera or TF missing                                                                                                                                                                                                   |
| `not_found`           | yes         | The scene camera found no container (with `wrist_survey`, nor did the wrist camera's survey: nothing, or two places that could be it); nothing was pressed                                                                                                               |
| `not_home`            | yes         | The arm could not get to its start pose                                                                                                                                                                                                                                  |
| `contact_early`       | yes         | The press met something more than 45 mm above the button's top as the cameras saw it: stopped there, nothing pressed, the arm went home                                                                                                                                  |
| `contact_late`        | yes         | The press met nothing down to 25 mm below that top, or at all: the box is not where it was seen; the arm went home                                                                                                                                                       |
| `watch_failed`        | no          | The press's contact watch failed: the press stopped there                                                                                                                                                                                                                |
| `box_moved`           | yes         | The press moved the container                                                                                                                                                                                                                                            |
| `not_popped`          | yes         | Pressed, but the knob did not come up (or the stroke pressed nothing); the arm went home                                                                                                                                                                                 |
| `no_button`           | yes         | The look found no button; nothing was pressed                                                                                                                                                                                                                            |
| `no_press`            | yes         | No press plans to the button; nothing was pressed or moved                                                                                                                                                                                                               |
| `lid_put_back`        | yes         | No carry for the lid: it is back on the box and the arm home                                                                                                                                                                                                             |
| `lid_dropped`         | no          | The arm got home, but the lid slipped out of the fingers on the way                                                                                                                                                                                                      |
| `arm_holds`           | no          | A motion stopped part of the way: the arm holds there until cleared                                                                                                                                                                                                      |
| `arm_halted`          | no          | The driver halted a motion (the e-stop) or took control away: the run sent nothing after it                                                                                                                                                                              |
| `stopped`             | no          | The run was ended from outside                                                                                                                                                                                                                                           |
| `failed`              | no          | Anything else, a traceback included; the message is the run's last STOP line (or line). Nothing says where the arm is                                                                                                                                                    |

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

The panel asks the user nothing and the run never waits on it. It is sent as
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

**With `line_up` on the arm never flies without the user's Open.** A goal that
has no UI to ask on (it came without the orchestrator's token, as one sent by
hand does, or the UI bridge is not there) is refused before anything starts:
`line_up_unavailable`, recoverable, nothing moved. Set `line_up` false to run
without lining up. Every goal's settings (`line_up`, whether it has the UI,
`line_up_far_m`, `air_control`, `wrist_first`, `wrist_alone`) are in the node's
log as it starts: a `ros2 param set` lives only in the running node, and a
restart puts the deployment's values back.

| Panel message                            | Buttons      | When                                                               |
| ---------------------------------------- | ------------ | ------------------------------------------------------------------ |
| `Looking for the container`              | Cancel       | No container seen in the last 1.5 s                                |
| `Move closer, about 15 cm`               | Cancel       | Outside the boundary: closer, back, left or right, the larger miss |
| `Hold still, checking`                   | Cancel       | Inside, not yet still for 1 s, or not yet judged                   |
| `Ready. Open the container?`             | Open, Cancel | Inside, still, and the run plans for it there                      |
| `Press open again when you are in place` | none         | 40 s passed: shown 3 s, then the action fails `not_lined_up`       |
| `The wrist is turning back to rest`      | Cancel       | The line-up is over and the wrist that looked turns back           |

`progress.phase` is `looking`, `move`, `hold`, `ready`, `go` or `timed_out`,
and `pct` how near the container is to the boundary (100 inside). The
orchestrator's progress reports phase `line_up` at 0 with the same message.
The run starts only on **Open**, and only where it is pressed on a `Ready`
panel whose container was seen within the last second; then the panel is the
run's. `Ready` is kept through half a second of frames that cannot be judged or
say no, so one bad frame does not take the button away. **Cancel** is the
orchestrator's cancel, as on the run's panel, and is read from the first panel
to the run's last. With `wrist_first` off nothing moves while lining up; with
it on the fingers close and the wrist turns to look as soon as the line-up
starts, and turns back if it ends without **Open** (see below). No world state
is written while lining up, but for an arm left holding (`arm_holds`).

The 40 s are what the orchestrator's 120 s leave: the run's 90 s budget less
the longest run logged (50 s with its start), so a run started on the last
second still ends its own way. A run's own budget is shortened by what its
line-up took. After the go of a line-up whose wrist held the container, the
run's first scene look is 2 s, not its usual 15: on both runs of 2026-11-12
that look found nothing in front of the arm, and the arm would stand through it
with the wrist turned. `not_lined_up` is recoverable, and its message carries
the watch's last word on why no container was seen, where that is why.

The boundary is not typed in. It is a grid of places where the run's own proof
plans a press (`reach_zone`, filled by `reach_probe` from the robot's URDF and
chair model, for the lid height the cameras read) and is kept under
`state_dir`'s `reach_zone`. `line_up_far_m` bounds how far out it is trusted:
a container further out is told to come closer at once, with no grid needed,
and no proof is spent past it. A lid height read a little either side of a
grid's 2 cm bucket keeps the grid it has.
A proof takes about half a second, and the grid is filled from the container
outward: on a height never met before, a container in place is judged after
some 4 s (flatbot's computer, 2026-11-16), and the whole grid, some 500 proofs,
takes 8 minutes across the line-ups that follow; each line-up keeps what it
asked as it ends. `reach_probe --urdf <the arm's URDF> --lid-z <height>` fills
one ahead of time: run in the image the node runs in and without `--table-z`
(a grid's key follows the code, and a fill on a table is another grid).
`reach_probe` runs at low priority (nice 10) on one thread; if it ends or
cannot start, the line-up ends at once (`line_up_unavailable`, with what it
said), never 40 s of `Hold still, checking`.

What follows the container while the panel is up depends on `wrist_first`:

- **`wrist_first` on (the wrist looks):** the run's own command line is started
  at once with `--line-up`, so that one process alone ever commands the arm. If
  the arm stands at rest, or at rest but for a wrist an earlier line-up left
  turned, it closes the fingers, as every run does at its start, and turns the
  wrist to the front view that shows the most of where the arm presses: the
  three wrist joints, once, flown as a look (slow and eased), the arm above
  them standing. From there it judges every wrist frame as it comes, the user
  driving the chair the while. A lid-sized top beyond the arm's reach is no
  rival of one within it (the run's own rule); where none is within reach yet
  the nearest alone is followed roughly, enough to say which way to move. With
  the orchestrator's selection a find is said only where the run would take it
  for the selected container (the run's own tests; `wrist_alone` for a list
  typed by hand). While the wrist sees the container the scene camera's
  sightings are left out. On **Open** the node tells it to go and the same
  process runs on from that view. An arm that stands anywhere else is not
  moved before **Open**: only the scene camera looks, and the panel's end
  reason says the arm is not at its rest pose (home it, then press open again).
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
in front of the arm (the reach map of 2026-11-11): the panel stays on
`Looking for the container` and the action ends `not_lined_up`, with the watch's
reason in the message. A container a camera's frame cuts is placed roughly,
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

A `CANCEL` on `/box_opening/control` that comes before its goal has no run to
cancel in the base class. With the token pushed ahead of the goal (the protocol
before 2026-11-16) the node kept it for that execution, which then ended
cancelled before anything ran; with the token in the goal there is nothing yet
to keep it for, and the goal that follows runs. With `line_up` on its first
panel has a Cancel.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file.

| Name                  | Default                | Meaning                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_orchestrator`    | `true`                 | `false` runs standalone: no registration, drive it by CLI                                                                                                                                                                                                                                                                                                                                           |
| `mock`                | `false`                | `true` skips hardware (the mock alternative sets it)                                                                                                                                                                                                                                                                                                                                                |
| `ui_bridge_node_name` | `"ui_bridge_node"`     | The UI bridge's node, whose `request_user_input` the base class's panel calls go to (see "The UI")                                                                                                                                                                                                                                                                                                  |
| `mock_phase_s`        | `0.0`                  | The mock only: how long each of its phases lasts (s); `0` and its run is over at once. A second or two rehearses the UI on it with no arm (the panel, its bar, Cancel)                                                                                                                                                                                                                              |
| `line_up`             | `false`                | Line up on the UI before a run: the panel says which way to move, asks, and the run goes on only on **Open** (see "Lining up first"). With `wrist_first` the wrist turns to look while the user lines up; without it only the scene camera follows the container                                                                                                                                    |
| `line_up_far_m`       | `0.0`                  | How far from the arm's base (m) the line-up trusts the boundary: a container further out is told to come closer. `0`: as far as the run's proof plans a press (on flatbot about 0.90 m; flown at 0.63 and 0.68 m)                                                                                                                                                                                   |
| `air_control`         | `"impedance"`          | How the arm flies the free air before the press: `impedance`, every leg in the driver's joint impedance, one motion to under the button (as flown 2026-11-12); `position`, the look and the way in's free-air lines in position control and slower, about a second's stand while the springs come in, then the press on them as ever (see "The free air before the press"). `position` is not flown |
| `allow_execute`       | `false`                | `true` only for attended runs; `false`: `open_box` unavailable, nothing runs                                                                                                                                                                                                                                                                                                                        |
| `container`           | `"ankou_pink.yaml"`    | The container's config, in the image's `config/containers`, or a path                                                                                                                                                                                                                                                                                                                               |
| `world`               | `"world_flatbot.yaml"` | The robot's planning frame, cameras and fixed body, in the image's `config`                                                                                                                                                                                                                                                                                                                         |
| `state_dir`           | `""`                   | Where runs keep their logs, captures and videos; `""` = `~/.ros/box_opening`                                                                                                                                                                                                                                                                                                                        |
| `video`               | `false`                | Record both cameras from the grip to the set-down (the newest 20 kept); on for attended data collection                                                                                                                                                                                                                                                                                             |
| `viz`                 | `false`                | Draw what the perception sees and decides for Foxglove                                                                                                                                                                                                                                                                                                                                              |
| `mission_frames`      | `false`                | Debug: each executed run keeps its camera frames (~10 MB, the newest 20), and its run record's flight files and room maps                                                                                                                                                                                                                                                                           |
| `speed_scale`         | `1.0`                  | Every motion slowed to this fraction of its speed (0 \< x \<= 1)                                                                                                                                                                                                                                                                                                                                    |
| `body_layers`         | see below              | The planner config whose chair and user the arm is held out of; `""` turns the check off (a bench with no chair)                                                                                                                                                                                                                                                                                    |
| `planner_worlds_as`   | `""`                   | The folder the planner's container mounts for box opening's worlds, as the planner names it; `""` = no world is sent (see "The planner's world")                                                                                                                                                                                                                                                    |
| `planner_worlds_dir`  | `""`                   | The same folder as this node writes it; `""` = `planner_worlds` under `state_dir`                                                                                                                                                                                                                                                                                                                   |
| `wrist_survey`        | `false`                | Where the scene camera finds no container, look for it with the wrist camera (see below); needs the body model on                                                                                                                                                                                                                                                                                   |
| `wrist_first`         | `false`                | Facing a table: the wrist camera finds the container in front of the arm, the wrist turned where the arm rests (see below)                                                                                                                                                                                                                                                                          |
| `wrist_alone`         | `false`                | With `wrist_first`: a selection the scene camera's detections do not hold (a list typed by hand) is looked for by the wrist alone, taken only where it sees one container (see below)                                                                                                                                                                                                               |

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
fix is used, and with neither the run stops with `not_found`, saying what each
camera said. It adds about 3 s where the first view shows the container and
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
selected container where that is a careful one, and stops with `not_found`
where it is only a rough one or there is none, saying what each camera said.
Where the selected container was in none of the scene camera's frames the wrist
is not turned: nothing would tie its find to the selection. The instance the
world state is written for is the selected one in every case.

`wrist_alone` (off by default) is for that last case: a selection that is no
detection of the scene camera's at all, as when the UI's objects come from a
list typed by hand (the orchestrator's `interactive_perception_node`) and the
selected id is in none of the segmenter's frames. With it on, the wrist looks
from its own views, tied to nothing, and what it finds is taken only where it is
the one container its frames hold; a second lid-sized top anywhere in them (a
place that passes, or a top a view cut), or two places the finder cannot choose
between, and the run stops with `not_found`, saying so. The look stops at the
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
