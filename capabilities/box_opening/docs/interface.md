# Box Opening Capability — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @ChrissCox
**Tier:** experimental

## What it does

Opens the food container: presses the button on its lid so the lid comes
loose, grips the lid by that button, lifts it off, sets it down on the table
beside the container, and goes home. It has one action, `open_box`:

1. **Ready:** the run checks the robot is ready (the planner, the arm driver
   and its link to the arm, that no other program drives the arm, the TF from
   the arm to the wrist camera's mount, the scene camera's calibration file) and
   refuses before anything moves if not; an arm away from its rest pose is
   brought there first (one parked inside the chair's padding steps out of it
   first, slowly: "The rest pose" below). No camera's stream is checked there: a camera that
   delivers nothing is met at the first look (see "A container no camera sees").
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
   the wrist camera aims at the button. Where it finds no button there, the
   wrist's own container finder is asked of a frame from that pose: a lid it
   finds elsewhere than the look was aimed (half a lid or more away), which is
   the selected container by the run's own rule, becomes the run's box; the
   look is planned again over it, as every look is, and flown, and the aim
   asked there (see "The look again"). At most two looks again; with no lid in
   that frame, or no button from them either, `no_button`.
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
   Where none of those spots has a carry that plans, more are looked for before
   anything else (owner 2026-11-19: "it should keep trying spots, assume tere
   will always be a spot for the lid"), every one through the same checks:
   the whole of the table the cameras measured (places 4 cm apart, the nearest
   where the hand holds the lid first; on a table with more than 48 such
   places, 48 of them evenly from the nearest to the farthest; the hand
   straight down and then leaning 20 deg; the spots round the container each
   stop after a few tries and none lies further than 0.45 m from it; a place
   taken straight down must have the room a clear spot has, the lid's and the
   hand's margins from what stands and from what was not seen), then
   every spot once more with the hand leaning 30 deg, as
   far as the press's own look pose leans. 20 deg was the most a set-down
   leaned, for how the lid lands and for the canister under the carry, and
   both are kept: a hand leaning 30 deg is taken only where the lid still
   lands tilted 20 deg at most (where the grip itself leaned), and its carry
   is refused where the lid's lowest point would pass under the top it came
   off, over the canister. Still none, and the map was not made with the
   wrist's frame from where the arm stands: the wrist looks at the table from
   there, its frame is pooled into the map, and the search goes once more.
   The carry to any of these further spots, and to every spot after the
   first (below), is asked more than the first spot's own, because the
   planner knows no table and a path of its own between two places over the
   table can sink a hand's width on the way. It is the arm's own walk first,
   a straight line from where the arm stands to over the spot at that
   height, and the planner's path only where no walk flies. Whichever it is,
   it is flown only if along the whole of it, between its waypoints too, the
   lid's lowest point stays within 10 mm of the height it is carried at (the
   lower of the path's two ends), over everything the map shows standing
   under the path the lid and the hand sweep, and never under the top the
   lid came off, over the canister. The lid is taken 13 mm lower than its
   model has it for all of this: that is how much sooner than planned the
   recorded set-downs met the table. The hand's own body is asked too: the
   hand's and the wrist's spheres of the arm's body model, all along that
   path, 20 mm clear of everything the map shows standing. A hand that leans
   to such a spot is taken only where those same spheres, at the set-down's
   own joints all the way down, stay 20 mm clear of everything standing; the
   on-table check above knows only a hand straight down. With no body model,
   or one whose sphere file gives the hand or either link of the wrist no
   sphere (the run says so once), no such leaning set-down is taken, and a
   carry is asked of the lid and a hand straight down over it alone. The first spot's own carry and set-down are
   planned as they have been flown: of the 22 first carries on record 18
   pass the path check and 4 do not (three sank 12-59 mm on the way, one
   passed its wrist camera by something standing), and the one recorded
   leaning set-down with a map kept 145 mm from everything standing.
1. **Set down:** fast to 80 mm over where the run has the lid meet the table,
   then slowly, to 8 mm under it at most. That height is the cameras' table
   moved by what the fingers found: the table is taken as far under the lid's
   top as the cameras saw it, the top where the fingers hold the knob (on
   2026-11-13 and 11-17 the wrist camera read the lid's top, and so the table,
   46-55 mm low). The arm feels for the table on the slow stretch, from 0.6 s
   after it begins (the arm is still catching up with its reference until then,
   which reads as a contact): the descent stops where the lid meets the table.
   A rise that began more than 35 mm over where the table should be is not the
   table, and the descent goes on, the arm settling again for 0.6 s. Where a
   rise began is where the pads last came freely down; what holds them up from
   there on is one thing. The lid is held up, whatever the settle says, where
   the pads come down less than 2 mm while their reference comes down 12, or
   stand 15 mm over where they last came freely down: that stops the descent.
   It is the table only where the pads had come freely down to within those
   35 mm of where the table should be; met higher, it is something else, and
   the lid is not let go. Nothing is planned while the set-down flies, so every sample of it is
   read as it comes; what was not read by its end is read then. With the spot
   chosen after the lift, the way home is planned while the arm rises off the
   lid.
1. **No table felt: another spot.** A set-down that ends without the arm
   feeling the lid meet the table (nothing met down to its end, or something
   holding the lid up well above where the table can be) does not let the lid
   go, and does not take it back to the box either (owner 2026-11-19: "it
   should keep trying spots, assume tere will always be a spot for the lid").
   The arm rises with the lid to the carry's height, and the spot is set
   aside for the run with the footprint that was left there. Where nothing
   was felt down to the touch's end, that is the lid's own: any spot whose
   centre is nearer than the lid is wide. Where something held the lid up,
   something stands there that the map did not show: the wrist looks at the
   table from where the arm stands before anything is chosen, and no part of
   the next lid or hand comes down over where this lid or hand did (two
   hands straight down stand 20 cm apart at the least). That look cannot
   show what held the lid: it stands under the lid in the fingers, and the
   wrist's frame leaves out everything within the lid's radius and 30 mm of
   the pads. It shows the table round it; what keeps the next spot off what
   held this lid is the footprint set aside, not the map. The next spot is
   chosen by the same rules in the same order, its carry asked what every
   carry after the first is (above: the walk first, at the carry's height
   over the table and what stands on it, off the open canister); and the lid
   is carried there and set down the same watched way. Spot after spot, each
   said on the console and in the run record (`lid_spot_left`, and a
   `lid_spot` row a spot with the spots left before it), but never without
   end:
   - at most 6 spots a run;
   - not where something held the lid up at the same height at two spots
     (within 15 mm): that is probably the table itself, standing higher than
     the run has it. Each such set-down presses 12-16 mm into what it met
     before it is told from the table, so no third spot is pressed on: the
     lid goes back on the box, and the run says so (`lid_spot_left`'s
     `same_height_as`). It is not let go there either: nothing the run has
     says where the table is but the plan that put it lower;
   - only while the run's time holds a further spot and the way back to the
     box after it. The engine stops a run at its budget with a Ctrl+C
     wherever it is (90 s; see "How long a line-up may take"), so it tells
     each run its budget as it starts it (`BOX_OPENING_RUN_BUDGET` in the
     run's environment: the seconds, the time they count from on the host's
     monotonic clock, and whether they start again at each go, as they do for
     a goal sent by hand). By the run records a spot takes 8 s to fly and 4 s
     to choose, and the way back to the box, the release and home 20 s, with
     3 s to spare. A further spot is started only with 35 s left. The clock
     is asked all through the choice, of every place looked at and before
     every reach check and every plan, and the moment less than 31 s are left
     the choosing stops and the lid goes back on the box, with the 23 s that
     takes still in hand. The carry chosen is sent only if the time left
     holds its own flight as the runner will fly it, 2.5 s back up, and those
     23 s. What the run says before each set-down is what is known then of
     what follows one that feels no table: another spot; the box, where that
     is certain (the last of the 6, a time that cannot be read); or, of a
     timed run, another spot only where its time then still holds one.
     A clean run reaches its first set-down with 51-63 s left, enough for three
     to five more spots; a run that had a full line-up inside its budget has
     none, and its first set-down that feels no table goes to the box. A run
     started by hand from the command line is handed no budget, is timed by
     nothing, and the 6 spots bound it. A budget that is there and cannot be
     read is not that: an engine is timing the run, the run does not know
     for how long, and no further spot is started. A budget that counts from
     later than now on the host's monotonic clock, or from more than its own
     seconds and an hour ago, is one that cannot be read: it was stamped on
     another clock, or is not this run's;
   - not after a set-down that could not be watched (its watch failed, which
     stops the descent where it failed, or where the slow stretch begins if
     it failed before that, and never the lift or the carry; or none of its
     stretch was fed): that says nothing of the spot, and the next would fly
     as blind.
1. **Release and home:** the fingers open, as the arm rises off the lid, only
   where the set-down felt the lid meet the table at about that height; then
   the arm goes home. The lid is never let go anywhere else.

The box is the last resort. If the run has to stop with the lid in the fingers
and the arm standing (no spot on the table the cameras measured that the hand
reaches and a carry plans to, the wider search and the wrist's fresh look
included; the run's own checks refuse the carry before it starts; or a
set-down felt no table and no other spot is tried: none is left, 6 were tried,
the run's time holds no other or cannot be read, something held the lid up at
the same height twice, or the set-down could not be watched),
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
It says `lid on the table` only of a set-down the run felt meet the table. A
set-down that felt none is a failed step with the lid kept, one line a spot as
the lid is taken on:

```text
 7 SET DOWN  FAIL  spot 1: no table felt, lid kept
 8 SET DOWN  FAIL  spot 2 (31 cm from the box): something held the lid up, lid kept
 9 SET DOWN  ok    spot 3 (25 cm from the box): let go where the lid met the table
```

and the result then reads `lid on the table (spot 3: no table felt at 2 before)`.
Where no other spot is tried, a `next` line says the lid is carried back to
the box and why (`no time left for another spot`, `as many spots as a run tries`, `a set-down that cannot be watched is tried nowhere else`, `held up at the same height twice: probably the table, higher than planned`, `the run's time cannot be read`), then `PUT BACK  ok`,
and the result's `lid: on the box`. Where the lid is put back from over a spot
it was left at, the `next` line says which it was: `no time left for another spot` or `no other spot found on the table`.
A run the engine's budget or a cancel cuts between two spots has the lid in
the air and no step of its own in flight; the view says so, and that is where
the run failed:

```text
 7 SET DOWN  FAIL  spot 1: no table felt, lid kept
 8 CARRY     FAIL  stopped: the run took too long, the lid on its way to spot 2
```

(or `8 SET DOWN  FAIL  stopped: the run took too long, the next spot for the lid being chosen`), with the result's `lid: in the hand`.
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
| `/arm/robot_description`             | `std_msgs/String`                              | yes      | The arm's URDF, rooted at `arm_mount`: the run's FK chain, and where the planner's tool frame is (`arm_tool_frame`)                                                                                                |
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

`lid_xyz` is where the lid was set down and let go, in `arm_mount` (m): the
spot that took it, not one it was taken to and left again. Its height is the
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
(`out of reach from here` where its wrist sees the container out of the arm's
reach, `no container in view` where no camera sees one: after a wrist look the
arm stands where it stood, and nothing says it is coming to rest),
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

| Code                  | Recoverable | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bad_params`          | no          | The goal had params; `open_box` takes none. Nothing ran                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `not_allowed`         | no          | `allow_execute` is off on this node. Nothing ran                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `bad_config`          | no          | A setting has a value it cannot have (`air_control` neither `impedance` nor `position`). Nothing ran                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `line_up_unavailable` | no          | What watches the container or judges its place could not start, or ended while lining up (the message carries what it said). The run was never told to go; a wrist that looked is back at rest. What judges the place (`reach_probe`) is started at the watch's first word; in a line-up that began because no camera saw the container, only at the first sighting, so such a line-up cannot end this way for a container that never comes into view                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `not_lined_up`        | yes         | The container was not where the arm opens it, or was never seen by a camera, in the time a line-up has (see "How long a line-up may take"), the UI went, a line-up in the middle of a run had no time left under the orchestrator's 120 s, or the run asked to be guided a fourth time in one goal (see "Lining up in the middle of a run"). Nothing was pressed; a wrist that looked is back at rest. The message says which it was (`no camera saw the container within ...` where none did), and carries the last instruction the user was given and, in the middle of a run, why the chair was guided; the next `open_box` then lines up first, unless no camera had seen the container during the line-up (it does all the same where the 120 s cut short, or left no time for, the guidance of a run whose own look had found no container), and not after a fourth request |
| `timeout`             | yes         | The run was still going after 90 s and was stopped its own way (one Ctrl+C); its stop's worst (a second Ctrl+C, then a kill) still ends before the 120 s of an orchestrator that counts them from the goal cancel; not recoverable with the lid still in the fingers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `not_ready`           | yes         | Refused before anything moved: the planner or the arm driver not up, the driver's link to the arm faulted or e-stopped, another program driving the arm, no TF to the wrist camera's mount, no scene camera calibration. No camera's stream is part of it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `not_found`           | yes         | A find, or a line-up's watch, with no container for a reason no moving of the chair cures: what it looks with delivers nothing, what it sees cannot be tied to the selected container, or (a find only) it sees two things that could be the container (see "A container no camera sees" for each, and for what the panel says). Nothing was pressed, the arm home (after a line-up: at rest, the wrist turned back). A container that is simply in no camera's view is not this: the chair is guided, and nothing but the line-up's own time ends that. A run by hand (`press_demo` without `--guide`) ends `not_found` for all of them, as it always has                                                                                                                                                                                                                        |
| `not_home`            | yes         | The arm could not get to its start pose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `contact_early`       | yes         | The press met something more than 45 mm above the button's top as the cameras saw it: stopped there, nothing pressed, the arm went home                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `contact_late`        | yes         | The press met nothing down to 25 mm below that top, or at all: the box is not where it was seen; the arm went home                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `watch_failed`        | no          | The press's contact watch failed: the press stopped there                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `box_moved`           | yes         | The press moved the container                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `not_popped`          | yes         | Pressed, but the knob did not come up (or the stroke pressed nothing); the arm went home                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `no_button`           | yes         | The container was found and the look from above it found no button on its lid, nor did a look again over a lid the wrist's own frame showed elsewhere (at most two; see "The look again"); nothing was pressed, the arm went home. (The run ends this with exit 2, as it ends a find with no container: until 2026-11-19 the engine read every exit 2 as `not_found`, and this code was never reported)                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `no_press`            | yes         | A stop before the press that names no other code (the container's config has no reach-out press, the fingers did not shut, the knob's height unread): nothing was pressed. A container no press plans to from where the chair stands is not this: the chair is guided (see "Lining up in the middle of a run")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `lid_put_back`        | yes         | No spot for the lid on the table the cameras measured (or no carry to one), or no table felt at the spots it was taken to and no other tried (none left, 6 tried, no time for another, held up at the same height twice, or a set-down that could not be watched): the lid was let go nowhere, it is back on the box and the arm home                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `lid_dropped`         | no          | The arm got home, but the lid slipped out of the fingers on the way                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `arm_holds`           | no          | A motion stopped part of the way: the arm holds there until cleared                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `arm_halted`          | no          | The driver halted a motion (the e-stop) or took control away: the run sent nothing after it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `stopped`             | no          | The run was ended from outside                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `failed`              | no          | Anything else, a traceback included; the message is the run's last STOP line (or line). Nothing says where the arm is                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

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
  "buttons": [{"button_id": "cancel", "label": "Cancel", "style": "warning"}]
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
nothing is sent. Every button carries its `style` (the bridge's
`USER_INPUT_JSON_SCHEMA.md`): Cancel `warning`, as on the orchestrator's own
progress panel, Open `normal`.

How the action ended is said on the panel for 5 s before the action ends
(`END_SHOW_S`), with no button, in one short line:

| The end                                                                                                                    | The panel says                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| the lid off and set down                                                                                                   | `Done. The lid is off, on the table`                                                                 |
| ended clean with the lid not off                                                                                           | `Finished. The lid was not taken off`                                                                |
| the person's Cancel (or `n`)                                                                                               | `Cancelled`                                                                                          |
| an error                                                                                                                   | what its code always means, then `You can try again` (recoverable) or `Ask someone to check the arm` |
| a line-up nobody answered at `Ready`                                                                                       | `No answer in time. Nothing was pressed. You can try again`                                          |
| a line-up in which no camera ever saw the container                                                                        | `The arm never saw the container. Nothing was pressed. You can try again`                            |
| a line-up whose panel the UI stopped showing                                                                               | `The screen lost the arm's panel. Nothing was pressed. You can try again`                            |
| something in view that cannot be tied to the selected container (`not_found`)                                              | `The arm cannot tell which container you chose. Choose it again. You can try again`                  |
| two things that could be the container, at the find (`not_found`)                                                          | `The arm cannot choose between two things that could be the container. You can try again`            |
| what watches delivers nothing: a camera, its pose or calibration, the scene segmenter, a watch fallen silent (`not_found`) | `The arm's vision is not working. Nothing was pressed. You can try again`                            |
| a fourth request to be guided in one goal (`not_lined_up`)                                                                 | `Ready three times, and the arm could not go on. Nothing was pressed. You can try again`             |
| any end with the lid still in the fingers                                                                                  | `Cancelled.` or `Stopped.` then `The lid is still in the fingers. Ask someone to check the arm`      |

An error's line (`END_SAYS` in `node.py`) says only what its code always means:
`timeout` is `Taking too long. The run was stopped`, never that the arm stands
still (a run that had to be killed may leave a motion in flight), and
`no_press` is `The button was not pressed`. An error's panel has the `style`
`error`. The whole reason stays in the action's result and the node's log.

The orchestrator shows no result of its own: its panel returns as the action
ends. The end is shown only for a goal whose line-ups wait for the person (see
"How long a line-up may take"), and not where nobody would see it: after the
orchestrator's own cancel (the screen is its own again), while the node is being
stopped, or once the bridge has not held the panel for 1.5 s (`END_UNSHOWN_S`).

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

| Panel message                                                     | Buttons           | When                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Looking for the container`                                       | Cancel            | Nothing has looked yet (the wrist is turning to its view, the scene camera's first look has not ended), or what watches has had no container for less than 1.5 s: one frame without it is not said as a container the arm cannot see                                                                                                                                                                                                                                                                                     |
| `The arm cannot see the container. Move the chair in front of it` | Cancel            | What watches delivers frames and has had nothing in them that could be the container for 1.5 s, or one that was seen has not been for 1.5 s. From the first panel where the run's own look had found none before it asked (and no yes was given yet in this goal). No direction, no distance: none is known. It never ends a line-up: only the line-up's time does                                                                                                                                                       |
| `The arm's vision is not working`                                 | Cancel, then none | What the line-up watches with has delivered nothing for 1.5 s: a camera with no frame, no pose or calibration for its frames, the scene segmenter down or not for this container; or the watch itself has said nothing for 5 s. No word of moving: no move cures it. Standing 10 s, its last 3 s with no button, the action fails `not_found`. Where the line-up watches with the wrist, the scene camera's sightings neither hide the wrist camera's fault nor start its 10 s again                                     |
| `The arm cannot tell which container you chose`                   | Cancel, then none | What watches HAS something that could be the container in view, and nothing ties it to the selected one: the wrist has a lid and the selected id is in none of the scene camera's detections (`wrist_alone` off). The same 10 s and end. (A lid that stands off the scene camera's line to a selected container the scene camera DOES hold is another thing on the table: the cannot-see sentence, which never ends.) Never of a selected id that is merely absent with nothing in view: that is the cannot-see sentence |
| `The arm sees two things that could be the container`             | Cancel            | What watches sees two lid-sized tops it cannot choose between (the second may be no container at all). Said while they stand, and it ends nothing: the line-up goes on reading sightings to its own time. Each of these sentences stands 3 s before another of them replaces it                                                                                                                                                                                                                                          |
| `Move closer, about 15 cm`                                        | Cancel            | Outside the boundary: closer, back, left or right, the larger miss                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `Hold still, checking`                                            | Cancel            | Inside, not yet still for 1 s, or not yet judged                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `Ready. Open the container?`                                      | Open, Cancel      | Inside, still, and the run plans for it there                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `Press open again when you are in place`                          | none              | The line-up's time passed (see "How long a line-up may take": 35 s from the goal's start, or `ui_line_up_s`; for a goal sent by hand, `Out of time. Start open_box again when you are in place` after 10 minutes): shown 3 s, then the action fails `not_lined_up`                                                                                                                                                                                                                                                       |
| `Container not seen in time. Press open again in front of it`     | none              | ... and no camera saw the container in all that time (with no UI, `Out of time: the container was never seen. Start open_box again in front of it`): the same end                                                                                                                                                                                                                                                                                                                                                        |
| `The arm is not at its rest pose: home it, then press open again` | none              | The arm stands off rest (more than a wrist a line-up left turned): shown 3 s, then the action fails `not_home`                                                                                                                                                                                                                                                                                                                                                                                                           |
| `The wrist is turning back to rest`                               | Cancel            | The line-up is over and the wrist that looked turns back                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `Ending`                                                          | Cancel            | The line-up is over and no wrist turned: the run is ending                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `The screen is not showing this any more`                         | none              | The UI stopped showing the line-up's panel for 20 s (see "How long a line-up may take"): the line-up ends `not_lined_up` and the wrist turns back. Nobody may see it: it is what the panel is sent while that ends                                                                                                                                                                                                                                                                                                       |

`progress.phase` is `looking` (the first five messages), `move`, `hold`,
`ready`, `go`, `timed_out` or `unable` (the last 3 s of a line-up that ends on
what no waiting cures),
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

The 35 s are what an orchestrator's 120 s leave where it counts them from the goal: the run's 90 s budget less
a long run after its go (55 s: about 37 s clean by the two runs of 2026-11-12
and the pacing since, more where a press or a grip is made twice), so a run
started on the last second still ends its own way and is not stopped part way.
Under that rule a run's own budget is shortened by what its line-up took. Either way the run is told the time
it has as it is started (`BOX_OPENING_RUN_BUDGET`: it takes the lid on to a further
spot only while that budget holds the spot and the way back to the box). After the go of a
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
  instructions). Where the run asks because its wrist sees the container out
  of reach, or because no camera sees one, the arm stands where it stood (at
  rest but for its wrist) and nothing says to wait for it: the panel says
  `Out of reach from here` or `No container in view`, the console
  `next  the run asks for the chair to be guided`.
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
- Three guidances a goal, no more (`GUIDES_MAX`). `Ready` is the watch's word
  (one frame's find, the boundary's proof); after the yes the run looks for
  itself, with its own views and the planner, and may find no container or no
  press where the watch had one. Nothing holds the two to agreeing, and each
  round has the line-up's whole time. A fourth request ends the action
  (`not_lined_up`, recoverable): the run is told to stop, its wrist turns back,
  the panel's end says
  `Ready three times, and the arm could not go on. Nothing was pressed. You can try again`,
  and the next `open_box` does not line up first (it would ask the same watch).
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
(`--guide`), a find that ends with **nothing that could be the container in any
camera's view** goes into the same guidance as a container out of reach: the
first look, the look after a line-up's yes, and the look after a guidance's yes
where the run's own look does not find what the watch had held.

Guidance is for what moving the chair can cure, and that is the one state it
is given for. A find with no container is one of four things (the `what` of
its `reach_refused` row). The other three end the run as they always have
(`NO BOX`, the arm home, `not_found`, recoverable: the arm is at rest and
nothing was pressed), each with its own sentence as the panel's end:

| `what`       | The find                                                                                                                                                                                                                                                                                               | What follows                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| `unseen`     | The camera a guidance would watch with delivered frames, and nothing in them could be the container: nothing lid-sized, a lid a view cuts; for the scene camera also a selected `instance_id` that is in none of its frames (the container may simply be out of its view)                              | The chair is guided (below)                                                                         |
| `not_chosen` | Something that could be the container IS in view, and nothing ties it to the selected one: the wrist has a lid (or two places, or a top out of reach) and the selected `instance_id` is in none of the scene camera's detections (typed by hand, or a track perception has dropped; `wrist_alone` off) | `NO BOX`; `The arm cannot tell which container you chose. Choose it again. You can try again`       |
| `several`    | Two things that could be the container, and nothing says which is meant: two places the finder cannot choose between, a second lid-sized top in the selected container's direction, two tops under `wrist_alone`                                                                                       | `NO BOX`; `The arm cannot choose between two things that could be the container. You can try again` |
| `fault`      | The camera a guidance would watch with delivered nothing: no wrist frame at a view the wrist stood at, or a scene camera's look the segmenter gave no frame of (streams or TF missing, the segmenter down, silent or showing another camera)                                                           | `NO BOX`; `The arm's vision is not working. Nothing was pressed. You can try again`                 |

Which camera that is: the wrist's, where `wrist_first` is on and the wrist
looked from a view. Its word then counts, however alive or dead the scene
segmenter is: a wrist camera with no frame is a fault whatever the scene camera
gave, and a wrist that looked and saw nothing is a container not in view
whatever the scene camera made of something further off. With `wrist_first`
off, or where the wrist could not look from any view (none to turn to, the
body model off), it is the scene camera's.

What none of this can tell: a camera that delivers frames and cannot find the
container in them (its mount off, holes in its depth where the lid is, a lid
the finder does not pass) from a container that is not in view. Both are
`unseen`: the person is told to move the chair, and no place of the chair
brings a `Ready`. Nothing in a frame says which of the two it is; such a
line-up lasts its whole time.

Two lid-sized tops side by side within the arm's reach are not parted by where
the chair stands, so they are not guided for: nothing box opening can say would
be both true and something to do with the chair, and the user is never told to
move the container or anything on the table (the scene is not changed, only
where the robot stands). The sentence says what the arm sees, and calls the
second a thing that could be the container, not a container: it may be a bowl.
A top out of the arm's reach is no rival of one within it (the finder's rule),
and lid-sized tops that all stand out of reach are guided to, as before
(`far`).

A selection the scene camera's detections do not hold (`wrist_alone` off):
nothing can tie any find to it, and the scene camera cannot tell a container
that is out of its view from an id no detection ever had (`was in none of its frames`
is all it knows). With `wrist_first` on, a run the engine started therefore
turns its wrist to look, as for every other find, and takes nothing it finds:
something that could be the container in front of the arm (a find, two places,
a lid-sized top out of reach) is `not_chosen` and ends the run with the person
told to choose again; nothing there is `unseen`, and guided. That guidance
reaches `Ready` only if the scene camera's detections come to hold the selected
id as the chair moves: the moment its watch has something that could be the
container and still nothing ties it to the selection, the panel says
`The arm cannot tell which container you chose`, and 10 s of that end the
line-up with the person told to choose again (the container is by then in
front of the arm, to be chosen). Until this, such a
run was guided with `The arm cannot see the container. Move the chair in front of it`
while its watch had a lid in every frame and took none of them: no place of the
chair could bring its `Ready`. A run by hand does not turn the wrist for it and
ends `NO BOX`, as it did.

With `wrist_first` off only the scene camera looks, and all it knows is that
the selected id is in none of its frames, as it is for a container that is
simply out of its view. That is a container the arm cannot see (`unseen`):
guided, with the cannot-see sentence, and never ended but by the line-up's own
time. What the scene camera cannot say is whether ANOTHER container stands in
its view under another id: the segmenter rung keeps only the selected
container's boxes of each frame. Where that is so the person is told to move
the chair for the whole wait, and the cure is to choose again; telling the two
apart needs the rung to count its classes' boxes whatever is selected
(`perception/segmenter_source.py`, not changed here).

A wrist find that the selection's tie does not take because it stands off the scene camera's line to the
selected container, which the scene camera holds, is another thing on the table (a bowl nearer the arm, a
second container): the chosen one is not in the wrist's view, and that is `unseen`, at the find and in a
line-up: it is guided for and never ends a guidance. For one round (2026-11-19) it was `not_chosen`, and a
healthy guided approach with a bowl in the wrist's view ended in 10 s with "Choose it again" (review of
2026-11-20). The price: with the scene camera's mount off, as on 2026-11-17, the selected container itself
reads as off the line, and the person is told the arm cannot see it for the whole wait.

The guidance itself:

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
  has looked, so not `Looking for the container` while the wrist turns). Not
  after a yes of the same goal: the watch had the container a moment ago where
  the person now stands, so there it looks first, and says so only once it has
  had none for 1.5 s. The user is never told to move the container or anything
  on the table.
- The moment a camera sees the container (the wrist's frames, or the scene
  camera's sightings while the wrist has none) the ordinary guidance takes
  over: which way and how far, `Hold still, checking`, `Ready. Open the container?`.
  Only what is known is said: a sighting the boundary cannot judge yet says
  `Hold still, checking`, and one with no known way in says
  `The arm cannot open it from here`. Lost again for 1.5 s, the user is told
  again that the arm cannot see it.
- What judges the boundary (`reach_probe`) is started at that first sighting,
  not when the line-up starts: it does nothing before a sighting (the lid's
  height names its grid), and one that could not start, or died, ended the
  action `line_up_unavailable`, not recoverable, for a container that might
  never come into view. The cost: its start-up comes after the first sighting,
  so the first direction comes later than in a line-up that began with the
  container seen (how much later is not measured).
- Where the watch comes to a state no moving of the chair cures, that is said
  as what it is, with **Cancel** and no word of moving, once it has stood 1.5 s
  (until then `Looking for the container`, or the sentence that stood before
  it; never, for these, that the arm cannot see the container):
  `The arm's vision is not working` (what watches delivers nothing),
  `The arm cannot tell which container you chose` (something in view that
  nothing ties to the selected container),
  `The arm sees two things that could be the container`.
- The first two END the line-up once they have stood 10 s without a break
  (`NO_CURE_S`; the last 3 with no button): the wrist turns back and the action
  fails `not_found`, recoverable, with that state's own end sentence (the table
  above). A sighting, or a word of another kind, starts the 10 s again: a
  segmenter message late for one look ends nothing. A line-up whose own time
  runs out first with such a sentence on the panel (under the 120 s it may have
  fewer than 10 s) ends the same way, that sentence its last word: not
  `Container not seen in time. Press open again in front of it`, which is not
  true of vision that is not working. This holds for every line-up, before a
  run, carried, or in the middle of one.
- Two tops in a frame end no line-up, as at 21aa7cc: their sentence stands
  while they do, the line-up goes on reading sightings (the container alone in
  view again is guided by at once), and its own time bounds it. (At the find, a
  run with two candidates ends `NO BOX`, as it always has.) Nor does a
  container that is merely not in view: that is what the guidance is for, and
  only the line-up's time ends it.
- Two states are the watch's own, whatever else is heard: said at once, and no
  sighting hides them or starts their 10 s again. Where the line-up watches
  with the wrist, the WRIST camera's fault (its `wrist camera: no new wrist frame`
  and the like): the scene camera may go on placing a lid in front of the arm,
  but on this robot only the wrist's frames bring a `Ready` there, so such a
  line-up said `Hold still, checking` and `Looking for the container` by turns
  for the whole wait. Only the wrist's own word that its camera delivers (a
  sighting of its own, a frame it judged and found nothing in) ends that
  state. And a watch that has fallen SILENT: 5 s with no word after it has said
  a camera's word (`SILENT_S`: its own constants allow 2.3 s between two words,
  a wrist frame waited for 1 s, the scene camera's look of 0.3 s and a `none` a
  second, besides the finder's time on a frame, which is not measured) is what
  watches not working, from then until it speaks again. Before its first such
  word a watch is not silent: its wrist is turning to the view.
- Which words are what watches delivering nothing: the wrist's watch's and
  `box_watch`'s own (`no new wrist frame`, `no camera pose for the frame (TF)`,
  `scene camera streams missing`, `the scene segmenter cannot be read`,
  `no frame newer than the look` and the like: `FAULT_MARKS` in `guide.py`,
  held to the modules that say them by a test). The run's own watch on the
  scene camera says the kind with each word, and knows one thing better: a look
  of its third of a second that the segmenter gave no frame newer than, the
  segmenter delivering by its own state (a message within 1 s, no more than 1 s
  behind the camera), has met a segmenter that is behind, not one that is down.
  That is `unseen`, whatever number of looks it lasts (the first frame comes
  0.04 to 0.24 s into a look by the logs the review read; the segmenter's own
  second is four times that).
- Everything else is the guidance's own: the same time (see "How long a
  line-up may take"), **Cancel** or `n` at any moment ending with the run's
  Ctrl+C, nothing moving toward the container without a yes, and at the yes a
  fresh find. A user who never brings the container into view gets the end of
  one who never brings it into reach: `not_lined_up`, recoverable, the wrist
  back at rest, with the words for what was missing (the panel's
  `Container not seen in time. Press open again in front of it`, the error's
  `no camera saw the container within ...`). After that end the next `open_box`
  does not line up first where the wait was the person's own (minutes, on the
  UI or by hand): a line-up that never saw the container has nothing to go on
  from.

Under an orchestrator whose 120 s count from the goal (see "How long a line-up
may take"), and with `wrist_first` on, this guidance is seldom given in the
goal that found no container: that find has spent the scene camera's whole
first look (15 s) and the wrist's views (up to about 15 s more, an estimate) by
then, which leaves under the 8 s a guidance needs of the 35. The action then
fails at once, `not_lined_up`, recoverable
(`no camera sees the container, and this run had no time left to guide the chair ...`).
On the UI the user sees `Looking for the container` while the run looks, then
(after an instant of `No container in view`) `The wrist is turning back to rest`,
and not yet that the chair is to move: the next `open_box` lines up first with
its whole 35 s, and that is where they are told that the arm cannot see the
container. With `wrist_first` off about 18 s of the 35 are left, and the user
is guided in the goal itself by the scene camera alone; if those seconds run
out with no camera having seen the container, the next `open_box` lines up
first too. The two ends are one: where the 35 s cut the guidance of a run
whose own look found no container, short or to nothing, "press open again"
goes on from a line-up with its whole time, not from another look that would
leave it as little. Once: a carried line-up that sees nothing carries nothing.
Where line-ups wait for the person (`ui_line_up_s`, or a goal sent by hand) the
user is guided in the goal itself, for up to 5 or 10 minutes each time, and a
line-up that ends there without the container seen carries nothing: nothing
was cut short. None of these times is changed by this.

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

| The goal                                                                                                     | A line-up before the run       | A line-up in the middle of it                                                                                                                                                                           | The run's own budget                              |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| with the orchestrator's token (from the UI, or the UI bridge down)                                           | ends 35 s after the goal began | ends 35 s after the goal began: what the run's look and proof left of them (run 2 of 2026-11-17 said "no press plans" 16 s in: about 19 s). With under 8 s left it is not given: `not_lined_up` at once | 90 s from the goal's start, the line-up inside it |
| no token (sent by hand, `open_box_terminal`)                                                                 | 10 minutes                     | 10 minutes, each time                                                                                                                                                                                   | 90 s from the last yes                            |
| with the orchestrator's token, its 120 s restarting at every progress item (`ui_line_up_s` 300, the default) | 5 minutes                      | 5 minutes, each time                                                                                                                                                                                    | 90 s from the last yes                            |

Whether a goal from the UI gets the first row or the third is the deployment's
word about its orchestrator. rammp-orchestrator before e17c625 (2026-11-16) ends
every action 120 s after its goal, whatever the engine reports: the first row.
From e17c625 on it counts its 120 s from the newest progress item, and only
120 s of silence cancel: the third row, which the deployment asks for with
`ui_line_up_s` above 0 (the package's fragment says `300.0`: the owner's "5
min", 2026-11-19). Box opening then sends its newest progress again every 30 s
with nothing new to say (`KEEP_ALIVE_S`), a line-up waits `ui_line_up_s` for the
person, and the run's 90 s start again at each yes, so that no wait can cut a
motion short.

The node cannot ask the orchestrator which of the two it is, so the third row
is taken only where nothing speaks against it. It is the first row:

- with `ui_line_up_s` `0.0`, the node's own default (an integer `0` is read the
  same);
- on a build whose goal does not bring the UI's token (it talks to the
  orchestrator before both changes);
- for a goal that has no UI as it comes (the bridge not there yet): it asks on
  the console of a node nobody sits at, and keeps its 35 s;
- from the moment a goal is cancelled 117 to 126 s after it came with its
  progress being sent and no cancel of the person's: the orchestrator is then
  taken not to restart its clock, the log says so, and it stays so until the
  node is started again. That first goal has still been cut; it does not happen
  twice.

A `ui_line_up_s` over 600 is taken as 600. Each goal's log line says which row
it has (`line_up_waits=300s`, or `35s-of-the-goal`). Proved against the real
orchestrator and UI bridge with the app stood in for (2026-11-19); not yet
with the app. Before the first use on a robot the deployment's word should be
proved there once: a line-up left unanswered for more than 120 s must not be
cancelled.

A line-up that waits minutes is not waited out on a screen that stopped showing
it. Under the third row a line-up ends `not_lined_up`, recoverable, its wrist
turned back as at any line-up's end, once the screen has not shown its panel
for 20 s (`UI_GONE_S`). "Shown" is: the bridge's server is there, the bridge
holds this goal's panel, and the bridge's own word on its app
(`/ui_bridge_node/ue_connection`, once a second) is no older than 5 s and says
connected. The 20 s count from when the node can tell, which is not when the
screen went: a bridge that stopped is noticed within about 5 s by its silence,
and one that restarted refuses the goal's token from its first answer. (Seen on
the rig: a bridge restarted in the middle of a line-up refuses the goal's token
from then on, and the app goes on showing the last panel with a Cancel that
reaches nobody.) The panel's words while it ends: `The screen is not showing this any more`.

Under the first row a line-up is given only where a run of 55 s still fits
after it inside the 90; the orchestrator cancels at 120 s whatever the engine
does. The 55 s are an estimate, not a measurement of a guided run (run 1 of
2026-11-17 took 33.5 s and the runs of 2026-11-13 about 40 s from the first row
to home): a guided run that takes longer still meets the run's own stop at
90 s, part way through a motion, as a run after a line-up before it can.
After an `open_box` that ended `not_lined_up` from a line-up in the middle of
its run, the next one within 5 minutes lines up before its first move whatever
`line_up` says, so that "press open again" goes on from the line-up with its
whole time and not from another look and proof. Once: a line-up that then never
sees the container does not ask for another. A line-up in which no camera saw
the container carries nothing, but for the one case in "A container no camera
sees": under the first row, the guidance of a run whose own look found no
container, cut short by the 35 s.

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
| `ui_line_up_s`        | `0.0`                  | How long (s) a line-up of a goal from the UI waits for the person. Above `0` it is the deployment's word that its orchestrator restarts its 120 s at every progress item (rammp-orchestrator e17c625 and later); the package's fragment says `300.0`. `0.0`: the orchestrator ends every action 120 s after its goal, and a line-up has what those leave (35 s of the goal). See "How long a line-up may take" for where it is `0.0` whatever is set                                                             |
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
| `rest_pose`           | see below              | A file in the arm driver's presets format whose `presets.home` is the robot's rest pose, where a run starts and ends; `""`: the built-in pose; `BOX_OPENING_REST_POSE` in the environment beats it                                                                                                                                                                                                                                                                                                               |
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

### The rest pose

A run starts and ends at the arm's rest pose, and everything it derives from it
follows: the wrist's front views (the rest pose with only the wrist turned), the
look the first-run observation flies, the side of the wrap the planner is handed
its joints on, the goal a home is planned to, and the key the line-up's reach
grids are kept under. The pose is the robot's own. `rest_pose` names a file in
the format the arm's driver reads its named poses from, by default
`/config/robot/arm_presets.yaml` (the robot's `config/robots/current`, which the
fragment mounts): ROS parameters, and box opening takes its `presets.home`, seven
angles in rad for `arm_joint_1`..`7`, the pose the driver's `go_to_preset` `home`
flies to. Box opening never calls that preset: it only reads the file.

**Which file.** `BOX_OPENING_REST_POSE` in a process's environment beats the
`rest_pose` parameter and every command line's `--rest-pose`, even empty. One
line in a deployment's box opening containers gives the node, its runs,
`home_arm`, `preflight` and the recorder the same file, whatever each was started
with. Without it: the parameter (the flag), and without that the default file.
`""` means the built-in pose (`constants.HOME`, flatbot's).

**What counts as the pose in the file.** Only the blocks the arm's driver would
take: `/**` and the driver's own node (`kinova_gen3_node`, under any namespace).
Another node's block is not the pose. Seven finite numbers, every one written as
a real (`0.0`, not `0`): ROS refuses a list that mixes whole numbers and reals,
so the driver does not start on such a file, and box opening refuses it too. Two
blocks that give different poses: neither is taken.

**Where there is none.**

- The default file not there, or with no `presets.home` the driver would take:
  the built-in pose. Where a robot's config is mounted (`/config/robot` exists)
  that is said as a WARNING with the numbers and the reason: the built-in pose
  was made for another robot.
- A file named (by the parameter, the flag or the environment, not the default)
  that is not there or gives no pose, and any `presets.home` that is malformed:
  the run stops before it moves. `open_box` fails `not_ready` before a line-up or
  a run is started.

**Every pose is judged before anything moves**, the built-in one included,
wherever a chair model is loaded (`body_layers`). Refused, with the link, the
box and the depth, or the joint and its room (`open_box`: `not_ready`), and never
flown to:

| Rule                                           | Margin   | Why that much                                                                                                                                           |
| ---------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Every limited joint inside its limits          | 0.23 rad | 0.18 is what the planner needs of a start beside a limit; 0.05 more is how far a hold may stand off its rest pose and still be where a run starts       |
| Every turning joint inside the planner's bound | none     | The planner's own bound on a joint that turns without end                                                                                               |
| Outside the chair and user's padding           | 10 mm    | Nearer than 10 mm, every way home begins with a step clear of the padding: such a pose would be stepped away from at the start of every run             |
| Clear of itself, by the planner's own rule     | 10 mm    | The planner plans nothing from a start that touches itself (the robot file's spheres, each link's `self_collision_buffer`, its `self_collision_ignore`) |

A pose that passes but stands inside the planner's `collision_activation_distance`
of the padding (30 mm) is taken with a WARNING: plans from it start under the
planner's push, and a hold that sags stands nearer still. With `body_layers ""`
there is no chair to judge against: a configured pose is judged on its joints
alone and the run's log says "NOT judged against any chair, nor against itself";
the built-in pose is then left as it always was. A configured pose with no arm
description to judge it by (`/arm/robot_description` not heard) is refused.

**One pose for every process.** Every command line says first which pose it
reads: `rest pose: [seven numbers], presets.home of <file> (digest <12 hex>)`, or
`the built-in one (<why>)`. The node hands every process it starts (the run, the
line-up's reach proof) the file it resolves itself and the digest of the seven
numbers it reads there (`--rest-pose`, `--rest-pose-digest`); a process that
reads other numbers stops before it moves ("the node that started it read
another"). Two box opening processes of one deployment cannot rest at two poses
unseen.

**An arm parked inside the padding** (RAMMP 1.5: the team's `presets.home` is
58 mm inside the padded armrest, and box opening rests at a pose of its own in a
private file) starts every run with a step out of it: slow, in the driver's
softest joint impedance, never deeper, the tool held within 5 mm of its height.
A run and `home_arm --execute` fly that step as soon as it is planned, asked
40 mm outside the padding, then wait until the arm stands still and read its
joints. It must stand 30 mm outside the padding by those measured joints; short
of that it steps again from where it stands (3 steps at most), and the way to
the rest pose is planned from the measured joints, never from the step's
commanded end. Where the arm cannot be brought clear by 30 mm the run stops
there: `STOP: the step out of the chair and user's padding did not bring the arm clear ... the arm holds where it stands`, `open_box` fails `arm_holds`. A preview
(`home_arm` without `--execute`) shows the step and a home planned from its
commanded end. So does the home at a run's end, which is planned from where
the run leaves the arm (in front of the chair, over the table), often while the
arm still flies: where that start is inside or within 10 mm of the padding its
step out still ends 10 mm outside by command, and home is planned from that
end.

**`preflight`** has a row for the rest pose (FAIL where a run would refuse it,
WARN for each warning) and two checks of what the rest of the stack holds, with
the chair model on:

- the TF tree's fixed mounts against the mounted robot config's own
  (`transforms/cameras.yaml`, `transforms/world.yaml`): FAIL where a mount in TF
  is more than 2 mm or 0.5 deg from the file's. It proves that the stack
  publishing TF was started on a config with these mounts. It does not prove a
  mount is right, and two robots whose files hold the same mounts are not told
  apart by it;
- the planner's canary, plans only, nothing flown: a plan from the rest pose
  must be taken, and a plan from a start the mounted chair model puts 20 mm or
  more inside one of the chair's boxes (drawn from the model on each run, never
  a constant) must be refused as a start in the planner's world. A planner that
  plans from the canary does not hold this robot's chair, whatever the path of
  its config says: FAIL. It is a one-way test: another chair with a box at the
  same place refuses the canary too. Measured offline (2026-11-19): gen_1_5's
  canaries all stand inside flatbot's padded armrest as well, and flatbot's
  inside gen_1_5's, so this check does not tell those two chairs apart. Of the
  two, it catches flatbot's files on a planner holding gen_1_5's chair (the rest
  pose is refused there); gen_1_5's files on a planner holding flatbot's chair
  pass it. The arm's mount in `transforms/world.yaml` differs between them by
  87 mm, which the first check sees where TF comes from the same stack.

Both were tested with stand-ins only; neither has been run against the robot's
TF tree or planner.

**What changed for a robot with nothing configured** (no `rest_pose`, no
environment), against the versions that took the built-in pose everywhere:

| Mounted at `/config/robot`                                    | Before            | Now                                                                                                                                                            |
| ------------------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| nothing (a bench, the tests)                                  | the built-in pose | the same; not judged without a chair model                                                                                                                     |
| a config with no `arm_presets.yaml`, or none the driver takes | the built-in pose | the built-in pose, judged against the chair model, with a WARNING; refused where it stands inside that chair's padding                                         |
| flatbot's (its `presets.home` is the built-in pose's numbers) | the built-in pose | the same numbers, read from the file and judged against flatbot's chair: taken, with the warning (11 mm outside its padding, inside the planner's 30 mm)       |
| dojo's (its `presets.home` is the arm's factory Home)         | the built-in pose | the factory Home, judged on its joints alone with `body_layers ""` (dojo's planner config names no chair layers); `rest_pose: ""` gives the built-in pose back |
| gen_1_5's (its `presets.home` is inside the chair's padding)  | the built-in pose | refused, `not_ready`, until `rest_pose` names a pose clear of the chair; the built-in pose (`""`) is refused there too                                         |

`home_arm` with a pose from a file and no arm description heard now refuses
("cannot be judged") where it used to fly to the built-in pose: a pose that
cannot be judged is not flown to. `--rest-pose ""` is the old behaviour.

The same mounted robot config and the arm's own description say where the hand is,
and a run says it in its first line (`... hand: ...`):

- box opening's own tool frame is 0.120 m out of the flange on every description of
  the arm: the frame its heights were tuned in by flight, so the flange stands for
  every target where it stood when it flew. The planner plans the description's own
  tool frame (its config's `ee_link`, `arm_tool_frame`, read from
  `/arm/robot_description` each start through the fixed joints that hang it on the
  flange): on a description with Kinova's sensor puck (`mount.yaml`'s
  `wrist: sensor_puck`) that is 14 mm further along the hand, and a pose handed to
  the planner is moved by that step. A tool frame the description does not fix to
  the flange stops the run before anything moves, with the body check on or off;
- the wrist camera's lens and axes are the robot's calibration, `wrist_camera` in
  `/config/robot/transforms/cameras.yaml`, its parent link placed from the flange by
  the description (`BOX_OPENING_ROBOT_CAMERAS` names another file; empty, none).
  Where no such file is mounted, the figure in `config/camera_oak_wrist.yaml` stands
  in. That file also keeps the two mounts presses have been flown on (a fit from a
  2026-10-22 run, and the mount of the 2026-11-12..17 flights): a run says how far
  its mount's aim, 0.30 m out, lands from theirs, warns where that is more than
  10 mm from both, or where the mounted robot config is not the description's
  robot, and goes on. A guarded run, which centres its press on the mount, does not
  start on such a mount;
- how far out of the flange the shut fingertips end is one number,
  `constants.PADS_FROM_FLANGE_M` (robot-body fact, 0.151 m, known to about
  150-161 mm until someone holds a ruler from the flange's face to the pads'
  ends). The press's contact band, the fingertips' spheres in
  `config/robot_body.yaml` and the depth taken for the hand itself follow from it.

`wrist_survey` is off by default: a run whose scene camera finds no container
has found none (what follows, guidance or `not_found`, is in "A container no
camera sees"; a run by hand stops with `not_found`). On, such a run surveys with the
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
camera said (what follows, guidance or `not_found`, is in "A container no
camera sees"; a run by hand stops with `not_found`). It adds about 3 s where the first view shows the container and
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
a neighbour beside it.

**One container in each camera's view is the same container.** The 75 mm takes
the scene camera's direction for right. On RAMMP 1.5 (2026-11-20, five sightings
of one container, the chair at four places) it was not: the wrist's lid stood
148-206 mm to the side of the scene camera's line and 47-60 mm nearer along it, a
steady 13-15 degrees about the scene lens, every find was refused for the
selected container, and every run reached for the scene camera's place, 16-19 cm
from the lid. Where the scene camera holds one container of the selected one's
kind and no other (every frame of its segmenter's last 2 s, three or more of
them, held at most one; a lid boxed on its holder counts with it), there is no
neighbour to tell it from, and what the wrist finds is the selected container
within 20 degrees about the scene lens of that line, and within 0.20 m of the
range the scene camera placed it at. The run's line then gives the millimetres
and the degrees and says that the cameras disagree, its record has them
(`wrist_first`, `wide`, `across_mm`, `about_lens_deg`, `nearer_mm`, both
cameras' places and the scene lens), and the console says `scene camera puts it N mm away; using the wrist's`. A second lid-sized top in that wider cone and
neither is taken, with two exceptions that are the finder's own rules: a top the
arm cannot reach is no rival of a find it can, and of two out of reach the one a
lid's width or more farther from the arm is not the one being brought to it. A
top within the 75 mm of the line is a rival of every find, as before. With two
or more containers of the kind in the scene camera's view the 75 mm stands
alone. Not flown.

A second lid-sized top in the same direction, a find that
is not tied, or no find: the run goes on with the scene camera's own fix of the
selected container where that is a careful one, and has found no container
where it is only a rough one or there is none, saying what each camera said.

### The look again

The look goes where the find put the container, and the aim takes a button only
within 150 mm of there. On RAMMP 1.5 (2026-11-20 13:40-13:44) the scene
camera's place was 16-19 cm from the lid: four looks stood short and left of
the container with the lid whole in the corner of the wrist's frame, and four
runs ended `no_button`. Replayed on those four frames, the wrist's finder has
the container in each, within 12 mm of where the wrist's look from rest had put
it.

Where the aim finds no button (a compliant run, which every run is by default):

- The wrist's finder (the survey's, by the lid's geometry; no button is asked
  of this frame) is asked of a still frame from the look pose, kept as a
  capture (`captures/wrist-survey-*`). Of several lid-sized tops it takes the
  one the selection's tie takes alone.
- What it finds is taken by the rule the wrist's look before the reach goes by:
  tied to the selected container (the 75 mm, or the one-container rule above),
  or with `wrist_alone` the one container in view, or with no selection the
  container it finds. A selection nothing ties a find to takes none.
- A lid within half a lid of where the look was aimed is not looked at again:
  the look stood over it.
- The lid it takes becomes the run's box, as a box the wrist found before the
  reach is: the planner's world has the box there, its own counter is the
  table (the counter's edges are read again on the wrist's points), and the
  scene camera, which has the box elsewhere, is not asked about it after the
  press. Its look is planned from where the arm stands, the press proven from
  it first, and flown; then the aim is asked with its prior there.
- At most two looks again in one reach (`RELOOKS_MAX`). One is flown only where
  the run's time still holds it and the rest of a run (`RELOOK_S` 8 s and
  `RELOOK_REST_S` 40 s left of the engine's budget; a run by hand has no
  budget).
- Where the lid's look does not plan, the arm goes home and the run says there
  is no press from where the chair stands for the lid the wrist sees: guided as
  any such run is ("Lining up in the middle of a run"), a stop in a run by
  hand. A look again that stops part way holds the arm (`arm_holds`).

The run says `LOOK AGAIN n of 2: no button where the scene camera had the box; the wrist sees the lid at [...], N mm from there`, its record has a `look_again`
row a look (`n`, `lid_xyz`, `from_aimed_mm`, `planned`, `tilt_deg`), and the
console shows the first look as `LOOK ??  no button where the scene camera had the box` and the next as a step of its own. A run that ends with no button
after them says how many it flew and why no further one. Not flown.

**A selected id the scene camera no longer has.** Perception gives a container
it loses and finds again a new id, and a goal may come with the old one
(2026-11-20 13:42: `food_container_2` was in none of the detections, and the goal
ended in 10 s with `The arm cannot tell which container you chose`, one
container on the table). Before the run's first fix of the container, where the
selected id is in none of the segmenter's frames and they hold one container of
its kind under one other id, for five frames over at least a second, that one is
read as the selected container from then on. The scene camera's line says it
once with both ids, the console notes it (`<id> is no longer in the scene camera's list; the one container in view (<id>) is taken for it`), and the
look's record has the id it read (`segmenter` `adopted`). With two or more containers of the kind and the selected id gone,
nothing says which was selected, and everything is as below. The look's record
also keeps the most containers of the kind any one of its frames held
(`containers`). Not flown.

Where the selected container was in none of the scene camera's frames nothing
would tie a wrist find to the selection: a run by hand does not turn the wrist,
and a run the engine started turns it only to tell whether anything that could
be the container stands in front of the arm, and takes nothing it finds. What
follows each of these (the chair guided where nothing that could be the
selected container is in view; `not_found` where a second top stands in its
direction, or where a lid that is in view cannot be tied to it) is in "A
container no camera sees"; a run by hand stops with `not_found`. A guidance's watch says a
wrist find only where the run would take it for the selected container. The
instance the world state is written for is the selected one in every case.

`wrist_alone` (off by default) is for that last case: a selection that is no
detection of the scene camera's at all, as when the UI's objects come from a
list typed by hand (the orchestrator's `interactive_perception_node`) and the
selected id is in none of the segmenter's frames. With it on, the wrist looks
from its own views, tied to nothing, and what it finds is taken only where it is
the one container its frames hold; a second lid-sized top anywhere in them (a
place that passes, or a top a view cut), or two places the finder cannot choose
between, and the run has found no container, saying so (`not_found`, the
panel's end `The arm cannot choose between two things that could be the container`:
no place of the chair parts two tops that stand side by side). The look stops at the
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
a selection does not survey; it has found no container, saying so (what
follows is in "A container no camera sees").
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
