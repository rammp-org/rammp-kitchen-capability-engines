# December demo — running the kitchen sequence (no UI)

How to start every capability on `integration/dec-demo` and run the demo's ten steps on flatbot, one goal at a
time, without the orchestrator or the UI. Each capability's own docs have the detail (options, failure codes, what
each check means); this page is the order, the commands and the hand-offs between steps.

| #   | Step                             | Capability        | Action(s)                        | Hand at the start          | Ends with                                      |
| --- | -------------------------------- | ----------------- | -------------------------------- | -------------------------- | ---------------------------------------------- |
| 1   | Open fridge                      | `fridge`          | `go_to_start`, `open`            | empty                      | door at 90°, hand off the door                 |
| 2   | Grab tupperware from fridge      | `microwave_place` | `take_out` (fridge viewing pose) | empty, open                | holding it, 30 cm back from where it stood     |
| 3   | Close fridge                     | `fridge`          | `close`                          | holding the container      | door shut, still holding it                    |
| 4   | Place tupperware on table        | `place_container` | `mark` (earlier), `place`        | holding the container      | container on the table, hand back at its start |
| 5   | Open tupperware                  | `box_opening`     | `open_box`                       | empty                      | lid beside the container, arm at HOME          |
| 6   | Open microwave                   | `microwave`       | `open` (`push_open`)             | empty                      | door open, arm at the microwave view pose      |
| 7   | Grab tupperware                  | `microwave_place` | `take_out` (table viewing pose)  | empty, open                | holding it, 30 cm back from where it stood     |
| 8   | Place tupperware in microwave    | `microwave_place` | `place`                          | holding the container      | container on the turntable, hand out           |
| 9   | Close microwave                  | `microwave`       | `close`                          | empty                      | door shut                                      |
| 10  | Press the microwave start button | `button_press`    | `press_button`                   | empty (it closes the hand) | button pressed, back at its start              |

`home_arm` (`go_home`) is the way back to a known pose between steps or after a stop.

**Every command that can move the arm is marked ARM MOVES.** Run them only attended, with someone on the e-stop
(T4). Every step has a dry run first; do it.

## Terminals

All on the robot's computer, from this repo's root unless it says otherwise:

- **T1** sheppy (arm driver, planner, cameras, perception)
- **N1–N7** one per capability node (tmux windows are easiest): fridge, microwave_place, place_container,
  box_opening, microwave, button_press, home_arm. They can all run at once; each waits idle for goals.
- **T3** goals and one-off ROS commands
- **T4** the e-stop, typed out and ready

## ROS environment

Nothing needs sourcing on the host. Every `ros2` command below runs inside one of this repo's images, whose
entrypoint sources `/opt/ros/humble` and the image's workspaces. Paste this block into **every** terminal (or save
it as `~/kitchen_env.sh` and `source ~/kitchen_env.sh`):

```bash
export ROS_DOMAIN_ID=0
export KCE=ghcr.io/rammp-org/rammp-kitchen-capability-engines
# Cyclone, as the images ship it, without the 10-participants-per-host cap (the Makefile's DDS_ENV)
export DDS='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>'
# The fridge node only: Cyclone on eno1, the sensor network the MIB (the base) is on, for /base/odom
export DDS_ENO1='<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery></Domain></CycloneDDS>'
# rammp-deployments' december_2026 directory on this computer (home_arm mounts its robot config)
export DEPLOY=~/rammp-deployments/december_2026

# goal <capability> <action> ['<params json>']: one ExecuteAction goal, feedback printed, as the orchestrator sends it
goal() {
  local cap=$1 action=$2 params=${3:-}
  [ -n "$params" ] || params='{}'
  docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" "$KCE/$cap:dev" \
    ros2 action send_goal --feedback "/$cap/execute_action" rammp_orchestrator_interfaces/action/ExecuteAction \
    "{engine_id: $cap, action_id: $action, params_json: '$params'}"
}
# ros <command ...>: any other ros2 command, in the fridge image (it has the arm, base, cuRobo and orchestrator messages)
ros() { docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" "$KCE/fridge:dev" "$@"; }
```

`goal` takes the params as plain JSON in single quotes, so no `\"` escaping:
`goal fridge open '{"execute": true, "speed": 0.2}'` sends the same goal as the fridge doc's long form.
Ctrl-C on a `goal` cancels it (the capability stops the arm where it is).

Only inside the arm's own container (the e-stop, below) do you source by hand:
`source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash`. If you'd rather run `ros2` natively, the
host needs ROS 2 Humble and a workspace with `rammp-interfaces-ros2` and `rammp_curobo_interfaces` built, and then
`source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI="$DDS"`.

## Set up

### Images

Build every image from this branch (needs internet; ~minutes each, a code-only change rebuilds fast). Rebuild an
image after any change to its capability or to `capability_engine/`:

```bash
for cap in fridge microwave_place place_container box_opening microwave button_press home_arm; do
  docker build --network host -f capabilities/$cap/Dockerfile -t $KCE/$cap:dev . || break
done
```

The fridge's offline rebuild (no internet, source-only change) is in
[fridge/docs/TESTING.md](../capabilities/fridge/docs/TESTING.md#building-offline-no-internet); the same recipe works
for the others with their own `COPY` lines (`shared/door_common` only for fridge and microwave).

### T1: sheppy

```bash
cd ~/rammp-deployments && git checkout dev && git pull
uv run scripts/set_robot.py december_2026/config/robots/flatbot
sheppy up flatbot --manifest december_2026/sheppy-manifest.yaml && sheppy status
```

Check `sheppy status` shows the arm, `curobo_planner`, both cameras, `static_tf`, the scene segmenter (`rfdetr`) and
`sam3` running (`sheppy logs <id>` if not). SAM 3's weights are mounted from `/home/abra/sam3-test/models/sam3`.
**No orchestrator, teleop or `rammp_adl_runtime` alongside**: `home_arm` and `box_opening` refuse to run next to them.

| Perception used                                   | By                                                                     |
| ------------------------------------------------- | ---------------------------------------------------------------------- |
| SAM 3 (`/perception/sam3/detect`)                 | fridge (handle), microwave_place (opening; the container for take_out) |
| Scene segmenter (`/perception/object_detections`) | box_opening (food container), microwave (microwave box), button_press  |
| Wrist camera (`/wrist_camera/...`)                | fridge, microwave, microwave_place take_out, box_opening, button_press |
| Scene camera (`/scene_camera/...`)                | microwave, microwave_place place, box_opening, button_press            |
| none                                              | place_container, home_arm                                              |

### T3: clear the arm's faults

Before any runs, and after any fault or e-stop:

```bash
ros ros2 service call /clear_faults std_srvs/srv/Trigger
```

### N1–N7: the capability nodes

Each starts standalone (`use_orchestrator:=false`). Start them all with `allow_execute:=false` for a first pass of
dry runs, then restart the ones you are about to run with `allow_execute:=true` (`home_arm` has no such switch:
its goals' `dry_run` is the only guard). Ctrl-C in a node's terminal stops it and any motion it is running.

The planner reads the worlds the fridge, microwave_place and button_press push from the `curobo_kernel_cache` volume
it already mounts at `/root/.cache`, so those nodes mount it too.

```bash
# N1 fridge (mount_tilt_deg 30 on flatbot; grasp side is the default, front the straight-in grasp)
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS_ENO1" \
  -v /tmp/fridge_data:/fridge_data -v curobo_kernel_cache:/planner_cache \
  $KCE/fridge:dev \
  ros2 run fridge fridge_node --ros-args -p use_orchestrator:=false -p allow_execute:=true \
  -p mount_tilt_deg:=30 -p grasp:=side \
  -p planner_world_dir:=/planner_cache/rammp_planner_worlds \
  -p planner_world_dir_in_planner:=/root/.cache/rammp_planner_worlds

# N2 microwave_place (both grabs and the microwave place). take_home_joints is the pose take_out looks from:
# set it per grab (step 2, step 7), which means restarting this node. Its parameters are read at startup.
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" \
  -v /tmp/microwave_place_data:/config/microwave_place -v curobo_kernel_cache:/root/.cache \
  $KCE/microwave_place:dev \
  ros2 run microwave_place microwave_place_node --ros-args -p use_orchestrator:=false -p allow_execute:=true \
  -p worlds_dir:=/root/.cache/rammp_planner_worlds -p arm.service_call_s:=30.0 \
  -p "take_home_joints:=[J1, J2, J3, J4, J5, J6, J7]"

# N3 place_container
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" \
  -v /tmp/place_container_data:/config/place_container \
  $KCE/place_container:dev \
  ros2 run place_container place_container_node --ros-args -p use_orchestrator:=false -p allow_execute:=true

# N4 box_opening (no ros2 run: as PID 1 it would not pass Ctrl-C on to the run). Its state dir keeps the
# finger calibration (gripper.json): keep the same host directory between sessions.
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" \
  -v $HOME/.ros/box_opening:/state \
  $KCE/box_opening:dev \
  /module_ws/install/box_opening/lib/box_opening/box_opening_node --ros-args \
  -p use_orchestrator:=false -p allow_execute:=true -p state_dir:=/state \
  -p container:=ankou_pink.yaml -p world:=world_flatbot.yaml

# N5 microwave
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" \
  -v /tmp/microwave_data:/microwave_data \
  $KCE/microwave:dev \
  ros2 run microwave microwave_node --ros-args -p use_orchestrator:=false -p allow_execute:=true

# N6 button_press (its code defaults are not flatbot's: these -p are the fragment's flatbot values)
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" \
  -v /tmp/button_press_data:/button_press_data -v curobo_kernel_cache:/planner_cache \
  $KCE/button_press:dev \
  ros2 run button_press button_press_node --ros-args -p use_orchestrator:=false -p allow_execute:=false \
  -p "mount_rpy_deg:=[0.0, 30.0, 0.0]" -p "camera_frame:=''" \
  -p depth_topic:=depth/image_rect_raw -p scene_depth_topic:=depth/image_rect_raw \
  -p log_dir:=/button_press_data -p arm.go_to_track_m:=0.05 \
  -p planner_world_dir:=/planner_cache/rammp_planner_worlds \
  -p planner_world_dir_in_planner:=/root/.cache/rammp_planner_worlds

# N7 home_arm (standalone only; it reads the chair and user boxes from the robot's planner config)
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" \
  -v $DEPLOY/config/robots/current:/config/robot:ro \
  $KCE/home_arm:dev \
  ros2 run home_arm home_arm_node --ros-args -p use_orchestrator:=false
```

Each node logs that it is up; `ros ros2 action list | grep execute_action` should list all seven.

Where these differ from the capabilities' own fragments, and why:

- **microwave_place `worlds_dir`**: its fragment writes the place's cuRobo world to
  `/home/snorlax/.ros/rammp_box_opening/worlds`, which the dev manifest's planner doesn't mount. Mounting the
  planner's own volume at the same path (`/root/.cache`) in both containers makes the file the planner loads the
  one the node wrote. **Not yet tried on the robot**: if the place's dry run fails at `set_world`, this is why.
- **button_press**: `allow_execute:=false` until step 10's first dry run passes (it has no per-goal `execute`), and
  the planner-world mount as the fridge's, for the same reason as above.

## How to stop the robot

| How                                      | Effect                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------ |
| **Software e-stop in T4** (first choice) | The arm **holds where it is**, powered. Motion is refused until cleared. |
| **Ctrl-C on the `goal`, or in the node** | Cancels the action; the arm stops and holds, powered.                    |
| Physical e-stop button                   | Last resort only: it cuts power and the arm drops.                       |

```bash
# T4 stop (repeats at 10 Hz; Ctrl-C once the arm has halted)
docker exec sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  ros2 topic pub -r 10 /estop rammp_common_interfaces/msg/EStop "{state: 1, source: operator, reason: stop}"'
# release, only right before you continue (if it does not release: `sheppy restart arm`); then clear the faults
docker exec sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  ros2 topic pub -t 10 -r 10 /estop rammp_common_interfaces/msg/EStop "{state: 0, source: operator, reason: clear, clear_target: operator}"'
```

The fridge doc has the e-stop's caveats (the new `EStop` message, test it at rest first).

## Utilities (T3)

```bash
# where the arm is: joint names and positions (rad); the arm's J1..J7 are arm_joint_1..7
ros ros2 topic echo --once /joint_states --field name
ros ros2 topic echo --once /joint_states --field position

# put the planner's own world back (microwave_place's place leaves its microwave box loaded; the fridge and
# button_press put theirs back themselves). Nothing moves.
ros ros2 service call /rammp_curobo/set_world rammp_curobo_interfaces/srv/SetWorld "{world: world.yaml}"

# home_arm: plan the way home (nothing sent), then go (ARM MOVES)
goal home_arm go_home '{"dry_run": true}'
goal home_arm go_home
```

A capability's `control` service pauses (`0`), resumes (`1`) or cancels (`2`) its running action, e.g.
`ros ros2 service call /microwave/control rammp_orchestrator_interfaces/srv/Control '{command: 2}'`.

## Before the run

1. **Record the two viewing poses for the grabs** (steps 2 and 7). `take_out` looks for the container with the wrist
   camera from `take_home_joints`, moves there itself, and then reaches straight in, level, 30 cm. Its default is the
   microwave's viewing pose (captured 2026-11-11). For the fridge and the table, teleop the arm (empty hand) to a pose
   where the wrist camera sees the whole container 0.4–0.5 m away, with a straight, level 30 cm path to it, and save
   the joints (`ros ros2 topic echo --once /joint_states --field position`, the seven arm joints, rad).

1. **Mark the table spot** (step 4) with the gripper closed and empty, because `mark` takes the fingertips' position
   (and it is the table's height). Teleop the closed fingertips onto the spot, then:

   ```bash
   goal place_container mark          # no motion: prints {"spot": [x, y], "table_z": z}
   ```

   Write the three numbers down; step 4 passes them explicitly (the saved spot expires after 30 min).

1. **Put the closed tupperware in the fridge** where the fridge viewing pose sees it, and the microwave door closed.

## The run

### 1. Open fridge (`fridge`)

Full detail: [fridge/docs/TESTING.md](../capabilities/fridge/docs/TESTING.md), "Without drive base", Open.

```bash
goal fridge go_to_start '{"execute": false}'                  # plan the start pose
goal fridge go_to_start '{"execute": true, "speed": 0.2}'     # ARM MOVES
```

Place the wheelchair so the camera looks square-on at the closed door, ~50–60 cm from it (side grasp: aim for the
logged `handle (mean)` x 0.70–0.85, y −0.40 to −0.20).

```bash
goal fridge open '{"execute": false}'                         # dry run: plans the whole open
goal fridge open '{"execute": true, "speed": 0.2}'            # ARM MOVES
```

It pulls to 60°, lets go, and pushes the door on to 90°. If it stops asking for a wheelchair move, make it (straight,
without turning) and push on:

```bash
goal fridge open '{"execute": true, "chair_moved": true, "speed": 0.2}'   # ARM MOVES
```

Or let the base make the move itself: add `"drive_base": true, "base_speed": 0.03` to the open (see the fridge doc's
"Arm + Drive Base" first, including its checks). For no wheelchair move at all: `"base_moves": false`.

### 2. Grab tupperware from the fridge (`microwave_place`, `take_out`)

Restart N2 with the **fridge** viewing pose: `-p "take_home_joints:=[...]"` (Before the run, step 1).

```bash
goal microwave_place take_out '{"execute": true}'             # ARM MOVES: to the viewing pose first
```

A dry run (`{"execute": false}`) refuses unless the arm is already at the viewing pose, so: execute once to get
there (it continues to detect and grasp: watch it), or teleop there first and dry-run from there. It opens the hand,
plans the approach with cuRobo, reaches in, closes, lifts 3 cm and backs out level to 30 cm short of the grasp.
Ends `Done: holding the container ...`.

**Not yet run anywhere but the microwave.** cuRobo's approach world has only the chair, the user and the floor:
**not the fridge, its shelves or its open door**, and the reach-in, lift and back-out are not collision-checked. Pick
a viewing pose with a clear straight path, and run it slowly the first time (the approach and exit run at
`take_approach_speed_scale` / `take_exit_speed_scale`, 0.3).

### 3. Close fridge (`fridge`, holding the container)

The close carries the container when the gripper is closed at its start. Hand ≥5 cm from the door and clear of the
chair; the wheelchair where it was for the open.

```bash
goal fridge close '{"execute": false, "speed": 0.7}'                        # plan: says which wheelchair move
goal fridge close '{"execute": true, "chair_moved": true, "speed": 0.7}'    # ARM MOVES, after the move
```

Keep `speed` ≥ 0.7 or the swing doesn't carry the door shut. With the base driving the move: replace
`"chair_moved": true` with `"drive_base": true, "base_speed": 0.03`. Move the wheelchair forward again after.

### 4. Place tupperware on the table (`place_container`)

With the x, y, table_z from `mark` (Before the run, step 2):

```bash
goal place_container place '{"execute": false, "x": X, "y": Y, "table_z": Z}'   # dry run: "all checks passed"
goal place_container place '{"execute": true, "x": X, "y": Y, "table_z": Z}'    # ARM MOVES
```

It carries the container over the spot, lowers it in impedance until the table stops it, opens the hand and goes
back to where it started (saved as its `start.json`). Ends `Done: container placed, hand clear. Back at the start position`. The table isn't in the planner's world: keep the path over the table clear.

### 5. Open the tupperware (`box_opening`)

Hand empty, container closed and upright on a level table, in the scene camera's view, with room beside it for the
lid. The run first goes to its HOME, and refuses (`not_home`) from more than 1.2 rad away, so go home first:

```bash
goal home_arm go_home '{"dry_run": true}'
goal home_arm go_home                          # ARM MOVES
goal box_opening open_box                      # ARM MOVES: no dry run, no params
```

It presses the lid's button, lifts the lid off and sets it down beside the container, then goes back to HOME. Result:
`"lid_off": true`. A no-motion check of its prerequisites (Cyclone, driver, planner, cameras, segmenter):

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" -v $HOME/.ros/box_opening:/state \
  $KCE/box_opening:dev ros2 run box_opening preflight --bench-world world_flatbot.yaml
```

The container config is `ankou_pink.yaml` (the pink canister), the only one the scene segmenter can find. If the demo
tupperware is another one, step 5 needs its own config first.

### 6. Open microwave (`microwave`)

```bash
goal microwave open '{"view_only": true, "execute": true}'    # ARM MOVES: to the fixed view pose only
goal microwave open '{"execute": false, "push_open": true}'   # dry run, from the view pose
goal microwave open '{"execute": true, "push_open": true}'    # ARM MOVES
```

`push_open` (off by default, on in the UI's Open Microwave Door) lets go after the swing and pushes the door fully
open, ends at the view pose and saves the door for the close. Without it the open ends **holding the handle**; let
go with `docker exec -it $(docker ps -qf ancestor=$KCE/microwave:dev) bash`, then
`source /module_ws/install/setup.bash && python3 -m microwave.open_door --release --execute` (ARM MOVES). The view
pose was set for the microwave's door face ~0.69 m out, the handle ~26 cm to the user's side.

### 7. Grab tupperware from the table (`microwave_place`, `take_out`)

Restart N2 with the **table** viewing pose. Then, as in step 2:

```bash
goal microwave_place take_out '{"execute": true}'             # ARM MOVES
```

The lid is off now; the grasp pinches the container's brim from the side. The open microwave door and the lid on the
table aren't in its approach world either.

### 8. Place tupperware in the microwave (`microwave_place`, `place`)

Door open (step 6), the opening in the scene camera's view. Don't move the arm between the dry run and the execute:
the executed plan must start within 1° of where it was planned.

```bash
goal microwave_place place '{"execute": false}'               # dry run: locate, plan, check
goal microwave_place place '{"execute": true}'                # ARM MOVES
```

It slides the container in, lowers it onto the turntable in impedance, lets go, tips it off the fingers and backs out.
Then put the planner's world back (Utilities): the place leaves its microwave box loaded.

```bash
ros ros2 service call /rammp_curobo/set_world rammp_curobo_interfaces/srv/SetWorld "{world: world.yaml}"
```

### 9. Close microwave (`microwave`, empty hand)

The close starts from the view pose (its dry run stops there otherwise) and needs the door the push-open saved
(`/tmp/microwave_data/door.json`). These are the settings that closed it on 11-12 (what the UI's Close Microwave
Door sends):

```bash
goal microwave open '{"view_only": true, "execute": true}'    # ARM MOVES: to the view pose
goal microwave close '{"execute": false, "face_door": true, "route_over": true, "over_above": 0.10, "side_past_hinge": 0.12, "push_radius": 0.31, "push_radius_end": 0.25, "push_z": 0.46}'
goal microwave close '{"execute": true, "face_door": true, "route_over": true, "over_above": 0.10, "side_past_hinge": 0.12, "push_radius": 0.31, "push_radius_end": 0.25, "push_z": 0.46}'   # ARM MOVES
```

Nothing checks the door latched: watch it.

### 10. Press the start button (`button_press`)

Door closed, the microwave's front facing the scene camera (it works from about 1.2 m). N6 starts with
`allow_execute:=false`:

```bash
goal button_press press_button '{"target": "start_30s"}'      # dry run: stops after gating the staging move
```

Then restart N6 with `-p allow_execute:=true`. First run: to the pre-press spot and stay, no press:

```bash
goal button_press press_button '{"target": "start_30s", "presses": 0, "no_return": true}'   # ARM MOVES
```

If the fingertips stop off the button's centre, restart N6 with `-p "prepress_aim_mm:=[x, y]"` (mm, x right, y up)
and repeat. Then the press:

```bash
goal button_press press_button '{"target": "start_30s", "presses": 1}'                      # ARM MOVES
```

It closes the hand, stages in front of the panel, re-detects the buttons from ~25 cm, presses until the wrist feels
the button (≤3 cm), backs off 15 cm and returns to its start. Other targets: `power_level`, `wgt_time_defrost`,
`timer_clock`, `stop_eco`. **Not yet run on the arm as an engine** (its README).

## Runs and logs

| Capability      | On the host                                                                    |
| --------------- | ------------------------------------------------------------------------------ |
| fridge          | `/tmp/fridge_data/` (`door.json`, `runs/`, `detections/`)                      |
| microwave       | `/tmp/microwave_data/` (`door.json`, `runs/`, `detections/`)                   |
| microwave_place | `/tmp/microwave_place_data/` (`runs/<action>_<time>/`, `plans/`)               |
| place_container | `/tmp/place_container_data/` (`start.json`, `spot.json`, `runs/`, `descents/`) |
| box_opening     | `~/.ros/box_opening/` (`runs/run-<time>.jsonl`, `gripper.json`)                |
| button_press    | `/tmp/button_press_data/press_<time>/`                                         |
| home_arm        | its terminal (N7) only                                                         |

Most runs carry a rosbag. `/tmp` is cleared on reboot: copy the session off after it.

## Known gaps

- **Grabs from the fridge and the table** use `take_out`, written and run for the microwave; the fridge and table
  viewing poses are not recorded yet, and its approach world doesn't model the fridge, the door or the table.
- **The container**: `take_out` prompts SAM 3 for "food container together with its blue sleeve"; box_opening is
  configured for the pink canister. Check both find the demo's tupperware.
- **Planner worlds**: microwave_place's `worlds_dir` mount above is untried; microwave_place's place doesn't put the
  planner's world back (step 8 does it by hand).
- **button_press** has not run on the arm as an engine; **home_arm** is experimental (standalone only).
- None of this runs from the UI yet; each capability's UI flow is in its own docs.
