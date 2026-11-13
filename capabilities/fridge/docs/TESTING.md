# Fridge open and close — how to test

- **Open** (`open`): detects the handle, grasps it **from the side** (the hand slides in along the door from the hinge
  side, one finger between the bar and the door, tilted anywhere from 40° nose-up to 40° nose-down, level first),
  pulls the door 60° open, lets go and pushes it on to 90°. With `"push_open": false`: a 90° pull, then let go and go
  back round the door. `-p grasp:=front` on the node (T2) grasps from the front instead (`feature/15-fridge`'s grasp);
  everything after the grasp is the same code.
- **Close** (`close`): pushes the door from behind with the side of the hand to 45°; it swings shut from there.
- **Wheelchair moves:** the close picks the smallest straight move (up to 35 cm back or forward) its plan needs; the open's push may need one between the pull and the push (up to 35 cm; the side grasp more often than the front: the 11-10 sim of this code: see the side sweep). You make them
  by hand when the goal asks, or the goal drives the base: [Driving the base](#driving-the-base-drive_base).

T1–T4 are four terminals on the robot's computer: T1 sheppy, T2 the fridge node, T3 goals, T4 the e-stop.

## Set up

```bash
# Once, and after any code change, from this repo's root (no internet: see "Building offline" below)
docker build --network host -f capabilities/fridge/Dockerfile -t ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev .

# T1, from rammp-deployments on dev
uv run scripts/set_robot.py december_2026/config/robots/flatbot
sheppy up flatbot --manifest december_2026/sheppy-manifest.yaml && sheppy status

# T2, the fridge node (mount_tilt_deg 30 on flatbot, 0 on dojo; allow_execute:=false makes every goal a dry run).
# On eno1, the sensor network the MIB (the base) is on; the default config picks enp3s0 and never sees the base.
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e 'CYCLONEDDS_URI=<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery></Domain></CycloneDDS>' \
  -v /tmp/fridge_data:/fridge_data -v curobo_kernel_cache:/planner_cache \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 run fridge fridge_node --ros-args -p use_orchestrator:=false -p allow_execute:=true \
  -p mount_tilt_deg:=30 -p grasp:=side \
  -p planner_world_dir:=/planner_cache/rammp_planner_worlds \
  -p planner_world_dir_in_planner:=/root/.cache/rammp_planner_worlds

# T3: the goal commands below (Open, Close), each a docker run

# T3, before any runs (and after any fault): clear the arm's faults
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 service call /clear_faults std_srvs/srv/Trigger
```

- **SAM 3** finds the handle; its weights are mounted from `/home/abra/sam3-test/models/sam3` (check it exists).
  `sheppy status` should show `sam3` running (`sheppy logs sam3` if not).
- **The door's angle:** the handle model fit on the wrist camera (`door_normal: handle_model`); `-p door_normal:=square`
  assumes it square to the arm (place the wheelchair square). `-p detect_camera:=scene` uses the scene camera.
- **The planner** reads the fridge world through the `curobo_kernel_cache` volume it already mounts at `/root/.cache`
  (T2 mounts it too), so the manifest needs no extra volume. `config ... fridge_world.yaml not found` = T2 lacks it. If
  every plan fails with `EXCEPTION`: `sheppy restart planner`.
- **Depth instead of SAM 3:** `-p handle_source:=depth -p door_normal:=plane`. Needs a matte door.
- **Wrist stereo (for FoundationStereo later, never yet seen live):** the bag records it only if the wrist OAK
  publishes it. Add to its params in the robot profile (compressed: the raw pair doesn't fit its 100 Mbit/s link):
  ```yaml
  depth.i_left_rect_publish_topic: true
  depth.i_right_rect_publish_topic: true
  depth.i_left_rect_low_bandwidth: true
  depth.i_right_rect_low_bandwidth: true
  depth.i_left_rect_low_bandwidth_quality: 95
  depth.i_right_rect_low_bandwidth_quality: 95
  ```
  Check `ros2 topic hz /wrist_camera/infra1/image_rect_raw` (and `infra2`) gives ~15 Hz and the colour and depth still
  do. The names come from the driver's source: if the camera's differ, set the `wrist_infra*_topic` parameters. After a
  run the console should say `M stereo pairs` with M close to N; 0 means nothing was recorded.

## UI

The fridge's buttons on the UI (`docs/interface.md`, "UI buttons"). Needs Unreal running and `~/ros2_ws` built with
rammp-orchestrator (`feat/orchestrator`) and rammp-UI-bridge (`init`).

```bash
# U1-U3, each: Cyclone on eno1, as T2 (Fast DDS, or another interface, doesn't see the fridge node)
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery></Domain></CycloneDDS>'
ros2 run rammp_ui_bridge ui_bridge_node                    # U1
ros2 run rammp_orchestrator orchestrator_node              # U2
ros2 run rammp_orchestrator interactive_perception_node    # U3, then type: add fridge_1 fridge
```

T1 as in Set up; T2 as in Set up but `-p use_orchestrator:=true`. The buttons move the arm (`allow_execute:=true`);
first try them with `-p mock:=true`: nothing moves and each ends "Mock: ... No arm motion."

In Unreal select **fridge_1**: it offers five buttons. Each shows what it is doing, with **Stop**, then a panel with
**Done** (it waits at most 30 s; a question at most 600 s). **Stop** stops the arm (and base) where it is and shows
"Stopped. ..."; after Stop during an open, if the hand is on the handle, send `release_only` before anything else.

- **Go to Fridge** → "The arm is in front of the fridge."
- **Open Fridge Door** → "Fridge door opened." If the push needs it, first: "Move your wheelchair straight back ...
  about N cm ..." Move it, press **Continue** (or **Stop**).
- **Close Fridge Door** (grip the container first, as in Close step 1) → "Fridge door closed." It may first ask for a
  wheelchair move, as the open does. Move the wheelchair forward again after.
- **... (wheelchair drives)**: the same, but the base drives the move at 0.03 m/s (Arm + Drive Base, "Before").
- A refusal: `Couldn't ... Press <button> to try again.` with that button and **Stop**. Any other failure: an error
  panel with **Done**; after a failed open, if it says the hand may still be holding the handle, send `release_only`.

No panels: T2 logs `standalone/no session token` (check `use_orchestrator:=true`) or
`request_user_input server unavailable` (U1 down, or not on Cyclone).

## How to stop the robot

| How                                      | Effect                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------ |
| **Software e-stop in T4** (first choice) | The arm **holds where it is**, powered. Motion is refused until cleared. |
| **Ctrl-C in T2**                         | Stops the node and its motion; the arm stops and holds, powered.         |
| Physical e-stop button                   | Last resort only: it cuts power and the arm drops.                       |

While the base drives, the same T4 e-stop stops it too (the node watches the arm's e-stop and stops the base),
and so does Ctrl-C in T2. See [Driving the base](#driving-the-base-drive_base).

Send the e-stop from the arm's own container (`docker exec sheppy-arm`, both `source` lines). Since the arm
image was updated (after 11-06) `EStop` is the NEW message: `uint8 state` (1 = engage, 0 = not), no `engaged` field
-- the old `{engaged: true}` fails with `Failed to populate field: 'EStop' object has no attribute 'engaged'`.
Check with `ros2 interface show rammp_common_interfaces/msg/EStop` in that container. **2026-11-09: these two are
NOT yet verified on the arm** -- test at rest first (`docker logs -f sheppy-arm` should print
`E-STOP engaged by 'operator'` then `e-stop RELEASED by 'operator'`), then while it moves. Keep the stop typed out in T4:

```bash
# stop (repeats at 10 Hz; Ctrl-C once the arm has halted)
docker exec -it sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  ros2 topic pub -r 10 /estop rammp_common_interfaces/msg/EStop "{state: 1, source: operator, reason: stop}"'
# release, only right before you continue (if it does not release: `sheppy restart arm`); then clear the faults (Set up, T3)
docker exec sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  ros2 topic pub -t 10 -r 10 /estop rammp_common_interfaces/msg/EStop "{state: 0, source: operator, reason: clear}"'
```

Two things the arm driver does (its `docs/interface.md`, "The e-stop"; both bit on 11-12):

- **Ctrl-C must reach the stop's publisher**: hence `docker exec -it`. Without `-t`, Ctrl-C ends only the docker
  client; `ros2 topic pub` keeps running inside `sheppy-arm`, keeps the arm stopped, and re-stops it after every
  release. Check with `docker exec sheppy-arm ps -eo pid,args | grep "topic pub"`; end one left over with
  `docker exec sheppy-arm kill -INT <pid>`.
- **The release is a plain `state: 0` from the same `source`, with no `clear_target`.** A message with
  `clear_target` set is the MIB's clear handshake and never releases anything. The driver logs
  `E-STOP engaged by 'operator'` on the stop and the release on a release; `/control_status` shows `estopped`.
  After the release, `/clear_faults` before the next action, or its first motion stops short.

## Arm + Drive Base

Add `"drive_base": true, "base_speed": 0.03` to an executing goal and it makes the wheelchair move itself, straight,
a fixed distance (at most 35 cm, ending within 1 cm), then carries on. No `chair_moved` goal. **Use 0.03 m/s while testing.**

**Before:** clear 40 cm in front of and behind the base, and check that the MIB is talking and odometry is live:

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery></Domain></CycloneDDS>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  bash -lc 'ros2 topic echo --once /rammp/mib/status rammp_base_interfaces/msg/MibStatus --field system_state; \
    echo "live odom (5 s):"; timeout 5 ros2 topic echo /base/odom nav_msgs/msg/Odometry \
    --qos-durability volatile --qos-reliability best_effort --field header.stamp.sec'
```

- **Pass:** a `system_state` (1 = IDLE), then a steady stream of odom stamps (20 Hz) for 5 s.
- **Status prints, odom prints nothing:** the MIB is up but the drive motor boards are not reporting to it. The MIB
  publishes `/base/odom` only from fresh drive-wheel motor states (RAMMP-mib-fw `publish_odometry`), even at rest, so
  `drive_base` refuses with `no /base/odom`. Fix it on the base (motor power, power-cycle the base, or the MIB panel
  `tools/mib_panel` in RAMMP-mib-fw, which shows each motor online/offline and its faults), not in this code.
- **Nothing prints, or `does not appear to be published yet`:** the command is not on `eno1`, the sensor network the
  MIB is on (the default Cyclone config picks `enp3s0`, the arm's cable), or the MIB is off.

Why this command: the message types are given because without them `echo` looks the topic up before the MIB has
been discovered and gives up (`Could not determine the type`). The odom echo is `volatile` because `/base/odom` is
transient-local: a plain `echo --once` gets the last saved message replayed and "passes" even when odometry stopped
long ago. "Malformed packet received from vendor 13.37" lines come from a device on the sensor network and can be
ignored here. T2 must run on `eno1` too (see Set up), or `drive_base` refuses with `no /base/odom`.

### Open

First send the arm to the fridge start pose (`go_to_start`, as in [Without drive base](#without-drive-base), step 1),
then:

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: open, params_json: "{\"execute\": true, \"speed\": 0.2, \"drive_base\": true, \"base_speed\": 0.03}"}'
```

### Close

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: close, params_json: "{\"execute\": true, \"speed\": 0.7, \"drive_base\": true, \"base_speed\": 0.03}"}'
```

T2 logs `drive_base: driving the wheelchair ...` and `drive_base: stopped ...`. The base stays where it drove: if it
is still back from a close, send the next close with `chair_moved` instead of `drive_base`, or it drives the picked move again.

**Stopping it:** the T4 e-stop or Ctrl-C in T2, or the joystick if the base has one. It also stops by itself, failing
the goal with `base_stopped` and leaving the arm holding, if it goes the wrong way or further than asked, takes more
than distance / speed + 5 s, ends more than 1 cm off, or loses odometry. Check where the base is before the next goal.

**First drive: test the e-stop.** Send the close, hit the T4 e-stop while the base moves, and check it stops at once.
Do this before relying on it.

## Without drive base

### Open

1. **The start pose:** in T3, plan it first (`"execute": false`), then send it with `"execute": true`:

   ```bash
   docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
     -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
     ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
     ros2 action send_goal --feedback /fridge/execute_action \
     rammp_orchestrator_interfaces/action/ExecuteAction \
     '{engine_id: fridge, action_id: go_to_start, params_json: "{\"execute\": true, \"speed\": 0.2}"}'
   ```

   cuRobo plans round the wheelchair's layers, not the fridge (not detected yet): keep the way clear of it.

1. **Point the camera square-on at the closed door**, ~50–60 cm from it, the handle in view, by placing the
   wheelchair (side grasp: aim for `handle (mean)` x 0.70–0.85, y −0.40 to −0.20, the middle of where the side
   grasp's planner planned in sim; front grasp: x 0.85–1.0, y −0.20 to −0.34).

1. **T3, the open:**

   ```bash
   docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
     -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
     ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
     ros2 action send_goal --feedback /fridge/execute_action \
     rammp_orchestrator_interfaces/action/ExecuteAction \
     '{engine_id: fridge, action_id: open, params_json: "{\"execute\": true, \"speed\": 0.2}"}'
   ```

   It pulls to 60° (with the side grasp the hand turns up to 30° less than the door, the bar turning in the gripper)
   and stops, lets go, slides back off the bar, moves out and round to in front of the handle (where the front grasp's
   back-off ends) and pushes on, or stops with
   `Pull done and the hand is off the handle. Move the WHEELCHAIR straight back ~5 cm ...`. A refused plan
   (dry run or not) then spends up to 90 s planning it again for wheelchair moves of 5 and 10 cm and ends
   `checked by planning it: the open plans with ... -- move the wheelchair ...` (or that no such move helps:
   the refusal isn't about placement). Prefer that over the `placement:` lines, which come from a sim.

   **Without any wheelchair move** add `\"base_moves\": false` to the params (e.g.
   `"{\"execute\": false, \"base_moves\": false}"` for the dry run): the push must plan from where the chair
   stands, or the open refuses before anything moves. The handle cells (x, y) (0.65, −0.15), (0.65, −0.10),
   (0.65, −0.05), (0.70, −0.15), (0.75, −0.25) and (0.90, −0.10) planned in sim only with the move, so the
   `placement:` advice leaves them out. Those cells are the **front** grasp's: the side grasp has no such list (its
   sim mostly planned only with a move), so with `grasp:=side` the advice may name a spot whose open then refuses,
   before anything moves.

1. **Only if asked:** move the wheelchair as it says (without turning it), then
   push the door on to 90°, in T3:

   ```bash
   docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
     -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
     ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
     ros2 action send_goal --feedback /fridge/execute_action \
     rammp_orchestrator_interfaces/action/ExecuteAction \
     '{engine_id: fridge, action_id: open, params_json: "{\"execute\": true, \"chair_moved\": true, \"speed\": 0.2}"}'
   ```

Any failed check refuses with nothing moved; a dry run (`"execute": false`) plans everything with cuRobo and says in
advance what it will do. cuRobo plans round the planner's wheelchair and occupant boxes, but those are best guesses:
**keep the space in front of the arm's base clear**, and start from a pose outside them (the web app's Fridge pose;
a start inside them refuses at once, naming the boxes). Watch where the fingers close (the gripper closes as the pull
starts) and whether the fridge drags during the pull (hinge values off). **Side grasp, first runs:** dry-run first and
watch the slide in at a low `speed`: the finger behind the bar must pass between it and the door. Its depth on the bar
is the one that worked on 11-06 (`SIDE_PADS_INTO_M`, `SIDE_PADS_PAST_BAR_M`), found with the scene camera's handle
point; the wrist camera's (the default `detect_camera`) is unchecked for it. If the finger meets the bar's front or the
door, stop (T4) and say how far off it was. The plan names the grasp it picked (`plan: side grasp, tilt +10 deg, ...`). A look failing with `handle model fit ...`
says why the door's angle couldn't be measured (the handle too far, too little of it seen).

Options: `execute` (`false`: dry run), `speed` (0–1, scales every motion), `push_open` (`false`: 90° pull, no push),
`swing_deg` (the pull, 30–110; at most 85 with the push), `grasp_offset_m` (grasp this far above the bar's middle),
`chair_moved`, `release_only` (only let go and back off, after a run stopped holding the handle), `save_door` (a dry
run that plans OK also writes `door.json`, for testing the close without an executed open).

### Close

1. **Grip the container** by teleop, the hand ≥5 cm from the door and clear of the chair.

1. **Plan, T3**, the wheelchair where it was for the open:

   ```bash
   docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
     -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
     ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
     ros2 action send_goal --feedback /fridge/execute_action \
     rammp_orchestrator_interfaces/action/ExecuteAction \
     '{engine_id: fridge, action_id: close, params_json: "{\"execute\": false, \"speed\": 0.7}"}'
   ```

   It ends
   `Plan OK. Move the WHEELCHAIR straight back ~N cm ...` (or forward; the smallest move, up to 35 cm, after which
   the close plans) or says which check failed.

1. **Move the wheelchair as it says**, straight, without turning it (the close goal then uses that move).

1. **Close, T3:**

   ```bash
   docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
     -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
     ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
     ros2 action send_goal --feedback /fridge/execute_action \
     rammp_orchestrator_interfaces/action/ExecuteAction \
     '{engine_id: fridge, action_id: close, params_json: "{\"execute\": true, \"chair_moved\": true, \"speed\": 0.7}"}'
   ```

   It goes round the door's free edge, behind it,
   swings the door to ~45° and backs out; the door swings shut from there. Move the wheelchair forward again after.

- **Speed:** the close runs at 0.7 by default. **Keep it at 0.7 or more**: slower, the swing doesn't carry the door
  shut.
- **Several closes in a row:** a close that swings the door saves it as 45° open in `door.json`
  (`/tmp/fridge_data/door.json` on the robot). The next close then plans for a door at 45° and, with the door really
  open further, refuses. Give the door's real
  angle in the close goal's `params_json`, e.g. `{\"execute\": true, \"chair_moved\": true, \"door_deg\": 90}` (or set
  `"door_open_deg"` in `door.json` back to it).
- **Without an executed open:** dry-run the open with `"save_door": true` until it plans OK, then open the door by hand
  to the angle it prints (90° after the push-open).

Options: `door_deg`, `chair_back` (m, + back, - forward; default: the plan picks it), `close_to_deg` (45), `push_radius`
(default: searched), `push_z` (default: searched), `speed` (0.7), `retreat_dist`. The close carries the container
when the gripper is closed at its start, and not when it is open.

## Runs

Saved in `/tmp/fridge_data/` (`door.json`, `runs/`, `detections/`). Each run has a `console.log` and a rosbag of the
cameras, boxer's boxes and TF (`-p record_bag:=false` turns it off). Copy them off after a session: `/tmp` is cleared on
reboot.

## Building offline (no internet)

The normal build needs the network. For a source-only change (Python or launch files, nothing in a `package.xml`, the
`pip` pins or the RAMMP-CuRobo commit), rebuild on top of the image you have:

```bash
cat > /tmp/fridge.offline.Dockerfile <<'EOF'
FROM ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev
WORKDIR /module_ws
COPY capability_engine/ src/capability_engine/
COPY shared/door_common/ src/door_common/
COPY capabilities/fridge/ src/fridge/
RUN . /opt/ros/humble/setup.sh && . /ros2_ws/install/setup.sh && colcon build && rm -rf build log
EOF
DOCKER_BUILDKIT=0 docker build --network host -f /tmp/fridge.offline.Dockerfile \
  -t ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev .
```
