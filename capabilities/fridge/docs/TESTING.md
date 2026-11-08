# Fridge open and close — how to test

- **Open** (`open`): detects the handle, grasps it, pulls the door 60° open, lets go and pushes it on to 90°. With
  `"push_open": false`: a 90° pull, then let go and go back round the door.
- **Close** (`close`): pushes the door from behind with the side of the hand to 45°; it swings shut from there.
- **Wheelchair moves:** the close picks the smallest straight move (up to 35 cm back or forward) its plan needs; the open's push sometimes needs 5–20 cm. You make them
  by hand when the goal asks, or the goal drives the base: [Driving the base](#driving-the-base-drive_base).

T1–T4 are four terminals on the robot's computer: T1 sheppy, T2 the fridge node, T3 goals, T4 the e-stop.

## Set up

```bash
# Once, and after any code change, from this repo's root (no internet: see "Building offline" below)
docker build --network host -f capabilities/fridge/Dockerfile -t ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev .

# T1, from rammp-deployments on dev
uv run scripts/set_robot.py december_2026/config/robots/flatbot
sheppy up flatbot --manifest december_2026/sheppy-manifest.yaml && sheppy status

# T2, the fridge node (mount_tilt_deg 30 on flatbot, 0 on dojo; allow_execute:=false makes every goal a dry run)
docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e 'CYCLONEDDS_URI=file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  -v /tmp/fridge_data:/fridge_data -v /tmp/rammp_planner_worlds:/rammp_planner_worlds \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 run fridge fridge_node --ros-args -p use_orchestrator:=false -p allow_execute:=true \
  -p mount_tilt_deg:=30

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
- **The planner** needs `- /tmp/rammp_planner_worlds:/rammp_planner_worlds:ro` in its manifest `volumes:`, or every
  plan refuses. If every plan fails with `EXCEPTION`: `sheppy restart planner`.
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

## Before the next run: one measurement for the grasp

Already measured (11-06): the handle is 8 in long (`handle_length_m`), its back edge 2.5 in from the door's face and
the bar ~0.5 in deep. **Still needed: how far the camera's handle point is from the bar's real front.** The grasp, the
door's face and the hinge are all placed from that point, and the camera gets little depth off the glossy bar, so it is
likely a few cm too deep (`GRIP_EXT` 0.126 compensates by trial). With it, the grasp can be set from your measurements
(fingers on the back edge). With the door closed and the wheelchair where it will be:

1. Dry-run the open (the Open command with `\"execute\": false`) and note T2's `handle (mean) [x y z]`.
1. Teleop the closed gripper straight onto the middle of the bar's front until it just touches; hold still.
1. Record the hand:
   `docker exec sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 topic echo --once /ee_state'`

Send both (the closed fingertips touch 4.1 cm past the tool frame, measured 11-06).

## How to stop the robot

| How                                      | Effect                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------ |
| **Software e-stop in T4** (first choice) | The arm **holds where it is**, powered. Motion is refused until cleared. |
| **Ctrl-C in T2**                         | Stops the node and its motion; the arm stops and holds, powered.         |
| Physical e-stop button                   | Last resort only: it cuts power and the arm drops.                       |

While the base drives, the same T4 e-stop stops it too (the node watches the arm's e-stop and stops the base),
and so does Ctrl-C in T2. See [Driving the base](#driving-the-base-drive_base).

Send the e-stop from the arm's own container: `~/ros2_ws` has a newer `EStop` than the driver (11-05), so a plain
`ros2 topic pub` from T4 fails. Keep the stop typed out in T4:

```bash
# stop (repeats at 10 Hz; Ctrl-C once the arm has halted)
docker exec sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  ros2 topic pub -r 10 /estop rammp_common_interfaces/msg/EStop "{engaged: true, source: operator, reason: stop}"'
# clear, only right before you continue; then clear the faults (Set up, T3), or `sheppy restart arm` if that fails
docker exec sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  ros2 topic pub --once /estop rammp_common_interfaces/msg/EStop "{engaged: false, source: operator, reason: clear}"'
```

## Open

1. **Teleop the camera square-on to the closed door**, ~50–60 cm from it, the handle in view, the arm clear of the
   chair (aim for `handle (mean)` x 0.85–1.0, y −0.20 to −0.34).

1. **T3, the open:**

   ```bash
   docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
     -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
     ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
     ros2 action send_goal --feedback /fridge/execute_action \
     rammp_orchestrator_interfaces/action/ExecuteAction \
     '{engine_id: fridge, action_id: open, params_json: "{\"execute\": true, \"speed\": 0.2}"}'
   ```

   It pulls, lets go, backs off the handle and pushes on, or stops with
   `Pull done and the hand is off the handle. Move the WHEELCHAIR straight back ~5 cm ...`.

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
starts) and whether the fridge drags during the pull (hinge values off). A look failing with `handle model fit ...`
says why the door's angle couldn't be measured (the handle too far, too little of it seen).

Options: `execute` (`false`: dry run), `speed` (0–1, scales every motion), `push_open` (`false`: 90° pull, no push),
`swing_deg` (the pull, 30–110; at most 85 with the push), `grasp_offset_m` (grasp this far above the bar's middle),
`chair_moved`, `release_only` (only let go and back off, after a run stopped holding the handle), `save_door` (a dry
run that plans OK also writes `door.json`, for testing the close without an executed open).

## Close

1. **Grip the container** by teleop, the hand ≥5 cm from the door and clear of the chair.

1. **Plan, T3**, the wheelchair where it was for the open:

   ```bash
   docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
     -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
     ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
     ros2 action send_goal --feedback /fridge/execute_action \
     rammp_orchestrator_interfaces/action/ExecuteAction \
     '{engine_id: fridge, action_id: close, params_json: "{\"execute\": false}"}'
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
     '{engine_id: fridge, action_id: close, params_json: "{\"execute\": true, \"chair_moved\": true}"}'
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

Options: `door_deg`, `chair_back` (m, + back, - forward; default: the plan picks it), `close_to_deg` (45), `push_radius` / `push_radius_end` (0.30 / 0.28),
`push_z`, `speed` (0.7), `hold_offset_x` (`0` if not holding the container), `side_past_hinge`, `retreat_dist`.

## Driving the base (`drive_base`)

Add `"drive_base": true, "base_speed": 0.03` to an executing goal and it makes the wheelchair move itself, straight,
a fixed distance (at most 35 cm, ending within 1 cm), then carries on. No `chair_moved` goal. **Use 0.03 m/s while testing.**

**Before:** clear 40 cm in front of and behind the base, and check that the MIB is talking (both should print):

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  bash -lc 'ros2 topic echo --once /rammp/mib/status && ros2 topic echo --once /base/odom'
```

**Open** (the push's move, if it needs one):

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: open, params_json: "{\"execute\": true, \"speed\": 0.2, \"drive_base\": true, \"base_speed\": 0.03}"}'
```

**Close** (the move its plan picked first), after its dry run passes:

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: close, params_json: "{\"execute\": true, \"drive_base\": true, \"base_speed\": 0.03}"}'
```

T2 logs `drive_base: driving the wheelchair ...` and `drive_base: stopped ...`. The base stays where it drove: if it
is still back from a close, send the next close with `chair_moved` instead of `drive_base`, or it drives the picked move again.

**Stopping it:** the T4 e-stop or Ctrl-C in T2, or the joystick if the base has one. It also stops by itself, failing
the goal with `base_stopped` and leaving the arm holding, if it goes the wrong way or further than asked, takes more
than distance / speed + 5 s, ends more than 1 cm off, or loses odometry. Check where the base is before the next goal.

**First drive: test the e-stop.** Send the close, hit the T4 e-stop while the base moves, and check it stops at once.
Do this before relying on it.

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
