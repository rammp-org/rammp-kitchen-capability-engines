# Fridge open and close — how to test

- **Open:** the `open` action. Detects the handle, grasps it, pulls the door 60° open, lets go and pushes it on to 90°.
- **Full-pull option:** `open` with `"push_open": false`. A 90° pull, then let go and go back round the door.
- **Close:** the `close` action. The microwave's push-close, reached round the door, stopping at 45°.

None of these has run on the arm from this repo yet.

## Run this first

The arm must be on the nightly driver and planner. Set `mount_tilt_deg` to `0` on dojo, `30` on flatbot.
`allow_execute:=false` makes every goal a dry run. T1–T4 are four terminals on the robot's computer: T1 sheppy, T2 the
fridge node, T3 goals, T4 the e-stop.

**IMPORTANT**: once you launch sheppy, there are several configs for the wrist camera. The default is *not* what we
want. There's a tuned one (I forget the exact name, ask Swap if it's not obvious) that we should always use instead.

**IMPORTANT**: check the wrist camera's IR dot projector is on (any shell), so the depth sees the door, not its
reflection: `ros2 param dump /wrist_camera | grep -i -E "enable_ir|laser|dot|flood|preset"`. With IR on, it prints
`i_enable_ir: true` and a dot-projector intensity above 0 (`false` or `0` means it's off). If it's not on, ask Swap about whether there's a way to get it on or whether this check is wrong.

```bash
# Once, and after any code change, from this repo's root
docker build --network host -f capabilities/fridge/Dockerfile -t ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev .

# T1, from rammp-deployments on dev. The flatbot profile doesn't start SAM 3: once, copy it with SAM 3 added
uv run scripts/set_robot.py december_2026/config/robots/flatbot
cp december_2026/profiles/flatbot.yaml december_2026/profiles/flatbot_sam3.yaml
sed -i 's/^  object_lifter: boxer$/&\n  sam3: sam3_docker/' december_2026/profiles/flatbot_sam3.yaml
sheppy up flatbot_sam3 --manifest december_2026/sheppy-manifest.yaml && sheppy status

# T2, the fridge node
docker run --rm -it --network host --ipc host \
  -e ROS_DOMAIN_ID=0 \
  -e 'CYCLONEDDS_URI=file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  -v /tmp/fridge_data:/fridge_data \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 run fridge fridge_node --ros-args \
  -p use_orchestrator:=false \
  -p allow_execute:=true \
  -p mount_tilt_deg:=30 \
  -p handle_source:=sam3 \
  -p door_square_to_arm:=true

# T3, for goals: any shell with docker

# T4, for the e-stop
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

- **SAM 3** finds the handle. Its weights are mounted from `/home/abra/sam3-test/models/sam3`: check that folder exists.
  `sheppy status` should show `sam3` running (`sheppy logs sam3` if not).
- **`door_square_to_arm`** (testing only): this door's glossy front gives the depth camera only its reflection, so the
  door is taken to face the arm squarely instead of being fit from depth. **Place the wheelchair square to the fridge.**
  Drop it once the door around the handle is matte.
- **Depth instead of SAM 3:** bring up `flatbot` in T1 and drop `-p handle_source:=sam3` from T2. Needs a matte door.
- **Sheppy's interactive UI:** `sheppy december_2026/sheppy-manifest.yaml` (no `up` or `status`), in another shell
  from `rammp-deployments`.

## Before the first run: measure the handle

Measure these on the fridge with the handle on, and pass them as `-p <name>:=<m>` on T2. The door's face, its inside
and the hinge are placed from the handle with them, so a wrong one drags the door or pushes air.

| Parameter                 | What to measure                                                                | Now: from the STL only |
| ------------------------- | ------------------------------------------------------------------------------ | ---------------------- |
| `handle_centre_to_face_m` | Middle of the handle bar (where the fingers close) to the door's front surface | 0.070                  |
| `handle_length_m`         | The bar's straight grippable length                                            | 0.188                  |

The door heights were measured on 11-01. Re-measure `hinge_along_door_m` and `pin_behind_face_m` only if the handle has
moved since 09-20.

## How to stop the robot

| How                                      | Effect                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------ |
| **Software e-stop in T4** (first choice) | The arm **holds where it is**, powered. Motion is refused until cleared. |
| **Ctrl-C in T2**                         | Stops the node and its motion; the arm stops and holds, powered.         |
| Physical e-stop button                   | Last resort only: it cuts power and the arm drops.                       |

**This one is the e-stop.** Keep it typed out in T4:

```bash
ros2 topic pub --once /estop rammp_common_interfaces/msg/EStop "{engaged: true, source: 'operator', reason: 'stop'}"
```

**This one clears it.** Send it only right before you continue:

```bash
ros2 topic pub --once /estop rammp_common_interfaces/msg/EStop "{engaged: false, source: 'operator', reason: 'clear'}"
```

If the driver is faulted afterwards, run `sheppy restart arm` in T1.

## Open: commands

| Step | Terminal | Command                                 | What it does                                                      |
| ---- | -------- | --------------------------------------- | ----------------------------------------------------------------- |
| 0    | —        | Teleop                                  | **Point the arm at the closed door.**                             |
| 1    | T1, T2   | "Run this first" above                  | Every node should show as running.                                |
| 2    | T3       | The open below                          | **Pull 60°, let go, back off the handle**, and push on if it can. |
| 3    | —        | Move the wheelchair, only if T3 says so | Back or forward by what it says, without turning it.              |
| 4    | T3       | The push below, only after step 3       | **Pushes the door on to 90°.**                                    |
| 5    | T2, T1   | Ctrl-C in T2, then `sheppy down`        | Shutdown.                                                         |

**Step 0:** the camera square-on to the door, ~50–60 cm from it, the handle in view. The open prints how to move the
wheelchair if the handle isn't where it planned in sim.

**Step 2**, in T3:

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: open, params_json: "{\"execute\": true, \"speed\": 0.2}"}'
```

After backing off the handle it either pushes straight on, or stops with the wheelchair move the push needs:
`Pull done and the hand is off the handle. Move the WHEELCHAIR straight back ~15 cm ..., then send the open with execute true and chair_moved true`. A dry run says the same in advance.

**Step 4**, after moving the wheelchair as asked, in T3:

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: open, params_json: "{\"execute\": true, \"chair_moved\": true, \"speed\": 0.2}"}'
```

Options go inside `params_json`, quotes escaped as above, e.g. `{\"execute\": true, \"swing_deg\": 60}`:

- `execute`: `false` is a dry run (detects and plans everything, nothing moves).
- `push_open`: `false` pulls 90° and goes round the door instead of pushing.
- `swing_deg`: the pull, 30–110 (at most 85 with the push). Default 60 with the push, 90 without.
- `grasp_offset_m`: grasp this far above the middle of the bar (negative: below), within ±(`handle_length_m`/2 − 3 cm).
- `chair_moved`: only the push, after the open asked for a wheelchair move.
- `release_only`: only let go of the handle and back off, after a run stopped holding it.
- `save_door`: a dry run that plans OK also writes `door.json`, for testing the close (below).
- `speed`: 0–1, scales every arm motion.

## Open: what to expect

Two camera looks and a plan; any failed check refuses with nothing moved. Then cuRobo to the line-up point, straight
in, grip, the pull, let go, back off 6 cm, (the wheelchair move and step 4, if asked,) round the door's free edge, in
behind it, and push the door with the side of the hand to 90°. **Keep the space in front of the arm's base clear:** the
sim has no wheelchair or person.

If a look fails with `no door plane fits the handle`, the door's depth is its reflection (see `door_square_to_arm`).

**Watch for:** where the fingers close (`GRIP_EXT`, bigger = shorter; tuned on depth detections), the swing pulling
toward the arm, and the fridge dragging (hinge values off).

Runs are saved in `/tmp/fridge_data/` (`door.json`, `runs/`, `detections/`); each run has a `console.log` and a rosbag
(`-p record_bag:=false` turns it off). Copy it off the robot after a session: `/tmp` is cleared on reboot.

## Close: commands

Two goals: one plans, one runs, with the wheelchair moved back in between.

| Step | Terminal | Command                                  | What it does                                              |
| ---- | -------- | ---------------------------------------- | --------------------------------------------------------- |
| 1    | T3       | Open step 2 above                        | Saves the door the close plans from.                      |
| 2    | —        | Teleop to grab the container             | **Grip the container.**                                   |
| 3    | T3       | Goal 1 below: plan                       | **Plans the close.** Nothing moves. Ends: move the chair. |
| 4    | —        | Move the wheelchair straight back ~30 cm | Away from the fridge, without turning it.                 |
| 5    | T3       | Goal 2 below: close                      | **Runs the close.**                                       |
| 6    | —        | Move the wheelchair forward again        | Before the next open.                                     |

**Without an executed open:** instead of step 1, dry-run the open with `\"save_door\": true` until it plans OK, then
open the door by hand to the angle it prints (90° after the push-open).

**Goal 1, plan** (wheelchair where it was for the open). It ends `Plan OK. Move the WHEELCHAIR straight back ~30 cm ...` or with the check that failed:

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: close, params_json: "{\"execute\": false}"}'
```

**Goal 2, close** (wheelchair moved back; without `chair_moved` it refuses):

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: close, params_json: "{\"execute\": true, \"chair_moved\": true}"}'
```

Options, in `params_json`: `door_deg`, `chair_back` (0.30), `close_to_deg` (45), `push_radius` / `push_radius_end`
(0.30 / 0.28), `push_z`, `speed`, `hold_offset_x` (`0` if not holding the container), `side_past_hinge`, `retreat_dist`.

## Close: what to expect

Backs out in front of the door's free edge, goes left past it, forward behind it, swings the door to ~45°, backs out.
The gripper is never touched and the hand never tilts. Start from a teleop-style pose, not where the open ended.

**Watch for:** the container (not in the collision model), the hand hitting the handle, and whether the door swings
shut from 45°.
