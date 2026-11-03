# Fridge open and close — how to test

- **Open:** the `open` action. Detects the handle, grasps it, pulls the door 70° open, lets go and pushes it on to 90°.
- **Full-pull option:** `open` with `"push_open": false`. A 90° pull, then let go and go back round the door.
- **Close:** the `close` action. The microwave's push-close, reached round the door, stopping at 45°.

None of these has run on the arm from this repo yet.

## Run this first

The arm must be on the nightly driver and planner. Set `mount_tilt_deg` to `0` on dojo, `30` on flatbot.
`allow_execute:=false` makes every goal a dry run.

Open four terminals on the robot's computer. T1–T4 below and in the tables are those terminals: T1 runs sheppy, T2
the fridge node, T3 sends the goals, T4 holds the e-stop.

```bash
# Once, and after any code change, from this repo's root
docker build --network host -f capabilities/fridge/Dockerfile -t ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev .

# T1, from rammp-deployments on dev (dojo needs its own profile, with the nightly arm, planner, TF and wrist camera)
uv run scripts/set_robot.py december_2026/config/robots/flatbot
sheppy up flatbot --manifest december_2026/sheppy-manifest.yaml && sheppy status

# T2, the fridge node
docker run --rm -it --network host --ipc host -v /tmp/fridge_data:/fridge_data \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 run fridge fridge_node --ros-args -p use_orchestrator:=false -p allow_execute:=true -p mount_tilt_deg:=30

# T3, for goals: any shell with docker (the goal commands below run in the fridge image)

# T4, for the e-stop
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## Before the first run

- **Measure the handle bar** before using `grasp_offset_m`: its grippable straight length between the mounts, passed as
  `-p handle_length_m:=<m>` on the node (or set in `rammp-alternative.yaml`). The default, 0.13, is an estimate read
  off the 11-01 depth frames, not a measurement. It sets how far up or down the bar the grasp may go.

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

| Step | Terminal | Command                          | What it does                          |
| ---- | -------- | -------------------------------- | ------------------------------------- |
| 0    | —        | Teleop                           | **Point the arm at the closed door.** |
| 1    | T1, T2   | "Run this first" above           | Every node should show as running.    |
| 2    | T3       | The open below                   | **Open: 70° pull, then push to 90°.** |
| 3    | T2, T1   | Ctrl-C in T2, then `sheppy down` | Shutdown.                             |

**Step 0:** the camera square-on to the door, ~50–60 cm from it, handle in view. The whole door should fill most of
the image: on 11-01, with the arm pulled back (camera ~25 cm from the base), the wall behind the fridge was taken for
the door (handle x ≈ 1.3 m) and the open refused.

**Step 2**, in T3:

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: open, params_json: "{\"execute\": true}"}'
```

Options go inside `params_json`, each quote escaped as above, e.g. `{\"execute\": true, \"swing_deg\": 60}`:

- `execute`: `true` moves the arm; `false` is a dry run (detects and plans every leg, nothing moves).
- `push_open`: `true` (default) pulls, lets go and pushes the door on to 90°; `false` only pulls, 90° by default,
  then lets go and goes back round the door.
- `swing_deg`: how far the pull opens the door, 30–110. Default 70 with the push, 90 without. Exactly this or it
  refuses.
- `grasp_offset_m`: grasp this far above the middle of the handle bar (m), negative for below. Default 0. It refuses
  beyond ±(`handle_length_m` / 2 − 3 cm), ±3.5 cm with the default length, so the finger pads stay on the bar.
- `release_only`: `true` only lets go of the handle and backs off, for when a run stopped holding it.
- `speed`: 0–1, scales every arm motion. Default 1.

### Handle detection with SAM 3

By default the handle is found from depth: the vertical bar sticking out of the door plane. To find it with SAM 3's
mask of "handle" on the wrist image instead (the door plane still comes from depth), set it **on the node**, not in the
goal:

1. **Start SAM 3.** The flatbot profile doesn't. In T1, once, make a local copy of it with SAM 3 added (don't commit
   it), and bring that up instead of `flatbot`:

   ```bash
   cp december_2026/profiles/flatbot.yaml december_2026/profiles/flatbot_sam3.yaml
   sed -i 's/^  object_lifter: boxer$/&\n  sam3: sam3_docker/' december_2026/profiles/flatbot_sam3.yaml
   sheppy up flatbot_sam3 --manifest december_2026/sheppy-manifest.yaml && sheppy status
   ```

   SAM 3's weights aren't in its image: the manifest mounts them from `/home/abra/sam3-test/models/sam3`, so check that
   folder exists on this computer first (`ls /home/abra/sam3-test/models/sam3`). `sheppy status` should show `sam3`
   running; `sheppy logs sam3` if not.

1. **Start the fridge node with `-p handle_source:=sam3`** added to the T2 command.

The open goal is the same. Each look in T2 then shows `SAM 3: 'handle' <score>` and `SAM 3 handle: <n> px, z ... -> middle ...`. If SAM 3 isn't up, every look says `no SAM 3 service /perception/sam3/detect` and the open refuses with
`NO DETECTION`, nothing moved. **Watch where the fingers close:** the grasp corrections were tuned on depth detections.

Back to depth: start the node without `handle_source`, and `sheppy up flatbot` again if you want SAM 3 off.

## Open: what to expect

Two camera looks and a plan, then: cuRobo to the line-up point, straight in, grip, a 70° pull, let go, back off 6 cm,
round the door's free edge, in behind it, and push the door with the side of the hand to 90°. With `"push_open": false`:
a 90° pull, let go, back off, round the free edge, in beside the door. Any failed check before the swing refuses with
nothing moved. **Keep the space in front of the arm's base clear:** the hand passes right beside it, and the sim has no
wheelchair or person.

**Watch for:** where the fingers close (`GRIP_EXT`, bigger = shorter), the swing pulling toward the arm, and the fridge
dragging (hinge values off).

Runs are saved in `/tmp/fridge_data/`: `door.json`, `runs/`, `detections/`. Each run's folder in `runs/` also has
`console.log` (everything T2 printed during it) and `bag/`, a rosbag of one wrist camera frame a second (colour, depth,
intrinsics) and TF, ~75 MB a minute (`-p record_bag:=false` on the node turns it off, `-p bag_frame_hz:=<n>` changes the
rate). Copy the folder off the robot after a session: `/tmp` is cleared on reboot.

## Close: commands

The close takes **two goals**: one to plan it, one to run it, with the wheelchair moved back in between.

| Step | Terminal | Command                                  | What it does                                                          |
| ---- | -------- | ---------------------------------------- | --------------------------------------------------------------------- |
| 1    | T3       | Open step 2 above                        | Saves the door the close plans from.                                  |
| 2    | —        | Teleop to grab the container             | **Grip the container.**                                               |
| 3    | T3       | Goal 1 below: plan                       | **Plans the close.** Nothing moves. Ends by saying to move the chair. |
| 4    | —        | Move the wheelchair straight back ~30 cm | Away from the fridge, without turning it.                             |
| 5    | T3       | Goal 2 below: close                      | **Runs the close.**                                                   |
| 6    | —        | Move the wheelchair forward again        | Before the next open.                                                 |

**Goal 1, plan** (step 3, wheelchair where it was for the open). It plans the whole close for where the fridge will be
once the wheelchair is 30 cm further back, and moves nothing. When the plan works, T3 ends with:

```text
Plan OK. Move the WHEELCHAIR straight back ~30 cm (away from the fridge, without turning it), then send the close with
execute true and chair_moved true.
```

That is your cue for step 4. If it refuses instead, the message says which check failed: fix that before moving the
chair.

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: close, params_json: "{\"execute\": false}"}'
```

**Goal 2, close** (step 5, wheelchair moved back). `chair_moved` tells it the wheelchair has moved. It plans again
from where the arm is, then closes.

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>' \
  ghcr.io/rammp-org/rammp-kitchen-capability-engines/fridge:dev \
  ros2 action send_goal --feedback /fridge/execute_action \
  rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: fridge, action_id: close, params_json: "{\"execute\": true, \"chair_moved\": true}"}'
```

Without `chair_moved`, goal 2 moves nothing and refuses with "Move the WHEELCHAIR straight back ~30 cm (away from the
fridge, without turning it), then send this goal again with chair_moved true." If the wheelchair is still back from the
last close, leave it there and send goal 2.

Options, in `params_json`: `door_deg`, `chair_back` (0.30), `close_to_deg` (45), `push_radius` / `push_radius_end`
(0.30 / 0.28), `push_z`, `speed`, `hold_offset_x` (`0` if not holding the container), `side_past_hinge`, `retreat_dist`.

## Close: what to expect

Plans everything first. Backs out in front of the door's free edge, goes left past it, forward behind it, swings the
door to ~45°, backs out. The gripper is never touched and the hand never tilts. Start from a teleop-style pose, not
where the open ended.

**Watch for:** the container (it isn't in the collision model), the hand hitting the handle (10-24), and whether the
door swings shut from 45°.
