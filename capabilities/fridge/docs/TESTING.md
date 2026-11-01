# Fridge open and close — how to test

- **Open:** the `open` action. Detects the handle, grasps it, swings the door open, goes back round the door.
- **Push-open option:** `open` with `push_open`. A short pull, then let go and push.
- **Close:** the `close` action. The microwave's push-close, reached round the door, stopping at 45°.

None of these has run on the arm from this repo yet.

## Run this first

The arm must be on the nightly driver and planner. Set `mount_tilt_deg` to `0` on dojo, `30` on flatbot.
`allow_execute:=false` makes every goal a dry run.

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

# T4, for the e-stop
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## Before the first run

- Measure the fridge if it isn't the 10-24 one, and pass the values as `-p name:=value` on the node: `door_width_m`
  (at the top of the door), `door_depth_m` (bins included), `door_above_handle_m`, `door_below_handle_m`,
  `hinge_along_door_m`, `pin_behind_face_m`, `handle_centre_to_face_m`. A wrong hinge drags the door or the fridge.
- **Wheelchair keep-out** (`chair_keep_out_behind_m`, 15 in): no part of the arm may be more than this behind the arm
  base while on its right. A plan that enters it refuses. cuRobo's move is checked only at its goal.
- **30° mount:** the fridge needs placing ~14–22 cm further from the arm (handle x 0.68–0.76, y −0.48 to −0.38). No
  fridge motion on flatbot until its cuRobo world has the wheelchair.

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

If the driver is faulted afterwards, run `sheppy restart arm` in T1. Never teleop while the arm driver is up: run
`sheppy down` first.

## Open: commands

| Step | Terminal | Command                                                                            | What it does                                     |
| ---- | -------- | ---------------------------------------------------------------------------------- | ------------------------------------------------ |
| 0    | —        | Teleop, with sheppy down                                                           | **Point the arm at the closed door.**            |
| 1    | T1, T2   | "Run this first" above                                                             | Every node should show as running.               |
| 2    | T3       | `make action CAP=fridge ACTION=open PARAMS='{"execute": true, "swing_deg": 90}'`   | **Open, 90°.** `"execute": false` for a dry run. |
| 2b   | T3       | `make action CAP=fridge ACTION=open PARAMS='{"execute": true, "push_open": true}'` | **Push-open instead (option).**                  |
| 3    | —        | `sheppy down`, then teleop                                                         | Grab the tupperware.                             |
| 4    | —        | Close the door by hand, back to step 0                                             | **Repeat.** `swing_deg` is 30–110.               |
| 5    | T2, T1   | Ctrl-C in T2, then `sheppy down`                                                   | Shutdown.                                        |

**Step 0:** the camera square-on to the door, ~50–60 cm from it, handle in view.

## Open: what to expect

Two camera looks and a plan, then: cuRobo to the line-up point, straight in, grip, swing, let go, back off 6 cm, round
the door's free edge, in beside the door. Any failed check before the swing refuses with nothing moved. **Keep the space
in front of the arm's base clear:** the hand passes right beside it, and the sim has no wheelchair or person.

If it stops holding the handle, let go with step 2's command using `"release_only": true` instead of `"swing_deg": 90`.

**Watch for:** where the fingers close (`GRIP_EXT`, bigger = shorter), the swing pulling toward the arm, and the fridge
dragging (hinge values off).

Runs are saved in `/tmp/fridge_data/`: `door.json`, `runs/`, `detections/`.

## Close: commands

| Step | Terminal | Command                                                                               | What it does                                       |
| ---- | -------- | ------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 1    | T3       | Open step 2 above (90° pull)                                                          | Saves the door the close plans from.               |
| 2    | T1, T2   | `sheppy down`, teleop to grab the container, then open step 1                         | **Grip the container.**                            |
| 3    | T3       | `make action CAP=fridge ACTION=close PARAMS='{"execute": false}'`                     | **Dry run.** Plans as if the chair has moved back. |
| 4    | —        | Move the wheelchair straight back ~30 cm                                              |                                                    |
| 5    | T3       | `make action CAP=fridge ACTION=close PARAMS='{"execute": true, "chair_moved": true}'` | **Close.** Refuses without `chair_moved`.          |
| 6    | —        | Move the wheelchair forward again                                                     | Before the next open.                              |

Options, in `PARAMS`: `door_deg`, `chair_back` (0.30), `close_to_deg` (45), `push_radius` / `push_radius_end`
(0.30 / 0.28), `push_z`, `speed`, `hold_offset_x` (`0` if not holding the container), `side_past_hinge`, `retreat_dist`.

## Close: what to expect

Plans everything first. Backs out in front of the door's free edge, goes left past it, forward behind it, swings the
door to ~45°, backs out. The gripper is never touched and the hand never tilts. Start from a teleop-style pose, not
where the open ended.

**Watch for:** the container (it isn't in the collision model), the hand hitting the handle (10-24), and whether the
door swings shut from 45°.
