# Testing the microwave

## Tests

```bash
make check
make lint
make test-arm
make test-arm ARGS="tests/arm/test_door_planning.py -k chair"
```

## Sheppy

From `rammp-deployments` on the Jetson:

```bash
sheppy up flatbot --manifest december_2026/sheppy-manifest.yaml
sheppy status
sheppy status microwave
sheppy logs microwave -n 50
sheppy restart microwave
sheppy down
```

## Let the arm move

In `december_2026/profiles/flatbot.yaml`, then rerun `sheppy up`:

```yaml
overrides:
  microwave:
    allow_execute: true
```

## Run

Paste this once per terminal:

```bash
mw() { docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 ghcr.io/rammp-org/rammp-kitchen-capability-engines/microwave:dev "$@"; }
```

```bash
# view pose (ARM MOVES)
mw ros2 action send_goal --feedback /microwave/execute_action rammp_orchestrator_interfaces/action/ExecuteAction '{engine_id: microwave, action_id: open, params_json: "{\"view_only\": true, \"execute\": true}"}'

# dry run
mw ros2 action send_goal --feedback /microwave/execute_action rammp_orchestrator_interfaces/action/ExecuteAction '{engine_id: microwave, action_id: open, params_json: "{\"execute\": false}"}'

# open, slow (ARM MOVES)
mw ros2 action send_goal --feedback /microwave/execute_action rammp_orchestrator_interfaces/action/ExecuteAction '{engine_id: microwave, action_id: open, params_json: "{\"execute\": true, \"speed\": 0.3}"}'

# open at full speed (ARM MOVES)
mw ros2 action send_goal --feedback /microwave/execute_action rammp_orchestrator_interfaces/action/ExecuteAction '{engine_id: microwave, action_id: open, params_json: "{\"execute\": true}"}'

# choose the swing, 45-75 (ARM MOVES)
mw ros2 action send_goal --feedback /microwave/execute_action rammp_orchestrator_interfaces/action/ExecuteAction '{engine_id: microwave, action_id: open, params_json: "{\"execute\": true, \"swing_deg\": 60}"}'

# the full open with the push-open -- off by default until the scene camera is in the checks (11-10 the
# push-open's 'up and back' hit it); without push_open the open stops after the swing, still holding the handle
mw ros2 action send_goal --feedback /microwave/execute_action rammp_orchestrator_interfaces/action/ExecuteAction '{engine_id: microwave, action_id: open, params_json: "{\"execute\": true, \"push_open\": true}"}'
```

## UI

The UI action **Open Microwave Door** (`open_door_ui`) runs the whole open from one press, executing (the node still
needs `allow_execute: true` and, for the UI, `use_orchestrator: true` in the profile's microwave overrides). A refused
check before the grasp offers the same button again; a stop with the hand on the handle says so and is not retried.

The orchestrator, UI bridge and object selection run natively from `~/ros2_ws` (`rammp-orchestrator`
`feat/orchestrator`, `rammp-UI-bridge` `init`, `rammp-interfaces-ros2` `dev`; built there, not from this repo); the
microwave runs in its sheppy container. In every native terminal:

```bash
source /opt/ros/humble/setup.zsh
source ~/ros2_ws/install/setup.zsh
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp   # the containers' middleware: Fast DDS can't see them
export PYTHONPATH="$HOME/ros2_ws/.panel-demo-python${PYTHONPATH:+:$PYTHONPATH}"   # NumPy 1.24 for system SciPy
ros2 daemon stop; ros2 daemon start            # once, after switching the middleware
```

```bash
# 1: UI bridge
ros2 run rammp_ui_bridge ui_bridge_node
# 2: orchestrator, its object list from the selection node below
ros2 run rammp_orchestrator orchestrator_node --ros-args -p object_detections_topic:=/ui/object_detections
# 3: object selection: a fixed microwave_1 for the UI; then type  add microwave_1 microwave
ros2 run rammp_orchestrator interactive_perception_node --ros-args -r /perception/object_detections:=/ui/object_detections
# 4: the RammpUI app
~/RammpUI-detection-markers-LinuxArm64/RammpUI-Arm64.sh
```

Selection goes on its own topic: the microwave reads the scene segmenter's real boxes on `/perception/object_detections`
to find the door, which a fixed fake box there would replace; and the orchestrator re-sends the UI's object list
whenever a box moves, so live boxes' pixel jitter made the action panel flash and drop taps (11-12). In the UI: select
`microwave_1`, then **Open Microwave Door**. The microwave registers with the orchestrator within ~15 s of it starting
(`sheppy logs microwave | grep registered`).

From the command line, through the orchestrator (panels still show):

```bash
mw ros2 service call /rammp_orchestrator/select_object rammp_orchestrator_interfaces/srv/SelectObject "{instance_id: 'microwave_1', perception_class: 'microwave'}"
mw ros2 action send_goal --feedback /rammp_orchestrator/execute_action rammp_orchestrator_interfaces/action/ExecuteAction '{engine_id: microwave, action_id: open_door_ui, params_json: "{}"}'
```

Under sheppy the microwave container bind-mounts this checkout's `capabilities/microwave/microwave` and
`shared/door_common/door_common` over the installed packages (`sheppy restart microwave` picks up a change);
`capability_engine` and the interface packages come from the image.

## Stop

```bash
# pause
mw ros2 service call /microwave/control rammp_orchestrator_interfaces/srv/Control '{command: 0}'
# resume
mw ros2 service call /microwave/control rammp_orchestrator_interfaces/srv/Control '{command: 1}'
# cancel
mw ros2 service call /microwave/control rammp_orchestrator_interfaces/srv/Control '{command: 2}'
```

## Inside the container

```bash
docker exec -it $(docker ps -qf ancestor=ghcr.io/rammp-org/rammp-kitchen-capability-engines/microwave:dev) bash
source /module_ws/install/setup.bash

# watch the planned run: writes <run>/plan.mp4
python3 -m microwave.render_plan /microwave_data/runs/<time>_open

# let go of the handle and back off (ARM MOVES)
python3 -m microwave.open_door --release --execute

# detection
python3 -m microwave.detection_debug
python3 -m microwave.detection_debug --live
```

## Refresh the chair and user boxes

```bash
make sync-chair
```
