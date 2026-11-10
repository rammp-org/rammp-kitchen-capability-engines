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
