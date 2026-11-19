#!/bin/bash
# Send one goal to the PoC microwave engine and wait for its result.
#   send.sh open '{"execute": false, "compare_planners": true}'
#   send.sh open '{"execute": true, "push_open": true, "planner": "curobo"}'
#   send.sh close_door_ui '{}'
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
ACTION=${1:?action id: open, close or close_door_ui}
PARAMS=${2:-"{}"}
GOAL="{engine_id: microwave, action_id: $ACTION, params_json: '$PARAMS'}"
docker run --rm --network host --ipc host "${DDS_ARGS[@]}" -e GOAL="$GOAL" --entrypoint bash "$MICROWAVE_IMAGE" -c \
  'source /opt/ros/humble/setup.bash; source /module_ws/install/setup.bash;
   ros2 action send_goal --feedback /microwave/execute_action rammp_orchestrator_interfaces/action/ExecuteAction "$GOAL"' \
  | grep -v '^pybullet build'
