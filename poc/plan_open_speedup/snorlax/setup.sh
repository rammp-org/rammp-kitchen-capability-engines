#!/bin/bash
# One-off, safe at any time (nothing here talks to the arm): pull the microwave image, save flatbot's live arm model
# for the IK server, and check the branch's code imports in the image.
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
mkdir -p "$POC/data/microwave" "$POC/sock"
docker pull -q "$MICROWAVE_IMAGE"
docker exec sheppy-arm_state_publisher bash -lc \
  "source /opt/ros/humble/setup.bash; ros2 param get /robot_state_publisher robot_description" \
  | sed '1s/^String value is: //' > "$POC/data/flatbot.urdf"
grep -q '<robot' "$POC/data/flatbot.urdf" || { echo "no arm model from /robot_state_publisher: is the sheppy stack up?"; exit 1; }
docker run --rm --network none -v "$POC/repo:/repo:ro" --entrypoint bash "$MICROWAVE_IMAGE" -c "$ROS_ENV
python3 -c 'import door_common.open_door as o, door_common.curobo_ik, microwave.node as n; print(\"imports from\", o.__file__, n.__file__)'" \
  | grep -v '^pybullet build'
echo "setup OK: arm model $(wc -c < "$POC/data/flatbot.urdf") bytes"
