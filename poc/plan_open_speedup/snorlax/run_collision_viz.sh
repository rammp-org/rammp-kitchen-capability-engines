#!/bin/bash
# cuRobo's collision model in Foxglove (collision_viz.py): the live planner's world and the arm's spheres at
# /joint_states, on /curobo_viz/markers. Read-only (subscribes and publishes markers); GPU for the arm's kinematics.
#   run_collision_viz.sh [robot data dir, default the planner's: config/robots/current]   stop: docker rm -f flatbot-poc-collision-viz
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
ROBOT=${1:-$HOME/rammp-deployments/december_2026/config/robots/current}
DDS=$HOME/rammp-deployments/december_2026/config
docker rm -f flatbot-poc-collision-viz >/dev/null 2>&1 || true
docker run -d --name flatbot-poc-collision-viz --network host --ipc host --runtime nvidia \
  -v "$ROBOT:/config/robot:ro" -v "$POC/repo/poc/plan_open_speedup/snorlax:/viz:ro" \
  -v "$DDS/robots/current/cyclonedds.xml:/etc/cyclonedds/robot.xml:ro" -v "$DDS/dds/cyclonedds-fleet.xml:/etc/cyclonedds/fleet.xml:ro" \
  -e CYCLONEDDS_URI=file:///etc/cyclonedds/fleet.xml,file:///etc/cyclonedds/robot.xml -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v curobo_kernel_cache:/root/.cache \
  "$CUROBO_IMAGE" python3 /viz/collision_viz.py /config/robot/curobo/planner.yaml >/dev/null
echo "started; logs: docker logs -f flatbot-poc-collision-viz"
