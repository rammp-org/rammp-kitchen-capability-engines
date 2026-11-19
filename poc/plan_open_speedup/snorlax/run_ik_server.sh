#!/bin/bash
# Start the cuRobo IK server that planner "curobo" asks (container flatbot-poc-ik, socket $POC/sock/ik.sock).
# GPU only: it computes joint angles, it can't command anything. Stop it with: docker rm -f flatbot-poc-ik
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
[ -s "$POC/data/flatbot.urdf" ] || { echo "run setup.sh first"; exit 1; }
[ -f "$ROBOT_CUROBO/robot.yaml" ] || { echo "no $ROBOT_CUROBO/robot.yaml: set ROBOT_CUROBO"; exit 1; }
docker rm -f flatbot-poc-ik >/dev/null 2>&1 || true
rm -f "$POC/sock/ik.sock"
docker run -d --name flatbot-poc-ik --network none --runtime nvidia \
  -v "$POC/repo:/repo:ro" -v "$POC/data:/data:ro" -v "$POC/sock:/sock" -v "$POC/curobo_core:/curobo_core:ro" \
  -v "$ROBOT_CUROBO:/robot:ro" -e PYTHONPATH=/curobo_core -e ROBOT_FILE=/robot/robot.yaml \
  -w /repo/poc/plan_open_speedup --entrypoint python3 "$CUROBO_IMAGE" \
  curobo_ik_server.py /data/flatbot.urdf /sock/ik.sock 1 1 1 12 512 >/dev/null
for _ in $(seq 1 100); do
  docker logs flatbot-poc-ik 2>&1 | grep -q READY && { echo "IK server ready (~1-2 min warm-up done)"; exit 0; }
  docker ps -q -f name=flatbot-poc-ik | grep -q . || { docker logs --tail 20 flatbot-poc-ik; exit 1; }
  sleep 3
done
echo "IK server not ready after 5 min:"; docker logs --tail 20 flatbot-poc-ik; exit 1
