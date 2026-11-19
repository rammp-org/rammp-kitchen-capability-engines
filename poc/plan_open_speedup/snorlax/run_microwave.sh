#!/bin/bash
# Start the microwave engine from the PoC branch, standalone (no orchestrator; goals from send.sh), as sheppy runs
# microwave_dev but with the branch's code mounted. Container flatbot-poc-microwave; logs: docker logs -f flatbot-poc-microwave
#   run_microwave.sh                  every goal is a dry run (allow_execute false): nothing moves
#   run_microwave.sh --allow-execute  goals with "execute": true MOVE THE ARM: only with someone at flatbot on the e-stop
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
ALLOW=false
[ "${1:-}" = --allow-execute ] && ALLOW=true
others=$(docker ps --format '{{.Names}}' | grep -i microwave | grep -v -x flatbot-poc-microwave | grep -v microwave_place || true)
[ -z "$others" ] || { echo "another microwave node is running ($others): stop it first (two nodes both take every goal)"; exit 1; }
docker rm -f flatbot-poc-microwave >/dev/null 2>&1 || true
docker run --rm --network none -v "$POC/repo:/repo:ro" -v "$MANIFEST:/manifest.yaml:ro" --entrypoint python3 "$MICROWAVE_IMAGE" \
  /repo/poc/plan_open_speedup/snorlax/make_params.py /manifest.yaml "$ALLOW" > "$POC/data/microwave_params.yaml"
docker run -d --name flatbot-poc-microwave --network host --ipc host --runtime nvidia "${DDS_ARGS[@]}" \
  -v "$POC/repo:/repo:ro" -v "$POC/data/microwave:/microwave_data" -v "$POC/sock:/poc" \
  -v "$POC/data/microwave_params.yaml:/params.yaml:ro" --entrypoint bash "$MICROWAVE_IMAGE" -c "$ROS_ENV
exec /module_ws/install/microwave/lib/microwave/microwave_node --ros-args --params-file /params.yaml" >/dev/null
echo "microwave engine started (allow_execute $ALLOW); waiting for it ..."
for _ in $(seq 1 40); do
  docker logs flatbot-poc-microwave 2>&1 | grep -q -e 'microwave up' && break
  docker ps -q -f name=flatbot-poc-microwave | grep -q . || { docker logs --tail 30 flatbot-poc-microwave; exit 1; }
  sleep 2
done
docker logs --tail 5 flatbot-poc-microwave
