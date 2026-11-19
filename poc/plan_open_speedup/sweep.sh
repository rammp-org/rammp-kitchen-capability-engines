#!/bin/bash
# Run an engine-side script under each IK server setting: one run_curobo.sh argument line per line of $1
# ("seeds reg particle iters batch"); $2: the script (default bench_batched.py, 3 repeats).
cd ~/flatbot-bot-poc/repo/poc/plan_open_speedup
script=${2:-bench_batched.py}
while read -r args; do
  [ -z "$args" ] && continue
  docker rm -f flatbot-poc-ik >/dev/null 2>&1
  (./run_curobo.sh $args > ~/flatbot-bot-poc/data/ik_server.log 2>&1 &)
  for i in $(seq 1 120); do grep -q -e READY -e Traceback ~/flatbot-bot-poc/data/ik_server.log && break; sleep 3; done
  echo "== server: $(grep -o 'seeds.*' ~/flatbot-bot-poc/data/ik_server.log | head -1)"
  tag=${args// /_}
  if [ "$script" = bench_batched.py ]; then
    ./run_engine.sh bench_batched.py /data/flatbot.urdf /data/batched_$tag.json 3
  else
    ./run_engine.sh $script /data/flatbot.urdf /data/${script%.py}_$tag.json
  fi 2>&1 | grep -E "RESULT|Error|Traceback" | cut -c1-900
done < "$1"
