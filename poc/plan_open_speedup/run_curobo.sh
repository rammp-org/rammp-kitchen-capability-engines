#!/bin/bash
# Start the cuRobo IK server (curobo_ik_server.py) in the rammp-curobo image on abra, socket /data/ik.sock.
# usage: run_curobo.sh [num_seeds] [regularization 0/1] [particle 0/1]
POC=~/flatbot-bot-poc
rm -f $POC/data/ik.sock
exec docker run --rm --name flatbot-poc-ik --network none --runtime nvidia -v $POC/repo:/repo:ro -v $POC/data:/data -v $POC/curobo_core:/curobo_core:ro -e PYTHONPATH=/curobo_core \
  -w /repo/poc/plan_open_speedup --entrypoint python3 ghcr.io/rammp-org/rammp-curobo:nightly \
  curobo_ik_server.py /data/flatbot.urdf /data/ik.sock "$@"
