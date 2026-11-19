#!/bin/bash
# Run a script from this directory in the engine image (PyBullet), on abra. No network, no ROS graph.
# usage: run_engine.sh <script.py> [args...]   (paths: /poc = this dir, /data = ~/flatbot-bot-poc/data)
POC=~/flatbot-bot-poc
exec docker run --rm --network none -v $POC/repo:/repo:ro -v $POC/data:/data -w /repo/poc/plan_open_speedup \
  --entrypoint bash ghcr.io/rammp-org/rammp-kitchen-capability-engines/microwave:dev -c \
  "source /opt/ros/humble/setup.bash; source /module_ws/install/setup.bash; export PYTHONPATH=/repo/poc/plan_open_speedup:/repo/capability_engine:/repo/shared/door_common:/repo/capabilities/microwave:\$PYTHONPATH; python3 $*"
