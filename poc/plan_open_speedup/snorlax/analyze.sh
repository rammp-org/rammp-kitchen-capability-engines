#!/bin/bash
# The side-by-side numbers from the PoC microwave runs (compare_runs.py). Read-only.
#   analyze.sh                      every run
#   analyze.sh --since 20261119_0900 only runs from then on (the run folders' names are their start times)
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
docker run --rm --network none -v "$POC/repo:/repo:ro" -v "$POC/data/microwave:/microwave_data:ro" --entrypoint python3 \
  "$MICROWAVE_IMAGE" /repo/poc/plan_open_speedup/snorlax/compare_runs.py /microwave_data/runs "$@" 2>&1 | grep -v '^pybullet build'
