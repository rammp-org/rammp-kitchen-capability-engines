#!/bin/bash
# Copy this branch and RAMMP-CuRobo's core/ to snorlax (~/flatbot-bot-poc) and run setup.sh there. Run it from a
# machine with both checkouts; nothing here talks to the arm.
#   stage.sh [path to RAMMP-CuRobo checkout] [ssh host]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
CUROBO=${1:-$REPO/../RAMMP-CuRobo}
HOST=${2:-snorlax}
[ -d "$CUROBO/core/rammp_curobo" ] || { echo "no RAMMP-CuRobo checkout at $CUROBO"; exit 1; }
ssh "$HOST" 'mkdir -p ~/flatbot-bot-poc'
rsync -az --delete --exclude .git "$REPO/" "$HOST:flatbot-bot-poc/repo/"
rsync -az --delete "$CUROBO/core/" "$HOST:flatbot-bot-poc/curobo_core/"
ssh "$HOST" 'bash ~/flatbot-bot-poc/repo/poc/plan_open_speedup/snorlax/setup.sh'
