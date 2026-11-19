#!/bin/bash
# button_press on the robot from this checkout, without building an image: the published :dev image with this
# branch's code mounted over it, and the terminal UI (cli_ui.py). Attended: with "execute" the arm MOVES.
#
#   capabilities/button_press/tools/jetson.sh node            # terminal 1: the node, dry run (nothing moves)
#   capabilities/button_press/tools/jetson.sh node execute    # terminal 1: the node, the arm moves (e-stop!)
#   capabilities/button_press/tools/jetson.sh ui              # terminal 2: the touchscreen's panels, in the terminal
#   capabilities/button_press/tools/jetson.sh pull            # git pull and the latest :dev image
#
# Not with the touchscreen stack (scripts/kitchen-ui/up.sh) or sheppy's own button_press node up: one of each.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
IMG=ghcr.io/rammp-org/rammp-kitchen-capability-engines/button_press:dev
CAP=$REPO/capabilities/button_press
SITE=/module_ws/install/button_press/lib/python3.10/site-packages/button_press
DOCKER=(docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=0
  -e "CYCLONEDDS_URI=file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>")

case "${1:-}" in
node)
  EXEC=false
  if [ "${2:-}" = execute ]; then
    EXEC=true
    echo "allow_execute:=true -- the arm will move. Someone on the e-stop."
  fi
  echo "branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"
  exec "${DOCKER[@]}" \
    -v "$CAP/button_press:$SITE:ro" \
    -v "$CAP/urdf:/module_ws/install/button_press/share/button_press/urdf:ro" \
    -v /tmp/button_press_data:/button_press_data -v curobo_kernel_cache:/planner_cache \
    "$IMG" ros2 run button_press button_press_node --ros-args \
    -p use_orchestrator:=false -p "allow_execute:=$EXEC" \
    -p "mount_rpy_deg:=[0.0, 30.0, 0.0]" -p "camera_frame:=''" \
    -p depth_topic:=depth/image_rect_raw -p scene_depth_topic:=depth/image_rect_raw \
    -p log_dir:=/button_press_data \
    -p planner_world_dir:=/planner_cache/rammp_planner_worlds \
    -p planner_world_dir_in_planner:=/root/.cache/rammp_planner_worlds
  ;;
ui)
  exec "${DOCKER[@]}" -v "$CAP/tools:/tools:ro" "$IMG" \
    bash -lc 'source /module_ws/install/setup.bash; python3 /tools/cli_ui.py "$@"' _ "${@:2}"
  ;;
pull)
  git -C "$REPO" pull
  docker pull "$IMG"
  ;;
*)
  sed -n '2,10p' "$0"
  exit 2
  ;;
esac
