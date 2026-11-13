#!/bin/bash
# The whole kitchen demo from the touchscreen, on the real arm: (re)start the UI stack and every capability's node
# on the orchestrator, in tmux session `kitchenui`, and put one tappable object per capability on the UI.
# Nothing here moves the arm: every node waits for a button press. docs/DEMO_TESTING.md, "Quick start".
#
#   scripts/kitchen-ui/up.sh
#
# Environment (defaults are snorlax's):
#   FRIDGE_GRASP=front      the fridge's handle grasp: front (straight in) or side
#   KUI_CAPS="fridge microwave ..."  only these capabilities (default: all six)
#   KUI_TAG=ui-4d916ca      the capability images' tag ($KCE/<capability>:$KUI_TAG; build them first)
#   ALLOW_EXECUTE=true      false: every button refuses to move the arm (and says so on the UI)
#   MOCK=false              true: every capability runs its mock (nothing detected or moved)
#   ROS_WS=~/ros2_ws        rammp-UI-bridge and rammp-orchestrator built natively (and their interfaces)
#   ORCH_WS=~/orch_hotfix_ws  rammp-orchestrator hotfix/execute-timeout-param (no 120 s limit, action labels),
#                           overlaid on ROS_WS; empty: ROS_WS's orchestrator
#   UE_DIR, VULKAN_LOADER, UE_DISPLAY (:0)  the Unreal app, its Vulkan loader, the screen it shows on (unreal.sh)
#
# Order matters to the Unreal app (11-12): it starts first, since objects pushed before it is up never appear, and
# restarting the orchestrator under a running Unreal left it taking no taps. So this always restarts all of it.
set -u
GRASP=${FRIDGE_GRASP:-front}
case "$GRASP" in front|side) ;; *) echo "kitchenui: FRIDGE_GRASP must be front or side, not '$GRASP'"; exit 2;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export KUI_TAG=${KUI_TAG:-ui-4d916ca} ALLOW_EXECUTE=${ALLOW_EXECUTE:-true} MOCK=${MOCK:-false} GRASP
ROS_WS=${ROS_WS:-$HOME/ros2_ws}
ORCH_WS=${ORCH_WS-$HOME/orch_hotfix_ws}
KCE=ghcr.io/rammp-org/rammp-kitchen-capability-engines
S=kitchenui
CAPS=${KUI_CAPS:-fridge microwave microwave_place button_press place_container box_opening}
# one object per capability to tap, each in its own place in the scene camera's 640 x 400 image (the default box is the
# same centred one for all: on the screen they lay on top of each other, 11-12). The UI draws each one's label just
# above its box's centre, and the image's top edge sits under the screen's status banner: centres at y 200 and 330
# keep the labels clear of it (at y 115 they covered it).
OBJECTS=("fridge_1 fridge 20 150 300 250" "microwave_1 microwave 340 150 620 250"
         "table_1 table 20 280 300 380" "tupperware_1 food_container 340 280 620 380")
ENO1='<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery></Domain></CycloneDDS>'
ROSENV="source /opt/ros/humble/setup.bash && source $ROS_WS/install/setup.bash && export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI='$ENO1'"
ORCHENV=$ROSENV
[ -n "$ORCH_WS" ] && ORCHENV="$ROSENV && source $ORCH_WS/install/setup.bash"

for c in $CAPS; do
  docker image inspect "$KCE/$c:$KUI_TAG" >/dev/null 2>&1 || { echo "kitchenui: no image $c:$KUI_TAG -- build it first"; exit 1; }
done
"$HERE/down.sh" quiet

win() {  # win <name> <command>: a bash window running <command>
  if tmux has-session -t "$S" 2>/dev/null; then tmux new-window -t "$S" -n "$1" bash; else tmux new-session -d -s "$S" -n "$1" bash; fi
  sleep 1
  tmux send-keys -t "$S:$1" "$2" Enter
}
wait_for() {  # wait_for <window> <text> <seconds>
  for _ in $(seq 1 "$3"); do tmux capture-pane -p -J -t "$S:$1" -S -400 | grep -q "$2" && return 0; sleep 1; done
  echo "kitchenui: '$2' not seen in window $1 after $3 s (tmux attach -t $S to look)"; return 1
}

win ue "$HERE/unreal.sh"
sleep 20
win bridge "$ROSENV && ros2 run rammp_ui_bridge ui_bridge_node"
wait_for bridge "GUI connected" 60
win orch "$ORCHENV && ros2 run rammp_orchestrator orchestrator_node --ros-args -p object_detections_topic:=/ui/object_detections"
wait_for orch "rammp_orchestrator ready" 60
win perc "$ROSENV && ros2 run rammp_orchestrator interactive_perception_node --ros-args -r /perception/object_detections:=/ui/object_detections"
wait_for perc "ready" 30
# the objects now: each shows on the screen once a capability that takes its class has registered
for obj in "${OBJECTS[@]}"; do
  tmux send-keys -t "$S:perc" "add $obj" Enter
  sleep 1
done
for c in $CAPS; do win "$c" "$HERE/node.sh $c"; done
for c in $CAPS; do wait_for "$c" "registered" 120 && echo "kitchenui: $c registered"; done
echo "kitchenui: up: $CAPS (fridge grasp $GRASP, images $KUI_TAG, allow_execute $ALLOW_EXECUTE, mock $MOCK)."
echo "Tap an object on the touchscreen. Watch: tmux attach -t $S. Stop it all: $HERE/down.sh"
