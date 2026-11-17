#!/bin/bash
# Stop everything up.sh started: the capability nodes, the UI stack, the Unreal app. Nothing moves; sheppy stays up.
S=kitchenui
CAPS="fridge microwave microwave_place button_press place_container box_opening"
for old in "$S" fridgeui0; do
  if tmux has-session -t "$old" 2>/dev/null; then
    for w in $CAPS perc orch bridge ue; do tmux send-keys -t "$old:$w" C-c 2>/dev/null; done
  fi
done
sleep 6
for old in "$S" fridgeui0; do tmux kill-session -t "$old" 2>/dev/null; done
for c in $CAPS; do docker stop -t 10 "kui-$c" >/dev/null 2>&1; done
docker stop -t 10 fridge-ui >/dev/null 2>&1
pkill -f 'Binaries/LinuxArm64/RammpUI' 2>/dev/null
pkill -f 'lib/rammp_ui_bridge/ui_bridge_node|lib/rammp_orchestrator/orchestrator_node|lib/rammp_orchestrator/interactive_perception_node' 2>/dev/null
sleep 2
[ "${1:-}" = quiet ] || echo "kitchenui: down (sheppy left running)"
