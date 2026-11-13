#!/bin/bash
# One capability's node for the kitchen UI (up.sh): domain 0, on the orchestrator, flatbot's settings
# (docs/DEMO_TESTING.md N1-N6 with use_orchestrator:=true). Idle until its button is pressed.
#
#   scripts/kitchen-ui/node.sh <fridge|microwave|microwave_place|button_press|place_container|box_opening>
#
# KUI_TAG, ALLOW_EXECUTE, MOCK as up.sh; FRIDGE_GRASP=front|side for the fridge (default front).
C=${1:?capability}
IMG=ghcr.io/rammp-org/rammp-kitchen-capability-engines/$C:${KUI_TAG:-ui-4d916ca}
DDS='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>'
# the fridge's Cyclone on eno1: the MIB's network, for /base/odom (the wheelchair-drives buttons)
ENO1='<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery></Domain></CycloneDDS>'
COMMON=(-p use_orchestrator:=true -p "allow_execute:=${ALLOW_EXECUTE:-true}" -p "mock:=${MOCK:-false}")
run() {  # run <cyclone uri> <docker args...> -- <command...>
  local uri=$1
  shift
  exec docker run --rm -it --name "kui-$C" --network host --ipc host -e ROS_DOMAIN_ID=0 -e "CYCLONEDDS_URI=$uri" "$@"
}
echo "kitchenui: $C from $IMG"
case "$C" in
  fridge)
    run "$ENO1" -v /tmp/fridge_data:/fridge_data -v curobo_kernel_cache:/planner_cache "$IMG" \
      ros2 run fridge fridge_node --ros-args "${COMMON[@]}" -p mount_tilt_deg:=30 -p "grasp:=${GRASP:-${FRIDGE_GRASP:-front}}" \
      -p planner_world_dir:=/planner_cache/rammp_planner_worlds \
      -p planner_world_dir_in_planner:=/root/.cache/rammp_planner_worlds ;;
  microwave)
    run "$DDS" -v /tmp/microwave_data:/microwave_data "$IMG" \
      ros2 run microwave microwave_node --ros-args "${COMMON[@]}" ;;
  microwave_place)
    # take_home_joints: the microwave's viewing pose (its default). Retrieve from Fridge refuses until
    # -p fridge_accept_unmodeled_door:=true (the open fridge door isn't in its collision world).
    run "$DDS" -v /tmp/microwave_place_data:/config/microwave_place -v curobo_kernel_cache:/root/.cache "$IMG" \
      ros2 run microwave_place microwave_place_node --ros-args "${COMMON[@]}" \
      -p worlds_dir:=/root/.cache/rammp_planner_worlds -p arm.service_call_s:=30.0 ;;
  button_press)
    run "$DDS" -v /tmp/button_press_data:/button_press_data -v curobo_kernel_cache:/planner_cache "$IMG" \
      ros2 run button_press button_press_node --ros-args "${COMMON[@]}" \
      -p "mount_rpy_deg:=[0.0, 30.0, 0.0]" -p "camera_frame:=''" \
      -p depth_topic:=depth/image_rect_raw -p scene_depth_topic:=depth/image_rect_raw \
      -p log_dir:=/button_press_data -p arm.go_to_track_m:=0.05 \
      -p planner_world_dir:=/planner_cache/rammp_planner_worlds \
      -p planner_world_dir_in_planner:=/root/.cache/rammp_planner_worlds ;;
  place_container)
    run "$DDS" -v /tmp/place_container_data:/config/place_container "$IMG" \
      ros2 run place_container place_container_node --ros-args "${COMMON[@]}" ;;
  box_opening)
    # no ros2 run: as PID 1 it would not pass Ctrl-C on to the run (N4); its state dir keeps the finger calibration
    mkdir -p "$HOME/.ros/box_opening"
    run "$DDS" -v "$HOME/.ros/box_opening:/state" "$IMG" \
      /module_ws/install/box_opening/lib/box_opening/box_opening_node --ros-args "${COMMON[@]}" \
      -p state_dir:=/state -p container:=ankou_pink.yaml -p world:=world_flatbot.yaml ;;
  *) echo "kitchenui: unknown capability $C"; exit 2 ;;
esac
