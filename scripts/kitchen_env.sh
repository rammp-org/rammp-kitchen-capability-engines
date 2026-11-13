# The December demo's shell environment on the robot's computer: source it in every terminal (docs/DEMO_TESTING.md,
# "ROS environment"):
#
#   source scripts/kitchen_env.sh
#
# Nothing here needs ROS on the host: every ros2 command runs inside one of this repo's images, whose entrypoint
# sources /opt/ros/humble and the image's workspaces.

export ROS_DOMAIN_ID=0
export KCE=ghcr.io/rammp-org/rammp-kitchen-capability-engines
# Cyclone, as the images ship it, without the 10-participants-per-host cap (the Makefile's DDS_ENV)
export DDS='file:///etc/cyclonedds/config.xml,<Discovery><ParticipantIndex>none</ParticipantIndex></Discovery>'
# The fridge node only: Cyclone on eno1, the sensor network the MIB (the base) is on, for /base/odom
export DDS_ENO1='<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General><Discovery><ParticipantIndex>none</ParticipantIndex></Discovery></Domain></CycloneDDS>'
# rammp-deployments' december_2026 directory on this computer (home_arm mounts its robot config)
export DEPLOY=~/rammp-deployments/december_2026

# goal <capability> <action> ['<params json>']: one ExecuteAction goal, feedback printed, as the orchestrator sends it.
# Params are plain JSON in single quotes: goal fridge open '{"execute": true, "speed": 0.2}'. Ctrl-C cancels it.
goal() {
  local cap=$1 action=$2 params=${3:-}
  [ -n "$params" ] || params='{}'
  docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" "$KCE/$cap:dev" \
    ros2 action send_goal --feedback "/$cap/execute_action" rammp_orchestrator_interfaces/action/ExecuteAction \
    "{engine_id: $cap, action_id: $action, params_json: '$params'}"
}

# ros <command ...>: any other ros2 command, in the fridge image (it has the arm, base, cuRobo and orchestrator messages)
ros() { docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI="$DDS" "$KCE/fridge:dev" "$@"; }

# The software e-stop, from the arm's own container (its EStop message is the arm image's). docs/DEMO_TESTING.md,
# "How to stop the robot".
# estop: stop the arm (it holds where it is), repeating at 10 Hz until Ctrl-C. -it so Ctrl-C reaches the publisher:
# without it the publisher keeps running inside sheppy-arm and re-stops the arm after every release (11-12).
estop() {
  docker exec -it sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
    ros2 topic pub -r 10 /estop rammp_common_interfaces/msg/EStop "{state: 1, source: operator, reason: stop}"'
}
# estop_release: release it. A plain state 0 from the same source, NO clear_target (with one it is the MIB's clear
# handshake and releases nothing). Ends any stop publisher left running inside sheppy-arm first. Then clear_faults.
estop_release() {
  local pid
  for pid in $(docker exec sheppy-arm pgrep -f "topic pub.*/estop.*state: 1"); do docker exec sheppy-arm kill -INT "$pid"; done
  docker exec sheppy-arm bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
    ros2 topic pub -t 10 -r 10 /estop rammp_common_interfaces/msg/EStop "{state: 0, source: operator, reason: clear}"'
}
# gripper_open / gripper_close: the 2F-85 open (0) or closed (1), and nothing else moves. It streams /setpoint/gripper
# at 50 Hz for 2 s, as capability_engine's self.arm.gripper() does; the gripper keeps the last command after. Refused
# by the driver while the arm is e-stopped or contact-stopped.
_gripper() {
  docker exec -it sheppy-arm bash -lc "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
    ros2 topic pub -r 50 -t 100 /setpoint/gripper rammp_arm_interfaces/msg/GripperSetpoint \
    '{position: $1, speed: 0.5, force: 0.5}' > /dev/null"
}
gripper_open() { _gripper 0.0; }
gripper_close() { _gripper 1.0; }
# clear_faults: the arm's latched faults and contact stops (after a release, before the next action)
clear_faults() { ros ros2 service call /clear_faults std_srvs/srv/Trigger; }
