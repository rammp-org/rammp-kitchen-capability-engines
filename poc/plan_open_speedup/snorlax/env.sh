# Shared by the scripts here. POC: where setup.sh staged everything; DEP: the read-only deployment checkout.
POC=${POC:-$HOME/flatbot-bot-poc}
DEP=${DEP:-$HOME/rammp-deployments/december_2026}
MICROWAVE_IMAGE=${MICROWAVE_IMAGE:-ghcr.io/rammp-org/rammp-kitchen-capability-engines/microwave:dev}
CUROBO_IMAGE=${CUROBO_IMAGE:-ghcr.io/rammp-org/rammp-curobo:nightly}
ROBOT_CUROBO=${ROBOT_CUROBO:-$DEP/config/robots/flatbot/curobo}  # flatbot's cuRobo robot file (robot.yaml) and its kinematics reference
MANIFEST=${MANIFEST:-$DEP/sheppy-manifest.yaml}  # the microwave_dev params come from here
DDS_ARGS=(
  -e "CYCLONEDDS_URI=file:///etc/cyclonedds/fleet.xml,file:///etc/cyclonedds/robot.xml" -e ROS_DOMAIN_ID=0
  -v "$DEP/config/robots/current/cyclonedds.xml:/etc/cyclonedds/robot.xml:ro"
  -v "$DEP/config/dds/cyclonedds-fleet.xml:/etc/cyclonedds/fleet.xml:ro"
)
# The branch's packages over the image's installed ones.
ROS_ENV='source /opt/ros/humble/setup.bash; source /module_ws/install/setup.bash;
export PYTHONPATH=/repo/capability_engine:/repo/shared/door_common:/repo/shared/capability_ui:/repo/capabilities/microwave:$PYTHONPATH'
