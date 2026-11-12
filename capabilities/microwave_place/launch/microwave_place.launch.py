"""Launches the capability. The image runs this same file, so a native
`ros2 launch` and the container start the same thing.

    ros2 launch microwave_place microwave_place.launch.py                          # with the orchestrator
    ros2 launch microwave_place microwave_place.launch.py use_orchestrator:=false  # standalone
    ros2 launch microwave_place microwave_place.launch.py use_orchestrator:=false allow_execute:=true \
        state_dir:=$HOME/.microwave_place                                          # attended, on the arm
    ros2 launch microwave_place microwave_place.launch.py mock:=true               # no hardware
"""

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch import LaunchDescription


def generate_launch_description():
    flags = ("use_orchestrator", "true"), ("mock", "false"), ("allow_execute", "false")
    strings = ("mock_urdf", ""), ("state_dir", "/config/microwave_place"), ("ui_bridge_node_name", "ui_bridge_node")
    # Reloading the cuRobo collision world took ~13 s on Snorlax; the shared 10 s default is too short.
    timeouts = (("service_call_s", "30.0"),)
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default) for name, default in flags + strings + timeouts]
        + [
            Node(
                package="microwave_place",
                executable="microwave_place_node",
                output="screen",
                parameters=[{name: ParameterValue(LaunchConfiguration(name), value_type=bool)} for name, _ in flags]
                # arm.mock_urdf: the URDF make_sim() loads in mock runs ("" loads none)
                + [{"arm.mock_urdf": ParameterValue(LaunchConfiguration("mock_urdf"), value_type=str)}]
                + [{"arm.service_call_s": ParameterValue(LaunchConfiguration("service_call_s"), value_type=float)}]
                + [
                    {name: ParameterValue(LaunchConfiguration(name), value_type=str)}
                    for name in ("state_dir", "ui_bridge_node_name")
                ],
            ),
        ]
    )
