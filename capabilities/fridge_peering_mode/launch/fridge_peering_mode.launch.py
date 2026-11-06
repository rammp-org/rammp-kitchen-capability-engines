"""Launches the capability. The image runs this same file, so a native
`ros2 launch` and the container start the same thing.

    ros2 launch fridge_peering_mode fridge_peering_mode.launch.py                          # with the orchestrator
    ros2 launch fridge_peering_mode fridge_peering_mode.launch.py use_orchestrator:=false  # standalone
    ros2 launch fridge_peering_mode fridge_peering_mode.launch.py mock:=true               # no hardware
    ros2 launch fridge_peering_mode fridge_peering_mode.launch.py allow_execute:=true      # the arm may move

It does not start joy_node: /joy comes from the robot's joystick node (see docs/interface.md).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = ("use_orchestrator", "true"), ("mock", "false"), ("allow_execute", "false")
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default) for name, default in args]
        + [
            # arm.mock_urdf: the URDF make_sim() loads in mock runs ("" loads none)
            DeclareLaunchArgument("mock_urdf", default_value=""),
            Node(
                package="fridge_peering_mode",
                executable="fridge_peering_mode_node",
                output="screen",
                parameters=[
                    {name: ParameterValue(LaunchConfiguration(name), value_type=bool)}
                    for name, _ in args
                ]
                + [
                    {
                        "arm.mock_urdf": ParameterValue(
                            LaunchConfiguration("mock_urdf"), value_type=str
                        )
                    }
                ],
            ),
        ]
    )
