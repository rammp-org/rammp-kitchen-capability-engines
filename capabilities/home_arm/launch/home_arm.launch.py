"""Launches the capability. The image runs this same file, so a native `ros2 launch` and the
container start the same thing.

    ros2 launch home_arm home_arm.launch.py                          # standalone, the default
    ros2 launch home_arm home_arm.launch.py use_orchestrator:=true   # registers; go_home refused
    ros2 launch home_arm home_arm.launch.py mock:=true               # no hardware
    ros2 launch home_arm home_arm.launch.py mock:=true mock_urdf:=/robot.urdf  # make_sim's model

Standalone is the default here, unlike the template's: the orchestrator cannot dispatch go_home
(it belongs to no object), and under it the engine refuses the goal (docs/interface.md).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = ("use_orchestrator", "false"), ("mock", "false")
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default) for name, default in args]
        + [
            # arm.mock_urdf: the URDF make_sim() loads in mock runs ("" loads none)
            DeclareLaunchArgument("mock_urdf", default_value=""),
            Node(
                package="home_arm",
                executable="home_arm_node",
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
