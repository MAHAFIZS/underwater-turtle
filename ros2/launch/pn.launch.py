from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory("underwater_turtle_ros")
    params = os.path.join(pkg, "config", "pn_params.yaml")

    return LaunchDescription([
        Node(
            package="underwater_turtle_ros",
            executable="pn_node",
            name="underwater_turtle_pn_node",
            output="screen",
            parameters=[params],
        ),
    ])
