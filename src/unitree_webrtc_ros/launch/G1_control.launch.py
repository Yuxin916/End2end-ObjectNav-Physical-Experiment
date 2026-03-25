#!/usr/bin/env python3
"""
Launch file for Unitree G1 WebRTC control node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Unitree G1 control node."""

    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='10.0.0.191',   # G1 IP (change to your default)
        description='IP address of the Unitree G1 robot'
    )

    connection_method_arg = DeclareLaunchArgument(
        'connection_method',
        default_value='LocalSTA',
        description='Connection method: LocalAP, LocalSTA, or Remote'
    )

    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value='wireless_controller',
        description='Control mode: sport_cmd or wireless_controller'
    )

    g1_control_node = Node(
        package='unitree_webrtc_ros',          # same package
        executable='g1_control',              # new entry point
        name='g1_control',
        output='screen',
        parameters=[{
            'robot_ip': LaunchConfiguration('robot_ip'),
            'connection_method': LaunchConfiguration('connection_method'),
            'control_mode': LaunchConfiguration('control_mode'),
        }],
        remappings=[
            # ('cmd_vel', '/g1/cmd_vel'),
        ]
    )

    return LaunchDescription([
        robot_ip_arg,
        connection_method_arg,
        control_mode_arg,
        g1_control_node,
    ])
