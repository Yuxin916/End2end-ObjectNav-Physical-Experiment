import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    network_interface = LaunchConfiguration('network_interface')
    enable_on_start = LaunchConfiguration('enable_on_start')

    return LaunchDescription([
        DeclareLaunchArgument(
            'network_interface', default_value='eth0',
            description='Wired NIC on the 192.168.123.x robot net (unitree_sdk2 DDS binds here).'),
        DeclareLaunchArgument(
            'enable_on_start', default_value='false',
            description='If true, forward cmd_vel immediately (UNSAFE). Default false: stand_up + enable required.'),
        Node(
            package='unitree_g1_sdk_bridge',
            executable='g1_sdk_control',
            name='g1_sdk_bridge',
            output='screen',
            parameters=[{
                'network_interface': network_interface,
                'enable_on_start': enable_on_start,
                'max_vx': 0.6,
                'max_vy': 0.4,
                'max_vyaw': 0.8,
                'cmd_timeout': 0.5,
                'control_rate': 50.0,
            }],
        ),
    ])
