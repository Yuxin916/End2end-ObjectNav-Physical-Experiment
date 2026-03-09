"""
Launch file for sam2_detector.

Run standalone (vlm_nav_bridge must already be running to receive /egocentric_rgb):
  ros2 launch sam2_detector sam2_detector.launch.py

Override parameters at launch time:
  ros2 launch sam2_detector sam2_detector.launch.py \\
      grounded_sam2_root:=/path/to/Grounded-SAM-2 \\
      use_full_prompt:=true \\
      inference_hz:=1.0
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('sam2_detector')
    default_config = os.path.join(pkg_share, 'config', 'sam2_detector.yaml')

    args = [
        DeclareLaunchArgument('grounded_sam2_root',
                              default_value='/home/tsaisplus/projects/VLN_CL_CoTNav/Grounded-SAM-2',
                              description='Absolute path to Grounded-SAM-2 repository'),
        DeclareLaunchArgument('use_full_prompt', default_value='false',
                              description='Detect all 21 indoor categories (true) or goal only (false)'),
        DeclareLaunchArgument('inference_hz', default_value='2.0',
                              description='Maximum inference rate (Hz)'),
        DeclareLaunchArgument('device', default_value='cuda:0',
                              description='Torch device for SAM2 + GroundingDINO'),
        DeclareLaunchArgument('camera_topic', default_value='/egocentric_rgb',
                              description='Input image topic (projected pinhole RGB)'),
        DeclareLaunchArgument('config_file', default_value=default_config,
                              description='Path to sam2_detector.yaml'),
    ]

    node = Node(
        package='sam2_detector',
        executable='sam2_detector',
        name='sam2_detector',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'grounded_sam2_root': LaunchConfiguration('grounded_sam2_root'),
                'use_full_prompt':    LaunchConfiguration('use_full_prompt'),
                'inference_hz':       LaunchConfiguration('inference_hz'),
                'device':             LaunchConfiguration('device'),
                'camera_topic':       LaunchConfiguration('camera_topic'),
            },
        ],
    )

    return LaunchDescription(args + [node])
