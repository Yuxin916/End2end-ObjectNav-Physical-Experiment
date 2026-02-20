"""
Launch file for vlm_nav_bridge.

Usage (standalone, after autonomy stack is already running):
  ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py \
      checkpoint:=/abs/path/to/checkpoint \
      object_goal:=chair

Or integrated with the real-robot system:
  ./system_real_robot.sh robot_ip:=192.168.1.120 \
      connection_method:=LocalSTA control_mode:=wireless_controller

  # In another terminal after SLAM is up:
  source install/setup.bash
  ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py checkpoint:=...
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vlm_nav_bridge')
    default_config = os.path.join(pkg_share, 'config', 'vlm_nav_bridge.yaml')

    vln_repo = '/home/tsaisplus/projects/VLN_CL_CoTNav'

    # Build PYTHONPATH for the VLN codebase
    vln_pythonpath = ':'.join([
        vln_repo,
        os.path.join(vln_repo, 'InternVL_cleaned', 'internvl_chat'),
        os.path.join(vln_repo, 'scripts'),
        os.path.join(vln_repo, 'habitat-lab', 'habitat-baselines'),
        os.path.join(vln_repo, 'habitat-lab', 'habitat-lab'),
    ])

    # Prepend to existing PYTHONPATH
    existing = os.environ.get('PYTHONPATH', '')
    full_pythonpath = vln_pythonpath + (':' + existing if existing else '')

    # ---- Launch arguments -----------------------------------------------
    args = [
        DeclareLaunchArgument(
            'checkpoint',
            default_value='/home/tsaisplus/projects/VLN_CL_CoTNav/all_log/experiments/mp3d_70k_few_S28_a6000_dualvit_bev-full_ego-lora32_mlp-train-patch-24-acc2_BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2_train/checkpoints',
                              description='Absolute path to InternVL checkpoint directory'),
        DeclareLaunchArgument('template',
                              default_value='BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
                              description='VLM template name'),
        DeclareLaunchArgument('device', default_value='cuda:0',
                              description='Torch device for VLM inference'),
        DeclareLaunchArgument('inference_interval', default_value='5.0',
                              description='Seconds between VLM queries'),
        DeclareLaunchArgument('vln_repo_path', default_value=vln_repo,
                              description='Path to VLN_CL_CoTNav repository'),
        DeclareLaunchArgument('target_detection_topic', default_value='/target_detection',
                              description='External detector topic (std_msgs/String JSON)'),
        DeclareLaunchArgument('target_confidence_threshold', default_value='0.30',
                              description='Minimum confidence for target detections'),
        DeclareLaunchArgument('target_require_sam2', default_value='true',
                              description='Only use detections marked as SAM2'),
        DeclareLaunchArgument('config_file', default_value=default_config,
                              description='Path to vlm_nav_bridge.yaml config'),
    ]

    # ---- Set PYTHONPATH so VLN imports work inside the node process ------
    env_action = SetEnvironmentVariable('PYTHONPATH', full_pythonpath)

    # ---- VLM navigator node ---------------------------------------------
    vlm_node = Node(
        package='vlm_nav_bridge',
        executable='vlm_navigator',
        name='vlm_navigator',
        output='screen',
        parameters=[
            # Load defaults from YAML
            LaunchConfiguration('config_file'),
            # Override select params via launch arguments.
            {
                'checkpoint': LaunchConfiguration('checkpoint'),
                'template': LaunchConfiguration('template'),
                'device': LaunchConfiguration('device'),
                'inference_interval': LaunchConfiguration('inference_interval'),
                'vln_repo_path': LaunchConfiguration('vln_repo_path'),
                'target_detection_topic': LaunchConfiguration('target_detection_topic'),
                'target_confidence_threshold': LaunchConfiguration('target_confidence_threshold'),
                'target_require_sam2': LaunchConfiguration('target_require_sam2'),
            },
        ],
        # Remappings if topic names differ
        remappings=[
            # Uncomment to remap topics if needed:
            # ('/registered_scan', '/your/lidar_topic'),
            # ('/state_estimation', '/your/odom_topic'),
        ],
        additional_env={'PYTHONPATH': full_pythonpath},
    )

    return LaunchDescription(args + [env_action, vlm_node])
