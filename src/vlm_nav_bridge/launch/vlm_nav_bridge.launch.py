"""
Launch file for vlm_nav_bridge.

Usage (standalone, after autonomy stack is already running):
  ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py \
      checkpoint:=/abs/path/to/checkpoint

  # Publish goal separately (topic-based, not launch arg):
  ros2 topic pub /object_goal std_msgs/String "data: 'chair'" -1

Or integrated with the real-robot system:
  ./system_real_robot.sh robot_ip:=192.168.1.120 \
      connection_method:=LocalSTA control_mode:=wireless_controller

  # In another terminal after SLAM is up:
  source install/setup.bash
  ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py checkpoint:=...
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vlm_nav_bridge')
    default_config = pkg_share + '/config/vlm_nav_bridge.yaml'

    # ---- Launch arguments -----------------------------------------------
    # Precedence: launch-argument overrides here > values loaded from YAML.
    args = [
        DeclareLaunchArgument(
            'checkpoint',
            default_value='/home/tsaisplus/projects/checkpoints/PosD/checkpoints',
                              description='Absolute path to InternVL checkpoint directory'),
        DeclareLaunchArgument('template',
                              default_value='BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
                              description='VLM template name'),
        DeclareLaunchArgument('device', default_value='cuda:0',
                              description='Torch device for VLM inference'),
        DeclareLaunchArgument('inference_interval', default_value='5.0',
                              description='Seconds between VLM queries'),
        DeclareLaunchArgument('pad2square', default_value='true',
                              description='Use pad2square in VLM preprocessing'),
        DeclareLaunchArgument('normalize_type', default_value='imagenet',
                              description='VLM preprocessing normalization type'),
        DeclareLaunchArgument('target_detection_topic', default_value='/target_detection',
                              description='External detector topic (std_msgs/String JSON)'),
        DeclareLaunchArgument('target_confidence_threshold', default_value='0.30',
                              description='Minimum confidence for target detections'),
        DeclareLaunchArgument('target_require_sam2', default_value='false',
                              description='Only use detections marked as SAM2'),
        DeclareLaunchArgument('config_file', default_value=default_config,
                              description='Path to vlm_nav_bridge.yaml config'),
        DeclareLaunchArgument('bev_only', default_value='false',
                              description='Skip VLM model loading; publish BEV debug only'),
    ]

    # ---- VLM navigator node ---------------------------------------------
    vlm_node = Node(
        package='vlm_nav_bridge',
        executable='vlm_navigator',
        name='vlm_navigator',
        output='screen',
        parameters=[
            # Load defaults from YAML
            LaunchConfiguration('config_file'),
            # Launch arguments override YAML when provided.
            {
                'checkpoint': LaunchConfiguration('checkpoint'),
                'template': LaunchConfiguration('template'),
                'device': LaunchConfiguration('device'),
                'inference_interval': LaunchConfiguration('inference_interval'),
                'pad2square': LaunchConfiguration('pad2square'),
                'normalize_type': LaunchConfiguration('normalize_type'),
                'target_detection_topic': LaunchConfiguration('target_detection_topic'),
                'target_confidence_threshold': LaunchConfiguration('target_confidence_threshold'),
                'target_require_sam2': LaunchConfiguration('target_require_sam2'),
                'bev_only': LaunchConfiguration('bev_only'),
            },
        ],
        # Remappings if topic names differ
        remappings=[
            # Uncomment to remap topics if needed:
            # ('/registered_scan', '/your/lidar_topic'),
            # ('/state_estimation', '/your/odom_topic'),
        ],
    )

    return LaunchDescription(args + [vlm_node])
