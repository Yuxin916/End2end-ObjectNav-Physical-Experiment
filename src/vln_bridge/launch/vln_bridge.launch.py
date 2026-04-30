"""
Launch file for vln_bridge.

Usage (standalone, after autonomy stack is already running):
    ros2 launch vln_bridge vln_bridge.launch.py

  # Publish instruction separately (topic-based, not launch arg):
  ros2 topic pub /instruction std_msgs/String "data: 'Go to the kitchen and stop near the table.'" -1

Or integrated with the real-robot system:
  ./system_real_robot.sh robot_ip:=192.168.1.120 \
      connection_method:=LocalSTA control_mode:=wireless_controller

  # In another terminal after SLAM is up:
  source install/setup.bash
    ros2 launch vln_bridge vln_bridge.launch.py
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vln_bridge')
    default_config = pkg_share + '/config/vln_bridge.yaml'

    # ---- Launch arguments -----------------------------------------------
    # Precedence: explicit parameter overrides here > values loaded from YAML.
    # Keep the VLM checkpoint path in vln_bridge.yaml so it has one source of truth.
    args = [
        DeclareLaunchArgument('instruction_topic', default_value='/instruction',
                              description='Instruction topic for VLN text commands'),
        DeclareLaunchArgument('template',
                              default_value='RGB_HisKFSingleColor',
                              description='VLM template name'),
        DeclareLaunchArgument('output_template', default_value='pixelintext',
                              description='VLM output template name'),
        DeclareLaunchArgument('device', default_value='cuda:0',
                              description='Torch device for VLM inference'),
        DeclareLaunchArgument('pad2square', default_value='true',
                              description='Use pad2square in VLM preprocessing'),
        DeclareLaunchArgument('normalize_type', default_value='imagenet',
                              description='VLM preprocessing normalization type'),
        DeclareLaunchArgument('vlm_load_delay_sec', default_value='2.0',
                              description='Delay before loading the VLM checkpoint'),
        DeclareLaunchArgument('vlm_auto_infer', default_value='true',
                              description='Automatically run VLM inference on a timer'),
        DeclareLaunchArgument('vlm_inference_interval_sec', default_value='2.0',
                              description='Minimum interval between VLM inference calls'),
        DeclareLaunchArgument('vlm_padding_px', default_value='8',
                              description='Gray padding width for current RGB VLM input'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/depth',
                              description='Depth topic aligned with the camera image'),
        DeclareLaunchArgument('waypoint_frame', default_value='map',
                              description='Frame used for /way_point publication'),
        DeclareLaunchArgument('target_detection_topic', default_value='/target_detection',
                              description='External detector topic (std_msgs/String JSON)'),
        DeclareLaunchArgument('target_confidence_threshold', default_value='0.30',
                              description='Minimum confidence for target detections'),
        DeclareLaunchArgument('target_require_sam2', default_value='false',
                              description='Only use detections marked as SAM2'),
        DeclareLaunchArgument('config_file', default_value=default_config,
                              description='Path to vln_bridge.yaml config'),
    ]

    # ---- VLM navigator node ---------------------------------------------
    vlm_node = Node(
        package='vln_bridge',
        executable='vlm_navigator',
        name='vlm_navigator',
        output='screen',
        parameters=[
            # Load defaults from YAML
            LaunchConfiguration('config_file'),
            {
                'instruction_topic': LaunchConfiguration('instruction_topic'),
                'template': LaunchConfiguration('template'),
                'output_template': LaunchConfiguration('output_template'),
                'device': LaunchConfiguration('device'),
                'pad2square': LaunchConfiguration('pad2square'),
                'normalize_type': LaunchConfiguration('normalize_type'),
                'vlm_load_delay_sec': LaunchConfiguration('vlm_load_delay_sec'),
                'vlm_auto_infer': LaunchConfiguration('vlm_auto_infer'),
                'vlm_inference_interval_sec': LaunchConfiguration('vlm_inference_interval_sec'),
                'vlm_padding_px': LaunchConfiguration('vlm_padding_px'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'waypoint_frame': LaunchConfiguration('waypoint_frame'),
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
    )

    return LaunchDescription(args + [vlm_node])
