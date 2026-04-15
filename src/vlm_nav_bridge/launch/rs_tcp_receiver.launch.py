from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('listen_ip', default_value='0.0.0.0'),
        DeclareLaunchArgument('port', default_value='9999'),
        DeclareLaunchArgument('output_topic', default_value='/camera/image'),
        DeclareLaunchArgument('frame_id', default_value='camera'),
        DeclareLaunchArgument('publish_rate_hz', default_value='30.0'),
        DeclareLaunchArgument('jpeg_decode_mode', default_value='color'),
        DeclareLaunchArgument('reconnect_delay_sec', default_value='1.0'),
    ]

    node = Node(
        package='vlm_nav_bridge',
        executable='rs_tcp_receiver',
        name='rs_tcp_receiver',
        output='screen',
        parameters=[{
            'listen_ip': LaunchConfiguration('listen_ip'),
            'port': LaunchConfiguration('port'),
            'output_topic': LaunchConfiguration('output_topic'),
            'frame_id': LaunchConfiguration('frame_id'),
            'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
            'jpeg_decode_mode': LaunchConfiguration('jpeg_decode_mode'),
            'reconnect_delay_sec': LaunchConfiguration('reconnect_delay_sec'),
        }],
    )

    return LaunchDescription(args + [node])
