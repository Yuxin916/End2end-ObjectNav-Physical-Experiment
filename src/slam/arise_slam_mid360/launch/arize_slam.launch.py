import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import launch_ros

def get_share_file(package_name, file_name):
    return os.path.join(get_package_share_directory(package_name), file_name)

def generate_launch_description():
    # Get robot config from environment variable or use default
    robot_config_env = os.environ.get('ROBOT_CONFIG_PATH', 'mechanum_drive')

    config_path = get_share_file(
        package_name="arise_slam_mid360",
        file_name="config/livox_mid360.yaml")
    calib_path = get_share_file(
        package_name="arise_slam_mid360",
        file_name="config/livox/livox_mid360_calibration.yaml"
    )
    home_directory = os.path.expanduser("~")
    
    config_path_arg = DeclareLaunchArgument(
        "config_file",
        default_value=config_path,
        description="Path to config file for arise_slam"
    )

    robot_config_arg = DeclareLaunchArgument(
        'robot_config',
        default_value=robot_config_env,
        description='Robot configuration file name (without .yaml)'
    )
    calib_path_arg = DeclareLaunchArgument(
        "calibration_file",
        default_value=calib_path,
    )
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="integrated_to_init"
    )
    world_frame_arg = DeclareLaunchArgument(
        "world_frame",
        default_value="map",
    )
    world_frame_rot_arg = DeclareLaunchArgument(
        "world_frame_rot",
        default_value="map_rot",
    )
    sensor_frame_arg = DeclareLaunchArgument(
        "sensor_frame",
        default_value="sensor",
    )
    sensor_frame_rot_arg = DeclareLaunchArgument(
        "sensor_frame_rot",
        default_value="sensor_rot",
    )
    # Relocalization against a pre-scanned map: local_mode=true loads map_dir
    # (text pointcloud, 'x y z intensity time' per line) as the prior map and
    # starts from init pose in that map's frame instead of building a fresh
    # map at the boot pose. Record the map with savePcd or
    # scripts/save_prior_map.py.
    local_mode_arg = DeclareLaunchArgument(
        "local_mode",
        default_value="false",
    )
    map_dir_arg = DeclareLaunchArgument(
        "map_dir",
        default_value=os.path.join(home_directory, "Desktop/pointcloud_local.txt"),
    )

    feature_extraction_node = Node(
        package="arise_slam_mid360",
        executable="feature_extraction_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[
            LaunchConfiguration("config_file"),
            PythonExpression([
                "'", FindPackageShare('local_planner'), "/config/",
                LaunchConfiguration('robot_config'), ".yaml'"
            ]),
            { "calibration_file": LaunchConfiguration("calibration_file"),
        }],
    )

    laser_mapping_node = Node(
        package="arise_slam_mid360",
        executable="laser_mapping_node",
        output={
            "stdout": "screen",
            #"stderr": "screen",
        },
        parameters=[LaunchConfiguration("config_file"),
            { "calibration_file": LaunchConfiguration("calibration_file"),
             "map_dir": LaunchConfiguration("map_dir"),
             "local_mode": ParameterValue(LaunchConfiguration("local_mode"), value_type=bool),
        }],
        remappings=[
            ("laser_odom_to_init", LaunchConfiguration("odom_topic")),
        ]
    )

    imu_preintegration_node = Node(
        package="arise_slam_mid360",
        executable="imu_preintegration_node",
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        parameters=[
            LaunchConfiguration("config_file"),
            PythonExpression([
                "'", FindPackageShare('local_planner'), "/config/",
                LaunchConfiguration('robot_config'), ".yaml'"
            ]),
            { "calibration_file": LaunchConfiguration("calibration_file")
        }],
    )

    
    return LaunchDescription([
        launch_ros.actions.SetParameter(name='use_sim_time', value='false'),
        config_path_arg,
        robot_config_arg,
        calib_path_arg,
        odom_topic_arg,
        world_frame_arg,
        world_frame_rot_arg,
        sensor_frame_arg,
        sensor_frame_rot_arg,
        local_mode_arg,
        map_dir_arg,
        feature_extraction_node,
        laser_mapping_node,
        imu_preintegration_node,
    ])