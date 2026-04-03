import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    local_planner_share = get_package_share_directory('local_planner')

    # Robot config from env or default
    robot_config_env = os.environ.get('ROBOT_CONFIG_PATH', 'unitree/unitree_go2_slow')

    # Launch arguments
    robot_config_arg = DeclareLaunchArgument(
        'robot_config',
        default_value=robot_config_env,
        description='Robot-specific config file (without .yaml extension)'
    )

    realRobot_arg = DeclareLaunchArgument(
        'realRobot',
        default_value='false'
    )

    twoWayDrive_arg = DeclareLaunchArgument(
        'twoWayDrive',
        default_value='false'
    )

    autonomyMode_arg = DeclareLaunchArgument(
        'autonomyMode',
        default_value='false'
    )

    joyToSpeedDelay_arg = DeclareLaunchArgument(
        'joyToSpeedDelay',
        default_value='2.0'
    )

    goalX_arg = DeclareLaunchArgument(
        'goalX',
        default_value='0.0'
    )

    goalY_arg = DeclareLaunchArgument(
        'goalY',
        default_value='0.0'
    )

    cameraOffsetZ_arg = DeclareLaunchArgument(
        'cameraOffsetZ',
        default_value='0.0'
    )

    # Read sensor offsets from robot config YAML for TF publisher
    sensor_offsets = {
        'sensorOffsetX': 0.0,
        'sensorOffsetY': 0.0,
        'sensorOffsetZ': 0.0
    }

    try:
        robot_config_path = os.path.join(
            local_planner_share, 'config', robot_config_env + '.yaml'
        )
        with open(robot_config_path, 'r') as file:
            config_data = yaml.safe_load(file)

        if (
            'sensorMountingOffsets' in config_data and
            'ros__parameters' in config_data['sensorMountingOffsets']
        ):
            mounting_offsets = config_data['sensorMountingOffsets']['ros__parameters']
            for key in sensor_offsets.keys():
                if key in mounting_offsets:
                    sensor_offsets[key] = mounting_offsets[key]
    except Exception as e:
        print(
            f"Warning: Could not read robot config from {robot_config_env}.yaml, "
            f"using defaults: {e}"
        )

    robot_config_file = os.path.join(
        local_planner_share, 'config', robot_config_env + '.yaml'
    )

    # XML-compatible defaults for localPlanner
    local_planner_defaults = {
        'pathFolder': os.path.join(local_planner_share, 'paths'),
        'vehicleLength': 0.5,
        'vehicleWidth': 0.5,
        'sensorOffsetX': 0.0,
        'sensorOffsetY': 0.0,
        'twoWayDrive': True,
        'laserVoxelSize': 0.05,
        'terrainVoxelSize': 0.2,
        'useTerrainAnalysis': True,
        'checkObstacle': True,
        'checkRotObstacle': False,
        'adjacentRange': 3.5,
        'obstacleHeightThre': 0.05,
        'groundHeightThre': 0.05,
        'costHeightThre1': 0.1,
        'costHeightThre2': 0.05,
        'useCost': False,
        'slowPathNumThre': 5,
        'slowGroupNumThre': 1,
        'pointPerPathThre': 2,
        'minRelZ': -0.4,
        'maxRelZ': 0.3,
        'maxSpeed': 0.875,
        'dirWeight': 0.02,
        'dirThre': 90.0,
        'dirToVehicle': False,
        'pathScale': 0.875,
        'minPathScale': 0.675,
        'pathScaleStep': 0.1,
        'pathScaleBySpeed': True,
        'minPathRange': 0.8,
        'pathRangeStep': 0.6,
        'pathRangeBySpeed': True,
        'pathCropByGoal': True,
        'autonomyMode': LaunchConfiguration('autonomyMode'),
        'autonomySpeed': 0.875,
        'joyToSpeedDelay': LaunchConfiguration('joyToSpeedDelay'),
        'joyToCheckObstacleDelay': 5.0,
        'goalClearRange': 0.35,
        'goalBehindRange': 0.35,
        'freezeAng': 90.0,
        'freezeTime': 0.0,
        'goalX': LaunchConfiguration('goalX'),
        'goalY': LaunchConfiguration('goalY'),
    }

    # XML-compatible defaults for pathFollower
    path_follower_defaults = {
        'realRobot': LaunchConfiguration('realRobot'),
        'serialPort': '/dev/ttyACM0',
        'baudrate': 115200,
        'sensorOffsetX': 0.0,
        'sensorOffsetY': 0.0,
        'pubSkipNum': 1,
        'twoWayDrive': LaunchConfiguration('twoWayDrive'),
        'lookAheadDis': 0.5,
        'maxSpeed': 0.875,
        'maxAccel': 2.0,
        'switchTimeThre': 1.0,
        'omniDirDiffThre': 1.5,
        'slowDwnDisThre': 0.875,
        'useInclRateToSlow': False,
        'inclRateThre': 120.0,
        'slowRate1': 0.25,
        'slowRate2': 0.5,
        'slowRate3': 0.75,
        'slowTime1': 2.0,
        'slowTime2': 2.0,
        'useInclToStop': False,
        'inclThre': 45.0,
        'stopTime': 5.0,
        'noRotAtStop': False,
        'noRotAtGoal': True,
        'autonomyMode': LaunchConfiguration('autonomyMode'),
        'autonomySpeed': 0.875,
        'joyToSpeedDelay': LaunchConfiguration('joyToSpeedDelay'),
    }

    localPlanner_node = Node(
        package='local_planner',
        executable='localPlanner',
        name='localPlanner',
        output='screen',
        parameters=[
            local_planner_defaults,
            robot_config_file,
        ]
    )

    pathFollower_node = Node(
        package='local_planner',
        executable='pathFollower',
        name='pathFollower',
        output='screen',
        parameters=[
            path_follower_defaults,
            robot_config_file,
        ]
    )

    vehicleTransPublisher_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='vehicleTransPublisher',
        arguments=[
            str(-sensor_offsets['sensorOffsetX']),
            str(-sensor_offsets['sensorOffsetY']),
            str(-sensor_offsets['sensorOffsetZ']),
            '0', '0', '0',
            '/sensor',
            '/vehicle'
        ]
    )

    sensorTransPublisher_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='sensorTransPublisher',
        arguments=[
            '0', '0', LaunchConfiguration('cameraOffsetZ'),
            '-1.5707963', '0', '-1.5707963',
            '/sensor', '/camera'
        ]
    )

    return LaunchDescription([
        robot_config_arg,
        realRobot_arg,
        twoWayDrive_arg,
        autonomyMode_arg,
        joyToSpeedDelay_arg,
        goalX_arg,
        goalY_arg,
        cameraOffsetZ_arg,
        localPlanner_node,
        pathFollower_node,
        vehicleTransPublisher_node,
        sensorTransPublisher_node,
    ])