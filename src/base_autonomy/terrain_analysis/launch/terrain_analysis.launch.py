import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get robot config from environment variable or use default
    robot_config_env = os.environ.get('ROBOT_CONFIG_PATH', 'unitree/unitree_g1')

    # Declare launch arguments
    robot_config_arg = DeclareLaunchArgument(
        'robot_config',
        default_value=robot_config_env,
        description='Robot-specific config file (without .yaml extension)'
    )

    # Base/default parameters (matching the original XML file)
    base_params = {
        'scanVoxelSize': 0.05,
        'decayTime': 1.0,
        'noDecayDis': 1.75,
        'clearingDis': 8.0,
        'useSorting': True,
        'quantileZ': 0.25,
        'considerDrop': False,
        'limitGroundLift': False,
        'maxGroundLift': 0.15,
        'clearDyObs': True,
        'sensorPitch': 0.0,
        'minDyObsDis': 0.14,
        'absDyObsRelZThre': 0.2,
        'minDyObsVFOV': -40.0,
        'maxDyObsVFOV': 10.0,
        'minDyObsPointNum': 1,
        'minOutOfFovPointNum': 10,
        'obstacleHeightThre': 0.1,
        'nearObstacle': True,
        'nearObstacleDis': 2.5,
        'nearObstacleRelZThre': -1.0,
        'negObstacle': -1,
        'negObstacleDis': 10.0,
        'negObstacleRelZThre': -0.2,
        'noDataObstacle': False,
        'noDataBlockShrinkNum': 0,
        'noDataBlockExpandNum': 1,
        'minBlockPointNum': 10,
        'vehicleHeight': 1.5,
        'voxelPointUpdateThre': 100,
        'voxelTimeUpdateThre': 2.0,
        'minRelZ': -1.5,
        'maxRelZ': 0.3,
        'disRatioZ': 0.2,
    }

    terrain_node = Node(
        package='terrain_analysis',          # pkg="terrain_analysis"
        executable='terrainAnalysis',        # exec="terrainAnalysis"
        name='terrainAnalysis',              # name="terrainAnalysis"
        output='screen',                     # output="screen"
        parameters=[
            base_params,
            # Robot-specific overrides from the same config folder as local_planner
            PythonExpression([
                "'", FindPackageShare('local_planner'), "/config/",
                LaunchConfiguration('robot_config'), ".yaml'"
            ]),
        ]
    )

    return LaunchDescription([
        robot_config_arg,
        terrain_node,
    ])
