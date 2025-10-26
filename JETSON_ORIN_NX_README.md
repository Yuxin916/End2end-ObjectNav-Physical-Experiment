
## Repository Overview

This is a ROS 2 Humble autonomy stack for Unitree Go2 and G1, containing:
- SLAM module with Mid-360 lidar support
- Base autonomy system (terrain analysis, collision avoidance, waypoint following)
- Route planner (FAR Planner based)
- Exploration planner (TARE Planner based)
- Unity simulation environment support


## Essential Build Commands

### Full Build
```bash
cd autonomy_stack_mecanum_wheel_platform
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Skip Build (Simulation only on x86)
```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-skip arise_slam_mid360 arise_slam_mid360_msgs livox_ros_driver2
```

### Single Package Build
```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select [package_name]
```

### Clean Build
```bash
rm -rf build install log
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Find a package 
```bash
sudo find /opt ~/ -type f -name "cv_bridgeConfig.cmake" 2>/dev/null | head
```

### If OOM: c++: fatal error: Killed signal terminated program cc1plus compilation terminated. 
```bash
# one compile at a time (both at the package and translation-unit level)
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS="-j1"

colcon build --packages-select xxxxx \
  --event-handlers console_cohesion+ \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### if Compatibility with CMake < 3.5 has been removed from CMake.
```bash
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

### cv_bridge missing
```bash
cd src

git clone -b humble https://github.com/ros-perception/vision_opencv.git

cd ..
rosdep update
rosdep install --from-paths src --ignore-src -r -y

colcon build --packages-select cv_bridge image_geometry \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

colcon build --packages-select arise_slam_mid360 \
  --cmake-clean-cache --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Env Variable 
```bash
export ROBOT_CONFIG_PATH="unitree/unitree_go2"
source install/setup.bash
```

## Packages Run
```bash
# one terminal
./system_real_robot.sh
```

## How to test the vel_to_sport_bridge → udp_to_sport_client → Go2 SDK?
```bash
# Terminal A
export DISABLE_SDK=1                 # DRY mode: prints, no motion
export VEL_DEADMAN_MS=2000
export VEL_CTRL_HZ=100
export VEL_LIMITS=0.3,0.0,0.6        # gentle: vx,vy,wz (vy=0 to avoid lateral)
ros2 run unitree_sport_tools udp_to_sport_client eth0

# You should see:
[udp_to_sport_client] listening on 0.0.0.0:50051 (binary) [DISABLE_SDK=1] [NIC=eth0]

```

```bash
# Terminal B
ros2 run unitree_cmd_bridge vel_to_sport_bridge \
  --ros-args \
  -p cmd_vel_topic:=/cmd_vel \
  -p target_ip:=127.0.0.1 \
  -p target_port:=50051 \
  -p use_csv:=false \
  -p qos_depth:=10

# You should see:
[vel_to_sport_bridge]: Forwarding '/cmd_vel' -> UDP 127.0.0.1:50051 (binary payload)

```

```bash
# Terminal C
ros2 topic pub -r 20 --times 10 /cmd_vel geometry_msgs/msg/TwistStamped "{header: {frame_id: base_link}, twist: {linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}"

# You should see Terminal A prints ~2 lines per second (DRY):
[udp_to_sport_client] (DRY) vx=0.300 vy=0.000 wz=0.000
and robot does not move.
```

```
unset DISABLE_SDK                  # enable motion
export VEL_DEADMAN_MS=2000
export VEL_CTRL_HZ=100
export VEL_LIMITS=0.3,0.0,0.6
ros2 run unitree_sport_tools udp_to_sport_client eth0
```

## Ethernet Network Configuration

### Built-in Go2 robot computer 

1. Go to IPv4 tab, Change Method from "Automatic (DHCP)" to "Manual", 
2. Add the following settings:
   - **Address**: 192.168.123.51
   - **Netmask**: 255.255.255.0
3. Therefore, can communicate with 192.168.123.161
4. ping it to double check


### MID360

1. Go to IPv4 tab, Change Method from "Automatic (DHCP)" to "Manual", 
2. Add the following settings:
   - **Address**: 192.168.1.5
   - **Netmask**: 255.255.255.0
   - **Gateway**: 192.168.1.1
3. Therefore, can communicate with 192.168.1.1xx (last 2 digits)
4. ping it to double check
5. edit [MID360_config.json](src/utilities/livox_ros_driver2/config/MID360_config.json), line 28:
    ```json
    "ip": "192.168.1.1xx",
    ```


## SLAM Configuration Modification 

### Localization Mode
Set in [livox_mid360.yaml](src/slam/arise_slam_mid360/config/livox_mid360.yaml):
```yaml
local_mode: true
init_x: 0.0
init_y: 0.0
init_yaw: 0.0
```
### Mapping Performance
```yaml
mapping_line_resolution: 0.1   # Decrease for higher quality
mapping_plane_resolution: 0.2  # Decrease for higher quality
max_iterations: 5               # Increase for better accuracy

mapping_skip_frame: 2  # 5Hz
```


## Kill all ROS2 Process
```
sudo pkill -9 -f '/home/tsaisplus/projects/autonomy_stack_mecanum_wheel_platform/install/.*\/lib\/|_ros2_daemon|^[[:space:]]*ros2$'
```

## Questions
1. What is 


## Code Architecture
### Package Structure
- **src/base_autonomy/**: Core navigation modules
  - `local_planner`: Main control and collision avoidance
    - dd
  - `terrain_analysis` & `terrain_analysis_ext`: Terrain processing
    - dd
  - `sensor_scan_generation`: Sensor data preprocessing
    - dd
  - `vehicle_simulator`: Unity integration and simulation launch
  - `visualization_tools`: RVIZ visualization utilities
    - dd


- **src/slam/**: SLAM implementation
  - `arise_slam_mid360`: Main SLAM node for Mid-360 lidar
    - dd
  - `arise_slam_mid360_msgs`: Custom message definitions
  - `dependency/`: External libraries (ceres-solver, gtsam, Sophus)

- **src/exploration_planner/**: TARE planner for autonomous exploration
  - `tare_planner`: Main exploration planning module

- **src/route_planner/**: FAR planner for goal navigation
  - `far_planner`: Global route planning module

- **src/utilities/**: Support packages
  - `livox_ros_driver2`: Mid-360 lidar driver
  - `teleop_*`: Teleoperation and control interfaces
  - `domain_bridge`: Network communication for base station
  - ROS plugins for RVIZ buttons and controls

## Useful for debugging 
1. 
