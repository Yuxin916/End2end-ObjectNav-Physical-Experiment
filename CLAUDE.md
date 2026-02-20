# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a ROS 2 Jazzy autonomy stack supporting multiple robot platforms (Unitree Go2, G1, B1, and Mecanum wheel platforms), containing:
- SLAM module with Mid-360 lidar support
- Base autonomy system (terrain analysis, collision avoidance, waypoint following)
- Route planner (FAR Planner based)
- Exploration planner (TARE Planner based)
- Unity simulation environment support

Current active branch: `unitree_g1`. Main branch: `jazzy`.

## Essential Build Commands

### Full Build (from repo root)
```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Simulation Build (skips SLAM and Mid-360 driver)
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

## System Launch Commands

### Simulation
- Base autonomy: `./system_simulation.sh`
- With route planner: `./system_simulation_with_route_planner.sh`
- With exploration planner: `./system_simulation_with_exploration_planner.sh`

### Real Robot
Set `ROBOT_CONFIG_PATH` before launching:
```bash
export ROBOT_CONFIG_PATH="unitree/unitree_g1"        # for G1
export ROBOT_CONFIG_PATH="unitree/unitree_go2_slow"  # for Go2 slow
export ROBOT_CONFIG_PATH="unitree/unitree_go2_fast"  # for Go2 fast
source install/setup.bash
```

Launch with robot-specific arguments:
```bash
# Go2 (LocalAP WiFi, wireless controller)
./system_real_robot.sh robot_ip:=192.168.12.1 connection_method:=LocalAP control_mode:=wireless_controller

# Go2 (LocalAP WiFi, sport_cmd control)
./system_real_robot.sh robot_ip:=192.168.12.1 connection_method:=LocalAP control_mode:=sport_cmd

# G1 (LocalSTA WiFi — must be in low-speed mode with advanced motion control enabled)
./system_real_robot.sh robot_ip:=192.168.1.120 connection_method:=LocalSTA control_mode:=wireless_controller
```

- With route planner: `./system_real_robot_with_route_planner.sh [args]`
- With exploration planner: `./system_real_robot_with_exploration_planner.sh [args]`

### Bagfile Processing
- Base autonomy: `./system_bagfile.sh`
- With route planner: `./system_bagfile_with_route_planner.sh`
- With exploration planner: `./system_bagfile_with_exploration_planner.sh`

### Kill All ROS Processes (if needed before launch)
```bash
sudo pkill -9 -f '/opt/ros/.*\/lib\/|\/install\/.*\/lib\/|_ros2_daemon|^[[:space:]]*ros2$'
```

## Code Architecture

### Package Structure
- **src/base_autonomy/**: Core navigation modules
  - `local_planner`: Main control and collision avoidance
  - `terrain_analysis` & `terrain_analysis_ext`: Terrain processing
  - `sensor_scan_generation`: Sensor data preprocessing
  - `vehicle_simulator`: Unity integration and simulation launch
  - `visualization_tools`: RVIZ visualization utilities

- **src/slam/**: SLAM implementation
  - `arise_slam_mid360`: Main SLAM node for Mid-360 lidar
  - `arise_slam_mid360_msgs`: Custom message definitions
  - `dependency/`: External libraries (ceres-solver, gtsam, Sophus)

- **src/exploration_planner/**: TARE planner for autonomous exploration
  - `tare_planner`: Main exploration planning module

- **src/route_planner/**: FAR planner for goal navigation
  - `far_planner`: Global route planning module
  - `boundary_handler`: Navigation boundary enforcement
  - `graph_decoder`: Visibility graph decoding
  - `visibility_graph_msg`: Custom graph message definitions

- **src/utilities/**: Support packages
  - `livox_ros_driver2`: Mid-360 lidar driver
  - `teleop_*`: Teleoperation and control interfaces
  - `domain_bridge`: Network communication for base station
  - `unitree_webrtc_ros`: Unitree robot wireless WebRTC control (Go2, G1, B1)
  - ROS plugins for RVIZ buttons and controls

### Key Configuration Files

**Robot Configuration** (`src/base_autonomy/local_planner/config/unitree/`):
- `unitree_g1.yaml`: Unitree G1 humanoid
- `unitree_go2.yaml` / `unitree_go2_slow.yaml` / `unitree_go2_fast.yaml`: Unitree Go2 quadruped variants
- `unitree_b1.yaml`: Unitree B1 quadruped
- `mechanum_drive.yaml`: Custom Mecanum platform
- Selected via `ROBOT_CONFIG_PATH` environment variable

**Local Planner** (`src/base_autonomy/local_planner/launch/local_planner.launch`):
- `maxSpeed`: Maximum vehicle speed in all modes
- `autonomySpeed`: Speed in waypoint mode
- `obstacleHeightThre`: Obstacle detection threshold
- `config`: "omniDir" for Mecanum wheels, "standard" for regular wheels

**SLAM** (`src/slam/arise_slam_mid360/config/livox_mid360.yaml`):
- `local_mode`: Enable localization mode with saved map
- `blindFront/Back/Left/Right`: Exclude regions from lidar FOV
- `init_x/y/z/yaw`: Initial pose for localization

**Mid-360 Lidar** (`src/utilities/livox_ros_driver2/config/MID360_config.json`):
- IP configuration (192.168.1.1xx where xx = last 2 digits of serial)

### Message Flow

1. **Sensor Input**:
   - Lidar scan → `/lidar/scan` (custom format)
   - IMU data → `/imu/data`

2. **SLAM Output**:
   - Vehicle pose → `/state_estimation`
   - Registered scan → `/registered_scan`

3. **Navigation**:
   - Waypoints → `/way_point`
   - Joystick commands → `/joy_cmd`
   - Navigation boundary → `/navigation_boundary`

4. **Control Output**:
   - Vehicle commands → `/cmd_vel` (to Unitree WebRTC or motor controller via serial)

### Operating Modes

1. **Smart Joystick Mode** (default): Follows joystick with collision avoidance
2. **Waypoint Mode**: Autonomous navigation to waypoints
3. **Manual Mode**: Direct joystick control without collision avoidance

Mode switching via:
- RVIZ control panel and waypoint button
- PS3/4 or Xbox controller buttons
- ROS topic commands

## VLM Object Navigation Bridge (Primary Research Objective)

The main research goal is to deploy a fine-tuned **InternVL object-navigation model** (from
`~/projects/VLN_CL_CoTNav`) on the Unitree G1 for real-world end-to-end object navigation.

### System Overview

```
[Livox Mid-360 Lidar]       [SLAM]
  /registered_scan   ──→  /state_estimation
         │                      │
         └────────┬─────────────┘
                  ▼
          [vlm_nav_bridge]          ← src/vlm_nav_bridge/
          LidarBEVMapper            builds 448×448 BEV occupancy map
          FrontierDetector          finds explored-boundary frontiers
          VLMInterface              runs InternVL inference
                  │
                  ▼/context
           /way_point  (PointStamped, map frame)
                  │
          [local_planner]           collision-aware path following
                  │
          /cmd_vel → Unitree G1 (WebRTC)
```

### VLM Bridge Package: `src/vlm_nav_bridge/`

**VLM model repo**: `~/projects/VLN_CL_CoTNav`

**Template**: `BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2`
- Input: 1 BEV image (448×448 RGB) + `position_info` dict (robot pixel pos, yaw, frontier pixels)
- Output: integer frontier index → converted to world coordinate → published as `/way_point`
- No RGB camera or semantic segmentation image required for this template

**BEV map parameters** (must match VLN training exactly):
- Resolution: 0.05 m/cell | Vision range: 100 cells (5 m) | Global map: 67.2 m × 67.2 m
- Crop radius: 150 cells | Output: 448 × 448 pixels
- Lidar range filter: [0.5 m, 5.0 m] | Height filter: [+0.1 m, +1.5 m] relative to robot z

**Key topics**:
| Topic | Type | Role |
|-------|------|------|
| `/registered_scan` | PointCloud2 (map frame) | Lidar input to BEV mapper |
| `/state_estimation` | nav_msgs/Odometry | Robot pose (SLAM output) |
| `/object_goal` | std_msgs/String | Target object (e.g. `"chair"`) — publish at runtime |
| `/way_point` | geometry_msgs/PointStamped | Bridge output to local_planner |
| `/vlm_bev_debug` | sensor_msgs/Image | BEV visualisation in RVIZ |

**Module files**:
- `vlm_nav_bridge/lidar_bev_mapper.py` — accumulates lidar into BEV grid, renders image
- `vlm_nav_bridge/frontier_detector.py` — explored-boundary frontier extraction
- `vlm_nav_bridge/vlm_interface.py` — loads VLMPolicy from VLN_CL_CoTNav via PYTHONPATH injection
- `vlm_nav_bridge/coord_utils.py` — BEV pixel ↔ ROS map frame conversion
- `vlm_nav_bridge/vlm_navigator_node.py` — ROS 2 orchestrator node
- `config/vlm_nav_bridge.yaml` — all tunable parameters
- `launch/vlm_nav_bridge.launch.py` — sets PYTHONPATH for VLN imports, launches node

### Full G1 Deployment Sequence

```bash
# 1. Launch autonomy stack
export ROBOT_CONFIG_PATH="unitree/unitree_g1"
source install/setup.bash
./system_real_robot.sh robot_ip:=192.168.1.120 connection_method:=LocalSTA control_mode:=wireless_controller

# 2. In a new terminal: launch the VLM bridge
source install/setup.bash
ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py

# 3. Set the navigation goal at runtime
ros2 topic pub /object_goal std_msgs/String "data: 'chair'" -1
```

### Future Work (Tracked)
- **FrontierRGB template** (`BEVftFOV_FrontierRGB_Pos*`, in a separate VLN branch): stores an RGB
  image captured the first time each frontier is born. `LidarBEVMapper` should add a
  `frontier_birth_rgb` dict keyed by frontier cluster ID when implementing this.
- **Target detection**: subscribe to camera topic + run object detector to populate
  `position_info["target_position"]` when the target object is visible.
- **Explored-region ray casting**: current implementation uses a disk approximation; upgrade to
  Bresenham ray casting to avoid marking cells through walls as explored.
- **BEV orientation validation**: verify BEV row↓ = map −Y direction matches Habitat training
  convention when testing with the real robot.

---

## Testing

No formal test framework configured. System testing done through:
- Simulation with Unity environment models
- Bagfile replay for offline testing
- Manual teleoperation testing with `teleop_joy_controller`
- VLM bridge: visualise `/vlm_bev_debug` in RVIZ; monitor `/way_point` output

## Development Notes

- Unity environment files must be placed in `src/base_autonomy/vehicle_simulator/mesh/unity/`
- Always source workspace before running: `source install/setup.bash`
- For ARM platforms, replace OR-Tools binaries in exploration planner
- System uses custom scan message format — must source workspace for bagfile operations
- X11 required for RVIZ on Ubuntu 24.04 (disable Wayland if needed)
- **Unitree G1 specific**: Robot must be in low-speed mode and advanced motion control (including arms) must be enabled before launch
- **Go2 connection methods**: `LocalAP` (robot creates hotspot, 192.168.12.1) or `LocalSTA` (robot joins existing WiFi)
- **Go2 control modes**: `wireless_controller` (joystick via WebRTC) or `sport_cmd` (ROS velocity commands via WebRTC)
