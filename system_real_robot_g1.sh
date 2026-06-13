#!/bin/bash

# Isolate the ROS2 stack from the robot's native DDS (it floods domain 0 with
# ~100 participants -> Fast DDS discovery congestion drops IMU/lidar frames ->
# SLAM drift). Move to domain 1 + cyclonedds RMW. (Borrowed from will_nx.)
# Also baked as ENV in docker/Dockerfile.sdk so `docker exec` shells match.
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR

# Legacy WebRTC backend needs a venv with unitree_webrtc_connect. The default
# SDK backend (unitree_sdk2 over DDS) does not -- so only activate if present.
[ -f .venv/bin/activate ] && source .venv/bin/activate

source ./install/setup.bash

# Only needed by the legacy WebRTC backend on G1 fw >= 1.5.1. Harmless otherwise.
[ -f "$HOME/.unitree_g1.env" ] && source "$HOME/.unitree_g1.env"

export ROBOT_CONFIG_PATH="unitree/unitree_g1"

# Pass all arguments through to the launch file. Default control backend is the
# unitree_sdk2 DDS bridge (no AES key). To use the legacy WebRTC path, append
# 'control_backend:=webrtc'. Example:
#   ./system_real_robot_g1.sh robot_ip:=192.168.123.161 connection_method:=LocalSTA control_mode:=wireless_controller
ros2 launch vehicle_simulator system_real_robot_g1.launch "$@" &

sleep 1
ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz
