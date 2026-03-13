#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR
source ./install/setup.bash

# Disable Unitree WebRTC for non-Unitree platforms (e.g. mechanum_drive)
EXTRA_ARGS=""
if [[ "${ROBOT_CONFIG_PATH}" == "mechanum_drive" ]]; then
  EXTRA_ARGS="launch_unitree_webrtc:=false"
fi

# Forward ALL arguments ($@) directly to the ROS launch file.
ROS_DOMAIN_ID=1 ros2 launch vehicle_simulator system_real_robot.launch $EXTRA_ARGS "$@"  &
sleep 1
ROS_DOMAIN_ID=1 ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz
