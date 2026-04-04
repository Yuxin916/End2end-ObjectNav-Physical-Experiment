#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR

# Hard-pin robot autonomy stack to local DDS defaults on Domain 79.
# This prevents host-side shell exports from leaking into the robot stack.
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=79

source ./install/setup.bash
ros2 launch vehicle_simulator system_real_robot.launch &
sleep 1
ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz
