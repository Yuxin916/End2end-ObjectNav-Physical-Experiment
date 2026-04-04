#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR

# Keep robot stack deterministic:
# - DDS mode: force Domain 79 and clear Cyclone interface pinning.
# - Zenoh mode: do not force ROS_DOMAIN_ID.
unset CYCLONEDDS_URI
if [[ "${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}" == "rmw_zenoh_cpp" ]]; then
  unset ROS_DOMAIN_ID
else
  export ROS_DOMAIN_ID=79
fi

source ./install/setup.bash
ros2 launch vehicle_simulator system_real_robot.launch &
sleep 1
ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz
