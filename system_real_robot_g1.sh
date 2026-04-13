#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR

# Activate venv so unitree_webrtc_connect is importable by launched nodes
source .venv/bin/activate

source ./install/setup.bash

export ROBOT_CONFIG_PATH="unitree/unitree_g1"

# Pass all arguments through to the launch file, e.g.:
#   ./system_real_robot_g1.sh robot_ip:=192.168.123.161 connection_method:=LocalSTA control_mode:=wireless_controller
ros2 launch vehicle_simulator system_real_robot_g1.launch "$@" &

sleep 1
ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz
