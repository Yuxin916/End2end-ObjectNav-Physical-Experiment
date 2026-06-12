#!/bin/bash

export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
#export ROS_LOCALHOST_ONLY=1
#export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR

# WebRTC control needs a venv with unitree_webrtc_connect. In the container it
# is baked at /opt/uvenv (the entrypoint activates it too; this covers shells
# that bypass the entrypoint, e.g. `docker exec`).
if [ -f /opt/uvenv/bin/activate ]; then
  source /opt/uvenv/bin/activate
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

source ./install/setup.bash

# WebRTC control on G1 fw >= 1.5.1 needs UNITREE_AES_KEY from this file.
[ -f "$HOME/.unitree_g1.env" ] && source "$HOME/.unitree_g1.env"

export ROBOT_CONFIG_PATH="unitree/unitree_g1"

# Pass all arguments through to the launch file. Default control backend is
# WebRTC (unitree_webrtc_connect; needs UNITREE_AES_KEY on fw >= 1.5.1). The
# 'sdk' backend is not available in the webrtc container image. Example:
#   ./system_real_robot_g1.sh robot_ip:=192.168.123.161 connection_method:=LocalSTA control_mode:=wireless_controller
ros2 launch vehicle_simulator system_real_robot_g1.launch "$@" &

sleep 1
ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz
