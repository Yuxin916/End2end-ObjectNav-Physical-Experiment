#!/bin/bash
# Run ONCE inside the container to build the ROS2 workspace into the
# host-mounted install/. Safe: compiles only, does not launch any node.
set -e
cd /workspace/autonomy_stack

source /opt/ros/jazzy/setup.bash

# Skip GPU/AI packages (need torch). Everything the G1 real-robot launch needs
# (base autonomy, SLAM, livox, sdk bridge) is built.
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --packages-skip sam2_detector vlm_nav_bridge

echo
echo "==> Workspace build finished. install/ is on the host."
