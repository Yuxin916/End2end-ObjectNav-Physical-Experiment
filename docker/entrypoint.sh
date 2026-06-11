#!/bin/bash
# Sources ROS + the workspace overlay, then execs the command.
# Robot control uses unitree_sdk2 over DDS (no venv / WebRTC / AES key).
set -e

source /opt/ros/jazzy/setup.bash

if [ -f /workspace/autonomy_stack/install/setup.bash ]; then
  source /workspace/autonomy_stack/install/setup.bash
fi

cd /workspace/autonomy_stack 2>/dev/null || cd /workspace
exec "$@"
