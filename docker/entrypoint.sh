#!/bin/bash
# Activates the WebRTC venv, sources ROS + the workspace overlay, then execs
# the command. Robot control uses unitree_webrtc_connect (WebRTC); on G1
# fw >= 1.5.1 it needs UNITREE_AES_KEY (from ~/.unitree_g1.env).
set -e

if [ -f /opt/uvenv/bin/activate ]; then
  source /opt/uvenv/bin/activate
fi

source /opt/ros/jazzy/setup.bash

if [ -f /workspace/autonomy_stack/install/setup.bash ]; then
  source /workspace/autonomy_stack/install/setup.bash
fi

if [ -f /root/.unitree_g1.env ]; then
  source /root/.unitree_g1.env
fi

cd /workspace/autonomy_stack 2>/dev/null || cd /workspace
exec "$@"
