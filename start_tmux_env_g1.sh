#!/usr/bin/env zsh
# G1 humanoid — single-machine setup (base autonomy + VLM on same host).
# Pane 0: base nav stack (SLAM + local planner + Livox + G1 WebRTC)
# Pane 1: VLM navigation bridge
# Pane 2: SAM2/YOLOE object detector
# Pane 3: RealSense camera driver
# Pane 4-5: free terminals
set -e

SESSION_NAME="sagan_nav_g1"
WORKDIR="./"

G1_IP="${G1_IP:-192.168.123.161}"
CONNECTION_METHOD="${CONNECTION_METHOD:-LocalSTA}"
CONTROL_MODE="${CONTROL_MODE:-wireless_controller}"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# cleanup debug images
echo "Removing ./debug_images ..."
rm -rf "./debug_images"

# create new detached session with 6 panes (2x3 tiled)
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

for i in $(seq 1 5); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

# Activate venv so unitree_webrtc_connect is available, then source ROS workspace
FULL_SETUP="cd \"$WORKDIR\" && source .venv/bin/activate"

for pane in 0 1 2 3 4 5; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$FULL_SETUP" C-m
done

sleep 5

# Pane 0: G1 base autonomy stack (SLAM + local planner + terrain analysis + Livox + G1 WebRTC)
tmux send-keys -t "$SESSION_NAME":0.0 \
    "./system_real_robot_g1.sh robot_ip:=$G1_IP connection_method:=$CONNECTION_METHOD control_mode:=$CONTROL_MODE" C-m

# Pane 1: VLM navigation bridge
tmux send-keys -t "$SESSION_NAME":0.1 "ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py" C-m

# Pane 2: SAM2/YOLOE object detector
tmux send-keys -t "$SESSION_NAME":0.2 "ros2 launch sam2_detector sam2_detector.launch.py" C-m

# Pane 3: RealSense camera driver
tmux send-keys -t "$SESSION_NAME":0.3 "ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=false depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30" C-m

# Pane 4: Relay RealSense RGB to the legacy /camera/image interface for downstream consumers
tmux send-keys -t "$SESSION_NAME":0.4 "ros2 run image_transport republish raw --ros-args -r in:=/camera/camera/color/image_raw -r out:=/camera/image" C-m

# Pane 5: free for manual commands (e.g. publishing /object_goal, arm gesture services)
tmux send-keys -t "$SESSION_NAME":0.5 "echo 'G1 environment ready — pane 5 free'" C-m

tmux attach -t "$SESSION_NAME"
