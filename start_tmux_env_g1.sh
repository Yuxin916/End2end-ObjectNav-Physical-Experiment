#!/usr/bin/env zsh
# G1 humanoid — single-machine setup (base autonomy + VLM on same host).
# Pane 0: base nav stack (SLAM + local planner + Livox + G1 WebRTC)
# Pane 1: VLM navigation bridge
# Pane 2: SAM2/YOLOE object detector
# Pane 3: TCP RealSense receiver -> /camera/image
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

# Activate venv so unitree_webrtc_connect is available, then source ROS workspace.
# Force ROS_DOMAIN_ID=1 + Fast DDS in EVERY pane so VLM bridge / detector / camera
# receiver (panes 1-3) share the same DDS domain as the nav stack (pane 0, set by
# system_real_robot_g1.sh). Without this they inherit .zshrc's ROS_DOMAIN_ID=80 and
# can't see pane 0's topics. RMW must match too.
FULL_SETUP="cd \"$WORKDIR\" && source /opt/ros/jazzy/setup.zsh && source install/setup.zsh && source .venv/bin/activate && export ROS_DOMAIN_ID=1 && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"

for pane in 0 1 2 3 4 5; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$FULL_SETUP" C-m
done

sleep 5

# Pane 0: G1 base autonomy stack (SLAM + local planner + terrain analysis + Livox + G1 WebRTC)
tmux send-keys -t "$SESSION_NAME":0.0 \
    "./system_real_robot_g1.sh robot_ip:=$G1_IP connection_method:=$CONNECTION_METHOD control_mode:=$CONTROL_MODE" C-m

# Pane 1: VLM navigation bridge
# Tee stdout to ./log so all get_logger() timing markers (pipeline_ms, bev_update_ms,
# lidar_hz, robot_speed, wp_exec_s, re_query_stats, cam_timing, GPU memory) are
# captured. NOTE: the node's target_debug_*.log only captures direct _file_logger
# calls (det_cb/target_state); get_logger() output goes to stdout, not that file.
tmux send-keys -t "$SESSION_NAME":0.1 \
    "mkdir -p log && ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py 2>&1 | tee log/vlm_$(date +%Y%m%d_%H%M%S).log" C-m

# Pane 2: SAM2/YOLOE object detector
# Tee stdout to ./log so the [detector_ms] timing markers are captured for the
# real-time profiling report (sam2_detector has no on-disk file logger of its own).
tmux send-keys -t "$SESSION_NAME":0.2 \
    "mkdir -p log && ros2 launch sam2_detector sam2_detector.launch.py 2>&1 | tee log/detector_$(date +%Y%m%d_%H%M%S).log" C-m

# Pane 3: Receive JPEG frames from Jetson over TCP and publish /camera/image
tmux send-keys -t "$SESSION_NAME":0.3 "ros2 launch vlm_nav_bridge rs_tcp_receiver.launch.py" C-m

# Pane 4: free for camera/topic diagnostics
tmux send-keys -t "$SESSION_NAME":0.4 "echo 'Camera TCP receiver ready — pane 4 free for diagnostics'" C-m

# Pane 5: free for manual commands (e.g. publishing /object_goal, arm gesture services)
tmux send-keys -t "$SESSION_NAME":0.5 "echo 'G1 environment ready — pane 5 free'" C-m

tmux attach -t "$SESSION_NAME"
