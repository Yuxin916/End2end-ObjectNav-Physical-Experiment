#!/usr/bin/env zsh
# Runs on the GPU HOST (Domain 80).
# Launches: domain_bridge (79↔80), VLM navigator, SAM2/YOLOE detector,
# camera decompressor, RViz.
# Robot runs its own autonomy stack on Domain 79.
set -e

SESSION_NAME="sagan_nav_host"
WORKDIR="./"
HOST_DOMAIN_ID="${HOST_DOMAIN_ID:-80}"
ROBOT_DOMAIN_ID="${ROBOT_DOMAIN_ID:-79}"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi


# create new detached session (7 panes: 0=bridge, 1=decompress, 2=vlm, 3=detector, 4=rviz, 5-6=free)
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

for i in $(seq 1 6); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

# Most panes use the host-side ROS domain. Pane 0 launches the domain bridge,
# which reads the bridge direction from its YAML config.
SETUP_BRIDGE="cd \"$WORKDIR\" && source ./install/setup.zsh && unset RMW_IMPLEMENTATION CYCLONEDDS_URI ROBOT_CYCLONEDDS_URI"
SETUP_HOST="cd \"$WORKDIR\" && source ./install/setup.zsh && unset RMW_IMPLEMENTATION CYCLONEDDS_URI ROBOT_CYCLONEDDS_URI && export ROS_DOMAIN_ID=$HOST_DOMAIN_ID"

# Pane 0: domain_bridge — no ROS_DOMAIN_ID so it can reach both Domain 79 and 80
tmux send-keys -t "$SESSION_NAME":0.0 "$SETUP_BRIDGE" C-m

# Panes 1-6: Domain 80
for pane in 1 2 3 4 5 6; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$SETUP_HOST" C-m
done

sleep 5

# Pane 0: domain_bridge — bridges robot ↔ host domains
tmux send-keys -t "$SESSION_NAME":0.0 \
    "ros2 launch domain_bridge domain_bridge.launch from_domain:=$ROBOT_DOMAIN_ID to_domain:=$HOST_DOMAIN_ID" C-m

# Pane 1: decompress /camera/image/compressed → /camera/image
# Required because domain_bridge forwards the compressed topic; VLM subscribes to raw.
tmux send-keys -t "$SESSION_NAME":0.1 \
    "source ./install/setup.zsh && ros2 run image_transport republish \
--ros-args \
-p in_transport:=compressed \
-p out_transport:=raw \
--remap in/compressed:=/camera/image/compressed \
--remap out:=/camera/image" C-m

# Pane 2: VLM navigation bridge
tmux send-keys -t "$SESSION_NAME":0.2 "ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py" C-m

# Pane 3: SAM2/YOLOE object detector
tmux send-keys -t "$SESSION_NAME":0.3 "ros2 launch sam2_detector sam2_detector.launch.py" C-m

# Pane 4: RViz visualization (base station config)
tmux send-keys -t "$SESSION_NAME":0.4 \
    "ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz" C-m

# Panes 5-6: free for manual commands (e.g., publishing /object_goal)
tmux send-keys -t "$SESSION_NAME":0.5 "echo 'Host environment ready in pane 5'" C-m
tmux send-keys -t "$SESSION_NAME":0.6 "echo 'Host environment ready in pane 6'" C-m

tmux attach -t "$SESSION_NAME"
