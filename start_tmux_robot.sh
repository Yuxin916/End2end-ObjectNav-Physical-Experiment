#!/usr/bin/env zsh
# Runs on the ROBOT NUC (Domain 79).
# Launches: navigation stack only (SLAM + local planner + Livox driver + WebRTC)
# plus the Theta camera driver. VLM/SAM/domain_bridge run on the host side.
set -e

SESSION_NAME="sagan_nav_robot"
WORKDIR="./"
ROBOT_CONFIG="unitree/unitree_go2_slow"
ROBOT_DOMAIN_ID="${ROBOT_DOMAIN_ID:-79}"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# create new detached session
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# create 4 panes total
for i in $(seq 1 3); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

# All robot-side panes stay on the robot ROS 2 domain.
FULL_SETUP="cd \"$WORKDIR\" && source ./install/setup.zsh && unset RMW_IMPLEMENTATION CYCLONEDDS_URI ROBOT_CYCLONEDDS_URI && export ROS_DOMAIN_ID=$ROBOT_DOMAIN_ID && export ROBOT_CONFIG_PATH=$ROBOT_CONFIG"

for pane in 0 1 2 3; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$FULL_SETUP" C-m
done

sleep 5

# Pane 0: robot-side navigation stack (SLAM + local planner + terrain analysis + Livox driver + WebRTC)
tmux send-keys -t "$SESSION_NAME":0.0 "./system_real_robot.sh" C-m

# Pane 1: Theta camera driver
tmux send-keys -t "$SESSION_NAME":0.1 "ros2 launch receive_theta receive_theta_sensorpod.launch" C-m

# Panes 2-3: free for manual commands
tmux send-keys -t "$SESSION_NAME":0.2 "echo 'Robot environment ready in pane 2'" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "echo 'Robot environment ready in pane 3'" C-m

tmux attach -t "$SESSION_NAME"
