#!/usr/bin/env zsh
# Runs on the ROBOT (Domain 79).
# Launches: autonomy stack (SLAM + local planner + Livox driver) and Theta camera driver.
# domain_bridge runs on the host side (start_tmux_host.sh).
set -e

SESSION_NAME="sagan_nav_robot"
WORKDIR="./"

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

# All panes use Domain 79. ~/.zshrc handles venv + ROS sourcing.
FULL_SETUP="cd \"$WORKDIR\" && export ROS_DOMAIN_ID=79"

for pane in 0 1 2 3; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$FULL_SETUP" C-m
done

sleep 5

# Pane 0: autonomy stack (SLAM + local planner + terrain analysis + Livox driver)
tmux send-keys -t "$SESSION_NAME":0.0 "./system_real_robot.sh" C-m

# Pane 1: Theta camera driver
tmux send-keys -t "$SESSION_NAME":0.1 "ros2 launch receive_theta receive_theta_sensorpod.launch" C-m

# Panes 2-3: free for manual commands
tmux send-keys -t "$SESSION_NAME":0.2 "echo 'Robot environment ready in pane 2'" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "echo 'Robot environment ready in pane 3'" C-m

tmux attach -t "$SESSION_NAME"
