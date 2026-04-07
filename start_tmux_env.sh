#!/usr/bin/env zsh
set -e

SESSION_NAME="sagan_nav_physical_experiment"
WORKDIR="./"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi


# cleanup debug images
echo "Removing ./debug_images ..."
rm -rf "./debug_images"

# create new detached session
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# create 6 panes total (2x3 tiled)
# apply tiled after each split to keep panes large enough for the next split
for i in $(seq 1 5); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

# setup for all panes
# tmux starts interactive zsh shells, so ~/.zshrc already handles venv + ROS sourcing.
FULL_SETUP="cd \"$WORKDIR\""

# commands to run
CMD_PANE0="./system_real_robot.sh"
CMD_PANE1="ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py"
CMD_PANE2="ros2 launch sam2_detector sam2_detector.launch.py"
CMD_PANE3="ros2 launch receive_theta receive_theta_sensorpod.launch"
CMD_PANE4="echo 'Pane 4 ready. Run ./record.sh manually when needed.'"

# send full setup to all 6 panes
for pane in 0 1 2 3 4 5; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$FULL_SETUP" C-m
done

# small pause to make sure setup is sourced cleanly
sleep 5

# auto-run commands
tmux send-keys -t "$SESSION_NAME":0.0 "$CMD_PANE0" C-m
tmux send-keys -t "$SESSION_NAME":0.1 "$CMD_PANE1" C-m
tmux send-keys -t "$SESSION_NAME":0.2 "$CMD_PANE2" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "$CMD_PANE3" C-m
tmux send-keys -t "$SESSION_NAME":0.4 "$CMD_PANE4" C-m
tmux send-keys -t "$SESSION_NAME":0.5 "echo 'Environment ready in pane 5'" C-m

# attach
tmux attach -t "$SESSION_NAME"
