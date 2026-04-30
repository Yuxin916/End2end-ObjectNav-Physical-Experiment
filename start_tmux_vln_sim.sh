#!/usr/bin/env zsh
set -e

SESSION_NAME="sagan_nav_physical_experiment"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"

export ROS_DOMAIN_ID=80
export ROS_LOG_DIR="$WORKDIR/.ros/log"
mkdir -p "$ROS_LOG_DIR"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# cleanup debug images
echo "Removing ./debug_images ..."
rm -rf "$WORKDIR/debug_images"

# create new detached session
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# create 6 panes total
for i in $(seq 1 5); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

# tmux starts interactive zsh shells, so ~/.zshrc should handle venv + ROS sourcing.
FULL_SETUP="cd \"$WORKDIR\" && export ROS_DOMAIN_ID=80 && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && mkdir -p \"$ROS_LOG_DIR\""

# commands to run
CMD_PANE0="./system_simulation.sh"
CMD_PANE1="ros2 launch vln_bridge vln_bridge.launch.py depth_topic:=/camera/depth instruction_topic:=/instruction template:=RGB_HisKFSingleColor"
CMD_PANE2="bash \"$WORKDIR/instruction_console.sh\""

# send setup to all 6 panes
for pane in 0 1 2 3 4 5; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$FULL_SETUP" C-m
done

sleep 2

# auto-run commands
tmux send-keys -t "$SESSION_NAME":0.0 "$CMD_PANE0" C-m
tmux send-keys -t "$SESSION_NAME":0.1 "$CMD_PANE1" C-m
tmux send-keys -t "$SESSION_NAME":0.2 "$CMD_PANE2" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "echo 'Environment ready in pane 3. ROS_DOMAIN_ID='$ROS_DOMAIN_ID" C-m
tmux send-keys -t "$SESSION_NAME":0.4 "echo 'Environment ready in pane 4. ROS_DOMAIN_ID='$ROS_DOMAIN_ID" C-m
tmux send-keys -t "$SESSION_NAME":0.5 "echo 'Environment ready in pane 5. ROS_DOMAIN_ID='$ROS_DOMAIN_ID" C-m

# attach
tmux attach -t "$SESSION_NAME"
