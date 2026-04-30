#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SESSION_NAME="sagan_nav_vln_real"
WORKDIR="$SCRIPT_DIR"
export ROS_LOG_DIR="$WORKDIR/.ros/log"
mkdir -p "$ROS_LOG_DIR"

CONFIG_FILE="$WORKDIR/src/vln_bridge/config/vln_bridge.yaml"
VENV_ACTIVATE="$WORKDIR/.venv/bin/activate"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# cleanup likely-stale processes from previous runs
echo "Cleaning up stale real-robot processes if any..."
pkill -f "rviz2" 2>/dev/null || true
pkill -f "vlm_navigator" 2>/dev/null || true
pkill -f "vln_bridge" 2>/dev/null || true
pkill -f "instruction_console.sh" 2>/dev/null || true
pkill -f "localPlanner" 2>/dev/null || true

# cleanup debug images
echo "Removing ./debug_images ..."
rm -rf "$WORKDIR/debug_images"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

# make sure latest nodes are installed
echo "Building vln_bridge ..."
source /opt/ros/jazzy/setup.bash
if [[ -f "$VENV_ACTIVATE" ]]; then
    source "$VENV_ACTIVATE"
fi
colcon build --packages-select vln_bridge --symlink-install

# create new detached session
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# create 6 panes total (2x3 tiled)
# apply tiled after each split to keep panes large enough for the next split
for i in $(seq 1 5); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

# setup for all panes
if [[ -f "$VENV_ACTIVATE" ]]; then
    FULL_SETUP="cd \"$WORKDIR\" && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && mkdir -p \"$ROS_LOG_DIR\" && source /opt/ros/jazzy/setup.bash && source \"$VENV_ACTIVATE\" && source \"$WORKDIR/install/setup.bash\""
else
    FULL_SETUP="cd \"$WORKDIR\" && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && mkdir -p \"$ROS_LOG_DIR\" && source /opt/ros/jazzy/setup.bash && source \"$WORKDIR/install/setup.bash\""
fi

# commands to run
CMD_PANE0="bash \"$WORKDIR/system_real_robot.sh\""
CMD_PANE1="ros2 launch vln_bridge vln_bridge.launch.py config_file:=$CONFIG_FILE instruction_topic:=/instruction template:=RGB_HisKFSingleColor"
CMD_PANE2="bash \"$WORKDIR/instruction_console.sh\""
CMD_PANE3="echo 'Pane 3 ready. Use for debug or manual ros2 commands.'"
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
