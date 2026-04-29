#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SESSION_NAME="sagan_nav_physical_experiment"
WORKDIR="$SCRIPT_DIR"
export ROS_LOG_DIR="$WORKDIR/.ros/log"
mkdir -p "$ROS_LOG_DIR"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# cleanup likely-stale simulation processes from previous runs
echo "Cleaning up stale simulation processes if any..."
pkill -f "default_server_endpoint" 2>/dev/null || true
pkill -f "Model.x86_64" 2>/dev/null || true
pkill -f "rviz2" 2>/dev/null || true
pkill -f "vlm_navigator" 2>/dev/null || true
pkill -f "sam2_detector" 2>/dev/null || true
pkill -f "localPlanner" 2>/dev/null || true
pkill -f "vehicleSimulator" 2>/dev/null || true


# cleanup debug images
echo "Removing ./debug_images ..."
rm -rf "./debug_images"

# make sure the latest bridge and simulator image republisher code are installed
echo "Building vln_bridge and vehicle_simulator ..."
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vln_bridge vehicle_simulator --symlink-install

# create new detached session
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# create 6 panes total (2x3 tiled)
# apply tiled after each split to keep panes large enough for the next split
for i in $(seq 1 5); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

# setup for all panes
FULL_SETUP="cd \"$WORKDIR\" && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && mkdir -p \"$ROS_LOG_DIR\" && source \"$WORKDIR/install/setup.bash\""

# commands to run
CMD_PANE0="./system_simulation.sh"
CMD_PANE1="ros2 launch vln_bridge vln_bridge.launch.py depth_topic:=/camera/depth instruction_topic:=/instruction template:=RGB_HisKFSingleColor"
CMD_PANE2="bash \"$WORKDIR/instruction_console.sh\""

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
tmux send-keys -t "$SESSION_NAME":0.3 "echo 'Environment ready in pane 3'" C-m
tmux send-keys -t "$SESSION_NAME":0.4 "echo 'Environment ready in pane 4'" C-m
tmux send-keys -t "$SESSION_NAME":0.5 "echo 'Environment ready in pane 5'" C-m

# attach
tmux attach -t "$SESSION_NAME"
