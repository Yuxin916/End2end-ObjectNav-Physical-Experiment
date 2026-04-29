#!/usr/bin/env bash
# Runs on the GPU HOST.
# Launches: domain_bridge (79↔80), camera decompressor, VLN bridge,
# instruction console, RViz.
# Robot runs its own autonomy stack on Domain 79.
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SESSION_NAME="sagan_nav_vln_host"
WORKDIR="$SCRIPT_DIR"
export ROS_LOG_DIR="$WORKDIR/.ros/log"
mkdir -p "$ROS_LOG_DIR"

CHECKPOINT_DIR="$WORKDIR/checkpoints/ALLdata_NL03_paddingEmbedding12400"
CONFIG_FILE="$WORKDIR/src/vln_bridge/config/vln_bridge.yaml"
VENV_ACTIVATE="$WORKDIR/.venv/bin/activate"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# cleanup likely-stale host-side processes from previous runs
echo "Cleaning up stale VLN host processes if any..."
pkill -f "rviz2" 2>/dev/null || true
pkill -f "vlm_navigator" 2>/dev/null || true
pkill -f "instruction_console.sh" 2>/dev/null || true
pkill -f "domain_bridge" 2>/dev/null || true
pkill -f "image_transport.*republish" 2>/dev/null || true

# cleanup debug images
echo "Removing ./debug_images ..."
rm -rf "$WORKDIR/debug_images"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "[ERROR] Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

# make sure latest host-side nodes are installed
echo "Building vln_bridge and domain_bridge ..."
source /opt/ros/jazzy/setup.bash
if [[ -f "$VENV_ACTIVATE" ]]; then
    source "$VENV_ACTIVATE"
fi
colcon build --packages-select vln_bridge domain_bridge --symlink-install

# create new detached session (7 panes:
# 0=bridge, 1=decompress, 2=vln, 3=instruction, 4=rviz, 5-6=free)
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

for i in $(seq 1 6); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

if [[ -f "$VENV_ACTIVATE" ]]; then
    SETUP_BRIDGE="cd \"$WORKDIR\" && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && mkdir -p \"$ROS_LOG_DIR\" && source /opt/ros/jazzy/setup.bash && source \"$VENV_ACTIVATE\" && source \"$WORKDIR/install/setup.bash\""
    SETUP_HOST="cd \"$WORKDIR\" && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && mkdir -p \"$ROS_LOG_DIR\" && export ROS_DOMAIN_ID=80 && source /opt/ros/jazzy/setup.bash && source \"$VENV_ACTIVATE\" && source \"$WORKDIR/install/setup.bash\""
else
    SETUP_BRIDGE="cd \"$WORKDIR\" && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && mkdir -p \"$ROS_LOG_DIR\" && source /opt/ros/jazzy/setup.bash && source \"$WORKDIR/install/setup.bash\""
    SETUP_HOST="cd \"$WORKDIR\" && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && mkdir -p \"$ROS_LOG_DIR\" && export ROS_DOMAIN_ID=80 && source /opt/ros/jazzy/setup.bash && source \"$WORKDIR/install/setup.bash\""
fi

# Pane 0: domain_bridge — no ROS_DOMAIN_ID so it can bridge Domain 79 and 80.
tmux send-keys -t "$SESSION_NAME":0.0 "$SETUP_BRIDGE" C-m

# Panes 1-6: host-side ROS domain.
for pane in 1 2 3 4 5 6; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$SETUP_HOST" C-m
done

sleep 5

CMD_PANE0="ros2 launch domain_bridge domain_bridge.launch"
CMD_PANE1="ros2 run image_transport republish --ros-args -p in_transport:=compressed -p out_transport:=raw --remap in/compressed:=/camera/image/compressed --remap out:=/camera/image"
CMD_PANE2="ros2 launch vln_bridge vln_bridge.launch.py config_file:=$CONFIG_FILE checkpoint:=$CHECKPOINT_DIR instruction_topic:=/instruction template:=RGB_HisKFSingleColor"
CMD_PANE3="bash \"$WORKDIR/instruction_console.sh\""
CMD_PANE4="ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz"
CMD_PANE5="echo 'Pane 5 ready. Run ./record.sh manually when needed.'"
CMD_PANE6="echo 'Pane 6 ready. Use for debug or manual ros2 commands.'"

tmux send-keys -t "$SESSION_NAME":0.0 "$CMD_PANE0" C-m
tmux send-keys -t "$SESSION_NAME":0.1 "$CMD_PANE1" C-m
tmux send-keys -t "$SESSION_NAME":0.2 "$CMD_PANE2" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "$CMD_PANE3" C-m
tmux send-keys -t "$SESSION_NAME":0.4 "$CMD_PANE4" C-m
tmux send-keys -t "$SESSION_NAME":0.5 "$CMD_PANE5" C-m
tmux send-keys -t "$SESSION_NAME":0.6 "$CMD_PANE6" C-m

tmux attach -t "$SESSION_NAME"
