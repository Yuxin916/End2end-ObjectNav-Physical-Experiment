#!/usr/bin/env zsh
# Runs on the ROBOT NUC.
# Launches:
# - robot navigation stack (Domain 79, default DDS interface selection)
# - Theta camera driver (Domain 79, default DDS interface selection)
# - dedicated cross-machine domain_bridge process (interface-pinned via CYCLONEDDS_URI)
set -e

SESSION_NAME="sagan_nav_robot"
WORKDIR="./"
ROBOT_CONFIG="unitree/unitree_go2_slow"
ROBOT_COMM_IFACE="${ROBOT_COMM_IFACE:-wlo1}"
ROBOT_BRIDGE_CONFIG="${ROBOT_BRIDGE_CONFIG:-src/utilities/domain_bridge/config/domain_bridge_minimal.yaml}"
ROBOT_ENABLE_BRIDGE="${ROBOT_ENABLE_BRIDGE:-1}"
ROBOT_BRIDGE_NICE="${ROBOT_BRIDGE_NICE:-15}"
ROBOT_BRIDGE_CPU="${ROBOT_BRIDGE_CPU:-}"
if ip link show "$ROBOT_COMM_IFACE" >/dev/null 2>&1; then
    ROBOT_CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${ROBOT_COMM_IFACE}\"/></Interfaces></General></Domain></CycloneDDS>"
    SETUP_COMM_DDS="export CYCLONEDDS_URI='$ROBOT_CYCLONEDDS_URI'"
else
    echo "Warning: interface '$ROBOT_COMM_IFACE' not found; comm pane will use default DDS interfaces."
    SETUP_COMM_DDS="unset CYCLONEDDS_URI"
fi

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# cleanup stale domain_bridge processes from previous experiments
echo "Cleaning stale domain_bridge processes ..."
pkill -f "/install/domain_bridge/lib/domain_bridge/domain_bridge" 2>/dev/null || true

# create new detached session
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# create 4 panes total
for i in $(seq 1 3); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

# Pane-specific setup:
# - Panes 0/1/3 keep default DDS selection to protect local SLAM/control timing.
# - Pane 2 is communication-only and pins DDS to Wi-Fi interface.
SETUP_LOCAL="cd \"$WORKDIR\" && unset CYCLONEDDS_URI && export ROS_DOMAIN_ID=79 && export ROBOT_CONFIG_PATH=$ROBOT_CONFIG"
SETUP_COMM="cd \"$WORKDIR\" && unset ROS_DOMAIN_ID && $SETUP_COMM_DDS && export ROBOT_CONFIG_PATH=$ROBOT_CONFIG"

tmux send-keys -t "$SESSION_NAME":0.0 "$SETUP_LOCAL" C-m
tmux send-keys -t "$SESSION_NAME":0.1 "$SETUP_LOCAL" C-m
tmux send-keys -t "$SESSION_NAME":0.2 "$SETUP_COMM" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "$SETUP_LOCAL" C-m

sleep 5

# Pane 0: robot-side navigation stack (SLAM + local planner + terrain analysis + Livox driver + WebRTC)
tmux send-keys -t "$SESSION_NAME":0.0 "./system_real_robot.sh" C-m

# Pane 1: Theta camera driver
tmux send-keys -t "$SESSION_NAME":0.1 "ros2 launch receive_theta receive_theta_sensorpod.launch" C-m

# Pane 2: communication-only domain_bridge (79 <-> 80), interface pinned
if [[ "$ROBOT_ENABLE_BRIDGE" == "1" ]]; then
    BRIDGE_CMD="nice -n $ROBOT_BRIDGE_NICE ros2 launch domain_bridge domain_bridge.launch config:=$ROBOT_BRIDGE_CONFIG"
    if [[ -n "$ROBOT_BRIDGE_CPU" ]]; then
        BRIDGE_CMD="taskset -c $ROBOT_BRIDGE_CPU $BRIDGE_CMD"
    fi
    tmux send-keys -t "$SESSION_NAME":0.2 \
        "source ./install/setup.bash && $BRIDGE_CMD" C-m
else
    tmux send-keys -t "$SESSION_NAME":0.2 "echo 'Bridge disabled (ROBOT_ENABLE_BRIDGE=0)'" C-m
fi

# Pane 3: free for manual commands
tmux send-keys -t "$SESSION_NAME":0.3 "echo 'Robot environment ready in pane 3'" C-m

tmux attach -t "$SESSION_NAME"
