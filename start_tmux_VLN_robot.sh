#!/usr/bin/env bash
# Runs on the ROBOT / onboard laptop.
# Launches: autonomy stack (SLAM + local planner + Livox driver) and Theta camera
# in Domain 79. Heavy VLN inference runs on the host side.
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SESSION_NAME="sagan_nav_vln_robot"
WORKDIR="$SCRIPT_DIR"
export ROS_LOG_DIR="$WORKDIR/.ros/log"
mkdir -p "$ROS_LOG_DIR"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# cleanup likely-stale robot-side processes from previous runs
echo "Cleaning up stale VLN robot processes if any..."
pkill -f "rviz2" 2>/dev/null || true
pkill -f "localPlanner" 2>/dev/null || true
pkill -f "vehicleSimulator" 2>/dev/null || true
pkill -f "receive_theta" 2>/dev/null || true
pkill -f "image_transport.*republish" 2>/dev/null || true

# create new detached session
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# create 5 panes total
for i in $(seq 1 4); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

FULL_SETUP="cd \"$WORKDIR\" && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && mkdir -p \"$ROS_LOG_DIR\" && export ROS_DOMAIN_ID=79 && source /opt/ros/jazzy/setup.bash && source \"$WORKDIR/install/setup.bash\""

for pane in 0 1 2 3 4; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$FULL_SETUP" C-m
done

sleep 5

CMD_PANE0="bash \"$WORKDIR/system_real_robot.sh\""
CMD_PANE1="ros2 launch receive_theta receive_theta_sensorpod.launch"
CMD_PANE2="ros2 run image_transport republish --ros-args -p in_transport:=compressed -p out_transport:=raw --remap in/compressed:=/camera/image/compressed --remap out:=/camera/image"
CMD_PANE3="echo 'Robot environment ready in pane 3'"
CMD_PANE4="echo 'Robot environment ready in pane 4'"

tmux send-keys -t "$SESSION_NAME":0.0 "$CMD_PANE0" C-m
tmux send-keys -t "$SESSION_NAME":0.1 "$CMD_PANE1" C-m
tmux send-keys -t "$SESSION_NAME":0.2 "$CMD_PANE2" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "$CMD_PANE3" C-m
tmux send-keys -t "$SESSION_NAME":0.4 "$CMD_PANE4" C-m

tmux attach -t "$SESSION_NAME"
