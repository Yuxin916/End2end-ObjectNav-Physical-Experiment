#!/usr/bin/env bash
set -e

SESSION_NAME="sagan_nav_physical_experiment"
WORKDIR="/home/all/yuxin/End2end-ObjectNav-Physical-Experiment"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# cleanup old processes
echo "Killing old ros/unity processes if any..."
pkill -9 -f ros || true
pkill -9 -f unity || true

# cleanup debug images
echo "Removing $WORKDIR/debug_images ..."
rm -rf "$WORKDIR/debug_images"

# create new detached session
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# create 4 panes total
tmux split-window -h -t "$SESSION_NAME":0 -c "$WORKDIR"
tmux split-window -v -t "$SESSION_NAME":0.0 -c "$WORKDIR"
tmux split-window -v -t "$SESSION_NAME":0.1 -c "$WORKDIR"
tmux select-layout -t "$SESSION_NAME":0 tiled

# common pre-build setup
PREBUILD_SETUP="cd \"$WORKDIR\" && conda deactivate && source .venv/bin/activate && source /opt/ros/jazzy/setup.bash"

# full post-build setup
FULL_SETUP="cd \"$WORKDIR\" && conda deactivate && source .venv/bin/activate && source /opt/ros/jazzy/setup.bash && source install/setup.bash"

# commands to auto-run after build
CMD_PANE0="ROS_DOMAIN_ID=1 ROBOT_CONFIG_PATH='omniDir' ./system_simulation.sh"
CMD_PANE1="ROS_DOMAIN_ID=1 ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py"
CMD_PANE2="ROS_DOMAIN_ID=1 ros2 launch sam2_detector sam2_detector.launch.py"

# run build in pane 0
tmux send-keys -t "$SESSION_NAME":0.0 "$PREBUILD_SETUP && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-skip arise_slam_mid360 arise_slam_mid360_msgs livox_ros_driver2" C-m

# wait for build to finish by polling pane output
echo "Waiting for colcon build to finish in pane 0..."
while true; do
    PANE_OUT="$(tmux capture-pane -pt "$SESSION_NAME":0.0 -S -50)"
    if echo "$PANE_OUT" | grep -q "Summary: .* packages finished"; then
        break
    fi
    if echo "$PANE_OUT" | grep -q "Failed   <<<\|Aborted  <<<\|colcon build: error"; then
        echo "Build failed. Attaching to tmux session for inspection."
        tmux attach -t "$SESSION_NAME"
        exit 1
    fi
    sleep 2
done

echo "Build finished successfully."

# send full setup to all 4 panes
for pane in 0 1 2 3; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$FULL_SETUP" C-m
done

# small pause to make sure setup is sourced cleanly
sleep 2

# auto-run commands
tmux send-keys -t "$SESSION_NAME":0.0 "$CMD_PANE0" C-m
tmux send-keys -t "$SESSION_NAME":0.1 "$CMD_PANE1" C-m
tmux send-keys -t "$SESSION_NAME":0.2 "$CMD_PANE2" C-m
tmux send-keys -t "$SESSION_NAME":0.3 "echo 'Environment ready in pane 3'" C-m

# attach
tmux attach -t "$SESSION_NAME"