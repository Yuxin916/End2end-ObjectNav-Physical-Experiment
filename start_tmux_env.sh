#!/usr/bin/env bash
set -e

SESSION_NAME="sagan_nav_physical_experiment"
WORKDIR="$HOME/projects/VLN_CL_CoTNav/End2end-ObjectNav-Physical-Experiment"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# create new detached session
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

# create 4 panes total
tmux split-window -h -t "$SESSION_NAME":0 -c "$WORKDIR"
tmux split-window -v -t "$SESSION_NAME":0.0 -c "$WORKDIR"
tmux split-window -v -t "$SESSION_NAME":0.1 -c "$WORKDIR"
tmux select-layout -t "$SESSION_NAME":0 tiled

# common pre-build setup
PREBUILD_SETUP="cd \"$WORKDIR\" && conda deactivate && source .venv/bin/activate && source /opt/ros/jazzy/setup.zsh"

# full post-build setup
FULL_SETUP="cd \"$WORKDIR\" && conda deactivate && source .venv/bin/activate && source /opt/ros/jazzy/setup.zsh && source install/setup.zsh"

# run build in pane 0
tmux send-keys -t "$SESSION_NAME":0.0 "$PREBUILD_SETUP && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-skip arise_slam_mid360 arise_slam_mid360_msgs livox_ros_driver2" C-m

# wait for build to finish by polling pane output
echo "Waiting for colcon build to finish in pane 0..."
while tmux capture-pane -pt "$SESSION_NAME":0.0 | tail -n 20 | grep -q "Starting >>>\|[[:space:]]\+\.\.\.\|Summary:"; do
    sleep 2
done

# send full setup to all 4 panes
for pane in 0 1 2 3; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$FULL_SETUP" C-m
done

# show a ready message in each pane
for pane in 0 1 2 3; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "echo 'Environment ready in pane $pane'" C-m
done

# attach
tmux attach -t "$SESSION_NAME"