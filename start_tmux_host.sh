#!/usr/bin/env zsh
# Runs on the GPU HOST (Domain 80).
# Launches: camera decompressor, VLM navigator, SAM2/YOLOE detector, RViz.
# Cross-machine domain_bridge is expected to run on robot side
# in a dedicated communication pane.
set -e

SESSION_NAME="sagan_nav_host"
WORKDIR="./"
HOST_COMM_IFACE="${HOST_COMM_IFACE:-wlp131s0}"
USE_ZENOH="${USE_ZENOH:-0}"
ZENOH_ROUTER_ENDPOINT="${ZENOH_ROUTER_ENDPOINT:-tcp/127.0.0.1:7447}"
ZENOH_ROUTER_CHECK_ATTEMPTS="${ZENOH_ROUTER_CHECK_ATTEMPTS:-20}"
ZENOH_OVERRIDE="mode=\"client\";connect/endpoints=[\"${ZENOH_ROUTER_ENDPOINT}\"]"

# kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# cleanup debug images
echo "Removing ./debug_images ..."
rm -rf "./debug_images"

# create new detached session (4 panes: 0=decompress, 1=vlm, 2=detector, 3=rviz)
tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"

for i in $(seq 1 3); do
    tmux split-window -t "$SESSION_NAME":0.0 -c "$WORKDIR"
    tmux select-layout -t "$SESSION_NAME":0 tiled
done

# Pane setup
if [[ "$USE_ZENOH" == "1" ]]; then
    SETUP_HOST="cd \"$WORKDIR\" && unset ROS_DOMAIN_ID && unset CYCLONEDDS_URI && export RMW_IMPLEMENTATION=rmw_zenoh_cpp && export ROS_LOCALHOST_ONLY=0 && export ZENOH_CONFIG_OVERRIDE='$ZENOH_OVERRIDE' && export ZENOH_ROUTER_CHECK_ATTEMPTS=$ZENOH_ROUTER_CHECK_ATTEMPTS"
else
    if ip link show "$HOST_COMM_IFACE" >/dev/null 2>&1; then
        HOST_CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${HOST_COMM_IFACE}\"/></Interfaces></General></Domain></CycloneDDS>"
        SETUP_HOST_DDS="export CYCLONEDDS_URI='$HOST_CYCLONEDDS_URI'"
    else
        echo "Warning: interface '$HOST_COMM_IFACE' not found; host panes will use default DDS interfaces."
        SETUP_HOST_DDS="unset CYCLONEDDS_URI"
    fi
    # DDS mode: host uses Domain 80
    SETUP_HOST="cd \"$WORKDIR\" && export ROS_DOMAIN_ID=80 && $SETUP_HOST_DDS"
fi

# Panes 0-3: Domain 80
for pane in 0 1 2 3; do
    tmux send-keys -t "$SESSION_NAME":0.$pane "$SETUP_HOST" C-m
done

sleep 5

# Pane 0: decompress /camera/image/compressed → /camera/image
# Required because domain_bridge forwards the compressed topic; VLM subscribes to raw.
tmux send-keys -t "$SESSION_NAME":0.0 \
    "source ./install/setup.bash && ros2 run image_transport republish \
--ros-args \
-p in_transport:=compressed \
-p out_transport:=raw \
--remap in/compressed:=/camera/image/compressed \
--remap out:=/camera/image" C-m

# Pane 1: VLM navigation bridge
tmux send-keys -t "$SESSION_NAME":0.1 "ros2 launch vlm_nav_bridge vlm_nav_bridge.launch.py" C-m

# Pane 2: SAM2/YOLOE object detector
tmux send-keys -t "$SESSION_NAME":0.2 "ros2 launch sam2_detector sam2_detector.launch.py" C-m

# Pane 3: RViz visualization (base station config)
tmux send-keys -t "$SESSION_NAME":0.3 \
    "source ./install/setup.bash && ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz" C-m

tmux attach -t "$SESSION_NAME"
