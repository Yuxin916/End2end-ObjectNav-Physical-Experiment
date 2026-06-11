#!/bin/bash
# Launch an interactive container for the autonomy_stack.
#   docker/run.sh           -> interactive bash (default)
#   docker/run.sh build     -> build the ROS2 workspace (one-time / after edits)
#   docker/run.sh <cmd...>  -> run an arbitrary command inside the env
#
# Uses host networking (Mid-360 UDP, WebRTC, ROS2 DDS), X11 for RViz, and
# passes through the joystick. Does NOT start the robot by itself.
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

xhost +local:root >/dev/null 2>&1 || true

DEV_ARGS=()
[ -e /dev/input/js0 ] && DEV_ARGS+=(--device /dev/input/js0)
for d in /dev/ttyACM0 /dev/ttyACM1 /dev/ttyUSB0; do
  [ -e "$d" ] && DEV_ARGS+=(--device "$d")
done

ENV_FILE_ARGS=()
[ -f "$HOME/.unitree_g1.env" ] && ENV_FILE_ARGS+=(-v "$HOME/.unitree_g1.env:/root/.unitree_g1.env:ro")

CMD=("$@")
if [ "${1:-}" = "build" ]; then
  CMD=(bash docker/setup_workspace.sh)
elif [ ${#CMD[@]} -eq 0 ]; then
  CMD=(bash)
fi

docker run -it --rm \
  --name autonomy_stack \
  --network host \
  --ipc host \
  --privileged \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$REPO_DIR":/workspace/autonomy_stack:rw \
  "${ENV_FILE_ARGS[@]}" \
  "${DEV_ARGS[@]}" \
  autonomy_stack:jazzy \
  "${CMD[@]}"
