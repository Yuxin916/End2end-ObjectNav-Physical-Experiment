#!/bin/bash
# Record a rosbag inside the running autonomy_stack container.
# Usage:
#   docker/rosbag_record.sh                  -> record all topics, auto-named
#   docker/rosbag_record.sh my_bag           -> record all topics, named my_bag
#   docker/rosbag_record.sh my_bag /topic1 /topic2  -> record specific topics

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

if ! docker ps --format '{{.Names}}' | grep -q '^autonomy_stack$'; then
  echo "ERROR: autonomy_stack container is not running. Start it with docker/run.sh first."
  exit 1
fi

BAG_NAME="${1:-bag_$(date +%Y%m%d_%H%M%S)}"
shift || true  # remaining args are topics (empty = record all)

if [ $# -gt 0 ]; then
  TOPICS="$*"
else
  TOPICS="/imu/data /lidar/scan"
fi

BAG_PATH="/workspace/autonomy_stack/bags/$BAG_NAME"

echo ">>> Recording to $BAG_PATH"
echo ">>> Topics: ${TOPICS}"
echo ">>> Ctrl+C to stop"

docker exec -it autonomy_stack bash -c "
  export ROS_DOMAIN_ID=1
  source /workspace/autonomy_stack/install/setup.bash
  mkdir -p /workspace/autonomy_stack/bags
  ros2 bag record $TOPICS -o $BAG_PATH
"
