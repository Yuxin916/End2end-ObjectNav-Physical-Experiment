#!/bin/bash
# Build the autonomy_stack image in two stages so iterating on the control
# layer never recompiles GTSAM:
#   Stage 1  docker/Dockerfile      -> autonomy_stack:heavy  (ROS+Sophus/Ceres/GTSAM/Livox)
#   Stage 2  docker/Dockerfile.sdk  -> autonomy_stack:jazzy  (+ unitree_sdk2 / cyclonedds)
# Build context = repo root (needs the vendored C++ sources).
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_DIR"

if ! docker image inspect autonomy_stack:heavy >/dev/null 2>&1; then
  echo ">>> Stage 1: heavy deps (one-time, ~30-40 min)"
  docker build -f docker/Dockerfile -t autonomy_stack:heavy .
else
  echo ">>> Stage 1: autonomy_stack:heavy exists, skipping (use --rebuild-heavy to force)"
fi
[ "$1" = "--rebuild-heavy" ] && docker build -f docker/Dockerfile -t autonomy_stack:heavy .

echo ">>> Stage 2: unitree_sdk2 control layer"
docker build -f docker/Dockerfile.sdk -t autonomy_stack:jazzy .
echo ">>> Done: autonomy_stack:jazzy"
