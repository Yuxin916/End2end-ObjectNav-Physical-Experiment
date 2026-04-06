#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BAG_ROOT_DIR="${BAG_ROOT_DIR:-bags}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${BAG_ROOT_DIR}/${TIMESTAMP}"

mkdir -p "$BAG_ROOT_DIR"

if [ "$#" -gt 0 ]; then
  RECORD_ARGS=("$@")
else
  RECORD_ARGS=(-a)
fi

echo "Recording bag to: $OUTPUT_DIR"
echo "ros2 bag record args: ${RECORD_ARGS[*]}"
echo "Press Ctrl+C to stop recording."

ros2 bag record "${RECORD_ARGS[@]}" -o "$OUTPUT_DIR"
