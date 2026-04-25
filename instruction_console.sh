#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$SCRIPT_DIR"
export ROS_LOG_DIR="$SCRIPT_DIR/.ros/log"
mkdir -p "$ROS_LOG_DIR"
source ./install/setup.bash

echo "Instruction console ready."
echo "Waiting for a subscriber on /instruction before accepting input..."
echo "Press Ctrl-C to exit this pane."

get_instruction_subscriber_count() {
    ros2 topic info /instruction 2>/dev/null | awk '/Subscription count:/ {print $3}'
}

wait_for_instruction_subscriber() {
    local count=""
    while true; do
        count="$(get_instruction_subscriber_count)"
        if [[ "$count" =~ ^[0-9]+$ ]] && (( count >= 1 )); then
            echo "Detected $count subscriber(s) on /instruction."
            echo "Type one instruction per line. It will be published to /instruction."
            return 0
        fi
        sleep 1
    done
}

wait_for_instruction_subscriber

while true; do
    read -r -p "Instruction> " USER_INSTR || break

    if [[ -z "${USER_INSTR// }" ]]; then
        continue
    fi

    if ! python3 "$SCRIPT_DIR/publish_instruction.py" "$USER_INSTR"; then
        echo "Failed to publish instruction. Check whether vln_bridge is running in pane1."
    fi
done
