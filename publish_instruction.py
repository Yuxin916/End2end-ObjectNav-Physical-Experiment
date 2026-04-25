#!/usr/bin/env python3

import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class InstructionPublisher(Node):
    def __init__(self) -> None:
        super().__init__('instruction_console_publisher')
        self.publisher = self.create_publisher(String, '/instruction', 10)

    def wait_for_subscribers(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.publisher.get_subscription_count() >= 1:
                return True
        return self.publisher.get_subscription_count() >= 1

    def publish_instruction(
        self,
        instruction: str,
        repeats: int = 5,
        interval_sec: float = 0.2,
    ) -> None:
        msg = String()
        msg.data = instruction
        repeats = max(1, int(repeats))
        interval_sec = max(0.05, float(interval_sec))
        for _ in range(repeats):
            self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(interval_sec)


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: publish_instruction.py "your instruction here"', file=sys.stderr)
        return 1

    instruction = sys.argv[1].strip()
    if not instruction:
        print('Instruction is empty.', file=sys.stderr)
        return 1

    rclpy.init()
    node = InstructionPublisher()
    try:
        if not node.wait_for_subscribers(timeout_sec=10.0):
            print(
                'Timed out waiting for a subscriber on /instruction.',
                file=sys.stderr,
            )
            return 2
        node.publish_instruction(instruction)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
