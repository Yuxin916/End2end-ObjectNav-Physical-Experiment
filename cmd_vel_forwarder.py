#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import socket, struct, os

SERVER_SOCK = '/tmp/cmdvel.sock'       # ROS side
CLIENT_SOCK = '/tmp/cmdvel_client.sock'# SDK side (the other proc will bind here)

class Forwarder(Node):
    def __init__(self):
        super().__init__('cmd_vel_forwarder')
        # Create UNIX datagram socket and bind
        try: os.unlink(SERVER_SOCK)
        except FileNotFoundError: pass
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(SERVER_SOCK)
        os.chmod(SERVER_SOCK, 0o666)   # permit other process to sendto()
        self.sub = self.create_subscription(Twist, '/cmd_vel', self.cb, 10)

    def cb(self, msg: Twist):
        # Pack (vx, vy, wz) as doubles
        payload = struct.pack('ddd', msg.linear.x, msg.linear.y, msg.angular.z)
        try:
            self.sock.sendto(payload, CLIENT_SOCK)
        except FileNotFoundError:
            # Client not up yet — ignore quietly
            pass

def main():
    rclpy.init()
    node = Forwarder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
