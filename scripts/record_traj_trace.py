#!/usr/bin/env python3
"""Record the robot's driven path + RViz waypoint clicks during a run.

Writes (append, flushed per line):
  /workspace/autonomy_stack/maps/trace_raw.txt      "t x y yaw"  (>=5 cm gap)
  /workspace/autonomy_stack/maps/waypoints_log.txt  "t x y"      (every /way_point)

Run inside the nav container. Ctrl-C safe; files survive container removal
(maps/ is on the host mount).
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

TRACE = '/workspace/autonomy_stack/maps/trace_raw.txt'
WPLOG = '/workspace/autonomy_stack/maps/waypoints_log.txt'
GAP = 0.05  # m


class Recorder(Node):
    def __init__(self):
        super().__init__('traj_trace_recorder')
        self.t0 = time.time()
        self.last = None
        self.tf = open(TRACE, 'a')
        self.wf = open(WPLOG, 'a')
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self.tf.write(f'# session {stamp}\n')
        self.wf.write(f'# session {stamp}\n')
        self.tf.flush()
        self.wf.flush()
        qos = QoSProfile(depth=10)
        self.create_subscription(Odometry, '/state_estimation', self.odom_cb, qos)
        # RViz waypoint tool publishes reliable; unitree_control homing carrot too.
        self.create_subscription(PointStamped, '/way_point', self.wp_cb, 10)
        self.get_logger().info('recording /state_estimation + /way_point')

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.last is not None:
            lx, ly = self.last
            if math.hypot(p.x - lx, p.y - ly) < GAP:
                return
        self.last = (p.x, p.y)
        self.tf.write(f'{time.time() - self.t0:.2f} {p.x:.4f} {p.y:.4f} {yaw:.4f}\n')
        self.tf.flush()

    def wp_cb(self, msg):
        self.wf.write(f'{time.time() - self.t0:.2f} {msg.point.x:.4f} {msg.point.y:.4f}\n')
        self.wf.flush()


def main():
    rclpy.init()
    rclpy.spin(Recorder())


if __name__ == '__main__':
    main()
