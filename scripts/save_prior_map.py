#!/usr/bin/env python3
"""Grab one PointCloud2 message and save it as the 5-column text format
(x y z intensity time) that arise_slam's laser_mapping map_dir loader reads
for local_mode relocalization.

Usage (inside the nav container):
  python3 scripts/save_prior_map.py [topic] [out.txt]
Defaults: topic=/explored_areas, out=/workspace/autonomy_stack/maps/prior_map.txt

/explored_areas holds the whole run's registered scans (0.1 m voxel), so this
can rescue a map at the END of a run even if savePcd wasn't recording.
"""
import osunitree@ubuntu [03:23:45 PM] [~/projects/autonomy_stack] [ntu_g1 *]
-> % ./docker/run.sh ./system_real_robot_g1.sh control_backend:=webrtc robot_ip:=192.168.123.161 connection_method:=LocalSTA control_mode:=wireless_controller localization:=true
import sys
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else '/explored_areas'
    out = sys.argv[2] if len(sys.argv) > 2 else '/workspace/autonomy_stack/maps/prior_map.txt'

    rclpy.init()
    node = Node('save_prior_map')
    got = {}

    def cb(msg):
        got['msg'] = msg

    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    node.create_subscription(PointCloud2, topic, cb, qos)
    # Also try reliable in parallel (publisher QoS unknown)
    qos2 = QoSProfile(depth=1)
    node.create_subscription(PointCloud2, topic, cb, qos2)

    print(f'waiting for one message on {topic} ...', flush=True)
    import time
    t0 = time.time()
    while 'msg' not in got and time.time() - t0 < 30.0:
        rclpy.spin_once(node, timeout_sec=0.5)
    if 'msg' not in got:
        print('TIMEOUT: no cloud received', flush=True)
        sys.exit(1)

    msg = got['msg']
    # Locate x/y/z/intensity field offsets
    off = {f.name: f.offset for f in msg.fields}
    step = msg.point_step
    data = bytes(msg.data)
    n = msg.width * msg.height
    has_int = 'intensity' in off

    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = 0
    with open(out, 'w') as fh:
        for i in range(n):
            b = i * step
            x, y, z = struct.unpack_from('<fff', data, b + off['x'])
            if x != x or y != y or z != z:  # NaN check
                continue
            inten = struct.unpack_from('<f', data, b + off['intensity'])[0] if has_int else 0.0
            fh.write('%f %f %f %f %f\n' % (x, y, z, inten, 0.0))
            kept += 1
    print(f'saved {kept} points -> {out}', flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
