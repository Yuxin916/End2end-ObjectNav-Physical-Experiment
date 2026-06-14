#!/usr/bin/env python3
"""Bridge ROS2 cmd_vel -> Unitree G1 high-level locomotion (ROS-facing half).

This node owns NO unitree LocoClient. unitree_sdk2's cyclonedds runtime cannot
live in the same process as rmw_cyclonedds_cpp (the ROS stack now uses cyclonedds
for SLAM stability) -- the pair segfaults at LocoClient() construction (verified
2026-06-14). So the actual SDK control runs in a separate rclpy-free child
process, g1_loco_driver.py, which this node spawns and feeds over localhost UDP.

SAFETY MODEL (unchanged; enforced in BOTH halves):
  * Starts DISABLED. cmd_vel is received but NOT forwarded until ~/enable true.
  * Velocities are clamped to max_vx / max_vy / max_vyaw (here AND in the driver).
  * If no cmd_vel arrives within cmd_timeout -> zero velocity. If the ROS node
    dies, the driver stops receiving packets and zeroes/Damps on its own.
  * ~/damp is a soft e-stop; ~/enable false also zeroes motion.

Services (std_srvs):
  ~/enable   (SetBool)  : true  -> forward cmd_vel ; false -> stop + hold
  ~/stand_up (Trigger)  : Damp -> Squat2StandUp (robot stands, still disabled)
  ~/sit      (Trigger)  : StandUp2Squat
  ~/damp     (Trigger)  : soft e-stop -> FSM damping (also disables)
  ~/start    (Trigger)  : enter main operation control
"""
import os
import sys
import json
import socket
import atexit
import threading
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger, SetBool


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class G1SdkBridge(Node):
    def __init__(self):
        super().__init__('g1_sdk_bridge')

        self.declare_parameter('network_interface', 'eth0')
        self.declare_parameter('max_vx', 0.6)
        self.declare_parameter('max_vy', 0.4)
        self.declare_parameter('max_vyaw', 0.8)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('enable_on_start', False)
        self.declare_parameter('driver_port', 43897)

        iface = self.get_parameter('network_interface').value
        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_vy = float(self.get_parameter('max_vy').value)
        self.max_vyaw = float(self.get_parameter('max_vyaw').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        rate = float(self.get_parameter('control_rate').value)
        port = int(self.get_parameter('driver_port').value)

        self._lock = threading.Lock()
        self.enabled = bool(self.get_parameter('enable_on_start').value)
        self.vx = self.vy = self.vyaw = 0.0
        self.last_cmd_t = self.get_clock().now()

        # ---- UDP link to the rclpy-free SDK driver child ----
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(('127.0.0.1', 0))          # ephemeral; receives acks
        self._sock.settimeout(0.2)
        self._driver_addr = ('127.0.0.1', port)
        self._acks = {}
        self._ack_lock = threading.Lock()
        self._cmd_id = 0

        driver = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'g1_loco_driver.py')
        if not os.path.exists(driver):
            raise FileNotFoundError('SDK driver not found: %s' % driver)
        self.get_logger().info('Spawning SDK driver %s (iface=%s udp=%d)'
                               % (driver, iface, port))
        self._child = subprocess.Popen(
            [sys.executable, driver, str(iface), str(port),
             str(self.max_vx), str(self.max_vy), str(self.max_vyaw),
             str(self.cmd_timeout), str(rate)])
        atexit.register(self._kill_child)

        threading.Thread(target=self._recv_acks, daemon=True).start()

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(TwistStamped, 'cmd_vel', self.cmd_cb, qos)

        self.create_service(SetBool, '~/enable', self.srv_enable)
        self.create_service(Trigger, '~/stand_up', self.srv_stand_up)
        self.create_service(Trigger, '~/sit', self.srv_sit)
        self.create_service(Trigger, '~/damp', self.srv_damp)
        self.create_service(Trigger, '~/start', self.srv_start)

        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().warn('Bridge up (driver PID %d). DISABLED until ~/enable true.'
                               % self._child.pid)

    # ---- child lifecycle ----
    def _kill_child(self):
        try:
            if getattr(self, '_child', None) and self._child.poll() is None:
                self._child.terminate()
                try:
                    self._child.wait(timeout=3)
                except Exception:  # noqa: BLE001
                    self._child.kill()
        except Exception:  # noqa: BLE001
            pass

    # ---- UDP ----
    def _send(self, obj):
        try:
            self._sock.sendto(json.dumps(obj).encode(), self._driver_addr)
        except OSError as e:
            self.get_logger().error('UDP send failed: %s' % e)

    def _recv_acks(self):
        while True:
            try:
                data, _addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode())
            except Exception:  # noqa: BLE001
                continue
            if msg.get('t') == 'ack':
                with self._ack_lock:
                    box = self._acks.get(msg.get('id'))
                if box is not None:
                    box['resp'] = msg
                    box['ev'].set()

    def _send_cmd(self, name, timeout=12.0):
        with self._ack_lock:
            self._cmd_id += 1
            cid = self._cmd_id
            ev = threading.Event()
            self._acks[cid] = {'ev': ev, 'resp': None}
        self._send({'t': 'cmd', 'id': cid, 'name': name})
        got = ev.wait(timeout)
        with self._ack_lock:
            box = self._acks.pop(cid, None)
        if not got or not box or not box['resp']:
            return False, '%s: no ack from SDK driver (timeout)' % name
        r = box['resp']
        return bool(r.get('ok')), str(r.get('msg'))

    # ---- cmd_vel ----
    def cmd_cb(self, msg: TwistStamped):
        with self._lock:
            self.vx = clamp(msg.twist.linear.x, -self.max_vx, self.max_vx)
            self.vy = clamp(msg.twist.linear.y, -self.max_vy, self.max_vy)
            self.vyaw = clamp(msg.twist.angular.z, -self.max_vyaw, self.max_vyaw)
            self.last_cmd_t = self.get_clock().now()

    def tick(self):
        if self._child.poll() is not None:
            with self._lock:
                was = self.enabled
                self.enabled = False
            if was:
                self.get_logger().error('SDK driver child died -- bridge DISABLED.')
        with self._lock:
            en = self.enabled
            stale = (self.get_clock().now() - self.last_cmd_t).nanoseconds > self.cmd_timeout * 1e9
            vx, vy, vyaw = (0.0, 0.0, 0.0) if stale else (self.vx, self.vy, self.vyaw)
        self._send({'t': 'vel', 'vx': vx, 'vy': vy, 'vyaw': vyaw, 'en': en})

    # ---- services ----
    def srv_enable(self, req, resp):
        with self._lock:
            self.enabled = req.data
        self._send({'t': 'vel', 'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0, 'en': req.data})
        resp.success = True
        resp.message = 'cmd_vel forwarding %s' % ('ENABLED' if req.data else 'DISABLED')
        self.get_logger().warn(resp.message)
        return resp

    def srv_stand_up(self, req, resp):
        resp.success, resp.message = self._send_cmd('stand_up')
        return resp

    def srv_sit(self, req, resp):
        resp.success, resp.message = self._send_cmd('sit')
        return resp

    def srv_damp(self, req, resp):
        with self._lock:
            self.enabled = False
        resp.success, resp.message = self._send_cmd('damp')
        self.get_logger().warn(resp.message)
        return resp

    def srv_start(self, req, resp):
        resp.success, resp.message = self._send_cmd('start')
        return resp


def main():
    rclpy.init()
    node = G1SdkBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._kill_child()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
