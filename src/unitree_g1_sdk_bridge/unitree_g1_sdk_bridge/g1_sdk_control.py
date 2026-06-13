#!/usr/bin/env python3
"""
Bridge ROS2 cmd_vel -> Unitree G1 high-level locomotion via unitree_sdk2.

Talks to the robot over DDS on the wired 192.168.123.x network (default iface
eth0). With the container started using --network host, the robot at
192.168.123.161 is directly reachable, so no WebRTC / AES key is needed.

SAFETY MODEL (the robot must NOT move unexpectedly):
  * Starts DISABLED. cmd_vel is received but NOT forwarded until enabled.
  * The robot does not stand on its own -- call the ~/stand_up service first.
  * Velocities are clamped to max_vx / max_vy / max_vyaw.
  * If no cmd_vel arrives within cmd_timeout seconds, zero velocity is sent.
  * ~/damp is a soft e-stop (FSM damping); ~/enable false also zeroes motion.

Services (std_srvs):
  ~/enable   (SetBool)  : true  -> forward cmd_vel ; false -> stop + hold stand
  ~/stand_up (Trigger)  : Damp -> Squat2StandUp (robot stands, still disabled)
  ~/sit      (Trigger)  : StandUp2Squat
  ~/damp     (Trigger)  : soft e-stop -> FSM damping (also disables)
  ~/start    (Trigger)  : enter main operation control (FSM 200)
"""
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger, SetBool

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


# ---------------------------------------------------------------------------
# cyclonedds config patch -- REQUIRED on this G1 firmware under Ubuntu 24.04.
#
# unitree_sdk2py's ChannelConfigHasInterface embeds a <Tracing><Verbosity>
# config</Verbosity></Tracing> block. Under 24.04's _FORTIFY_SOURCE, cyclonedds'
# do_print_uint32_bitset -> __snprintf_chk aborts with
# "*** buffer overflow detected ***" inside ChannelFactoryInitialize, so the
# bridge dies before it ever talks to the robot. The same DDS config WITHOUT the
# <Tracing> block initializes cleanly. We monkeypatch the value the SDK reads
# (channel module global, keeping the $__IF_NAME__$ placeholder) before init.
# ---------------------------------------------------------------------------
_CYCLONEDDS_CONFIG_NO_TRACING = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
                </Interfaces>
            </General>
        </Domain>
    </CycloneDDS>'''


def _patch_cyclonedds_tracing():
    """Strip the SDK's <Tracing> DDS config so ChannelFactoryInitialize won't
    abort with a fortify buffer-overflow on Ubuntu 24.04. Idempotent."""
    import unitree_sdk2py.core.channel as _ch
    _ch.ChannelConfigHasInterface = _CYCLONEDDS_CONFIG_NO_TRACING


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class G1SdkBridge(Node):
    def __init__(self):
        super().__init__('g1_sdk_bridge')

        self.declare_parameter('network_interface', 'eth0')
        self.declare_parameter('max_vx', 0.6)      # m/s forward/back
        self.declare_parameter('max_vy', 0.4)      # m/s lateral
        self.declare_parameter('max_vyaw', 0.8)    # rad/s yaw
        self.declare_parameter('cmd_timeout', 0.5) # s; zero motion if stale
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('enable_on_start', False)

        iface = self.get_parameter('network_interface').value
        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_vy = float(self.get_parameter('max_vy').value)
        self.max_vyaw = float(self.get_parameter('max_vyaw').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        rate = float(self.get_parameter('control_rate').value)

        self._lock = threading.Lock()
        self.enabled = bool(self.get_parameter('enable_on_start').value)
        self.vx = self.vy = self.vyaw = 0.0
        self.last_cmd_t = self.get_clock().now()

        # ---- unitree_sdk2 channel + loco client ----
        self.get_logger().info(f'Initializing unitree_sdk2 DDS on interface "{iface}" ...')
        _patch_cyclonedds_tracing()  # must run before ChannelFactoryInitialize on 24.04
        ChannelFactoryInitialize(0, iface)
        self.client = LocoClient()
        self.client.SetTimeout(10.0)
        self.client.Init()
        self.get_logger().info('LocoClient ready. Bridge is DISABLED until ~/enable true.')

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

    # ---- cmd_vel ----
    def cmd_cb(self, msg: TwistStamped):
        with self._lock:
            self.vx = clamp(msg.twist.linear.x, -self.max_vx, self.max_vx)
            self.vy = clamp(msg.twist.linear.y, -self.max_vy, self.max_vy)
            self.vyaw = clamp(msg.twist.angular.z, -self.max_vyaw, self.max_vyaw)
            self.last_cmd_t = self.get_clock().now()

    def tick(self):
        with self._lock:
            if not self.enabled:
                return
            stale = (self.get_clock().now() - self.last_cmd_t).nanoseconds > self.cmd_timeout * 1e9
            vx, vy, vyaw = (0.0, 0.0, 0.0) if stale else (self.vx, self.vy, self.vyaw)
        try:
            self.client.Move(vx, vy, vyaw)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'Move failed: {e}')

    # ---- services ----
    def srv_enable(self, req, resp):
        with self._lock:
            self.enabled = req.data
            if not self.enabled:
                try:
                    self.client.Move(0.0, 0.0, 0.0)
                except Exception:  # noqa: BLE001
                    pass
        resp.success = True
        resp.message = f'cmd_vel forwarding {"ENABLED" if req.data else "DISABLED"}'
        self.get_logger().warn(resp.message)
        return resp

    def srv_stand_up(self, req, resp):
        import time
        self.get_logger().warn('stand_up: Damp -> Squat2StandUp')
        try:
            self.client.Damp()
            time.sleep(0.5)
            self.client.Squat2StandUp()
            resp.success, resp.message = True, 'standing up'
        except Exception as e:  # noqa: BLE001
            resp.success, resp.message = False, str(e)
        return resp

    def srv_sit(self, req, resp):
        try:
            self.client.StandUp2Squat()
            resp.success, resp.message = True, 'sitting / squatting'
        except Exception as e:  # noqa: BLE001
            resp.success, resp.message = False, str(e)
        return resp

    def srv_damp(self, req, resp):
        with self._lock:
            self.enabled = False
        try:
            self.client.Damp()
            resp.success, resp.message = True, 'DAMPING (soft e-stop), bridge disabled'
        except Exception as e:  # noqa: BLE001
            resp.success, resp.message = False, str(e)
        self.get_logger().warn(resp.message)
        return resp

    def srv_start(self, req, resp):
        try:
            self.client.Start()
            resp.success, resp.message = True, 'main operation control (FSM 200)'
        except Exception as e:  # noqa: BLE001
            resp.success, resp.message = False, str(e)
        return resp


def main():
    rclpy.init()
    node = G1SdkBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.client.Damp()  # leave the robot in a safe damping state
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
