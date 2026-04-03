#!/usr/bin/env python3

import asyncio
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)


class G1ControlNode(Node):
    """ROS 2 node for Unitree G1 robot control via WebRTC."""

    def __init__(self):
        super().__init__('g1_control')

        # Parameters
        self.declare_parameter('robot_ip', '10.0.0.191')
        self.declare_parameter('connection_method', 'LocalSTA')
        self.declare_parameter('control_mode', 'wireless_controller')  # or 'sport_cmd'

        self.robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        connection_method_str = self.get_parameter('connection_method').get_parameter_value().string_value
        self.control_mode = self.get_parameter('control_mode').get_parameter_value().string_value

        connection_method_map = {
            'LocalAP': WebRTCConnectionMethod.LocalAP,
            'LocalSTA': WebRTCConnectionMethod.LocalSTA,
            'Remote': WebRTCConnectionMethod.Remote,
        }
        self.connection_method = connection_method_map.get(
            connection_method_str, WebRTCConnectionMethod.LocalSTA
        )

        self.get_logger().info(f'Connecting to G1 at {self.robot_ip} using {connection_method_str}')
        self.get_logger().info(f'Control mode: {self.control_mode}')

        # Async / connection stuff
        self.conn = None
        self.loop = None
        self.connected = threading.Event()
        self.connection_thread = threading.Thread(
            target=self._connection_worker, daemon=True
        )
        self.connection_thread.start()

        if not self.connected.wait(timeout=30.0):
            self.get_logger().error('Failed to connect to G1 within timeout')
            raise RuntimeError('Connection timeout')

        self.get_logger().info('Successfully connected to G1')

        # QoS for cmd_vel
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # cmd_vel subscriber
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped,
            'cmd_vel',
            self.cmd_vel_callback,
            qos_profile,
        )

        # Services for arm gestures
        self.handshake_srv = self.create_service(Trigger, 'g1/handshake', self.handshake_callback)
        self.highfive_srv = self.create_service(Trigger, 'g1/high_five', self.high_five_callback)
        self.hug_srv = self.create_service(Trigger, 'g1/hug', self.hug_callback)
        self.cancel_arm_srv = self.create_service(Trigger, 'g1/cancel_arm_action', self.cancel_arm_callback)

        # Services for locomotion modes
        self.walk_mode_srv = self.create_service(Trigger, 'g1/mode_walk', self.mode_walk_callback)
        self.run_mode_srv = self.create_service(Trigger, 'g1/mode_run', self.mode_run_callback)

        self.get_logger().info('G1 control node started')
        self.get_logger().info('Subscribed to: cmd_vel')
        self.get_logger().info('Services: '
                               'g1/handshake, g1/high_five, g1/hug, '
                               'g1/cancel_arm_action, g1/mode_walk, g1/mode_run')

    # --- Connection worker -------------------------------------------------

    def _connection_worker(self):
        """Background thread for asyncio event loop and WebRTC connection."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.conn = UnitreeWebRTCConnection(
                self.connection_method,
                ip=self.robot_ip,
            )
            self.loop.run_until_complete(self.conn.connect())
            self.connected.set()
            self.loop.run_forever()
        except Exception as e:
            self.get_logger().error(f'Connection error: {e}')
            self.connected.set()
        finally:
            self.loop.close()

    # --- cmd_vel -> wirelesscontroller or sport_cmd -----------------------

    def cmd_vel_callback(self, msg: TwistStamped):
        """Handle incoming cmd_vel messages."""
        if not self.conn or not self.loop:
            self.get_logger().warn('Connection not ready, ignoring cmd_vel')
            return

        x = msg.twist.linear.x
        y = msg.twist.linear.y
        yaw = msg.twist.angular.z

        if self.control_mode == 'wireless_controller':
            # Map ROS -> virtual gamepad
            # lx: +right, -left  (ROS y)
            # ly: +forward, -back (ROS x)
            # rx: +turn right, -turn left (ROS yaw)
            lx = -y
            ly = x
            rx = -yaw
            ry = 0.0

            async def async_move():
                self.conn.datachannel.pub_sub.publish_without_callback(
                    "rt/wirelesscontroller",
                    {
                        "lx": lx,
                        "ly": ly,
                        "rx": rx,
                        "ry": ry,
                        "keys": 0,
                    },
                )
        else:  # 'sport_cmd' – using sport move API (if supported similarly)
            async def async_move():
                await self.conn.datachannel.pub_sub.publish_request_new(
                    "rt/api/sport/request",
                    {
                        "api_id": 7102,  # example Move API for G1; adjust if docs differ
                        "parameter": {"x": x, "y": y, "z": yaw},
                    },
                )

        try:
            future = asyncio.run_coroutine_threadsafe(async_move(), self.loop)
            if self.control_mode != 'wireless_controller':
                future.result(timeout=0.5)
        except Exception as e:
            self.get_logger().error(f'Failed to send cmd_vel: {e}')

    # --- Helper: send arm action via 7106 ---------------------------------

    def _arm_action(self, name: str, action_id: int) -> Trigger.Response:
        resp = Trigger.Response()

        if not self.conn or not self.loop:
            resp.success = False
            resp.message = 'Connection not ready'
            return resp

        async def async_cmd():
            await self.conn.datachannel.pub_sub.publish_request_new(
                "rt/api/arm/request",
                {
                    "api_id": 7106,
                    "parameter": {"data": action_id},
                },
            )

        try:
            future = asyncio.run_coroutine_threadsafe(async_cmd(), self.loop)
            future.result(timeout=5.0)
            resp.success = True
            resp.message = f'{name} command sent'
            self.get_logger().info(resp.message)
        except Exception as e:
            resp.success = False
            resp.message = f'Failed to execute {name}: {e}'
            self.get_logger().error(resp.message)

        return resp

    # Arm gesture services (IDs from your example comments)
    def handshake_callback(self, request, response):
        return self._arm_action('Handshake', 27)

    def high_five_callback(self, request, response):
        return self._arm_action('HighFive', 18)

    def hug_callback(self, request, response):
        return self._arm_action('Hug', 19)

    def cancel_arm_callback(self, request, response):
        return self._arm_action('CancelArm', 99)

    # --- Locomotion mode services (7101) ----------------------------------

    def _mode_action(self, name: str, mode_code: int) -> Trigger.Response:
        resp = Trigger.Response()

        if not self.conn or not self.loop:
            resp.success = False
            resp.message = 'Connection not ready'
            return resp

        async def async_cmd():
            await self.conn.datachannel.pub_sub.publish_request_new(
                "rt/api/sport/request",
                {
                    "api_id": 7101,  # set locomotion mode
                    "parameter": {"data": mode_code},
                },
            )

        try:
            future = asyncio.run_coroutine_threadsafe(async_cmd(), self.loop)
            future.result(timeout=5.0)
            resp.success = True
            resp.message = f'{name} mode command sent'
            self.get_logger().info(resp.message)
        except Exception as e:
            resp.success = False
            resp.message = f'Failed to execute {name}: {e}'
            self.get_logger().error(resp.message)

        return resp

    def mode_walk_callback(self, request, response):
        # 500 - Walk
        return self._mode_action('Walk', 500)

    def mode_run_callback(self, request, response):
        # 801 - Run
        return self._mode_action('Run', 801)

    # --- Shutdown ----------------------------------------------------------

    def destroy_node(self):
        self.get_logger().info('Shutting down G1 control node...')
        if self.conn and self.loop:
            try:
                async def async_disconnect():
                    await self.conn.disconnect()
                asyncio.run_coroutine_threadsafe(async_disconnect(), self.loop)
            except Exception as e:
                self.get_logger().error(f'Error during disconnect: {e}')

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.connection_thread.is_alive():
            self.connection_thread.join(timeout=2.0)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = G1ControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
