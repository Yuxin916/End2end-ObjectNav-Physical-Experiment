import socket
import struct
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


def recv_exact(conn: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            raise ConnectionError('Socket closed')
        buf.extend(chunk)
    return bytes(buf)


class RSTCPReceiverNode(Node):
    def __init__(self) -> None:
        super().__init__('rs_tcp_receiver')

        self.declare_parameter('listen_ip', '0.0.0.0')
        self.declare_parameter('port', 9999)
        self.declare_parameter('output_topic', '/camera/image')
        self.declare_parameter('frame_id', 'camera')
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('jpeg_decode_mode', 'color')
        self.declare_parameter('reconnect_delay_sec', 1.0)

        self.listen_ip = str(self.get_parameter('listen_ip').value)
        self.port = int(self.get_parameter('port').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.jpeg_decode_mode = str(self.get_parameter('jpeg_decode_mode').value).lower()
        self.reconnect_delay_sec = float(self.get_parameter('reconnect_delay_sec').value)

        pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(Image, self.output_topic, pub_qos)

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_peer: Optional[Tuple[str, int]] = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()

        publish_period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.create_timer(publish_period, self._publish_latest_frame)

        self.get_logger().info(
            f'Listening for RealSense TCP frames on {self.listen_ip}:{self.port}, '
            f'publishing {self.output_topic}'
        )

    def _decode_flag(self) -> int:
        if self.jpeg_decode_mode == 'gray':
            return cv2.IMREAD_GRAYSCALE
        return cv2.IMREAD_COLOR

    def _receiver_loop(self) -> None:
        while not self._stop_event.is_set():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind((self.listen_ip, self.port))
                server.listen(1)
                server.settimeout(1.0)

                while not self._stop_event.is_set():
                    try:
                        conn, addr = server.accept()
                    except socket.timeout:
                        continue

                    self._latest_peer = addr
                    self.get_logger().info(f'TCP camera client connected from {addr[0]}:{addr[1]}')
                    try:
                        self._handle_client(conn)
                    except Exception as exc:
                        self.get_logger().warn(f'TCP camera client disconnected: {exc}')
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
            except Exception as exc:
                self.get_logger().error(f'Failed to start TCP receiver on {self.listen_ip}:{self.port}: {exc}')
                time.sleep(self.reconnect_delay_sec)
            finally:
                try:
                    server.close()
                except Exception:
                    pass

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(None)
        decode_flag = self._decode_flag()
        while not self._stop_event.is_set():
            header = recv_exact(conn, 8)
            msg_len = struct.unpack('!Q', header)[0]
            jpg_data = recv_exact(conn, msg_len)

            arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, decode_flag)
            if frame is None:
                self.get_logger().warn('Dropped undecodable JPEG frame')
                continue

            with self._frame_lock:
                self._latest_frame = frame.copy()

    def _publish_latest_frame(self) -> None:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        if frame is None:
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])

        if frame.ndim == 2:
            msg.encoding = 'mono8'
            msg.step = int(frame.shape[1])
        else:
            msg.encoding = 'bgr8'
            msg.step = int(frame.shape[1] * frame.shape[2])

        msg.is_bigendian = False
        msg.data = frame.tobytes()
        self.publisher.publish(msg)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RSTCPReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
