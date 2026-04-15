import socket
import struct

import cv2
import numpy as np


LISTEN_IP = "0.0.0.0"
PORT = 9999


def recv_exact(conn: socket.socket, size: int) -> bytes:
    buf = b""
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_IP, PORT))
    server.listen(1)

    print(f"[INFO] Listening on {LISTEN_IP}:{PORT}")
    conn, addr = server.accept()
    print(f"[INFO] Client connected from {addr}")

    try:
        while True:
            header = recv_exact(conn, 8)
            msg_len = struct.unpack("!Q", header)[0]

            jpg_data = recv_exact(conn, msg_len)

            arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            cv2.imshow("RealSense from Jetson", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"[ERROR] Receiver crashed: {e}")
    finally:
        conn.close()
        server.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()