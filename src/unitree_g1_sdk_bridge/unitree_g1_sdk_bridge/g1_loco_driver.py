#!/usr/bin/env python3
"""Standalone Unitree G1 loco driver -- the SDK-facing half of the bridge.

Runs as a child process of g1_sdk_control (the ROS node) and is deliberately
rclpy-FREE: unitree_sdk2's bundled cyclonedds cannot share a process with
rmw_cyclonedds_cpp -- the combination segfaults at LocoClient() construction
(verified 2026-06-14). Keeping the SDK in its own process sidesteps that while
the ROS stack keeps using cyclonedds (needed for SLAM stability).

It receives control over localhost UDP and owns ALL robot safety:
  * starts DISABLED; only Moves while 'en' is true,
  * clamps velocities,
  * if the ROS parent stops feeding packets (> cmd_timeout) -> zero velocity,
  * on SIGINT/SIGTERM/exit -> Damp.

argv: <iface> <udp_port> <max_vx> <max_vy> <max_vyaw> <cmd_timeout> <rate>
"""
import sys
import json
import time
import socket
import signal
import threading

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
    import unitree_sdk2py.core.channel as _ch
    _ch.ChannelConfigHasInterface = _CYCLONEDDS_CONFIG_NO_TRACING


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    iface       = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    port        = int(sys.argv[2]) if len(sys.argv) > 2 else 43897
    max_vx      = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    max_vy      = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
    max_vyaw    = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
    cmd_timeout = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    rate        = float(sys.argv[7]) if len(sys.argv) > 7 else 50.0

    _patch_cyclonedds_tracing()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print('[loco_driver] DDS init on "%s" ...' % iface, flush=True)
    ChannelFactoryInitialize(0, iface)
    client = LocoClient()
    client.SetTimeout(10.0)
    client.Init()
    print("[loco_driver] LocoClient ready. DISABLED until parent sends en=true.", flush=True)

    state = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "en": False, "last": 0.0}
    st_lock = threading.Lock()        # guards state dict
    cl_lock = threading.Lock()        # serializes every LocoClient call
    stop = {"flag": False}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.2)

    def do_cmd(name):
        try:
            with cl_lock:
                if name == "stand_up":
                    client.Damp(); time.sleep(0.5); client.Squat2StandUp()
                    return True, "stand_up: Damp -> Squat2StandUp"
                if name == "sit":
                    client.StandUp2Squat();  return True, "sit: StandUp2Squat"
                if name == "damp":
                    client.Damp();           return True, "DAMPING (soft e-stop)"
                if name == "start":
                    client.Start();          return True, "start: Start() (FSM main op)"
            return False, "unknown cmd: %s" % name
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    def recv_loop():
        while not stop["flag"]:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode())
            except Exception:  # noqa: BLE001
                continue
            t = msg.get("t")
            if t == "vel":
                with st_lock:
                    state["vx"]   = clamp(float(msg.get("vx", 0.0)),  -max_vx,   max_vx)
                    state["vy"]   = clamp(float(msg.get("vy", 0.0)),  -max_vy,   max_vy)
                    state["vyaw"] = clamp(float(msg.get("vyaw", 0.0)),-max_vyaw, max_vyaw)
                    state["en"]   = bool(msg.get("en", False))
                    state["last"] = time.monotonic()
            elif t == "cmd":
                if msg.get("name") == "damp":
                    with st_lock:
                        state["en"] = False
                ok, m = do_cmd(msg.get("name", ""))
                try:
                    sock.sendto(json.dumps({"t": "ack", "id": msg.get("id"),
                                            "ok": ok, "msg": m}).encode(), addr)
                except OSError:
                    pass

    threading.Thread(target=recv_loop, daemon=True).start()

    def shutdown(*_a):
        stop["flag"] = True
        try:
            with cl_lock:
                client.Move(0.0, 0.0, 0.0)
                client.Damp()
        except Exception:  # noqa: BLE001
            pass
        print("[loco_driver] shutdown -> Damp", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    period = 1.0 / rate
    prev_en = False
    while not stop["flag"]:
        t0 = time.monotonic()
        with st_lock:
            en, vx, vy, vyaw, last = (state["en"], state["vx"], state["vy"],
                                      state["vyaw"], state["last"])
        stale = (time.monotonic() - last) > cmd_timeout
        try:
            with cl_lock:
                if en:
                    client.Move(0.0, 0.0, 0.0) if stale else client.Move(vx, vy, vyaw)
                elif prev_en:
                    client.Move(0.0, 0.0, 0.0)   # one stop on the enable->disable edge
        except Exception as e:  # noqa: BLE001
            print("[loco_driver] Move failed: %s" % e, flush=True)
        prev_en = en
        dt = period - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)


if __name__ == "__main__":
    main()
