import serial
import time
import math
from contextlib import AbstractContextManager

def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

class RobotControl(AbstractContextManager):
    """
    Text-protocol robot controller over serial (Arduino SBUS bridge).
    - Commands like: 'move:STRAFE,FORWARD,TURN,PITCH\\n', 'unlock\\n', etc.
    - Maps normalized axes [-1, 1] -> SBUS [172, 1811] with center 992.
    """

    # Canonical SBUS rail values (Futaba/FrSky style)
    SBUS_MIN = 172
    SBUS_MID = 992
    SBUS_MAX = 1811

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        read_timeout: float = 0.05,
        write_timeout: float = 0.05,
        dtr_reset_sleep: float = 2.0,
        dead_zone: float = 0.06,           # ~6% stick dead-zone
        invert_forward: bool = True,       # many rigs use -Y as "forward"
        invert_strafe: bool = False,
        invert_turn: bool = False,
        invert_pitch: bool = False,
        heartbeat_hz: float | None = None  # e.g. 5.0 to periodically send MID hold
    ):
        self.port = port
        self.baudrate = baudrate
        self.dead_zone = float(dead_zone)
        self.axis_sign = {
            "strafe": -1.0 if invert_strafe else 1.0,
            "forward": -1.0 if invert_forward else 1.0,
            "turn": -1.0 if invert_turn else 1.0,
            "pitch": -1.0 if invert_pitch else 1.0,
        }
        self.heartbeat_hz = heartbeat_hz
        self._last_tx = 0.0
        self._ser = None

        # Open serial robustly
        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=read_timeout,
            write_timeout=write_timeout,
            dsrdtr=False,
            xonxoff=False,
            rtscts=False,
        )
        # Typical Arduino resets on DTR toggle; give it time
        time.sleep(dtr_reset_sleep)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    # ---- Helpers ----------------------------------------------------------

    @staticmethod
    def _apply_dead_zone(x: float, dz: float) -> float:
        """Center dead-zone, remap remaining range back to [-1, 1]."""
        x = _clamp(x, -1.0, 1.0)
        dz = abs(dz)
        if dz <= 0.0:
            return x
        if abs(x) <= dz:
            return 0.0
        # Rescale so that |x| in [dz, 1] -> [0, 1]
        scale = (abs(x) - dz) / (1.0 - dz)
        return math.copysign(scale, x)

    @classmethod
    def _axis_to_sbus(cls, x: float) -> int:
        """
        Map normalized axis [-1, 1] to SBUS [MIN, MAX] with MID at 992.
        Linear mapping: x=-1 -> MIN, 0 -> MID, +1 -> MAX.
        """
        x = _clamp(x, -1.0, 1.0)
        if x >= 0:
            val = cls.SBUS_MID + x * (cls.SBUS_MAX - cls.SBUS_MID)
        else:
            val = cls.SBUS_MID + x * (cls.SBUS_MID - cls.SBUS_MIN)
        return int(_clamp(round(val), cls.SBUS_MIN, cls.SBUS_MAX))

    def _axis_pipeline(self, name: str, value: float) -> int:
        signed = self.axis_sign[name] * float(value)
        shaped = self._apply_dead_zone(signed, self.dead_zone)
        return self._axis_to_sbus(shaped)

    def _send(self, line: str):
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("Serial port is not open.")
        # Always newline-terminate; Arduino side should read until '\n'
        payload = (line.rstrip("\r\n") + "\n").encode("ascii", errors="strict")
        self._ser.write(payload)
        self._ser.flush()  # honor write_timeout; keep buffers clean
        self._last_tx = time.monotonic()

    # ---- Public API -------------------------------------------------------

    def send_command(self, cmd: str):
        """Send a raw single-word command (e.g., 'unlock', 'recover')."""
        self._send(cmd)

    def move(self, strafe: float, forward: float, turn: float, pitch: float = 0.0):
        """Send movement command using normalized axes in [-1, 1]."""
        sv = self._axis_pipeline("strafe", strafe)
        fv = self._axis_pipeline("forward", forward)
        tv = self._axis_pipeline("turn", turn)
        pv = self._axis_pipeline("pitch", pitch)
        self._send(f"move:{sv},{fv},{tv},{pv}")

    # Convenience actions (keeping your names)
    def unlock(self): self.send_command("unlock")
    def toggle_damping(self): self.send_command("damping")
    def recover(self): self.send_command("recover")
    def toggle_continuous_movement(self): self.send_command("continuous")
    def toggle_stand_height(self): self.send_command("stand")
    def toggle_obstacle_avoidance(self): self.send_command("obstacle")
    def switch_gait(self): self.send_command("gait")
    def toggle_light(self): self.send_command("light")
    def dance(self): self.send_command("dance")
    def jump(self): self.send_command("jump")

    def stop(self):
        """Send neutral channels (all mid)."""
        self._send(f"move:{self.SBUS_MID},{self.SBUS_MID},{self.SBUS_MID},{self.SBUS_MID}")

    def close(self):
        """Graceful stop + close."""
        try:
            if self._ser and self._ser.is_open:
                # Best-effort neutral
                self.stop()
                time.sleep(0.02)
        finally:
            if self._ser and self._ser.is_open:
                self._ser.close()

    # Context manager support
    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # don't suppress exceptions

    # Optional: keep-alive/heartbeat you can call in your loop
    def maybe_heartbeat(self):
        """If heartbeat_hz set, ensure at least MID command at that rate."""
        if not self.heartbeat_hz or self.heartbeat_hz <= 0:
            return
        interval = 1.0 / self.heartbeat_hz
        now = time.monotonic()
        if now - self._last_tx >= interval:
            self.stop()

def run_for(duration_s: float, rate_hz: float, step_fn):
    """
    Fixed-rate loop helper. Calls step_fn() each tick for duration_s seconds.
    """
    period = 1.0 / rate_hz
    t0 = time.monotonic()
    next_t = t0
    while True:
        now = time.monotonic()
        if now - t0 >= duration_s:
            break
        step_fn()
        next_t += period
        sleep_time = next_t - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

def main():
    robot = None
    try:
        with RobotControl(
            port="/dev/ttyACM0",
            baudrate=115200,
            dead_zone=0.08,          # tune as you like
            invert_forward=True,     # your code used -0.5 for "forward"
            heartbeat_hz=5.0         # optional safety keep-alive
        ) as robot:

            print("Unlocking robot...")
            robot.unlock()
            time.sleep(1.0)

            print("Moving forward & turning (2s @ 20 Hz)...")
            def step():
                # 50% forward, 50% turn-right (matching your intent)
                robot.move(strafe=0.0, forward=+0.5, turn=+0.5, pitch=0.0)
                robot.maybe_heartbeat()

            run_for(duration_s=2.0, rate_hz=20.0, step_fn=step)

            print("Stopping...")
            robot.stop()
            time.sleep(0.2)

            # Example: toggle features if you want
            # robot.switch_gait()
            # time.sleep(0.5)
            # robot.toggle_light()

    except KeyboardInterrupt:
        print("\n[Ctrl-C] Stopping robot...")
        try:
            if robot:
                robot.stop()
        except Exception:
            pass
    except (serial.SerialException, OSError) as e:
        print(f"[Serial Error] {e}")
    finally:
        if robot:
            robot.close()

if __name__ == "__main__":
    main()

