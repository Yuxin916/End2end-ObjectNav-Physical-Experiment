#!/usr/bin/env python3
"""
ROS 2 node for controlling Unitree Go2 robot via WebRTC.
Subscribes to cmd_vel and provides sport mode command services.
"""

import asyncio
import math
import os
import threading
import time

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32, Int8, String
from std_srvs.srv import Trigger

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD


class UnitreeControlNode(Node):
    """ROS 2 node for Unitree Go2 robot control via WebRTC."""

    def __init__(self):
        super().__init__('unitree_control')

        # Declare parameters
        self.declare_parameter('robot_ip', '192.168.8.181')
        self.declare_parameter('connection_method', 'LocalSTA')
        self.declare_parameter('control_mode', 'sport_cmd')  # Options: 'sport_cmd' or 'wireless_controller'
        # Per-device AES-128 key (32 hex chars). Required for the LAN flow on
        # G1 firmware >= 1.5.1 / Go2 >= 1.1.15, where con_notify returns
        # data2 == 3. Empty falls back to the UNITREE_AES_KEY env var. Fetch
        # once via unitree_webrtc_connect.unitree_cloud.fetch_aes_key().
        self.declare_parameter('aes_128_key', '')
        self.declare_parameter('device_type', 'G1')  # 'Go2' or 'G1'
        # Joystick gesture buttons (招手 etc.). The nav stack holds the only
        # WebRTC connection, so the phone app can't trigger gestures -- do it
        # from the same /joy the driver already uses. Rising edge fires the
        # mapped G1 arm action. Format "button:action_id,..."; empty disables.
        # Xbox layout: 0=A 1=B 2=X 3=Y 4=LB 5=RB. LT/RT are the pathFollower
        # autonomy/manual mode switches (axes 2/5) -- never map gestures
        # there, and NOTE: this gamepad ALSO reports the held trigger on
        # buttons[6]/[7] (hybrid trigger-as-button), so 6/7 fire spuriously
        # while driving (robot did the arms-crossed 'reject' mid-walk).
        # Keep 6/7 unmapped. Arm actions: 11=two-hand kiss, 12/13=left/right
        # kiss, 15=hands up, 17=clap, 18=high five, 19=hug, 20=heart,
        # 21=right heart, 22=reject, 23=right hand up, 24=x-ray,
        # 25=face wave, 26=high wave, 27=shake hand, 99=release arm (reset).
        # Default: A=face wave, B=shake hand, X=release arm, Y=high wave,
        #          LB=clap. RB is the head-LED color cycle (see color_button).
        self.declare_parameter('gesture_buttons',
                               '0:25,1:27,2:99,3:26,4:17')
        # While a gesture plays, mute cmd_vel forwarding for this many seconds.
        # 0 (default) disables muting -- G1 arm actions are an upper-body
        # overlay, so it can gesture while walking. >0 makes the robot pause.
        self.declare_parameter('wave_mute_sec', 0.0)
        # --- "Go home" (fixed end pose) ---
        # Rising edge on buttons[home_button] toggles homing: publish the end
        # pose from end_pose_file as /way_point (drive there under the local
        # planner -- hold the autonomy trigger), then rotate in place to the
        # recorded yaw and stop. Requires localization:=true so the map frame
        # matches the recorded pose. -1 disables. Default 10 = right stick click.
        self.declare_parameter('home_button', 10)
        self.declare_parameter('end_pose_file', '/workspace/autonomy_stack/maps/end_pose.yaml')
        self.declare_parameter('goal_tolerance', 0.35)     # m: switch to fine-align phase
        self.declare_parameter('yaw_tolerance', 0.08)      # rad (~5 deg): done
        self.declare_parameter('homing_yaw_rate', 0.5)     # rad/s cap for final rotation
        # The local planner only plans nearby -- feed it a moving carrot
        # waypoint at most this far ahead along the straight line to the goal.
        self.declare_parameter('waypoint_step', 3.0)       # m
        # Autonomy speed while homing/retracing, as a fraction of pathFollower
        # maxSpeed. It rides on the synthetic /joy axes[4] (joySpeed) -- which
        # also blocks the /speed topic override while nonzero. 07-10 field
        # test: 0.5 looked too slow on stage -> back to 1.0; the near-goal
        # taper + fine-align keep the stop clean. Drop to ~0.8 if the
        # crab-walk / S-wobble reappears at full stick.
        self.declare_parameter('homing_speed', 1.0)
        # Near-goal approach: taper joySpeed from homing_speed down to
        # approach_speed inside slowdown_dis of the real goal (pathFollower's
        # slowDwnDisThre only sees the moving carrot, not the goal).
        self.declare_parameter('slowdown_dis', 2.0)        # m
        self.declare_parameter('approach_speed', 0.3)      # joySpeed floor near goal
        # Fine-align: inside goal_tolerance, bypass planner+follower (their
        # cmd_vel is dropped) and P-servo the map-frame position error with
        # small omni steps, then rotate to the recorded yaw.
        self.declare_parameter('fine_tolerance', 0.10)     # m: accept position
        self.declare_parameter('fine_speed', 0.2)          # cap for fine steps
        self.declare_parameter('fine_timeout', 15.0)       # s: give up -> rotate
        # --- taught trajectory (recorded fixed route) ---
        # The straight-line carrot walks INTO anything sitting between the
        # robot and the goal. If traj_file exists, homing follows this
        # recorded polyline instead (and return follows it reversed) -- the
        # route goes around obstacles the way the operator drove it.
        # Record (under localization:=true so the frame matches the map):
        #   ros2 service call /traj_record_start std_srvs/srv/Trigger
        #   ... joystick-drive the ideal route start -> stage center ...
        #   ros2 service call /traj_record_stop std_srvs/srv/Trigger
        # Stop saves the file and activates it immediately.
        self.declare_parameter('traj_file', '/workspace/autonomy_stack/maps/stage_traj.txt')
        self.declare_parameter('traj_lookahead', 1.5)      # m: carrot ahead along the route
        self.declare_parameter('traj_min_gap', 0.2)        # m between recorded breadcrumbs
        # --- "Return to start" (回起点) ---
        # Rising edge on buttons[return_button] drives back to where this run
        # started (first odom fix), with the same carrot guidance as homing.
        # Works in any frame (no prior map needed). -1 disables.
        # Default 9 = left stick click.
        self.declare_parameter('return_button', 9)
        # --- "Perform" (演讲模式) ---
        # Optional: after Homing DONE, cycle through arm actions every
        # perform_period seconds until homing/return is pressed again (which
        # releases the arms and drives off). DISABLED by default (empty) --
        # the operator triggers gestures manually on the joystick instead.
        # Enable with e.g. perform_actions:='23,15,26,20'.
        self.declare_parameter('perform_actions', '')
        self.declare_parameter('perform_period', 6.0)      # s between actions
        # --- head LED color ---
        # G1's head RGB light is set through the 'voice' service over the same
        # WebRTC connection the nav stack already owns (topic
        # rt/api/voice/request, api_id 1010 SET_RGB_LED, parameter
        # {"R","G","B"} each 0-255) -- verified 2026-07-16, response code 0.
        # A rising edge on buttons[color_button] steps to the next palette
        # entry; the same step is exposed as the /cycle_led service (for RViz /
        # CLI). -1 disables. Default 5 = RB (was the heart gesture).
        self.declare_parameter('color_button', 5)
        # Palette to cycle through: "name:R,G,B;name:R,G,B;..." (0-255 each).
        self.declare_parameter('color_palette',
                               'red:255,0,0;green:0,255,0;blue:0,0,255;'
                               'yellow:255,255,0;cyan:0,255,255;'
                               'purple:160,0,255;white:255,255,255')
        # Blink mode: while active, step to the next palette color every
        # blink_period seconds (a festive rainbow flash). Toggled by the
        # /led_blink_start and /led_blink_stop services (RViz / CLI).
        self.declare_parameter('blink_period', 0.4)        # s between colors
        # --- RViz control surface ---
        # Primary: the docked g1_rviz_panel plugin publishes a command string
        # on /g1_panel_cmd; this node dispatches it to the same actions as the
        # joystick (see _run_panel_cmd). The joystick still works in parallel.
        self.declare_parameter('panel_cmd_topic', 'g1_panel_cmd')
        # Optional legacy: a column of interactive-marker BUTTONs floating in
        # the 3D scene (add an "InteractiveMarkers" display on /g1_buttons/
        # update). OFF by default -- they need panel_frame TF (only after
        # localization locks) and are easy to lose. The docked panel is better.
        self.declare_parameter('rviz_buttons', False)
        # Button label language. RViz2's built-in 3D font (Liberation Sans) has
        # NO CJK glyphs, so 'zh' labels may show as boxes on some builds --
        # switch to 'pinyin' or 'en' with panel_lang:=pinyin if that happens.
        self.declare_parameter('panel_lang', 'zh')        # 'zh' | 'pinyin' | 'en'
        self.declare_parameter('panel_frame', 'vehicle')
        self.declare_parameter('panel_x', 0.0)             # m, in panel_frame
        self.declare_parameter('panel_y', 1.2)             # m to the robot's left
        self.declare_parameter('panel_z_top', 1.6)         # m, top button height
        self.declare_parameter('panel_dz', 0.32)           # m vertical pitch
        self.declare_parameter('panel_scale', 0.28)        # m button half-size

        # Get parameters
        self.robot_ip = self.get_parameter('robot_ip').value
        connection_method_str = self.get_parameter('connection_method').value
        self.control_mode = self.get_parameter('control_mode').value
        self.device_type = self.get_parameter('device_type').value
        self.aes_128_key = self.get_parameter('aes_128_key').value or os.environ.get('UNITREE_AES_KEY', '')
        self.wave_mute_sec = float(self.get_parameter('wave_mute_sec').value)
        # Parse "button:action_id,..." -> {button_index: action_id}
        self.gesture_map = {}
        spec = str(self.get_parameter('gesture_buttons').value).strip()
        if spec:
            for pair in spec.split(','):
                btn, _, act = pair.partition(':')
                self.gesture_map[int(btn)] = int(act)
        self._prev_buttons = ()           # edge-detect state for gesture buttons
        self._wave_mute_until = 0.0       # monotonic deadline; cmd_vel muted until then
        # Head-LED color cycle
        self.color_button = int(self.get_parameter('color_button').value)
        self.color_palette = self._parse_palette(
            str(self.get_parameter('color_palette').value))
        self._color_idx = -1              # -1 -> first press selects palette[0]
        self.blink_period = max(0.05, float(self.get_parameter('blink_period').value))
        self.blinking = False             # rainbow-flash mode active
        self._blink_timer = None

        # --- homing state ---
        self.home_button = int(self.get_parameter('home_button').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.yaw_tolerance = float(self.get_parameter('yaw_tolerance').value)
        self.homing_yaw_rate = float(self.get_parameter('homing_yaw_rate').value)
        self.waypoint_step = float(self.get_parameter('waypoint_step').value)
        self.homing_speed = float(self.get_parameter('homing_speed').value)
        self.slowdown_dis = float(self.get_parameter('slowdown_dis').value)
        self.approach_speed = float(self.get_parameter('approach_speed').value)
        self.fine_tolerance = float(self.get_parameter('fine_tolerance').value)
        self.fine_speed = float(self.get_parameter('fine_speed').value)
        self.fine_timeout = float(self.get_parameter('fine_timeout').value)
        self._fine_deadline = 0.0
        self._fine_rounds = 0
        self._last_latch_speed = self.homing_speed
        self.traj_file = str(self.get_parameter('traj_file').value)
        self.traj_lookahead = float(self.get_parameter('traj_lookahead').value)
        self.traj_min_gap = float(self.get_parameter('traj_min_gap').value)
        self.traj = self._load_traj(self.traj_file)   # [(x, y), ...] map frame
        self.recording = False            # taught-route recording in progress
        self._record = []
        self._route = []                  # active route for this homing/return
        self._traj_idx = 0                # monotonic progress along _route
        self._spike_count = 0             # consecutive rejected odom teleports
        self.return_button = int(self.get_parameter('return_button').value)
        self.start_pose = None            # (x, y) of the first odom fix this run
        spec = str(self.get_parameter('perform_actions').value).strip()
        self.perform_actions = [int(a) for a in spec.split(',')] if spec else []
        self.perform_period = float(self.get_parameter('perform_period').value)
        self.performing = False           # cycling talk-gestures at the end pose
        self._perform_i = 0
        self._next_perform_t = 0.0
        self.end_pose = self._load_end_pose(str(self.get_parameter('end_pose_file').value))
        self.homing = None                # None | 'drive' | 'fine' | 'rotate' | 'return'
        self._estop = False               # emergency stop latched (safetyStop=2)
        self.cur_pose = None              # (x, y, yaw) from /state_estimation

        # Map connection method string to enum
        connection_method_map = {
            'LocalAP': WebRTCConnectionMethod.LocalAP,
            'LocalSTA': WebRTCConnectionMethod.LocalSTA,
            'Remote': WebRTCConnectionMethod.Remote
        }
        self.connection_method = connection_method_map.get(
            connection_method_str, WebRTCConnectionMethod.LocalSTA
        )

        self.get_logger().info(f'Connecting to robot at {self.robot_ip} using {connection_method_str}')
        self.get_logger().info(f'Control mode: {self.control_mode}')
        self.get_logger().info(f'Device type: {self.device_type}')
        if self.connection_method == WebRTCConnectionMethod.LocalSTA and not self.aes_128_key:
            self.get_logger().warn(
                'No aes_128_key set. LAN connection will FAIL on G1 firmware '
                '>= 1.5.1 (con_notify data2==3). Set the aes_128_key param or '
                'the UNITREE_AES_KEY env var.'
            )

        # Initialize WebRTC connection
        self.conn = None
        self.loop = None
        self.connected = threading.Event()
        self.connection_error = None

        # Start connection in background thread
        self.connection_thread = threading.Thread(target=self._connection_worker, daemon=True)
        self.connection_thread.start()

        # Wait for connection
        if not self.connected.wait(timeout=30.0):
            self.get_logger().error('Failed to connect to robot within timeout')
            raise RuntimeError('Connection timeout')
        if self.connection_error is not None:
            raise RuntimeError(f'Connection failed: {self.connection_error}')

        self.get_logger().info('Successfully connected to robot')

        # QoS profile for cmd_vel (use best effort for real-time control)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscriber for cmd_vel
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped,
            'cmd_vel',
            self.cmd_vel_callback,
            qos_profile
        )

        # Subscribe to /joy so controller buttons can trigger gestures while
        # the nav stack owns the (single) WebRTC connection.
        if self.gesture_map or self.home_button >= 0 or self.color_button >= 0:
            self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
            self.get_logger().info(
                f'Gestures mapped on /joy: {self.gesture_map}'
                f' (mute cmd_vel {self.wave_mute_sec:.1f}s during gesture)'
            )

        # Homing/return: waypoint/speed/stop/joy out, odom in, 20 Hz tick.
        want_home = self.home_button >= 0 and self.end_pose
        want_return = self.return_button >= 0
        if want_home or want_return:
            self.waypoint_pub = self.create_publisher(PointStamped, 'way_point', 5)
            self.speed_pub = self.create_publisher(Float32, 'speed', 5)
            self.stop_pub = self.create_publisher(Int8, 'stop', 5)
            self.joy_pub = self.create_publisher(Joy, 'joy', 5)
            self._tick_count = 0
            self.create_subscription(Odometry, 'state_estimation', self.odom_callback, 10)
            self.create_timer(0.05, self.homing_tick)
        if want_home:
            ex, ey, eyaw = self.end_pose
            self.get_logger().info(
                f'Homing: buttons[{self.home_button}] -> ({ex:.2f}, {ey:.2f}, yaw {math.degrees(eyaw):.0f} deg)'
            )
        elif self.home_button >= 0:
            self.get_logger().warn('Homing disabled: end_pose_file missing/unreadable')
        if want_return:
            self.get_logger().info(f'Return: buttons[{self.return_button}] -> back to the run start')
        if want_home or want_return:
            self.create_service(Trigger, 'traj_record_start', self.traj_record_start_cb)
            self.create_service(Trigger, 'traj_record_stop', self.traj_record_stop_cb)
            if self.traj:
                self.get_logger().info(
                    f'Taught route: {len(self.traj)} points from {self.traj_file}; '
                    'homing/return will follow it instead of the straight line.')
            else:
                self.get_logger().info(
                    f'No taught route ({self.traj_file}); homing/return use the '
                    'straight-line carrot. Record one via /traj_record_start|stop.')

        # Create services for sport commands
        self.standup_srv = self.create_service(Trigger, 'standup', self.standup_callback)
        self.liedown_srv = self.create_service(Trigger, 'liedown', self.liedown_callback)
        self.hello_srv = self.create_service(Trigger, 'hello', self.hello_callback)
        self.stretch_srv = self.create_service(Trigger, 'stretch', self.stretch_callback)
        self.recovery_stand_srv = self.create_service(Trigger, 'recovery_stand', self.recovery_stand_callback)
        self.cycle_led_srv = self.create_service(Trigger, 'cycle_led', self.cycle_led_callback)
        self.blink_start_srv = self.create_service(Trigger, 'led_blink_start', self.led_blink_start_callback)
        self.blink_stop_srv = self.create_service(Trigger, 'led_blink_stop', self.led_blink_stop_callback)

        # Docked RViz panel (g1_rviz_panel) publishes a command string here.
        self.panel_cmd_sub = self.create_subscription(
            String, str(self.get_parameter('panel_cmd_topic').value),
            self._panel_cmd_cb, 10)

        if bool(self.get_parameter('rviz_buttons').value):
            self._setup_rviz_buttons()

        self.get_logger().info('Unitree control node started')
        self.get_logger().info('Subscribed to: cmd_vel')
        self.get_logger().info('Services: standup, liedown, hello, stretch, recovery_stand')

    def _connection_worker(self):
        """Background thread for asyncio event loop and WebRTC connection."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            # Create connection
            self.conn = UnitreeWebRTCConnection(
                self.connection_method,
                ip=self.robot_ip,
                aes_128_key=self.aes_128_key or None,
                device_type=self.device_type,
            )

            # Connect
            self.loop.run_until_complete(self.conn.connect())

            # Signal that connection is ready
            self.connected.set()

            # Keep the loop running
            self.loop.run_forever()
        except Exception as e:
            self.connection_error = str(e)
            self.get_logger().error(f'Connection error: {e}')
            self.connected.set()  # Release waiting thread even on error
        finally:
            self.loop.close()

    def _gesture_request(self, action_id: int):
        """Device-specific gesture (topic, request) for the datachannel.

        G1 (fw >= 1.5.x): arm gestures live on the dedicated 'arm' service
        (rt/api/arm/request), api 7106 EXECUTE_ACTION with {"data": action_id}.
        The loco SetTaskId path (same api number 7106 but on
        rt/api/sport/request) is ACKed by the robot but not executed on this
        firmware -- verified 2026-07-08. Go2 has no arm: any gesture button
        falls back to the classic sport Hello=1016.
        """
        if self.device_type == 'G1':
            return ("rt/api/arm/request",
                    {"api_id": 7106, "parameter": {"data": action_id}})
        return (RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["Hello"]})

    def joy_callback(self, msg: Joy):
        """Fire arm gestures on rising edges of the mapped buttons."""
        # Synthetic joy messages (RViz waypoint/goalpoint tools set
        # buttons[7]=1 and axes[2]=-1 to latch autonomy; our own homing
        # re-latch does the same) are NOT human button presses -- they once
        # made the robot cross its arms (buttons[7] was mapped to 'reject').
        if msg.header.frame_id in ('waypoint_tool', 'goalpoint_tool', 'homing'):
            return
        prev = self._prev_buttons
        self._prev_buttons = tuple(msg.buttons)
        if not prev:
            # First message is baseline only. The joydev driver replays the
            # current button state on open -- a stale/phantom 'pressed' there
            # must not fire a gesture (robot crossed arms at startup once).
            return
        # Manual takeover releases a latched e-stop: real stick deflection
        # (left/right stick translate axes, past a deadzone) clears it so the
        # operator is never trapped. The 50 Hz idle stream stays near 0.
        if self._estop and msg.axes:
            sticks = [msg.axes[i] for i in (0, 1, 3) if i < len(msg.axes)]
            if any(abs(v) > 0.25 for v in sticks):
                self._estop = False
                if hasattr(self, 'stop_pub'):
                    self.stop_pub.publish(Int8(data=0))
                self.get_logger().warn('E-STOP released (manual joystick input)')
        # Home button (right stick click by default): toggle go-to-end-pose.
        if (self.end_pose and 0 <= self.home_button < len(msg.buttons)
                and msg.buttons[self.home_button] == 1
                and not (self.home_button < len(prev) and prev[self.home_button] == 1)):
            self._toggle_homing()
        # Return button (left stick click by default): toggle back-to-start.
        if (0 <= self.return_button < len(msg.buttons)
                and msg.buttons[self.return_button] == 1
                and not (self.return_button < len(prev) and prev[self.return_button] == 1)):
            self._toggle_return()
        # Color button (RB by default): step to the next head-LED color.
        if (self.color_palette and 0 <= self.color_button < len(msg.buttons)
                and msg.buttons[self.color_button] == 1
                and not (self.color_button < len(prev) and prev[self.color_button] == 1)):
            self._cycle_led()
        for btn, action_id in self.gesture_map.items():
            if btn >= len(msg.buttons):
                continue
            if msg.buttons[btn] == 1 and not (btn < len(prev) and prev[btn] == 1):
                self._wave_mute_until = time.monotonic() + self.wave_mute_sec
                self.get_logger().info(f'Gesture button {btn} -> arm action {action_id}')
                self._fire_action(action_id)

    def _fire_action(self, action_id: int):
        """Fire-and-forget an arm action on the WebRTC loop; never blocks the
        ROS executor. The robot's response is logged so a no-op is visible."""
        if not self.conn or not self.loop:
            return
        topic, request = self._gesture_request(action_id)
        async def async_gesture(t=topic, r=request):
            return await self.conn.datachannel.pub_sub.publish_request_new(t, r)
        fut = asyncio.run_coroutine_threadsafe(async_gesture(), self.loop)
        def _log_result(f, act=action_id):
            try:
                self.get_logger().info(f'action {act} response: {f.result()}')
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'action {act} failed: {e}')
        fut.add_done_callback(_log_result)

    # ---- head LED color ----

    def _parse_palette(self, spec):
        """Parse "name:R,G,B;..." into [(name, r, g, b), ...]; [] if empty."""
        out = []
        for entry in spec.strip().split(';'):
            entry = entry.strip()
            if not entry:
                continue
            try:
                name, _, rgb = entry.partition(':')
                r, g, b = (int(v) for v in rgb.split(','))
                out.append((name.strip() or f'{r},{g},{b}', r, g, b))
            except ValueError:
                self.get_logger().warn(f'color_palette: bad entry {entry!r}, skipped')
        return out

    def _fire_led(self, r, g, b):
        """Fire-and-forget a head-LED color over the WebRTC loop (G1 'voice'
        service, api_id 1010). Never blocks the ROS executor."""
        if not self.conn or not self.loop:
            return
        request = {"api_id": 1010, "parameter": {"R": int(r), "G": int(g), "B": int(b)}}
        async def async_led(req=request):
            return await self.conn.datachannel.pub_sub.publish_request_new(
                "rt/api/voice/request", req)
        fut = asyncio.run_coroutine_threadsafe(async_led(), self.loop)
        def _log_result(f):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'LED set failed: {e}')
        fut.add_done_callback(_log_result)

    def _cycle_led(self):
        """Advance to the next palette color and send it."""
        if not self.color_palette:
            return
        self._color_idx = (self._color_idx + 1) % len(self.color_palette)
        name, r, g, b = self.color_palette[self._color_idx]
        self.get_logger().info(f'Head LED -> {name} ({r},{g},{b})')
        self._fire_led(r, g, b)
        return name

    def _blink_tick(self):
        if self.blinking:
            self._cycle_led()

    def _start_blink(self):
        if not self.color_palette:
            return False
        if self._blink_timer is None:
            self._blink_timer = self.create_timer(self.blink_period, self._blink_tick)
        self.blinking = True
        self.get_logger().warn(f'Head LED blink ON ({self.blink_period:.2f}s/color)')
        return True

    def _stop_blink(self):
        self.blinking = False
        self.get_logger().warn('Head LED blink OFF')

    # ---- RViz control dispatch (docked panel -> action) ----

    def _command_map(self):
        """cmd string -> action. Shared by the docked panel (/g1_panel_cmd) and
        the optional interactive-marker buttons."""
        return {
            'wave_hi': lambda: self._fire_action(26),   # 大招呼 big wave
            'wave_lo': lambda: self._fire_action(25),   # 小招呼 small wave
            'clap':    lambda: self._fire_action(17),   # 鼓掌 clap
            'reset':   lambda: self._fire_action(99),   # 返回默认 release arm
            'stage':   self._toggle_homing,             # 去舞台中心
            'back':    self._toggle_return,             # 回后台
            'stop':    self._emergency_stop,            # 急停：清 waypoint + 停下
            'color':   self._cycle_led,                 # step palette
            'blink_on':  self._start_blink,             # 颜色开始
            'blink_off': self._stop_blink,              # 颜色结束
        }

    def _run_panel_cmd(self, cmd):
        fn = self._command_map().get(cmd)
        if fn is None:
            self.get_logger().warn(f'panel cmd unknown: {cmd!r}')
            return
        if cmd == 'stage' and not (hasattr(self, 'waypoint_pub') and self.end_pose):
            self.get_logger().warn('去舞台中心 unavailable (homing not configured)')
            return
        if cmd == 'back' and not hasattr(self, 'joy_pub'):
            self.get_logger().warn('回后台 unavailable (return not configured)')
            return
        self.get_logger().info(f'Panel cmd: {cmd}')
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'panel cmd {cmd} failed: {e}')

    def _panel_cmd_cb(self, msg: String):
        self._run_panel_cmd(msg.data.strip())

    # ---- RViz clickable-button panel ----

    def _setup_rviz_buttons(self):
        """Publish a column of interactive-marker BUTTONs so every joystick
        action is also one left-click away in RViz. Folded into this node so it
        can call the action methods directly (no service round-trip) and needs
        no extra executable / colcon build. Missing viz packages just disable
        the panel with a warning."""
        try:
            from interactive_markers import InteractiveMarkerServer
            from visualization_msgs.msg import (
                InteractiveMarker, InteractiveMarkerControl,
                InteractiveMarkerFeedback, Marker)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'RViz buttons disabled (import failed): {e}')
            return

        self._IM_BUTTON = InteractiveMarkerControl.BUTTON
        self._IM_CLICK = InteractiveMarkerFeedback.BUTTON_CLICK
        self._Marker = Marker
        self._InteractiveMarker = InteractiveMarker
        self._InteractiveMarkerControl = InteractiveMarkerControl

        # Label sets. RViz2's 3D font lacks CJK glyphs on some builds, so keep
        # pinyin / en fallbacks selectable via panel_lang.
        LABELS = {
            'zh': {'wave_hi': '大招呼', 'wave_lo': '小招呼', 'clap': '鼓掌',
                   'reset': '返回默认', 'stage': '去舞台中心', 'back': '回后台',
                   'blink_on': '颜色开始', 'blink_off': '颜色结束'},
            'pinyin': {'wave_hi': 'Da Zhao Hu', 'wave_lo': 'Xiao Zhao Hu',
                       'clap': 'Gu Zhang', 'reset': 'Fan Hui Mo Ren',
                       'stage': 'Qu Wu Tai Zhong Xin', 'back': 'Hui Hou Tai',
                       'blink_on': 'Yan Se Kai Shi', 'blink_off': 'Yan Se Jie Shu'},
            'en': {'wave_hi': 'Big Wave', 'wave_lo': 'Small Wave', 'clap': 'Clap',
                   'reset': 'Reset Arm', 'stage': 'To Stage', 'back': 'Backstage',
                   'blink_on': 'Color ON', 'blink_off': 'Color OFF'},
        }
        lang = str(self.get_parameter('panel_lang').value)
        label = LABELS.get(lang, LABELS['zh'])

        # (name, (r,g,b) button color, handler). Home/back only when the homing
        # infrastructure was created (home_button/return_button >= 0).
        specs = [
            ('wave_hi', (0.20, 0.50, 0.90), lambda: self._fire_action(26)),
            ('wave_lo', (0.20, 0.50, 0.90), lambda: self._fire_action(25)),
            ('clap',    (0.20, 0.50, 0.90), lambda: self._fire_action(17)),
            ('reset',   (0.45, 0.45, 0.50), lambda: self._fire_action(99)),
        ]
        if hasattr(self, 'waypoint_pub') and self.end_pose:
            specs.append(('stage', (0.95, 0.55, 0.10), self._toggle_homing))
        if hasattr(self, 'joy_pub'):
            specs.append(('back', (0.95, 0.55, 0.10), self._toggle_return))
        specs += [
            ('blink_on',  (0.15, 0.75, 0.25), self._start_blink),
            ('blink_off', (0.85, 0.20, 0.20), self._stop_blink),
        ]
        self._button_handlers = {name: fn for name, _, fn in specs}

        frame = str(self.get_parameter('panel_frame').value)
        px = float(self.get_parameter('panel_x').value)
        py = float(self.get_parameter('panel_y').value)
        z_top = float(self.get_parameter('panel_z_top').value)
        dz = float(self.get_parameter('panel_dz').value)
        s = float(self.get_parameter('panel_scale').value)

        self.im_server = InteractiveMarkerServer(self, 'g1_buttons')
        for i, (name, color, _fn) in enumerate(specs):
            im = self._make_button(frame, name, label[name], color, px, py,
                                   z_top - i * dz, s)
            self.im_server.insert(im, feedback_callback=self._im_feedback)
        self.im_server.applyChanges()
        self.get_logger().info(
            f'RViz buttons: {len(specs)} in frame "{frame}" '
            '(add an InteractiveMarkers display on /g1_buttons/update)')

    def _make_button(self, frame, name, label, color, x, y, z, s):
        Marker = self._Marker
        im = self._InteractiveMarker()
        im.header.frame_id = frame
        im.name = name
        im.scale = 2.5 * s
        im.pose.position.x = x
        im.pose.position.y = y
        im.pose.position.z = z

        ctrl = self._InteractiveMarkerControl()
        ctrl.interaction_mode = self._IM_BUTTON
        ctrl.always_visible = True
        ctrl.name = 'btn'

        box = Marker()
        box.type = Marker.CUBE
        box.scale.x = 2.4 * s
        box.scale.y = 0.9 * s
        box.scale.z = 0.9 * s
        box.color.r, box.color.g, box.color.b, box.color.a = (*color, 0.9)
        ctrl.markers.append(box)

        txt = Marker()
        txt.type = Marker.TEXT_VIEW_FACING
        txt.text = label
        txt.scale.z = 0.7 * s
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.pose.position.z = 0.55 * s
        ctrl.markers.append(txt)

        im.controls.append(ctrl)
        return im

    def _im_feedback(self, feedback):
        """Fire the mapped action on a button left-click."""
        if feedback.event_type != self._IM_CLICK:
            return
        fn = self._button_handlers.get(feedback.marker_name)
        if fn is None:
            return
        self.get_logger().info(f'RViz button: {feedback.marker_name}')
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'RViz button {feedback.marker_name} failed: {e}')

    def _stop_performing(self, release: bool = True):
        if self.performing:
            self.performing = False
            if release:
                self._fire_action(99)     # release arm -> neutral
            self.get_logger().warn('Perform mode OFF')

    # ---- homing (fixed end pose) ----

    def _load_end_pose(self, path):
        """Read (x, y, yaw) from end_pose.yaml; None if unavailable."""
        try:
            with open(path) as fh:
                d = yaml.safe_load(fh)['end_pose']
            return (float(d['x']), float(d['y']), float(d['yaw']))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'end_pose_file {path}: {e}')
            return None

    def _load_traj(self, path):
        """Read the taught route ("x y" per line, map frame); [] if missing."""
        try:
            pts = []
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    x, y = line.split()[:2]
                    pts.append((float(x), float(y)))
            return pts if len(pts) >= 2 else []
        except OSError:
            return []

    def traj_record_start_cb(self, request, response):
        if not self.cur_pose:
            response.success = False
            response.message = 'no /state_estimation yet'
            return response
        self.recording = True
        self._record = [self.cur_pose[:2]]
        response.success = True
        response.message = 'recording taught route; drive the ideal path now'
        self.get_logger().warn('Taught route: RECORDING started')
        return response

    def traj_record_stop_cb(self, request, response):
        self.recording = False
        if self.cur_pose and self._record:
            self._record.append(self.cur_pose[:2])   # exact stop spot
        if len(self._record) < 2:
            response.success = False
            response.message = 'too few points recorded, route unchanged'
            return response
        try:
            with open(self.traj_file, 'w') as fh:
                fh.write('# taught route, map frame, one "x y" per line\n')
                for px, py in self._record:
                    fh.write(f'{px:.4f} {py:.4f}\n')
        except OSError as e:
            response.success = False
            response.message = f'cannot write {self.traj_file}: {e}'
            return response
        self.traj = list(self._record)
        response.success = True
        response.message = f'saved {len(self.traj)} points to {self.traj_file}, active now'
        self.get_logger().warn(f'Taught route: SAVED {len(self.traj)} points -> {self.traj_file}')
        return response

    def _start_route(self, x, y, reverse):
        """Arm _route for this homing/return: taught route (reversed for
        return) densified to 0.1 m spacing, starting nearest the robot.
        Densifying matters: with sparse points the carrot could land inside
        pathFollower's stopDisThre and park the robot for good."""
        raw = list(reversed(self.traj)) if reverse else list(self.traj)
        dense = []
        for a, b in zip(raw, raw[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(seg / 0.1))
            for k in range(n):
                dense.append((a[0] + (b[0] - a[0]) * k / n,
                              a[1] + (b[1] - a[1]) * k / n))
        if raw:
            dense.append(raw[-1])
        self._route = dense
        if self._route:
            self._traj_idx = min(
                range(len(self._route)),
                key=lambda k: (self._route[k][0] - x) ** 2 + (self._route[k][1] - y) ** 2)

    def _traj_carrot(self, x, y, gx, gy):
        """Carrot along the taught route: advance the (monotonic) nearest
        index, then walk traj_lookahead meters forward ALONG the route.
        The carrot is measured from route progress, not from the robot, so
        it stays a live goal ahead of the planner even if the robot stalls
        (a robot-relative window once deadlocked: robot stops -> carrot
        stops -> robot never restarts). Near the route end it snaps to the
        exact goal."""
        pts = self._route
        i = self._traj_idx
        while (i + 1 < len(pts)
               and math.hypot(pts[i + 1][0] - x, pts[i + 1][1] - y)
               <= math.hypot(pts[i][0] - x, pts[i][1] - y)):
            i += 1
        self._traj_idx = i
        j = i
        acc = 0.0
        while j + 1 < len(pts) and acc < self.traj_lookahead:
            acc += math.hypot(pts[j + 1][0] - pts[j][0], pts[j + 1][1] - pts[j][1])
            j += 1
        if j >= len(pts) - 1:
            return gx, gy
        return pts[j]

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # /state_estimation throws transient 0.4-2 m spikes while walking
        # (observed 2026-07-10). A teleport between consecutive ~100 Hz
        # samples is physically impossible for the G1 -- drop it so it can't
        # fake goal arrival or yank the route index forward. >20 consecutive
        # "spikes" means a real relocalization correction: accept it.
        if self.cur_pose is not None:
            if math.hypot(p.x - self.cur_pose[0], p.y - self.cur_pose[1]) > 0.5:
                self._spike_count += 1
                if self._spike_count <= 20:
                    return
            self._spike_count = 0
        self.cur_pose = (p.x, p.y, yaw)
        if self.start_pose is None:
            self.start_pose = (p.x, p.y)  # where this run began
        if self.recording:
            lx, ly = self._record[-1]
            if math.hypot(p.x - lx, p.y - ly) >= self.traj_min_gap:
                self._record.append((p.x, p.y))

    def _publish_autonomy_latch(self, speed=None):
        """Synthetic /joy like the RViz waypoint tool: axes[2]=-1 latches
        autonomyMode in localPlanner/pathFollower until real joystick input,
        axes[4] sets joySpeed. This is how the stack engages autonomy without
        physically holding LT."""
        sp = min(1.0, self.homing_speed if speed is None else speed)
        j = Joy()
        j.header.stamp = self.get_clock().now().to_msg()
        j.header.frame_id = 'homing'
        j.axes = [0.0, 0.0, -1.0, 0.0, sp, 1.0, 0.0, 0.0]
        j.buttons = [0] * 11
        self.joy_pub.publish(j)
        self._last_latch_speed = sp

    def _toggle_homing(self):
        self._stop_performing()
        if self.homing in ('drive', 'fine', 'rotate'):
            self.homing = None
            self._send_vel(0.0, 0.0, 0.0)
            self.get_logger().warn('Homing CANCELLED')
            return
        if not self.cur_pose:
            self.get_logger().warn('Homing: no /state_estimation yet, ignored')
            return
        self.homing = 'drive'
        self._estop = False
        self._start_route(*self.cur_pose[:2], reverse=False)
        # Resume navigation in case a safety stop is latched (RViz 'Resume').
        self.stop_pub.publish(Int8(data=0))
        self._publish_autonomy_latch()
        ex, ey, eyaw = self.end_pose
        self.get_logger().warn(
            ('Homing STARTED (taught route, %d pts)' % len(self._route)
             if self._route else 'Homing STARTED (straight-line carrot)')
            + f' -> ({ex:.2f}, {ey:.2f}); autonomy latched, no LT needed. '
            'Touching the joystick pauses it (re-latched within 1 s); press the home '
            'button again to cancel. Auto-rotate to %.0f deg on arrival.'
            % math.degrees(eyaw))

    def _toggle_return(self):
        self._stop_performing()
        if self.homing == 'return':
            self.homing = None
            self.get_logger().warn('Return CANCELLED')
            return
        if not self.cur_pose or not self.start_pose:
            self.get_logger().warn('Return: no /state_estimation yet, ignored')
            return
        self.homing = 'return'
        self._estop = False
        self._start_route(*self.cur_pose[:2], reverse=True)
        self.stop_pub.publish(Int8(data=0))
        self._publish_autonomy_latch()
        sx, sy = self.start_pose
        self.get_logger().warn(
            ('Return STARTED (taught route reversed)' if self._route
             else 'Return STARTED (straight-line carrot)')
            + f' -> run start ({sx:.2f}, {sy:.2f}). Press again to cancel.')

    def _emergency_stop(self):
        """Stop NOW: cancel any homing/return (kills the carrot/way_point stream
        and the autonomy re-latch), full-halt pathFollower via safetyStop=2, and
        send zero velocity immediately. Latched: released when the operator moves
        the joystick (manual takeover) or presses 去舞台中心 / 回后台 again."""
        self._stop_performing()
        self.homing = None
        self._estop = True
        if hasattr(self, 'stop_pub'):
            self.stop_pub.publish(Int8(data=2))    # >=2: zero speed AND yaw rate
        self._send_vel(0.0, 0.0, 0.0)
        self.get_logger().warn(
            'E-STOP: homing cancelled, waypoint cleared, robot halted. '
            'Move the joystick or press 去舞台中心/回后台 to release.')

    def _publish_carrot(self, wx, wy, speed=None):
        sp = self.homing_speed if speed is None else speed
        wp = PointStamped()
        wp.header.stamp = self.get_clock().now().to_msg()
        wp.header.frame_id = 'map'
        wp.point.x, wp.point.y, wp.point.z = wx, wy, 0.0
        self.waypoint_pub.publish(wp)
        self.speed_pub.publish(Float32(data=sp))
        # Re-latch autonomy at 1 Hz: any real joystick event drops
        # autonomyMode; this brings the drive back within a second. Also
        # re-latch immediately when the taper moved joySpeed (axes[4]) --
        # that's the only channel pathFollower takes speed from.
        self._tick_count += 1
        if self._tick_count % 20 == 0 or abs(sp - self._last_latch_speed) > 0.05:
            self._publish_autonomy_latch(sp)

    def _approach_speed(self, dist):
        """joySpeed for the drive phase: homing_speed far out, tapering
        linearly to approach_speed inside slowdown_dis of the goal."""
        if dist >= self.slowdown_dis:
            return self.homing_speed
        return max(self.approach_speed, self.homing_speed * dist / self.slowdown_dis)

    def homing_tick(self):
        """20 Hz: carrot waypoint + speed while driving; fine-align then
        rotate in place at the goal; cycle talk-gestures while performing."""
        if self.performing:
            now = time.monotonic()
            if now >= self._next_perform_t:
                action = self.perform_actions[self._perform_i % len(self.perform_actions)]
                self._perform_i += 1
                self._next_perform_t = now + self.perform_period
                self.get_logger().info(f'Perform: arm action {action}')
                self._fire_action(action)
        if not self.homing or not self.cur_pose:
            return
        x, y, yaw = self.cur_pose

        if self.homing == 'return':
            # Same carrot guidance as homing, aimed at the run's start point.
            sx, sy = self.start_pose
            dx, dy = sx - x, sy - y
            dist = math.hypot(dx, dy)
            if dist < self.goal_tolerance:
                self.homing = None
                self._send_vel(0.0, 0.0, 0.0)
                self.get_logger().warn('Return DONE: back at the run start. Robot stopped.')
                return
            if self._route:
                wx, wy = self._traj_carrot(x, y, sx, sy)
            elif dist > self.waypoint_step:
                s = self.waypoint_step / dist
                wx, wy = x + dx * s, y + dy * s
            else:
                wx, wy = sx, sy
            self._publish_carrot(wx, wy, self._approach_speed(dist))
            return

        ex, ey, eyaw = self.end_pose

        if self.homing == 'drive':
            # Carrot guidance: along the taught route if one is loaded,
            # otherwise along the straight line to the goal (the local
            # planner only plans nearby either way).
            dx, dy = ex - x, ey - y
            dist = math.hypot(dx, dy)
            if dist < self.goal_tolerance:
                self.homing = 'fine'
                self._fine_rounds = 0
                self._fine_deadline = time.monotonic() + self.fine_timeout
                self.get_logger().warn('Homing: near goal, fine-aligning position...')
                return
            if self._route:
                wx, wy = self._traj_carrot(x, y, ex, ey)
            elif dist > self.waypoint_step:
                s = self.waypoint_step / dist
                wx, wy = x + dx * s, y + dy * s
            else:
                wx, wy = ex, ey
            self._publish_carrot(wx, wy, self._approach_speed(dist))
            if self._tick_count % 40 == 0:   # ~2 s heartbeat for diagnosis
                self.get_logger().info(
                    f'Homing drive: {dist:.2f} m to goal, route {self._traj_idx}/'
                    f'{len(self._route)}, carrot ({wx:.2f}, {wy:.2f})')
            return

        if self.homing == 'fine':
            # Planner+follower cmd_vel is dropped (cmd_vel_callback): P-servo
            # the map-frame position error with small omni steps. This is what
            # turns the ~0.35 m carrot-stop scatter into a few cm on stage.
            dx, dy = ex - x, ey - y
            dist = math.hypot(dx, dy)
            if dist < self.fine_tolerance:
                self._send_vel(0.0, 0.0, 0.0)
                self.homing = 'rotate'
                self.get_logger().warn(
                    f'Homing: position aligned ({100 * dist:.0f} cm off), aligning yaw...')
                return
            if time.monotonic() > self._fine_deadline:
                self._send_vel(0.0, 0.0, 0.0)
                self.homing = 'rotate'
                self.get_logger().warn(
                    f'Homing: fine-align TIMEOUT at {dist:.2f} m, aligning yaw anyway')
                return
            bx = math.cos(yaw) * dx + math.sin(yaw) * dy
            by = -math.sin(yaw) * dx + math.cos(yaw) * dy
            vx, vy = 1.2 * bx, 1.2 * by
            n = math.hypot(vx, vy)
            if n > self.fine_speed:
                vx, vy = vx / n * self.fine_speed, vy / n * self.fine_speed
            elif 0.0 < n < 0.06:
                # Too-small stick values don't move the gait at all.
                vx, vy = vx / n * 0.06, vy / n * 0.06
            self._send_vel(vx, vy, 0.0)
            return

        # rotate phase: external cmd_vel is dropped in cmd_vel_callback;
        # we own the velocity stream until aligned.
        err = math.atan2(math.sin(eyaw - yaw), math.cos(eyaw - yaw))
        if abs(err) < self.yaw_tolerance:
            # Rotating a humanoid in place shifts it a little: re-check the
            # position and do another fine pass if it stepped off the marker.
            dist = math.hypot(ex - x, ey - y)
            if dist > 1.5 * self.fine_tolerance and self._fine_rounds < 2:
                self._fine_rounds += 1
                self._fine_deadline = time.monotonic() + self.fine_timeout
                self.homing = 'fine'
                self.get_logger().warn(
                    f'Homing: rotation drifted {100 * dist:.0f} cm off, fine pass {self._fine_rounds}')
                return
            self._send_vel(0.0, 0.0, 0.0)
            self.homing = None
            self.get_logger().warn('Homing DONE: at end pose, aligned. Robot stopped.')
            # External audio plays now; do natural talk-gestures instead of
            # standing frozen. Ends when homing/return is pressed.
            if self.perform_actions:
                self.performing = True
                self._perform_i = 0
                self._next_perform_t = time.monotonic() + 2.0
                self.get_logger().warn(
                    f'Perform mode ON: cycling actions {self.perform_actions} '
                    f'every {self.perform_period:.0f}s until homing/return pressed.')
            return
        rate = max(-self.homing_yaw_rate, min(self.homing_yaw_rate, 1.5 * err))
        self._send_vel(0.0, 0.0, rate)

    def cmd_vel_callback(self, msg: TwistStamped):
        """Handle incoming cmd_vel messages."""
        if not self.conn or not self.loop:
            self.get_logger().warn('Connection not ready, ignoring cmd_vel')
            return

        # Hold still while a wave is in progress so the 50 Hz velocity stream
        # doesn't fight the arm gesture.
        if time.monotonic() < self._wave_mute_until:
            return

        # During the homing fine-align/rotate phases this node owns the
        # velocity stream; drop pathFollower's cmd_vel.
        if self.homing in ('fine', 'rotate'):
            return

        x, y, yaw = msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z
        self._send_vel(x, y, yaw)

    def _send_vel(self, x, y, yaw):
        """Send a velocity command over the active control mode."""
        if not self.conn or not self.loop:
            return
        # Choose control mode based on parameter
        if self.control_mode == 'wireless_controller':
            # WebRTC coordinate mapping for wireless controller:
            # lx - Positive right, negative left (maps to ROS y)
            # ly - Positive forward, negative backwards (maps to ROS x)
            # rx - Positive rotate right, negative rotate left (maps to ROS yaw)
            async def async_move():
                self.conn.datachannel.pub_sub.publish_without_callback(
                    RTC_TOPIC["WIRELESS_CONTROLLER"],
                    data={"lx": -y, "ly": x, "rx": -yaw, "ry": 0},
                )
        else:  # sport_cmd (default)
            # SPORT_CMD["Move"] coordinate mapping:
            # x - forward/backward
            # y - left/right
            # z - yaw rotation
            async def async_move():
                await self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {
                        "api_id": SPORT_CMD["Move"],
                        "parameter": {"x": x, "y": y, "z": yaw}
                    }
                )

        try:
            future = asyncio.run_coroutine_threadsafe(async_move(), self.loop)
            if self.control_mode == 'sport_cmd':
                future.result()  # Wait for sport_cmd
            # wireless_controller uses publish_without_callback, no need to wait
        except Exception as e:
            self.get_logger().error(f'Failed to send cmd_vel: {e}')

    def _execute_sport_command(self, command_name: str, api_id: int, parameter: dict = None,
                               topic: str = None) -> Trigger.Response:
        """Execute a sport/arm mode command and log the robot's response."""
        response = Trigger.Response()

        if not self.conn or not self.loop:
            response.success = False
            response.message = 'Connection not ready'
            return response

        try:
            request_data = {"api_id": api_id}
            if parameter:
                request_data["parameter"] = parameter

            async def async_command():
                return await self.conn.datachannel.pub_sub.publish_request_new(
                    topic or RTC_TOPIC["SPORT_MOD"],
                    request_data
                )

            future = asyncio.run_coroutine_threadsafe(async_command(), self.loop)
            result = future.result(timeout=5.0)

            response.success = True
            response.message = f'{command_name} command sent successfully'
            self.get_logger().info(f'{response.message}; response: {result}')

        except Exception as e:
            response.success = False
            response.message = f'Failed to execute {command_name}: {str(e)}'
            self.get_logger().error(response.message)

        return response

    def standup_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Service callback to make robot stand up."""
        return self._execute_sport_command('StandUp', SPORT_CMD["StandUp"])

    def liedown_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Service callback to make robot lie down."""
        return self._execute_sport_command('StandDown', SPORT_CMD["StandDown"])

    def hello_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Service callback to make robot wave hello (G1: arm EXECUTE_ACTION 26)."""
        topic, req = self._gesture_request(26)
        return self._execute_sport_command('Hello', req["api_id"], req.get("parameter"), topic)

    def stretch_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Service callback to make robot stretch."""
        return self._execute_sport_command('Stretch', SPORT_CMD["Stretch"])

    def recovery_stand_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Service callback to recovery stand position."""
        return self._execute_sport_command('RecoveryStand', SPORT_CMD["RecoveryStand"])

    def cycle_led_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Step the head LED to the next palette color (for RViz / CLI)."""
        response = Trigger.Response()
        if not self.color_palette:
            response.success = False
            response.message = 'color_palette empty'
            return response
        name = self._cycle_led()
        response.success = True
        response.message = f'LED -> {name}'
        return response

    def led_blink_start_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Start the head-LED rainbow blink (for RViz / CLI)."""
        response = Trigger.Response()
        response.success = self._start_blink()
        response.message = 'blink started' if response.success else 'color_palette empty'
        return response

    def led_blink_stop_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Stop the head-LED blink (for RViz / CLI)."""
        response = Trigger.Response()
        self._stop_blink()
        response.success = True
        response.message = 'blink stopped'
        return response

    def destroy_node(self):
        """Clean up resources before node shutdown."""
        self.get_logger().info('Shutting down...')

        # Disconnect WebRTC
        if self.conn and self.loop:
            try:
                async def async_disconnect():
                    await self.conn.disconnect()

                asyncio.run_coroutine_threadsafe(async_disconnect(), self.loop)
            except Exception as e:
                self.get_logger().error(f'Error during disconnect: {e}')

        # Stop event loop
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        # Wait for thread to finish
        if self.connection_thread.is_alive():
            self.connection_thread.join(timeout=2.0)

        super().destroy_node()


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)

    try:
        node = UnitreeControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
