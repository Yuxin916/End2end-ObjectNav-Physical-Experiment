# autonomy_stack on Unitree G1 — Docker 运行指南

ROS2 **Jazzy** 导航栈跑在 Unitree G1 的 Jetson 上的完整说明：安装、编译、模块、
容器内外的启动方式，以及雷达起前的网络配置与自检。

---

## 🚀 快速启动（每次开机，TL;DR）

> 这是日常起系统的**实际操作序列**。前 3 条在 **宿主机 Jetson** 上跑（需 sudo 密码 `123`），
> 后 2 条把你带进容器再起整套系统。想看原理 / 排错，往下翻 §0–§8。

**① 宿主机：每次开机做一次**
```bash
# (a) 锁 Jetson CPU/GPU 频率到最高 —— 稳住 Mid-360 雷达时间戳，避免 SLAM 漂移
#     （docker/run.sh 里也会跑一次，这里再跑无妨）
sudo jetson_clocks

# (b) 把发往 192.168.123.120 的流量强制走有线 eth0（源 IP 用 eth0 的 .164）。
#     eth0(.164) 和 wlan0(.110) 都在 192.168.123.0/24，不加这条路由内核可能把 .120
#     漏到 wlan0（详见 §6.3）
sudo ip route add 192.168.123.120/32 dev eth0 src 192.168.123.164

# (b2) 笔记本网线直连调试时(笔记本静态 IP 192.168.123.103),Jetson 回包也会漏到
#      wlan0(同一个坑的反方向)—— ssh/scp/NoMachine 全部超时。加同样的 host 路由:
sudo ip route add 192.168.123.103/32 dev eth0 src 192.168.123.164

# (c) 放开 X 服务器本地访问，让容器里(以 root 运行)的 RViz 能在物理显示器(:0)弹窗
sudo env DISPLAY=:0 XAUTHORITY=/var/run/lightdm/root/:0 xhost +local:
```

**② 进容器**
```bash
./docker/run.sh            # 进容器交互 bash（镜像 autonomy_stack:jazzy）
```

**③ 容器内：起整套系统 + RViz**
```bash
./system_real_robot_g1.sh \
  control_backend:=webrtc \
  robot_ip:=192.168.123.161 \
  connection_method:=LocalSTA \
  control_mode:=wireless_controller
```

参数含义：

| 参数 | 含义 |
|---|---|
| `control_backend:=webrtc` | 用旧 WebRTC 控制路径（需 `~/.unitree_g1.env` 里的 AES key）。换 `:=sdk` 走 unitree_sdk2 DDS（无需 AES key，启动即 DISABLED，要 §5 调 service 才动）；换 `:=none` 不起控制，纯可视化最安全 |
| `robot_ip:=192.168.123.161` | G1 在有线网上的 IP |
| `connection_method:=LocalSTA` | 走有线局域网（G1 作为局域网内的 station） |
| `control_mode:=wireless_controller` | 用手柄遥控 |

> 停止：起系统的那个终端按 `Ctrl+C`。验证话题频率、控制机器人站起/放行速度等，见 §3 / §5。

---

## 0. 背景：为什么必须用 Docker

| | 实际情况 |
|---|---|
| Jetson 系统 | Ubuntu **20.04 Focal** / aarch64 / L4T **R35.3.1**（JetPack 5.1.1），15GB RAM |
| 本机 ROS | 只有 `/opt/ros/foxy` |
| 本栈要求 | Ubuntu **24.04 + ROS2 Jazzy**（20.04 上 apt 装不了 Jazzy） |

解决办法：在一个**原生 arm64 的 Ubuntu 24.04 / ROS2 Jazzy 容器**里跑整套栈。
目标命令（导航 + SLAM + 雷达 + 控制）**全是 CPU 模块，不需要 GPU**；GPU 模块
（`sam2_detector`、`vlm_nav_bridge`）**没有编进来**。

机器人速度控制走 **`unitree_sdk2`（DDS over eth0）**，不是 WebRTC —— 所以**不需要 AES key**。

---

## 1. 架构总览

```
宿主机 Jetson (Ubuntu 20.04, Docker)
│
├─ autonomy_stack:heavy   ← docker/Dockerfile       （ROS Jazzy + Sophus/Ceres/GTSAM/Livox-SDK2）
│      └─ autonomy_stack:jazzy ← docker/Dockerfile.sdk （+ cyclonedds 0.10.2 + unitree_sdk2_python）
│
└─ docker run --network host  （容器）
       ├─ 挂载宿主仓库 → /workspace/autonomy_stack   （源码 + install/ 实时同步）
       ├─ 共享宿主网卡 eth0(192.168.123.x)            → 连机器人 .161 / 雷达 .120
       ├─ /dev/input/js0（手柄）、X11（RViz）
       └─ colcon 工作区 + ros2 launch
```

- **两段式镜像**：重依赖（编一次，慢）和控制层（改起来快）分开，改控制代码不会重编 GTSAM。
- **install/ 挂在宿主**：换/重建镜像不丢编译产物；改代码即时生效。

---

## 2. 安装与构建（都在宿主机、仓库根目录执行）

### 2.1 前置
- 宿主机已装 Docker（`docker --version`，本机为 26.1.3）。
- 桌面要看 RViz：先在 Jetson 图形终端跑一次 `xhost +local:root`。

### 2.2 构建镜像
```bash
bash docker/build.sh
```
- 自动判断：`autonomy_stack:heavy` 不存在 → 先建（**~30–40 分钟**，编 ROS desktop + GTSAM 等）；
  已存在 → 跳过，只建 `autonomy_stack:jazzy`（**几分钟**，加 cyclonedds + SDK）。
- 强制重建重依赖层：`bash docker/build.sh --rebuild-heavy`。
- 后台构建看进度：`tail -f docker/build_sdk.log`。

### 2.3 编译 ROS 工作区（一次性；改了 C++ 后再跑）
```bash
bash docker/run.sh build
```
- 在容器里 `colcon build`，产物落宿主 `install/`，跳过 `sam2_detector`、`vlm_nav_bridge`。
- ⚠️ **OOM 注意**：默认全核并行编译时，GTSAM 重的 `arise_slam_mid360` 可能被内核 OOM 杀
  （日志出现 `Killed signal terminated program cc1plus`）。单独限内存重编即可：
  ```bash
  bash docker/run.sh   # 进容器后：
  MAKEFLAGS="-j2" colcon build --symlink-install --parallel-workers 1 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select arise_slam_mid360
  ```

---

## 2.5 看画面：显示器接口 / RViz / 远程可视化

RViz 是 GPU 3D 程序，怎么看到它有三条路，**强烈推荐物理显示器**（本机 GPU 渲染，最流畅）。

### 2.5.1 ⚠️ 物理显示器必须插对口：**[9] 号 Type-C**
G1 脖子后面那排 Type-C 口里：
- **[6][7][8] = 纯 USB3.0 Host**（只有数据，**没有视频**）
- **[9] = Alt-Mode Type-C，USB3.2 + DisplayPort 1.4 ← 只有这个口能出视频**

用 **Type-C→HDMI 转接头插到 [9] 号口**，再接 HDMI 线和显示器，才能看到 Jetson 桌面。
插错口（6–8）的症状：USB 设备（手柄等）正常枚举，但 Jetson 的 `DP-0/DP-1` 读不到 EDID、
显示器不亮（`xrandr` 看 `DP-0 disconnected`、分辨率掉到 640x480）。插对 [9] 后 `DP-0 connected 1920x1080`。
> Tegra/NVIDIA 显示栈对热插拔不总是灵敏；最好**开机时就插着显示器**。插对口后桌面在物理 `:0`。

### 2.5.2 在物理显示器上看 RViz
直接在那台显示器的桌面终端起系统（见 §4.2），RViz 自动出现在 `:0`。

若**从 SSH 起**、想把窗口送到物理屏（`:0`）：SSH 会话不是物理屏的 X server 属主，
直接 `xhost` 会报 `cannot open display`，必须用 `sudo env` 把 `DISPLAY` 带进去给 X 授权：
```bash
sudo env DISPLAY=:0 xhost +local:root   # 关键：给物理屏 :0 的 X server 放行 root/容器
docker exec -e DISPLAY=:0 -it autonomy_stack bash   # 之后容器里起的 RViz 才会画到物理屏
```
> 在物理显示器自己的桌面终端跑则不用 `sudo env`，`xhost +local:root` 即可（见 §2.1）。

### 2.5.3 远程看（没有物理显示器时）
- **Foxglove（推荐的远程方案，GPU 在你本机渲染，不卡）**：容器里跑 foxglove_bridge，
  本机浏览器/桌面端连 `ws://192.168.123.164:8765`。
  ```bash
  # 容器内（镜像若没预装，先 apt-get install -y ros-jazzy-foxglove-bridge）：
  ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 -p address:=0.0.0.0
  ```
  那些 `unitree_api/unitree_go` 的 schema 报错可忽略（机器人自带 DDS 话题缺消息定义，不影响标准话题）。
- **X11 转发**（`ssh -Y` 把容器里 RViz 画到本机）：能用但 3D **非常卡**，不推荐。
- **NoMachine**：连的是 NoMachine 虚拟桌面 `:1001`，默认分辨率可能只有 768x576（壁纸被放大、没任务栏）。
  修：`DISPLAY=:1001 XAUTHORITY=~/.nx/M-ubuntu-*/authority xrandr --output nxoutput0 --mode 1920x1080`。

---

## 3. 模块说明（`src/`）

| 目录 | 作用 | 目标 launch 用到 |
|---|---|---|
| `base_autonomy/local_planner` | 局部规划 + 避障 + 路点跟随 | ✅ |
| `base_autonomy/terrain_analysis(_ext)` | 地形可通行性分析（近/远） | ✅ |
| `base_autonomy/sensor_scan_generation` | 点云配准/扫描生成 | ✅ |
| `base_autonomy/vehicle_simulator` | 系统总 launch + RViz 配置 | ✅ |
| `base_autonomy/visualization_tools` | 可视化、指标 | ✅ |
| `slam/arise_slam_mid360(_msgs)` | Mid-360 LiDAR-Inertial SLAM（GTSAM/Ceres/Sophus） | ✅ |
| `utilities/livox_ros_driver2` | Livox Mid-360 雷达驱动（+ `Livox-SDK2`） | ✅ |
| `utilities/serial`, `teleop_joy_controller` | 串口、手柄遥操 | ✅ |
| `utilities/*_rviz_plugin`, `goalpoint_rviz_plugin` | RViz 交互插件（设目标点等） | ✅ |
| **`unitree_g1_sdk_bridge`** | **cmd_vel → G1 `unitree_sdk2` LocoClient（DDS，本项目新增）** | ✅ 默认 |
| `unitree_webrtc_ros` | 旧 WebRTC 控制（需 AES key），保留做回退 | 可选 |
| `route_planner/*`（far_planner 等） | 全局路由规划（可视图） | 否 |
| `exploration_planner/tare_planner` | 自主探索 | 否 |
| `sam2_detector`, `vlm_nav_bridge` | GPU AI（**未编入容器**） | 否 |

---

## 4. 启动：容器外 vs 容器内

### 4.1 容器外（宿主机）
```bash
bash docker/run.sh                  # 进容器交互 bash（默认）
bash docker/run.sh build            # 编译工作区
bash docker/run.sh ros2 topic list  # 在容器环境里跑任意一条命令
```
`run.sh` 自动带：`--network host`、`/dev/input/js0`、X11、（存在则）挂 `~/.unitree_g1.env`。
> 它还会在起容器前自动 `sudo jetson_clocks`（锁 CPU/GPU 到最高频，降低 livox 时间戳抖动→
> 减少 SLAM 漂移）。会弹一次 sudo 密码（`123`）；失败不致命，只是时序抖动可能变大。
> 想手动单独跑：`sudo jetson_clocks`（查看当前状态：`sudo jetson_clocks --show`）。

### 4.2 ✅ 标准启动流程（一步步，建议在物理显示器的桌面终端跑）

**阶段 0｜清掉旧容器**（同名会冲突）
```bash
docker rm -f autonomy_stack 2>/dev/null
```

**阶段 1｜起前网络自检（每次开机做一次，需 sudo 密码 `123`）**
雷达路由不持久，重启后要重做（详见 §6）：
```bash
sudo ip route add 192.168.123.120/32 dev eth0          # 雷达只走 eth0
sudo ip neigh del 192.168.123.120 dev wlan0 2>/dev/null # 清掉漏到无线的坏 ARP
ping -c 3 192.168.123.161    # 机器人，应通 ~0.1ms
ping -c 3 192.168.123.120    # 雷达，应通（不通别往下走）
```

**阶段 2｜起整套系统 + RViz**
```bash
cd ~/projects/autonomy_stack

# A) 纯可视化 / 调试（机器人物理上不可能动，推荐先用这个）：
bash docker/run.sh ./system_real_robot_g1.sh control_backend:=none network_interface:=eth0

# B) 带控制能力（要驱动机器人时，务必场地空旷+握急停）：
bash docker/run.sh ./system_real_robot_g1.sh control_backend:=sdk network_interface:=eth0 \
    robot_ip:=192.168.123.161 connection_method:=LocalSTA control_mode:=wireless_controller
```
- `control_backend:=none` → 控制节点不启动；`:=sdk` → 起 `g1_sdk_bridge`（unitree_sdk2，无需 AES key，
  **启动即 DISABLED**，机器人仍不动，要调 service 才动，见 §5）；`:=webrtc` → 旧 WebRTC 回退（需 AES key）。
- 拉起：local_planner、terrain、sensor_scan、arise_slam、visualization、joy、livox(Mid-360)、控制后端，并开 RViz。
- RViz 出现在物理显示器（`:0`）。**停止**：该终端 `Ctrl+C`。

**阶段 3｜验证（另开一个终端）**
```bash
docker exec -it autonomy_stack bash
source install/setup.bash
ros2 topic hz /state_estimation   # SLAM 里程计 ~42Hz
ros2 topic hz /terrain_map        # 地形 ~3Hz
```
> ⚠️ **机器人吊在龙门架上、脚不沾地时**：arise_slam(LIO) 约束不足 → `/state_estimation` 漂移、`/tf` 不发
> 动态变换（RViz 里看不到机器人本体帧）。这是正常的，**等机器人落地站稳、走两步给运动激励后** SLAM 才收敛。
> 这跟 cyclonedds 无关（ROS2 栈走 Fast DDS，cyclonedds 只给 §5 的控制桥用）。

---

## 5. G1 控制流程（安全第一，默认机器人不动）

`g1_sdk_bridge` 启动后是 **DISABLED**：收 `cmd_vel` 但**不下发**，机器人不站不动。
要让它动，另开一个终端（`docker exec -it autonomy_stack bash`）按顺序调 service：

```bash
# 1) 站起来（Damp -> Squat2StandUp）
ros2 service call /g1_sdk_bridge/stand_up std_srvs/srv/Trigger
# 2) 放行 cmd_vel —— 此后导航/手柄速度才真正驱动机器人
ros2 service call /g1_sdk_bridge/enable std_srvs/srv/SetBool "{data: true}"

# —— 软急停（进阻尼，并自动 disable）——
ros2 service call /g1_sdk_bridge/damp std_srvs/srv/Trigger
# 其它：坐下 / 进运控模式
ros2 service call /g1_sdk_bridge/sit   std_srvs/srv/Trigger
ros2 service call /g1_sdk_bridge/start std_srvs/srv/Trigger
```

保护参数（`src/unitree_g1_sdk_bridge/launch/g1_sdk_control.launch.py`）：
- 速度夹紧：`max_vx=0.6` `max_vy=0.4` `max_vyaw=0.8`
- `cmd_vel` 超过 `0.5s` 没更新 → 自动发 0
- 节点退出 → 自动 `Damp()`

G1 FSM id 参考：ZeroTorque=0, Damp=1, Sit=3, Start=200, Lie2StandUp=702, Squat2StandUp=706。

> **cyclonedds 崩溃补丁（已内置，勿删）**：本机 G1 固件 + Ubuntu 24.04 fortify 下，unitree_sdk2 的
> `ChannelConfigHasInterface` 里那段 `<Tracing>` 会让 `ChannelFactoryInitialize` 直接
> `*** buffer overflow detected ***` 崩。`g1_sdk_control.py` 已内置 `_patch_cyclonedds_tracing()`
> 在初始化前把那段去掉（容器内实测：不打补丁崩、打了正常）。所以 `control_backend:=sdk` 现在能正常起。

---

## 6. 雷达（Livox Mid-360）—— 起之前必看

### 6.1 网络规划
`src/utilities/livox_ros_driver2/config/MID360_config.json`：
- **电脑接收 IP（host_net_info）= `192.168.123.164`** ✅（已改好；原厂默认是 `.103`，但 eth0 是 `.164`，
  接收 IP 必须等于 eth0 实际 IP，否则雷达把点云发到一个不存在的地址 → ping 通也没点云）
- **雷达 IP = `192.168.123.120`**（注意：Weston 文档写 `.20` 是笔误，本机实测是 `.120`）

而 Jetson 现状是 eth0 `192.168.123.164` / wlan0 `192.168.123.110`，**两块网卡都在 123 网段**。
路由表里 wlan0 的 `192.168.123.0/24` metric=50 比 eth0 的 metric=100 小 → 内核默认拿 wlan0 解析
123.x。机器人 `.161` 能通，是因为另有一条专门的主机路由 `192.168.123.161 dev eth0`；而雷达
`.120` 没有这条路由，于是 ARP 漏到 wlan0（`ip neigh` 显示 `192.168.123.120 dev wlan0 INCOMPLETE`）
→ ping 全丢、雷达不出点云。**eth0 物理链路本身是好的**（`carrier=1`、1000Mb/s）。

### 6.2 ✅ 起系统前自检（在宿主机做，不连机器人控制 / 不发移动指令）
**这是每次起 `system_real_robot_g1.sh` 之前必做的一步。**
```bash
# 1) 看网卡（确认 eth0 在 123.x 且 UP；注意 wlan0 是否也在 123.x）
ip -brief addr

# 2) 确认 eth0 网线物理在线（应为 1 / yes）
cat /sys/class/net/eth0/carrier            # 1 = 网线插着且通
ethtool eth0 | grep -E "Speed|Link detected"

# 3) ping 机器人（应通，走 eth0，~0.1ms）
ping -c 3 192.168.123.161

# 4) ping 雷达（首次大概率不通，见 6.3 修复后必须通）
ping -c 3 192.168.123.120

# 5) 关键：看 ARP 是从哪块网卡解析雷达的（必须是 eth0，绝不能是 wlan0）
ip neigh | grep 192.168.123.120
```
**判读**：`.161` 不通 → 查 eth0 网线/机器人电源；`.120` 不通且 ARP 落在 `wlan0` →
路由漏到无线网卡，按 6.3 修；`.120` 走 eth0 仍 INCOMPLETE/不通 → 才是雷达供电/网线问题。

### 6.3 修复路由（实测有效，**保留 wifi**，不需要关 wlan0）
给雷达 `.120` 加一条强制走 eth0 的主机路由，并清掉 wlan0 上那条坏 ARP：
```bash
sudo ip neigh del 192.168.123.120 dev wlan0 2>/dev/null   # 清掉漏到无线的坏 ARP
sudo ip route add 192.168.123.120/32 dev eth0             # 雷达只走 eth0
ping -c 3 192.168.123.120                                 # 现在应当 ping 通
```
> 注：路由不持久，Jetson 重启后要重跑。备选方案：`sudo ip link set wlan0 down`（会断 wifi/联网），
> 或把 wlan0 换到别的网段。

**点云还需要 host 接收 IP（ping 通≠出点云）**：`MID360_config.json` 写死电脑接收 IP=`192.168.123.103`，
而 eth0 是 `.164`，否则雷达没有目标地址发点云。二选一：
```bash
# A. 给 eth0 加上 .103 别名（临时，重启失效）
sudo ip addr add 192.168.123.103/24 dev eth0
# B. 把 json 里的 192.168.123.103 改成 192.168.123.164，再 `bash docker/run.sh build` 重编 livox 配置（持久）
```

### 6.4 单独验证雷达出点云（不启动整套系统）
```bash
bash docker/run.sh ros2 launch livox_ros_driver2 msg_MID360_launch.py
# 另一终端： docker exec -it autonomy_stack bash -lc "source install/setup.bash && ros2 topic hz /livox/lidar"
```

> 机器人 `.161` 走 eth0 已可达，`unitree_sdk2` DDS 不受雷达网络问题影响。

---

## 7. 常用命令速查
```bash
bash docker/build.sh                 # 建镜像（两段式）
bash docker/run.sh build             # 编译工作区
bash docker/run.sh                   # 进容器
docker exec -it autonomy_stack bash  # 再开一个终端进同一容器
docker ps                            # 看容器
xhost +local:root                    # RViz 起不来时（X11 权限）
```

## 8. 安全提示
- **建镜像、编工作区都不会让机器人动。**
- 只有跑 `system_real_robot_g1.sh` **且** 调了 `stand_up` + `enable true` 之后机器人才会动。
- 首次务必：场地空旷、人手急停、先用小速度测试。
