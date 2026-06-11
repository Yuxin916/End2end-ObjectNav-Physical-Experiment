# autonomy_stack on Unitree G1 — Docker 运行指南

ROS2 **Jazzy** 导航栈跑在 Unitree G1 的 Jetson 上的完整说明：安装、编译、模块、
容器内外的启动方式，以及雷达起前的网络配置与自检。

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

### 4.2 容器内：启动整套系统
```bash
# 进容器后：
./system_real_robot_g1.sh robot_ip:=192.168.123.161 connection_method:=LocalSTA control_mode:=wireless_controller
```
- 默认 `control_backend:=sdk`（unitree_sdk2，无需 AES key）。
- 回退旧 WebRTC：追加 `control_backend:=webrtc`（需 `~/.unitree_g1.env` 里的 `UNITREE_AES_KEY`）。
- 启动会拉起：local_planner、terrain、sensor_scan、arise_slam、visualization、joy、
  livox(Mid-360)、g1_sdk_bridge，并打开 RViz。

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

---

## 6. 雷达（Livox Mid-360）—— 起之前必看

### 6.1 网络规划
`src/utilities/livox_ros_driver2/config/MID360_config.json` 里写死：
- **电脑接收 IP（host_net_info）= `192.168.123.103`**
- **雷达 IP = `192.168.123.120`**

而 Jetson 现状是 eth0 `192.168.123.164` / wlan0 `192.168.123.110`，**两块网卡都在 123 网段**，
会导致发往雷达 `.120` 的包从 wlan0 漏出去 → 雷达不出点云。

### 6.2 起雷达前的自检（在宿主机做，不连机器人控制）
```bash
# 1) 看网卡（确认 eth0 在 123.x；注意 wlan0 是否也在 123.x）
ip -brief addr

# 2) ping 雷达（应通；不通见下方排查）
ping -c 3 192.168.123.120

# 3) ping 机器人（应通，走 eth0）
ping -c 3 192.168.123.161

# 4) 看 ARP 是从哪块网卡解析雷达的（应是 eth0，不该是 wlan0）
ip neigh | grep 192.168.123.120
```

### 6.3 修复（二选一 / 通常都要）
```bash
# A. 让 123.x 只走 eth0（关掉同段的 wlan0，或把 wlan0 改到别的网段）
sudo ip link set wlan0 down

# B. 给 eth0 配上 MID360_config 要的接收 IP .103
sudo ip addr add 192.168.123.103/24 dev eth0
#   或者：把 json 里的 192.168.123.103 改成 192.168.123.164，然后 `bash docker/run.sh build` 重编 livox 配置
```
改完再 `ping 192.168.123.120` 确认 ARP 走 eth0、能通。

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
