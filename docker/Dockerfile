# Stage 1 (heavy deps) of the autonomy_stack image for Unitree G1 Jetson.
# Builds ROS2 Jazzy + the vendored C++ deps (Sophus, Ceres, GTSAM, Livox-SDK2)
# into /usr/local. Produces the slow-changing base; the WebRTC control layer
# is added by docker/Dockerfile.webrtc (FROM this image, tagged :heavy).
#
# Host is Ubuntu 20.04/Focal + L4T R35, which can't run Jazzy natively, so the
# whole stack runs in this native arm64 Jazzy container. The target launch is
# CPU-only -- no CUDA/nvidia runtime; GPU modules (sam2_detector,
# vlm_nav_bridge) are intentionally NOT built.
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# System + ROS dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ninja-build pkg-config \
      python3-pip python3-venv python3-colcon-common-extensions python3-rosdep \
      libeigen3-dev libboost-all-dev libtbb-dev \
      libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev gfortran \
      libpcl-dev libopencv-dev libyaml-cpp-dev libzstd-dev \
      libapr1-dev libaprutil1-dev \
      qtbase5-dev libqt5core5a libqt5gui5 libqt5widgets5 \
      ros-jazzy-desktop \
      ros-jazzy-pcl-ros ros-jazzy-pcl-conversions \
      ros-jazzy-cv-bridge ros-jazzy-image-transport \
      ros-jazzy-joy ros-jazzy-tf2-geometry-msgs ros-jazzy-tf2-sensor-msgs \
      ros-jazzy-rosbag2 ros-jazzy-rosbag2-cpp \
      ros-jazzy-rviz2 ros-jazzy-rviz-common ros-jazzy-rviz-rendering \
      ros-jazzy-rviz-default-plugins ros-jazzy-pluginlib ros-jazzy-message-filters \
      x11-apps mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Vendored C++ dependencies -> /usr/local
# ---------------------------------------------------------------------------
WORKDIR /deps

COPY src/slam/dependency/Sophus /deps/Sophus
RUN cd /deps/Sophus && mkdir -p build && cd build \
    && cmake .. -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" && make install

COPY src/slam/dependency/ceres-solver /deps/ceres-solver
RUN cd /deps/ceres-solver && mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" && make install

COPY src/slam/dependency/gtsam /deps/gtsam
RUN cd /deps/gtsam && mkdir -p build && cd build \
    && cmake .. -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
                -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
                -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" && make install

COPY src/utilities/livox_ros_driver2/Livox-SDK2 /deps/Livox-SDK2
RUN cd /deps/Livox-SDK2 && mkdir -p build && cd build \
    && cmake .. && make -j"$(nproc)" && make install

RUN ldconfig

WORKDIR /workspace
CMD ["bash"]
