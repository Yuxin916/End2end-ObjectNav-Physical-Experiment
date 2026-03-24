# docker/Dockerfile
FROM osrf/ros:jazzy-desktop

# --- Build args (override at build to match your host) ---
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
ARG DEBIAN_FRONTEND=noninteractive

# Use bash for RUN so future steps can `source` things if needed
SHELL ["/bin/bash", "-lc"]

# --- Base utilities, locales, build chain, ROS dev tools ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      sudo git curl ca-certificates wget nano tzdata locales \
      build-essential cmake \
      python3-pip python3-venv python3-colcon-common-extensions \
      ros-jazzy-desktop-full \
      ros-jazzy-pcl-ros libpcl-dev \
      # QoL
      ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-tf2-tools \
      bash-completion && \
    rm -rf /var/lib/apt/lists/*

# Locale (avoid weird UTF-8 issues)
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# --- GUI / GL bits for RViz & friends ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 libglx-mesa0 libglvnd0 \
      libxext6 libx11-6 libxrender1 libxi6 libxrandr2 \
      mesa-utils vulkan-tools && \
    rm -rf /var/lib/apt/lists/*

# --- Non-root user (reuse existing GID if present) ---
RUN set -eux; \
    # Pick or create the group for ${GID}
    if getent group "${GID}" >/dev/null; then \
      GROUP_NAME="$(getent group ${GID} | cut -d: -f1)"; \
    else \
      GROUP_NAME="${USERNAME}"; \
      groupadd -g "${GID}" "${GROUP_NAME}"; \
    fi; \
    # Create or rename the user to ${USERNAME} with UID ${UID}
    if getent passwd "${UID}" >/dev/null; then \
      # A user with this UID exists; rename it to ${USERNAME}
      OLD_USER="$(getent passwd ${UID} | cut -d: -f1)"; \
      usermod -l "${USERNAME}" -d "/home/${USERNAME}" -m "${OLD_USER}" || true; \
      usermod -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    else \
      useradd -m -s /bin/bash -u "${UID}" -g "${GID}" "${USERNAME}"; \
    fi; \
    usermod -aG sudo "${USERNAME}"; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-"${USERNAME}"


# --- ROS env defaults ---
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_DOMAIN_ID=42

# System-wide shell QoL
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc && \
    echo "export RMW_IMPLEMENTATION=\$RMW_IMPLEMENTATION" >> /etc/bash.bashrc && \
    echo "export ROS_DOMAIN_ID=\$ROS_DOMAIN_ID" >> /etc/bash.bashrc && \
    (echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /etc/bash.bashrc || true)

# --- Workspace layout (renamed) ---
USER ${USERNAME}
WORKDIR /home/${USERNAME}
RUN mkdir -p /home/${USERNAME}/ros2_jazzy_ubuntu_2404/src
ENV COLCON_DEFAULTS_FILE=/home/${USERNAME}/.colcon_defaults.yaml

# colcon defaults (faster multi-core builds)
RUN printf -- "---\n" \
            "build:\n" \
            "  parallel-workers: auto\n" \
            "  symlink-install: true\n" \
            "test:\n" \
            "  parallel-workers: auto\n" \
            > ${COLCON_DEFAULTS_FILE}

# --- Developer aliases ---
RUN echo "# Aliases for git + GPU" >> /home/${USERNAME}/.bashrc && \
    echo "alias gs='git status'" >> /home/${USERNAME}/.bashrc && \
    echo "alias ga='git add'" >> /home/${USERNAME}/.bashrc && \
    echo "alias gp='git push'" >> /home/${USERNAME}/.bashrc && \
    echo "alias gcm='git commit -m'" >> /home/${USERNAME}/.bashrc && \
    echo "alias gpu='watch -n1 nvidia-smi'" >> /home/${USERNAME}/.bashrc && \
    echo "alias gr='git remote -v'" >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${GID} /home/${USERNAME}/.bashrc

# Final working directory = workspace root
WORKDIR /home/${USERNAME}/ros2_jazzy_ubuntu_2404
