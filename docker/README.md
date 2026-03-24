

## Build the Docker image (one-time, or when change the Dockerfile)
```
cd autonomy_stack_mecanum_wheel_platform/docker
docker build \
  --no-cache \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USERNAME=$USER \
  -t jazzy-dev:latest .
```

## Run the container (every time when want to work inside it)
```
xhost +local:root

docker run --rm -it \
  --name autonomy_jazzy \
  --net=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/..:/home/$USER/ros2_jazzy_ubuntu_2404/src/autonomy_stack_mecanum_wheel_platform \
  jazzy-dev:latest \
  bash
```

## Inside the container, build the repo
```
cd ~/ros2_jazzy_ubuntu_2404
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --packages-skip arise_slam_mid360 arise_slam_mid360_msgs livox_ros_driver2
source install/setup.bash
```

## Attach to a running container
```
docker exec -it autonomy_jazzy bash
```