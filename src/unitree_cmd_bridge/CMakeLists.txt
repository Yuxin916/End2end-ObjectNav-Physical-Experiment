cmake_minimum_required(VERSION 3.10)
project(unitree_cmd_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(vel_to_sport_bridge src/vel_to_sport_bridge.cpp)
ament_target_dependencies(vel_to_sport_bridge rclcpp geometry_msgs)

# Linux sockets are in libc; no extra link libs needed on Ubuntu
install(TARGETS vel_to_sport_bridge
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
