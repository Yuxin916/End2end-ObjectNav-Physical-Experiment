cmake_minimum_required(VERSION 3.5)
project(vehicle_simulator)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  "${PROJECT_SOURCE_DIR}/include"
  /usr/local/include
  /usr/include
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
# find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)

add_executable(sim_image_repub src/sim_image_repub.cpp)
add_executable(vehicleSimulator src/vehicleSimulator.cpp)
ament_target_dependencies(vehicleSimulator rclcpp std_msgs sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs message_filters pcl_ros pcl_conversions)
ament_target_dependencies(sim_image_repub rclcpp std_msgs sensor_msgs cv_bridge)
target_link_libraries(vehicleSimulator ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(sim_image_repub ${OpenCV_LIBRARIES})

add_executable(vel_to_sport_client src/vel_to_sport_client.cpp)
ament_target_dependencies(vel_to_sport_client rclcpp geometry_msgs)

# --- Where your SDK lives (override with -DUNITREE_SDK2_ROOT=/path/to/sdk if needed)
set(UNITREE_SDK2_ROOT "$ENV{UNITREE_SDK2_ROOT}" CACHE PATH "Unitree SDK2 root")
if(NOT UNITREE_SDK2_ROOT)
  set(UNITREE_SDK2_ROOT "$ENV{HOME}/unitree_sdk2")
endif()

# aarch64 (Jetson) lib dir
set(UNITREE_SDK2_LIBDIR "${UNITREE_SDK2_ROOT}/lib/aarch64")

# Includes
target_include_directories(vel_to_sport_client PRIVATE
  ${UNITREE_SDK2_ROOT}/include
)

# Static archive (.a)
find_library(UNITREE_SDK2_LIB
  NAMES unitree_sdk2
  PATHS ${UNITREE_SDK2_LIBDIR}
  NO_DEFAULT_PATH
)

if(NOT UNITREE_SDK2_LIB)
  message(FATAL_ERROR "Could not find libunitree_sdk2.a under ${UNITREE_SDK2_LIBDIR}")
endif()

# Threads (and likely dl/rt on 20.04 aarch64)
find_package(Threads REQUIRED)

# PLAIN signature (matches ament_target_dependencies)
target_link_libraries(vel_to_sport_client ${UNITREE_SDK2_LIB} Threads::Threads dl rt)

# If you see undefined references from the SDK due to static linking,
# try force-linking all objects in the archive:
# target_link_options(vel_to_sport_client PRIVATE -Wl,--whole-archive ${UNITREE_SDK2_LIB} -Wl,--no-whole-archive)

install(TARGETS vel_to_sport_client
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)


install(TARGETS
  vehicleSimulator
  sim_image_repub
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY
  launch
  log
  mesh
  rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
