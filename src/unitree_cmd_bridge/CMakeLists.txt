cmake_minimum_required(VERSION 3.14)
project(unitree_cmd_bridge)

# --- ROS 2 deps
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Threads REQUIRED)

# --- Executable
add_executable(vel_to_sport_client src/vel_to_sport_client.cpp)
ament_target_dependencies(vel_to_sport_client rclcpp geometry_msgs)

# --- Unitree SDK2 locations (override with -DUNITREE_SDK2_ROOT=/path/to/unitree_sdk2)
set(UNITREE_SDK2_ROOT "$ENV{UNITREE_SDK2_ROOT}" CACHE PATH "Unitree SDK2 root")
if(NOT UNITREE_SDK2_ROOT)
  set(UNITREE_SDK2_ROOT "$ENV{HOME}/unitree_sdk2")
endif()

# Includes:
#  1) SDK headers
#  2) Cyclone DDS C++ shim root: provides <dds/topic/TopicTraits.hpp>
#  3) (also the parent) org/eclipse/cyclonedds/...
target_include_directories(vel_to_sport_client PRIVATE
  ${UNITREE_SDK2_ROOT}/include
  ${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx
  ${UNITREE_SDK2_ROOT}/thirdparty/include
)

# Static Unitree lib (Jetson aarch64)
set(UNITREE_SDK2_LIBDIR "${UNITREE_SDK2_ROOT}/lib/aarch64")
find_library(UNITREE_SDK2_LIB
  NAMES unitree_sdk2
  PATHS ${UNITREE_SDK2_LIBDIR}
  NO_DEFAULT_PATH
)
if(NOT UNITREE_SDK2_LIB)
  message(FATAL_ERROR "Could not find libunitree_sdk2.a under ${UNITREE_SDK2_LIBDIR}")
endif()

# Optionally/automatically link Cyclone DDS C lib if present in thirdparty
set(UNITREE_TP_LIBDIRS
  ${UNITREE_SDK2_ROOT}/thirdparty/lib
  ${UNITREE_SDK2_ROOT}/thirdparty/lib/aarch64
)
find_library(CYCLONEDDS_C
  NAMES ddsc
  PATHS ${UNITREE_TP_LIBDIRS}
  NO_DEFAULT_PATH
)

# --- Link (PLAIN signature to match ament_target_dependencies)
if(CYCLONEDDS_C)
  message(STATUS "Found Cyclone DDS C lib: ${CYCLONEDDS_C}")
  target_link_libraries(vel_to_sport_client
    ${UNITREE_SDK2_LIB}
    ${CYCLONEDDS_C}
    Threads::Threads dl rt
  )
else()
  target_link_libraries(vel_to_sport_client
    ${UNITREE_SDK2_LIB}
    Threads::Threads dl rt
  )
endif()

# If you encounter unresolved symbols from the Unitree static archive, try enabling whole-archive:
# set(USE_WHOLE_ARCHIVE ON)
if(DEFINED USE_WHOLE_ARCHIVE AND USE_WHOLE_ARCHIVE)
  target_link_options(vel_to_sport_client PRIVATE -Wl,--whole-archive ${UNITREE_SDK2_LIB} -Wl,--no-whole-archive)
endif()

# --- Install
install(TARGETS vel_to_sport_client
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
