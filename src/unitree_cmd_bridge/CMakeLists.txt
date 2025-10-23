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

# --- Unitree SDK2 root (override with -DUNITREE_SDK2_ROOT=/path/to/unitree_sdk2)
set(UNITREE_SDK2_ROOT "$ENV{UNITREE_SDK2_ROOT}" CACHE PATH "Unitree SDK2 root")
if(NOT UNITREE_SDK2_ROOT)
  set(UNITREE_SDK2_ROOT "$ENV{HOME}/unitree_sdk2")
endif()

# Include dirs:
#  1) Unitree SDK headers
#  2) Cyclone DDS C++ “ddscxx” include root (gives <dds/topic/TopicTraits.hpp>)
#  3) Parent include for org/eclipse/cyclonedds/...
target_include_directories(vel_to_sport_client PRIVATE
  ${UNITREE_SDK2_ROOT}/include
  ${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx
  ${UNITREE_SDK2_ROOT}/thirdparty/include
)

# --- Static Unitree lib (Jetson aarch64)
set(UNITREE_SDK2_LIBDIR "${UNITREE_SDK2_ROOT}/lib/aarch64")
find_library(UNITREE_SDK2_LIB
  NAMES unitree_sdk2
  PATHS ${UNITREE_SDK2_LIBDIR}
  NO_DEFAULT_PATH
)
if(NOT UNITREE_SDK2_LIB)
  message(FATAL_ERROR "Could not find libunitree_sdk2.a under ${UNITREE_SDK2_LIBDIR}")
endif()

# --- Cyclone DDS shared libs (from SDK thirdparty)
set(UNITREE_TP_LIBDIR "${UNITREE_SDK2_ROOT}/thirdparty/lib/aarch64")
find_library(CYCLONEDDSXX_LIB NAMES ddscxx PATHS ${UNITREE_TP_LIBDIR} NO_DEFAULT_PATH)
find_library(CYCLONEDDS_C_LIB NAMES ddsc   PATHS ${UNITREE_TP_LIBDIR} NO_DEFAULT_PATH)
if(NOT CYCLONEDDSXX_LIB OR NOT CYCLONEDDS_C_LIB)
  message(FATAL_ERROR "Could not find ddscxx and/or ddsc in ${UNITREE_TP_LIBDIR}")
endif()

# --- Link (PLAIN signature; order matters for static -> shared deps)
target_link_libraries(vel_to_sport_client
  ${UNITREE_SDK2_LIB}
  ${CYCLONEDDSXX_LIB}
  ${CYCLONEDDS_C_LIB}
  Threads::Threads dl rt
)

# --- RPATH: make the installed node find ddscxx/ddsc without env vars
set(CMAKE_BUILD_RPATH ${UNITREE_TP_LIBDIR})
set(CMAKE_INSTALL_RPATH ${UNITREE_TP_LIBDIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# If you still see unresolved symbols from unitree_sdk2.a, you can force-include the archive:
# target_link_options(vel_to_sport_client PRIVATE -Wl,--whole-archive ${UNITREE_SDK2_LIB} -Wl,--no-whole-archive)

# --- Install
install(TARGETS vel_to_sport_client
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
