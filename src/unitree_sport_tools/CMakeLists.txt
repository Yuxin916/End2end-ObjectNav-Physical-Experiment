cmake_minimum_required(VERSION 3.10)
project(unitree_sport_tools CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Make sure CMake can find Unitree’s config.
# Adjust these paths to match your install.
list(APPEND CMAKE_PREFIX_PATH
  "/opt/unitree_robotics/lib/cmake"
)

find_package(ament_cmake REQUIRED)
find_package(unitree_sdk2 REQUIRED)  # Provided by Unitree SDK2

add_executable(stand_example_go2 src/stand_example_go2.cpp)
target_link_libraries(stand_example_go2 PRIVATE unitree_sdk2)

add_executable(sports_client src/sports_client.cpp)
target_link_libraries(sports_client PRIVATE unitree_sdk2)

add_executable(udp_to_sport_client src/udp_to_sport_client.cpp)
target_link_libraries(udp_to_sport_client PRIVATE unitree_sdk2)

# Optional: if your SDK needs ddscxx headers/libs explicitly, Unitree’s config
# should already handle that. If not, you could add include/link dirs here.

# Install so `ros2 run` works
install(TARGETS stand_example_go2
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS sports_client
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS udp_to_sport_client
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
