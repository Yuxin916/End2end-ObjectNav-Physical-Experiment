cmake_minimum_required(VERSION 3.8)
project(g1_rviz_panel)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

set(CMAKE_AUTOMOC ON)

qt5_wrap_cpp(MOC_FILES
  include/g1_rviz_panel/g1_control_panel.hpp
)

add_library(${PROJECT_NAME} SHARED
  src/g1_control_panel.cpp
  ${MOC_FILES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rviz_common
  pluginlib
  std_msgs
)
target_link_libraries(${PROJECT_NAME} Qt5::Widgets)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/
  DESTINATION include
)
install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME})
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

ament_package()
