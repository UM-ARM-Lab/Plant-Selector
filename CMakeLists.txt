cmake_minimum_required(VERSION 3.0.2)
project(plant_selector)

find_package(catkin REQUIRED COMPONENTS 
  roscpp
  rospy
  std_msgs
  interactive_markers
  rosbag
  pcl_conversions
  pcl_ros
  rviz
  pluginlib
  )

catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

set(CMAKE_AUTOMOC ON)

find_package(Qt5 ${rviz_QT_VERSION} EXACT REQUIRED Core Widgets Quick)
set(QT_LIBRARIES Qt5::Widgets)
qt5_wrap_ui(${PROJECT_NAME}_UIS_H ${${PROJECT_NAME}_UIS})
qt5_wrap_cpp(MOC_FILES_SELECTOR include/plant_selector/publishing_selector.hpp)
qt5_wrap_cpp(MOC_FILES_MAIN include/plant_selector/main_panel.hpp)
qt5_wrap_cpp(MOC_FILES_ROSBAG include/plant_selector/rosbag_panel.hpp)
add_definitions(-DQT_NO_KEYWORDS)

add_library(publishing_selector
  src/publishing_selector.cpp
  ${MOC_FILES_SELECTOR}
  )

add_library(main_panel
  src/main_panel.cpp
  ${${PROJECT_NAME}_UIS_H}
  ${MOC_FILES_MAIN}
  )

add_library(rosbag_panel
  src/rosbag_panel.cpp
  ${${PROJECT_NAME}_UIS_H}
  ${MOC_FILES_ROSBAG}
  )

add_executable(real_time_handler 
  src/real_time_handler.cpp
  include/plant_selector/real_time_handler.hpp
  )

add_executable(send_pcd_msg 
  scripts/send_pcd_msg.cpp
  )

target_link_libraries(publishing_selector
  ${QT_LIBRARIES}
  ${catkin_LIBRARIES}
  )

target_link_libraries(main_panel
  ${QT_LIBRARIES}
  ${catkin_LIBRARIES}
  )

target_link_libraries(rosbag_panel
  ${QT_LIBRARIES}
  ${catkin_LIBRARIES}
  )

target_link_libraries(real_time_handler
  ${catkin_LIBRARIES}
  )

target_link_libraries(send_pcd_msg
  ${catkin_LIBRARIES}

  )
install(TARGETS publishing_selector main_panel rosbag_panel real_time_handler send_pcd_msg
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(FILES
  publishing_selector_plugin_description.xml
  main_panel_plugin_description.xml
  rosbag_panel_plugin_description.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})