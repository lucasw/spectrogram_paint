cmake_minimum_required(VERSION 2.8.3)
project(spectrogram_paint_ros)

find_package(catkin REQUIRED COMPONENTS
  # dynamic_reconfigure
  # nodelet
  geometry_msgs
  message_generation
  roscpp
  roslint
  sensor_msgs
  std_msgs
)

# roslint_cpp()

# generate_dynamic_reconfigure_options(
#   cfg/Delay.cfg
# )

add_message_files(
  FILES
  Audio.msg
)

# add_service_files(
#   FILES
# )

generate_messages(
  DEPENDENCIES
  std_msgs
#  geometry_msgs
)

catkin_package(
  # INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS nodelet roscpp sensor_msgs std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# add_executable(${PROJECT_NAME}
# )
# target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
# add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
# add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencpp)

# TODO(lucasw) is this needed?
if(catkin_EXPORTED_LIBRARIES)
  add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_LIBRARIES})
endif()

# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# install(FILES
#   nodelet_${PROJECT_NAME}.xml
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )
