cmake_minimum_required(VERSION 3.0.2)
project(slda_bci)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  processing_bci
  rosneuro_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs processing_bci rosneuro_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/slda.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# test rosnode NODE
add_executable(test_publisher_slda test/publisher.cpp)
target_link_libraries(test_publisher_slda 
  ${catkin_LIBRARIES}
)

add_executable(test_logger_slda test/logger_fbcsp.cpp)
target_link_libraries(test_logger_slda 
  ${catkin_LIBRARIES}
)