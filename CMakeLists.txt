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
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/slda.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
