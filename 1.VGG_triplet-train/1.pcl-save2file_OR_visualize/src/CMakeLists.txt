cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(pcl_visualizer_viewports)

#########################################################
# FIND PCL
#########################################################
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#########################################################
# Include Files
#########################################################
add_executable (matrix_transform matrix_transform.cpp)
target_link_libraries (matrix_transform ${OPENGL_LIBRARIES} ${PCL_LIBRARIES} )
