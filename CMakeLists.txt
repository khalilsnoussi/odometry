cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)


project(odometry)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(odometry odometry.cpp)
target_link_libraries(odometry ${OpenCV_LIBS})



