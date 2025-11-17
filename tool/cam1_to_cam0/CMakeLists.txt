cmake_minimum_required(VERSION 2.8.3)
project(cam1_to_cam0)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(Eigen3)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(cam1_to_cam0 cam1_to_cam0.cpp)
