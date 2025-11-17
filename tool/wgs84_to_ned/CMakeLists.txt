cmake_minimum_required(VERSION 2.8.3)
project(llt2ned)
set(CMAKE_CXX_FLAGS "-std=c++11")


find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include ${CMAKE_CURRENT_SOURCE_DIR})

find_package(GeographicLib REQUIRED)
include_directories(${GeographicLib_INCLUDE_DIRS})

add_executable(llt2ned llt2ned.cpp)
target_link_libraries(llt2ned  ${GeographicLib_LIBRARIES})
