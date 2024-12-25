cmake_minimum_required(VERSION 2.8.3)
project(CyperstereoRos)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_CXX_FLAGS "-std=c++11")

set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport sensor_msgs
)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(OpenCV_DIR C:\\opencv3.4.16\\opencv\\build)
endif()

# find_package(OpenCV  REQUIRED)
set(OpenCV_DIR /usr/local/opencv-3.4.16/lib/cmake/opencv4)
find_package(OpenCV 3.4.16  REQUIRED)

find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message("OpenCV_LIBRARIES: " ${OpenCV_LIBRARIES})
message("OpenCV_LIBRARIES: " ${catkin_INCLUDE_DIRS})
message("OpenCV_LIBRARIES: " ${catkin_LIBRARIES})

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message("WIN OS.")
  add_library(Cyperlib STATIC
    ../../src/usb/uvc/uvc.h
    ../../src/usb/uvc/types.h
    ../../src/usb/uvc/win/uvc-wmf.cc)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message("MAC OS.")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message("LINUX OS.")
  add_library(Cyperlib STATIC
    ../../src/usb/uvc/uvc.h
    ../../src/usb/uvc/types.h
    ../../src/usb/uvc/linux/uvc-v4l2.cc)
else()
  message(FATAL_ERROR "Unsupported OS.")
endif()


include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

catkin_package()


add_executable(capture_image_imu capture_image_imu.cpp)
target_link_libraries(capture_image_imu
Cyperlib
${OpenCV_LIBRARIES}
${catkin_LIBRARIES}
)
