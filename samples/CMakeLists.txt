# Copyright 2018 Slightech Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0)

project(cyperstereo_samples)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS -pthread)

message(STATUS "C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")


#set(OpenCV_DIR /usr/local/opencv-3.4.16/lib/cmake/opencv4)
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(OpenCV_DIR C:\\opencv3.4.16\\opencv\\build)
endif()

find_package(OpenCV  REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message("WIN OS.")
  add_library(Cyperlib STATIC
    ../src/usb/uvc/uvc.h
    ../src/usb/uvc/types.h
    ../src/usb/uvc/win/uvc-wmf.cc)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message("MAC OS.")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message("LINUX OS.")
  add_library(Cyperlib STATIC
    ../src/usb/uvc/uvc.h
    ../src/usb/uvc/types.h
    ../src/usb/uvc/linux/uvc-v4l2.cc)
else()
  message(FATAL_ERROR "Unsupported OS.")
endif()
add_executable(capture_image_imu
  capture_image_imu.cc)
add_executable(save_image_imu
  save_image_imu.cc)
target_link_libraries(capture_image_imu Cyperlib ${OpenCV_LIBRARIES})
target_link_libraries(save_image_imu Cyperlib ${OpenCV_LIBRARIES})
