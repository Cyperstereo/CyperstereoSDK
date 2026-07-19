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

# ---- Architecture-aware optimisation / NEON auto-detection -----------------
# Detects the target CPU and enables NEON/SIMD accordingly. On x86 it keeps
# -march=native. On ARM it selects the best -mcpu the toolchain accepts so the
# NEON de-interleave path (CYPERSTEREO_HAVE_NEON) is compiled in.
#
# The board is auto-detected; override it when cross-compiling, e.g.:
#   cmake -DTARGET_BOARD=rk3588 ..         # RK3588 (Cortex-A76+A55)
#   cmake -DTARGET_BOARD=sunrise-x5 ..     # Horizon Sunrise/Journey X5 (A55)
#   cmake -DTARGET_BOARD=pi5 ..            # Raspberry Pi 5 (Cortex-A76)
#   cmake -DTARGET_BOARD=pi4 ..            # Raspberry Pi 4 (Cortex-A72)
#   cmake -DTARGET_BOARD=pi3 ..            # Raspberry Pi 3 (Cortex-A53)
#   cmake -DTARGET_BOARD=jetson-xavier ..  # Jetson AGX/NX Xavier (NVIDIA Carmel, ARMv8.2)
#   cmake -DTARGET_BOARD=jetson-orin ..    # Jetson AGX/Orin NX/Nano (Cortex-A78AE)
#   cmake -DTARGET_BOARD=jetson-thor ..    # Jetson Thor (Neoverse-V3AE, ARMv9)
# On a Jetson (or any board) that you build ON-DEVICE, the default 'auto' uses
# -mcpu=native, which is ideal; the explicit boards above are for cross-builds.
include(CheckCXXCompilerFlag)

set(TARGET_BOARD "auto" CACHE STRING
    "CPU tuning: auto|native|rk3588|sunrise-x5|pi5|pi4|pi3|jetson-xavier|jetson-orin|jetson-thor|generic")

# Keep the first candidate flag string the compiler actually accepts.
function(pick_first_supported_flag OUT_VAR)
  set(_result "")
  foreach(_flag ${ARGN})
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _cache_name "HAVE_FLAG_${_flag}")
    set(CMAKE_REQUIRED_FLAGS "${_flag}")
    check_cxx_compiler_flag("${_flag}" ${_cache_name})
    if(${_cache_name})
      set(_result "${_flag}")
      break()
    endif()
  endforeach()
  set(${OUT_VAR} "${_result}" PARENT_SCOPE)
endfunction()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _proc)
set(ARCH_FLAGS "")

if(_proc MATCHES "aarch64|arm64|armv8")
  # 64-bit ARM: NEON (ASIMD) is always present; pick the best CPU tuning.
  if(TARGET_BOARD STREQUAL "rk3588")
    set(_cands "-mcpu=cortex-a76.cortex-a55" "-mcpu=cortex-a76" "-march=armv8.2-a+simd")
  elseif(TARGET_BOARD STREQUAL "sunrise-x5")
    set(_cands "-mcpu=cortex-a55" "-march=armv8.2-a+simd")
  elseif(TARGET_BOARD STREQUAL "pi5")
    set(_cands "-mcpu=cortex-a76" "-march=armv8.2-a+simd")
  elseif(TARGET_BOARD STREQUAL "pi4")
    set(_cands "-mcpu=cortex-a72" "-march=armv8-a+simd")
  elseif(TARGET_BOARD STREQUAL "pi3")
    set(_cands "-mcpu=cortex-a53" "-march=armv8-a+simd")
  elseif(TARGET_BOARD STREQUAL "jetson-xavier")
    # NVIDIA Carmel (ARMv8.2-A, has fp16 + dotprod); GCC has no -mcpu=carmel.
    set(_cands "-march=armv8.2-a+fp16+dotprod" "-march=armv8.2-a+simd" "-march=armv8-a+simd")
  elseif(TARGET_BOARD STREQUAL "jetson-orin")
    # Cortex-A78AE (ARMv8.2-A).
    set(_cands "-mcpu=cortex-a78ae" "-mcpu=cortex-a78" "-march=armv8.2-a+fp16+dotprod" "-march=armv8-a+simd")
  elseif(TARGET_BOARD STREQUAL "jetson-thor")
    # Neoverse-V3AE (ARMv9-A); fall back progressively for older toolchains.
    set(_cands "-mcpu=neoverse-v3ae" "-mcpu=neoverse-v2" "-march=armv9-a" "-march=armv8-a+simd")
  elseif(TARGET_BOARD STREQUAL "generic")
    set(_cands "-march=armv8-a+simd")
  else() # auto / native
    set(_cands "-mcpu=native" "-march=armv8-a+simd")
  endif()
  pick_first_supported_flag(ARCH_FLAGS ${_cands})
elseif(_proc MATCHES "arm|armv7")
  # 32-bit ARM: NEON must be requested explicitly via -mfpu.
  if(TARGET_BOARD STREQUAL "pi3")
    set(_cands "-mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard"
               "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")
  else()
    set(_cands "-mcpu=native -mfpu=neon-vfpv4 -mfloat-abi=hard"
               "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard"
               "-march=armv7-a -mfpu=neon -mfloat-abi=hard")
  endif()
  pick_first_supported_flag(ARCH_FLAGS ${_cands})
else()
  # x86 / other: native tuning (unchanged behaviour).
  pick_first_supported_flag(ARCH_FLAGS "-march=native")
endif()

set(OPT_FLAGS "-O3 -funroll-loops ${ARCH_FLAGS}")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_FLAGS} -pthread")

# RK3588 Linux enumerates the four Cortex-A55 efficiency cores as CPU0-3 and
# the four Cortex-A76 performance cores as CPU4-7. Expose the selected board
# to the header-only ISP scheduler so its four camera workers can be pinned
# one per A76. Keep this board-specific: other ARM SoCs use different layouts.
if(TARGET_BOARD STREQUAL "rk3588")
  add_definitions(-DCYPERSTEREO_RK3588=1)
endif()

message(STATUS "Target processor: ${CMAKE_SYSTEM_PROCESSOR}  (board=${TARGET_BOARD})")
message(STATUS "Arch opt flags:   ${ARCH_FLAGS}")
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
