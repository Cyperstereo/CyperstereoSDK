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

cmake_minimum_required(VERSION 3.1)

project(cyperstereo_samples)

# The fast ISP uses C++14 only (rather than compiler-specific extensions), so
# keep the language mode identical across MSVC, GCC and Clang.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# This samples project is a production/performance build by default.  Keep
# optimisation Release-only, but do not leave single-config Linux generators
# silently unoptimised when the caller omits CMAKE_BUILD_TYPE.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the build type (Release is recommended for the ISP)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---- Architecture-aware optimisation / NEON auto-detection -----------------
# Detects the target CPU and enables NEON/SIMD accordingly. Linux x86 has an
# explicit native/AVX2/portable policy; ARM selects the best -mcpu the
# toolchain accepts so the NEON paths are compiled in.
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

# Linux x86 deployment policy:
#   native   Build and run on the same host (default, normally fastest).
#   avx2     Deterministic AVX2 baseline, suitable for an AVX2-only fleet.
#   portable Baseline compiler ISA plus target("avx2") runtime dispatch. This
#            runs on pre-AVX2 CPUs, although a few compile-time AVX2 loops use
#            their SSE/scalar fallback.
# This option is intentionally ignored by MSVC (which retains its existing
# Release-only /arch:AVX2 target configuration) and by ARM.
set(CYPERSTEREO_X86_MODE "native" CACHE STRING
    "Linux x86 ISA policy: native|avx2|portable")
set_property(CACHE CYPERSTEREO_X86_MODE PROPERTY STRINGS native avx2 portable)
string(TOLOWER "${CYPERSTEREO_X86_MODE}" CYPERSTEREO_X86_MODE)
if(NOT CYPERSTEREO_X86_MODE MATCHES "^(native|avx2|portable)$")
  message(FATAL_ERROR
      "Invalid CYPERSTEREO_X86_MODE='${CYPERSTEREO_X86_MODE}'; use native, avx2, or portable")
endif()

# Keep the first candidate flag string the compiler actually accepts.
function(pick_first_supported_flag OUT_VAR)
  set(_result "")
  set(_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
  foreach(_flag ${ARGN})
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _cache_name "HAVE_FLAG_${_flag}")
    set(CMAKE_REQUIRED_FLAGS "${_flag}")
    check_cxx_compiler_flag("${_flag}" ${_cache_name})
    if(${_cache_name})
      set(_result "${_flag}")
      break()
    endif()
  endforeach()
  # Do not leak a candidate ISA option into later compiler/link probes.
  set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")
  set(${OUT_VAR} "${_result}" PARENT_SCOPE)
endfunction()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _proc)
set(ARCH_FLAGS "")

# -mcpu=native describes the compiler host, not the target. Reject that
# ambiguous request for ARM cross builds; TARGET_BOARD=auto selects the
# generic NEON baseline below when CMake is cross-compiling.
if(CMAKE_CROSSCOMPILING AND
   _proc MATCHES "^(aarch64|arm64|arm|armv[0-9])" AND
   TARGET_BOARD STREQUAL "native")
  message(FATAL_ERROR
      "TARGET_BOARD=native is invalid for an ARM cross build; select the actual board or use TARGET_BOARD=auto/generic")
endif()

if(MSVC)
  # Architecture options for MSVC are attached to the fast-ISP consumers
  # below. In particular, never feed GCC-style flags to cl.exe.
elseif(_proc MATCHES "^(aarch64|arm64)")
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
  elseif(TARGET_BOARD STREQUAL "generic" OR
         (TARGET_BOARD STREQUAL "auto" AND CMAKE_CROSSCOMPILING))
    # A cross compiler's -mcpu=native would tune for the build host. Keep
    # auto cross builds on the portable AArch64+NEON baseline instead.
    set(_cands "-march=armv8-a+simd")
  else() # on-device auto / native
    set(_cands "-mcpu=native" "-march=armv8-a+simd")
  endif()
  pick_first_supported_flag(ARCH_FLAGS ${_cands})
elseif(_proc MATCHES "^(arm|armv[0-9])")
  # 32-bit ARM: NEON must be requested explicitly via -mfpu.
  # Keep this branch after aarch64/arm64: names such as armv8l describe a
  # 32-bit userspace and must not receive AArch64 -mcpu/-march options.
  if(TARGET_BOARD STREQUAL "pi3")
    set(_cands "-mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard"
               "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")
  elseif(TARGET_BOARD STREQUAL "generic" OR
         (TARGET_BOARD STREQUAL "auto" AND CMAKE_CROSSCOMPILING))
    # armv8l is a 32-bit userspace. Use an ARMv7-A+NEON baseline for an
    # unspecified cross target rather than probing the build host's CPU.
    set(_cands "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard"
               "-march=armv7-a -mfpu=neon -mfloat-abi=hard")
  else() # on-device auto / native
    set(_cands "-mcpu=native -mfpu=neon-vfpv4 -mfloat-abi=hard"
               "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard"
               "-march=armv7-a -mfpu=neon -mfloat-abi=hard")
  endif()
  pick_first_supported_flag(ARCH_FLAGS ${_cands})
elseif(_proc MATCHES "^(x86_64|amd64|x86|i[3-6]86)")
  if(CYPERSTEREO_X86_MODE STREQUAL "native")
    pick_first_supported_flag(ARCH_FLAGS "-march=native")
    if(NOT ARCH_FLAGS)
      message(FATAL_ERROR "The compiler does not support -march=native")
    endif()
  elseif(CYPERSTEREO_X86_MODE STREQUAL "avx2")
    # -mavx2 deliberately does not imply FMA. Prefer host scheduling when it
    # is accepted, while keeping the generated ISA usable on any AVX2 CPU.
    pick_first_supported_flag(ARCH_FLAGS
        "-mavx2 -mtune=native"
        "-mavx2")
    if(NOT ARCH_FLAGS)
      message(FATAL_ERROR
          "CYPERSTEREO_X86_MODE=avx2 requires compiler support for -mavx2")
    endif()
  else()
    # No global -march/-mavx2: explicitly target-attributed kernels still
    # runtime-dispatch through __builtin_cpu_supports("avx2").
    set(ARCH_FLAGS "")
  endif()
else()
  # Unknown architecture: favour a compiler baseline over accidentally
  # emitting build-host instructions into a cross-compiled binary.
  set(ARCH_FLAGS "")
endif()

if(MSVC)
  # Release already supplies /O2. Do not pass GCC flags to cl.exe: apart
  # from noisy D9002 warnings they make it easy to miss a genuinely ignored
  # option. /utf-8 also keeps the bilingual ISP comments intact on Windows.
  set(OPT_FLAGS "MSVC Release /O2")
  add_compile_options(/utf-8)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Preserve the integer/floating-point equations used by the bit-exact ISP
  # comparisons. In particular, do not allow -march=native to introduce FMA
  # contraction, and override an accidental caller-supplied -ffast-math.
  set(OPT_FLAGS
      "-O3 -funroll-loops -fno-fast-math -ffp-contract=off ${ARCH_FLAGS}")
  set(CMAKE_C_FLAGS_RELEASE
      "${CMAKE_C_FLAGS_RELEASE} ${OPT_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
  # pthread is a correctness/link requirement, not a Release optimisation.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
else()
  set(OPT_FLAGS "compiler-default Release optimisation")
  message(WARNING
      "No explicit optimisation policy for compiler '${CMAKE_CXX_COMPILER_ID}'")
endif()

# RK3588 Linux enumerates the four Cortex-A55 efficiency cores as CPU0-3 and
# the four Cortex-A76 performance cores as CPU4-7. Expose the selected board
# to the header-only ISP scheduler so its four camera workers can be pinned
# one per A76. Keep this board-specific: other ARM SoCs use different layouts.
if(TARGET_BOARD STREQUAL "rk3588")
  add_definitions(-DCYPERSTEREO_RK3588=1)
endif()

message(STATUS "Target processor: ${CMAKE_SYSTEM_PROCESSOR}  (board=${TARGET_BOARD})")
if(_proc MATCHES "^(x86_64|amd64|x86|i[3-6]86)" AND NOT MSVC)
  message(STATUS "Linux x86 mode:   ${CYPERSTEREO_X86_MODE}")
endif()
message(STATUS "Arch opt flags:   ${ARCH_FLAGS}")
message(STATUS "Release opt:      ${OPT_FLAGS}")
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
# ---- Fast-ISP quality pipeline (stored under src/ISP) ----------------------
# Compile the pipeline objects directly into Cyperlib so the SDK has one
# self-contained static archive. main.cpp and the JSON parser are excluded;
# the SDK bridge supplies parameters directly in code.
set(FAST_ISP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../src/ISP")
if(NOT EXISTS "${FAST_ISP_ROOT}/srcs/include/common/pipeline.h")
  message(FATAL_ERROR
      "Fast-ISP sources were not found under '${FAST_ISP_ROOT}'")
endif()
file(GLOB FAST_ISP_MODULE_SOURCES
  "${FAST_ISP_ROOT}/srcs/sources/modules/*.cpp")
set(FAST_ISP_SOURCES
  ${FAST_ISP_MODULE_SOURCES}
  ${FAST_ISP_ROOT}/srcs/sources/common/common.cpp
  ${FAST_ISP_ROOT}/srcs/sources/common/frame.cpp
  ${FAST_ISP_ROOT}/srcs/sources/common/modules.cpp
  ${FAST_ISP_ROOT}/srcs/sources/common/pipeline.cpp
  ${FAST_ISP_ROOT}/thirdparty/easylog/easylogging++.cc
  ../src/usb/uvc/fast_isp.cc)
target_sources(Cyperlib PRIVATE ${FAST_ISP_SOURCES})
target_include_directories(Cyperlib
  PRIVATE
    ${FAST_ISP_ROOT}/srcs/include
    ${FAST_ISP_ROOT}/thirdparty/easylog
  PUBLIC
    ${OpenCV_INCLUDE_DIRS})
# ELPP_THREAD_SAFE: 4 camera pipelines log from their own threads.
# ELPP_NO_DEFAULT_LOG_FILE: don't litter myeasylog.log next to the binary.
target_compile_definitions(Cyperlib PRIVATE
  ELPP_THREAD_SAFE
  ELPP_NO_DEFAULT_LOG_FILE)
if (MSVC)
  target_compile_definitions(Cyperlib PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
target_link_libraries(Cyperlib PUBLIC ${OpenCV_LIBRARIES})

add_executable(capture_image_imu
  capture_image_imu.cc)
add_executable(save_image_imu
  save_image_imu.cc)
target_link_libraries(capture_image_imu Cyperlib ${OpenCV_LIBRARIES})
target_link_libraries(save_image_imu Cyperlib ${OpenCV_LIBRARIES})

# The deployed Windows x64 platform (Intel family 6 model 158) supports AVX2.
# Compile the header-only fast ISP consumers accordingly; without this option
# MSVC never defines __AVX2__ and the hand-written fused output backend is
# silently excluded even though the source contains it.
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  foreach(_fast_target capture_image_imu save_image_imu)
    target_compile_options(${_fast_target} PRIVATE
      $<$<CONFIG:Release>:/arch:AVX2>)
  endforeach()
endif()
