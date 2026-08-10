cmake_minimum_required(VERSION 3.8)
project(data_cap)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

include(CheckCXXCompilerFlag)
find_package(Threads REQUIRED)

# Keep the ROS 2 executable's ISA policy explicit.  "native" maximises the
# on-device build, "avx2" makes the x86 deployment requirement deterministic,
# and "portable" avoids whole-binary ISA assumptions.
set(CYPERSTEREO_CPU_TUNE "native" CACHE STRING
    "Linux CPU tuning: native|avx2|portable")
set_property(CACHE CYPERSTEREO_CPU_TUNE PROPERTY STRINGS native avx2 portable)
set(TARGET_BOARD "auto" CACHE STRING
    "ARM CPU tuning: auto|native|rk3588|sunrise-x5|pi5|pi4|pi3|jetson-xavier|jetson-orin|jetson-thor|generic")
set_property(CACHE TARGET_BOARD PROPERTY STRINGS
    auto native rk3588 sunrise-x5 pi5 pi4 pi3
    jetson-xavier jetson-orin jetson-thor generic)

function(cyperstereo_pick_first_supported_flag out_var)
  set(_selected "")
  set(_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
  foreach(_candidate ${ARGN})
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _cache_name
           "CYPERSTEREO_HAS_${_candidate}")
    set(CMAKE_REQUIRED_FLAGS "${_candidate}")
    check_cxx_compiler_flag("${_candidate}" ${_cache_name})
    if(${_cache_name})
      set(_selected "${_candidate}")
      break()
    endif()
  endforeach()
  set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")
  set(${out_var} "${_selected}" PARENT_SCOPE)
endfunction()

set(CYPERSTEREO_ARCH_OPTION "")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _cyper_cpu)
  if(CMAKE_CROSSCOMPILING AND TARGET_BOARD STREQUAL "native")
    message(FATAL_ERROR
            "TARGET_BOARD=native is invalid while cross-compiling; select the target board or generic")
  endif()
  if(_cyper_cpu MATCHES "x86_64|amd64|i[3-6]86")
    if(CYPERSTEREO_CPU_TUNE STREQUAL "native")
      check_cxx_compiler_flag("-march=native" CYPERSTEREO_HAS_MARCH_NATIVE)
      if(CYPERSTEREO_HAS_MARCH_NATIVE)
        set(CYPERSTEREO_ARCH_OPTION "-march=native")
      else()
        message(WARNING "-march=native is unavailable; using portable x86")
      endif()
    elseif(CYPERSTEREO_CPU_TUNE STREQUAL "avx2")
      check_cxx_compiler_flag("-mavx2" CYPERSTEREO_HAS_MAVX2)
      if(NOT CYPERSTEREO_HAS_MAVX2)
        message(FATAL_ERROR "CYPERSTEREO_CPU_TUNE=avx2 requires -mavx2")
      endif()
      set(CYPERSTEREO_ARCH_OPTION "-mavx2")
    elseif(NOT CYPERSTEREO_CPU_TUNE STREQUAL "portable")
      message(FATAL_ERROR
              "Unknown CYPERSTEREO_CPU_TUNE=${CYPERSTEREO_CPU_TUNE}")
    endif()
  elseif(_cyper_cpu MATCHES "^(aarch64|arm64)")
    if(NOT CYPERSTEREO_CPU_TUNE MATCHES "^(native|portable)$")
      message(FATAL_ERROR
              "ARM supports CYPERSTEREO_CPU_TUNE=native or portable")
    endif()
    if(CYPERSTEREO_CPU_TUNE STREQUAL "portable")
      set(_cyper_arm_candidates "-march=armv8-a+simd")
    elseif(TARGET_BOARD STREQUAL "native" OR
       (TARGET_BOARD STREQUAL "auto" AND NOT CMAKE_CROSSCOMPILING))
      set(_cyper_arm_candidates "-mcpu=native" "-march=armv8-a+simd")
    elseif(TARGET_BOARD STREQUAL "rk3588")
      set(_cyper_arm_candidates "-mcpu=cortex-a76.cortex-a55"
          "-mcpu=cortex-a76" "-march=armv8.2-a+simd")
    elseif(TARGET_BOARD STREQUAL "sunrise-x5")
      set(_cyper_arm_candidates "-mcpu=cortex-a55" "-march=armv8.2-a+simd")
    elseif(TARGET_BOARD STREQUAL "pi5")
      set(_cyper_arm_candidates "-mcpu=cortex-a76" "-march=armv8.2-a+simd")
    elseif(TARGET_BOARD STREQUAL "pi4")
      set(_cyper_arm_candidates "-mcpu=cortex-a72" "-march=armv8-a+simd")
    elseif(TARGET_BOARD STREQUAL "pi3")
      set(_cyper_arm_candidates "-mcpu=cortex-a53" "-march=armv8-a+simd")
    elseif(TARGET_BOARD STREQUAL "jetson-xavier")
      set(_cyper_arm_candidates "-march=armv8.2-a+fp16+dotprod"
          "-march=armv8.2-a+simd" "-march=armv8-a+simd")
    elseif(TARGET_BOARD STREQUAL "jetson-orin")
      set(_cyper_arm_candidates "-mcpu=cortex-a78ae" "-mcpu=cortex-a78"
          "-march=armv8.2-a+fp16+dotprod" "-march=armv8-a+simd")
    elseif(TARGET_BOARD STREQUAL "jetson-thor")
      set(_cyper_arm_candidates "-mcpu=neoverse-v3ae" "-mcpu=neoverse-v2"
          "-march=armv9-a" "-march=armv8-a+simd")
    elseif(TARGET_BOARD STREQUAL "auto" OR TARGET_BOARD STREQUAL "generic")
      set(_cyper_arm_candidates "-march=armv8-a+simd")
    else()
      message(FATAL_ERROR "Unknown TARGET_BOARD=${TARGET_BOARD}")
    endif()
    cyperstereo_pick_first_supported_flag(
        CYPERSTEREO_ARCH_OPTION ${_cyper_arm_candidates})
    if(NOT CYPERSTEREO_ARCH_OPTION)
      message(FATAL_ERROR "No supported ARM64 SIMD option for TARGET_BOARD=${TARGET_BOARD}")
    endif()
  elseif(_cyper_cpu MATCHES "^(arm|armv[0-9])")
    if(NOT CYPERSTEREO_CPU_TUNE MATCHES "^(native|portable)$")
      message(FATAL_ERROR
              "ARM supports CYPERSTEREO_CPU_TUNE=native or portable")
    endif()
    if(CYPERSTEREO_CPU_TUNE STREQUAL "portable")
      set(_cyper_arm_candidates
          "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard"
          "-march=armv7-a -mfpu=neon -mfloat-abi=hard")
    elseif(TARGET_BOARD STREQUAL "native" OR
       (TARGET_BOARD STREQUAL "auto" AND NOT CMAKE_CROSSCOMPILING))
      set(_cyper_arm_candidates
          "-mcpu=native -mfpu=neon-vfpv4 -mfloat-abi=hard"
          "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")
    elseif(TARGET_BOARD MATCHES "^(rk3588|sunrise-x5|pi5|pi4|pi3)$")
      set(_cyper_arm_candidates
          "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard"
          "-march=armv7-a -mfpu=neon -mfloat-abi=hard")
    elseif(TARGET_BOARD STREQUAL "auto" OR TARGET_BOARD STREQUAL "generic")
      set(_cyper_arm_candidates
          "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard"
          "-march=armv7-a -mfpu=neon -mfloat-abi=hard")
    else()
      message(FATAL_ERROR
              "TARGET_BOARD=${TARGET_BOARD} has no supported ARMv7 profile")
    endif()
    cyperstereo_pick_first_supported_flag(
        CYPERSTEREO_ARCH_OPTION ${_cyper_arm_candidates})
    if(NOT CYPERSTEREO_ARCH_OPTION)
      message(FATAL_ERROR "The ARMv7 compiler must support -mfpu=neon")
    endif()
  elseif(NOT CYPERSTEREO_CPU_TUNE STREQUAL "portable")
    message(WARNING "Unknown CPU '${CMAKE_SYSTEM_PROCESSOR}'; portable build")
  endif()
endif()

set(CYPERSTEREO_ARCH_OPTIONS "")
if(CYPERSTEREO_ARCH_OPTION)
  separate_arguments(CYPERSTEREO_ARCH_OPTIONS NATIVE_COMMAND
                     "${CYPERSTEREO_ARCH_OPTION}")
endif()

function(cyperstereo_optimise_target target_name)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target_name} PRIVATE $<$<CONFIG:Release>:-O3>)
    # Preserve the fixed rounding boundaries used by the shared ISP source.
    target_compile_options(${target_name} PRIVATE
      $<$<CONFIG:Release>:-fno-fast-math>
      $<$<CONFIG:Release>:-ffp-contract=off>)
    foreach(_arch_option IN LISTS CYPERSTEREO_ARCH_OPTIONS)
      target_compile_options(${target_name} PRIVATE
        "$<$<CONFIG:Release>:${_arch_option}>")
    endforeach()
  endif()
endfunction()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message("WIN OS.")
  add_library(Cyperlib STATIC
    ../../../src/usb/uvc/uvc.h
    ../../../src/usb/uvc/types.h
    ../../../src/usb/uvc/win/uvc-wmf.cc)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(FATAL_ERROR "The ROS 2 target has no macOS UVC library definition")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message("LINUX OS.")
  add_library(Cyperlib STATIC
    ../../../src/usb/uvc/uvc.h
    ../../../src/usb/uvc/types.h
    ../../../src/usb/uvc/linux/uvc-v4l2.cc)
else()
  message(FATAL_ERROR "Unsupported OS.")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(capture_image_imu src/capture_image_imu.cpp)
ament_target_dependencies(capture_image_imu
  rclcpp
  cv_bridge
  image_transport
  sensor_msgs
)
cyperstereo_optimise_target(Cyperlib)
cyperstereo_optimise_target(capture_image_imu)
target_link_libraries(capture_image_imu
  Cyperlib
  ${OpenCV_LIBRARIES}
  Threads::Threads
)

install(
  TARGETS capture_image_imu
  DESTINATION lib/${PROJECT_NAME}
)

message(STATUS
  "Cyperstereo CPU tune=${CYPERSTEREO_CPU_TUNE} "
  "board=${TARGET_BOARD} arch='${CYPERSTEREO_ARCH_OPTION}' "
  "build=${CMAKE_BUILD_TYPE}")

ament_package()
