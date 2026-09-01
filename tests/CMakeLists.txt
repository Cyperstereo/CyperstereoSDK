cmake_minimum_required(VERSION 3.5)
project(cyperstereo_portable_metadata_tests LANGUAGES CXX)

# The production SDK is C++14.  This small compatibility layer deliberately
# also compiles as C++11 so older ARM Linux toolchains can consume the layout
# definitions without OpenCV, UVC or architecture-specific intrinsics.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_executable(smartsens_metadata_test smartsens_metadata_test.cc)
target_include_directories(smartsens_metadata_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/usb/uvc")

add_test(NAME smartsens_serial_s0_s1_s2_and_metadata_2_3_2_4_2_5_2_6
         COMMAND smartsens_metadata_test)
