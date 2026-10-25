# Use the identical SDK ISP implementation in samples, ROS1 and ROS2.
function(cyperstereo_add_isp target sdk_root)
  get_filename_component(_sdk "${sdk_root}" ABSOLUTE)
  set(_isp "${_sdk}/src/ISP")
  if(NOT EXISTS "${_isp}/srcs/include/common/pipeline.h")
    message(FATAL_ERROR "Fast-ISP sources not found under '${_isp}'")
  endif()
  file(GLOB _modules "${_isp}/srcs/sources/modules/*.cpp")
  target_sources(${target} PRIVATE
    ${_modules}
    ${_isp}/srcs/sources/common/common.cpp
    ${_isp}/srcs/sources/common/frame.cpp
    ${_isp}/srcs/sources/common/modules.cpp
    ${_isp}/srcs/sources/common/pipeline.cpp
    ${_isp}/thirdparty/easylog/easylogging++.cc
    ${_sdk}/src/usb/uvc/fast_isp.cc)
  target_include_directories(${target}
    PRIVATE ${_isp}/srcs/include ${_isp}/thirdparty/easylog
    PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(${target} PRIVATE
    ELPP_THREAD_SAFE ELPP_NO_DEFAULT_LOG_FILE)
  if(MSVC)
    target_compile_definitions(${target} PUBLIC _CRT_SECURE_NO_WARNINGS)
  endif()
  target_link_libraries(${target} PUBLIC ${OpenCV_LIBRARIES})
endfunction()
