cmake_minimum_required(VERSION 2.8.3)
project(img_to_bag)
set(CMAKE_CXX_FLAGS "-std=c++11")


set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport sensor_msgs
)

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



include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

catkin_package()


add_executable(img_to_bag img_to_bag.cpp)
target_link_libraries(img_to_bag  
${OpenCV_LIBRARIES}
${catkin_LIBRARIES} 
)
