// Copyright 2018 Slightech Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iostream>
#include "string"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include "../src/usb/uvc/cyperstereo_api.h"


const double g = 9.7887;

CYPERSTEREO_USE_NAMESPACE

int main(int argc, char *argv[]) {
  int count = 0;
  std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
  if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device)) {
    return 0;
  }
  cyperstereo::FrameInfo frame_info{};
  cyperstereo::uvc::set_device_mode(
      *cyperstereo_device, 752, 480, static_cast<int>(cyperstereo::Format::YUYV), 60,
      [&frame_info](const void *data, std::function<void()> continuation) {
        cyperstereo::SetStreamData(frame_info, data, continuation);
      });
  cyperstereo::uvc::start_streaming(*cyperstereo_device, 0);
  while (true) {
    cyperstereo::WaitForStream(frame_info);
    double image_timestamp = frame_info.framestream.image_timestamp;
    cv::Mat left_image = frame_info.framestream.left_image;
    cv::Mat right_image = frame_info.framestream.right_image;
    if (count++ % 2 != 0) {
      std::cout << "image_timestamp " << image_timestamp << std::endl;
      cv::imshow("left", left_image);
      cv::imshow("right", right_image);
      cv::waitKey(1);
    } 
    for (int i = 0; i <= frame_info.framestream.imu.imu_count; ++i) {
        double imu_timestamp = frame_info.framestream.imu.imu_timestamp[i];
        double gyro_x = frame_info.framestream.imu.gyro_x[i];
        double gyro_y = frame_info.framestream.imu.gyro_y[i];
        double gyro_z = frame_info.framestream.imu.gyro_z[i];
        double acc_x = frame_info.framestream.imu.acc_x[i] * g;
        double acc_y = frame_info.framestream.imu.acc_y[i] * g;
        double acc_z = frame_info.framestream.imu.acc_z[i] * g;
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout.precision(6);
        std::cout << "imu_timestamp " << imu_timestamp << " " << gyro_x << " "<< gyro_y << " "<< gyro_z << " " << acc_x << " "<< acc_y << " " << acc_z << std::endl;
    }
  }
  cyperstereo::uvc::stop_streaming(*cyperstereo_device);
  
  return 0;
}

