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
#include <mutex>
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
  TicToc t_frame;
  while (true) {
    cyperstereo::WaitForStream(frame_info);

    double image_timestamp = 0.0;
    cv::Mat left_image;
    cv::Mat right_image;
    cyperstereo::IMUStreamData imu_data{};
    cyperstereo::GNSSStreamData gnss_data{};

    {
      std::lock_guard<std::mutex> lock(frame_info.mtx);
      image_timestamp = frame_info.framestream.image_timestamp;
      frame_info.framestream.left_image.copyTo(left_image);
      frame_info.framestream.right_image.copyTo(right_image);
      imu_data = frame_info.framestream.imu;
      gnss_data = frame_info.framestream.gnss;
    }

    if (count % 2 != 0) {
      std::cout << "image_timestamp " << image_timestamp << std::endl;
      cv::imshow("left", left_image);
      cv::imshow("right", right_image);
      cv::waitKey(1);
    } 

    //imu data
    for (int i = 0; i <= imu_data.imu_count; ++i) {
        double imu_timestamp = imu_data.imu_timestamp[i];
        double gyro_x = imu_data.gyro_x[i];
        double gyro_y = imu_data.gyro_y[i];
        double gyro_z = imu_data.gyro_z[i];
        double acc_x = imu_data.acc_x[i] * g;
        double acc_y = imu_data.acc_y[i] * g;
        double acc_z = imu_data.acc_z[i] * g;
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout.precision(6);
        std::cout << "imu_timestamp " << imu_timestamp << " " << gyro_x << " "<< gyro_y << " "<< gyro_z << " " << acc_x << " "<< acc_y << " " << acc_z << std::endl;
    }

    //gnss data
    static std::string last_gnss_time;
    if (!gnss_data.gnss_utc_time.empty() && gnss_data.valid == true && gnss_data.gnss_utc_time != last_gnss_time) {
      last_gnss_time = gnss_data.gnss_utc_time;
      std::cout << std::fixed << std::setprecision(10)
                << "  gnss_timestamp: " << gnss_data.gnss_timestamp
                << "  gnss_utc_time: " << gnss_data.gnss_utc_time
                << "  latitude: " << gnss_data.latitude << "  longitude: " << gnss_data.longitude
                << "  altitude: " << gnss_data.altitude
                << "  fix_type: " << gnss_data.fix_type << "  satellites_used: " << gnss_data.satellites_used
                << "  gps_geoid_height: " << gnss_data.gps_geoid_height
                << "  hdop: " << gnss_data.hdop << "  vdop: " << gnss_data.vdop << "  pdop: " << gnss_data.pdop
                << "  velocity: " << gnss_data.velocity << "  heading: " << gnss_data.heading
                << std::endl;
    }

    ++count;
    if (count % 100 == 0) {
    	double frame_rate = 100 / (t_frame.toc() / 1000);
    	t_frame.tic();
    	std::cout << "frame_rate " << frame_rate << std::endl;
    }
    
  }
  cyperstereo::uvc::stop_streaming(*cyperstereo_device);
  
  return 0;
}
