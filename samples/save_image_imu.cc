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
#include <condition_variable>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <mutex>
#include "string"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <array>
#include <condition_variable>
#include <thread>
#include "../src/usb/uvc/cyperstereo_api.h"


const double g = 9.7887;

CYPERSTEREO_USE_NAMESPACE

std::queue<std::pair<double, std::array<double, 6>> > IMU;
std::queue<std::pair<double, std::vector<cv::Mat>> > IMAGE;
std::mutex m_buf;
std::condition_variable con;
void GetData(std::queue<std::pair<double, std::array<double, 6>> > &imu_data, std::queue<std::pair<double, std::vector<cv::Mat>> > &image_data);
void DataFlow();
void InputIMAGE(const double timestamp, const cv::Mat& cam0_img, const cv::Mat& cam1_img);
void InputIMU(const double timestamp, double gyro_x, double gyro_y, double gyro_z, double acc_x, double acc_y, double acc_z);

int main(int argc, char *argv[]) {
  std::thread data_flow{DataFlow};
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
    InputIMAGE(image_timestamp, left_image, right_image);
    for (int i = 0; i <= frame_info.framestream.imu.imu_count; ++i) {
      InputIMU(frame_info.framestream.imu.imu_timestamp[i], frame_info.framestream.imu.gyro_x[i], frame_info.framestream.imu.gyro_y[i], frame_info.framestream.imu.gyro_z[i],
                frame_info.framestream.imu.acc_x[i], frame_info.framestream.imu.acc_y[i], frame_info.framestream.imu.acc_z[i]);
    }
  }
  cyperstereo::uvc::stop_streaming(*cyperstereo_device);
  
  return 0;
}

void InputIMU(const double timestamp, double gyro_x, double gyro_y, double gyro_z, double acc_x, double acc_y, double acc_z) {
    m_buf.lock();
    std::array<double, 6> imu_data{gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z};
    IMU.push(make_pair(timestamp, imu_data));
    m_buf.unlock();
    con.notify_one();
}

void InputIMAGE(const double timestamp, const cv::Mat& cam0_img, const cv::Mat& cam1_img) {
    m_buf.lock();
    std::vector<cv::Mat> image_data;
    image_data.push_back(cam0_img.clone());
    image_data.push_back(cam1_img.clone());
    IMAGE.push(make_pair(timestamp, image_data));
    m_buf.unlock();
    con.notify_one();
}

void DataFlow() {
  int count = 0;
  while (true) {
      std::queue<std::pair<double, std::array<double, 6>> > imu_data;
      std::queue<std::pair<double, std::vector<cv::Mat>> > image_data;
      std::unique_lock<std::mutex> lk(m_buf);
      con.wait(lk, [&] {
        GetData(imu_data, image_data);
        return imu_data.size() != 0 || image_data.size() != 0;});
      lk.unlock();
      while (!imu_data.empty()) {
        double imu_timestamp = imu_data.front().first;
        double gyro_x = imu_data.front().second[0];
        double gyro_y = imu_data.front().second[1];
        double gyro_z = imu_data.front().second[2];
        double acc_x = imu_data.front().second[3];
        double acc_y = imu_data.front().second[4];
        double acc_z = imu_data.front().second[5];
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout.precision(6);
        std::cout << "imu_timestamp " << imu_timestamp << " " << gyro_x << " "<< gyro_y << " "<< gyro_z << " " << acc_x << " "<< acc_y << " " << acc_z << std::endl;
        std::ofstream foutC("./imu/imu.csv", std::ios::app);
        foutC.setf(std::ios::fixed, std::ios::floatfield);
        foutC.precision(4);
        foutC << imu_timestamp << ",";
        foutC.precision(6);
        foutC << gyro_x << ","
            << gyro_y << ","
            << gyro_z << ","
            << acc_x * g << ","
            << acc_y * g << ","
            << acc_z * g
            << std::endl;
        foutC.close();
        imu_data.pop();
      }
      while (!image_data.empty()) {
        double image_timestamp = image_data.front().first;
        cv::Mat left_image = image_data.front().second[0];
        cv::Mat right_image = image_data.front().second[1];
        if (!left_image.empty() && !right_image.empty()) {
          std::cout << "image_timestamp " << image_timestamp << std::endl;
          // TicToc tf;
          // float eps = 0.0001;//eps的取值很关键（乘于255的平方）
          // cv::Mat left_image_res = FastGuidedfilter(left_image, 9, eps, 3);
          // cv::Mat right_image_res = FastGuidedfilter(right_image, 9, eps, 3);
          // double tic2 = static_cast<double>(cv::getTickCount());
          // std::cout <<tf.toc() << std::endl;
          if (count % 2 != 0) {         
            cv::imwrite("./left/" + std::to_string(static_cast<int>(image_timestamp * 10000)) + ".png", left_image);
            cv::imwrite("./right/" + std::to_string(static_cast<int>(image_timestamp * 10000)) + ".png", right_image);
          }
          count++;
        }
        image_data.pop();
      }
  }
}

void GetData(std::queue<std::pair<double, std::array<double, 6>> > &imu_data, std::queue<std::pair<double, std::vector<cv::Mat>> > &image_data) {
  while (!IMU.empty()) {
    imu_data.push(IMU.front());
    IMU.pop();
  }
  while (!IMAGE.empty()) {
    image_data.push(IMAGE.front());
    IMAGE.pop();
  }
}