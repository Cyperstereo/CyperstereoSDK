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
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include "../src/usb/uvc/cyperstereo_api.h"
#include "../src/usb/uvc/hdr_isp.h"
#include "../src/usb/uvc/tic_toc.h"
#include "../src/usb/uvc/thread_priority.h"


const double g = 9.7887;

CYPERSTEREO_USE_NAMESPACE

std::queue<std::pair<double, std::array<double, 6>> > IMU;
struct ImageRecord {
  double timestamp;
  uint32_t hardware_version;
  uint32_t software_version;
  std::array<double, 4> camera_gain{};
  std::vector<cv::Mat> images;
};
std::queue<ImageRecord> IMAGE;
struct GnssRecord {
  double gnss_timestamp;
  std::string gnss_utc_time;
  double latitude;
  double longitude;
  double altitude;
  int fix_type;
  int satellites_used;
  double gps_geoid_height;
  double velocity;
  double heading;
  double hdop;
  double vdop;
  double pdop;
};
std::queue<GnssRecord> GNSS;
std::mutex m_buf;
std::condition_variable con;
void GetData(std::queue<std::pair<double, std::array<double, 6>> > &imu_data, std::queue<ImageRecord> &image_data, std::queue<GnssRecord> &gnss_data);
void DataFlow();
void InputIMAGE(const double timestamp, uint32_t hardware_version,
                uint32_t software_version,
                const std::array<double, 4> &camera_gain,
                const std::vector<cv::Mat>& images);
void InputIMU(const double timestamp, double gyro_x, double gyro_y, double gyro_z, double acc_x, double acc_y, double acc_z);
void InputGNSS(const cyperstereo::GNSSStreamData &gnss);

int main(int argc, char *argv[]) {
#ifdef _WIN32
  _mkdir("left");
  _mkdir("right");
  _mkdir("left_front");
  _mkdir("right_front");
  _mkdir("imu");
  _mkdir("gnss");
#else
  mkdir("left", 0755);
  mkdir("right", 0755);
  mkdir("left_front", 0755);
  mkdir("right_front", 0755);
  mkdir("imu", 0755);
  mkdir("gnss", 0755);
#endif
  // We parallelise the per-camera work ourselves, so disable OpenCV's internal
  // threading to avoid oversubscribing the cores.
  cv::setNumThreads(1);
  // Raise the main (capture) thread to real-time priority, just below the poll
  // and worker threads. All priorities are configured in thread_priority.h.
  cyperstereo::ApplyThreadPriority(cyperstereo::ThreadRole::kMain, "main");
  std::thread data_flow{DataFlow};
  std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
  if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device)) {
    return 0;
  }
  cyperstereo::FrameInfo frame_info{};
  // The USB serial number is a static device property, so read it once and use
  // it (or, when unset, the advertised UVC size) to auto-select the camera
  // profile (MT9V034 vs SmartSens) before starting the stream.
  const std::string serial_num =
      cyperstereo::uvc::get_serial_number(*cyperstereo_device);
  const cyperstereo::CameraProfile &profile =
      cyperstereo::SelectProfile(serial_num, *cyperstereo_device);
  frame_info.Init(profile);
  frame_info.framestream.serial_num = serial_num;
  std::cout << "camera: " << profile.name << "  serial: "
            << (serial_num.empty() ? "(none)" : serial_num) << "  "
            << profile.frame_width << "x" << profile.frame_height << "@"
            << profile.fps << "  cameras: " << profile.num_cameras << std::endl;
  const int num_cameras = profile.num_cameras;
  cyperstereo::uvc::set_device_mode(
      *cyperstereo_device, profile.frame_width, profile.frame_height,
      static_cast<int>(cyperstereo::Format::YUYV), profile.fps,
      [&frame_info](const void *data, std::function<void()> continuation) {
        cyperstereo::SetStreamData(frame_info, data, continuation);
      });
  cyperstereo::uvc::start_streaming(*cyperstereo_device, 0);

  // Persistent per-camera buffers, swapped with the framestream planes under
  // the lock (O(1), like capture_image_imu) so the USB poll thread is never
  // blocked while we copy pixels. The unused planes stay allocated but idle for
  // the 2-camera (MT9V034) profile.
  cv::Mat left_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat right_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat left_front_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat right_front_image(profile.frame_height, profile.cam_width, CV_8U);

  while (true) {
    cyperstereo::WaitForStream(frame_info);

    double image_timestamp = 0.0;
    uint32_t hardware_version = 0;
    uint32_t software_version = 0;
    std::array<double, 4> camera_gain{};
    cyperstereo::IMUStreamData imu_data{};
    cyperstereo::GNSSStreamData gnss_data{};

    {
      std::lock_guard<std::mutex> lock(frame_info.mtx);
      image_timestamp = frame_info.framestream.image_timestamp;
      hardware_version = frame_info.framestream.hardware_version;
      software_version = frame_info.framestream.software_version;
      cv::swap(frame_info.framestream.left_image, left_image);
      cv::swap(frame_info.framestream.right_image, right_image);
      if (num_cameras >= 4) {
        cv::swap(frame_info.framestream.left_front_image, left_front_image);
        cv::swap(frame_info.framestream.right_front_image, right_front_image);
        for (int i = 0; i < 4; ++i)
          camera_gain[i] = frame_info.framestream.camera_gain[i];
      }
      imu_data = frame_info.framestream.imu;
      gnss_data = frame_info.framestream.gnss;
    }

    std::vector<cv::Mat> images;
    images.push_back(left_image);
    images.push_back(right_image);
    if (num_cameras >= 4) {
      images.push_back(left_front_image);
      images.push_back(right_front_image);
    }
    InputIMAGE(image_timestamp, hardware_version, software_version,
               camera_gain, images);  // clones into the save queue
    for (int i = 0; i < imu_data.imu_count; ++i) {
      InputIMU(imu_data.imu_timestamp[i], imu_data.gyro_x[i], imu_data.gyro_y[i],
               imu_data.gyro_z[i], imu_data.acc_x[i], imu_data.acc_y[i],
               imu_data.acc_z[i]);
    }
    static std::string last_gnss_time;
    if (!gnss_data.gnss_utc_time.empty() && gnss_data.valid == true &&
        gnss_data.gnss_utc_time != last_gnss_time) {
      InputGNSS(gnss_data);
      last_gnss_time = gnss_data.gnss_utc_time;
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

void InputGNSS(const cyperstereo::GNSSStreamData &gnss) {
    std::lock_guard<std::mutex> lk(m_buf);
    GnssRecord record;
    record.gnss_timestamp = gnss.gnss_timestamp;
    record.gnss_utc_time = gnss.gnss_utc_time;
    record.latitude = gnss.latitude;
    record.longitude = gnss.longitude;
    record.altitude = gnss.altitude;
    record.fix_type = gnss.fix_type;
    record.satellites_used = gnss.satellites_used;
    record.gps_geoid_height = gnss.gps_geoid_height;
    record.velocity = gnss.velocity;
    record.heading = gnss.heading;
    record.hdop = gnss.hdop;
    record.vdop = gnss.vdop;
    record.pdop = gnss.pdop;
    GNSS.push(record);
    con.notify_one();
}


void InputIMAGE(const double timestamp, uint32_t hardware_version,
                 uint32_t software_version,
                 const std::array<double, 4> &camera_gain,
                 const std::vector<cv::Mat>& images) {
    m_buf.lock();
    ImageRecord record{
        timestamp, hardware_version, software_version, camera_gain, {}};
    record.images.reserve(images.size());
    for (const auto &img : images) {
      record.images.push_back(img.clone());
    }
    IMAGE.push(std::move(record));
    m_buf.unlock();
    con.notify_one();
}

void DataFlow() {
  // This thread does the heavy per-camera work (WB + demosaic + PNG encode),
  // so run it at real-time worker priority (configured in thread_priority.h).
  cyperstereo::ApplyThreadPriority(cyperstereo::ThreadRole::kWorker, "dataflow");

  int count = 0;
  // Pre-allocated demosaic output buffers for the 4-camera (SmartSens) path,
  // reused every processed frame to avoid reallocating on the hot path. Sized
  // lazily on first use so the same binary works for either camera profile.
  cv::Mat left_color, right_color, left_front_color, right_front_color;
  WhiteBalance fast_wb[4];
  const bool use_fast_balanced = FastBalancedIspEnabled();
  std::cout << "ISP mode: "
            << (use_fast_balanced ? "fast-balanced" : "quality-reference")
            << std::endl;
  if (use_fast_balanced) {
    std::cout << "Fast ISP: gamma="
              << (FastBalancedGammaEnabled() ? "on" : "off")
              << " hue_guard="
              << (FastBalancedHueGuardEnabled() ? "on" : "off")
              << " bayer_nr="
              << (FastBalancedBayerNrEnabled() ? "adaptive>=3x" : "off")
              << " demosaic="
              << FastBalancedDemosaicName()
              << " saturation=" << IspSaturation() << std::endl;
  }
  while (true) {
      std::queue<std::pair<double, std::array<double, 6>> > imu_data;
      std::queue<ImageRecord> image_data;
      std::queue<GnssRecord> gnss_data;
      std::unique_lock<std::mutex> lk(m_buf);
      con.wait(lk, [&] {
        GetData(imu_data, image_data, gnss_data);
        return imu_data.size() != 0 || image_data.size() != 0 || gnss_data.size() != 0;});
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
       // std::cout << "imu_timestamp " << imu_timestamp << " " << gyro_x << " "<< gyro_y << " "<< gyro_z << " " << acc_x << " "<< acc_y << " " << acc_z << std::endl;
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
      while (!gnss_data.empty()) {
        const auto &record = gnss_data.front();
        std::ofstream foutG("./gnss/gnss.csv", std::ios::app);
        foutG.setf(std::ios::fixed, std::ios::floatfield);
        foutG.precision(6);
        foutG << record.gnss_timestamp << ","
              << record.gnss_utc_time << ","
              << record.latitude << ","
              << record.longitude << ","
              << record.altitude << ","
              << record.fix_type << ","
              << record.satellites_used << ","
              << record.gps_geoid_height << ","
              << record.velocity << ","
              << record.heading << ","
              << record.hdop << ","
              << record.vdop << ","
              << record.pdop
              << std::endl;
        foutG.close();
        gnss_data.pop();
      }
      while (!image_data.empty()) {
        const ImageRecord &record = image_data.front();
        double image_timestamp = record.timestamp;
        const std::vector<cv::Mat> &imgs = record.images;
        const size_t n = imgs.size();
        // 4-camera (SmartSens) path: RAW Bayer -> white balance + demosaic, save
        // as colour. 2-camera (MT9V034) path: monochrome, saved as-is.
        const bool four_ok =
            n >= 4 && !imgs[0].empty() && !imgs[1].empty() &&
            !imgs[2].empty() && !imgs[3].empty();
        const bool two_ok =
            n >= 2 && !imgs[0].empty() && !imgs[1].empty();
        if (four_ok) {
          if (count % 2 == 0) {
            cv::Mat left_image = imgs[0];
            cv::Mat right_image = imgs[1];
            cv::Mat left_front_image = imgs[2];
            cv::Mat right_front_image = imgs[3];
            
            // Run the selected ISP pipeline; fast-balanced is the default.
            const std::string image_name = std::to_string(static_cast<int>(image_timestamp * 10000)) + ".png";
            const BayerConversion image13_bayer = SelectBayerConversion(
                record.hardware_version, record.software_version, 0);
            if (use_fast_balanced) {
              ApplyFastBalancedISPParallel({
                  {left_image, left_color, fast_wb[0], "fast-cam1",
                   record.camera_gain[0], image13_bayer},
                  {right_image, right_color, fast_wb[1], "fast-cam2",
                   record.camera_gain[1]},
                  {left_front_image, left_front_color, fast_wb[2], "fast-cam3",
                   record.camera_gain[2], image13_bayer},
                  {right_front_image, right_front_color, fast_wb[3], "fast-cam4",
                   record.camera_gain[3]},
              });
            } else {
              std::thread t2([&] {
                ApplyHdrIsp(right_image, right_color, "cam2",
                            BayerConversion::kColorBayerRg2Bgr,
                            record.camera_gain[1]);
              });
              std::thread t3([&] {
                ApplyHdrIsp(left_front_image, left_front_color, "cam3",
                            image13_bayer, record.camera_gain[2]);
              });
              std::thread t4([&] {
                ApplyHdrIsp(right_front_image, right_front_color, "cam4",
                            BayerConversion::kColorBayerRg2Bgr,
                            record.camera_gain[3]);
              });
              ApplyHdrIsp(left_image, left_color, "cam1", image13_bayer,
                          record.camera_gain[0]);
              t2.join();
              t3.join();
              t4.join();
            }
            std::thread t2([&] {
              cv::imwrite("./right/" + image_name, right_color);
            });
            std::thread t3([&] {
              cv::imwrite("./left_front/" + image_name, left_front_color);
            });
            std::thread t4([&] {
              cv::imwrite("./right_front/" + image_name, right_front_color);
            });
            cv::imwrite("./left/" + image_name, left_color);
            t2.join();
            t3.join();
            t4.join();

          }
          count++;
        } else if (two_ok) {
          if (count % 3 == 0) {
            // MT9V034 is monochrome: save the two raw planes directly.
            const std::string image_name = std::to_string(static_cast<int>(image_timestamp * 10000)) + ".png";
            cv::imwrite("./left/" + image_name, imgs[0]);
            cv::imwrite("./right/" + image_name, imgs[1]);
          }
          count++;
        }
        image_data.pop();
      }
  }
}

void GetData(std::queue<std::pair<double, std::array<double, 6>> > &imu_data, std::queue<ImageRecord> &image_data, std::queue<GnssRecord> &gnss_data) {
  while (!IMU.empty()) {
    imu_data.push(IMU.front());
    IMU.pop();
  }
  while (!IMAGE.empty()) {
    image_data.push(IMAGE.front());
    IMAGE.pop();
  }
  while (!GNSS.empty()) {
    gnss_data.push(GNSS.front());
    GNSS.pop();
  }
}
