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
#include <csignal>
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
#include "../src/usb/uvc/camera_output.h"
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
  std::array<double, 2> c56_gain{};  // six-camera C5/C6 AE gain
  int process_cameras{0};
  std::vector<cv::Mat> images;
  std::array<bool, 4> fpga_bayer_nr{};
  RawTelemetrySnapshot raw_telemetry{};
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
bool capture_finished = false;  // guarded by m_buf
static volatile std::sig_atomic_t stop_requested = 0;
extern "C" void save_stop_handler(int) { stop_requested = 1; }
void GetData(std::queue<std::pair<double, std::array<double, 6>> > &imu_data, std::queue<ImageRecord> &image_data, std::queue<GnssRecord> &gnss_data);
void DataFlow();
void InputIMAGE(const double timestamp, uint32_t hardware_version,
                uint32_t software_version,
                const std::array<double, 4> &camera_gain,
                const std::array<double, 2> &c56_gain,
                const std::array<bool, 4> &fpga_bayer_nr,
                 const RawTelemetrySnapshot &raw_telemetry,
                int process_cameras,
                const std::vector<cv::Mat>& images);
void InputIMU(const double timestamp, double gyro_x, double gyro_y, double gyro_z, double acc_x, double acc_y, double acc_z);
void InputGNSS(const cyperstereo::GNSSStreamData &gnss);

int main(int argc, char *argv[]) {
#ifdef _WIN32
  _mkdir("cam1");
  _mkdir("cam2");
  _mkdir("cam3");
  _mkdir("cam4");
  _mkdir("cam5");
  _mkdir("cam6");
  _mkdir("imu");
  _mkdir("gnss");
#else
  mkdir("cam1", 0755);
  mkdir("cam2", 0755);
  mkdir("cam3", 0755);
  mkdir("cam4", 0755);
  mkdir("cam5", 0755);
  mkdir("cam6", 0755);
  mkdir("imu", 0755);
  mkdir("gnss", 0755);
#endif
  // We parallelise the per-camera work ourselves, so disable OpenCV's internal
  // threading to avoid oversubscribing the cores.
  cv::setNumThreads(1);
  // Raise the main (capture) thread to real-time priority, just below the poll
  // and worker threads. All priorities are configured in thread_priority.h.
  cyperstereo::ApplyThreadPriority(cyperstereo::ThreadRole::kMain, "main");
  std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
  if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device)) {
    return 1;
  }
  cyperstereo::FrameInfo frame_info{};
  // The USB serial number is a static device property, so read it once and use
  // it (or, when unset, the advertised UVC size) to auto-select the camera
  // profile (MT9V034, SC136HGS or AR0234) before starting the stream.
  const std::string serial_num =
      cyperstereo::uvc::get_serial_number(*cyperstereo_device);
  const cyperstereo::CameraProfile &profile =
      cyperstereo::SelectProfile(serial_num, *cyperstereo_device);
  frame_info.Init(profile);
  frame_info.framestream.serial_num = serial_num;
  const int num_cameras = profile.num_cameras;
  const int process_cameras =
      cyperstereo::SmartSensProcessedCameraCount(serial_num, num_cameras);
  std::cout << "camera: " << profile.name << "  serial: "
            << (serial_num.empty() ? "(none)" : serial_num) << "  "
            << profile.frame_width << "x" << profile.frame_height << "@"
            << profile.fps << "  cameras: " << num_cameras;
  if (process_cameras == 1 && num_cameras > 1)
    std::cout << "  process: C1 only (S1 color SKU)";
  std::cout << std::endl;
  const bool is_bayer = cyperstereo::IsSmartSensProfile(profile);
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

  std::signal(SIGINT, save_stop_handler);
  std::signal(SIGTERM, save_stop_handler);
  std::thread data_flow{DataFlow};
  while (!stop_requested) {
    if (!WaitForCameraFrame(frame_info, [] { return !stop_requested; })) break;

    double image_timestamp = 0.0;
    RawTelemetrySnapshot raw_telemetry;
    uint32_t hardware_version = 0;
    uint32_t software_version = 0;
    std::array<double, 4> camera_gain{};
    std::array<double, 2> c56_gain{};
    std::array<bool, 4> fpga_bayer_nr{};
    cyperstereo::IMUStreamData imu_data{};
    cyperstereo::GNSSStreamData gnss_data{};

    {
      std::lock_guard<std::mutex> lock(frame_info.mtx);
      image_timestamp = frame_info.framestream.image_timestamp;
      raw_telemetry = frame_info.framestream.raw_telemetry;
      hardware_version = frame_info.framestream.hardware_version;
      software_version = frame_info.framestream.software_version;
      cv::swap(frame_info.framestream.left_image, left_image);
      if (process_cameras >= 2)
        cv::swap(frame_info.framestream.right_image, right_image);
      if (process_cameras >= 4) {
        cv::swap(frame_info.framestream.left_front_image, left_front_image);
        cv::swap(frame_info.framestream.right_front_image, right_front_image);
      }
      if (is_bayer) {
        for (int i = 0; i < process_cameras; ++i) {
          camera_gain[i] = frame_info.framestream.isp_camera_gain[i];
          fpga_bayer_nr[i] = frame_info.framestream.fpga_bayer_nr[i];
        }
        for (int c = 0; c < 2; ++c)
          c56_gain[c] = frame_info.framestream.c56_camera_gain[c];
      }
      imu_data = frame_info.framestream.imu;
      gnss_data = frame_info.framestream.gnss;
    }

    std::vector<cv::Mat> images;
    images.push_back(left_image);
    if (process_cameras >= 2)
      images.push_back(right_image);
    if (process_cameras >= 4) {
      images.push_back(left_front_image);
      images.push_back(right_front_image);
    }
    InputIMAGE(image_timestamp, hardware_version, software_version,
               camera_gain, c56_gain, fpga_bayer_nr, raw_telemetry, process_cameras,
               images);  // clones into the save queue
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
  {
    std::lock_guard<std::mutex> lock(m_buf);
    capture_finished = true;
  }
  con.notify_one();
  data_flow.join();  // drain queued images/IMU before exiting
  
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
                 const std::array<double, 2> &c56_gain,
                 const std::array<bool, 4> &fpga_bayer_nr,
                 const RawTelemetrySnapshot &raw_telemetry,
                 int process_cameras,
                 const std::vector<cv::Mat>& images) {
    m_buf.lock();
    ImageRecord record{
        timestamp, hardware_version, software_version, camera_gain,
        c56_gain, process_cameras, {}, fpga_bayer_nr, raw_telemetry};
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
  const bool use_fast_balanced = FastBalancedIspEnabled();
  IspProcessor isp(use_fast_balanced ? IspMode::kFastBalancedBgr888
                                   : IspMode::kQualityReferenceBgr888);
  // Six-image firmware (04/07, 05/07): the four half images of lanes 2/3 keep
  // their own AWB/exposure state, separate from the full-width C1/C2.
  IspProcessor half_isp(isp.mode());
  cv::Mat half_raw[4], half_color[4];
  bool half_res_announced = false;
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
        return capture_finished || !imu_data.empty() || !image_data.empty() || !gnss_data.empty();});
      if (capture_finished && imu_data.empty() && image_data.empty() && gnss_data.empty()) break;
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
        // SC136HGS S0/S1/S2 and AR0234: RAW Bayer -> white balance + demosaic, save as
        // colour. S1 processes and writes only C1. MT9V034 remains monochrome
        // and is saved as-is.
        const int process_n = record.process_cameras > 0
                                  ? record.process_cameras
                                  : static_cast<int>(n);
        const bool four_ok =
            process_n >= 4 && n >= 4 && !imgs[0].empty() &&
            !imgs[1].empty() && !imgs[2].empty() && !imgs[3].empty();
        const bool two_ok =
            process_n >= 2 && n >= 2 && !imgs[0].empty() && !imgs[1].empty();
        const bool one_ok = process_n >= 1 && n >= 1 && !imgs[0].empty();
        const bool is_bayer =
            IsSmartSensFamilyHardware(static_cast<int>(record.hardware_version));
        // Six-image: lanes 2/3 are NOT full-width C4/C3 but four horizontally
        // binned images of C3..C6 (C3h|C4h, C5h|C6h).
        const bool half_res_lanes = four_ok &&
            IsSmartSensHalfResLaneLayout(
                static_cast<int>(record.hardware_version),
                static_cast<int>(record.software_version));
        if (half_res_lanes && !half_res_announced) {
          std::cout << "[half-res] " << record.hardware_version << "/"
                    << record.software_version << " six-image "
                    << (IsAr0234FamilyHardware(
                            static_cast<int>(record.hardware_version))
                            ? "AR0234" : "SC136HGS")
                    << ": lanes 2/3 -> C3h/C4h/C5h/C6h "
                    << SmartSensHalfResWidth(imgs[2].cols) << "x"
                    << SmartSensHalfResHeight(imgs[2].rows)
                    << " (saved to ./cam3..6)" << std::endl;
          half_res_announced = true;
        }
        if (is_bayer && one_ok) {
          if (count % 2 == 0) {
            cv::Mat left_image = imgs[0];
            cv::Mat right_image = two_ok ? imgs[1] : cv::Mat{};
            cv::Mat left_front_image = four_ok ? imgs[2] : cv::Mat{};
            cv::Mat right_front_image = four_ok ? imgs[3] : cv::Mat{};
            
            // Run the selected ISP pipeline; fast-balanced is the default.
            const std::string image_name = std::to_string(static_cast<int64_t>(image_timestamp * 10000)) + ".png";
            cv::Mat *raws[4] = {&left_image, &right_image,
                               &left_front_image, &right_front_image};
            cv::Mat *colors[4] = {&left_color, &right_color,
                                 &left_front_color, &right_front_color};
            const int active =
                half_res_lanes ? 2 : (four_ok ? 4 : (two_ok ? 2 : 1));
            ApplyBayerCameraIsp(isp, raws, colors, active,
                record.hardware_version, record.software_version,
                record.camera_gain, record.fpga_bayer_nr, record.raw_telemetry);
            if (half_res_lanes) {
              SplitAndVBinHalfResLanes(imgs[2], imgs[3], half_raw);
              // camera_gain[] is display-plane order C1,C2,C4,C3; C5/C6 come
              // from their own metadata columns.
              const double cam_gain[4] = {record.camera_gain[3], record.camera_gain[2],
                                          record.c56_gain[0], record.c56_gain[1]};
              const uint32_t hw = record.hardware_version;
              const uint32_t sw = record.software_version;
              half_isp.ApplyParallel({
                  {half_raw[0], half_color[0], "half-cam3", cam_gain[0],
                   SelectCameraBayerConversion(hw, sw, SixCamHalfImageCamera(0))},
                  {half_raw[1], half_color[1], "half-cam4", cam_gain[1],
                   SelectCameraBayerConversion(hw, sw, SixCamHalfImageCamera(1))},
                  {half_raw[2], half_color[2], "half-cam5", cam_gain[2],
                   SelectCameraBayerConversion(hw, sw, SixCamHalfImageCamera(2))},
                  {half_raw[3], half_color[3], "half-cam6", cam_gain[3],
                   SelectCameraBayerConversion(hw, sw, SixCamHalfImageCamera(3))},
              });
              MirrorImage5Vertical(half_raw[2], half_color[2]);
            }
            std::thread t2;
            if (two_ok) {
              t2 = std::thread([&] {
                cv::imwrite("./cam2/" + image_name, right_color);
              });
            }
            std::thread th[4];
            if (half_res_lanes) {
              static const char *const kHalfDirs[4] = {
                  "./cam3/", "./cam4/", "./cam5/", "./cam6/"};
              for (int i = 0; i < 4; ++i) {
                th[i] = std::thread([&, i] {
                  cv::imwrite(kHalfDirs[i] + image_name, half_color[i]);
                });
              }
            }
            cv::imwrite("./cam1/" + image_name, left_color);
            if (t2.joinable()) t2.join();
            for (auto &t : th)
              if (t.joinable()) t.join();

          }
          count++;
        } else if (two_ok) {
          if (count % 3 == 0) {
            // MT9V034 is monochrome: save the two raw planes directly.
            const std::string image_name = std::to_string(static_cast<int64_t>(image_timestamp * 10000)) + ".png";
            cv::imwrite("./cam1/" + image_name, imgs[0]);
            cv::imwrite("./cam2/" + image_name, imgs[1]);
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
