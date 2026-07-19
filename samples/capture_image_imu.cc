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
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <iomanip>
#include <iostream>
#include "string"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mutex>
#include "../src/usb/uvc/cyperstereo_api.h"
#include "../src/usb/uvc/tic_toc.h"
#include "../src/usb/uvc/thread_priority.h"

const double g = 9.7887;

CYPERSTEREO_USE_NAMESPACE

// Crash triage without gdb on the target: dump raw return addresses from the
// faulting thread, resolvable offline with addr2line against this binary.
extern "C" void crash_handler(int sig) {
  void *frames[32];
  const int n = backtrace(frames, 32);
  fprintf(stderr, "\n[crash] signal %d, backtrace (%d frames):\n", sig, n);
  backtrace_symbols_fd(frames, n, 2);
  signal(sig, SIG_DFL);
  raise(sig);
}

int main(int argc, char *argv[]) {
  signal(SIGSEGV, crash_handler);
  signal(SIGABRT, crash_handler);
  signal(SIGBUS, crash_handler);
  cv::setNumThreads(1);

  // --no-display: skip all imshow/GUI work. The preview path (X11/GTK) can
  // block the consumer loop for tens of ms (much worse over SSH X-forwarding);
  // when the pipeline falls behind, the camera's FX3 firmware overruns its
  // internal FIFO and the stream stalls ("v4l2 get stream time out").
  bool show_preview = true;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--no-display")
      show_preview = false;
  }
  if (show_preview && std::getenv("DISPLAY") == nullptr) {
    std::cout << "[gui] DISPLAY not set, disabling preview (use X forwarding "
                 "or a local session to enable)" << std::endl;
    show_preview = false;
  }
  
  // camera config init
  std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
  if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device)) {
    return 0;
  }
  cyperstereo::FrameInfo frame_info{};
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
  cyperstereo::uvc::set_device_mode(
    *cyperstereo_device, profile.frame_width, profile.frame_height,
    static_cast<int>(cyperstereo::Format::YUYV), profile.fps,
    [&frame_info](const void *data, std::function<void()> continuation) {
      cyperstereo::SetStreamData(frame_info, data, continuation);
    });
  cyperstereo::uvc::start_streaming(*cyperstereo_device, 0);
  
  
  TicToc t_frame;
  int count = 0;

  const int num_cameras = profile.num_cameras;

  cv::Mat left_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat right_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat left_front_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat right_front_image(profile.frame_height, profile.cam_width, CV_8U);

  cv::Mat left_color(profile.frame_height, profile.cam_width, CV_8UC3);
  cv::Mat right_color(profile.frame_height, profile.cam_width, CV_8UC3);
  cv::Mat left_front_color(profile.frame_height, profile.cam_width, CV_8UC3);
  cv::Mat right_front_color(profile.frame_height, profile.cam_width, CV_8UC3);

  //imshow windows init
  constexpr int kShowEvery = 5;
  if (show_preview) {
    const int win_w = profile.cam_width  / 2.0;
    const int win_h = profile.frame_height / 2.0;
    const char *wins[4] = {"image1", "image2", "image3", "image4"};
    for (int i = 0; i < num_cameras; ++i) {
      cv::namedWindow(wins[i], cv::WINDOW_NORMAL);
      cv::resizeWindow(wins[i], win_w, win_h);
    }
    cv::startWindowThread();
  }
  
  while (true) {
    cyperstereo::WaitForStream(frame_info);

    double image_timestamp = 0.0;
    cyperstereo::IMUStreamData imu_data{};
    cyperstereo::GNSSStreamData gnss_data{};

    {
      std::lock_guard<std::mutex> lock(frame_info.mtx);
      image_timestamp = frame_info.framestream.image_timestamp;
      cv::swap(frame_info.framestream.left_image, left_image);
      cv::swap(frame_info.framestream.right_image, right_image);
      if (num_cameras >= 4) {
        cv::swap(frame_info.framestream.left_front_image, left_front_image);
        cv::swap(frame_info.framestream.right_front_image, right_front_image);
      }
      imu_data = frame_info.framestream.imu;
      gnss_data = frame_info.framestream.gnss;
    }

    if (num_cameras >= 4)
    {
      TicToc proc;
      static WhiteBalance wb1, wb2, wb3, wb4;
      // jobs[0] on this thread; rest on persistent IspWorkers (see ApplyISPParallel).
      ApplyISPParallel({
          {left_image, left_color, wb1, "wb-cam1"},
          {right_image, right_color, wb2, "wb-cam2"},
          {left_front_image, left_front_color, wb3, "wb-cam3"},
          {right_front_image, right_front_color, wb4, "wb-cam4"},
      });
      //std::cout << "proc(wb+cvt) " << proc.toc() << std::endl;
      
     if (show_preview && count % kShowEvery == 0) {
        cv::imshow("image1", left_color);
        cv::imshow("image2", right_color);
        cv::imshow("image3", left_front_color);
        cv::imshow("image4", right_front_color);
        cv::waitKey(1);
      }
    }
    else
    {
      // MT9V034 is monochrome (no Bayer): display the two raw planes directly.
      if (show_preview && count % kShowEvery == 0) {
        cv::imshow("image1", left_image);
        cv::imshow("image2", right_image);
        cv::waitKey(1);
      }
    }

    // Image timestamp + IMU samples, printed every frame (no count%2 gate).
    /* std::cout << std::fixed << std::setprecision(6)
              << "[meta] image_ts=" << image_timestamp
              << "  imu_n=" << imu_data.imu_count << std::endl;
    if (imu_data.imu_count > 0) {
      for (int i = 0; i < imu_data.imu_count; ++i) {
        std::cout << std::fixed << std::setprecision(4)
                  << "  imu_ts[" << i << "]=" << imu_data.imu_timestamp[i]
                  << "  acc[" << i << "]=(" << imu_data.acc_x[i] * g << ","
                  << imu_data.acc_y[i] * g << "," << imu_data.acc_z[i] * g << ")"
                  << std::setprecision(6)
                  << "  gyro[" << i << "]=(" << imu_data.gyro_x[i] << ","
                  << imu_data.gyro_y[i] << "," << imu_data.gyro_z[i] << ")"
                  << "  T[" << i << "]=" << imu_data.temperature[i] << std::endl;
      }
    }*/

    // GNSS: print valid, non-duplicate records, every frame.
    if (gnss_data.valid && !gnss_data.gnss_utc_time.empty()) {
      static std::string last_gnss_time;
      if (gnss_data.gnss_utc_time != last_gnss_time) {
        std::cout << std::fixed << std::setprecision(6)
                  << "[gnss] ts=" << gnss_data.gnss_timestamp
                  << "  utc=" << gnss_data.gnss_utc_time
                  << "  lat=" << gnss_data.latitude
                  << "  lon=" << gnss_data.longitude
                  << "  alt=" << gnss_data.altitude
                  << "  fix=" << gnss_data.fix_type
                  << "  sat=" << gnss_data.satellites_used
                  << "  vel=" << gnss_data.velocity
                  << "  heading=" << gnss_data.heading
                  << "  hdop=" << gnss_data.hdop
                  << "  vdop=" << gnss_data.vdop
                  << "  pdop=" << gnss_data.pdop << std::endl;
        last_gnss_time = gnss_data.gnss_utc_time;
      }
    }
    ++count;
    if (count % 1000 == 0) {
    	double frame_rate = 1000 / (t_frame.toc() / 1000);
    	t_frame.tic();
    	std::cout << "frame_rate " << frame_rate
                << "  image_drops=" << frame_info.image_drop_count
                << "  imu_drops=" << frame_info.imu_drop_count << std::endl;
    }
  }
  cyperstereo::uvc::stop_streaming(*cyperstereo_device);
  
  return 0;
}
