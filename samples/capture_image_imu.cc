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
#ifndef _WIN32
#include <execinfo.h>
#endif
#include <iomanip>
#include <iostream>
#include <sstream>
#include "string"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mutex>
#include "../src/usb/uvc/cyperstereo_api.h"
#include "../src/usb/uvc/tic_toc.h"
#include "../src/usb/uvc/thread_priority.h"

const double g = 9.7887;

CYPERSTEREO_USE_NAMESPACE

enum class CaptureColorOutput {
  kAuto,
  kBgr888,
  kUyvy422Bt601FullRange,
};

// Signal handlers may only perform async-signal-safe operations.  Setting a
// sig_atomic_t lets the main loop stop the UVC thread and issue STREAMOFF
// normally instead of having the kernel tear the process down while streaming.
static volatile std::sig_atomic_t g_stop_requested = 0;

extern "C" void stop_handler(int) {
  g_stop_requested = 1;
}

static bool WaitForStreamInterruptible(cyperstereo::FrameInfo &frame_info) {
  std::unique_lock<std::mutex> lock(frame_info.mtx);
  const auto frame_ready = [&frame_info]() {
    return frame_info.frame != nullptr;
  };
  int waited_ms = 0;
  while (!frame_ready() && !g_stop_requested) {
    if (frame_info.con.wait_for(lock, std::chrono::milliseconds(200),
                                frame_ready))
      break;
    waited_ms += 200;
    if (waited_ms % 5000 == 0) {
      std::cout << "[api] WARN no frame for " << waited_ms / 1000
                << " s, capture layer is retrying"
                << "  (check USB link / dmesg if this persists)" << std::endl;
    }
  }
  if (!frame_ready())
    return false;
  frame_info.frame = nullptr;
  return true;
}

class ScopedUvcStream {
 public:
  explicit ScopedUvcStream(cyperstereo::uvc::device &device)
      : device_(&device) {}

  ScopedUvcStream(const ScopedUvcStream &) = delete;
  ScopedUvcStream &operator=(const ScopedUvcStream &) = delete;

  ~ScopedUvcStream() { Stop(); }

  void Start() {
    cyperstereo::uvc::start_streaming(*device_, 0);
    active_ = true;
  }

  void Stop() {
    if (!active_)
      return;
    active_ = false;
    cyperstereo::uvc::stop_streaming(*device_);
  }

 private:
  cyperstereo::uvc::device *device_;
  bool active_{false};
};

// OpenCV 4.2's packed 4:2:2 conversion uses video/limited-range YUV. The
// fast ISP exposes native full-range BT.601 values, so using
// cv::COLOR_YUV2BGR_UYVY would render incorrect black/white levels. Keep this
// conversion out of the capture hot path: it is called only for a sampled
// preview frame when --display is active.
static void Uyvy422Bt601FullRangeToBgrPreview(const cv::Mat &uyvy,
                                              cv::Mat &bgr) {
  CV_Assert(uyvy.type() == CV_8UC2 && (uyvy.cols & 1) == 0);
  bgr.create(uyvy.rows, uyvy.cols, CV_8UC3);
  for (int y = 0; y < uyvy.rows; ++y) {
    const uchar *src = uyvy.ptr<uchar>(y);
    uchar *dst = bgr.ptr<uchar>(y);
    for (int x = 0; x < uyvy.cols; x += 2, src += 4, dst += 6) {
      // UYVY byte order is Cb, Y0, Cr, Y1. These equations intentionally
      // match the fast ISP's full-range YCrCb-to-BGR output kernel.
      const int dcb = static_cast<int>(src[0]) - 128;
      const int dcr = static_cast<int>(src[2]) - 128;
      const int tb = (454 * dcb) >> 8;
      const int tg = (183 * dcr + 88 * dcb) >> 8;
      const int tr = (359 * dcr) >> 8;
      for (int k = 0; k < 2; ++k) {
        const int luma = src[1 + 2 * k];
        dst[3 * k + 0] = cv::saturate_cast<uchar>(luma + tb);
        dst[3 * k + 1] = cv::saturate_cast<uchar>(luma - tg);
        dst[3 * k + 2] = cv::saturate_cast<uchar>(luma + tr);
      }
    }
  }
}

static IspMode ConfigureCaptureIsp(bool use_fast_balanced,
                                   bool use_uyvy422,
                                   bool show_preview,
                                   int capture_log_every,
                                   size_t output_stride,
                                   int num_cameras) {
  const IspMode mode =
      !use_fast_balanced
          ? IspMode::kQualityReferenceBgr888
          : (use_uyvy422
                 ? IspMode::kFastBalancedUyvy422Bt601FullRange
                 : IspMode::kFastBalancedBgr888);

  std::cout << "ISP mode: "
            << (use_fast_balanced ? "fast-balanced" : "quality-reference")
            << std::endl;
  if (use_uyvy422) {
    std::cout << "ISP output: UYVY422 CV_8UC2, bytes=[Cb Y0 Cr Y1], "
                 "BT.601 full-range, chroma=4:2:0-derived/vertically "
                 "duplicated, stride="
              << output_stride << " bytes; output RGB tone deferred"
              << std::endl;
    if (show_preview) {
      std::cout << "Preview: sampled full-range UYVY-to-BGR conversion; "
                << (FastBalancedGammaEnabled() ? "RGB tone applied"
                                               : "RGB tone disabled")
                << std::endl;
    }
  } else {
    std::cout << "ISP output: BGR888 CV_8UC3"
              << (use_fast_balanced ? " (RGB tone follows ISP setting)"
                                    : " (quality-reference/HDR)")
              << std::endl;
  }

  if (capture_log_every == 0)
    std::cout << "Capture metadata log: quiet" << std::endl;
  else
    std::cout << "Capture metadata log: every " << capture_log_every
              << " frame(s)" << std::endl;

  if (use_fast_balanced) {
    std::cout << "Fast ISP: rgb_tone="
              << (use_uyvy422
                      ? "bypassed-for-direct-YUV"
                      : (FastBalancedGammaEnabled() ? "on" : "off"))
              << " hue_guard="
              << (FastBalancedHueGuardEnabled() ? "on" : "off")
              << " bayer_nr="
              << (FastBalancedBayerNrEnabled() ? "adaptive>=3x" : "off")
              << " demosaic=" << FastBalancedDemosaicName()
              << " saturation=" << IspSaturation() << std::endl;
#if defined(CYPERSTEREO_RK3588) && defined(__aarch64__) && defined(__linux__)
    if (num_cameras >= 4) {
      std::cout << "RK3588 capture CPUs: CPU4-7 only "
                   "(four Cortex-A76; A55 disabled)"
                << std::endl;
    }
#else
    (void)num_cameras;
#endif
  }
  return mode;
}

static void LogCaptureDiagnostics(
    int count, int capture_log_every, int num_cameras, bool is_smartsens,
    double image_timestamp, const double exposure_time[4],
    const uint16_t exposure_lines[4], const double camera_gain[4],
    const double camera_temperature[4],
    const cyperstereo::IMUStreamData &imu_data,
    const cyperstereo::GNSSStreamData &gnss_data,
    const cyperstereo::FrameInfo &frame_info, TicToc &frame_timer,
    std::ostringstream &capture_log) {
  const bool log_this_frame =
      capture_log_every > 0 && count % capture_log_every == 0;
  bool log_new_gnss = false;
  if (gnss_data.valid && !gnss_data.gnss_utc_time.empty()) {
    static std::string last_gnss_time;
    if (gnss_data.gnss_utc_time != last_gnss_time) {
      log_new_gnss = capture_log_every > 0;
      last_gnss_time = gnss_data.gnss_utc_time;
    }
  }

  const bool log_status = (count + 1) % 1000 == 0;
  double frame_rate = 0.0;
  if (log_status) {
    frame_rate = 1000 / (frame_timer.toc() / 1000);
    frame_timer.tic();
  }

  const bool have_output =
      log_this_frame || log_new_gnss ||
      (log_status && capture_log_every > 0);
  if (!have_output) return;

  capture_log.str(std::string());
  capture_log.clear();
  if (log_this_frame) {
    capture_log << std::fixed << std::setprecision(6)
                << "[meta] image_ts=" << image_timestamp
                << "  imu_n=" << imu_data.imu_count << '\n';
    for (int i = 0; i < imu_data.imu_count; ++i) {
      capture_log << std::fixed << std::setprecision(4)
                  << "  imu_ts[" << i << "]=" << imu_data.imu_timestamp[i]
                  << "  acc[" << i << "]=(" << imu_data.acc_x[i] * g << ","
                  << imu_data.acc_y[i] * g << ","
                  << imu_data.acc_z[i] * g << ")"
                  << std::setprecision(6)
                  << "  gyro[" << i << "]=(" << imu_data.gyro_x[i] << ","
                  << imu_data.gyro_y[i] << "," << imu_data.gyro_z[i] << ")"
                  << "  T[" << i << "]=" << imu_data.temperature[i] << '\n';
    }

    // FPGA AEC targets are controller state, not sensor-register readback.
    if (is_smartsens) {
      static const char *const kCameraNames[4] = {
          "image1(L,C1)", "image2(R,C2)",
          "image3(LF,C4)", "image4(RF,C3)"};
      capture_log << std::fixed << std::setprecision(6)
                  << "[cam] frame_end_ts=" << image_timestamp << '\n';
      for (int i = 0; i < num_cameras; ++i) {
        capture_log << "  " << kCameraNames[i]
                    << "  aec_exp=" << std::setprecision(3)
                    << exposure_time[i] * 1000.0 << "ms ("
                    << exposure_lines[i] << " lines)"
                    << "  aec_gain=" << std::setprecision(2)
                    << camera_gain[i] << "x"
                    << "  sensor_temp=" << camera_temperature[i] << "C\n";
      }
    }
  }

  if (log_new_gnss) {
    capture_log << std::fixed << std::setprecision(6)
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
                << "  pdop=" << gnss_data.pdop << '\n';
  }
  if (log_status && capture_log_every > 0) {
    capture_log << "frame_rate " << frame_rate
                << "  image_drops=" << frame_info.image_drop_count
                << "  imu_drops=" << frame_info.imu_drop_count << '\n';
  }
  std::cout << capture_log.str() << std::flush;
}

#ifndef _WIN32
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
#endif

static int run(int argc, char *argv[]) {
#if defined(CYPERSTEREO_RK3588) && defined(__aarch64__) && defined(__linux__)
  // Establish the cluster boundary before OpenCV, UVC, or ISP creates any
  // threads. Linux threads inherit this mask, so the entire capture process
  // stays on RK3588's four Cortex-A76 cores and never schedules on CPU0..3.
  const int affinity_rc = RestrictCurrentThreadToCpuRange(4, 7);
  if (affinity_rc != 0) {
    std::cerr << "[affinity] cannot restrict capture to RK3588 CPU4-7 ("
              << std::strerror(affinity_rc) << "). Aborting instead of using "
                 "Cortex-A55 cores."
              << std::endl;
    return 1;
  }

  // The four-camera scheduler already assigns one camera lane to each A76.
  // Force off both LITTLE-core assistance and nested per-frame sharding.
  if (::setenv("CYPERSTEREO_FAST_BIG_LITTLE", "0", 1) != 0 ||
      ::setenv("CYPERSTEREO_FAST_INTRA_FRAME", "0", 1) != 0) {
    std::cerr << "[affinity] cannot disable auxiliary ISP workers ("
              << std::strerror(errno) << "). Aborting instead of using "
                 "Cortex-A55 cores."
              << std::endl;
    return 1;
  }
#endif
  cv::setNumThreads(1);

  // --no-display: skip all imshow/GUI work. The preview path (X11/GTK) can
  // block the consumer loop for tens of ms (much worse over SSH X-forwarding);
  // when the pipeline falls behind, the camera's FX3 firmware overruns its
  // internal FIFO and the stream stalls ("v4l2 get stream time out").
#if defined(CYPERSTEREO_RK3588) && defined(__aarch64__) && defined(__linux__)
  // Keep the production RK3588 capture loop headless by default. Rendering
  // can block the consumer long enough to overrun the FX3 FIFO; --display is
  // still available for an explicitly requested local preview.
  bool show_preview = false;
#else
  bool show_preview = true;
#endif
  bool use_fast_balanced = FastBalancedIspEnabled();
  CaptureColorOutput output_request = CaptureColorOutput::kAuto;
#if defined(CYPERSTEREO_RK3588) && defined(__aarch64__) && defined(__linux__)
  constexpr int kDefaultCaptureLogEvery = 30;
#else
  constexpr int kDefaultCaptureLogEvery = 1;
#endif
  int capture_log_every = kDefaultCaptureLogEvery;
  if (const char *value = std::getenv("CYPERSTEREO_CAPTURE_LOG_EVERY")) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && *end == '\0' && parsed >= 0 && parsed <= 1000000) {
      capture_log_every = static_cast<int>(parsed);
    } else {
      std::cerr << "[log] invalid CYPERSTEREO_CAPTURE_LOG_EVERY='"
                << value << "', using " << kDefaultCaptureLogEvery
                << std::endl;
    }
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--no-display")
      show_preview = false;
    else if (arg == "--display")
      show_preview = true;
    else if (arg == "--verbose")
      capture_log_every = 1;
    else if (arg == "--quiet")
      capture_log_every = 0;
    else if (arg == "--isp-fast" || arg == "--isp-fast-balanced")
      use_fast_balanced = true;
    else if (arg == "--isp-quality")
      use_fast_balanced = false;
    else if (arg == "--output-yuv422" || arg == "--output-uyvy" ||
             arg == "--output-uyvy422")
      output_request = CaptureColorOutput::kUyvy422Bt601FullRange;
    else if (arg == "--output-bgr" || arg == "--output-bgr888")
      output_request = CaptureColorOutput::kBgr888;
  }
  // Fast-balanced capture uses the processed full-range UYVY422 contract on
  // every platform by default.  Keep --output-bgr as an explicit compatibility
  // override; the quality-reference/HDR path remains BGR-only below.
  constexpr bool kAutoUsesUyvy422 = true;
  // The quality-reference/HDR pipeline intentionally retains its existing
  // BGR888 contract. Never pass a CV_8UC2 destination to that API.
  const bool requested_uyvy422 =
      output_request == CaptureColorOutput::kUyvy422Bt601FullRange ||
      (output_request == CaptureColorOutput::kAuto && kAutoUsesUyvy422);
  const bool use_uyvy422 = use_fast_balanced && requested_uyvy422;
  if (!use_fast_balanced &&
      output_request == CaptureColorOutput::kUyvy422Bt601FullRange) {
    std::cerr << "[output] --output-yuv422 is available only with the "
                 "fast-balanced ISP; quality-reference keeps BGR888"
              << std::endl;
  }
#ifndef _WIN32
  // X11-only check; Windows has no DISPLAY variable and HighGUI works natively.
  if (show_preview && std::getenv("DISPLAY") == nullptr) {
    std::cout << "[gui] DISPLAY not set, disabling preview (use X forwarding "
                 "or a local session to enable)" << std::endl;
    show_preview = false;
  }
#endif
  
  // camera config init
  std::shared_ptr<cyperstereo::uvc::device> cyperstereo_device{nullptr};
  if (!cyperstereo::FindCyperstereoDevices(cyperstereo_device)) {
    return 0;
  }
  cyperstereo::FrameInfo frame_info{};
  // Declared after frame_info so its destructor stops the callback thread
  // before frame_info is destroyed on every return/exception path.
  ScopedUvcStream stream(*cyperstereo_device);
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
  cyperstereo::uvc::set_device_mode(
    *cyperstereo_device, profile.frame_width, profile.frame_height,
    static_cast<int>(cyperstereo::Format::YUYV), profile.fps,
    [&frame_info](const void *data, std::function<void()> continuation) {
      cyperstereo::SetStreamData(frame_info, data, continuation);
    });
  stream.Start();
  
  
  TicToc t_frame;
  int count = 0;

  const bool is_smartsens = cyperstereo::IsSmartSensProfile(profile);

  cv::Mat left_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat right_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat left_front_image(profile.frame_height, profile.cam_width, CV_8U);
  cv::Mat right_front_image(profile.frame_height, profile.cam_width, CV_8U);

  const int color_type = use_uyvy422 ? CV_8UC2 : CV_8UC3;
  cv::Mat left_color(profile.frame_height, profile.cam_width, color_type);
  cv::Mat right_color(profile.frame_height, profile.cam_width, color_type);
  cv::Mat left_front_color(profile.frame_height, profile.cam_width, color_type);
  cv::Mat right_front_color(profile.frame_height, profile.cam_width,
                            color_type);
  // Empty in headless operation. Full-range UYVY is converted only for the
  // sampled frames that are actually sent to HighGUI.
  cv::Mat preview_color[4];
  const IspMode isp_mode = ConfigureCaptureIsp(
      use_fast_balanced, use_uyvy422, show_preview, capture_log_every,
      left_color.step[0], process_cameras);
  IspProcessor isp(isp_mode);

  //imshow windows init
  constexpr int kShowEvery = 1;
  static const char *const kWindowNames[4] = {
      "image1", "image2", "image3", "image4"};
  if (show_preview) {
    for (int i = 0; i < process_cameras; ++i) {
      // WINDOW_AUTOSIZE keeps the preview at the Mat's native resolution.
      // Do not resize the window: scaling can hide fine noise and false color.
      cv::namedWindow(kWindowNames[i], cv::WINDOW_AUTOSIZE);
    }
    cv::startWindowThread();
  }
  bool preview_visible_seen[4]{};
  bool preview_autosize_seen[4]{};
  const auto preview_window_closed = [&]() {
    for (int i = 0; i < process_cameras; ++i) {
      // OpenCV 4.2's GTK backend does not implement WND_PROP_VISIBLE and
      // returns -1 even while a window is displayed. AUTOSIZE is supported:
      // it is 0/1 for an existing window and -1 after the user closes it. A
      // backend may transiently report an unsupported/hidden value while its
      // first window is being mapped, so only treat a state transition as a
      // close after that same property was observed valid for this window.
      const double visible =
          cv::getWindowProperty(kWindowNames[i], cv::WND_PROP_VISIBLE);
      if (visible > 0.0)
        preview_visible_seen[i] = true;
      else if (visible == 0.0 && preview_visible_seen[i])
        return true;

      const double autosize =
          cv::getWindowProperty(kWindowNames[i], cv::WND_PROP_AUTOSIZE);
      if (autosize >= 0.0)
        preview_autosize_seen[i] = true;
      else if (preview_autosize_seen[i])
        return true;
    }
    return false;
  };

  // Reuse one formatting buffer; constructing iostream/locale state on every
  // frame would replace stdout overhead with allocator/locale overhead.
  std::ostringstream capture_log;
  
  while (!g_stop_requested) {
    if (!WaitForStreamInterruptible(frame_info))
      break;

    int preview_key = -1;

    double image_timestamp = 0.0;
    double exposure_time[4]{};
    uint16_t exposure_lines[4]{};
    double camera_gain[4]{};
    double camera_temperature[4]{};
    uint32_t hardware_version = 0;
    uint32_t software_version = 0;
    cyperstereo::IMUStreamData imu_data{};
    cyperstereo::GNSSStreamData gnss_data{};

    {
      std::lock_guard<std::mutex> lock(frame_info.mtx);
      image_timestamp = frame_info.framestream.image_timestamp;
      hardware_version = frame_info.framestream.hardware_version;
      software_version = frame_info.framestream.software_version;
      cv::swap(frame_info.framestream.left_image, left_image);
      if (process_cameras >= 2)
        cv::swap(frame_info.framestream.right_image, right_image);
      if (process_cameras >= 4) {
        cv::swap(frame_info.framestream.left_front_image, left_front_image);
        cv::swap(frame_info.framestream.right_front_image, right_front_image);
      }
      if (is_smartsens) {
        for (int i = 0; i < process_cameras; ++i) {
          exposure_time[i] = frame_info.framestream.exposure_time[i];
          exposure_lines[i] = frame_info.framestream.exposure_lines[i];
          camera_gain[i] = frame_info.framestream.camera_gain[i];
          camera_temperature[i] =
              frame_info.framestream.camera_temperature[i];
        }
      }
      imu_data = frame_info.framestream.imu;
      gnss_data = frame_info.framestream.gnss;
    }

    // Check before imshow as well as after waitKey. Otherwise HighGUI can
    // recreate a just-closed window when the next sampled frame is shown,
    // hiding the user's close request.
    if (show_preview && count % kShowEvery == 0 &&
        preview_window_closed()) {
      std::cout << "[capture] preview closed; stopping stream" << std::endl;
      break;
    }

    if (is_smartsens)
    {
      TicToc proc;
      // Run the selected ISP pipeline. Fast-balanced is the default; the
      // HDR-ISP quality reference remains available through --isp-quality.
      // Each camera uses its own live AEC gain from the metadata row.
      const BayerConversion image13_bayer =
          SelectBayerConversion(hardware_version, software_version, 0);
      if (process_cameras <= 1) {
        // S1: C1-only color SKU. Unused FX3 lanes are not ISP'd.
        isp.ApplyParallel({
            {left_image, left_color, "cam1", camera_gain[0], image13_bayer},
        });
      } else if (process_cameras >= 4) {
        isp.ApplyParallel({
            {left_image, left_color, "cam1", camera_gain[0], image13_bayer},
            {right_image, right_color, "cam2", camera_gain[1]},
            {left_front_image, left_front_color, "cam3", camera_gain[2],
             image13_bayer},
            {right_front_image, right_front_color, "cam4", camera_gain[3]},
        });
      } else {
        // S2 exposes only the C1/C2 Bayer planes.
        isp.ApplyParallel({
            {left_image, left_color, "cam1", camera_gain[0], image13_bayer},
            {right_image, right_color, "cam2", camera_gain[1]},
        });
      }
      //std::cout << "proc(wb+cvt) " << proc.toc() << std::endl;
      
      if (show_preview && count % kShowEvery == 0) {
        const cv::Mat *display_images[4] = {
            &left_color, &right_color, &left_front_color, &right_front_color};
        if (use_uyvy422) {
          for (int i = 0; i < process_cameras; ++i) {
            Uyvy422Bt601FullRangeToBgrPreview(*display_images[i],
                                              preview_color[i]);
            if (FastBalancedGammaEnabled())
              ApplyFastBalancedGamma(preview_color[i]);
            display_images[i] = &preview_color[i];
          }
        }
        for (int i = 0; i < process_cameras; ++i)
          cv::imshow(kWindowNames[i], *display_images[i]);
        preview_key = cv::waitKey(1);
      }
    }
    else
    {
      // MT9V034 is monochrome (no Bayer): display the two raw planes directly.
      if (show_preview && count % kShowEvery == 0) {
        cv::imshow("image1", left_image);
        cv::imshow("image2", right_image);
        preview_key = cv::waitKey(1);
      }
    }

    if (show_preview && count % kShowEvery == 0) {
      if (preview_key == 27 || preview_key == 'q' || preview_key == 'Q' ||
          preview_window_closed()) {
        std::cout << "[capture] preview closed; stopping stream" << std::endl;
        break;
      }
    }

    LogCaptureDiagnostics(
        count, capture_log_every, process_cameras, is_smartsens, image_timestamp,
        exposure_time, exposure_lines, camera_gain, camera_temperature,
        imu_data, gnss_data, frame_info, t_frame, capture_log);
    ++count;
  }
  stream.Stop();
  if (show_preview)
    cv::destroyAllWindows();

  return 0;
}

int main(int argc, char *argv[]) {
  std::signal(SIGINT, stop_handler);
  std::signal(SIGTERM, stop_handler);
#ifndef _WIN32
  std::signal(SIGSEGV, crash_handler);
  std::signal(SIGABRT, crash_handler);
  std::signal(SIGBUS, crash_handler);
#endif
  // The UVC backends report unrecoverable setup errors (no matching media
  // type, COM/V4L2 failures) as std::runtime_error.  Without this handler the
  // program dies with no message ("flash crash"); print the reason instead.
  try {
    return run(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "[fatal] " << e.what() << std::endl;
    return 1;
  }
}
