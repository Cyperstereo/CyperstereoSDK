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
#include "../src/usb/uvc/six_camera_isp.h"
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
                 "BT.601 full-range, post-tone pair-averaged chroma, stride="
              << output_stride << " bytes; exposure/tone matched to BGR"
              << std::endl;
    if (show_preview) {
      std::cout << "Preview: sampled full-range UYVY-to-BGR conversion; "
                   "tone already handled by ISP" << std::endl;
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
              << (FastBalancedGammaEnabled()
                      ? (FastToneFilmic() ? "filmic" : "gamma") : "off")
              << " hue_guard="
              << (FastBalancedHueGuardEnabled() ? "on" : "off")
              << " bayer_nr="
              << (FastBalancedBayerNrEnabled() ? "adaptive>=3x" : "off")
              << " demosaic=" << FastBalancedDemosaicName()
              << " saturation=" << IspSaturation()
              << " ar0234_tune=on-detect"
              << " ar0234_lsc=" << (FastAr0234LscEnabled() ? "on" : "off")
              << std::endl;
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

// Six-camera C5/C6 AE telemetry copied out of the framestream under its lock.
struct C56Telemetry {
  bool valid{false};
  uint16_t exposure_lines[2]{};
  double exposure_time[2]{};
  double camera_gain[2]{};
  double digital_gain[2]{1.0, 1.0};
  double camera_temperature[2]{};
};

static void LogCaptureDiagnostics(
    int count, int capture_log_every, int num_cameras, bool is_smartsens,
    double image_timestamp, const double exposure_time[4],
    const uint16_t exposure_lines[4], const double camera_gain[4],
    const double digital_gain[4], const double camera_temperature[4],
    const C56Telemetry &c56,
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
      // image3(LF,C4) is the longest label. Numeric fields are right-aligned
      // so the decimal points stay in the same column when widths differ.
      const auto log_camera = [&](const char *name, double exposure_s,
                                  int lines, double aec, double digital,
                                  double temp) {
        capture_log << "  " << std::left << std::setw(13) << name
                    << std::right << std::setprecision(3)
                    << "  aec_exp=" << std::setw(7) << exposure_s * 1000.0
                    << "ms (" << std::setw(4) << lines << " lines)"
                    << std::setprecision(2)
                    << "  aec_gain=" << std::setw(6) << aec << "x"
                    << "  digital_gain=" << std::setw(6) << digital << "x"
                    << "  sensor_temp=";
        if (std::isfinite(temp))
          capture_log << std::setw(6) << temp << "C\n";
        else capture_log << "unavailable\n";
      };
      const auto log_flags = capture_log.flags();
      capture_log << std::fixed << std::setprecision(6)
                  << "[cam] frame_end_ts=" << image_timestamp << '\n';
      for (int i = 0; i < num_cameras; ++i) {
        log_camera(kCameraNames[i], exposure_time[i], exposure_lines[i],
                   camera_gain[i], digital_gain[i], camera_temperature[i]);
      }
      static const char *const kC56Names[2] = {"image5(C5)", "image6(C6)"};
      for (int c = 0; c < 2 && c56.valid; ++c) {
        log_camera(kC56Names[c], c56.exposure_time[c], c56.exposure_lines[c],
                   c56.camera_gain[c], c56.digital_gain[c],
                   c56.camera_temperature[c]);
      }
      capture_log.flags(log_flags);
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

// RAW8 plane statistics before the ISP (black level and display exposure not
// applied), so sensors can be compared independently of the display path.
static void LogRawStats(int count, int every, const cv::Mat *const raw[4],
                        int num_cameras) {
  if (every <= 0 || count % every != 0) return;
  std::ostringstream out;
  out << "[raw] frame=" << count;
  for (int c = 0; c < num_cameras; ++c) {
    const cv::Mat &m = *raw[c];
    if (m.empty() || m.type() != CV_8U) continue;
    uint32_t hist[256] = {};
    uint64_t sum = 0;
    uint32_t n = 0;
    for (int y = 0; y < m.rows; y += 4) {
      const uchar *p = m.ptr<uchar>(y);
      for (int x = 0; x < m.cols; x += 4) {
        ++hist[p[x]];
        sum += p[x];
        ++n;
      }
    }
    if (n == 0) continue;
    const auto pct = [&](double q) {
      const uint32_t target = static_cast<uint32_t>(n * q);
      uint32_t acc = 0;
      for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc > target) return v;
      }
      return 255;
    };
    uint32_t sat = 0;
    for (int v = 250; v < 256; ++v) sat += hist[v];
    out << "  image" << c + 1 << " mean=" << std::fixed
        << std::setprecision(1) << static_cast<double>(sum) / n
        << " p10=" << pct(0.10) << " p50=" << pct(0.50)
        << " p90=" << pct(0.90) << " p99=" << pct(0.99)
        << " sat=" << std::setprecision(2) << 100.0 * sat / n << "%";
  }
  out << "  isp_blc=" << std::setprecision(3) << cyperstereo::IspBlackLevel();
  std::cout << out.str() << std::endl;
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
  // --show-raw: preview the RAW8 Bayer planes instead of the ISP output.
  // Key 's' saves raw_imageN_<frame>.png (lossless RAW8) and the matching
  // color_imageN_<frame>.png to the working directory in either mode.
  // --save-at N does the same automatically at frames N and N+1.
  bool show_raw = false;
  long save_at = -1;
  long max_frames = -1;
  long save_every = 0;
  int raw_stats_every = 0;  // RAW statistics are opt-in diagnostics.
  if (const char *value = std::getenv("CYPERSTEREO_RAW_STATS_EVERY")) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && *end == '\0' && parsed >= 0 && parsed <= 1000000)
      raw_stats_every = static_cast<int>(parsed);
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--no-display")
      show_preview = false;
    else if (arg == "--show-raw")
      show_raw = true;
    else if (arg == "--save-at" && i + 1 < argc)
      save_at = std::strtol(argv[++i], nullptr, 10);
    else if (arg == "--frames" && i + 1 < argc)
      max_frames = std::strtol(argv[++i], nullptr, 10);
    else if (arg == "--save-every" && i + 1 < argc)
      save_every = std::strtol(argv[++i], nullptr, 10);
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
  // Fast-balanced defaults to BGR888. --output-yuv422 selects full-range
  // UYVY422 with the same digital exposure and tone. Quality/HDR is BGR-only.
  constexpr bool kAutoUsesUyvy422 = false;
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
  cv::Mat preview_color[6];
  const IspMode isp_mode = ConfigureCaptureIsp(
      use_fast_balanced, use_uyvy422, show_preview, capture_log_every,
      left_color.step[0], process_cameras);
  IspProcessor isp(isp_mode);
  // Six-image firmware (04/07, 05/07) packs four half-width binned cameras
  // into image3/image4.  A second processor keeps those halves on their own
  // AWB/exposure state so C1/C2's full-width state is not overwritten.
  IspProcessor half_isp(isp_mode);
  std::unique_ptr<SixCameraIspProcessor> six_isp;
  const bool use_six_scheduler = use_fast_balanced &&
      !FastEnvTruthy("CYPERSTEREO_DISABLE_SIX_CAMERA_ISP");
  cv::Mat half_raw[4];
  cv::Mat half_color[4];
  for (int i = 0; i < 4; ++i) {
    half_color[i].create(SmartSensHalfResHeight(profile.frame_height),
                         SmartSensHalfResWidth(profile.cam_width), color_type);
  }
  bool half_res_announced = false;

  //imshow windows init
  constexpr int kShowEvery = 1;
  static const char *const kQuadWindowNames[4] = {
      "image1", "image2", "image3", "image4"};
  static const char *const kHalfWindowNames[6] = {
      "image1", "image2", "image3", "image4", "image5", "image6"};
  int n_windows = process_cameras;
  const char *const *window_names = kQuadWindowNames;
  if (show_preview) {
    for (int i = 0; i < n_windows; ++i) {
      // WINDOW_AUTOSIZE keeps the preview at the Mat's native resolution.
      // Do not resize the window: scaling can hide fine noise and false color.
      cv::namedWindow(window_names[i], cv::WINDOW_AUTOSIZE);
    }
    cv::startWindowThread();
  }
  bool preview_visible_seen[6]{};
  bool preview_autosize_seen[6]{};
  const auto preview_window_closed = [&]() {
    for (int i = 0; i < n_windows; ++i) {
      // OpenCV 4.2's GTK backend does not implement WND_PROP_VISIBLE and
      // returns -1 even while a window is displayed. AUTOSIZE is supported:
      // it is 0/1 for an existing window and -1 after the user closes it. A
      // backend may transiently report an unsupported/hidden value while its
      // first window is being mapped, so only treat a state transition as a
      // close after that same property was observed valid for this window.
      const double visible =
          cv::getWindowProperty(window_names[i], cv::WND_PROP_VISIBLE);
      if (visible > 0.0)
        preview_visible_seen[i] = true;
      else if (visible == 0.0 && preview_visible_seen[i])
        return true;

      const double autosize =
          cv::getWindowProperty(window_names[i], cv::WND_PROP_AUTOSIZE);
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
    double isp_camera_gain[4]{};
    bool fpga_bayer_nr[4]{};
    double camera_temperature[4]{};
    C56Telemetry c56;
    RawTelemetrySnapshot raw_telemetry;
    uint32_t hardware_version = 0;
    uint32_t software_version = 0;
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
      if (is_smartsens) {
        for (int i = 0; i < process_cameras; ++i) {
          exposure_time[i] = frame_info.framestream.exposure_time[i];
          exposure_lines[i] = frame_info.framestream.exposure_lines[i];
          camera_gain[i] = frame_info.framestream.camera_gain[i];
          isp_camera_gain[i] = frame_info.framestream.isp_camera_gain[i];
          fpga_bayer_nr[i] = frame_info.framestream.fpga_bayer_nr[i];
          camera_temperature[i] =
              frame_info.framestream.camera_temperature[i];
        }
        const auto &fs = frame_info.framestream;
        c56.valid = fs.c56_telemetry_valid;
        for (int c = 0; c < 2; ++c) {
          c56.exposure_lines[c] = fs.c56_exposure_lines[c];
          c56.exposure_time[c] = fs.c56_exposure_time[c];
          c56.camera_gain[c] = fs.c56_camera_gain[c];
          c56.digital_gain[c] = 1.0;
          c56.camera_temperature[c] = fs.c56_camera_temperature[c];
        }
      }
      imu_data = frame_info.framestream.imu;
      gnss_data = frame_info.framestream.gnss;
    }
    if (is_smartsens) {
      const cv::Mat *raw_planes[4] = {
          &left_image, &right_image, &left_front_image, &right_front_image};
      LogRawStats(count, raw_stats_every, raw_planes, process_cameras);
    }

    // Check before imshow as well as after waitKey. Otherwise HighGUI can
    // recreate a just-closed window when the next sampled frame is shown,
    // hiding the user's close request.
    if (show_preview && count % kShowEvery == 0 &&
        preview_window_closed()) {
      std::cout << "[capture] preview closed; stopping stream" << std::endl;
      break;
    }

    bool six_this_frame = false;
    if (is_smartsens)
    {
      TicToc proc;
      // Run the selected ISP pipeline. Fast-balanced is the default; the
      // HDR-ISP quality reference remains available through --isp-quality.
      // Each camera uses its own live AEC gain from the metadata row.
      // image1/image2 use the delivered phase: AR0234 rotates those two RAW
      // planes 180 degrees. image3/image4 stay in transport orientation.
      const BayerConversion image1_bayer =
          SelectDeliveredPlaneBayer(hardware_version, software_version, 0);
      const BayerConversion image2_bayer =
          SelectDeliveredPlaneBayer(hardware_version, software_version, 1);
      const BayerConversion image3_bayer =
          SelectBayerConversion(hardware_version, software_version, 2);
      const BayerConversion image4_bayer =
          SelectBayerConversion(hardware_version, software_version, 3);
      const bool half_res_lanes =
          process_cameras >= 4 &&
          IsSmartSensHalfResLaneLayout(static_cast<int>(hardware_version),
                                       static_cast<int>(software_version));
      if (process_cameras <= 1) {
        // S1: C1-only color SKU. Unused FX3 lanes are not ISP'd.
        isp.ApplyParallel({
            {left_image, left_color, "cam1", isp_camera_gain[0], image1_bayer,
             fpga_bayer_nr[0], raw_telemetry.RegionForPlane(0)},
        });
      } else if (half_res_lanes) {
        // image3/image4 are concat half-width RAW pairs of C3|C4 and C5|C6.
        // Split them into four cameras, then vertically 2:1 same-colour bin in
        // RAW (the FPGA already did horizontal 2:1) so the ISP runs on
        // quarter-size images.
        if (!half_res_announced) {
          std::cout << "[half-res] " << hardware_version << "/"
                    << software_version << " six-image "
                    << (IsAr0234FamilyHardware(static_cast<int>(hardware_version))
                            ? "AR0234" : "SC136HGS")
                    << ": split image3/image4 -> C3h/C4h/C5h/C6h "
                    << half_color[0].cols << "x" << half_color[0].rows
                    << " (RAW vbin " << left_front_image.rows << "->"
                    << half_color[0].rows << ")" << std::endl;
          half_res_announced = true;
          if (show_preview && n_windows != 6) {
            cv::destroyWindow("image3");
            cv::destroyWindow("image4");
            window_names = kHalfWindowNames;
            n_windows = 6;
            for (int i = 2; i < 6; ++i) {
              cv::namedWindow(window_names[i], cv::WINDOW_AUTOSIZE);
              preview_visible_seen[i] = false;
              preview_autosize_seen[i] = false;
            }
          }
        }
        six_this_frame = use_six_scheduler &&
            IsAr0234FamilyHardware(static_cast<int>(hardware_version));
        if (!six_this_frame) {
          isp.ApplyParallel({
              {left_image, left_color, "cam1", isp_camera_gain[0], image1_bayer,
               fpga_bayer_nr[0], raw_telemetry.RegionForPlane(0)},
              {right_image, right_color, "cam2", isp_camera_gain[1],
               image2_bayer, fpga_bayer_nr[1], raw_telemetry.RegionForPlane(1)},
          });
        }
        SplitAndVBinHalfResLanes(left_front_image, right_front_image, half_raw);
        BayerConversion cam_bayer[4];
        for (int h = 0; h < 4; ++h)
          cam_bayer[h] = SelectCameraBayerConversion(
              hardware_version, software_version, SixCamHalfImageCamera(h));
        // isp_camera_gain[] is display-plane order C1,C2,C4,C3; C5/C6 come
        // from their own metadata columns.
        const double cam_gain[4] = {isp_camera_gain[3], isp_camera_gain[2],
                                    c56.camera_gain[0], c56.camera_gain[1]};
        // The FPGA leaves the half lanes unfiltered (nr_active = 0), and they
        // carry no metadata row, so the host NR runs and no region is masked.
        if (six_this_frame) {
          if (!six_isp) six_isp.reset(new SixCameraIspProcessor(
              use_uyvy422 ? IspPixelFormat::kUyvy422Bt601FullRange
                          : IspPixelFormat::kBgr888));
          const IspFrameJob jobs[] = {
              {left_image, left_color, "cam1", isp_camera_gain[0], image1_bayer,
               fpga_bayer_nr[0], raw_telemetry.RegionForPlane(0)},
              {right_image, right_color, "cam2", isp_camera_gain[1], image2_bayer,
               fpga_bayer_nr[1], raw_telemetry.RegionForPlane(1)},
              {half_raw[0], half_color[0], "half-cam3", cam_gain[0], cam_bayer[0]},
              {half_raw[1], half_color[1], "half-cam4", cam_gain[1], cam_bayer[1]},
              {half_raw[2], half_color[2], "half-cam5", cam_gain[2], cam_bayer[2]},
              {half_raw[3], half_color[3], "half-cam6", cam_gain[3], cam_bayer[3]}};
          six_isp->Apply(jobs);
        } else {
          half_isp.ApplyParallel({
              {half_raw[0], half_color[0], "half-cam3", cam_gain[0], cam_bayer[0]},
              {half_raw[1], half_color[1], "half-cam4", cam_gain[1], cam_bayer[1]},
              {half_raw[2], half_color[2], "half-cam5", cam_gain[2], cam_bayer[2]},
              {half_raw[3], half_color[3], "half-cam6", cam_gain[3], cam_bayer[3]},
          });
        }
        MirrorImage5Vertical(half_raw[2], half_color[2]);
        // half_isp slots 2/3 are physical C5/C6, same digital exposure as image1.
        c56.digital_gain[0] = six_this_frame ? six_isp->DigitalGain(4) : half_isp.DigitalGain(2);
        c56.digital_gain[1] = six_this_frame ? six_isp->DigitalGain(5) : half_isp.DigitalGain(3);
      } else if (process_cameras >= 4) {
        isp.ApplyParallel({
            {left_image, left_color, "cam1", isp_camera_gain[0], image1_bayer,
             fpga_bayer_nr[0], raw_telemetry.RegionForPlane(0)},
            {right_image, right_color, "cam2", isp_camera_gain[1],
             image2_bayer, fpga_bayer_nr[1], raw_telemetry.RegionForPlane(1)},
            {left_front_image, left_front_color, "cam3", isp_camera_gain[2],
             image3_bayer, fpga_bayer_nr[2], raw_telemetry.RegionForPlane(2)},
            {right_front_image, right_front_color, "cam4", isp_camera_gain[3],
             image4_bayer, fpga_bayer_nr[3], raw_telemetry.RegionForPlane(3)},
        });
      } else {
        // S2 exposes only the C1/C2 Bayer planes.
        isp.ApplyParallel({
            {left_image, left_color, "cam1", isp_camera_gain[0], image1_bayer,
             fpga_bayer_nr[0], raw_telemetry.RegionForPlane(0)},
            {right_image, right_color, "cam2", isp_camera_gain[1],
             image2_bayer, fpga_bayer_nr[1], raw_telemetry.RegionForPlane(1)},
        });
      }
      //std::cout << "proc(wb+cvt) " << proc.toc() << std::endl;
      
      // Six-image mode shows/saves C1, C2 and the four half images; otherwise
      // the four full planes.
      const int n_images = half_res_lanes ? 6 : process_cameras;
      const cv::Mat *color_images[6] = {
          &left_color, &right_color, &left_front_color, &right_front_color,
          nullptr, nullptr};
      const cv::Mat *raw_images[6] = {
          &left_image, &right_image, &left_front_image, &right_front_image,
          nullptr, nullptr};
      // Display/save slot -> camera tag.  In six-image mode slots 4/5 show
      // C5h/C4h.  Processing above is unchanged.
      std::string image_tags[6] = {"image1", "image2", "image3",
                                   "image4", "image5", "image6"};
      if (half_res_lanes) {
        static const int kHalfOrder[4] = {0, 2, 1, 3};
        static const char *const kHalfTags[4] = {"C3h", "C4h", "C5h", "C6h"};
        for (int i = 0; i < 4; ++i) {
          color_images[2 + i] = &half_color[kHalfOrder[i]];
          raw_images[2 + i] = &half_raw[kHalfOrder[i]];
          image_tags[2 + i] = std::string("image_") + kHalfTags[kHalfOrder[i]];
        }
      }
      if (show_preview && count % kShowEvery == 0) {
        const cv::Mat *display_images[6];
        for (int i = 0; i < n_images; ++i) {
          display_images[i] = color_images[i];
          if (use_uyvy422) {
            Uyvy422Bt601FullRangeToBgrPreview(*display_images[i],
                                              preview_color[i]);
            display_images[i] = &preview_color[i];
          }
        }
        for (int i = 0; i < n_images && i < n_windows; ++i)
          cv::imshow(window_names[i],
                     show_raw ? *raw_images[i] : *display_images[i]);
        preview_key = cv::waitKey(1);
      }
      // Save independently of HighGUI, including consecutive frames for
      // temporal-noise measurements. Automatic saving requires an explicit
      // option (save_at defaults to -1).
      if (preview_key == 's' || preview_key == 'S' ||
          (save_at >= 0 && (count == save_at || count == save_at + 1)) ||
          (save_every > 0 && count > 0 && count % save_every == 0)) {
          const cv::Mat *display_images[6];
          for (int i = 0; i < n_images; ++i) {
            display_images[i] = color_images[i];
            if (use_uyvy422) {
              Uyvy422Bt601FullRangeToBgrPreview(*display_images[i], preview_color[i]);
              display_images[i] = &preview_color[i];
            }
          }
          for (int i = 0; i < n_images; ++i) {
            const std::string tag = image_tags[i] + "_" + std::to_string(count);
            cv::imwrite("raw_" + tag + ".png", *raw_images[i]);
            cv::imwrite("color_" + tag + ".png", *display_images[i]);
          }
          if (raw_telemetry.valid) {
            std::ofstream telemetry_file("telemetry_" + std::to_string(count) + ".csv");
            WriteRawTelemetryCsv(telemetry_file, raw_telemetry);
            if (!telemetry_file) std::cerr << "[save] telemetry write failed" << std::endl;
          }
          std::cout << "[save] raw_/color_image*_" << count
                    << ".png written" << std::endl;
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

    double digital_gain[4];
    for (int i = 0; i < 4; ++i) digital_gain[i] = isp.DigitalGain(i);
    if (six_this_frame) {
      // Diagnostic plane order remains C1,C2,C4,C3.
      const int slots[4] = {0, 1, 3, 2};
      for (int i = 0; i < 4; ++i) digital_gain[i] = six_isp->DigitalGain(slots[i]);
    }
    LogCaptureDiagnostics(
        count, capture_log_every, process_cameras, is_smartsens, image_timestamp,
        exposure_time, exposure_lines, camera_gain, digital_gain,
        camera_temperature, c56, imu_data, gnss_data, frame_info, t_frame,
        capture_log);
    ++count;
    if (max_frames > 0 && count >= max_frames) break;
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
