#ifndef CYPERSTEREO_API_H_
#define CYPERSTEREO_API_H_

#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include "string"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "uvc.h"
#include "tic_toc.h"
#include "thread_priority.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define CYPERSTEREO_HAVE_NEON 1
#endif

CYPERSTEREO_BEGIN_NAMESPACE

static inline void DeinterleaveFourPlanes(
    const unsigned char *src, int src_stride,
    unsigned char *p0, unsigned char *p1,
    unsigned char *p2, unsigned char *p3,
    int dst_stride, int width, int height) {
  for (int row = 0; row < height; ++row) {
    const unsigned char *s = src + static_cast<size_t>(row) * src_stride;
    const size_t off = static_cast<size_t>(row) * dst_stride;
    unsigned char *d0 = p0 + off;
    unsigned char *d1 = p1 + off;
    unsigned char *d2 = p2 + off;
    unsigned char *d3 = p3 + off;
    int col = 0;
#if defined(CYPERSTEREO_HAVE_NEON)
    for (; col + 16 <= width; col += 16) {
      const uint8x16x4_t v = vld4q_u8(s + 4 * col);
      vst1q_u8(d0 + col, v.val[0]);
      vst1q_u8(d1 + col, v.val[1]);
      vst1q_u8(d2 + col, v.val[2]);
      vst1q_u8(d3 + col, v.val[3]);
    }
#endif
    for (; col < width; ++col) {
      const int si = 4 * col;
      d0[col] = s[si];
      d1[col] = s[si + 1];
      d2[col] = s[si + 2];
      d3[col] = s[si + 3];
    }
  }
}

static inline void DeinterleaveTwoPlanes(
    const unsigned char *src, int src_stride,
    unsigned char *p0, unsigned char *p1,
    int dst_stride, int width, int height) {
  for (int row = 0; row < height; ++row) {
    const unsigned char *s = src + static_cast<size_t>(row) * src_stride;
    const size_t off = static_cast<size_t>(row) * dst_stride;
    unsigned char *d0 = p0 + off;
    unsigned char *d1 = p1 + off;
    int col = 0;
#if defined(CYPERSTEREO_HAVE_NEON)
    for (; col + 16 <= width; col += 16) {
      const uint8x16x2_t v = vld2q_u8(s + 2 * col);
      vst1q_u8(d0 + col, v.val[0]);
      vst1q_u8(d1 + col, v.val[1]);
    }
#endif
    for (; col < width; ++col) {
      d0[col] = s[2 * col];
      d1[col] = s[2 * col + 1];
    }
  }
}

static constexpr int kFx3FrameWidth = 2560;
static constexpr int kFx3FrameHeight = 1024;
static constexpr int kFx3FrameFps = 30;
static constexpr int kFx3FramePixels = kFx3FrameWidth * kFx3FrameHeight;
static constexpr int kFx3HalfFrameWidth = kFx3FrameWidth / 2;
static constexpr int kFx3HalfFramePixels = kFx3HalfFrameWidth * kFx3FrameHeight;

static constexpr int kMt9FrameWidth = 752;
static constexpr int kMt9FrameHeight = 480;
static constexpr int kMt9FrameFps = 60;

static constexpr int kMetaImuBaseCol = 5;
static constexpr int kImuWordsPerSample = 9;
static constexpr int kImuMaxSamplesPerFrame = 7;

static constexpr double kImuGapThresholdSec = 0.007;
static constexpr double kImageGapThresholdSec = 0.040;

struct CameraProfile {
  const char *name;
  int hardware_version;
  int software_version;
  int frame_width;
  int frame_height;
  int fps;
  int num_cameras;
  int cam_width;
  int meta_row;
  int imu_samples_per_frame;
  int gnss_base_col;
};

static constexpr CameraProfile kProfileSmartSens{
    "SmartSens(Cyper-ego)", 2, 3, kFx3FrameWidth, kFx3FrameHeight, kFx3FrameFps,
    4, kFx3HalfFrameWidth, kFx3FrameHeight - 1, 7, 72};

static constexpr CameraProfile kProfileM150{
    "MT9V034(M150)", 0, 2, kMt9FrameWidth, kMt9FrameHeight, kMt9FrameFps,
    2, kMt9FrameWidth, kMt9FrameHeight - 1, 4, 48};

static constexpr CameraProfile kProfileM60{
    "MT9V034(M60)", 1, 2, kMt9FrameWidth, kMt9FrameHeight, kMt9FrameFps,
    2, kMt9FrameWidth, kMt9FrameHeight - 1, 4, 48};

inline const CameraProfile &SelectProfileBySerial(const std::string &serial_num) {
  if (!serial_num.empty()) {
    const char c = static_cast<char>(
        std::toupper(static_cast<unsigned char>(serial_num[0])));
    if (c == 'S') return kProfileSmartSens;
    if (c == 'C') return kProfileM150;
    if (c == 'M') return kProfileM60;
  }
  return kProfileM60;
}

#define BMI088_ACCEL_24G_SEN 0.000732421875
#define BMI088_ACCEL_12G_SEN 0.0003662109375
#define BMI088_ACCEL_6G_SEN 0.00018310546875
#define BMI088_ACCEL_3G_SEN 0.000091552734375
#define BMI088_GYRO_2000_SEN 0.0010652644178602
#define BMI088_GYRO_1000_SEN 0.0005326322089301215
#define BMI088_GYRO_500_SEN 0.0002663161044650608
#define BMI088_GYRO_250_SEN 0.0001331580522325304
#define BMI088_GYRO_125_SEN 0.00006657902611626519
double BMI088_ACCEL_SEN = BMI088_ACCEL_6G_SEN;
double BMI088_GYRO_SEN = BMI088_GYRO_2000_SEN;


struct IMUStreamData {
  double acc_x[kImuMaxSamplesPerFrame];
	double acc_y[kImuMaxSamplesPerFrame];
	double acc_z[kImuMaxSamplesPerFrame];
	double gyro_x[kImuMaxSamplesPerFrame];
	double gyro_y[kImuMaxSamplesPerFrame];
	double gyro_z[kImuMaxSamplesPerFrame];
  double temperature[kImuMaxSamplesPerFrame];
  double imu_timestamp[kImuMaxSamplesPerFrame];
  int imu_count;
};

struct GNSSStreamData {
  bool valid;
  double gnss_timestamp;
  std::string gnss_utc_time;
  double latitude;
  double longitude;
  double altitude;
  // 0 - 无效 1 - 单点定位 2 - 差分定位 4 - RTK 固定解 5 - RTK 浮点解 6 - 惯导定位 50 - UWB 定位
  int fix_type;
  int satellites_used;
  double gps_geoid_height;
  double velocity;
  double heading;
  double hdop;
  double vdop;
  double pdop;
};

struct FrameStreamData {
  cv::Mat left_image;
  cv::Mat right_image;
  cv::Mat left_front_image;
  cv::Mat right_front_image;
  IMUStreamData imu;
  GNSSStreamData gnss;
  std::string serial_num;
  uint32_t hardware_version{0};
  uint32_t software_version{0};
	double image_timestamp;
};

struct Frame {
  const void *data = nullptr;
  std::function<void()> continuation = nullptr;
  Frame() {
  }
  ~Frame() {
    data = nullptr;
    if (continuation) {
      continuation();
      continuation = nullptr;
    }
  }
};

struct FrameInfo {
  std::mutex mtx;
  std::condition_variable con;
  std::shared_ptr<Frame> frame{nullptr};
  CameraProfile profile{kProfileM60};
  FrameStreamData framestream{};
  // Host-side (steady_clock) arrival time of the previous frame. Used to tell
  // a REAL transport drop (host gap ~= camera-timestamp gap) apart from a
  // camera/FPGA timestamp jump (host gap stays ~33ms while the embedded
  // timestamp leaps).
  std::chrono::steady_clock::time_point last_arrival{};
  double host_gap_ms{-1.0};
  double last_imu_timestamp{0};
  double last_image_timestamp{0};
  double last_imu_count_s{0};
  double last_imu_count_ms{0};
  double last_image_count_s{0};
  int image_drop_count{0};
  int imu_drop_count{0};

  void Init(const CameraProfile &p) {
    profile = p;
    framestream.left_image.create(p.frame_height, p.cam_width, CV_8U);
    framestream.right_image.create(p.frame_height, p.cam_width, CV_8U);
    if (p.num_cameras >= 4) {
      framestream.left_front_image.create(p.frame_height, p.cam_width, CV_8U);
      framestream.right_front_image.create(p.frame_height, p.cam_width, CV_8U);
    } else {
      framestream.left_front_image.release();
      framestream.right_front_image.release();
    }
  }

  FrameInfo() { Init(kProfileM60); }
  explicit FrameInfo(const CameraProfile &p) { Init(p); }
};

class WhiteBalance {
 public:
  void Apply(cv::Mat &raw) {
    EstimateGains(raw);

    double bg = b_gain_ > 0.0 ? b_gain_ : 1.0;
    double rg = r_gain_ > 0.0 ? r_gain_ : 1.0;
    BuildLuts(bg, rg);

    const int w = raw.cols;
    for (int y = 0; y < raw.rows; ++y) {
      uchar *p = raw.ptr<uchar>(y);
      const uchar *even_col = (y & 1) ? lut_g_ : lut_b_;
      const uchar *odd_col = (y & 1) ? lut_r_ : lut_g_;
      int x = 0;
      for (; x + 1 < w; x += 2) {
        p[x] = even_col[p[x]];
        p[x + 1] = odd_col[p[x + 1]];
      }
      if (x < w) p[x] = even_col[p[x]];
    }
  }

 private:
  void EstimateGains(const cv::Mat &raw) {
    const int lo = static_cast<int>(kBlackLevel) + 24;
    const int hi = 250;
    double sum_b = 0, sum_g = 0, sum_r = 0;
    int n = 0, n_total = 0;
    for (int y = 0; y + 1 < raw.rows; y += kEstStep) {
      const uchar *r0 = raw.ptr<uchar>(y);
      const uchar *r1 = raw.ptr<uchar>(y + 1);
      for (int x = 0; x + 1 < raw.cols; x += kEstStep) {
        int b = r0[x];
        int g = (r0[x + 1] + r1[x]) >> 1;
        int r = r1[x + 1];
        ++n_total;
        int luma = (b + 2 * g + r) >> 2;
        if (luma < lo || luma > hi) continue;
        sum_b += b;
        sum_g += g;
        sum_r += r;
        ++n;
      }
    }
    if (n < n_total / 100) return;

    double b = sum_b / n - kBlackLevel;
    double gg = sum_g / n - kBlackLevel;
    double r = sum_r / n - kBlackLevel;
    if (b < 1.0 || gg < 1.0 || r < 1.0) return;

    double b_gain = gg / b;
    double r_gain = gg / r;
    if (b_gain_ <= 0.0) {
      b_gain_ = b_gain;
      r_gain_ = r_gain;
    } else {
      b_gain_ = (1.0 - kSmooth) * b_gain_ + kSmooth * b_gain;
      r_gain_ = (1.0 - kSmooth) * r_gain_ + kSmooth * r_gain;
    }
  }

  void BuildLuts(double bg, double rg) {
    if (built_ && bg == lut_bg_ && rg == lut_rg_) return;
    built_ = true;
    lut_bg_ = bg;
    lut_rg_ = rg;
    for (int i = 0; i < 256; ++i) {
      lut_b_[i] = cv::saturate_cast<uchar>((i - kBlackLevel) * bg);
      lut_g_[i] = cv::saturate_cast<uchar>((i - kBlackLevel) * 1.0);
      lut_r_[i] = cv::saturate_cast<uchar>((i - kBlackLevel) * rg);
    }
  }

  static constexpr double kBlackLevel = 16.0;
  static constexpr double kSmooth = 0.05;
  static constexpr int kEstStep = 8;
  double b_gain_ = -1.0;
  double r_gain_ = -1.0;
  double lut_bg_ = -1.0;
  double lut_rg_ = -1.0;
  bool built_ = false;
  uchar lut_b_[256], lut_g_[256], lut_r_[256];
};

inline void ApplyISP(cv::Mat &raw, cv::Mat &color, WhiteBalance &wb,
                     const char *name) {
  ApplyThreadPriority(ThreadRole::kWorker, name);
  wb.Apply(raw);
  cv::cvtColor(raw, color, cv::COLOR_BayerRG2BGR);
}

cv::Mat FastGuidedfilter(cv::Mat &I, int r, float eps, int size) {
    r = r / size;
    int wsize = 2 * r + 1;
    I.convertTo(I, CV_32FC1, 1/255.0);

    cv::Mat small_I, small_p;
    cv::resize(I, small_I, I.size()/size, 0, 0, cv::INTER_AREA);
    small_p = small_I;

    cv::Mat mean_I, mean_p;
    cv::boxFilter(small_I, mean_I, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    mean_p = mean_I;

    cv::Mat mean_II, mean_Ip;
    mean_II = small_I.mul(small_I);
    cv::boxFilter(mean_II, mean_II, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    mean_Ip = mean_II;

    cv::Mat var_I, cov_Ip, mean_mul_I;
    mean_mul_I=mean_I.mul(mean_I);
    cv::subtract(mean_II, mean_mul_I, var_I);
    cov_Ip = var_I;
    
    cv::Mat a, b;
    cv::divide(cov_Ip, (var_I+eps),a);
    cv::subtract(mean_p, a.mul(mean_I), b);

    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    cv::boxFilter(b, mean_b, -1, cv::Size(wsize, wsize), cv::Point(-1, -1), true, cv::BORDER_REFLECT);

    cv::resize(mean_a, mean_a, I.size());
    cv::resize(mean_b, mean_b, I.size());

    cv::Mat out = mean_a.mul(I) + mean_b;
    out.convertTo(out, CV_8UC1, 255);
    return out;
}


bool FindCyperstereoDevices(std::shared_ptr<uvc::device>& cyperstereo_device) {
  std::vector<std::shared_ptr<uvc::device>> cyperstereo_devices;

  auto context = uvc::create_context();
  auto devices = uvc::query_devices(context);
  if (devices.size() <= 0) {
    std::cout << "No devices :(" << std::endl;
    return false;
  }

  for (auto &&device : devices) {
    auto vid = uvc::get_vendor_id(*device);
    if (vid == Cyperstereo_VID) {
      cyperstereo_devices.push_back(device);
    }
  }

  size_t n = cyperstereo_devices.size();
  if (n <= 0) {
    std::cout << "No Cyperstereo devices :(" << std::endl;
    return false;
  }

  std::cout  << "usb devices: " << std::endl;
  for (size_t i = 0; i < n; i++) {
    auto device = cyperstereo_devices[i];
    auto name = uvc::get_video_name(*device);
    auto vid = uvc::get_vendor_id(*device);
    auto pid = uvc::get_product_id(*device);
    auto rawname = uvc::get_name(*device);
    auto serial_num = uvc::get_serial_number(*device);
    std::cout << "  index: " << i << ", rawname: " << rawname << ", name: " << name << ", vid: 0x"
              << std::hex << vid << ", pid: 0x" << std::hex << pid
              << std::dec << ", serial_num: " << serial_num << std::endl;
  }

  if (n <= 1) {
    cyperstereo_device = cyperstereo_devices[0];
    std::cout << "Only one Cyperstereo device, select index: 0" << std::endl;
  } else {
    while (true) {
      size_t i;
      std::cout << "There are " << n << " Cyperstereo devices, select index: " << std::endl;
      std::cin >> i;
      if (i >= n) {
        std::cout << "Index out of range :(" << std::endl;
        continue;
      }
      cyperstereo_device = cyperstereo_devices[i];
      break;
    }
  }
  return true;
}

void WaitForStream(FrameInfo& frame_info) {
  std::unique_lock<std::mutex> lock(frame_info.mtx);
  const auto frame_ready = [&frame_info]() { return frame_info.frame != nullptr; };
  if (frame_info.frame == nullptr) {
    if (!frame_info.con.wait_for(lock, std::chrono::seconds(5), frame_ready))
      throw std::runtime_error("Timeout waiting for frame.");
  }  
  frame_info.frame = nullptr;
}

void SetStreamData(FrameInfo& frame_info, const void *data, std::function<void()> continuation) {
  const auto t_cb_start = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(frame_info.mtx);
  frame_info.host_gap_ms =
      frame_info.last_arrival.time_since_epoch().count() == 0
          ? -1.0
          : std::chrono::duration<double, std::milli>(
                t_cb_start - frame_info.last_arrival).count();
  frame_info.last_arrival = t_cb_start;
        if (frame_info.frame == nullptr) {
          frame_info.frame = std::make_shared<struct Frame>();
        } else {
          if (frame_info.frame->continuation) {
            frame_info.frame->continuation();
          }
        }
        frame_info.frame->data = data;
        frame_info.frame->continuation = continuation;

        const CameraProfile &prof = frame_info.profile;
        FrameStreamData &fs = frame_info.framestream;

        cv::Mat img(prof.frame_height, prof.frame_width, CV_8UC2,
                    const_cast<void *>(frame_info.frame->data));

        cv::Mat *lo_plane;
        cv::Mat *hi_plane;
        if (prof.num_cameras >= 4) {
          DeinterleaveFourPlanes(
              img.ptr<unsigned char>(0, 0), static_cast<int>(img.step),
              fs.left_image.ptr<unsigned char>(0, 0),
              fs.right_image.ptr<unsigned char>(0, 0),
              fs.left_front_image.ptr<unsigned char>(0, 0),
              fs.right_front_image.ptr<unsigned char>(0, 0),
              static_cast<int>(fs.left_image.step),
              prof.cam_width, prof.frame_height);
          lo_plane = &fs.left_image;
          hi_plane = &fs.right_image;
        } else {
          DeinterleaveTwoPlanes(
              img.ptr<unsigned char>(0, 0), static_cast<int>(img.step),
              fs.right_image.ptr<unsigned char>(0, 0),
              fs.left_image.ptr<unsigned char>(0, 0),
              static_cast<int>(fs.right_image.step),
              prof.cam_width, prof.frame_height);
          lo_plane = &fs.right_image;
          hi_plane = &fs.left_image;
        }

        const int imu_end_col =
            kMetaImuBaseCol + prof.imu_samples_per_frame * kImuWordsPerSample;
        const int gnss_end_col = prof.gnss_base_col + 23;
        const int needed_cols = imu_end_col > gnss_end_col ? imu_end_col : gnss_end_col;
        if (prof.meta_row < lo_plane->rows && lo_plane->cols >= needed_cols) {
          const uchar *hi = hi_plane->ptr<uchar>(prof.meta_row);
          const uchar *lo = lo_plane->ptr<uchar>(prof.meta_row);
          auto meta_u = [&](int col) -> int {
            return (static_cast<int>(hi[col]) << 8) | static_cast<int>(lo[col]);
          };
          auto meta_s = [&](int col) -> int {
            return static_cast<int16_t>((hi[col] << 8) | lo[col]);
          };

          const int ver0 = meta_u(0);
          const int ver1 = meta_u(1);
          const bool marker_ok =
              (ver0 == prof.hardware_version) && (ver1 == prof.software_version);

          {
            static int dbg = 0;
            if (dbg < 5 || (!marker_ok && dbg % 100 == 0)) {
              std::cout << "[meta] profile=" << prof.name
                        << " serial=" << fs.serial_num
                        << " res=" << prof.frame_width << "x" << prof.frame_height
                        << " ver0=" << ver0 << " ver1=" << ver1
                        << " (expect " << prof.hardware_version << "/"
                        << prof.software_version << ")"
                        << (marker_ok ? " OK" : " MISMATCH") << std::endl;
            }
            ++dbg;
          }

          if (marker_ok) {
            fs.hardware_version = static_cast<uint32_t>(ver0);
            fs.software_version = static_cast<uint32_t>(ver1);
          }
          double image_count_hour = meta_u(2);
          double image_count_ms = meta_u(3);
          double image_count_s = meta_u(4);
          if (!marker_ok) {
            image_count_hour = 0;
            image_count_ms = 0;
            image_count_s = 0;
          }

          fs.image_timestamp =
              image_count_hour * 12 * 3600 + image_count_s + image_count_ms / 10000.0;

          if (frame_info.last_image_timestamp > 0.0) {
            const double image_gap =
                fs.image_timestamp - frame_info.last_image_timestamp;
            if (image_gap > kImageGapThresholdSec) {
              ++frame_info.image_drop_count;
              std::cout << "[drop] image gap=" << std::fixed
                        << std::setprecision(3) << image_gap * 1000.0
                        << " ms (threshold " << kImageGapThresholdSec * 1000.0
                        << " ms)  prev_ts=" << frame_info.last_image_timestamp
                        << "  cur_ts=" << fs.image_timestamp
                        << "  host_gap=" << frame_info.host_gap_ms << " ms"
                        << std::endl;
              // Camera-timestamp gap vs host arrival gap disagree by a lot ->
              // frames kept flowing and only the embedded timestamp leapt.
              if (frame_info.host_gap_ms >= 0.0 &&
                  frame_info.host_gap_ms < image_gap * 1000.0 * 0.5) {
                std::cout << "[drop] note: frames arrived continuously "
                             "(host_gap ~ frame period) -> camera/FPGA "
                             "TIMESTAMP JUMP, not a real transport drop"
                          << std::endl;
              } else if (frame_info.host_gap_ms >= 0.0) {
                std::cout << "[drop] note: host arrival gap matches -> REAL "
                             "transport drop (USB/device stopped delivering)"
                          << std::endl;
              }
            }
          }
          frame_info.last_image_count_s = image_count_s;
          frame_info.last_image_timestamp = fs.image_timestamp;

          const int imu_n = prof.imu_samples_per_frame;
          for (int i = 0; i < imu_n; ++i) {
            const int base = kMetaImuBaseCol + i * kImuWordsPerSample;
            const double imu_count_ms = meta_u(base + 0);
            const double imu_count_s = meta_u(base + 1);
            fs.imu.imu_timestamp[i] =
                image_count_hour * 12 * 3600 + imu_count_s + imu_count_ms / 10000.0;
            fs.imu.acc_x[i] = meta_s(base + 2) * BMI088_ACCEL_SEN;
            fs.imu.acc_y[i] = meta_s(base + 3) * BMI088_ACCEL_SEN;
            fs.imu.acc_z[i] = meta_s(base + 4) * BMI088_ACCEL_SEN;
            fs.imu.gyro_x[i] = meta_s(base + 5) * BMI088_GYRO_SEN;
            fs.imu.gyro_y[i] = meta_s(base + 6) * BMI088_GYRO_SEN;
            fs.imu.gyro_z[i] = meta_s(base + 7) * BMI088_GYRO_SEN;
            double t = meta_s(base + 8);
            if (t > 1023) t -= 2048;
            fs.imu.temperature[i] = t * 0.125 + 23;
          }

          int valid_samples = imu_n;
          if (imu_n >= 1) {
            const int last_base = kMetaImuBaseCol + (imu_n - 1) * kImuWordsPerSample;
            bool last_all_zero = true;
            for (int w = 0; w < kImuWordsPerSample; ++w) {
              if (meta_u(last_base + w) != 0) { last_all_zero = false; break; }
            }
            if (last_all_zero) valid_samples = imu_n - 1;
          }
          fs.imu.imu_count = valid_samples;

          for (int i = 0; i < fs.imu.imu_count; ++i) {
            const double ts = fs.imu.imu_timestamp[i];
            if (frame_info.last_imu_timestamp > 0.0) {
              const double imu_gap = ts - frame_info.last_imu_timestamp;
              if (imu_gap > kImuGapThresholdSec) {
                ++frame_info.imu_drop_count;
                std::cout << "[drop] imu gap=" << std::fixed
                          << std::setprecision(3) << imu_gap * 1000.0
                          << " ms (threshold " << kImuGapThresholdSec * 1000.0
                          << " ms)  prev_ts=" << frame_info.last_imu_timestamp
                          << "  cur_ts=" << ts
                          << "  host_gap=" << frame_info.host_gap_ms << " ms"
                          << std::endl;
              }
            }
            frame_info.last_imu_timestamp = ts;
          }

          if (marker_ok) {
            const int gnss_shift = prof.gnss_base_col * 2 - 96;
            auto g8 = [&](int ref_byte) -> int {
              const int b = ref_byte + gnss_shift;
              return (b & 1) ? hi[b >> 1] : lo[b >> 1];
            };

            double gnss_count_ms = ((g8(97) << 8) | g8(96));
            double gnss_count_s = ((g8(99) << 8) | g8(98));
            double gnss_timestamp =
                image_count_hour * 12 * 3600 + gnss_count_s + gnss_count_ms / 10000.0;

            uint16_t gnss_utc_hour = (g8(101) >> 3);
            uint16_t gps_utc_minute = (g8(100) >> 2);
            uint16_t gps_utc_second = (g8(103) >> 2);
            uint16_t gps_utc_second_ms = (g8(102) >> 1) * 10;
            std::ostringstream gnss_utc_stream;
            gnss_utc_stream << std::setfill('0') << std::setw(2) << gnss_utc_hour << ":"
                            << std::setw(2) << gps_utc_minute << ":"
                            << std::setw(2) << gps_utc_second << "."
                            << std::setw(3) << gps_utc_second_ms;
            std::string gnss_utc_time = gnss_utc_stream.str();
            bool gnss_time_unchanged = (gnss_utc_time == fs.gnss.gnss_utc_time);
            fs.gnss.valid = false;

            double gps_lat_degree = (g8(105) >> 1);
            double gps_lat_minute = (g8(104) >> 2);
            double gps_lat_fraction =
                ((g8(107) << 9) | (g8(106) << 1) | ((g8(104) & 0x02) >> 1));
            uint16_t gps_lat_is_north = (g8(104) & 0x01);
            double latitude =
                gps_lat_degree + (gps_lat_minute + gps_lat_fraction / 100000.0) / 60.0;
            if (gps_lat_is_north == 0) latitude = -latitude;
            double gps_lon_degree = (g8(109));
            double gps_lon_minute = (g8(108) >> 2);
            double gps_lon_fraction =
                ((g8(111) << 9) | (g8(110) << 1) | ((g8(108) & 0x02) >> 1));
            uint16_t gps_lon_is_east = (g8(108) & 0x01);
            double longitude =
                gps_lon_degree + (gps_lon_minute + gps_lon_fraction / 100000.0) / 60.0;
            if (gps_lon_is_east == 0) longitude = -longitude;
            double gps_alt_integer = (g8(113) << 6 | g8(112) >> 2);
            double gps_alt_fraction = ((g8(115) << 8) | g8(114));
            double altitude = gps_alt_integer + gps_alt_fraction / 10000.0;
            uint16_t gps_fix_quality = (g8(117) >> 5);
            uint16_t gps_satellites = (g8(116) >> 1);
            double gps_geoid_integer = ((g8(119) << 6) | (g8(118) >> 2));
            double gps_geoid_fraction = ((g8(121) << 4) | (g8(120) >> 4));
            uint16_t gps_geoid_negative = (g8(118) & 0x01);
            double gps_geoid_height = gps_geoid_integer + gps_geoid_fraction / 10000.0;
            if (gps_geoid_negative == 1) gps_geoid_height = -gps_geoid_height;
            double gps_lat_err_integer = ((g8(123) << 8) | g8(122));
            double gps_lat_err_fraction = ((g8(125) << 2) | g8(124) >> 6);
            double gps_lat_err = gps_lat_err_integer + gps_lat_err_fraction / 1000.0;
            double gps_lon_err_integer = ((g8(127) << 8) | g8(126));
            double gps_lon_err_fraction = ((g8(129) << 2) | g8(128) >> 6);
            double gps_lon_err = gps_lon_err_integer + gps_lon_err_fraction / 1000.0;
            double gps_alt_err_integer = ((g8(131) << 8) | g8(130));
            double gps_alt_err_fraction = ((g8(133) << 2) | g8(132) >> 6);
            double gps_alt_err = gps_alt_err_integer + gps_alt_err_fraction / 1000.0;
            double hdop = sqrt(gps_lat_err * gps_lat_err + gps_lon_err * gps_lon_err);
            double vdop = gps_alt_err;
            double pdop = sqrt(hdop * hdop + vdop * vdop);
            double gps_speed_kmh_integer = ((g8(135) << 8) | g8(134));
            double gps_speed_kmh_fraction = ((g8(137) << 2) | g8(136) >> 6);
            double gps_speed_kmh = gps_speed_kmh_integer + gps_speed_kmh_fraction / 1000.0;
            double gps_angle_integer = ((g8(139) << 8) | g8(138));
            double gps_angle_fraction = ((g8(141) << 2) | g8(140) >> 6);
            double gps_angle = gps_angle_integer + gps_angle_fraction / 1000.0;

            if (!gnss_time_unchanged && gps_fix_quality >= 1 && hdop < 50 &&
                vdop < 50 && pdop < 50) {
              fs.gnss.valid = true;
              fs.gnss.gnss_timestamp = gnss_timestamp;
              fs.gnss.gnss_utc_time = gnss_utc_time;
              fs.gnss.latitude = latitude;
              fs.gnss.longitude = longitude;
              fs.gnss.altitude = altitude;
              fs.gnss.fix_type = gps_fix_quality;
              fs.gnss.satellites_used = gps_satellites;
              fs.gnss.gps_geoid_height = gps_geoid_height;
              fs.gnss.hdop = hdop;
              fs.gnss.vdop = vdop;
              fs.gnss.pdop = pdop;
              fs.gnss.velocity = gps_speed_kmh;
              fs.gnss.heading = gps_angle;
            }
          } else {
            fs.gnss.valid = false;
          }
        }

        const double cb_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t_cb_start)
                                 .count();
        static int cb_calls = 0;
        static int cb_slow = 0;
        static double cb_ms_sum = 0.0;
        static double cb_ms_max = 0.0;
        ++cb_calls;
        cb_ms_sum += cb_ms;
        if (cb_ms > cb_ms_max) cb_ms_max = cb_ms;
        if (cb_ms > 33.0) {
          ++cb_slow;
          std::cout << "[cb] SetStreamData took " << std::dec << std::fixed
                    << std::setprecision(2) << cb_ms << " ms (>33ms)"
                    << std::endl;
        }
        frame_info.con.notify_one();
}

CYPERSTEREO_END_NAMESPACE
#endif
