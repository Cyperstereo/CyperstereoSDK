#ifndef CYPERSTEREO_API_H_
#define CYPERSTEREO_API_H_

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <condition_variable>
#include <iomanip>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include "string"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include "uvc.h"
#include "tic_toc.h"
#include "thread_priority.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#define CYPERSTEREO_HAVE_AVX2_OUTPUT 1
#define CYPERSTEREO_AVX2_TARGET __attribute__((target("avx2")))
#elif defined(__AVX2__)
#define CYPERSTEREO_HAVE_AVX2_OUTPUT 1
#define CYPERSTEREO_AVX2_TARGET
#endif
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define CYPERSTEREO_HAVE_NEON 1
#endif

CYPERSTEREO_BEGIN_NAMESPACE

// OpenCV 4.11 (PR opencv#26109) removed the arithmetic/bitwise/shift
// operators on universal intrinsics in favour of v_add/v_sub/v_mul/v_and/
// v_shl/v_shr (they clash with the RISC-V RVV built-in vector types). Our
// CV_SIMD128 fallback loops are written with operators -- which keeps them
// bit-exact and still builds against the 4.5.x that ships with most
// distros. Restore the operators for newer OpenCV, scoped to this SDK's
// namespace so ordinary lookup finds them inside our code while we never
// redefine (and clash with) OpenCV's own operators on < 4.11.
#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 11)
#define CYPERSTEREO_VOP_BIN(OP, FUN, TYPE)             \
  static inline cv::TYPE operator OP(const cv::TYPE &a, \
                                     const cv::TYPE &b) { return cv::FUN(a, b); }
#define CYPERSTEREO_VOP_ALL(TYPE)     \
  CYPERSTEREO_VOP_BIN(+, v_add, TYPE) \
  CYPERSTEREO_VOP_BIN(-, v_sub, TYPE) \
  CYPERSTEREO_VOP_BIN(*, v_mul, TYPE) \
  CYPERSTEREO_VOP_BIN(&, v_and, TYPE)
CYPERSTEREO_VOP_ALL(v_uint16x8)
CYPERSTEREO_VOP_ALL(v_uint32x4)
CYPERSTEREO_VOP_ALL(v_int16x8)
CYPERSTEREO_VOP_ALL(v_int32x4)
#undef CYPERSTEREO_VOP_ALL
#undef CYPERSTEREO_VOP_BIN
#define CYPERSTEREO_VOP_SHIFT(TYPE)                                       \
  static inline cv::TYPE operator<<(const cv::TYPE &a, int n) {           \
    return cv::v_shl(a, n);                                               \
  }                                                                      \
  static inline cv::TYPE operator>>(const cv::TYPE &a, int n) {           \
    return cv::v_shr(a, n);                                               \
  }
CYPERSTEREO_VOP_SHIFT(v_uint16x8)
CYPERSTEREO_VOP_SHIFT(v_uint32x4)
CYPERSTEREO_VOP_SHIFT(v_int16x8)
CYPERSTEREO_VOP_SHIFT(v_int32x4)
#undef CYPERSTEREO_VOP_SHIFT
#endif

static inline void DeinterleaveFourPlanes(
    const unsigned char *src, int src_stride,
    unsigned char *p0, unsigned char *p1,
    unsigned char *p2, unsigned char *p3,
    int dst_stride, int width, int height) {
#if defined(CYPERSTEREO_HAVE_NEON)
  for (int row = 0; row < height; ++row) {
    const unsigned char *s = src + static_cast<size_t>(row) * src_stride;
    const size_t off = static_cast<size_t>(row) * dst_stride;
    unsigned char *d0 = p0 + off;
    unsigned char *d1 = p1 + off;
    unsigned char *d2 = p2 + off;
    unsigned char *d3 = p3 + off;
    int col = 0;
    for (; col + 16 <= width; col += 16) {
      const uint8x16x4_t v = vld4q_u8(s + 4 * col);
      vst1q_u8(d0 + col, v.val[0]);
      vst1q_u8(d1 + col, v.val[1]);
      vst1q_u8(d2 + col, v.val[2]);
      vst1q_u8(d3 + col, v.val[3]);
    }
    for (; col < width; ++col) {
      const int si = 4 * col;
      d0[col] = s[si];
      d1[col] = s[si + 1];
      d2[col] = s[si + 2];
      d3[col] = s[si + 3];
    }
  }
#else
  // x86 and other targets: cv::split on a CV_8UC4 view is SIMD-optimized
  // (measured 0.32 ms vs 0.47 ms scalar for 1280x1024x4). This runs on the
  // capture thread at the highest RT priority, so every saved cycle there
  // is a cycle it cannot steal from the ISP workers.
  const cv::Mat s4(height, width, CV_8UC4, const_cast<unsigned char *>(src),
                   static_cast<size_t>(src_stride));
  cv::Mat outs[4] = {
      cv::Mat(height, width, CV_8U, p0, static_cast<size_t>(dst_stride)),
      cv::Mat(height, width, CV_8U, p1, static_cast<size_t>(dst_stride)),
      cv::Mat(height, width, CV_8U, p2, static_cast<size_t>(dst_stride)),
      cv::Mat(height, width, CV_8U, p3, static_cast<size_t>(dst_stride))};
  cv::split(s4, outs);
#endif
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

// Dynamic IMU sample-count detection. This firmware packs a VARIABLE number of
// IMU samples per frame (e.g. ~200Hz IMU vs ~54fps image => 3 or 4 per frame)
// and does NOT zero unused metadata slots -- they retain a stale value. So the
// genuine sample count is found by validating each slot against:
//   (a) timestamp continuity: within a window of the image timestamp, and a
//       small forward step from the previous accepted sample; and
//   (b) temperature plausibility: within the BMI088 range and only a small
//       change from the running temperature reference.
// The first slot that fails ends the genuine run (stale slots only trail).
static constexpr double kImuTsToImageWindowSec = 0.5;
static constexpr double kImuSampleStepMaxSec = 0.050;
static constexpr double kImuTempMinC = -45.0;
static constexpr double kImuTempMaxC = 130.0;
static constexpr double kImuTempStepC = 8.0;

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

// Trusted Cyperstereo USB serial prefixes: S=SmartSens, C=M150, M=M60.
inline bool IsValidCyperstereoSerial(const std::string &serial_num) {
  if (serial_num.empty()) return false;
  const char c = static_cast<char>(
      std::toupper(static_cast<unsigned char>(serial_num[0])));
  return c == 'S' || c == 'C' || c == 'M';
}

inline const CameraProfile &SelectProfileBySerial(const std::string &serial_num) {
  if (IsValidCyperstereoSerial(serial_num)) {
    const char c = static_cast<char>(
        std::toupper(static_cast<unsigned char>(serial_num[0])));
    if (c == 'S') return kProfileSmartSens;
    if (c == 'C') return kProfileM150;
    if (c == 'M') return kProfileM60;
  }
  // No (valid) SN burned into FX3: default to MT9V034(M60). Prefer
  // SelectProfile(serial, device) so SmartSens units without SN are still
  // detected via advertised UVC resolution. M150 vs M60 is refined from the
  // first metadata marker in SetStreamData.
  return kProfileM60;
}

// SN-first profile pick; when SN is missing/invalid, discriminate MT9V034
// (752x480) vs SmartSens (2560x1024) from the device's UVC frame sizes.
inline const CameraProfile &SelectProfile(const std::string &serial_num,
                                          const uvc::device &device) {
  if (IsValidCyperstereoSerial(serial_num))
    return SelectProfileBySerial(serial_num);

  const bool has_smartsens =
      uvc::has_frame_size(device, kFx3FrameWidth, kFx3FrameHeight);
  const bool has_mt9 =
      uvc::has_frame_size(device, kMt9FrameWidth, kMt9FrameHeight);

  if (has_smartsens && !has_mt9) {
    std::cout << "[api] no USB serial; UVC size " << kFx3FrameWidth << "x"
              << kFx3FrameHeight << " -> " << kProfileSmartSens.name
              << std::endl;
    return kProfileSmartSens;
  }
  if (has_mt9) {
    std::cout << "[api] no USB serial; UVC size " << kMt9FrameWidth << "x"
              << kMt9FrameHeight << " -> " << kProfileM60.name
              << " (M150/M60 refined from metadata)" << std::endl;
    return kProfileM60;
  }
  if (has_smartsens) {
    std::cout << "[api] no USB serial; UVC advertises both families, preferring "
              << kProfileSmartSens.name << std::endl;
    return kProfileSmartSens;
  }

  std::cout << "[api] no USB serial and no known UVC size; defaulting to "
            << kProfileM60.name << std::endl;
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

// One frame's worth of raw metadata, kept in a small ring buffer so that when
// a timestamp anomaly fires we can replay the frames AROUND it.  Needed to
// tell apart (a) the FPGA counter really jumping (one +X step between two
// adjacent 5ms IMU samples inside one frame), (b) a stretch of STALE
// timestamps followed by recovery (previous frames' samples all old, then a
// snap forward), and (c) metadata corruption (raw fields inconsistent).
struct FrameDiagRec {
  uint64_t seq{0};
  double host_ms{0};      // host arrival gap vs previous frame
  int img_hour{0}, img_ms{0}, img_s{0};   // raw meta cols 2/3/4
  double image_ts{0};
  int imu_n{0};
  int imu_ms[kImuMaxSamplesPerFrame]{};   // raw per-sample subsecond field
  int imu_s[kImuMaxSamplesPerFrame]{};    // raw per-sample second field
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
  // Running IMU die-temperature reference for stale-slot rejection. The BMI088
  // temperature changes very slowly, so a genuine sample stays close to this;
  // a stale/held-over metadata slot (this firmware does not zero unused IMU
  // slots) jumps away from it. Sentinel < -900 means "no reference yet".
  double last_imu_temperature{-1000.0};
  // Anomaly replay ring: last kDiagRingLen frames of raw metadata.
  static constexpr int kDiagRingLen = 10;
  FrameDiagRec diag_ring[kDiagRingLen]{};
  uint64_t frame_seq{0};
  // When an anomaly fires we dump the ring (past frames) and keep dumping
  // this many FUTURE frames to see whether the jump persists or reverts.
  int dump_frames_left{0};
  double last_imu_count_s{0};
  double last_imu_count_ms{0};
  double last_image_count_s{0};
  int image_drop_count{0};
  int imu_drop_count{0};
  // Frames rejected because the metadata marker (ver0/ver1) didn't match:
  // full-length transfers whose CONTENT is misaligned stream data (the FX3
  // framing-offset failure mode).  They are dropped before publication so
  // consumers never see image_timestamp=0 / garbage pixels.
  uint64_t meta_bad_count{0};
  // Snapshot of meta_bad_count at the previous PUBLISHED frame: if it moved,
  // the current image gap spans frames we discarded (framing storm), which
  // is real data loss even though host arrivals looked continuous.
  uint64_t meta_bad_at_last_good{0};
  uint64_t meta_bad_suppressed{0};
  int meta_bad_logged_in_window{0};
  std::chrono::steady_clock::time_point meta_bad_window_start{};

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
    // Estimating on every 2nd frame halves the estimator cost; the gains
    // are EMA-smoothed (kSmooth=0.05, ~0.7s time constant) so the update
    // rate change is imperceptible.
    if ((frame_idx_++ & 1) == 0 || b_gain_ <= 0.0) EstimateGains(raw);

    double bg = b_gain_ > 0.0 ? b_gain_ : 1.0;
    double rg = r_gain_ > 0.0 ? r_gain_ : 1.0;
    BuildLuts(bg, rg);

#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
    if (UseNeonWbLut()) {
      ApplyLutsNeon(raw);
      return;
    }
#endif
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
  // Robust gray-world: MEAN-based gray-world forces the frame AVERAGE to
  // neutral, so under warm indoor light (wood floor, skin, warm lamps
  // dominating the average) it over-corrects and pushes truly neutral
  // surfaces -- white walls -- toward blue. Using the MEDIAN of per-sample
  // g/b and g/r ratios instead makes the DOMINANT surface neutral (indoor
  // scenes are usually dominated by white/gray walls and desks), and a few
  // strongly colored patches (blue dusk window, orange cloth) cannot skew
  // it. Gains are also clamped to a plausible illuminant range.
  void EstimateGains(const cv::Mat &raw) {
    const int lo = static_cast<int>(kBlackLevel) + 24;
    const int hi = 250;
    ratios_bg_.clear();
    ratios_rg_.clear();
    int n_total = 0;
    for (int y = 0; y + 1 < raw.rows; y += kEstStep) {
      const uchar *r0 = raw.ptr<uchar>(y);
      const uchar *r1 = raw.ptr<uchar>(y + 1);
      for (int x = 0; x + 1 < raw.cols; x += kEstStep) {
        int b = r0[x];
        int g0 = r0[x + 1];
        int g1 = r1[x];
        int r = r1[x + 1];
        int g = (g0 + g1) >> 1;
        ++n_total;
        // Reject samples where ANY channel is at/near sensor clipping: a
        // clipped channel flattens the ratio toward 1.0, and large bright
        // areas (a white desk under the lamp fills ~1/4 of the night frame)
        // would drag the median toward "no correction", leaving a blue cast
        // on the walls the estimator should have neutralized.
        if (b >= 250 || g0 >= 250 || g1 >= 250 || r >= 250) continue;
        int luma = (b + 2 * g + r) >> 2;
        if (luma < lo || luma > hi) continue;
        double bb = b - kBlackLevel;
        double gg = g - kBlackLevel;
        double rr = r - kBlackLevel;
        // Reject samples where any channel is too close to the black level:
        // the ratio would be dominated by noise.
        if (bb < 4.0 || gg < 4.0 || rr < 4.0) continue;
        ratios_bg_.push_back(static_cast<float>(gg / bb));
        ratios_rg_.push_back(static_cast<float>(gg / rr));
      }
    }
    if (static_cast<int>(ratios_bg_.size()) < n_total / 100) return;

    const auto median = [](std::vector<float> &v) {
      const size_t mid = v.size() / 2;
      std::nth_element(v.begin(), v.begin() + mid, v.end());
      return static_cast<double>(v[mid]);
    };
    double b_gain = std::min(kGainMax, std::max(kGainMin, median(ratios_bg_)));
    double r_gain = std::min(kGainMax, std::max(kGainMin, median(ratios_rg_)));
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

#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
  static bool UseNeonWbLut() {
    // Cached A/B switch, same convention as the output-kernel switches.
    static const bool enabled =
        std::getenv("CYPERSTEREO_DISABLE_NEON_WB") == nullptr;
    return enabled;
  }

  // 256-entry LUT as four 64-byte TBL/TBX lookups per vector (A64-only
  // instructions). TBX leaves lanes whose adjusted index is out of [0,63]
  // untouched, so the four stages each translate exactly their quarter of
  // the value range; bit-exact vs the scalar loop. RK3588 A76: 1.08 ->
  // 0.98 ms for a 1280x1024 frame.
  struct NeonLut256 {
    uint8x16x4_t t0, t1, t2, t3;
    void Load(const uchar *lut) {
      t0 = vld1q_u8_x4(lut);
      t1 = vld1q_u8_x4(lut + 64);
      t2 = vld1q_u8_x4(lut + 128);
      t3 = vld1q_u8_x4(lut + 192);
    }
    inline uint8x16_t Apply(uint8x16_t idx) const {
      uint8x16_t r = vqtbl4q_u8(t0, idx);
      r = vqtbx4q_u8(r, t1, vsubq_u8(idx, vdupq_n_u8(64)));
      r = vqtbx4q_u8(r, t2, vsubq_u8(idx, vdupq_n_u8(128)));
      r = vqtbx4q_u8(r, t3, vsubq_u8(idx, vdupq_n_u8(192)));
      return r;
    }
  };

  void ApplyLutsNeon(cv::Mat &raw) {
    NeonLut256 lb, lg, lr;
    lb.Load(lut_b_);
    lg.Load(lut_g_);
    lr.Load(lut_r_);
    const int w = raw.cols;
    for (int y = 0; y < raw.rows; ++y) {
      uchar *p = raw.ptr<uchar>(y);
      const NeonLut256 &even_col = (y & 1) ? lg : lb;
      const NeonLut256 &odd_col = (y & 1) ? lr : lg;
      const uchar *lut_even = (y & 1) ? lut_g_ : lut_b_;
      const uchar *lut_odd = (y & 1) ? lut_r_ : lut_g_;
      int x = 0;
      for (; x + 32 <= w; x += 32) {
        uint8x16x2_t v = vld2q_u8(p + x);
        v.val[0] = even_col.Apply(v.val[0]);
        v.val[1] = odd_col.Apply(v.val[1]);
        vst2q_u8(p + x, v);
      }
      for (; x + 1 < w; x += 2) {
        p[x] = lut_even[p[x]];
        p[x + 1] = lut_odd[p[x + 1]];
      }
      if (x < w) p[x] = lut_even[p[x]];
    }
  }
#endif

  static constexpr double kBlackLevel = 16.0;
  static constexpr double kSmooth = 0.05;
  static constexpr int kEstStep = 8;
  static constexpr double kGainMin = 0.6;
  static constexpr double kGainMax = 2.6;
  uint32_t frame_idx_ = 0;
  double b_gain_ = -1.0;
  double r_gain_ = -1.0;
  double lut_bg_ = -1.0;
  double lut_rg_ = -1.0;
  bool built_ = false;
  uchar lut_b_[256], lut_g_[256], lut_r_[256];
  std::vector<float> ratios_bg_, ratios_rg_;
};

// False color suppression (the standard ISP YUV-domain block; we have no
// optical low-pass filter, so detail near the CFA Nyquist limit aliases
// into chroma during demosaic and can only be detected + desaturated, never
// truly recovered). Two complementary mechanisms, each validated against a
// failure mode the other cannot handle:
//
// 1. Texture-gated desaturation, for color moire on dense fine texture
//    (fan grilles, distant building windows). The artifact appears as
//    LOW-frequency red/blue bands (beat between texture period and pixel
//    grid), so pulling chroma toward an 11px local average fails -- the
//    local average IS the moire color (measured: only 14% of the artifact
//    amplitude removable that way). The correct target is neutral gray,
//    and the correct gate is band-pass luma texture energy rather than
//    gradient magnitude: dense grille/window texture measures ~7-13 while
//    smooth-but-noisy dim skin measures ~1-3, so pulling texture-gated
//    pixels to neutral cannot wash out skin tone the way a gradient gate
//    did (noise edges fire Sobel, but carry almost no band-pass energy).
//    Cost: real color on dense fine texture (fabric weave) is partially
//    muted; every hardware ISP's false-color-suppression makes the same
//    trade.
//
// 2. Shadow desaturation, for chroma blotches on near-black regions
//    (motion-blurred fan blades at night: texture energy ~0.5, so gate 1
//    never fires there). At those signal levels chroma is demosaic/noise
//    artifact amplified asymmetrically by WB gains, not real color, so
//    scale chroma toward neutral as (blurred) luma falls below kShadowHi,
//    fully neutral at kShadowLo. Skin sits at Y>=85 in our dim captures,
//    above this range.
//
// A final 3x3 chroma median mops up isolated speckle (hot pixels, demosaic
// outliers) in flat regions where neither gate fires. (Removal was tested:
// residual chroma HF energy rises 15-24% on every validation scene and
// isolated-speckle p99.9 jumps 3 -> 12 DN on the night scenes, so its
// ~0.5 ms stays.)
//
// 3. Luma + chroma noise reduction (the pipeline otherwise has NO denoise
//    stage anywhere -- neither RAW NR nor YUV NR -- so night captures at
//    high sensor gain show plain grain on flat walls). Both planes go
//    through a fast self-guided filter (He et al., box filters on a
//    subsampled stats grid, O(N)): smooths variation whose local variance
//    is well below eps and preserves higher-contrast structure. Luma gets
//    a light touch (grain halves on walls, edges kept); chroma tolerates a
//    stronger setting since real chroma varies slowly. Residual LOW-
//    frequency color mottling under very high gain cannot be fully removed
//    spatially without bleeding real color -- that would need temporal NR
//    or RAW-domain NR upstream.
//
// Performance: the requirement is <= 15 ms/frame on x86 with FOUR cameras
// processed in parallel threads. History on i7-7700HQ, 1280x1024, single
// thread: naive full-res implementation ~105 ms; restructured (half-res
// chroma, preallocated buffers) ~23 ms. On i7-7700HQ the latest live,
// same-binary A/B is post 9.54 ms (SSE) -> 9.00 ms (AVX2), with total ISP
// ~10.7-11.3 ms in the AVX2 run; 29.98 fps and zero image/IMU drops.
// Desktop load and missing SCHED_FIFO capability can add ~1 ms tail jitter,
// so compare 300-frame averages rather than individual frames. RK3588 uses
// the bit-exact NEON backend and, when configured with
// -DTARGET_BOARD=rk3588, pins the four workers to A76 cores CPU4-7.
// What changed relative to the 23 ms version, with quality re-validated on
// the same captures each step:
//
//  a. The full-res YCrCb round-trip is gone (was ~5 ms of cvtColor/split/
//     merge and 27 MB of traffic). Luma comes from BGR2GRAY (same BT.601
//     weights as YCrCb's Y), chroma from ONE half-res BGR->YCrCb.
//  b. Luma guided NR keeps the same math (self-guided, eps=20, ~12px
//     support) but runs its stats on a 1/4 grid and emits the per-pixel
//     linear coefficients a,b quantized to 12-bit fixed point on the
//     half grid; the final y' = a*y + b runs inside the fused output
//     loop. 12-bit keeps quantization error < 0.1 DN (8-bit coeffs showed
//     ~1 DN blocky error on flat walls).
//  c. Chroma NR switched from guided filter to gauss5 + median3 at half
//     res: chroma is heavily low-passed anyway and the measured output
//     difference vs the guided version was below the gate quantization
//     step, at ~1/4 the cost.
//  d. All gate arithmetic collapsed into one u8 fixed-point loop at
//     quarter res; the texture band-pass itself stays FULL resolution
//     (half/quarter-res band-pass reads ~45% lower energy and separates
//     grille from skin much worse); only the energy MAP is downsampled.
//  e. One fused SIMD output loop does luma MAC + chroma NN-upsample +
//     YCrCb->BGR and writes final BGR directly, replacing luma convertTo +
//     2x chroma resize + merge + full-res cvtColor. It dispatches to AVX2
//     on capable x86 (output kernel 2.738 -> 2.270 ms), explicit NEON on
//     ARM/RK3588, and the original universal-intrinsics fallback elsewhere.
//     Random full-frame verification: AVX2 and emulated A76/NEON each
//     differ from the scalar integer reference at 0 of 3,932,160 channels.
//  f. Three memory-traffic merges, each bit-identical or below the gate
//     quantization step: AbsDiffPool4 fuses the band-pass absdiff with its
//     4x4 pooling (0.75 -> 0.25 ms, no full-res hf plane); the guided
//     var/a/b arithmetic runs as one loop instead of six whole-plane cv::
//     ops (1.0 -> 0.55 ms); the chroma keep gate is read NN off the
//     quarter grid inside the SIMD loop, dropping its bilinear upsample.
//     Tested-and-REJECTED removals: the 3x3 chroma median (speckle p99.9
//     jumps 3 -> 12 DN on night scenes) and the gauss3 on the texture map
//     (no measurable speed gain; keeps the gate spatially stable).
//  g. ARM/RK3588 front-end: FusedFrontYCbCr420 produces y8 + raw half-res
//     Cr/Cb in one BGR pass, replacing BGR2GRAY + resize + BGR2YCrCb +
//     split, which are disproportionately slow in OpenCV 4.5.4's aarch64
//     kernels (5.67 -> 2.30 ms on an A76; x86 keeps the OpenCV path,
//     which is faster there). The WB LUT also gains a TBL/TBX NEON path
//     (1.08 -> 0.98 ms), bit-exact vs the scalar LUT loop.
//
// Metrics on the validation frames (night fan / dusk skin / day balcony),
// current SDK -> this version: fan-grille false color and dark-region
// blotches both remain fully suppressed (residual chroma energy 2.83 ->
// 2.92, still far below the 4.19 unsuppressed level); flat-wall noise
// sigma 0.409 -> 0.415 (unfiltered: 0.731); edge sharpness 247.5 -> 246.8
// (unfiltered: 257.5); skin tone preserved (chroma 5.58 -> 5.81 vs 6.65
// unfiltered -- slightly MORE color kept than before).

// One-pass |a-b| with 4x4 average pooling to the quarter grid. Replaces
// cv::absdiff (full-res write) + cv::resize INTER_AREA (full-res re-read):
// same output within rounding, one read of each input, no full-res
// intermediate (measured 0.75 -> 0.25 ms).
inline void AbsDiffPool4(const cv::Mat &y8, const cv::Mat &y_bl,
                         cv::Mat &out_q, cv::Mat &colsum) {
  const int W = y8.cols, qW = W / 4, qH = y8.rows / 4;
  out_q.create(qH, qW, CV_8U);
  colsum.create(1, W, CV_16U);
  ushort *cs = colsum.ptr<ushort>(0);
  for (int yq = 0; yq < qH; ++yq) {
    std::memset(cs, 0, W * sizeof(ushort));
    for (int r = 0; r < 4; ++r) {
      const uchar *pa = y8.ptr<uchar>(4 * yq + r);
      const uchar *pb = y_bl.ptr<uchar>(4 * yq + r);
      int x = 0;
#if CV_SIMD128
      for (; x + 16 <= W; x += 16) {
        cv::v_uint8x16 d =
            cv::v_absdiff(cv::v_load(pa + x), cv::v_load(pb + x));
        cv::v_uint16x8 d_lo, d_hi;
        cv::v_expand(d, d_lo, d_hi);
        cv::v_store(cs + x, cv::v_load(cs + x) + d_lo);
        cv::v_store(cs + x + 8, cv::v_load(cs + x + 8) + d_hi);
      }
#endif
      for (; x < W; ++x)
        cs[x] = static_cast<ushort>(cs[x] + std::abs(pa[x] - pb[x]));
    }
    uchar *po = out_q.ptr<uchar>(yq);
    for (int xq = 0; xq < qW; ++xq) {
      const int s =
          cs[4 * xq] + cs[4 * xq + 1] + cs[4 * xq + 2] + cs[4 * xq + 3];
      po[xq] = static_cast<uchar>((s + 8) >> 4);
    }
  }
}

// Fused front-end: ONE pass over the demosaiced BGR produces the full-res
// luma plane AND the half-res Cr/Cb planes (4:2:0, 2x2 box like INTER_AREA,
// chroma evaluated in the 2x2-SUM domain so there is no intermediate
// rounding). Replaces cvtColor(BGR2GRAY) + resize(INTER_AREA) +
// cvtColor(BGR2YCrCb) + split, i.e. four whole-plane walks of the 3.9 MB
// BGR image collapse into one.
//
// Same BT.601 arithmetic as OpenCV within +-1 DN (Q8 luma weights
// 77/150/29 vs OpenCV's Q14; chroma 183/1024 and 144/1024 in the 4x sum
// domain equal 0.7148/0.5625 vs 0.713/0.564); verified on real frames.
// Dispatch is per-architecture and runtime-switchable:
//  - ARM/RK3588: ON by default. OpenCV 4.5.4's aarch64 kernels for this
//    front measured 5.67 ms on an A76 (gray 2.20 + prep 3.34) vs 2.30 ms
//    fused -- a 3.4 ms/frame saving. Opt out: CYPERSTEREO_DISABLE_FUSED_FRONT.
//  - x86: OFF by default. The IPP/AVX2 cvtColor+resize path is faster than
//    this universal-intrinsics loop there (1.65 vs 2.83 ms measured);
//    opt in with CYPERSTEREO_ENABLE_FUSED_FRONT for quality A/B on desktop.
inline bool UseFusedFront() {
#if defined(CYPERSTEREO_HAVE_NEON)
  static const bool enabled =
      std::getenv("CYPERSTEREO_DISABLE_FUSED_FRONT") == nullptr;
#else
  static const bool enabled =
      std::getenv("CYPERSTEREO_ENABLE_FUSED_FRONT") != nullptr;
#endif
  return enabled;
}

#if defined(CYPERSTEREO_HAVE_NEON)
// Explicit-NEON body of the fused front-end. Bit-exact vs the universal-
// intrinsics loop below (verified full-frame on the board: 0 differing
// pixels in Y/Cr/Cb), but 2.23 -> 1.44 ms on an A76: vmlal_u8 does the
// 8-bit luma MAC without the u8->u16 expand step, and vpaddlq_u8 does the
// horizontal 2x2-sum halves in one instruction.
inline void FusedFrontYCbCr420Neon(const cv::Mat &color, cv::Mat &y8,
                                   cv::Mat &cr_h, cv::Mat &cb_h) {
  const int width = color.cols;
  const int height = color.rows;
  const int half_width = width / 2;
  const int half_height = height / 2;
  y8.create(height, width, CV_8U);
  cr_h.create(half_height, half_width, CV_8U);
  cb_h.create(half_height, half_width, CV_8U);
  const uint8x8_t k29 = vdup_n_u8(29), k150 = vdup_n_u8(150),
                  k77 = vdup_n_u8(77);
  const uint16x8_t r128 = vdupq_n_u16(128);
  for (int yh = 0; yh < half_height; ++yh) {
    const uchar *row0 = color.ptr<uchar>(2 * yh);
    const uchar *row1 = color.ptr<uchar>(2 * yh + 1);
    uchar *yrow0 = y8.ptr<uchar>(2 * yh);
    uchar *yrow1 = y8.ptr<uchar>(2 * yh + 1);
    uchar *pcr = cr_h.ptr<uchar>(yh);
    uchar *pcb = cb_h.ptr<uchar>(yh);
    int xh = 0;
    for (; xh + 8 <= half_width; xh += 8) {
      const int x0 = xh << 1;
      const uint8x16x3_t v0 = vld3q_u8(row0 + 3 * x0);
      const uint8x16x3_t v1 = vld3q_u8(row1 + 3 * x0);
      const auto gray16 = [&](const uint8x16x3_t &v) {
        const uint16x8_t lo = vmlal_u8(
            vmlal_u8(vmlal_u8(r128, vget_low_u8(v.val[0]), k29),
                     vget_low_u8(v.val[1]), k150),
            vget_low_u8(v.val[2]), k77);
        const uint16x8_t hi = vmlal_u8(
            vmlal_u8(vmlal_u8(r128, vget_high_u8(v.val[0]), k29),
                     vget_high_u8(v.val[1]), k150),
            vget_high_u8(v.val[2]), k77);
        return vcombine_u8(vshrn_n_u16(lo, 8), vshrn_n_u16(hi, 8));
      };
      vst1q_u8(yrow0 + x0, gray16(v0));
      vst1q_u8(yrow1 + x0, gray16(v1));

      const uint16x8_t bsum =
          vaddq_u16(vpaddlq_u8(v0.val[0]), vpaddlq_u8(v1.val[0]));
      const uint16x8_t gsum =
          vaddq_u16(vpaddlq_u8(v0.val[1]), vpaddlq_u8(v1.val[1]));
      const uint16x8_t rsum =
          vaddq_u16(vpaddlq_u8(v0.val[2]), vpaddlq_u8(v1.val[2]));
      const auto ymac = [&](const uint16x4_t b, const uint16x4_t g,
                            const uint16x4_t r) {
        uint32x4_t s = vmull_n_u16(b, 29);
        s = vmlal_n_u16(s, g, 150);
        s = vmlal_n_u16(s, r, 77);
        return vshrq_n_u32(vaddq_u32(s, vdupq_n_u32(128)), 8);
      };
      const uint32x4_t ysum_lo =
          ymac(vget_low_u16(bsum), vget_low_u16(gsum), vget_low_u16(rsum));
      const uint32x4_t ysum_hi = ymac(vget_high_u16(bsum),
                                      vget_high_u16(gsum),
                                      vget_high_u16(rsum));
      const int32x4_t dcr_lo = vsubq_s32(
          vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(rsum))),
          vreinterpretq_s32_u32(ysum_lo));
      const int32x4_t dcr_hi = vsubq_s32(
          vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(rsum))),
          vreinterpretq_s32_u32(ysum_hi));
      const int32x4_t dcb_lo = vsubq_s32(
          vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(bsum))),
          vreinterpretq_s32_u32(ysum_lo));
      const int32x4_t dcb_hi = vsubq_s32(
          vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(bsum))),
          vreinterpretq_s32_u32(ysum_hi));
      const int32x4_t c128 = vdupq_n_s32(128);
      const int32x4_t rnd = vdupq_n_s32(512);
      const auto chroma = [&](const int32x4_t lo, const int32x4_t hi,
                              int coefficient) {
        const int32x4_t clo = vaddq_s32(
            c128,
            vshrq_n_s32(vaddq_s32(vmulq_n_s32(lo, coefficient), rnd), 10));
        const int32x4_t chi = vaddq_s32(
            c128,
            vshrq_n_s32(vaddq_s32(vmulq_n_s32(hi, coefficient), rnd), 10));
        return vqmovun_s16(vcombine_s16(vqmovn_s32(clo), vqmovn_s32(chi)));
      };
      vst1_u8(pcr + xh, chroma(dcr_lo, dcr_hi, 183));
      vst1_u8(pcb + xh, chroma(dcb_lo, dcb_hi, 144));
    }
    for (; xh < half_width; ++xh) {
      const int x0 = xh << 1;
      int bsum = 0, gsum = 0, rsum = 0;
      for (int row = 0; row < 2; ++row) {
        const uchar *src = row ? row1 : row0;
        uchar *ydst = row ? yrow1 : yrow0;
        for (int dx = 0; dx < 2; ++dx) {
          const int x = x0 + dx;
          const int b = src[3 * x];
          const int g = src[3 * x + 1];
          const int r = src[3 * x + 2];
          ydst[x] =
              static_cast<uchar>((29 * b + 150 * g + 77 * r + 128) >> 8);
          bsum += b;
          gsum += g;
          rsum += r;
        }
      }
      const int ysum = (29 * bsum + 150 * gsum + 77 * rsum + 128) >> 8;
      pcr[xh] = cv::saturate_cast<uchar>(
          128 + (((rsum - ysum) * 183 + 512) >> 10));
      pcb[xh] = cv::saturate_cast<uchar>(
          128 + (((bsum - ysum) * 144 + 512) >> 10));
    }
  }
}
#endif

inline void FusedFrontYCbCr420(const cv::Mat &color, cv::Mat &y8,
                               cv::Mat &cr_h, cv::Mat &cb_h) {
#if defined(CYPERSTEREO_HAVE_NEON)
  // ARM always takes the explicit kernel (bit-exact, 2.23 -> 1.44 ms).
  FusedFrontYCbCr420Neon(color, y8, cr_h, cb_h);
#else
  const int width = color.cols;
  const int height = color.rows;
  const int half_width = width / 2;
  const int half_height = height / 2;
  y8.create(height, width, CV_8U);
  cr_h.create(half_height, half_width, CV_8U);
  cb_h.create(half_height, half_width, CV_8U);

  for (int yh = 0; yh < half_height; ++yh) {
    const uchar *row0 = color.ptr<uchar>(2 * yh);
    const uchar *row1 = color.ptr<uchar>(2 * yh + 1);
    uchar *yrow0 = y8.ptr<uchar>(2 * yh);
    uchar *yrow1 = y8.ptr<uchar>(2 * yh + 1);
    uchar *pcr = cr_h.ptr<uchar>(yh);
    uchar *pcb = cb_h.ptr<uchar>(yh);
    int xh = 0;
#if CV_SIMD128
    const cv::v_uint16x8 k29 = cv::v_setall_u16(29);
    const cv::v_uint16x8 k150 = cv::v_setall_u16(150);
    const cv::v_uint16x8 k77 = cv::v_setall_u16(77);
    const cv::v_uint16x8 round128 = cv::v_setall_u16(128);
    const cv::v_uint32x4 mask16 = cv::v_setall_u32(0xffff);
    for (; xh + 8 <= half_width; xh += 8) {
      const int x0 = xh << 1;
      cv::v_uint8x16 b0, g0, r0, b1, g1, r1;
      cv::v_load_deinterleave(row0 + 3 * x0, b0, g0, r0);
      cv::v_load_deinterleave(row1 + 3 * x0, b1, g1, r1);
      cv::v_uint16x8 b0l, b0h, g0l, g0h, r0l, r0h;
      cv::v_uint16x8 b1l, b1h, g1l, g1h, r1l, r1h;
      cv::v_expand(b0, b0l, b0h);
      cv::v_expand(g0, g0l, g0h);
      cv::v_expand(r0, r0l, r0h);
      cv::v_expand(b1, b1l, b1h);
      cv::v_expand(g1, g1l, g1h);
      cv::v_expand(r1, r1l, r1h);

      // Full-res luma for both rows. Weights sum to 256 so u16 never
      // overflows: max 255*256 + 128 < 65536.
      const auto gray8 = [&](const cv::v_uint16x8 &bb,
                             const cv::v_uint16x8 &gg,
                             const cv::v_uint16x8 &rr) {
        return (bb * k29 + gg * k150 + rr * k77 + round128) >> 8;
      };
      cv::v_store(yrow0 + x0,
                  cv::v_pack(gray8(b0l, g0l, r0l), gray8(b0h, g0h, r0h)));
      cv::v_store(yrow1 + x0,
                  cv::v_pack(gray8(b1l, g1l, r1l), gray8(b1h, g1h, r1h)));

      // 2x2 sums per channel: vertical add (u16, max 510), then horizontal
      // pair-add through a u32 view (max 1020).
      const auto horizontal_pairs = [&](const cv::v_uint16x8 &v) {
        const cv::v_uint32x4 words = cv::v_reinterpret_as_u32(v);
        return (words & mask16) + (words >> 16);
      };
      const cv::v_uint32x4 bsum0 = horizontal_pairs(b0l + b1l);
      const cv::v_uint32x4 bsum1 = horizontal_pairs(b0h + b1h);
      const cv::v_uint32x4 gsum0 = horizontal_pairs(g0l + g1l);
      const cv::v_uint32x4 gsum1 = horizontal_pairs(g0h + g1h);
      const cv::v_uint32x4 rsum0 = horizontal_pairs(r0l + r1l);
      const cv::v_uint32x4 rsum1 = horizontal_pairs(r0h + r1h);
      const cv::v_uint32x4 ysum0 =
          (bsum0 * cv::v_setall_u32(29) + gsum0 * cv::v_setall_u32(150) +
           rsum0 * cv::v_setall_u32(77) + cv::v_setall_u32(128)) >> 8;
      const cv::v_uint32x4 ysum1 =
          (bsum1 * cv::v_setall_u32(29) + gsum1 * cv::v_setall_u32(150) +
           rsum1 * cv::v_setall_u32(77) + cv::v_setall_u32(128)) >> 8;
      const cv::v_int32x4 dcr0 =
          cv::v_reinterpret_as_s32(rsum0) - cv::v_reinterpret_as_s32(ysum0);
      const cv::v_int32x4 dcr1 =
          cv::v_reinterpret_as_s32(rsum1) - cv::v_reinterpret_as_s32(ysum1);
      const cv::v_int32x4 dcb0 =
          cv::v_reinterpret_as_s32(bsum0) - cv::v_reinterpret_as_s32(ysum0);
      const cv::v_int32x4 dcb1 =
          cv::v_reinterpret_as_s32(bsum1) - cv::v_reinterpret_as_s32(ysum1);
      const cv::v_int32x4 center = cv::v_setall_s32(128);
      const cv::v_int32x4 round = cv::v_setall_s32(512);
      const cv::v_int16x8 cr16 = cv::v_pack(
          center + ((dcr0 * cv::v_setall_s32(183) + round) >> 10),
          center + ((dcr1 * cv::v_setall_s32(183) + round) >> 10));
      const cv::v_int16x8 cb16 = cv::v_pack(
          center + ((dcb0 * cv::v_setall_s32(144) + round) >> 10),
          center + ((dcb1 * cv::v_setall_s32(144) + round) >> 10));
      cv::v_pack_u_store(pcr + xh, cr16);
      cv::v_pack_u_store(pcb + xh, cb16);
    }
#endif
    for (; xh < half_width; ++xh) {
      const int x0 = xh << 1;
      int bsum = 0, gsum = 0, rsum = 0;
      for (int row = 0; row < 2; ++row) {
        const uchar *src = row ? row1 : row0;
        uchar *ydst = row ? yrow1 : yrow0;
        for (int dx = 0; dx < 2; ++dx) {
          const int x = x0 + dx;
          const int b = src[3 * x];
          const int g = src[3 * x + 1];
          const int r = src[3 * x + 2];
          ydst[x] =
              static_cast<uchar>((29 * b + 150 * g + 77 * r + 128) >> 8);
          bsum += b;
          gsum += g;
          rsum += r;
        }
      }
      const int ysum = (29 * bsum + 150 * gsum + 77 * rsum + 128) >> 8;
      pcr[xh] = cv::saturate_cast<uchar>(
          128 + (((rsum - ysum) * 183 + 512) >> 10));
      pcb[xh] = cv::saturate_cast<uchar>(
          128 + (((bsum - ysum) * 144 + 512) >> 10));
    }
  }
#endif  // !CYPERSTEREO_HAVE_NEON
}

// ---------------------------------------------------------------------------
// Fused EA demosaic + front-end (ARM). The interleaved BGR frame produced by
// cv::cvtColor(COLOR_BayerRG2BGR_EA) has exactly ONE consumer -- the fused
// front above (the output kernel rebuilds BGR from processed Y/Cr/Cb). This
// kernel replicates OpenCV's edge-aware interpolation bit-exactly (verified
// full-frame: 0 differing pixels in Y/Cr/Cb vs cvtColor+front), keeps B/G/R
// planar in registers, and emits y8 + half-res Cr/Cb straight from the
// white-balanced Bayer plane. Board: 2.29 (cv EA) + 1.44 (front) -> 1.67 ms.
//
// EA formulas (OpenCV demosaicing.cpp, Bayer2RGB_EdgeAware_T_Invoker; our
// BayerRG order = even rows B G, odd rows G R):
//   G at R/B sites : |left-right| > |down-up| ? avg2(up,down) : avg2(l,r)
//   R/B at G sites : avg2 of the two same-color axis neighbours
//   R at B / B at R: avg4 of the four diagonal neighbours
//   avg2 = (a+b+1)>>1, avg4 = (sum+2)>>2
// Borders: interior is rows/cols 1..N-2, then OpenCV replicates the
// demosaiced col1 -> col0, colW-2 -> colW-1, row1 -> row0, rowH-2 -> rowH-1;
// the scalar edge handling below folds that replication in directly.
inline bool UseFusedDemosaic() {
#if !defined(CYPERSTEREO_HAVE_NEON)
  return false;
#else
  static const bool enabled = [] {
    const char *disable = std::getenv("CYPERSTEREO_DISABLE_FUSED_DEMOSAIC");
    return !(disable && disable[0] == '1');
  }();
  return enabled;
#endif
}

#if defined(CYPERSTEREO_HAVE_NEON)
// Scalar EA demosaic of one interior pixel (row/col 1..N-2).
inline void EaDemosaicPixel(const cv::Mat &raw, int r, int x, int &B, int &G,
                            int &R) {
  const uchar *rm = raw.ptr<uchar>(r - 1);
  const uchar *rc = raw.ptr<uchar>(r);
  const uchar *rp = raw.ptr<uchar>(r + 1);
  const bool row_gr = (r & 1) != 0;  // odd rows: G R
  const bool odd = (x & 1) != 0;
  const auto ea_green = [&]() {
    const int h = std::abs(rc[x - 1] - rc[x + 1]);
    const int v = std::abs(rp[x] - rm[x]);
    return (h > v ? rp[x] + rm[x] + 1 : rc[x - 1] + rc[x + 1] + 1) >> 1;
  };
  if (row_gr) {
    if (odd) {  // R site
      R = rc[x];
      G = ea_green();
      B = (rm[x - 1] + rm[x + 1] + rp[x - 1] + rp[x + 1] + 2) >> 2;
    } else {  // G site on an R row: R horizontal, B vertical
      G = rc[x];
      R = (rc[x - 1] + rc[x + 1] + 1) >> 1;
      B = (rm[x] + rp[x] + 1) >> 1;
    }
  } else {
    if (!odd) {  // B site
      B = rc[x];
      G = ea_green();
      R = (rm[x - 1] + rm[x + 1] + rp[x - 1] + rp[x + 1] + 2) >> 2;
    } else {  // G site on a B row: B horizontal, R vertical
      G = rc[x];
      B = (rc[x - 1] + rc[x + 1] + 1) >> 1;
      R = (rm[x] + rp[x] + 1) >> 1;
    }
  }
}

// Planar demosaiced row pair for 32 columns starting at even column x0:
// lane j of *e covers column x0+2j, lane j of *o covers x0+2j+1.
struct EaPlanar {
  uint8x16_t Be, Bo, Ge, Go, Re, Ro;
};

inline uint8x16_t EaGreenNeon(uint8x16_t hl, uint8x16_t hr, uint8x16_t vu,
                              uint8x16_t vd) {
  const uint8x16_t gh = vabdq_u8(hl, hr);
  const uint8x16_t gv = vabdq_u8(vd, vu);
  const uint8x16_t havg = vrhaddq_u8(hl, hr);
  const uint8x16_t vavg = vrhaddq_u8(vu, vd);
  return vbslq_u8(vcgtq_u8(gh, gv), vavg, havg);
}

inline uint8x16_t EaDiag4Neon(uint8x16_t a, uint8x16_t b, uint8x16_t c,
                              uint8x16_t d) {
  const uint16x8_t lo = vaddq_u16(vaddl_u8(vget_low_u8(a), vget_low_u8(b)),
                                  vaddl_u8(vget_low_u8(c), vget_low_u8(d)));
  const uint16x8_t hi =
      vaddq_u16(vaddl_u8(vget_high_u8(a), vget_high_u8(b)),
                vaddl_u8(vget_high_u8(c), vget_high_u8(d)));
  return vcombine_u8(vrshrn_n_u16(lo, 2), vrshrn_n_u16(hi, 2));
}

// Even ("B G") row: B at even cols. rm/rp are the G-R rows above/below.
inline EaPlanar EaRowBG(const uchar *rm, const uchar *rc, const uchar *rp,
                        int x0) {
  const uint8x16x2_t cL = vld2q_u8(rc + x0 - 2);
  const uint8x16x2_t cC = vld2q_u8(rc + x0);
  const uint8x16x2_t cR = vld2q_u8(rc + x0 + 2);
  const uint8x16x2_t mL = vld2q_u8(rm + x0 - 2);
  const uint8x16x2_t mC = vld2q_u8(rm + x0);
  const uint8x16x2_t pL = vld2q_u8(rp + x0 - 2);
  const uint8x16x2_t pC = vld2q_u8(rp + x0);
  EaPlanar o;
  o.Be = cC.val[0];
  o.Ge = EaGreenNeon(cL.val[1], cC.val[1], mC.val[0], pC.val[0]);
  o.Re = EaDiag4Neon(mL.val[1], mC.val[1], pL.val[1], pC.val[1]);
  o.Go = cC.val[1];
  o.Bo = vrhaddq_u8(cC.val[0], cR.val[0]);
  o.Ro = vrhaddq_u8(mC.val[1], pC.val[1]);
  return o;
}

// Odd ("G R") row: R at odd cols. rm/rp are the B-G rows above/below.
inline EaPlanar EaRowGR(const uchar *rm, const uchar *rc, const uchar *rp,
                        int x0) {
  const uint8x16x2_t cL = vld2q_u8(rc + x0 - 2);
  const uint8x16x2_t cC = vld2q_u8(rc + x0);
  const uint8x16x2_t cR = vld2q_u8(rc + x0 + 2);
  const uint8x16x2_t mC = vld2q_u8(rm + x0);
  const uint8x16x2_t mR = vld2q_u8(rm + x0 + 2);
  const uint8x16x2_t pC = vld2q_u8(rp + x0);
  const uint8x16x2_t pR = vld2q_u8(rp + x0 + 2);
  EaPlanar o;
  o.Ro = cC.val[1];
  o.Go = EaGreenNeon(cC.val[0], cR.val[0], mC.val[1], pC.val[1]);
  o.Bo = EaDiag4Neon(mC.val[0], mR.val[0], pC.val[0], pR.val[0]);
  o.Ge = cC.val[0];
  o.Re = vrhaddq_u8(cL.val[1], cC.val[1]);
  o.Be = vrhaddq_u8(mC.val[0], pC.val[0]);
  return o;
}

inline uint8x16_t EaGrayNeon(const uint8x16_t b, const uint8x16_t g,
                             const uint8x16_t r) {
  const uint8x8_t k29 = vdup_n_u8(29), k150 = vdup_n_u8(150),
                  k77 = vdup_n_u8(77);
  const uint16x8_t r128 = vdupq_n_u16(128);
  const uint16x8_t lo = vmlal_u8(
      vmlal_u8(vmlal_u8(r128, vget_low_u8(b), k29), vget_low_u8(g), k150),
      vget_low_u8(r), k77);
  const uint16x8_t hi = vmlal_u8(
      vmlal_u8(vmlal_u8(r128, vget_high_u8(b), k29), vget_high_u8(g), k150),
      vget_high_u8(r), k77);
  return vcombine_u8(vshrn_n_u16(lo, 8), vshrn_n_u16(hi, 8));
}

// 2x2 block sums of one plane over a row pair (u16, lo/hi halves).
inline void EaSum4(const uint8x16_t e0, const uint8x16_t o0,
                   const uint8x16_t e1, const uint8x16_t o1, uint16x8_t &lo,
                   uint16x8_t &hi) {
  lo = vaddq_u16(vaddl_u8(vget_low_u8(e0), vget_low_u8(o0)),
                 vaddl_u8(vget_low_u8(e1), vget_low_u8(o1)));
  hi = vaddq_u16(vaddl_u8(vget_high_u8(e0), vget_high_u8(o0)),
                 vaddl_u8(vget_high_u8(e1), vget_high_u8(o1)));
}

inline uint32x4_t EaYMac(const uint16x4_t b, const uint16x4_t g,
                         const uint16x4_t r) {
  uint32x4_t s = vmull_n_u16(b, 29);
  s = vmlal_n_u16(s, g, 150);
  s = vmlal_n_u16(s, r, 77);
  return vshrq_n_u32(vaddq_u32(s, vdupq_n_u32(128)), 8);
}

// Chroma from 2x2 sums -- identical math to FusedFrontYCbCr420Neon.
inline uint8x8_t EaChroma8(const uint16x8_t csum, const uint32x4_t ys_lo,
                           const uint32x4_t ys_hi, int coefficient) {
  const int32x4_t c128 = vdupq_n_s32(128);
  const int32x4_t rnd = vdupq_n_s32(512);
  const int32x4_t d_lo =
      vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(csum))),
                vreinterpretq_s32_u32(ys_lo));
  const int32x4_t d_hi =
      vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(csum))),
                vreinterpretq_s32_u32(ys_hi));
  const int32x4_t lo = vaddq_s32(
      c128, vshrq_n_s32(vaddq_s32(vmulq_n_s32(d_lo, coefficient), rnd), 10));
  const int32x4_t hi = vaddq_s32(
      c128, vshrq_n_s32(vaddq_s32(vmulq_n_s32(d_hi, coefficient), rnd), 10));
  return vqmovun_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)));
}

inline void FusedDemosaicFrontNeon(const cv::Mat &raw, cv::Mat &y8,
                                   cv::Mat &cr_h, cv::Mat &cb_h) {
  const int W = raw.cols, H = raw.rows, hw = W / 2, hh = H / 2;
  y8.create(H, W, CV_8U);
  cr_h.create(hh, hw, CV_8U);
  cb_h.create(hh, hw, CV_8U);

  // Scalar fallback for the first/last half-column, with the output-border
  // replication (col0 := col1, colW-1 := colW-2, row clamps) baked in.
  const auto pix = [&](int r, int x, int &B, int &G, int &R) {
    const int rr = r < 1 ? 1 : (r > H - 2 ? H - 2 : r);
    const int xx = x < 1 ? 1 : (x > W - 2 ? W - 2 : x);
    EaDemosaicPixel(raw, rr, xx, B, G, R);
  };
  const auto scalar_cols = [&](int yh, int xh_lo, int xh_hi) {
    uchar *y0 = y8.ptr<uchar>(2 * yh);
    uchar *y1 = y8.ptr<uchar>(2 * yh + 1);
    uchar *pcr = cr_h.ptr<uchar>(yh);
    uchar *pcb = cb_h.ptr<uchar>(yh);
    for (int xh = xh_lo; xh < xh_hi; ++xh) {
      int bsum = 0, gsum = 0, rsum = 0;
      for (int row = 0; row < 2; ++row) {
        uchar *yd = row ? y1 : y0;
        for (int dx = 0; dx < 2; ++dx) {
          const int x = 2 * xh + dx;
          int B, G, R;
          pix(2 * yh + row, x, B, G, R);
          yd[x] = static_cast<uchar>((29 * B + 150 * G + 77 * R + 128) >> 8);
          bsum += B;
          gsum += G;
          rsum += R;
        }
      }
      const int ysum = (29 * bsum + 150 * gsum + 77 * rsum + 128) >> 8;
      pcr[xh] = cv::saturate_cast<uchar>(
          128 + (((rsum - ysum) * 183 + 512) >> 10));
      pcb[xh] = cv::saturate_cast<uchar>(
          128 + (((bsum - ysum) * 144 + 512) >> 10));
    }
  };

  const int x0_last = W - 34;  // last (overlapping) 32-column chunk
  for (int yh = 0; yh < hh; ++yh) {
    const int r0 = 2 * yh;      // even row (B G); row0 replicates row1
    const int r1 = 2 * yh + 1;  // odd row (G R); rowH-1 replicates rowH-2
    const bool rep0 = r0 == 0, rep1 = r1 == H - 1;
    uchar *yrow0 = y8.ptr<uchar>(r0);
    uchar *yrow1 = y8.ptr<uchar>(r1);
    uchar *pcr = cr_h.ptr<uchar>(yh);
    uchar *pcb = cb_h.ptr<uchar>(yh);
    int x0 = 2;
    while (true) {
      EaPlanar p0, p1;
      if (rep0) {
        p1 = EaRowGR(raw.ptr<uchar>(r1 - 1), raw.ptr<uchar>(r1),
                     raw.ptr<uchar>(r1 + 1), x0);
        p0 = p1;
      } else if (rep1) {
        p0 = EaRowBG(raw.ptr<uchar>(r0 - 1), raw.ptr<uchar>(r0),
                     raw.ptr<uchar>(r0 + 1), x0);
        p1 = p0;
      } else {
        p0 = EaRowBG(raw.ptr<uchar>(r0 - 1), raw.ptr<uchar>(r0),
                     raw.ptr<uchar>(r0 + 1), x0);
        p1 = EaRowGR(raw.ptr<uchar>(r1 - 1), raw.ptr<uchar>(r1),
                     raw.ptr<uchar>(r1 + 1), x0);
      }
      uint8x16x2_t yst;
      yst.val[0] = EaGrayNeon(p0.Be, p0.Ge, p0.Re);
      yst.val[1] = EaGrayNeon(p0.Bo, p0.Go, p0.Ro);
      vst2q_u8(yrow0 + x0, yst);
      yst.val[0] = EaGrayNeon(p1.Be, p1.Ge, p1.Re);
      yst.val[1] = EaGrayNeon(p1.Bo, p1.Go, p1.Ro);
      vst2q_u8(yrow1 + x0, yst);

      uint16x8_t bs_lo, bs_hi, gs_lo, gs_hi, rs_lo, rs_hi;
      EaSum4(p0.Be, p0.Bo, p1.Be, p1.Bo, bs_lo, bs_hi);
      EaSum4(p0.Ge, p0.Go, p1.Ge, p1.Go, gs_lo, gs_hi);
      EaSum4(p0.Re, p0.Ro, p1.Re, p1.Ro, rs_lo, rs_hi);
      const uint32x4_t ys0 = EaYMac(vget_low_u16(bs_lo), vget_low_u16(gs_lo),
                                    vget_low_u16(rs_lo));
      const uint32x4_t ys1 = EaYMac(vget_high_u16(bs_lo),
                                    vget_high_u16(gs_lo),
                                    vget_high_u16(rs_lo));
      const uint32x4_t ys2 = EaYMac(vget_low_u16(bs_hi), vget_low_u16(gs_hi),
                                    vget_low_u16(rs_hi));
      const uint32x4_t ys3 = EaYMac(vget_high_u16(bs_hi),
                                    vget_high_u16(gs_hi),
                                    vget_high_u16(rs_hi));
      const int xh = x0 >> 1;
      vst1q_u8(pcr + xh, vcombine_u8(EaChroma8(rs_lo, ys0, ys1, 183),
                                     EaChroma8(rs_hi, ys2, ys3, 183)));
      vst1q_u8(pcb + xh, vcombine_u8(EaChroma8(bs_lo, ys0, ys1, 144),
                                     EaChroma8(bs_hi, ys2, ys3, 144)));
      if (x0 == x0_last) break;
      x0 += 32;
      if (x0 > x0_last) x0 = x0_last;
    }
    scalar_cols(yh, 0, 1);
    scalar_cols(yh, (W - 2) / 2, hw);
  }
}
#endif  // CYPERSTEREO_HAVE_NEON

#if defined(CYPERSTEREO_HAVE_NEON)
inline bool UseNeonGauss5() {
  // Cached A/B switch, same convention as the other NEON kernels.
  static const bool enabled =
      std::getenv("CYPERSTEREO_DISABLE_NEON_GAUSS5") == nullptr;
  return enabled;
}

// Separable [1 4 6 4 1]/16 Gaussian, u8 -> u8, BORDER_REFLECT_101 --
// exactly cv::GaussianBlur(src, dst, Size(5,5), 0) for CV_8U (OpenCV's
// bit-exact fixed-point path; verified maxdiff 0 on the board). OpenCV
// 4.5.4's aarch64 row filter leaves ~30% on the table: measured on an A76
// 1.66 -> 1.17 ms full res, 0.84 -> 0.59 ms for the two half-res chroma
// planes. Horizontal pass emits u16 sums (max 255*16 = 4080), vertical
// pass accumulates u16*[1 4 6 4 1] in u32 and rounds once: (v + 128) >> 8.
// Horizontal pass for one row: u8 src -> u16 [1 4 6 4 1] sums with
// REFLECT_101 ends.
inline void Gauss5HRowNeon(const uchar *s, ushort *t, int W) {
  const auto at = [&](int x) {
    x = x < 0 ? -x : (x >= W ? 2 * W - 2 - x : x);  // reflect101
    return static_cast<int>(s[x]);
  };
  for (int x = 0; x < 2; ++x)
    t[x] = static_cast<ushort>(at(x - 2) + 4 * at(x - 1) + 6 * at(x) +
                               4 * at(x + 1) + at(x + 2));
  int x = 2;
  for (; x + 16 + 2 <= W; x += 16) {
    const uint8x16_t m2 = vld1q_u8(s + x - 2);
    const uint8x16_t m1 = vld1q_u8(s + x - 1);
    const uint8x16_t c0 = vld1q_u8(s + x);
    const uint8x16_t p1 = vld1q_u8(s + x + 1);
    const uint8x16_t p2 = vld1q_u8(s + x + 2);
    uint16x8_t lo = vaddl_u8(vget_low_u8(m2), vget_low_u8(p2));
    uint16x8_t hi = vaddl_u8(vget_high_u8(m2), vget_high_u8(p2));
    const uint16x8_t s1lo = vaddl_u8(vget_low_u8(m1), vget_low_u8(p1));
    const uint16x8_t s1hi = vaddl_u8(vget_high_u8(m1), vget_high_u8(p1));
    lo = vmlaq_n_u16(lo, s1lo, 4);
    hi = vmlaq_n_u16(hi, s1hi, 4);
    const uint8x8_t six = vdup_n_u8(6);
    lo = vmlal_u8(lo, vget_low_u8(c0), six);
    hi = vmlal_u8(hi, vget_high_u8(c0), six);
    vst1q_u16(t + x, lo);
    vst1q_u16(t + x + 8, hi);
  }
  for (; x < W - 2; ++x)
    t[x] = static_cast<ushort>(s[x - 2] + 4 * s[x - 1] + 6 * s[x] +
                               4 * s[x + 1] + s[x + 2]);
  for (int xb = std::max(2, W - 2); xb < W; ++xb)
    t[xb] = static_cast<ushort>(at(xb - 2) + 4 * at(xb - 1) + 6 * at(xb) +
                                4 * at(xb + 1) + at(xb + 2));
}

// Vertical pass for one output row from five h-sum rows.
inline void Gauss5VRowNeon(const ushort *r0, const ushort *r1,
                           const ushort *r2, const ushort *r3,
                           const ushort *r4, uchar *d, int W) {
  int x = 0;
  for (; x + 8 <= W; x += 8) {
    const uint16x8_t a0 = vld1q_u16(r0 + x);
    const uint16x8_t a1 = vld1q_u16(r1 + x);
    const uint16x8_t a2 = vld1q_u16(r2 + x);
    const uint16x8_t a3 = vld1q_u16(r3 + x);
    const uint16x8_t a4 = vld1q_u16(r4 + x);
    uint32x4_t lo = vaddl_u16(vget_low_u16(a0), vget_low_u16(a4));
    uint32x4_t hi = vaddl_u16(vget_high_u16(a0), vget_high_u16(a4));
    const uint32x4_t b1lo = vaddl_u16(vget_low_u16(a1), vget_low_u16(a3));
    const uint32x4_t b1hi = vaddl_u16(vget_high_u16(a1), vget_high_u16(a3));
    lo = vmlaq_n_u32(lo, b1lo, 4);
    hi = vmlaq_n_u32(hi, b1hi, 4);
    lo = vmlal_n_u16(lo, vget_low_u16(a2), 6);
    hi = vmlal_n_u16(hi, vget_high_u16(a2), 6);
    lo = vaddq_u32(lo, vdupq_n_u32(128));
    hi = vaddq_u32(hi, vdupq_n_u32(128));
    const uint16x4_t nlo = vshrn_n_u32(lo, 8);
    const uint16x4_t nhi = vshrn_n_u32(hi, 8);
    vst1_u8(d + x, vmovn_u16(vcombine_u16(nlo, nhi)));
  }
  for (; x < W; ++x) {
    const int v = r0[x] + 4 * r1[x] + 6 * r2[x] + 4 * r3[x] + r4[x];
    d[x] = static_cast<uchar>((v + 128) >> 8);
  }
}

inline void Gauss5NeonU8(const cv::Mat &src, cv::Mat &dst, cv::Mat &tmp16) {
  const int W = src.cols, H = src.rows;
  dst.create(H, W, CV_8U);
  tmp16.create(H, W, CV_16U);
  for (int y = 0; y < H; ++y)
    Gauss5HRowNeon(src.ptr<uchar>(y), tmp16.ptr<ushort>(y), W);
  for (int y = 0; y < H; ++y) {
    const int ym2 = y - 2 < 0 ? 2 - y : y - 2;
    const int ym1 = y - 1 < 0 ? 1 - y : y - 1;
    const int yp1 = y + 1 >= H ? 2 * H - 2 - (y + 1) : y + 1;
    const int yp2 = y + 2 >= H ? 2 * H - 2 - (y + 2) : y + 2;
    Gauss5VRowNeon(tmp16.ptr<ushort>(ym2), tmp16.ptr<ushort>(ym1),
                   tmp16.ptr<ushort>(y), tmp16.ptr<ushort>(yp1),
                   tmp16.ptr<ushort>(yp2), dst.ptr<uchar>(y), W);
  }
}

// Fully-fused luma chain (gauss5 + |y8-blur| 4x4 pooling + guided-stats
// decimation) built on the row kernels above. The blurred plane lives only
// as an 8-row u16 ring + a single u8 row: the 2.6MB tmp16 and the 1.3MB
// y_bl plane are never written or re-read. Outputs are bit-exact vs the
// separate stages (same kernels, same order). Board, 4 workers in
// parallel: 1.90 -> 1.47 ms per camera; 1-core 1.41 -> 1.38 ms (the win
// is DRAM bandwidth, not ALU).
inline void FusedLumaChainNeon(const cv::Mat &y8, cv::Mat &hf_q8,
                               cv::Mat &colsum, cv::Mat &yq8, cv::Mat &sq16,
                               cv::Mat &ring, cv::Mat &row_buf) {
  const int W = y8.cols, H = y8.rows, qW = W / 4, qH = H / 4;
  hf_q8.create(qH, qW, CV_8U);
  colsum.create(1, W, CV_16U);
  yq8.create(qH, qW, CV_8U);
  sq16.create(qH, qW, CV_16U);
  ring.create(8, W, CV_16U);
  row_buf.create(1, W, CV_8U);
  ushort *rr[8];
  for (int i = 0; i < 8; ++i) rr[i] = ring.ptr<ushort>(i);
  ushort *cs = colsum.ptr<ushort>(0);
  uchar *rb = row_buf.ptr<uchar>(0);

  int next_h = 0;
  const auto fill_h = [&](int upto) {
    for (; next_h <= upto && next_h < H; ++next_h)
      Gauss5HRowNeon(y8.ptr<uchar>(next_h), rr[next_h & 7], W);
  };
  for (int yq = 0; yq < qH; ++yq) {
    std::memset(cs, 0, W * sizeof(ushort));
    for (int r = 0; r < 4; ++r) {
      const int y = 4 * yq + r;
      fill_h(y + 2);
      const int ym2 = y - 2 < 0 ? 2 - y : y - 2;
      const int ym1 = y - 1 < 0 ? 1 - y : y - 1;
      const int yp1 = y + 1 >= H ? 2 * H - 2 - (y + 1) : y + 1;
      const int yp2 = y + 2 >= H ? 2 * H - 2 - (y + 2) : y + 2;
      Gauss5VRowNeon(rr[ym2 & 7], rr[ym1 & 7], rr[y & 7], rr[yp1 & 7],
                     rr[yp2 & 7], rb, W);
      // band-pass energy accumulate: colsum += |y8_row - blur_row|
      const uchar *py = y8.ptr<uchar>(y);
      int x = 0;
      for (; x + 16 <= W; x += 16) {
        const uint8x16_t d = vabdq_u8(vld1q_u8(py + x), vld1q_u8(rb + x));
        vst1q_u16(cs + x, vaddw_u8(vld1q_u16(cs + x), vget_low_u8(d)));
        vst1q_u16(cs + x + 8,
                  vaddw_u8(vld1q_u16(cs + x + 8), vget_high_u8(d)));
      }
      for (; x < W; ++x)
        cs[x] = static_cast<ushort>(cs[x] + std::abs(py[x] - rb[x]));
      if (r == 0) {
        // guided-stats source: NN decimation of blurred row 4*yq + squares
        uchar *d = yq8.ptr<uchar>(yq);
        ushort *sq = sq16.ptr<ushort>(yq);
        int xq = 0;
        for (; xq + 16 <= qW; xq += 16) {
          const uint8x16x4_t v4 = vld4q_u8(rb + 4 * xq);
          const uint8x16_t v = v4.val[0];
          vst1q_u8(d + xq, v);
          vst1q_u16(sq + xq, vmull_u8(vget_low_u8(v), vget_low_u8(v)));
          vst1q_u16(sq + xq + 8,
                    vmull_u8(vget_high_u8(v), vget_high_u8(v)));
        }
        for (; xq < qW; ++xq) {
          const uchar v = rb[4 * xq];
          d[xq] = v;
          sq[xq] = static_cast<ushort>(v * v);
        }
      }
    }
    uchar *po = hf_q8.ptr<uchar>(yq);
    for (int xq = 0; xq < qW; ++xq) {
      const int s =
          cs[4 * xq] + cs[4 * xq + 1] + cs[4 * xq + 2] + cs[4 * xq + 3];
      po[xq] = static_cast<uchar>((s + 8) >> 4);
    }
  }
}
#endif

// gauss5 dispatch: NEON kernel on ARM, cv::GaussianBlur elsewhere.
inline void Gauss5U8(const cv::Mat &src, cv::Mat &dst, cv::Mat &tmp16) {
#if defined(CYPERSTEREO_HAVE_NEON)
  if (UseNeonGauss5()) {
    Gauss5NeonU8(src, dst, tmp16);
    return;
  }
#endif
  (void)tmp16;
  cv::GaussianBlur(src, dst, cv::Size(5, 5), 0);
}

#if defined(CYPERSTEREO_HAVE_NEON)
inline bool UseNeonGuided() {
  static const bool enabled =
      std::getenv("CYPERSTEREO_DISABLE_NEON_GUIDED") == nullptr;
  return enabled;
}

inline bool UseFusedLumaChain() {
  static const bool enabled =
      std::getenv("CYPERSTEREO_DISABLE_FUSED_LUMA") == nullptr;
  return enabled;
}

// Integer rewrite of the guided-filter stats block for ARM. Replaces
// resize(INTER_NEAREST) + two f32 boxFilters + cv::multiply + the scalar
// a/b loop + two u16 boxFilters (six OpenCV dispatches on an 82K-px grid)
// with three fused passes over integer sliding sums. Box3 with
// BORDER_REFLECT == index clamping for a 3-tap kernel, so borders match
// cv::boxFilter exactly. The variance is computed EXACTLY as integers
// (num = 9*sum(x^2) - sum(x)^2 <= 5.3M, f32-24bit-exact) and
// a = num/(num+9^2*eps) equals var/(var+eps) algebraically; only the f32
// evaluation ORDER differs from the reference, measured a/b maxdiff 1 LSB
// (of 4096) on 0.1%/0.05% of cells -- less than the coefficient
// quantization step. mean_i (f32, used by the shadow gate) matches the
// reference within f32 rounding (measured maxdiff 0.0000). A76: 1.24 ->
// 0.51 ms.
// Stats core taking the already-decimated quarter grid (yq8 = NN sites of
// the blurred plane, sq16 = their squares). Split out so the fused luma
// chain (which produces yq8/sq16 as a by-product) can call it directly.
inline void GuidedStatsFromDecimNeon(const cv::Mat &yq8, const cv::Mat &sq16,
                                     cv::Mat &a_q16, cv::Mat &b_q16,
                                     cv::Mat &mean_i) {
  static thread_local cv::Mat vs16, vs32, a_q, b_q, hs_a, hs_b;
  const int qW = yq8.cols, qH = yq8.rows;
  const cv::Size quarter(qW, qH);
  vs16.create(quarter, CV_16U);
  vs32.create(quarter, CV_32S);
  a_q.create(quarter, CV_16U);
  b_q.create(quarter, CV_16U);
  mean_i.create(quarter, CV_32F);

  // Pass 2: vertical 3-row sums (rows clamped == BORDER_REFLECT for r=1).
  for (int y = 0; y < qH; ++y) {
    const uchar *r0 = yq8.ptr<uchar>(y > 0 ? y - 1 : 0);
    const uchar *r1 = yq8.ptr<uchar>(y);
    const uchar *r2 = yq8.ptr<uchar>(y < qH - 1 ? y + 1 : qH - 1);
    const ushort *q0 = sq16.ptr<ushort>(y > 0 ? y - 1 : 0);
    const ushort *q1 = sq16.ptr<ushort>(y);
    const ushort *q2 = sq16.ptr<ushort>(y < qH - 1 ? y + 1 : qH - 1);
    ushort *v16 = vs16.ptr<ushort>(y);
    int *v32 = vs32.ptr<int>(y);
    int x = 0;
    for (; x + 8 <= qW; x += 8) {
      const uint8x8_t a = vld1_u8(r0 + x), b = vld1_u8(r1 + x),
                      c = vld1_u8(r2 + x);
      vst1q_u16(v16 + x, vaddw_u8(vaddl_u8(a, b), c));
      const uint16x8_t s0 = vld1q_u16(q0 + x);
      const uint16x8_t s1 = vld1q_u16(q1 + x);
      const uint16x8_t s2 = vld1q_u16(q2 + x);
      uint32x4_t lo = vaddl_u16(vget_low_u16(s0), vget_low_u16(s1));
      uint32x4_t hi = vaddl_u16(vget_high_u16(s0), vget_high_u16(s1));
      lo = vaddw_u16(lo, vget_low_u16(s2));
      hi = vaddw_u16(hi, vget_high_u16(s2));
      vst1q_s32(v32 + x, vreinterpretq_s32_u32(lo));
      vst1q_s32(v32 + x + 4, vreinterpretq_s32_u32(hi));
    }
    for (; x < qW; ++x) {
      v16[x] = static_cast<ushort>(r0[x] + r1[x] + r2[x]);
      v32[x] = q0[x] + q1[x] + q2[x];
    }
  }

  // Pass 3: horizontal 3-sums + a/b/mean arithmetic in one walk.
  for (int y = 0; y < qH; ++y) {
    const ushort *v16 = vs16.ptr<ushort>(y);
    const int *v32 = vs32.ptr<int>(y);
    ushort *pa = a_q.ptr<ushort>(y);
    ushort *pb = b_q.ptr<ushort>(y);
    float *pm = mean_i.ptr<float>(y);
    const auto scalar_px = [&](int x) {
      const int xl = x > 0 ? x - 1 : 0, xr = x < qW - 1 ? x + 1 : qW - 1;
      const int m9 = v16[xl] + v16[x] + v16[xr];
      const int s9 = v32[xl] + v32[x] + v32[xr];
      int num = 9 * s9 - m9 * m9;
      if (num < 0) num = 0;
      const float af =
          static_cast<float>(num) / (static_cast<float>(num) + 1620.0f);
      const float m = static_cast<float>(m9) * (1.0f / 9.0f);
      pa[x] = static_cast<ushort>(af * 4096.0f + 0.5f);
      pb[x] = static_cast<ushort>((m - af * m) * 16.0f + 0.5f);
      pm[x] = m;
    };
    scalar_px(0);
    int x = 1;
    const float32x4_t k1620 = vdupq_n_f32(1620.0f);
    const float32x4_t k4096 = vdupq_n_f32(4096.0f);
    const float32x4_t kinv9x16 = vdupq_n_f32(16.0f / 9.0f);
    const float32x4_t kinv9 = vdupq_n_f32(1.0f / 9.0f);
    const float32x4_t khalf = vdupq_n_f32(0.5f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; x + 8 <= qW - 1; x += 8) {
      const uint16x8_t l = vld1q_u16(v16 + x - 1);
      const uint16x8_t c = vld1q_u16(v16 + x);
      const uint16x8_t r = vld1q_u16(v16 + x + 1);
      const uint16x8_t m9v = vaddq_u16(vaddq_u16(l, c), r);
      const int32x4_t s9lo = vaddq_s32(
          vaddq_s32(vld1q_s32(v32 + x - 1), vld1q_s32(v32 + x)),
          vld1q_s32(v32 + x + 1));
      const int32x4_t s9hi = vaddq_s32(
          vaddq_s32(vld1q_s32(v32 + x + 3), vld1q_s32(v32 + x + 4)),
          vld1q_s32(v32 + x + 5));
      const uint32x4_t m9lo = vmovl_u16(vget_low_u16(m9v));
      const uint32x4_t m9hi = vmovl_u16(vget_high_u16(m9v));
      const int32x4_t numlo = vmaxq_s32(
          vsubq_s32(vmulq_n_s32(s9lo, 9),
                    vreinterpretq_s32_u32(vmulq_u32(m9lo, m9lo))),
          vdupq_n_s32(0));
      const int32x4_t numhi = vmaxq_s32(
          vsubq_s32(vmulq_n_s32(s9hi, 9),
                    vreinterpretq_s32_u32(vmulq_u32(m9hi, m9hi))),
          vdupq_n_s32(0));
      const float32x4_t nlo = vcvtq_f32_s32(numlo);
      const float32x4_t nhi = vcvtq_f32_s32(numhi);
      const float32x4_t alo = vdivq_f32(nlo, vaddq_f32(nlo, k1620));
      const float32x4_t ahi = vdivq_f32(nhi, vaddq_f32(nhi, k1620));
      const float32x4_t mlo = vcvtq_f32_u32(m9lo);
      const float32x4_t mhi = vcvtq_f32_u32(m9hi);
      const uint32x4_t palo = vcvtq_u32_f32(vmlaq_f32(khalf, alo, k4096));
      const uint32x4_t pahi = vcvtq_u32_f32(vmlaq_f32(khalf, ahi, k4096));
      const uint32x4_t pblo = vcvtq_u32_f32(
          vmlaq_f32(khalf, vmulq_f32(mlo, kinv9x16), vsubq_f32(one, alo)));
      const uint32x4_t pbhi = vcvtq_u32_f32(
          vmlaq_f32(khalf, vmulq_f32(mhi, kinv9x16), vsubq_f32(one, ahi)));
      vst1q_u16(pa + x, vcombine_u16(vmovn_u32(palo), vmovn_u32(pahi)));
      vst1q_u16(pb + x, vcombine_u16(vmovn_u32(pblo), vmovn_u32(pbhi)));
      vst1q_f32(pm + x, vmulq_f32(mlo, kinv9));
      vst1q_f32(pm + x + 4, vmulq_f32(mhi, kinv9));
    }
    for (; x < qW - 1; ++x) scalar_px(x);
    scalar_px(qW - 1);
  }

  // Pass 4: box3 smoothing of a/b as integer sliding sums; the exact
  // round(sum/9) matches cv::boxFilter's normalized u16 output (sum <=
  // 9*4096 is f32-exact, and sum/9 never lands on a .5 tie).
  hs_a.create(quarter, CV_32S);
  hs_b.create(quarter, CV_32S);
  for (int y = 0; y < qH; ++y) {
    const ushort *sa = a_q.ptr<ushort>(y);
    const ushort *sb = b_q.ptr<ushort>(y);
    int *ha = hs_a.ptr<int>(y);
    int *hb = hs_b.ptr<int>(y);
    ha[0] = 2 * sa[0] + sa[1];
    hb[0] = 2 * sb[0] + sb[1];
    for (int x = 1; x < qW - 1; ++x) {
      ha[x] = sa[x - 1] + sa[x] + sa[x + 1];
      hb[x] = sb[x - 1] + sb[x] + sb[x + 1];
    }
    ha[qW - 1] = sa[qW - 2] + 2 * sa[qW - 1];
    hb[qW - 1] = sb[qW - 2] + 2 * sb[qW - 1];
  }
  a_q16.create(quarter, CV_16U);
  b_q16.create(quarter, CV_16U);
  const float32x4_t inv9 = vdupq_n_f32(1.0f / 9.0f);
  const float32x4_t half = vdupq_n_f32(0.5f);
  for (int y = 0; y < qH; ++y) {
    const int *a0 = hs_a.ptr<int>(y > 0 ? y - 1 : 0);
    const int *a1 = hs_a.ptr<int>(y);
    const int *a2 = hs_a.ptr<int>(y < qH - 1 ? y + 1 : qH - 1);
    const int *b0 = hs_b.ptr<int>(y > 0 ? y - 1 : 0);
    const int *b1 = hs_b.ptr<int>(y);
    const int *b2 = hs_b.ptr<int>(y < qH - 1 ? y + 1 : qH - 1);
    ushort *da = a_q16.ptr<ushort>(y);
    ushort *db = b_q16.ptr<ushort>(y);
    int x = 0;
    for (; x + 4 <= qW; x += 4) {
      const int32x4_t s9a = vaddq_s32(
          vaddq_s32(vld1q_s32(a0 + x), vld1q_s32(a1 + x)),
          vld1q_s32(a2 + x));
      const int32x4_t s9b = vaddq_s32(
          vaddq_s32(vld1q_s32(b0 + x), vld1q_s32(b1 + x)),
          vld1q_s32(b2 + x));
      const uint32x4_t ra =
          vcvtq_u32_f32(vmlaq_f32(half, vcvtq_f32_s32(s9a), inv9));
      const uint32x4_t rb =
          vcvtq_u32_f32(vmlaq_f32(half, vcvtq_f32_s32(s9b), inv9));
      vst1_u16(da + x, vmovn_u32(ra));
      vst1_u16(db + x, vmovn_u32(rb));
    }
    for (; x < qW; ++x) {
      da[x] = static_cast<ushort>((a0[x] + a1[x] + a2[x]) / 9.0f + 0.5f);
      db[x] = static_cast<ushort>((b0[x] + b1[x] + b2[x]) / 9.0f + 0.5f);
    }
  }
}

inline void GuidedStatsIntNeon(const cv::Mat &y_bl, cv::Mat &a_q16,
                               cv::Mat &b_q16, cv::Mat &mean_i) {
  static thread_local cv::Mat yq8, sq16;
  const int qW = y_bl.cols / 4, qH = y_bl.rows / 4;
  const cv::Size quarter(qW, qH);
  yq8.create(quarter, CV_8U);
  sq16.create(quarter, CV_16U);

  // Pass 1: NN decimation (same sites as cv::resize INTER_NEAREST) plus
  // squares.
  for (int y = 0; y < qH; ++y) {
    const uchar *s = y_bl.ptr<uchar>(4 * y);
    uchar *d = yq8.ptr<uchar>(y);
    ushort *sq = sq16.ptr<ushort>(y);
    int x = 0;
    for (; x + 16 <= qW; x += 16) {
      const uint8x16x4_t v4 = vld4q_u8(s + 4 * x);
      const uint8x16_t v = v4.val[0];
      vst1q_u8(d + x, v);
      vst1q_u16(sq + x, vmull_u8(vget_low_u8(v), vget_low_u8(v)));
      vst1q_u16(sq + x + 8, vmull_u8(vget_high_u8(v), vget_high_u8(v)));
    }
    for (; x < qW; ++x) {
      const uchar v = s[4 * x];
      d[x] = v;
      sq[x] = static_cast<ushort>(v * v);
    }
  }
  GuidedStatsFromDecimNeon(yq8, sq16, a_q16, b_q16, mean_i);
}
#endif

#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
inline bool CpuHasAvx2Output() {
#if defined(__GNUC__) || defined(__clang__)
  // The opt-out is intentionally cached: it provides a zero-overhead way to
  // run whole-pipeline A/B tests on the same binary, including any AVX2
  // frequency effect under four-camera load.
  static const bool has_avx2 =
      __builtin_cpu_supports("avx2") &&
      std::getenv("CYPERSTEREO_DISABLE_AVX2_OUTPUT") == nullptr;
  return has_avx2;
#else
  return true;
#endif
}

CYPERSTEREO_AVX2_TARGET inline __m256i PackAndRepeat2Avx2(__m256i v) {
  const __m128i v8 = _mm_packs_epi32(
      _mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  return _mm256_set_m128i(_mm_unpackhi_epi16(v8, v8),
                          _mm_unpacklo_epi16(v8, v8));
}

// AVX2 backend for the final luma-MAC + 4:2:0 chroma upsample +
// YCrCb-to-BGR stage. It evaluates 16 full-resolution pixels per dependency
// chain using 256-bit arithmetic, then uses OpenCV's proven 3-channel SSE
// store. The integer equations and rounding are exactly the same as the
// portable backend below.
CYPERSTEREO_AVX2_TARGET inline void FusedOutputAvx2(
    const cv::Mat &y8, const cv::Mat &a_q16, const cv::Mat &b_q16,
    const cv::Mat &cr_h, const cv::Mat &cb_h, cv::Mat &color) {
  const int full_width = color.cols;
  const int full_height = color.rows;
  const int half_width = cr_h.cols;
  const int half_height = cr_h.rows;
  const int quarter_height = a_q16.rows;
  const __m256i round_y = _mm256_set1_epi32(2048);
  const __m256i c128_32 = _mm256_set1_epi32(128);
  const __m256i k454 = _mm256_set1_epi32(454);
  const __m256i k183 = _mm256_set1_epi32(183);
  const __m256i k88 = _mm256_set1_epi32(88);
  const __m256i k359 = _mm256_set1_epi32(359);

  for (int y = 0; y < full_height; ++y) {
    const int yh = std::min(y >> 1, half_height - 1);
    const int yq = std::min(y >> 2, quarter_height - 1);
    const uchar *py = y8.ptr<uchar>(y);
    const ushort *pa = a_q16.ptr<ushort>(yq);
    const ushort *pb = b_q16.ptr<ushort>(yq);
    const uchar *pcr = cr_h.ptr<uchar>(yh);
    const uchar *pcb = cb_h.ptr<uchar>(yh);
    uchar *pc = color.ptr<uchar>(y);
    int xh = 0;

    for (; xh + 8 <= half_width; xh += 8) {
      const int x0 = xh << 1;
      const int xq = xh >> 1;

      // Four quarter-grid values -> sixteen u16 lanes (each repeated 4x).
      const __m128i a4 =
          _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pa + xq));
      const __m128i b4 =
          _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pb + xq));
      const __m128i a8 = _mm_unpacklo_epi16(a4, a4);
      const __m128i b8 = _mm_unpacklo_epi16(b4, b4);
      const __m256i a16 = _mm256_set_m128i(
          _mm_unpackhi_epi16(a8, a8), _mm_unpacklo_epi16(a8, a8));
      const __m256i b16 = _mm256_set_m128i(
          _mm_unpackhi_epi16(b8, b8), _mm_unpacklo_epi16(b8, b8));
      const __m256i y16 = _mm256_cvtepu8_epi16(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(py + x0)));

      const __m256i y32_lo =
          _mm256_cvtepu16_epi32(_mm256_castsi256_si128(y16));
      const __m256i y32_hi =
          _mm256_cvtepu16_epi32(_mm256_extracti128_si256(y16, 1));
      const __m256i a32_lo =
          _mm256_cvtepu16_epi32(_mm256_castsi256_si128(a16));
      const __m256i a32_hi =
          _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a16, 1));
      const __m256i b32_lo =
          _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b16));
      const __m256i b32_hi =
          _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b16, 1));

      const __m256i yn32_lo = _mm256_srli_epi32(
          _mm256_add_epi32(
              _mm256_mullo_epi32(a32_lo, y32_lo),
              _mm256_add_epi32(_mm256_slli_epi32(b32_lo, 8), round_y)),
          12);
      const __m256i yn32_hi = _mm256_srli_epi32(
          _mm256_add_epi32(
              _mm256_mullo_epi32(a32_hi, y32_hi),
              _mm256_add_epi32(_mm256_slli_epi32(b32_hi, 8), round_y)),
          12);
      // vpackusdw is lane-local; reorder 64-bit chunks back to pixel order.
      const __m256i yn16 = _mm256_permute4x64_epi64(
          _mm256_packus_epi32(yn32_lo, yn32_hi), 0xd8);

      const __m256i dcr = _mm256_sub_epi32(
          _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pcr + xh))),
          c128_32);
      const __m256i dcb = _mm256_sub_epi32(
          _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pcb + xh))),
          c128_32);
      const __m256i tb32 =
          _mm256_srai_epi32(_mm256_mullo_epi32(dcb, k454), 8);
      const __m256i tg32 = _mm256_srai_epi32(
          _mm256_add_epi32(_mm256_mullo_epi32(dcr, k183),
                           _mm256_mullo_epi32(dcb, k88)),
          8);
      const __m256i tr32 =
          _mm256_srai_epi32(_mm256_mullo_epi32(dcr, k359), 8);

      const __m256i tb16 = PackAndRepeat2Avx2(tb32);
      const __m256i tg16 = PackAndRepeat2Avx2(tg32);
      const __m256i tr16 = PackAndRepeat2Avx2(tr32);

      const __m256i outb16 = _mm256_add_epi16(yn16, tb16);
      const __m256i outg16 = _mm256_sub_epi16(yn16, tg16);
      const __m256i outr16 = _mm256_add_epi16(yn16, tr16);
      const __m128i outb = _mm_packus_epi16(
          _mm256_castsi256_si128(outb16),
          _mm256_extracti128_si256(outb16, 1));
      const __m128i outg = _mm_packus_epi16(
          _mm256_castsi256_si128(outg16),
          _mm256_extracti128_si256(outg16, 1));
      const __m128i outr = _mm_packus_epi16(
          _mm256_castsi256_si128(outr16),
          _mm256_extracti128_si256(outr16, 1));
      cv::v_store_interleave(pc + 3 * x0, cv::v_uint8x16(outb),
                             cv::v_uint8x16(outg), cv::v_uint8x16(outr));
    }

    // Generic odd-size tail. The camera's 1280-wide frames have no tail.
    for (; xh < half_width; ++xh) {
      const int a = pa[xh >> 1];
      const int b256 = pb[xh >> 1] << 8;
      const int dcr = static_cast<int>(pcr[xh]) - 128;
      const int dcb = static_cast<int>(pcb[xh]) - 128;
      const int tb = (454 * dcb) >> 8;
      const int tg = (183 * dcr + 88 * dcb) >> 8;
      const int tr = (359 * dcr) >> 8;
      const int x0 = xh << 1;
      for (int k = 0; k < 2 && x0 + k < full_width; ++k) {
        const int yn = (a * py[x0 + k] + b256 + 2048) >> 12;
        uchar *o = pc + 3 * (x0 + k);
        o[0] = cv::saturate_cast<uchar>(yn + tb);
        o[1] = cv::saturate_cast<uchar>(yn - tg);
        o[2] = cv::saturate_cast<uchar>(yn + tr);
      }
    }
  }
}
#endif

#if defined(CYPERSTEREO_HAVE_NEON)
inline bool UseNeonOutput() {
  // Cached A/B switch for measuring the explicit NEON kernel against the
  // universal-intrinsics fallback on the actual ARM board.
  static const bool enabled =
      std::getenv("CYPERSTEREO_DISABLE_NEON_OUTPUT") == nullptr;
  return enabled;
}

// AArch64/ARM NEON backend for RK3588 and other ARM targets. NEON is
// 128-bit (16 output pixels per vector step like the portable path), but
// this kernel walks TWO image rows per iteration: rows 2k and 2k+1 share
// one chroma row and one coefficient row, so the a/b unpack and the whole
// Cr/Cb term computation run once per row PAIR instead of once per row
// (measured on A76: 2.27 -> 2.01 ms, bit-exact). Native widening MACs and
// a direct vst3q_u8 BGR store; bit-exact vs the scalar reference.
inline void FusedOutputNeon(
    const cv::Mat &y8, const cv::Mat &a_q16, const cv::Mat &b_q16,
    const cv::Mat &cr_h, const cv::Mat &cb_h, cv::Mat &color) {
  const int full_width = color.cols;
  const int full_height = color.rows;
  const int half_width = cr_h.cols;
  const int half_height = cr_h.rows;
  const int quarter_height = a_q16.rows;
  const uint32x4_t round_y = vdupq_n_u32(2048);
  const int16x8_t c128_16 = vdupq_n_s16(128);

  const int paired_height = full_height & ~1;
  for (int y = 0; y < paired_height; y += 2) {
    // Rows y and y+1 (y even) always share yh = y>>1 and yq = y>>2.
    const int yh = std::min(y >> 1, half_height - 1);
    const int yq = std::min(y >> 2, quarter_height - 1);
    const uchar *py0 = y8.ptr<uchar>(y);
    const uchar *py1 = y8.ptr<uchar>(y + 1);
    const ushort *pa = a_q16.ptr<ushort>(yq);
    const ushort *pb = b_q16.ptr<ushort>(yq);
    const uchar *pcr = cr_h.ptr<uchar>(yh);
    const uchar *pcb = cb_h.ptr<uchar>(yh);
    uchar *pc0 = color.ptr<uchar>(y);
    uchar *pc1 = color.ptr<uchar>(y + 1);
    int xh = 0;

    for (; xh + 8 <= half_width; xh += 8) {
      const int x0 = xh << 1;
      const int xq = xh >> 1;
      const uint16x4_t a4 = vld1_u16(pa + xq);
      const uint16x4_t b4 = vld1_u16(pb + xq);
      const uint16x4x2_t a8 = vzip_u16(a4, a4);
      const uint16x4x2_t b8 = vzip_u16(b4, b4);
      const uint16x4x2_t a01 = vzip_u16(a8.val[0], a8.val[0]);
      const uint16x4x2_t a23 = vzip_u16(a8.val[1], a8.val[1]);
      const uint16x4x2_t b01 = vzip_u16(b8.val[0], b8.val[0]);
      const uint16x4x2_t b23 = vzip_u16(b8.val[1], b8.val[1]);
      const uint16x8_t av0 = vcombine_u16(a01.val[0], a01.val[1]);
      const uint16x8_t av1 = vcombine_u16(a23.val[0], a23.val[1]);
      const uint16x8_t bv0 = vcombine_u16(b01.val[0], b01.val[1]);
      const uint16x8_t bv1 = vcombine_u16(b23.val[0], b23.val[1]);

      // Chroma terms: once per row pair.
      const int16x8_t dcr = vsubq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pcr + xh))), c128_16);
      const int16x8_t dcb = vsubq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pcb + xh))), c128_16);
      const auto mul_shift = [](int16x8_t v, int coefficient) {
        const int32x4_t lo =
            vshrq_n_s32(vmull_n_s16(vget_low_s16(v), coefficient), 8);
        const int32x4_t hi =
            vshrq_n_s32(vmull_n_s16(vget_high_s16(v), coefficient), 8);
        return vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
      };
      const int16x8_t tb = mul_shift(dcb, 454);
      const int16x8_t tr = mul_shift(dcr, 359);
      const int32x4_t tg_lo = vshrq_n_s32(
          vmlal_n_s16(vmull_n_s16(vget_low_s16(dcr), 183),
                      vget_low_s16(dcb), 88),
          8);
      const int32x4_t tg_hi = vshrq_n_s32(
          vmlal_n_s16(vmull_n_s16(vget_high_s16(dcr), 183),
                      vget_high_s16(dcb), 88),
          8);
      const int16x8_t tg =
          vcombine_s16(vqmovn_s32(tg_lo), vqmovn_s32(tg_hi));
      const int16x8x2_t tb2 = vzipq_s16(tb, tb);
      const int16x8x2_t tg2 = vzipq_s16(tg, tg);
      const int16x8x2_t tr2 = vzipq_s16(tr, tr);

      const auto luma_mac = [&](uint16x8_t a, uint16x8_t b,
                                uint16x8_t yy) {
        uint32x4_t lo =
            vmlal_u16(vshll_n_u16(vget_low_u16(b), 8),
                      vget_low_u16(a), vget_low_u16(yy));
        uint32x4_t hi =
            vmlal_u16(vshll_n_u16(vget_high_u16(b), 8),
                      vget_high_u16(a), vget_high_u16(yy));
        lo = vshrq_n_u32(vaddq_u32(lo, round_y), 12);
        hi = vshrq_n_u32(vaddq_u32(hi, round_y), 12);
        return vcombine_u16(vmovn_u32(lo), vmovn_u32(hi));
      };
      const auto emit_row = [&](const uchar *py, uchar *pc) {
        const uint8x16_t yv = vld1q_u8(py + x0);
        const int16x8_t yns0 = vreinterpretq_s16_u16(
            luma_mac(av0, bv0, vmovl_u8(vget_low_u8(yv))));
        const int16x8_t yns1 = vreinterpretq_s16_u16(
            luma_mac(av1, bv1, vmovl_u8(vget_high_u8(yv))));
        uint8x16x3_t out;
        out.val[0] = vcombine_u8(
            vqmovun_s16(vaddq_s16(yns0, tb2.val[0])),
            vqmovun_s16(vaddq_s16(yns1, tb2.val[1])));
        out.val[1] = vcombine_u8(
            vqmovun_s16(vsubq_s16(yns0, tg2.val[0])),
            vqmovun_s16(vsubq_s16(yns1, tg2.val[1])));
        out.val[2] = vcombine_u8(
            vqmovun_s16(vaddq_s16(yns0, tr2.val[0])),
            vqmovun_s16(vaddq_s16(yns1, tr2.val[1])));
        vst3q_u8(pc + 3 * x0, out);
      };
      emit_row(py0, pc0);
      emit_row(py1, pc1);
    }

    for (; xh < half_width; ++xh) {
      const int a = pa[xh >> 1];
      const int b256 = pb[xh >> 1] << 8;
      const int dcr = static_cast<int>(pcr[xh]) - 128;
      const int dcb = static_cast<int>(pcb[xh]) - 128;
      const int tb = (454 * dcb) >> 8;
      const int tg = (183 * dcr + 88 * dcb) >> 8;
      const int tr = (359 * dcr) >> 8;
      const int x0 = xh << 1;
      for (int r = 0; r < 2; ++r) {
        const uchar *py = r ? py1 : py0;
        uchar *pc = r ? pc1 : pc0;
        for (int k = 0; k < 2 && x0 + k < full_width; ++k) {
          const int yn = (a * py[x0 + k] + b256 + 2048) >> 12;
          uchar *o = pc + 3 * (x0 + k);
          o[0] = cv::saturate_cast<uchar>(yn + tb);
          o[1] = cv::saturate_cast<uchar>(yn - tg);
          o[2] = cv::saturate_cast<uchar>(yn + tr);
        }
      }
    }
  }

  // Odd trailing row (not hit by the 1024/480-high sensor modes).
  for (int y = paired_height; y < full_height; ++y) {
    const int yh = std::min(y >> 1, half_height - 1);
    const int yq = std::min(y >> 2, quarter_height - 1);
    const uchar *py = y8.ptr<uchar>(y);
    const ushort *pa = a_q16.ptr<ushort>(yq);
    const ushort *pb = b_q16.ptr<ushort>(yq);
    const uchar *pcr = cr_h.ptr<uchar>(yh);
    const uchar *pcb = cb_h.ptr<uchar>(yh);
    uchar *pc = color.ptr<uchar>(y);
    for (int xh = 0; xh < half_width; ++xh) {
      const int a = pa[xh >> 1];
      const int b256 = pb[xh >> 1] << 8;
      const int dcr = static_cast<int>(pcr[xh]) - 128;
      const int dcb = static_cast<int>(pcb[xh]) - 128;
      const int tb = (454 * dcb) >> 8;
      const int tg = (183 * dcr + 88 * dcb) >> 8;
      const int tr = (359 * dcr) >> 8;
      const int x0 = xh << 1;
      for (int k = 0; k < 2 && x0 + k < full_width; ++k) {
        const int yn = (a * py[x0 + k] + b256 + 2048) >> 12;
        uchar *o = pc + 3 * (x0 + k);
        o[0] = cv::saturate_cast<uchar>(yn + tb);
        o[1] = cv::saturate_cast<uchar>(yn - tg);
        o[2] = cv::saturate_cast<uchar>(yn + tr);
      }
    }
  }
}
#endif

// raw_wb: white-balanced Bayer plane, or nullptr. When given (ARM fused
// demosaic path) the EA demosaic runs fused with the front-end and the
// interleaved BGR frame is never materialized; `color` is only written by
// the output kernel at the end.
inline void SuppressFalseColorImpl(const cv::Mat *raw_wb, cv::Mat &color) {
  static thread_local cv::Mat y8, y_bl, colsum;              // u8 full res
  static thread_local cv::Mat yq8, sq_q, mean_i, mean_ii, a_q, b_q;
  static thread_local cv::Mat a_q16, b_q16;                  // u16 coeffs
  static thread_local cv::Mat hf_q8, tex_q;                  // u8 quarter
  static thread_local cv::Mat keep_q8;                       // u8 gate
  static thread_local cv::Mat bgr_h, ycc_h, ch_h[3];         // half res
  static thread_local cv::Mat cr_h, cb_h;                    // u8 half
  static thread_local cv::Mat g5_tmp_full, g5_tmp_half;      // u16 scratch
  static thread_local cv::Mat g5_row_buf;                    // u8 one row

  if (raw_wb) color.create(raw_wb->rows, raw_wb->cols, CV_8UC3);
  const cv::Size full(color.cols, color.rows);
  const cv::Size half(color.cols / 2, color.rows / 2);
  const cv::Size quarter(color.cols / 4, color.rows / 4);

  // Luma plane + its gauss5 low-pass (shared by the texture band-pass
  // below and by the guided-filter stats). With the fused front-end the
  // same pass also emits the raw half-res chroma (into ch_h[1]/ch_h[2]);
  // its gauss5 NR happens further down, exactly where the split-path
  // chroma gets blurred.
  const bool fused_front = UseFusedFront();
  const bool have_chroma = raw_wb != nullptr || fused_front;
#if defined(CYPERSTEREO_HAVE_NEON)
  if (raw_wb) {
    FusedDemosaicFrontNeon(*raw_wb, y8, ch_h[1], ch_h[2]);
  } else
#endif
  if (fused_front) {
    FusedFrontYCbCr420(color, y8, ch_h[1], ch_h[2]);
  } else {
    cv::cvtColor(color, y8, cv::COLOR_BGR2GRAY);
  }

  // Luma NR (mechanism 3): self-guided filter, stats on the 1/4 grid
  // (box3 there ~= 12px full-res support). eps ~= (edge threshold)^2:
  // variation below ~4.5 DN local contrast is treated as noise. Guided
  // filtering is edge-preserving by construction (a->1 at edges, a->0 on
  // flats), so luma needs no texture gating. The stats source is a NN
  // decimation of the ALREADY-COMPUTED gauss5 plane (y_bl) -- reusing it
  // costs 0.05 ms where INTER_AREA of y8 cost 0.67 ms, and the measured
  // guided coefficients match within 1% (mean a: 0.415 vs 0.419).
  // var/a/b arithmetic runs as ONE scalar loop over the 80K-float grid
  // instead of six whole-plane cv:: ops (multiply/subtract/add/divide each
  // re-walked 320 KB; measured 1.0 -> 0.55 ms, output identical).
  //
  // ARM takes the fully-fused row-streaming version of this whole stretch
  // (gauss5 + band-pass pooling + guided-stats source; the blurred plane
  // never exists in memory) -- outputs verified bit-exact, and under
  // 4-worker DRAM contention it is 1.90 -> 1.47 ms. The AbsDiffPool4 call
  // further down stays for the non-fused paths only.
  bool luma_fused = false;
#if defined(CYPERSTEREO_HAVE_NEON)
  if (UseNeonGauss5() && UseNeonGuided() && UseFusedLumaChain()) {
    FusedLumaChainNeon(y8, hf_q8, colsum, yq8, sq_q, g5_tmp_full,
                       g5_row_buf);
    GuidedStatsFromDecimNeon(yq8, sq_q, a_q16, b_q16, mean_i);
    luma_fused = true;
  }
#endif
  if (!luma_fused) Gauss5U8(y8, y_bl, g5_tmp_full);
  constexpr float kLumaNREps = 20.0f;
#if defined(CYPERSTEREO_HAVE_NEON)
  if (!luma_fused && UseNeonGuided()) {
    GuidedStatsIntNeon(y_bl, a_q16, b_q16, mean_i);
  } else if (!luma_fused)
#else
  if (!luma_fused)
#endif
  {
    cv::resize(y_bl, yq8, quarter, 0, 0, cv::INTER_NEAREST);
    const cv::Size w3(3, 3);
    cv::boxFilter(yq8, mean_i, CV_32F, w3, cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
    cv::multiply(yq8, yq8, sq_q, 1.0, CV_16U);
    cv::boxFilter(sq_q, mean_ii, CV_32F, w3, cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
    // a,b are quantized to fixed point HERE (a Q12, b Q4) and the smoothing
    // box3 runs on u16 planes: vs f32 boxes + convertTo this is 1.08 ->
    // 0.70 ms and the rounding lands within 1 LSB of the f32 path (the box
    // average smooths the quantization noise).
    a_q.create(quarter, CV_16U);
    b_q.create(quarter, CV_16U);
    for (int y = 0; y < quarter.height; ++y) {
      const float *pi = mean_i.ptr<float>(y);
      const float *pii = mean_ii.ptr<float>(y);
      ushort *pa = a_q.ptr<ushort>(y);
      ushort *pb = b_q.ptr<ushort>(y);
      for (int x = 0; x < quarter.width; ++x) {
        const float m = pi[x];
        float v = pii[x] - m * m;
        if (v < 0.f) v = 0.f;
        const float a = v / (v + kLumaNREps);
        pa[x] = static_cast<ushort>(a * 4096.0f + 0.5f);
        pb[x] = static_cast<ushort>((m - a * m) * 16.0f + 0.5f);
      }
    }
    cv::boxFilter(a_q, a_q16, -1, w3, cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
    cv::boxFilter(b_q, b_q16, -1, w3, cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
  }
  // a in [0,1] -> Q12; b in [0,255] -> Q4. y' = (a*y + (b<<8) + 2048) >> 12.
  // The coefficient planes stay on the quarter grid; the output loop
  // expands them NN (bilinear upsampling cost 0.65 ms/frame and measurably
  // changed < 0.11% of pixels; residual 4px-seam energy on flat walls is
  // 0.43 vs 0.30 DN -- invisible at unity zoom).

  // Gate maps at quarter res. strength: 0 below kTexLo=6.5 (flat/smooth
  // keeps full color), 1 above kTexHi=15 (dense texture fully
  // desaturated); grille aliasing measures T>=10.9 (p10), skin p90 ~5.7.
  // shadow confidence: 0 at/below Y=16, 1 at/above Y=64 (mechanism 2);
  // skin sits at Y>=85 in our dim captures. keep = shade*(1-strength),
  // all in u8 fixed point (strength255 = 30*T - 195 == (T-6.5)*255/8.5).
  // The band-pass energy |y8 - y_bl| is STILL evaluated at full res (see
  // note d in the header comment) -- AbsDiffPool4 only fuses the pooling.
  // (On the ARM fused-luma path hf_q8 was already produced above.)
  if (!luma_fused) AbsDiffPool4(y8, y_bl, hf_q8, colsum);
  cv::GaussianBlur(hf_q8, tex_q, cv::Size(3, 3), 0);
  keep_q8.create(quarter, CV_8U);
  for (int y = 0; y < quarter.height; ++y) {
    const uchar *pt = tex_q.ptr<uchar>(y);
    const float *pl = mean_i.ptr<float>(y);  // box3 quarter luma ~ 12px LP
    uchar *pk = keep_q8.ptr<uchar>(y);
    for (int x = 0; x < quarter.width; ++x) {
      int strength = pt[x] * 30 - 195;
      strength = strength < 0 ? 0 : (strength > 255 ? 255 : strength);
      int shade = static_cast<int>((pl[x] - 16.0f) * (255.0f / 48.0f));
      shade = shade < 0 ? 0 : (shade > 255 ? 255 : shade);
      pk[x] = static_cast<uchar>(shade * (255 - strength) / 255);
    }
  }

  // Chroma path at half res (standard 4:2:0 treatment): gauss5 NR, gated
  // pull toward neutral 128, median3 for isolated speckle. The keep gate
  // is read NN straight off the quarter grid (u8 zip -> 16 lanes), same
  // trick as the a/b coefficients; the former bilinear upsample to half
  // res changed the output by < 1 gate quantization step.
  if (!have_chroma) {
    cv::resize(color, bgr_h, half, 0, 0, cv::INTER_AREA);
    cv::cvtColor(bgr_h, ycc_h, cv::COLOR_BGR2YCrCb);
    cv::split(ycc_h, ch_h);
  }
  Gauss5U8(ch_h[1], cr_h, g5_tmp_half);
  Gauss5U8(ch_h[2], cb_h, g5_tmp_half);
  for (int y = 0; y < half.height; ++y) {
    const int yq = (y >> 1) < quarter.height ? (y >> 1) : quarter.height - 1;
    const uchar *pk = keep_q8.ptr<uchar>(yq);
    for (cv::Mat *plane : {&cr_h, &cb_h}) {
      uchar *pv = plane->ptr<uchar>(y);
      int x = 0;
#if CV_SIMD128
      const cv::v_int16x8 c128 = cv::v_setall_s16(128);
      const cv::v_int32x4 rnd = cv::v_setall_s32(128);
      for (; x + 16 <= half.width; x += 16) {
        cv::v_uint16x8 c_lo, c_hi, k_lo, k_hi;
        cv::v_expand(cv::v_load(pv + x), c_lo, c_hi);
        cv::v_uint8x16 k8 = cv::v_load_low(pk + (x >> 1));  // 8 quarter px
        cv::v_uint8x16 k16, kdum;
        cv::v_zip(k8, k8, k16, kdum);                       // -> 16 lanes
        cv::v_expand(k16, k_lo, k_hi);
        cv::v_int16x8 d_lo = cv::v_reinterpret_as_s16(c_lo) - c128;
        cv::v_int16x8 d_hi = cv::v_reinterpret_as_s16(c_hi) - c128;
        cv::v_int32x4 d0, d1, d2, d3, k0, k1, k2, k3;
        cv::v_expand(d_lo, d0, d1);
        cv::v_expand(d_hi, d2, d3);
        cv::v_expand(cv::v_reinterpret_as_s16(k_lo), k0, k1);
        cv::v_expand(cv::v_reinterpret_as_s16(k_hi), k2, k3);
        d0 = (d0 * k0 + rnd) >> 8;
        d1 = (d1 * k1 + rnd) >> 8;
        d2 = (d2 * k2 + rnd) >> 8;
        d3 = (d3 * k3 + rnd) >> 8;
        cv::v_int16x8 o_lo = cv::v_pack(d0, d1) + c128;
        cv::v_int16x8 o_hi = cv::v_pack(d2, d3) + c128;
        cv::v_store(pv + x, cv::v_pack_u(o_lo, o_hi));
      }
#endif
      for (; x < half.width; ++x) {
        const int k = pk[x >> 1];
        pv[x] = static_cast<uchar>(128 + (((pv[x] - 128) * k + 128) >> 8));
      }
    }
  }
  cv::medianBlur(cr_h, cr_h, 3);
  cv::medianBlur(cb_h, cb_h, 3);

  // Fused output: y' = a*y + b (coeffs NN from the quarter grid), chroma
  // NN-upsampled from the half grid, YCrCb->BGR in BT.601 integer math,
  // written straight into `color`. Use an architecture-specific backend
  // where available; the existing universal-intrinsics loop below remains
  // the bit-exact fallback for SSE-only x86 and other architectures.
#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
  if (CpuHasAvx2Output()) {
    FusedOutputAvx2(y8, a_q16, b_q16, cr_h, cb_h, color);
    return;
  }
#endif
#if defined(CYPERSTEREO_HAVE_NEON)
  if (UseNeonOutput()) {
    FusedOutputNeon(y8, a_q16, b_q16, cr_h, cb_h, color);
    return;
  }
#endif
  for (int y = 0; y < full.height; ++y) {
    const int yh = (y >> 1) < half.height ? (y >> 1) : half.height - 1;
    const int yq = (y >> 2) < quarter.height ? (y >> 2) : quarter.height - 1;
    const uchar *py = y8.ptr<uchar>(y);
    const ushort *pa = a_q16.ptr<ushort>(yq);
    const ushort *pb = b_q16.ptr<ushort>(yq);
    const uchar *pcr = cr_h.ptr<uchar>(yh);
    const uchar *pcb = cb_h.ptr<uchar>(yh);
    uchar *pc = color.ptr<uchar>(y);
    int xh = 0;
#if CV_SIMD128
    // 8 half-grid samples -> 16 output pixels per iteration.
    for (; xh + 8 <= half.width; xh += 8) {
      const int x0 = 2 * xh;
      const int xq = xh >> 1;
      // 4 quarter coeffs -> 8 half lanes -> (below) 16 full lanes.
      cv::v_uint16x8 a4 = cv::v_load_low(pa + xq);
      cv::v_uint16x8 b4 = cv::v_load_low(pb + xq);
      cv::v_uint16x8 a8, b8, zip_hi;
      cv::v_zip(a4, a4, a8, zip_hi);
      cv::v_zip(b4, b4, b8, zip_hi);
      cv::v_uint16x8 a_lo, a_hi, b_lo, b_hi;
      cv::v_zip(a8, a8, a_lo, a_hi);
      cv::v_zip(b8, b8, b_lo, b_hi);

      cv::v_uint16x8 y_lo, y_hi;
      cv::v_expand(cv::v_load(py + x0), y_lo, y_hi);

      cv::v_uint32x4 ay0, ay1, ay2, ay3;
      cv::v_mul_expand(a_lo, y_lo, ay0, ay1);
      cv::v_mul_expand(a_hi, y_hi, ay2, ay3);
      cv::v_uint32x4 b0, b1, b2, b3;
      cv::v_expand(b_lo, b0, b1);
      cv::v_expand(b_hi, b2, b3);
      const cv::v_uint32x4 rnd = cv::v_setall_u32(2048);
      ay0 = (ay0 + (b0 << 8) + rnd) >> 12;
      ay1 = (ay1 + (b1 << 8) + rnd) >> 12;
      ay2 = (ay2 + (b2 << 8) + rnd) >> 12;
      ay3 = (ay3 + (b3 << 8) + rnd) >> 12;
      cv::v_int16x8 yn_lo = cv::v_reinterpret_as_s16(cv::v_pack(ay0, ay1));
      cv::v_int16x8 yn_hi = cv::v_reinterpret_as_s16(cv::v_pack(ay2, ay3));

      const cv::v_int16x8 c128 = cv::v_setall_s16(128);
      cv::v_int16x8 dcr =
          cv::v_reinterpret_as_s16(cv::v_load_expand(pcr + xh)) - c128;
      cv::v_int16x8 dcb =
          cv::v_reinterpret_as_s16(cv::v_load_expand(pcb + xh)) - c128;

      // BT.601: B += (454*dCb)>>8; G -= (183*dCr + 88*dCb)>>8;
      //         R += (359*dCr)>>8   (i32 to avoid i16 overflow).
      cv::v_int32x4 dcr0, dcr1, dcb0, dcb1;
      cv::v_expand(dcr, dcr0, dcr1);
      cv::v_expand(dcb, dcb0, dcb1);
      const cv::v_int32x4 k454 = cv::v_setall_s32(454);
      const cv::v_int32x4 k183 = cv::v_setall_s32(183);
      const cv::v_int32x4 k88 = cv::v_setall_s32(88);
      const cv::v_int32x4 k359 = cv::v_setall_s32(359);
      cv::v_int16x8 tb = cv::v_pack((dcb0 * k454) >> 8, (dcb1 * k454) >> 8);
      cv::v_int16x8 tg = cv::v_pack((dcr0 * k183 + dcb0 * k88) >> 8,
                                    (dcr1 * k183 + dcb1 * k88) >> 8);
      cv::v_int16x8 tr = cv::v_pack((dcr0 * k359) >> 8, (dcr1 * k359) >> 8);

      cv::v_int16x8 tb_lo, tb_hi, tg_lo, tg_hi, tr_lo, tr_hi;
      cv::v_zip(tb, tb, tb_lo, tb_hi);
      cv::v_zip(tg, tg, tg_lo, tg_hi);
      cv::v_zip(tr, tr, tr_lo, tr_hi);

      cv::v_uint8x16 outb = cv::v_pack_u(yn_lo + tb_lo, yn_hi + tb_hi);
      cv::v_uint8x16 outg = cv::v_pack_u(yn_lo - tg_lo, yn_hi - tg_hi);
      cv::v_uint8x16 outr = cv::v_pack_u(yn_lo + tr_lo, yn_hi + tr_hi);
      cv::v_store_interleave(pc + 3 * x0, outb, outg, outr);
    }
#endif
    for (; xh < half.width; ++xh) {
      const int a = pa[xh >> 1];
      const int b256 = pb[xh >> 1] << 8;
      const int dcr = static_cast<int>(pcr[xh]) - 128;
      const int dcb = static_cast<int>(pcb[xh]) - 128;
      const int tb = (454 * dcb) >> 8;
      const int tg = (183 * dcr + 88 * dcb) >> 8;
      const int tr = (359 * dcr) >> 8;
      const int x0 = 2 * xh;
      for (int k = 0; k < 2; ++k) {
        const int yn = (a * py[x0 + k] + b256 + 2048) >> 12;
        uchar *o = pc + 3 * (x0 + k);
        const int vb = yn + tb, vg = yn - tg, vr = yn + tr;
        o[0] = static_cast<uchar>(vb < 0 ? 0 : (vb > 255 ? 255 : vb));
        o[1] = static_cast<uchar>(vg < 0 ? 0 : (vg > 255 ? 255 : vg));
        o[2] = static_cast<uchar>(vr < 0 ? 0 : (vr > 255 ? 255 : vr));
      }
    }
  }
}

inline void SuppressFalseColor(cv::Mat &color) {
  SuppressFalseColorImpl(nullptr, color);
}

// Per-stage ISP timing. Enable with CYPERSTEREO_ISP_PROFILE=1; averaged
// per-camera stats are printed every kReportEvery frames.
class IspProfiler {
 public:
  static bool Enabled() {
    static const bool enabled =
        std::getenv("CYPERSTEREO_ISP_PROFILE") != nullptr;
    return enabled;
  }
  static void Add(const char *name, double wb_ms, double demosaic_ms,
                  double post_ms) {
    static std::mutex mtx;
    static std::map<std::string, Acc> accs;
    constexpr int kReportEvery = 300;
    std::lock_guard<std::mutex> lock(mtx);
    Acc &a = accs[name];
    a.wb += wb_ms;
    a.demosaic += demosaic_ms;
    a.post += post_ms;
    if (++a.n >= kReportEvery) {
      std::cout << "[isp] " << name << std::fixed << std::setprecision(2)
                << "  wb " << a.wb / a.n << "  demosaic " << a.demosaic / a.n
                << "  post " << a.post / a.n << "  total "
                << (a.wb + a.demosaic + a.post) / a.n << " ms (avg of " << a.n
                << ")" << std::endl;
      a = Acc{};
    }
  }

 private:
  struct Acc {
    double wb{0}, demosaic{0}, post{0};
    int n{0};
  };
};

inline void ApplyISP(cv::Mat &raw, cv::Mat &color, WhiteBalance &wb,
                     const char *name) {
  // Priority and board-aware core affinity only need to be applied once per
  // thread; pthread_setschedparam is a syscall we don't want 4x per frame.
  static thread_local bool sched_done = false;
  if (!sched_done) {
    sched_done = true;
    ApplyThreadPriority(ThreadRole::kWorker, name);
#if defined(__x86_64__) || defined(__i386__) || \
    defined(CYPERSTEREO_RK3588)
    // name is "wb-camN" (N=1..4). x86 uses physical cores CPU0-3;
    // RK3588 uses the four Cortex-A76 performance cores CPU4-7. The latter
    // is enabled only by -DTARGET_BOARD=rk3588 so other big.LITTLE layouts
    // are never guessed.
    const size_t len = std::strlen(name);
    if (len > 0) {
      const char last = name[len - 1];
      if (last >= '1' && last <= '4') {
#if defined(CYPERSTEREO_RK3588)
        PinThreadToCpu(4 + last - '1');
#else
        PinThreadToCpu(last - '1');
#endif
      }
    }
#endif
  }
  // ARM: EA demosaic runs fused with the YCbCr front-end inside
  // SuppressFalseColorImpl (bit-exact vs cvtColor(EA) + front; the BGR
  // intermediate is never materialized). Its cost is accounted under
  // "post" in the profile; "demosaic" reads ~0 on that path.
  const bool fused_demosaic = UseFusedDemosaic();
  if (!IspProfiler::Enabled()) {
    wb.Apply(raw);
    if (fused_demosaic) {
      SuppressFalseColorImpl(&raw, color);
    } else {
      cv::cvtColor(raw, color, cv::COLOR_BayerRG2BGR_EA);
      SuppressFalseColor(color);
    }
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  wb.Apply(raw);
  const auto t1 = std::chrono::steady_clock::now();
  if (!fused_demosaic)
    cv::cvtColor(raw, color, cv::COLOR_BayerRG2BGR_EA);
  const auto t2 = std::chrono::steady_clock::now();
  if (fused_demosaic)
    SuppressFalseColorImpl(&raw, color);
  else
    SuppressFalseColor(color);
  const auto t3 = std::chrono::steady_clock::now();
  const auto ms = [](std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  IspProfiler::Add(name, ms(t1 - t0), ms(t2 - t1), ms(t3 - t2));
}

// One long-lived ISP worker thread. Prefer this over per-frame
// std::thread spawn/join (~0.3-0.5 ms) so RT priority / core affinity
// set inside ApplyISP stick across frames.
class IspWorker {
 public:
  IspWorker() {
    thr_ = std::thread([this] {
      for (;;) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_go_.wait(lk, [this] { return has_job_; });
        Job j = job_;
        lk.unlock();
        ApplyISP(*j.raw, *j.color, *j.wb, j.name);
        lk.lock();
        has_job_ = false;
        lk.unlock();
        cv_done_.notify_one();
      }
    });
    thr_.detach();
  }

  void Submit(cv::Mat &raw, cv::Mat &color, WhiteBalance &wb,
              const char *name) {
    std::lock_guard<std::mutex> lk(mtx_);
    job_ = Job{&raw, &color, &wb, name};
    has_job_ = true;
    cv_go_.notify_one();
  }

  void Wait() {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_done_.wait(lk, [this] { return !has_job_; });
  }

 private:
  struct Job {
    cv::Mat *raw;
    cv::Mat *color;
    WhiteBalance *wb;
    const char *name;
  };
  std::mutex mtx_;
  std::condition_variable cv_go_, cv_done_;
  Job job_{};
  bool has_job_ = false;
  std::thread thr_;
};

struct IspJob {
  cv::Mat *raw;
  cv::Mat *color;
  WhiteBalance *wb;
  const char *name;

  IspJob(cv::Mat &r, cv::Mat &c, WhiteBalance &w, const char *n)
      : raw(&r), color(&c), wb(&w), name(n) {}
};

// jobs[0] runs on the calling thread; jobs[1..] on persistent workers.
inline void ApplyISPParallel(const IspJob *jobs, int n) {
  constexpr int kMaxWorkers = 7;
  static IspWorker workers[kMaxWorkers];
  if (n <= 0) return;
  if (n > kMaxWorkers + 1) n = kMaxWorkers + 1;
  for (int i = 1; i < n; ++i)
    workers[i - 1].Submit(*jobs[i].raw, *jobs[i].color, *jobs[i].wb,
                           jobs[i].name);
  ApplyISP(*jobs[0].raw, *jobs[0].color, *jobs[0].wb, jobs[0].name);
  for (int i = 1; i < n; ++i) workers[i - 1].Wait();
}

inline void ApplyISPParallel(std::initializer_list<IspJob> jobs) {
  ApplyISPParallel(jobs.begin(), static_cast<int>(jobs.size()));
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
  // Never throw on timeout: the capture layer restarts the stream (and, if
  // the driver is wedged, reopens the device node) on its own.  Throwing
  // here turned a recoverable multi-second USB outage into a process abort
  // (observed on Orange Pi 5: EPROTO burst -> storm restart -> uncaught
  // "Timeout waiting for frame").  Warn periodically instead.
  int waited_s = 0;
  while (!frame_info.con.wait_for(lock, std::chrono::seconds(5), frame_ready)) {
    waited_s += 5;
    std::cout << "[api] WARN no frame for " << waited_s
              << " s, capture layer is retrying"
              << "  (check USB link / dmesg if this persists)" << std::endl;
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

        const CameraProfile &prof = frame_info.profile;
        FrameStreamData &fs = frame_info.framestream;

        cv::Mat img(prof.frame_height, prof.frame_width, CV_8UC2,
                    const_cast<void *>(data));

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
          bool marker_ok =
              (ver0 == prof.hardware_version) && (ver1 == prof.software_version);

          // MT9V034 family (2-cam, 752x480): hardware 0=M150 / 1=M60, and
          // software 1 (older FX3 image) or 2 (current). Layout is the same;
          // accept any of these markers and latch onto the live profile so
          // subsequent frames are not rejected (no-SN units often report
          // 0/1 which matches neither of the baked-in 0/2 or 1/2 expects).
          if (!marker_ok && prof.num_cameras < 4 &&
              (ver0 == kProfileM150.hardware_version ||
               ver0 == kProfileM60.hardware_version) &&
              (ver1 == 1 || ver1 == kProfileM60.software_version)) {
            if (ver0 != frame_info.profile.hardware_version ||
                ver1 != frame_info.profile.software_version) {
              frame_info.profile.hardware_version = ver0;
              frame_info.profile.software_version = ver1;
              frame_info.profile.name = (ver0 == kProfileM150.hardware_version)
                                            ? kProfileM150.name
                                            : kProfileM60.name;
              std::cout << "[api] MT9V034 variant from metadata: "
                        << frame_info.profile.name << " marker=" << ver0 << "/"
                        << ver1 << std::endl;
            }
            marker_ok = true;
          }

          {
            static int dbg = 0;
            if (dbg < 5) {
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

          if (!marker_ok) {
            // Full-length transfer whose content is misaligned stream data
            // (FX3 framing offset): the metadata row is garbage and the
            // pixels belong to two different frames.  Reject the frame here
            // so consumers never receive image_timestamp=0 / corrupt images.
            // Raw metadata is still recorded in the replay ring for ts-diag.
            ++frame_info.meta_bad_count;
            FrameDiagRec &bad_rec =
                frame_info.diag_ring[frame_info.frame_seq % FrameInfo::kDiagRingLen];
            bad_rec = FrameDiagRec{};
            bad_rec.seq = frame_info.frame_seq;
            bad_rec.host_ms = frame_info.host_gap_ms;
            bad_rec.img_hour = meta_u(2);
            bad_rec.img_ms = meta_u(3);
            bad_rec.img_s = meta_u(4);
            bad_rec.image_ts = 0.0;
            bad_rec.imu_n = prof.imu_samples_per_frame;
            for (int i = 0; i < prof.imu_samples_per_frame; ++i) {
              const int base = kMetaImuBaseCol + i * kImuWordsPerSample;
              bad_rec.imu_ms[i] = meta_u(base + 0);
              bad_rec.imu_s[i] = meta_u(base + 1);
            }
            ++frame_info.frame_seq;

            // Throttled logging: first 3 per 5 s window, then a summary count.
            const auto now = std::chrono::steady_clock::now();
            if (frame_info.meta_bad_window_start.time_since_epoch().count() == 0 ||
                now - frame_info.meta_bad_window_start > std::chrono::seconds(5)) {
              if (frame_info.meta_bad_suppressed > 0) {
                std::cout << "[api] meta-bad burst: "
                          << frame_info.meta_bad_suppressed
                          << " further marker-mismatch frames suppressed"
                          << "  (total=" << frame_info.meta_bad_count << ")"
                          << std::endl;
              }
              frame_info.meta_bad_window_start = now;
              frame_info.meta_bad_logged_in_window = 0;
              frame_info.meta_bad_suppressed = 0;
            }
            if (frame_info.meta_bad_logged_in_window < 3) {
              ++frame_info.meta_bad_logged_in_window;
              std::cout << "[api] BAD FRAME skipped (metadata marker mismatch):"
                        << " ver0=" << ver0 << " ver1=" << ver1
                        << " (expect " << prof.hardware_version << "/"
                        << prof.software_version << ")"
                        << "  total=" << frame_info.meta_bad_count << std::endl;
            } else {
              ++frame_info.meta_bad_suppressed;
            }

            if (continuation) continuation();
            return;
          }

          fs.hardware_version = static_cast<uint32_t>(ver0);
          fs.software_version = static_cast<uint32_t>(ver1);
          double image_count_hour = meta_u(2);
          double image_count_ms = meta_u(3);
          double image_count_s = meta_u(4);

          fs.image_timestamp =
              image_count_hour * 12 * 3600 + image_count_s + image_count_ms / 10000.0;

          // Record this frame's raw metadata into the replay ring (image
          // timestamp fields now; per-sample IMU fields added just below,
          // before any anomaly can fire on them).
          FrameDiagRec &rec =
              frame_info.diag_ring[frame_info.frame_seq % FrameInfo::kDiagRingLen];
          rec = FrameDiagRec{};
          rec.seq = frame_info.frame_seq;
          rec.host_ms = frame_info.host_gap_ms;
          rec.img_hour = static_cast<int>(image_count_hour);
          rec.img_ms = static_cast<int>(image_count_ms);
          rec.img_s = static_cast<int>(image_count_s);
          rec.image_ts = fs.image_timestamp;
          rec.imu_n = prof.imu_samples_per_frame;
          for (int i = 0; i < prof.imu_samples_per_frame; ++i) {
            const int base = kMetaImuBaseCol + i * kImuWordsPerSample;
            rec.imu_ms[i] = meta_u(base + 0);
            rec.imu_s[i] = meta_u(base + 1);
          }

          auto print_diag_rec = [](const FrameDiagRec &r) {
            std::cout << "[ts-diag] seq=" << r.seq << std::fixed
                      << std::setprecision(1) << "  host_gap=" << r.host_ms
                      << "ms  img(h/s/sub)=" << r.img_hour << "/" << r.img_s
                      << "/" << r.img_ms << std::setprecision(4)
                      << "  img_ts=" << r.image_ts << "  imu(s/sub):";
            for (int i = 0; i < r.imu_n; ++i)
              std::cout << " " << r.imu_s[i] << "/" << r.imu_ms[i];
            std::cout << std::endl;
          };
          bool anomaly = false;

          if (frame_info.last_image_timestamp > 0.0) {
            const double image_gap =
                fs.image_timestamp - frame_info.last_image_timestamp;
            // Flag BOTH directions: a backward step (negative gap) was
            // previously silent, hiding the "stale timestamps then snap
            // forward" failure mode (the forward snap alone looks identical
            // to a genuine counter jump).
            if (std::fabs(image_gap) > kImageGapThresholdSec) {
              anomaly = true;
              ++frame_info.image_drop_count;
              std::cout << "[drop] image gap=" << std::fixed
                        << std::setprecision(3) << image_gap * 1000.0
                        << " ms (threshold " << kImageGapThresholdSec * 1000.0
                        << " ms)  prev_ts=" << frame_info.last_image_timestamp
                        << "  cur_ts=" << fs.image_timestamp
                        << "  host_gap=" << frame_info.host_gap_ms << " ms"
                        << (image_gap < 0 ? "  [BACKWARD jump]" : "")
                        << std::endl;
              const uint64_t discarded =
                  frame_info.meta_bad_count - frame_info.meta_bad_at_last_good;
              if (image_gap < 0 && fs.image_timestamp < 60.0 &&
                  frame_info.last_image_timestamp > 120.0) {
                // Timestamp restarted from near zero after running for
                // minutes: the CAMERA rebooted (FPGA delay_reset cycled,
                // which also hard-resets the FX3 via USB_RESET_N).  Seen as
                // ENODEV + [fx3-stats] counters restarting in the same event.
                std::cout << "[drop] note: timestamp restarted from ~0 -> "
                             "camera HARDWARE REBOOT (FPGA reconfigured / "
                             "power dip; FX3 reset by FPGA USB_RESET_N)"
                          << std::endl;
              } else if (discarded > 0) {
                // Frames DID keep arriving but were corrupt (marker mismatch)
                // and rejected -- a framing storm.  host_gap stays near the
                // frame period, so without this check the case below would
                // mislabel it a timestamp jump.
                std::cout << "[drop] note: gap spans " << discarded
                          << " discarded corrupt frames (framing storm) -> "
                             "REAL data loss, frames rejected not missing"
                          << std::endl;
              } else if (frame_info.host_gap_ms >= 0.0 &&
                  frame_info.host_gap_ms < std::fabs(image_gap) * 1000.0 * 0.5) {
                // Camera-timestamp gap vs host arrival gap disagree by a lot
                // -> frames kept flowing and only the embedded timestamp
                // leapt.
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
          frame_info.meta_bad_at_last_good = frame_info.meta_bad_count;

          // IMU (and GNSS) metadata samples carry only second+subsecond; the
          // 12-hour period counter is borrowed from the IMAGE metadata.  At
          // the 43200s rollover a frame can straddle the wrap: e.g. image
          // fields still hour=0/s=43199 while a later IMU sample in the same
          // frame has already wrapped to s=0 (or the reverse).  Blindly using
          // the image hour then mis-places that sample by a full 43200s.
          // Correct it by snapping the reconstructed timestamp to within half
          // a period (21600s) of the image timestamp.
          constexpr double kHalfDaySec = 12 * 3600;  // one hour-counter period
          auto wrap_fix = [&](double ts) {
            if (ts - fs.image_timestamp > kHalfDaySec / 2) return ts - kHalfDaySec;
            if (fs.image_timestamp - ts > kHalfDaySec / 2) return ts + kHalfDaySec;
            return ts;
          };

          // prof.imu_samples_per_frame is the MAX slots the layout reserves;
          // the genuine count varies per frame. Walk the slots and accept a
          // sample only while both the timestamp and the temperature stay
          // continuous; the first slot that breaks continuity ends the run
          // (stale/held-over slots only ever trail the genuine samples).
          const int imu_slots = prof.imu_samples_per_frame;
          int valid_samples = 0;
          double prev_ts = -1.0;
          double prev_temp = frame_info.last_imu_temperature;
          for (int i = 0; i < imu_slots; ++i) {
            const int base = kMetaImuBaseCol + i * kImuWordsPerSample;
            const double imu_count_ms = meta_u(base + 0);
            const double imu_count_s = meta_u(base + 1);
            const double ts = wrap_fix(
                image_count_hour * 12 * 3600 + imu_count_s + imu_count_ms / 10000.0);
            double t = meta_s(base + 8);
            if (t > 1023) t -= 2048;
            const double temp = t * 0.125 + 23;

            // (a) timestamp continuity: near the image timestamp, and a small
            //     forward step from the previous accepted sample.
            bool ts_ok =
                std::fabs(ts - fs.image_timestamp) <= kImuTsToImageWindowSec;
            if (ts_ok && prev_ts > 0.0) {
              const double d = ts - prev_ts;
              ts_ok = (d > 0.0) && (d <= kImuSampleStepMaxSec);
            }
            // (b) temperature plausibility + slow-change constraint.
            bool temp_ok = (temp >= kImuTempMinC && temp <= kImuTempMaxC);
            if (temp_ok && prev_temp > -900.0)
              temp_ok = std::fabs(temp - prev_temp) <= kImuTempStepC;

            if (!(ts_ok && temp_ok)) break;

            const int k = valid_samples;
            fs.imu.imu_timestamp[k] = ts;
            fs.imu.acc_x[k] = meta_s(base + 2) * BMI088_ACCEL_SEN;
            fs.imu.acc_y[k] = meta_s(base + 3) * BMI088_ACCEL_SEN;
            fs.imu.acc_z[k] = meta_s(base + 4) * BMI088_ACCEL_SEN;
            fs.imu.gyro_x[k] = meta_s(base + 5) * BMI088_GYRO_SEN;
            fs.imu.gyro_y[k] = meta_s(base + 6) * BMI088_GYRO_SEN;
            fs.imu.gyro_z[k] = meta_s(base + 7) * BMI088_GYRO_SEN;
            fs.imu.temperature[k] = temp;
            prev_ts = ts;
            prev_temp = temp;
            ++valid_samples;
          }
          fs.imu.imu_count = valid_samples;
          if (valid_samples > 0) frame_info.last_imu_temperature = prev_temp;
          rec.imu_n = valid_samples;

          for (int i = 0; i < fs.imu.imu_count; ++i) {
            const double ts = fs.imu.imu_timestamp[i];
            if (frame_info.last_imu_timestamp > 0.0) {
              const double imu_gap = ts - frame_info.last_imu_timestamp;
              if (std::fabs(imu_gap) > kImuGapThresholdSec) {
                anomaly = true;
                ++frame_info.imu_drop_count;
                std::cout << "[drop] imu gap=" << std::fixed
                          << std::setprecision(3) << imu_gap * 1000.0
                          << " ms (threshold " << kImuGapThresholdSec * 1000.0
                          << " ms)  prev_ts=" << frame_info.last_imu_timestamp
                          << "  cur_ts=" << ts << "  sample_idx=" << i
                          << "  host_gap=" << frame_info.host_gap_ms << " ms"
                          << (imu_gap < 0 ? "  [BACKWARD jump]" : "")
                          << std::endl;
              }
            }
            frame_info.last_imu_timestamp = ts;
          }

          // On any timestamp anomaly: replay the raw metadata of the frames
          // BEFORE it (from the ring) and keep dumping the frames AFTER it,
          // so the log shows whether the jump is a single step between two
          // adjacent IMU samples (real counter jump), a stretch of stale
          // values that snaps back (FIFO/reset glitch in FPGA), or garbage
          // fields (metadata corruption).
          if (anomaly) {
            std::cout << "[ts-diag] ---- anomaly at seq=" << frame_info.frame_seq
                      << ", replaying last frames ----" << std::endl;
            const uint64_t cur = frame_info.frame_seq;
            const uint64_t first =
                cur >= FrameInfo::kDiagRingLen - 1 ? cur - (FrameInfo::kDiagRingLen - 1) : 0;
            for (uint64_t s = first; s <= cur; ++s) {
              const FrameDiagRec &r =
                  frame_info.diag_ring[s % FrameInfo::kDiagRingLen];
              if (r.seq == s) print_diag_rec(r);
            }
            frame_info.dump_frames_left = 6;
          } else if (frame_info.dump_frames_left > 0) {
            print_diag_rec(rec);
            if (--frame_info.dump_frames_left == 0)
              std::cout << "[ts-diag] ---- post-anomaly dump end ----" << std::endl;
          }
          ++frame_info.frame_seq;

          {
            const int gnss_shift = prof.gnss_base_col * 2 - 96;
            auto g8 = [&](int ref_byte) -> int {
              const int b = ref_byte + gnss_shift;
              return (b & 1) ? hi[b >> 1] : lo[b >> 1];
            };

            double gnss_count_ms = ((g8(97) << 8) | g8(96));
            double gnss_count_s = ((g8(99) << 8) | g8(98));
            // Same 43200s-rollover correction as the IMU samples above.
            double gnss_timestamp = wrap_fix(
                image_count_hour * 12 * 3600 + gnss_count_s + gnss_count_ms / 10000.0);

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
          }
        }

        // Publish only after validation: marker-mismatch frames returned
        // above and were requeued without ever becoming visible here.
        if (frame_info.frame == nullptr) {
          frame_info.frame = std::make_shared<struct Frame>();
        } else {
          if (frame_info.frame->continuation) {
            frame_info.frame->continuation();
          }
        }
        frame_info.frame->data = data;
        frame_info.frame->continuation = continuation;

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
