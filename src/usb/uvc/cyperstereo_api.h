#ifndef CYPERSTEREO_API_H_
#define CYPERSTEREO_API_H_

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
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
#include "bayer_format.h"
#include "tic_toc.h"
#include "thread_priority.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
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

// Opt-in single-frame scheduler for RK3588's four Cortex-A76 cores. Three
// persistent helpers are pinned to CPU4..6; the submitting thread executes
// shard 3 on CPU7. No worker is spawned per frame or per stage.
class FastIspStagePool {
 public:
  FastIspStagePool() {
    try {
      for (int i = 0; i < 3; ++i)
        workers_[i] = std::thread([this, i] { WorkerLoop(i); });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        ++generation_;
      }
      go_.notify_all();
      for (auto &worker : workers_)
        if (worker.joinable()) worker.join();
      throw;
    }
  }

  ~FastIspStagePool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    go_.notify_all();
    for (auto &worker : workers_)
      if (worker.joinable()) worker.join();
  }

  FastIspStagePool(const FastIspStagePool &) = delete;
  FastIspStagePool &operator=(const FastIspStagePool &) = delete;

  bool TryAcquireFrame() {
    std::lock_guard<std::mutex> lock(mode_mutex_);
    if (batch_users_ != 0 || frame_active_) return false;
    frame_active_ = true;
    return true;
  }

  void ReleaseFrame() {
    {
      std::lock_guard<std::mutex> lock(mode_mutex_);
      frame_active_ = false;
    }
    mode_changed_.notify_all();
  }

  void BeginMultiCameraBatch() {
    std::unique_lock<std::mutex> lock(mode_mutex_);
    ++batch_users_;
    mode_changed_.wait(lock, [this] { return !frame_active_; });
  }

  void EndMultiCameraBatch() {
    {
      std::lock_guard<std::mutex> lock(mode_mutex_);
      --batch_users_;
    }
    mode_changed_.notify_all();
  }

  template <class Function>
  void Run(Function function) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_ = function;
      error_ = nullptr;
      pending_ = 3;
      ++generation_;
    }
    go_.notify_all();
    std::exception_ptr caller_error;
    try {
      function(3);
    } catch (...) {
      caller_error = std::current_exception();
    }
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
    const std::exception_ptr helper_error = error_;
    job_ = nullptr;
    lock.unlock();
    if (caller_error) std::rethrow_exception(caller_error);
    if (helper_error) std::rethrow_exception(helper_error);
  }

 private:
  void WorkerLoop(int worker) {
    ApplyThreadPriority(ThreadRole::kWorker, "isp-frame-shard");
#if defined(CYPERSTEREO_RK3588)
    PinThreadToCpu(4 + worker);
#endif
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      go_.wait(lock, [this, seen] {
        return stop_ || generation_ != seen;
      });
      if (stop_) return;
      seen = generation_;
      const std::function<void(int)> job = job_;
      lock.unlock();
      std::exception_ptr error;
      try {
        job(worker);
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();
      if (error && !error_) error_ = error;
      --pending_;
      if (pending_ == 0) done_.notify_one();
    }
  }

  std::array<std::thread, 3> workers_;
  std::mutex mutex_;
  std::condition_variable go_, done_;
  std::mutex mode_mutex_;
  std::condition_variable mode_changed_;
  std::function<void(int)> job_;
  std::exception_ptr error_;
  uint64_t generation_ = 0;
  int pending_ = 0;
  int batch_users_ = 0;
  bool stop_ = false;
  bool frame_active_ = false;
};

inline bool FastIspSingleFrameParallelEnabled() {
#if defined(CYPERSTEREO_RK3588) && defined(__aarch64__)
  static const bool enabled = [] {
    const char *value = std::getenv("CYPERSTEREO_FAST_INTRA_FRAME");
    return value && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "false") != 0;
  }();
  return enabled;
#else
  return false;
#endif
}

inline FastIspStagePool &GetFastIspStagePool() {
  static FastIspStagePool pool;
  return pool;
}

inline bool &FastIspFrameParallelActive() {
  static thread_local bool active = false;
  return active;
}

inline void FastIspShardRange(int count, int shard,
                              int &begin, int &end) {
  begin = count * shard / 4;
  end = count * (shard + 1) / 4;
}

class FastIspFrameParallelGuard {
 public:
  FastIspFrameParallelGuard() {
    if (!FastIspSingleFrameParallelEnabled()) return;
    FastIspStagePool &candidate = GetFastIspStagePool();
    if (!candidate.TryAcquireFrame()) return;
    pool_ = &candidate;
    FastIspFrameParallelActive() = true;
#if defined(CYPERSTEREO_RK3588)
    static thread_local bool pinned = false;
    if (!pinned) {
      PinThreadToCpu(7);
      pinned = true;
    }
#endif
  }

  ~FastIspFrameParallelGuard() {
    if (!pool_) return;
    FastIspFrameParallelActive() = false;
    pool_->ReleaseFrame();
  }

  FastIspFrameParallelGuard(const FastIspFrameParallelGuard &) = delete;
  FastIspFrameParallelGuard &operator=(const FastIspFrameParallelGuard &) =
      delete;

 private:
  FastIspStagePool *pool_ = nullptr;
};

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

// SmartSens SC136HGS row(line) time in the 30Hz continuous build:
//   Current FPGA table writes HTS = 0x016a = 362 at 15 MHz EXTCLK.
//   Relative to the characterized 364-row/24us timing:
//   Tline = 24us * 362 / 364 ~= 23.868132us.
//   exposure_time[s] = exposure_lines * kSmartSensLineTimeSec.
static constexpr double kSmartSensLineTimeSec = 23.868131868e-6;

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
	double image_timestamp;   // exposure END, common to all sensors (seconds)

  // Per-display-plane AE telemetry, parsed from the SmartSens metadata row.
  // Index: 0=image1(left), 1=image2(right), 2=image3(left_front),
  // 3=image4(right_front).  SmartSens 4-cam only; left 0 for the MT9V families.
  // Each sensor runs its own AEC, so these generally differ between cameras.
  // These are the FPGA AEC targets sent to I2C, not sensor-register readback.
  uint16_t exposure_lines[4]{};          // target exposure, in rows
  double   exposure_time[4]{};           // seconds = exposure_lines * Tline
  double   camera_temperature[4]{};      // Celsius, from sensor reg {0x4c10,0x4c11}
  double   camera_gain[4]{};             // target real gain (x), code / 64

  // Per-camera capture timestamp taken at the exposure MIDPOINT:
  //   image_midpoint_timestamp[i] = image_timestamp - exposure_time[i]/2.
  // All four share the common exposure END (image_timestamp), but each has its
  // own exposure length, so each midpoint differs.
  double   image_midpoint_timestamp[4]{};
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

// Sensor pedestal in the 8-bit domain (SC136HGS BLC target after the
// FPGA's 10->8 bit truncation). Subtracted in the WB LUT; also used by
// the WB gain estimator so the g/b, g/r ratios are pedestal-free.
constexpr double kIspBlackLevel = 16.0;

// Output tone pipeline, folded into the WB LUT so it costs nothing per
// pixel. Canonical ISP order (openISP, in-camera rendering tutorials):
// full BLC in linear -> WB gains -> ... -> encode curve -> photo-finishing
// tone curve. The sensor output is LINEAR; shown as-is on an sRGB display
// a 10%-luminance surface renders as ~26/255, which is why shadows looked
// crushed even though the data was there.
//
// Stage 1, encode curve: sRGB piece-wise (IEC 61966-2-1) by default, NOT a
// pure power gamma. The linear toe (slope 12.92 below 0.0031308) exists
// precisely to limit noise amplification at black; the pure 1/2.2 power
// curve tried first has infinite slope at 0 and lifted the read-noise
// floor into a gray haze (the "washed out" A/B result).
// Stage 2, photo-finishing: black point + gentle smoothstep S-curve, the
// standard fix for the flat look of encode-gamma-only output.
//
// Env knobs (all cached once):
//   CYPERSTEREO_GAMMA:      unset = sRGB encode (default); 1.0 = legacy
//                           fully-linear output; 1.0<g<=4.0 = pure power
//                           1/g encode (A/B experiments).
//   CYPERSTEREO_BLACKPOINT: encoded-domain black point, default 6,
//                           range 0..32. Higher = deeper blacks.
//   CYPERSTEREO_CONTRAST:   S-curve blend 0..1, default 0.3. 0 disables.
inline double IspGamma() {
  static const double g = [] {
    if (const char *e = std::getenv("CYPERSTEREO_GAMMA")) {
      const double v = std::atof(e);
      if (v == 1.0) return 1.0;
      if (v > 1.0 && v <= 4.0) return v;
    }
    return 0.0;  // 0 = sRGB piece-wise encode
  }();
  return g;
}

inline double IspBlackPoint() {
  static const double bp = [] {
    if (const char *e = std::getenv("CYPERSTEREO_BLACKPOINT")) {
      const double v = std::atof(e);
      if (v >= 0.0 && v <= 32.0) return v;
    }
    return 6.0;
  }();
  return bp;
}

inline double IspContrast() {
  static const double a = [] {
    if (const char *e = std::getenv("CYPERSTEREO_CONTRAST")) {
      const double v = std::atof(e);
      if (v >= 0.0 && v <= 1.0) return v;
    }
    return 0.3;
  }();
  return a;
}

// Global chroma saturation gain (the HSC block of a standard ISP). With
// the CCM stage below now doing the colorimetric work in the linear
// domain, this is only a YUV-domain fine-trim knob; default 1.0 (off).
// Env CYPERSTEREO_SATURATION: range 0.5..1.99 (Q7 limit).
inline double IspSaturation() {
  static const double s = [] {
    if (const char *e = std::getenv("CYPERSTEREO_SATURATION")) {
      const double v = std::atof(e);
      if (v >= 0.5 && v <= 1.99) return v;
    }
    return 1.0;
  }();
  return s;
}

// ---------------------------------------------------------------------------
// CCM: 3x3 color correction matrix, the canonical post-demosaic color stage
// (BLC -> WB -> demosaic -> CCM in LINEAR RGB -> encode/tone -> YUV). This
// is the module that actually restores saturation and hue: the sensor's
// color filters overlap spectrally (a "red" photosite also responds to
// green light), so raw colors are globally mixed toward gray until the
// matrix unmixes them. Rows sum to 1 so neutrals (and thus WB) survive.
//
// Env:
//   CYPERSTEREO_CCM      "off" = identity, or 9 R-major values
//                        "rr,rg,rb,gr,gg,gb,br,bg,bb" (rows applied to
//                        [R,G,B]); overrides CCM_SAT. Obtain calibrated
//                        values with tools/ccm_calibrate.py and a 24-patch
//                        ColorChecker capture.
//   CYPERSTEREO_CCM_SAT  legacy fallback (generic saturation matrix) when
//                        CYPERSTEREO_CCM is unset; ignored if the baked-in
//                        default below is used.
//
// Default: calibrated from ColorChecker capture 741.png (SC136HGS, linear
// Bayer->RGB, tools/ccm_calibrate.py). Override with CYPERSTEREO_CCM.
inline const cv::Matx33f &CcmMatrixBgr() {
  static const cv::Matx33f m = [] {
    double rgb[3][3];
    bool have = false;
    if (const char *e = std::getenv("CYPERSTEREO_CCM")) {
      double v[9];
      int n = 0;
      const char *p = e;
      char *end = nullptr;
      while (n < 9) {
        const double d = std::strtod(p, &end);
        if (end == p) break;
        v[n++] = d;
        p = end;
        while (*p == ',' || *p == ';' || *p == ' ') ++p;
      }
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          rgb[i][j] = (n == 9) ? v[3 * i + j] : (i == j ? 1.0 : 0.0);
      have = true;  // "off" or unparseable input -> identity
    }
    if (!have) {
      // R-major CCM from 741.png calibration (20/24 patches, fit err ~48 DN).
      static const double kDefaultCcm[9] = {
          1.7020, -0.6295, -0.0725, -0.4929, 1.7391, -0.2462,
          0.1409, -0.7787, 1.6378,
      };
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          rgb[i][j] = kDefaultCcm[3 * i + j];
    }
    // Frames are BGR: remap the R-major matrix to BGR in/out order.
    cv::Matx33f bgr;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        bgr(i, j) = static_cast<float>(rgb[2 - i][2 - j]);
    return bgr;
  }();
  return m;
}

inline bool CcmIsIdentity() {
  static const bool ident = [] {
    const cv::Matx33f &m = CcmMatrixBgr();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        if (std::fabs(m(i, j) - (i == j ? 1.f : 0.f)) > 1e-4f) return false;
    return true;
  }();
  return ident;
}

// True when the whole post-demosaic color/tone stage is a no-op, i.e. the
// original fully-linear pipeline. The ARM fused Bayer->YCbCr demosaic can
// only run in that mode (it never materializes the BGR frame the CCM and
// tone curve apply to).
inline bool IspLegacyLinear() { return IspGamma() == 1.0 && CcmIsIdentity(); }

// CCM (linear domain) + tone curve, applied to the demosaiced BGR frame.
// cv::transform is OpenCV's SIMD per-pixel 3x3 multiply with u8 saturation;
// the tone curve is one shared 256-entry cv::LUT. ~1-2 ms per 1.3 MP frame.
inline double ToneEncode(double linear_val);

inline void ApplyCcmTone(cv::Mat &bgr) {
  if (!CcmIsIdentity()) cv::transform(bgr, bgr, CcmMatrixBgr());
  if (IspGamma() == 1.0) return;  // legacy linear output: no tone curve
  static const cv::Mat tone_lut = [] {
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i)
      lut.at<uchar>(0, i) = cv::saturate_cast<uchar>(ToneEncode(i));
    return lut;
  }();
  cv::LUT(bgr, tone_lut, bgr);
}

// Maps a value from the legacy linear LUT-output domain ((i-16)*gain,
// full scale 255-16=239) to the final tone-mapped 0..255 output domain.
// Used both to build the WB LUT and to remap YUV-stage gate thresholds
// that were tuned on the linear pipeline.
inline double ToneEncode(double linear_val) {
  const double g = IspGamma();
  if (g == 1.0) return linear_val;  // bit-compatible legacy path
  if (linear_val <= 0.0) return 0.0;
  double n = linear_val / (255.0 - kIspBlackLevel);
  if (n > 1.0) n = 1.0;
  double v;
  if (g == 0.0) {
    v = n <= 0.0031308 ? 12.92 * n : 1.055 * std::pow(n, 1.0 / 2.4) - 0.055;
  } else {
    v = std::pow(n, 1.0 / g);
  }
  // Photo-finishing: re-anchor the black point (the encode toe still lifts
  // residual flare/noise a little), then a gentle S for midtone contrast.
  const double bp = IspBlackPoint() / 255.0;
  double t = (v - bp) / (1.0 - bp);
  if (t < 0.0) t = 0.0;
  const double s = t * t * (3.0 - 2.0 * t);  // smoothstep
  const double a = IspContrast();
  return 255.0 * ((1.0 - a) * t + a * s);
}

// Conservative tone used by the HDR quality path, expressed as a shared
// luminance gain for the fast-balanced path.  Keeping one gain for B/G/R is
// essential: applying the same nonlinear LUT to each channel independently
// lifts the smaller green component of a dark red pixel much more than red
// and rotates the hue toward orange.
struct FastBalancedToneLuts {
  std::array<uint16_t, 256> gain_q12{};
  std::array<uint16_t, 256> clip_q12{};
  // AVX2 gathers operate on 32-bit elements. Keeping widened mirrors avoids
  // scalar lane gathers in the fused output kernel.
  std::array<int32_t, 256> gain_q12_i32{};
  std::array<int32_t, 256> clip_q12_i32{};
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
  // AArch64 NEON has no vector gather. Collapse the two scalar LUT reads and
  // min into one lookup indexed by [max_color:luma]. Each entry stores the
  // signed Q15 multiplier for the gain above identity. gain==0 is encoded as
  // -32768, so value + SQRDMULH(value, -32768) is exactly zero.
  std::array<int16_t, 65536> gain_delta_q15{};
#endif
};

inline const FastBalancedToneLuts &GetFastBalancedToneLuts() {
  static const FastBalancedToneLuts luts = [] {
    FastBalancedToneLuts v;
    constexpr double kLinearFullScale8 = 255.0 - kIspBlackLevel;  // 239
    constexpr double kGamma = 1.20;
    constexpr double kMaxGain = 1.35;
    // HDR values 32..128 are in its 10-bit working range, hence 8..32 here.
    constexpr double kNoiseFloor8 = 8.0;
    constexpr double kRampEnd8 = 32.0;
    for (int y = 0; y < 256; ++y) {
      double n = y / kLinearFullScale8;
      if (n > 1.0) n = 1.0;
      double tone = n > 0.0 ? std::pow(n, 1.0 / kGamma) : 0.0;
      if (n > 0.0) tone = (std::min)(tone, n * kMaxGain);
      double ramp = (y - kNoiseFloor8) / (kRampEnd8 - kNoiseFloor8);
      ramp = (std::max)(0.0, (std::min)(ramp, 1.0));
      tone = n + ramp * (tone - n);
      const int out_y = (std::max)(0, (std::min)(
          255, static_cast<int>(tone * 255.0 + 0.5)));
      v.gain_q12[y] = static_cast<uint16_t>(
          y > 0 ? (out_y * 4096 + y / 2) / y : 0);
      v.clip_q12[y] = static_cast<uint16_t>(
          y > 0 ? (255 * 4096) / y : 65535);
      v.gain_q12_i32[y] = v.gain_q12[y];
      v.clip_q12_i32[y] = v.clip_q12[y];
    }
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
    for (int max_color = 0; max_color < 256; ++max_color) {
      for (int luma = 0; luma < 256; ++luma) {
        const int gain = (std::min)(
            static_cast<int>(v.gain_q12[luma]),
            static_cast<int>(v.clip_q12[max_color]));
        v.gain_delta_q15[(max_color << 8) | luma] =
            gain == 0 ? static_cast<int16_t>(-32768)
                      : static_cast<int16_t>((gain - 4096) << 3);
      }
    }
#endif
    return v;
  }();
  return luts;
}

// Apply the shared-luminance tone to one already-clipped BGR pixel. Keep
// this helper as the single source of truth for the reference full-frame
// pass and the fused output kernels below: reconstruction must clip first,
// then derive luma/max from those clipped values.
inline void ApplyFastBalancedTonePixel(
    uchar &blue, uchar &green, uchar &red,
    const FastBalancedToneLuts &luts) {
  const int b = blue, g = green, r = red;
  const int luma = (29 * b + 150 * g + 77 * r + 128) >> 8;
  const int max_color = (std::max)(b, (std::max)(g, r));
  const int gain_q12 = (std::min)(
      static_cast<int>(luts.gain_q12[luma]),
      static_cast<int>(luts.clip_q12[max_color]));
  blue = static_cast<uchar>((b * gain_q12 + 2048) >> 12);
  green = static_cast<uchar>((g * gain_q12 + 2048) >> 12);
  red = static_cast<uchar>((r * gain_q12 + 2048) >> 12);
}

#if CV_SIMD128
// Exact 16-pixel form of ApplyFastBalancedTonePixel. The final YCrCb->BGR
// kernel already has planar B/G/R vectors in registers, so applying the tone
// here avoids another full-frame interleaved read/deinterleave/write pass.
inline void ApplyFastBalancedTone16(
    cv::v_uint8x16 &blue, cv::v_uint8x16 &green, cv::v_uint8x16 &red,
    const FastBalancedToneLuts &luts) {
  cv::v_uint16x8 b_lo, b_hi, g_lo, g_hi, r_lo, r_hi;
  cv::v_expand(blue, b_lo, b_hi);
  cv::v_expand(green, g_lo, g_hi);
  cv::v_expand(red, r_lo, r_hi);
  const cv::v_uint16x8 k29 = cv::v_setall_u16(29);
  const cv::v_uint16x8 k150 = cv::v_setall_u16(150);
  const cv::v_uint16x8 k77 = cv::v_setall_u16(77);
  const cv::v_uint16x8 round128 = cv::v_setall_u16(128);
  const cv::v_uint16x8 y_lo =
      (b_lo * k29 + g_lo * k150 + r_lo * k77 + round128) >> 8;
  const cv::v_uint16x8 y_hi =
      (b_hi * k29 + g_hi * k150 + r_hi * k77 + round128) >> 8;
  const cv::v_uint16x8 max_lo = cv::v_max(b_lo, cv::v_max(g_lo, r_lo));
  const cv::v_uint16x8 max_hi = cv::v_max(b_hi, cv::v_max(g_hi, r_hi));

  uchar y_values[16], max_values[16];
  cv::v_store(y_values, cv::v_pack(y_lo, y_hi));
  cv::v_store(max_values, cv::v_pack(max_lo, max_hi));
  ushort gains[16];
  for (int lane = 0; lane < 16; ++lane) {
    gains[lane] = static_cast<ushort>((std::min)(
        static_cast<int>(luts.gain_q12[y_values[lane]]),
        static_cast<int>(luts.clip_q12[max_values[lane]])));
  }
  const cv::v_uint16x8 gain_lo = cv::v_load(gains);
  const cv::v_uint16x8 gain_hi = cv::v_load(gains + 8);
  const cv::v_uint32x4 round2048 = cv::v_setall_u32(2048);
  const auto tone8 = [&](const cv::v_uint16x8 &value,
                         const cv::v_uint16x8 &gain) {
    cv::v_uint32x4 product_lo, product_hi;
    cv::v_mul_expand(value, gain, product_lo, product_hi);
    return cv::v_pack((product_lo + round2048) >> 12,
                      (product_hi + round2048) >> 12);
  };
  blue = cv::v_pack(tone8(b_lo, gain_lo), tone8(b_hi, gain_hi));
  green = cv::v_pack(tone8(g_lo, gain_lo), tone8(g_hi, gain_hi));
  red = cv::v_pack(tone8(r_lo, gain_lo), tone8(r_hi, gain_hi));
}
#endif

#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
// Constant-lane extraction lets AArch64 lower each lookup to UMOV + LDRH +
// INS, without materialising the index/result vectors on the stack.
inline int16x8_t FastToneGatherDelta8Neon(
    const int16_t *table, const uint16x8_t index) {
  int16x8_t result = vdupq_n_s16(table[vgetq_lane_u16(index, 0)]);
  result = vsetq_lane_s16(table[vgetq_lane_u16(index, 1)], result, 1);
  result = vsetq_lane_s16(table[vgetq_lane_u16(index, 2)], result, 2);
  result = vsetq_lane_s16(table[vgetq_lane_u16(index, 3)], result, 3);
  result = vsetq_lane_s16(table[vgetq_lane_u16(index, 4)], result, 4);
  result = vsetq_lane_s16(table[vgetq_lane_u16(index, 5)], result, 5);
  result = vsetq_lane_s16(table[vgetq_lane_u16(index, 6)], result, 6);
  result = vsetq_lane_s16(table[vgetq_lane_u16(index, 7)], result, 7);
  return result;
}

// Exact AArch64 form of ApplyFastBalancedTonePixel. For every valid gain:
//   round(value * gain / 4096)
// = value + SQRDMULH(value, (gain - 4096) * 8).
inline void ApplyFastBalancedTone16Neon(
    uint8x16_t &blue, uint8x16_t &green, uint8x16_t &red,
    const FastBalancedToneLuts &luts) {
  const uint16x8_t b_lo = vmovl_u8(vget_low_u8(blue));
  const uint16x8_t b_hi = vmovl_u8(vget_high_u8(blue));
  const uint16x8_t g_lo = vmovl_u8(vget_low_u8(green));
  const uint16x8_t g_hi = vmovl_u8(vget_high_u8(green));
  const uint16x8_t r_lo = vmovl_u8(vget_low_u8(red));
  const uint16x8_t r_hi = vmovl_u8(vget_high_u8(red));

  uint16x8_t y_lo = vmulq_n_u16(b_lo, 29);
  uint16x8_t y_hi = vmulq_n_u16(b_hi, 29);
  y_lo = vmlaq_n_u16(y_lo, g_lo, 150);
  y_hi = vmlaq_n_u16(y_hi, g_hi, 150);
  y_lo = vmlaq_n_u16(y_lo, r_lo, 77);
  y_hi = vmlaq_n_u16(y_hi, r_hi, 77);
  y_lo = vshrq_n_u16(vaddq_u16(y_lo, vdupq_n_u16(128)), 8);
  y_hi = vshrq_n_u16(vaddq_u16(y_hi, vdupq_n_u16(128)), 8);
  const uint16x8_t max_lo = vmaxq_u16(b_lo, vmaxq_u16(g_lo, r_lo));
  const uint16x8_t max_hi = vmaxq_u16(b_hi, vmaxq_u16(g_hi, r_hi));
  const uint16x8_t index_lo =
      vorrq_u16(y_lo, vshlq_n_u16(max_lo, 8));
  const uint16x8_t index_hi =
      vorrq_u16(y_hi, vshlq_n_u16(max_hi, 8));
  const int16_t *table = luts.gain_delta_q15.data();
  const int16x8_t delta_lo = FastToneGatherDelta8Neon(table, index_lo);
  const int16x8_t delta_hi = FastToneGatherDelta8Neon(table, index_hi);

  const auto tone8 = [](const uint16x8_t value,
                        const int16x8_t gain_delta) {
    const int16x8_t value_s = vreinterpretq_s16_u16(value);
    return vreinterpretq_u16_s16(
        vaddq_s16(value_s, vqrdmulhq_s16(value_s, gain_delta)));
  };
  blue = vcombine_u8(vqmovn_u16(tone8(b_lo, delta_lo)),
                     vqmovn_u16(tone8(b_hi, delta_hi)));
  green = vcombine_u8(vqmovn_u16(tone8(g_lo, delta_lo)),
                      vqmovn_u16(tone8(g_hi, delta_hi)));
  red = vcombine_u8(vqmovn_u16(tone8(r_lo, delta_lo)),
                    vqmovn_u16(tone8(r_hi, delta_hi)));
}
#endif

inline void ApplyFastBalancedGamma(cv::Mat &bgr) {
  CV_Assert(bgr.type() == CV_8UC3);
  const FastBalancedToneLuts &luts = GetFastBalancedToneLuts();
  for (int y = 0; y < bgr.rows; ++y) {
    uchar *p = bgr.ptr<uchar>(y);
    for (int x = 0; x < bgr.cols; ++x, p += 3) {
      ApplyFastBalancedTonePixel(p[0], p[1], p[2], luts);
    }
  }
}

inline bool FastEnvTruthy(const char *name) {
  const char *value = std::getenv(name);
  return value &&
      (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
       std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0 ||
       std::strcmp(value, "ON") == 0);
}

// Optional single-frame latency mode for big ARM cores.  The normal capture
// path processes independent cameras in parallel, one ISP per core; enabling
// this switch instead splits one frame across the SDK's persistent four-shard
// pool.  The RK3588 pool pins its helpers to CPU4..6 and the submitting ISP
// thread to CPU7.  Other ARM topologies remain disabled rather than guessed.
inline bool FastIntraFrameParallelEnabled(const char *stage) {
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
  const char *mode = std::getenv("CYPERSTEREO_FAST_INTRA_FRAME");
  if (!mode) return false;
  return std::strcmp(mode, "1") == 0 || std::strcmp(mode, "all") == 0 ||
         std::strcmp(mode, stage) == 0;
#else
  (void)stage;
  return false;
#endif
}

template <class RowBody>
inline void FastParallelForRows(int rows, const char *stage,
                                const RowBody &body) {
  if (rows <= 0) return;
  if (!FastIspFrameParallelActive() ||
      !FastIntraFrameParallelEnabled(stage)) {
    for (int row = 0; row < rows; ++row) body(row);
    return;
  }
  GetFastIspStagePool().Run([&](int shard) {
    int begin, end;
    FastIspShardRange(rows, shard, begin, end);
    for (int row = begin; row < end; ++row) body(row);
  });
}

inline bool FastBalancedGammaEnabled() {
  static const bool enabled = [] {
    return !FastEnvTruthy("CYPERSTEREO_FAST_DISABLE_GAMMA");
  }();
  return enabled;
}

inline bool FastBalancedHueGuardEnabled() {
  static const bool enabled = [] {
    return !FastEnvTruthy("CYPERSTEREO_FAST_DISABLE_HUE_GUARD");
  }();
  return enabled;
}

inline bool FastBalancedBayerNrEnabled() {
  static const bool enabled = [] {
    return !FastEnvTruthy("CYPERSTEREO_FAST_DISABLE_BAYERNR");
  }();
  return enabled;
}

inline bool FastBalancedFusedWbCopyEnabled() {
  static const bool enabled = [] {
    return !FastEnvTruthy("CYPERSTEREO_FAST_DISABLE_FUSED_WB_COPY");
  }();
  return enabled;
}

inline const char *FastBalancedDemosaicName() { return "ea"; }

class WhiteBalance {
 public:
  void Apply(
      cv::Mat &raw,
      BayerConversion bayer = BayerConversion::kColorBayerRg2Bgr) {
    // Estimating on every 2nd frame halves the estimator cost; the gains
    // are EMA-smoothed (kSmooth=0.05, ~0.7s time constant) so the update
    // rate change is imperceptible.
    if ((frame_idx_++ & 1) == 0 || b_gain_ <= 0.0)
      EstimateGains(raw, bayer);

    double bg = b_gain_ > 0.0 ? b_gain_ : 1.0;
    double rg = r_gain_ > 0.0 ? r_gain_ : 1.0;
    BuildLuts(bg, rg);

#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
    if (UseNeonWbLut()) {
      ApplyLutsNeon(raw, bayer);
      return;
    }
#endif
    const int w = raw.cols;
    for (int y = 0; y < raw.rows; ++y) {
      uchar *p = raw.ptr<uchar>(y);
      const bool rggb = bayer == BayerConversion::kColorBayerBg2Bgr;
      const uchar *even_col =
          (y & 1) ? lut_g_ : (rggb ? lut_r_ : lut_b_);
      const uchar *odd_col =
          (y & 1) ? (rggb ? lut_b_ : lut_r_) : lut_g_;
      int x = 0;
      for (; x + 1 < w; x += 2) {
        p[x] = even_col[p[x]];
        p[x + 1] = odd_col[p[x + 1]];
      }
      if (x < w) p[x] = even_col[p[x]];
    }
  }

  // Out-of-place BLC/AWB for the low-gain fast path.  This is deliberately
  // bit-exact with raw.copyTo(dst) followed by Apply(dst): gain estimation
  // observes the same source samples and every output byte visits the same
  // channel LUT.  On AVX2, consecutive even/odd CFA values are split into
  // u16 pairs and gathered only from their own channel tables, fusing the
  // copy and LUT walks without changing any ISP arithmetic.
  void ApplyTo(
      const cv::Mat &raw, cv::Mat &dst,
      BayerConversion bayer = BayerConversion::kColorBayerRg2Bgr) {
    CV_Assert(raw.type() == CV_8UC1);
    CV_Assert(raw.data != dst.data);
    if ((frame_idx_++ & 1) == 0 || b_gain_ <= 0.0)
      EstimateGains(raw, bayer);

    const double bg = b_gain_ > 0.0 ? b_gain_ : 1.0;
    const double rg = r_gain_ > 0.0 ? r_gain_ : 1.0;
    BuildLuts(bg, rg);
    dst.create(raw.size(), CV_8UC1);

    const bool rggb = bayer == BayerConversion::kColorBayerBg2Bgr;
    const int width = raw.cols;
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
    // Out-of-place counterpart of ApplyLutsNeon.  vld2/vst2 keeps the
    // alternating Bayer phases separate while the same 256-entry TBL/TBX
    // lookup maps every byte.  This is bit-exact with copyTo()+Apply(), but
    // writes the destination only once and never mutates the caller's RAW.
    const bool use_neon_lut = UseNeonWbLut();
    NeonLut256 neon_b, neon_g, neon_r;
    if (use_neon_lut) {
      neon_b.Load(lut_b_);
      neon_g.Load(lut_g_);
      neon_r.Load(lut_r_);
    }
#endif
    for (int y = 0; y < raw.rows; ++y) {
      const uchar *src = raw.ptr<uchar>(y);
      uchar *out = dst.ptr<uchar>(y);
      const uchar *even_col =
          (y & 1) ? lut_g_ : (rggb ? lut_r_ : lut_b_);
      const uchar *odd_col =
          (y & 1) ? (rggb ? lut_b_ : lut_r_) : lut_g_;
      int x = 0;
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
      if (use_neon_lut) {
        const NeonLut256 &even_neon =
            (y & 1) ? neon_g : (rggb ? neon_r : neon_b);
        const NeonLut256 &odd_neon =
            (y & 1) ? (rggb ? neon_b : neon_r) : neon_g;
        for (; x + 32 <= width; x += 32) {
          uint8x16x2_t values = vld2q_u8(src + x);
          values.val[0] = even_neon.Apply(values.val[0]);
          values.val[1] = odd_neon.Apply(values.val[1]);
          vst2q_u8(out + x, values);
        }
      }
#endif
#if defined(__AVX2__) || defined(_M_AVX2)
      const int32_t *even_col_i32 =
          (y & 1) ? lut_g_i32_ : (rggb ? lut_r_i32_ : lut_b_i32_);
      const int32_t *odd_col_i32 =
          (y & 1) ? (rggb ? lut_b_i32_ : lut_r_i32_) : lut_g_i32_;
      const __m256i mask_u16 = _mm256_set1_epi32(0xffff);
      const __m256i black32 =
          _mm256_set1_epi32(static_cast<int>(kBlackLevel));
      const bool even_is_green = (y & 1) != 0;
      const auto apply_luts16 = [&](const __m256i samples16) {
        const __m256i even_indices =
            _mm256_and_si256(samples16, mask_u16);
        const __m256i odd_indices = _mm256_srli_epi32(samples16, 16);
        // Half of a Bayer row is green and its LUT is exactly
        // max(sample-black, 0). Evaluate that half arithmetically, leaving
        // only the gain-bearing R/B half as a gather (two rather than four
        // gathers per 32 pixels).
        const __m256i even_values = even_is_green
            ? _mm256_sub_epi32(_mm256_max_epi32(even_indices, black32),
                               black32)
            : _mm256_i32gather_epi32(even_col_i32, even_indices, 4);
        const __m256i odd_values = even_is_green
            ? _mm256_i32gather_epi32(odd_col_i32, odd_indices, 4)
            : _mm256_sub_epi32(_mm256_max_epi32(odd_indices, black32),
                               black32);
        return _mm256_packus_epi32(
            _mm256_unpacklo_epi32(even_values, odd_values),
            _mm256_unpackhi_epi32(even_values, odd_values));
      };
      for (; x + 32 <= width; x += 32) {
        const __m256i samples8 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + x));
        const __m256i values_lo = apply_luts16(_mm256_cvtepu8_epi16(
            _mm256_castsi256_si128(samples8)));
        const __m256i values_hi = apply_luts16(_mm256_cvtepu8_epi16(
            _mm256_extracti128_si256(samples8, 1)));
        const __m256i packed = _mm256_permute4x64_epi64(
            _mm256_packus_epi16(values_lo, values_hi), 0xd8);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + x), packed);
      }
#endif
      for (; x + 1 < width; x += 2) {
        out[x] = even_col[src[x]];
        out[x + 1] = odd_col[src[x + 1]];
      }
      if (x < width) out[x] = even_col[src[x]];
    }
  }

  // Gain-adaptive Bayer same-colour NR fused with the BLC/AWB LUT write.
  // Reading from `raw` and writing a separate `raw_wb` avoids both an int32
  // RAW expansion and an additional full-frame copy.  The +/-2 neighbours
  // stay on the exact R/Gr/Gb/B sub-lattice, so no colour planes are mixed.
  void ApplyWithBayerNr(
      const cv::Mat &raw, cv::Mat &raw_wb, double sensor_gain,
      BayerConversion bayer = BayerConversion::kColorBayerRg2Bgr) {
    CV_Assert(raw.type() == CV_8UC1);
    if ((frame_idx_++ & 1) == 0 || b_gain_ <= 0.0)
      EstimateGains(raw, bayer);

    const double bg = b_gain_ > 0.0 ? b_gain_ : 1.0;
    const double rg = r_gain_ > 0.0 ? r_gain_ : 1.0;
    BuildLuts(bg, rg);

    if (!(sensor_gain >= 1.0)) sensor_gain = 1.0;
    sensor_gain = (std::min)(sensor_gain, 8.0);
    const int t_q8 = static_cast<int>(
        (sensor_gain - 1.0) * (256.0 / 7.0) + 0.5);
    const int threshold_base = 2 + ((4 - 2) * t_q8 + 128) / 256;
    const int threshold_signal_q8 =
        2 + ((3 - 2) * t_q8 + 128) / 256;
    // OpenCV EA is slightly more sensitive to prefiltering than the HDR
    // colour-difference demosaic. Keep the blend conservative while retaining
    // enough high-gain strength to remove random RAW noise.
    const int strength_q8 =
        96 + ((144 - 96) * t_q8 + 128) / 256;

    raw_wb.create(raw.size(), CV_8UC1);
    const bool rggb = bayer == BayerConversion::kColorBayerBg2Bgr;
    const int width = raw.cols;
    const int height = raw.rows;
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
    // Keep the BayerNR output in registers and map two vectors together.
    // UZP separates the CFA phases; green is the exact lut_g_ saturating
    // black subtraction and the R/B phase uses the existing 256-entry TBL.
    const bool use_neon_bayer_nr_lut = UseNeonWbLut();
    NeonLut256 neon_b, neon_r;
    if (use_neon_bayer_nr_lut) {
      neon_b.Load(lut_b_);
      neon_r.Load(lut_r_);
    }
#endif
    const auto process_row = [&](int y) {
      const uchar *src = raw.ptr<uchar>(y);
      uchar *dst = raw_wb.ptr<uchar>(y);
      const uchar *even_col =
          (y & 1) ? lut_g_ : (rggb ? lut_r_ : lut_b_);
      const uchar *odd_col =
          (y & 1) ? (rggb ? lut_b_ : lut_r_) : lut_g_;
#if defined(__AVX2__)
      const int32_t *even_col_i32 =
          (y & 1) ? lut_g_i32_ : (rggb ? lut_r_i32_ : lut_b_i32_);
      const int32_t *odd_col_i32 =
          (y & 1) ? (rggb ? lut_b_i32_ : lut_r_i32_) : lut_g_i32_;
#endif
      if (y < 2 || y >= height - 2) {
        for (int x = 0; x < width; ++x)
          dst[x] = (x & 1 ? odd_col : even_col)[src[x]];
        return;
      }

      const uchar *up = raw.ptr<uchar>(y - 2);
      const uchar *down = raw.ptr<uchar>(y + 2);
      int x = 0;
      for (; x < (std::min)(2, width); ++x)
        dst[x] = (x & 1 ? odd_col : even_col)[src[x]];
#if defined(__AVX2__)
      // 32 Bayer samples per iteration. Filtering stays in u16 lanes; the
      // alternating WB LUT is applied with AVX2 gathers, eliminating the
      // scalar 16-lane table loop from the SSE fallback below.
      const __m256i black16 =
          _mm256_set1_epi16(static_cast<short>(kBlackLevel));
      const __m256i base16 = _mm256_set1_epi16(threshold_base);
      const __m256i slope16 = _mm256_set1_epi16(threshold_signal_q8);
      const __m256i four16 = _mm256_set1_epi16(4);
      const __m256i strength16 = _mm256_set1_epi16(strength_q8);
      const __m256i round127_16 = _mm256_set1_epi16(127);
      const __m256i round128_16 = _mm256_set1_epi16(128);
      const __m256i zero16 = _mm256_setzero_si256();
      const __m256i mask_u16 = _mm256_set1_epi32(0xffff);
      const __m256i black32 =
          _mm256_set1_epi32(static_cast<int>(kBlackLevel));
      const bool even_is_green = (y & 1) != 0;
      const auto filter16 = [&](const __m256i center, const __m256i left,
                                const __m256i right, const __m256i above,
                                const __m256i below) {
        const __m256i signal =
            _mm256_sub_epi16(_mm256_max_epu16(center, black16), black16);
        const __m256i threshold = _mm256_add_epi16(
            base16, _mm256_srli_epi16(
                        _mm256_add_epi16(_mm256_mullo_epi16(signal, slope16),
                                         round128_16),
                        8));
        const auto accepted = [&](const __m256i sample) {
          const __m256i difference = _mm256_sub_epi16(
              _mm256_max_epu16(sample, center),
              _mm256_min_epu16(sample, center));
          const __m256i reject = _mm256_cmpgt_epi16(difference, threshold);
          return _mm256_blendv_epi8(sample, center, reject);
        };
        __m256i sum = _mm256_mullo_epi16(center, four16);
        sum = _mm256_add_epi16(sum, accepted(left));
        sum = _mm256_add_epi16(sum, accepted(right));
        sum = _mm256_add_epi16(sum, accepted(above));
        sum = _mm256_add_epi16(sum, accepted(below));
        const __m256i filtered =
            _mm256_srli_epi16(_mm256_add_epi16(sum, four16), 3);
        __m256i product = _mm256_mullo_epi16(
            _mm256_sub_epi16(filtered, center), strength16);
        const __m256i negative = _mm256_cmpgt_epi16(zero16, product);
        const __m256i rounding =
            _mm256_blendv_epi8(round128_16, round127_16, negative);
        product = _mm256_srai_epi16(_mm256_add_epi16(product, rounding), 8);
        return _mm256_add_epi16(center, product);
      };
      // Each dword contains one consecutive [even, odd] u16 Bayer pair.
      // Split the pair before lookup so each index visits only its own CFA
      // LUT: green is exactly max(sample-black, 0), so evaluate that half in
      // lanes and gather only the gain-bearing R/B half. This reduces the
      // mapping to two gathers per 32 pixels. The pack order restores
      // e0,o0,... exactly.
      const auto apply_luts16 = [&](const __m256i filtered16) {
        const __m256i even_indices =
            _mm256_and_si256(filtered16, mask_u16);
        const __m256i odd_indices = _mm256_srli_epi32(filtered16, 16);
        const __m256i even_values = even_is_green
            ? _mm256_sub_epi32(
                  _mm256_max_epi32(even_indices, black32), black32)
            : _mm256_i32gather_epi32(even_col_i32, even_indices, 4);
        const __m256i odd_values = even_is_green
            ? _mm256_i32gather_epi32(odd_col_i32, odd_indices, 4)
            : _mm256_sub_epi32(
                  _mm256_max_epi32(odd_indices, black32), black32);
        return _mm256_packus_epi32(
            _mm256_unpacklo_epi32(even_values, odd_values),
            _mm256_unpackhi_epi32(even_values, odd_values));
      };
      for (; x + 32 <= width - 2; x += 32) {
        const __m256i center8 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + x));
        const __m256i left8 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(src + x - 2));
        const __m256i right8 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(src + x + 2));
        const __m256i above8 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(up + x));
        const __m256i below8 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(down + x));
        const __m256i center_lo = _mm256_cvtepu8_epi16(
            _mm256_castsi256_si128(center8));
        const __m256i center_hi = _mm256_cvtepu8_epi16(
            _mm256_extracti128_si256(center8, 1));
        const __m256i filtered_lo = filter16(
            center_lo,
            _mm256_cvtepu8_epi16(_mm256_castsi256_si128(left8)),
            _mm256_cvtepu8_epi16(_mm256_castsi256_si128(right8)),
            _mm256_cvtepu8_epi16(_mm256_castsi256_si128(above8)),
            _mm256_cvtepu8_epi16(_mm256_castsi256_si128(below8)));
        const __m256i filtered_hi = filter16(
            center_hi,
            _mm256_cvtepu8_epi16(_mm256_extracti128_si256(left8, 1)),
            _mm256_cvtepu8_epi16(_mm256_extracti128_si256(right8, 1)),
            _mm256_cvtepu8_epi16(_mm256_extracti128_si256(above8, 1)),
            _mm256_cvtepu8_epi16(_mm256_extracti128_si256(below8, 1)));

        const __m256i values_lo = apply_luts16(filtered_lo);
        const __m256i values_hi = apply_luts16(filtered_hi);
        const __m256i packed = _mm256_permute4x64_epi64(
            _mm256_packus_epi16(values_lo, values_hi), 0xd8);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + x), packed);
      }
#endif
#if CV_SIMD128
      // Rejected neighbours are replaced by the centre. This keeps a fixed
      // denominator of eight (4*centre + four neighbours), eliminating the
      // original scalar variable divide while still making an independent
      // edge decision for every same-colour neighbour.
      const cv::v_uint16x8 black = cv::v_setall_u16(
          static_cast<unsigned short>(kBlackLevel));
      const cv::v_uint16x8 base_u =
          cv::v_setall_u16(static_cast<unsigned short>(threshold_base));
      const cv::v_uint16x8 slope_u = cv::v_setall_u16(
          static_cast<unsigned short>(threshold_signal_q8));
      const cv::v_uint16x8 round128_u = cv::v_setall_u16(128);
      const cv::v_uint16x8 four_u = cv::v_setall_u16(4);
      const cv::v_int16x8 strength =
          cv::v_setall_s16(static_cast<short>(strength_q8));
      const cv::v_int16x8 zero_s = cv::v_setzero_s16();
      const cv::v_int16x8 pos_round = cv::v_setall_s16(128);
      const cv::v_int16x8 neg_round = cv::v_setall_s16(127);
      const auto filter8 = [&](const cv::v_uint16x8 &center,
                               const cv::v_uint16x8 &left,
                               const cv::v_uint16x8 &right,
                               const cv::v_uint16x8 &above,
                               const cv::v_uint16x8 &below) {
        const cv::v_uint16x8 signal = cv::v_max(center, black) - black;
        const cv::v_uint16x8 threshold =
            base_u + ((signal * slope_u + round128_u) >> 8);
        const auto accepted = [&](const cv::v_uint16x8 &sample) {
          const cv::v_uint16x8 difference =
              cv::v_max(sample, center) - cv::v_min(sample, center);
          const cv::v_uint16x8 mask = ~(difference > threshold);
          return (mask & sample) | (~mask & center);
        };
        cv::v_uint16x8 sum = center * four_u;
        sum = sum + accepted(left) + accepted(right) + accepted(above) +
              accepted(below);
        return (sum + four_u) >> 3;
      };
      const auto filter_pixels16 = [&](int offset) {
        cv::v_uint16x8 c_lo, c_hi, l_lo, l_hi, r_lo, r_hi;
        cv::v_uint16x8 u_lo, u_hi, d_lo, d_hi;
        cv::v_expand(cv::v_load(src + offset), c_lo, c_hi);
        cv::v_expand(cv::v_load(src + offset - 2), l_lo, l_hi);
        cv::v_expand(cv::v_load(src + offset + 2), r_lo, r_hi);
        cv::v_expand(cv::v_load(up + offset), u_lo, u_hi);
        cv::v_expand(cv::v_load(down + offset), d_lo, d_hi);
        const cv::v_uint16x8 f_lo = filter8(c_lo, l_lo, r_lo, u_lo, d_lo);
        const cv::v_uint16x8 f_hi = filter8(c_hi, l_hi, r_hi, u_hi, d_hi);
        const cv::v_int16x8 c_s_lo = cv::v_reinterpret_as_s16(c_lo);
        const cv::v_int16x8 c_s_hi = cv::v_reinterpret_as_s16(c_hi);
        cv::v_int16x8 p_lo =
            (cv::v_reinterpret_as_s16(f_lo) - c_s_lo) * strength;
        cv::v_int16x8 p_hi =
            (cv::v_reinterpret_as_s16(f_hi) - c_s_hi) * strength;
        const cv::v_int16x8 neg_lo = p_lo < zero_s;
        const cv::v_int16x8 neg_hi = p_hi < zero_s;
        p_lo = (p_lo + ((neg_lo & neg_round) | (~neg_lo & pos_round))) >> 8;
        p_hi = (p_hi + ((neg_hi & neg_round) | (~neg_hi & pos_round))) >> 8;
        return cv::v_pack_u(c_s_lo + p_lo, c_s_hi + p_hi);
      };
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
      if (use_neon_bayer_nr_lut) {
        const bool even_is_green = (y & 1) != 0;
        const NeonLut256 &color_neon =
            (y & 1) ? (rggb ? neon_b : neon_r)
                    : (rggb ? neon_r : neon_b);
        const uint8x16_t black8 =
            vdupq_n_u8(static_cast<unsigned char>(kBlackLevel));
        const uint8x16_t threshold_base8 =
            vdupq_n_u8(static_cast<unsigned char>(threshold_base));
        const uint16x8_t round4_u16 = vdupq_n_u16(4);
        const int16x8_t strength_s16 =
            vdupq_n_s16(static_cast<short>(strength_q8));
        const int16x8_t zero_s16 = vdupq_n_s16(0);
        const int16x8_t pos_round_s16 = vdupq_n_s16(128);
        const int16x8_t neg_round_s16 = vdupq_n_s16(127);
        // Native u8 filter. The old universal path widened center and all
        // four neighbours before doing even the threshold comparisons. Here
        // abs-difference, threshold and neighbour selection stay in u8; only
        // the guaranteed-positive weighted sum widens to u16. The threshold
        // slope is always 2 or 3 for the clamped 1x..8x gain range. Counting
        // the exact rounded transition points avoids widening the signal.
        const auto filter_neon16 = [&](int offset) {
          const uint8x16_t center = vld1q_u8(src + offset);
          const uint8x16_t signal = vqsubq_u8(center, black8);
          uint8x16_t threshold_extra;
          if (threshold_signal_q8 == 2) {
            threshold_extra = vaddq_u8(
                vshrq_n_u8(vcgeq_u8(signal, vdupq_n_u8(64)), 7),
                vshrq_n_u8(vcgeq_u8(signal, vdupq_n_u8(192)), 7));
          } else {
            threshold_extra = vaddq_u8(
                vaddq_u8(
                    vshrq_n_u8(vcgeq_u8(signal, vdupq_n_u8(43)), 7),
                    vshrq_n_u8(vcgeq_u8(signal, vdupq_n_u8(128)), 7)),
                vshrq_n_u8(vcgeq_u8(signal, vdupq_n_u8(214)), 7));
          }
          const uint8x16_t threshold =
              vaddq_u8(threshold_base8, threshold_extra);
          const auto accepted = [&](const uint8x16_t sample) {
            return vbslq_u8(vcleq_u8(vabdq_u8(sample, center), threshold),
                            sample, center);
          };
          const uint8x16_t left = accepted(vld1q_u8(src + offset - 2));
          const uint8x16_t right = accepted(vld1q_u8(src + offset + 2));
          const uint8x16_t above = accepted(vld1q_u8(up + offset));
          const uint8x16_t below = accepted(vld1q_u8(down + offset));
          const uint16x8_t center_lo =
              vmovl_u8(vget_low_u8(center));
          const uint16x8_t center_hi = vmovl_high_u8(center);
          uint16x8_t sum_lo = vshlq_n_u16(center_lo, 2);
          uint16x8_t sum_hi = vshlq_n_u16(center_hi, 2);
          sum_lo = vaddw_u8(sum_lo, vget_low_u8(left));
          sum_hi = vaddw_high_u8(sum_hi, left);
          sum_lo = vaddw_u8(sum_lo, vget_low_u8(right));
          sum_hi = vaddw_high_u8(sum_hi, right);
          sum_lo = vaddw_u8(sum_lo, vget_low_u8(above));
          sum_hi = vaddw_high_u8(sum_hi, above);
          sum_lo = vaddw_u8(sum_lo, vget_low_u8(below));
          sum_hi = vaddw_high_u8(sum_hi, below);
          const uint16x8_t filtered_lo =
              vshrq_n_u16(vaddq_u16(sum_lo, round4_u16), 3);
          const uint16x8_t filtered_hi =
              vshrq_n_u16(vaddq_u16(sum_hi, round4_u16), 3);
          int16x8_t product_lo = vmulq_s16(
              vreinterpretq_s16_u16(vsubq_u16(filtered_lo, center_lo)),
              strength_s16);
          int16x8_t product_hi = vmulq_s16(
              vreinterpretq_s16_u16(vsubq_u16(filtered_hi, center_hi)),
              strength_s16);
          const uint16x8_t negative_lo = vcltq_s16(product_lo, zero_s16);
          const uint16x8_t negative_hi = vcltq_s16(product_hi, zero_s16);
          product_lo = vshrq_n_s16(
              vaddq_s16(product_lo,
                        vbslq_s16(negative_lo, neg_round_s16,
                                  pos_round_s16)),
              8);
          product_hi = vshrq_n_s16(
              vaddq_s16(product_hi,
                        vbslq_s16(negative_hi, neg_round_s16,
                                  pos_round_s16)),
              8);
          return vcombine_u8(
              vqmovun_s16(vaddq_s16(vreinterpretq_s16_u16(center_lo),
                                     product_lo)),
              vqmovun_s16(vaddq_s16(vreinterpretq_s16_u16(center_hi),
                                     product_hi)));
        };
        for (; x + 32 <= width - 2; x += 32) {
          const uint8x16_t filtered0 = filter_neon16(x);
          const uint8x16_t filtered1 = filter_neon16(x + 16);
          const uint8x16_t even_idx = vuzp1q_u8(filtered0, filtered1);
          const uint8x16_t odd_idx = vuzp2q_u8(filtered0, filtered1);
          const uint8x16_t even_values = even_is_green
              ? vqsubq_u8(even_idx, black8)
              : color_neon.Apply(even_idx);
          const uint8x16_t odd_values = even_is_green
              ? color_neon.Apply(odd_idx)
              : vqsubq_u8(odd_idx, black8);
          vst1q_u8(dst + x, vzip1q_u8(even_values, odd_values));
          vst1q_u8(dst + x + 16, vzip2q_u8(even_values, odd_values));
        }
      }
#endif
      for (; x + 16 <= width - 2; x += 16) {
        const cv::v_uint8x16 filtered = filter_pixels16(x);
        uchar values[16];
        cv::v_store(values, filtered);
        for (int lane = 0; lane < 16; ++lane) {
          const int px = x + lane;
          dst[px] = (px & 1 ? odd_col : even_col)[values[lane]];
        }
      }
#endif
      for (; x < width - 2; ++x) {
        int value = src[x];
        const int center = value;
        const int signal =
            (std::max)(center - static_cast<int>(kBlackLevel), 0);
        const int threshold = threshold_base +
            ((signal * threshold_signal_q8 + 128) >> 8);
        int sum = 4 * center;
        const int samples[4] = {src[x - 2], src[x + 2], up[x], down[x]};
        for (int sample : samples)
          sum += std::abs(sample - center) <= threshold ? sample : center;
        const int filtered = (sum + 4) >> 3;
        const int delta = (filtered - center) * strength_q8;
        value = center +
            (delta >= 0 ? (delta + 128) >> 8 : -((-delta + 128) >> 8));
        dst[x] = (x & 1 ? odd_col : even_col)[
            static_cast<uchar>((std::max)(0, (std::min)(value, 255)))];
      }
      for (; x < width; ++x) {
        dst[x] = (x & 1 ? odd_col : even_col)[src[x]];
      }
    };
    FastParallelForRows(height, "wb", process_row);
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
  void EstimateGains(const cv::Mat &raw, BayerConversion bayer) {
    const int lo = static_cast<int>(kBlackLevel) + 24;
    const int hi = 250;
    ratios_bg_.clear();
    ratios_rg_.clear();
    int n_total = 0;
    for (int y = 0; y + 1 < raw.rows; y += kEstStep) {
      const uchar *r0 = raw.ptr<uchar>(y);
      const uchar *r1 = raw.ptr<uchar>(y + 1);
      for (int x = 0; x + 1 < raw.cols; x += kEstStep) {
        const bool rggb = bayer == BayerConversion::kColorBayerBg2Bgr;
        int b = rggb ? r1[x + 1] : r0[x];
        int g0 = r0[x + 1];
        int g1 = r1[x];
        int r = rggb ? r0[x] : r1[x + 1];
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
    // The WB LUT stays LINEAR (pedestal subtraction + WB gain only): the
    // canonical pipeline needs linear data for demosaic and the CCM, so
    // the encode/tone curve now runs post-CCM in ApplyCcmTone(). (The NEON
    // TBL path reads these same tables.)
    for (int i = 0; i < 256; ++i) {
      lut_b_[i] = cv::saturate_cast<uchar>((i - kBlackLevel) * bg);
      lut_g_[i] = cv::saturate_cast<uchar>((i - kBlackLevel) * 1.0);
      lut_r_[i] = cv::saturate_cast<uchar>((i - kBlackLevel) * rg);
      lut_b_i32_[i] = lut_b_[i];
      lut_g_i32_[i] = lut_g_[i];
      lut_r_i32_[i] = lut_r_[i];
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

  void ApplyLutsNeon(cv::Mat &raw, BayerConversion bayer) {
    NeonLut256 lb, lg, lr;
    lb.Load(lut_b_);
    lg.Load(lut_g_);
    lr.Load(lut_r_);
    const int w = raw.cols;
    for (int y = 0; y < raw.rows; ++y) {
      uchar *p = raw.ptr<uchar>(y);
      const bool rggb = bayer == BayerConversion::kColorBayerBg2Bgr;
      const NeonLut256 &even_col = (y & 1) ? lg : (rggb ? lr : lb);
      const NeonLut256 &odd_col = (y & 1) ? (rggb ? lb : lr) : lg;
      const uchar *lut_even =
          (y & 1) ? lut_g_ : (rggb ? lut_r_ : lut_b_);
      const uchar *lut_odd =
          (y & 1) ? (rggb ? lut_b_ : lut_r_) : lut_g_;
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

  // Restored to the sensor pedestal: the kBlackLevel=0 experiment showed a
  // green cast (the estimator's g/b, g/r ratios get pulled toward 1 by the
  // common pedestal, under-driving the B/R gains). Shadow detail now comes
  // from the encode gamma instead of from keeping the pedestal.
  static constexpr double kBlackLevel = kIspBlackLevel;
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
  int32_t lut_b_i32_[256], lut_g_i32_[256], lut_r_i32_[256];
  std::vector<float> ratios_bg_, ratios_rg_;
};

// False color suppression (the standard ISP YUV-domain block; we have no
// optical low-pass filter, so detail near the CFA Nyquist limit aliases
// into chroma during demosaic and can only be detected + desaturated, never
// truly recovered). Two complementary mechanisms, each validated against a
// failure mode the other cannot handle:
//
// 1. Texture-gated residual suppression, for color moire on dense fine texture
//    (fan grilles, distant building windows). The artifact appears as
//    LOW-frequency red/blue bands (beat between texture period and pixel
//    grid). A wide stable chroma base is kept, while only the high-frequency
//    residual around it is attenuated. Band-pass luma texture energy gates the
//    correction; coherent local/wide colour protects genuine fabric and skin.
//
// 2. Shadow residual suppression, for chroma blotches on near-black regions
//    (motion-blurred fan blades at night: texture energy ~0.5, so gate 1
//    never fires there). At those signal levels chroma is demosaic/noise
//    artifact amplified asymmetrically by WB gains. The noisy residual fades
//    with luma, but the stable base colour remains intact.
//
// A final 3x3 chroma median mops up isolated speckle (hot pixels, demosaic
// outliers) in flat regions where neither gate fires. (Removal was tested:
// residual chroma HF energy rises 15-24% on every validation scene and
// isolated-speckle p99.9 jumps 3 -> 12 DN on the night scenes, so its
// ~0.5 ms stays.)
//
// 3. Luma + chroma noise reduction complements the gain-adaptive BayerNR
//    upstream. Both planes go
//    through a fast self-guided filter (He et al., box filters on a
//    subsampled stats grid, O(N)): smooths variation whose local variance
//    is well below eps and preserves higher-contrast structure. Luma gets
//    a light touch (grain halves on walls, edges kept); chroma tolerates a
//    stronger setting since real chroma varies slowly. Residual low-frequency
//    mottling under very high gain still requires temporal NR to remove safely.
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
//  - x86 AVX2: ON by default. The explicit BGR24 pshufb + Q14 AVX2 body is
//    bit-exact against OpenCV's gray + half-area + YCrCb chain and measured
//    1.71 -> 0.80 ms at 1280x1024 on the deployed i7-7700HQ. The older
//    universal-intrinsics body remains the non-AVX2 fallback. Opt out with
//    CYPERSTEREO_DISABLE_FUSED_FRONT for same-binary A/B.
inline bool UseFusedFront() {
#if defined(CYPERSTEREO_HAVE_NEON)
  static const bool enabled =
      std::getenv("CYPERSTEREO_DISABLE_FUSED_FRONT") == nullptr;
#elif defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
  static const bool enabled = [] {
    if (std::getenv("CYPERSTEREO_DISABLE_FUSED_FRONT") != nullptr)
      return false;
#if defined(__GNUC__) || defined(__clang__)
    return !!__builtin_cpu_supports("avx2");
#else
    // MSVC consumers compile this header with /arch:AVX2 for Release.
    return true;
#endif
  }();
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

#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
struct FastReconstructCoefficients16Neon {
  int16x8_t base[2];
  int16x8_t keep[2];
  int32x4_t base_delta[4];
};

// Prepare the quarter-grid coefficients once for the two half-grid rows that
// share them.  Keeping the base term in i32 avoids repeating four widening
// multiplies for the second row.
inline void FastPrepareReconstruct16Neon(
    const uchar *base_q, const uchar *keep_q, const int base_q7,
    FastReconstructCoefficients16Neon &coefficient) {
  const uint8x8_t base_q8 = vld1_u8(base_q);
  const uint8x8_t keep_q8 = vld1_u8(keep_q);
  const uint8x8x2_t base_repeated = vzip_u8(base_q8, base_q8);
  const uint8x8x2_t keep_repeated = vzip_u8(keep_q8, keep_q8);
  const uint8x16_t base8 =
      vcombine_u8(base_repeated.val[0], base_repeated.val[1]);
  const uint8x16_t keep8 =
      vcombine_u8(keep_repeated.val[0], keep_repeated.val[1]);
  coefficient.base[0] =
      vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(base8)));
  coefficient.base[1] =
      vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(base8)));
  coefficient.keep[0] =
      vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(keep8)));
  coefficient.keep[1] =
      vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(keep8)));
  const int16x8_t center = vdupq_n_s16(128);
  const int16x8_t centered0 = vsubq_s16(coefficient.base[0], center);
  const int16x8_t centered1 = vsubq_s16(coefficient.base[1], center);
  coefficient.base_delta[0] =
      vmull_n_s16(vget_low_s16(centered0), base_q7);
  coefficient.base_delta[1] =
      vmull_n_s16(vget_high_s16(centered0), base_q7);
  coefficient.base_delta[2] =
      vmull_n_s16(vget_low_s16(centered1), base_q7);
  coefficient.base_delta[3] =
      vmull_n_s16(vget_high_s16(centered1), base_q7);
}

inline int16x4_t FastRoundReconstruct4Neon(const int32x4_t value) {
  // vrshrn adds 64 before shifting.  Subtract one from negative lanes first
  // so ties round away from zero exactly like the scalar +/-64 expression.
  return vrshrn_n_s32(vaddq_s32(value, vshrq_n_s32(value, 31)), 7);
}

inline uint8x16_t FastReconstructChroma16Neon(
    const uchar *value, const FastReconstructCoefficients16Neon &coefficient) {
  const uint8x16_t value8 = vld1q_u8(value);
  const int16x8_t value0 =
      vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(value8)));
  const int16x8_t value1 =
      vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(value8)));
  const int16x8_t residual0 = vsubq_s16(value0, coefficient.base[0]);
  const int16x8_t residual1 = vsubq_s16(value1, coefficient.base[1]);
  const int32x4_t delta0 = vmlal_s16(
      coefficient.base_delta[0], vget_low_s16(residual0),
      vget_low_s16(coefficient.keep[0]));
  const int32x4_t delta1 = vmlal_s16(
      coefficient.base_delta[1], vget_high_s16(residual0),
      vget_high_s16(coefficient.keep[0]));
  const int32x4_t delta2 = vmlal_s16(
      coefficient.base_delta[2], vget_low_s16(residual1),
      vget_low_s16(coefficient.keep[1]));
  const int32x4_t delta3 = vmlal_s16(
      coefficient.base_delta[3], vget_high_s16(residual1),
      vget_high_s16(coefficient.keep[1]));
  const int16x8_t rounded0 = vcombine_s16(
      FastRoundReconstruct4Neon(delta0),
      FastRoundReconstruct4Neon(delta1));
  const int16x8_t rounded1 = vcombine_s16(
      FastRoundReconstruct4Neon(delta2),
      FastRoundReconstruct4Neon(delta3));
  const int16x8_t center = vdupq_n_s16(128);
  return vcombine_u8(vqmovun_s16(vaddq_s16(rounded0, center)),
                     vqmovun_s16(vaddq_s16(rounded1, center)));
}

inline void FastSort2U8Neon(uint8x16_t &lo, uint8x16_t &hi) {
  const uint8x16_t old_lo = lo;
  lo = vminq_u8(old_lo, hi);
  hi = vmaxq_u8(old_lo, hi);
}

inline uint8x16_t FastMedian9U8Neon(
    uint8x16_t a0, uint8x16_t a1, uint8x16_t a2,
    uint8x16_t b0, uint8x16_t b1, uint8x16_t b2,
    uint8x16_t c0, uint8x16_t c1, uint8x16_t c2) {
  FastSort2U8Neon(a0, a1);
  FastSort2U8Neon(a1, a2);
  FastSort2U8Neon(a0, a1);
  FastSort2U8Neon(b0, b1);
  FastSort2U8Neon(b1, b2);
  FastSort2U8Neon(b0, b1);
  FastSort2U8Neon(c0, c1);
  FastSort2U8Neon(c1, c2);
  FastSort2U8Neon(c0, c1);
  uint8x16_t low = vmaxq_u8(vmaxq_u8(a0, b0), c0);
  FastSort2U8Neon(a1, b1);
  FastSort2U8Neon(b1, c1);
  FastSort2U8Neon(a1, b1);
  uint8x16_t high = vminq_u8(vminq_u8(a2, b2), c2);
  FastSort2U8Neon(low, b1);
  FastSort2U8Neon(b1, high);
  FastSort2U8Neon(low, b1);
  return b1;
}

inline uchar FastMedian9U8Scalar(
    uchar a0, uchar a1, uchar a2, uchar b0, uchar b1, uchar b2,
    uchar c0, uchar c1, uchar c2) {
  const auto sort2 = [](uchar &lo, uchar &hi) {
    if (lo > hi) std::swap(lo, hi);
  };
  sort2(a0, a1); sort2(a1, a2); sort2(a0, a1);
  sort2(b0, b1); sort2(b1, b2); sort2(b0, b1);
  sort2(c0, c1); sort2(c1, c2); sort2(c0, c1);
  uchar low = (std::max)((std::max)(a0, b0), c0);
  sort2(a1, b1); sort2(b1, c1); sort2(a1, b1);
  uchar high = (std::min)((std::min)(a2, b2), c2);
  sort2(low, b1); sort2(b1, high); sort2(low, b1);
  return b1;
}

// Bit-exact median3 for one u8 plane.  OpenCV's documented replicate border
// is reproduced explicitly; the overlapping final vector only rewrites
// already-correct interior pixels and avoids a scalar right tail.
inline void FastMedian3U8Neon(const cv::Mat &source, cv::Mat &destination) {
  destination.create(source.size(), CV_8U);
  const int width = source.cols;
  const int height = source.rows;
  if (width < 18) {
    cv::medianBlur(source, destination, 3);
    return;
  }
  // source/destination can refer to caller-thread TLS Mats.  Resolve them
  // once and pass ordinary pointers into persistent worker threads so the
  // helpers never perform a TLS name lookup of their own.
  const cv::Mat *const median_source = &source;
  cv::Mat *const median_destination = &destination;
  FastParallelForRows(height, "median", [&](int y) {
    const uchar *row0 =
        median_source->ptr<uchar>((std::max)(y - 1, 0));
    const uchar *row1 = median_source->ptr<uchar>(y);
    const uchar *row2 =
        median_source->ptr<uchar>((std::min)(y + 1, height - 1));
    uchar *output = median_destination->ptr<uchar>(y);
    output[0] = FastMedian9U8Scalar(
        row0[0], row0[0], row0[1], row1[0], row1[0], row1[1],
        row2[0], row2[0], row2[1]);
    int x = 1;
    for (; x + 16 <= width - 1; x += 16) {
      vst1q_u8(output + x, FastMedian9U8Neon(
          vld1q_u8(row0 + x - 1), vld1q_u8(row0 + x),
          vld1q_u8(row0 + x + 1), vld1q_u8(row1 + x - 1),
          vld1q_u8(row1 + x), vld1q_u8(row1 + x + 1),
          vld1q_u8(row2 + x - 1), vld1q_u8(row2 + x),
          vld1q_u8(row2 + x + 1)));
    }
    if (x < width - 1) {
      x = width - 17;
      vst1q_u8(output + x, FastMedian9U8Neon(
          vld1q_u8(row0 + x - 1), vld1q_u8(row0 + x),
          vld1q_u8(row0 + x + 1), vld1q_u8(row1 + x - 1),
          vld1q_u8(row1 + x), vld1q_u8(row1 + x + 1),
          vld1q_u8(row2 + x - 1), vld1q_u8(row2 + x),
          vld1q_u8(row2 + x + 1)));
    }
    const int right = width - 1;
    output[right] = FastMedian9U8Scalar(
        row0[right - 1], row0[right], row0[right],
        row1[right - 1], row1[right], row1[right],
        row2[right - 1], row2[right], row2[right]);
  });
}
#endif

#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
// Exact x86 fused front-end. OpenCV's three operations use Q14 BT.601 and,
// for an exact 2x reduction, INTER_AREA is (sum4 + 2) >> 2 per B/G/R channel.
// Keeping those two rounding points makes all three output planes bit-exact;
// unlike the older Q8 universal body, this can therefore be enabled without
// any colour-quality delta.
struct FastFrontBgr16 {
  __m128i b, g, r;
};

CYPERSTEREO_AVX2_TARGET inline FastFrontBgr16 FastFrontLoadBgr16(
    const uchar *p) {
  const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
  const __m128i b =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 16));
  const __m128i c =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 32));
  const char z = static_cast<char>(-1);
  const __m128i mb0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, z, z, z, z, z, z,
                                     z, z, z, z);
  const __m128i mb1 = _mm_setr_epi8(z, z, z, z, z, z, 2, 5, 8, 11, 14, z,
                                     z, z, z, z);
  const __m128i mb2 = _mm_setr_epi8(z, z, z, z, z, z, z, z, z, z, z, 1, 4,
                                     7, 10, 13);
  const __m128i mg0 = _mm_setr_epi8(1, 4, 7, 10, 13, z, z, z, z, z, z, z, z,
                                     z, z, z);
  const __m128i mg1 = _mm_setr_epi8(z, z, z, z, z, 0, 3, 6, 9, 12, 15, z, z,
                                     z, z, z);
  const __m128i mg2 = _mm_setr_epi8(z, z, z, z, z, z, z, z, z, z, z, 2, 5,
                                     8, 11, 14);
  const __m128i mr0 = _mm_setr_epi8(2, 5, 8, 11, 14, z, z, z, z, z, z, z, z,
                                     z, z, z);
  const __m128i mr1 = _mm_setr_epi8(z, z, z, z, z, 1, 4, 7, 10, 13, z, z, z,
                                     z, z, z);
  const __m128i mr2 = _mm_setr_epi8(z, z, z, z, z, z, z, z, z, z, 0, 3, 6,
                                     9, 12, 15);
  FastFrontBgr16 out;
  out.b = _mm_or_si128(_mm_shuffle_epi8(a, mb0),
                       _mm_or_si128(_mm_shuffle_epi8(b, mb1),
                                    _mm_shuffle_epi8(c, mb2)));
  out.g = _mm_or_si128(_mm_shuffle_epi8(a, mg0),
                       _mm_or_si128(_mm_shuffle_epi8(b, mg1),
                                    _mm_shuffle_epi8(c, mg2)));
  out.r = _mm_or_si128(_mm_shuffle_epi8(a, mr0),
                       _mm_or_si128(_mm_shuffle_epi8(b, mr1),
                                    _mm_shuffle_epi8(c, mr2)));
  return out;
}

CYPERSTEREO_AVX2_TARGET inline void FastFrontGrayQ14(
    const FastFrontBgr16 &v, __m256i &y_lo, __m256i &y_hi) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i b = _mm256_cvtepu8_epi16(v.b);
  const __m256i g = _mm256_cvtepu8_epi16(v.g);
  const __m256i r = _mm256_cvtepu8_epi16(v.r);
  const __m256i kbg = _mm256_set1_epi32((9617 << 16) | 1868);
  const __m256i kr = _mm256_set1_epi32(4899);
  const __m256i round = _mm256_set1_epi32(8192);
  y_lo = _mm256_srai_epi32(
      _mm256_add_epi32(
          round,
          _mm256_add_epi32(
              _mm256_madd_epi16(_mm256_unpacklo_epi16(b, g), kbg),
              _mm256_madd_epi16(_mm256_unpacklo_epi16(r, zero), kr))),
      14);
  y_hi = _mm256_srai_epi32(
      _mm256_add_epi32(
          round,
          _mm256_add_epi32(
              _mm256_madd_epi16(_mm256_unpackhi_epi16(b, g), kbg),
              _mm256_madd_epi16(_mm256_unpackhi_epi16(r, zero), kr))),
      14);
}

CYPERSTEREO_AVX2_TARGET inline __m128i FastFrontPack16(
    const __m256i lo, const __m256i hi) {
  const __m256i p16 = _mm256_packs_epi32(lo, hi);
  return _mm_packus_epi16(_mm256_castsi256_si128(p16),
                          _mm256_extracti128_si256(p16, 1));
}

CYPERSTEREO_AVX2_TARGET inline __m128i FastFrontAverage2x2(
    const __m128i row0, const __m128i row1) {
  const __m128i ones = _mm_set1_epi16(0x0101);
  __m128i s = _mm_add_epi16(_mm_maddubs_epi16(row0, ones),
                            _mm_maddubs_epi16(row1, ones));
  s = _mm_srli_epi16(_mm_add_epi16(s, _mm_set1_epi16(2)), 2);
  return _mm_packus_epi16(s, _mm_setzero_si128());
}

CYPERSTEREO_AVX2_TARGET inline void FastFrontChromaQ14(
    const FastFrontBgr16 &v, __m128i &cr8, __m128i &cb8) {
  __m256i ylo, yhi;
  FastFrontGrayQ14(v, ylo, yhi);
  const __m256i zero = _mm256_setzero_si256();
  const __m256i r = _mm256_cvtepu8_epi16(v.r);
  const __m256i b = _mm256_cvtepu8_epi16(v.b);
  const __m256i one = _mm256_set1_epi32(1);
  const __m256i rlo =
      _mm256_madd_epi16(_mm256_unpacklo_epi16(r, zero), one);
  const __m256i rhi =
      _mm256_madd_epi16(_mm256_unpackhi_epi16(r, zero), one);
  const __m256i blo =
      _mm256_madd_epi16(_mm256_unpacklo_epi16(b, zero), one);
  const __m256i bhi =
      _mm256_madd_epi16(_mm256_unpackhi_epi16(b, zero), one);
  const __m256i delta = _mm256_set1_epi32((128 << 14) + 8192);
  const __m256i crlo = _mm256_srai_epi32(
      _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(rlo, ylo),
                                          _mm256_set1_epi32(11682)),
                       delta),
      14);
  const __m256i crhi = _mm256_srai_epi32(
      _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(rhi, yhi),
                                          _mm256_set1_epi32(11682)),
                       delta),
      14);
  const __m256i cblo = _mm256_srai_epi32(
      _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(blo, ylo),
                                          _mm256_set1_epi32(9241)),
                       delta),
      14);
  const __m256i cbhi = _mm256_srai_epi32(
      _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(bhi, yhi),
                                          _mm256_set1_epi32(9241)),
                       delta),
      14);
  cr8 = FastFrontPack16(crlo, crhi);
  cb8 = FastFrontPack16(cblo, cbhi);
}

CYPERSTEREO_AVX2_TARGET inline void FusedFrontYCbCr420Avx2(
    const cv::Mat &color, cv::Mat &y8, cv::Mat &cr_h, cv::Mat &cb_h) {
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
    int x = 0;
    for (; x + 32 <= width; x += 32) {
      const FastFrontBgr16 a0 = FastFrontLoadBgr16(row0 + 3 * x);
      const FastFrontBgr16 a1 = FastFrontLoadBgr16(row0 + 3 * (x + 16));
      const FastFrontBgr16 b0 = FastFrontLoadBgr16(row1 + 3 * x);
      const FastFrontBgr16 b1 = FastFrontLoadBgr16(row1 + 3 * (x + 16));
      __m256i ya0l, ya0h, ya1l, ya1h, yb0l, yb0h, yb1l, yb1h;
      FastFrontGrayQ14(a0, ya0l, ya0h);
      FastFrontGrayQ14(a1, ya1l, ya1h);
      FastFrontGrayQ14(b0, yb0l, yb0h);
      FastFrontGrayQ14(b1, yb1l, yb1h);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(yrow0 + x),
                       FastFrontPack16(ya0l, ya0h));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(yrow0 + x + 16),
                       FastFrontPack16(ya1l, ya1h));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(yrow1 + x),
                       FastFrontPack16(yb0l, yb0h));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(yrow1 + x + 16),
                       FastFrontPack16(yb1l, yb1h));
      FastFrontBgr16 half;
      half.b = _mm_unpacklo_epi64(FastFrontAverage2x2(a0.b, b0.b),
                                  FastFrontAverage2x2(a1.b, b1.b));
      half.g = _mm_unpacklo_epi64(FastFrontAverage2x2(a0.g, b0.g),
                                  FastFrontAverage2x2(a1.g, b1.g));
      half.r = _mm_unpacklo_epi64(FastFrontAverage2x2(a0.r, b0.r),
                                  FastFrontAverage2x2(a1.r, b1.r));
      __m128i cr, cb;
      FastFrontChromaQ14(half, cr, cb);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pcr + x / 2), cr);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pcb + x / 2), cb);
    }
    // 752-wide MT9 modes end with exactly this 16-pixel vector tail.
    if (x + 16 <= width) {
      const FastFrontBgr16 a = FastFrontLoadBgr16(row0 + 3 * x);
      const FastFrontBgr16 b = FastFrontLoadBgr16(row1 + 3 * x);
      __m256i yal, yah, ybl, ybh;
      FastFrontGrayQ14(a, yal, yah);
      FastFrontGrayQ14(b, ybl, ybh);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(yrow0 + x),
                       FastFrontPack16(yal, yah));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(yrow1 + x),
                       FastFrontPack16(ybl, ybh));
      FastFrontBgr16 half;
      half.b = FastFrontAverage2x2(a.b, b.b);
      half.g = FastFrontAverage2x2(a.g, b.g);
      half.r = FastFrontAverage2x2(a.r, b.r);
      __m128i cr, cb;
      FastFrontChromaQ14(half, cr, cb);
      _mm_storel_epi64(reinterpret_cast<__m128i *>(pcr + x / 2), cr);
      _mm_storel_epi64(reinterpret_cast<__m128i *>(pcb + x / 2), cb);
      x += 16;
    }
    // Generic even-width safety tail. Fast ISP pads normal camera modes to
    // a multiple of four, so this is only for unusual external callers.
    for (; x < width; x += 2) {
      int bs = 0, gs = 0, rs = 0;
      for (int row = 0; row < 2; ++row) {
        const uchar *src = row ? row1 : row0;
        uchar *dst = row ? yrow1 : yrow0;
        for (int dx = 0; dx < 2; ++dx) {
          const int xx = x + dx;
          const int bb = src[3 * xx];
          const int gg = src[3 * xx + 1];
          const int rr = src[3 * xx + 2];
          dst[xx] = static_cast<uchar>(
              (1868 * bb + 9617 * gg + 4899 * rr + 8192) >> 14);
          bs += bb;
          gs += gg;
          rs += rr;
        }
      }
      const int bb = (bs + 2) >> 2;
      const int gg = (gs + 2) >> 2;
      const int rr = (rs + 2) >> 2;
      const int yy = (1868 * bb + 9617 * gg + 4899 * rr + 8192) >> 14;
      pcr[x / 2] = cv::saturate_cast<uchar>(
          ((rr - yy) * 11682 + (128 << 14) + 8192) >> 14);
      pcb[x / 2] = cv::saturate_cast<uchar>(
          ((bb - yy) * 9241 + (128 << 14) + 8192) >> 14);
    }
  }
}

// Fast-only fused EA demosaic + Y/Cr/Cb front-end for the primary BGGR
// sensor phase (COLOR_BayerRG2BGR_EA).  The demosaiced BGR image has no
// other consumer on this path: the final output kernel reconstructs BGR
// after luma/chroma NR and shared-luminance gamma.  Keeping the three EA
// planes in registers avoids a 3.9 MB BGR write/read at 1280x1024.
//
// Unlike the older ARM Q8 front-end below, Windows OpenCV 3.4.16 uses Q14
// BT.601 and rounds the 2x2 BGR area average before YCrCb conversion.  Those
// two rounding points are preserved here.  Random full-range comparisons at
// 1280x1024, 1280x996, 752x480 and small boundary-heavy frames therefore
// produce zero differing Y/Cr/Cb samples against cvtColor(EA)+FusedFront.
struct FastEaPlanar32 {
  __m256i b, g, r;
};

CYPERSTEREO_AVX2_TARGET inline __m256i FastEaAbsDiffU8(__m256i a,
                                                       __m256i b) {
  return _mm256_sub_epi8(_mm256_max_epu8(a, b), _mm256_min_epu8(a, b));
}

CYPERSTEREO_AVX2_TARGET inline __m256i FastEaCmpGtU8(__m256i a,
                                                     __m256i b) {
  const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));
  return _mm256_cmpgt_epi8(_mm256_xor_si256(a, bias),
                           _mm256_xor_si256(b, bias));
}

CYPERSTEREO_AVX2_TARGET inline __m256i FastEaPackU16x32(__m256i lo,
                                                        __m256i hi) {
  // packus is lane-local: [lo0..7,hi0..7 | lo8..15,hi8..15].
  // Reorder 64-bit groups back to the original 0..31 pixel order.
  return _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0xd8);
}

CYPERSTEREO_AVX2_TARGET inline __m256i FastEaAverage4U8(
    __m256i a, __m256i b, __m256i c, __m256i d) {
  const __m128i al = _mm256_castsi256_si128(a);
  const __m128i ah = _mm256_extracti128_si256(a, 1);
  const __m128i bl = _mm256_castsi256_si128(b);
  const __m128i bh = _mm256_extracti128_si256(b, 1);
  const __m128i cl = _mm256_castsi256_si128(c);
  const __m128i ch = _mm256_extracti128_si256(c, 1);
  const __m128i dl = _mm256_castsi256_si128(d);
  const __m128i dh = _mm256_extracti128_si256(d, 1);
  __m256i lo = _mm256_add_epi16(_mm256_cvtepu8_epi16(al),
                                 _mm256_cvtepu8_epi16(bl));
  lo = _mm256_add_epi16(lo, _mm256_cvtepu8_epi16(cl));
  lo = _mm256_add_epi16(lo, _mm256_cvtepu8_epi16(dl));
  __m256i hi = _mm256_add_epi16(_mm256_cvtepu8_epi16(ah),
                                 _mm256_cvtepu8_epi16(bh));
  hi = _mm256_add_epi16(hi, _mm256_cvtepu8_epi16(ch));
  hi = _mm256_add_epi16(hi, _mm256_cvtepu8_epi16(dh));
  const __m256i two = _mm256_set1_epi16(2);
  lo = _mm256_srli_epi16(_mm256_add_epi16(lo, two), 2);
  hi = _mm256_srli_epi16(_mm256_add_epi16(hi, two), 2);
  return FastEaPackU16x32(lo, hi);
}

// x0 is even and covers 32 interior output columns.  Primary BGGR means
// even rows are B G and odd rows are G R.
CYPERSTEREO_AVX2_TARGET inline FastEaPlanar32 FastEaDemosaicRow32(
    const cv::Mat &raw, int y, int x0) {
  const uchar *up = raw.ptr<uchar>(y - 1);
  const uchar *cc = raw.ptr<uchar>(y);
  const uchar *dn = raw.ptr<uchar>(y + 1);
  const __m256i center = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(cc + x0));
  const __m256i left = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(cc + x0 - 1));
  const __m256i right = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(cc + x0 + 1));
  const __m256i above = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(up + x0));
  const __m256i below = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(dn + x0));
  const __m256i above_left = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(up + x0 - 1));
  const __m256i above_right = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(up + x0 + 1));
  const __m256i below_left = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(dn + x0 - 1));
  const __m256i below_right = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(dn + x0 + 1));
  const __m256i horizontal = _mm256_avg_epu8(left, right);
  const __m256i vertical = _mm256_avg_epu8(above, below);
  const __m256i green = _mm256_blendv_epi8(
      horizontal, vertical,
      FastEaCmpGtU8(FastEaAbsDiffU8(left, right),
                    FastEaAbsDiffU8(above, below)));
  const __m256i diagonal = FastEaAverage4U8(
      above_left, above_right, below_left, below_right);
  const __m256i odd = _mm256_set1_epi16(static_cast<short>(0xff00));
  FastEaPlanar32 out;
  if ((y & 1) == 0) {  // B G row
    out.b = _mm256_blendv_epi8(center, horizontal, odd);
    out.g = _mm256_blendv_epi8(green, center, odd);
    out.r = _mm256_blendv_epi8(diagonal, vertical, odd);
  } else {             // G R row
    out.b = _mm256_blendv_epi8(vertical, diagonal, odd);
    out.g = _mm256_blendv_epi8(center, green, odd);
    out.r = _mm256_blendv_epi8(horizontal, center, odd);
  }
  return out;
}

CYPERSTEREO_AVX2_TARGET inline __m256i FastEaGray32(
    const FastEaPlanar32 &p) {
  FastFrontBgr16 lo, hi;
  lo.b = _mm256_castsi256_si128(p.b);
  lo.g = _mm256_castsi256_si128(p.g);
  lo.r = _mm256_castsi256_si128(p.r);
  hi.b = _mm256_extracti128_si256(p.b, 1);
  hi.g = _mm256_extracti128_si256(p.g, 1);
  hi.r = _mm256_extracti128_si256(p.r, 1);
  __m256i y0l, y0h, y1l, y1h;
  FastFrontGrayQ14(lo, y0l, y0h);
  FastFrontGrayQ14(hi, y1l, y1h);
  return _mm256_inserti128_si256(
      _mm256_castsi128_si256(FastFrontPack16(y0l, y0h)),
      FastFrontPack16(y1l, y1h), 1);
}

CYPERSTEREO_AVX2_TARGET inline __m128i FastEaAverage2x2(
    __m256i row0, __m256i row1) {
  const __m256i ones = _mm256_set1_epi16(0x0101);
  __m256i sum = _mm256_add_epi16(
      _mm256_maddubs_epi16(row0, ones),
      _mm256_maddubs_epi16(row1, ones));
  sum = _mm256_srli_epi16(
      _mm256_add_epi16(sum, _mm256_set1_epi16(2)), 2);
  return _mm_packus_epi16(_mm256_castsi256_si128(sum),
                          _mm256_extracti128_si256(sum, 1));
}

CYPERSTEREO_AVX2_TARGET inline void FastEaDemosaicPixel(
    const cv::Mat &raw, int y, int x, int &B, int &G, int &R) {
  const uchar *up = raw.ptr<uchar>(y - 1);
  const uchar *cc = raw.ptr<uchar>(y);
  const uchar *dn = raw.ptr<uchar>(y + 1);
  const auto ea_green = [&] {
    const int horizontal =
        std::abs(static_cast<int>(cc[x - 1]) - cc[x + 1]);
    const int vertical =
        std::abs(static_cast<int>(dn[x]) - up[x]);
    return (horizontal > vertical ? dn[x] + up[x] + 1
                                  : cc[x - 1] + cc[x + 1] + 1) >> 1;
  };
  if ((y & 1) == 0) {
    if ((x & 1) == 0) {
      B = cc[x];
      G = ea_green();
      R = (up[x - 1] + up[x + 1] + dn[x - 1] + dn[x + 1] + 2) >> 2;
    } else {
      G = cc[x];
      B = (cc[x - 1] + cc[x + 1] + 1) >> 1;
      R = (up[x] + dn[x] + 1) >> 1;
    }
  } else {
    if ((x & 1) == 0) {
      G = cc[x];
      R = (cc[x - 1] + cc[x + 1] + 1) >> 1;
      B = (up[x] + dn[x] + 1) >> 1;
    } else {
      R = cc[x];
      G = ea_green();
      B = (up[x - 1] + up[x + 1] + dn[x - 1] + dn[x + 1] + 2) >> 2;
    }
  }
}

CYPERSTEREO_AVX2_TARGET inline void FusedDemosaicFrontAvx2(
    const cv::Mat &raw, cv::Mat &y8, cv::Mat &cr_h, cv::Mat &cb_h) {
  CV_Assert(raw.type() == CV_8U && (raw.cols & 1) == 0 &&
            (raw.rows & 1) == 0 && raw.cols >= 4 && raw.rows >= 4);
  const int width = raw.cols;
  const int height = raw.rows;
  const int half_width = width / 2;
  const int half_height = height / 2;
  y8.create(height, width, CV_8U);
  cr_h.create(half_height, half_width, CV_8U);
  cb_h.create(half_height, half_width, CV_8U);

  const auto pixel = [&](int y, int x, int &B, int &G, int &R) {
    y = (std::max)(1, (std::min)(y, height - 2));
    x = (std::max)(1, (std::min)(x, width - 2));
    FastEaDemosaicPixel(raw, y, x, B, G, R);
  };
  const auto scalar_block = [&](int yh, int xh) {
    int bsum = 0, gsum = 0, rsum = 0;
    for (int dy = 0; dy < 2; ++dy) {
      const int y = 2 * yh + dy;
      uchar *ydst = y8.ptr<uchar>(y);
      for (int dx = 0; dx < 2; ++dx) {
        const int x = 2 * xh + dx;
        int B, G, R;
        pixel(y, x, B, G, R);
        ydst[x] = static_cast<uchar>(
            (1868 * B + 9617 * G + 4899 * R + 8192) >> 14);
        bsum += B;
        gsum += G;
        rsum += R;
      }
    }
    const int B = (bsum + 2) >> 2;
    const int G = (gsum + 2) >> 2;
    const int R = (rsum + 2) >> 2;
    const int Y = (1868 * B + 9617 * G + 4899 * R + 8192) >> 14;
    cr_h.ptr<uchar>(yh)[xh] = cv::saturate_cast<uchar>(
        ((R - Y) * 11682 + (128 << 14) + 8192) >> 14);
    cb_h.ptr<uchar>(yh)[xh] = cv::saturate_cast<uchar>(
        ((B - Y) * 9241 + (128 << 14) + 8192) >> 14);
  };

  const int last_vector_x = width - 34;
  for (int yh = 0; yh < half_height; ++yh) {
    const int y0 = 2 * yh;
    const int y1 = y0 + 1;
    uchar *yrow0 = y8.ptr<uchar>(y0);
    uchar *yrow1 = y8.ptr<uchar>(y1);
    uchar *pcr = cr_h.ptr<uchar>(yh);
    uchar *pcb = cb_h.ptr<uchar>(yh);
    if (width >= 36) {
      int x = 2;
      for (;;) {
        FastEaPlanar32 p0, p1;
        if (y0 == 0) {
          p1 = FastEaDemosaicRow32(raw, y1, x);
          p0 = p1;  // OpenCV demosaiced row0 := row1.
        } else if (y1 == height - 1) {
          p0 = FastEaDemosaicRow32(raw, y0, x);
          p1 = p0;  // OpenCV demosaiced last row := preceding row.
        } else {
          p0 = FastEaDemosaicRow32(raw, y0, x);
          p1 = FastEaDemosaicRow32(raw, y1, x);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(yrow0 + x),
                            FastEaGray32(p0));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(yrow1 + x),
                            FastEaGray32(p1));
        FastFrontBgr16 half;
        half.b = FastEaAverage2x2(p0.b, p1.b);
        half.g = FastEaAverage2x2(p0.g, p1.g);
        half.r = FastEaAverage2x2(p0.r, p1.r);
        __m128i cr, cb;
        FastFrontChromaQ14(half, cr, cb);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(pcr + x / 2), cr);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(pcb + x / 2), cb);
        if (x == last_vector_x) break;
        x += 32;
        if (x > last_vector_x) x = last_vector_x;
      }
      // The vector body covers columns 2..width-3.  Reproduce OpenCV's
      // demosaiced column replication in the first and last 2x2 blocks.
      scalar_block(yh, 0);
      scalar_block(yh, half_width - 1);
    } else {
      // Safety path for tiny external test images.  Camera modes never use
      // this branch, but it keeps every neighbour load in bounds.
      for (int xh = 0; xh < half_width; ++xh) scalar_block(yh, xh);
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
#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
  FusedFrontYCbCr420Avx2(color, y8, cr_h, cb_h);
  return;
#endif
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
  // The fused kernel goes white-balanced Bayer -> Y/CbCr directly, so
  // there is nowhere to apply the post-demosaic CCM + tone stage; it is
  // only valid on the legacy fully-linear pipeline.
  return enabled && IspLegacyLinear();
#endif
}

// The x86 kernel is deliberately fast-path-only.  The quality ISP applies
// its own post-demosaic CCM/tone operations to the BGR image and must keep
// materialising that image.  Fast-balanced instead applies shared-luminance
// gamma in its final reconstruction, so its intermediate BGR is disposable.
inline bool UseFastFusedDemosaic() {
#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
  static const bool enabled = [] {
    if (std::getenv("CYPERSTEREO_DISABLE_FUSED_DEMOSAIC") != nullptr)
      return false;
#if defined(__GNUC__) || defined(__clang__)
    return !!__builtin_cpu_supports("avx2");
#else
    // Windows Release consumers are compiled with /arch:AVX2.
    return true;
#endif
  }();
  return enabled;
#elif defined(CYPERSTEREO_HAVE_NEON)
  // Fast-balanced never applies the quality path's post-demosaic CCM: its
  // shared-luminance tone is deliberately performed in the final output
  // kernel.  The NEON Bayer->Y/Cr/Cb kernel is therefore valid here even when
  // the global quality configuration is not legacy-linear.  Keep the quality
  // path's stricter UseFusedDemosaic()/IspLegacyLinear() contract untouched.
  static const bool enabled =
      !FastEnvTruthy("CYPERSTEREO_DISABLE_FUSED_DEMOSAIC");
  return enabled;
#else
  return false;
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
  FastParallelForRows(hh, "front", [&](int yh) {
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
  });
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

inline bool UseFusedQuarterTextureAvx2() {
  static const bool enabled =
      std::getenv("CYPERSTEREO_DISABLE_FUSED_QUARTER_TEXTURE") == nullptr;
  return enabled;
}

// Windows/x86 luma preparation. The guided filter needs NN samples and their
// squares from y_bl, while the false-colour gate needs a Gaussian3-smoothed
// 4x4 average of |y8-y_bl|. Produce all three in one 4x4-tile walk. The raw
// high-frequency map is streamed through three rows, so neither the old
// full-width colsum nor a full quarter-resolution hf plane is materialized.
// Integer equations, BORDER_REFLECT_101 handling and rounding are bit-exact
// with resize(INTER_NEAREST) + multiply + AbsDiffPool4 + GaussianBlur(3x3).
CYPERSTEREO_AVX2_TARGET inline void FusedQuarterPoolDecimRowAvx2(
    const cv::Mat &y8, const cv::Mat &y_bl, int yq, uchar *hf,
    uchar *yq_row, ushort *sq_row) {
  const int qW = y8.cols / 4;
  const __m256i ones8 = _mm256_set1_epi8(1);
  const __m256i ones16 = _mm256_set1_epi16(1);
  const __m256i round8 = _mm256_set1_epi32(8);
  const __m256i pick4 = _mm256_setr_epi8(
      0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  int xq = 0;
  for (; xq + 8 <= qW; xq += 8) {
    const int x = 4 * xq;
    __m256i sums = _mm256_setzero_si256();
    for (int r = 0; r < 4; ++r) {
      const uchar *pa = y8.ptr<uchar>(4 * yq + r) + x;
      const uchar *pb = y_bl.ptr<uchar>(4 * yq + r) + x;
      const __m256i a = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(pa));
      const __m256i b = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(pb));
      const __m256i d = _mm256_sub_epi8(_mm256_max_epu8(a, b),
                                         _mm256_min_epu8(a, b));
      sums = _mm256_add_epi32(
          sums, _mm256_madd_epi16(_mm256_maddubs_epi16(d, ones8), ones16));
      if (r == 0) {
        const __m256i picked = _mm256_shuffle_epi8(b, pick4);
        const __m128i packed8 = _mm_unpacklo_epi32(
            _mm256_castsi256_si128(picked),
            _mm256_extracti128_si256(picked, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(yq_row + xq), packed8);
        const __m128i values16 = _mm_cvtepu8_epi16(packed8);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(sq_row + xq),
                         _mm_mullo_epi16(values16, values16));
      }
    }
    sums = _mm256_srli_epi32(_mm256_add_epi32(sums, round8), 4);
    const __m128i values16 = _mm_packus_epi32(
        _mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    _mm_storel_epi64(reinterpret_cast<__m128i *>(hf + xq),
                     _mm_packus_epi16(values16, _mm_setzero_si128()));
  }
  for (; xq < qW; ++xq) {
    const int x = 4 * xq;
    int sum = 0;
    for (int r = 0; r < 4; ++r) {
      const uchar *pa = y8.ptr<uchar>(4 * yq + r);
      const uchar *pb = y_bl.ptr<uchar>(4 * yq + r);
      for (int k = 0; k < 4; ++k)
        sum += std::abs(static_cast<int>(pa[x + k]) -
                        static_cast<int>(pb[x + k]));
    }
    const uchar value = y_bl.ptr<uchar>(4 * yq)[x];
    yq_row[xq] = value;
    sq_row[xq] = static_cast<ushort>(value * value);
    hf[xq] = static_cast<uchar>((sum + 8) >> 4);
  }
}

CYPERSTEREO_AVX2_TARGET inline void FusedQuarterGauss3HRowAvx2(
    const uchar *src, ushort *dst, int width) {
  dst[0] = static_cast<ushort>(2 * src[0] + 2 * src[1]);
  int x = 1;
  for (; x + 16 <= width - 1; x += 16) {
    const __m256i left = _mm256_cvtepu8_epi16(_mm_loadu_si128(
        reinterpret_cast<const __m128i *>(src + x - 1)));
    const __m256i center = _mm256_cvtepu8_epi16(_mm_loadu_si128(
        reinterpret_cast<const __m128i *>(src + x)));
    const __m256i right = _mm256_cvtepu8_epi16(_mm_loadu_si128(
        reinterpret_cast<const __m128i *>(src + x + 1)));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(dst + x),
        _mm256_add_epi16(_mm256_add_epi16(left, right),
                         _mm256_slli_epi16(center, 1)));
  }
  for (; x < width - 1; ++x)
    dst[x] = static_cast<ushort>(src[x - 1] + 2 * src[x] + src[x + 1]);
  dst[width - 1] =
      static_cast<ushort>(2 * src[width - 2] + 2 * src[width - 1]);
}

CYPERSTEREO_AVX2_TARGET inline void FusedQuarterGauss3VRowAvx2(
    const ushort *top, const ushort *center, const ushort *bottom,
    uchar *dst, int width) {
  const __m256i round = _mm256_set1_epi16(8);
  int x = 0;
  for (; x + 16 <= width; x += 16) {
    __m256i value = _mm256_add_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(top + x)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bottom + x)));
    value = _mm256_add_epi16(
        value, _mm256_slli_epi16(_mm256_loadu_si256(
                                      reinterpret_cast<const __m256i *>(
                                          center + x)),
                                  1));
    value = _mm256_srli_epi16(_mm256_add_epi16(value, round), 4);
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(dst + x),
        _mm_packus_epi16(_mm256_castsi256_si128(value),
                         _mm256_extracti128_si256(value, 1)));
  }
  for (; x < width; ++x)
    dst[x] = static_cast<uchar>(
        (top[x] + 2 * center[x] + bottom[x] + 8) >> 4);
}

CYPERSTEREO_AVX2_TARGET inline void FusedQuarterTextureAvx2(
    const cv::Mat &y8, const cv::Mat &y_bl, cv::Mat &tex_q, cv::Mat &yq8,
    cv::Mat &sq16, cv::Mat &hf_ring, cv::Mat &hs_ring) {
  const int qW = y8.cols / 4, qH = y8.rows / 4;
  CV_Assert(qW >= 2 && qH >= 2);
  tex_q.create(qH, qW, CV_8U);
  yq8.create(qH, qW, CV_8U);
  sq16.create(qH, qW, CV_16U);
  hf_ring.create(3, qW, CV_8U);
  hs_ring.create(3, qW, CV_16U);
  const auto fill = [&](int y) {
    uchar *hf = hf_ring.ptr<uchar>(y % 3);
    FusedQuarterPoolDecimRowAvx2(y8, y_bl, y, hf, yq8.ptr<uchar>(y),
                                 sq16.ptr<ushort>(y));
    FusedQuarterGauss3HRowAvx2(hf, hs_ring.ptr<ushort>(y % 3), qW);
  };
  fill(0);
  fill(1);
  FusedQuarterGauss3VRowAvx2(
      hs_ring.ptr<ushort>(1), hs_ring.ptr<ushort>(0),
      hs_ring.ptr<ushort>(1), tex_q.ptr<uchar>(0), qW);
  for (int y = 1; y < qH - 1; ++y) {
    fill(y + 1);
    FusedQuarterGauss3VRowAvx2(
        hs_ring.ptr<ushort>((y - 1) % 3), hs_ring.ptr<ushort>(y % 3),
        hs_ring.ptr<ushort>((y + 1) % 3), tex_q.ptr<uchar>(y), qW);
  }
  FusedQuarterGauss3VRowAvx2(
      hs_ring.ptr<ushort>((qH - 2) % 3),
      hs_ring.ptr<ushort>((qH - 1) % 3),
      hs_ring.ptr<ushort>((qH - 2) % 3), tex_q.ptr<uchar>(qH - 1), qW);
}

// Bit-exact AVX2 implementation of the quarter-grid guided-filter
// coefficient quantization. Keep the subtraction in b = (m - a*m): the
// algebraically equivalent m*(1-a) changes a few Q4 coefficients by one LSB.
CYPERSTEREO_AVX2_TARGET inline void FastGuidedRawCoeffAvx2(
    const cv::Mat &mean_i, const cv::Mat &mean_ii, float eps,
    cv::Mat &a_q, cv::Mat &b_q) {
  a_q.create(mean_i.size(), CV_16U);
  b_q.create(mean_i.size(), CV_16U);
  const __m256 veps = _mm256_set1_ps(eps);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 q12 = _mm256_set1_ps(4096.0f);
  const __m256 q4 = _mm256_set1_ps(16.0f);
  for (int y = 0; y < mean_i.rows; ++y) {
    const float *pi = mean_i.ptr<float>(y);
    const float *pii = mean_ii.ptr<float>(y);
    ushort *pa = a_q.ptr<ushort>(y);
    ushort *pb = b_q.ptr<ushort>(y);
    int x = 0;
    for (; x + 8 <= mean_i.cols; x += 8) {
      const __m256 m = _mm256_loadu_ps(pi + x);
      __m256 v = _mm256_sub_ps(_mm256_loadu_ps(pii + x),
                               _mm256_mul_ps(m, m));
      v = _mm256_max_ps(v, zero);
      const __m256 a = _mm256_div_ps(v, _mm256_add_ps(v, veps));
      const __m256i ai = _mm256_cvttps_epi32(
          _mm256_add_ps(_mm256_mul_ps(a, q12), half));
      const __m256i bi = _mm256_cvttps_epi32(_mm256_add_ps(
          _mm256_mul_ps(_mm256_sub_ps(m, _mm256_mul_ps(a, m)), q4), half));
      const __m128i a16 = _mm_packus_epi32(
          _mm256_castsi256_si128(ai), _mm256_extracti128_si256(ai, 1));
      const __m128i b16 = _mm_packus_epi32(
          _mm256_castsi256_si128(bi), _mm256_extracti128_si256(bi, 1));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pa + x), a16);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pb + x), b16);
    }
    for (; x < mean_i.cols; ++x) {
      const float m = pi[x];
      float v = pii[x] - m * m;
      if (v < 0.f) v = 0.f;
      const float a = v / (v + eps);
      pa[x] = static_cast<ushort>(a * 4096.0f + 0.5f);
      pb[x] = static_cast<ushort>((m - a * m) * 16.0f + 0.5f);
    }
  }
}

// Apply the gain-adaptive guided-filter strength to 16 Q12/Q4 coefficient
// pairs at a time. All intermediates fit in unsigned 20 bits.
CYPERSTEREO_AVX2_TARGET inline void FastGuidedStrengthAvx2(
    cv::Mat &a_q16, cv::Mat &b_q16, int strength_q8) {
  const __m256i strength = _mm256_set1_epi32(strength_q8);
  const __m256i round = _mm256_set1_epi32(128);
  const __m256i one = _mm256_set1_epi32(4096);
  for (int y = 0; y < a_q16.rows; ++y) {
    ushort *pa = a_q16.ptr<ushort>(y);
    ushort *pb = b_q16.ptr<ushort>(y);
    int x = 0;
    for (; x + 16 <= a_q16.cols; x += 16) {
      const __m256i av16 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(pa + x));
      const __m256i bv16 = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(pb + x));
      __m256i alo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(av16));
      __m256i ahi =
          _mm256_cvtepu16_epi32(_mm256_extracti128_si256(av16, 1));
      __m256i blo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(bv16));
      __m256i bhi =
          _mm256_cvtepu16_epi32(_mm256_extracti128_si256(bv16, 1));
      alo = _mm256_sub_epi32(
          one, _mm256_srli_epi32(
                   _mm256_add_epi32(
                       _mm256_mullo_epi32(_mm256_sub_epi32(one, alo),
                                          strength),
                       round),
                   8));
      ahi = _mm256_sub_epi32(
          one, _mm256_srli_epi32(
                   _mm256_add_epi32(
                       _mm256_mullo_epi32(_mm256_sub_epi32(one, ahi),
                                          strength),
                       round),
                   8));
      blo = _mm256_srli_epi32(
          _mm256_add_epi32(_mm256_mullo_epi32(blo, strength), round), 8);
      bhi = _mm256_srli_epi32(
          _mm256_add_epi32(_mm256_mullo_epi32(bhi, strength), round), 8);
      const __m128i ao0 = _mm_packus_epi32(
          _mm256_castsi256_si128(alo), _mm256_extracti128_si256(alo, 1));
      const __m128i ao1 = _mm_packus_epi32(
          _mm256_castsi256_si128(ahi), _mm256_extracti128_si256(ahi, 1));
      const __m128i bo0 = _mm_packus_epi32(
          _mm256_castsi256_si128(blo), _mm256_extracti128_si256(blo, 1));
      const __m128i bo1 = _mm_packus_epi32(
          _mm256_castsi256_si128(bhi), _mm256_extracti128_si256(bhi, 1));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pa + x), ao0);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pa + x + 8), ao1);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pb + x), bo0);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pb + x + 8), bo1);
    }
    for (; x < a_q16.cols; ++x) {
      const int one_minus_a = 4096 - pa[x];
      pa[x] = static_cast<ushort>(
          4096 - ((one_minus_a * strength_q8 + 128) >> 8));
      pb[x] = static_cast<ushort>(
          (static_cast<int>(pb[x]) * strength_q8 + 128) >> 8);
    }
  }
}

CYPERSTEREO_AVX2_TARGET inline __m256i PackAndRepeat2Avx2(__m256i v) {
  const __m128i v8 = _mm_packs_epi32(
      _mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  return _mm256_set_m128i(_mm_unpackhi_epi16(v8, v8),
                          _mm_unpacklo_epi16(v8, v8));
}

// OpenCV 4.2 expands v_store_interleave into a long unpack/shift/or chain.
// Fixed byte shuffles write the same 16 BGR pixels with fewer instructions.
CYPERSTEREO_AVX2_TARGET inline void FastOutputStoreBgr16Avx2(
    uchar *dst, __m128i blue, __m128i green, __m128i red) {
  constexpr char Z = static_cast<char>(-128);
  const __m128i b0 =
      _mm_setr_epi8(0, Z, Z, 1, Z, Z, 2, Z, Z, 3, Z, Z, 4, Z, Z, 5);
  const __m128i g0 =
      _mm_setr_epi8(Z, 0, Z, Z, 1, Z, Z, 2, Z, Z, 3, Z, Z, 4, Z, Z);
  const __m128i r0 =
      _mm_setr_epi8(Z, Z, 0, Z, Z, 1, Z, Z, 2, Z, Z, 3, Z, Z, 4, Z);

  const __m128i b1 =
      _mm_setr_epi8(Z, Z, 6, Z, Z, 7, Z, Z, 8, Z, Z, 9, Z, Z, 10, Z);
  const __m128i g1 =
      _mm_setr_epi8(5, Z, Z, 6, Z, Z, 7, Z, Z, 8, Z, Z, 9, Z, Z, 10);
  const __m128i r1 =
      _mm_setr_epi8(Z, 5, Z, Z, 6, Z, Z, 7, Z, Z, 8, Z, Z, 9, Z, Z);

  const __m128i b2 =
      _mm_setr_epi8(Z, 11, Z, Z, 12, Z, Z, 13, Z, Z, 14, Z, Z, 15, Z, Z);
  const __m128i g2 =
      _mm_setr_epi8(Z, Z, 11, Z, Z, 12, Z, Z, 13, Z, Z, 14, Z, Z, 15, Z);
  const __m128i r2 =
      _mm_setr_epi8(10, Z, Z, 11, Z, Z, 12, Z, Z, 13, Z, Z, 14, Z, Z, 15);

  const __m128i o0 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(blue, b0),
                   _mm_shuffle_epi8(green, g0)),
      _mm_shuffle_epi8(red, r0));
  const __m128i o1 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(blue, b1),
                   _mm_shuffle_epi8(green, g1)),
      _mm_shuffle_epi8(red, r1));
  const __m128i o2 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(blue, b2),
                   _mm_shuffle_epi8(green, g2)),
      _mm_shuffle_epi8(red, r2));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), o0);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 16), o1);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 32), o2);
}

// AVX2 backend for the final luma-MAC + 4:2:0 chroma upsample +
// YCrCb-to-BGR stage. It evaluates 16 full-resolution pixels per dependency
// chain using 256-bit arithmetic, then uses OpenCV's proven 3-channel SSE
// store. The integer equations and rounding are exactly the same as the
// portable backend below.
CYPERSTEREO_AVX2_TARGET inline void FusedOutputAvx2(
    const cv::Mat &y8, const cv::Mat &a_q16, const cv::Mat &b_q16,
    const cv::Mat &cr_h, const cv::Mat &cb_h, cv::Mat &color,
    bool apply_tone = false) {
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
  const FastBalancedToneLuts *tone_luts =
      apply_tone ? &GetFastBalancedToneLuts() : nullptr;

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
      __m128i outb, outg, outr;
      if (tone_luts) {
        const __m256i zero16 = _mm256_setzero_si256();
        const __m256i max255_16 = _mm256_set1_epi16(255);
        const auto clip16 = [&](const __m256i value) CYPERSTEREO_AVX2_TARGET {
          return _mm256_min_epi16(max255_16,
                                  _mm256_max_epi16(zero16, value));
        };
        const __m256i b16 = clip16(outb16);
        const __m256i g16 = clip16(outg16);
        const __m256i r16 = clip16(outr16);
        const __m256i b32_lo =
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b16));
        const __m256i b32_hi =
            _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b16, 1));
        const __m256i g32_lo =
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(g16));
        const __m256i g32_hi =
            _mm256_cvtepu16_epi32(_mm256_extracti128_si256(g16, 1));
        const __m256i r32_lo =
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r16));
        const __m256i r32_hi =
            _mm256_cvtepu16_epi32(_mm256_extracti128_si256(r16, 1));
        const __m256i k29_32 = _mm256_set1_epi32(29);
        const __m256i k150_32 = _mm256_set1_epi32(150);
        const __m256i k77_32 = _mm256_set1_epi32(77);
        const __m256i round128_32 = _mm256_set1_epi32(128);
        const auto luma8 = [&](const __m256i bb, const __m256i gg,
                               const __m256i rr) CYPERSTEREO_AVX2_TARGET {
          return _mm256_srli_epi32(
              _mm256_add_epi32(
                  _mm256_add_epi32(_mm256_mullo_epi32(bb, k29_32),
                                   _mm256_mullo_epi32(gg, k150_32)),
                  _mm256_add_epi32(_mm256_mullo_epi32(rr, k77_32),
                                   round128_32)),
              8);
        };
        const __m256i y32_lo = luma8(b32_lo, g32_lo, r32_lo);
        const __m256i y32_hi = luma8(b32_hi, g32_hi, r32_hi);
        const __m256i max32_lo =
            _mm256_max_epi32(b32_lo, _mm256_max_epi32(g32_lo, r32_lo));
        const __m256i max32_hi =
            _mm256_max_epi32(b32_hi, _mm256_max_epi32(g32_hi, r32_hi));
        const int *gain_table = tone_luts->gain_q12_i32.data();
        const int *clip_table = tone_luts->clip_q12_i32.data();
        const __m256i gain_lo = _mm256_min_epi32(
            _mm256_i32gather_epi32(gain_table, y32_lo, 4),
            _mm256_i32gather_epi32(clip_table, max32_lo, 4));
        const __m256i gain_hi = _mm256_min_epi32(
            _mm256_i32gather_epi32(gain_table, y32_hi, 4),
            _mm256_i32gather_epi32(clip_table, max32_hi, 4));
        const __m256i round2048_32 = _mm256_set1_epi32(2048);
        const auto tone_channel = [&](const __m256i lo,
                                      const __m256i hi)
            CYPERSTEREO_AVX2_TARGET {
          const __m256i toned_lo = _mm256_srli_epi32(
              _mm256_add_epi32(_mm256_mullo_epi32(lo, gain_lo),
                               round2048_32),
              12);
          const __m256i toned_hi = _mm256_srli_epi32(
              _mm256_add_epi32(_mm256_mullo_epi32(hi, gain_hi),
                               round2048_32),
              12);
          return _mm256_permute4x64_epi64(
              _mm256_packus_epi32(toned_lo, toned_hi), 0xd8);
        };
        const __m256i toned_b16 = tone_channel(b32_lo, b32_hi);
        const __m256i toned_g16 = tone_channel(g32_lo, g32_hi);
        const __m256i toned_r16 = tone_channel(r32_lo, r32_hi);
        outb = _mm_packus_epi16(_mm256_castsi256_si128(toned_b16),
                                _mm256_extracti128_si256(toned_b16, 1));
        outg = _mm_packus_epi16(_mm256_castsi256_si128(toned_g16),
                                _mm256_extracti128_si256(toned_g16, 1));
        outr = _mm_packus_epi16(_mm256_castsi256_si128(toned_r16),
                                _mm256_extracti128_si256(toned_r16, 1));
      } else {
        outb = _mm_packus_epi16(_mm256_castsi256_si128(outb16),
                                _mm256_extracti128_si256(outb16, 1));
        outg = _mm_packus_epi16(_mm256_castsi256_si128(outg16),
                                _mm256_extracti128_si256(outg16, 1));
        outr = _mm_packus_epi16(_mm256_castsi256_si128(outr16),
                                _mm256_extracti128_si256(outr16, 1));
      }
      cv::v_uint8x16 outb8(outb), outg8(outg), outr8(outr);
      cv::v_store_interleave(pc + 3 * x0, outb8, outg8, outr8);
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
        if (tone_luts)
          ApplyFastBalancedTonePixel(o[0], o[1], o[2], *tone_luts);
      }
    }
  }
}

// Two adjacent full-resolution rows share the same 4:2:0 chroma row and
// quarter-grid guided coefficients. Compute those terms once per pair while
// retaining the exact per-row tone and interleaved store equations.
CYPERSTEREO_AVX2_TARGET inline void FusedOutputAvx2Paired(
    const cv::Mat &y8, const cv::Mat &a_q16, const cv::Mat &b_q16,
    const cv::Mat &cr_h, const cv::Mat &cb_h, cv::Mat &color,
    bool apply_tone = false) {
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
  const __m256i zero16 = _mm256_setzero_si256();
  const __m256i max255_16 = _mm256_set1_epi16(255);
  const __m256i k29_16 = _mm256_set1_epi16(29);
  const __m256i k150_16 = _mm256_set1_epi16(150);
  const __m256i k77_16 = _mm256_set1_epi16(77);
  const __m256i round128_16 = _mm256_set1_epi16(128);
  const FastBalancedToneLuts *tone_luts =
      apply_tone ? &GetFastBalancedToneLuts() : nullptr;

  const int paired_height = full_height & ~1;
  for (int y = 0; y < paired_height; y += 2) {
    const int yh = (std::min)(y >> 1, half_height - 1);
    const int yq = (std::min)(y >> 2, quarter_height - 1);
    const uchar *py0 = y8.ptr<uchar>(y);
    const uchar *py1 = y8.ptr<uchar>(y + 1);
    uchar *pc0 = color.ptr<uchar>(y);
    uchar *pc1 = color.ptr<uchar>(y + 1);
    const ushort *pa = a_q16.ptr<ushort>(yq);
    const ushort *pb = b_q16.ptr<ushort>(yq);
    const uchar *pcr = cr_h.ptr<uchar>(yh);
    const uchar *pcb = cb_h.ptr<uchar>(yh);
    int xh = 0;

    for (; xh + 8 <= half_width; xh += 8) {
      const int x0 = xh << 1;
      const int xq = xh >> 1;
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
      const __m128i a_lo128 = _mm256_castsi256_si128(a16);
      const __m128i a_hi128 = _mm256_extracti128_si256(a16, 1);
      const __m128i b_lo128 = _mm256_castsi256_si128(b16);
      const __m128i b_hi128 = _mm256_extracti128_si256(b16, 1);
      const __m256i ab_lo = _mm256_set_m128i(
          _mm_unpackhi_epi16(a_lo128, b_lo128),
          _mm_unpacklo_epi16(a_lo128, b_lo128));
      const __m256i ab_hi = _mm256_set_m128i(
          _mm_unpackhi_epi16(a_hi128, b_hi128),
          _mm_unpacklo_epi16(a_hi128, b_hi128));

      const auto guided_luma16 = [&](const uchar *py)
          CYPERSTEREO_AVX2_TARGET {
        const __m256i y16 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(py + x0)));
        const __m128i y_lo128 = _mm256_castsi256_si128(y16);
        const __m128i y_hi128 = _mm256_extracti128_si256(y16, 1);
        const __m128i c256 = _mm_set1_epi16(256);
        const __m256i yb_lo = _mm256_set_m128i(
            _mm_unpackhi_epi16(y_lo128, c256),
            _mm_unpacklo_epi16(y_lo128, c256));
        const __m256i yb_hi = _mm256_set_m128i(
            _mm_unpackhi_epi16(y_hi128, c256),
            _mm_unpacklo_epi16(y_hi128, c256));
        const __m256i yn32_lo = _mm256_srli_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(ab_lo, yb_lo), round_y),
            12);
        const __m256i yn32_hi = _mm256_srli_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(ab_hi, yb_hi), round_y),
            12);
        return _mm256_permute4x64_epi64(
            _mm256_packus_epi32(yn32_lo, yn32_hi), 0xd8);
      };
      const __m256i yn16_0 = guided_luma16(py0);
      const __m256i yn16_1 = guided_luma16(py1);

      const __m256i dcr = _mm256_sub_epi32(
          _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pcr + xh))),
          c128_32);
      const __m256i dcb = _mm256_sub_epi32(
          _mm256_cvtepu8_epi32(
              _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pcb + xh))),
          c128_32);
      const __m256i tb16 = PackAndRepeat2Avx2(_mm256_srai_epi32(
          _mm256_mullo_epi32(dcb, k454), 8));
      const __m256i tg16 = PackAndRepeat2Avx2(_mm256_srai_epi32(
          _mm256_add_epi32(_mm256_mullo_epi32(dcr, k183),
                           _mm256_mullo_epi32(dcb, k88)),
          8));
      const __m256i tr16 = PackAndRepeat2Avx2(_mm256_srai_epi32(
          _mm256_mullo_epi32(dcr, k359), 8));

      const auto emit_row = [&](const __m256i yn16, uchar *pc)
          CYPERSTEREO_AVX2_TARGET {
        const __m256i outb16 = _mm256_add_epi16(yn16, tb16);
        const __m256i outg16 = _mm256_sub_epi16(yn16, tg16);
        const __m256i outr16 = _mm256_add_epi16(yn16, tr16);
        __m128i outb, outg, outr;
        if (tone_luts) {
          const auto clip16 = [&](const __m256i value)
              CYPERSTEREO_AVX2_TARGET {
            return _mm256_min_epi16(max255_16,
                                    _mm256_max_epi16(zero16, value));
          };
          const __m256i b_clipped = clip16(outb16);
          const __m256i g_clipped = clip16(outg16);
          const __m256i r_clipped = clip16(outr16);
          // The BT.601 numerator tops out at 65408, so luma can stay in
          // unsigned u16. Only the two 8-lane LUT indices need widening.
          const __m256i luma16 = _mm256_srli_epi16(
              _mm256_add_epi16(
                  _mm256_add_epi16(_mm256_mullo_epi16(b_clipped, k29_16),
                                   _mm256_mullo_epi16(g_clipped, k150_16)),
                  _mm256_add_epi16(_mm256_mullo_epi16(r_clipped, k77_16),
                                   round128_16)),
              8);
          const __m256i max16 = _mm256_max_epu16(
              b_clipped, _mm256_max_epu16(g_clipped, r_clipped));
          const __m256i luma_lo = _mm256_cvtepu16_epi32(
              _mm256_castsi256_si128(luma16));
          const __m256i luma_hi = _mm256_cvtepu16_epi32(
              _mm256_extracti128_si256(luma16, 1));
          const __m256i max_lo = _mm256_cvtepu16_epi32(
              _mm256_castsi256_si128(max16));
          const __m256i max_hi = _mm256_cvtepu16_epi32(
              _mm256_extracti128_si256(max16, 1));
          const int *gain_table = tone_luts->gain_q12_i32.data();
          const int *clip_table = tone_luts->clip_q12_i32.data();
          const __m256i gain_lo = _mm256_min_epi32(
              _mm256_i32gather_epi32(gain_table, luma_lo, 4),
              _mm256_i32gather_epi32(clip_table, max_lo, 4));
          const __m256i gain_hi = _mm256_min_epi32(
              _mm256_i32gather_epi32(gain_table, luma_hi, 4),
              _mm256_i32gather_epi32(clip_table, max_hi, 4));
          const __m256i gain16 = _mm256_permute4x64_epi64(
              _mm256_packus_epi32(gain_lo, gain_hi), 0xd8);
          // mulhrs performs the same rounded Q12 product after scaling the
          // gain by eight. The current effective gains are 0 or 4096..5958;
          // adding 2*value exactly compensates their signed high-bit wrap.
          const __m256i gain_scale = _mm256_slli_epi16(gain16, 3);
          const __m256i gain_scale_wrapped =
              _mm256_cmpgt_epi16(zero16, gain_scale);
          const auto tone_channel = [&](const __m256i value)
              CYPERSTEREO_AVX2_TARGET {
            return _mm256_add_epi16(
                _mm256_mulhrs_epi16(value, gain_scale),
                _mm256_and_si256(_mm256_slli_epi16(value, 1),
                                 gain_scale_wrapped));
          };
          const __m256i toned_b = tone_channel(b_clipped);
          const __m256i toned_g = tone_channel(g_clipped);
          const __m256i toned_r = tone_channel(r_clipped);
          outb = _mm_packus_epi16(_mm256_castsi256_si128(toned_b),
                                  _mm256_extracti128_si256(toned_b, 1));
          outg = _mm_packus_epi16(_mm256_castsi256_si128(toned_g),
                                  _mm256_extracti128_si256(toned_g, 1));
          outr = _mm_packus_epi16(_mm256_castsi256_si128(toned_r),
                                  _mm256_extracti128_si256(toned_r, 1));
        } else {
          outb = _mm_packus_epi16(_mm256_castsi256_si128(outb16),
                                  _mm256_extracti128_si256(outb16, 1));
          outg = _mm_packus_epi16(_mm256_castsi256_si128(outg16),
                                  _mm256_extracti128_si256(outg16, 1));
          outr = _mm_packus_epi16(_mm256_castsi256_si128(outr16),
                                  _mm256_extracti128_si256(outr16, 1));
        }
        FastOutputStoreBgr16Avx2(pc + 3 * x0, outb, outg, outr);
      };
      emit_row(yn16_0, pc0);
      emit_row(yn16_1, pc1);
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
      for (int row = 0; row < 2; ++row) {
        const uchar *py = row ? py1 : py0;
        uchar *pc = row ? pc1 : pc0;
        for (int k = 0; k < 2 && x0 + k < full_width; ++k) {
          const int yn = (a * py[x0 + k] + b256 + 2048) >> 12;
          uchar *out = pc + 3 * (x0 + k);
          out[0] = cv::saturate_cast<uchar>(yn + tb);
          out[1] = cv::saturate_cast<uchar>(yn - tg);
          out[2] = cv::saturate_cast<uchar>(yn + tr);
          if (tone_luts)
            ApplyFastBalancedTonePixel(out[0], out[1], out[2], *tone_luts);
        }
      }
    }
  }

  // Aligned camera frames never enter this tail; keep arbitrary SDK input
  // safe and identical to the scalar reference.
  for (int y = paired_height; y < full_height; ++y) {
    const int yh = (std::min)(y >> 1, half_height - 1);
    const int yq = (std::min)(y >> 2, quarter_height - 1);
    const uchar *py = y8.ptr<uchar>(y);
    uchar *pc = color.ptr<uchar>(y);
    const ushort *pa = a_q16.ptr<ushort>(yq);
    const ushort *pb = b_q16.ptr<ushort>(yq);
    const uchar *pcr = cr_h.ptr<uchar>(yh);
    const uchar *pcb = cb_h.ptr<uchar>(yh);
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
        uchar *out = pc + 3 * (x0 + k);
        out[0] = cv::saturate_cast<uchar>(yn + tb);
        out[1] = cv::saturate_cast<uchar>(yn - tg);
        out[2] = cv::saturate_cast<uchar>(yn + tr);
        if (tone_luts)
          ApplyFastBalancedTonePixel(out[0], out[1], out[2], *tone_luts);
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
    const cv::Mat &cr_h, const cv::Mat &cb_h, cv::Mat &color,
    bool apply_tone) {
  const int full_width = color.cols;
  const int full_height = color.rows;
  const int half_width = cr_h.cols;
  const int half_height = cr_h.rows;
  const int quarter_height = a_q16.rows;
  const uint32x4_t round_y = vdupq_n_u32(2048);
  const int16x8_t c128_16 = vdupq_n_s16(128);
  const FastBalancedToneLuts *tone_luts =
      apply_tone ? &GetFastBalancedToneLuts() : nullptr;

  const int paired_height = full_height & ~1;
  FastParallelForRows(paired_height / 2, "output", [&](int pair_row) {
    const int y = pair_row << 1;
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
        if (tone_luts) {
#if defined(__aarch64__)
          ApplyFastBalancedTone16Neon(out.val[0], out.val[1], out.val[2],
                                      *tone_luts);
#elif CV_SIMD128
          // Reuse the architecture-neutral reference helper after clipping,
          // while B/G/R are still planar in registers.  The helper uses the
          // same luma/max LUT indices and Q12 rounding as the scalar reference,
          // so fusing this step changes memory traffic, not colour arithmetic.
          cv::v_uint8x16 blue(out.val[0]);
          cv::v_uint8x16 green(out.val[1]);
          cv::v_uint8x16 red(out.val[2]);
          ApplyFastBalancedTone16(blue, green, red, *tone_luts);
          out.val[0] = blue.val;
          out.val[1] = green.val;
          out.val[2] = red.val;
#else
          alignas(16) uchar blue[16], green[16], red[16];
          vst1q_u8(blue, out.val[0]);
          vst1q_u8(green, out.val[1]);
          vst1q_u8(red, out.val[2]);
          for (int lane = 0; lane < 16; ++lane)
            ApplyFastBalancedTonePixel(blue[lane], green[lane], red[lane],
                                       *tone_luts);
          out.val[0] = vld1q_u8(blue);
          out.val[1] = vld1q_u8(green);
          out.val[2] = vld1q_u8(red);
#endif
        }
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
          if (tone_luts)
            ApplyFastBalancedTonePixel(o[0], o[1], o[2], *tone_luts);
        }
      }
    }
  });

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
        if (tone_luts)
          ApplyFastBalancedTonePixel(o[0], o[1], o[2], *tone_luts);
      }
    }
  }
}
#endif

// raw_wb: white-balanced Bayer plane, or nullptr. When given (ARM fused
// demosaic path) the EA demosaic runs fused with the front-end and the
// interleaved BGR frame is never materialized; `color` is only written by
// the output kernel at the end.
inline bool FastPostDetailProfileEnabled() {
  static const bool enabled =
      std::getenv("CYPERSTEREO_ISP_PROFILE_DETAIL") != nullptr;
  return enabled;
}

inline void AddFastPostDetailProfile(const std::array<double, 9> &ms) {
  static std::mutex mutex;
  static std::array<double, 9> sums{};
  static int count = 0;
  std::lock_guard<std::mutex> lock(mutex);
  for (size_t i = 0; i < sums.size(); ++i) sums[i] += ms[i];
  if (++count < 300) return;
  static const char *names[9] = {
      "front",          "luma",          "chroma-front",
      "chroma-filter",  "chroma-texture", "chroma-gate",
      "reconstruct",    "blend-hue",      "output-tone"};
  std::cout << "[isp-detail]";
  double total = 0.0;
  for (size_t i = 0; i < sums.size(); ++i) {
    const double value = sums[i] / count;
    total += value;
    std::cout << "  " << names[i] << " " << std::fixed
              << std::setprecision(2) << value;
  }
  std::cout << "  total " << total << " ms" << std::endl;
  sums.fill(0.0);
  count = 0;
}

#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
// Exact floor(value / 255) for every non-negative value used by the quarter
// chroma gate (the largest one is 255*255).  The identity avoids scalar
// integer divides while retaining C++ truncation exactly.
inline uint32x4_t FastDiv255U32Neon(const uint32x4_t value) {
  return vshrq_n_u32(
      vaddq_u32(vaddq_u32(value, vdupq_n_u32(1)),
                vshrq_n_u32(value, 8)),
      8);
}

inline int32x4_t FastClamp255S32Neon(const int32x4_t value) {
  return vminq_s32(vdupq_n_s32(255),
                   vmaxq_s32(vdupq_n_s32(0), value));
}

inline void FastLoad8U32Neon(const uchar *source, uint32x4_t &lo,
                             uint32x4_t &hi) {
  const uint16x8_t value16 = vmovl_u8(vld1_u8(source));
  lo = vmovl_u16(vget_low_u16(value16));
  hi = vmovl_u16(vget_high_u16(value16));
}

inline int16x8_t FastCenter8U8Neon(const uint8x8_t value) {
  return vreinterpretq_s16_u16(
      vsubq_u16(vmovl_u8(value), vdupq_n_u16(128)));
}

inline void FastChromaPairStats16Neon(
    const uchar *row0, const uchar *row1, int32x4_t &sum_lo,
    int32x4_t &sum_hi, uint32x4_t &abs_lo, uint32x4_t &abs_hi) {
  const uint8x16_t source0 = vld1q_u8(row0);
  const uint8x16_t source1 = vld1q_u8(row1);
  const int16x8_t value00 = FastCenter8U8Neon(vget_low_u8(source0));
  const int16x8_t value01 = FastCenter8U8Neon(vget_high_u8(source0));
  const int16x8_t value10 = FastCenter8U8Neon(vget_low_u8(source1));
  const int16x8_t value11 = FastCenter8U8Neon(vget_high_u8(source1));
  sum_lo = vaddq_s32(vpaddlq_s16(value00), vpaddlq_s16(value10));
  sum_hi = vaddq_s32(vpaddlq_s16(value01), vpaddlq_s16(value11));
  abs_lo = vaddq_u32(
      vpaddlq_u16(vreinterpretq_u16_s16(vabsq_s16(value00))),
      vpaddlq_u16(vreinterpretq_u16_s16(vabsq_s16(value10))));
  abs_hi = vaddq_u32(
      vpaddlq_u16(vreinterpretq_u16_s16(vabsq_s16(value01))),
      vpaddlq_u16(vreinterpretq_u16_s16(vabsq_s16(value11))));
}

inline void FastChromaGate4Neon(
    const int32x4_t sum_cr, const int32x4_t sum_cb,
    const uint32x4_t abs_sum, const uint32x4_t texture_source,
    const float32x4_t mean_luma, const uint32x4_t base_cr_source,
    const uint32x4_t base_cb_source, const float shadow_lo,
    const float shadow_scale, const int sat_q128, uint32x4_t &residual_q7,
    uint32x4_t &blend, uint32x4_t &hue, uint32x4_t &min_keep) {
  const int32x4_t zero32 = vdupq_n_s32(0);
  const uint32x4_t c255 = vdupq_n_u32(255);
  const uint32x4_t vector_sum = vaddq_u32(
      vreinterpretq_u32_s32(vabsq_s32(sum_cr)),
      vreinterpretq_u32_s32(vabsq_s32(sum_cb)));

  const uint32x4_t texture = vreinterpretq_u32_s32(FastClamp255S32Neon(
      vsubq_s32(vreinterpretq_s32_u32(vmulq_n_u32(texture_source, 30)),
                vdupq_n_s32(195))));
  const float32x4_t shade_float = vmulq_f32(
      vsubq_f32(mean_luma, vdupq_n_f32(shadow_lo)),
      vdupq_n_f32(shadow_scale));
  const uint32x4_t shade = vreinterpretq_u32_s32(
      FastClamp255S32Neon(vcvtq_s32_f32(shade_float)));

  const int32x4_t base_cr = vsubq_s32(
      vreinterpretq_s32_u32(base_cr_source), vdupq_n_s32(128));
  const int32x4_t base_cb = vsubq_s32(
      vreinterpretq_s32_u32(base_cb_source), vdupq_n_s32(128));
  const int32x4_t base_chroma = vaddq_s32(vabsq_s32(base_cr),
                                           vabsq_s32(base_cb));
  const uint32x4_t protect_delta = vreinterpretq_u32_s32(vminq_s32(
      vdupq_n_s32(18),
      vmaxq_s32(zero32, vsubq_s32(base_chroma, vdupq_n_s32(10)))));
  // floor(delta*255/18), delta in [0,18].  This reciprocal is exact for all
  // nineteen possible inputs (the same identity as the AVX2 path).
  const uint32x4_t protect =
      vshrq_n_u32(vmulq_n_u32(protect_delta, 928455), 16);

  uint32x4_t residual_keep = FastDiv255U32Neon(
      vmulq_u32(shade, vsubq_u32(c255, texture)));
  const uint32x4_t local_chroma =
      vshrq_n_u32(vaddq_u32(vector_sum, vdupq_n_u32(2)), 2);
  const uint32x4_t local_delta = vreinterpretq_u32_s32(vminq_s32(
      vdupq_n_s32(16),
      vmaxq_s32(zero32,
                vsubq_s32(vreinterpretq_s32_u32(local_chroma),
                          vdupq_n_s32(4)))));
  uint32x4_t local_protect =
      vshrq_n_u32(vmulq_n_u32(local_delta, 255), 4);
  const uint32x4_t adjusted_local =
      FastDiv255U32Neon(vmulq_u32(local_protect, protect));
  local_protect = vbslq_u32(vcgtq_u32(texture, vdupq_n_u32(0)),
                            adjusted_local, local_protect);
  const uint32x4_t coherent = vcgeq_u32(
      vshlq_n_u32(vector_sum, 2), vmulq_n_u32(abs_sum, 3));
  local_protect = vandq_u32(local_protect, coherent);
  residual_keep = vmaxq_u32(residual_keep, local_protect);
  const uint32x4_t color_protect = vmaxq_u32(protect, local_protect);

  residual_q7 = vminq_u32(
      c255, FastDiv255U32Neon(
                vmulq_u32(residual_keep,
                          vdupq_n_u32(static_cast<unsigned>(sat_q128)))));
  const uint32x4_t base_blend = vsubq_u32(
      vdupq_n_u32(144),
      FastDiv255U32Neon(vmulq_n_u32(color_protect, 104)));
  const uint32x4_t texture_blend = FastDiv255U32Neon(
      vmulq_u32(texture, vsubq_u32(c255, protect)));
  blend = vmaxq_u32(base_blend, texture_blend);
  hue = color_protect;
  min_keep = FastDiv255U32Neon(vmulq_n_u32(protect, 216));
}

inline void FastStoreGate8Neon(uchar *destination, const uint32x4_t lo,
                               const uint32x4_t hi) {
  vst1_u8(destination,
          vmovn_u16(vcombine_u16(vmovn_u32(lo), vmovn_u32(hi))));
}

// Evaluate eight quarter cells (a 16x2 block of half-resolution Cr/Cb) and
// emit all four gate maps in one NEON pass.
inline void FastChromaGate8Neon(
    const uchar *src_cr0, const uchar *src_cr1, const uchar *src_cb0,
    const uchar *src_cb1, const uchar *texture, const float *mean_luma,
    const uchar *base_cr, const uchar *base_cb, const float shadow_lo,
    const float shadow_scale, const int sat_q128, uchar *residual_q7,
    uchar *blend, uchar *hue, uchar *min_keep) {
  int32x4_t sum_cr_lo, sum_cr_hi, sum_cb_lo, sum_cb_hi;
  uint32x4_t abs_cr_lo, abs_cr_hi, abs_cb_lo, abs_cb_hi;
  FastChromaPairStats16Neon(src_cr0, src_cr1, sum_cr_lo, sum_cr_hi,
                            abs_cr_lo, abs_cr_hi);
  FastChromaPairStats16Neon(src_cb0, src_cb1, sum_cb_lo, sum_cb_hi,
                            abs_cb_lo, abs_cb_hi);
  uint32x4_t texture_lo, texture_hi, base_cr_lo, base_cr_hi;
  uint32x4_t base_cb_lo, base_cb_hi;
  FastLoad8U32Neon(texture, texture_lo, texture_hi);
  FastLoad8U32Neon(base_cr, base_cr_lo, base_cr_hi);
  FastLoad8U32Neon(base_cb, base_cb_lo, base_cb_hi);
  uint32x4_t residual_lo, residual_hi, blend_lo, blend_hi;
  uint32x4_t hue_lo, hue_hi, min_lo, min_hi;
  FastChromaGate4Neon(
      sum_cr_lo, sum_cb_lo, vaddq_u32(abs_cr_lo, abs_cb_lo), texture_lo,
      vld1q_f32(mean_luma), base_cr_lo, base_cb_lo, shadow_lo, shadow_scale,
      sat_q128, residual_lo, blend_lo, hue_lo, min_lo);
  FastChromaGate4Neon(
      sum_cr_hi, sum_cb_hi, vaddq_u32(abs_cr_hi, abs_cb_hi), texture_hi,
      vld1q_f32(mean_luma + 4), base_cr_hi, base_cb_hi, shadow_lo,
      shadow_scale, sat_q128, residual_hi, blend_hi, hue_hi, min_hi);
  FastStoreGate8Neon(residual_q7, residual_lo, residual_hi);
  FastStoreGate8Neon(blend, blend_lo, blend_hi);
  FastStoreGate8Neon(hue, hue_lo, hue_hi);
  FastStoreGate8Neon(min_keep, min_lo, min_hi);
}

inline uint8x16_t FastRepeatEach2U8Neon(const uint8x8_t value) {
  const uint8x8x2_t repeated = vzip_u8(value, value);
  return vcombine_u8(repeated.val[0], repeated.val[1]);
}

inline int16x8_t FastBlendChroma8Neon(const uint16x8_t source,
                                      const uint16x8_t filtered,
                                      const uint16x8_t alpha) {
  const int16x8_t source_s = vreinterpretq_s16_u16(source);
  const int16x8_t delta = vreinterpretq_s16_u16(vsubq_u16(filtered, source));
  const uint16x8_t magnitude =
      vreinterpretq_u16_s16(vabsq_s16(delta));
  const uint16x8_t rounded = vshrq_n_u16(
      vaddq_u16(vmulq_u16(magnitude, alpha), vdupq_n_u16(128)), 8);
  const int16x8_t sign = vshrq_n_s16(delta, 15);
  const int16x8_t signed_rounded = vsubq_s16(
      vreinterpretq_s16_u16(
          veorq_u16(rounded, vreinterpretq_u16_s16(sign))),
      sign);
  return vaddq_s16(source_s, signed_rounded);
}

inline uint16x8_t FastHueViolations8Neon(
    const int16x8_t cr0, const int16x8_t cb0, const int16x8_t cr1,
    const int16x8_t cb1, const uint16x8_t active,
    const uint16x8_t low_mag) {
  const int32x4_t dot_lo = vmlal_s16(
      vmull_s16(vget_low_s16(cr0), vget_low_s16(cr1)),
      vget_low_s16(cb0), vget_low_s16(cb1));
  const int32x4_t dot_hi = vmlal_s16(
      vmull_s16(vget_high_s16(cr0), vget_high_s16(cr1)),
      vget_high_s16(cb0), vget_high_s16(cb1));
  const int32x4_t cross_lo = vabsq_s32(vsubq_s32(
      vmull_s16(vget_low_s16(cr0), vget_low_s16(cb1)),
      vmull_s16(vget_low_s16(cb0), vget_low_s16(cr1))));
  const int32x4_t cross_hi = vabsq_s32(vsubq_s32(
      vmull_s16(vget_high_s16(cr0), vget_high_s16(cb1)),
      vmull_s16(vget_high_s16(cb0), vget_high_s16(cr1))));
  uint32x4_t bad_lo = vorrq_u32(
      vcleq_s32(dot_lo, vdupq_n_s32(0)),
      vcgtq_s32(vshlq_n_s32(cross_lo, 8), vmulq_n_s32(dot_lo, 9)));
  uint32x4_t bad_hi = vorrq_u32(
      vcleq_s32(dot_hi, vdupq_n_s32(0)),
      vcgtq_s32(vshlq_n_s32(cross_hi, 8), vmulq_n_s32(dot_hi, 9)));
  const int16x8_t low_s = vreinterpretq_s16_u16(low_mag);
  const int16x8_t active_s = vreinterpretq_s16_u16(active);
  bad_lo = vandq_u32(
      vorrq_u32(bad_lo, vreinterpretq_u32_s32(vmovl_s16(
                              vget_low_s16(low_s)))),
      vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(active_s))));
  bad_hi = vandq_u32(
      vorrq_u32(bad_hi, vreinterpretq_u32_s32(vmovl_s16(
                              vget_high_s16(low_s)))),
      vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(active_s))));
  return vcombine_u16(vmovn_u32(bad_lo), vmovn_u32(bad_hi));
}

// For d=1..256 this produces exactly (32768+d/2)/d.  There is no half-way
// quotient in this divisor range, and binary32 division error is far below
// the nearest quotient boundary; an exhaustive host-side check guards the
// identity in the accompanying validation program.
inline void FastHueReciprocal8Neon(const uint16x8_t divisor,
                                   uint32x4_t &reciprocal_lo,
                                   uint32x4_t &reciprocal_hi) {
  const uint16x8_t safe_divisor = vmaxq_u16(divisor, vdupq_n_u16(1));
  const uint32x4_t divisor_lo = vmovl_u16(vget_low_u16(safe_divisor));
  const uint32x4_t divisor_hi = vmovl_u16(vget_high_u16(safe_divisor));
  const float32x4_t numerator = vdupq_n_f32(32768.0f);
  const float32x4_t half = vdupq_n_f32(0.5f);
  reciprocal_lo = vcvtq_u32_f32(vaddq_f32(
      vdivq_f32(numerator, vcvtq_f32_u32(divisor_lo)), half));
  reciprocal_hi = vcvtq_u32_f32(vaddq_f32(
      vdivq_f32(numerator, vcvtq_f32_u32(divisor_hi)), half));
}

inline int16x8_t FastScaleHue8Neon(const int16x8_t value,
                                   const uint16x8_t target,
                                   const uint32x4_t reciprocal_lo,
                                   const uint32x4_t reciprocal_hi) {
  const uint16x8_t magnitude =
      vreinterpretq_u16_s16(vabsq_s16(value));
  uint32x4_t scaled_lo = vmull_u16(vget_low_u16(magnitude),
                                    vget_low_u16(target));
  uint32x4_t scaled_hi = vmull_u16(vget_high_u16(magnitude),
                                    vget_high_u16(target));
  scaled_lo = vshrq_n_u32(
      vaddq_u32(vmulq_u32(scaled_lo, reciprocal_lo),
                vdupq_n_u32(16384)),
      15);
  scaled_hi = vshrq_n_u32(
      vaddq_u32(vmulq_u32(scaled_hi, reciprocal_hi),
                vdupq_n_u32(16384)),
      15);
  int32x4_t signed_lo = vreinterpretq_s32_u32(scaled_lo);
  int32x4_t signed_hi = vreinterpretq_s32_u32(scaled_hi);
  signed_lo = vbslq_s32(
      vcltq_s32(vmovl_s16(vget_low_s16(value)), vdupq_n_s32(0)),
      vnegq_s32(signed_lo), signed_lo);
  signed_hi = vbslq_s32(
      vcltq_s32(vmovl_s16(vget_high_s16(value)), vdupq_n_s32(0)),
      vnegq_s32(signed_hi), signed_hi);
  return vcombine_s16(vmovn_s32(signed_lo), vmovn_s32(signed_hi));
}

inline void FastBlendHue16Neon(
    const uchar *source_cr, const uchar *source_cb,
    const uchar *filtered_cr, const uchar *filtered_cb,
    const uchar *blend_q8, const uchar *hue_active,
    const uchar *hue_min_keep, const bool apply_hue, uchar *output_cr,
    uchar *output_cb) {
  const uint8x16_t alpha8 = FastRepeatEach2U8Neon(vld1_u8(blend_q8));
  const uint16x8_t alpha_lo = vmovl_u8(vget_low_u8(alpha8));
  const uint16x8_t alpha_hi = vmovl_u8(vget_high_u8(alpha8));
  const uint8x16_t source_cr8 = vld1q_u8(source_cr);
  const uint8x16_t source_cb8 = vld1q_u8(source_cb);
  const uint8x16_t filtered_cr8 = vld1q_u8(filtered_cr);
  const uint8x16_t filtered_cb8 = vld1q_u8(filtered_cb);
  const uint16x8_t source_cr_lo = vmovl_u8(vget_low_u8(source_cr8));
  const uint16x8_t source_cr_hi = vmovl_u8(vget_high_u8(source_cr8));
  const uint16x8_t source_cb_lo = vmovl_u8(vget_low_u8(source_cb8));
  const uint16x8_t source_cb_hi = vmovl_u8(vget_high_u8(source_cb8));
  int16x8_t cr_out_lo = FastBlendChroma8Neon(
      source_cr_lo, vmovl_u8(vget_low_u8(filtered_cr8)), alpha_lo);
  int16x8_t cr_out_hi = FastBlendChroma8Neon(
      source_cr_hi, vmovl_u8(vget_high_u8(filtered_cr8)), alpha_hi);
  int16x8_t cb_out_lo = FastBlendChroma8Neon(
      source_cb_lo, vmovl_u8(vget_low_u8(filtered_cb8)), alpha_lo);
  int16x8_t cb_out_hi = FastBlendChroma8Neon(
      source_cb_hi, vmovl_u8(vget_high_u8(filtered_cb8)), alpha_hi);

  if (apply_hue) {
    const uint8x8_t active_q = vld1_u8(hue_active);
    if (vget_lane_u64(vreinterpret_u64_u8(active_q), 0) != 0) {
      const uint8x16_t active8 = FastRepeatEach2U8Neon(active_q);
      const uint8x16_t min_keep8 =
          FastRepeatEach2U8Neon(vld1_u8(hue_min_keep));
      const int16x8_t center = vdupq_n_s16(128);
      const int16x8_t cr0_lo =
          vsubq_s16(vreinterpretq_s16_u16(source_cr_lo), center);
      const int16x8_t cr0_hi =
          vsubq_s16(vreinterpretq_s16_u16(source_cr_hi), center);
      const int16x8_t cb0_lo =
          vsubq_s16(vreinterpretq_s16_u16(source_cb_lo), center);
      const int16x8_t cb0_hi =
          vsubq_s16(vreinterpretq_s16_u16(source_cb_hi), center);
      int16x8_t cr1_lo = vsubq_s16(cr_out_lo, center);
      int16x8_t cr1_hi = vsubq_s16(cr_out_hi, center);
      int16x8_t cb1_lo = vsubq_s16(cb_out_lo, center);
      int16x8_t cb1_hi = vsubq_s16(cb_out_hi, center);
      const uint16x8_t mag0_lo = vaddq_u16(
          vreinterpretq_u16_s16(vabsq_s16(cr0_lo)),
          vreinterpretq_u16_s16(vabsq_s16(cb0_lo)));
      const uint16x8_t mag0_hi = vaddq_u16(
          vreinterpretq_u16_s16(vabsq_s16(cr0_hi)),
          vreinterpretq_u16_s16(vabsq_s16(cb0_hi)));
      const uint16x8_t mag1_lo = vaddq_u16(
          vreinterpretq_u16_s16(vabsq_s16(cr1_lo)),
          vreinterpretq_u16_s16(vabsq_s16(cb1_lo)));
      const uint16x8_t mag1_hi = vaddq_u16(
          vreinterpretq_u16_s16(vabsq_s16(cr1_hi)),
          vreinterpretq_u16_s16(vabsq_s16(cb1_hi)));
      const uint16x8_t active_lo = vandq_u16(
          vcgtq_u16(vmovl_u8(vget_low_u8(active8)), vdupq_n_u16(0)),
          vcgtq_u16(mag0_lo, vdupq_n_u16(7)));
      const uint16x8_t active_hi = vandq_u16(
          vcgtq_u16(vmovl_u8(vget_high_u8(active8)), vdupq_n_u16(0)),
          vcgtq_u16(mag0_hi, vdupq_n_u16(7)));
      const uint16x8_t min_lo = vshrq_n_u16(
          vaddq_u16(
              vmulq_u16(mag0_lo, vmovl_u8(vget_low_u8(min_keep8))),
              vdupq_n_u16(128)),
          8);
      const uint16x8_t min_hi = vshrq_n_u16(
          vaddq_u16(
              vmulq_u16(mag0_hi, vmovl_u8(vget_high_u8(min_keep8))),
              vdupq_n_u16(128)),
          8);
      const uint16x8_t violation_lo = FastHueViolations8Neon(
          cr0_lo, cb0_lo, cr1_lo, cb1_lo, active_lo,
          vcgtq_u16(min_lo, mag1_lo));
      const uint16x8_t violation_hi = FastHueViolations8Neon(
          cr0_hi, cb0_hi, cr1_hi, cb1_hi, active_hi,
          vcgtq_u16(min_hi, mag1_hi));
      const uint64x2_t violations = vreinterpretq_u64_u16(
          vorrq_u16(violation_lo, violation_hi));
      if ((vgetq_lane_u64(violations, 0) |
           vgetq_lane_u64(violations, 1)) != 0) {
        uint32x4_t reciprocal_lo0, reciprocal_lo1;
        uint32x4_t reciprocal_hi0, reciprocal_hi1;
        FastHueReciprocal8Neon(mag0_lo, reciprocal_lo0, reciprocal_lo1);
        FastHueReciprocal8Neon(mag0_hi, reciprocal_hi0, reciprocal_hi1);
        const uint16x8_t target_lo = vmaxq_u16(mag1_lo, min_lo);
        const uint16x8_t target_hi = vmaxq_u16(mag1_hi, min_hi);
        int16x8_t scaled_cr_lo = FastScaleHue8Neon(
            cr0_lo, target_lo, reciprocal_lo0, reciprocal_lo1);
        int16x8_t scaled_cr_hi = FastScaleHue8Neon(
            cr0_hi, target_hi, reciprocal_hi0, reciprocal_hi1);
        int16x8_t scaled_cb_lo = FastScaleHue8Neon(
            cb0_lo, target_lo, reciprocal_lo0, reciprocal_lo1);
        int16x8_t scaled_cb_hi = FastScaleHue8Neon(
            cb0_hi, target_hi, reciprocal_hi0, reciprocal_hi1);
        const int16x8_t min_chroma = vdupq_n_s16(-128);
        const int16x8_t max_chroma = vdupq_n_s16(127);
        scaled_cr_lo = vmaxq_s16(
            min_chroma, vminq_s16(max_chroma, scaled_cr_lo));
        scaled_cr_hi = vmaxq_s16(
            min_chroma, vminq_s16(max_chroma, scaled_cr_hi));
        scaled_cb_lo = vmaxq_s16(
            min_chroma, vminq_s16(max_chroma, scaled_cb_lo));
        scaled_cb_hi = vmaxq_s16(
            min_chroma, vminq_s16(max_chroma, scaled_cb_hi));
        cr1_lo = vbslq_s16(violation_lo, scaled_cr_lo, cr1_lo);
        cr1_hi = vbslq_s16(violation_hi, scaled_cr_hi, cr1_hi);
        cb1_lo = vbslq_s16(violation_lo, scaled_cb_lo, cb1_lo);
        cb1_hi = vbslq_s16(violation_hi, scaled_cb_hi, cb1_hi);
        cr_out_lo = vaddq_s16(cr1_lo, center);
        cr_out_hi = vaddq_s16(cr1_hi, center);
        cb_out_lo = vaddq_s16(cb1_lo, center);
        cb_out_hi = vaddq_s16(cb1_hi, center);
      }
    }
  }
  vst1q_u8(output_cr, vcombine_u8(vqmovun_s16(cr_out_lo),
                                   vqmovun_s16(cr_out_hi)));
  vst1q_u8(output_cb, vcombine_u8(vqmovun_s16(cb_out_lo),
                                   vqmovun_s16(cb_out_hi)));
}
#endif

#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
// Reconstruct sixteen half-grid chroma samples. Pairing [base,residual] with
// [base gain,keep] lets vpmaddwd evaluate both products and their sum in one
// instruction while preserving the scalar signed rounding exactly.
CYPERSTEREO_AVX2_TARGET inline __m128i FastReconstructChroma16Avx2(
    const __m128i value8, const __m128i base8, const __m128i keep8,
    const int base_q7) {
  const __m256i value16 = _mm256_cvtepu8_epi16(value8);
  const __m256i base16 = _mm256_cvtepu8_epi16(base8);
  const __m256i keep16 = _mm256_cvtepu8_epi16(keep8);
  const __m256i base_centered =
      _mm256_sub_epi16(base16, _mm256_set1_epi16(128));
  const __m256i residual = _mm256_sub_epi16(value16, base16);
  const __m256i base_gain =
      _mm256_set1_epi16(static_cast<short>(base_q7));
  const __m256i coefficient_lo =
      _mm256_unpacklo_epi16(base_gain, keep16);
  const __m256i coefficient_hi =
      _mm256_unpackhi_epi16(base_gain, keep16);
  __m256i delta_lo = _mm256_madd_epi16(
      _mm256_unpacklo_epi16(base_centered, residual), coefficient_lo);
  __m256i delta_hi = _mm256_madd_epi16(
      _mm256_unpackhi_epi16(base_centered, residual), coefficient_hi);
  const __m256i round64 = _mm256_set1_epi32(64);
  delta_lo = _mm256_srai_epi32(
      _mm256_add_epi32(
          delta_lo,
          _mm256_add_epi32(round64, _mm256_srai_epi32(delta_lo, 31))),
      7);
  delta_hi = _mm256_srai_epi32(
      _mm256_add_epi32(
          delta_hi,
          _mm256_add_epi32(round64, _mm256_srai_epi32(delta_hi, 31))),
      7);
  const __m256i result16 = _mm256_add_epi16(
      _mm256_packs_epi32(delta_lo, delta_hi),
      _mm256_set1_epi16(128));
  return _mm_packus_epi16(_mm256_castsi256_si128(result16),
                          _mm256_extracti128_si256(result16, 1));
}

CYPERSTEREO_AVX2_TARGET inline __m256i FastHueViolations8Avx2(
    const __m128i cr0, const __m128i cb0, const __m128i cr1,
    const __m128i cb1, const __m128i enabled,
    const __m128i low_mag) {
  const __m256i lhs = _mm256_inserti128_si256(
      _mm256_castsi128_si256(_mm_unpacklo_epi16(cr0, cb0)),
      _mm_unpackhi_epi16(cr0, cb0), 1);
  const __m256i dot_rhs = _mm256_inserti128_si256(
      _mm256_castsi128_si256(_mm_unpacklo_epi16(cr1, cb1)),
      _mm_unpackhi_epi16(cr1, cb1), 1);
  const __m128i neg_cr1 = _mm_sub_epi16(_mm_setzero_si128(), cr1);
  const __m256i cross_rhs = _mm256_inserti128_si256(
      _mm256_castsi128_si256(_mm_unpacklo_epi16(cb1, neg_cr1)),
      _mm_unpackhi_epi16(cb1, neg_cr1), 1);
  const __m256i dot = _mm256_madd_epi16(lhs, dot_rhs);
  const __m256i cross =
      _mm256_abs_epi32(_mm256_madd_epi16(lhs, cross_rhs));
  const __m256i direction_bad = _mm256_or_si256(
      _mm256_cmpgt_epi32(_mm256_set1_epi32(1), dot),
      _mm256_cmpgt_epi32(
          _mm256_slli_epi32(cross, 8),
          _mm256_mullo_epi32(dot, _mm256_set1_epi32(9))));
  return _mm256_and_si256(
      _mm256_or_si256(direction_bad,
                      _mm256_cvtepi16_epi32(low_mag)),
      _mm256_cvtepi16_epi32(enabled));
}

CYPERSTEREO_AVX2_TARGET inline __m256i FastScaleHue8Avx2(
    const __m128i value16, const __m128i target16,
    const __m128i divisor16, const int32_t *reciprocal_q15) {
  const __m256i value32 = _mm256_cvtepi16_epi32(value16);
  const __m256i target32 = _mm256_cvtepu16_epi32(target16);
  const __m256i divisor32 = _mm256_cvtepu16_epi32(divisor16);
  const __m256i reciprocal =
      _mm256_i32gather_epi32(reciprocal_q15, divisor32, 4);
  const __m256i product =
      _mm256_mullo_epi32(_mm256_abs_epi32(value32), target32);
  const __m256i scaled = _mm256_srli_epi32(
      _mm256_add_epi32(
          _mm256_mullo_epi32(product, reciprocal),
          _mm256_set1_epi32(16384)),
      15);
  return _mm256_sign_epi32(scaled, value32);
}

// Apply the sparse hue-protection correction to a whole blend block. The
// blend vectors remain live, dot/cross use vpmaddwd, and corrected lanes are
// selected before the block's single final store.
CYPERSTEREO_AVX2_TARGET inline void FastApplyHueGuard16Avx2(
    const __m256i src_cr_u16, const __m256i src_cb_u16,
    const __m256i active16, const __m256i min_keep16,
    const int32_t *reciprocal_q15, __m256i &dst_cr_u16,
    __m256i &dst_cb_u16) {
  const __m256i center16 = _mm256_set1_epi16(128);
  const __m256i cr0 = _mm256_sub_epi16(src_cr_u16, center16);
  const __m256i cb0 = _mm256_sub_epi16(src_cb_u16, center16);
  __m256i cr1 = _mm256_sub_epi16(dst_cr_u16, center16);
  __m256i cb1 = _mm256_sub_epi16(dst_cb_u16, center16);
  const __m256i mag0 = _mm256_add_epi16(_mm256_abs_epi16(cr0),
                                         _mm256_abs_epi16(cb0));
  const __m256i mag1 = _mm256_add_epi16(_mm256_abs_epi16(cr1),
                                         _mm256_abs_epi16(cb1));
  const __m256i min_mag = _mm256_srli_epi16(
      _mm256_add_epi16(_mm256_mullo_epi16(mag0, min_keep16),
                       _mm256_set1_epi16(128)),
      8);
  const __m256i enabled = _mm256_and_si256(
      _mm256_cmpgt_epi16(active16, _mm256_setzero_si256()),
      _mm256_cmpgt_epi16(mag0, _mm256_set1_epi16(7)));
  const __m256i low_mag = _mm256_cmpgt_epi16(min_mag, mag1);
  const __m256i violations_lo = FastHueViolations8Avx2(
      _mm256_castsi256_si128(cr0), _mm256_castsi256_si128(cb0),
      _mm256_castsi256_si128(cr1), _mm256_castsi256_si128(cb1),
      _mm256_castsi256_si128(enabled),
      _mm256_castsi256_si128(low_mag));
  const __m256i violations_hi = FastHueViolations8Avx2(
      _mm256_extracti128_si256(cr0, 1),
      _mm256_extracti128_si256(cb0, 1),
      _mm256_extracti128_si256(cr1, 1),
      _mm256_extracti128_si256(cb1, 1),
      _mm256_extracti128_si256(enabled, 1),
      _mm256_extracti128_si256(low_mag, 1));
  const __m256i violation16 = _mm256_permute4x64_epi64(
      _mm256_packs_epi32(violations_lo, violations_hi), 0xd8);
  if (_mm256_testz_si256(violation16, violation16)) return;

  const __m256i target_mag = _mm256_max_epi16(mag1, min_mag);
  const __m256i cr_scaled_lo = FastScaleHue8Avx2(
      _mm256_castsi256_si128(cr0),
      _mm256_castsi256_si128(target_mag),
      _mm256_castsi256_si128(mag0), reciprocal_q15);
  const __m256i cr_scaled_hi = FastScaleHue8Avx2(
      _mm256_extracti128_si256(cr0, 1),
      _mm256_extracti128_si256(target_mag, 1),
      _mm256_extracti128_si256(mag0, 1), reciprocal_q15);
  const __m256i cb_scaled_lo = FastScaleHue8Avx2(
      _mm256_castsi256_si128(cb0),
      _mm256_castsi256_si128(target_mag),
      _mm256_castsi256_si128(mag0), reciprocal_q15);
  const __m256i cb_scaled_hi = FastScaleHue8Avx2(
      _mm256_extracti128_si256(cb0, 1),
      _mm256_extracti128_si256(target_mag, 1),
      _mm256_extracti128_si256(mag0, 1), reciprocal_q15);
  __m256i cr_scaled = _mm256_permute4x64_epi64(
      _mm256_packs_epi32(cr_scaled_lo, cr_scaled_hi), 0xd8);
  __m256i cb_scaled = _mm256_permute4x64_epi64(
      _mm256_packs_epi32(cb_scaled_lo, cb_scaled_hi), 0xd8);
  const __m256i lo16 = _mm256_set1_epi16(-128);
  const __m256i hi16 = _mm256_set1_epi16(127);
  cr_scaled = _mm256_min_epi16(hi16, _mm256_max_epi16(lo16, cr_scaled));
  cb_scaled = _mm256_min_epi16(hi16, _mm256_max_epi16(lo16, cb_scaled));
  cr1 = _mm256_blendv_epi8(cr1, cr_scaled, violation16);
  cb1 = _mm256_blendv_epi8(cb1, cb_scaled, violation16);
  dst_cr_u16 = _mm256_add_epi16(cr1, center16);
  dst_cb_u16 = _mm256_add_epi16(cb1, center16);
}
#endif

inline void SuppressFalseColorImpl(const cv::Mat *raw_wb, cv::Mat &color,
                                   double sensor_gain = 1.0,
                                   bool linear_input = false,
                                   bool apply_tone = false) {
  static thread_local cv::Mat y8, y_bl, colsum;              // u8 full res
  static thread_local cv::Mat yq8, sq_q, mean_i, mean_ii, a_q, b_q;
  static thread_local cv::Mat a_q16, b_q16;                  // u16 coeffs
  static thread_local cv::Mat hf_q8, tex_q;                  // u8 quarter
  static thread_local cv::Mat luma_hf_ring, luma_hs_ring;    // AVX2 3-row rings
  static thread_local cv::Mat residual_keep_q8;              // u8 Q7 gate
  static thread_local cv::Mat chroma_blend_q8;               // u8 HDR-like mix
  static thread_local cv::Mat hue_protect_q8;                // u8 sparse guard
  static thread_local cv::Mat hue_min_keep_q8;               // u8 magnitude floor
  static thread_local cv::Mat bgr_h, ycc_h, ch_h[3];         // half res
  static thread_local cv::Mat cr_h, cb_h;                    // u8 half
  // Keep median3 out-of-place. OpenCV must clone the whole source when its
  // input/output alias; these persistent ping-pong planes remove that hidden
  // copy while executing the exact same optimized median sorting network.
  static thread_local cv::Mat cr_med_h, cb_med_h;            // u8 half
  static thread_local cv::Mat base_cr_q, base_cb_q;          // u8 quarter
  static thread_local cv::Mat g5_tmp_full, g5_tmp_half;      // u16 scratch
  static thread_local cv::Mat g5_row_buf;                    // u8 one row

  if (raw_wb) color.create(raw_wb->rows, raw_wb->cols, CV_8UC3);
  const cv::Size full(color.cols, color.rows);
  CV_Assert(full.width >= 4 && full.height >= 4);
  // The fused 4:2:0/quarter-grid kernels process 4x4 tiles. Camera modes are
  // naturally aligned, but SDK callers and offline tests may supply arbitrary
  // geometry. Replicate-pad at most three right/bottom pixels, process the
  // aligned frame, then crop; this keeps every tail access safe without
  // changing the visible image dimensions.
  if ((full.width & 3) != 0 || (full.height & 3) != 0) {
    const int padded_width = (full.width + 3) & ~3;
    const int padded_height = (full.height + 3) & ~3;
    const int right = padded_width - full.width;
    const int bottom = padded_height - full.height;
    cv::Mat padded_color;
    if (raw_wb) {
      cv::Mat padded_raw;
      cv::copyMakeBorder(*raw_wb, padded_raw, 0, bottom, 0, right,
                         cv::BORDER_REPLICATE);
      SuppressFalseColorImpl(&padded_raw, padded_color, sensor_gain,
                             linear_input, apply_tone);
    } else {
      cv::copyMakeBorder(color, padded_color, 0, bottom, 0, right,
                         cv::BORDER_REPLICATE);
      SuppressFalseColorImpl(nullptr, padded_color, sensor_gain, linear_input,
                             apply_tone);
    }
    padded_color(cv::Rect(0, 0, full.width, full.height)).copyTo(color);
    return;
  }
  const cv::Size half(color.cols / 2, color.rows / 2);
  const cv::Size quarter(color.cols / 4, color.rows / 4);
  int gain_t_q8 = 0;
  if (linear_input) {
    if (!(sensor_gain >= 1.0)) sensor_gain = 1.0;  // also rejects NaN
    sensor_gain = (std::min)(sensor_gain, 8.0);
    gain_t_q8 = static_cast<int>(
        (sensor_gain - 1.0) * (256.0 / 7.0) + 0.5);
  }
  using DetailClock = std::chrono::steady_clock;
  const bool detail_profile = FastPostDetailProfileEnabled();
  const auto detail_now = [&] {
    return detail_profile ? DetailClock::now() : DetailClock::time_point{};
  };
  const auto detail_t0 = detail_now();

  // Luma plane + its gauss5 low-pass (shared by the texture band-pass
  // below and by the guided-filter stats). With the fused front-end the
  // same pass also emits the raw half-res chroma (into ch_h[1]/ch_h[2]);
  // its gauss5 NR happens further down, exactly where the split-path
  // chroma gets blurred.
  const bool fused_front = UseFusedFront();
  const bool have_chroma = raw_wb != nullptr || fused_front;
#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
  if (raw_wb) {
    FusedDemosaicFrontAvx2(*raw_wb, y8, ch_h[1], ch_h[2]);
  } else
#elif defined(CYPERSTEREO_HAVE_NEON)
  if (raw_wb) {
    FusedDemosaicFrontNeon(*raw_wb, y8, ch_h[1], ch_h[2]);
  } else
#endif
  if (fused_front) {
    FusedFrontYCbCr420(color, y8, ch_h[1], ch_h[2]);
  } else {
    cv::cvtColor(color, y8, cv::COLOR_BGR2GRAY);
  }
  const auto detail_t1 = detail_now();

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
  bool quarter_texture_fused = false;
#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
  // y_bl is still retained because OpenCV's x86 Gaussian5 is faster than a
  // hand-written row kernel. Consume it only once for all quarter-grid luma
  // products; the opt-out supports bit-exact same-binary performance A/B.
  if (!luma_fused && quarter.width >= 2 && quarter.height >= 2 &&
      CpuHasAvx2Output() && UseFusedQuarterTextureAvx2()) {
    FusedQuarterTextureAvx2(y8, y_bl, tex_q, yq8, sq_q, luma_hf_ring,
                            luma_hs_ring);
    quarter_texture_fused = true;
  }
#endif
  // Gain-adaptive variance threshold. The quarter-grid fast implementation
  // is slightly stronger than the HDR reference at the same numerical eps,
  // so cap the high-gain value at 20 to retain the same strong-edge contrast.
  const float luma_nr_eps =
      linear_input ? static_cast<float>(8 + ((20 - 8) * gain_t_q8 + 128) / 256)
                   : 20.0f;
#if defined(CYPERSTEREO_HAVE_NEON)
  if (!luma_fused && UseNeonGuided()) {
    GuidedStatsIntNeon(y_bl, a_q16, b_q16, mean_i);
  } else if (!luma_fused)
#else
  if (!luma_fused)
#endif
  {
    if (!quarter_texture_fused)
      cv::resize(y_bl, yq8, quarter, 0, 0, cv::INTER_NEAREST);
    const cv::Size w3(3, 3);
    cv::boxFilter(yq8, mean_i, CV_32F, w3, cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
    if (!quarter_texture_fused)
      cv::multiply(yq8, yq8, sq_q, 1.0, CV_16U);
    cv::boxFilter(sq_q, mean_ii, CV_32F, w3, cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
    // a,b are quantized to fixed point HERE (a Q12, b Q4) and the smoothing
    // box3 runs on u16 planes: vs f32 boxes + convertTo this is 1.08 ->
    // 0.70 ms and the rounding lands within 1 LSB of the f32 path (the box
    // average smooths the quantization noise).
    bool guided_coeff_done = false;
#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
    if (CpuHasAvx2Output()) {
      FastGuidedRawCoeffAvx2(mean_i, mean_ii, luma_nr_eps, a_q, b_q);
      guided_coeff_done = true;
    }
#endif
    if (!guided_coeff_done) {
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
          const float a = v / (v + luma_nr_eps);
          pa[x] = static_cast<ushort>(a * 4096.0f + 0.5f);
          pb[x] = static_cast<ushort>((m - a * m) * 16.0f + 0.5f);
        }
      }
    }
    cv::boxFilter(a_q, a_q16, -1, w3, cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
    cv::boxFilter(b_q, b_q16, -1, w3, cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
  }
  // The quality/reference HDR path blends guided luma NR from 25% at 1x to
  // 75% at the sensor's 8x ceiling.  The legacy fused path used the full
  // guided result at every gain.  Blend the quarter-grid coefficients here,
  // before the fused output, so the fast-balanced path follows the same
  // gain-adaptive intent without adding a full-resolution pass.
  if (linear_input) {
    const int strength_q8 =
        64 + ((192 - 64) * gain_t_q8 + 128) / 256;
    bool guided_strength_done = false;
#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
    if (CpuHasAvx2Output()) {
      FastGuidedStrengthAvx2(a_q16, b_q16, strength_q8);
      guided_strength_done = true;
    }
#endif
    if (!guided_strength_done) {
      for (int y = 0; y < quarter.height; ++y) {
        ushort *pa = a_q16.ptr<ushort>(y);
        ushort *pb = b_q16.ptr<ushort>(y);
        for (int x = 0; x < quarter.width; ++x) {
          const int one_minus_a = 4096 - pa[x];
          pa[x] = static_cast<ushort>(
              4096 - ((one_minus_a * strength_q8 + 128) >> 8));
          pb[x] = static_cast<ushort>(
              (static_cast<int>(pb[x]) * strength_q8 + 128) >> 8);
        }
      }
    }
  }
  const auto detail_t2 = detail_now();

  // a in [0,1] -> Q12; b in [0,255] -> Q4. y' = (a*y + (b<<8) + 2048) >> 12.
  // The coefficient planes stay on the quarter grid; the output loop
  // expands them NN (bilinear upsampling cost 0.65 ms/frame and measurably
  // changed < 0.11% of pixels; residual 4px-seam energy on flat walls is
  // 0.43 vs 0.30 DN -- invisible at unity zoom).

  // Materialize the half-resolution source chroma before building the gate.
  // Besides feeding the NR below, a coherent 2x2 half-grid colour vector is
  // the evidence that a dark area contains real colour rather than random
  // red/blue sensor noise.  The old gate looked only at luma, so saturated
  // dark blue/red patches were forced all the way to grey before gamma.
  if (!have_chroma) {
    cv::resize(color, bgr_h, half, 0, 0, cv::INTER_AREA);
    cv::cvtColor(bgr_h, ycc_h, cv::COLOR_BGR2YCrCb);
    cv::split(ycc_h, ch_h);
  }
  const auto detail_t2a = detail_now();

  // Ordinary half-grid chroma NR candidate plus a much wider stable base
  // colour. The latter spans roughly 28x28 full-resolution pixels, wide
  // enough to average out isolated sensor speckles and demosaic colour beats
  // without erasing a genuine coloured surface.
  Gauss5U8(ch_h[1], cr_h, g5_tmp_half);
  Gauss5U8(ch_h[2], cb_h, g5_tmp_half);
  cv::resize(cr_h, base_cr_q, quarter, 0, 0, cv::INTER_AREA);
  cv::resize(cb_h, base_cb_q, quarter, 0, 0, cv::INTER_AREA);
  const cv::Size base_window(7, 7);
  cv::boxFilter(base_cr_q, base_cr_q, -1, base_window,
                cv::Point(-1, -1), true, cv::BORDER_REFLECT);
  cv::boxFilter(base_cb_q, base_cb_q, -1, base_window,
                cv::Point(-1, -1), true, cv::BORDER_REFLECT);
  const auto detail_t2b = detail_now();

  // Gate maps at quarter res. Texture suppression remains available for
  // dense near-neutral aliasing, while coherent real chroma establishes a
  // minimum keep value.  This mirrors the HDR quality path: strong colour is
  // only lightly filtered; aggressive attenuation is reserved for locally
  // neutral noise/false colour.
  // The band-pass energy |y8 - y_bl| is STILL evaluated at full res (see
  // note d in the header comment) -- AbsDiffPool4 only fuses the pooling.
  // (On the ARM fused-luma path hf_q8 was already produced above.)
  if (!quarter_texture_fused) {
    if (!luma_fused) AbsDiffPool4(y8, y_bl, hf_q8, colsum);
    cv::GaussianBlur(hf_q8, tex_q, cv::Size(3, 3), 0);
  }
  const auto detail_t2c = detail_now();
  residual_keep_q8.create(quarter, CV_8U);
  chroma_blend_q8.create(quarter, CV_8U);
  hue_protect_q8.create(quarter, CV_8U);
  hue_min_keep_q8.create(quarter, CV_8U);
  // The shadow gate (0 at/below Y=16, 1 at/above Y=64) was tuned on the
  // LINEAR pipeline; luma now arrives tone-mapped by the WB LUT, so map
  // the thresholds through the same curve to keep the gate firing on the
  // same scene content.
  const float kShadowLo = static_cast<float>(
      linear_input ? 16.0 : ToneEncode(16.0));
  const float kShadowHi = static_cast<float>(
      linear_input ? 64.0 : ToneEncode(64.0));
  const float kShadowScale = 255.0f / (kShadowHi - kShadowLo);
  // The saturation gain (HSC) is folded into the keep gate, which turns it
  // from a Q8 attenuator (255 = 1.0x) into a Q7 chroma gain (128 = 1.0x,
  // 255 = ~2.0x); the multiply loop below shifts by 7 accordingly. Shadow /
  // texture desaturation still wins where the gate is low.
  static const int kSatQ128 =
      static_cast<int>(IspSaturation() * 128.0 + 0.5);
  // These planes are thread_local scratch owned by the submitting ISP
  // thread. Capture explicit addresses before dispatch: referring to the TLS
  // names from a worker would otherwise resolve that worker's empty scratch.
  const cv::Mat *gate_source_cr = &ch_h[1];
  const cv::Mat *gate_source_cb = &ch_h[2];
  const cv::Mat *gate_texture = &tex_q;
  const cv::Mat *gate_mean_luma = &mean_i;
  const cv::Mat *gate_base_cr = &base_cr_q;
  const cv::Mat *gate_base_cb = &base_cb_q;
  cv::Mat *gate_residual = &residual_keep_q8;
  cv::Mat *gate_blend = &chroma_blend_q8;
  cv::Mat *gate_hue = &hue_protect_q8;
  cv::Mat *gate_min_keep = &hue_min_keep_q8;
  FastParallelForRows(quarter.height, "gate", [&](int y) {
    const int hy = y << 1;
    const uchar *src_cr0 = gate_source_cr->ptr<uchar>(hy);
    const uchar *src_cr1 = gate_source_cr->ptr<uchar>(hy + 1);
    const uchar *src_cb0 = gate_source_cb->ptr<uchar>(hy);
    const uchar *src_cb1 = gate_source_cb->ptr<uchar>(hy + 1);
    const uchar *pt = gate_texture->ptr<uchar>(y);
    const float *pl =
        gate_mean_luma->ptr<float>(y);  // box3 quarter luma ~ 12px LP
    const uchar *pbase_cr = gate_base_cr->ptr<uchar>(y);
    const uchar *pbase_cb = gate_base_cb->ptr<uchar>(y);
    uchar *pr = gate_residual->ptr<uchar>(y);
    uchar *pblend = gate_blend->ptr<uchar>(y);
    uchar *phue = gate_hue->ptr<uchar>(y);
    uchar *pmin = gate_min_keep->ptr<uchar>(y);
    int x = 0;
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
    for (; x + 8 <= quarter.width; x += 8) {
      const int hx = x << 1;
      FastChromaGate8Neon(
          src_cr0 + hx, src_cr1 + hx, src_cb0 + hx, src_cb1 + hx,
          pt + x, pl + x, pbase_cr + x, pbase_cb + x, kShadowLo,
          kShadowScale, kSatQ128, pr + x, pblend + x, phue + x, pmin + x);
    }
#elif defined(__AVX2__) || defined(_M_AVX2)
    // Eight quarter cells describe sixteen half-grid samples. Vectorize the
    // exact scalar gate equations; only the final 8-byte maps are stored.
    // All integer divisions here have non-negative operands and fixed
    // divisors, so div255/shift forms preserve C++ truncation exactly.
    const __m256i zero32 = _mm256_setzero_si256();
    const __m256i all32 = _mm256_cmpeq_epi32(zero32, zero32);
    const __m256i c255 = _mm256_set1_epi32(255);
    const auto clamp255 = [&](const __m256i value) {
      return _mm256_min_epi32(c255, _mm256_max_epi32(zero32, value));
    };
    const auto div255 = [&](const __m256i value) {
      return _mm256_srli_epi32(
          _mm256_add_epi32(_mm256_add_epi32(value,
                                            _mm256_set1_epi32(1)),
                           _mm256_srli_epi32(value, 8)),
          8);
    };
    const auto load8u32 = [&](const uchar *values) {
      return _mm256_cvtepu8_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i *>(values)));
    };
    const auto load16centered = [&](const uchar *values) {
      return _mm256_sub_epi16(
          _mm256_cvtepu8_epi16(
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(values))),
          _mm256_set1_epi16(128));
    };
    const auto store8u8 = [&](uchar *destination, const __m256i value) {
      const __m128i packed16 = _mm_packs_epi32(
          _mm256_castsi256_si128(value),
          _mm256_extracti128_si256(value, 1));
      const __m128i packed8 =
          _mm_packus_epi16(packed16, _mm_setzero_si128());
      _mm_storel_epi64(reinterpret_cast<__m128i *>(destination), packed8);
    };
    const __m256i pair_ones = _mm256_set1_epi16(1);
    for (; x + 8 <= quarter.width; x += 8) {
      const int hx = x << 1;
      const __m256i cr00 = load16centered(src_cr0 + hx);
      const __m256i cr01 = load16centered(src_cr1 + hx);
      const __m256i cb00 = load16centered(src_cb0 + hx);
      const __m256i cb01 = load16centered(src_cb1 + hx);
      const __m256i sum_cr = _mm256_add_epi32(
          _mm256_madd_epi16(cr00, pair_ones),
          _mm256_madd_epi16(cr01, pair_ones));
      const __m256i sum_cb = _mm256_add_epi32(
          _mm256_madd_epi16(cb00, pair_ones),
          _mm256_madd_epi16(cb01, pair_ones));
      const __m256i vector_sum = _mm256_add_epi32(
          _mm256_abs_epi32(sum_cr), _mm256_abs_epi32(sum_cb));
      __m256i abs_sum = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_abs_epi16(cr00), pair_ones),
          _mm256_madd_epi16(_mm256_abs_epi16(cr01), pair_ones));
      abs_sum = _mm256_add_epi32(
          abs_sum,
          _mm256_add_epi32(
              _mm256_madd_epi16(_mm256_abs_epi16(cb00), pair_ones),
              _mm256_madd_epi16(_mm256_abs_epi16(cb01), pair_ones)));

      const __m256i texture = clamp255(_mm256_sub_epi32(
          _mm256_mullo_epi32(load8u32(pt + x), _mm256_set1_epi32(30)),
          _mm256_set1_epi32(195)));
      const __m256 shade_f = _mm256_mul_ps(
          _mm256_sub_ps(_mm256_loadu_ps(pl + x),
                        _mm256_set1_ps(kShadowLo)),
          _mm256_set1_ps(kShadowScale));
      const __m256i shade = clamp255(_mm256_cvttps_epi32(shade_f));

      const __m256i base_cr = _mm256_sub_epi32(
          load8u32(pbase_cr + x), _mm256_set1_epi32(128));
      const __m256i base_cb = _mm256_sub_epi32(
          load8u32(pbase_cb + x), _mm256_set1_epi32(128));
      const __m256i base_chroma = _mm256_add_epi32(
          _mm256_abs_epi32(base_cr), _mm256_abs_epi32(base_cb));
      // floor(d*255/18), d in [0,18]. The fixed-point reciprocal is exact
      // for every possible d and avoids a random gather in the hot loop.
      const __m256i protect_delta = _mm256_min_epi32(
          _mm256_set1_epi32(18),
          _mm256_max_epi32(
              zero32,
              _mm256_sub_epi32(base_chroma, _mm256_set1_epi32(10))));
      const __m256i protect = _mm256_srli_epi32(
          _mm256_mullo_epi32(protect_delta, _mm256_set1_epi32(928455)),
          16);

      __m256i residual_keep = div255(_mm256_mullo_epi32(
          shade, _mm256_sub_epi32(c255, texture)));
      const __m256i local_chroma =
          _mm256_srli_epi32(_mm256_add_epi32(vector_sum,
                                             _mm256_set1_epi32(2)),
                            2);
      __m256i local_protect = clamp255(_mm256_srli_epi32(
          _mm256_mullo_epi32(
              _mm256_max_epi32(
                  zero32,
                  _mm256_sub_epi32(local_chroma, _mm256_set1_epi32(4))),
              c255),
          4));
      const __m256i adjusted_local = div255(
          _mm256_mullo_epi32(local_protect, protect));
      const __m256i textured =
          _mm256_cmpgt_epi32(texture, zero32);
      local_protect = _mm256_blendv_epi8(local_protect, adjusted_local,
                                         textured);
      const __m256i coherent = _mm256_xor_si256(
          _mm256_cmpgt_epi32(
              _mm256_mullo_epi32(abs_sum, _mm256_set1_epi32(3)),
              _mm256_slli_epi32(vector_sum, 2)),
          all32);
      local_protect = _mm256_and_si256(local_protect, coherent);
      residual_keep = _mm256_max_epi32(residual_keep, local_protect);
      const __m256i color_protect =
          _mm256_max_epi32(protect, local_protect);

      const __m256i residual_q7 = clamp255(div255(_mm256_mullo_epi32(
          residual_keep, _mm256_set1_epi32(kSatQ128))));
      const __m256i base_blend = _mm256_sub_epi32(
          _mm256_set1_epi32(144),
          div255(_mm256_mullo_epi32(color_protect,
                                    _mm256_set1_epi32(104))));
      const __m256i texture_blend = div255(_mm256_mullo_epi32(
          texture, _mm256_sub_epi32(c255, protect)));
      const __m256i blend =
          _mm256_max_epi32(base_blend, texture_blend);
      store8u8(pr + x, residual_q7);
      store8u8(pblend + x, blend);
      store8u8(phue + x, color_protect);
      store8u8(pmin + x, div255(_mm256_mullo_epi32(
                                protect, _mm256_set1_epi32(216))));
    }
#endif
    for (; x < quarter.width; ++x) {
      int strength = pt[x] * 30 - 195;
      strength = strength < 0 ? 0 : (strength > 255 ? 255 : strength);
      int shade = static_cast<int>((pl[x] - kShadowLo) * kShadowScale);
      shade = shade < 0 ? 0 : (shade > 255 ? 255 : shade);
      constexpr int kColorProtectLo = 10;
      constexpr int kColorProtectHi = 28;
      const int base_chroma =
          std::abs(static_cast<int>(pbase_cr[x]) - 128) +
          std::abs(static_cast<int>(pbase_cb[x]) - 128);
      int protect = (base_chroma - kColorProtectLo) * 255 /
                    (kColorProtectHi - kColorProtectLo);
      protect = protect < 0 ? 0 : (protect > 255 ? 255 : protect);

      // A stable low-frequency base is not random noise, even when it is dark
      // and only weakly coloured. Keep it exactly as the HDR joint ChromaNR
      // does; use shadow/texture confidence only on the residual around it.
      int residual_keep = shade * (255 - strength) / 255;

      // At a genuine colour edge the wide base is diluted, while the local
      // 2x2 half-grid vector remains coherent. Protect that residual in flat
      // or ordinary edge regions. On dense high-frequency texture require a
      // coloured wide base too, otherwise the vector is likely CFA moire.
      const int hx = x << 1;
      const int crv[4] = {static_cast<int>(src_cr0[hx]) - 128,
                          static_cast<int>(src_cr0[hx + 1]) - 128,
                          static_cast<int>(src_cr1[hx]) - 128,
                          static_cast<int>(src_cr1[hx + 1]) - 128};
      const int cbv[4] = {static_cast<int>(src_cb0[hx]) - 128,
                          static_cast<int>(src_cb0[hx + 1]) - 128,
                          static_cast<int>(src_cb1[hx]) - 128,
                          static_cast<int>(src_cb1[hx + 1]) - 128};
      const int sum_cr = crv[0] + crv[1] + crv[2] + crv[3];
      const int sum_cb = cbv[0] + cbv[1] + cbv[2] + cbv[3];
      const int vector_sum = std::abs(sum_cr) + std::abs(sum_cb);
      const int abs_sum = std::abs(crv[0]) + std::abs(crv[1]) +
                          std::abs(crv[2]) + std::abs(crv[3]) +
                          std::abs(cbv[0]) + std::abs(cbv[1]) +
                          std::abs(cbv[2]) + std::abs(cbv[3]);
      int color_protect = protect;
      if (vector_sum * 4 >= abs_sum * 3) {
        constexpr int kLocalProtectLo = 4;
        constexpr int kLocalProtectHi = 20;
        const int local_chroma = (vector_sum + 2) >> 2;
        int local_protect =
            (local_chroma - kLocalProtectLo) * 255 /
            (kLocalProtectHi - kLocalProtectLo);
        local_protect = local_protect < 0
                            ? 0
                            : (local_protect > 255 ? 255 : local_protect);
        if (strength > 0)
          local_protect = local_protect * protect / 255;
        residual_keep = (std::max)(residual_keep, local_protect);
        color_protect = (std::max)(color_protect, local_protect);
      }
      const int residual_q7 = residual_keep * kSatQ128 / 255;
      pr[x] = static_cast<uchar>(residual_q7 > 255 ? 255 : residual_q7);
      // HDR ChromaNR blends 56.25% around neutral and only 15.625% on
      // stable saturated colour. Dense near-neutral texture may use the full
      // false-colour correction, while colour confidence reduces the blend.
      int blend_q8 = 144 + (40 - 144) * color_protect / 255;
      // Fade texture suppression continuously as stable colour confidence
      // rises.  A binary protect==0 gate caused a large blend discontinuity
      // around weakly coloured high-frequency detail.
      const int texture_blend = strength * (255 - protect) / 255;
      blend_q8 = (std::max)(blend_q8, texture_blend);
      pblend[x] = static_cast<uchar>(blend_q8);
      phue[x] = static_cast<uchar>(color_protect);
      pmin[x] = static_cast<uchar>(protect * 216 / 255);
    }
  });
  const auto detail_t3 = detail_now();

  // Reconstruct the filtered chroma candidate on the half grid, then remove
  // isolated speckle with median3. The source half-grid chroma remains the
  // dominant term on real colour and edges in the blend below.
  const int base_q7 = (std::min)(kSatQ128, 255);
#if CV_SIMD128
  const cv::v_int32x4 round63 = cv::v_setall_s32(63);
  const cv::v_int32x4 round64 = cv::v_setall_s32(64);
  const cv::v_int32x4 base_gain = cv::v_setall_s32(base_q7);
  const cv::v_int16x8 offset128 = cv::v_setall_s16(128);
  const auto reconstruct8 = [&](const cv::v_uint16x8 &value_u,
                                const cv::v_uint16x8 &base_u,
                                const cv::v_uint16x8 &keep_u) {
    const cv::v_int16x8 base_s =
        cv::v_reinterpret_as_s16(base_u) - offset128;
    const cv::v_int16x8 residual_s =
        cv::v_reinterpret_as_s16(value_u) -
        cv::v_reinterpret_as_s16(base_u);
    cv::v_int32x4 base0, base1, residual0, residual1;
    cv::v_uint32x4 keep0_u, keep1_u;
    cv::v_expand(base_s, base0, base1);
    cv::v_expand(residual_s, residual0, residual1);
    cv::v_expand(keep_u, keep0_u, keep1_u);
    cv::v_int32x4 delta0 =
        base0 * base_gain +
        residual0 * cv::v_reinterpret_as_s32(keep0_u);
    cv::v_int32x4 delta1 =
        base1 * base_gain +
        residual1 * cv::v_reinterpret_as_s32(keep1_u);
    const cv::v_int32x4 neg0 = delta0 < cv::v_setzero_s32();
    const cv::v_int32x4 neg1 = delta1 < cv::v_setzero_s32();
    delta0 = (delta0 + ((neg0 & round63) | (~neg0 & round64))) >> 7;
    delta1 = (delta1 + ((neg1 & round63) | (~neg1 & round64))) >> 7;
    return cv::v_pack(delta0, delta1) + offset128;
  };
#endif
  // These planes are static thread_local scratch in the caller.  Capture
  // their resolved addresses before dispatch; referring to the Mat names in
  // a helper would instead select that helper thread's empty TLS instance.
  cv::Mat *const reconstruct_planes[2] = {&cr_h, &cb_h};
  const cv::Mat *const reconstruct_bases[2] = {&base_cr_q, &base_cb_q};
  const cv::Mat *const reconstruct_keep = &residual_keep_q8;
  FastParallelForRows(half.height / 2, "reconstruct", [&](int pair_row) {
    const int y = pair_row << 1;
    const int yq = (std::min)(y >> 1, quarter.height - 1);
    const uchar *pr = reconstruct_keep->ptr<uchar>(yq);
    for (int channel = 0; channel < 2; ++channel) {
      cv::Mat *plane = reconstruct_planes[channel];
      const uchar *pbase = reconstruct_bases[channel]->ptr<uchar>(yq);
      uchar *pv0 = plane->ptr<uchar>(y);
      uchar *pv1 = plane->ptr<uchar>(y + 1);
      int x = 0;
#if CV_SIMD128 || \
    (defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__))
      for (; x + 16 <= half.width; x += 16) {
        const int xq = x >> 1;
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
        FastReconstructCoefficients16Neon coefficient;
        FastPrepareReconstruct16Neon(pbase + xq, pr + xq, base_q7,
                                     coefficient);
        vst1q_u8(pv0 + x,
                 FastReconstructChroma16Neon(pv0 + x, coefficient));
        vst1q_u8(pv1 + x,
                 FastReconstructChroma16Neon(pv1 + x, coefficient));
#elif defined(__AVX2__) || defined(_M_AVX2)
        const __m128i base8 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(pbase + xq));
        const __m128i keep8 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(pr + xq));
        const __m128i base16 = _mm_unpacklo_epi8(base8, base8);
        const __m128i keep16 = _mm_unpacklo_epi8(keep8, keep8);
        const auto reconstruct_row = [&](uchar *pv) {
          const __m128i value = _mm_loadu_si128(
              reinterpret_cast<const __m128i *>(pv + x));
          _mm_storeu_si128(
              reinterpret_cast<__m128i *>(pv + x),
              FastReconstructChroma16Avx2(value, base16, keep16, base_q7));
        };
#else
        cv::v_uint8x16 base8 = cv::v_load_low(pbase + xq);
        cv::v_uint8x16 keep8 = cv::v_load_low(pr + xq);
        cv::v_uint8x16 base16, keep16, unused;
        cv::v_zip(base8, base8, base16, unused);
        cv::v_zip(keep8, keep8, keep16, unused);
        cv::v_uint16x8 base_lo, base_hi;
        cv::v_uint16x8 keep_lo, keep_hi;
        cv::v_expand(base16, base_lo, base_hi);
        cv::v_expand(keep16, keep_lo, keep_hi);
        const auto reconstruct_row = [&](uchar *pv) {
          cv::v_uint16x8 value_lo, value_hi;
          cv::v_expand(cv::v_load(pv + x), value_lo, value_hi);
          cv::v_store(pv + x,
                      cv::v_pack_u(reconstruct8(value_lo, base_lo, keep_lo),
                                   reconstruct8(value_hi, base_hi, keep_hi)));
        };
#endif
#if !defined(CYPERSTEREO_HAVE_NEON) || !defined(__aarch64__)
        reconstruct_row(pv0);
        reconstruct_row(pv1);
#endif
      }
#endif
      for (; x < half.width; ++x) {
        const int xq = (std::min)(x >> 1, quarter.width - 1);
        const int base = pbase[xq];
        for (int row = 0; row < 2; ++row) {
          uchar *pv = row == 0 ? pv0 : pv1;
          const int delta = (base - 128) * base_q7 +
                            (static_cast<int>(pv[x]) - base) * pr[xq];
          const int rounded = delta >= 0 ? (delta + 64) >> 7
                                         : -((-delta + 64) >> 7);
          pv[x] = cv::saturate_cast<uchar>(128 + rounded);
        }
      }
    }
  });
  if ((half.height & 1) != 0) {
    const int reconstruction_y = half.height - 1;
    const int yq = (std::min)(reconstruction_y >> 1,
                              quarter.height - 1);
    const uchar *pr = reconstruct_keep->ptr<uchar>(yq);
    for (int channel = 0; channel < 2; ++channel) {
      cv::Mat *plane = reconstruct_planes[channel];
      const uchar *pbase = reconstruct_bases[channel]->ptr<uchar>(yq);
      uchar *pv = plane->ptr<uchar>(reconstruction_y);
      for (int x = 0; x < half.width; ++x) {
        const int xq = (std::min)(x >> 1, quarter.width - 1);
        const int base = pbase[xq];
        const int delta = (base - 128) * base_q7 +
                          (static_cast<int>(pv[x]) - base) * pr[xq];
        const int rounded = delta >= 0 ? (delta + 64) >> 7
                                       : -((-delta + 64) >> 7);
        pv[x] = cv::saturate_cast<uchar>(128 + rounded);
      }
    }
  }
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
  FastMedian3U8Neon(cr_h, cr_med_h);
  FastMedian3U8Neon(cb_h, cb_med_h);
#else
  cv::medianBlur(cr_h, cr_med_h, 3);
  cv::medianBlur(cb_h, cb_med_h, 3);
#endif
  std::swap(cr_h, cr_med_h);
  std::swap(cb_h, cb_med_h);
  const auto detail_t4 = detail_now();

  // Match the HDR ChromaNR strength policy instead of replacing every pixel
  // by the half-resolution candidate. Blend and hue protection share this one
  // half-grid pass, avoiding a second read/write of both chroma planes while
  // preserving the exact scalar equations and rounding order.
  const auto round_shift8_signed = [](int value) {
    return value >= 0 ? (value + 128) >> 8 : -((-value + 128) >> 8);
  };
  const bool hue_guard = linear_input && FastBalancedHueGuardEnabled();
  constexpr int kSaturatedChroma = 8;
  constexpr int kMaxHueTanQ8 = 9;  // tan(about 2 degrees) * 256
  static const std::array<int32_t, 257> reciprocal_q15 = [] {
    std::array<int32_t, 257> v{};
    for (int d = 1; d <= 256; ++d)
      v[d] = (32768 + d / 2) / d;
    return v;
  }();
  const auto scale_signed = [&](int value, int magnitude, int divisor) {
    const int product = std::abs(value * magnitude);
    const int scaled =
        (product * reciprocal_q15[divisor] + 16384) >> 15;
    return value < 0 ? -scaled : scaled;
  };
  const auto apply_hue_guard = [&](int cr0, int cb0, int xq,
                                   const uchar *pbase_cr,
                                   const uchar *pbase_cb, int &cr, int &cb) {
    const int mag0 = std::abs(cr0) + std::abs(cb0);
    if (mag0 < kSaturatedChroma) return;
    const int dot = cr0 * cr + cb0 * cb;
    const int cross = std::abs(cr0 * cb - cb0 * cr);
    const int mag1 = std::abs(cr) + std::abs(cb);
    const int base_chroma =
        std::abs(static_cast<int>(pbase_cr[xq]) - 128) +
        std::abs(static_cast<int>(pbase_cb[xq]) - 128);
    constexpr int kProtectLo = 10;
    constexpr int kProtectHi = 28;
    int protect = (base_chroma - kProtectLo) * 255 /
                  (kProtectHi - kProtectLo);
    protect = protect < 0 ? 0 : (protect > 255 ? 255 : protect);
    const int min_keep_q8 = protect * 216 / 255;
    const int min_mag = (mag0 * min_keep_q8 + 128) >> 8;
    const bool hue_ok = dot > 0 && cross * 256 <= kMaxHueTanQ8 * dot;
    if (!hue_ok || mag1 < min_mag) {
      const int target_mag = (std::max)(mag1, min_mag);
      cr = scale_signed(cr0, target_mag, mag0);
      cb = scale_signed(cb0, target_mag, mag0);
    }
  };
  const cv::Mat *blend_source_cr = &ch_h[1];
  const cv::Mat *blend_source_cb = &ch_h[2];
  cv::Mat *blend_filtered_cr = &cr_h;
  cv::Mat *blend_filtered_cb = &cb_h;
  const cv::Mat *blend_map = &chroma_blend_q8;
  const cv::Mat *blend_hue_map = &hue_protect_q8;
  const cv::Mat *blend_min_map = &hue_min_keep_q8;
  const cv::Mat *blend_base_cr = &base_cr_q;
  const cv::Mat *blend_base_cb = &base_cb_q;
  FastParallelForRows(half.height, "blend", [&](int y) {
    const uchar *src_cr = blend_source_cr->ptr<uchar>(y);
    const uchar *src_cb = blend_source_cb->ptr<uchar>(y);
    uchar *dst_cr = blend_filtered_cr->ptr<uchar>(y);
    uchar *dst_cb = blend_filtered_cb->ptr<uchar>(y);
    const int yq = (std::min)(y >> 1, quarter.height - 1);
    const uchar *pblend = blend_map->ptr<uchar>(yq);
    const uchar *phue = blend_hue_map->ptr<uchar>(yq);
    const uchar *pmin = blend_min_map->ptr<uchar>(yq);
    const uchar *pbase_cr = blend_base_cr->ptr<uchar>(yq);
    const uchar *pbase_cb = blend_base_cb->ptr<uchar>(yq);
    int x = 0;
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
    for (; x + 16 <= half.width; x += 16) {
      const int xq = x >> 1;
      FastBlendHue16Neon(
          src_cr + x, src_cb + x, dst_cr + x, dst_cb + x, pblend + xq,
          phue + xq, pmin + xq, hue_guard, dst_cr + x, dst_cb + x);
    }
#elif CV_SIMD128
#if defined(__AVX2__) || defined(_M_AVX2)
    // Bit-exact signed round((filtered-source)*alpha/256) without expanding
    // to i32.  abs(delta)*alpha is at most 255*255=65025, and +128 is at
    // most 65153, so the unsigned u16 multiply/round cannot overflow.  The
    // sign restore reproduces the scalar +128/-128 symmetric rounding.
    const auto blend16_avx2 = [&](const __m256i source_u16,
                                  const __m256i filtered_u16,
                                  const __m256i alpha_u16) {
      const __m256i delta =
          _mm256_sub_epi16(filtered_u16, source_u16);
      const __m256i sign = _mm256_srai_epi16(delta, 15);
      const __m256i magnitude = _mm256_abs_epi16(delta);
      const __m256i product =
          _mm256_mullo_epi16(magnitude, alpha_u16);
      const __m256i rounded = _mm256_srli_epi16(
          _mm256_add_epi16(product, _mm256_set1_epi16(128)), 8);
      const __m256i signed_rounded = _mm256_sub_epi16(
          _mm256_xor_si256(rounded, sign), sign);
      return _mm256_add_epi16(source_u16, signed_rounded);
    };
#else
    const cv::v_int32x4 round127 = cv::v_setall_s32(127);
    const cv::v_int32x4 round128 = cv::v_setall_s32(128);
    const auto blend8 = [&](const cv::v_uint16x8 &source_u,
                            const cv::v_uint16x8 &filtered_u,
                            const cv::v_uint16x8 &alpha_u) {
      const cv::v_int16x8 source_s = cv::v_reinterpret_as_s16(source_u);
      const cv::v_int16x8 delta_s =
          cv::v_reinterpret_as_s16(filtered_u) - source_s;
      cv::v_int32x4 d0, d1;
      cv::v_uint32x4 a0_u, a1_u;
      cv::v_expand(delta_s, d0, d1);
      cv::v_expand(alpha_u, a0_u, a1_u);
      cv::v_int32x4 p0 = d0 * cv::v_reinterpret_as_s32(a0_u);
      cv::v_int32x4 p1 = d1 * cv::v_reinterpret_as_s32(a1_u);
      const cv::v_int32x4 neg0 = p0 < cv::v_setzero_s32();
      const cv::v_int32x4 neg1 = p1 < cv::v_setzero_s32();
      p0 = (p0 + ((neg0 & round127) | (~neg0 & round128))) >> 8;
      p1 = (p1 + ((neg1 & round127) | (~neg1 & round128))) >> 8;
      return source_s + cv::v_pack(p0, p1);
    };
#endif
    for (; x + 16 <= half.width; x += 16) {
      const int xq = x >> 1;
#if defined(__AVX2__) || defined(_M_AVX2)
      const __m128i alpha8 = _mm_loadl_epi64(
          reinterpret_cast<const __m128i *>(pblend + xq));
      const __m128i alpha16 = _mm_unpacklo_epi8(alpha8, alpha8);
      const __m256i alpha_u16 = _mm256_cvtepu8_epi16(alpha16);
      const __m256i src_cr_u16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
          reinterpret_cast<const __m128i *>(src_cr + x)));
      const __m256i src_cb_u16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
          reinterpret_cast<const __m128i *>(src_cb + x)));
      const __m256i dst_cr_u16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
          reinterpret_cast<const __m128i *>(dst_cr + x)));
      const __m256i dst_cb_u16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
          reinterpret_cast<const __m128i *>(dst_cb + x)));
      __m256i cr_out_u16 =
          blend16_avx2(src_cr_u16, dst_cr_u16, alpha_u16);
      __m256i cb_out_u16 =
          blend16_avx2(src_cb_u16, dst_cb_u16, alpha_u16);
#else
      cv::v_uint8x16 alpha8 = cv::v_load_low(pblend + xq);
      cv::v_uint8x16 alpha16, unused;
      cv::v_zip(alpha8, alpha8, alpha16, unused);
      cv::v_uint16x8 alpha_lo, alpha_hi;
      cv::v_uint16x8 src_cr_lo, src_cr_hi, src_cb_lo, src_cb_hi;
      cv::v_uint16x8 dst_cr_lo, dst_cr_hi, dst_cb_lo, dst_cb_hi;
      cv::v_expand(alpha16, alpha_lo, alpha_hi);
      cv::v_expand(cv::v_load(src_cr + x), src_cr_lo, src_cr_hi);
      cv::v_expand(cv::v_load(src_cb + x), src_cb_lo, src_cb_hi);
      cv::v_expand(cv::v_load(dst_cr + x), dst_cr_lo, dst_cr_hi);
      cv::v_expand(cv::v_load(dst_cb + x), dst_cb_lo, dst_cb_hi);
      const cv::v_uint8x16 cr_out = cv::v_pack_u(
          blend8(src_cr_lo, dst_cr_lo, alpha_lo),
          blend8(src_cr_hi, dst_cr_hi, alpha_hi));
      const cv::v_uint8x16 cb_out = cv::v_pack_u(
          blend8(src_cb_lo, dst_cb_lo, alpha_lo),
          blend8(src_cb_hi, dst_cb_hi, alpha_hi));
      cv::v_store(dst_cr + x, cr_out);
      cv::v_store(dst_cb + x, cb_out);
#endif

      unsigned guard_lanes = 0;
      if (hue_guard) {
#if defined(__AVX2__) || defined(_M_AVX2)
        const __m128i active_q = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(phue + xq));
        if (_mm_cvtsi128_si64(active_q) != 0) {
          const __m128i min_q = _mm_loadl_epi64(
              reinterpret_cast<const __m128i *>(pmin + xq));
          const __m128i active_bytes = _mm_unpacklo_epi8(active_q, active_q);
          const __m128i min_bytes = _mm_unpacklo_epi8(min_q, min_q);
          const __m256i active16 = _mm256_cvtepu8_epi16(active_bytes);
          const __m256i min_keep16 = _mm256_cvtepu8_epi16(min_bytes);
          FastApplyHueGuard16Avx2(
              src_cr_u16, src_cb_u16, active16, min_keep16,
              reciprocal_q15.data(), cr_out_u16, cb_out_u16);
        }
#else
        for (int lane = 0; lane < 16; ++lane)
          if (phue[(x + lane) >> 1] != 0) guard_lanes |= 1u << lane;
#endif
      }
#if defined(__AVX2__) || defined(_M_AVX2)
      const __m128i cr_out = _mm_packus_epi16(
          _mm256_castsi256_si128(cr_out_u16),
          _mm256_extracti128_si256(cr_out_u16, 1));
      const __m128i cb_out = _mm_packus_epi16(
          _mm256_castsi256_si128(cb_out_u16),
          _mm256_extracti128_si256(cb_out_u16, 1));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_cr + x), cr_out);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_cb + x), cb_out);
#endif
      if (guard_lanes != 0) {
        for (int lane = 0; lane < 16; ++lane) {
          if ((guard_lanes & (1u << lane)) == 0) continue;
          const int px = x + lane;
          const int q = px >> 1;
          const int cr0 = static_cast<int>(src_cr[px]) - 128;
          const int cb0 = static_cast<int>(src_cb[px]) - 128;
          int cr = static_cast<int>(dst_cr[px]) - 128;
          int cb = static_cast<int>(dst_cb[px]) - 128;
          apply_hue_guard(cr0, cb0, q, pbase_cr, pbase_cb, cr, cb);
          cr = (std::max)(-128, (std::min)(cr, 127));
          cb = (std::max)(-128, (std::min)(cb, 127));
          dst_cr[px] = static_cast<uchar>(128 + cr);
          dst_cb[px] = static_cast<uchar>(128 + cb);
        }
      }
    }
#endif
    for (; x < half.width; ++x) {
      const int xq = (std::min)(x >> 1, quarter.width - 1);
      const int cr0 = static_cast<int>(src_cr[x]) - 128;
      const int cb0 = static_cast<int>(src_cb[x]) - 128;
      const int cr1 = static_cast<int>(dst_cr[x]) - 128;
      const int cb1 = static_cast<int>(dst_cb[x]) - 128;
      const int alpha_q8 = pblend[xq];
      int cr = cr0 + round_shift8_signed((cr1 - cr0) * alpha_q8);
      int cb = cb0 + round_shift8_signed((cb1 - cb0) * alpha_q8);

      if (hue_guard && phue[xq] != 0)
        apply_hue_guard(cr0, cb0, xq, pbase_cr, pbase_cb, cr, cb);
      cr = (std::max)(-128, (std::min)(cr, 127));
      cb = (std::max)(-128, (std::min)(cb, 127));
      dst_cr[x] = static_cast<uchar>(128 + cr);
      dst_cb[x] = static_cast<uchar>(128 + cb);
    }
  });
  const auto detail_t5 = detail_now();

  // Fused output: y' = a*y + b (coeffs NN from the quarter grid), chroma
  // NN-upsampled from the half grid, YCrCb->BGR in BT.601 integer math,
  // written straight into `color`. Use an architecture-specific backend
  // where available; the existing universal-intrinsics loop below remains
  // the bit-exact fallback for SSE-only x86 and other architectures.
  const auto report_detail = [&] {
    if (!detail_profile) return;
    const auto detail_t6 = DetailClock::now();
    const auto ms = [](DetailClock::duration duration) {
      return std::chrono::duration<double, std::milli>(duration).count();
    };
    AddFastPostDetailProfile({ms(detail_t1 - detail_t0),
                              ms(detail_t2 - detail_t1),
                              ms(detail_t2a - detail_t2),
                              ms(detail_t2b - detail_t2a),
                              ms(detail_t2c - detail_t2b),
                              ms(detail_t3 - detail_t2c),
                              ms(detail_t4 - detail_t3),
                              ms(detail_t5 - detail_t4),
                              ms(detail_t6 - detail_t5)});
  };
#if defined(CYPERSTEREO_HAVE_AVX2_OUTPUT)
  if (CpuHasAvx2Output()) {
    FusedOutputAvx2Paired(y8, a_q16, b_q16, cr_h, cb_h, color, apply_tone);
    report_detail();
    return;
  }
#endif
#if defined(CYPERSTEREO_HAVE_NEON)
  if (UseNeonOutput()) {
    FusedOutputNeon(y8, a_q16, b_q16, cr_h, cb_h, color, apply_tone);
    report_detail();
    return;
  }
#endif
  const FastBalancedToneLuts *tone_luts =
      apply_tone ? &GetFastBalancedToneLuts() : nullptr;
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
      if (tone_luts)
        ApplyFastBalancedTone16(outb, outg, outr, *tone_luts);
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
        if (tone_luts)
          ApplyFastBalancedTonePixel(o[0], o[1], o[2], *tone_luts);
      }
    }
  }
  report_detail();
}

inline void SuppressFalseColor(cv::Mat &color) {
  SuppressFalseColorImpl(nullptr, color);
}

inline void SuppressFalseColorLinear(cv::Mat &color, double sensor_gain,
                                     bool apply_tone = false) {
  SuppressFalseColorImpl(nullptr, color, sensor_gain, true, apply_tone);
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
                     const char *name,
                     BayerConversion bayer =
                         BayerConversion::kColorBayerRg2Bgr) {
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
  // The hand-fused ARM kernel implements the legacy RG2BGR-equivalent CFA.
  // Use OpenCV EA for the opposite phase until that kernel has a pattern-aware
  // implementation; correctness takes priority for software version 04.
  const bool fused_demosaic =
      UseFusedDemosaic() &&
      bayer == BayerConversion::kColorBayerRg2Bgr;
  const int bayer_code =
      bayer == BayerConversion::kColorBayerBg2Bgr
          ? cv::COLOR_BayerBG2BGR_EA
          : cv::COLOR_BayerRG2BGR_EA;
  if (!IspProfiler::Enabled()) {
    wb.Apply(raw, bayer);
    if (fused_demosaic) {
      SuppressFalseColorImpl(&raw, color);
    } else {
      cv::cvtColor(raw, color, bayer_code);
      ApplyCcmTone(color);
      SuppressFalseColor(color);
    }
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  wb.Apply(raw, bayer);
  const auto t1 = std::chrono::steady_clock::now();
  if (!fused_demosaic) {
    cv::cvtColor(raw, color, bayer_code);
    ApplyCcmTone(color);  // accounted under the "demosaic" bucket
  }
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

// Fast path with the colour-safe ordering used by the HDR quality pipeline:
// linear BLC/WB (+ BayerNR at gain >=3x) -> EA demosaic -> linear
// luma/chroma/false-colour NR -> one shared-luminance gamma. Stable and local
// colour vectors control a joint Cr/Cb blend, preventing the blanket shadow
// desaturation that previously made the fast result grey/dark. The caller's
// RAW frame is left untouched.
inline void ApplyFastBalancedISP(
    cv::Mat &raw, cv::Mat &color, WhiteBalance &wb, const char *name,
    double sensor_gain = 1.0,
    BayerConversion bayer = BayerConversion::kColorBayerRg2Bgr) {
  static thread_local bool sched_done = false;
  if (!sched_done) {
    sched_done = true;
    ApplyThreadPriority(ThreadRole::kWorker, name);
  }
  FastIspFrameParallelGuard frame_parallel_guard;
#if defined(CYPERSTEREO_HAVE_NEON)
  // FusedDemosaicFrontNeon processes a 32-column interior block beginning at
  // x=2 and therefore needs W>=36 (loads through x+33). H>=4 supplies the
  // two interior rows used to reproduce OpenCV's top/bottom border. Public
  // callers may pass images as small as 4x4, so keep those on the safe,
  // bit-compatible OpenCV EA fallback instead of entering the NEON kernel.
  const bool fused_geometry_safe = raw.cols >= 36 && raw.rows >= 4;
#else
  const bool fused_geometry_safe = true;
#endif
  const bool fused_demosaic =
      UseFastFusedDemosaic() && fused_geometry_safe &&
      bayer == BayerConversion::kColorBayerRg2Bgr;
  const int bayer_code = bayer == BayerConversion::kColorBayerBg2Bgr
      ? cv::COLOR_BayerBG2BGR_EA
      : cv::COLOR_BayerRG2BGR_EA;
  const bool use_bayer_nr = FastBalancedBayerNrEnabled() &&
                            sensor_gain >= 3.0;
  const bool apply_tone = FastBalancedGammaEnabled();
  static thread_local cv::Mat raw_wb;

  if (!IspProfiler::Enabled()) {
    if (use_bayer_nr) {
      wb.ApplyWithBayerNr(raw, raw_wb, sensor_gain, bayer);
    } else if (FastBalancedFusedWbCopyEnabled()) {
      wb.ApplyTo(raw, raw_wb, bayer);
    } else {
      raw.copyTo(raw_wb);
      wb.Apply(raw_wb, bayer);
    }
    if (fused_demosaic)
      SuppressFalseColorImpl(&raw_wb, color, sensor_gain, true, apply_tone);
    else {
      cv::cvtColor(raw_wb, color, bayer_code);
      SuppressFalseColorLinear(color, sensor_gain, apply_tone);
    }
    return;
  }

  const auto t0 = std::chrono::steady_clock::now();
  if (use_bayer_nr) {
    wb.ApplyWithBayerNr(raw, raw_wb, sensor_gain, bayer);
  } else if (FastBalancedFusedWbCopyEnabled()) {
    wb.ApplyTo(raw, raw_wb, bayer);
  } else {
    raw.copyTo(raw_wb);
    wb.Apply(raw_wb, bayer);
  }
  const auto t1 = std::chrono::steady_clock::now();
  if (!fused_demosaic) cv::cvtColor(raw_wb, color, bayer_code);
  const auto t2 = std::chrono::steady_clock::now();
  if (fused_demosaic)
    SuppressFalseColorImpl(&raw_wb, color, sensor_gain, true, apply_tone);
  else
    SuppressFalseColorLinear(color, sensor_gain, apply_tone);
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
        ApplyISP(*j.raw, *j.color, *j.wb, j.name, j.bayer);
        lk.lock();
        has_job_ = false;
        lk.unlock();
        cv_done_.notify_one();
      }
    });
    thr_.detach();
  }

  void Submit(cv::Mat &raw, cv::Mat &color, WhiteBalance &wb,
              const char *name, BayerConversion bayer) {
    std::lock_guard<std::mutex> lk(mtx_);
    job_ = Job{&raw, &color, &wb, name, bayer};
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
    BayerConversion bayer;
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
  BayerConversion bayer;

  IspJob(cv::Mat &r, cv::Mat &c, WhiteBalance &w, const char *n,
         BayerConversion b = BayerConversion::kColorBayerRg2Bgr)
      : raw(&r), color(&c), wb(&w), name(n), bayer(b) {}
};

// jobs[0] runs on the calling thread; jobs[1..] on persistent workers.
inline void ApplyISPParallel(const IspJob *jobs, int n) {
  constexpr int kMaxWorkers = 7;
  static IspWorker workers[kMaxWorkers];
  if (n <= 0) return;
  if (n > kMaxWorkers + 1) n = kMaxWorkers + 1;
  for (int i = 1; i < n; ++i)
    workers[i - 1].Submit(*jobs[i].raw, *jobs[i].color, *jobs[i].wb,
                           jobs[i].name, jobs[i].bayer);
  ApplyISP(*jobs[0].raw, *jobs[0].color, *jobs[0].wb, jobs[0].name,
           jobs[0].bayer);
  for (int i = 1; i < n; ++i) workers[i - 1].Wait();
}

inline void ApplyISPParallel(std::initializer_list<IspJob> jobs) {
  ApplyISPParallel(jobs.begin(), static_cast<int>(jobs.size()));
}

class FastBalancedIspWorker {
 public:
  explicit FastBalancedIspWorker(int cpu = -1)
      : cpu_(cpu), thread_([this] { Run(); }) {}
  ~FastBalancedIspWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    go_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

  FastBalancedIspWorker(const FastBalancedIspWorker &) = delete;
  FastBalancedIspWorker &operator=(const FastBalancedIspWorker &) = delete;

  void Submit(cv::Mat &raw, cv::Mat &color, WhiteBalance &wb,
              const char *name, double sensor_gain,
              BayerConversion bayer) {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return !has_job_; });
    job_ = Job{&raw, &color, &wb, name, sensor_gain, bayer};
    error_ = nullptr;
    has_job_ = true;
    lock.unlock();
    go_.notify_one();
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return !has_job_; });
    std::exception_ptr error = error_;
    lock.unlock();
    if (error) std::rethrow_exception(error);
  }

 private:
  struct Job {
    cv::Mat *raw;
    cv::Mat *color;
    WhiteBalance *wb;
    const char *name;
    double sensor_gain;
    BayerConversion bayer;
  };

  void Run() {
    if (cpu_ >= 0) PinThreadToCpu(cpu_);
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      go_.wait(lock, [this] { return stop_ || has_job_; });
      if (stop_) return;
      const Job job = job_;
      lock.unlock();
      std::exception_ptr error;
      try {
        ApplyFastBalancedISP(*job.raw, *job.color, *job.wb, job.name,
                             job.sensor_gain, job.bayer);
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();
      error_ = error;
      has_job_ = false;
      lock.unlock();
      done_.notify_all();
    }
  }

  std::mutex mutex_;
  std::condition_variable go_, done_;
  Job job_{};
  bool has_job_ = false;
  bool stop_ = false;
  std::exception_ptr error_;
  int cpu_ = -1;
  std::thread thread_;
};

struct FastBalancedIspJob {
  cv::Mat *raw;
  cv::Mat *color;
  WhiteBalance *wb;
  const char *name;
  double sensor_gain;
  BayerConversion bayer;

  FastBalancedIspJob(
      cv::Mat &r, cv::Mat &c, WhiteBalance &w, const char *n,
      double gain = 1.0,
      BayerConversion b = BayerConversion::kColorBayerRg2Bgr)
      : raw(&r), color(&c), wb(&w), name(n), sensor_gain(gain), bayer(b) {}
};

inline void ApplyFastBalancedISPParallel(const FastBalancedIspJob *jobs,
                                         int n) {
  constexpr int kMaxWorkers = 3;
  if (n <= 0) return;
  CV_Assert(jobs != nullptr);
#if defined(CYPERSTEREO_RK3588)
  // jobs[0] runs on the caller at CPU4. The three persistent workers process
  // jobs[1..3] on CPU5..7, respectively. Slot-based mapping is intentional:
  // callers such as fast_isp_benchmark use the same name for all four jobs.
  static FastBalancedIspWorker worker1(5);
  static FastBalancedIspWorker worker2(6);
  static FastBalancedIspWorker worker3(7);
#else
  static FastBalancedIspWorker worker1;
  static FastBalancedIspWorker worker2;
  static FastBalancedIspWorker worker3;
#endif
  FastBalancedIspWorker *const workers[kMaxWorkers] = {
      &worker1, &worker2, &worker3};
  // Preserve batch ownership until every copied job pointer is no longer in
  // use. This is uncontended in capture/save, and also makes diagnostic tools
  // with multiple submitting threads deterministic.
  static std::mutex batch_mutex;
  std::lock_guard<std::mutex> batch_lock(batch_mutex);
  if (n > kMaxWorkers + 1) n = kMaxWorkers + 1;
  struct BatchModeGuard {
    explicit BatchModeGuard(bool enabled)
        : enabled_(enabled && FastIspSingleFrameParallelEnabled()) {
      if (enabled_) GetFastIspStagePool().BeginMultiCameraBatch();
    }
    ~BatchModeGuard() {
      if (enabled_) GetFastIspStagePool().EndMultiCameraBatch();
    }
    bool enabled_;
  } multi_camera_guard(n > 1);
#if defined(CYPERSTEREO_RK3588)
  // The application owns the caller, so bind it only while it executes job0.
  // This also overrides a previous single-frame low-latency CPU7 binding for
  // the duration of the batch, then restores the caller's original mask.
  ScopedThreadAffinity caller_affinity(n > 1 ? 4 : -1);
#endif
  for (int i = 1; i < n; ++i)
    workers[i - 1]->Submit(*jobs[i].raw, *jobs[i].color, *jobs[i].wb,
                           jobs[i].name, jobs[i].sensor_gain, jobs[i].bayer);
  std::exception_ptr caller_error;
  try {
    ApplyFastBalancedISP(*jobs[0].raw, *jobs[0].color, *jobs[0].wb,
                         jobs[0].name, jobs[0].sensor_gain, jobs[0].bayer);
  } catch (...) {
    caller_error = std::current_exception();
  }
  std::exception_ptr worker_error;
  for (int i = 1; i < n; ++i) {
    try {
      workers[i - 1]->Wait();
    } catch (...) {
      if (!worker_error) worker_error = std::current_exception();
    }
  }
  if (caller_error) std::rethrow_exception(caller_error);
  if (worker_error) std::rethrow_exception(worker_error);
}

inline void ApplyFastBalancedISPParallel(
    std::initializer_list<FastBalancedIspJob> jobs) {
  ApplyFastBalancedISPParallel(jobs.begin(), static_cast<int>(jobs.size()));
}

inline bool FastBalancedIspEnabled() {
  const char *mode = std::getenv("CYPERSTEREO_ISP_MODE");
  // Fast-balanced is the production default.  Setting any explicit non-fast
  // mode (for example "quality") selects the HDR-ISP reference path; the
  // capture sample also exposes --isp-quality as a command-line override.
  if (!mode) return true;
  return std::strcmp(mode, "fast") == 0 ||
         std::strcmp(mode, "fast-balanced") == 0;
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

          // SmartSens hardware 02 has two supported FPGA metadata/software
          // layouts.  Version 03 uses the legacy Bayer conversion on all four
          // planes; version 04 marks C1/C4 (images 1/3) as the opposite Bayer
          // phase after sensor mirror+flip.  Their metadata layout is otherwise
          // compatible, so accept and latch either marker.
          if (prof.num_cameras >= 4 &&
              IsSupportedSmartSensFirmware(ver0, ver1)) {
            if (ver0 != frame_info.profile.hardware_version ||
                ver1 != frame_info.profile.software_version) {
              frame_info.profile.hardware_version = ver0;
              frame_info.profile.software_version = ver1;
              std::cout << "[api] SmartSens firmware from metadata: "
                        << ver0 << "/" << ver1 << std::endl;
            }
            marker_ok = true;
          }

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

          // Per-camera AE telemetry (SmartSens SC136HGS 4-cam only).  The FPGA
          // metadata row packs a full 16-bit value per column into the two low
          // byte lanes, so meta_u(col) returns it directly:
          //   col 69..72   = FPGA AEC target exposure lines C1,C2,C3,C4
          //   col 73..76   = temperature raw C1,C2,C3,C4 ({0x4c10,0x4c11})
          //   col 77..80   = FPGA AEC target gain code C1,C2,C3,C4
          // Values are forwarded directly from the controller state; no
          // exposure/gain sensor-register readback is performed.
          // FPGA byte-lane wiring is DQ[7:0]=C1, DQ[15:8]=C2,
          // DQ[23:16]=C4, DQ[31:24]=C3.  DeinterleaveFourPlanes therefore maps
          // image1..4 to C1,C2,C4,C3 respectively.  All metadata arrays below
          // use that display-plane order so each print follows its own image.
          if (prof.num_cameras >= 4) {
            // FX3 image-plane order is C1,C2,C4,C3.  Metadata columns remain
            // in physical-sensor order C1,C2,C3,C4, so map each plane here.
            static const int kExpColByPlane[4]     = {69, 70, 72, 71};
            static const int kTempColByPlane[4]    = {73, 74, 76, 75};
            static const int kGainColByPlane[4]    = {77, 78, 80, 79};
            for (int p = 0; p < 4; ++p) {
              const int lines = meta_u(kExpColByPlane[p]);
              fs.exposure_lines[p] = static_cast<uint16_t>(lines);
              fs.exposure_time[p]  = lines * kSmartSensLineTimeSec;
              const int traw = meta_u(kTempColByPlane[p]);   // {0x4c10,0x4c11}
              fs.camera_temperature[p] =
                  ((traw >> 8) * 8 + (traw & 0x7)) / 4.0 - 273.15;
              fs.camera_gain[p] = meta_u(kGainColByPlane[p]) / 64.0;
              // Estimated capture midpoint from the target exposure.
              fs.image_midpoint_timestamp[p] =
                  fs.image_timestamp - fs.exposure_time[p] / 2.0;
            }
          }

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

          // SmartSens 4-cam has no GNSS module, and its metadata cols 72..80 now
          // carry the per-camera AE telemetry parsed above, which would collide
          // with the GNSS byte layout.  Only parse GNSS on the MT9V families.
          if (prof.num_cameras >= 4) {
            fs.gnss.valid = false;
          } else {
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
