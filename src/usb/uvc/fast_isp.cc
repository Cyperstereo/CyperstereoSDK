// Fast-ISP quality bridge, see hdr_isp.h for the stable public interface and
// environment knobs.
//
// Parameter provenance:
//   BLC  = 16 DN (8-bit domain)      -- kIspBlackLevel of the old pipeline
//                                       (SC136HGS pedestal after the FPGA's
//                                       10->8 bit truncation).
//   CCM  = calibrated R-major matrix -- same default as the old ApplyCcmTone
//                                       (ColorChecker capture 741.png,
//                                       tools/ccm_calibrate.py).
//   WB   = per-camera robust gray-world (median of g/b, g/r ratios), the
//          estimator used by the old WhiteBalance class.
//   Tone = mild shared-luminance gamma (1.20, max 1.35x lift) by default;
//          explicit tone env knobs select the ported sRGB/power + S-curve.
//
// Sensor analogue gain is passed to gain-adaptive BayerNR and LumaNR. Exposure
// time remains camera metadata; the bridge is a renderer, not an AE stage.
#include "hdr_isp.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/pipeline.h"  // HDR-ISP-main

// easylogging++ needs exactly one initialization TU in the final binary;
// HDR-ISP-main's own main.cpp is not compiled into the SDK, so it lives here.
INITIALIZE_EASYLOGGINGPP

namespace cyperstereo {
namespace {

// ---------------------------------------------------------------------------
// Fixed sensor / pipeline constants.

// Sensor pedestal in the 8-bit domain (== kIspBlackLevel of the old ISP).
constexpr int kBlackLevel8 = 16;

// The 8-bit Bayer input is expanded by <<kBitShift into HDR-ISP's int32 raw
// domain so the integer modules (WB, CCM) keep 2 extra fraction bits.
constexpr int kBitShift = 2;
constexpr int kMaxVal = (256 << kBitShift) - 1;  // 1023
// Post-BLC full scale: a saturated pixel is (255<<2) - blc = 956, not 1023.
// The rgbgamma curve is built against this so white stays at 255 out.
constexpr int kBlc = kBlackLevel8 << kBitShift;
constexpr int kLinearFullScale = (255 << kBitShift) - kBlc;

// Default CCM, R-major rows applied to [R,G,B] -- identical convention in
// the old CcmMatrixBgr() and in HDR-ISP's ccm module. From the 741.png
// ColorChecker calibration (20/24 patches, fit err ~48 DN).
constexpr double kDefaultCcm[9] = {
    1.7020, -0.6295, -0.0725,
    -0.4929, 1.7391, -0.2462,
    0.1409, -0.7787, 1.6378,
};

// ---------------------------------------------------------------------------
// Env helpers (same semantics as the old pipeline's knobs).

double EnvGamma() {
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

double EnvBlackPoint() {
  static const double bp = [] {
    if (const char *e = std::getenv("CYPERSTEREO_BLACKPOINT")) {
      const double v = std::atof(e);
      if (v >= 0.0 && v <= 32.0) return v;
    }
    return 6.0;
  }();
  return bp;
}

double EnvContrast() {
  static const double a = [] {
    if (const char *e = std::getenv("CYPERSTEREO_CONTRAST")) {
      const double v = std::atof(e);
      if (v >= 0.0 && v <= 1.0) return v;
    }
    return 0.3;
  }();
  return a;
}

// Mild default curve for rgbgamma.  The module applies it to luminance and
// uses that gain for B/G/R equally, so it reveals a little shadow detail
// without changing hue or materially lifting chroma noise.
constexpr double kDefaultRgbGamma = 1.20;
constexpr double kDefaultRgbGammaMaxGain = 1.35;
// Do not lift the sensor's darkest samples.  The transition ends at 32 DN in
// the original 8-bit post-BLC domain, so shadow detail rises smoothly above
// the noise floor rather than turning it into visible colour speckle.
constexpr double kDefaultRgbGammaNoiseFloor = 32.0;
constexpr double kDefaultRgbGammaRampEnd = 128.0;

// True when the user explicitly asked for the old ApplyISPParallel tone
// pipeline via any of its env knobs; otherwise the conservative default
// luminance curve above is used.
bool UsePortedToneCurve() {
  static const bool on = std::getenv("CYPERSTEREO_GAMMA") != nullptr ||
                         std::getenv("CYPERSTEREO_BLACKPOINT") != nullptr ||
                         std::getenv("CYPERSTEREO_CONTRAST") != nullptr;
  return on;
}

// Encode curve on normalized linear input, 0..1 -> 0..1: sRGB (or power)
// encode followed by black point + smoothstep S (the "photo-finishing"
// stage of the old ToneEncode()).
double ToneCurve01(double n) {
  if (n <= 0.0) return 0.0;
  if (n > 1.0) n = 1.0;
  const double g = EnvGamma();
  if (g == 1.0) return n;  // legacy fully-linear output
  double v;
  if (g == 0.0)
    v = n <= 0.0031308 ? 12.92 * n : 1.055 * std::pow(n, 1.0 / 2.4) - 0.055;
  else
    v = std::pow(n, 1.0 / g);
  const double bp = EnvBlackPoint() / 255.0;
  double t = (v - bp) / (1.0 - bp);
  if (t < 0.0) t = 0.0;
  const double s = t * t * (3.0 - 2.0 * t);  // smoothstep
  const double a = EnvContrast();
  return (1.0 - a) * t + a * s;
}

// R-major 3x3 from CYPERSTEREO_CCM ("off"/garbage -> identity), else the
// calibrated default.
void FillCcm(float ccm[3][3]) {
  double v[9];
  int n = 0;
  if (const char *e = std::getenv("CYPERSTEREO_CCM")) {
    const char *p = e;
    char *end = nullptr;
    while (n < 9) {
      const double d = std::strtod(p, &end);
      if (end == p) break;
      v[n++] = d;
      p = end;
      while (*p == ',' || *p == ';' || *p == ' ') ++p;
    }
    if (n != 9) {  // "off" or unparseable -> identity
      for (int i = 0; i < 9; ++i) v[i] = (i % 4 == 0) ? 1.0 : 0.0;
      n = 9;
    }
  } else {
    for (int i = 0; i < 9; ++i) v[i] = kDefaultCcm[i];
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ccm[i][j] = static_cast<float>(v[3 * i + j]);
}

std::list<std::string> PipeModules() {
  // Default: BLC -> same-colour Bayer NR -> WB -> edge-aware demosaic ->
  // linear guided luma NR -> joint chroma NR -> texture-gated false-colour
  // suppression -> rgbgamma.
  // Both BGR-domain stages keep Dr/Db coupled, so no unsafe uint8 YUV round
  // trip or independent U/V filtering can rotate red toward orange.
  // CCM stays OFF so the sensor's native demosaiced colour is retained.
  // Full chain example via CYPERSTEREO_HDRISP_PIPE (uses HDR-ISP's own tone):
  //   set CYPERSTEREO_HDRISP_PIPE=blc|wbgain|demoasic|ccm|rgbgamma|rgb2yuv|contrast|sharpen|cns|yuv2rgb
  const char *e = std::getenv("CYPERSTEREO_HDRISP_PIPE");
  const std::string s = e ? e :
      "blc|bayernr|wbgain|demoasic|lumanr|chromanr|falsecolor|rgbgamma";
  std::list<std::string> mods;
  std::string cur;
  for (char c : s) {
    if (c == '|') {
      if (!cur.empty()) mods.push_back(cur);
      cur.clear();
    } else if (!std::isspace(static_cast<unsigned char>(c))) {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) mods.push_back(cur);
  return mods;
}

bool ProfileEnabled() {
  static const bool on = std::getenv("CYPERSTEREO_HDRISP_PROFILE") != nullptr;
  return on;
}

// ---------------------------------------------------------------------------
// Per-camera context.

struct CamCtx {
  IspPrms prms{};
  std::unique_ptr<Frame> frame;
  IspPipeline pipeline;

  // Where the final BGR result lands depends on the last module:
  //   yuv2rgb  -> 8-bit interleaved BGR in data.bgr_u8_o
  //   otherwise-> int32 BGR in data.bgr_s32_i (demosaic/ccm/rgbgamma/ltm all
  //               leave their result there after the swap)
  bool output_u8 = true;
  // True when the pipe contains rgbgamma: the int32 BGR result is already in
  // 0..255 and read 1:1. When false (bare demosaic/ccm output), the bridge
  // linearly rescales the post-BLC range to BGR8 at output.
  bool gamma_scaled = false;

  // AWB state (robust gray-world, carried over from WhiteBalance).
  uint32_t frame_idx = 0;
  double b_gain = -1.0, r_gain = -1.0;
  std::vector<float> ratios_bg, ratios_rg;

  // profiling
  double acc_ms = 0.0;
  int acc_n = 0;
};

void InitPrms(IspPrms &prms, int width, int height, CfaTypes cfa) {
  prms.info.width = width;
  prms.info.height = height;
  prms.info.bpp = 16;
  prms.info.max_val = kMaxVal;
  prms.info.mipi_packed = false;
  prms.info.cfa = cfa;
  prms.info.dt = RawDataTypes::RAW8;
  prms.info.domain = ColorDomains::RAW;

  prms.blc = kBlc;

  FillCcm(prms.ccm_prms.ccm);

  // Neutral start; overwritten every frame by the AWB estimator.
  for (int i = 0; i < 4; ++i) {
    prms.wb_gains.d65_gain[i] = 1.0f;
    prms.wb_gains.d50_gain[i] = 1.0f;
    prms.wb_gains.f11_gain[i] = 1.0f;
    prms.wb_gains.f12_gain[i] = 1.0f;
  }

  // rgbgamma: 10-bit linear in -> 8-bit encoded out. HDR-ISP interpolates
  // `nums` knots evenly over the input span [0, 1<<in_bits).
  //
  // Default: a mild gamma-1.20 luminance curve whose gain is capped at 1.35x.
  // When a CYPERSTEREO_GAMMA/_BLACKPOINT/_CONTRAST knob is set we instead bake
  // the ported ApplyISPParallel sRGB+blackpoint+S curve into 21 knots,
  // rescaled so post-BLC full scale (956) maps to 1.0 (white stays white).
  prms.rgb_gamma.in_bits = 10;
  prms.rgb_gamma.out_bits = 8;
  if (UsePortedToneCurve()) {
    prms.rgb_gamma.nums = MAX_GAMMA_NUMS;  // 21
    for (int k = 0; k < MAX_GAMMA_NUMS; ++k) {
      const double linear = k * ((1 << 10) / double(MAX_GAMMA_NUMS - 1));
      prms.rgb_gamma.curve[k] =
          static_cast<float>(ToneCurve01(linear / kLinearFullScale));
    }
  } else {
    prms.rgb_gamma.nums = MAX_GAMMA_NUMS;
    for (int k = 0; k < MAX_GAMMA_NUMS; ++k) {
      const double linear = k * ((1 << 10) / double(MAX_GAMMA_NUMS - 1));
      double n = linear / kLinearFullScale;
      if (n > 1.0) n = 1.0;
      double tone = std::pow(n, 1.0 / kDefaultRgbGamma);
      if (n > 0.0)
        tone = (std::min)(tone, n * kDefaultRgbGammaMaxGain);
      // Keep the noise floor linear and blend into the lift gradually.
      double ramp = (linear - kDefaultRgbGammaNoiseFloor) /
                    (kDefaultRgbGammaRampEnd - kDefaultRgbGammaNoiseFloor);
      ramp = (std::min)(1.0, (std::max)(0.0, ramp));
      tone = n + ramp * (tone - n);
      prms.rgb_gamma.curve[k] = static_cast<float>(tone);
    }
  }
  // ygamma identity (only used if "ygamma" is added to the pipe by env).
  prms.y_gamma.nums = MAX_GAMMA_NUMS;
  prms.y_gamma.in_bits = 8;
  prms.y_gamma.out_bits = 8;
  for (int k = 0; k < MAX_GAMMA_NUMS; ++k)
    prms.y_gamma.curve[k] = k / float(MAX_GAMMA_NUMS - 1);

  // YUV-domain finishing, HDR-ISP defaults (dsc config), sharpen softened
  // for the SC136HGS noise floor.
  prms.contrast_prms.ratio = 0.1f;
  prms.sharpen_prms.ratio = 0.3f;
  prms.sat_prms.rotate_angle = 0.0f;

  // Only used if the env pipe inserts them.
  prms.ltm_prms.in_bits = 10;
  prms.ltm_prms.out_bits = 10;
  prms.dpc_prms.thres = 30;
  prms.dpc_prms.mode = DpcMode::GRADIENT;
}

// Adapt RAW denoise to each sensor's live AEC gain. The metadata reports a
// nominal real gain in [1, 8]. Invalid/missing telemetry stays at the original
// conservative 1x parameters. Integer interpolation avoids frame-to-frame
// floating-point differences in the Bayer module itself.
void UpdateGainAdaptiveBayerNr(IspPrms &prms, double sensor_gain) {
  if (!(sensor_gain >= 1.0)) sensor_gain = 1.0;  // also rejects NaN
  sensor_gain = (std::min)(sensor_gain, 8.0);
  const int t_q8 = static_cast<int>(
      (sensor_gain - 1.0) * (256.0 / 7.0) + 0.5);

  BayerNrPrms &p = prms.bayer_nr_prms;
  p.threshold_base = 16 + ((24 - 16) * t_q8 + 128) / 256;
  p.threshold_signal_q8 = 4 + ((6 - 4) * t_q8 + 128) / 256;
  p.strength_q8 = 128 + ((176 - 128) * t_q8 + 128) / 256;
  p.center_weight = sensor_gain >= 4.0 ? 3 : 4;

  // Luma guided NR is deliberately light at low gain and reaches a 75%
  // blend at the sensor's 8x ceiling. eps is a variance in the 10-bit linear
  // domain: 448 corresponds to a noise/edge transition of about 5.3 RAW8 DN.
  LumaNrPrms &l = prms.luma_nr_prms;
  l.eps = 128 + ((448 - 128) * t_q8 + 128) / 256;
  l.strength_q8 = 64 + ((192 - 64) * t_q8 + 128) / 256;
}

// Robust gray-world on 8-bit BGGR/RGGB raw: median of per-sample g/b and g/r
// ratios (pedestal-subtracted), clamped to a plausible illuminant range and
// EMA-smoothed. Verbatim port of WhiteBalance::EstimateGains.
void EstimateWbGains(CamCtx &ctx, const cv::Mat &raw) {
  constexpr int kEstStep = 8;
  constexpr double kSmooth = 0.05;
  constexpr double kGainMin = 0.6, kGainMax = 2.6;
  const int lo = kBlackLevel8 + 24;
  const int hi = 250;

  ctx.ratios_bg.clear();
  ctx.ratios_rg.clear();
  int n_total = 0;
  for (int y = 0; y + 1 < raw.rows; y += kEstStep) {
    const uchar *r0 = raw.ptr<uchar>(y);
    const uchar *r1 = raw.ptr<uchar>(y + 1);
    for (int x = 0; x + 1 < raw.cols; x += kEstStep) {
      const bool rggb = ctx.prms.info.cfa == CfaTypes::RGGB;
      const int b = rggb ? r1[x + 1] : r0[x];
      const int g0 = r0[x + 1];
      const int g1 = r1[x];
      const int r = rggb ? r0[x] : r1[x + 1];
      const int g = (g0 + g1) >> 1;
      ++n_total;
      // Reject clipped samples: a clipped channel drags the ratio to 1.0.
      if (b >= 250 || g0 >= 250 || g1 >= 250 || r >= 250) continue;
      const int luma = (b + 2 * g + r) >> 2;
      if (luma < lo || luma > hi) continue;
      const double bb = b - kBlackLevel8;
      const double gg = g - kBlackLevel8;
      const double rr = r - kBlackLevel8;
      if (bb < 4.0 || gg < 4.0 || rr < 4.0) continue;  // noise-dominated
      ctx.ratios_bg.push_back(static_cast<float>(gg / bb));
      ctx.ratios_rg.push_back(static_cast<float>(gg / rr));
    }
  }
  if (static_cast<int>(ctx.ratios_bg.size()) < n_total / 100) return;

  const auto median = [](std::vector<float> &v) {
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return static_cast<double>(v[mid]);
  };
  // (std::min) parentheses: easylogging++ pulls in windows.h whose min/max
  // macros would otherwise mangle these calls under MSVC.
  const double b_gain =
      (std::min)(kGainMax, (std::max)(kGainMin, median(ctx.ratios_bg)));
  const double r_gain =
      (std::min)(kGainMax, (std::max)(kGainMin, median(ctx.ratios_rg)));
  if (ctx.b_gain <= 0.0) {
    ctx.b_gain = b_gain;
    ctx.r_gain = r_gain;
  } else {
    ctx.b_gain = (1.0 - kSmooth) * ctx.b_gain + kSmooth * b_gain;
    ctx.r_gain = (1.0 - kSmooth) * ctx.r_gain + kSmooth * r_gain;
  }
}

// Registry / pipeline construction in HDR-ISP-main is not thread-safe
// (global module map, easylogging setup), so context creation is serialized.
std::mutex g_ctx_mutex;
std::map<std::string, std::unique_ptr<CamCtx>> g_ctxs;

void QuietLogsOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    // HDR-ISP logs one INFO line per module per frame; that is unusable at
    // 30 fps x 4 cameras. Warnings/errors stay visible.
    el::Loggers::reconfigureAllLoggers(el::Level::Info,
                                       el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureAllLoggers(el::Level::Debug,
                                       el::ConfigurationType::Enabled, "false");
  });
}

inline CfaTypes HdrCfa(BayerConversion bayer) {
  // The existing COLOR_BayerRG2BGR-equivalent path is BGGR in HDR-ISP's
  // physical-CFA naming; the opposite conversion is RGGB.
  return bayer == BayerConversion::kColorBayerBg2Bgr ? CfaTypes::RGGB
                                                      : CfaTypes::BGGR;
}

CamCtx *GetCtx(const char *name, int width, int height,
               BayerConversion bayer) {
  std::lock_guard<std::mutex> lk(g_ctx_mutex);
  QuietLogsOnce();
  const CfaTypes cfa = HdrCfa(bayer);
  auto &slot = g_ctxs[name];
  if (slot && (slot->prms.info.width != width ||
               slot->prms.info.height != height ||
               slot->prms.info.cfa != cfa))
    slot.reset();  // resolution/CFA changed: rebuild buffers and AWB state
  if (!slot) {
    slot.reset(new CamCtx());
    InitPrms(slot->prms, width, height, cfa);
    slot->frame.reset(new Frame(slot->prms.info));
    const std::list<std::string> mods = PipeModules();
    slot->output_u8 = !mods.empty() && mods.back() == "yuv2rgb";
    // rgbgamma rescales to out_bits (0..255) and is read 1:1. Without it the
    // bare linear BGR result is directly normalized during extraction.
    slot->gamma_scaled =
        std::find(mods.begin(), mods.end(), "rgbgamma") != mods.end();
    if (slot->pipeline.MakePipe(mods) != 0) {
      std::fprintf(stderr,
                   "[hdr-isp] invalid pipe (CYPERSTEREO_HDRISP_PIPE), check "
                   "module order/domains\n");
    }
  }
  return slot.get();
}

// ApplyHdrIspParallel processes at most four camera jobs in normal use. Keep
// the three non-calling-thread lanes alive between frames so the sizeable
// thread_local scratch buffers owned by demosaic/NR modules are reused rather
// than allocated again for every frame. Lane i always processes job i (and,
// for defensive compatibility with callers passing more than four jobs,
// i+4, i+8, ...), matching the old job-to-thread layout for the 1..4 case.
class HdrIspWorkerPool {
 public:
  HdrIspWorkerPool() {
    workers_.reserve(kWorkerCount);
    try {
      for (int lane = 1; lane <= kWorkerCount; ++lane)
        workers_.emplace_back(&HdrIspWorkerPool::WorkerLoop, this, lane);
    } catch (...) {
      {
        std::lock_guard<std::mutex> lk(state_mutex_);
        stopping_ = true;
      }
      start_cv_.notify_all();
      for (auto &worker : workers_)
        if (worker.joinable()) worker.join();
      throw;
    }
  }

  ~HdrIspWorkerPool() {
    // Waiting for the batch lock also makes destruction safe if shutdown
    // races an in-flight call: no worker can still be using caller-owned job
    // descriptors when the stop flag is published.
    std::lock_guard<std::mutex> batch_lk(batch_mutex_);
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      stopping_ = true;
    }
    start_cv_.notify_all();
    for (auto &worker : workers_)
      if (worker.joinable()) worker.join();
  }

  HdrIspWorkerPool(const HdrIspWorkerPool &) = delete;
  HdrIspWorkerPool &operator=(const HdrIspWorkerPool &) = delete;

  void Run(const HdrIspJob *jobs, int n) {
    CV_Assert(jobs != nullptr && n > 1);

    // The public API is normally called by one capture thread. Serializing
    // batches also makes concurrent callers safe while keeping the published
    // jobs pointer valid until every worker has completed.
    std::lock_guard<std::mutex> batch_lk(batch_mutex_);
    const int active_workers = (std::min)(kWorkerCount, n - 1);
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      jobs_ = jobs;
      job_count_ = n;
      pending_workers_ = active_workers;
      worker_error_ = nullptr;
      ++generation_;
    }
    start_cv_.notify_all();

    std::exception_ptr caller_error;
    try {
      // For the supported 1..4 job case this loop executes job 0 only.
      for (int i = 0; i < n; i += kLaneCount) RunJob(jobs[i]);
    } catch (...) {
      caller_error = std::current_exception();
    }

    std::exception_ptr worker_error;
    {
      std::unique_lock<std::mutex> lk(state_mutex_);
      done_cv_.wait(lk, [this] { return pending_workers_ == 0; });
      worker_error = worker_error_;
      jobs_ = nullptr;
      job_count_ = 0;
    }

    // Always wait for every lane before propagating an exception. Otherwise
    // workers could dereference stack-backed HdrIspJob entries after return.
    if (caller_error) std::rethrow_exception(caller_error);
    if (worker_error) std::rethrow_exception(worker_error);
  }

 private:
  static constexpr int kWorkerCount = 3;
  static constexpr int kLaneCount = kWorkerCount + 1;

  static void RunJob(const HdrIspJob &job) {
    ApplyHdrIsp(*job.raw, *job.color, job.name, job.bayer, job.sensor_gain);
  }

  void WorkerLoop(int lane) {
    uint64_t seen_generation = 0;
    for (;;) {
      const HdrIspJob *jobs;
      int job_count;
      {
        std::unique_lock<std::mutex> lk(state_mutex_);
        start_cv_.wait(lk, [this, seen_generation] {
          return stopping_ || generation_ != seen_generation;
        });
        if (stopping_) return;
        seen_generation = generation_;
        jobs = jobs_;
        job_count = job_count_;
      }

      // Lanes beyond the current job count still consume the generation but
      // are not included in pending_workers_.
      if (lane >= job_count) continue;

      std::exception_ptr error;
      try {
        for (int i = lane; i < job_count; i += kLaneCount) RunJob(jobs[i]);
      } catch (...) {
        error = std::current_exception();
      }

      {
        std::lock_guard<std::mutex> lk(state_mutex_);
        if (error && !worker_error_) worker_error_ = error;
        if (--pending_workers_ == 0) done_cv_.notify_one();
      }
    }
  }

  std::mutex batch_mutex_;
  std::mutex state_mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  std::vector<std::thread> workers_;
  const HdrIspJob *jobs_ = nullptr;
  int job_count_ = 0;
  int pending_workers_ = 0;
  uint64_t generation_ = 0;
  bool stopping_ = false;
  std::exception_ptr worker_error_;
};

HdrIspWorkerPool &WorkerPool() {
  // Constructed after the namespace-static camera contexts and therefore
  // destroyed before them. The destructor joins all workers before Frame,
  // pipeline, logging, or context globals begin teardown.
  static HdrIspWorkerPool pool;
  return pool;
}

}  // namespace

void ApplyHdrIsp(const cv::Mat &raw, cv::Mat &color, const char *name,
                 BayerConversion bayer, double sensor_gain) {
  CV_Assert(raw.type() == CV_8UC1);
  CamCtx *ctx = GetCtx(name, raw.cols, raw.rows, bayer);

  UpdateGainAdaptiveBayerNr(ctx->prms, sensor_gain);

  const auto t0 = std::chrono::steady_clock::now();

  // AWB: estimate on every 2nd frame (EMA makes the rate change invisible).
  if ((ctx->frame_idx++ & 1) == 0 || ctx->b_gain <= 0.0)
    EstimateWbGains(*ctx, raw);
  // WbGain order is R, GR, GB, B; the wbgain module reads d65_gain.
  ctx->prms.wb_gains.d65_gain[0] =
      static_cast<float>(ctx->r_gain > 0.0 ? ctx->r_gain : 1.0);
  ctx->prms.wb_gains.d65_gain[1] = 1.0f;
  ctx->prms.wb_gains.d65_gain[2] = 1.0f;
  ctx->prms.wb_gains.d65_gain[3] =
      static_cast<float>(ctx->b_gain > 0.0 ? ctx->b_gain : 1.0);

  // 8-bit Bayer -> HDR-ISP int32 raw domain (<<2, see kBitShift).
  {
    int32_t *dst = reinterpret_cast<int32_t *>(ctx->frame->data.raw_s32_i);
    for (int y = 0; y < raw.rows; ++y) {
      const uchar *src = raw.ptr<uchar>(y);
      int32_t *d = dst + static_cast<size_t>(y) * raw.cols;
      for (int x = 0; x < raw.cols; ++x) d[x] = src[x] << kBitShift;
    }
  }

  if (ctx->pipeline.RunPipe(ctx->frame.get(), &ctx->prms) != 0) {
    std::fprintf(stderr, "[hdr-isp] pipeline failed for %s\n", name);
    return;
  }

  color.create(raw.rows, raw.cols, CV_8UC3);
  if (ctx->output_u8) {
    // yuv2rgb wrote interleaved BGR u8 -- exactly the cv::Mat 8UC3 layout.
    const uint8_t *src = reinterpret_cast<uint8_t *>(ctx->frame->data.bgr_u8_o);
    for (int y = 0; y < raw.rows; ++y)
      std::memcpy(color.ptr<uchar>(y),
                  src + static_cast<size_t>(y) * raw.cols * 3,
                  static_cast<size_t>(raw.cols) * 3);
  } else if (ctx->gamma_scaled) {
    // rgbgamma already applied one shared-luminance gain and rescaled to
    // 0..255, preserving channel ratios unless gamut clipping is required.
    const int32_t *src =
        reinterpret_cast<int32_t *>(ctx->frame->data.bgr_s32_i);
    for (int y = 0; y < raw.rows; ++y) {
      uchar *d = color.ptr<uchar>(y);
      const int32_t *s = src + static_cast<size_t>(y) * raw.cols * 3;
      for (int i = 0; i < raw.cols * 3; ++i) {
        const int v = s[i];
        d[i] = static_cast<uchar>(v < 0 ? 0 : (v > 255 ? 255 : v));
      }
    }
  } else {
    // Pipeline stopped in the BGR int32 linear domain (blc|wbgain|demoasic
    // [|ccm]). Directly quantize it to BGR8: no gamma or tone mapping.
    const int32_t *src =
        reinterpret_cast<int32_t *>(ctx->frame->data.bgr_s32_i);
    constexpr float kLinearToU8 = 255.0f / kLinearFullScale;
    for (int y = 0; y < raw.rows; ++y) {
      uchar *d = color.ptr<uchar>(y);
      const int32_t *s = src + static_cast<size_t>(y) * raw.cols * 3;
      for (int x = 0; x < raw.cols; ++x) {
        int ob = static_cast<int>(s[3 * x + 0] * kLinearToU8 + 0.5f);
        int og = static_cast<int>(s[3 * x + 1] * kLinearToU8 + 0.5f);
        int orr = static_cast<int>(s[3 * x + 2] * kLinearToU8 + 0.5f);
        d[3 * x + 0] = static_cast<uchar>(ob < 0 ? 0 : (ob > 255 ? 255 : ob));
        d[3 * x + 1] = static_cast<uchar>(og < 0 ? 0 : (og > 255 ? 255 : og));
        d[3 * x + 2] = static_cast<uchar>(orr < 0 ? 0 : (orr > 255 ? 255 : orr));
      }
    }
  }

  if (ProfileEnabled()) {
    const auto t1 = std::chrono::steady_clock::now();
    ctx->acc_ms +=
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (++ctx->acc_n >= 100) {
      std::printf("[hdr-isp] %s avg %.2f ms/frame (last 100)\n", name,
                  ctx->acc_ms / ctx->acc_n);
      ctx->acc_ms = 0.0;
      ctx->acc_n = 0;
    }
  }
}

void ApplyHdrIspParallel(const HdrIspJob *jobs, int n) {
  if (n <= 0) return;
  CV_Assert(jobs != nullptr);
  if (n == 1) {
    ApplyHdrIsp(*jobs[0].raw, *jobs[0].color, jobs[0].name, jobs[0].bayer,
                jobs[0].sensor_gain);
    return;
  }
  WorkerPool().Run(jobs, n);
}

void ApplyHdrIspParallel(std::initializer_list<HdrIspJob> jobs) {
  ApplyHdrIspParallel(jobs.begin(), static_cast<int>(jobs.size()));
}

namespace detail {
void ApplyQualityReferenceISPParallel(
    const cv::Mat *const *raws, cv::Mat *const *outputs,
    const char *const *names, const double *sensor_gains,
    const BayerConversion *bayers, int n) {
  if (n <= 0) return;
  CV_Assert(raws != nullptr && outputs != nullptr && names != nullptr &&
            sensor_gains != nullptr && bayers != nullptr);
  n = std::min(n, 4);
  std::array<HdrIspJob, 4> jobs{};
  for (int i = 0; i < n; ++i) {
    CV_Assert(raws[i] != nullptr && outputs[i] != nullptr &&
              names[i] != nullptr);
    jobs[i].raw = raws[i];
    jobs[i].color = outputs[i];
    jobs[i].name = names[i];
    jobs[i].sensor_gain = sensor_gains[i];
    jobs[i].bayer = bayers[i];
  }
  ApplyHdrIspParallel(jobs.data(), n);
}
}  // namespace detail

}  // namespace cyperstereo
