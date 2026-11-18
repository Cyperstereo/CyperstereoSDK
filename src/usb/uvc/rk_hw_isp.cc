// Rockchip ISP backend for the fast-balanced ISP: ISP39 on RK3576, ISP30 on
// RK3588.
//
// Each RAW8 frame is replayed from memory through the ISP core
// (rawrd2_s -> isp -> mainpath UYVY / selfpath BGR) without rkaiq. The
// colour stages follow the software ISP of the same frame:
//   BLS      = WhiteBalance black level
//   AWB gain = WhiteBalance gray-world gains
//   GOC      = display exposure + highlight roll-off, linear 8-bit output
// (the fast-balanced path has no CCM, so neither has this one).
// The tone stays on the CPU, fused into the copy out of the capture buffer:
// it is a shared-luminance gain (GetFastBalancedToneLuts), which a
// per-channel gamma table cannot express without shifting shadow hues.
// Demosaic, DPCC, GIC and the colour-space stages use the fixed configs in
// rkisp/rk_hw_isp_tuning.inc; YNR/CNR take the row of the tuning's ISO
// ladder that matches the frame's NR gain (NrLevel). There is no hardware
// BayerNR stand-in: an image that needs the software BayerNR (NR gain
// >= CYPERSTEREO_RKISP_MAX_NR_GAIN, default 3, and no FPGA BayerNR) gets it
// on the CPU, fused with the WB LUT as in the software ISP, and the ISP
// takes that linear RAW with identity BLS/AWB/GOC.
//
// Six-image halves are (cam_width/2) x frame_height RAW, demosaiced
// at that size, then the linear rows are averaged 2:1 and toned at half
// height (IspFrameJob::half_height). They always take the CPU WB above, so
// two of them share one frame side by side (PackGap() reflected columns
// apart, fewer up to the 1920 selfpath width) and the six images run as one
// batch in the full images' geometry, four frames like the quad. A half of
// the other CFA phase is placed one row and column in, so the frames split
// evenly between the two virtual ISPs. YNR for a half uses the software
// post-average NR gain (kHalfHeightPostFrontNrRatio). A batch larger than
// HwBudgetPixels() (AR0234 quad and six) leaves frames to the software ISP,
// which runs them concurrently with the core.
//
// One virtual ISP (rkisp-virN) serves one CFA phase, the first two free ones
// are taken for RGGB and BGGR. They share the hardware core, which runs
// them back to back. An open ISP is kept for any frame that fits it. The
// core runs at 594 MHz instead of the driver's 396 MHz (CoreMhz()).
//
// RK3588 differences (ISP30): two cores (rkisp0-virN, rkisp1-virN; enabled
// with scripts/setup_rk3588_hw_isp.sh). Each core gets one virtual ISP per
// phase and a phase's frames alternate between the cores, so the cores run
// concurrently, each at 600 MHz (two streaming virtual ISPs are needed for
// that step). The selfpath has no RGB output, so BGR is converted on the CPU
// from the mainpath UYVY during the copy out. The ISP30 takes the ISP39
// DPCC/CSM/CGC configs (same registers), the debayer of the imx415 ISO-50
// tuning and no GIC, YNR, CNR or sharpening.
#include "cyperstereo_api.h"

#if defined(CYPERSTEREO_RK_HW_ISP)

#include <fcntl.h>
#include <linux/media-bus-format.h>
#include <linux/media.h>
#include <linux/v4l2-subdev.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef BIT_ULL
#define BIT_ULL(nr) (1ULL << (nr))
#endif
#if defined(CYPERSTEREO_RK3576)
#include "linux/rk-isp39-config.h"
#define CYPERSTEREO_RKISP_V39 1
#else
#include "linux/rk-isp3-config.h"
#endif

namespace cyperstereo {
namespace detail {
namespace {

#include "rkisp/rk_hw_isp_tuning.inc"

#if defined(CYPERSTEREO_RKISP_V39)
using IspParams = isp39_isp_params_cfg;
#define RKISP_MODULE(m) ISP39_MODULE_##m
constexpr int kGammaSamples = ISP39_GAMMA_OUT_MAX_SAMPLES;
constexpr uint64_t kAllModules = (ISP39_MODULE_RGBIR << 1) - 1;
constexpr bool kSelfPathBgr = true;
#else
using IspParams = isp3x_isp_params_cfg;
#define RKISP_MODULE(m) ISP3X_MODULE_##m
constexpr int kGammaSamples = ISP3X_GAMMA_OUT_MAX_SAMPLES;
constexpr uint64_t kAllModules =
    ((ISP3X_MODULE_CAC << 1) - 1) & ~ISP3X_MODULE_FORCE;
constexpr bool kSelfPathBgr = false;
#endif
// Kernels older than the uapi header lack trailing members (the ISP30's
// `exposure`): the driver's buffer only has to hold the module configs.
constexpr size_t kParamsMinSize =
    offsetof(IspParams, others) + sizeof(IspParams::others);

struct Devices {
  std::string subdev, rawrd, params, mainpath, selfpath;
  int core = 0;  // hardware ISP core behind the virtual ISP
};

struct Color {
  double black = 0.0;
  double r_gain = 1.0, g_gain = 1.0, b_gain = 1.0;
  double exposure = 1.0;
  double knee = 0.0;
  int nr_level = 0;          // row of kYnrIso / kCnrIso
  uint16_t nr_gain_q4 = 16;  // YNR/CNR noise-model gain, Q4
};

constexpr int kNrLevels =
    static_cast<int>(sizeof(kYnrIso) / sizeof(kYnrIso[0]));
static_assert(sizeof(kCnrIso) / sizeof(kCnrIso[0]) == kNrLevels,
              "YNR and CNR ISO ladders differ in length");

constexpr uint64_t kColorModules =
    RKISP_MODULE(BLS) | RKISP_MODULE(AWB_GAIN) | RKISP_MODULE(GOC);
#if defined(CYPERSTEREO_RKISP_V39)
constexpr uint64_t kNrModules = ISP39_MODULE_YNR | ISP39_MODULE_CNR;
constexpr uint64_t kEnabledModules =
    kColorModules | kNrModules | ISP39_MODULE_DPCC | ISP39_MODULE_GIC |
    ISP39_MODULE_DEBAYER | ISP39_MODULE_CSM | ISP39_MODULE_CGC |
    ISP39_MODULE_SHARP;
#else
constexpr uint64_t kNrModules = 0;
constexpr uint64_t kEnabledModules =
    kColorModules | ISP3X_MODULE_DPCC | ISP3X_MODULE_DEBAYER |
    ISP3X_MODULE_CSM | ISP3X_MODULE_CGC;
#endif

// 12-bit input of the gamma segments (equ_segm = 0, finalx4_dense_en = 1).
constexpr int kGammaX[kGammaSamples] = {
    0,    1,    2,    3,    4,    5,    6,    7,    8,    10,
    12,   14,   16,   20,   24,   28,   32,   40,   48,   56,
    64,   80,   96,   112,  128,  160,  192,  224,  256,  320,
    384,  448,  512,  640,  768,  896,  1024, 1280, 1536, 1792,
    2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4095};

template <typename T, size_t N>
void LoadConfig(T &dst, const unsigned char (&blob)[N]) {
  static_assert(sizeof(T) == N, "ISP uapi layout differs from the tuning");
  std::memcpy(&dst, blob, N);
}

const IspParams &Template() {
  static const IspParams t = [] {
    IspParams p;
    std::memset(&p, 0, sizeof(p));
#if defined(CYPERSTEREO_RKISP_V39)
    LoadConfig(p.others.dpcc_cfg, kDpcc);
    LoadConfig(p.others.gic_cfg, kGic);
    LoadConfig(p.others.debayer_cfg, kDebayer);
    LoadConfig(p.others.ynr_cfg, kYnrIso[0]);
    LoadConfig(p.others.cnr_cfg, kCnrIso[0]);
    LoadConfig(p.others.sharp_cfg, kSharp);
#else
    // The ISP39 DPCC adds border_bypass_mode as its third byte.
    static_assert(sizeof(p.others.dpcc_cfg) + 1 == sizeof(kDpcc),
                  "ISP30 DPCC layout differs from the ISP39 tuning");
    auto *dpcc = reinterpret_cast<unsigned char *>(&p.others.dpcc_cfg);
    std::memcpy(dpcc, kDpcc, 2);
    std::memcpy(dpcc + 2, kDpcc + 3, sizeof(kDpcc) - 3);
    // imx415 ISO-50 debayer (scene_isp30 "debayer" of the RK3588 IQ).
    static const int8_t filter1[5] = {2, -6, 0, 6, -2};
    static const int8_t filter2[5] = {2, -4, 4, -4, 2};
    auto &d = p.others.debayer_cfg;
    d.filter_c_en = 1;
    d.filter_g_en = 1;
    d.thed0 = 3;
    d.thed1 = 6;
    d.dist_scale = 8;
    d.max_ratio = 4;
    d.clip_en = 1;
    d.filter1_coe1 = filter1[0];
    d.filter1_coe2 = filter1[1];
    d.filter1_coe3 = filter1[2];
    d.filter1_coe4 = filter1[3];
    d.filter1_coe5 = filter1[4];
    d.filter2_coe1 = filter2[0];
    d.filter2_coe2 = filter2[1];
    d.filter2_coe3 = filter2[2];
    d.filter2_coe4 = filter2[3];
    d.filter2_coe5 = filter2[4];
    d.hf_offset = 1;
    d.gain_offset = 4;
    d.offset = 1;
    d.shift_num = 2;
    d.order_max = 10;
    d.order_min = 3;
#endif
    LoadConfig(p.others.csm_cfg, kCsm);
    LoadConfig(p.others.cgc_cfg, kCgc);
    return p;
  }();
  return t;
}

// The GOC input is the 12-bit signal after BLS and AWB, i.e. 16x the 8-bit
// linear WB output of the software ISP before exposure. The table applies
// the exposure and roll-off of its WB LUT.
void BuildGamma(const Color &c, uint16_t *y) {
  const double e = c.exposure;
  const double xm = (255.0 - c.black) * e;
  const bool roll_off = c.knee > 0.0 && xm > 255.0;
  const double s = roll_off ? (xm - c.knee) / (255.0 - c.knee) : 1.0;
  const double white = (std::min)(255.0, xm);
  for (int i = 0; i < kGammaSamples; ++i) {
    double v = kGammaX[i] / 16.0 * e;
    if (roll_off && v > c.knee) {
      const double t = (std::min)(1.0, (v - c.knee) / (xm - c.knee));
      v = c.knee + (255.0 - c.knee) * t * s / (1.0 + (s - 1.0) * t);
    } else {
      v = (std::min)(v, white);
    }
    y[i] = static_cast<uint16_t>(
        (std::max)(0.0, (std::min)(4095.0, std::round(v * 16.0))));
  }
}

void FillParams(IspParams &p, const Color &c, uint32_t id, bool first,
                bool nr_update) {
  p = Template();
  p.frame_id = id;
  p.module_en_update = kAllModules;
  p.module_ens = kEnabledModules;
  p.module_cfg_update =
      first ? (kEnabledModules | RKISP_MODULE(FORCE)) : kColorModules;
  if (nr_update) p.module_cfg_update |= kNrModules;
#if defined(CYPERSTEREO_RKISP_V39)
  LoadConfig(p.others.ynr_cfg, kYnrIso[c.nr_level]);
  LoadConfig(p.others.cnr_cfg, kCnrIso[c.nr_level]);
  p.others.ynr_cfg.global_set_gain = c.nr_gain_q4;
  p.others.cnr_cfg.global_gain = c.nr_gain_q4;
#endif

  auto &bls = p.others.bls_cfg;
  const uint16_t black12 = static_cast<uint16_t>(std::lround(c.black * 16.0));
  bls.fixed_val.r = bls.fixed_val.gr = bls.fixed_val.gb = bls.fixed_val.b =
      black12;
#if defined(CYPERSTEREO_RKISP_V39)
  bls.isp_ob_predgain = 256;
  bls.isp_ob_max = 4095;
#endif

  auto &awb = p.others.awb_gain_cfg;
  const auto q8 = [](double g) {
    return static_cast<uint16_t>(
        (std::max)(1L, (std::min)(65535L, std::lround(g * 256.0))));
  };
#if defined(CYPERSTEREO_RKISP_V39)
  // gain0..2 are the per-exposure HDR gains; awb1 is the white balance.
  awb.gain0_red = awb.gain0_blue = awb.gain0_green_r = awb.gain0_green_b = 256;
  awb.gain1_red = awb.gain1_blue = awb.gain1_green_r = awb.gain1_green_b = 256;
  awb.gain2_red = awb.gain2_blue = awb.gain2_green_r = awb.gain2_green_b = 256;
  awb.awb1_gain_r = q8(c.r_gain);
  awb.awb1_gain_gr = awb.awb1_gain_gb = q8(c.g_gain);
  awb.awb1_gain_b = q8(c.b_gain);
#else
  // ISP30 has no separate white-balance stage: gain0 (and the HDR gains)
  // carry it.
  awb.gain0_red = awb.gain1_red = awb.gain2_red = q8(c.r_gain);
  awb.gain0_green_r = awb.gain1_green_r = awb.gain2_green_r =
      awb.gain0_green_b = awb.gain1_green_b = awb.gain2_green_b =
          q8(c.g_gain);
  awb.gain0_blue = awb.gain1_blue = awb.gain2_blue = q8(c.b_gain);
#endif

  auto &goc = p.others.gammaout_cfg;
  goc.equ_segm = 0;
  goc.finalx4_dense_en = 1;
  goc.offset = 0;
  uint16_t gamma[kGammaSamples];
  BuildGamma(c, gamma);
  std::memcpy(goc.gamma_y, gamma, sizeof(gamma));
}

// Shared-luminance tone of the software output stage on linear BGR.
void ToneRowBgr(const uint8_t *src, uint8_t *dst, int w,
                const FastBalancedToneLuts &luts) {
  int x = 0;
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
  for (; x + 16 <= w; x += 16) {
    uint8x16x3_t v = vld3q_u8(src + 3 * x);
    ApplyFastBalancedTone16Neon(v.val[0], v.val[1], v.val[2], luts);
    vst3q_u8(dst + 3 * x, v);
  }
#endif
  for (; x < w; ++x) {
    uchar b = src[3 * x], g = src[3 * x + 1], r = src[3 * x + 2];
    ApplyFastBalancedTonePixel(b, g, r, luts);
    dst[3 * x] = b;
    dst[3 * x + 1] = g;
    dst[3 * x + 2] = r;
  }
}

// The same tone on full-range BT.601 UYVY of linear BGR. Scaling BGR by a
// gain scales Y and Cb/Cr - 128 alike, so each Y takes its own pixel's gain
// and the pair's chroma the mean of both gains, as the software path
// averages the chroma of the toned pair. max(B, G, R) for the highlight
// clip is reconstructed with the preview decoder's coefficients.
void ToneRowUyvy(const uint8_t *src, uint8_t *dst, int w,
                 const FastBalancedToneLuts &luts) {
  const auto gain = [&](int y, int tb, int tg, int tr) {
    const int m = (std::max)(
        0, (std::min)(255, y + (std::max)(tb, (std::max)(-tg, tr))));
    return (std::min)(static_cast<int>(luts.gain_q12[y]),
                      static_cast<int>(luts.clip_q12[m]));
  };
  int x = 0;
#if defined(CYPERSTEREO_HAVE_NEON) && defined(__aarch64__)
  const int16_t *table = luts.gain_min_q12.data();
  for (; x + 32 <= w; x += 32) {
    uint8x16x4_t v = vld4q_u8(src + 2 * x);  // Cb, Y0, Cr, Y1
    const int16x8_t cb_lo = vreinterpretq_s16_u16(
        vsubl_u8(vget_low_u8(v.val[0]), vdup_n_u8(128)));
    const int16x8_t cb_hi =
        vreinterpretq_s16_u16(vsubl_high_u8(v.val[0], vdupq_n_u8(128)));
    const int16x8_t cr_lo = vreinterpretq_s16_u16(
        vsubl_u8(vget_low_u8(v.val[2]), vdup_n_u8(128)));
    const int16x8_t cr_hi =
        vreinterpretq_s16_u16(vsubl_high_u8(v.val[2], vdupq_n_u8(128)));
    // Largest of the B, G, R offsets from Y: (454 cb, -(183 cr + 88 cb),
    // 359 cr) >> 8.
    const auto offset_max = [](int16x8_t cb, int16x8_t cr) {
      const int32x4_t cb_l = vmovl_s16(vget_low_s16(cb));
      const int32x4_t cb_h = vmovl_high_s16(cb);
      const int32x4_t cr_l = vmovl_s16(vget_low_s16(cr));
      const int32x4_t cr_h = vmovl_high_s16(cr);
      const auto one = [](int32x4_t b, int32x4_t r) {
        const int32x4_t tb = vshrq_n_s32(vmulq_n_s32(b, 454), 8);
        const int32x4_t tr = vshrq_n_s32(vmulq_n_s32(r, 359), 8);
        const int32x4_t tg = vshrq_n_s32(
            vmlaq_n_s32(vmulq_n_s32(r, 183), b, 88), 8);
        return vmaxq_s32(tb, vmaxq_s32(vnegq_s32(tg), tr));
      };
      return vcombine_s16(vmovn_s32(one(cb_l, cr_l)),
                          vmovn_s32(one(cb_h, cr_h)));
    };
    const int16x8_t off_lo = offset_max(cb_lo, cr_lo);
    const int16x8_t off_hi = offset_max(cb_hi, cr_hi);
    int16x8_t gains[2][2];
    for (int k = 0; k < 2; ++k) {
      const uint8x16_t y = v.val[1 + 2 * k];
      const int16x8_t m_lo = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y))), off_lo);
      const int16x8_t m_hi = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_high_u8(y)), off_hi);
      const uint8x16_t m =
          vcombine_u8(vqmovun_s16(m_lo), vqmovun_s16(m_hi));
      const uint16x8_t index_lo = vreinterpretq_u16_u8(vzip1q_u8(y, m));
      const uint16x8_t index_hi = vreinterpretq_u16_u8(vzip2q_u8(y, m));
      gains[k][0] = FastToneGather8Neon(table, index_lo);
      gains[k][1] = FastToneGather8Neon(table, index_hi);
      v.val[1 + 2 * k] = FastToneScale16Neon(y, gains[k][0], gains[k][1]);
    }
    const int16x8_t pair_lo = vreinterpretq_s16_u16(
        vrhaddq_u16(vreinterpretq_u16_s16(gains[0][0]),
                    vreinterpretq_u16_s16(gains[1][0])));
    const int16x8_t pair_hi = vreinterpretq_s16_u16(
        vrhaddq_u16(vreinterpretq_u16_s16(gains[0][1]),
                    vreinterpretq_u16_s16(gains[1][1])));
    // SQRDMULH(8 c, g) = (c g + 2048) >> 12 as in FastToneScale16Neon.
    const auto scale_chroma = [&](int16x8_t c_lo, int16x8_t c_hi) {
      const int16x8_t lo = vaddq_s16(
          vqrdmulhq_s16(vshlq_n_s16(c_lo, 3), pair_lo), vdupq_n_s16(128));
      const int16x8_t hi = vaddq_s16(
          vqrdmulhq_s16(vshlq_n_s16(c_hi, 3), pair_hi), vdupq_n_s16(128));
      return vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi));
    };
    v.val[0] = scale_chroma(cb_lo, cb_hi);
    v.val[2] = scale_chroma(cr_lo, cr_hi);
    vst4q_u8(dst + 2 * x, v);
  }
#endif
  for (; x + 1 < w; x += 2) {
    const uint8_t *s = src + 2 * x;
    uint8_t *d = dst + 2 * x;
    const int dcb = s[0] - 128, dcr = s[2] - 128;
    const int tb = (454 * dcb) >> 8;
    const int tg = (183 * dcr + 88 * dcb) >> 8;
    const int tr = (359 * dcr) >> 8;
    const int g0 = gain(s[1], tb, tg, tr), g1 = gain(s[3], tb, tg, tr);
    const int gp = (g0 + g1 + 1) >> 1;
    const auto scale = [](int c, int g) {
      return (std::max)(-128, (std::min)(127, (c * g + 2048) >> 12));
    };
    d[0] = static_cast<uint8_t>(128 + scale(dcb, gp));
    d[1] = static_cast<uint8_t>((std::min)(255, (s[1] * g0 + 2048) >> 12));
    d[2] = static_cast<uint8_t>(128 + scale(dcr, gp));
    d[3] = static_cast<uint8_t>((std::min)(255, (s[3] * g1 + 2048) >> 12));
  }
}

// Full-range BT.601 UYVY to BGR with the preview decoder's coefficients,
// for ISPs without RGB output paths.
void UyvyRowToBgr(const uint8_t *src, uint8_t *dst, int w) {
  int x = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
  for (; x + 32 <= w; x += 32) {
    const uint8x16x4_t v = vld4q_u8(src + 2 * x);  // Cb, Y0, Cr, Y1
    const auto offsets = [](uint8x8_t cb8, uint8x8_t cr8, int16x8_t &tb,
                            int16x8_t &tg, int16x8_t &tr) {
      const int16x8_t cb =
          vreinterpretq_s16_u16(vsubl_u8(cb8, vdup_n_u8(128)));
      const int16x8_t cr =
          vreinterpretq_s16_u16(vsubl_u8(cr8, vdup_n_u8(128)));
      // (454 cb) >> 8, (183 cr + 88 cb) >> 8, (359 cr) >> 8 in Q15 halves.
      tb = vaddq_s16(cb, vqrdmulhq_n_s16(cb, (454 - 256) * 128));
      tg = vqrdmulhq_n_s16(cr, 183 * 128);
      tg = vaddq_s16(tg, vqrdmulhq_n_s16(cb, 88 * 128));
      tr = vaddq_s16(cr, vqrdmulhq_n_s16(cr, (359 - 256) * 128));
    };
    int16x8_t tb[2], tg[2], tr[2];
    offsets(vget_low_u8(v.val[0]), vget_low_u8(v.val[2]), tb[0], tg[0], tr[0]);
    offsets(vget_high_u8(v.val[0]), vget_high_u8(v.val[2]), tb[1], tg[1],
            tr[1]);
    uint8x16_t b[2], g[2], r[2];  // [even / odd pixel]
    for (int k = 0; k < 2; ++k) {
      const uint8x16_t y8 = v.val[1 + 2 * k];
      const int16x8_t ylo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y8)));
      const int16x8_t yhi = vreinterpretq_s16_u16(vmovl_high_u8(y8));
      b[k] = vcombine_u8(vqmovun_s16(vaddq_s16(ylo, tb[0])),
                         vqmovun_s16(vaddq_s16(yhi, tb[1])));
      g[k] = vcombine_u8(vqmovun_s16(vsubq_s16(ylo, tg[0])),
                         vqmovun_s16(vsubq_s16(yhi, tg[1])));
      r[k] = vcombine_u8(vqmovun_s16(vaddq_s16(ylo, tr[0])),
                         vqmovun_s16(vaddq_s16(yhi, tr[1])));
    }
    uint8x16x3_t o;
    o.val[0] = vzip1q_u8(b[0], b[1]);
    o.val[1] = vzip1q_u8(g[0], g[1]);
    o.val[2] = vzip1q_u8(r[0], r[1]);
    vst3q_u8(dst + 3 * x, o);
    o.val[0] = vzip2q_u8(b[0], b[1]);
    o.val[1] = vzip2q_u8(g[0], g[1]);
    o.val[2] = vzip2q_u8(r[0], r[1]);
    vst3q_u8(dst + 3 * x + 48, o);
  }
#endif
  const auto clamp = [](int v) {
    return static_cast<uint8_t>((std::max)(0, (std::min)(255, v)));
  };
  for (; x + 1 < w; x += 2) {
    const uint8_t *s = src + 2 * x;
    uint8_t *d = dst + 3 * x;
    const int dcb = s[0] - 128, dcr = s[2] - 128;
    const int tb = (454 * dcb + 128) >> 8;
    const int tg = (183 * dcr + 88 * dcb + 128) >> 8;
    const int tr = (359 * dcr + 128) >> 8;
    for (int k = 0; k < 2; ++k) {
      const int y = s[1 + 2 * k];
      d[3 * k] = clamp(y + tb);
      d[3 * k + 1] = clamp(y - tg);
      d[3 * k + 2] = clamp(y + tr);
    }
  }
}

int Xioctl(int fd, unsigned long req, void *arg) {
  int r;
  do r = ioctl(fd, req, arg);
  while (r < 0 && errno == EINTR);
  return r;
}

std::string CharDevicePath(uint32_t major, uint32_t minor) {
  std::ifstream uevent("/sys/dev/char/" + std::to_string(major) + ":" +
                       std::to_string(minor) + "/uevent");
  std::string line;
  while (std::getline(uevent, line))
    if (line.compare(0, 8, "DEVNAME=") == 0) return "/dev/" + line.substr(8);
  return std::string();
}

// Nodes of the media graph whose driver is `driver` (e.g. "rkisp-vir0").
bool FindDevices(const std::string &driver, Devices &d) {
  for (int m = 0; m < 64; ++m) {
    const std::string path = "/dev/media" + std::to_string(m);
    const int fd = open(path.c_str(), O_RDWR);
    if (fd < 0) continue;
    media_device_info info{};
    if (Xioctl(fd, MEDIA_IOC_DEVICE_INFO, &info) < 0 ||
        driver != info.driver) {
      close(fd);
      continue;
    }
    media_entity_desc e{};
    e.id = MEDIA_ENT_ID_FLAG_NEXT;
    uint32_t isp_id = 0, rawrd_id = 0;
    while (Xioctl(fd, MEDIA_IOC_ENUM_ENTITIES, &e) == 0) {
      const std::string name = e.name;
      std::string *slot = name == "rkisp-isp-subdev"     ? &d.subdev
                          : name == "rkisp_rawrd2_s"     ? &d.rawrd
                          : name == "rkisp-input-params" ? &d.params
                          : name == "rkisp_mainpath"     ? &d.mainpath
                          : name == "rkisp_selfpath"     ? &d.selfpath
                                                         : nullptr;
      if (slot) *slot = CharDevicePath(e.dev.major, e.dev.minor);
      if (name == "rkisp-isp-subdev") isp_id = e.id;
      if (name == "rkisp_rawrd2_s") rawrd_id = e.id;
      e.id |= MEDIA_ENT_ID_FLAG_NEXT;
    }
    // Without a sensor the ISP30 leaves its input unlinked; the ISP39 has
    // it linked already, so a failure here is left to STREAMON to report.
    if (isp_id && rawrd_id) {
      media_link_desc link{};
      link.source.entity = rawrd_id;
      link.source.index = 0;
      link.sink.entity = isp_id;
      link.sink.index = 0;
      link.flags = MEDIA_LNK_FL_ENABLED;
      Xioctl(fd, MEDIA_IOC_SETUP_LINK, &link);
    }
    close(fd);
    return !d.subdev.empty() && !d.rawrd.empty() && !d.params.empty() &&
           !d.mainpath.empty() && !d.selfpath.empty();
  }
  return false;
}

class VirtualIsp {
 public:
  VirtualIsp() = default;
  VirtualIsp(const VirtualIsp &) = delete;
  VirtualIsp &operator=(const VirtualIsp &) = delete;
  ~VirtualIsp() { Close(); }

  // Smaller frames are padded, so an open size at or above w x h serves.
  bool IsOpen(int w, int h, IspPixelFormat f) const {
    return out_ >= 0 && w <= w_ && h <= h_ && f == format_;
  }
  int width() const { return w_; }
  int height() const { return h_; }
  IspPixelFormat format() const { return format_; }

  bool Open(const Devices &dev, int w, int h, bool rggb, IspPixelFormat f) {
    Close();
    w_ = w;
    h_ = h;
    format_ = f;
    const bool uyvy = HwUyvy();
    subdev_ = open(dev.subdev.c_str(), O_RDWR);
    rawrd_ = open(dev.rawrd.c_str(), O_RDWR | O_NONBLOCK);
    params_ = open(dev.params.c_str(), O_RDWR | O_NONBLOCK);
    out_ = open((uyvy ? dev.mainpath : dev.selfpath).c_str(),
                O_RDWR | O_NONBLOCK);
    if (subdev_ < 0 || rawrd_ < 0 || params_ < 0 || out_ < 0)
      return Fail("open");
    if (!SetPad(0, rggb ? MEDIA_BUS_FMT_SRGGB8_1X8
                        : MEDIA_BUS_FMT_SBGGR8_1X8) ||
        !SetPad(2, MEDIA_BUS_FMT_YUYV8_2X8))
      return false;
    // The paths default to an 800x600 crop of the ISP output.
    v4l2_selection sel{};
    sel.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    sel.target = V4L2_SEL_TGT_CROP;
    sel.r.width = w_;
    sel.r.height = h_;
    if (Xioctl(out_, VIDIOC_S_SELECTION, &sel) < 0) return Fail("crop");
    if (!SetupImageQueue(out_, kCap,
                         uyvy ? V4L2_PIX_FMT_UYVY : V4L2_PIX_FMT_RGB24, 4,
                         out_bufs_, out_bpl_) ||
        !SetupImageQueue(rawrd_, kOut,
                         rggb ? V4L2_PIX_FMT_SRGGB8 : V4L2_PIX_FMT_SBGGR8, 2,
                         rd_bufs_, rd_bpl_))
      return false;
    if (rd_bpl_ < static_cast<uint32_t>(w_) ||
        out_bpl_ < static_cast<uint32_t>(w_) * (uyvy ? 2 : 3)) {
      errno = EINVAL;
      return Fail("stride");
    }
    if (!SetupParamsQueue()) return false;
    // The first parameter buffer initialises every module and has to be
    // queued before the image streams start.
    if (!QueueParams(Color(), 0, true)) return false;
    for (int i = 0; i < static_cast<int>(out_bufs_.size()); ++i)
      if (!QueueImage(out_, kCap, i, 0)) return false;
    int t = kCap;
    if (Xioctl(out_, VIDIOC_STREAMON, &t) < 0) return Fail("streamon");
    t = kOut;
    if (Xioctl(rawrd_, VIDIOC_STREAMON, &t) < 0) return Fail("rawrd streamon");
    frame_id_ = 1;
    return true;
  }

  // The RAW buffer of the next Submit(): width() x h_ bytes, rd_stride()
  // apart. Its previous frame has been collected.
  uint8_t *RawBuffer() { return static_cast<uint8_t *>(rd_bufs_[rd_next_].ptr); }
  size_t rd_stride() const { return rd_bpl_; }

  // Starts the ISP on RawBuffer(); Collect() returns the frame.
  bool Submit(const Color &c) {
    ReclaimParams();
    const uint32_t id = frame_id_++;
    if (!QueueParams(c, id, false)) return false;
    if (!QueueImage(rawrd_, kOut, rd_next_, rd_bpl_ * h_)) return false;
    rd_next_ = (rd_next_ + 1) % static_cast<int>(rd_bufs_.size());
    isp2x_csi_trigger tg{};
    tg.frame_id = id;
    tg.mode = T_START_X1;
    if (Xioctl(subdev_, RKISP_CMD_TRIGGER_READ_BACK, &tg) < 0)
      return Fail("trigger");
    return true;
  }

  // One image of a frame: the w x h output at (x, y) into `out`.
  // `half` averages each pair of linear rows first (six-image demosaic-first
  // halves): tone then sees the half-height image, as the software path tones
  // after its row average.
  struct Region {
    cv::Mat *out;
    int x, y, w, h;
    bool half;
  };

  // Copies the frame's images out, applying `tone` (nullptr = linear).
  bool Collect(const Region *regions, int n, const FastBalancedToneLuts *tone) {
    const int oi = DequeueImage(out_, kCap, 2000);
    if (oi < 0) return Fail("output timeout");
    const bool uyvy = format_ == IspPixelFormat::kUyvy422Bt601FullRange;
    const bool convert = !uyvy && HwUyvy();
    const int bpp = HwUyvy() ? 2 : 3;
    for (int r = 0; r < n; ++r) {
      const Region &g = regions[r];
      // Selfpath "RGB24" stores B, G, R in memory order.
      const int rows = g.half ? g.h / 2 : g.h;
      g.out->create(rows, g.w, uyvy ? CV_8UC2 : CV_8UC3);
      const uint8_t *src = static_cast<const uint8_t *>(out_bufs_[oi].ptr) +
                           static_cast<size_t>(g.y) * out_bpl_ +
                           static_cast<size_t>(g.x) * bpp;
      const size_t row_bytes = static_cast<size_t>(g.w) * bpp;
      if (g.half) row_.resize(row_bytes);
      for (int y = 0; y < rows; ++y) {
        const uint8_t *s;
        if (!g.half) {
          s = src + static_cast<size_t>(y) * out_bpl_;
        } else {
          const uint8_t *a = src + static_cast<size_t>(2 * y) * out_bpl_;
          const uint8_t *b = src + static_cast<size_t>(2 * y + 1) * out_bpl_;
          int x = 0;
#if defined(__ARM_NEON)
          for (; x + 16 <= static_cast<int>(row_bytes); x += 16)
            vst1q_u8(row_.data() + x,
                     vrhaddq_u8(vld1q_u8(a + x), vld1q_u8(b + x)));
#endif
          for (; x < static_cast<int>(row_bytes); ++x)
            row_[x] = static_cast<uint8_t>((a[x] + b[x] + 1) >> 1);
          s = row_.data();
        }
        uint8_t *d = g.out->ptr(y);
        if (convert) {
          UyvyRowToBgr(s, d, g.w);
          if (tone) ToneRowBgr(d, d, g.w, *tone);
        } else if (!tone) {
          std::memcpy(d, s, row_bytes);
        } else if (uyvy) {
          ToneRowUyvy(s, d, g.w, *tone);
        } else {
          ToneRowBgr(s, d, g.w, *tone);
        }
      }
    }
    if (!QueueImage(out_, kCap, oi, 0)) return false;
    DequeueImage(rawrd_, kOut, 100);
    return true;
  }

  void Close() {
    // Params must go off first. Stopping the capture node waits inside
    // rkisp_stop_3a_run until the params node is already off, and that wait
    // is a 1s timeout when userspace is blocked in the capture STREAMOFF.
    int t = kMeta;
    if (params_ >= 0) Xioctl(params_, VIDIOC_STREAMOFF, &t);
    t = kOut;
    if (rawrd_ >= 0) Xioctl(rawrd_, VIDIOC_STREAMOFF, &t);
    t = kCap;
    if (out_ >= 0) Xioctl(out_, VIDIOC_STREAMOFF, &t);
    for (auto *v : {&out_bufs_, &rd_bufs_, &par_bufs_}) {
      for (auto &b : *v)
        if (b.ptr) munmap(b.ptr, b.len);
      v->clear();
    }
    for (int *fd : {&out_, &rawrd_, &params_, &subdev_}) {
      if (*fd >= 0) close(*fd);
      *fd = -1;
    }
    par_free_.clear();
    rd_next_ = 0;
  }

  const std::string &error() const { return error_; }

 private:
  static constexpr int kCap = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  static constexpr int kOut = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  static constexpr int kMeta = V4L2_BUF_TYPE_META_OUTPUT;

  struct Buf {
    void *ptr = nullptr;
    size_t len = 0;
  };

  bool Fail(const char *what) {
    error_ = std::string(what) + ": " + std::strerror(errno);
    return false;
  }

  // The mainpath UYVY also serves BGR where the selfpath has no RGB.
  bool HwUyvy() const {
    return format_ == IspPixelFormat::kUyvy422Bt601FullRange || !kSelfPathBgr;
  }

 public:
  // Sets a closed virtual ISP's input size, which the driver's clock estimate
  // reads at the next ForceEnum.
  static bool SetInputSize(const Devices &dev, int w, int h) {
    VirtualIsp v;
    v.w_ = w;
    v.h_ = h;
    v.subdev_ = open(dev.subdev.c_str(), O_RDWR);
    const bool ok = v.subdev_ >= 0 && v.SetPad(0, MEDIA_BUS_FMT_SRGGB8_1X8);
    return ok;
  }
  // Re-reads every virtual ISP's input size; only while none is streaming.
  static bool ForceEnum(const Devices &dev) {
    const int fd = open(dev.subdev.c_str(), O_RDWR);
    if (fd < 0) return false;
    const bool ok = Xioctl(fd, RKISP_CMD_MULTI_DEV_FORCE_ENUM, nullptr) == 0;
    close(fd);
    return ok;
  }

 private:
  bool SetPad(uint32_t pad, uint32_t code) {
    v4l2_subdev_format f{};
    f.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    f.pad = pad;
    f.format.width = w_;
    f.format.height = h_;
    f.format.code = code;
    f.format.field = V4L2_FIELD_NONE;
    if (pad == 2) {
      f.format.colorspace = V4L2_COLORSPACE_SMPTE170M;
      f.format.quantization = V4L2_QUANTIZATION_FULL_RANGE;
    }
    if (Xioctl(subdev_, VIDIOC_SUBDEV_S_FMT, &f) < 0) return Fail("pad fmt");
    v4l2_subdev_selection s{};
    s.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    s.pad = pad;
    s.target = V4L2_SEL_TGT_CROP;
    s.r.width = w_;
    s.r.height = h_;
    if (Xioctl(subdev_, VIDIOC_SUBDEV_S_SELECTION, &s) < 0)
      return Fail("pad crop");
    return true;
  }

  bool MapBuffers(int fd, int type, int count, std::vector<Buf> &bufs,
                  bool mplane) {
    v4l2_requestbuffers req{};
    req.count = count;
    req.type = type;
    req.memory = V4L2_MEMORY_MMAP;
    if (Xioctl(fd, VIDIOC_REQBUFS, &req) < 0) return Fail("reqbufs");
    bufs.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
      v4l2_plane plane{};
      v4l2_buffer b{};
      b.type = type;
      b.memory = V4L2_MEMORY_MMAP;
      b.index = i;
      if (mplane) {
        b.m.planes = &plane;
        b.length = 1;
      }
      if (Xioctl(fd, VIDIOC_QUERYBUF, &b) < 0) return Fail("querybuf");
      const size_t len = mplane ? plane.length : b.length;
      const off_t off = mplane ? plane.m.mem_offset : b.m.offset;
      void *p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, off);
      if (p == MAP_FAILED) return Fail("mmap");
      bufs[i].ptr = p;
      bufs[i].len = len;
    }
    return true;
  }

  bool SetupImageQueue(int fd, int type, uint32_t fourcc, int count,
                       std::vector<Buf> &bufs, uint32_t &bpl) {
    v4l2_format fmt{};
    fmt.type = type;
    fmt.fmt.pix_mp.width = w_;
    fmt.fmt.pix_mp.height = h_;
    fmt.fmt.pix_mp.pixelformat = fourcc;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;
    if (fourcc == V4L2_PIX_FMT_UYVY) {
      fmt.fmt.pix_mp.colorspace = V4L2_COLORSPACE_SMPTE170M;
      fmt.fmt.pix_mp.quantization = V4L2_QUANTIZATION_FULL_RANGE;
    }
    if (Xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) return Fail("s_fmt");
    if (fmt.fmt.pix_mp.width != static_cast<uint32_t>(w_) ||
        fmt.fmt.pix_mp.height != static_cast<uint32_t>(h_)) {
      errno = EINVAL;
      return Fail("s_fmt size");
    }
    bpl = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
    return MapBuffers(fd, type, count, bufs, true);
  }

  bool SetupParamsQueue() {
    if (!MapBuffers(params_, kMeta, 4, par_bufs_, false)) return false;
    for (auto &b : par_bufs_)
      if (b.len < kParamsMinSize) {
        errno = EINVAL;
        return Fail("params size");
      }
    int t = kMeta;
    if (Xioctl(params_, VIDIOC_STREAMON, &t) < 0)
      return Fail("params streamon");
    for (int i = 0; i < static_cast<int>(par_bufs_.size()); ++i)
      par_free_.push_back(i);
    return true;
  }

  bool QueueImage(int fd, int type, int index, uint32_t bytesused) {
    v4l2_plane plane{};
    plane.bytesused = bytesused;
    v4l2_buffer b{};
    b.type = type;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = index;
    b.m.planes = &plane;
    b.length = 1;
    // The CPU only reads capture buffers, so they hold no dirty lines.
    if (type == kCap) b.flags |= V4L2_BUF_FLAG_NO_CACHE_CLEAN;
    if (Xioctl(fd, VIDIOC_QBUF, &b) < 0) return Fail("qbuf");
    return true;
  }

  int DequeueImage(int fd, int type, int timeout_ms) {
    pollfd p{fd, POLLIN | POLLOUT | POLLPRI, 0};
    if (poll(&p, 1, timeout_ms) <= 0) return -1;
    v4l2_plane plane{};
    v4l2_buffer b{};
    b.type = type;
    b.memory = V4L2_MEMORY_MMAP;
    b.m.planes = &plane;
    b.length = 1;
    if (Xioctl(fd, VIDIOC_DQBUF, &b) < 0) return -1;
    return b.index;
  }

  void ReclaimParams() {
    for (;;) {
      v4l2_buffer b{};
      b.type = kMeta;
      b.memory = V4L2_MEMORY_MMAP;
      if (Xioctl(params_, VIDIOC_DQBUF, &b) < 0) break;
      par_free_.push_back(b.index);
    }
  }

  bool QueueParams(const Color &c, uint32_t id, bool first) {
    if (par_free_.empty()) {
      pollfd p{params_, POLLOUT, 0};
      poll(&p, 1, 100);
      ReclaimParams();
      if (par_free_.empty()) {
        errno = EBUSY;
        return Fail("params buffers");
      }
    }
    const int i = par_free_.back();
    par_free_.pop_back();
    // Built off the mapping, which may be shorter than IspParams.
    FillParams(params_scratch_, c, id, first,
               first || c.nr_level != nr_level_ || c.nr_gain_q4 != nr_gain_q4_);
    const size_t size = (std::min)(sizeof(IspParams), par_bufs_[i].len);
    std::memcpy(par_bufs_[i].ptr, &params_scratch_, size);
    nr_level_ = c.nr_level;
    nr_gain_q4_ = c.nr_gain_q4;
    v4l2_buffer b{};
    b.type = kMeta;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    b.bytesused = static_cast<uint32_t>(size);
    if (Xioctl(params_, VIDIOC_QBUF, &b) < 0) return Fail("params qbuf");
    return true;
  }

  int w_ = 0, h_ = 0;
  IspPixelFormat format_ = IspPixelFormat::kBgr888;
  int subdev_ = -1, rawrd_ = -1, params_ = -1, out_ = -1;
  std::vector<Buf> out_bufs_, rd_bufs_, par_bufs_;
  std::vector<int> par_free_;
  uint32_t out_bpl_ = 0, rd_bpl_ = 0;
  int rd_next_ = 0;
  uint32_t frame_id_ = 0;
  std::vector<uint8_t> row_;  // averaged row for half-height output
  IspParams params_scratch_;
  // YNR/CNR state of the last queued parameters.
  int nr_level_ = 0;
  uint16_t nr_gain_q4_ = 16;
  std::string error_;
};

// Persistent thread pinned to `cpu` running one task at a time.
class Worker {
 public:
  explicit Worker(int cpu) : thread_([this, cpu] { Loop(cpu); }) {}
  ~Worker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      quit_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  void Start(std::function<bool()> task) {
    std::lock_guard<std::mutex> lock(mutex_);
    task_ = std::move(task);
    busy_ = true;
    cv_.notify_all();
  }
  bool Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !busy_; });
    return result_;
  }

 private:
  void Loop(int cpu) {
    ScopedThreadAffinity pin(cpu);
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      cv_.wait(lock, [&] { return quit_ || (busy_ && task_); });
      if (quit_) return;
      std::function<bool()> task = std::move(task_);
      task_ = nullptr;
      lock.unlock();
      const bool result = task();
      lock.lock();
      result_ = result;
      busy_ = false;
      cv_.notify_all();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::function<bool()> task_;
  bool busy_ = false, quit_ = false, result_ = false;
  std::thread thread_;
};

double MaxNrGain() {
  static const double v = [] {
    const char *e = std::getenv("CYPERSTEREO_RKISP_MAX_NR_GAIN");
    const double g = e ? std::atof(e) : 3.0;
    return g > 0.0 ? g : 3.0;
  }();
  return v;
}

// YNR/CNR for NR gain g (sensor gain x WhiteBalance::NrGainScale): row
// round(log2 g) of the ISO ladder, and noise-model gain
// 16 * max(1, g * CYPERSTEREO_RKISP_NR_GAIN_SCALE) in Q4. The ladder was
// captured at the FakeCamera's 1x, which left gain-10 walls about 3x noisier
// than the software ISP. YNR trades wall grain against fine texture with no
// selectivity: at gain 10, 0.4 halves the wall noise for ~15% of the woven
// mat's detail, 0.6 matches the software walls for ~30%. Gains up to 1/scale
// keep the ISO-50 template.
void SelectNr(double nr_gain, Color &c) {
  static const double scale = [] {
    const char *e = std::getenv("CYPERSTEREO_RKISP_NR_GAIN_SCALE");
    const double s = e ? std::atof(e) : 0.4;
    return s > 0.0 ? s : 0.4;
  }();
  const double g = (std::max)(1.0, nr_gain);
  c.nr_level = (std::min)(kNrLevels - 1,
                          static_cast<int>(std::lround(std::log2(g))));
  c.nr_gain_q4 = static_cast<uint16_t>(
      std::lround(16.0 * (std::min)(63.0, (std::max)(1.0, g * scale))));
}

// On by default in RK3576 builds; CYPERSTEREO_ISP_BACKEND=sw keeps every
// batch on the software ISP.
std::atomic<bool> backend_requested{[] {
  const char *e = std::getenv("CYPERSTEREO_ISP_BACKEND");
  return !(e && (std::strcmp(e, "sw") == 0 || std::strcmp(e, "software") == 0 ||
                 std::strcmp(e, "cpu") == 0));
}()};
std::atomic<bool> last_batch_on_hardware{false};

// Half images are packed two to a frame, PackGap() reflected columns apart,
// so the six-image halves share the full images' geometry: 2 x 640 + 32
// against 1280 opens the ISP once at 1312 and full frames are padded. Past
// 16 columns the gap no longer changes the halves' edges.
// CYPERSTEREO_RKISP_PACK_GAP sets the gap (multiple of 4, up to 256),
// CYPERSTEREO_RKISP_PACK=0 gives every half its own padded frame.
int PackGap() {
  static const int v = [] {
    const char *e = std::getenv("CYPERSTEREO_RKISP_PACK_GAP");
    const int g = e ? std::atoi(e) : 32;
    return (std::max)(0, (std::min)(256, g)) & ~3;
  }();
  return v;
}

bool PackHalves() {
  static const bool v = [] {
    const char *e = std::getenv("CYPERSTEREO_RKISP_PACK");
    return !(e && std::strcmp(e, "0") == 0);
  }();
  return v;
}

// ISP core clock to ask the driver for, default 594 MHz; 0 leaves the
// driver's choice (396 MHz for these sizes), 702 is the top step. With rawrd
// input the driver sizes the clock from the streaming virtual ISPs, (sum of
// their enumerated w x h) x 30 fps x (1 + 0.25 per ISP), against
// 400/400/500/600/702 MHz steps (594 = GPLL / 2, 702 = SPLL); ApplyBatch
// makes the enumerated sizes large enough for the requested step. 702 is no
// faster for these batches. CYPERSTEREO_RKISP_CORE_MHZ.
int CoreMhz() {
  static const int v = [] {
    const char *e = std::getenv("CYPERSTEREO_RKISP_CORE_MHZ");
    const int m = e ? std::atoi(e) : 594;
    return m >= 700 ? 702 : m >= 590 ? 594 : 0;
  }();
  return v;
}

// Hardware frame pixels per batch (frames x padded width x height). The core
// at 396 MHz takes ~3.7 ms per megapixel, so the default 6.4 MP is ~24 ms
// and leaves the live stream room within 33 ms. SC136HGS batches (4 x
// 1312 x 1026) fit; AR0234's (4 x 1920 x 1080) do not. A faster core
// (CoreMhz()) keeps the same budget: the halves' CPU front end, not the core,
// then bounds an all-hardware batch, so the offload still shortens it.
// The budget is per core (RK3588 has two). CYPERSTEREO_RKISP_HW_BUDGET_MP
// sets it, 0 = no limit.
double HwBudgetPixels() {
  static const double v = [] {
    const char *e = std::getenv("CYPERSTEREO_RKISP_HW_BUDGET_MP");
    const double mp = e ? std::atof(e) : 6.4;
    return mp > 0.0 ? mp * 1e6 : 1e18;
  }();
  return v;
}

// Column of a w-wide image that stands in for column c outside it: mirror
// about the edge pixel with the even period 2w - 2, which keeps the CFA phase.
int ReflectColumn(int c, int w) {
  const int p = 2 * w - 2;
  int m = c % p;
  if (m < 0) m += p;
  return m < w ? m : p - m;
}

constexpr int kMaxBatch = 8;
constexpr int kSelfPathMaxWidth = 1920;
#if defined(CYPERSTEREO_RKISP_V39)
constexpr int kMainPathMaxWidth = 4096;
#else
// Per virtual ISP of a core with three (the overlay's): wider inputs make the
// driver read every frame back twice.
constexpr int kMainPathMaxWidth = 2560;
#endif
// Virtual ISPs driven at once: one per CFA phase and core (lane = 2 x core
// slot + phase).
constexpr int kLanes = 4;

// Images of one hardware frame, side by side. A half image whose CFA phase
// is not the frame's sits one row and one column in (`shift`), which turns
// RGGB into BGGR and back; frames are then two rows taller.
struct Frame {
  int job[2];
  bool shift[2] = {false, false};
  int x[2] = {0, 0}, y[2] = {0, 0};
  int count = 0;
};

bool ApplyBatch(const FastBalancedIspDispatchJob *jobs, int n) {
  static std::mutex mutex;
  static VirtualIsp isp[kLanes];  // even RGGB, odd BGGR
  static int bound[kLanes] = {-1, -1, -1, -1};  // index into `virtual_isps`
  static std::vector<Devices> virtual_isps;
  static std::vector<int> cores;  // distinct Devices::core, one per slot
  static bool disabled = false;
  std::lock_guard<std::mutex> lock(mutex);
  if (disabled || n <= 0 || n > kMaxBatch) return false;
  // RK3576: rkisp-virN on one core. RK3588: rkisp0-virN and rkisp1-virN.
  if (virtual_isps.empty()) {
    for (int v = 0; v < 8; ++v) {
      bool found = false;
      for (const char *prefix : {"rkisp", "rkisp0", "rkisp1"}) {
        Devices d;
        if (!FindDevices(prefix + std::string("-vir") + std::to_string(v), d))
          continue;
        d.core = std::strcmp(prefix, "rkisp1") == 0 ? 1 : 0;
        if (std::find(cores.begin(), cores.end(), d.core) == cores.end())
          cores.push_back(d.core);
        virtual_isps.push_back(d);
        found = true;
      }
      if (!found) break;
    }
    if (cores.empty()) cores.push_back(0);
  }
  const int slots = (std::min)(kLanes / 2, static_cast<int>(cores.size()));
  // Full images share one width, half images another; all share the height.
  int full_w = 0, half_w = 0;
  const int h = jobs[0].raw->rows;
  bool cpu_front[kMaxBatch], bayer_nr[kMaxBatch];
  for (int i = 0; i < n; ++i) {
    const FastBalancedIspDispatchJob &j = jobs[i];
    if (j.raw->type() != CV_8UC1 || j.raw->cols < 4 || (j.raw->cols & 1) ||
        j.raw->rows != h || (h & 1) || h < 2 ||
        j.output_format != jobs[0].output_format)
      return false;
    int &w = j.half_height ? half_w : full_w;
    if (w && w != j.raw->cols) return false;
    w = j.raw->cols;
    // No hardware BayerNR stand-in: the software one runs on the CPU, fused
    // with the WB LUT, and the ISP takes that linear RAW with an identity
    // BLS/AWB/GOC. Half images always take that route so two cameras can
    // share a frame.
    bayer_nr[i] = FastBalancedBayerNrEnabled() && !j.raw_bayer_nr_done &&
                  j.sensor_gain * j.wb->NrGainScale() >= MaxNrGain();
    cpu_front[i] = bayer_nr[i] || j.half_height;
  }
  // The selfpath (BGR) takes frames up to 1920 wide: AR0234's 960-wide halves
  // pack only without a gap.
  const bool selfpath =
      kSelfPathBgr && jobs[0].output_format == IspPixelFormat::kBgr888;
  const int max_w = selfpath ? kSelfPathMaxWidth : kMainPathMaxWidth;
  if (full_w > max_w || half_w > max_w) return false;
  const bool pack = PackHalves() && 2 * half_w <= max_w;
  const int gap = pack ? (std::min)(PackGap(), (max_w - 2 * half_w) & ~3) : 0;
  const int w = (std::max)(full_w, half_w ? (pack ? 2 * half_w + gap : half_w)
                                          : 0);
  // Frames per phase: full images alone, half images in pairs. A shifted
  // UYVY image would pair chroma across two of its columns, so only selfpath
  // BGR shifts.
  const auto phase = [&](int i) {
    return jobs[i].bayer == BayerConversion::kColorBayerBg2Bgr ? 0 : 1;
  };
  const bool can_shift = pack && gap >= 4 && selfpath;
  std::vector<Frame> by_phase[2];
  std::vector<int> halves[2];
  for (int i = 0; i < n; ++i) {
    if (jobs[i].half_height) {
      halves[phase(i)].push_back(i);
      continue;
    }
    Frame f;
    f.job[0] = i;
    f.count = 1;
    by_phase[phase(i)].push_back(f);
  }
  if (can_shift) {
    // A frame has one YNR/CNR strength, so pair the halves of nearest NR
    // gain, then even out the frames of the two virtual ISPs (the core runs
    // them back to back).
    std::vector<int> l(halves[0]);
    l.insert(l.end(), halves[1].begin(), halves[1].end());
    std::stable_sort(l.begin(), l.end(), [&](int a, int b) {
      return jobs[a].sensor_gain * jobs[a].wb->NrGainScale() <
             jobs[b].sensor_gain * jobs[b].wb->NrGainScale();
    });
    for (size_t i = 0; i < l.size(); i += 2) {
      Frame f;
      f.job[0] = l[i];
      f.count = i + 1 < l.size() ? 2 : 1;
      if (f.count == 2) f.job[1] = l[i + 1];
      const int native = phase(f.job[0]);
      const int k = by_phase[native].size() <= by_phase[1 - native].size()
                        ? native : 1 - native;
      for (int s = 0; s < f.count; ++s) f.shift[s] = phase(f.job[s]) != k;
      by_phase[k].push_back(f);
    }
  } else {
    for (int k = 0; k < 2; ++k) {
      const std::vector<int> &l = halves[k];
      for (size_t i = 0; i < l.size(); i += pack ? 2 : 1) {
        Frame f;
        f.job[0] = l[i];
        f.count = pack && i + 1 < l.size() ? 2 : 1;
        if (f.count == 2) f.job[1] = l[i + 1];
        by_phase[k].push_back(f);
      }
    }
  }
  // A phase's frames alternate between the cores.
  std::vector<Frame> frames[kLanes];
  for (int k = 0; k < 2; ++k)
    for (size_t i = 0; i < by_phase[k].size(); ++i)
      frames[2 * (i % slots) + k].push_back(by_phase[k][i]);
  // Past the budget (per core) the cores alone would not keep 30 fps
  // (AR0234 on one ISP39: four 1920x1080 frames take ~31 ms): the software
  // ISP takes frames off the busiest virtual ISP, half pairs first, on
  // CPU6/7 while the cores run.
  std::vector<int> software;
  {
    const double budget = HwBudgetPixels() * slots;
    const double frame_px = static_cast<double>(w) * (h + 2);
    for (;;) {
      size_t total = 0;
      int k = 0;
      for (int l = 0; l < kLanes; ++l) {
        total += frames[l].size();
        if (frames[l].size() > frames[k].size()) k = l;
      }
      if (total <= 1 || total * frame_px <= budget) break;
      std::vector<Frame> &fr = frames[k];
      size_t pick = fr.size() - 1;
      for (size_t i = 0; i < fr.size(); ++i)
        if (fr[i].count == 2) pick = i;
      for (int s = 0; s < fr[pick].count; ++s)
        software.push_back(fr[pick].job[s]);
      fr.erase(fr.begin() + static_cast<std::ptrdiff_t>(pick));
    }
  }
  bool shifted = false;
  for (std::vector<Frame> &fr : frames)
    for (Frame &f : fr)
      for (int s = 0; s < f.count; ++s) {
        f.x[s] = (s ? half_w + gap : 0) + (f.shift[s] ? (s ? -1 : 1) : 0);
        f.y[s] = f.shift[s] ? 1 : 0;
        shifted |= f.shift[s];
      }
  const int frame_h = h + (shifted ? 2 : 0);
  const auto give_up = [](const char *what, const std::string &why) {
    std::fprintf(stderr,
                 "[rkisp] %s%s%s; using the software ISP from now on\n", what,
                 why.empty() ? "" : ": ", why.c_str());
    for (VirtualIsp &v : isp) v.Close();
    disabled = true;
    return false;
  };
  // Statistics and the RAW and output copies are CPU work: keep them on the
  // big cores (A72 on RK3576, A76 on RK3588) like the software path's caller.
  ScopedThreadAffinity caller_affinity(4);
  const auto any_open = [&] {
    for (const VirtualIsp &v : isp)
      if (v.IsOpen(2, 2, v.format())) return true;
    return false;
  };
  // CoreMhz(): the driver sizes the clock from the input sizes it last
  // enumerated, which it re-reads only on RKISP_CMD_MULTI_DEV_FORCE_ENUM (or
  // when the ISP powers up) and only while no virtual ISP streams. Before the
  // first open every virtual ISP is given a large input size and enumerated;
  // Open() then sets the real size, which the estimate does not see.
  if (CoreMhz() && !any_open()) {
#if defined(CYPERSTEREO_RKISP_V39)
    const int phases = (std::max)(1, !frames[0].empty() + !frames[1].empty());
    const double target = CoreMhz() == 702 ? 800.0 : 550.0;
    const double each = target / (1.0 + 0.25 * phases) / 30.0 * 1e6 / phases;
    const int pw = 3840;
    const int rows = (static_cast<int>(std::ceil(each / pw)) + 1) & ~1;
    const int ph = (std::max)(frame_h, (std::min)(3504, rows));
    for (const Devices &d : virtual_isps)
      VirtualIsp::SetInputSize(d, (std::max)(pw, w), ph);
#else
    // Without the double read-back a core's three virtual ISPs may be one
    // 3840x2160 and two 2560x1536 inputs. Its first two stream, so the
    // estimate is (3840x2160 + 2560x1536) x 30 x 1.5 = 550 -> 600 MHz.
    bool first[2] = {true, true};
    for (const Devices &d : virtual_isps) {
      const bool big = first[d.core];
      first[d.core] = false;
      VirtualIsp::SetInputSize(d, big ? 3840 : 2560, big ? 2160 : 1536);
    }
#endif
    bool enumerated[2] = {false, false};  // per core
    for (const Devices &d : virtual_isps) {
      if (enumerated[d.core]) continue;
      enumerated[d.core] = true;
      if (!VirtualIsp::ForceEnum(d))
        std::fprintf(stderr,
                     "[rkisp] ISP clock left to the driver: sizes not re-read "
                     "(%s; another ISP client is streaming)\n",
                     std::strerror(errno));
    }
  }
  // An open ISP at least as wide is kept (narrower frames are padded), so a
  // six-image stream opens once. The virtual ISPs must be reconfigured one
  // after the other: stopping two at once deadlocks in rkisp_stop_3a_run.
  for (int l = 0; l < kLanes; ++l) {
    if (frames[l].empty() || isp[l].IsOpen(w, frame_h, jobs[0].output_format))
      continue;
    // A virtual ISP held by another client (e.g. rkaiq_3A waiting for a
    // sensor) refuses the formats with EBUSY; try the next one of the core.
    std::string why = "no rkisp-vir media device";
    bool opened = false;
    bound[l] = -1;  // Open() closes it
    for (int v = 0; v < static_cast<int>(virtual_isps.size()) && !opened;
         ++v) {
      if (virtual_isps[v].core != cores[l / 2] ||
          std::find(bound, bound + kLanes, v) != bound + kLanes)
        continue;
      opened = isp[l].Open(virtual_isps[v], w, frame_h, (l & 1) == 0,
                           jobs[0].output_format);
      if (opened)
        bound[l] = v;
      else
        why = "rkisp-vir" + std::to_string(v) + " " + isp[l].error();
    }
    if (!opened) return give_up("no usable virtual ISP", why);
  }
  // Lane 0 runs on the caller (CPU4), lane l on CPU 4 + l.
  static std::unique_ptr<Worker> lane_worker[kLanes];
  for (int l = 1; l < kLanes; ++l)
    if (!frames[l].empty() && !lane_worker[l])
      lane_worker[l].reset(new Worker(4 + l));

  const bool apply_tone = FastBalancedGammaEnabled();
  const bool exposure = apply_tone && GetFastExposureConfig().enabled;
  const FastBalancedToneLuts *tone =
      apply_tone ? &GetFastBalancedToneLuts() : nullptr;

  // Writes frame f into the phase's RAW buffer and starts the ISP on it.
  const auto submit = [&](VirtualIsp &v, const Frame &f) {
    uint8_t *buf = v.RawBuffer();
    const size_t stride = v.rd_stride();
    const int fw = v.width(), fh = v.height();
    Color c;
    double nr_sum = 0.0;
    for (int s = 0; s < f.count; ++s) {
      const FastBalancedIspDispatchJob &j = jobs[f.job[s]];
      WhiteBalance &wb = *j.wb;
      wb.PrepareRawMetadata(*j.raw, j.metadata_region);
      wb.SetDisplayExposure(exposure, j.sensor_gain);
      const double nr_gain = j.sensor_gain * wb.NrGainScale();
      // Half-height YNR matches the software post-average NR gain, not the
      // tall-RAW BayerNR gain.
      nr_sum += j.half_height
                    ? (std::max)(1.0, nr_gain * kHalfHeightPostFrontNrRatio)
                    : nr_gain;
      uint8_t *dst = buf + static_cast<size_t>(f.y[s]) * stride + f.x[s];
      const int cols = j.raw->cols;
      if (cpu_front[f.job[s]]) {
        cv::Mat linear(h, cols, CV_8UC1, dst, stride);
        if (bayer_nr[f.job[s]])
          wb.ApplyWithBayerNr(*j.raw, linear, nr_gain, j.bayer);
        else
          wb.ApplyTo(*j.raw, linear, j.bayer);
      } else {
        const WhiteBalance::LinearStage st =
            wb.PrepareLinearStage(*j.raw, j.bayer);
        c.black = st.black;
        c.r_gain = st.r_gain;
        c.g_gain = st.g_gain;
        c.b_gain = st.b_gain;
        c.exposure = st.exposure;
        c.knee = st.knee;
        for (int y = 0; y < h; ++y)
          std::memcpy(dst + static_cast<size_t>(y) * stride,
                      wb.RawRow(*j.raw, y), cols);
      }
      // Rows above and below the image reflect it.
      for (int y = 0; y < fh; ++y) {
        if (y >= f.y[s] && y < f.y[s] + h) continue;
        std::memcpy(buf + static_cast<size_t>(y) * stride + f.x[s],
                    dst + static_cast<size_t>(ReflectColumn(y - f.y[s], h)) *
                              stride,
                    cols);
      }
    }
    // Columns outside the images reflect the nearest one.
    int end[2];
    for (int s = 0; s < f.count; ++s) end[s] = f.x[s] + jobs[f.job[s]].raw->cols;
    std::vector<int> pad;  // (column, source column) pairs
    for (int x = 0; x < fw; ++x) {
      int best = -1, dist = 0;
      for (int s = 0; s < f.count; ++s) {
        const int d = x < f.x[s] ? f.x[s] - x : x >= end[s] ? x - end[s] + 1 : 0;
        if (best < 0 || d < dist) best = s, dist = d;
      }
      if (dist == 0) continue;
      pad.push_back(x);
      pad.push_back(f.x[best] + ReflectColumn(x - f.x[best],
                                              end[best] - f.x[best]));
    }
    for (int y = 0; y < fh && !pad.empty(); ++y) {
      uint8_t *row = buf + static_cast<size_t>(y) * stride;
      for (size_t p = 0; p < pad.size(); p += 2) row[pad[p]] = row[pad[p + 1]];
    }
    SelectNr(nr_sum / f.count, c);
    return v.Submit(c);
  };
  const auto collect = [&](VirtualIsp &v, const Frame &f) {
    VirtualIsp::Region r[2];
    for (int s = 0; s < f.count; ++s) {
      const FastBalancedIspDispatchJob &j = jobs[f.job[s]];
      r[s] = {j.output, f.x[s], f.y[s], j.raw->cols, h, j.half_height};
    }
    return v.Collect(r, f.count, tone);
  };
  // Each virtual ISP is driven by its own thread with up to two frames in
  // flight, so one frame's tone/copy overlaps the core working on another.
  const auto run = [&](int k) {
    const std::vector<Frame> &fr = frames[k];
    for (size_t i = 0; i < fr.size(); ++i) {
      if (i >= 2 && !collect(isp[k], fr[i - 2])) return false;
      if (!submit(isp[k], fr[i])) return false;
    }
    for (size_t i = fr.size() > 2 ? fr.size() - 2 : 0; i < fr.size(); ++i)
      if (!collect(isp[k], fr[i])) return false;
    return true;
  };
  static Worker software_worker6(6), software_worker7(7);
  Worker *const software_worker[2] = {&software_worker6, &software_worker7};
  const int software_lanes = (std::min)(2, static_cast<int>(software.size()));
  for (int l = 0; l < software_lanes; ++l)
    software_worker[l]->Start([&, l] {
      // One job per core: no intra-frame stage threads on the ISP's CPUs.
      GetFastIspStagePool().BeginMultiCameraBatch();
      bool ok = true;
      try {
        for (size_t i = l; i < software.size(); i += 2) {
          const FastBalancedIspDispatchJob &j = jobs[software[i]];
          ApplyFastBalancedISPImpl(*j.raw, *j.output, *j.wb, j.name,
                                   j.sensor_gain, j.bayer, j.output_format,
                                   j.raw_bayer_nr_done, j.metadata_region,
                                   nullptr, j.half_height);
        }
      } catch (...) {
        ok = false;
      }
      GetFastIspStagePool().EndMultiCameraBatch();
      return ok;
    });
  for (int l = 1; l < kLanes; ++l)
    if (!frames[l].empty()) lane_worker[l]->Start([&, l] { return run(l); });
  bool ok[kLanes];
  ok[0] = run(0);
  for (int l = 1; l < kLanes; ++l)
    ok[l] = frames[l].empty() || lane_worker[l]->Wait();
  bool software_ok = true;
  for (int l = 0; l < software_lanes; ++l)
    software_ok &= software_worker[l]->Wait();
  if (!software_ok) return false;
  for (int l = 0; l < kLanes; ++l)
    if (!ok[l])
      return give_up((l & 1) ? "rkisp BGGR" : "rkisp RGGB", isp[l].error());
  return true;
}

}  // namespace

bool RkHwIspBackendRequested() {
  const bool on = backend_requested.load(std::memory_order_relaxed);
  if (!on) last_batch_on_hardware.store(false, std::memory_order_relaxed);
  return on;
}

void SetRkHwIspBackendRequested(bool on) {
  backend_requested.store(on, std::memory_order_relaxed);
}

bool RkHwIspLastBatchOnHardware() {
  return last_batch_on_hardware.load(std::memory_order_relaxed);
}

bool ApplyRkHwIspParallel(const FastBalancedIspDispatchJob *jobs, int n) {
  const bool ok = ApplyBatch(jobs, n);
  last_batch_on_hardware.store(ok, std::memory_order_relaxed);
  return ok;
}

}  // namespace detail
}  // namespace cyperstereo

#endif  // CYPERSTEREO_RK_HW_ISP
