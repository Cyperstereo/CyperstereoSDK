// RK3576 ISP39 backend for the fast-balanced ISP.
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
// BayerNR stand-in: batches that would need the software BayerNR (NR gain
// >= CYPERSTEREO_RKISP_MAX_NR_GAIN, default 3, and no FPGA BayerNR) stay on
// the CPU.
//
// One virtual ISP (rkisp-virN) serves one CFA phase, the first two free ones
// are taken for RGGB and BGGR. They share the hardware core, which runs
// them back to back.
#include "cyperstereo_api.h"

#if defined(CYPERSTEREO_RK3576) && defined(__linux__)

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
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef BIT_ULL
#define BIT_ULL(nr) (1ULL << (nr))
#endif
#include "linux/rk-isp39-config.h"

namespace cyperstereo {
namespace detail {
namespace {

#include "rkisp/rk_hw_isp_tuning.inc"

struct Devices {
  std::string subdev, rawrd, params, mainpath, selfpath;
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
    ISP39_MODULE_BLS | ISP39_MODULE_AWB_GAIN | ISP39_MODULE_GOC;
constexpr uint64_t kEnabledModules =
    kColorModules | ISP39_MODULE_DPCC | ISP39_MODULE_GIC |
    ISP39_MODULE_DEBAYER | ISP39_MODULE_CSM | ISP39_MODULE_CGC |
    ISP39_MODULE_YNR | ISP39_MODULE_CNR | ISP39_MODULE_SHARP;

// 12-bit input of the gamma segments (equ_segm = 0, finalx4_dense_en = 1).
constexpr int kGammaX[ISP39_GAMMA_OUT_MAX_SAMPLES] = {
    0,    1,    2,    3,    4,    5,    6,    7,    8,    10,
    12,   14,   16,   20,   24,   28,   32,   40,   48,   56,
    64,   80,   96,   112,  128,  160,  192,  224,  256,  320,
    384,  448,  512,  640,  768,  896,  1024, 1280, 1536, 1792,
    2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4095};

template <typename T, size_t N>
void LoadConfig(T &dst, const unsigned char (&blob)[N]) {
  static_assert(sizeof(T) == N, "ISP39 uapi layout differs from the tuning");
  std::memcpy(&dst, blob, N);
}

const isp39_isp_params_cfg &Template() {
  static const isp39_isp_params_cfg t = [] {
    isp39_isp_params_cfg p;
    std::memset(&p, 0, sizeof(p));
    LoadConfig(p.others.dpcc_cfg, kDpcc);
    LoadConfig(p.others.gic_cfg, kGic);
    LoadConfig(p.others.debayer_cfg, kDebayer);
    LoadConfig(p.others.csm_cfg, kCsm);
    LoadConfig(p.others.cgc_cfg, kCgc);
    LoadConfig(p.others.ynr_cfg, kYnrIso[0]);
    LoadConfig(p.others.cnr_cfg, kCnrIso[0]);
    LoadConfig(p.others.sharp_cfg, kSharp);
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
  for (int i = 0; i < ISP39_GAMMA_OUT_MAX_SAMPLES; ++i) {
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

void FillParams(isp39_isp_params_cfg &p, const Color &c, uint32_t id,
                bool first, bool nr_update) {
  p = Template();
  p.frame_id = id;
  p.module_en_update = (ISP39_MODULE_RGBIR << 1) - 1;
  p.module_ens = kEnabledModules;
  p.module_cfg_update =
      first ? (kEnabledModules | ISP39_MODULE_FORCE) : kColorModules;
  if (nr_update) p.module_cfg_update |= ISP39_MODULE_YNR | ISP39_MODULE_CNR;
  LoadConfig(p.others.ynr_cfg, kYnrIso[c.nr_level]);
  LoadConfig(p.others.cnr_cfg, kCnrIso[c.nr_level]);
  p.others.ynr_cfg.global_set_gain = c.nr_gain_q4;
  p.others.cnr_cfg.global_gain = c.nr_gain_q4;

  auto &bls = p.others.bls_cfg;
  const uint16_t black12 = static_cast<uint16_t>(std::lround(c.black * 16.0));
  bls.fixed_val.r = bls.fixed_val.gr = bls.fixed_val.gb = bls.fixed_val.b =
      black12;
  bls.isp_ob_predgain = 256;
  bls.isp_ob_max = 4095;

  // gain0..2 are the per-exposure HDR gains; awb1 is the white balance.
  auto &awb = p.others.awb_gain_cfg;
  const auto q8 = [](double g) {
    return static_cast<uint16_t>(
        (std::max)(1L, (std::min)(65535L, std::lround(g * 256.0))));
  };
  awb.gain0_red = awb.gain0_blue = awb.gain0_green_r = awb.gain0_green_b = 256;
  awb.gain1_red = awb.gain1_blue = awb.gain1_green_r = awb.gain1_green_b = 256;
  awb.gain2_red = awb.gain2_blue = awb.gain2_green_r = awb.gain2_green_b = 256;
  awb.awb1_gain_r = q8(c.r_gain);
  awb.awb1_gain_gr = awb.awb1_gain_gb = q8(c.g_gain);
  awb.awb1_gain_b = q8(c.b_gain);

  auto &goc = p.others.gammaout_cfg;
  goc.equ_segm = 0;
  goc.finalx4_dense_en = 1;
  goc.offset = 0;
  uint16_t gamma[ISP39_GAMMA_OUT_MAX_SAMPLES];
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
    while (Xioctl(fd, MEDIA_IOC_ENUM_ENTITIES, &e) == 0) {
      const std::string name = e.name;
      std::string *slot = name == "rkisp-isp-subdev"     ? &d.subdev
                          : name == "rkisp_rawrd2_s"     ? &d.rawrd
                          : name == "rkisp-input-params" ? &d.params
                          : name == "rkisp_mainpath"     ? &d.mainpath
                          : name == "rkisp_selfpath"     ? &d.selfpath
                                                         : nullptr;
      if (slot) *slot = CharDevicePath(e.dev.major, e.dev.minor);
      e.id |= MEDIA_ENT_ID_FLAG_NEXT;
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

  bool IsOpen(int w, int h, IspPixelFormat f) const {
    return out_ >= 0 && w == w_ && h == h_ && f == format_;
  }

  bool Open(const Devices &dev, int w, int h, bool rggb, IspPixelFormat f) {
    Close();
    w_ = w;
    h_ = h;
    format_ = f;
    const bool uyvy = f == IspPixelFormat::kUyvy422Bt601FullRange;
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

  // Starts the ISP on one frame; Collect() returns it.
  bool Submit(const cv::Mat &raw, const WhiteBalance &rows, const Color &c) {
    ReclaimParams();
    const uint32_t id = frame_id_++;
    if (!QueueParams(c, id, false)) return false;
    uint8_t *rd = static_cast<uint8_t *>(rd_bufs_[rd_next_].ptr);
    for (int y = 0; y < h_; ++y)
      std::memcpy(rd + static_cast<size_t>(y) * rd_bpl_, rows.RawRow(raw, y),
                  w_);
    if (!QueueImage(rawrd_, kOut, rd_next_, rd_bpl_ * h_)) return false;
    rd_next_ = (rd_next_ + 1) % static_cast<int>(rd_bufs_.size());
    isp2x_csi_trigger tg{};
    tg.frame_id = id;
    tg.mode = T_START_X1;
    if (Xioctl(subdev_, RKISP_CMD_TRIGGER_READ_BACK, &tg) < 0)
      return Fail("trigger");
    return true;
  }

  // Copies the frame out, applying `tone` (nullptr = linear output).
  bool Collect(cv::Mat &out, const FastBalancedToneLuts *tone) {
    const int oi = DequeueImage(out_, kCap, 2000);
    if (oi < 0) return Fail("output timeout");
    const bool uyvy = format_ == IspPixelFormat::kUyvy422Bt601FullRange;
    // Selfpath "RGB24" stores B, G, R in memory order.
    out.create(h_, w_, uyvy ? CV_8UC2 : CV_8UC3);
    const uint8_t *src = static_cast<const uint8_t *>(out_bufs_[oi].ptr);
    const size_t row_bytes = static_cast<size_t>(w_) * (uyvy ? 2 : 3);
    for (int y = 0; y < h_; ++y) {
      const uint8_t *s = src + static_cast<size_t>(y) * out_bpl_;
      if (!tone)
        std::memcpy(out.ptr(y), s, row_bytes);
      else if (uyvy)
        ToneRowUyvy(s, out.ptr(y), w_, *tone);
      else
        ToneRowBgr(s, out.ptr(y), w_, *tone);
    }
    if (!QueueImage(out_, kCap, oi, 0)) return false;
    DequeueImage(rawrd_, kOut, 100);
    return true;
  }

  void Close() {
    int t = kCap;
    if (out_ >= 0) Xioctl(out_, VIDIOC_STREAMOFF, &t);
    t = kOut;
    if (rawrd_ >= 0) Xioctl(rawrd_, VIDIOC_STREAMOFF, &t);
    t = kMeta;
    if (params_ >= 0) Xioctl(params_, VIDIOC_STREAMOFF, &t);
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
      if (b.len < sizeof(isp39_isp_params_cfg)) {
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
    FillParams(*static_cast<isp39_isp_params_cfg *>(par_bufs_[i].ptr), c, id,
               first,
               first || c.nr_level != nr_level_ || c.nr_gain_q4 != nr_gain_q4_);
    nr_level_ = c.nr_level;
    nr_gain_q4_ = c.nr_gain_q4;
    v4l2_buffer b{};
    b.type = kMeta;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    b.bytesused = sizeof(isp39_isp_params_cfg);
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

bool ApplyBatch(const FastBalancedIspDispatchJob *jobs, int n) {
  static std::mutex mutex;
  static VirtualIsp isp[2];  // [0] RGGB, [1] BGGR
  static int bound[2] = {-1, -1};  // index into `virtual_isps`
  static std::vector<Devices> virtual_isps;
  static bool disabled = false;
  std::lock_guard<std::mutex> lock(mutex);
  if (disabled || n <= 0) return false;
  for (int i = 0; i < n; ++i) {
    const FastBalancedIspDispatchJob &j = jobs[i];
    const bool nr_high = FastBalancedBayerNrEnabled() &&
                         !j.raw_bayer_nr_done &&
                         j.sensor_gain * j.wb->NrGainScale() >= MaxNrGain();
    if (j.half_height || nr_high || j.raw->type() != CV_8UC1 ||
        (j.raw->cols & 1) || (j.raw->rows & 1) ||
        j.raw->size() != jobs[0].raw->size() ||
        j.output_format != jobs[0].output_format)
      return false;
  }
  const auto give_up = [](const char *what, const std::string &why) {
    std::fprintf(stderr,
                 "[rkisp] %s%s%s; using the software ISP from now on\n", what,
                 why.empty() ? "" : ": ", why.c_str());
    for (VirtualIsp &v : isp) v.Close();
    disabled = true;
    return false;
  };
  // Statistics and the RAW and output copies are CPU work: keep them on an
  // A72 like the software path's caller.
  ScopedThreadAffinity caller_affinity(4);
  const int w = jobs[0].raw->cols, h = jobs[0].raw->rows;
  if (virtual_isps.empty()) {
    for (int v = 0; v < 8; ++v) {
      Devices d;
      if (!FindDevices("rkisp-vir" + std::to_string(v), d)) break;
      virtual_isps.push_back(d);
    }
  }
  for (int k = 0; k < 2; ++k) {
    const bool used = std::any_of(jobs, jobs + n, [&](const auto &j) {
      return (j.bayer == BayerConversion::kColorBayerBg2Bgr) == (k == 0);
    });
    if (!used || isp[k].IsOpen(w, h, jobs[0].output_format)) continue;
    // A virtual ISP held by another client (e.g. rkaiq_3A waiting for a
    // sensor) refuses the formats with EBUSY; try the next one.
    std::string why = "no rkisp-vir media device";
    bool opened = false;
    for (int v = 0; v < static_cast<int>(virtual_isps.size()) && !opened;
         ++v) {
      if (v == bound[1 - k]) continue;
      opened = isp[k].Open(virtual_isps[v], w, h, k == 0,
                           jobs[0].output_format);
      if (opened)
        bound[k] = v;
      else
        why = "rkisp-vir" + std::to_string(v) + " " + isp[k].error();
    }
    if (!opened) return give_up("no usable virtual ISP", why);
  }

  const bool apply_tone = FastBalancedGammaEnabled();
  const bool exposure = apply_tone && GetFastExposureConfig().enabled;
  const FastBalancedToneLuts *tone =
      apply_tone ? &GetFastBalancedToneLuts() : nullptr;

  // Each virtual ISP is driven by its own thread with up to two frames in
  // flight, so one camera's tone/copy overlaps the core working on another.
  const auto run = [&](int k) {
    int pending[2], count = 0;
    for (int i = 0; i < n; ++i) {
      const FastBalancedIspDispatchJob &j = jobs[i];
      if ((j.bayer == BayerConversion::kColorBayerBg2Bgr) != (k == 0))
        continue;
      if (count == 2) {
        if (!isp[k].Collect(*jobs[pending[0]].output, tone)) return false;
        pending[0] = pending[1];
        count = 1;
      }
      WhiteBalance &wb = *j.wb;
      wb.PrepareRawMetadata(*j.raw, j.metadata_region);
      wb.SetDisplayExposure(exposure, j.sensor_gain);
      const WhiteBalance::LinearStage st =
          wb.PrepareLinearStage(*j.raw, j.bayer);
      Color c;
      c.black = st.black;
      c.r_gain = st.r_gain;
      c.g_gain = st.g_gain;
      c.b_gain = st.b_gain;
      c.exposure = st.exposure;
      c.knee = st.knee;
      SelectNr(j.sensor_gain * wb.NrGainScale(), c);
      if (!isp[k].Submit(*j.raw, wb, c)) return false;
      pending[count++] = i;
    }
    for (int p = 0; p < count; ++p)
      if (!isp[k].Collect(*jobs[pending[p]].output, tone)) return false;
    return true;
  };
  static Worker worker(5);
  worker.Start([&] { return run(1); });
  const bool ok0 = run(0);
  const bool ok1 = worker.Wait();
  if (!ok0) return give_up("rkisp RGGB", isp[0].error());
  if (!ok1) return give_up("rkisp BGGR", isp[1].error());
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

#endif  // CYPERSTEREO_RK3576 && __linux__
