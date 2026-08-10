// HDR-ISP-main integration: full replacement for the legacy ApplyISP /
// ApplyISPParallel software ISP in cyperstereo_api.h.
//
// The per-frame processing is delegated to the HDR-ISP-main pipeline
// (HDR-ISP-main/srcs), configured for the SC136HGS 8-bit BGGR Bayer output.
// Default: blc -> bayernr -> wbgain -> demoasic -> lumanr -> chromanr ->
// falsecolor -> rgbgamma. rgbgamma is a
// color-preserving luminance-ratio gain with a noise-floor guard and 1.35x
// maximum lift; CCM and YUV finishing are opt-in through
// CYPERSTEREO_HDRISP_PIPE.  BLC (16 DN in the 8-bit domain) and white balance
// are calibrated per camera.
//
// Env knobs (all read once):
//   CYPERSTEREO_HDRISP_PIPE  '|'-separated module list overriding the default
//                            pipe above (e.g. to insert "ltm" for stills).
//   CYPERSTEREO_CCM          "off" = identity, or 9 R-major values
//                            "rr,rg,rb,gr,gg,gb,br,bg,bb" (same convention as
//                            before; tools/ccm_calibrate.py output).
//   (Tone curve) By default a mild color-preserving luminance gamma is used.
//                Setting ANY of the three knobs below switches its LUT to the
//                ported ApplyISPParallel sRGB+blackpoint+S curve:
//   CYPERSTEREO_GAMMA        unset = sRGB encode; 1.0 = linear; 1<g<=4 = 1/g.
//   CYPERSTEREO_BLACKPOINT   encoded-domain black point 0..32, default 6.
//   CYPERSTEREO_CONTRAST     S-curve blend 0..1, default 0.3 (folded into the
//                            rgbgamma curve; the HDR-ISP "contrast" module is
//                            a separate Y-domain stretch).
//   CYPERSTEREO_HDRISP_PROFILE  set to print per-camera pipeline ms.
#ifndef CYPERSTEREO_HDR_ISP_H_
#define CYPERSTEREO_HDR_ISP_H_

#include <initializer_list>

#include <opencv2/core/core.hpp>

#include "bayer_format.h"

namespace cyperstereo {

struct HdrIspJob {
  const cv::Mat *raw;  // 8-bit Bayer input (NOT modified, unlike the old
                       // ApplyISP which wrote the WB LUT back into raw)
  cv::Mat *color;      // 8UC3 BGR output, (re)allocated as needed
  const char *name;    // stable per-camera key, e.g. "cam1".."cam4"
  BayerConversion bayer;
  double sensor_gain;  // per-camera analogue/AEC target gain, nominally 1..8x

  HdrIspJob(
      const cv::Mat &r, cv::Mat &c, const char *n,
      BayerConversion b = BayerConversion::kColorBayerRg2Bgr)
      : raw(&r), color(&c), name(n), bayer(b), sensor_gain(1.0) {}

  HdrIspJob(
      const cv::Mat &r, cv::Mat &c, const char *n, double gain,
      BayerConversion b = BayerConversion::kColorBayerRg2Bgr)
      : raw(&r), color(&c), name(n), bayer(b), sensor_gain(gain) {}
};

// Runs the HDR-ISP-main pipeline on one camera. Per-camera state (frame
// buffers, AWB gains) is keyed by `name`, so always pass the same name for
// the same physical camera.
void ApplyHdrIsp(
    const cv::Mat &raw, cv::Mat &color, const char *name,
    BayerConversion bayer = BayerConversion::kColorBayerRg2Bgr,
    double sensor_gain = 1.0);

// jobs[0] runs on the calling thread, the rest on their own threads.
void ApplyHdrIspParallel(const HdrIspJob *jobs, int n);
void ApplyHdrIspParallel(std::initializer_list<HdrIspJob> jobs);

}  // namespace cyperstereo

#endif  // CYPERSTEREO_HDR_ISP_H_
