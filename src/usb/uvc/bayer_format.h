#ifndef CYPERSTEREO_BAYER_FORMAT_H_
#define CYPERSTEREO_BAYER_FORMAT_H_

#include <cstdint>

namespace cyperstereo {

// OpenCV-style conversion names.  The names intentionally describe the
// conversion code used by the SDK rather than the first physical CFA pixel:
// OpenCV's Bayer-to-BGR aliases make those two descriptions easy to confuse.
enum class BayerConversion {
  kColorBayerRg2Bgr,
  kColorBayerBg2Bgr,
};

inline bool IsSupportedSmartSensFirmware(int hardware_version,
                                         int software_version) {
  return hardware_version == 2 &&
         (software_version == 3 || software_version == 4);
}

// FPGA software 04 mirror+flips C1/C4 in the sensor, which reverses their
// red/blue Bayer phase.  USB image order is C1,C2,C4,C3, so this applies to
// display images 1 and 3 (zero-based indices 0 and 2).  Software 03 keeps the
// legacy conversion on all four images.
inline BayerConversion SelectBayerConversion(uint32_t hardware_version,
                                              uint32_t software_version,
                                              int image_index) {
  const bool image_1_or_3 = image_index == 0 || image_index == 2;
  if (hardware_version == 2 && software_version == 4 && image_1_or_3)
    return BayerConversion::kColorBayerBg2Bgr;
  return BayerConversion::kColorBayerRg2Bgr;
}

}  // namespace cyperstereo

#endif  // CYPERSTEREO_BAYER_FORMAT_H_
