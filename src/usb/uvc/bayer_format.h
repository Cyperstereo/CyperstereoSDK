#ifndef CYPERSTEREO_BAYER_FORMAT_H_
#define CYPERSTEREO_BAYER_FORMAT_H_

#include <cstdint>
#include "smartsens_metadata.h"

namespace cyperstereo {

// OpenCV-style conversion names.  The names intentionally describe the
// conversion code used by the SDK rather than the first physical CFA pixel:
// OpenCV's Bayer-to-BGR aliases make those two descriptions easy to confuse.
enum class BayerConversion {
  kColorBayerRg2Bgr,
  kColorBayerBg2Bgr,
};

// FPGA software 04 introduced sensor mirror+flip on C1/C4, which reverses
// their red/blue Bayer phase; software 05/06 keep that image orientation.
// USB image order is C1,C2,C4,C3, so the opposite phase applies to display
// images 1 and 3 (indices 0 and 2). Software 03 keeps the legacy conversion on
// all four images.
inline BayerConversion SelectBayerConversion(uint32_t hardware_version,
                                              uint32_t software_version,
                                              int image_index) {
  const bool image_1_or_3 = image_index == 0 || image_index == 2;
  const bool mirrored_smartsens =
      hardware_version ==
          static_cast<uint32_t>(kSmartSensHardwareVersion) &&
      (software_version ==
           static_cast<uint32_t>(kSmartSensSoftwareVersion4) ||
       software_version ==
           static_cast<uint32_t>(kSmartSensSoftwareVersion5) ||
       software_version ==
           static_cast<uint32_t>(kSmartSensSoftwareVersion6));
  if (mirrored_smartsens && image_1_or_3)
    return BayerConversion::kColorBayerBg2Bgr;
  return BayerConversion::kColorBayerRg2Bgr;
}

}  // namespace cyperstereo

#endif  // CYPERSTEREO_BAYER_FORMAT_H_
