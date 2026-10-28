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
// their red/blue Bayer phase; software 05 through 08 keep that orientation.
// USB image order is C1,C2,C4,C3, so the opposite phase applies to display
// images 1 and 3 (indices 0 and 2). Software 03 keeps the legacy conversion on
// all four images.  AR0234 (hardware 03) streams RGGB on every plane: its
// window is already the full 1920 active columns (x 8..1927), so the column
// phase cannot be moved to reach BGGR, and its mirror/flip keep the starting
// Bayer colour.  All four AR0234 images use the RGGB conversion.
// Six-image markers 04/05 use the phase of their sensor family (02/03).
inline BayerConversion SelectBayerConversion(uint32_t hardware_version,
                                              uint32_t software_version,
                                              int image_index) {
  hardware_version = static_cast<uint32_t>(
      SensorFamilyHardwareVersion(static_cast<int>(hardware_version)));
  if (hardware_version == static_cast<uint32_t>(kAr0234HardwareVersion))
    return BayerConversion::kColorBayerBg2Bgr;
  const bool image_1_or_3 = image_index == 0 || image_index == 2;
  const bool mirrored_smartsens =
      hardware_version ==
          static_cast<uint32_t>(kSmartSensHardwareVersion) &&
      (software_version ==
           static_cast<uint32_t>(kSmartSensSoftwareVersion4) ||
       software_version ==
           static_cast<uint32_t>(kSmartSensSoftwareVersion5) ||
       software_version ==
           static_cast<uint32_t>(kSmartSensSoftwareVersion6) ||
       software_version ==
           static_cast<uint32_t>(kSmartSensSoftwareVersion7) ||
       software_version ==
           static_cast<uint32_t>(kSmartSensSoftwareVersion8));
  if (mirrored_smartsens && image_1_or_3)
    return BayerConversion::kColorBayerBg2Bgr;
  return BayerConversion::kColorBayerRg2Bgr;
}

// AR0234 image1/image2 are rotated 180 degrees after telemetry is captured.
// Both dimensions are even, so that rotation swaps RGGB and BGGR. image3,
// image4, and the six-image half cameras are not rotated and keep
// SelectBayerConversion.
inline BayerConversion SelectDeliveredPlaneBayer(uint32_t hardware_version,
                                                 uint32_t software_version,
                                                 int image_index) {
  const BayerConversion native =
      SelectBayerConversion(hardware_version, software_version, image_index);
  if ((image_index == 0 || image_index == 1) &&
      IsAr0234FamilyHardware(static_cast<int>(hardware_version))) {
    return native == BayerConversion::kColorBayerBg2Bgr
               ? BayerConversion::kColorBayerRg2Bgr
               : BayerConversion::kColorBayerBg2Bgr;
  }
  return native;
}

// Bayer phase of a PHYSICAL camera (1-based C1..C6) rather than of a display
// plane (plane order C1,C2,C4,C3).  Horizontal 2:1 same-colour binning (the
// six-image half lanes) preserves the phase, so the half images use exactly
// the conversion of their full-width parent camera.  Only C1/C4 are
// mirror+flipped; C5/C6 keep the C2/C3 phase.
inline BayerConversion SelectCameraBayerConversion(uint32_t hardware_version,
                                                    uint32_t software_version,
                                                    int camera_number) {
  static const int kPlaneByCamera[6] = {0, 1, 3, 2, 1, 3};
  const int plane =
      (camera_number >= 1 && camera_number <= 6)
          ? kPlaneByCamera[camera_number - 1] : 0;
  return SelectBayerConversion(hardware_version, software_version, plane);
}

}  // namespace cyperstereo

#endif  // CYPERSTEREO_BAYER_FORMAT_H_
