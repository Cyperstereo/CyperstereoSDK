#ifndef CYPERSTEREO_AR0234_EXPOSURE_METER_H_
#define CYPERSTEREO_AR0234_EXPOSURE_METER_H_
#include <cstdint>

namespace cyperstereo {
struct Ar0234ShadowMeter {
  bool valid = false;
  int signal_p35 = 0;
  double requested_gain = 1.0;
};

// Histogram units are RAW8 DN after pedestal subtraction. Ignore near-black
// noise (0..3 DN) and the top overflow/saturation bin. At least 25% of the
// ROI must carry usable signal; a covered lens must not request digital lift.
inline Ar0234ShadowMeter MeterAr0234Shadows(
    const uint32_t *histogram, int bins, int total, double key = 32.0) {
  Ar0234ShadowMeter result;
  if (!histogram || bins < 6 || total <= 0 || !(key > 0.0)) return result;
  uint64_t useful = 0;
  for (int i = 4; i < bins - 1; ++i) useful += histogram[i];
  if (!useful || useful * 4 < static_cast<uint64_t>(total)) return result;
  const uint64_t rank = (useful * 35 + 99) / 100;
  uint64_t cumulative = 0;
  for (int i = 4; i < bins - 1; ++i) {
    cumulative += histogram[i];
    if (cumulative >= rank) {
      result.valid = true;
      result.signal_p35 = i;
      result.requested_gain = key / i;
      break;
    }
  }
  return result;
}
}  // namespace cyperstereo
#endif
