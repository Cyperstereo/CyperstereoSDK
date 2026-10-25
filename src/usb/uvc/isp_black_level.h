#ifndef CYPERSTEREO_ISP_BLACK_LEVEL_H_
#define CYPERSTEREO_ISP_BLACK_LEVEL_H_
#include <atomic>
#include <cmath>
#include <cstdlib>
namespace cyperstereo {
// SC136 baseline matches develop10; AR0234 uses its separate calibration.
// Internal ISP code must use IspBlackLevel() for the detected sensor profile.
constexpr double kIspBlackLevel = 16.0;
inline std::atomic<bool> &FastIspAr0234Profile() {
  static std::atomic<bool> value{false};
  return value;
}
inline double Ar0234CalibratedBlackLevel() {
  static const double value = [] {
    if (const char *s = std::getenv("CYPERSTEREO_AR0234_BLACK_LEVEL")) {
      char *end = nullptr;
      const double v = std::strtod(s, &end);
      if (end != s && *end == '\0' && std::isfinite(v) && v >= 0 && v <= 32)
        return v;
    }
    // Measured RAW8 dark baseline with the current auto AE/AGC table,
    // DATA_PEDESTAL=42, NOISE_PEDESTAL=42, digital gain=1x. This is NOT
    // DATA_PEDESTAL/4: internal noise-pedestal/gain processing shifts it.
    // Recalibrate when sensor gain mode, digital gain or pedestal changes.
    return 3.5;
  }();
  return value;
}
inline double IspBlackLevel() {
  return FastIspAr0234Profile().load(std::memory_order_relaxed)
             ? Ar0234CalibratedBlackLevel() : kIspBlackLevel;
}
}  // namespace cyperstereo
#endif
