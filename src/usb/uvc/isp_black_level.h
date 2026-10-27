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
// CYPERSTEREO_AR0234_BLACK_LEVEL (0..32) fixes the AR0234 black and turns
// the corner measurement off; negative = not set.
inline double Ar0234BlackLevelOverride() {
  static const double value = [] {
    if (const char *s = std::getenv("CYPERSTEREO_AR0234_BLACK_LEVEL")) {
      char *end = nullptr;
      const double v = std::strtod(s, &end);
      if (end != s && *end == '\0' && std::isfinite(v) && v >= 0 && v <= 32)
        return v;
    }
    return -1.0;
  }();
  return value;
}
// RAW8 black read from the 7761 fisheye corners (outside the image circle),
// 0 = no measurement yet.  The sensor black after ABLC differs per board and
// gain table (3.5 on one board, ~6.4 on another with the same pedestal), and
// every DN left in is multiplied by the blue WB gain and digital exposure,
// which turns the shadows blue.
inline std::atomic<double> &Ar0234MeasuredBlackLevel() {
  static std::atomic<double> value{0.0};
  return value;
}
// Shared by all camera workers: a slow average over cameras and frames,
// published only when it moves by 0.25 DN so the LUTs are not rebuilt for
// noise.  Races between workers only drop an update.
inline void Ar0234UpdateMeasuredBlackLevel(double sample) {
  static std::atomic<double> avg{0.0};
  const double prev = avg.load(std::memory_order_relaxed);
  const double next = prev > 0.0 ? prev + 0.05 * (sample - prev) : sample;
  avg.store(next, std::memory_order_relaxed);
  const double q = std::round(next * 8.0) / 8.0;
  std::atomic<double> &pub = Ar0234MeasuredBlackLevel();
  const double cur = pub.load(std::memory_order_relaxed);
  if (cur <= 0.0 || std::fabs(q - cur) >= 0.25)
    pub.store(q, std::memory_order_relaxed);
}
inline double Ar0234CalibratedBlackLevel() {
  const double fixed = Ar0234BlackLevelOverride();
  if (fixed >= 0.0) return fixed;
  const double measured =
      Ar0234MeasuredBlackLevel().load(std::memory_order_relaxed);
  if (measured > 0.0) return measured;
  // Cap-on RAW8 dark baseline (DATA_PEDESTAL=42, NOISE_PEDESTAL=42, digital
  // gain 1x), used until the first corner measurement.
  return 3.5;
}
inline double IspBlackLevel() {
  return FastIspAr0234Profile().load(std::memory_order_relaxed)
             ? Ar0234CalibratedBlackLevel() : kIspBlackLevel;
}
}  // namespace cyperstereo
#endif
