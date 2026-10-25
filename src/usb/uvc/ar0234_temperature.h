#ifndef CYPERSTEREO_AR0234_TEMPERATURE_H_
#define CYPERSTEREO_AR0234_TEMPERATURE_H_
#include <cmath>
#include <cstdint>
#include <limits>
#include "ar0234_diagnostics.h"

namespace cyperstereo {
// AND9820 temperature sensor: R30B4[4,0] enables conversion/power,
// R30B2[9:0] is ADC, R30C6[9:0] is this sensor's 55 C calibration.
// T = 55 + 0.7 * (ADC - calibration). Host monotonic seconds track freshness.
class Ar0234TemperatureState {
 public:
  static double Unavailable() { return std::numeric_limits<double>::quiet_NaN(); }
  void Observe(const Ar0234Readback &read, double now) {
    if (!std::isfinite(now) || read.status == 0) return;
    if (seen_sequence_) {
      const auto delta = static_cast<uint16_t>(read.sequence - sequence_);
      if (delta == 0) return; // Same sample repeated in several USB frames.
      if (delta >= 32768) Reset(); // Counter restarted; discard old calibration.
    }
    seen_sequence_ = true;
    sequence_ = read.sequence;
    switch (read.address) {
      case 0x30B4:
        control_valid_ = read.ok() && (read.value & 0x0031) == 0x0011;
        control_time_ = now;
        break;
      case 0x30C6:
        calibration_ = read.value & 0x03FF;
        calibration_valid_ = read.ok() && calibration_ > 0 && calibration_ < 1023;
        calibration_time_ = now;
        break;
      case 0x30B2:
        adc_ = read.value & 0x03FF;
        adc_valid_ = read.ok() && adc_ > 0 && adc_ < 1023;
        adc_time_ = now;
        break;
      default: break;
    }
  }
  bool Valid(double now) const {
    // A scan takes 19 / 8 = 2.375 seconds; allow several missed scans,
    // then report unavailable rather than presenting an old value as current.
    return control_valid_ && calibration_valid_ && adc_valid_ &&
           Fresh(now, control_time_) && Fresh(now, calibration_time_) &&
           Fresh(now, adc_time_);
  }
  double Celsius(double now) const {
    return Valid(now) ? 55.0 + 0.7 * (int(adc_) - int(calibration_)) : Unavailable();
  }
  double AgeSeconds(double now) const {
    return adc_valid_ && std::isfinite(now) && now >= adc_time_
               ? now - adc_time_ : Unavailable();
  }
  uint16_t Adc() const { return adc_; }
  uint16_t Calibration() const { return calibration_; }
  void Reset() { *this = Ar0234TemperatureState{}; }
 private:
  static bool Fresh(double now, double sample) {
    return std::isfinite(now) && now >= sample && now - sample <= 10.0;
  }
  bool seen_sequence_ = false;
  uint16_t sequence_ = 0, adc_ = 0, calibration_ = 0;
  bool control_valid_ = false, calibration_valid_ = false, adc_valid_ = false;
  double control_time_ = 0, calibration_time_ = 0, adc_time_ = 0;
};
}  // namespace cyperstereo
#endif
