#ifndef CYPERSTEREO_SC136_AEC_GUARD_H_
#define CYPERSTEREO_SC136_AEC_GUARD_H_
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
namespace cyperstereo {
// Optional extension to 02/07 (and the six-image 04/07, same metadata row).
// Existing columns 0..105 stay unchanged.
// Columns 106/107 = A915/0106; six words per physical sensor C1,C2,C3,C4.
struct Sc136AecTelemetry {
  bool valid = false;
  uint16_t frame = 0, sample = 0, write = 0, status = 0, ratio_q8 = 256;
  uint8_t lumin = 0, reason = 0;
  unsigned Delay() const { return (status >> 5) & 7; }
  // Conservative control still has ACK-confirmed gain and a longer delay.
  // Only actual fault, active pause, or incomplete statistics invalidate it.
  bool Unsafe() const { return (status & 0x4100) != 0 || (status & 2) == 0; }
};
template<class ReadWord>
inline std::array<Sc136AecTelemetry, 4> DecodeSc136Aec(
    int hardware, int software, ReadWord word) {
  std::array<Sc136AecTelemetry, 4> result{};
  if ((hardware != 2 && hardware != 4) || software != 7 || word(106) != 0xA915 ||
      word(107) != 0x0106) return result;
  // A damaged header/body must not partially activate the new contract.
  for (int s = 0; s < 4; ++s)
    if ((word(111 + s * 6) & 0x8000) == 0) return result;
  static const int physical[4] = {0, 1, 3, 2};
  for (int p = 0; p < 4; ++p) {
    const int b = 108 + physical[p] * 6;
    auto &d = result[p];
    d.valid = true; d.frame = word(b); d.sample = word(b + 1);
    d.write = word(b + 2); d.status = word(b + 3);
    d.lumin = word(b + 4) & 255; d.reason = (word(b + 4) >> 8) & 255;
    d.ratio_q8 = word(b + 5);
    if (d.Delay() < 1 || d.Delay() > 3) return {};
  }
  return result;
}
// Hold the last confirmed gain while the next target matures. Returning zero
// on EVERY exposure write would repeatedly reset digital lift and cause pumping.
// Zero is reserved for startup, invalid metadata and FPGA communication faults or an active pause.
class Sc136GainAlignment {
 public:
  double Update(double target, const Sc136AecTelemetry &d) {
    if (!(target >= 1.0 && target <= 12.0) || !std::isfinite(target)) {
      DropEstimate(); return 0.0;
    }
    if (d.valid) {
      if (!seen_extended_) DropEstimate(); // never carry legacy certainty across an upgrade
      seen_extended_ = true;
      if (d.Unsafe()) { DropEstimate(); return 0.0; }
      const uint16_t age = static_cast<uint16_t>(d.frame - d.write);
      if (age >= 32768) { DropEstimate(); return 0.0; }
      // A repeated frame cannot confirm a new target; retain the old estimate.
      if (have_frame_) {
        const uint16_t step = static_cast<uint16_t>(d.frame - last_frame_);
        if (step == 0) return confirmed_gain_;
        if (step >= 32768) { DropEstimate(); return 0.0; }
      }
      last_frame_ = d.frame; have_frame_ = true;
      if (age < d.Delay()) return confirmed_gain_;
      confirmed_gain_ = target; return confirmed_gain_;
    }
    // Once advertised, a missing/corrupt extension is not a legacy frame.
    if (seen_extended_) { DropEstimate(); return 0.0; }
    // Legacy firmware: no claim of exact alignment. Require three consecutive
    // identical targets before replacing the last confirmed estimate.
    have_frame_ = false;
    if (std::fabs(target - candidate_gain_) > 1e-9) {
      candidate_gain_ = target; legacy_count_ = 1;
    } else if (legacy_count_ < 3) ++legacy_count_;
    if (legacy_count_ >= 3) confirmed_gain_ = target;
    return confirmed_gain_;
  }
  void Reset() { DropEstimate(); seen_extended_ = false; }
 private:
  void DropEstimate() { confirmed_gain_ = 0; candidate_gain_ = 0; legacy_count_ = 0; have_frame_ = false; }
  double confirmed_gain_ = 0, candidate_gain_ = 0;
  unsigned legacy_count_ = 0;
  bool have_frame_ = false;
  bool seen_extended_ = false;
  uint16_t last_frame_ = 0;
};
inline double Sc136LimitedLogExposure(double previous, double target) {
  const double filtered = previous + 0.5 * (target - previous);
  // No overshoot for a fixed target; bounded output changes when it moves.
  return (std::max)(previous - std::log(1.125),
                    (std::min)(previous + std::log(1.0625), filtered));
}
// Ring keeps >8 seconds at 30fps. Event dump includes prehistory plus five
// seconds after an event. Set CYPERSTEREO_AEC_TRACE=all/off to override.
class Sc136AecTrace {
 public:
  void Record(double timestamp, const Sc136AecTelemetry d[4],
              const uint16_t exposure[4], const double gain[4],
              const double isp_gain[4], const double raw_mean[4], int cameras) {
    const char *setting = std::getenv("CYPERSTEREO_AEC_TRACE");
    if (setting && std::string(setting) == "off") return;
    bool valid = false, trigger = false;
    std::ostringstream row;
    row << std::fixed << std::setprecision(6);
    for (int c = 0; c < cameras; ++c) {
      if (!d[c].valid) continue;
      valid = true;
      const unsigned alarm = d[c].status & 0x6100;
      trigger |= (alarm & ~previous_alarm_[c]) != 0;
      trigger |= previous_mean_[c] >= 0 && std::fabs(raw_mean[c] - previous_mean_[c]) > 40;
      previous_alarm_[c] = alarm; previous_mean_[c] = raw_mean[c];
      // Also capture slow/non-recovery; it need not produce a one-frame jump.
      if ((d[c].status & 2) && d[c].lumin < 64) {
        if (dark_frames_[c] < 61) ++dark_frames_[c];
        trigger |= dark_frames_[c] == 60;
      } else dark_frames_[c] = 0;
      row << timestamp << ',' << c << ',' << d[c].frame << ',' << d[c].sample
          << ',' << d[c].write << ',' << d[c].status << ',' << unsigned(d[c].reason)
          << ',' << unsigned(d[c].lumin) << ',' << d[c].ratio_q8 << ','
          << exposure[c] << ',' << gain[c] << ',' << isp_gain[c] << ',' << raw_mean[c] << '\n';
    }
    if (!valid) return;
    const bool all = setting && std::string(setting) == "all";
    if (trigger) tail_ = 150;
    if (all || tail_ > 0) {
      if (!file_.is_open()) {
        file_.open("sc136_aec_trace.csv", std::ios::app);
        if (file_) file_ << "timestamp,plane,frame,sample,write,status,reason,score,ratio_q8,exp_lines,target_gain,isp_gain,raw_mean\n";
      }
      if (file_) {
        for (const auto &old : ring_) file_ << old;
        file_ << row.str();
        if (trigger || tail_ == 1) file_.flush();
      }
      ring_.clear(); if (tail_) --tail_;
    } else {
      ring_.push_back(row.str()); if (ring_.size() > 256) ring_.pop_front();
    }
  }
 private:
  std::array<unsigned, 4> previous_alarm_{};
  std::array<unsigned, 4> dark_frames_{};
  std::array<double, 4> previous_mean_{{-1, -1, -1, -1}};
  std::deque<std::string> ring_;
  std::ofstream file_;
  unsigned tail_ = 0;
};
} // namespace cyperstereo
#endif
