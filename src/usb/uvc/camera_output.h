#ifndef CYPERSTEREO_CAMERA_OUTPUT_H_
#define CYPERSTEREO_CAMERA_OUTPUT_H_

#include "cyperstereo_api.h"
#include <cmath>
#include <iomanip>
#include <limits>

namespace cyperstereo {

// Small value-owned state, copied under FrameInfo::mtx with the image planes.
// No cv::Mat here: enqueueing diagnostics never copies a RAW image.
struct CameraSensorSnapshot {
  std::array<double, 4> temperature{}, temperature_age_s{}, exposure_s{};
  std::array<bool, 4> temperature_valid{};
  std::array<uint16_t, 4> temperature_adc{}, temperature_calibration{};

  static CameraSensorSnapshot FromFrame(const FrameStreamData &frame) {
    CameraSensorSnapshot result;
    for (int i = 0; i < 4; ++i) {
      result.temperature_valid[i] = frame.camera_temperature_valid[i] &&
          std::isfinite(frame.camera_temperature[i]);
      result.temperature[i] = result.temperature_valid[i]
          ? frame.camera_temperature[i] : std::numeric_limits<double>::quiet_NaN();
      result.temperature_age_s[i] = frame.camera_temperature_age_s[i];
      result.temperature_adc[i] = frame.camera_temperature_adc[i];
      result.temperature_calibration[i] = frame.camera_temperature_calibration[i];
      result.exposure_s[i] = frame.exposure_time[i];
    }
    return result;
  }
};

inline IspMode CameraBgrIspMode() {
  return FastBalancedIspEnabled() ? IspMode::kFastBalancedBgr888
                                : IspMode::kQualityReferenceBgr888;
}

// Plane order is C1/C2/C4/C3, identical to capture_image_imu. In particular,
// AR0234 must not inherit the old hard-coded SC136 Bayer phase on C2/C3.
// Metadata is passed as a row overlay; the original RAW stays untouched.
inline void ApplyBayerCameraIsp(
    IspProcessor &isp, cv::Mat *const raw[4], cv::Mat *const color[4], int count,
    uint32_t hardware, uint32_t software, const std::array<double, 4> &gain,
    const std::array<bool, 4> &nr_done, const RawTelemetrySnapshot &telemetry) {
  CV_Assert(count >= 1 && count <= 4);
  IspFrameJob jobs[] = {
      {*raw[0], *color[0], "cam1", gain[0], SelectBayerConversion(hardware, software, 0), nr_done[0], telemetry.RegionForPlane(0)},
      {*raw[1], *color[1], "cam2", gain[1], SelectBayerConversion(hardware, software, 1), nr_done[1], telemetry.RegionForPlane(1)},
      {*raw[2], *color[2], "cam3", gain[2], SelectBayerConversion(hardware, software, 2), nr_done[2], telemetry.RegionForPlane(2)},
      {*raw[3], *color[3], "cam4", gain[3], SelectBayerConversion(hardware, software, 3), nr_done[3], telemetry.RegionForPlane(3)}};
  isp.ApplyParallel(jobs, count);
}

inline void WriteCameraSensorsCsv(
    std::ostream &out, double timestamp, int count, uint32_t hardware,
    uint32_t software, const std::array<double, 4> &isp_gain,
    const CameraSensorSnapshot &sensors) {
  out << "timestamp_s,plane,hardware,software,exposure_s,isp_sensor_gain,temperature_c,temperature_valid,temperature_age_s,temperature_adc,temperature_calibration\n";
  out << std::setprecision(12);
  for (int i = 0; i < count && i < 4; ++i)
    out << timestamp << ',' << i << ',' << hardware << ',' << software << ','
        << sensors.exposure_s[i] << ',' << isp_gain[i] << ','
        << sensors.temperature[i] << ',' << sensors.temperature_valid[i] << ','
        << sensors.temperature_age_s[i] << ',' << sensors.temperature_adc[i] << ','
        << sensors.temperature_calibration[i] << '\n';
}

// ROS shutdown / Ctrl+C must also work when the USB camera has no frames.
template <typename KeepRunning>
inline bool WaitForCameraFrame(FrameInfo &frame_info, KeepRunning keep_running) {
  std::unique_lock<std::mutex> lock(frame_info.mtx);
  while (keep_running()) {
    if (frame_info.con.wait_for(lock, std::chrono::milliseconds(200),
                              [&] { return frame_info.frame != nullptr; })) {
      frame_info.frame = nullptr;
      return keep_running();
    }
  }
  return false;
}

}  // namespace cyperstereo
#endif
