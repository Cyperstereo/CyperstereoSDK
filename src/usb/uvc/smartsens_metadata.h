#ifndef CYPERSTEREO_SMARTSENS_METADATA_H_
#define CYPERSTEREO_SMARTSENS_METADATA_H_

#include <string>

// Platform-neutral SmartSens firmware and metadata description.
//
// Keep this header free of OpenCV, UVC and SIMD dependencies: the same layout
// decisions must be compiled into the Windows sample, Linux/V4L2, ROS/ROS 2
// and both 32-bit and 64-bit ARM builds.
namespace cyperstereo {

// Cyperstereo SmartSens serials encode the camera count in character 1:
// S0xxxxxx=quad and S2xxxxxx=stereo. Return 0 for another family/SKU so the
// caller can use UVC-size or legacy fallback logic.
inline int SmartSensCameraCountFromSerial(const std::string &serial_number) {
  if (serial_number.size() < 2 ||
      (serial_number[0] != 'S' && serial_number[0] != 's'))
    return 0;
  if (serial_number[1] == '0') return 4;
  if (serial_number[1] == '2') return 2;
  return 0;
}

static constexpr int kSmartSensHardwareVersion = 2;
static constexpr int kSmartSensSoftwareVersion3 = 3;
static constexpr int kSmartSensSoftwareVersion4 = 4;
static constexpr int kSmartSensSoftwareVersion5 = 5;
static constexpr int kSmartSensSoftwareVersion6 = 6;

static constexpr int kMetaImuBaseCol = 5;
static constexpr int kImuWordsPerSample = 9;
static constexpr int kSmartSensLegacyImuSamplesPerFrame = 7;
static constexpr int kSmartSensV5ImuSamplesPerFrame = 13;
static constexpr int kImuMaxSamplesPerFrame =
    kSmartSensV5ImuSamplesPerFrame;

static constexpr double kImageGapThresholdSec = 0.040;
static constexpr double kSmartSensV5ImageGapThresholdSec = 0.080;

// SC136HGS row (line) time depends on the FPGA register table. Software 03/04
// use the legacy HTS=362 timing. Software 05 writes HTS=358 and runs SCLK at
// about 15.1875 MHz. Software 06 keeps that SCLK but raises HTS to 452, so its
// exposure telemetry must not be converted with the legacy line time.
static constexpr double kSmartSensLegacyLineTimeSec = 23.868131868e-6;
static constexpr double kSmartSensV5LineTimeSec = 23.572016461e-6;
static constexpr double kSmartSensV6LineTimeSec = 452.0 / 15187500.0;

// Metadata is selected from the marker in columns 0/1. Software 03 has seven
// IMU slots and no AE telemetry; software 04 adds telemetry at columns 68..80.
// Software 05 carries twelve mandatory plus one optional IMU sample and moves
// that telemetry to columns 122..134. Software 06 returns to the software-04
// seven-slot/columns-68..80 layout, but uses the longer HTS=452 line time.
struct SmartSensMetadataLayout {
  int imu_samples_per_frame;
  int ae_marker_col;
  int exposure_base_col;
  int temperature_base_col;
  int gain_base_col;
  int end_col;  // exclusive
  double image_gap_threshold_sec;
  double line_time_sec;
  bool has_ae_telemetry;
  bool zero_fills_unused_imu;
};

inline bool IsSupportedSmartSensFirmware(int hardware_version,
                                         int software_version) {
  return hardware_version == kSmartSensHardwareVersion &&
         (software_version == kSmartSensSoftwareVersion3 ||
          software_version == kSmartSensSoftwareVersion4 ||
          software_version == kSmartSensSoftwareVersion5 ||
          software_version == kSmartSensSoftwareVersion6);
}

inline SmartSensMetadataLayout GetSmartSensMetadataLayout(
    int hardware_version, int software_version) {
  if (hardware_version == kSmartSensHardwareVersion &&
      software_version == kSmartSensSoftwareVersion6) {
    return SmartSensMetadataLayout{
        kSmartSensLegacyImuSamplesPerFrame,
        68, 69, 73, 77, 81,
        kImageGapThresholdSec,
        kSmartSensV6LineTimeSec,
        true,
        false};
  }
  if (hardware_version == kSmartSensHardwareVersion &&
      software_version == kSmartSensSoftwareVersion5) {
    return SmartSensMetadataLayout{
        kSmartSensV5ImuSamplesPerFrame,
        122, 123, 127, 131, 135,
        kSmartSensV5ImageGapThresholdSec,
        kSmartSensV5LineTimeSec,
        true,
        true};
  }
  if (hardware_version == kSmartSensHardwareVersion &&
      software_version == kSmartSensSoftwareVersion4) {
    return SmartSensMetadataLayout{
        kSmartSensLegacyImuSamplesPerFrame,
        68, 69, 73, 77, 81,
        kImageGapThresholdSec,
        kSmartSensLegacyLineTimeSec,
        true,
        false};
  }
  return SmartSensMetadataLayout{
      kSmartSensLegacyImuSamplesPerFrame,
      -1, -1, -1, -1, 68,
      kImageGapThresholdSec,
      kSmartSensLegacyLineTimeSec,
      false,
      false};
}

static constexpr int kSmartSensMaxMetadataCols = 135;

}  // namespace cyperstereo

#endif  // CYPERSTEREO_SMARTSENS_METADATA_H_
