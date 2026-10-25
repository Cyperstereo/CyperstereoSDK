#ifndef CYPERSTEREO_SMARTSENS_METADATA_H_
#define CYPERSTEREO_SMARTSENS_METADATA_H_

#include <string>

// Platform-neutral SmartSens firmware and metadata description.
//
// Keep this header free of OpenCV, UVC and SIMD dependencies: the same layout
// decisions must be compiled into the Windows sample, Linux/V4L2, ROS/ROS 2
// and both 32-bit and 64-bit ARM builds.
namespace cyperstereo {

// Cyperstereo SmartSens serials encode the SKU in character 1:
// S0xxxxxx=quad (C1/C2/C4/C3), S1xxxxxx=mono color (process C1 only),
// S2xxxxxx=stereo (C1/C2). The FX3 transport may still carry unused lanes
// for S1; consumers must skip those planes. Return 0 for another family/SKU
// so the caller can use UVC-size or legacy fallback logic.
inline int SmartSensCameraCountFromSerial(const std::string &serial_number) {
  if (serial_number.size() < 2 ||
      (serial_number[0] != 'S' && serial_number[0] != 's'))
    return 0;
  if (serial_number[1] == '0') return 4;
  if (serial_number[1] == '1') return 1;
  if (serial_number[1] == '2') return 2;
  return 0;
}

inline bool IsSmartSensC1OnlySerial(const std::string &serial_number) {
  return SmartSensCameraCountFromSerial(serial_number) == 1;
}

// How many image planes a consumer should ISP / preview / save.
// S1 still arrives on the S0/S2 lane layout; only C1 (image1/left) is used.
inline int SmartSensProcessedCameraCount(const std::string &serial_number,
                                         int profile_num_cameras) {
  if (IsSmartSensC1OnlySerial(serial_number)) return 1;
  return profile_num_cameras;
}

// AR0234 serials encode the SKU in character 1: A0xxxxx=quad,
// A6xxxxx=six-camera. Both use the four-lane 3840x1080 transport; the
// six-image lane layout itself is still selected per frame by metadata
// column 0 (03 / 05). Return 0 for another family/SKU.
inline int Ar0234CameraCountFromSerial(const std::string &serial_number) {
  if (serial_number.size() < 2 ||
      (serial_number[0] != 'A' && serial_number[0] != 'a'))
    return 0;
  if (serial_number[1] == '0') return 4;
  if (serial_number[1] == '6') return 6;
  return 0;
}

static constexpr int kSmartSensHardwareVersion = 2;
static constexpr int kSmartSensSoftwareVersion3 = 3;
static constexpr int kSmartSensSoftwareVersion4 = 4;
static constexpr int kSmartSensSoftwareVersion5 = 5;
static constexpr int kSmartSensSoftwareVersion6 = 6;
static constexpr int kSmartSensSoftwareVersion7 = 7;
static constexpr int kSmartSensSoftwareVersion8 = 8;

// Metadata column 0 is the sensor family: 02 = SC136HGS, 03 = AR0234.  Both
// share the FX3 four-lane transport (C1/C2/C4/C3), the timestamp/IMU columns
// and the column-68 marker; only the plane size and telemetry differ.
static constexpr int kAr0234HardwareVersion = 3;
static constexpr int kAr0234SoftwareVersion7 = 7;

// Six-camera boards keep software 07 and mark the mode in column 0:
// 04 = SC136HGS six-camera, 05 = AR0234 six-camera.  Lanes 0/1 (C1/C2 full)
// and metadata columns 0..131 are identical to 02/07 resp. 03/07, but byte
// lanes 2/3 carry FOUR horizontally 2:1 binned images of the physical
// sensors C3..C6 instead of full-width C4/C3:
//   lane2 (third_image)  = C3 half [left]  | C4 half [right]
//   lane3 (fourth_image) = C5 half [left]  | C6 half [right]
static constexpr int kSc136SixCamHardwareVersion = 4;
static constexpr int kAr0234SixCamHardwareVersion = 5;
// Physical camera (1-based) of half image 0..3 (lane2 left/right, lane3
// left/right).
inline int SixCamHalfImageCamera(int half_index) { return half_index + 3; }
// SC136HGS six-camera C5/C6 AE telemetry, same encodings as columns 69..80:
// column 132 = {0xC6, cam6/cam5 present}; then exposure (C5, C6),
// temperature (C5, C6) and gain (C5, C6).
static constexpr int kSixCamC56MarkerCol = 132;
static constexpr int kSixCamC56MarkerHigh = 0xC6;
static constexpr int kSixCamC56ExposureCol = 133;
static constexpr int kSixCamC56TemperatureCol = 135;
static constexpr int kSixCamC56GainCol = 137;
static constexpr int kSixCamC56EndCol = 139;  // exclusive
// AR0234 six-camera (05/07) uses the same marker/exposure/gain columns with
// the AR0234 encodings (R0x3164 lines, R0x312A gain); columns 135/136 are 0
// and the C5/C6 register-readback mailboxes {addr, value, seq, status} follow
// at 139..142 (C5) and 143..146 (C6).
static constexpr int kAr0234SixCamC56MailboxCol = 139;
static constexpr int kAr0234SixCamC56EndCol = 147;  // exclusive

// Sensor family of a column-0 marker: six-image codes map to the quad code
// whose sensor, plane size, Bayer phase and telemetry they share.
inline int SensorFamilyHardwareVersion(int hardware_version) {
  if (hardware_version == kSc136SixCamHardwareVersion)
    return kSmartSensHardwareVersion;
  if (hardware_version == kAr0234SixCamHardwareVersion)
    return kAr0234HardwareVersion;
  return hardware_version;
}

inline bool IsSmartSensFamilyHardware(int hardware_version) {
  const int family = SensorFamilyHardwareVersion(hardware_version);
  return family == kSmartSensHardwareVersion ||
         family == kAr0234HardwareVersion;
}

inline bool IsAr0234FamilyHardware(int hardware_version) {
  return SensorFamilyHardwareVersion(hardware_version) ==
         kAr0234HardwareVersion;
}

inline bool IsSc136FamilyHardware(int hardware_version) {
  return SensorFamilyHardwareVersion(hardware_version) ==
         kSmartSensHardwareVersion;
}

inline bool IsSixCamHardware(int hardware_version) {
  return hardware_version == kSc136SixCamHardwareVersion ||
         hardware_version == kAr0234SixCamHardwareVersion;
}

// True when byte lanes 2/3 carry the four half-width images (04/07, 05/07).
inline bool IsSmartSensHalfResLaneLayout(int hardware_version,
                                         int software_version) {
  return IsSixCamHardware(hardware_version) &&
         software_version == kSmartSensSoftwareVersion7;
}

// The host additionally bins 2:1 vertically in RAW, so each half image is
// (cam_width / 2) x (frame_height / 2): 640x512 for SC136HGS, 960x540 for
// AR0234.
inline int SmartSensHalfResWidth(int cam_width) { return cam_width / 2; }
inline int SmartSensHalfResHeight(int frame_height) { return frame_height / 2; }

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
// AR0234 parallel row = 4 * LINE_LENGTH_PCK(612) PIXCLK at 85.714 MHz
// (24 MHz EXTCLK, PLL 50/2, VCO 600 MHz, vt_pix_clk_div 7).
static constexpr double kAr0234LineTimeSec = 2448.0 * 7.0 / 600000000.0;
// AR0234 03/07 runs the sensor AE: exposure 2..700 rows (19.992 ms).
// Sensor digital gain is 1x (R0x305E = 0x0080).  Columns 69..72 are each
// camera's R0x3164 exposure and columns 73..76 its R0x312A gain, in host
// order C1, C2, C4, C3.
// R0x3060=0x0026: fine code t=6; applied coarse cap stays 4x.
// Total maximum 4*32/26=4.92308x is the nearest setting not exceeding5x.
// Pair this decoder with AGC5 firmware. SDK digital exposure begins
// opening above 90% of that ceiling (~4.431x).
static constexpr int kAr0234FixedExposureLines = 700;
static constexpr int kAr0234FineGainCode = 6;
static constexpr int kAr0234CoarseCap = 2;  // Table 23, 4x = 2^2
static constexpr double kAr0234AnalogGainMax = 4.0 * 32.0 / 26.0;

// gain_raw == 0 means the metadata word has not been read yet.
inline double Ar0234AnalogGainFromReadback(int gain_raw) {
  if (gain_raw == 0) return kAr0234AnalogGainMax;
  int coarse = (gain_raw >> 11) & 7;
  if (coarse > kAr0234CoarseCap) coarse = kAr0234CoarseCap;
  const int t = kAr0234FineGainCode;
  const double fine = (coarse == 0 || coarse == 2)
                          ? 32.0 / (32.0 - t)
                          : 1.0 / (1.0 - (t / 2) / 16.0);
  return static_cast<double>(1 << coarse) * fine;
}
// 1.5 frame periods of the 33.33 ms AR0234 frame.
static constexpr double kAr0234ImageGapThresholdSec = 0.050;
// Host image order C1, C2, C4, C3.  Each camera runs its own AE.
static constexpr int kAr0234AeExpBaseCol = 69;    // R0x3164 lines
static constexpr int kAr0234AeGainBaseCol = 73;   // R0x312A
static constexpr int kAr0234FlashWidthUsCol = 77;
static constexpr int kAr0234FlashPeriod16UsCol = 78;
static constexpr int kAr0234C2PixDivCol = 79;     // C2 R0x302A
static constexpr int kAr0234C2FllCol = 80;        // C2 R0x300A

// Metadata is selected from the marker in columns 0/1. Software 03 has seven
// IMU slots and no AE telemetry; software 04 adds telemetry at columns 68..80.
// Software 05 carries twelve mandatory plus one optional IMU sample and moves
// that telemetry to columns 122..134. Software 06 returns to the software-04
// seven-slot/columns-68..80 layout, but uses the longer HTS=452 line time.
// Software 07 adds USB online upgrade and GNSS at columns 81..105.
// Software 08 adds USB online upgrade to 05: same 13 IMU slots, AE columns,
// 16 Hz timing and zero-filled unused slots; it does not contain GNSS.
// AR0234 03/07 keeps the software-06 seven-slot layout and the column-68
// marker but has no FPGA AEC (the sensor runs its own AE): columns 69..80
// hold the sensor exposure/gain readback. Readback layout 0x1005 carries the
// optional GNSS at 81..105 exactly as SC136HGS 02/07 and the register
// mailboxes at 106..125 (ar0234_diagnostics.h); layouts up to 0x1004 put the
// mailboxes at 81..100 and have no GNSS.
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
  if (IsSixCamHardware(hardware_version))
    return software_version == kSmartSensSoftwareVersion7;
  if (hardware_version == kAr0234HardwareVersion)
    return software_version == kAr0234SoftwareVersion7;
  return hardware_version == kSmartSensHardwareVersion &&
         (software_version == kSmartSensSoftwareVersion3 ||
          software_version == kSmartSensSoftwareVersion4 ||
          software_version == kSmartSensSoftwareVersion5 ||
          software_version == kSmartSensSoftwareVersion6 ||
          software_version == kSmartSensSoftwareVersion7 ||
          software_version == kSmartSensSoftwareVersion8);
}

inline SmartSensMetadataLayout GetSmartSensMetadataLayout(
    int hardware_version, int software_version) {
  if (IsSmartSensHalfResLaneLayout(hardware_version, software_version))
    hardware_version = SensorFamilyHardwareVersion(hardware_version);
  if (hardware_version == kAr0234HardwareVersion &&
      software_version == kAr0234SoftwareVersion7) {
    return SmartSensMetadataLayout{
        kSmartSensLegacyImuSamplesPerFrame,
        -1, -1, -1, -1, 81,
        kAr0234ImageGapThresholdSec,
        kAr0234LineTimeSec,
        false,
        false};
  }
  if (hardware_version == kSmartSensHardwareVersion &&
      (software_version == kSmartSensSoftwareVersion6 ||
       software_version == kSmartSensSoftwareVersion7)) {
    return SmartSensMetadataLayout{
        kSmartSensLegacyImuSamplesPerFrame,
        68, 69, 73, 77, software_version == kSmartSensSoftwareVersion7 ? 106 : 81,
        kImageGapThresholdSec,
        kSmartSensV6LineTimeSec,
        true,
        false};
  }
  if (hardware_version == kSmartSensHardwareVersion &&
      (software_version == kSmartSensSoftwareVersion5 ||
       software_version == kSmartSensSoftwareVersion8)) {
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

// AE marker column (ae_marker_col) = {0xAE, lo}. lo[3:0] = cam4/3/2/1 present.
// lo[7:4] = per physical sensor C4/C3/C2/C1: the FPGA already applied the
// fast-ISP BayerNR to that sensor's RAW in this frame (bit-exact with the host
// filter), so the host must not filter it again. Older bitstreams send 0.
static constexpr int kSmartSensAeMarkerHigh = 0xAE;
static constexpr int kSmartSensFpgaBayerNrShift = 4;

}  // namespace cyperstereo

#endif  // CYPERSTEREO_SMARTSENS_METADATA_H_
