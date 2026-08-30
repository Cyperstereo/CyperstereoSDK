#include <cmath>
#include <iostream>

#include "bayer_format.h"
#include "smartsens_metadata.h"

namespace {

bool Near(double a, double b) {
  return std::fabs(a - b) < 1e-15;
}

int Check(bool condition, const char *message) {
  if (condition) return 0;
  std::cerr << "FAILED: " << message << std::endl;
  return 1;
}

}  // namespace

int main() {
  using namespace cyperstereo;
  int failures = 0;

  failures += Check(SmartSensCameraCountFromSerial("S200010") == 2,
                    "uppercase S2 selects stereo");
  failures += Check(SmartSensCameraCountFromSerial("s200010") == 2,
                    "lowercase s2 selects stereo");
  failures += Check(SmartSensCameraCountFromSerial("S000061") == 4,
                    "uppercase S0 selects quad");
  failures += Check(SmartSensCameraCountFromSerial("s000061") == 4,
                    "lowercase s0 selects quad");
  failures += Check(SmartSensCameraCountFromSerial("S3") == 0,
                    "unsupported SmartSens SKU uses fallback");
  failures += Check(SmartSensCameraCountFromSerial("C200010") == 0,
                    "non-SmartSens serial is not classified");

  failures += Check(IsSupportedSmartSensFirmware(2, 3), "support 2/3");
  failures += Check(IsSupportedSmartSensFirmware(2, 4), "support 2/4");
  failures += Check(IsSupportedSmartSensFirmware(2, 5), "support 2/5");
  failures += Check(IsSupportedSmartSensFirmware(2, 6), "support 2/6");
  failures += Check(!IsSupportedSmartSensFirmware(2, 2), "reject 2/2");
  failures += Check(!IsSupportedSmartSensFirmware(2, 7), "reject 2/7");
  failures += Check(!IsSupportedSmartSensFirmware(1, 5), "reject 1/5");

  const SmartSensMetadataLayout v3 = GetSmartSensMetadataLayout(2, 3);
  failures += Check(v3.imu_samples_per_frame == 7, "2/3 IMU slots");
  failures += Check(v3.ae_marker_col == -1 && v3.end_col == 68,
                    "2/3 metadata bounds");
  failures += Check(!v3.has_ae_telemetry && !v3.zero_fills_unused_imu,
                    "2/3 metadata flags");

  const SmartSensMetadataLayout v4 = GetSmartSensMetadataLayout(2, 4);
  failures += Check(v4.imu_samples_per_frame == 7, "2/4 IMU slots");
  failures += Check(v4.ae_marker_col == 68 &&
                        v4.exposure_base_col == 69 &&
                        v4.temperature_base_col == 73 &&
                        v4.gain_base_col == 77 && v4.end_col == 81,
                    "2/4 metadata columns");
  failures += Check(v4.has_ae_telemetry && !v4.zero_fills_unused_imu,
                    "2/4 metadata flags");

  const SmartSensMetadataLayout v5 = GetSmartSensMetadataLayout(2, 5);
  failures += Check(v5.imu_samples_per_frame == 13, "2/5 IMU slots");
  failures += Check(v5.ae_marker_col == 122 &&
                        v5.exposure_base_col == 123 &&
                        v5.temperature_base_col == 127 &&
                        v5.gain_base_col == 131 && v5.end_col == 135,
                    "2/5 metadata columns");
  failures += Check(v5.has_ae_telemetry && v5.zero_fills_unused_imu,
                    "2/5 metadata flags");
  failures += Check(Near(v5.image_gap_threshold_sec, 0.080),
                    "2/5 image-gap threshold");
  failures += Check(Near(v5.line_time_sec, 23.572016461e-6),
                    "2/5 line time");
  failures += Check(kMetaImuBaseCol +
                            v5.imu_samples_per_frame * kImuWordsPerSample ==
                        v5.ae_marker_col,
                    "2/5 IMU range ends at AE marker");
  failures += Check(v5.end_col == kSmartSensMaxMetadataCols,
                    "2/5 maximum metadata width");
  failures += Check(kImuMaxSamplesPerFrame >= v3.imu_samples_per_frame &&
                        kImuMaxSamplesPerFrame >= v4.imu_samples_per_frame &&
                        kImuMaxSamplesPerFrame >= v5.imu_samples_per_frame,
                    "public IMU arrays cover every firmware layout");

  const SmartSensMetadataLayout v6 = GetSmartSensMetadataLayout(2, 6);
  failures += Check(v6.imu_samples_per_frame == 7, "2/6 IMU slots");
  failures += Check(v6.ae_marker_col == 68 &&
                        v6.exposure_base_col == 69 &&
                        v6.temperature_base_col == 73 &&
                        v6.gain_base_col == 77 && v6.end_col == 81,
                    "2/6 metadata columns");
  failures += Check(v6.has_ae_telemetry && !v6.zero_fills_unused_imu,
                    "2/6 metadata flags");
  failures += Check(Near(v6.image_gap_threshold_sec, 0.040),
                    "2/6 image-gap threshold");
  failures += Check(Near(v6.line_time_sec, 29.761316872427983e-6),
                    "2/6 HTS=452 line time");
  failures += Check(Near(168.0 * v6.line_time_sec,
                         0.004999901234567901),
                    "2/6 168-line exposure is approximately 5 ms");
  failures += Check(kImuMaxSamplesPerFrame >= v6.imu_samples_per_frame,
                    "public IMU arrays cover firmware 2/6");

  failures += Check(
      SelectBayerConversion(2, 3, 0) == BayerConversion::kColorBayerRg2Bgr &&
          SelectBayerConversion(2, 3, 2) ==
              BayerConversion::kColorBayerRg2Bgr,
      "2/3 Bayer phase");
  failures += Check(
      SelectBayerConversion(2, 4, 0) == BayerConversion::kColorBayerBg2Bgr &&
          SelectBayerConversion(2, 4, 1) ==
              BayerConversion::kColorBayerRg2Bgr &&
          SelectBayerConversion(2, 4, 2) ==
              BayerConversion::kColorBayerBg2Bgr,
      "2/4 Bayer phase");
  failures += Check(
      SelectBayerConversion(2, 5, 0) == BayerConversion::kColorBayerBg2Bgr &&
          SelectBayerConversion(2, 5, 1) ==
              BayerConversion::kColorBayerRg2Bgr &&
          SelectBayerConversion(2, 5, 2) ==
              BayerConversion::kColorBayerBg2Bgr,
      "2/5 Bayer phase");
  failures += Check(
      SelectBayerConversion(2, 6, 0) == BayerConversion::kColorBayerBg2Bgr &&
          SelectBayerConversion(2, 6, 1) ==
              BayerConversion::kColorBayerRg2Bgr &&
          SelectBayerConversion(2, 6, 2) ==
              BayerConversion::kColorBayerBg2Bgr,
      "2/6 Bayer phase");

  if (failures != 0) return 1;
  std::cout << "SmartSens S0/S2 serial and 2/3, 2/4, 2/5, 2/6 metadata tests passed"
            << std::endl;
  return 0;
}
