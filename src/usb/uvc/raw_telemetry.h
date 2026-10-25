#ifndef CYPERSTEREO_RAW_TELEMETRY_H_
#define CYPERSTEREO_RAW_TELEMETRY_H_
#include <array>
#include <cstdint>
#include <ostream>
#include <opencv2/core/core.hpp>
#include "smartsens_metadata.h"

namespace cyperstereo {
// Half-open column interval in the original transport plane. It describes
// overwritten bytes, not image content detected by a pixel heuristic.
struct RawMetadataRegion {
  int row = -1;
  int first_col = 0;
  int end_col = 0;
  bool active() const { return row >= 0 && end_col > first_col; }
};

// Value-owned snapshot travels with its frame, including through save queues.
// words[c] preserves both original wire bytes: low=C1, high=C2.
struct RawTelemetrySnapshot {
  bool valid = false;
  int hardware = 0, software = 0, row = -1, word_count = 0;
  std::array<uint16_t, kSmartSensMaxMetadataCols> words{};
  RawMetadataRegion RegionForPlane(int plane) const {
    RawMetadataRegion region;
    if (valid && (plane == 0 || plane == 1)) {
      region.row = row;
      region.end_col = word_count;
    }
    return region;
  }
};

inline RawTelemetrySnapshot DecodeRawTelemetrySnapshot(
    const unsigned char *lo, const unsigned char *hi, int columns,
    int row, int hardware, int software) {
  RawTelemetrySnapshot result;
  if (!lo || !hi || columns < 2 || row < 0 ||
      !IsSupportedSmartSensFirmware(hardware, software)) return result;
  const auto word = [&](int c) -> uint16_t {
    return static_cast<uint16_t>((static_cast<unsigned>(hi[c]) << 8) | lo[c]);
  };
  if (word(0) != hardware || word(1) != software) return result;
  int end = GetSmartSensMetadataLayout(hardware, software).end_col;
  const int family = SensorFamilyHardwareVersion(hardware);
  if (family == kAr0234HardwareVersion && columns >= 108 &&
      word(106) == 0xB1C1 && word(107) == 0x1005) {
    end = 126;  // GNSS 81..105 + register readback 106..125
  } else if (family == kAr0234HardwareVersion && columns >= 83 &&
      word(81) == 0xB1C1) {
    const auto revision = word(82);
    if (revision == 0x1001) end = 99;
    else if (revision >= 0x1002 && revision <= 0x1004) end = 101;
  }
  if (family == kSmartSensHardwareVersion && software == 7 &&
      columns >= 108 && word(106) == 0xA915 && word(107) == 0x0106)
    end = 132;
  // Fault/status bits inside an advertised extension do not make its bytes
  // into image pixels. Detect occupancy separately from telemetry validity.
  if (columns < end || end > static_cast<int>(result.words.size())) return result;
  result.valid = true;
  result.hardware = hardware;
  result.software = software;
  result.row = row;
  result.word_count = end;
  for (int c = 0; c < end; ++c) result.words[c] = word(c);
  return result;
}

// Only the ISP copy is repaired. The donor row is two rows above, retaining
// the Bayer phase; all other pixels (including the valid row tail) survive.
// Read-only overlay for ISP front ends: allocate/copy one row, never a frame.
// A per-camera owner prepares it before launching row workers; Row() is then
// read-only and safe to share until all those workers have completed.
class IspRawRows {
 public:
  void Prepare(const cv::Mat &raw, const RawMetadataRegion &region) {
    source_ = nullptr;
    row_ = -1;
    if (!region.active()) return;
    CV_Assert(raw.type() == CV_8UC1 && region.row >= 2 &&
              region.row < raw.rows && region.first_col >= 0 &&
              region.end_col <= raw.cols);
    raw.row(region.row).copyTo(repaired_);
    raw.row(region.row - 2).colRange(region.first_col, region.end_col)
        .copyTo(repaired_.colRange(region.first_col, region.end_col));
    source_ = raw.data;
    step_ = raw.step;
    columns_ = raw.cols;
    row_ = region.row;
  }
  const uchar *Row(const cv::Mat &raw, int y) const {
    return y == row_ && raw.data == source_ && raw.step == step_ &&
                   raw.cols == columns_
               ? repaired_.ptr<uchar>(0) : raw.ptr<uchar>(y);
  }
  // The legacy WB fallback already copies RAW; repair that existing buffer.
  void RepairExistingCopy(const cv::Mat &source, cv::Mat &copy) const {
    if (row_ >= 0 && source.data == source_) {
      CV_Assert(copy.data != source.data && copy.size() == source.size());
      repaired_.copyTo(copy.row(row_));
    }
  }
 private:
  cv::Mat repaired_;
  const uchar *source_ = nullptr;
  size_t step_ = 0;
  int row_ = -1, columns_ = 0;
};

inline void PrepareIspRaw(const cv::Mat &raw, cv::Mat &out,
                          const RawMetadataRegion &region) {
  CV_Assert(raw.type() == CV_8UC1);
  if (region.active()) {
    CV_Assert(region.row >= 2 && region.row < raw.rows &&
              region.first_col >= 0 && region.end_col <= raw.cols);
  }
  CV_Assert(&raw != &out);
  // Detach even when a caller supplied a shallow alias of the input.
  if (out.datastart && out.datastart == raw.datastart) out.release();
  raw.copyTo(out);
  if (!region.active()) return;
  raw.row(region.row - 2).colRange(region.first_col, region.end_col)
      .copyTo(out.row(region.row).colRange(region.first_col, region.end_col));
}

inline void WriteRawTelemetryCsv(std::ostream &out,
                                 const RawTelemetrySnapshot &snapshot) {
  out << "hardware,software,row,column,low_C1,high_C2,word16\n";
  if (!snapshot.valid) return;
  for (int c = 0; c < snapshot.word_count; ++c) {
    const auto w = snapshot.words[c];
    out << snapshot.hardware << ',' << snapshot.software << ',' << snapshot.row
        << ',' << c << ',' << (w & 255) << ',' << (w >> 8) << ',' << w << '\n';
  }
}
}  // namespace cyperstereo
#endif
