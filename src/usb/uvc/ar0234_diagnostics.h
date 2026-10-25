#ifndef CYPERSTEREO_AR0234_DIAGNOSTICS_H_
#define CYPERSTEREO_AR0234_DIAGNOSTICS_H_
#include <cstdint>
// Wire format of the register-readback extension, independent of the legacy
// 3/7 profile: {0xB1C1, layout} followed by four C1..C4 mailboxes. Layouts
// 0x1001..0x1004 (1010 AR project) put it at columns 81..98; layout 0x1005
// moves it to 106..123 so that columns 81..105 carry GNSS as on the SC136HGS.
// Sensor order in the extension is C1,C2,C3,C4, not display order.
namespace cyperstereo {
struct Ar0234Readback {
  uint16_t address{0}, value{0}, sequence{0}, status{0};
  bool ok() const { return status == 1; }
};
static constexpr int kAr0234ReadbackLegacyCol = 81;
static constexpr int kAr0234ReadbackCol = 106;
static constexpr uint16_t kAr0234ReadbackMagic = 0xB1C1;
static constexpr uint16_t kAr0234ReadbackLayout = 0x1005;
// Column of the 0xB1C1 magic word, or -1 when no extension is present.
// ReadWord: int column -> uint16_t word.
template <typename ReadWord>
inline int Ar0234ReadbackHeaderCol(int columns, ReadWord word) {
  if (columns >= kAr0234ReadbackCol + 18 &&
      word(kAr0234ReadbackCol) == kAr0234ReadbackMagic &&
      word(kAr0234ReadbackCol + 1) == kAr0234ReadbackLayout)
    return kAr0234ReadbackCol;
  if (columns >= kAr0234ReadbackLegacyCol + 18 &&
      word(kAr0234ReadbackLegacyCol) == kAr0234ReadbackMagic &&
      word(kAr0234ReadbackLegacyCol + 1) >= 0x1001 &&
      word(kAr0234ReadbackLegacyCol + 1) <= 0x1004)
    return kAr0234ReadbackLegacyCol;
  return -1;
}
inline bool DecodeAr0234Readback(const unsigned char *lo,
                                const unsigned char *hi, int columns,
                                int plane, Ar0234Readback &sample) {
  sample = Ar0234Readback{};
  if (!lo || !hi || plane < 0 || plane >= 4) return false;
  const auto word = [&](int c) -> uint16_t {
    return static_cast<uint16_t>((static_cast<unsigned>(hi[c]) << 8) | lo[c]);
  };
  const int header = Ar0234ReadbackHeaderCol(columns, word);
  if (header < 0) return false;
  static const int sensor_by_plane[4] = {0, 1, 3, 2};
  const int base = header + 2 + 4 * sensor_by_plane[plane];
  sample.address = word(base);
  sample.value = word(base + 1);
  sample.sequence = word(base + 2);
  sample.status = word(base + 3);
  return true;
}
// AR0234 six-camera (05/07) C5/C6 mailboxes at columns 139..146 (camera 0 =
// C5, 1 = C6), present only together with the column-132 0xC6 marker and a
// six-camera readback layout (0x1004 at 81 or 0x1005 at 106).
inline bool DecodeAr0234C56Readback(const unsigned char *lo,
                                    const unsigned char *hi, int columns,
                                    int camera, Ar0234Readback &sample) {
  sample = Ar0234Readback{};
  if (!lo || !hi || columns < 147 || camera < 0 || camera >= 2) return false;
  const auto word = [&](int c) -> uint16_t {
    return static_cast<uint16_t>((static_cast<unsigned>(hi[c]) << 8) | lo[c]);
  };
  const int header = Ar0234ReadbackHeaderCol(columns, word);
  if (header < 0) return false;
  if (header == kAr0234ReadbackLegacyCol && word(header + 1) != 0x1004)
    return false;
  if ((word(132) >> 8) != 0xC6) return false;
  const int base = 139 + 4 * camera;
  sample.address = word(base);
  sample.value = word(base + 1);
  sample.sequence = word(base + 2);
  sample.status = word(base + 3);
  return true;
}
}  // namespace cyperstereo
#endif
