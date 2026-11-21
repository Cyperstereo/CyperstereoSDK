#ifndef CYPERSTEREO_CAMERA_FRAME_RATE_H_
#define CYPERSTEREO_CAMERA_FRAME_RATE_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <thread>

#include "smartsens_metadata.h"
#include "uvc.h"

// Frame rate of SC136HGS bitstreams with a selectable rate. The rate is kept
// in the camera's Flash parameter sector and applied by the FPGA at power-up,
// so a stored rate takes effect after the camera is power-cycled.
//
// Requests travel as 32-byte packets on FX3 XU selector 5 to the FPGA
// parameter bridge (the same channel as the maximum exposure/gain tool):
//   {0x46, op, seq, count, address[3], 0, payload[22], crc16}
// Reply:
//   {0x46, 0x01, status, seq, length, op, 0, 0, data[length], ..., crc16}
// CRC16 is CCITT-FALSE over the whole packet, big-endian at the end.
//   op 30h, count 0, address 'F','P','S': query, 16 bytes
//     {'F', 'R', 2, active fps, default fps, mask, imu slots, from_flash, 0...}
//     mask bit0..4 = 10/15/20/30/60 Hz; from_flash = active rate came from the
//     stored record. Older bitstreams answer status 3.
//   op 05h: Flash status register (bit0 busy, bits 2..6 write protection).
//   op 31h, address 0: erase the parameter sector (both records).
//   op 32h/33h, count 16, address 0 or 10h: program/read a record.
// Records (16 bytes, CRC16 over bytes 0..13 in bytes 14..15):
//   offset 0   maximum exposure/gain: {'C', 'P', 1, sensor, ..., 6 zero bytes}
//   offset 10h frame rate:            {'F', 'R', 1, fps, 10 zero bytes}
// An erased, corrupt or unsupported frame-rate record selects the default.
CYPERSTEREO_BEGIN_NAMESPACE

struct CameraFrameRateInfo {
  int active_fps = 0;
  int default_fps = 0;   // build-time rate (EZ_CAMERA_LINKER_Params.v)
  unsigned supported_mask = 0;  // bit0..4 = 10/15/20/30/60 Hz
  int imu_slots = 0;     // metadata IMU slots per frame below 30 Hz
  bool from_flash = false;  // active_fps came from the stored record
  int stored_fps = 0;    // Flash record: fps, 0 = none, -1 = corrupt

  bool Supports(int fps) const {
    switch (fps) {
      case 10: return (supported_mask & 1) != 0;
      case 15: return (supported_mask & 2) != 0;
      case 20: return (supported_mask & 4) != 0;
      case 30: return (supported_mask & 8) != 0;
      case 60: return (supported_mask & 16) != 0;
      default: return false;
    }
  }
  // Rate the camera selects at its next power-up.
  int NextBootFps() const {
    return Supports(stored_fps) ? stored_fps : default_fps;
  }
  bool RestartPending() const { return NextBootFps() != active_fps; }
};

namespace frame_rate_detail {

static constexpr uint8_t kXuUnit = 3;
static constexpr uint8_t kXuSelector = 5;
static constexpr int kPacketSize = 32;
static constexpr int kRecordSize = 16;
static constexpr uint8_t kOpStatus = 0x05, kOpInfo = 0x30, kOpErase = 0x31,
                         kOpProgram = 0x32, kOpRead = 0x33;
static constexpr uint32_t kFpsAddress = 0x465053;  // 'F','P','S'
static constexpr uint32_t kExposureOffset = 0x00, kFpsOffset = 0x10;
static constexpr uint8_t kStatusPending = 1;

inline uint16_t Crc16(const uint8_t *data, int size) {
  uint16_t crc = 0xffff;
  for (int i = 0; i < size; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                           : static_cast<uint16_t>(crc << 1);
  }
  return crc;
}

inline const char *StatusText(int status) {
  switch (status) {
    case 2: return "request CRC error";
    case 3: return "FPGA bitstream cannot store the frame rate (upgrade FPGA)";
    case 0x81: return "FX3 busy (old FX3 firmware: stop streaming or upgrade FX3)";
    case 0x82: return "FX3 firmware has no FPGA parameter channel";
    case 0x83: return "FX3 to FPGA I2C write failed";
    case 0x84: return "FX3 to FPGA I2C read failed";
    case 0x85: return "malformed FPGA reply";
    case 0x87: return "FPGA reply CRC error";
    default: return "unknown status";
  }
}

inline bool Transfer(const uvc::device &device, uvc::xu &xu, bool write,
                     uint8_t *data) {
  try {
    return uvc::xu_control_query(
        device, xu, kXuSelector,
        write ? uvc::XU_QUERY_SET : uvc::XU_QUERY_GET, kPacketSize, data);
  } catch (const std::exception &) {
    return false;
  }
}

// Linux/macOS address the XU by unit id; Windows by topology node, which is
// found by probing for a node whose selector 5 answers with the bridge header.
inline bool FindXu(const uvc::device &device, uvc::xu *xu) {
  *xu = uvc::xu{kXuUnit, 0, {0xffffffff, 0xffff, 0xffff,
                             {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}}};
#ifdef _WIN32
  for (int node = 0; node < 32; ++node) {
    xu->node = node;
    uint8_t probe[kPacketSize] = {};
    if (Transfer(device, *xu, false, probe) && probe[0] == 0x46 &&
        probe[1] == 1)
      return true;
  }
  return false;
#else
  return true;
#endif
}

class Bridge {
 public:
  Bridge(const uvc::device &device, std::string *error)
      : device_(device), error_(error) {}

  bool Open() {
    return FindXu(device_, &xu_) ||
           Fail("camera has no FX3 parameter control (XU selector 5)");
  }

  bool Fail(const std::string &text) {
    if (error_) *error_ = text;
    return false;
  }

  // Sends one request; copies `length` reply data bytes to out.
  bool Command(uint8_t op, int count, uint32_t address, const uint8_t *data,
               int length, uint8_t *out) {
    static std::atomic<unsigned> sequence{0};
    const uint8_t seq = static_cast<uint8_t>(++sequence);
    uint8_t packet[kPacketSize] = {0x46,
                                   op,
                                   seq,
                                   static_cast<uint8_t>(count),
                                   static_cast<uint8_t>(address >> 16),
                                   static_cast<uint8_t>(address >> 8),
                                   static_cast<uint8_t>(address),
                                   0};
    if (data) std::memcpy(packet + 8, data, kRecordSize);
    const uint16_t crc = Crc16(packet, kPacketSize - 2);
    packet[kPacketSize - 2] = static_cast<uint8_t>(crc >> 8);
    packet[kPacketSize - 1] = static_cast<uint8_t>(crc);
    if (!Transfer(device_, xu_, true, packet)) return Fail("XU SET failed");

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    uint8_t reply[kPacketSize];
    while (true) {
      if (!Transfer(device_, xu_, false, reply)) return Fail("XU GET failed");
      if (reply[0] != 0x46 || reply[1] != 1 || Crc16(reply, kPacketSize) != 0)
        return Fail("corrupt camera reply");
      if (reply[3] == seq && reply[5] == op && reply[2] != kStatusPending)
        break;
      if (std::chrono::steady_clock::now() > deadline)
        return Fail("camera reply timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (reply[2] != 0)
      return Fail(std::string("op 0x") + Hex(op) + " status 0x" +
                  Hex(reply[2]) + ": " + StatusText(reply[2]));
    if (reply[4] != length) return Fail("camera reply has a wrong length");
    if (out) std::memcpy(out, reply + 8, length);
    return true;
  }

  bool Query(CameraFrameRateInfo *info) {
    uint8_t body[kRecordSize], record[kRecordSize];
    if (!Command(kOpInfo, 0, kFpsAddress, nullptr, kRecordSize, body))
      return false;
    if (body[0] != 'F' || body[1] != 'R' || body[2] != 2 ||
        !IsSupportedFrameRate(body[3]))
      return Fail(body[0] == 'F' && body[1] == 'R'
                      ? StatusText(3)
                      : "camera reply is not a frame-rate record");
    if (!Read(kFpsOffset, record)) return false;
    if (info) {
      info->active_fps = body[3];
      info->default_fps = body[4];
      info->supported_mask = body[5] & 0x1f;
      info->imu_slots = body[6];
      info->from_flash = (body[7] & 1) != 0;
      info->stored_fps = StoredFps(record);
    }
    return true;
  }

  bool Read(uint32_t offset, uint8_t *record) {
    return Command(kOpRead, kRecordSize, offset, nullptr, kRecordSize, record);
  }

  bool WaitReady(uint8_t *status) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (true) {
      if (!Command(kOpStatus, 0, 0, nullptr, 1, status)) return false;
      if (!(*status & 1)) return true;
      if (std::chrono::steady_clock::now() > deadline)
        return Fail("Flash stays busy");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  bool Program(uint32_t offset, const uint8_t *record) {
    uint8_t status, back[kRecordSize];
    return Command(kOpProgram, kRecordSize, offset, record, kRecordSize,
                   nullptr) &&
           WaitReady(&status) && Read(offset, back) &&
           (std::memcmp(back, record, kRecordSize) == 0 ||
            Fail("Flash read-back mismatch"));
  }

  bool Erase() {
    uint8_t status, back[kRecordSize];
    if (!WaitReady(&status)) return false;
    if (status & 0x7c) return Fail("Flash is write protected, nothing written");
    if (!Command(kOpErase, 0, 0, nullptr, 0, nullptr) || !WaitReady(&status) ||
        !Read(kExposureOffset, back))
      return false;
    for (uint8_t b : back)
      if (b != 0xff) return Fail("parameter sector erase failed");
    return true;
  }

  // 0 erased, -1 corrupt, else the stored fps.
  static int StoredFps(const uint8_t *r) {
    if (Erased(r)) return 0;
    if (r[0] != 'F' || r[1] != 'R' || r[2] != 1 || !ZeroCrc(r, 4)) return -1;
    return r[3];
  }
  static bool ValidExposureRecord(const uint8_t *r) {
    return r[0] == 'C' && r[1] == 'P' && r[2] == 1 && ZeroCrc(r, 8);
  }
  static void MakeFpsRecord(int fps, uint8_t *r) {
    std::memset(r, 0, kRecordSize);
    r[0] = 'F';
    r[1] = 'R';
    r[2] = 1;
    r[3] = static_cast<uint8_t>(fps);
    const uint16_t crc = Crc16(r, 14);
    r[14] = static_cast<uint8_t>(crc >> 8);
    r[15] = static_cast<uint8_t>(crc);
  }

 private:
  static bool Erased(const uint8_t *r) {
    for (int i = 0; i < kRecordSize; ++i)
      if (r[i] != 0xff) return false;
    return true;
  }
  // Reserved bytes [first, 14) are zero and the record CRC checks out.
  static bool ZeroCrc(const uint8_t *r, int first) {
    for (int i = first; i < 14; ++i)
      if (r[i]) return false;
    return Crc16(r, kRecordSize) == 0;
  }
  static std::string Hex(uint8_t v) {
    return {"0123456789abcdef"[v >> 4], "0123456789abcdef"[v & 15]};
  }

  const uvc::device &device_;
  std::string *error_;
  uvc::xu xu_{};
};

}  // namespace frame_rate_detail

// Reads the active, default, supported and stored frame rates. Works while
// streaming with current FX3 firmware.
inline bool QueryCameraFrameRate(const uvc::device &device,
                                 CameraFrameRateInfo *info,
                                 std::string *error = nullptr) {
  frame_rate_detail::Bridge bridge(device, error);
  return bridge.Open() && bridge.Query(info);
}

// Stores fps (10/15/20/30/60, or 0 to clear the record and use the default) in
// the camera Flash; it takes effect after the camera is power-cycled. The
// stored maximum exposure/gain record is kept. *info receives the state after
// writing (active_fps is unchanged until the power cycle).
inline bool StoreCameraFrameRate(const uvc::device &device, int fps,
                                 CameraFrameRateInfo *info = nullptr,
                                 std::string *error = nullptr) {
  using namespace frame_rate_detail;
  Bridge bridge(device, error);
  CameraFrameRateInfo current;
  if (!bridge.Open() || !bridge.Query(&current)) return false;
  if (fps != 0 && !current.Supports(fps))
    return bridge.Fail("this camera does not support " + std::to_string(fps) +
                       " Hz");
  uint8_t exposure[kRecordSize], rate[kRecordSize];
  if (!bridge.Read(kExposureOffset, exposure)) return false;
  const bool keep_exposure = Bridge::ValidExposureRecord(exposure);
  if (!bridge.Erase()) return false;
  if (keep_exposure && !bridge.Program(kExposureOffset, exposure)) return false;
  if (fps != 0) {
    Bridge::MakeFpsRecord(fps, rate);
    if (!bridge.Program(kFpsOffset, rate)) return false;
  }
  return bridge.Query(info ? info : &current);
}

CYPERSTEREO_END_NAMESPACE

#endif  // CYPERSTEREO_CAMERA_FRAME_RATE_H_
